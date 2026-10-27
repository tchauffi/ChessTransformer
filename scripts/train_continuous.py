"""Continuous training on a live self-play directory, after Braun 2026, "Engineering Efficient
Self-Play Chess: Search, Replay, and Throughput Under Limited Compute" (arXiv 2609.37447).

No generations: the trainer follows self-play's output, keeps a replay window of the most
recent positions, trains as many samples as the new positions allow (--reuse presentations
each: 4 in the paper, where 4, 6.25 and 8 performed alike) and publishes its EMA weights every
--block-steps steps; self-play loads them in place (selfplay_rust.py --reload-from). One
optimizer runs throughout, where the generational loop started a fresh AdamW, warm-up and
decay for every candidate and saw each position once.

    uv run python scripts/train_continuous.py --live data/rl/continuous/selfplay \
        --seed-data data/rl/selfplay/gen16 --base data/rl/champion \
        --state data/rl/continuous/trainer --publish data/rl/continuous/published

The targets are train_expert_iter.py's (visit counts at --policy-temp, WDL value on
(1 - soft_mix) z + soft_mix root_v, human replay), from the same loss code.

--state holds checkpoint.pt -- weights, EMA, optimizer, counters and the shards admitted --
rewritten after every block, so a stop costs at most one block; SIGTERM saves first.
--rollback-to <model dir> restarts from those weights with a fresh optimizer (what the
orchestrator does after two failed evaluations). Games whose id hashes into --heldout-frac are
never trained on; their positions give the held-out line logged at every publication.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_expert_iter import (ACTION_SPACE, compute_loss, create_ema, human_loss,  # noqa: E402
                               load_model, swap_ema, update_ema)
from chesstransformer.datasets.flat_shard_dataset import FlatShardDataset, make_dataloader  # noqa: E402

HASH = np.uint64(0x9E3779B97F4A7C15)


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def held_out(game_id: np.ndarray, frac: float) -> np.ndarray:
    """The games kept out of training for good, by a hash of the game id: no file to keep, and
    the same games whichever process asks."""
    top = (game_id.astype(np.uint64) * HASH) >> np.uint64(40)  # top 24 bits of the product
    return top < np.uint64(int(frac * (1 << 24)))


class Shard:
    """One npz shard in RAM: its training rows and its held-out rows (whole games)."""

    def __init__(self, path: Path, policy_temp: float, heldout_frac: float):
        d = np.load(path)
        self.name = f"{path.parent.name}/{path.name}"
        self.boards, self.player = d["boards"], d["player"]
        self.castling, self.ep = d["castling"], d["ep"]
        self.halfmove = d["halfmove"].astype(np.float32)
        self.z, self.root_v = d["z"].astype(np.float32), d["root_v"].astype(np.float32)
        self.full = d["full"].astype(np.float32)
        self.v_idx, self.v_ptr = d["visit_idx"].astype(np.int64), d["visit_ptr"]
        cnt = d["visit_cnt"].astype(np.float32)
        # T > 1 softens the visit target (visits^(1/T)); normalised per position in make_batch.
        self.v_cnt = cnt if policy_temp == 1.0 else cnt ** (1.0 / policy_temp)
        held = held_out(d["game_id"], heldout_frac)
        self.train, self.held = np.flatnonzero(~held), np.flatnonzero(held)


def make_batch(parts: list[tuple[Shard, np.ndarray]]):
    """train_expert_iter.collate's tensors for rows drawn from several shards, the dense
    policy target built without a Python loop over positions."""
    cols = collections.defaultdict(list)
    n_total = sum(len(r) for _, r in parts)
    policy = np.zeros((n_total, ACTION_SPACE), dtype=np.float32)
    off = 0
    for s, rows in parts:
        n = len(rows)
        for c in ("boards", "player", "castling", "ep", "halfmove", "z", "root_v", "full"):
            cols[c].append(getattr(s, c)[rows])
        starts = s.v_ptr[rows]
        lens = s.v_ptr[rows + 1] - starts
        r = np.repeat(np.arange(n), lens)
        src = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
        cnt = s.v_cnt[src]
        tot = np.bincount(r, weights=cnt, minlength=n)
        policy[off + r, s.v_idx[src]] = cnt / np.maximum(tot[r], 1e-12)
        off += n
    cat = {c: np.concatenate(v) for c, v in cols.items()}
    long = lambda a: torch.from_numpy(a.astype(np.int64))
    flt = lambda a: torch.from_numpy(a.astype(np.float32))
    return (long(cat["boards"]), long(cat["player"]), long(cat["castling"]), long(cat["ep"]),
            flt(cat["halfmove"]), flt(cat["z"]), flt(cat["root_v"]), flt(cat["full"]),
            torch.from_numpy(policy))


class Window:
    """The most recent shards, first in first out, up to max_rows training rows."""

    def __init__(self, max_rows: int):
        self.max_rows = max_rows
        self.shards: collections.deque[Shard] = collections.deque()
        self.rows = 0

    def add(self, s: Shard) -> None:
        self.shards.append(s)
        self.rows += len(s.train)
        while len(self.shards) > 1 and self.rows - len(self.shards[0].train) >= self.max_rows:
            self.rows -= len(self.shards.popleft().train)

    def _parts(self, rng: np.random.Generator, n: int, attr: str):
        sizes = np.fromiter((len(getattr(s, attr)) for s in self.shards), dtype=np.int64)
        cum = np.cumsum(sizes)
        g = np.sort(rng.integers(0, cum[-1], n))
        si = np.searchsorted(cum, g, side="right")
        local = g - (cum - sizes)[si]
        return [(self.shards[k], getattr(self.shards[k], attr)[local[si == k]]) for k in np.unique(si)]

    def sample(self, rng: np.random.Generator, n: int):
        return make_batch(self._parts(rng, n, "train"))

    def held_batches(self, n: int, batch: int, seed: int = 0):
        """A fixed draw of n held-out positions from the current window, in batches."""
        if sum(len(s.held) for s in self.shards) == 0:
            return []
        batches, cur, cur_n = [], [], 0
        for s, rows in self._parts(np.random.default_rng(seed), n, "held"):
            i = 0
            while i < len(rows):
                take = min(batch - cur_n, len(rows) - i)
                cur.append((s, rows[i:i + take]))
                cur_n, i = cur_n + take, i + take
                if cur_n == batch:
                    batches.append(make_batch(cur))
                    cur, cur_n = [], 0
        if cur:
            batches.append(make_batch(cur))
        return batches


def shard_paths(d: Path) -> list[Path]:
    return sorted(d.glob("positions_*.npz"))


def param_groups(model, lr: float, lr_embedding: float, weight_decay: float):
    """train_expert_iter's groups: embeddings at their own lr, no decay on 1-d tensors or the
    full attention-bias table."""
    emb, decay, nodecay = [], [], []
    for n, t in model.named_parameters():
        if "embedding" in n:
            emb.append(t)
        elif t.ndim >= 2 and "bias_full" not in n:
            decay.append(t)
        else:
            nodecay.append(t)
    return [{"params": emb, "lr": lr_embedding, "weight_decay": weight_decay},
            {"params": decay, "lr": lr, "weight_decay": weight_decay},
            {"params": nodecay, "lr": lr, "weight_decay": 0.0}]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", required=True, help="self-play output directory to follow")
    p.add_argument("--seed-data", nargs="*", default=[],
                   help="directories loaded into the window at the start (older games)")
    p.add_argument("--base", required=True, help="initial network (unused once a checkpoint exists)")
    p.add_argument("--state", required=True, help="checkpoint directory")
    p.add_argument("--publish", required=True, help="published networks: step_NNNNNNN/ and latest")
    p.add_argument("--rollback-to", default=None, help="restart from this model dir's weights")
    p.add_argument("--window-rows", type=int, default=10_000_000)
    p.add_argument("--reuse", type=float, default=4.0, help="presentations per new position")
    p.add_argument("--seed-reuse", type=float, default=0.0,
                   help="presentations owed to the --seed-data positions (0: they only fill the window)")
    p.add_argument("--min-rows", type=int, default=200_000, help="no training below this window")
    p.add_argument("--block-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-embedding", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--soft-mix", type=float, default=0.5)
    p.add_argument("--policy-temp", type=float, default=1.0)
    p.add_argument("--ramp-halfmoves", type=float, default=40.0)
    p.add_argument("--kl-coef", type=float, default=0.0,
                   help="KL anchor to the last published network (0: none, as in the paper)")
    p.add_argument("--human-shards", default=None)
    p.add_argument("--human-frac", type=float, default=0.2)
    p.add_argument("--heldout-frac", type=float, default=0.02)
    p.add_argument("--heldout-positions", type=int, default=8192)
    p.add_argument("--keep", type=int, default=24, help="publications kept on disk")
    p.add_argument("--poll", type=float, default=30.0, help="seconds between scans while waiting")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = args.device
    state_dir, pub_dir, live = Path(args.state), Path(args.publish), Path(args.live)
    state_dir.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)
    ck_path = state_dir / "checkpoint.pt"

    stopping = False

    def on_term(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    ck = torch.load(ck_path, map_location="cpu", weights_only=False) if ck_path.exists() else None
    base = Path(ck["base"]) if ck else Path(args.base).resolve()
    model = load_model(base, device)
    cfg_src = base / "model_config.json"
    fwd = torch.compile(model) if args.compile else model
    ema = create_ema(model)
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.lr_embedding, args.weight_decay))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))
    step = trained = admitted = seed_rows = 0
    seen: list[str] = []
    rng = np.random.default_rng(args.seed)
    if ck:
        model.load_state_dict(ck["model"])
        ema = {k: v.to(device) for k, v in ck["ema"].items()}
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, trained, admitted, seed_rows = ck["step"], ck["trained"], ck["admitted"], ck["seed_rows"]
        seen = ck["seen"]
        rng.bit_generator.state = ck["rng"]
        log(f"resumed at step {step}: {trained:,} samples trained, {admitted:,} positions admitted")
    if args.rollback_to:
        state = load_file(str(Path(args.rollback_to) / "model.safetensors"))
        model.load_state_dict(state)
        ema = create_ema(model)
        opt = torch.optim.AdamW(param_groups(model, args.lr, args.lr_embedding, args.weight_decay))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))
        log(f"rolled back to {args.rollback_to}: fresh optimizer, counters kept")

    # The window: the seed directories, then every live shard admitted so far, in order.
    window = Window(args.window_rows)
    t0 = time.time()
    for d in args.seed_data:
        for path in shard_paths(Path(d)):
            s = Shard(path, args.policy_temp, args.heldout_frac)
            window.add(s)
            if not ck:
                seed_rows += len(s.train)
    seen_set = set(seen)
    for name in seen:
        window.add(Shard(live.parent / name, args.policy_temp, args.heldout_frac))
    log(f"window {window.rows:,} rows in {len(window.shards)} shards ({time.time() - t0:.0f} s); "
        f"{seed_rows:,} seed rows, {admitted:,} admitted")

    human_iter = None
    if args.human_shards and args.human_frac > 0:
        human_bs = max(1, round(args.batch_size * args.human_frac / (1.0 - args.human_frac)))
        human_ds = FlatShardDataset(args.human_shards, "train")
        loader = make_dataloader(human_ds, human_bs, shuffle=True, num_workers=2, drop_last=True,
                                 seed=args.seed + step)

        def cycle():
            while True:
                yield from loader
        human_iter = cycle()
        log(f"human replay {args.human_shards}: {human_bs}/step, loss share {args.human_frac:.0%}")

    ref = None
    if args.kl_coef > 0:
        ref = load_model(base, device).eval()
        for t in ref.parameters():
            t.requires_grad_(False)

    def publish(suffix: str = "") -> str:
        swap_ema(model, ema)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        swap_ema(model, ema)
        name = f"step_{step:07d}{suffix}"
        tmp = pub_dir / f".{name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        save_file(state, str(tmp / "model.safetensors"))
        shutil.copy(cfg_src, tmp / "model_config.json")
        (tmp / "meta.json").write_text(json.dumps({
            "step": step, "trained": trained, "admitted": admitted, "window_rows": window.rows,
            "base": str(base), "time": time.strftime("%F %T"),
            "args": {k: v for k, v in vars(args).items() if k != "rollback_to"}}, indent=1))
        shutil.rmtree(pub_dir / name, ignore_errors=True)
        os.replace(tmp, pub_dir / name)
        link = pub_dir / ".latest.tmp"
        link.unlink(missing_ok=True)
        os.symlink(name, link)
        os.replace(link, pub_dir / "latest")
        for old in sorted(pub_dir.glob("step_*"))[:-args.keep]:
            shutil.rmtree(old, ignore_errors=True)
        if ref is not None:
            ref.load_state_dict(state)
        return name

    def checkpoint():
        tmp = state_dir / "checkpoint.pt.tmp"
        torch.save({"model": model.state_dict(), "ema": ema, "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "trained": trained,
                    "admitted": admitted, "seed_rows": seed_rows, "seen": seen,
                    "rng": rng.bit_generator.state, "base": str(base)}, tmp)
        os.replace(tmp, ck_path)

    @torch.no_grad()
    def held_line() -> str:
        batches = window.held_batches(args.heldout_positions, args.batch_size)
        if not batches:
            return "no held-out games yet"
        swap_ema(model, ema)
        model.eval()
        tot = collections.defaultdict(float)
        n = 0
        for b in batches:
            board, player, castling, ep, halfmove, z, root_v, full, policy = (t.to(device) for t in b)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits, value = model(board, player, castling, ep)
                _, m = compute_loss(logits, value, policy, full, z, root_v, halfmove, logits,
                                    args.value_weight, 0.0, args.soft_mix, args.ramp_halfmoves)
            for k in ("policy", "value", "sign_acc"):
                tot[k] += m[k].item() * len(z)
            n += len(z)
        model.train()
        swap_ema(model, ema)
        return " ".join(f"{k} {v / n:.4f}" for k, v in tot.items()) + f" on {n} held-out positions"

    if args.rollback_to:  # a new name, or self-play would keep this step's rejected weights
        log(f"published {publish('_rollback')} (the rolled-back weights)")
    elif not (pub_dir / "latest").exists():
        log(f"published {publish()} (the base network)")

    block_samples = args.block_steps * args.batch_size
    model.train()
    while not stopping:
        for path in shard_paths(live):
            name = f"{live.name}/{path.name}"
            if name in seen_set:
                continue
            try:
                s = Shard(path, args.policy_temp, args.heldout_frac)
            except Exception as e:  # a shard still being renamed into place, or damaged
                log(f"skipped {name} for now: {e}")
                continue
            window.add(s)
            admitted += len(s.train)
            seen.append(name)
            seen_set.add(name)
        budget = args.seed_reuse * seed_rows + args.reuse * admitted
        if window.rows < args.min_rows or trained + block_samples > budget:
            for _ in range(int(args.poll)):
                if stopping:
                    break
                time.sleep(1)
            continue

        run = collections.defaultdict(float)
        t_block = time.time()
        done = 0
        for _ in range(args.block_steps):
            if stopping:
                break
            board, player, castling, ep, halfmove, z, root_v, full, policy = (
                t.to(device, non_blocking=True) for t in window.sample(rng, args.batch_size))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits, value = fwd(board, player, castling, ep)
                if ref is not None:
                    with torch.no_grad():
                        ref_logits, _ = ref(board, player, castling, ep)
                else:
                    ref_logits = logits.detach()
                loss, m = compute_loss(logits, value, policy, full, z, root_v, halfmove, ref_logits,
                                       args.value_weight, args.kl_coef, args.soft_mix, args.ramp_halfmoves)
                if human_iter is not None:
                    h = human_loss(fwd, next(human_iter), device, args.value_weight)
                    loss = (1.0 - args.human_frac) * loss + args.human_frac * h
                    m["human"] = h.detach()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            sched.step()
            update_ema(model, ema, args.ema_decay)
            for k, v in m.items():
                run[k] += v.item()
            done += 1
            step += 1
        trained += done * args.batch_size
        if done < args.block_steps:  # stopped mid-block: keep the progress, publish nothing
            break
        name = publish()
        checkpoint()
        took = time.time() - t_block
        log(f"step {step} | {name} | trained {trained:,} of budget {budget:,.0f} | window {window.rows:,} "
            f"rows, {len(window.shards)} shards | admitted {admitted:,} | "
            + " ".join(f"{k} {v / done:.4f}" for k, v in run.items() if k in ("policy", "value", "kl", "human"))
            + f" | {done * args.batch_size / took:,.0f} samples/s")
        log(f"  held out (EMA): {held_line()}")
    checkpoint()
    log(f"stopped at step {step}: checkpoint saved")


if __name__ == "__main__":
    main()
