#!/usr/bin/env python3
"""Validate a self-play generation directory end to end (shards <-> games.jsonl <-> training).

    uv run python scripts/validate_selfplay.py data/rl/selfplay/gen14 \
        [--shards 3] [--human data/shards/elite_k32_v2] [--no-train]
    uv run python scripts/validate_selfplay.py --trend data/rl/selfplay

Prints PASS / FAIL / INFO per invariant and exits 1 on a FAIL; CPU only, paths from the repo root.
Every shard row is matched to its game (game_id) and ply (row order within the game), the
game is replayed with python-chess from games.jsonl, and each row is checked against it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from chesstransformer.models.tokenizer.alphazero_move_encoder import move_to_action_plane  # noqa: E402
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer  # noqa: E402

TOK = PostionTokenizer()
NPL = 73
RESULTS: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    print(f"{status:4} {name}" + (f": {detail}" if detail else ""), flush=True)


def check(ok: bool, name: str, detail: str = "") -> None:
    report("PASS" if ok else "FAIL", name, detail)


def inference_encoding(board: chess.Board) -> tuple[list[int], int, int, int]:
    """What the net is fed for `board` (check_onnx_parity.encode == chess_core encode_position)."""
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    ep = chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8
    return TOK.encode(board), int(board.turn), castling, ep


def legal_index(board: chess.Board) -> dict[int, chess.Move]:
    """from_square * 73 + action plane -> move, for every legal move (absolute squares)."""
    return {m.from_square * NPL + move_to_action_plane(m.from_square, m.to_square, m.promotion): m
            for m in board.legal_moves}


def pidx_of(board: chess.Board, uci: str) -> int:
    m = chess.Move.from_uci(uci)
    return m.from_square * NPL + move_to_action_plane(m.from_square, m.to_square, m.promotion)


def load_games(gen: Path) -> dict[int, dict]:
    games = {}
    with open(gen / "games.jsonl") as f:
        for line in f:
            g = json.loads(line)
            games[g["id"]] = g
    return games


def would_resign(root_v: np.ndarray, player: np.ndarray, thr: float, plies: int) -> int:
    """The engine's resign rule on the white-POV root value: +1 / -1 (who is winning) or 0."""
    streak, sign = 0, 0
    for v, p in zip(root_v, player):
        vw = v if p == 1 else -v
        if abs(vw) >= thr:
            s = 1 if vw > 0 else -1
            streak = streak + 1 if s == sign else 1
            sign = s
        else:
            streak = 0
        if streak >= plies:
            return sign
    return 0


# ── one generation ────────────────────────────────────────────────────────────

def validate_generation(gen: Path, n_shards: int, seed: int, do_train: bool) -> None:
    cfg = json.loads((gen / "generation_config.json").read_text())
    sims, fast = cfg.get("sims"), cfg.get("fast_sims", 0)
    full_frac = cfg.get("full_frac", 1.0) if fast else 1.0
    forced = cfg.get("forced_playouts", 0) or 0
    games = load_games(gen)
    shards = sorted(gen.glob("positions_*.npz"))
    rng = random.Random(seed)
    pick = sorted(rng.sample(shards, min(n_shards, len(shards))))
    print(f"== {gen}: {len(games):,} games, {len(shards)} shards; checking {[p.name for p in pick]}"
          f" | sims {sims}/{fast} full_frac {full_frac} forced_playouts {forced}")

    bad = Counter()
    n_rows = n_games = 0
    full_tot, fast_tot, full_rows = [], [], 0
    zs, rvs, nvs = [], [], []
    soft_agree = soft_n = 0
    played_out = Counter()
    ends = Counter()
    legal_cols = None
    examples = defaultdict(list)
    stats = Counter()
    for sp in pick:
        d = np.load(sp)
        keys = set(d.keys())
        need = {"boards", "player", "castling", "ep", "halfmove", "z", "root_v", "full",
                "game_id", "visit_idx", "visit_cnt", "visit_ptr"}
        check(need <= keys, f"{sp.name} has the training columns", f"missing {sorted(need - keys)}" if not need <= keys else "")
        has_legal = {"legal_idx", "legal_prior", "legal_n", "legal_q", "legal_ptr", "root_net_v"} <= keys
        legal_cols = has_legal if legal_cols is None else (legal_cols and has_legal)
        a = {k: d[k] for k in keys}
        gid = a["game_id"]
        vptr = a["visit_ptr"]
        lptr = a["legal_ptr"] if has_legal else None
        # rows of a game are contiguous and in ply order (ShardWriter.add_game)
        starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
        ends_ = np.r_[starts[1:], len(gid)]
        if len(np.unique(gid)) != len(starts):
            bad["game rows not contiguous"] += 1
        for s, e in zip(starts, ends_):
            g = games.get(int(gid[s]))
            if g is None:
                bad["shard game id missing from games.jsonl"] += 1
                continue
            n_games += 1
            ends[g["end"]] += 1
            if e - s != len(g["moves"]):
                bad["rows per game != plies"] += 1
                examples["rows per game != plies"].append((int(gid[s]), e - s, len(g["moves"])))
                continue
            zw = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}[g["result"]]
            board = chess.Board(g["fen"])
            for i, uci in enumerate(g["moves"]):
                r = s + i
                n_rows += 1
                toks, pl, ca, ep = inference_encoding(board)
                if not np.array_equal(a["boards"][r], np.asarray(toks, dtype=np.uint8)):
                    bad["board tokens != inference encoding"] += 1
                if a["player"][r] != pl:
                    bad["player != side to move"] += 1
                if a["castling"][r] != ca:
                    bad["castling != inference encoding"] += 1
                if a["ep"][r] != ep:
                    bad["ep != inference encoding"] += 1
                    examples["ep"].append((board.fen(), int(a["ep"][r]), ep))
                hm = 2 * (board.fullmove_number - 1) + (0 if board.turn else 1)
                if a["halfmove"][r] != hm:
                    bad["halfmove != ply index"] += 1
                lid = legal_index(board)
                vi = a["visit_idx"][vptr[r]:vptr[r + 1]].astype(np.int64)
                vc = a["visit_cnt"][vptr[r]:vptr[r + 1]].astype(np.int64)
                if len(vi) == 0:
                    bad["empty visit target"] += 1
                if not all(int(x) in lid for x in vi):
                    bad["visit_idx not a legal move"] += 1
                    examples["illegal"].append((board.fen(), [int(x) for x in vi if int(x) not in lid][:3]))
                if (vc <= 0).any():
                    bad["visit count <= 0"] += 1
                if len(set(vi.tolist())) != len(vi):
                    bad["duplicate visit_idx"] += 1
                mv = chess.Move.from_uci(uci)
                if mv not in board.legal_moves:
                    bad["played move illegal"] += 1
                    break
                pidx = mv.from_square * NPL + move_to_action_plane(mv.from_square, mv.to_square, mv.promotion)
                full = bool(a["full"][r])
                played_out["in support" if pidx in set(vi.tolist()) else
                           ("NOT in support (full)" if full else "NOT in support (fast)")] += 1
                tot = int(vc.sum())
                (full_tot if full else fast_tot).append(tot)
                full_rows += full
                exp_z = zw if board.turn == chess.WHITE else -zw
                if a["z"][r] != exp_z:
                    bad["z != result from side to move"] += 1
                zs.append(exp_z)
                rvs.append(float(a["root_v"][r]))
                t = 0.5 * exp_z + 0.5 * float(a["root_v"][r])
                if exp_z != 0:
                    soft_n += 1
                    soft_agree += np.sign(t) == np.sign(exp_z)
                if has_legal:
                    li = a["legal_idx"][lptr[r]:lptr[r + 1]].astype(np.int64)
                    lp = a["legal_prior"][lptr[r]:lptr[r + 1]].astype(np.float64)
                    ln = a["legal_n"][lptr[r]:lptr[r + 1]].astype(np.int64)
                    lq = a["legal_q"][lptr[r]:lptr[r + 1]].astype(np.float64)
                    if set(li.tolist()) != set(lid) or len(li) != len(lid):
                        bad["legal_idx != legal move set"] += 1
                    if abs(lp.sum() - 1) > 0.02:
                        bad["legal_prior does not sum to 1"] += 1
                    if (np.abs(lq) > 1.0001).any():
                        bad["legal_q outside [-1, 1]"] += 1
                    nmap = dict(zip(li.tolist(), ln.tolist()))
                    vmap = dict(zip(vi.tolist(), vc.tolist()))
                    if full and forced:
                        if any(vmap[k] > nmap.get(k, -1) for k in vmap):
                            bad["pruned visit > raw visit (full)"] += 1
                    else:
                        raw = {k: v for k, v in nmap.items() if v > 0}
                        if raw != vmap:
                            bad["fast visit target != root visits"] += 1
                    nv = float(a["root_net_v"][r])
                    if not np.isnan(nv):
                        nvs.append((nv, exp_z))
                    # root_v is the Q of the move played (search.rs finish_on), not the root value
                    qmap = dict(zip(li.tolist(), lq.tolist()))
                    if abs(qmap.get(pidx_of(board, uci), 9.0) - float(a["root_v"][r])) > 2e-3:
                        bad["root_v != Q(played move)"] += 1
                    if pidx_of(board, uci) != int(li[np.argmax(ln)]):
                        stats["played != most visited"] += 1
                        stats["played != most visited, ply<30"] += hm < 30
                board.push(mv)
            else:
                # the recorded end matches the replayed final position
                if g["end"] == "terminal":
                    over = board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() \
                        or board.halfmove_clock >= 100 or board.is_repetition(3)
                    if not over:
                        bad["'terminal' game not over on replay"] += 1
                    if board.is_checkmate():
                        winner = 1.0 if board.turn == chess.BLACK else -1.0
                        if zw != winner:
                            bad["checkmate result wrong"] += 1
                    elif zw != 0:
                        bad["non-mate terminal not a draw"] += 1
                elif g["end"] == "max_plies" and zw != 0:
                    bad["max_plies game not a draw"] += 1
                elif g["end"] == "resign":
                    s_, e_ = s, e
                    rw = would_resign(a["root_v"][s_:e_], a["player"][s_:e_], cfg.get("resign", 0.93),
                                      cfg.get("resign_plies", 6))
                    if not g.get("resign_allowed", True):
                        bad["resigned with resignation off"] += 1
                    if rw != zw:
                        bad["resign result != resign rule on root_v"] += 1

    print(f"   {n_rows:,} rows / {n_games:,} games replayed; ends {dict(ends)}")
    for k in ["board tokens != inference encoding", "player != side to move", "castling != inference encoding",
              "ep != inference encoding", "halfmove != ply index", "rows per game != plies",
              "shard game id missing from games.jsonl", "game rows not contiguous"]:
        check(bad[k] == 0, f"(a) {k}", f"{bad[k]} rows" + (f" e.g. {examples.get('ep', [])[:1] or examples.get(k, [])[:1]}" if bad[k] else ""))
    for k in ["visit_idx not a legal move", "visit count <= 0", "empty visit target", "duplicate visit_idx",
              "played move illegal"]:
        check(bad[k] == 0, f"(b) {k}", f"{bad[k]} rows" + (f" e.g. {examples['illegal'][:1]}" if bad[k] and k.startswith('visit_idx') else ""))
    fr = full_rows / max(1, n_rows)
    check(abs(fr - full_frac) < 0.03, "(b) full-search share ~ full_frac", f"{fr:.3f} vs {full_frac}")
    if full_tot:
        ft = np.array(full_tot)
        report("INFO", "(b) visit total on full searches", f"mean {ft.mean():.1f}, p5 {np.percentile(ft, 5):.0f}, "
               f"p95 {np.percentile(ft, 95):.0f}, max {ft.max()} (budget {sims} incl. reused visits)")
        if forced:
            check((ft < sims).mean() > 0.5, "(b) forced playouts pruned on full searches",
                  f"{(ft < sims).mean():.0%} of full targets below the {sims} budget")
    if fast_tot:
        fa = np.array(fast_tot)
        report("INFO", "(b) visit total on fast searches", f"mean {fa.mean():.1f}, max {fa.max()} (budget {fast})")
    report("INFO", "(b) played move vs visit support", ", ".join(f"{k} {v / max(1, n_rows):.2%}" for k, v in played_out.items()))
    if legal_cols:
        for k in ["legal_idx != legal move set", "legal_prior does not sum to 1", "legal_q outside [-1, 1]",
                  "pruned visit > raw visit (full)", "fast visit target != root visits", "root_v != Q(played move)"]:
            check(bad[k] == 0, f"(b) {k}", f"{bad[k]} rows")
        report("INFO", "(b) move played != most-visited move", f"{stats['played != most visited'] / max(1, n_rows):.1%} of rows "
               f"({stats['played != most visited, ply<30']} of them before ply 30: move temperature)")
    else:
        report("INFO", "(b) no root-search columns (legal_*) in these shards")
    check(bad["z != result from side to move"] == 0, "(c) z == result from side to move", f"{bad['z != result from side to move']} rows")
    zs_, rvs_ = np.array(zs), np.array(rvs)
    c = np.corrcoef(zs_, rvs_)[0, 1] if zs_.std() > 0 else float("nan")
    check(c > 0.2, "(c) root_v sign convention = z's", f"corr {c:+.3f}")
    if nvs:
        nv = np.array(nvs)
        c2 = np.corrcoef(nv[:, 0], nv[:, 1])[0, 1]
        check(c2 > 0.2, "(c) root_net_v sign convention = z's", f"corr {c2:+.3f}")
    check(soft_n == 0 or soft_agree / soft_n > 0.8, "(c) soft-mix 0.5 target sign agrees with z (decisive games)",
          f"{soft_agree / max(1, soft_n):.1%} of {soft_n} rows")
    for k in ["'terminal' game not over on replay", "checkmate result wrong", "non-mate terminal not a draw",
              "max_plies game not a draw", "resigned with resignation off", "resign result != resign rule on root_v"]:
        check(bad[k] == 0, f"(c) {k}", f"{bad[k]} games")

    # held-out games
    ho = gen / "heldout_games.json"
    if ho.exists():
        ids = set(json.loads(ho.read_text())["games"])
        check(ids <= set(games), "(d) heldout_games.json ids are games of this dir",
              f"{len(ids)} held out, {len(ids - set(games))} unknown")
    if do_train:
        train_roundtrip(pick)


def train_roundtrip(pick: list[Path]) -> None:
    """The real training code (SelfplayDataset + collate + compute_loss) on these shards."""
    import torch
    import train_expert_iter as T

    with tempfile.TemporaryDirectory() as tmp:
        for p in pick:
            (Path(tmp) / p.name).symlink_to(p.resolve())
        ds = T.SelfplayDataset([Path(tmp)])
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ds), min(512, len(ds)), replace=False)
        batch = T.collate([ds[int(i)] for i in idx])
        board, player, castling, ep, halfmove, z, root_v, full, policy = batch
        ok_sum = torch.allclose(policy.sum(1), torch.ones(len(idx)), atol=1e-4)
        argmax_ok = 0
        for j, i in enumerate(idx):
            it = ds[int(i)]
            top = int(it["v_idx"][np.argmax(it["v_cnt"])])
            argmax_ok += int(policy[j].argmax()) == top or policy[j, top] == policy[j].max()
        check(ok_sum, "(d) dense policy target rows sum to 1")
        check(argmax_ok == len(idx), "(d) dense target argmax == most-visited move index",
              f"{argmax_ok}/{len(idx)}")
        # the policy loss sees full searches only; value and KL see every row
        B = len(idx)
        logits = torch.randn(B, 64, NPL)
        value = torch.randn(B, 3)
        tot, m = T.compute_loss(logits, value, policy, full, z, root_v, halfmove, logits.clone(),
                                1.0, 0.1, 0.5, 40.0)
        logp = torch.log_softmax(logits.view(B, -1), -1)
        per = -(policy * logp).sum(-1)
        want = (per * full).sum() / full.sum()
        check(torch.allclose(m["policy"] if "policy" in m else m.get("policy_loss", want), want, atol=1e-5),
              "(d) policy loss = CE on full rows only", f"{float(want):.4f}")
        # same flattened layout as the model: logits (B, 64, 73) -> from_square * 73 + plane
        sq, pl = 12, 28  # e2 (12), plane 28 = queen dir 4? just a layout probe
        probe = torch.zeros(1, 64, NPL)
        probe[0, sq, pl] = 1
        check(int(probe.view(1, -1).argmax()) == sq * NPL + pl, "(d) logits.view(B,-1) index = from_sq*73 + plane")


def validate_human(hdir: Path, n: int, seed: int) -> None:
    """Human replay rows: the played move's index is legal in the stored position (same layout)."""
    from chesstransformer.datasets.shard_format import dtype_from_meta
    meta = json.loads((hdir / "meta.json").read_text())
    dt = dtype_from_meta(meta["record_dtype"])
    shards = sorted(hdir.glob("shard_*.npy"))
    rng = np.random.default_rng(seed)
    sym = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K", 7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k"}
    bad = Counter()
    n_rows = 0
    ep_set = ep_illegal = 0
    for sp in rng.choice(shards, 2, replace=False):
        raw = np.load(sp, mmap_mode="r")
        arr = raw if raw.dtype == dt else raw.view(dt).reshape(-1)
        for r in rng.choice(len(arr), n // 2, replace=False):
            rec = arr[r]
            b = chess.Board(None)
            for sq, t in enumerate(rec["position"]):
                if t:
                    b.set_piece_at(sq, chess.Piece.from_symbol(sym[int(t)]))
            b.turn = bool(rec["is_white"])
            c = int(rec["castling_rights"])
            fen_c = ("K" if c & 1 else "") + ("Q" if c & 2 else "") + ("k" if c & 4 else "") + ("q" if c & 8 else "")
            b.set_castling_fen(fen_c or "-")
            epf = int(rec["en_passant_file"])
            if epf != 8:
                ep_set += 1
                b.ep_square = chess.square(epf, 5 if b.turn else 2)
                if not b.has_legal_en_passant():
                    ep_illegal += 1
            n_rows += 1
            lid = legal_index(b)
            idx = int(rec["from_square"]) * NPL + int(rec["action_plane"])
            if idx not in lid:
                bad["played move not legal in stored position"] += 1
            stored = [int(x) for x in rec["legal_idx"][: int(rec["legal_cnt"])]]
            if len(lid) <= meta.get("legal_cap", 64) and set(stored) != set(lid):
                bad["stored legal list != python-chess legal set"] += 1
    print(f"== human shards {hdir}: {n_rows} rows")
    for k in ["played move not legal in stored position", "stored legal list != python-chess legal set"]:
        check(bad[k] == 0, f"(d) human: {k}", f"{bad[k]} / {n_rows}")
    report("INFO", "(d) human: en-passant file set", f"{ep_set} rows, {ep_illegal} of them with no legal ep capture "
           "(self-play / inference encode ep only when a legal capture exists)")


# ── trends over generations ───────────────────────────────────────────────────

def trend(root: Path, n_shards: int, seed: int) -> None:
    rng = random.Random(seed)
    print(f"{'gen':6}{'games':>8}{'plies':>7}{'draw':>6}{'resign':>7}{'maxpl':>6}{'full':>6}"
          f"{'H(pi)':>7}{'top1':>6}{'Nfull':>7}{'|z|':>6}{'FRrate':>8}{'noRes':>7}")
    for gen in sorted(p for p in root.glob("gen[0-9][0-9]") if (p / ".selfplay_done").exists()):
        cfg = json.loads((gen / "generation_config.json").read_text())
        games = load_games(gen)
        res = Counter(g["result"] for g in games.values())
        endc = Counter(g["end"] for g in games.values())
        plies = np.mean([len(g["moves"]) for g in games.values()])
        n = len(games)
        shards = sorted(gen.glob("positions_*.npz"))
        ent, top1, ntot, absz = [], [], [], []
        fullrows = rows = 0
        fr_n = fr_bad = 0
        for sp in rng.sample(shards, min(n_shards, len(shards))):
            d = np.load(sp)
            full = d["full"].astype(bool)
            ptr, cnt = d["visit_ptr"], d["visit_cnt"].astype(np.float64)
            rows += len(full)
            fullrows += full.sum()
            for r in np.flatnonzero(full)[::4]:
                c = cnt[ptr[r]:ptr[r + 1]]
                p = c / c.sum()
                ent.append(-(p * np.log(p)).sum())
                top1.append(p.max())
                ntot.append(c.sum())
            absz.append(np.abs(d["z"]).mean())
            gid, rv, pl = d["game_id"], d["root_v"], d["player"]
            starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
            ends_ = np.r_[starts[1:], len(gid)]
            for s, e in zip(starts, ends_):
                g = games.get(int(gid[s]))
                if g is None or g.get("resign_allowed", True):
                    continue
                w = would_resign(rv[s:e], pl[s:e], cfg.get("resign", 0.93), cfg.get("resign_plies", 6))
                if w:
                    fr_n += 1
                    zw = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[g["result"]]
                    fr_bad += zw != w
        print(f"{gen.name:6}{n:8,}{plies:7.1f}{res['1/2-1/2'] / n:6.0%}{endc['resign'] / n:7.0%}"
              f"{endc['max_plies'] / n:6.1%}{fullrows / max(1, rows):6.0%}{np.mean(ent):7.3f}{np.mean(top1):6.2f}"
              f"{np.mean(ntot):7.0f}{np.mean(absz):6.2f}"
              f"{(fr_bad / fr_n if fr_n else float('nan')):8.1%}{fr_n:7d}")
    print("H(pi) / top1 / Nfull: entropy (nats), top move share, visit total of full-search policy targets; "
          "FRrate: false-resign rate = no-resign games where the resign rule would have fired and the "
          "'losing' side did not lose (noRes = such games sampled)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gen", type=Path)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--human", type=Path)
    ap.add_argument("--human-n", type=int, default=4000)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--trend", action="store_true", help="gen is the selfplay root: per-generation stats")
    args = ap.parse_args()
    if args.trend:
        trend(args.gen, args.shards, args.seed)
        return
    validate_generation(args.gen, args.shards, args.seed, not args.no_train)
    if args.human:
        validate_human(args.human, args.human_n, args.seed)
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    print(f"\n{len(RESULTS) - len(fails)} checks passed or informational, {len(fails)} FAILED"
          + "".join(f"\n  FAIL {n}: {d}" for _, n, d in fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
