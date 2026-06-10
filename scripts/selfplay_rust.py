"""Self-play generation driven by the Rust core (rust/selfplay-core).

Same output contract as selfplay_value_games.py (npz shards + games.jsonl,
loadable by train_value_head.py / train_value_trunk.py) plus **MCTS visit
distributions** per position in CSR form (visit_idx/visit_cnt/visit_ptr,
indices = from_square * 73 + action_plane) — the policy targets needed for
expert-iteration training.

The Rust engine runs N games concurrently and batches leaf evaluations across
all of them; Python only runs the batched NN forward. Build the core with:

    uvx maturin build --release -m rust/selfplay-core/Cargo.toml -o dist
    uv pip install --force-reinstall dist/selfplay_core-*.whl

Usage
-----
    uv run python scripts/selfplay_rust.py \
        --model data/models/pos2move_v2.1 --out data/selfplay/v2.1-rust \
        --games 2000 --sims 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import selfplay_core
from engine_match import OPENINGS

from train_value_head import load_model

_BUCKETS = (32, 64, 128, 256, 512, 1024)


class Forward:
    """Batched bf16 forward with bucketed static shapes (compile-friendly)."""

    def __init__(self, model_dir: str, device: str, use_ema: bool, compile_model: bool):
        self.device = device
        self.model = load_model(Path(model_dir), device).bfloat16().eval()
        if use_ema:
            ema = torch.load(Path(model_dir) / "ema_state.pt", map_location="cpu", weights_only=True)
            ema = {k.replace("_orig_mod.", ""): v for k, v in ema.items()}
            self.model.load_state_dict(ema, strict=False)
            self.model = self.model.bfloat16()
            print(f"EMA weights loaded ({len(ema)} tensors)")
        if compile_model and device == "cuda":
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.bufs = {}
        for b in _BUCKETS:  # warm up every bucket so CUDA-graph capture is up front
            self._buffers(b)
            bt, pl, ca, ep = self.bufs[b]
            for _ in range(3):
                self.model(bt, pl, ca, ep)
        if device == "cuda":
            torch.cuda.synchronize()

    def _buffers(self, bucket: int):
        if bucket not in self.bufs:
            self.bufs[bucket] = (
                torch.zeros(bucket, 64, dtype=torch.long, device=self.device),
                torch.zeros(bucket, dtype=torch.long, device=self.device),
                torch.zeros(bucket, dtype=torch.long, device=self.device),
                torch.zeros(bucket, dtype=torch.long, device=self.device),
            )
        return self.bufs[bucket]

    @torch.no_grad()
    def __call__(self, boards: np.ndarray, player: np.ndarray, castling: np.ndarray,
                 ep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(player)
        bucket = next((b for b in _BUCKETS if n <= b), _BUCKETS[-1])
        if n > _BUCKETS[-1]:  # split oversized batches
            l1, v1 = self(boards[:bucket], player[:bucket], castling[:bucket], ep[:bucket])
            l2, v2 = self(boards[bucket:], player[bucket:], castling[bucket:], ep[bucket:])
            return np.concatenate([l1, l2]), np.concatenate([v1, v2])
        bt, pl, ca, e = self._buffers(bucket)
        bt.zero_(); pl.zero_(); ca.zero_(); e.zero_()
        bt[:n] = torch.from_numpy(boards.astype(np.int64)).to(self.device)
        pl[:n] = torch.from_numpy(player.astype(np.int64)).to(self.device)
        ca[:n] = torch.from_numpy(castling.astype(np.int64)).to(self.device)
        e[:n] = torch.from_numpy(ep.astype(np.int64)).to(self.device)
        logits, value = self.model(bt, pl, ca, e)
        return (
            logits[:n].float().cpu().numpy(),
            value[:n].squeeze(-1).float().cpu().numpy(),
        )


class ShardWriter:
    """npz shards with the selfplay_value_games.py columns + CSR visit dists."""

    SCALAR_COLS = ("player", "castling", "ep", "halfmove", "z", "root_v")

    def __init__(self, out_dir: Path, flush_positions: int = 10_000):
        self.out_dir = out_dir
        self.flush_positions = flush_positions
        self.shard_idx = len(list(out_dir.glob("positions_*.npz")))
        self._reset()

    def _reset(self):
        self.cols = {c: [] for c in self.SCALAR_COLS}
        self.boards, self.game_id = [], []
        self.visit_idx, self.visit_cnt = [], []
        self.visit_lens = []
        self.n = 0

    def add_game(self, g: dict, game_id: int):
        p = len(g["z"])
        self.boards.append(np.asarray(g["boards"], dtype=np.uint8))
        for c in self.SCALAR_COLS:
            self.cols[c].append(np.asarray(g[c]))
        self.game_id.append(np.full(p, game_id, dtype=np.uint32))
        self.visit_idx.append(np.asarray(g["visit_idx"], dtype=np.uint16))
        self.visit_cnt.append(np.asarray(g["visit_cnt"], dtype=np.uint16))
        ptr = np.asarray(g["visit_ptr"], dtype=np.int64)
        self.visit_lens.append(np.diff(ptr))
        self.n += p
        if self.n >= self.flush_positions:
            self.flush()

    def flush(self):
        if self.n == 0:
            return
        lens = np.concatenate(self.visit_lens)
        ptr = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=ptr[1:])
        arrays = {
            "boards": np.concatenate(self.boards),
            "game_id": np.concatenate(self.game_id),
            "visit_idx": np.concatenate(self.visit_idx),
            "visit_cnt": np.concatenate(self.visit_cnt),
            "visit_ptr": ptr,
        }
        casts = {"player": np.uint8, "castling": np.uint8, "ep": np.uint8,
                 "halfmove": np.uint16, "z": np.float32, "root_v": np.float32}
        for c in self.SCALAR_COLS:
            arrays[c] = np.concatenate(self.cols[c]).astype(casts[c])
        path = self.out_dir / f"positions_{self.shard_idx:04d}.npz"
        np.savez_compressed(path, **arrays)
        print(f"  wrote {path.name} ({self.n} positions)", flush=True)
        self.shard_idx += 1
        self._reset()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="data/models/pos2move_v2.1")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--sims", type=int, default=128)
    p.add_argument("--parallel-games", type=int, default=64)
    p.add_argument("--leaves-per-wave", type=int, default=4)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--fpu", type=float, default=0.2)
    p.add_argument("--move-temp", type=float, default=1.0)
    p.add_argument("--temp-plies", type=int, default=30)
    p.add_argument("--resign", type=float, default=0.93)
    p.add_argument("--resign-plies", type=int, default=6)
    p.add_argument("--no-resign-frac", type=float, default=0.1)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--tt-cap", type=int, default=2_000_000)
    p.add_argument("--flush-positions", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    games_log = out_dir / "games.jsonl"
    id_offset = sum(1 for _ in open(games_log)) if games_log.exists() else 0
    if id_offset:
        print(f"Resuming after {id_offset} existing games in {out_dir}")
    (out_dir / "generation_config.json").write_text(json.dumps(vars(args), indent=2))

    cfg = {
        "num_sims": args.sims,
        "num_parallel_games": args.parallel_games,
        "leaves_per_wave": args.leaves_per_wave,
        "c_puct": args.c_puct,
        "fpu": args.fpu,
        "move_temp": args.move_temp,
        "temp_plies": args.temp_plies,
        "resign_threshold": args.resign,
        "resign_plies": args.resign_plies,
        "no_resign_frac": args.no_resign_frac,
        "max_plies": args.max_plies,
        "tt_cap": args.tt_cap,
    }
    openings = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] + list(OPENINGS)
    engine = selfplay_core.Engine(json.dumps(cfg), openings, args.seed + id_offset, args.games)
    forward = Forward(args.model, args.device, args.ema, not args.no_compile)
    writer = ShardWriter(out_dir, flush_positions=args.flush_positions)

    counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    done_games, n_positions = 0, 0
    t0 = time.time()
    log = open(games_log, "a")

    while not engine.done():
        boards, player, castling, ep = engine.collect()
        if len(player) > 0:
            logits, values = forward(boards, player, castling, ep)
        else:
            logits = np.zeros((0, 64, 73), dtype=np.float32)
            values = np.zeros(0, dtype=np.float32)
        engine.apply(logits, values)

        for g in engine.drain_finished():
            gid = id_offset + g["id"]
            z = g["z_white"]
            result = "1-0" if z > 0 else "0-1" if z < 0 else "1/2-1/2"
            counts[result] += 1
            n_positions += len(g["z"])
            writer.add_game(g, gid)
            log.write(json.dumps({
                "id": gid, "fen": g["fen"], "moves": g["moves"], "result": result,
                "end": g["end"], "plies": len(g["moves"]),
                "resign_allowed": g["resign_allowed"],
            }) + "\n")
            log.flush()
            done_games += 1
            if done_games % 20 == 0:
                el = time.time() - t0
                _, _, _, tt = engine.stats()
                print(f"{done_games}/{args.games} games | {n_positions} pos | "
                      f"{done_games / el * 3600:.0f} games/h | tt={tt}", flush=True)

    writer.flush()
    log.close()
    el = time.time() - t0
    print(f"\nDone: {done_games} games, {n_positions} positions | "
          f"W {counts['1-0']} / D {counts['1/2-1/2']} / B {counts['0-1']} | "
          f"{el / 60:.1f} min ({done_games / el * 3600:.0f} games/h)")


if __name__ == "__main__":
    main()
