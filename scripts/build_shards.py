"""One-shot preprocessing pass: HDF5 games -> flat, memory-mappable sample shards.

Implements ``doc/dataset_shards.md`` §3. The point is to move every cost that
``HDF5ChessDataset.__getitem__`` pays per sample — gzip chunk decompression, the
python-chess replay, the legal-move enumeration, the position tokenize — to build time,
paid once, so the training-time loader is a memmap gather and nothing else.

Three things drive the design:

* **Games are visited in order.** ``moves`` is gzipped in chunks of 10,000 games, so a
  random read decompresses a whole chunk to extract one game: 0.352 ms random vs 0.017 ms
  sequential, a 13× penalty. Each shard is therefore a *contiguous game range*, and each
  worker walks its ranges front to back.
* **Each game is replayed once.** ``__getitem__`` pays a fresh O(ply) replay per sample;
  here one replay emits all K samples for that game.
* **The sampling distribution is frozen into the shard.** The K plies are drawn at build
  time with the same triangular middlegame weights as
  ``HDF5ChessDataset._sample_move_idx``, so the training loader shuffles uniformly and
  carries no weighting logic in the hot path.

The honest cost: a shard freezes K positions per game forever, where the HDF5 path
redraws a fresh position every epoch. At K=16 a run consuming 40M samples sees each
position ~1.6×. If that shows up as a widening train/val gap, rebuild with a different
``--seed`` rather than raising K.

Usage::

    uv run scripts/build_shards.py --h5 data/elite_db.h5 --out data/shards/elite_k16
    uv run scripts/build_shards.py --h5 data/elite_db.h5 --out /tmp/smoke --limit-games 5000
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import chess
import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chesstransformer.datasets.shard_format import (  # noqa: E402
    FORMAT_VERSION,
    LEGAL_CAP,
    NUM_ACTION_PLANES,
    PAD_ACTION,
    RECORD_BYTES,
    RECORD_DTYPE,
    dtype_to_meta,
)
from chesstransformer.models.tokenizer.alphazero_move_encoder import (  # noqa: E402
    move_to_action_plane,
)
from chesstransformer.models.tokenizer.move_tokenizer import MoveTokenizer  # noqa: E402
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer  # noqa: E402

# Per-worker globals, initialised once in _worker_init. Building these per game would
# dominate the replay cost.
_TOKEN_TO_MOVE: list = []
_POS_TOKENIZER: PostionTokenizer = None
_CFG: dict = {}


def _worker_init(cfg: dict):
    """Set up the per-process decode tables.

    ``_TOKEN_TO_MOVE`` replaces ``MoveTokenizer.decode`` + ``chess.Move.from_uci`` on the
    hot path with a list index. It is exactly equivalent — the same two calls, hoisted out
    of the loop — and the UCI parse is a meaningful share of a 2,000-ply-per-second budget.
    """
    global _TOKEN_TO_MOVE, _POS_TOKENIZER, _CFG
    mt = MoveTokenizer()
    _TOKEN_TO_MOVE = [None] * mt.vocab_size
    for uci, tok in mt.vocab.items():
        try:
            _TOKEN_TO_MOVE[tok] = chess.Move.from_uci(uci)
        except ValueError:
            _TOKEN_TO_MOVE[tok] = None
    _POS_TOKENIZER = PostionTokenizer()
    _CFG = cfg


def sample_plies(num_moves: int, k: int, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Choose which plies of a game become samples.

    Mirrors ``HDF5ChessDataset._sample_move_idx`` exactly — same bounds, same triangular
    weights — but draws k of them at once instead of one per epoch. Returns sorted ply
    indices so the caller can emit them during a single forward replay.

    Ply ``p`` means "the position *before* move p", so the last usable ply is
    ``num_moves - 2``; ``max_move_idx`` below is the exclusive upper bound, matching the
    ``hi`` in the online sampler.
    """
    max_move_idx = int(num_moves) - 1
    if max_move_idx <= 0:
        return np.zeros(1, dtype=np.int64)

    skip = cfg["skip_opening_plies"]
    lo = min(skip, max_move_idx - 1) if max_move_idx > 1 else 0
    hi = max_move_idx
    plies = np.arange(lo, hi, dtype=np.int64)

    if cfg["sample_weighting"] == "uniform" or hi - lo <= 8:
        weights = np.ones(len(plies), dtype=np.float64)
    else:
        ramp_up = np.minimum(plies / 20.0, 1.0)
        taper = np.minimum(np.maximum((80.0 - plies) / 40.0, 0.1), 1.0)
        weights = ramp_up * taper

    total = weights.sum()
    if total <= 0:  # every candidate ply has zero weight; fall back to uniform
        weights = np.ones(len(plies), dtype=np.float64)
        total = weights.sum()
    weights = weights / total

    if cfg["sample_mode"] == "iid":
        # Exactly reproduces the online sampler's marginal (measured TVD 0.014, at the
        # 0.005 noise floor of a 30k-sample histogram) at the cost of 13.7 % duplicate
        # rows — two draws landing on the same ply of the same game.
        return np.sort(rng.choice(plies, size=k, replace=True, p=weights))

    # Default. Without replacement there are no duplicate rows, and the marginal drift is
    # measured rather than assumed: TVD 0.037, which flattens the middlegame peak slightly
    # into both tails but moves the mean ply by 0.07 (33.73 vs 33.80). That is cosmetic
    # next to spending 13.7 % of the page-cache budget on exact repeats.
    # verify_shards.py check 2 re-measures this on whatever you actually build.
    n_available = int(np.count_nonzero(weights))
    k_eff = min(k, n_available)
    return np.sort(rng.choice(plies, size=k_eff, replace=False, p=weights))


def _encode_position(board: chess.Board, target: chess.Move, out: np.ndarray):
    """Fill one record from a board and the move played from it.

    ``out`` is a single element of a :data:`RECORD_DTYPE` array. Returns True if the
    legal-move list overflowed :data:`LEGAL_CAP` (the caller counts these).
    """
    out["position"] = _POS_TOKENIZER.encode(board)

    from_sq = target.from_square
    plane = move_to_action_plane(from_sq, target.to_square, target.promotion)
    flat_target = from_sq * NUM_ACTION_PLANES + plane

    # The target index goes first, so clipping at LEGAL_CAP can only ever drop an
    # alternative legal move — never the label the model is trained to predict.
    idxs = [flat_target]
    for mv in board.legal_moves:
        flat = mv.from_square * NUM_ACTION_PLANES + move_to_action_plane(
            mv.from_square, mv.to_square, mv.promotion
        )
        if flat != flat_target:
            idxs.append(flat)

    overflow = len(idxs) > LEGAL_CAP
    n = min(len(idxs), LEGAL_CAP)
    buf = np.full(LEGAL_CAP, PAD_ACTION, dtype=np.uint16)
    buf[:n] = idxs[:n]
    out["legal_idx"] = buf
    out["legal_cnt"] = n

    out["from_square"] = from_sq
    out["action_plane"] = plane

    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    out["castling_rights"] = castling

    out["en_passant_file"] = (
        chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8
    )
    out["is_white"] = 1 if board.turn else 0
    return overflow


def build_shard(task: dict) -> dict:
    """Build one shard: a contiguous game range -> one ``.npy`` of fixed-width records."""
    shard_id = task["shard_id"]
    g0, g1 = task["game_start"], task["game_end"]
    k = _CFG["k"]
    out_path = Path(_CFG["out_dir"]) / f"shard_{shard_id:04d}.npy"

    rng = np.random.default_rng([_CFG["seed"], shard_id])

    with h5py.File(_CFG["h5_path"], "r", rdcc_nbytes=64 << 20, rdcc_nslots=10007) as f:
        moves_ds = f["moves"]
        num_moves = f["num_moves"][g0:g1].astype(np.int64)
        results = f["result"][g0:g1].astype(np.int8)

        # ELO filter, applied to games exactly as HDF5ChessDataset.__init__ does. It has to
        # happen here rather than in the loader: the shard freezes the game population the
        # same way it freezes the ply sampling.
        min_elo, max_elo = _CFG["min_elo"], _CFG["max_elo"]
        if min_elo is not None or max_elo is not None:
            avg_elo = (f["white_elo"][g0:g1].astype(np.float64) + f["black_elo"][g0:g1]) / 2
            keep = np.ones(g1 - g0, dtype=bool)
            if min_elo is not None:
                keep &= avg_elo >= min_elo
            if max_elo is not None:
                keep &= avg_elo <= max_elo
        else:
            keep = None

        records = np.zeros((g1 - g0) * k, dtype=RECORD_DTYPE)
        n_written = 0
        n_overflow = 0
        n_bad_push = 0
        n_skipped_games = 0
        n_filtered_games = 0

        for local_g in range(g1 - g0):
            nm = int(num_moves[local_g])
            if nm < 2:  # nothing to predict from
                n_skipped_games += 1
                continue
            if keep is not None and not keep[local_g]:
                n_filtered_games += 1
                continue

            # Sequential read: this is the access pattern the whole layout exists to keep.
            game_moves = moves_ds[g0 + local_g]
            plies = sample_plies(nm, k, rng, _CFG)
            plies = plies[plies < len(game_moves)]
            if len(plies) == 0:
                n_skipped_games += 1
                continue

            result = results[local_g]
            board = chess.Board()
            next_wanted = 0
            for ply in range(int(plies[-1]) + 1):
                while next_wanted < len(plies) and plies[next_wanted] == ply:
                    target = _TOKEN_TO_MOVE[int(game_moves[ply])]
                    if target is not None and board.is_legal(target):
                        rec = records[n_written]
                        if _encode_position(board, target, rec):
                            n_overflow += 1
                        rec["result"] = result
                        rec["move_number"] = ply
                        n_written += 1
                    next_wanted += 1

                mv = _TOKEN_TO_MOVE[int(game_moves[ply])]
                if mv is None or not board.is_legal(mv):
                    # HDF5ChessDataset swallows an unpushable move and carries on; the
                    # rest of the replay is then meaningless, so drop the game instead.
                    n_bad_push += 1
                    break
                board.push(mv)

        records = records[:n_written]
        np.save(out_path, records)

    return {
        "shard_id": shard_id,
        "path": out_path.name,
        "game_start": int(g0),
        "game_end": int(g1),
        "num_samples": int(n_written),
        "num_overflow": int(n_overflow),
        "num_bad_push": int(n_bad_push),
        "num_skipped_games": int(n_skipped_games),
        "num_filtered_games": int(n_filtered_games),
    }


def choose_split_shards(num_shards: int, n_val: int, n_test: int):
    """Reserve whole shards for val/test, spread evenly across the game range.

    Splitting on *samples* would put positions from the same game on both sides of the
    split — the model would validate on openings it trained on and val loss, which drives
    best-model selection, would go quietly optimistic. Shards are contiguous game ranges,
    so reserving whole shards makes the split game-disjoint by construction.

    They are spread rather than taken from the front because games sit in the HDF5 in
    ingestion (roughly chronological) order; shards 0-9 would be an unrepresentative slice.
    """
    picks = np.linspace(0, num_shards - 1, n_val + n_test + 2)[1:-1].round().astype(int)
    picks = list(dict.fromkeys(int(p) for p in picks))
    return sorted(picks[:n_val]), sorted(picks[n_val : n_val + n_test])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="data/elite_db.h5")
    ap.add_argument("--out", default="data/shards/elite_k16")
    ap.add_argument("-k", "--positions-per-game", type=int, default=16)
    ap.add_argument("--num-shards", type=int, default=256,
                    help="256 divides every world_size x num_workers you will plausibly use")
    ap.add_argument("--val-shards", type=int, default=5)
    ap.add_argument("--test-shards", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--sample-weighting", choices=["uniform", "middlegame"], default="middlegame")
    ap.add_argument("--skip-opening-plies", type=int, default=0)
    ap.add_argument("--min-elo", type=int, default=None,
                    help="Average-ELO floor, applied to games at build time exactly as "
                         "HDF5ChessDataset does. The trainer's HDF5 path defaults to 2400; pass "
                         "the same value here if you want the shards to hold the same population.")
    ap.add_argument("--max-elo", type=int, default=None)
    ap.add_argument("--sample-mode", choices=["unique", "iid"], default="unique",
                    help="'unique' draws K distinct plies per game; 'iid' reproduces the "
                         "online sampler's marginal exactly but writes duplicate rows")
    ap.add_argument("--limit-games", type=int, default=0, help="smoke-test on the first N games")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.glob("shard_*.npy")) and not args.overwrite:
        sys.exit(f"{out_dir} already holds shards; pass --overwrite to rebuild")
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        n_games = f["moves"].shape[0]
        num_moves_all = f["num_moves"][:].astype(np.int64)
    if args.limit_games:
        n_games = min(n_games, args.limit_games)
        num_moves_all = num_moves_all[:n_games]

    num_shards = min(args.num_shards, n_games)
    bounds = np.linspace(0, n_games, num_shards + 1).round().astype(np.int64)
    tasks = [
        {"shard_id": i, "game_start": int(bounds[i]), "game_end": int(bounds[i + 1])}
        for i in range(num_shards)
        if bounds[i + 1] > bounds[i]
    ]

    val_shards, test_shards = choose_split_shards(len(tasks), args.val_shards, args.test_shards)
    reserved = set(val_shards) | set(test_shards)
    train_shards = [t["shard_id"] for t in tasks if t["shard_id"] not in reserved]

    cfg = {
        "h5_path": str(Path(args.h5).resolve()),
        "out_dir": str(out_dir.resolve()),
        "k": args.positions_per_game,
        "seed": args.seed,
        "sample_weighting": args.sample_weighting,
        "skip_opening_plies": args.skip_opening_plies,
        "sample_mode": args.sample_mode,
        "min_elo": args.min_elo,
        "max_elo": args.max_elo,
    }

    est_samples = min(args.positions_per_game * n_games, int((num_moves_all - 1).clip(min=0).sum()))
    print("=" * 74)
    print("BUILD SHARDS")
    print("=" * 74)
    print(f"source        : {args.h5}  ({n_games:,} games, {num_moves_all.sum():,} plies)")
    print(f"output        : {out_dir}")
    print(f"K             : {args.positions_per_game} positions/game  ({args.sample_mode})")
    print(f"elo filter    : {args.min_elo or 'none'} - {args.max_elo or 'none'}")
    print(f"shards        : {len(tasks)}  (train {len(train_shards)} / val {val_shards} / test {test_shards})")
    print(f"record        : {RECORD_BYTES} B  (legal cap C={LEGAL_CAP})")
    print(f"estimated     : <= {est_samples:,} samples, {est_samples * RECORD_BYTES / 1e9:.2f} GB")
    print(f"workers       : {args.workers}")
    print()

    t0 = time.perf_counter()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_worker_init, initargs=(cfg,)) as pool:
        shard_meta = list(
            tqdm(pool.imap_unordered(build_shard, tasks), total=len(tasks), desc="shards", unit="shard")
        )
    elapsed = time.perf_counter() - t0

    shard_meta.sort(key=lambda m: m["shard_id"])
    total = sum(m["num_samples"] for m in shard_meta)
    overflow = sum(m["num_overflow"] for m in shard_meta)
    bad_push = sum(m["num_bad_push"] for m in shard_meta)
    skipped = sum(m["num_skipped_games"] for m in shard_meta)
    filtered = sum(m["num_filtered_games"] for m in shard_meta)

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unknown"

    meta = {
        "format_version": FORMAT_VERSION,
        "record_dtype": dtype_to_meta(),
        "legal_cap": LEGAL_CAP,
        "pad_action": PAD_ACTION,
        "source_h5": cfg["h5_path"],
        "source_games": int(n_games),
        "k": args.positions_per_game,
        "seed": args.seed,
        "sample_mode": args.sample_mode,
        "sample_weighting": args.sample_weighting,
        "skip_opening_plies": args.skip_opening_plies,
        "min_elo": args.min_elo,
        "max_elo": args.max_elo,
        "num_samples": int(total),
        "splits": {"train": train_shards, "val": val_shards, "test": test_shards},
        "shards": shard_meta,
        "git_sha": git_sha,
        "build_seconds": round(elapsed, 1),
    }
    with (out_dir / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    def split_count(ids):
        return sum(m["num_samples"] for m in shard_meta if m["shard_id"] in set(ids))

    print()
    print(f"samples       : {total:,}  ({total * RECORD_BYTES / 1e9:.2f} GB on disk)")
    print(f"  train       : {split_count(train_shards):,}")
    print(f"  val         : {split_count(val_shards):,}")
    print(f"  test        : {split_count(test_shards):,}")
    print(f"legal cap hits: {overflow:,}  ({100 * overflow / max(total, 1):.4f} % — raise C if > 0.01 %)")
    print(f"games dropped : {skipped:,} too short, {bad_push:,} unreplayable, "
          f"{filtered:,} outside ELO range")
    print(f"build time    : {elapsed / 60:.1f} min on {args.workers} workers "
          f"({total / max(elapsed, 1e-9):,.0f} samples/s)")
    print(f"meta          : {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
