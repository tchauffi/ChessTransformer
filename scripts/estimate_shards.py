"""Estimate the Phase 1 flat-shard baseline before building anything.

Answers four questions with measurements on the real dataset, not guesses:

  1. How big is one record, for each candidate legal-mask layout?
  2. How much disk does the whole shard set take at K positions/game?
  3. How long does the one-shot build take (and can it be parallelised)?
  4. What loader throughput does a memmap-backed dataset actually reach?

Usage:
    uv run scripts/estimate_shards.py                       # default 400 games
    uv run scripts/estimate_shards.py --games 2000 --workers 12
"""

import argparse
import sys
import time
from pathlib import Path

import chess
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chesstransformer.datasets.h5_lichess_dataset import HDF5ChessDataset  # noqa: E402
from chesstransformer.models.tokenizer.alphazero_move_encoder import (  # noqa: E402
    NUM_ACTION_PLANES,
    move_to_action_plane,
)

# The ten keys the trainer actually reads (pos2move_v2_trainer.py:463-472).
# Everything else HDF5ChessDataset returns is dead weight.
SCALAR_FIELDS = [
    ("is_white", 1),
    ("castling_rights", 1),
    ("en_passant_file", 1),
    ("from_square", 1),
    ("action_plane", 1),
    ("result", 1),
    ("move_number", 2),
]
POSITION_BYTES = 64  # 64 squares, 13 token ids -> uint8


def measure_encoding(h5_path: str, n_games: int, seed: int = 0):
    """Replay n_games sampled positions through the current encode path.

    Returns per-sample legal-move counts and the wall-clock cost of the
    replay+encode work that build_shards.py will have to do once.
    """
    ds = HDF5ChessDataset(h5_path, cache_size=n_games)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=n_games, replace=False)

    legal_counts = []
    plies_replayed = 0
    t0 = time.perf_counter()
    for i in idxs:
        sample = ds[int(i)]
        legal_counts.append(int(sample["legal_moves_planes"].sum().item()))
        plies_replayed += int(sample["move_number"])
    elapsed = time.perf_counter() - t0

    return np.asarray(legal_counts), elapsed, plies_replayed


def measure_sequential_encoding(h5_path: str, n_games: int, seed: int = 0):
    """Split the build cost into its two components.

    build_shards.py should walk each game once and emit K samples from that
    single replay, instead of paying a fresh O(ply) replay per sample the way
    __getitem__ does. Cost then decomposes into:

      * replay: decode token + push, paid for *every* ply
      * encode: position tokens + legal-move planes, paid only at the K
        plies actually emitted

    Returns (plies, replay_seconds, encode_seconds) so the caller can model
    ``T = plies/replay_rate + samples/encode_rate`` for any K.
    """
    with h5py.File(h5_path, "r") as f:
        num_moves = f["num_moves"][:]
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(num_moves), size=n_games, replace=False)
        raw = [f["moves"][int(i)] for i in idxs]

    from chesstransformer.models.tokenizer.move_tokenizer import MoveTokenizer
    from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer

    mt, pt = MoveTokenizer(), PostionTokenizer()

    # Pass A: replay only.
    plies = 0
    boards = []
    t0 = time.perf_counter()
    for moves in raw:
        board = chess.Board()
        for tok in moves:
            try:
                mv = chess.Move.from_uci(mt.decode(int(tok)))
            except ValueError:
                break
            board.push(mv)
            plies += 1
        boards.append(board)
    replay_s = time.perf_counter() - t0

    # Pass B: encode only, on positions reached by replaying to a mid-game ply.
    encode_boards = []
    for moves in raw:
        board = chess.Board()
        stop = max(1, len(moves) // 2)
        for tok in moves[:stop]:
            try:
                board.push(chess.Move.from_uci(mt.decode(int(tok))))
            except ValueError:
                break
        encode_boards.append(board)

    t0 = time.perf_counter()
    for board in encode_boards:
        pt.encode(board)
        planes = np.zeros((64, NUM_ACTION_PLANES), dtype=bool)
        for lm in board.legal_moves:
            planes[lm.from_square, move_to_action_plane(lm.from_square, lm.to_square, lm.promotion)] = True
        np.packbits(planes)
    encode_s = time.perf_counter() - t0

    return plies, replay_s, len(encode_boards), encode_s


def record_layouts(legal_counts: np.ndarray):
    """Bytes per record under each candidate legal-mask encoding."""
    scalars = sum(w for _, w in SCALAR_FIELDS)
    base = POSITION_BYTES + scalars

    p999 = int(np.percentile(legal_counts, 99.9))
    layouts = {
        "raw bool (64,73)": 64 * NUM_ACTION_PLANES,
        "packbits (64,73)": int(np.ceil(64 * NUM_ACTION_PLANES / 8)),
        f"int16 index list, fixed K={p999}": 2 * p999 + 2,  # +2 for the count
        "int16 index list, mean (ragged)": 2 * float(legal_counts.mean()) + 2,
    }
    return base, layouts, p999


def measure_memmap_read(record_bytes: int, n_samples: int, batch_size: int, tmpdir: Path):
    """Throughput ceiling of an O(1) memmap __getitem__ + collate.

    Writes a synthetic shard of the target record size, then random-reads it
    the way FlatShardDataset would, to bound what Phase 1 can deliver per core.
    """
    tmpdir.mkdir(parents=True, exist_ok=True)
    path = tmpdir / f"synthetic_{record_bytes}b.bin"
    if not path.exists() or path.stat().st_size != record_bytes * n_samples:
        with open(path, "wb") as fh:
            chunk = np.zeros(record_bytes * 10_000, dtype=np.uint8)
            written = 0
            while written < record_bytes * n_samples:
                take = min(len(chunk), record_bytes * n_samples - written)
                fh.write(chunk[:take].tobytes())
                written += take

    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(n_samples, record_bytes))
    rng = np.random.default_rng(0)
    # Warm the page cache so we measure decode, not cold IO.
    _ = np.asarray(mm[rng.integers(0, n_samples, size=min(n_samples, 50_000))]).sum()

    n_batches = 200
    t0 = time.perf_counter()
    total = 0
    for _ in range(n_batches):
        idx = np.sort(rng.integers(0, n_samples, size=batch_size))
        batch = np.asarray(mm[idx])  # the actual per-sample work
        total += int(batch[0, 0])
        _ = np.unpackbits(batch[:, -584:], axis=1)  # mask unpack, the one real cost
    elapsed = time.perf_counter() - t0
    return batch_size * n_batches / elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/elite_db.h5")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seq-games", type=int, default=150)
    ap.add_argument("--workers", type=int, default=12, help="cores assumed available for the build")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--tmpdir", default="/home/tchauffi/.claude/jobs/760377e3/tmp/shard_estimate")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        n_games_total = f["moves"].shape[0]
        total_plies = int(f["num_moves"][:].astype(np.int64).sum())

    print("=" * 74)
    print("PHASE 1 SHARD SIZING ESTIMATE")
    print("=" * 74)
    print(f"source          : {args.h5}")
    print(f"games           : {n_games_total:,}")
    print(f"plies           : {total_plies:,}")
    print()

    print(f"[1/4] Measuring legal-move distribution over {args.games} sampled positions...")
    legal_counts, enc_elapsed, plies = measure_encoding(args.h5, args.games)
    print(f"      legal moves/pos : mean {legal_counts.mean():.1f}  "
          f"p50 {np.percentile(legal_counts, 50):.0f}  "
          f"p99 {np.percentile(legal_counts, 99):.0f}  "
          f"p99.9 {np.percentile(legal_counts, 99.9):.0f}  "
          f"max {legal_counts.max()}")
    print(f"      current __getitem__ path: {args.games / enc_elapsed:.0f} samples/s/core "
          f"({plies / args.games:.0f} plies replayed per sample)")
    print()

    print("[2/4] Record layouts (trainer-consumed fields only)")
    base, layouts, p999 = record_layouts(legal_counts)
    print(f"      position(64) + 7 scalars = {base} B")
    for name, mask_bytes in layouts.items():
        total = base + mask_bytes
        print(f"      {name:<36} mask {mask_bytes:>7.0f} B  ->  record {total:>7.0f} B")
    print()

    packed_record = base + layouts["packbits (64,73)"]
    idx_record = base + layouts[f"int16 index list, fixed K={p999}"]

    print("[3/4] Footprint and build cost")
    plies, replay_s, n_enc, encode_s = measure_sequential_encoding(args.h5, args.seq_games)
    replay_rate = plies / replay_s
    encode_rate = n_enc / encode_s
    print(f"      replay (decode+push) : {replay_rate:>8,.0f} plies/s/core   "
          f"[{plies:,} plies measured]")
    print(f"      encode (tokens+mask) : {encode_rate:>8,.0f} samples/s/core "
          f"[{n_enc:,} positions measured]")
    print()
    print(f"      {'K/game':>7} {'samples':>13} {'packed':>10} {'idx-list':>10} "
          f"{'build 1-core':>13} {'build ' + str(args.workers) + '-core':>14}")
    for k in (8, 12, 16):
        n = min(k * n_games_total, total_plies)
        gb_packed = n * packed_record / 1e9
        gb_idx = n * idx_record / 1e9
        # One full replay of every game, plus an encode at each emitted ply.
        build_1 = total_plies / replay_rate + n / encode_rate
        h1 = build_1 / 3600
        hw = h1 / args.workers
        print(f"      {k:>7} {n:>13,} {gb_packed:>9.1f}G {gb_idx:>9.1f}G "
              f"{h1:>12.1f}h {hw:>13.1f}h")
    print()

    print("[4/4] Loader ceiling with a memmap shard")
    n_syn = 1_000_000
    rate = measure_memmap_read(packed_record, n_syn, args.batch_size, Path(args.tmpdir))
    print(f"      packed record ({packed_record} B): {rate:,.0f} samples/s/core")
    print(f"      bytes per collated batch of {args.batch_size}: "
          f"{args.batch_size * packed_record / 1e6:.2f} MB "
          f"(current HDF5 path: 5.8 MB)")
    print()
    print("Targets from Phase 0: end-to-end > 5,400 samples/s, <= 4 workers, "
          "< 1.5 MB/batch")


if __name__ == "__main__":
    main()
