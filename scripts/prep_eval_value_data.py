"""Build a value-training set from the Lichess Stockfish evaluation dataset.

Downloads parquet shards of `Lichess/chess-position-evaluations` (CC0, 388M
positions, columns: fen / line / depth / knodes / cp / mate), samples and
dedupes them, encodes each FEN exactly like `Pos2MoveV2Bot._encode_position`,
and converts the Stockfish score to a **side-to-move-POV value in [-1, 1]** for
the value head. Output is a single `.npz` consumed by
`scripts/train_value_head.py`.

The cp/mate point-of-view is **auto-detected** (the HF card does not state it):
we correlate cp against white-POV and stm-POV material balance and pick the
convention that lines up. Getting this wrong trains the value head inverted, so
the detection is printed loudly and can be overridden with --cp-pov.

Usage (deps are not in pyproject — pull them ephemerally):
    uv run --with pyarrow,huggingface_hub python scripts/prep_eval_value_data.py \
        --shards 2 --max-positions 3000000 --out data/eval/lichess-sf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chess  # noqa: E402
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer  # noqa: E402

REPO_ID = "Lichess/chess-position-evaluations"
PARQUET_REVISION = "refs/convert/parquet"
PARQUET_FMT = "default/train/{:04d}.parquet"

# Lichess win%/eval curve: P(win) = 1/(1+exp(-k*cp)); value = 2*P - 1.
CP_K = 0.00368208
# Standard piece values for the cp-sign material-correlation test (king omitted).
_PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def download_shards(n: int, cache_dir: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download
    paths = []
    for i in range(n):
        print(f"  downloading shard {i} ...", flush=True)
        p = hf_hub_download(REPO_ID, PARQUET_FMT.format(i), repo_type="dataset",
                            revision=PARQUET_REVISION, cache_dir=str(cache_dir))
        paths.append(Path(p))
    return paths


def sample_rows(shards: list[Path], min_depth: int, target: int, seed: int):
    """Stream parquet row-batches, depth-filter, Bernoulli-subsample to ~target.

    Returns parallel numpy arrays (fen, cp, mate, depth). Memory stays bounded
    because we keep only a sampled fraction, never a whole shard in RAM.
    """
    import pyarrow.parquet as pq

    total = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    keep_p = min(1.0, target / max(1, total))
    print(f"  {total:,} rows across {len(shards)} shard(s); keep_p={keep_p:.4g}")
    rng = np.random.default_rng(seed)

    fens, cps, mates, depths = [], [], [], []
    for s in shards:
        pf = pq.ParquetFile(s)
        for batch in pf.iter_batches(batch_size=131072, columns=["fen", "cp", "mate", "depth"]):
            depth = np.asarray(batch.column("depth"), dtype=np.float32)
            mask = depth >= min_depth
            if keep_p < 1.0:
                mask &= rng.random(len(depth)) < keep_p
            if not mask.any():
                continue
            idx = np.nonzero(mask)[0]
            fen = np.asarray(batch.column("fen").to_pylist(), dtype=object)[idx]
            cp = batch.column("cp").to_numpy(zero_copy_only=False).astype(np.float32)[idx]
            mate = batch.column("mate").to_numpy(zero_copy_only=False).astype(np.float32)[idx]
            fens.append(fen); cps.append(cp); mates.append(mate); depths.append(depth[idx])
    if not fens:
        raise RuntimeError("no rows survived the depth filter")
    return (np.concatenate(fens), np.concatenate(cps),
            np.concatenate(mates), np.concatenate(depths))


def dedupe_max_depth(fen, cp, mate, depth):
    """Keep one row per FEN — the deepest analysis."""
    order = np.argsort(depth, kind="stable")[::-1]  # deepest first
    fen, cp, mate, depth = fen[order], cp[order], mate[order], depth[order]
    _, first = np.unique(fen, return_index=True)     # first occ = deepest
    return fen[first], cp[first], mate[first], depth[first]


def encode_fens(fens):
    """Mirror Pos2MoveV2Bot._encode_position; also return material balance.

    Returns boards (N,64), player, castling, ep, white_balance, valid mask.
    """
    tok = PostionTokenizer()
    n = len(fens)
    boards = np.zeros((n, 64), dtype=np.int64)
    player = np.zeros(n, dtype=np.int64)
    castling = np.zeros(n, dtype=np.int64)
    ep = np.full(n, 8, dtype=np.int64)
    white_bal = np.zeros(n, dtype=np.float32)
    valid = np.ones(n, dtype=bool)

    for i, fen in enumerate(fens):
        try:
            board = chess.Board(fen)
        except ValueError:
            valid[i] = False
            continue
        boards[i] = tok.encode(board)
        player[i] = int(board.turn)
        c = 0
        if board.has_kingside_castling_rights(chess.WHITE):  c |= 1
        if board.has_queenside_castling_rights(chess.WHITE): c |= 2
        if board.has_kingside_castling_rights(chess.BLACK):  c |= 4
        if board.has_queenside_castling_rights(chess.BLACK): c |= 8
        castling[i] = c
        ep[i] = chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8
        bal = 0
        for piece in board.piece_map().values():
            v = _PIECE_VALUE[piece.piece_type]
            bal += v if piece.color == chess.WHITE else -v
        white_bal[i] = bal
        if (i + 1) % 200000 == 0:
            print(f"  encoded {i + 1:,}/{n:,}", end="\r")
    print(f"  encoded {n:,}/{n:,}        ")
    return boards, player, castling, ep, white_bal, valid


def detect_cp_pov(cp, player, white_bal):
    """Decide whether cp is White-POV or side-to-move-POV via material corr."""
    finite = np.isfinite(cp) & (np.abs(white_bal) >= 3)  # decisive material only
    if finite.sum() < 1000:
        raise RuntimeError("too few decisive-material positions to detect cp POV")
    cpf = cp[finite]
    white = white_bal[finite]
    stm = np.where(player[finite] == 1, white, -white)
    corr_white = np.corrcoef(cpf, white)[0, 1]
    corr_stm = np.corrcoef(cpf, stm)[0, 1]
    pov = "white" if corr_white >= corr_stm else "stm"
    print(f"  cp-POV detection: corr(cp, white_balance)={corr_white:+.3f} | "
          f"corr(cp, stm_balance)={corr_stm:+.3f} -> '{pov}'")
    if max(corr_white, corr_stm) < 0.2:
        print("  WARNING: weak correlation; cp-POV detection is unreliable.")
    return pov


def to_stm_value(cp, mate, player, pov):
    """cp/mate (in the detected POV) -> stm-POV value in [-1, 1]."""
    to_white = player == 1
    if pov == "white":
        flip = np.where(to_white, 1.0, -1.0)  # white-POV -> stm-POV
    else:
        flip = np.ones_like(cp)               # already stm-POV
    cp_stm = cp * flip
    mate_stm = mate * flip

    value = np.full(len(cp), np.nan, dtype=np.float32)
    has_mate = np.isfinite(mate)
    value[has_mate] = np.sign(mate_stm[has_mate]).astype(np.float32)
    has_cp = np.isfinite(cp) & ~has_mate
    value[has_cp] = (2.0 / (1.0 + np.exp(-CP_K * cp_stm[has_cp])) - 1.0).astype(np.float32)
    is_mate = has_mate
    decisive = has_mate | (np.isfinite(cp) & (np.abs(cp) >= 100))
    return value, is_mate, decisive


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", type=int, default=2, help="number of parquet shards (of 20) to use")
    p.add_argument("--out", default="data/eval/lichess-sf")
    p.add_argument("--min-depth", type=int, default=12)
    p.add_argument("--max-positions", type=int, default=3_000_000)
    p.add_argument("--oversample", type=float, default=2.0,
                   help="sample this multiple of max-positions before dedup/trim")
    p.add_argument("--cp-pov", choices=["auto", "white", "stm"], default="auto")
    p.add_argument("--cache-dir", default="data/eval/_hf_cache")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("Downloading shards ...")
    shards = download_shards(args.shards, Path(args.cache_dir))

    print("Sampling rows ...")
    fen, cp, mate, depth = sample_rows(
        shards, args.min_depth, int(args.max_positions * args.oversample), args.seed)
    print(f"  sampled {len(fen):,} eligible rows")

    print("Deduping by FEN (max depth) ...")
    fen, cp, mate, depth = dedupe_max_depth(fen, cp, mate, depth)
    print(f"  {len(fen):,} unique positions")

    if len(fen) > args.max_positions:
        sel = np.random.default_rng(args.seed).choice(len(fen), args.max_positions, replace=False)
        fen, cp, mate, depth = fen[sel], cp[sel], mate[sel], depth[sel]
        print(f"  trimmed to {len(fen):,}")

    print("Encoding FENs ...")
    boards, player, castling, ep, white_bal, valid = encode_fens(fen)
    if not valid.all():
        print(f"  dropping {int((~valid).sum()):,} unparseable FEN(s)")
        boards, player, castling, ep, white_bal = (
            boards[valid], player[valid], castling[valid], ep[valid], white_bal[valid])
        cp, mate, depth = cp[valid], mate[valid], depth[valid]

    pov = detect_cp_pov(cp, player, white_bal) if args.cp_pov == "auto" else args.cp_pov
    if args.cp_pov != "auto":
        print(f"  cp-POV forced to '{pov}'")

    value, is_mate, decisive = to_stm_value(cp, mate, player, pov)
    keep = np.isfinite(value)
    if not keep.all():
        print(f"  dropping {int((~keep).sum()):,} position(s) with no usable eval")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "positions_eval.npz",
        boards=boards[keep], player=player[keep], castling=castling[keep], ep=ep[keep],
        value=value[keep], cp=cp[keep], is_mate=is_mate[keep], decisive=decisive[keep],
        depth=depth[keep].astype(np.int16))
    meta = {"repo": REPO_ID, "shards": args.shards, "min_depth": args.min_depth,
            "positions": int(keep.sum()), "cp_pov": pov, "cp_k": CP_K,
            "decisive_frac": float(decisive[keep].mean()), "mate_frac": float(is_mate[keep].mean())}
    (out / "prep_meta.json").write_text(__import__("json").dumps(meta, indent=2))
    print(f"\nSaved {int(keep.sum()):,} positions to {out}/positions_eval.npz")
    print(f"  value range [{value[keep].min():.3f}, {value[keep].max():.3f}] | "
          f"mean {value[keep].mean():+.3f} | decisive {decisive[keep].mean():.1%} | "
          f"mate {is_mate[keep].mean():.1%}")


if __name__ == "__main__":
    main()
