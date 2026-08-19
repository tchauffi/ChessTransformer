#!/usr/bin/env python3
"""Build a large, balanced opening book for engine matches.

Why this exists
---------------
At a fixed node budget this repo's MCTS is deterministic: no root noise, no
move temperature during matches (``move_temp=0.0``), and the docstring on
``head_to_head.py --workers`` notes the match is node-limited so CPU contention
cannot change which moves get played. Replaying an opening therefore reproduces
the *same game*, move for move. The hand-written books cap the two harnesses at
132 and 184 games, and no amount of re-running adds information.

Simulating this repo's SPRT against a 66%-draw match says what that ceiling
costs, in games needed to reach a decision on H0=0 vs H1=+15 Elo:

    true effect     median      p90
      +80 Elo        96        142      <- inside the old book
      +40 Elo       220        378
      +25 Elo       388        738
      +15 Elo       850       1876
        0 Elo       834       2102

So the old book could only ever resolve effects at or above roughly +80 Elo.
That is why the -89 Elo blitz regression was detected cleanly while every
+20..+50 Elo question in this project came back "not significant" -- those were
not weak results, they were unmeasurable ones. A few thousand distinct openings
removes the ceiling.

What makes a good book
----------------------
Two properties, both of which this script enforces:

* **Distinct.** Deduped by the FEN reached, so transpositions collapse rather
  than quietly replaying one line twice.
* **Balanced.** Openings are screened with Stockfish and kept only when the
  evaluation is near equal (``--max-cp``). A line that is already winning
  decides the game regardless of which engine plays it, which adds variance
  without adding information -- and it does so asymmetrically once colours are
  swapped, which is exactly the noise pentanomial pairing then has to cancel.

Usage
-----
    uv run python scripts/build_opening_book.py --out data/openings/book2k.json \
        --openings 2000 --plies 8 --max-cp 80 --workers 12
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import chess
import chess.engine
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.models.tokenizer.move_tokenizer import MoveTokenizer  # noqa: E402

SF_PATH = "/usr/games/stockfish"

# Set once per worker process: one engine per worker, as in gen_sf_policy_labels.py.
_ENGINE: chess.engine.SimpleEngine | None = None
_DEPTH = 12


def _worker_init(sf_path: str, depth: int) -> None:
    global _ENGINE, _DEPTH
    _DEPTH = depth
    _ENGINE = chess.engine.SimpleEngine.popen_uci(sf_path)
    _ENGINE.configure({"Threads": 1, "Hash": 64})


def _score_line(item: tuple[str, str]) -> tuple[str, str, int] | None:
    """Evaluate one candidate line. Returns (uci_line, fen, cp) or None."""
    uci_line, fen = item
    board = chess.Board(fen)
    try:
        info = _ENGINE.analyse(board, chess.engine.Limit(depth=_DEPTH))
    except chess.engine.EngineError:
        return None
    score = info["score"].white()
    if score.is_mate():
        return None
    return uci_line, fen, score.score()


def candidate_lines(h5_path: Path, n_candidates: int, plies: int,
                    min_elo: int, seed: int) -> list[tuple[str, str]]:
    """Sample distinct opening lines from the game database.

    Dedupes on the FEN *reached*, so two different move orders into the same
    position count once.
    """
    tok = MoveTokenizer()
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    with h5py.File(h5_path, "r") as f:
        moves, num_moves = f["moves"], f["num_moves"]
        white_elo, black_elo = f["white_elo"], f["black_elo"]
        total = len(moves)
        # Sample indices without replacement; games are cheap to reject.
        order = rng.sample(range(total), min(total, n_candidates * 12))

        for idx in order:
            if len(out) >= n_candidates:
                break
            if num_moves[idx] < plies + 10:
                continue  # too short to be a real game past the opening
            if min_elo and (white_elo[idx] < min_elo or black_elo[idx] < min_elo):
                continue

            board = chess.Board()
            ucis = []
            ok = True
            for token in moves[idx][:plies]:
                uci = tok.decode(int(token))
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    ok = False
                    break
                board.push(move)
                ucis.append(uci)
            if not ok or len(ucis) < plies or board.is_game_over():
                continue

            fen = board.fen()
            key = " ".join(fen.split()[:4])  # ignore clocks for dedup
            if key in seen:
                continue
            seen.add(key)
            out.append((" ".join(ucis), fen))

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", type=Path, default=Path("data/elite_db.h5"))
    p.add_argument("--out", type=Path, required=True, help="destination .json")
    p.add_argument("--openings", type=int, default=2000,
                   help="balanced openings to keep. Each gives 2 games in a match.")
    p.add_argument("--plies", type=int, default=8,
                   help="opening depth. Deeper is more diverse but drifts further "
                        "from equality, so more lines get rejected by --max-cp.")
    p.add_argument("--max-cp", type=int, default=80,
                   help="keep lines whose Stockfish eval is within this many "
                        "centipawns of equal. 0 disables the balance screen.")
    p.add_argument("--depth", type=int, default=12, help="Stockfish screening depth")
    p.add_argument("--min-elo", type=int, default=2200,
                   help="require both players at least this rating (0 = off)")
    p.add_argument("--workers", type=int, default=min(12, mp.cpu_count()))
    p.add_argument("--sf-path", default=SF_PATH)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.h5.exists():
        p.error(f"{args.h5} not found")

    # Oversample: the balance screen rejects a large fraction at typical --max-cp.
    want = args.openings if args.max_cp <= 0 else int(args.openings * 3.5)
    print(f"sampling {want} candidate lines at {args.plies} plies "
          f"(min_elo={args.min_elo}) ...", flush=True)
    cands = candidate_lines(args.h5, want, args.plies, args.min_elo, args.seed)
    print(f"  {len(cands)} distinct lines", flush=True)

    if args.max_cp <= 0:
        kept = [{"uci": u, "fen": f, "cp": None} for u, f in cands[: args.openings]]
    else:
        print(f"screening with Stockfish depth {args.depth} on {args.workers} "
              f"workers, keeping |cp| <= {args.max_cp} ...", flush=True)
        kept = []
        with mp.Pool(args.workers, initializer=_worker_init,
                     initargs=(args.sf_path, args.depth)) as pool:
            for i, res in enumerate(pool.imap_unordered(_score_line, cands, chunksize=16)):
                if i % 500 == 0:
                    print(f"  {i}/{len(cands)} screened, {len(kept)} kept", flush=True)
                if res is None:
                    continue
                uci, fen, cp = res
                if abs(cp) <= args.max_cp:
                    kept.append({"uci": uci, "fen": fen, "cp": cp})
                if len(kept) >= args.openings:
                    break

    if not kept:
        print("no openings survived the screen", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "openings": kept,
        "meta": {
            "source": str(args.h5), "plies": args.plies, "max_cp": args.max_cp,
            "depth": args.depth, "min_elo": args.min_elo, "seed": args.seed,
            "count": len(kept),
        },
    }
    args.out.write_text(json.dumps(payload, indent=1))
    cps = [k["cp"] for k in kept if k["cp"] is not None]
    print(f"\nwrote {len(kept)} openings to {args.out}")
    if cps:
        print(f"  cp: mean {sum(cps)/len(cps):+.1f}, "
              f"|cp| max {max(abs(c) for c in cps)}")
    print(f"  supports matches up to {2 * len(kept)} games")
    return 0


if __name__ == "__main__":
    sys.exit(main())
