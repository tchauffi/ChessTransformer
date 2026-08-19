"""Engine-vs-engine match between two ct-bot UCI configs (model and/or node
budget). Both engines are deterministic (move_temp=0), so we vary the start via
an opening book and play each line twice with colours swapped. Reports W/D/L and
an approximate Elo difference (config A relative to config B).

Usage
-----
    # default: base model against itself at two node budgets (the original A/B)
    uv run python scripts/head_to_head.py

    # candidate vs released baseline, equal budget, full opening book
    uv run python scripts/head_to_head.py \
        --model-a /path/to/candidate/model.int8.onnx --label-a run_007 \
        --nodes-a 1800 --nodes-b 1800 --openings 32

Results are scored **pentanomially** -- the two colour-swapped games of one
opening are one sample -- and can be stopped early by an SPRT (``--sprt``).

Because both engines are deterministic, replaying an opening reproduces the same
game exactly, so the size of the book is a hard ceiling on how much can ever be
learned from a match. The built-in book below caps out at 132 games, which
simulation puts at roughly the +80 Elo resolution mark; resolving +15 Elo needs
around 850 games (p90 ~1900). Use ``--book`` with a book from
``scripts/build_opening_book.py`` for anything smaller than a large effect.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.evaluation.sprt import (  # noqa: E402
    format_report,
    pair_up,
    sprt_decision,
    sprt_llr,
    summarize,
)

BASE = "data/models/pos2move_v2.1/model.int8.onnx"  # base (non-EMA) int8

# Ordered so that the first 8 reproduce the original hardcoded book. Each line is
# played twice (colours swapped), so games = 2 * --openings.
OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6",   # open game
    "e2e4 c7c5",             # Sicilian
    "d2d4 d7d5 c2c4",        # QGD
    "d2d4 g8f6 c2c4 g7g6",   # KID
    "e2e4 e7e6",             # French
    "c2c4 e7e5",             # English
    "g1f3 d7d5 g2g3",        # Reti
    "e2e4 c7c6",             # Caro-Kann
    "e2e4 d7d5",             # Scandinavian
    "e2e4 g8f6",             # Alekhine
    "e2e4 d7d6",             # Pirc
    "e2e4 g7g6",             # Modern
    "d2d4 g8f6 c2c4 e7e6",   # Nimzo/QID complex
    "d2d4 g8f6 c2c4 c7c5",   # Benoni
    "d2d4 d7d5 g1f3 g8f6",   # symmetrical queen's pawn
    "d2d4 f7f5",             # Dutch
    "d2d4 d7d5 c2c4 c7c6",   # Slav
    "d2d4 d7d5 c2c4 d5c4",   # QGA
    "e2e4 e7e5 g1f3 g8f6",   # Petrov
    "e2e4 e7e5 f1c4",        # Bishop's opening
    "e2e4 e7e5 b1c3",        # Vienna
    "e2e4 c7c5 g1f3 d7d6",   # Sicilian, ...d6
    "e2e4 c7c5 b1c3",        # closed Sicilian
    "c2c4 g8f6",             # English vs Nf6
    "c2c4 c7c5",             # symmetrical English
    "g1f3 g8f6 c2c4",        # flexible
    "d2d4 e7e6",             # Franco-Indian
    "e2e4 e7e5 g1f3 b8c6 f1b5",  # Ruy Lopez
    "e2e4 e7e5 g1f3 b8c6 f1c4",  # Italian
    "d2d4 g8f6 g1f3 e7e6",   # colourless Indian
    "b2b3",                  # Larsen
    "g2g3",                  # King's fianchetto
    # --- appended 2026-08-18 to double the book for tighter Elo intervals. The engines are
    # deterministic, so replaying the same line adds no information -- more *distinct* lines
    # is the only way to buy games. Appended, never reordered: indices 0-31 above are
    # unchanged, so any earlier --openings N <= 32 result stays reproducible.
    "e2e4 c7c5 g1f3 b8c6",   # Sicilian, ...Nc6
    "e2e4 c7c5 g1f3 e7e6",   # Sicilian, ...e6
    "e2e4 c7c5 g1f3 g7g6",   # accelerated fianchetto
    "e2e4 c7c5 c2c3",        # Alapin
    "e2e4 c7c5 d2d4",        # open Sicilian gambit
    "e2e4 e7e5 f2f4",        # King's Gambit
    "e2e4 e7e5 d2d4",        # Centre game
    "e2e4 e7e5 g1f3 b8c6 d2d4",  # Scotch
    "e2e4 e7e5 g1f3 b8c6 b1c3",  # Four Knights
    "e2e4 e7e5 g1f3 d7d6",   # Philidor
    "e2e4 b8c6",             # Nimzowitsch
    "e2e4 b7b6",             # Owen
    "d2d4 g8f6 c2c4 e7e6 b1c3",  # Nimzo-Indian complex
    "d2d4 g8f6 c2c4 e7e6 g1f3",  # Queen's Indian complex
    "d2d4 g8f6 c1g5",        # Trompowsky
    "d2d4 g8f6 g1f3 g7g6",   # Indian, ...g6
    "d2d4 d7d5 c1f4",        # London
    "d2d4 d7d5 e2e3",        # Colle
    "d2d4 d7d5 g1f3 c7c5",   # symmetrical, ...c5
    "d2d4 b7b6",             # English defence
    "d2d4 g7g6",             # Modern vs d4
    "c2c4 e7e6",             # English/QGD
    "c2c4 c7c6",             # Caro-English
    "c2c4 g7g6",             # English fianchetto
    "g1f3 d7d5 d2d4",        # Reti into d4
    "g1f3 c7c5",             # Reti vs ...c5
    "g1f3 g8f6 g2g3",        # double fianchetto
    "e2e3",                  # Van 't Kruijs
    "f2f4",                  # Bird
    "b2b4",                  # Sokolsky
    "d2d3",                  # Mieses
    "e2e4 g8f6 e4e5",        # Alekhine, main
]


def play(white, white_nodes, black, black_nodes, opening: str, max_plies: int) -> str:
    board = chess.Board()
    for uci in opening.split():
        board.push(chess.Move.from_uci(uci))
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        eng, nodes = (white, white_nodes) if board.turn == chess.WHITE else (black, black_nodes)
        board.push(eng.play(board, chess.engine.Limit(nodes=nodes)).move)
    return board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bot", default="rust/target/release/ct-bot")
    p.add_argument("--model-a", default=BASE, help="ONNX model for side A")
    p.add_argument("--model-b", default=BASE, help="ONNX model for side B")
    p.add_argument("--nodes-a", type=int, default=3600, help="MCTS node budget, side A")
    p.add_argument("--nodes-b", type=int, default=1800, help="MCTS node budget, side B")
    p.add_argument("--label-a", default=None, help="display name for side A")
    p.add_argument("--label-b", default=None, help="display name for side B")
    p.add_argument("--openings", type=int, default=8,
                   help="opening lines to use, from the front of the book "
                        f"(built-in book: {len(OPENINGS)}). Each is played twice, "
                        "colours swapped.")
    p.add_argument("--book", type=Path, default=None,
                   help="JSON book from scripts/build_opening_book.py. The built-in "
                        "book only supports 132 games, which cannot resolve anything "
                        "under roughly +80 Elo.")
    p.add_argument("--sprt", action="store_true",
                   help="stop as soon as the match decides H0 vs H1 instead of "
                        "playing every opening.")
    p.add_argument("--elo0", type=float, default=0.0, help="SPRT null hypothesis")
    p.add_argument("--elo1", type=float, default=15.0,
                   help="SPRT alternative: the smallest gain worth promoting")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--min-pairs", type=int, default=20,
                   help="never stop before this many completed pairs; the normal "
                        "approximation is unreliable on a handful of samples.")
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--workers", type=int, default=1,
                   help="game-pairs to play concurrently, each with its own engine pair. "
                        "Safe to raise: the match is node-limited, not time-limited, so CPU "
                        "contention changes wall-clock but not which moves get played.")
    p.add_argument("--threads", type=int, default=1,
                   help="ONNX Runtime intra-op threads per engine (0 = auto). Keep at 1 and "
                        "raise --workers instead: intra-op scaling is sublinear (~2.4x for 4 "
                        "threads), so one thread per game gives the best total throughput, and "
                        "'auto' has every engine grab the whole machine.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="inference device for both engines. 'cuda' needs a ct-bot built with "
                        "--features cuda plus ORT_DYLIB_PATH/LD_LIBRARY_PATH pointing at an "
                        "onnxruntime-gpu that has kernels for this GPU -- and fp32 .onnx models, "
                        "not the int8 ones.")
    for side in ("a", "b"):
        p.add_argument(f"--cpuct-{side}", type=float, default=None,
                       help=f"PUCT exploration constant for side {side.upper()} "
                            "(ct-bot default 1.0). Left unset the engine default applies.")
        p.add_argument(f"--fpu-{side}", type=float, default=None,
                       help=f"first-play-urgency penalty for side {side.upper()} (default 0.2)")
        p.add_argument(f"--prior-temp-{side}", type=float, default=None,
                       help=f"policy-prior softmax temperature for side {side.upper()} (default 1.0)")
    p.add_argument("--sim-batch", type=int, default=None,
                   help="leaves per NN wave (ct-bot default 16). Larger batches feed the GPU "
                        "better but change search behaviour -- more collisions per wave -- so "
                        "leave unset when the match is meant to mirror production search.")
    args = p.parse_args()
    args.lines = load_book(args.book)
    if not 1 <= args.openings <= len(args.lines):
        p.error(f"--openings must be in 1..{len(args.lines)}"
                + ("" if args.book else "; pass --book for a larger book"))
    args.label_a = args.label_a or f"A@{args.nodes_a}"
    args.label_b = args.label_b or f"B@{args.nodes_b}"
    return args


def load_book(path: Path | None) -> list[str]:
    """UCI opening lines, from a generated book or the built-in list."""
    if path is None:
        return list(OPENINGS)
    payload = json.loads(path.read_text())
    return [o["uci"] for o in payload["openings"]]


def main() -> None:
    args = parse_args()
    la, lb = args.label_a, args.label_b
    common = ["--threads", str(args.threads), "--device", args.device]
    if args.sim_batch is not None:
        common += ["--sim-batch", str(args.sim_batch)]

    def search_flags(cpuct, fpu, prior_temp):
        """Per-side PUCT knobs, so a tuned config can be matched against an untuned one."""
        out = []
        for flag, val in (("--c-puct", cpuct), ("--fpu", fpu), ("--prior-temp", prior_temp)):
            if val is not None:
                out += [flag, str(val)]
        return out

    A = (la, [args.bot, "uci", "--model", args.model_a, *common,
              *search_flags(args.cpuct_a, args.fpu_a, args.prior_temp_a)], args.nodes_a)
    B = (lb, [args.bot, "uci", "--model", args.model_b, *common,
              *search_flags(args.cpuct_b, args.fpu_b, args.prior_temp_b)], args.nodes_b)
    book = args.lines[: args.openings]

    jobs = [(i, op, a_white) for i, op in enumerate(book) for a_white in (True, False)]
    workers = max(1, min(args.workers, len(jobs)))

    print(f"{la}: {args.model_a} @ {args.nodes_a} nodes")
    print(f"{lb}: {args.model_b} @ {args.nodes_b} nodes")
    print(f"book: {args.book or 'built-in'} ({len(args.lines)} lines available)")
    print(f"{len(book)} openings x 2 colours = {len(jobs)} games, {workers} worker(s)")
    if args.sprt:
        print(f"SPRT H0={args.elo0:+.0f} vs H1={args.elo1:+.0f} Elo, "
              f"alpha={args.alpha}, beta={args.beta}, min {args.min_pairs} pairs")
    print(flush=True)

    lock = threading.Lock()
    done = [0]
    # Per-game scores keyed by (opening index, A played white). A pair is only
    # scored once both colours of that opening are in, so an early stop or an
    # interrupted match never contributes a one-sided opening.
    results: dict[tuple[int, bool], float] = {}
    pairs: list[float] = []
    stop = threading.Event()
    verdict = [None]

    def run_shard(shard):
        """Play a slice of the job list with a private pair of engine processes."""
        ea = chess.engine.SimpleEngine.popen_uci(A[1])
        eb = chess.engine.SimpleEngine.popen_uci(B[1])
        out = []
        try:
            for idx, op, a_white in shard:
                if stop.is_set():
                    break
                if a_white:
                    white, wn, black, bn = ea, A[2], eb, B[2]
                else:
                    white, wn, black, bn = eb, B[2], ea, A[2]
                r = play(white, wn, black, bn, op, args.max_plies)
                s = 0.5 if r == "1/2-1/2" else (1.0 if (r == "1-0") == a_white else 0.0)
                tag = "=" if s == 0.5 else ("+" if s == 1.0 else "-")
                out.append(s)
                with lock:
                    done[0] += 1
                    results[(idx, a_white)] = s
                    other = results.get((idx, not a_white))
                    note = ""
                    if other is not None:
                        pairs.append(s + other)
                        if args.sprt:
                            llr = sprt_llr(pairs, args.elo0, args.elo1)
                            note = f"  LLR {llr:+.2f}"
                            decision = sprt_decision(
                                llr, args.alpha, args.beta,
                                min_pairs_met=len(pairs) >= args.min_pairs)
                            if decision:
                                verdict[0] = decision
                                stop.set()
                                note += f"  -> {decision}, stopping"
                    print(f"[{done[0]:4}/{len(jobs)}] {la if a_white else lb:10}=W  "
                          f"{op[:11]:11} -> {r:7} ({la} {tag}){note}", flush=True)
        finally:
            ea.quit()
            eb.quit()
        return out

    shards = [jobs[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = [s for shard in pool.map(run_shard, shards) for s in shard]

    # Score the completed pairs, not the raw games: the two colour-swapped games
    # of one opening share that opening's difficulty, so pairing removes it from
    # the variance instead of letting it inflate the interval.
    final_pairs = pair_up(results)
    stats = summarize(final_pairs, scores)
    llr = sprt_llr(final_pairs, args.elo0, args.elo1) if args.sprt else None

    print()
    if len(scores) > stats.n_games:
        print(f"note: {len(scores) - stats.n_games} game(s) had no colour-swapped "
              f"partner and were dropped from the estimate")
    print(format_report(stats, la, lb, llr, args.elo0, args.elo1,
                        args.alpha, args.beta))


if __name__ == "__main__":
    sys.exit(main())
