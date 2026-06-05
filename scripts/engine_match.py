"""Deterministic engine-vs-engine match for A/B-testing Pos2MoveV2Bot configs.

Both engines are (essentially) deterministic, so variety comes from a fixed
suite of opening positions played from both colours. Useful for validating a
strength change (e.g. quiescence on/off, EMA vs base) much faster and with less
variance than a Stockfish gauntlet.

Usage
-----
    # Quiescence A (qdepth=4) vs B (qdepth=0), same weights:
    uv run python scripts/engine_match.py --a-quiescence 4 --b-quiescence 0 --depth 3

    # EMA vs base weights:
    uv run python scripts/engine_match.py --a-ema --b-quiescence 4 --depth 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.bots.pos2move_v2_bot import Pos2MoveV2Bot
from chesstransformer.bots.pos2move_v2_mcts_bot import Pos2MoveV2MctsBot

# Common openings a few moves in, to diversify otherwise-deterministic play.
OPENINGS: list[str] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",          # start
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",     # Sicilian
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",     # QP
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",     # QGambit
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",      # Indian
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2",  # open game
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",     # KID/Gruen
    "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",     # Slav
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",     # QGD
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Sicilian Nc6
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",     # Nimzo/QID
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",     # Scandi
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", # Ruy Lopez
    "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 1 3",  # QGD/Indian
    "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",    # KID
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", # Italian
    "rnbqkbnr/1ppppppp/p7/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",       # 1.e4 a6
    "rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",      # Modern
    "rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 2",     # 1.e4 f5
    "rnbqkbnr/pp1ppppp/8/2p5/3P4/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 2",     # 1.d4 c5
    "rnbqkb1r/pppppppp/5n2/8/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 2 2",    # 1.d4 Nf6 2.Nf3
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 2",    # closed Sicilian
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/2N5/PPPP1PPP/R1BQK1NR w KQkq - 4 4", # 4 knights-ish
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR b KQkq - 2 3",  # London
]


def play(white: Pos2MoveV2Bot, black: Pos2MoveV2Bot, start_fen: str, max_moves: int):
    board = chess.Board(start_fen)
    nodes = 0
    move_time = 0.0
    n_moves = 0
    while board.fullmove_number <= max_moves:
        if not any(board.legal_moves) or board.is_insufficient_material() \
                or board.is_repetition(3) or board.halfmove_clock >= 100:
            break
        bot = white if board.turn == chess.WHITE else black
        t0 = time.perf_counter()
        mv, _ = bot.predict(board)
        move_time += time.perf_counter() - t0
        nodes += bot.nodes_searched
        n_moves += 1
        board.push_uci(mv)

    if not any(board.legal_moves) and board.is_check():
        result = "0-1" if board.turn == chess.WHITE else "1-0"
    else:
        result = "1/2-1/2"
    return result, n_moves, nodes, move_time


def build_bot(args, ema, quiescence, depth, mcts, sims, model_dir,
              cpuct=1.5, prior_temp=1.0, fpu=None, tree_reuse=True):
    if mcts:
        return Pos2MoveV2MctsBot(
            model_dir=model_dir,
            num_simulations=sims,
            sim_batch=args.sim_batch,
            c_puct=cpuct,
            prior_temp=prior_temp,
            fpu=fpu,
            tree_reuse=tree_reuse,
            time_limit=args.time_limit,
            use_ema=ema,
            compile=not args.no_compile,
        )
    return Pos2MoveV2Bot(
        model_dir=model_dir,
        depth=depth,
        top_p=args.top_p,
        time_limit=args.time_limit,
        use_ema=ema,
        quiescence_depth=quiescence,
        compile=not args.no_compile,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir", nargs="?",
                   default=str(Path(__file__).resolve().parents[1] / "data" / "models" / "pos2move_v2"))
    p.add_argument("--depth", type=int, default=3, help="Shared depth if --a-depth/--b-depth unset")
    p.add_argument("--a-depth", type=int, default=None)
    p.add_argument("--b-depth", type=int, default=None)
    p.add_argument("--top-p", type=float, default=0.90)
    p.add_argument("--time-limit", type=float, default=0.0, help="0 = fixed depth (deterministic)")
    p.add_argument("--max-moves", type=int, default=120)
    p.add_argument("--a-ema", action="store_true")
    p.add_argument("--b-ema", action="store_true")
    p.add_argument("--a-quiescence", type=int, default=4)
    p.add_argument("--b-quiescence", type=int, default=0)
    p.add_argument("--a-mcts", action="store_true", help="Engine A uses MCTS/PUCT")
    p.add_argument("--b-mcts", action="store_true", help="Engine B uses MCTS/PUCT")
    p.add_argument("--a-sims", type=int, default=200)
    p.add_argument("--b-sims", type=int, default=200)
    p.add_argument("--sim-batch", type=int, default=16, help="MCTS leaves per batch")
    p.add_argument("--a-cpuct", type=float, default=1.5)
    p.add_argument("--b-cpuct", type=float, default=1.5)
    p.add_argument("--a-prior-temp", type=float, default=1.0)
    p.add_argument("--b-prior-temp", type=float, default=1.0)
    p.add_argument("--a-fpu", type=float, default=None)
    p.add_argument("--b-fpu", type=float, default=None)
    p.add_argument("--a-no-reuse", action="store_true", help="disable tree reuse for engine A")
    p.add_argument("--b-no-reuse", action="store_true", help="disable tree reuse for engine B")
    p.add_argument("--a-model-dir", default=None, help="Override model dir for engine A")
    p.add_argument("--b-model-dir", default=None, help="Override model dir for engine B")
    p.add_argument("--openings", type=int, default=len(OPENINGS), help="How many openings to use")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    a_depth = args.a_depth if args.a_depth is not None else args.depth
    b_depth = args.b_depth if args.b_depth is not None else args.depth
    a_model = args.a_model_dir or args.model_dir
    b_model = args.b_model_dir or args.model_dir
    a_desc = (f"MCTS sims={args.a_sims} cpuct={args.a_cpuct} ptemp={args.a_prior_temp} fpu={args.a_fpu}"
              if args.a_mcts else f"depth={a_depth} quiescence={args.a_quiescence}")
    b_desc = (f"MCTS sims={args.b_sims} cpuct={args.b_cpuct} ptemp={args.b_prior_temp} fpu={args.b_fpu}"
              if args.b_mcts else f"depth={b_depth} quiescence={args.b_quiescence}")
    print(f"Engine A: {a_desc} ema={args.a_ema} model={a_model}")
    print(f"Engine B: {b_desc} ema={args.b_ema} model={b_model}")
    bot_a = build_bot(args, args.a_ema, args.a_quiescence, a_depth, args.a_mcts, args.a_sims, a_model,
                      args.a_cpuct, args.a_prior_temp, args.a_fpu, not args.a_no_reuse)
    bot_b = build_bot(args, args.b_ema, args.b_quiescence, b_depth, args.b_mcts, args.b_sims, b_model,
                      args.b_cpuct, args.b_prior_temp, args.b_fpu, not args.b_no_reuse)

    openings = OPENINGS[: args.openings]
    a_score = 0.0
    a_w = a_d = a_l = 0
    a_nodes = a_time = 0.0
    a_n_moves = 0
    games = 0

    for fen in openings:
        # Game 1: A=White, B=Black ; Game 2: A=Black, B=White
        for a_is_white in (True, False):
            white, black = (bot_a, bot_b) if a_is_white else (bot_b, bot_a)
            result, n_moves, nodes, mtime = play(white, black, fen, args.max_moves)
            games += 1
            if result == "1/2-1/2":
                sc = 0.5
            elif (result == "1-0") == a_is_white:
                sc = 1.0
            else:
                sc = 0.0
            a_score += sc
            a_w += sc == 1.0
            a_d += sc == 0.5
            a_l += sc == 0.0
            # Track A's own move cost (half the moves are A's, roughly).
            a_n_moves += n_moves
            a_nodes += nodes
            a_time += mtime

    print("\n" + "=" * 56)
    print(f"MATCH RESULT (engine A perspective), {games} games")
    print("=" * 56)
    print(f"  A: +{a_w} ={a_d} -{a_l}   score {a_score}/{games} = {a_score / games:.1%}")
    print(f"  total move time: {a_time:.1f}s over {a_n_moves} half-moves "
          f"({a_time / max(a_n_moves,1) * 1000:.0f} ms/move, {a_nodes / max(a_time,1e-9):.0f} nodes/s)")
    print("=" * 56)


if __name__ == "__main__":
    main()
