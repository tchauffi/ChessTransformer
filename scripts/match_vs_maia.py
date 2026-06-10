"""Head-to-head: Pos2MoveV2 (MCTS) vs Maia — two engines both trained on human games.

Maia (CSSLab `maia2`, rapid) predicts the human move conditioned on a rating
level; we take its most-likely legal move (argmax) — the standard way to play
Maia at a given strength. Our bot plays deterministically (move_temp=0) so each
(opening, colour) is one reproducible game; variety comes from the opening suite.

CPU-only (Maia is tiny and our model is 11.7M), so this won't touch a training GPU.

Usage
-----
    uv run python scripts/match_vs_maia.py --maia-elo 1500 1900 --sims 400 --openings 12
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))           # engine_match.OPENINGS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.bots.pos2move_v2_mcts_bot import Pos2MoveV2MctsBot
from engine_match import OPENINGS


def maia_pick(m, prep, board: chess.Board, elo_self: int, elo_oppo: int) -> chess.Move:
    """Maia's move = argmax of its move distribution over the *legal* moves."""
    from maia2 import inference as maia_inf
    probs, _ = maia_inf.inference_each(m, prep, board.fen(), elo_self, elo_oppo)
    best, best_p = None, -1.0
    for mv in board.legal_moves:
        p = probs.get(mv.uci(), 0.0)
        if p > best_p:
            best_p, best = p, mv
    return best


def play(bot, m, prep, bot_is_white: bool, opening: str, maia_elo: int,
         oppo_elo: int, max_plies: int) -> str:
    """Return 'W' (bot win), 'D', or 'L' (bot loss)."""
    board = chess.Board(opening)
    bot._reuse_root = None          # fresh search tree per game
    bot._reuse_moves = None
    bot_color = chess.WHITE if bot_is_white else chess.BLACK

    plies = 0
    while not board.is_game_over() and plies < max_plies:
        if board.turn == bot_color:
            uci, _ = bot.predict(board)
            board.push_uci(uci)
        else:
            board.push(maia_pick(m, prep, board, maia_elo, oppo_elo))
        plies += 1

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return "D"
    return "W" if outcome.winner == bot_color else "L"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="data/models/pos2move_v2.1")
    p.add_argument("--maia-elo", type=int, nargs="+", default=[1500, 1900])
    p.add_argument("--oppo-elo", type=int, default=1900,
                   help="Rating Maia *thinks* its opponent is (conditions its play)")
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--cpuct", type=float, default=1.0)
    p.add_argument("--openings", type=int, default=12, help="Openings used (×2 colours)")
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--maia-type", default="rapid", choices=["rapid", "blitz"])
    args = p.parse_args()

    from maia2 import model as maia_model
    print("Loading Maia …")
    m = maia_model.from_pretrained(type=args.maia_type, device="cpu")
    from maia2 import inference as maia_inf
    prep = maia_inf.prepare()

    bot = Pos2MoveV2MctsBot(model_dir=args.model, num_simulations=args.sims,
                            c_puct=args.cpuct, device="cpu", compile=False,
                            time_limit=0.0, move_temp=0.0)

    openings = OPENINGS[: args.openings]
    for elo in args.maia_elo:
        w = d = l = 0
        t0 = time.time()
        print(f"\n=== Pos2MoveV2 (MCTS {args.sims}) vs Maia-{args.maia_type} {elo} "
              f"({len(openings) * 2} games) ===")
        for i, opening in enumerate(openings):
            for bot_white in (True, False):
                r = play(bot, m, prep, bot_white, opening, elo, args.oppo_elo, args.max_plies)
                w += r == "W"; d += r == "D"; l += r == "L"
        n = w + d + l
        score = w + 0.5 * d
        pct = 100 * score / n
        # rough Elo gap from score (logistic), clamped
        import math
        s = min(max(score / n, 1e-3), 1 - 1e-3)
        gap = -400 * math.log10(1 / s - 1)
        print(f"  Bot: +{w} ={d} -{l}   score {score:.1f}/{n} = {pct:.1f}%   "
              f"(~{gap:+.0f} Elo vs Maia {elo})   {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
