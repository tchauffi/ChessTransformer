"""Estimate Elo by playing the TRT bot against Stockfish at various skill levels.

Stockfish skill levels 0–20 map roughly to known Elo ratings.
We play N games at each level, compute win rates, and use the
standard Elo expectation formula to estimate the bot's rating.

Usage
-----
    export LD_LIBRARY_PATH=".venv/lib/python3.12/site-packages/tensorrt_libs"
    python scripts/elo_gauntlet.py <model_dir> [--depth 5] [--games 10] [--stockfish-path /usr/games/stockfish]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.bots.pos2move_v2_bot import Pos2MoveV2Bot


# Approximate Elo for Stockfish skill levels (SF 16 on modern hardware).
# These are rough community-consensus values; exact numbers vary by hardware/version.
SKILL_ELO = {
    0: 800,
    1: 900,
    2: 1000,
    3: 1100,
    4: 1200,
    5: 1350,
    6: 1500,
    7: 1600,
    8: 1700,
    9: 1800,
    10: 1900,
    11: 2000,
    12: 2100,
    13: 2200,
    14: 2300,
    15: 2400,
    16: 2500,
    17: 2600,
    18: 2700,
    19: 2850,
    20: 3200,
}


def play_game(
    bot: Pos2MoveV2Bot,
    engine: chess.engine.SimpleEngine,
    bot_color: chess.Color,
    sf_time_limit: float,
    max_moves: int = 200,
) -> tuple[str, str, list[str]]:
    """Play one game. Returns (result, termination, move_list).

    Note: we deliberately do NOT use ``claim_draw=True`` to terminate the game.
    Threefold-repetition and 50-move are *claimable* draws — only the side to
    move may invoke them in a real game, and a winning side will never do so.
    Auto-claiming would unfairly cut short games where (e.g.) Stockfish is
    winning but a temporary repetition pops up. We only stop on conditions
    that end the game automatically: checkmate, stalemate, insufficient
    material, fivefold repetition, or the 75-move rule.
    """
    board = chess.Board()
    moves: list[str] = []

    while not board.is_game_over(claim_draw=False) and board.fullmove_number <= max_moves:
        if board.turn == bot_color:
            move_uci, _ = bot.predict(board)
            move = chess.Move.from_uci(move_uci)
        else:
            result = engine.play(board, chess.engine.Limit(time=sf_time_limit))
            move = result.move

        moves.append(move.uci())
        board.push(move)

    if board.is_checkmate():
        # Side to move is checkmated
        result = "0-1" if board.turn == chess.WHITE else "1-0"
        termination = "checkmate"
    elif board.is_stalemate():
        result = "1/2-1/2"
        termination = "stalemate"
    elif board.is_insufficient_material():
        result = "1/2-1/2"
        termination = "insufficient_material"
    elif board.is_fivefold_repetition():
        result = "1/2-1/2"
        termination = "fivefold_repetition"
    elif board.is_seventyfive_moves():
        result = "1/2-1/2"
        termination = "seventyfive_moves"
    elif board.fullmove_number > max_moves:
        result = "1/2-1/2"
        termination = "max_moves"
    else:
        result = board.result(claim_draw=False)
        termination = "other"

    return result, termination, moves


def score_from_result(result: str, bot_color: chess.Color) -> float:
    """Return score from the bot's perspective: 1=win, 0.5=draw, 0=loss."""
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if bot_color == chess.WHITE else 0.0
    if result == "0-1":
        return 1.0 if bot_color == chess.BLACK else 0.0
    return 0.5


def elo_from_score(score: float, opponent_elo: int) -> float | None:
    """Estimate Elo from win rate against a known-Elo opponent.

    Uses the inverse of the expected-score formula:
        E = 1 / (1 + 10^((Ro - R) / 400))
        R = Ro - 400 * log10(1/E - 1)
    """
    if score <= 0.0:
        return opponent_elo - 400  # lost all games
    if score >= 1.0:
        return opponent_elo + 400  # won all games
    return opponent_elo - 400.0 * math.log10(1.0 / score - 1.0)


def run_gauntlet(
    bot: Pos2MoveV2Bot,
    stockfish_path: str,
    skill_levels: list[int],
    games_per_level: int,
    sf_time_limit: float,
    pgn_path: Path | None,
) -> list[dict]:
    """Play games at each skill level and return results."""
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    all_results = []
    pgn_games = []

    for skill in skill_levels:
        engine.configure({"Skill Level": skill, "Threads": 1, "Hash": 16})
        ref_elo = SKILL_ELO.get(skill, 1500)

        wins = draws = losses = 0
        level_start = time.time()

        print(f"\n{'='*60}")
        print(f"Stockfish Skill {skill}  (approx. {ref_elo} Elo)")
        print(f"{'='*60}")

        for g in range(games_per_level):
            bot_color = chess.WHITE if g % 2 == 0 else chess.BLACK
            color_str = "White" if bot_color == chess.WHITE else "Black"

            t0 = time.time()
            result, termination, moves = play_game(
                bot, engine, bot_color, sf_time_limit
            )
            elapsed = time.time() - t0
            sc = score_from_result(result, bot_color)

            if sc == 1.0:
                wins += 1
                outcome = "WIN"
            elif sc == 0.0:
                losses += 1
                outcome = "LOSS"
            else:
                draws += 1
                outcome = "DRAW"

            print(
                f"  Game {g+1}/{games_per_level}: Bot={color_str}  "
                f"Result={result} ({outcome})  {termination}  "
                f"{len(moves)} moves  {elapsed:.1f}s"
            )

            # Build PGN game
            pgn_game = chess.pgn.Game()
            pgn_game.headers["Event"] = "Elo Gauntlet"
            pgn_game.headers["Date"] = time.strftime("%Y.%m.%d")
            depth_str = getattr(bot, "depth", None) or f"s{getattr(bot, 'num_simulations', '?')}"
            bot_name = f"ChessTransformer_{depth_str}"
            sf_name = f"Stockfish_skill{skill}"
            if bot_color == chess.WHITE:
                pgn_game.headers["White"] = bot_name
                pgn_game.headers["Black"] = sf_name
            else:
                pgn_game.headers["White"] = sf_name
                pgn_game.headers["Black"] = bot_name
            pgn_game.headers["Result"] = result
            pgn_game.headers["Termination"] = termination

            node = pgn_game
            board = chess.Board()
            for m in moves:
                move = chess.Move.from_uci(m)
                node = node.add_variation(move)
                board.push(move)

            pgn_games.append(pgn_game)

        total_games = wins + draws + losses
        total_score = wins + 0.5 * draws
        avg_score = total_score / total_games if total_games > 0 else 0.5
        est_elo = elo_from_score(avg_score, ref_elo)
        level_time = time.time() - level_start

        print(f"\n  Results vs Skill {skill}: "
              f"+{wins} ={draws} -{losses}  "
              f"Score: {total_score}/{total_games} ({avg_score:.1%})")
        print(f"  Estimated Elo: {est_elo:.0f}  (ref={ref_elo})")
        print(f"  Time: {level_time:.0f}s")

        all_results.append({
            "skill": skill,
            "ref_elo": ref_elo,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": avg_score,
            "est_elo": round(est_elo) if est_elo is not None else None,
        })

        # Stop early if bot is losing badly
        if avg_score < 0.15 and total_games >= 4:
            print(f"\n  Bot is losing heavily — stopping gauntlet here.")
            break

    engine.quit()

    # Write PGN file
    if pgn_path and pgn_games:
        with open(pgn_path, "w") as f:
            for g in pgn_games:
                print(g, file=f, end="\n\n")
        print(f"\nPGN saved to {pgn_path}")

    return all_results


def print_summary(results: list[dict]):
    """Print final Elo estimation summary."""
    print(f"\n{'='*60}")
    print("GAUNTLET SUMMARY")
    print(f"{'='*60}")
    print(f"{'Skill':>6} {'Ref Elo':>8} {'W':>3} {'D':>3} {'L':>3} {'Score':>7} {'Est Elo':>8}")
    print("-" * 50)

    elo_estimates = []
    weights = []
    for r in results:
        total = r["wins"] + r["draws"] + r["losses"]
        score_str = f"{r['score']:.0%}"
        elo_str = str(r["est_elo"]) if r["est_elo"] else "N/A"
        print(f"{r['skill']:>6} {r['ref_elo']:>8} {r['wins']:>3} {r['draws']:>3} "
              f"{r['losses']:>3} {score_str:>7} {elo_str:>8}")
        if r["est_elo"] is not None:
            elo_estimates.append(r["est_elo"])
            # Weight by number of games and closeness to 50% score
            closeness = 1.0 - abs(r["score"] - 0.5) * 2
            weights.append(total * max(closeness, 0.1))

    if elo_estimates:
        # Weighted average Elo (closer-to-50% matchups are more informative)
        total_w = sum(weights)
        weighted_elo = sum(e * w for e, w in zip(elo_estimates, weights)) / total_w
        simple_avg = sum(elo_estimates) / len(elo_estimates)

        print("-" * 50)
        print(f"  Simple average Elo:   {simple_avg:.0f}")
        print(f"  Weighted average Elo: {weighted_elo:.0f}")
        print(f"\n  ** Estimated Elo: ~{round(weighted_elo / 25) * 25} **")


def main():
    parser = argparse.ArgumentParser(description="Estimate bot Elo via Stockfish gauntlet")
    parser.add_argument("model_dir", help="Path to model directory with model.onnx")
    parser.add_argument("--depth", type=int, default=5, help="Search depth (default: 5)")
    parser.add_argument("--top-p", type=float, default=0.90, help="Nucleus probability threshold (default: 0.90)")
    parser.add_argument("--games", type=int, default=10, help="Games per skill level (default: 10)")
    parser.add_argument("--time-limit", type=float, default=30.0, help="Bot time limit per move (default: 30s)")
    parser.add_argument("--sf-time", type=float, default=0.1, help="Stockfish time per move in seconds (default: 0.1)")
    parser.add_argument("--sf-path", default="/usr/games/stockfish", help="Path to Stockfish binary")
    parser.add_argument("--skills", type=int, nargs="+", default=None,
                        help="Skill levels to test (default: auto-scan 0,2,4,...)")
    parser.add_argument("--pgn", default=None, help="Output PGN file path")
    parser.add_argument("--ema", action="store_true", help="Load EMA weights (requires ema_state.pt in model_dir)")
    args = parser.parse_args()

    if args.skills is None:
        # Start from low skill, stop when losing heavily
        skill_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    else:
        skill_levels = args.skills

    pgn_path = Path(args.pgn) if args.pgn else Path(args.model_dir) / "gauntlet.pgn"

    mode_str = f"alpha-beta (depth={args.depth}, top_p={args.top_p:.0%})"
    print(f"Elo Gauntlet: {mode_str}, "
          f"games/level={args.games}, bot_time={args.time_limit}s, sf_time={args.sf_time}s")
    print(f"Stockfish: {args.sf_path}")
    print(f"Skills: {skill_levels}")
    print(f"PGN output: {pgn_path}")

    bot = Pos2MoveV2Bot(
        model_dir=args.model_dir,
        depth=args.depth,
        top_p=args.top_p,
        time_limit=args.time_limit,
        use_ema=args.ema,
    )

    results = run_gauntlet(
        bot=bot,
        stockfish_path=args.sf_path,
        skill_levels=skill_levels,
        games_per_level=args.games,
        sf_time_limit=args.sf_time,
        pgn_path=pgn_path,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
