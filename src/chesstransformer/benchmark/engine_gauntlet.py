"""
Engine Gauntlet — estimate absolute Elo by playing against Stockfish.

This is the method used by AlphaZero (Silver et al., 2018): play direct
matches against an engine of known strength, then derive Elo from the
Win / Draw / Loss record.

Stockfish skill‑levels 0‑20 map (approximately) to 800‑3200 Elo.  The
gauntlet plays N games at each requested level, alternating colours, and
produces per‑level statistics plus an overall estimated Elo.

Requirements:
    pip install stockfish   (Python wrapper)
    Stockfish binary on PATH, or pass the path explicitly.

Usage:
    python -m chesstransformer.benchmark.engine_gauntlet \\
        --bot-type position2move \\
        --bot-checkpoint path/to/model.safetensors \\
        --num-games 20 \\
        --skill-levels 0 5 10 15 20 \\
        --output gauntlet_results.json
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Stockfish skill‑level → approximate Elo mapping
# Derived from community testing (CCRL/Lichess).  The exact numbers depend
# on hardware and binary version, but the curve is well‑established.
# ---------------------------------------------------------------------------
SKILL_LEVEL_ELO: Dict[int, int] = {
    0: 800,
    1: 900,
    2: 1000,
    3: 1100,
    4: 1200,
    5: 1300,
    6: 1400,
    7: 1500,
    8: 1600,
    9: 1700,
    10: 1800,
    11: 1900,
    12: 2000,
    13: 2100,
    14: 2200,
    15: 2300,
    16: 2400,
    17: 2500,
    18: 2600,
    19: 2800,
    20: 3200,
}


# ── dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class LevelResult:
    """Statistics for one Stockfish skill‑level."""

    skill_level: int
    stockfish_elo: int
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Winning fraction (1 for win, 0.5 for draw)."""
        if self.total == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total

    @property
    def elo_diff(self) -> float:
        """Elo difference estimated from the score.

        Δ = −400 · log₁₀(1/p − 1)
        Returns 0 when score is 0 or 1 (undefined).
        """
        p = self.score
        if p <= 0.0 or p >= 1.0:
            return 400.0 if p >= 1.0 else -400.0
        return -400.0 * math.log10(1.0 / p - 1.0)

    @property
    def estimated_elo(self) -> float:
        """Bot's estimated Elo anchored to this level's Stockfish Elo."""
        return self.stockfish_elo + self.elo_diff

    @property
    def los(self) -> float:
        """Likelihood Of Superiority.

        LOS = ½ [1 + erf((W − L) / √(2(W + L)))]
        """
        w, l = self.wins, self.losses
        if w + l == 0:
            return 0.5
        return 0.5 * (1.0 + math.erf((w - l) / math.sqrt(2.0 * (w + l))))

    @property
    def elo_error_margin(self) -> float:
        """95 % confidence interval half‑width on the Elo difference.

        Uses the normal approximation:
            σ_p = √(p(1−p)/N)
            σ_Elo ≈ 400 · σ_p / (p(1−p) · ln10)
        Returns ±∞ when N is 0 or score is 0/1.
        """
        p = self.score
        n = self.total
        if n == 0 or p <= 0.0 or p >= 1.0:
            return float("inf")
        sigma_p = math.sqrt(p * (1 - p) / n)
        # derivative of Elo w.r.t. p: dElo/dp = 400 / (p(1-p) ln10)
        d_elo_dp = 400.0 / (p * (1 - p) * math.log(10))
        sigma_elo = abs(d_elo_dp) * sigma_p
        return 1.96 * sigma_elo  # 95% CI

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "total": self.total,
            "score": round(self.score, 4),
            "elo_diff": round(self.elo_diff, 1),
            "estimated_elo": round(self.estimated_elo, 1),
            "los": round(self.los, 4),
            "elo_error_margin_95": round(self.elo_error_margin, 1),
        }


@dataclass
class GauntletResult:
    """Full gauntlet result across all skill‑levels."""

    bot_name: str
    levels: List[LevelResult] = field(default_factory=list)
    total_time_s: float = 0.0

    @property
    def overall_estimated_elo(self) -> float:
        """Weighted average of per‑level Elo estimates.

        Levels with more games and tighter error bars contribute more.
        """
        weights: List[float] = []
        elos: List[float] = []
        for lvl in self.levels:
            if lvl.total == 0:
                continue
            margin = lvl.elo_error_margin
            if margin <= 0 or math.isinf(margin):
                continue
            w = 1.0 / (margin ** 2)
            weights.append(w)
            elos.append(lvl.estimated_elo)
        if not weights:
            # fallback: simple average
            valid = [l for l in self.levels if l.total > 0]
            if not valid:
                return 0.0
            return sum(l.estimated_elo for l in valid) / len(valid)
        return sum(w * e for w, e in zip(weights, elos)) / sum(weights)

    @property
    def total_games(self) -> int:
        return sum(l.total for l in self.levels)

    @property
    def total_wins(self) -> int:
        return sum(l.wins for l in self.levels)

    @property
    def total_draws(self) -> int:
        return sum(l.draws for l in self.levels)

    @property
    def total_losses(self) -> int:
        return sum(l.losses for l in self.levels)

    def to_dict(self) -> dict:
        return {
            "bot_name": self.bot_name,
            "overall_estimated_elo": round(self.overall_estimated_elo, 1),
            "total_games": self.total_games,
            "total_wins": self.total_wins,
            "total_draws": self.total_draws,
            "total_losses": self.total_losses,
            "total_time_s": round(self.total_time_s, 1),
            "levels": [l.to_dict() for l in self.levels],
        }


# ── main class ───────────────────────────────────────────────────────────────


class StockfishGauntlet:
    """Play a chess bot against Stockfish at various skill levels to estimate Elo.

    This mirrors the AlphaZero methodology: derive Elo from actual game
    results against an engine of known strength.

    Parameters
    ----------
    stockfish_path : str or None
        Path to the Stockfish binary.  When *None* the ``stockfish`` Python
        package will try to locate it automatically.
    stockfish_time_limit : float
        Seconds Stockfish is allowed per move (default 0.1 s).
    stockfish_depth : int or None
        Optional fixed search depth instead of a time limit.
    max_moves : int
        Maximum half‑moves per game before declaring a draw.
    skill_elo_map : dict or None
        Override the default skill‑level → Elo mapping.
    """

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        stockfish_time_limit: float = 0.1,
        stockfish_depth: Optional[int] = None,
        max_moves: int = 500,
        skill_elo_map: Optional[Dict[int, int]] = None,
    ):
        self.stockfish_path = stockfish_path
        self.stockfish_time_limit = stockfish_time_limit
        self.stockfish_depth = stockfish_depth
        self.max_moves = max_moves
        self.skill_elo_map = skill_elo_map or SKILL_LEVEL_ELO

    # ── internal helpers ─────────────────────────────────────────────────

    def _create_stockfish(self, skill_level: int):
        """Create a fresh Stockfish instance at the given skill level."""
        try:
            from stockfish import Stockfish
        except ImportError:
            raise ImportError(
                "The 'stockfish' Python package is required.\n"
                "Install it with:  pip install stockfish\n"
                "You also need the Stockfish binary installed on your system:\n"
                "  - Linux:   sudo apt install stockfish\n"
                "  - macOS:   brew install stockfish\n"
                "  - Windows: download from https://stockfishchess.org/download/"
            )

        params: dict = {"Skill Level": skill_level, "Threads": 1, "Hash": 64}

        if self.stockfish_path:
            sf = Stockfish(path=self.stockfish_path, parameters=params)
        else:
            sf = Stockfish(parameters=params)

        return sf

    @staticmethod
    def _greedy_predict(bot, board: chess.Board):
        """Greedy (argmax) wrapper around a bot's predict method.

        Instead of sampling from the probability distribution, always
        pick the move with the highest probability.  This gives the
        bot's true ceiling strength rather than a stochastic estimate.
        """
        import torch

        tokens_ids = bot.position_tokenizer.encode(board)
        torch_input = (
            torch.tensor(tokens_ids).unsqueeze(0).long().to(bot.device)
        )
        is_white = torch.tensor([board.turn]).bool().to(bot.device)
        logits = bot.model(torch_input, is_white)

        # Handle value head models that return (logits, value) tuple
        if isinstance(logits, tuple):
            logits = logits[0]

        legal_moves = [m.uci() for m in board.legal_moves]
        mask = torch.full((logits.size(-1),), float("-inf")).to(bot.device)
        for move in legal_moves:
            move_id = bot.move_tokenizer.encode(move)
            mask[move_id] = 0.0

        masked_logits = logits[0] + mask
        best_id = masked_logits.argmax(dim=-1).item()
        proba = torch.softmax(masked_logits, dim=-1)[best_id].item()
        return bot.move_tokenizer.decode(best_id), proba

    def _play_game(
        self,
        bot,
        skill_level: int,
        bot_plays_white: bool,
        greedy: bool = True,
    ) -> str:
        """Play one game.  Returns ``"1-0"``, ``"0-1"``, or ``"1/2-1/2"``."""
        sf = self._create_stockfish(skill_level)
        board = chess.Board()
        move_count = 0

        while not board.is_game_over() and move_count < self.max_moves:
            is_bot_turn = (board.turn == chess.WHITE) == bot_plays_white

            if is_bot_turn:
                if greedy and hasattr(bot, "position_tokenizer"):
                    move_uci, _ = self._greedy_predict(bot, board)
                else:
                    move_uci, _ = bot.predict(board)
            else:
                sf.set_fen_position(board.fen())
                if self.stockfish_depth is not None:
                    move_uci = sf.get_best_move_time(
                        int(self.stockfish_time_limit * 1000)
                    )
                    # Fallback: use depth‑limited search
                    if move_uci is None:
                        move_uci = sf.get_best_move()
                else:
                    move_uci = sf.get_best_move_time(
                        int(self.stockfish_time_limit * 1000)
                    )
                if move_uci is None:
                    # Stockfish resigned or could not find a move
                    break

            board.push(chess.Move.from_uci(move_uci))
            move_count += 1

        # Determine result
        if board.is_checkmate():
            return "0-1" if board.turn == chess.WHITE else "1-0"
        return "1/2-1/2"

    # ── public API ───────────────────────────────────────────────────────

    def run(
        self,
        bot,
        skill_levels: Optional[List[int]] = None,
        num_games_per_level: int = 20,
        verbose: bool = True,
        greedy: bool = True,
    ) -> GauntletResult:
        """Run the full gauntlet.

        Parameters
        ----------
        bot
            Any bot implementing ``predict(board) -> (move_uci, proba)``.
        skill_levels : list[int]
            Stockfish skill‑levels to test.  Defaults to ``[0, 5, 10, 15, 20]``.
        num_games_per_level : int
            Games per level (colours are alternated).
        verbose : bool
            Print live results.
        greedy : bool
            Use argmax decoding instead of the bot's own sampling
            strategy.  This evaluates the bot at its ceiling strength
            (default True).

        Returns
        -------
        GauntletResult
        """
        if skill_levels is None:
            skill_levels = [0, 5, 10, 15, 20]

        bot_name = getattr(bot, "_benchmark_id", bot.__class__.__name__)
        result = GauntletResult(bot_name=bot_name)
        t0 = time.time()

        for skill in skill_levels:
            sf_elo = self.skill_elo_map.get(skill, 1500)
            lvl = LevelResult(skill_level=skill, stockfish_elo=sf_elo)

            desc = f"Skill {skill:>2} (~{sf_elo} Elo)"
            pbar = tqdm(
                range(num_games_per_level), desc=desc, disable=not verbose
            )

            for game_i in pbar:
                bot_plays_white = game_i % 2 == 0
                game_result = self._play_game(bot, skill, bot_plays_white, greedy=greedy)

                # Attribute W/D/L from bot's perspective
                if game_result == "1/2-1/2":
                    lvl.draws += 1
                elif (game_result == "1-0") == bot_plays_white:
                    lvl.wins += 1
                else:
                    lvl.losses += 1

                pbar.set_postfix(
                    W=lvl.wins, D=lvl.draws, L=lvl.losses,
                    score=f"{lvl.score:.0%}",
                )

            result.levels.append(lvl)

            if verbose:
                sign = "+" if lvl.elo_diff >= 0 else ""
                print(
                    f"  → {lvl.wins}W / {lvl.draws}D / {lvl.losses}L  "
                    f"score={lvl.score:.1%}  "
                    f"Δ={sign}{lvl.elo_diff:.0f}  "
                    f"est. Elo={lvl.estimated_elo:.0f} ± {lvl.elo_error_margin:.0f}  "
                    f"LOS={lvl.los:.1%}"
                )

        result.total_time_s = time.time() - t0

        if verbose:
            self.print_summary(result)

        return result

    # ── display ──────────────────────────────────────────────────────────

    @staticmethod
    def print_summary(result: GauntletResult):
        """Pretty‑print the gauntlet results."""
        print("\n" + "=" * 80)
        print(f"  ENGINE GAUNTLET — {result.bot_name}")
        print("=" * 80)
        print(
            f"{'Level':>6}  {'SF Elo':>7}  {'W':>4} {'D':>4} {'L':>4}  "
            f"{'Score':>6}  {'ΔElo':>7}  {'Est.Elo':>8}  {'±95%':>6}  {'LOS':>6}"
        )
        print("-" * 80)

        for lvl in result.levels:
            sign = "+" if lvl.elo_diff >= 0 else ""
            margin = (
                f"{lvl.elo_error_margin:.0f}"
                if not math.isinf(lvl.elo_error_margin)
                else "∞"
            )
            print(
                f"{lvl.skill_level:>6}  {lvl.stockfish_elo:>7}  "
                f"{lvl.wins:>4} {lvl.draws:>4} {lvl.losses:>4}  "
                f"{lvl.score:>5.1%}  "
                f"{sign}{lvl.elo_diff:>6.0f}  "
                f"{lvl.estimated_elo:>8.0f}  "
                f"{margin:>6}  "
                f"{lvl.los:>5.1%}"
            )

        print("-" * 80)
        print(
            f"  Overall estimated Elo : {result.overall_estimated_elo:.0f}  "
            f"({result.total_games} games in {result.total_time_s:.0f}s)"
        )
        print(
            f"  Totals                : "
            f"{result.total_wins}W / {result.total_draws}D / {result.total_losses}L"
        )
        print("=" * 80 + "\n")

    # ── persistence ──────────────────────────────────────────────────────

    @staticmethod
    def save_results(result: GauntletResult, output_path: str):
        """Save gauntlet results to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    """Command‑line interface for the engine gauntlet."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate bot Elo by playing against Stockfish (AlphaZero method)."
    )
    parser.add_argument(
        "--bot-type",
        type=str,
        required=True,
        choices=["position2move", "legacy_position2move", "random"],
        help="Type of bot to evaluate.",
    )
    parser.add_argument(
        "--bot-checkpoint",
        type=str,
        default=str(
            Path(__file__).parents[1]
            / "data/models/position2moveV2.1/best_model/model.safetensors"
        ),
        help="Path to model checkpoint (default: position2moveV2.1).",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        help="Path to Stockfish binary (auto‑detected if omitted).",
    )
    parser.add_argument(
        "--stockfish-time",
        type=float,
        default=0.1,
        help="Seconds Stockfish is allowed per move (default 0.1).",
    )
    parser.add_argument(
        "--skill-levels",
        type=int,
        nargs="+",
        default=[0, 3, 5, 8, 10, 13, 15, 18, 20],
        help="Stockfish skill‑levels to test (default: 0 3 5 8 10 13 15 18 20).",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=20,
        help="Games per skill‑level (default 20).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=500,
        help="Max half‑moves per game before draw (default 500).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gauntlet_results.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--no-greedy",
        action="store_true",
        help="Use the bot's own sampling strategy instead of argmax.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Device for model inference (default: cuda if available, else cpu).",
    )

    args = parser.parse_args()

    # Build bot via BotBenchmark factory
    from chesstransformer.benchmark.bot_benchmark import BotBenchmark

    factory = BotBenchmark()
    bot = factory.create_bot(args.bot_type, checkpoint_path=args.bot_checkpoint, device=args.device)

    greedy = not args.no_greedy
    print(f"Bot        : {bot._benchmark_id}")
    print(f"Device     : {args.device}")
    print(f"SF time/mv : {args.stockfish_time}s")
    print(f"Levels     : {args.skill_levels}")
    print(f"Games/level: {args.num_games}")
    print(f"Greedy     : {greedy}")
    print()

    gauntlet = StockfishGauntlet(
        stockfish_path=args.stockfish_path,
        stockfish_time_limit=args.stockfish_time,
        max_moves=args.max_moves,
    )

    result = gauntlet.run(
        bot,
        skill_levels=args.skill_levels,
        num_games_per_level=args.num_games,
        verbose=True,
        greedy=greedy,
    )

    StockfishGauntlet.save_results(result, args.output)


if __name__ == "__main__":
    main()
