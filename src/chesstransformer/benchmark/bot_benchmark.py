"""
Bot benchmarking system for comparing chess bot performance.

Usage:
    python -m chesstransformer.benchmark.bot_benchmark \\
        --bot1-type position2move --bot1-checkpoint path/to/checkpoint \\
        --bot2-type random \\
        --num-games 100 \\
        --output results.json
"""

import chess
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from chesstransformer.bots import Position2MoveBot, LegacyPosition2MoveBot, RandomBot


@dataclass
class GameResult:
    """Single game result statistics."""
    game_id: int
    white_bot: str
    black_bot: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    num_moves: int
    white_time: float
    black_time: float
    termination: str  # "checkmate", "stalemate", "insufficient_material", "max_moves"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class BenchmarkStats:
    """Aggregate statistics for a bot."""
    bot_name: str
    total_games: int
    wins: int
    losses: int
    draws: int
    avg_moves_per_game: float
    avg_time_per_move: float
    total_time: float
    win_rate: float
    win_as_white: int
    win_as_black: int
    
    def to_dict(self):
        return asdict(self)


class BotBenchmark:
    """Benchmark system for comparing chess bots."""
    
    def __init__(self, max_moves: int = 500, time_limit: Optional[float] = None):
        """
        Initialize benchmark.
        
        Args:
            max_moves: Maximum moves per game before declaring draw
            time_limit: Optional time limit per move in seconds
        """
        self.max_moves = max_moves
        self.time_limit = time_limit
        self.game_results: List[GameResult] = []
        self.bot_counter = 0  # Counter for unique bot IDs
    
    def create_bot(self, bot_type: str, checkpoint_path: Optional[str] = None, bot_name: Optional[str] = None, **kwargs):
        """
        Factory method to create bots.
        
        Args:
            bot_type: Type of bot ("position2move", "legacy_position2move", "random")
            checkpoint_path: Path to model checkpoint (for Position2MoveBot)
            bot_name: Optional custom name for the bot (for tracking multiple instances)
            **kwargs: Additional bot-specific arguments
        """
        if bot_type.lower() == "position2move":
            if checkpoint_path:
                bot = Position2MoveBot(model_path=checkpoint_path)
            else:
                bot = Position2MoveBot()
        elif bot_type.lower() == "legacy_position2move":
            if checkpoint_path:
                bot = LegacyPosition2MoveBot(model_path=checkpoint_path)
            else:
                bot = LegacyPosition2MoveBot()
        elif bot_type.lower() == "random":
            bot = RandomBot()
        else:
            raise ValueError(f"Unknown bot type: {bot_type}")
        
        # Assign unique identifier
        if bot_name:
            bot._benchmark_id = bot_name
        else:
            self.bot_counter += 1
            bot._benchmark_id = f"{bot.__class__.__name__}_{self.bot_counter}"
        
        return bot
    
    def play_game(self, white_bot, black_bot, game_id: int, verbose: bool = False) -> GameResult:
        """
        Play a single game between two bots.
        
        Args:
            white_bot: Bot playing white
            black_bot: Bot playing black
            game_id: Unique game identifier
            verbose: Print move-by-move progress
        
        Returns:
            GameResult with statistics
        """
        board = chess.Board()
        white_time = 0.0
        black_time = 0.0
        move_count = 0
        
        while not board.is_game_over() and move_count < self.max_moves:
            current_bot = white_bot if board.turn == chess.WHITE else black_bot
            
            start_time = time.time()

            move, _ = current_bot.predict(board)

            elapsed = time.time() - start_time
            
            if board.turn == chess.WHITE:
                white_time += elapsed
            else:
                black_time += elapsed
            
            board.push(chess.Move.from_uci(move))
            move_count += 1
            
            if verbose and move_count % 10 == 0:
                print(f"Move {move_count}: {move}")
        
        # Determine result and termination reason
        if board.is_checkmate():
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            termination = "checkmate"
        elif board.is_stalemate():
            result = "1/2-1/2"
            termination = "stalemate"
        elif board.is_insufficient_material():
            result = "1/2-1/2"
            termination = "insufficient_material"
        elif board.is_seventyfive_moves():
            result = "1/2-1/2"
            termination = "75_move_rule"
        elif board.is_fivefold_repetition():
            result = "1/2-1/2"
            termination = "repetition"
        elif move_count >= self.max_moves:
            result = "1/2-1/2"
            termination = "max_moves"
        else:
            # Should not reach here if game ended normally
            result = "1/2-1/2"
            termination = "unknown"
        
        return GameResult(
            game_id=game_id,
            white_bot=white_bot._benchmark_id,
            black_bot=black_bot._benchmark_id,
            result=result,
            num_moves=move_count,
            white_time=white_time,
            black_time=black_time,
            termination=termination
        )
    
    def run_benchmark(
        self,
        bot1,
        bot2,
        num_games: int,
        alternate_colors: bool = True,
        verbose: bool = False
    ) -> Tuple[BenchmarkStats, BenchmarkStats]:
        """
        Run full benchmark between two bots.
        
        Args:
            bot1: First bot
            bot2: Second bot
            num_games: Number of games to play
            alternate_colors: Alternate colors each game
            verbose: Print game-by-game results
        
        Returns:
            Tuple of (bot1_stats, bot2_stats)
        """
        self.game_results = []
        
        pbar = tqdm(range(num_games), desc="Playing games")
        for game_id in pbar:
            if alternate_colors and game_id % 2 == 1:
                # Swap colors
                white_bot, black_bot = bot2, bot1
            else:
                white_bot, black_bot = bot1, bot2
            
            result = self.play_game(white_bot, black_bot, game_id, verbose=verbose)
            self.game_results.append(result)
            
            if verbose:
                print(f"Game {game_id + 1}: {result.white_bot} vs {result.black_bot} = {result.result}")
            
            # Update progress bar with current stats
            bot1_wins = sum(1 for r in self.game_results if self._bot_won(r, bot1._benchmark_id))
            bot2_wins = sum(1 for r in self.game_results if self._bot_won(r, bot2._benchmark_id))
            pbar.set_postfix({
                bot1._benchmark_id: bot1_wins,
                bot2._benchmark_id: bot2_wins
            })
        
        # Calculate statistics
        bot1_stats = self._calculate_stats(bot1._benchmark_id)
        bot2_stats = self._calculate_stats(bot2._benchmark_id)
        
        return bot1_stats, bot2_stats
    
    def _bot_won(self, result: GameResult, bot_name: str) -> bool:
        """Check if bot won the game."""
        if bot_name == result.white_bot and result.result == "1-0":
            return True
        if bot_name == result.black_bot and result.result == "0-1":
            return True
        return False
    
    def _calculate_stats(self, bot_name: str) -> BenchmarkStats:
        """Calculate aggregate statistics for a bot."""
        bot_results = []
        wins = 0
        losses = 0
        draws = 0
        win_as_white = 0
        win_as_black = 0
        total_time = 0.0
        total_moves = 0
        
        for result in self.game_results:
            is_white = result.white_bot == bot_name
            is_black = result.black_bot == bot_name
            
            if not (is_white or is_black):
                continue
            
            bot_results.append(result)
            total_moves += result.num_moves
            
            if is_white:
                total_time += result.white_time
                if result.result == "1-0":
                    wins += 1
                    win_as_white += 1
                elif result.result == "0-1":
                    losses += 1
                else:
                    draws += 1
            else:
                total_time += result.black_time
                if result.result == "0-1":
                    wins += 1
                    win_as_black += 1
                elif result.result == "1-0":
                    losses += 1
                else:
                    draws += 1
        
        total_games = len(bot_results)
        avg_moves = total_moves / total_games if total_games > 0 else 0
        avg_time_per_move = total_time / total_moves if total_moves > 0 else 0
        win_rate = wins / total_games if total_games > 0 else 0
        
        return BenchmarkStats(
            bot_name=bot_name,
            total_games=total_games,
            wins=wins,
            losses=losses,
            draws=draws,
            avg_moves_per_game=avg_moves,
            avg_time_per_move=avg_time_per_move,
            total_time=total_time,
            win_rate=win_rate,
            win_as_white=win_as_white,
            win_as_black=win_as_black
        )
    
    def save_results(self, output_path: str):
        """Save benchmark results to JSON file."""
        output = {
            "config": {
                "max_moves": self.max_moves,
                "time_limit": self.time_limit,
                "num_games": len(self.game_results)
            },
            "games": [r.to_dict() for r in self.game_results],
            "stats": {
                bot_name: self._calculate_stats(bot_name).to_dict()
                for bot_name in set(
                    [r.white_bot for r in self.game_results] + 
                    [r.black_bot for r in self.game_results]
                )
            }
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def print_summary(self, bot1_stats: BenchmarkStats, bot2_stats: BenchmarkStats):
        """Print formatted summary of benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        
        for stats in [bot1_stats, bot2_stats]:
            print(f"\n{stats.bot_name}:")
            print(f"  Total Games:       {stats.total_games}")
            print(f"  Wins:              {stats.wins} ({stats.win_rate:.1%})")
            print(f"    - As White:      {stats.win_as_white}")
            print(f"    - As Black:      {stats.win_as_black}")
            print(f"  Losses:            {stats.losses}")
            print(f"  Draws:             {stats.draws}")
            print(f"  Avg Moves/Game:    {stats.avg_moves_per_game:.1f}")
            print(f"  Avg Time/Move:     {stats.avg_time_per_move:.3f}s")
            print(f"  Total Time:        {stats.total_time:.1f}s")
        
        print("\n" + "=" * 80)


def main():
    """CLI entry point for bot benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark chess bots")
    parser.add_argument("--bot1-type", type=str, required=True, 
                       choices=["position2move", "legacy_position2move", "random"],
                       help="Type of first bot")
    parser.add_argument("--bot1-checkpoint", type=str, default=None,
                       help="Checkpoint path for bot1 (if applicable)")
    parser.add_argument("--bot1-name", type=str, default=None,
                       help="Custom name for bot1 (useful when comparing same bot types)")
    parser.add_argument("--bot2-type", type=str, required=True,
                       choices=["position2move", "legacy_position2move", "random"],
                       help="Type of second bot")
    parser.add_argument("--bot2-checkpoint", type=str, default=None,
                       help="Checkpoint path for bot2 (if applicable)")
    parser.add_argument("--bot2-name", type=str, default=None,
                       help="Custom name for bot2 (useful when comparing same bot types)")
    parser.add_argument("--num-games", type=int, default=100,
                       help="Number of games to play")
    parser.add_argument("--max-moves", type=int, default=500,
                       help="Maximum moves per game")
    parser.add_argument("--time-limit", type=float, default=None,
                       help="Time limit per move in seconds")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--no-alternate", action="store_true",
                       help="Don't alternate colors between games")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed game information")
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = BotBenchmark(max_moves=args.max_moves, time_limit=args.time_limit)
    
    # Create bots
    print(f"Creating {args.bot1_type} bot...")
    bot1 = benchmark.create_bot(args.bot1_type, checkpoint_path=args.bot1_checkpoint, bot_name=args.bot1_name)
    
    print(f"Creating {args.bot2_type} bot...")
    bot2 = benchmark.create_bot(args.bot2_type, checkpoint_path=args.bot2_checkpoint, bot_name=args.bot2_name)
    
    # Run benchmark
    print(f"\nRunning benchmark: {args.num_games} games")
    print(f"Max moves per game: {args.max_moves}")
    if args.time_limit:
        print(f"Time limit per move: {args.time_limit}s")
    
    bot1_stats, bot2_stats = benchmark.run_benchmark(
        bot1,
        bot2,
        num_games=args.num_games,
        alternate_colors=not args.no_alternate,
        verbose=args.verbose
    )
    
    # Print summary
    benchmark.print_summary(bot1_stats, bot2_stats)
    
    # Save results
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
