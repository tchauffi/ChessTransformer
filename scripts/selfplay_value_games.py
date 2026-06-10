"""Self-play game generation for value-head retraining (step 1 of self-play RL).

Plays games with the *frozen* current policy via the MCTS bot and records every
position with the eventual game outcome. The hypothesis: the value head trained
on human game results is the weak link (humans blunder won positions), and
self-play-with-search outcomes are a much cleaner label. Train the value head
alone on this data with scripts/train_value_head.py, then A/B with
scripts/engine_match.py.

Diversity comes from the opening suite + visit-count temperature sampling over
the early plies (move_temp). Cheap-game tricks: resign when both sides agree
the position is lost (a fraction of games never resign, to monitor false
resignations), and a hard ply cap adjudicated as a draw.

Output: data shards ``positions_NNNN.npz`` (encoded positions + stm-POV outcome
``z`` + root MCTS value) and ``games.jsonl`` (one line per game, for sanity /
PGN reconstruction). Re-running with the same --out appends.

Usage
-----
    uv run python scripts/selfplay_value_games.py \
        --model data/models/pos2move_v2.1 --out data/selfplay/v2.1 \
        --games 2000 --sims 128
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine_match import OPENINGS

from chesstransformer.bots.pos2move_v2_mcts_bot import Pos2MoveV2MctsBot


def game_result(board: chess.Board) -> float | None:
    """White-POV result if the game is over (cheap checks, mirrors the bot's
    _terminal_value), else None."""
    if not any(board.legal_moves):
        if board.is_check():  # side to move is mated
            return 1.0 if board.turn == chess.BLACK else -1.0
        return 0.0
    if (
        board.is_insufficient_material()
        or board.is_repetition(3)
        or board.halfmove_clock >= 100
    ):
        return 0.0
    return None


class ShardWriter:
    """Accumulates per-position records and flushes npz shards to disk."""

    def __init__(self, out_dir: Path, flush_positions: int = 50_000):
        self.out_dir = out_dir
        self.flush_positions = flush_positions
        self.shard_idx = len(list(out_dir.glob("positions_*.npz")))
        self._reset()

    def _reset(self):
        self.boards: list[np.ndarray] = []
        self.player: list[int] = []
        self.castling: list[int] = []
        self.ep: list[int] = []
        self.halfmove: list[int] = []
        self.z: list[float] = []          # stm-POV outcome
        self.root_v: list[float] = []     # stm-POV root MCTS value
        self.game_id: list[int] = []

    def add_game(self, records: list[tuple], z_white: float, game_id: int):
        for tokens, player, castling, ep, halfmove, root_v in records:
            self.boards.append(np.asarray(tokens, dtype=np.uint8))
            self.player.append(player)
            self.castling.append(castling)
            self.ep.append(ep)
            self.halfmove.append(halfmove)
            self.z.append(z_white if player == 1 else -z_white)
            self.root_v.append(root_v)
            self.game_id.append(game_id)
        if len(self.z) >= self.flush_positions:
            self.flush()

    def flush(self):
        if not self.z:
            return
        path = self.out_dir / f"positions_{self.shard_idx:04d}.npz"
        np.savez_compressed(
            path,
            boards=np.stack(self.boards),
            player=np.asarray(self.player, dtype=np.uint8),
            castling=np.asarray(self.castling, dtype=np.uint8),
            ep=np.asarray(self.ep, dtype=np.uint8),
            halfmove=np.asarray(self.halfmove, dtype=np.uint16),
            z=np.asarray(self.z, dtype=np.float32),
            root_v=np.asarray(self.root_v, dtype=np.float32),
            game_id=np.asarray(self.game_id, dtype=np.uint32),
        )
        print(f"  wrote {path.name} ({len(self.z)} positions)")
        self.shard_idx += 1
        self._reset()


def play_game(
    bot: Pos2MoveV2MctsBot,
    start_fen: str,
    max_plies: int,
    resign_threshold: float,
    resign_plies: int,
    allow_resign: bool,
) -> tuple[list[tuple], float, list[str], str]:
    """Play one self-play game. Returns (records, z_white, uci_moves, end_reason)."""
    board = chess.Board(start_fen)
    bot._reuse_root = None  # never re-root a tree carried over from the previous game
    records: list[tuple] = []
    moves: list[str] = []
    resign_streak = 0
    resign_sign = 0

    for _ in range(max_plies):
        result = game_result(board)
        if result is not None:
            return records, result, moves, "terminal"

        tokens, player, castling, ep = bot._encode_position(board)
        halfmove = 2 * (board.fullmove_number - 1) + (0 if board.turn == chess.WHITE else 1)

        uci, value = bot.predict(board)  # value is stm-POV root value
        records.append((tokens, player, castling, ep, halfmove, value))
        moves.append(uci)
        board.push_uci(uci)

        v_white = value if player == 1 else -value
        if abs(v_white) >= resign_threshold:
            sign = 1 if v_white > 0 else -1
            resign_streak = resign_streak + 1 if sign == resign_sign else 1
            resign_sign = sign
        else:
            resign_streak = 0
        if allow_resign and resign_streak >= resign_plies:
            return records, float(resign_sign), moves, "resign"

    result = game_result(board)
    if result is not None:
        return records, result, moves, "terminal"
    return records, 0.0, moves, "max_plies"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="data/models/pos2move_v2.1")
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--sims", type=int, default=128)
    p.add_argument("--sim-batch", type=int, default=16)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--fpu", type=float, default=0.2)
    p.add_argument("--move-temp", type=float, default=1.0)
    p.add_argument("--temp-plies", type=int, default=30,
                   help="sample moves ∝ visits^(1/temp) for this many half-moves, then argmax")
    p.add_argument("--resign", type=float, default=0.93,
                   help="resign when |white-POV root value| stays above this")
    p.add_argument("--resign-plies", type=int, default=6,
                   help="consecutive plies above threshold before resigning")
    p.add_argument("--no-resign-frac", type=float, default=0.1,
                   help="fraction of games played out fully, to monitor false resignations")
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--tt-size", type=int, default=150_000,
                   help="bot transposition-table entries (~19KB each — RAM!); "
                        "lower this when running several workers in parallel")
    p.add_argument("--flush-positions", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    games_log = out_dir / "games.jsonl"
    game_id = sum(1 for _ in open(games_log)) if games_log.exists() else 0
    if game_id:
        print(f"Resuming after {game_id} existing games in {out_dir}")
    random.seed(args.seed + game_id)
    np.random.seed(args.seed + game_id)  # move_temp sampling uses np.random

    (out_dir / "generation_config.json").write_text(json.dumps(vars(args), indent=2))

    bot = Pos2MoveV2MctsBot(
        model_dir=args.model,
        num_simulations=args.sims,
        c_puct=args.c_puct,
        fpu=args.fpu,
        sim_batch=args.sim_batch,
        tt_size=args.tt_size,
        move_temp=args.move_temp,
        move_temp_plies=args.temp_plies,
        time_limit=0.0,
    )
    writer = ShardWriter(out_dir, flush_positions=args.flush_positions)

    openings = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] + list(OPENINGS)
    counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    n_positions = 0
    t0 = time.time()

    with open(games_log, "a") as log:
        for i in range(args.games):
            fen = random.choice(openings)
            allow_resign = random.random() >= args.no_resign_frac
            t_game = time.time()
            records, z_white, moves, reason = play_game(
                bot, fen, args.max_plies, args.resign, args.resign_plies, allow_resign
            )
            result = "1-0" if z_white > 0 else "0-1" if z_white < 0 else "1/2-1/2"
            counts[result] += 1
            n_positions += len(records)
            writer.add_game(records, z_white, game_id)
            log.write(json.dumps({
                "id": game_id, "fen": fen, "moves": moves, "result": result,
                "end": reason, "plies": len(moves), "resign_allowed": allow_resign,
            }) + "\n")
            log.flush()
            game_id += 1

            dt = time.time() - t_game
            elapsed = time.time() - t0
            print(
                f"game {i + 1}/{args.games} (id={game_id - 1}): {result} in {len(moves)} plies "
                f"[{reason}] {dt:.1f}s | total {n_positions} pos, "
                f"{(i + 1) / elapsed * 3600:.0f} games/h",
                flush=True,
            )

    writer.flush()
    total = sum(counts.values())
    print(
        f"\nDone: {total} games, {n_positions} positions | "
        f"W {counts['1-0']} / D {counts['1/2-1/2']} / B {counts['0-1']} | "
        f"{(time.time() - t0) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
