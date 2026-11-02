"""Blunders in head_to_head games: every move scored by a full-strength Stockfish.

Every position of a game, from the first move the engines chose to the last one, is
evaluated by Stockfish at a fixed node budget. A move costs the drop in the mover's
winning chances, on Lichess's scale: winning chances = 2 / (1 + exp(-0.00368208 * cp)) - 1,
cp clamped to +-1000 (a mate counts as 1000). A drop of 0.3 or more is a blunder, 0.2 a
mistake: the thresholds of Lichess's game analysis.

The report counts them per game for each side, the men on the board when they happen,
and how many of a side's losses hold at least one of its own blunders.

It needs games written with their moves (head_to_head.py --games-out records them;
older files hold results only). head_to_head.py --blunder-nodes runs it at the end of a
match; scripts/sf_ladder.sh does so at every level.

Usage:
    .venv/bin/python scripts/blunder_report.py GAMES.jsonl [...] [--nodes 200000]
        [--workers 4] [--stockfish /usr/games/stockfish] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine

CP_CLAMP = 1000
BLUNDER, MISTAKE = 0.3, 0.2
MEN_BUCKETS = ((2, 5), (6, 10), (11, 15), (16, 20), (21, 32))


def winning_chances(cp: float) -> float:
    """Lichess's winning chances in [-1, 1] for a score in centipawns (mover's view)."""
    cp = max(-CP_CLAMP, min(CP_CLAMP, cp))
    return 2 / (1 + math.exp(-0.00368208 * cp)) - 1


def white_cp(engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int) -> float:
    """The position's score for White, clamped; finished games scored by their result."""
    if board.is_checkmate():
        return -CP_CLAMP if board.turn == chess.WHITE else CP_CLAMP
    if board.is_game_over() or board.is_repetition(3) or board.halfmove_clock >= 100:
        return 0.0
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
    cp = info["score"].white().score(mate_score=100 * CP_CLAMP)
    return float(max(-CP_CLAMP, min(CP_CLAMP, cp)))


def engine_moves(rec: dict) -> tuple[chess.Board, list[chess.Move]]:
    """The board at the engines' first move and the moves they chose (book moves skipped)."""
    board = chess.Board(rec["start_fen"])
    moves = [chess.Move.from_uci(u) for u in rec["moves"]]
    book = rec.get("book_plies", 0)
    for m in moves[:book]:
        board.push(m)
    return board, moves[book:]


def analyse_game(engine: chess.engine.SimpleEngine, rec: dict, nodes: int) -> dict:
    """Every engine move of one game with its cost to the mover; side "a" or "b"."""
    board, moves = engine_moves(rec)
    scores = [white_cp(engine, board, nodes)]
    plies = []
    for m in moves:
        mover = board.turn
        men = len(board.piece_map())
        san = board.san(m)
        board.push(m)
        scores.append(white_cp(engine, board, nodes))
        sign = 1 if mover == chess.WHITE else -1
        loss = winning_chances(sign * scores[-2]) - winning_chances(sign * scores[-1])
        side = "a" if (mover == chess.WHITE) == rec["a_white"] else "b"
        plies.append({"ply": len(board.move_stack), "side": side, "san": san, "men": men,
                      "cp_before": scores[-2], "cp_after": scores[-1], "loss": round(loss, 3)})
    return {"idx": rec.get("idx"), "a_white": rec["a_white"], "result": rec["result"],
            "score_a": rec["score_a"], "end": rec.get("end"), "moves": plies}


def analyse_records(records: list[dict], stockfish: str, nodes: int, workers: int = 1) -> list[dict]:
    """analyse_game over the records, `workers` Stockfish processes of one thread each."""
    workers = max(1, min(workers, len(records)))
    shards = [list(range(len(records)))[i::workers] for i in range(workers)]
    out: list[dict | None] = [None] * len(records)

    def run(shard: list[int]):
        engine = chess.engine.SimpleEngine.popen_uci(stockfish)
        try:
            engine.configure({"Threads": 1, "Hash": 64})
            for i in shard:
                out[i] = analyse_game(engine, records[i], nodes)
        finally:
            engine.quit()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, shards))
    return [g for g in out if g is not None]


def own_blunders(game: dict, side: str) -> list[dict]:
    return [p for p in game["moves"] if p["side"] == side and p["loss"] >= BLUNDER]


def summarise(games: list[dict]) -> dict:
    """Per side: moves, blunders and mistakes per game, the men at blunders, and losses."""
    report = {}
    for side in ("a", "b"):
        moves = [p for g in games for p in g["moves"] if p["side"] == side]
        blunders = [p for p in moves if p["loss"] >= BLUNDER]
        mistakes = [p for p in moves if MISTAKE <= p["loss"] < BLUNDER]
        lost = [g for g in games if g["score_a"] == (0.0 if side == "a" else 1.0)]
        men = [p["men"] for p in blunders]
        n = max(len(games), 1)
        report[side] = {
            "games": len(games), "moves": len(moves),
            "blunders": len(blunders), "mistakes": len(mistakes),
            "blunders_per_game": len(blunders) / n, "mistakes_per_game": len(mistakes) / n,
            "blunders_per_100_moves": 100 * len(blunders) / max(len(moves), 1),
            "games_with_blunder": sum(1 for g in games if own_blunders(g, side)),
            "losses": len(lost),
            "losses_with_blunder": sum(1 for g in lost if own_blunders(g, side)),
            "men_median": statistics.median(men) if men else None,
            "men_buckets": {f"{lo}-{hi}": sum(1 for m in men if lo <= m <= hi)
                            for lo, hi in MEN_BUCKETS},
        }
    return report


def format_summary(report: dict, label_a: str = "A", label_b: str = "B") -> str:
    lines = []
    for side, label in (("a", label_a), ("b", label_b)):
        r = report[side]
        men = (f"at {', '.join(f'{k} men: {v}' for k, v in r['men_buckets'].items() if v)}"
               f" (median {r['men_median']:g})" if r["blunders"] else "none")
        lines.append(
            f"  {label:12} {r['blunders_per_game']:.2f} blunders / game "
            f"({r['blunders_per_100_moves']:.1f} per 100 moves), "
            f"{r['mistakes_per_game']:.2f} mistakes / game; "
            f"{r['games_with_blunder']}/{r['games']} games with one; "
            f"{r['losses_with_blunder']}/{r['losses']} losses hold one")
        lines.append(f"  {'':12} blunders {men}")
    return "\n".join(lines)


def header(nodes: int) -> str:
    return (f"Blunders: every move scored by Stockfish at {nodes:,} nodes a position; "
            f"a drop in the mover's winning chances of {BLUNDER} = blunder, "
            f"{MISTAKE} = mistake (Lichess)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("games", type=Path, nargs="+", help="head_to_head --games-out files")
    p.add_argument("--stockfish", default="/usr/games/stockfish")
    p.add_argument("--nodes", type=int, default=200_000, help="Stockfish nodes per position")
    p.add_argument("--workers", type=int, default=4, help="Stockfish processes, one thread each")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out", type=Path, default=None, help="write every scored move as JSON")
    args = p.parse_args()

    print(header(args.nodes))
    dump = {}
    for path in args.games:
        records = [json.loads(line) for line in path.open() if line.strip()]
        with_moves = [r for r in records if "moves" in r and "start_fen" in r]
        if len(with_moves) < len(records):
            print(f"{path}: {len(records) - len(with_moves)} of {len(records)} games have no "
                  f"moves (written before head_to_head recorded them); skipped")
        if not with_moves:
            continue
        games = analyse_records(with_moves, args.stockfish, args.nodes, args.workers)
        report = summarise(games)
        print(f"\n{path} ({len(games)} games)")
        print(format_summary(report, args.label_a, args.label_b))
        dump[str(path)] = {"summary": report, "games": games}
    if args.out:
        args.out.write_text(json.dumps(dump, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
