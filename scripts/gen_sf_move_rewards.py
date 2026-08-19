#!/usr/bin/env python3
"""Stockfish centipawn rewards for candidate moves in self-play positions.

Builds the reward table that ``scripts/grpo_selfplay.py`` trains against: for
each position, a small set of candidate moves and what Stockfish thinks of each.

Why this is not the distillation that failed
--------------------------------------------
``scripts/distill_policy.py`` minimised cross-entropy against Stockfish's move
distribution over the full 4672-way action space, and did not move the policy
even at 200k labels -- the conclusion recorded in ``doc/selfplay_rl.md`` was
that the 11.7M policy head is at capacity. Asking a small network to *reproduce*
a stronger player's whole distribution is a much harder ask than getting it to
**re-rank the handful of moves it already considers**. A cp reward supports the
second without demanding the first, and the gradient only ever touches moves
that are actually in play.

The candidate set
-----------------
Candidates are the union of

* the policy's own top-K legal moves -- the moves it actually plays, which is
  where a re-ranking gradient has to act; and
* Stockfish's top-M -- because a table containing only the policy's preferences
  can never teach it about a good move it currently ranks low.

That second half matters more than it looks. ``eval_search_coverage.py`` shows
Stockfish's best move sits below the PUCT visibility floor 1.8% of the time at
800 sims, i.e. the search cannot see it at all. Restricting the reward table to
the policy's own top-K would bake that same blind spot into training -- the
exact failure this whole line of work is trying to undo.

Scoring
-------
Every candidate is scored by one Stockfish search restricted to the candidate
set (``root_moves`` + ``multipv``), so all cp values in a row come from the same
search at the same depth and are directly comparable. Scores are stored from the
**mover's** point of view, and converted to a value in [-1, 1] downstream with
the same Lichess win curve ``prep_eval_value_data.py`` uses (CP_K = 0.00368208).

Usage
-----
    uv run python scripts/gen_sf_move_rewards.py \
        --selfplay data/selfplay/v2.1-exit-128 --model data/models/pos2move_v2.1 \
        --out data/rewards/exit128-sf10.npz --positions 150000 --workers 12
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.models.tokenizer.alphazero_move_encoder import (  # noqa: E402
    move_to_action_plane,
)
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer  # noqa: E402
from chesstransformer.models.transformer.pos2move_v2 import (  # noqa: E402
    NUM_ACTION_PLANES,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grpo_puzzles import encode_state, load_model  # noqa: E402

SF_PATH = "/usr/games/stockfish"
TOK = PostionTokenizer()

# Mate scores are clipped to this many centipawns before the win curve. Left raw
# they dominate every advantage in the group and turn the reward into a mate
# detector.
MATE_CP = 2000

_ENGINE: chess.engine.SimpleEngine | None = None
_DEPTH = 10
_TOP_M = 4


def _worker_init(sf_path: str, depth: int, top_m: int) -> None:
    global _ENGINE, _DEPTH, _TOP_M
    _DEPTH, _TOP_M = depth, top_m
    _ENGINE = chess.engine.SimpleEngine.popen_uci(sf_path)
    _ENGINE.configure({"Threads": 1, "Hash": 64})


def _cp_of(score: chess.engine.PovScore) -> int:
    """Mover-POV centipawns, with mates clipped rather than infinite."""
    rel = score.relative
    if rel.is_mate():
        m = rel.mate()
        return MATE_CP if m > 0 else -MATE_CP
    return int(rel.score())


def _score_position(item: tuple[int, str, list[str]]):
    """Score one position's candidates. Returns (row, ucis, cps) or None."""
    row, fen, policy_ucis = item
    board = chess.Board(fen)
    legal = {m.uci(): m for m in board.legal_moves}
    if len(legal) < 2:
        return None

    cands = [legal[u] for u in policy_ucis if u in legal]

    # Discover Stockfish's own preferences so the table is not limited to what
    # the policy already likes.
    try:
        top = _ENGINE.analyse(board, chess.engine.Limit(depth=_DEPTH), multipv=_TOP_M)
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
        return None
    for line in top:
        pv = line.get("pv")
        if pv and pv[0] not in cands:
            cands.append(pv[0])

    if len(cands) < 2:
        return None

    # One search over exactly the candidate set: every cp in this row comes from
    # the same search at the same depth, so they are comparable to each other.
    try:
        info = _ENGINE.analyse(board, chess.engine.Limit(depth=_DEPTH),
                               multipv=len(cands), root_moves=cands)
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
        return None

    ucis, cps = [], []
    for line in info:
        pv = line.get("pv")
        if not pv:
            continue
        ucis.append(pv[0].uci())
        cps.append(_cp_of(line["score"]))
    if len(ucis) < 2:
        return None
    return row, ucis, cps


def sample_positions(selfplay: Path, n: int, min_ply: int, seed: int) -> list[tuple[str, int, int]]:
    """(fen, game_id, ply) sampled from the recorded self-play games."""
    jsonl = selfplay / "games.jsonl"
    if not jsonl.exists():
        raise SystemExit(f"{jsonl} not found")

    games = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(games)

    out: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for g in games:
        if len(out) >= n:
            break
        board = chess.Board(g.get("fen") or chess.STARTING_FEN)
        for ply, uci in enumerate(g["moves"]):
            if ply >= min_ply and not board.is_game_over():
                key = " ".join(board.fen().split()[:4])
                if key not in seen:
                    seen.add(key)
                    out.append((board.fen(), int(g["id"]), ply))
                    if len(out) >= n:
                        break
            try:
                board.push(chess.Move.from_uci(uci))
            except (ValueError, AssertionError):
                break
    return out


@torch.no_grad()
def policy_topk(model, positions: list[str], k: int, device: str,
                batch_size: int = 512) -> tuple[list[list[str]], list[float]]:
    """Top-k legal moves per position, plus the probability mass they cover.

    The covered mass is the number that says whether k is large enough: if the
    policy routinely puts weight outside its own top-k, a table built from top-k
    is missing moves the trainer will sample.
    """
    all_ucis: list[list[str]] = []
    covered: list[float] = []
    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start + batch_size]
        boards = [chess.Board(f) for f in chunk]
        n = len(boards)
        bt = torch.zeros(n, 64, dtype=torch.long)
        pl = torch.zeros(n, dtype=torch.long)
        ca = torch.zeros(n, dtype=torch.long)
        ep = torch.zeros(n, dtype=torch.long)
        for i, b in enumerate(boards):
            toks, player, castling, epf = encode_state(b)
            bt[i] = torch.tensor(toks, dtype=torch.long)
            pl[i], ca[i], ep[i] = player, castling, epf
        logits, _ = model(bt.to(device), pl.to(device), ca.to(device), ep.to(device))
        flat = logits.view(n, -1).float().cpu()

        for i, b in enumerate(boards):
            moves = list(b.legal_moves)
            idx = [m.from_square * NUM_ACTION_PLANES
                   + move_to_action_plane(m.from_square, m.to_square, m.promotion)
                   for m in moves]
            probs = torch.softmax(flat[i, idx], dim=0)
            order = torch.argsort(probs, descending=True)[:k]
            all_ucis.append([moves[j].uci() for j in order.tolist()])
            covered.append(float(probs[order].sum()))
        print(f"  policy top-{k}: {min(start + batch_size, len(positions))}/"
              f"{len(positions)}", end="\r", flush=True)
    print()
    return all_ucis, covered


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selfplay", type=Path, required=True,
                   help="self-play run directory containing games.jsonl")
    p.add_argument("--model", default="data/models/pos2move_v2.1",
                   help="policy whose top-K forms half the candidate set")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--positions", type=int, default=150_000)
    p.add_argument("--top-k", type=int, default=8,
                   help="policy candidates per position")
    p.add_argument("--top-m", type=int, default=4,
                   help="Stockfish candidates to add, so the table can teach "
                        "moves the policy currently ranks low")
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--min-ply", type=int, default=6,
                   help="skip book-ish opening plies")
    p.add_argument("--workers", type=int, default=min(12, mp.cpu_count()))
    p.add_argument("--sf-path", default=SF_PATH)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"sampling up to {args.positions} positions from {args.selfplay} ...", flush=True)
    sampled = sample_positions(args.selfplay, args.positions, args.min_ply, args.seed)
    fens = [s[0] for s in sampled]
    print(f"  {len(fens)} distinct positions")

    print(f"policy top-{args.top_k} on {args.device} ...", flush=True)
    model = load_model(args.model, args.device).eval()
    topk_ucis, covered = policy_topk(model, fens, args.top_k, args.device)
    print(f"  policy mass covered by top-{args.top_k}: "
          f"mean {np.mean(covered):.4f}, p05 {np.percentile(covered, 5):.4f}")
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    print(f"scoring with Stockfish depth {args.depth} on {args.workers} workers ...",
          flush=True)
    work = [(i, fens[i], topk_ucis[i]) for i in range(len(fens))]
    rows, t0, done = [], time.time(), 0
    with mp.Pool(args.workers, initializer=_worker_init,
                 initargs=(args.sf_path, args.depth, args.top_m)) as pool:
        for res in pool.imap_unordered(_score_position, work, chunksize=8):
            done += 1
            if done % 2000 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(work) - done) / max(rate, 1e-9) / 60
                print(f"  {done}/{len(work)} scored ({rate:.0f}/s, ETA {eta:.0f} min)",
                      flush=True)
            if res is not None:
                rows.append(res)

    if not rows:
        print("no positions scored", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r[0])
    max_c = max(len(r[1]) for r in rows)
    n = len(rows)

    boards_a = np.zeros((n, 64), dtype=np.uint8)
    player_a = np.zeros(n, dtype=np.uint8)
    castle_a = np.zeros(n, dtype=np.uint8)
    ep_a = np.zeros(n, dtype=np.uint8)
    cand_idx = np.full((n, max_c), -1, dtype=np.int32)
    cand_cp = np.zeros((n, max_c), dtype=np.float32)
    cand_mask = np.zeros((n, max_c), dtype=bool)
    game_id = np.zeros(n, dtype=np.int32)
    ply_a = np.zeros(n, dtype=np.uint16)

    # Legal moves as a CSR, so the trainer can softmax over the true legal set
    # without running python-chess inside the training loop. Renormalising over
    # candidates alone would optimise a different distribution than the one the
    # engine actually plays.
    legal_flat: list[int] = []
    legal_ptr = np.zeros(n + 1, dtype=np.int64)

    for i, (row, ucis, cps) in enumerate(rows):
        fen, gid, ply = sampled[row]
        board = chess.Board(fen)
        toks, player, castling, epf = encode_state(board)
        boards_a[i] = np.asarray(toks, dtype=np.uint8)
        player_a[i], castle_a[i], ep_a[i] = player, castling, epf
        game_id[i], ply_a[i] = gid, ply
        for j, (u, cp) in enumerate(zip(ucis, cps)):
            m = chess.Move.from_uci(u)
            cand_idx[i, j] = (m.from_square * NUM_ACTION_PLANES
                              + move_to_action_plane(m.from_square, m.to_square, m.promotion))
            cand_cp[i, j] = cp
            cand_mask[i, j] = True
        for m in board.legal_moves:
            legal_flat.append(m.from_square * NUM_ACTION_PLANES
                              + move_to_action_plane(m.from_square, m.to_square, m.promotion))
        legal_ptr[i + 1] = len(legal_flat)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        boards=boards_a, player=player_a, castling=castle_a, ep=ep_a,
        cand_idx=cand_idx, cand_cp=cand_cp, cand_mask=cand_mask,
        legal_idx=np.asarray(legal_flat, dtype=np.int32), legal_ptr=legal_ptr,
        game_id=game_id, ply=ply_a,
        meta=np.array([args.depth, args.top_k, args.top_m, MATE_CP], dtype=np.int32),
    )
    n_c = cand_mask.sum(1)
    print(f"\nwrote {n} positions to {args.out}")
    print(f"  candidates/position: mean {n_c.mean():.2f}, min {n_c.min()}, max {n_c.max()}")
    print(f"  cp spread within a position: mean "
          f"{np.mean([np.ptp(cand_cp[i, cand_mask[i]]) for i in range(n)]):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
