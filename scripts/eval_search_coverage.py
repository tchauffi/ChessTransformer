#!/usr/bin/env python3
"""Does the search ever *look at* the good move? A prior-visibility audit.

PUCT child selection (``rust/chess-core/src/tree.rs:47``) scores an unvisited child

    parent_Q - fpu + c_puct * sqrt(1 + N) * P(a)

so a move only earns its first visit once ``P(a) > fpu / (c_puct * sqrt(1 + N))``.
At the deployed ``fpu=0.2``, ``c_puct=1.0`` (``rust/ct-bot/src/search.rs:38``) and the
1800-sim cap, that floor is ~0.47 %: a move the policy scores below it is **invisible**
to the search -- never visited once, not merely under-explored.

This reports, per model, how wide the policy actually is and how often Stockfish's best
move falls below that floor. Companion to ``scripts/eval_policy.py``, which measures
whether the top-1 move is right; this measures whether the right move is *reachable*.

    uv run python scripts/eval_search_coverage.py \
        --models logs/.../run_019_.../checkpoints/best_model \
                 logs/.../run_023_.../checkpoints/best_model \
                 data/models/pos2move_v2.1 \
        --positions 300
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chess
import chess.engine

from eval_policy import TOK, load_model as _load_model


def load_model_anywhere(base_dir: str, device: str):
    """Load a checkpoint whose model_config.json may live further up the tree.

    Checkpoint dirs written by the trainer hold weights but no config -- only the run
    root does. (run_019's best_model has one solely because export_onnx.py was run there.)
    Resolve it by walking up, then hand a directory that has both to eval_policy's loader.
    """
    base = Path(base_dir)
    if (base / "model_config.json").exists():
        return _load_model(base_dir, device)
    for parent in base.parents:
        cfg = parent / "model_config.json"
        if cfg.exists():
            link = base / "model_config.json"
            link.write_text(cfg.read_text())      # materialise it next to the weights
            return _load_model(base_dir, device)
        if parent.name == "logs":
            break
    raise FileNotFoundError(f"no model_config.json at or above {base}")
from chesstransformer.datasets.h5_lichess_dataset import HDF5ChessDataset
from chesstransformer.models.tokenizer.alphazero_move_encoder import move_to_action_plane


def visibility_floor(sims: int, fpu: float, c_puct: float) -> float:
    """Minimum prior for a move to be visited at all within `sims` simulations."""
    return fpu / (c_puct * math.sqrt(1.0 + sims))


@torch.no_grad()
def policy_priors(model, board, device, prior_temp: float):
    """Legal moves and their priors, exactly as the Rust bot builds them.

    ``priors_for`` (rust/ct-bot/src/search.rs:368) gathers the logits of the legal moves
    only, divides by prior_temp and softmaxes over that subset -- not over the full
    64x73 action space. Matching that here is the whole point.
    """
    tok = torch.tensor(TOK.encode(board), dtype=torch.long, device=device).unsqueeze(0)
    player = torch.tensor([int(board.turn)], dtype=torch.long, device=device)
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE): castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE): castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK): castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK): castling |= 8
    ep = chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8
    logits, _ = model(tok, player,
                      torch.tensor([castling], dtype=torch.long, device=device),
                      torch.tensor([ep], dtype=torch.long, device=device))
    logits = logits[0].float().cpu().numpy()
    moves = list(board.legal_moves)
    s = np.array([logits[m.from_square,
                         move_to_action_plane(m.from_square, m.to_square, m.promotion)]
                  for m in moves], dtype=np.float64) / prior_temp
    s -= s.max()
    p = np.exp(s)
    return moves, p / p.sum()


def build_positions(ds, rng, eng, n, depth):
    """Fixed held-out set of (fen, stockfish_best_uci). Same recipe as eval_policy.py."""
    out = []
    while len(out) < n:
        idx = int(rng.integers(0, len(ds)))
        moves = ds._get_game_moves(ds.valid_game_indices[idx])
        if len(moves) <= 8:
            continue
        ply = int(rng.integers(6, len(moves) - 1))
        b = chess.Board()
        ok = True
        for i in range(ply):
            try:
                b.push(chess.Move.from_uci(ds._decode_move_token(int(moves[i]))))
            except (ValueError, AssertionError):
                ok = False
                break
        if not ok or b.is_game_over() or not any(b.legal_moves):
            continue
        info = eng.analyse(b, chess.engine.Limit(depth=depth))
        out.append((b.fen(), info["pv"][0].uci()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--h5", default="data/elite_db.h5")
    p.add_argument("--positions", type=int, default=300)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--eval-depth", type=int, default=10)
    p.add_argument("--seed", type=int, default=99999, help="held-out sampling seed")
    p.add_argument("--sf-path", default="/usr/games/stockfish")
    p.add_argument("--sims", type=int, nargs="+", default=[800, 1800],
                   help="simulation budgets whose visibility floor to report")
    p.add_argument("--fpu", type=float, default=0.2)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--prior-temp", type=float, default=1.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    floors = {n: visibility_floor(n, args.fpu, args.c_puct) for n in args.sims}
    print(f"PUCT: fpu={args.fpu} c_puct={args.c_puct} prior_temp={args.prior_temp}")
    for n, f in floors.items():
        print(f"  visibility floor @ {n:5d} sims: prior > {f*100:.3f} %")

    ds = HDF5ChessDataset(args.h5, sample_weighting="uniform")
    rng = np.random.default_rng(args.seed)
    eng = chess.engine.SimpleEngine.popen_uci(args.sf_path)
    eng.configure({"Threads": 1, "Hash": 64})

    positions = build_positions(ds, rng, eng, args.positions, args.depth)
    print(f"\nHeld-out positions: {len(positions)} (sf depth {args.depth})")

    cache: dict[tuple[str, str], float] = {}

    def eval_after(board, move):
        key = (board.fen(), move.uci())
        if key not in cache:
            board.push(move)
            cache[key] = -eng.analyse(
                board, chess.engine.Limit(depth=args.eval_depth)
            )["score"].relative.score(mate_score=2000)
            board.pop()
        return cache[key]

    rows = []
    for mdir in args.models:
        model = load_model_anywhere(mdir, device)
        H, eff, nlegal, visible = [], [], [], {n: [] for n in args.sims}
        rank, pbest, cp = [], [], []
        invisible = {n: 0 for n in args.sims}
        for fen, sf_best in positions:
            b = chess.Board(fen)
            moves, pri = policy_priors(model, b, device, args.prior_temp)
            h = float(-(pri * np.log(pri + 1e-12)).sum())
            H.append(h); eff.append(math.exp(h)); nlegal.append(len(moves))
            for n, f in floors.items():
                visible[n].append(int((pri >= f).sum()))
            order = np.argsort(-pri)
            uci = [moves[i].uci() for i in order]
            r = uci.index(sf_best) + 1 if sf_best in uci else len(uci) + 1
            rank.append(r)
            pb = float(pri[[m.uci() for m in moves].index(sf_best)]) if sf_best in uci else 0.0
            pbest.append(pb)
            for n, f in floors.items():
                invisible[n] += int(pb < f)
            cp.append(eval_after(b, chess.Move.from_uci(sf_best))
                      - eval_after(b, moves[int(order[0])]))
        n_pos = len(positions)
        rows.append(dict(
            name=Path(mdir).name if Path(mdir).name != "best_model"
                 else Path(mdir).parents[1].name,
            H=np.mean(H), eff=np.mean(eff), nlegal=np.mean(nlegal),
            visible={n: np.mean(v) for n, v in visible.items()},
            top1=np.mean([r == 1 for r in rank]), top3=np.mean([r <= 3 for r in rank]),
            top5=np.mean([r <= 5 for r in rank]), medrank=np.median(rank),
            medp=np.median(pbest), cp=np.mean(cp),
            invisible={n: v / n_pos for n, v in invisible.items()},
        ))
        del model
        torch.cuda.empty_cache()
    eng.quit()

    w = max(len(r["name"]) for r in rows) + 2
    print(f"\n{'':<{w}}{'entropy':>9}{'eff.mv':>8}{'legal':>7}" +
          "".join(f"{'vis@'+str(n):>9}" for n in args.sims))
    print("-" * (w + 24 + 9 * len(args.sims)))
    for r in rows:
        print(f"{r['name']:<{w}}{r['H']:>9.3f}{r['eff']:>8.2f}{r['nlegal']:>7.1f}" +
              "".join(f"{r['visible'][n]:>9.1f}" for n in args.sims))

    print(f"\nStockfish's best move, as seen by the policy:")
    print(f"{'':<{w}}{'top1':>8}{'top3':>8}{'top5':>8}{'medrank':>9}{'med prior':>11}{'cp_loss':>9}" +
          "".join(f"{'INVIS@'+str(n):>12}" for n in args.sims))
    print("-" * (w + 53 + 12 * len(args.sims)))
    for r in rows:
        print(f"{r['name']:<{w}}{r['top1']:>7.1%}{r['top3']:>8.1%}{r['top5']:>8.1%}"
              f"{r['medrank']:>9.0f}{r['medp']:>10.2%}{r['cp']:>8.0f}c" +
              "".join(f"{r['invisible'][n]:>11.1%}" for n in args.sims))
    print("\nINVIS@N = share of positions where Stockfish's best move has a prior below the\n"
          "visibility floor, i.e. the search never visits it once within N simulations.")


if __name__ == "__main__":
    main()
