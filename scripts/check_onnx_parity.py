"""Parity check: ONNX export vs the PyTorch fp32 reference (and, optionally,
the Rust ct-bot evaluator vs python-onnxruntime).

Plays random games with python-chess, samples positions, and compares
move_logits / value between a fp32 torch forward and an onnxruntime CPU
session on the exported model. The deployed Python bot runs bf16, so the
parity target here is torch-fp32 vs onnx-fp32 (the export's numerics).

With --rust-bin, additionally runs `ct-bot eval --fen <fen>` per position and
compares its value + per-move priors against a python-onnxruntime softmax —
this pins down the Rust ort plumbing (input dtypes/order, logits layout).

Run after re-exporting the model:
    uv run python scripts/check_onnx_parity.py --model data/models/pos2move_v2.1 --ema
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.models.tokenizer.alphazero_move_encoder import move_to_action_plane
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2

TOK = PostionTokenizer()


def load_torch_model(model_dir: Path, use_ema: bool) -> Pos2MoveV2:
    with open(model_dir / "model_config.json") as f:
        config = json.load(f)
    model = Pos2MoveV2(**config)
    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if use_ema:
        ema = torch.load(str(model_dir / "ema_state.pt"), map_location="cpu", weights_only=True)
        ema = {k.replace("_orig_mod.", ""): v for k, v in ema.items()}
        model.load_state_dict(ema, strict=False)
    return model.eval().float()


def encode(board: chess.Board) -> tuple[list[int], int, int, int]:
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    ep = chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8
    return TOK.encode(board), int(board.turn), castling, ep


def sample_positions(n: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    while len(boards) < n:
        board = chess.Board()
        while not board.is_game_over(claim_draw=True) and board.ply() < 200:
            boards.append(board.copy(stack=False))
            board.push(rng.choice(list(board.legal_moves)))
            if len(boards) >= n:
                break
    return boards


def batch_inputs(boards: list[chess.Board]):
    enc = [encode(b) for b in boards]
    return (
        np.array([e[0] for e in enc], dtype=np.int64),
        np.array([e[1] for e in enc], dtype=np.int64),
        np.array([e[2] for e in enc], dtype=np.int64),
        np.array([e[3] for e in enc], dtype=np.int64),
    )


def check_rust(rust_bin: str, onnx_path: Path, boards: list[chess.Board], tol: float) -> list[str]:
    """Compare `ct-bot eval --fen` priors/value vs a python-onnxruntime softmax."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    errors = []
    for board in boards:
        fen = board.fen()
        out = subprocess.run(
            [*rust_bin.split(), "eval", "--model", str(onnx_path), "--fen", fen],
            capture_output=True, text=True, check=True,
        )
        rs = json.loads(out.stdout)
        bt, pl, ca, ep = batch_inputs([board])
        logits, value = sess.run(
            ["move_logits", "value"],
            {"board_tokens": bt, "player_token": pl, "castling_token": ca, "en_passant_token": ep},
        )
        moves = list(board.legal_moves)
        scores = np.array([
            logits[0, m.from_square, move_to_action_plane(m.from_square, m.to_square, m.promotion)]
            for m in moves
        ])
        scores -= scores.max()
        priors = np.exp(scores) / np.exp(scores).sum()
        py = {m.uci(): p for m, p in zip(moves, priors)}
        rs_moves = {m["uci"]: m["prior"] for m in rs["moves"]}
        if set(py) != set(rs_moves):
            errors.append(f"move set mismatch at {fen}")
            continue
        dv = abs(rs["value"] - float(value[0, 0]))
        dp = max(abs(py[u] - rs_moves[u]) for u in py)
        if dv > tol or dp > tol:
            errors.append(f"rust mismatch at {fen}: dvalue={dv:.2e} dprior={dp:.2e}")
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="data/models/pos2move_v2.1")
    p.add_argument("--onnx", default=None, help="default: <model>/model[_ema].onnx")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--rust-bin", default=None,
                   help="ct-bot binary (+args) to also check the Rust evaluator (~50 positions)")
    args = p.parse_args()

    model_dir = Path(args.model)
    onnx_path = Path(args.onnx) if args.onnx else \
        model_dir / ("model_ema.onnx" if args.ema else "model.onnx")

    import onnxruntime as ort

    boards = sample_positions(args.n, args.seed)
    bt, pl, ca, ep = batch_inputs(boards)

    model = load_torch_model(model_dir, args.ema)
    with torch.no_grad():
        t_logits, t_value = model(
            torch.from_numpy(bt), torch.from_numpy(pl),
            torch.from_numpy(ca), torch.from_numpy(ep),
        )
    t_logits = t_logits.numpy()
    t_value = t_value.numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o_logits, o_value = sess.run(
        ["move_logits", "value"],
        {"board_tokens": bt, "player_token": pl, "castling_token": ca, "en_passant_token": ep},
    )

    d_logits = np.abs(t_logits - o_logits).max()
    d_value = np.abs(t_value - o_value).max()
    print(f"torch vs onnx on {args.n} positions: "
          f"max|Δlogits|={d_logits:.2e} max|Δvalue|={d_value:.2e} (tol {args.tol:.0e})")
    ok = d_logits < args.tol and d_value < args.tol

    if args.rust_bin:
        errors = check_rust(args.rust_bin, onnx_path, boards[:50], args.tol)
        for e in errors[:10]:
            print(f"FAIL: {e}")
        print(f"rust vs python-onnxruntime on 50 positions: "
              f"{'OK' if not errors else f'{len(errors)} mismatches'}")
        ok = ok and not errors

    if not ok:
        print("PARITY FAILED")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
