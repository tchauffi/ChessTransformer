"""GRPO fine-tuning on chess puzzles (verifiable reward).

Group Relative Policy Optimization on the policy head: at each solver-to-move
puzzle position we sample K candidate moves from the policy, score each with a
*verifiable* reward (matches the puzzle solution — pluggable to a Stockfish
"is-it-winning" verifier), turn the K rewards into group-relative advantages,
and do a policy-gradient update with a KL anchor to a frozen reference model
(prevents collapse / catastrophic forgetting).

No critic needed (GRPO uses the group baseline). The value head is left frozen.

Usage
-----
    uv run python scripts/grpo_puzzles.py \
        --base data/models/pos2move_v2.1 \
        --puzzles data/lichess_puzzles.csv.zst \
        --out data/models/pos2move_v2.1-grpo \
        --k 8 --batch 64 --steps 2000 --lr 1e-5 --beta-kl 0.02
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.datasets.puzzle_dataset import LichessPuzzleDataset
from chesstransformer.models.tokenizer.alphazero_move_encoder import move_to_action_plane
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer
from chesstransformer.models.transformer.pos2move_v2 import NUM_ACTION_PLANES, Pos2MoveV2

TOK = PostionTokenizer()


def load_model(base: str, device: str) -> Pos2MoveV2:
    cfg = json.loads((Path(base) / "model_config.json").read_text())
    m = Pos2MoveV2(**cfg)
    with safe_open(str(Path(base) / "model.safetensors"), framework="pt", device="cpu") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.to(device)


def solver_positions(puzzles) -> list[tuple[str, str]]:
    """Flatten puzzles into (fen, correct_uci) at every *solver*-to-move ply.

    Lichess convention: moves[0] is the opponent move played to reach the
    puzzle; the solver's moves are at odd indices (1, 3, 5, ...)."""
    out = []
    for p in puzzles:
        board = chess.Board(p["fen"])
        mv = p["moves"]
        try:
            board.push(chess.Move.from_uci(mv[0]))  # opponent setup move
        except (ValueError, AssertionError):
            continue
        for j in range(1, len(mv), 2):
            if not any(board.legal_moves):
                break
            out.append((board.fen(), mv[j]))
            try:
                board.push(chess.Move.from_uci(mv[j]))
                if j + 1 < len(mv):
                    board.push(chess.Move.from_uci(mv[j + 1]))
            except (ValueError, AssertionError):
                break
    return out


def encode_state(board: chess.Board):
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


def batch_tensors(boards, device):
    n = len(boards)
    bt = torch.zeros(n, 64, dtype=torch.long)
    pl = torch.zeros(n, dtype=torch.long)
    ca = torch.zeros(n, dtype=torch.long)
    ep = torch.zeros(n, dtype=torch.long)
    legal_idx, correct_local = [], []
    for i, (board, correct) in enumerate(boards):
        toks, player, castling, epf = encode_state(board)
        bt[i] = torch.tensor(toks, dtype=torch.long)
        pl[i], ca[i], ep[i] = player, castling, epf
        moves = list(board.legal_moves)
        idx = [m.from_square * NUM_ACTION_PLANES + move_to_action_plane(m.from_square, m.to_square, m.promotion)
               for m in moves]
        legal_idx.append(idx)
        correct_local.append(next((j for j, m in enumerate(moves) if m.uci() == correct), -1))
    return (bt.to(device), pl.to(device), ca.to(device), ep.to(device), legal_idx, correct_local)


def grpo_step(model, ref, batch, device, k, beta_kl, temperature):
    boards = batch  # list of (board, correct_uci)
    bt, pl, ca, ep, legal_idx, correct_local = batch_tensors(boards, device)
    move_logits, _ = model(bt, pl, ca, ep)
    flat = move_logits.view(move_logits.size(0), -1).float()
    with torch.no_grad():
        ref_logits, _ = ref(bt, pl, ca, ep)
        ref_flat = ref_logits.view(ref_logits.size(0), -1).float()

    total_loss = 0.0
    total_kl = 0.0
    solved = 0
    n = 0
    for i, (board, correct) in enumerate(boards):
        idx = torch.tensor(legal_idx[i], dtype=torch.long, device=device)
        if idx.numel() < 2 or correct_local[i] < 0:
            continue
        logp = F.log_softmax(flat[i, idx] / temperature, dim=0)            # differentiable
        ref_logp = F.log_softmax(ref_flat[i, idx] / temperature, dim=0)    # frozen
        probs = logp.detach().exp()
        samples = torch.multinomial(probs, k, replacement=True)            # K move indices (local)
        rewards = (samples == correct_local[i]).float()                    # verifiable: solution match
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        pg = -(adv * logp[samples]).mean()                                  # GRPO policy gradient
        kl = (logp.exp() * (logp - ref_logp)).sum()                        # KL(pi || ref)
        total_loss = total_loss + pg + beta_kl * kl
        total_kl = total_kl + kl.detach()
        solved += int(logp.argmax().item() == correct_local[i])            # greedy-solve metric
        n += 1
    if n == 0:
        return None
    return total_loss / n, (total_kl / n).item(), solved / n


@torch.no_grad()
def solve_rate(model, positions, device, max_n=1000):
    """Greedy puzzle solve-rate: argmax legal policy move == solution."""
    correct = 0
    n = 0
    for start in range(0, min(len(positions), max_n), 256):
        chunk = [(chess.Board(f), c) for f, c in positions[start:start + 256]]
        bt, pl, ca, ep, legal_idx, correct_local = batch_tensors(chunk, device)
        flat = model(bt, pl, ca, ep)[0].view(len(chunk), -1).float()
        for i in range(len(chunk)):
            if correct_local[i] < 0:
                continue
            idx = torch.tensor(legal_idx[i], device=device)
            pred = int(idx[flat[i, idx].argmax()].item())
            correct += int(pred == legal_idx[i][correct_local[i]])
            n += 1
    return correct / max(n, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="data/models/pos2move_v2.1")
    p.add_argument("--puzzles", required=True, help=".csv.zst Lichess puzzle file")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=8, help="samples per position (group size)")
    p.add_argument("--batch", type=int, default=64, help="positions per step")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta-kl", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--min-rating", type=int, default=None)
    p.add_argument("--max-rating", type=int, default=None)
    p.add_argument("--max-puzzles", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-every", type=int, default=200)
    args = p.parse_args()

    random.seed(0); torch.manual_seed(0)
    ds = LichessPuzzleDataset(args.puzzles, min_rating=args.min_rating,
                              max_rating=args.max_rating, max_puzzles=args.max_puzzles)
    positions = solver_positions(ds.puzzles)
    random.shuffle(positions)
    split = max(1, int(0.05 * len(positions)))
    val_pos, train_pos = positions[:split], positions[split:]
    print(f"Solver positions: {len(train_pos)} train / {len(val_pos)} val")

    model = load_model(args.base, args.device)
    ref = load_model(args.base, args.device).eval()
    for pr in ref.parameters():
        pr.requires_grad_(False)
    # Freeze the value head — GRPO only updates the policy.
    for name, pr in model.named_parameters():
        if name.startswith("value_head"):
            pr.requires_grad_(False)
    opt = torch.optim.AdamW([pr for pr in model.parameters() if pr.requires_grad], lr=args.lr)

    print(f"Initial greedy solve-rate (val): {solve_rate(model, val_pos, args.device):.1%}")
    model.train()
    for step in range(1, args.steps + 1):
        batch = [(chess.Board(f), c) for f, c in random.sample(train_pos, min(args.batch, len(train_pos)))]
        out = grpo_step(model, ref, batch, args.device, args.k, args.beta_kl, args.temperature)
        if out is None:
            continue
        loss, kl, greedy = out
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step}: loss={loss.item():.4f} kl={kl:.4f} batch_greedy_solve={greedy:.2f}")
        if step % args.eval_every == 0:
            model.eval()
            print(f"  [eval] val solve-rate: {solve_rate(model, val_pos, args.device):.1%}")
            model.train()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.eval()
    save_file(model.state_dict(), str(out / "model.safetensors"))
    import shutil
    shutil.copy(Path(args.base) / "model_config.json", out / "model_config.json")
    print(f"Final greedy solve-rate (val): {solve_rate(model, val_pos, args.device):.1%}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
