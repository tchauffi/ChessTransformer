#!/usr/bin/env python3
"""
Training script for Position2Move transformer model.

This script trains a transformer to predict chess moves from board positions.
Uses the new efficient HDF5 dataset format with game-based storage and random sampling.

Usage:
    python src/chesstransformer/trainers/position2move_trainer.py --data data/Lichess2017.h5
    
Options:
    --data: Path to HDF5 dataset
    --min-elo: Minimum average ELO for filtering games
    --max-elo: Maximum average ELO for filtering games
    --batch-size: Batch size (default: 32)
    --epochs: Number of epochs (default: 10)
    --lr: Learning rate (default: 3e-4)
    --save-every: Save checkpoint every N epochs (default: 5)
"""

from pathlib import Path
from datetime import datetime
import argparse
import copy
import random
import numpy as np
import json

from tqdm.auto import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import accelerate
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import LambdaLR

from chesstransformer.datasets.h5_lichess_dataset import HDF5ChessDataset
from chesstransformer.models.transformer.position2move import Position2MoveModel


def result_to_value_target(result, is_white):
    """Convert game result to value target from the current player's perspective.
    
    Args:
        result: Tensor of game outcomes (0=draw, 1=white win, 2=black win)
        is_white: Tensor of booleans (True if current player is white)
    
    Returns:
        Tensor of float values in [-1, 1]:
            +1.0 = current player won
             0.0 = draw
            -1.0 = current player lost
    """
    # Map: 0 (draw) -> 0.0, 1 (white win) -> +1.0, 2 (black win) -> -1.0
    value = torch.zeros_like(result, dtype=torch.float32)
    value[result == 1] = 1.0   # white win
    value[result == 2] = -1.0  # black win (= white loss)
    
    # Flip sign for black's perspective: black winning is +1 for black
    is_black = ~is_white.bool()
    value[is_black] = -value[is_black]
    
    return value


def get_next_run_number(log_dir="logs"):
    """Get the next run number by checking existing run directories."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return 1

    existing_runs = [
        d.name for d in log_path.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]

    if not existing_runs:
        return 1

    run_numbers = []
    for run_name in existing_runs:
        try:
            num = int(run_name.split("_")[1])
            run_numbers.append(num)
        except (IndexError, ValueError):
            continue

    return max(run_numbers) + 1 if run_numbers else 1

def compute_loss(logits, moves, legal_moves_mask, beta, legal_move_loss_weight, value=None, value_target=None, value_loss_weight=0.0):
    llm_loss = F.cross_entropy(logits, moves)
    illegal_prob = F.softmax(logits, dim=-1) * (1. - legal_moves_mask.float())
    legal_move_loss = torch.sum(illegal_prob ** 2, dim=-1).mean()
    filtered_logits = torch.where(legal_moves_mask.bool(), logits, -torch.inf)
    legal_move_cross_entropy = F.cross_entropy(filtered_logits, moves)
    policy_loss = beta * llm_loss + legal_move_loss_weight * legal_move_loss + (1 - beta) * legal_move_cross_entropy
    
    # Value loss (MSE between predicted value and game outcome from current player's perspective)
    v_loss = torch.tensor(0.0, device=logits.device)
    if value is not None and value_target is not None and value_loss_weight > 0:
        v_loss = F.mse_loss(value, value_target)
    
    loss = policy_loss + value_loss_weight * v_loss
    return loss, llm_loss, legal_move_loss, legal_move_cross_entropy, v_loss

def save_trainer_state(checkpoint_dir, epoch, global_step, best_val_loss, scheduler_config=None):
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    if scheduler_config:
        state["scheduler_config"] = scheduler_config
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    with state_path.open("w", encoding="utf-8") as fp:
        json.dump(state, fp)

def load_trainer_state(checkpoint_dir):
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.exists():
        return None
    with state_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)

def cleanup_old_step_checkpoints(checkpoint_dir, max_checkpoints):
    """Remove old step checkpoints, keeping only the most recent ones."""
    checkpoint_path = Path(checkpoint_dir)
    step_checkpoints = sorted(
        [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint_step_")],
        key=lambda x: int(x.name.split("_")[-1])
    )
    
    # Remove oldest checkpoints if we exceed max_checkpoints
    while len(step_checkpoints) > max_checkpoints:
        oldest = step_checkpoints.pop(0)
        import shutil
        shutil.rmtree(oldest)
        print(f"  Removed old checkpoint: {oldest.name}")

def create_linear_warmup_decay_scheduler(optimizer, warmup_steps, total_steps, final_lr_ratio):
    final_lr_ratio = max(0.0, final_lr_ratio)
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = (current_step - warmup_steps + 1) / float(max(1, total_steps - warmup_steps))
        return 1.0 - (1.0 - final_lr_ratio) * min(1.0, progress)

    return LambdaLR(optimizer, lr_lambda)

def find_best_learning_rate(model, train_loader, accelerator, lr_min=1e-6, lr_max=1e-2, num_steps=100, beta=0.7, legal_move_loss_weight=10.0, use_value_head=False, value_loss_weight=0.0):
    model.train()
    device = accelerator.device
    initial_state = copy.deepcopy(model.state_dict())
    # Use base learning rate for LR finder
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_min)
    multiplier = (lr_max / lr_min) ** (1 / max(num_steps - 1, 1))
    best_lr = lr_min
    best_loss = float("inf")
    lr = lr_min
    data_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        optimizer.zero_grad()
        positions = batch["position"].to(device)
        moves = batch["move"].to(device)
        is_white = batch["is_white"].to(device)
        legal_moves_mask = batch["legal_moves_mask"].to(device)
        castling_rights = batch["castling_rights"].to(device)
        en_passant_file = batch["en_passant_file"].to(device)
        halfmove_clock = batch["halfmove_clock"].to(device)

        output = model(positions, is_white, castling_rights, en_passant_file, halfmove_clock)
        
        value = None
        value_target = None
        if use_value_head:
            logits, value = output
            result = batch["result"].to(device)
            value_target = result_to_value_target(result, is_white)
        else:
            logits = output

        loss, llm_loss, legal_move_loss, legal_move_cross_entropy, v_loss = compute_loss(
            logits, moves, legal_moves_mask, beta, legal_move_loss_weight,
            value=value, value_target=value_target, value_loss_weight=value_loss_weight,
        )

        accelerator.backward(loss)
        optimizer.step()

        loss_value = accelerator.gather_for_metrics(loss.detach()).mean().item()
        if loss_value < best_loss:
            best_loss = loss_value
            best_lr = lr

        lr *= multiplier
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    model.load_state_dict(initial_state)
    model.zero_grad(set_to_none=True)
    accelerator.free_memory()
    return best_lr, best_loss

def main():
    default_data_path = Path(__file__).parents[2] / "data" / "Lichess2017.h5"

    parser = argparse.ArgumentParser(description="Train Position2Move Transformer")
    parser.add_argument("--data", type=str, default=default_data_path, help="Path to HDF5 dataset")
    parser.add_argument("--min-elo", type=int, default=None, help="Minimum average ELO for filtering games")
    parser.add_argument("--max-elo", type=int, default=None, help="Maximum average ELO for filtering games")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for transformers (base LR)")
    parser.add_argument("--lr-embedding", type=float, default=None, help="Learning rate for embeddings (default: same as --lr)")
    parser.add_argument("--lr-head", type=float, default=None, help="Learning rate for output head (default: same as --lr)")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--lr-find", action="store_true", help="Run learning rate finder before training")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum LR for finder sweep")
    parser.add_argument("--lr-max", type=float, default=1e-2, help="Maximum LR for finder sweep")
    parser.add_argument("--lr-steps", type=int, default=100, help="Number of steps for LR finder sweep")
    parser.add_argument("--beta", type=float, default=0.7, help="Weight for LLM loss in combined loss function")
    parser.add_argument("--legal-move-loss-weight", type=float, default=10.0, help="Weight for legal move loss in combined loss function")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--final-lr-ratio", type=float, default=0.01, help="Final LR multiplier after linear decay")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm for clipping (0 to disable)")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to use from the dataset")
    parser.add_argument("--save-steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--max-checkpoints", type=int, default=5, help="Maximum number of step checkpoints to keep (excluding best model)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["bf16", "fp16", "fp32"], help="Precision for training")
    parser.add_argument("--use-value-head", action="store_true", help="Enable value head for position evaluation training")
    parser.add_argument("--value-loss-weight", type=float, default=1.0, help="Weight for value head loss (default: 1.0)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"], help="Optimizer: 'adamw' or 'muon' (Muon for 2D weights + AdamW for rest, requires PyTorch >=2.10)")
    parser.add_argument("--lr-value-head", type=float, default=None, help="Learning rate for value head (default: 10x --lr when fine-tuning with value head, else same as --lr-head)")
    parser.add_argument("--muon-momentum", type=float, default=0.95, help="Momentum for Muon optimizer (default: 0.95)")
    args = parser.parse_args()

    set_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # Setup dataset and dataloaders
    dataset = HDF5ChessDataset(
        hdf5_path=args.data,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
    )

    MAX_VALIDATION_SAMPLES = 10000

    val_size = min(int(0.1 * len(dataset)), MAX_VALIDATION_SAMPLES)
    test_size = min(int(0.1 * len(dataset)), MAX_VALIDATION_SAMPLES)
    train_size = len(dataset) - val_size - test_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=12, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=12)

    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}, Test samples: {len(test_set)}")

    # Model configuration
    model_config = {
        "vocab_size": dataset.position_tokenizer.vocab_size,
        "move_vocab_size": dataset.move_tokenizer.vocab_size,
        "embed_dim": 512,
        "nb_transformer_layers": 8,
        "num_heads": 8,
        "num_kv_groups": 4,  # GQA: 8 heads / 4 groups = 2 heads per KV group
        "dropout": 0.1,
        "kvq_bias": False,
        "mask_future": False,
        "rope": True,  # Use RoPE for positional encoding
        "use_swiglu": True,  # Use SwiGLU activation (Llama-style)
        "use_col_row_emb": True,  # Use separate column and row embeddings
        "use_value_head": args.use_value_head,  # Enable value head for position evaluation
    }

    # Setup logging
    log_dir = Path("logs") / "position2move"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_number = get_next_run_number(str(log_dir))
    log_dir_full =  f"run_{run_number:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = log_dir / log_dir_full
    log_path.mkdir(parents=True, exist_ok=False)
    print(f"Logging to: {log_path}")

    # Create checkpoints directory
    checkpoint_dir = log_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save model config to JSON
    config_path = log_path / "model_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"Model config saved to: {config_path}")

    precision = args.precision if torch.cuda.is_available() or torch.backends.mps.is_available() else "fp32"
    print(f"Using precision: {precision}")

    accelerator = accelerate.Accelerator(log_with="tensorboard", project_dir=str(log_path), mixed_precision=precision)
    
    # Pre-compute discriminative learning rates (needed for tracker config below)
    use_muon = args.optimizer == "muon"
    lr_transformer = args.lr  # Muon LR for 2D weights, AdamW LR otherwise

    # When using Muon, AdamW params (embeddings, norms, biases) need a much
    # smaller LR.  Typical Muon LR ≈ 0.02, typical AdamW LR ≈ 3e-4.
    # Auto-scale with a 0.015× ratio unless the user explicitly overrides.
    _muon_adamw_ratio = 0.015  # 0.02 * 0.015 = 3e-4
    if use_muon:
        _adamw_default_lr = args.lr * _muon_adamw_ratio
        lr_embedding = args.lr_embedding if args.lr_embedding is not None else _adamw_default_lr
        lr_head = args.lr_head if args.lr_head is not None else _adamw_default_lr
    else:
        lr_embedding = args.lr_embedding if args.lr_embedding is not None else args.lr
        lr_head = args.lr_head if args.lr_head is not None else args.lr

    # Value head LR: use explicit flag, or 10x base LR when fine-tuning (value head is randomly initialised)
    if args.lr_value_head is not None:
        lr_value_head = args.lr_value_head
    elif args.use_value_head and args.resume_from:
        lr_value_head = (args.lr if use_muon else args.lr * 10)  # Muon already fast; don't 10× it
    else:
        lr_value_head = lr_head

    # Initialize trackers
    accelerator.init_trackers(
        project_name="position2move",
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "lr_embedding": lr_embedding,
            "lr_head": lr_head,
            "min_elo": args.min_elo,
            "max_elo": args.max_elo,
            "max_grad_norm": args.max_grad_norm,
            **model_config,
            "warmup_steps": args.warmup_steps,
            "final_lr_ratio": args.final_lr_ratio,
            "precision": args.precision,
            "value_loss_weight": args.value_loss_weight if args.use_value_head else 0.0,
            "optimizer": args.optimizer,
            "lr_value_head": lr_value_head if args.use_value_head else 0.0,
        }
    )

    device = accelerator.device
    print(f"Using device: {device}")

    model = Position2MoveModel(**model_config)
    
    # torch.compile has limited support on MPS backend - disable if compilation fails
    if device.type == "mps":
        print("Warning: Skipping torch.compile on MPS device (Mac GPU) due to backend limitations")
        print("Training will proceed without compilation optimizations")
    else:
        try:
            model = torch.compile(model, )
            print("Model compiled successfully")
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}), proceeding without compilation")
    
    model.to(device)

    if args.lr_find:
        if accelerator.is_main_process:
            print("Running learning rate finder...")
        best_lr, best_loss = find_best_learning_rate(
            model,
            train_loader,
            accelerator,
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            num_steps=args.lr_steps,
            beta=args.beta,
            legal_move_loss_weight=args.legal_move_loss_weight,
            use_value_head=args.use_value_head,
            value_loss_weight=args.value_loss_weight if args.use_value_head else 0.0,
        )
        accelerator.log({"lr_finder/best_lr": best_lr, "lr_finder/best_loss": best_loss}, step=0)
        args.lr = best_lr
        if accelerator.is_main_process:
            print(f"Best LR found: {best_lr:.6g} (loss {best_loss:.4f})")

    # Get underlying model (unwrap from accelerator if needed)
    unwrapped_model = accelerator.unwrap_model(model)
    
    if use_muon:
        from torch.optim import Muon
        # Muon: 2D Linear weights → Muon optimizer
        # Everything else (embeddings, norms, biases, cls_token, 1D params) → AdamW
        muon_params = []          # 2D weight matrices for Muon
        adamw_embedding_params = []  # embeddings → AdamW
        adamw_head_params = []       # lm_head non-2D → AdamW
        adamw_value_head_params = [] # value_head non-2D → AdamW (separate LR)
        adamw_other_params = []      # norms, biases, cls_token → AdamW
        muon_value_head_params = []  # value_head 2D weights → Muon (separate LR)
        
        for name, param in unwrapped_model.named_parameters():
            if not param.requires_grad:
                continue
            is_2d = param.ndim == 2
            if 'value_head' in name:
                if is_2d:
                    muon_value_head_params.append(param)
                else:
                    adamw_value_head_params.append(param)
            elif 'embedding' in name or 'cls_token' in name:
                adamw_embedding_params.append(param)
            elif 'lm_head' in name:
                if is_2d:
                    muon_params.append(param)
                else:
                    adamw_head_params.append(param)
            elif is_2d:
                muon_params.append(param)
            else:
                adamw_other_params.append(param)
        
        # AdamW for non-Muon params
        _adamw_norms_lr = _adamw_default_lr  # norms/biases use scaled-down LR
        adamw_groups = [
            {'params': adamw_embedding_params, 'lr': lr_embedding, 'name': 'adamw_embeddings'},
            {'params': adamw_other_params, 'lr': _adamw_norms_lr, 'name': 'adamw_norms_biases'},
            {'params': adamw_head_params, 'lr': lr_head, 'name': 'adamw_head'},
        ]
        if adamw_value_head_params:
            adamw_groups.append({'params': adamw_value_head_params, 'lr': lr_value_head, 'name': 'adamw_value_head'})
        adamw_groups = [g for g in adamw_groups if g['params']]
        
        # Muon for 2D weight matrices
        muon_groups = [
            {'params': muon_params, 'lr': lr_transformer, 'name': 'muon_weights'},
        ]
        if muon_value_head_params:
            muon_groups.append({'params': muon_value_head_params, 'lr': lr_value_head, 'name': 'muon_value_head'})
        muon_groups = [g for g in muon_groups if g['params']]
        
        muon_optimizer = Muon(muon_groups, lr=lr_transformer, momentum=args.muon_momentum)
        adamw_optimizer = torch.optim.AdamW(adamw_groups, lr=lr_transformer)
        
        class DualOptimizer:
            """Wraps Muon + AdamW so the training loop can call a single optimizer."""
            def __init__(self, muon_opt, adamw_opt):
                self.muon = muon_opt
                self.adamw = adamw_opt
                # Expose param_groups for LR scheduler (union of both)
                self.param_groups = muon_opt.param_groups + adamw_opt.param_groups
            def zero_grad(self, set_to_none=True):
                self.muon.zero_grad(set_to_none=set_to_none)
                self.adamw.zero_grad(set_to_none=set_to_none)
            def step(self, closure=None):
                self.muon.step(closure)
                self.adamw.step(closure)
            def state_dict(self):
                return {'muon': self.muon.state_dict(), 'adamw': self.adamw.state_dict()}
            def load_state_dict(self, state_dict):
                self.muon.load_state_dict(state_dict['muon'])
                self.adamw.load_state_dict(state_dict['adamw'])

        class DualScheduler:
            """Wraps two LambdaLR schedulers for Muon + AdamW.
            
            get_last_lr() returns a semantically ordered list:
              [emb_lr, transformer_lr, head_lr, (value_head_lr)]
            so that logging indices [0],[1],[2] stay consistent with the
            plain-AdamW path.
            """
            def __init__(self, muon_scheduler, adamw_scheduler):
                self.muon_scheduler = muon_scheduler
                self.adamw_scheduler = adamw_scheduler
            def step(self):
                self.muon_scheduler.step()
                self.adamw_scheduler.step()
            def get_last_lr(self):
                # AdamW groups order: emb, norms/biases, head, (value_head)
                # Muon groups order: weights, (value_head)
                # Return semantic order: emb_lr, transformer_lr, head_lr, (value_head_lr)
                adamw_lrs = self.adamw_scheduler.get_last_lr()
                muon_lrs = self.muon_scheduler.get_last_lr()
                # emb comes from adamw[0], transformer from muon[0], head from adamw[2]
                result = [adamw_lrs[0], muon_lrs[0], adamw_lrs[2] if len(adamw_lrs) > 2 else adamw_lrs[-1]]
                # If value head exists, average the muon and adamw value_head LRs
                if len(muon_lrs) > 1:
                    result.append(muon_lrs[1])
                elif len(adamw_lrs) > 3:
                    result.append(adamw_lrs[3])
                return result
            @property
            def last_epoch(self):
                return self.muon_scheduler.last_epoch
            @last_epoch.setter
            def last_epoch(self, value):
                self.muon_scheduler.last_epoch = value
                self.adamw_scheduler.last_epoch = value
            def state_dict(self):
                return {'muon': self.muon_scheduler.state_dict(), 'adamw': self.adamw_scheduler.state_dict()}
            def load_state_dict(self, state_dict):
                self.muon_scheduler.load_state_dict(state_dict['muon'])
                self.adamw_scheduler.load_state_dict(state_dict['adamw'])
        
        optimizer = DualOptimizer(muon_optimizer, adamw_optimizer)
        
        if accelerator.is_main_process:
            muon_count = sum(p.numel() for g in muon_groups for p in g['params'])
            adamw_count = sum(p.numel() for g in adamw_groups for p in g['params'])
            print(f"\nOptimizer: Muon + AdamW (hybrid)")
            print(f"  Muon params (2D weights): {muon_count:,}")
            print(f"  AdamW params (emb/norm/bias): {adamw_count:,}")
            print(f"  Muon LR: {lr_transformer:.6g}, momentum: {args.muon_momentum}")
            print(f"  AdamW LRs: emb={lr_embedding:.6g}, norms={_adamw_norms_lr:.6g}, head={lr_head:.6g}")
            if args.use_value_head:
                print(f"  Value head LR: {lr_value_head:.6g}")
    else:
        # Standard AdamW with discriminative learning rates
        embedding_params = []
        transformer_params = []
        head_params = []
        value_head_params = []
        
        for name, param in unwrapped_model.named_parameters():
            if 'embedding' in name:
                embedding_params.append(param)
            elif 'value_head' in name:
                value_head_params.append(param)
            elif 'lm_head' in name:
                head_params.append(param)
            else:
                transformer_params.append(param)
        
        param_groups = [
            {'params': embedding_params, 'lr': lr_embedding, 'name': 'embeddings'},
            {'params': transformer_params, 'lr': lr_transformer, 'name': 'transformers'},
            {'params': head_params, 'lr': lr_head, 'name': 'head'},
        ]
        if value_head_params:
            param_groups.append({'params': value_head_params, 'lr': lr_value_head, 'name': 'value_head'})
        
        if accelerator.is_main_process:
            emb_count = sum(p.numel() for p in embedding_params)
            trans_count = sum(p.numel() for p in transformer_params)
            head_count = sum(p.numel() for p in head_params)
            vh_count = sum(p.numel() for p in value_head_params)
            print(f"\nOptimizer: AdamW")
            print(f"  Embeddings:   {lr_embedding:.6g} ({emb_count:,} params)")
            print(f"  Transformers: {lr_transformer:.6g} ({trans_count:,} params)")
            print(f"  Head:         {lr_head:.6g} ({head_count:,} params)")
            if vh_count:
                print(f"  Value head:   {lr_value_head:.6g} ({vh_count:,} params)")
        
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    if use_muon:
        # Accelerate needs real Optimizer objects — prepare each separately
        model, train_loader, val_loader, test_loader = accelerator.prepare(
            model, train_loader, val_loader, test_loader
        )
        optimizer.muon = accelerator.prepare(optimizer.muon)
        optimizer.adamw = accelerator.prepare(optimizer.adamw)
        optimizer.param_groups = optimizer.muon.param_groups + optimizer.adamw.param_groups
    else:
        model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader
        )
    
    # Check if resuming to load scheduler config from checkpoint
    best_val_loss = float("inf")
    start_epoch = 1
    global_step = 0
    resume_path = Path(args.resume_from) if args.resume_from else None
    trainer_state = None
    
    if resume_path:
        trainer_state = load_trainer_state(resume_path)
        if trainer_state:
            start_epoch = trainer_state.get("epoch", 0) + 1
            global_step = trainer_state.get("global_step", 0)
            best_val_loss = trainer_state.get("best_val_loss", float("inf"))
    
    # Determine scheduler parameters - use saved config when resuming for consistency
    has_scheduler_config = trainer_state and "scheduler_config" in trainer_state
    
    if has_scheduler_config:
        saved_config = trainer_state["scheduler_config"]
        scheduler_warmup_steps = saved_config["warmup_steps"]
        scheduler_total_steps = saved_config["total_steps"]
        scheduler_final_lr_ratio = saved_config["final_lr_ratio"]
        if accelerator.is_main_process:
            print(f"Using scheduler config from checkpoint: warmup={scheduler_warmup_steps}, total={scheduler_total_steps}, final_lr_ratio={scheduler_final_lr_ratio}")
    else:
        scheduler_warmup_steps = args.warmup_steps
        scheduler_total_steps = max(1, len(train_loader)) * max(1, args.epochs)
        scheduler_final_lr_ratio = args.final_lr_ratio
        if resume_path and accelerator.is_main_process:
            print(f"Old checkpoint without scheduler_config - using new scheduler params: warmup={scheduler_warmup_steps}, total={scheduler_total_steps}")
    
    # Store scheduler config for saving in checkpoints
    scheduler_config = {
        "warmup_steps": scheduler_warmup_steps,
        "total_steps": scheduler_total_steps,
        "final_lr_ratio": scheduler_final_lr_ratio,
    }
    
    if use_muon:
        muon_scheduler = create_linear_warmup_decay_scheduler(
            optimizer.muon,
            warmup_steps=scheduler_warmup_steps,
            total_steps=scheduler_total_steps,
            final_lr_ratio=scheduler_final_lr_ratio,
        )
        adamw_scheduler = create_linear_warmup_decay_scheduler(
            optimizer.adamw,
            warmup_steps=scheduler_warmup_steps,
            total_steps=scheduler_total_steps,
            final_lr_ratio=scheduler_final_lr_ratio,
        )
        scheduler = DualScheduler(muon_scheduler, adamw_scheduler)
        accelerator.register_for_checkpointing(muon_scheduler)
        accelerator.register_for_checkpointing(adamw_scheduler)
    else:
        scheduler = create_linear_warmup_decay_scheduler(
            optimizer,
            warmup_steps=scheduler_warmup_steps,
            total_steps=scheduler_total_steps,
            final_lr_ratio=scheduler_final_lr_ratio,
        )
        # Always register scheduler for checkpointing (required for load_state to work)
        accelerator.register_for_checkpointing(scheduler)

    # Load the full accelerator state (model, optimizer, scheduler) if resuming
    if resume_path:
        # When adding a value head to a checkpoint that was saved without one,
        # accelerator.load_state() would fail on strict key matching.
        # In that case, load only the model weights with strict=False so the
        # new value_head parameters stay randomly initialised.
        _needs_partial_load = False
        if args.use_value_head:
            # Peek at the saved model state to check for value_head keys
            import glob, os
            _sf_pattern = os.path.join(str(resume_path), "**", "*.safetensors")
            _sf_files = glob.glob(_sf_pattern, recursive=True)
            if _sf_files:
                from safetensors.torch import load_file as _load_sf
                _keys = list(_load_sf(_sf_files[0], device="cpu").keys())
                _needs_partial_load = not any("value_head" in k for k in _keys)
            else:
                # Try .bin format
                _bin_pattern = os.path.join(str(resume_path), "**", "*.bin")
                _bin_files = glob.glob(_bin_pattern, recursive=True)
                if _bin_files:
                    _keys = list(torch.load(_bin_files[0], map_location="cpu", weights_only=True).keys())
                    _needs_partial_load = not any("value_head" in k for k in _keys)

        if _needs_partial_load:
            # Load model weights only (strict=False), skip optimizer/scheduler
            from safetensors.torch import load_file as _load_sf
            _state = _load_sf(_sf_files[0], device="cpu")
            _unwrapped = accelerator.unwrap_model(model)
            missing, unexpected = _unwrapped.load_state_dict(_state, strict=False)
            if accelerator.is_main_process:
                print(f"Partial checkpoint load (value head is new):")
                print(f"  Missing keys (randomly initialised): {missing}")
                if unexpected:
                    print(f"  Unexpected keys (ignored): {unexpected}")
                print(f"  Optimizer and scheduler start fresh.")
            # Reset resume state — we're not truly resuming, just warm-starting the backbone
            start_epoch = 1
            global_step = 0
            best_val_loss = float("inf")
        else:
            accelerator.load_state(str(resume_path))
        
        # For old checkpoints without scheduler_config, the loaded scheduler state 
        # has a mismatched lr_lambda (old total_steps). We need to recreate it.
        if not _needs_partial_load and not has_scheduler_config:
            # Recreate scheduler with correct parameters
            if use_muon:
                muon_scheduler = create_linear_warmup_decay_scheduler(
                    optimizer.muon, warmup_steps=scheduler_warmup_steps,
                    total_steps=scheduler_total_steps, final_lr_ratio=scheduler_final_lr_ratio,
                )
                adamw_scheduler = create_linear_warmup_decay_scheduler(
                    optimizer.adamw, warmup_steps=scheduler_warmup_steps,
                    total_steps=scheduler_total_steps, final_lr_ratio=scheduler_final_lr_ratio,
                )
                scheduler = DualScheduler(muon_scheduler, adamw_scheduler)
            else:
                scheduler = create_linear_warmup_decay_scheduler(
                    optimizer, warmup_steps=scheduler_warmup_steps,
                    total_steps=scheduler_total_steps, final_lr_ratio=scheduler_final_lr_ratio,
                )
            # Manually advance to the correct step
            scheduler.last_epoch = global_step - 1  # -1 because step() will increment it
            scheduler.step()  # This sets it to global_step and computes correct LR
            if accelerator.is_main_process:
                current_lrs = scheduler.get_last_lr()
                print(f"  Recreated scheduler and advanced to step {global_step}")
                print(f"  Current LRs: emb={current_lrs[0]:.6g}, trans={current_lrs[1]:.6g}, head={current_lrs[2]:.6g}")
        
        if accelerator.is_main_process:
            print(f"Resumed from checkpoint: {resume_path}")
            print(f"  Starting from epoch {start_epoch}, global_step {global_step}, best_val_loss {best_val_loss:.4f}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_llm_loss = 0.0
        train_legal_move_loss = 0.0
        train_legal_move_ce = 0.0
        train_value_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} - Training", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            positions = batch['position']
            moves = batch['move']
            is_white = batch['is_white']
            legal_moves_mask = batch['legal_moves_mask']
            castling_rights = batch['castling_rights']
            en_passant_file = batch['en_passant_file']
            halfmove_clock = batch['halfmove_clock']

            output = model(positions, is_white, castling_rights, en_passant_file, halfmove_clock)
            
            value = None
            value_target = None
            if args.use_value_head:
                logits, value = output
                result = batch['result']
                value_target = result_to_value_target(result, is_white)
            else:
                logits = output

            loss, llm_loss, legal_move_loss, legal_move_cross_entropy, v_loss = compute_loss(
                logits, moves, legal_moves_mask, args.beta, args.legal_move_loss_weight,
                value=value, value_target=value_target,
                value_loss_weight=args.value_loss_weight if args.use_value_head else 0.0,
            )

            accelerator.backward(loss)
            
            # Gradient clipping — skip Muon params (Newton-Schulz already normalises)
            if args.max_grad_norm > 0:
                if use_muon:
                    # Only clip AdamW parameters (embeddings, norms, biases)
                    adamw_params = [p for g in optimizer.adamw.param_groups for p in g['params']]
                    accelerator.clip_grad_norm_(adamw_params, args.max_grad_norm)
                else:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Compute gradient norms for monitoring (every 100 steps to avoid overhead)
            if global_step % 100 == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                _grad_norm = total_norm.item()
            else:
                _grad_norm = None
            
            optimizer.step()
            scheduler.step()

            batch_size = positions.size(0)
            train_loss += loss.item() * batch_size
            train_llm_loss += llm_loss.item() * batch_size
            train_legal_move_loss += legal_move_loss.item() * batch_size
            train_legal_move_ce += legal_move_cross_entropy.item() * batch_size
            train_value_loss += v_loss.item() * batch_size
            
            postfix = {
                "loss": loss.item(), 
                "llm_loss": llm_loss.item(), 
                "legal_move_loss": legal_move_loss.item(), 
                "legal_move_ce": legal_move_cross_entropy.item(),
            }
            if args.use_value_head:
                postfix["value_loss"] = v_loss.item()
            progress_bar.set_postfix(postfix)
            
            # Log step-level metrics
            current_lrs = scheduler.get_last_lr()
            log_dict = {
                "train/step_loss": loss.item(),
                "train/step_llm_loss": llm_loss.item(),
                "train/step_legal_move_loss": legal_move_loss.item(),
                "train/step_legal_move_ce": legal_move_cross_entropy.item(),
                "train/lr_embeddings": current_lrs[0],
                "train/lr_transformers": current_lrs[1],
                "train/lr_head": current_lrs[2],
            }
            if args.use_value_head:
                log_dict["train/step_value_loss"] = v_loss.item()
            if _grad_norm is not None:
                log_dict["train/grad_norm"] = _grad_norm
            accelerator.log(log_dict, step=global_step)
            
            global_step += 1
            
            # Save step checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                step_checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step:07d}"
                accelerator.save_state(str(step_checkpoint_path))
                save_trainer_state(step_checkpoint_path, epoch, global_step, best_val_loss, scheduler_config)
                if accelerator.is_main_process:
                    print(f"\n  Saved step checkpoint at step {global_step}")
                    cleanup_old_step_checkpoints(checkpoint_dir, args.max_checkpoints)

        # Calculate average training metrics
        train_loss /= len(train_set)
        train_llm_loss /= len(train_set)
        train_legal_move_loss /= len(train_set)
        train_legal_move_ce /= len(train_set)
        train_value_loss /= len(train_set)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_llm_loss = 0.0
        val_legal_move_loss = 0.0
        val_legal_move_ce = 0.0
        val_value_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} - Validation", leave=False)
            for batch in progress_bar:
                positions = batch['position']
                moves = batch['move']
                is_white = batch['is_white']
                legal_moves_mask = batch['legal_moves_mask']
                castling_rights = batch['castling_rights']
                en_passant_file = batch['en_passant_file']
                halfmove_clock = batch['halfmove_clock']

                output = model(positions, is_white, castling_rights, en_passant_file, halfmove_clock)
                
                value = None
                value_target = None
                if args.use_value_head:
                    logits, value = output
                    result = batch['result']
                    value_target = result_to_value_target(result, is_white)
                else:
                    logits = output

                loss, llm_loss, legal_move_loss, legal_move_cross_entropy, v_loss = compute_loss(
                    logits, moves, legal_moves_mask, args.beta, args.legal_move_loss_weight,
                    value=value, value_target=value_target,
                    value_loss_weight=args.value_loss_weight if args.use_value_head else 0.0,
                )

                batch_size = positions.size(0)
                val_loss += loss.item() * batch_size
                val_llm_loss += llm_loss.item() * batch_size
                val_legal_move_loss += legal_move_loss.item() * batch_size
                val_legal_move_ce += legal_move_cross_entropy.item() * batch_size
                val_value_loss += v_loss.item() * batch_size
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                val_correct += (predictions == moves).sum().item()
                val_total += batch_size
                
                progress_bar.set_postfix({"val_loss": loss.item()})

        val_loss /= len(val_set)
        val_llm_loss /= len(val_set)
        val_legal_move_loss /= len(val_set)
        val_legal_move_ce /= len(val_set)
        val_value_loss /= len(val_set)
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0

        # Log epoch-level metrics
        epoch_log = {
            "train/epoch_loss": train_loss,
            "train/epoch_llm_loss": train_llm_loss,
            "train/epoch_legal_move_loss": train_legal_move_loss,
            "train/epoch_legal_move_ce": train_legal_move_ce,
            "val/loss": val_loss,
            "val/llm_loss": val_llm_loss,
            "val/legal_move_loss": val_legal_move_loss,
            "val/legal_move_ce": val_legal_move_ce,
            "val/accuracy": val_accuracy,
            "epoch": epoch,
            "train/lr_embeddings_epoch_end": scheduler.get_last_lr()[0],
            "train/lr_transformers_epoch_end": scheduler.get_last_lr()[1],
            "train/lr_head_epoch_end": scheduler.get_last_lr()[2],
        }
        if args.use_value_head:
            epoch_log["train/epoch_value_loss"] = train_value_loss
            epoch_log["val/value_loss"] = val_value_loss
        accelerator.log(epoch_log, step=global_step)

        print(f"\nEpoch {epoch}/{args.epochs}:")
        value_info = f" | Val Value Loss: {val_value_loss:.4f}" if args.use_value_head else ""
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}{value_info}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoint_dir / "best_model"
            accelerator.save_state(str(best_checkpoint_path))
            save_trainer_state(best_checkpoint_path, epoch, global_step, best_val_loss, scheduler_config)
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

        # Periodic checkpoint saving
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}"
            accelerator.save_state(str(checkpoint_path))
            save_trainer_state(checkpoint_path, epoch, global_step, best_val_loss, scheduler_config)
            print(f"  Saved checkpoint at epoch {epoch}")

    # Final test evaluation
    print("\nRunning final test evaluation...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_value_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in progress_bar:
            positions = batch['position']
            moves = batch['move']
            is_white = batch['is_white']
            legal_moves_mask = batch['legal_moves_mask']
            castling_rights = batch['castling_rights']
            en_passant_file = batch['en_passant_file']
            halfmove_clock = batch['halfmove_clock']

            output = model(positions, is_white, castling_rights, en_passant_file, halfmove_clock)
            
            value = None
            value_target = None
            if args.use_value_head:
                logits, value = output
                result = batch['result']
                value_target = result_to_value_target(result, is_white)
            else:
                logits = output

            loss, llm_loss, legal_move_loss, legal_move_cross_entropy, v_loss = compute_loss(
                logits, moves, legal_moves_mask, args.beta, args.legal_move_loss_weight,
                value=value, value_target=value_target,
                value_loss_weight=args.value_loss_weight if args.use_value_head else 0.0,
            )

            batch_size = positions.size(0)
            test_loss += loss.item() * batch_size
            test_value_loss += v_loss.item() * batch_size
            
            predictions = torch.argmax(logits, dim=-1)
            test_correct += (predictions == moves).sum().item()
            test_total += batch_size

    test_loss /= len(test_set)
    test_accuracy = test_correct / test_total if test_total > 0 else 0.0
    test_value_loss /= len(test_set)

    test_log = {
        "test/loss": test_loss,
        "test/accuracy": test_accuracy,
    }
    if args.use_value_head:
        test_log["test/value_loss"] = test_value_loss
    accelerator.log(test_log, step=global_step)

    print(f"\nTest Results:")
    value_info = f" | Test Value Loss: {test_value_loss:.4f}" if args.use_value_head else ""
    print(f"  Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}{value_info}")

    # End tracking
    accelerator.end_training()
    print(f"\nTraining complete! Logs saved to: {log_path}")


if __name__ == "__main__":
    main()
