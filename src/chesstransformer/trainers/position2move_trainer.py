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

def compute_loss(logits, moves, legal_moves_mask, beta, legal_move_loss_weight):
    llm_loss = F.cross_entropy(logits, moves)
    illegal_prob = F.softmax(logits, dim=-1) * (1. - legal_moves_mask.float())
    legal_move_loss = torch.sum(illegal_prob ** 2, dim=-1).mean()
    filtered_logits = torch.where(legal_moves_mask.bool(), logits, -torch.inf)
    legal_move_cross_entropy = F.cross_entropy(filtered_logits, moves)
    loss = beta * llm_loss + legal_move_loss_weight * legal_move_loss + (1 - beta) * legal_move_cross_entropy
    return loss, llm_loss, legal_move_loss, legal_move_cross_entropy

def save_trainer_state(checkpoint_dir, epoch, global_step, best_val_loss):
    state = {"epoch": epoch, "global_step": global_step, "best_val_loss": best_val_loss}
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    with state_path.open("w", encoding="utf-8") as fp:
        json.dump(state, fp)

def load_trainer_state(checkpoint_dir):
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.exists():
        return None
    with state_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)

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

def find_best_learning_rate(model, train_loader, accelerator, lr_min=1e-6, lr_max=1e-2, num_steps=100, beta=0.7, legal_move_loss_weight=10.0):
    model.train()
    device = accelerator.device
    initial_state = copy.deepcopy(model.state_dict())
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

        logits = model(positions, is_white)
        loss, llm_loss, legal_move_loss, legal_move_cross_entropy = compute_loss(
            logits, moves, legal_moves_mask, beta, legal_move_loss_weight
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
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
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
    val_size = int(0.1 * len(dataset))
    test_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}, Test samples: {len(test_set)}")

    # Model configuration
    model_config = {
        "vocab_size": dataset.position_tokenizer.vocab_size,
        "move_vocab_size": dataset.move_tokenizer.vocab_size,
        "embed_dim": 512,
        "nb_transformer_layers": 8,
        "num_heads": 8,
        "dropout": 0.1,
        "kvq_bias": False,
        "mask_future": False,
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

    accelerator = accelerate.Accelerator(log_with="tensorboard", project_dir=str(log_path))
    
    # Initialize trackers
    accelerator.init_trackers(
        project_name="position2move",
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "min_elo": args.min_elo,
            "max_elo": args.max_elo,
            "max_grad_norm": args.max_grad_norm,
            **model_config,
            "warmup_steps": args.warmup_steps,
            "final_lr_ratio": args.final_lr_ratio,
        }
    )

    device = accelerator.device
    print(f"Using device: {device}")

    model = Position2MoveModel(**model_config)
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
        )
        accelerator.log({"lr_finder/best_lr": best_lr, "lr_finder/best_loss": best_loss}, step=0)
        args.lr = best_lr
        if accelerator.is_main_process:
            print(f"Best LR found: {best_lr:.6g} (loss {best_loss:.4f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    total_training_steps = max(1, len(train_loader)) * max(1, args.epochs)
    scheduler = create_linear_warmup_decay_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_training_steps,
        final_lr_ratio=args.final_lr_ratio,
    )
    accelerator.register_for_checkpointing(scheduler)

    best_val_loss = float("inf")
    start_epoch = 1
    global_step = 0
    resume_path = Path(args.resume_from) if args.resume_from else None
    if resume_path:
        accelerator.load_state(str(resume_path))
        trainer_state = load_trainer_state(resume_path)
        if trainer_state:
            start_epoch = trainer_state.get("epoch", 0) + 1
            global_step = trainer_state.get("global_step", 0)
            best_val_loss = trainer_state.get("best_val_loss", float("inf"))
        if accelerator.is_main_process:
            print(f"Resumed from checkpoint: {resume_path}")

    # Training loop
    best_val_loss = float("inf")
    global_step = 0
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_llm_loss = 0.0
        train_legal_move_loss = 0.0
        train_legal_move_ce = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} - Training", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            positions = batch['position']
            moves = batch['move']
            is_white = batch['is_white']
            legal_moves_mask = batch['legal_moves_mask']

            logits = model(positions, is_white)
            loss, llm_loss, legal_move_loss, legal_move_cross_entropy = compute_loss(
                logits, moves, legal_moves_mask, args.beta, args.legal_move_loss_weight
            )

            accelerator.backward(loss)
            
            # Gradient clipping
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()

            batch_size = positions.size(0)
            train_loss += loss.item() * batch_size
            train_llm_loss += llm_loss.item() * batch_size
            train_legal_move_loss += legal_move_loss.item() * batch_size
            train_legal_move_ce += legal_move_cross_entropy.item() * batch_size
            
            progress_bar.set_postfix({
                "loss": loss.item(), 
                "llm_loss": llm_loss.item(), 
                "legal_move_loss": legal_move_loss.item(), 
                "legal_move_ce": legal_move_cross_entropy.item()
            })
            
            # Log step-level metrics
            accelerator.log({
                "train/step_loss": loss.item(),
                "train/step_llm_loss": llm_loss.item(),
                "train/step_legal_move_loss": legal_move_loss.item(),
                "train/step_legal_move_ce": legal_move_cross_entropy.item(),
                "train/lr": scheduler.get_last_lr()[0],
            }, step=global_step)
            
            global_step += 1

        # Calculate average training metrics
        train_loss /= len(train_set)
        train_llm_loss /= len(train_set)
        train_legal_move_loss /= len(train_set)
        train_legal_move_ce /= len(train_set)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_llm_loss = 0.0
        val_legal_move_loss = 0.0
        val_legal_move_ce = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} - Validation", leave=False)
            for batch in progress_bar:
                positions = batch['position']
                moves = batch['move']
                is_white = batch['is_white']
                legal_moves_mask = batch['legal_moves_mask']

                logits = model(positions, is_white)
                loss, llm_loss, legal_move_loss, legal_move_cross_entropy = compute_loss(
                    logits, moves, legal_moves_mask, args.beta, args.legal_move_loss_weight
                )

                batch_size = positions.size(0)
                val_loss += loss.item() * batch_size
                val_llm_loss += llm_loss.item() * batch_size
                val_legal_move_loss += legal_move_loss.item() * batch_size
                val_legal_move_ce += legal_move_cross_entropy.item() * batch_size
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                val_correct += (predictions == moves).sum().item()
                val_total += batch_size
                
                progress_bar.set_postfix({"val_loss": loss.item()})

        val_loss /= len(val_set)
        val_llm_loss /= len(val_set)
        val_legal_move_loss /= len(val_set)
        val_legal_move_ce /= len(val_set)
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0

        # Log epoch-level metrics
        accelerator.log({
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
            "train/lr_epoch_end": scheduler.get_last_lr()[0],
        }, step=global_step)

        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoint_dir / "best_model"
            accelerator.save_state(str(best_checkpoint_path))
            save_trainer_state(best_checkpoint_path, epoch, global_step, best_val_loss)
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

        # Periodic checkpoint saving
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}"
            accelerator.save_state(str(checkpoint_path))
            save_trainer_state(checkpoint_path, epoch, global_step, best_val_loss)
            print(f"  Saved checkpoint at epoch {epoch}")

    # Final test evaluation
    print("\nRunning final test evaluation...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in progress_bar:
            positions = batch['position']
            moves = batch['move']
            is_white = batch['is_white']
            legal_moves_mask = batch['legal_moves_mask']

            logits = model(positions, is_white)
            loss, llm_loss, legal_move_loss, legal_move_cross_entropy = compute_loss(
                logits, moves, legal_moves_mask, args.beta, args.legal_move_loss_weight
            )

            batch_size = positions.size(0)
            test_loss += loss.item() * batch_size
            
            predictions = torch.argmax(logits, dim=-1)
            test_correct += (predictions == moves).sum().item()
            test_total += batch_size

    test_loss /= len(test_set)
    test_accuracy = test_correct / test_total if test_total > 0 else 0.0

    accelerator.log({
        "test/loss": test_loss,
        "test/accuracy": test_accuracy,
    }, step=global_step)

    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

    # End tracking
    accelerator.end_training()
    print(f"\nTraining complete! Logs saved to: {log_path}")


if __name__ == "__main__":
    main()
