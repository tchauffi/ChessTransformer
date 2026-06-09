#!/usr/bin/env python3
"""Training script for Pos2MoveV2.

Trains the v2 transformer with GQA + QK-norm + chess-geometry attention bias,
AlphaZero-style 64×73 action output and value head on HDF5 chess datasets.

Usage:
    python src/chesstransformer/trainers/pos2move_v2_trainer.py --data data/elite_db.h5
"""

from pathlib import Path
from datetime import datetime
import argparse
import json
import random
import shutil

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
import accelerate
from accelerate.utils import set_seed

from chesstransformer.datasets.h5_lichess_dataset import HDF5ChessDataset
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2, NUM_ACTION_PLANES


def get_next_run_number(log_dir: str) -> int:
    log_path = Path(log_dir)
    if not log_path.exists():
        return 1
    run_numbers = []
    for d in log_path.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            try:
                run_numbers.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return max(run_numbers, default=0) + 1


def compute_loss(
    move_logits,
    value,
    from_sq,
    action_plane,
    legal_moves_planes,
    result,
    is_white,
    move_number,
    value_loss_weight,
    value_rampup_halfmoves=40,
    label_smoothing: float = 0.0,
):
    """Total loss = (label-smoothed) policy CE + value MSE.

    Returns
    -------
    total_loss : scalar tensor (the only thing that gets backprop'd)
    metrics    : dict of detached scalar tensors for logging
        - ce          : standard cross-entropy on the full action space
        - legal_ce    : cross-entropy after masking to legal moves only
        - value_loss  : weighted MSE on the value head
        - acc         : top-1 accuracy on the full action space
        - legal_acc   : top-1 accuracy after masking to legal moves only
    """
    B = move_logits.size(0)

    # Flatten (B, 64, 73) -> (B, 64*73). Upcast to fp32 for loss stability under bf16.
    flat_logits = move_logits.view(B, -1).float()
    flat_target = from_sq * NUM_ACTION_PLANES + action_plane
    flat_mask = legal_moves_planes.view(B, -1).float()

    # ── Policy loss: label-smoothed CE over LEGAL moves only ────────────
    if label_smoothing > 0:
        eps = label_smoothing
        legal_count = flat_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        smooth = eps / legal_count
        target_dist = flat_mask * smooth
        idx = flat_target.unsqueeze(1)
        target_dist.scatter_add_(1, idx, torch.full_like(idx, 1.0 - eps, dtype=target_dist.dtype))
        log_pred = F.log_softmax(flat_logits, dim=-1)
        policy_loss = F.kl_div(log_pred, target_dist, reduction="batchmean")
    else:
        policy_loss = F.cross_entropy(flat_logits, flat_target)

    # ── Value loss: per-sample MSE, ramped up by move number ────────────
    target_value = torch.zeros_like(value)
    white_win = result == 1
    black_win = result == 2
    target_value[white_win & is_white] = 1.0
    target_value[white_win & ~is_white] = -1.0
    target_value[black_win & is_white] = -1.0
    target_value[black_win & ~is_white] = 1.0

    progress = (move_number.float() / value_rampup_halfmoves).clamp(max=1.0)
    per_sample_value_loss = (value.float() - target_value) ** 2
    value_loss = (progress * per_sample_value_loss).mean()

    total_loss = policy_loss + value_loss_weight * value_loss

    # ── Metrics (no grad) ───────────────────────────────────────────────
    with torch.no_grad():
        ce_metric = F.cross_entropy(flat_logits, flat_target)

        masked_logits = flat_logits.masked_fill(flat_mask == 0, float("-inf"))
        legal_ce_metric = F.cross_entropy(masked_logits, flat_target)

        preds = flat_logits.argmax(dim=-1)
        acc = (preds == flat_target).float().mean()

        legal_preds = masked_logits.argmax(dim=-1)
        legal_acc = (legal_preds == flat_target).float().mean()

    metrics = {
        "ce": ce_metric,
        "legal_ce": legal_ce_metric,
        "value_loss": value_loss.detach(),
        "acc": acc,
        "legal_acc": legal_acc,
    }
    return total_loss, metrics


def create_lr_scheduler(optimizer, warmup_steps, total_steps, final_lr_ratio):
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps + 1) / float(max(1, total_steps - warmup_steps))
        return 1.0 - (1.0 - final_lr_ratio) * min(1.0, progress)

    return LambdaLR(optimizer, lr_lambda)


def save_trainer_state(checkpoint_dir, epoch, global_step, best_val_loss, scheduler_config=None):
    state = {"epoch": epoch, "global_step": global_step, "best_val_loss": best_val_loss}
    if scheduler_config:
        state["scheduler_config"] = scheduler_config
    with (Path(checkpoint_dir) / "trainer_state.json").open("w") as f:
        json.dump(state, f)


def load_trainer_state(checkpoint_dir):
    path = Path(checkpoint_dir) / "trainer_state.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


# ── EMA helpers ──────────────────────────────────────────────────────────
# EMA state is kept in fp32 to avoid bf16 precision loss on the
# (1 - decay) * param update term (which is ~0.001 * param at decay=0.999
# and would round to zero for many small values in bf16).

def create_ema_state(model):
    return {name: p.data.detach().float().clone() for name, p in model.named_parameters()}


@torch.no_grad()
def update_ema(model, ema_state, decay):
    for name, param in model.named_parameters():
        # Cast live param to fp32 for the lerp; ema_state stays fp32.
        ema_state[name].lerp_(param.data.float(), 1.0 - decay)


def swap_ema_weights(model, ema_state):
    # Backup current (compute-dtype) weights into ema_state, load EMA weights
    # into the model — cast to the model's dtype on the way in.
    for name, param in model.named_parameters():
        tmp = param.data.detach().float().clone()
        param.data.copy_(ema_state[name].to(param.dtype))
        ema_state[name].copy_(tmp)


def save_ema_state(ema_state, path):
    torch.save(ema_state, path)


def load_ema_state(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def get_raw_model(model, accelerator):
    m = accelerator.unwrap_model(model)
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def cleanup_old_checkpoints(checkpoint_dir, max_checkpoints):
    checkpoints = sorted(
        [d for d in Path(checkpoint_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint_step_")],
        key=lambda x: int(x.name.split("_")[-1]),
    )
    while len(checkpoints) > max_checkpoints:
        shutil.rmtree(checkpoints.pop(0))

class RepeatingDataloader:
    """Repeats the batch N times to avoid underloaded GPU."""
    def __init__(self, dataloader, repeat_factor):
        self.dataloader = dataloader
        self.repeat_factor = repeat_factor

    def __iter__(self):
        for batch in self.dataloader:
            for _ in range(self.repeat_factor):
                yield batch

    def __len__(self):
        return len(self.dataloader) * self.repeat_factor


def main():
    default_data = Path(__file__).parents[3] / "data" / "elite_db.h5"

    parser = argparse.ArgumentParser(description="Train Pos2MoveV2 (clean)")
    # Data
    parser.add_argument("--data", type=str, default=str(default_data))
    parser.add_argument("--min-elo", type=int, default=None)
    parser.add_argument("--max-elo", type=int, default=None)
    parser.add_argument("--sample-weighting", type=str, default="uniform", choices=["uniform", "middlegame"])
    parser.add_argument("--skip-opening-plies", type=int, default=0)
    # Training
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-muon", type=float, default=1e-3)
    parser.add_argument("--lr-embedding", type=float, default=2e-4)
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--final-lr-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    # Loss weights
    parser.add_argument("--value-loss-weight", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    # Model
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)    
    parser.add_argument("--layer-drop", type=float, default=0.0,
                        help="Stochastic depth rate. Linearly scales from 0 (first layer) to this value (last layer). Try 0.1.")    # Checkpointing
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--resume-from", type=str, default=None)
    # EMA
    parser.add_argument("--ema-decay", type=float, default=0.999)
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--compile-mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = HDF5ChessDataset(
        hdf5_path=args.data,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        sample_weighting=args.sample_weighting,
        skip_opening_plies=args.skip_opening_plies,
    )

    MAX_VAL_SAMPLES = 10_000
    val_size = min(int(0.1 * len(dataset)), MAX_VAL_SAMPLES)
    test_size = min(int(0.1 * len(dataset)), MAX_VAL_SAMPLES)
    train_size = len(dataset) - val_size - test_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train: {len(train_set):,} | Val: {len(val_set):,} | Test: {len(test_set):,}")

    # ── Model ────────────────────────────────────────────────────────────
    model_config = {
        "vocab_size": dataset.position_tokenizer.vocab_size,
        "embed_dim": args.embed_dim,
        "nb_transformer_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "kvq_bias": False,
        "layer_drop": args.layer_drop,
    }

    # ── Logging ──────────────────────────────────────────────────────────
    log_dir = Path("logs") / "pos2move_v2"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_number = get_next_run_number(str(log_dir))
    run_name = f"run_{run_number:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = log_dir / run_name
    log_path.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = log_path / "checkpoints"
    checkpoint_dir.mkdir()

    with (log_path / "model_config.json").open("w") as f:
        json.dump(model_config, f, indent=2)

    precision = args.precision if torch.cuda.is_available() else "fp32"
    accelerator = accelerate.Accelerator(
        log_with="tensorboard",
        project_dir=str(log_path),
        mixed_precision=precision,
        gradient_accumulation_steps=args.grad_accum,
    )
    effective_bs = args.batch_size * args.grad_accum
    print(f"Effective batch size: {effective_bs} (micro={args.batch_size} × accum={args.grad_accum})")
    accelerator.init_trackers(
        project_name="pos2move_v2",
        config={
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": effective_bs,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "value_loss_weight": args.value_loss_weight,
            **model_config,
        },
    )

    device = accelerator.device
    print(f"Device: {device} | Precision: {precision} | Logging to: {log_path}")

    model = Pos2MoveV2(**model_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    model.to(device)

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
        print(f"Model compiled with torch.compile (mode={args.compile_mode})")

    # ── Optimizer (Muon for 2D weights, AdamW for embeddings/heads/1D) ──
    lr_emb = args.lr_embedding or args.lr
    lr_head = args.lr_head or args.lr

    muon_params = []
    adamw_emb_params = []
    adamw_head_params = []
    adamw_other_params = []

    for name, param in model.named_parameters():
        if "embedding" in name:
            adamw_emb_params.append(param)
        elif "move_head" in name or "value_head" in name:
            adamw_head_params.append(param)
        elif param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_other_params.append(param)

    print(f"LR: emb={lr_emb:.2e} | muon={args.lr_muon:.2e} | head={lr_head:.2e}")
    print(
        f"Params: emb={sum(p.numel() for p in adamw_emb_params):,} | "
        f"muon={sum(p.numel() for p in muon_params):,} | "
        f"adamw_other={sum(p.numel() for p in adamw_other_params):,} | "
        f"head={sum(p.numel() for p in adamw_head_params):,}"
    )

    muon_optimizer = torch.optim.Muon(
        muon_params,
        lr=args.lr_muon,
        momentum=0.95,
        weight_decay=args.weight_decay,
    )
    adamw_optimizer = torch.optim.AdamW([
        {"params": adamw_emb_params, "lr": lr_emb, "weight_decay": args.weight_decay},
        {"params": adamw_other_params, "lr": args.lr, "weight_decay": 0.0},
        {"params": adamw_head_params, "lr": lr_head, "weight_decay": 0.0},
    ])

    model, muon_optimizer, adamw_optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, muon_optimizer, adamw_optimizer, train_loader, val_loader, test_loader
    )

    # Wrap after prepare so Accelerate handles device placement first
    train_loader = RepeatingDataloader(train_loader, repeat_factor=2)

    # ── EMA ──────────────────────────────────────────────────────────────
    use_ema = args.ema_decay > 0
    ema_state = None
    if use_ema:
        ema_state = create_ema_state(get_raw_model(model, accelerator))
        print(f"EMA enabled (decay={args.ema_decay})")

    # ── Resume ───────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    start_epoch = 1
    global_step = 0
    trainer_state = None

    if args.resume_from:
        trainer_state = load_trainer_state(args.resume_from)
        if trainer_state:
            start_epoch = trainer_state["epoch"] + 1
            global_step = trainer_state["global_step"]
            best_val_loss = trainer_state["best_val_loss"]

    # ── Scheduler ────────────────────────────────────────────────────────
    total_steps_from_epochs = (len(train_loader) * args.epochs) // max(1, args.grad_accum)
    total_steps = args.max_steps if args.max_steps else total_steps_from_epochs
    scheduler_config = {
        "warmup_steps": args.warmup_steps,
        "total_steps": total_steps,
        "final_lr_ratio": args.final_lr_ratio,
    }
    if trainer_state and "scheduler_config" in trainer_state:
        scheduler_config = trainer_state["scheduler_config"]

    muon_scheduler = create_lr_scheduler(muon_optimizer, **scheduler_config)
    adamw_scheduler = create_lr_scheduler(adamw_optimizer, **scheduler_config)
    accelerator.register_for_checkpointing(muon_scheduler, adamw_scheduler)

    if args.resume_from:
        accelerator.load_state(args.resume_from)
        if use_ema:
            ema_path = Path(args.resume_from) / "ema_state.pt"
            if ema_path.exists():
                ema_state = load_ema_state(ema_path, device)
                print(f"Restored EMA state from {ema_path}")
            else:
                ema_state = create_ema_state(get_raw_model(model, accelerator))
                print("EMA state not found in checkpoint, re-initialized from model")
        print(f"Resumed from {args.resume_from} (epoch {start_epoch}, step {global_step})")

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_legal_ce = 0.0
        epoch_value = 0.0
        epoch_acc = 0.0
        epoch_legal_acc = 0.0
        epoch_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            with accelerator.accumulate(model):
                board = batch["position"]
                player = batch["is_white"].long()
                castling = batch["castling_rights"]
                en_passant = batch["en_passant_file"]
                from_sq = batch["from_square"]
                action_plane = batch["action_plane"]
                legal_planes = batch["legal_moves_planes"]
                result = batch["result"]
                is_white = batch["is_white"]
                move_number = batch["move_number"]

                move_logits, value = model(board, player, castling, en_passant)
                loss, metrics = compute_loss(
                    move_logits, value,
                    from_sq, action_plane,
                    legal_planes, result, is_white, move_number,
                    args.value_loss_weight,
                    label_smoothing=args.label_smoothing,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                muon_optimizer.step()
                adamw_optimizer.step()
                if accelerator.sync_gradients:
                    muon_scheduler.step()
                    adamw_scheduler.step()
                muon_optimizer.zero_grad()
                adamw_optimizer.zero_grad()

                if use_ema and accelerator.sync_gradients:
                    update_ema(get_raw_model(model, accelerator), ema_state, args.ema_decay)

            bs = board.size(0)
            epoch_loss += loss.item() * bs
            epoch_ce += metrics["ce"].item() * bs
            epoch_legal_ce += metrics["legal_ce"].item() * bs
            epoch_value += metrics["value_loss"].item() * bs
            epoch_acc += metrics["acc"].item() * bs
            epoch_legal_acc += metrics["legal_acc"].item() * bs
            epoch_samples += bs

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{metrics['ce'].item():.4f}",
                acc=f"{metrics['acc'].item():.3f}",
                val_l=f"{metrics['value_loss'].item():.4f}",
            )

            if not accelerator.sync_gradients:
                continue

            muon_lrs = muon_scheduler.get_last_lr()
            adamw_lrs = adamw_scheduler.get_last_lr()
            accelerator.log(
                {
                    "train/step_loss": loss.item(),
                    "train/step_ce": metrics["ce"].item(),
                    "train/step_legal_ce": metrics["legal_ce"].item(),
                    "train/step_value_loss": metrics["value_loss"].item(),
                    "train/step_acc": metrics["acc"].item(),
                    "train/step_legal_acc": metrics["legal_acc"].item(),
                    "train/lr_muon": muon_lrs[0],
                    "train/lr_emb": adamw_lrs[0],
                    "train/lr_head": adamw_lrs[2],
                },
                step=global_step,
            )

            global_step += 1

            if args.max_steps and global_step >= args.max_steps:
                break

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                ckpt = checkpoint_dir / f"checkpoint_step_{global_step:07d}"
                accelerator.save_state(str(ckpt))
                save_trainer_state(ckpt, epoch, global_step, best_val_loss, scheduler_config)
                if use_ema and accelerator.is_main_process:
                    save_ema_state(ema_state, ckpt / "ema_state.pt")
                if accelerator.is_main_process:
                    print(f"\n  Saved step checkpoint at step {global_step}")
                    cleanup_old_checkpoints(checkpoint_dir, args.max_checkpoints)

        max_steps_reached = args.max_steps and global_step >= args.max_steps

        if max_steps_reached and accelerator.is_main_process:
            print(f"\nReached max-steps ({args.max_steps}), stopping after validation.")

        epoch_loss /= epoch_samples
        epoch_ce /= epoch_samples
        epoch_legal_ce /= epoch_samples
        epoch_value /= epoch_samples
        epoch_acc /= epoch_samples
        epoch_legal_acc /= epoch_samples

        # ── Validation (with EMA weights if enabled) ────────────────────
        model.eval()
        unwrapped = get_raw_model(model, accelerator)

        if use_ema:
            swap_ema_weights(unwrapped, ema_state)

        val_loss_sum = 0.0
        val_ce_sum = 0.0
        val_legal_ce_sum = 0.0
        val_value_sum = 0.0
        val_correct = 0
        val_legal_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                board = batch["position"]
                player = batch["is_white"].long()
                castling = batch["castling_rights"]
                en_passant = batch["en_passant_file"]
                from_sq = batch["from_square"]
                action_plane = batch["action_plane"]
                legal_planes = batch["legal_moves_planes"]
                result = batch["result"]
                is_white = batch["is_white"]
                move_number = batch["move_number"]

                move_logits, value = model(board, player, castling, en_passant)
                loss, metrics = compute_loss(
                    move_logits, value,
                    from_sq, action_plane,
                    legal_planes, result, is_white, move_number,
                    args.value_loss_weight,
                )

                B = move_logits.size(0)
                val_loss_sum += loss.item() * B
                val_ce_sum += metrics["ce"].item() * B
                val_legal_ce_sum += metrics["legal_ce"].item() * B
                val_value_sum += metrics["value_loss"].item() * B
                val_correct += int(metrics["acc"].item() * B)
                val_legal_correct += int(metrics["legal_acc"].item() * B)
                val_total += B

        val_loss = val_loss_sum / val_total
        val_ce = val_ce_sum / val_total
        val_legal_ce = val_legal_ce_sum / val_total
        val_value = val_value_sum / val_total
        val_acc = val_correct / val_total
        val_legal_acc = val_legal_correct / val_total

        log_dict = {
            "train/epoch_loss": epoch_loss,
            "train/epoch_ce": epoch_ce,
            "train/epoch_legal_ce": epoch_legal_ce,
            "train/epoch_value_loss": epoch_value,
            "train/epoch_acc": epoch_acc,
            "train/epoch_legal_acc": epoch_legal_acc,
            "val/loss": val_loss,
            "val/ce": val_ce,
            "val/legal_ce": val_legal_ce,
            "val/value_loss": val_value,
            "val/accuracy": val_acc,
            "val/legal_accuracy": val_legal_acc,
            "epoch": epoch,
        }
        if use_ema:
            log_dict["val/ema_loss"] = val_loss
            log_dict["val/ema_accuracy"] = val_acc
        accelerator.log(log_dict, step=global_step)

        ema_tag = " (EMA)" if use_ema else ""
        print(
            f"Epoch {epoch}: train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f}{ema_tag}"
            f" | val_acc={val_acc:.4f} | val_legal_acc={val_legal_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "best_model"
            accelerator.save_state(str(best_path))
            save_trainer_state(best_path, epoch, global_step, best_val_loss, scheduler_config)
            if use_ema and accelerator.is_main_process:
                save_ema_state(ema_state, best_path / "ema_state.pt")
            print(f"  -> New best model (val_loss={val_loss:.4f})")

        if epoch % args.save_every == 0:
            ckpt = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}"
            accelerator.save_state(str(ckpt))
            save_trainer_state(ckpt, epoch, global_step, best_val_loss, scheduler_config)
            if use_ema and accelerator.is_main_process:
                save_ema_state(ema_state, ckpt / "ema_state.pt")

        if use_ema:
            swap_ema_weights(unwrapped, ema_state)

        if max_steps_reached:
            break

    # ── Final test (using EMA weights) ───────────────────────────────────
    print("\nFinal test evaluation...")
    model.eval()
    if use_ema:
        swap_ema_weights(get_raw_model(model, accelerator), ema_state)
    test_loss_sum = 0.0
    test_correct = 0
    test_legal_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            board = batch["position"]
            player = batch["is_white"].long()
            castling = batch["castling_rights"]
            en_passant = batch["en_passant_file"]
            from_sq = batch["from_square"]
            action_plane = batch["action_plane"]
            legal_planes = batch["legal_moves_planes"]
            result = batch["result"]
            is_white = batch["is_white"]
            move_number = batch["move_number"]

            move_logits, value = model(board, player, castling, en_passant)
            loss, metrics = compute_loss(
                move_logits, value,
                from_sq, action_plane,
                legal_planes, result, is_white, move_number,
                args.value_loss_weight,
            )

            B = move_logits.size(0)
            test_loss_sum += loss.item() * B
            test_correct += int(metrics["acc"].item() * B)
            test_legal_correct += int(metrics["legal_acc"].item() * B)
            test_total += B

    test_loss = test_loss_sum / test_total
    test_acc = test_correct / test_total
    test_legal_acc = test_legal_correct / test_total

    accelerator.log(
        {"test/loss": test_loss, "test/accuracy": test_acc, "test/legal_accuracy": test_legal_acc},
        step=global_step,
    )
    print(f"Test: loss={test_loss:.4f} | accuracy={test_acc:.4f} | legal_accuracy={test_legal_acc:.4f}")

    accelerator.end_training()
    print(f"\nTraining complete. Logs: {log_path}")


if __name__ == "__main__":
    main()
