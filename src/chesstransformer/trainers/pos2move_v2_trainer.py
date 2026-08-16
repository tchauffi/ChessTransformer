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
import os
import random
import shutil
import sys

# Make the checkout runnable without installing the package. `uv run` installs it into the
# venv, so this never fires locally -- but a bare `git clone` on Kaggle or a rented box has
# no chesstransformer on sys.path and every import below fails. scripts/build_shards.py,
# scripts/verify_shards.py and tests/conftest.py already do this; the trainer did not, which
# is exactly where it was first hit.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
import accelerate
from accelerate.utils import broadcast_object_list, set_seed

from chesstransformer.datasets.flat_shard_dataset import (
    FlatShardDataset,
    make_dataloader as make_shard_dataloader,
    scatter_legal_planes,
    worker_init_fn as shard_worker_init_fn,
)
from chesstransformer.datasets.h5_lichess_dataset import HDF5ChessDataset
from chesstransformer.distributed_muon import (
    DistributedMuon,
    muon_momentum,
    muon_weight_decay,
)
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2, NUM_ACTION_PLANES


#: Model sizes. DDP replicates weights *and* optimizer state on every rank.
#:
#: The static column is **measured**, not derived: one optimizer step then
#: torch.cuda.memory_allocated(), on a 16 GB card. It comes out at ~8.4 bytes/param, not
#: the ~18 a generic "fp32 master + two Adam moments + grads" estimate gives, because the
#: optimizer is split -- 2D params go to Muon, which keeps a single momentum buffer, and
#: the AdamW params (norms, biases, embeddings' non-2D parts) are a small remainder.
#: An earlier version of this table used the 18 B/param estimate and overstated xl by 2.3x.
#:
#: Activations dominate and scale with micro-batch: measured ~76 MB/sample for `large`
#: under torch.compile (~90 MB uncompiled). That is what actually decides whether a config
#: fits, so the static column alone is not a fit test.
#:
#: ===== ============ ======== ============= =============================================
#: name  dim x layers   params  static/rank   notes
#: ===== ============ ======== ============= =============================================
#: base     256 x 16    11.7 M       0.10 GB  anything
#: 46m      512 x 16    46.5 M       0.35 GB  anything
#: large    768 x 24   156.4 M       1.23 GB  measured 8.5 GB/rank total at micro-batch 96
#:                                            on a T4; micro-batch 128 fits, 160 OOMs on 16 GB
#: xl      1536 x 32   832.8 M     ~6.4 GB    extrapolated. Static alone would fit a 40 GB
#:                                            card under DDP; whether activations do is
#:                                            unmeasured -- measure before assuming FSDP.
#: ===== ============ ======== ============= =============================================
#:
#: `xl` is deliberately oversized as a sharding/scaling testbed. It is not a bid for
#: playing strength -- 46M already lost its head-to-head against v2.1 at 44.8%.
PRESETS = {
    "base":  {"embed_dim": 256,  "num_layers": 16, "num_heads": 8},
    "46m":   {"embed_dim": 512,  "num_layers": 16, "num_heads": 8},
    "large": {"embed_dim": 768,  "num_layers": 24, "num_heads": 12},
    "xl":    {"embed_dim": 1536, "num_layers": 32, "num_heads": 16},
}


def rank0_print(*a, **kw):
    """print() on the main process only.

    Keyed off the RANK env var rather than accelerator.is_main_process so it also works
    before the Accelerator is constructed -- dataset loading logs before that point. N
    ranks echoing the same line is noise; N ranks echoing *different* lines is a
    debugging trap, and the dataset banners are exactly where they would differ.
    """
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, **kw)


def resolve_precision(requested: str) -> str:
    """Pick a mixed-precision mode this GPU can actually run.

    bf16 needs compute capability 8.0 (Ampere). Kaggle's T4 is Turing, 7.5 -- so the
    trainer's bf16 default silently does not apply there, and a run that looks fine is
    quietly not using tensor cores as intended. Fall back to fp16 (which Turing does have
    tensor cores for) and say so, rather than let the first multi-GPU session discover it.
    """
    # Accelerate spells "no mixed precision" as "no"; "fp32" is not a value it accepts.
    # Passing it through raises ValueError, which means --precision fp32 has never worked
    # and a CPU-only box (where the old code forced "fp32") could never start at all.
    if not torch.cuda.is_available() or os.environ.get("ACCELERATE_USE_CPU"):
        return "no"
    if requested == "fp32":
        return "no"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        cap = torch.cuda.get_device_capability(0)
        rank0_print(f"WARNING: {torch.cuda.get_device_name(0)} is compute capability "
              f"{cap[0]}.{cap[1]}; bf16 needs 8.0+. Falling back to fp16 "
              f"(Accelerate adds a GradScaler). Pass --precision fp32 to opt out.")
        return "fp16"
    return requested


def unpack_batch(batch, device):
    """Normalise a batch from either data path into the tensors the loss expects.

    The shard loader ships the legal mask as a ``(B, 64)`` uint16 index list — 64 KB per
    batch of 512 against 2.4 MB for the dense ``(64, 73)`` bool mask — so the mask is
    rebuilt here, on device, after the H2D copy. Doing it in the dataset would put the
    2.4 MB straight back on the wire and undo the entire point of the shard format.
    """
    if "legal_idx" in batch:
        legal_planes = scatter_legal_planes(batch["legal_idx"].to(device, non_blocking=True))
    else:
        legal_planes = batch["legal_moves_planes"].to(device, non_blocking=True)
    return {
        "board": batch["position"].to(device, non_blocking=True).long(),
        "player": batch["is_white"].to(device, non_blocking=True).long(),
        "castling": batch["castling_rights"].to(device, non_blocking=True).long(),
        "en_passant": batch["en_passant_file"].to(device, non_blocking=True).long(),
        "from_sq": batch["from_square"].to(device, non_blocking=True).long(),
        "action_plane": batch["action_plane"].to(device, non_blocking=True).long(),
        "legal_planes": legal_planes,
        # Narrow on the wire (int8/int16 out of the shard), widened here so both data
        # paths hand compute_loss exactly the same dtypes.
        "result": batch["result"].to(device, non_blocking=True).long(),
        # Must stay bool: compute_loss uses it as a *mask* (`target_value[white_win &
        # is_white]`). An int tensor there would silently become fancy indexing and
        # scatter the value targets onto the wrong rows.
        "is_white": batch["is_white"].to(device, non_blocking=True).bool(),
        "move_number": batch["move_number"].to(device, non_blocking=True).long(),
    }


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


def save_trainer_state(checkpoint_dir, data_pass, global_step, best_val_loss, scheduler_config=None):
    # "epoch" is written alongside "data_pass" only so a checkpoint from this trainer still
    # loads in anything that predates the step paradigm. Nothing here reads it back.
    state = {"data_pass": data_pass, "epoch": data_pass,
             "global_step": global_step, "best_val_loss": best_val_loss}
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

def main():
    default_data = Path(__file__).parents[3] / "data" / "elite_db.h5"

    parser = argparse.ArgumentParser(description="Train Pos2MoveV2 (clean)")
    # Data
    parser.add_argument("--data", type=str, default=str(default_data))
    parser.add_argument("--shards", type=str, default=None,
                        help="Directory of flat shards from scripts/build_shards.py. Bypasses "
                             "--data/--min-elo/--sample-weighting entirely: the sampling "
                             "distribution is frozen into the shard at build time.")
    parser.add_argument("--min-elo", type=int, default=None)
    parser.add_argument("--max-elo", type=int, default=None)
    parser.add_argument("--sample-weighting", type=str, default="middlegame", choices=["uniform", "middlegame"])
    parser.add_argument("--skip-opening-plies", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=50_000,
                        help="Cap on val/test size. The HDF5 path has always applied this; "
                             "the shard path needs it too since a reserved shard is much larger.")
    # Training
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None,
                        help="DEPRECATED and ignored. Training length is --max-steps. Kept so "
                             "existing invocations do not hard-fail; it warns and does nothing.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-muon", type=float, default=2e-3)
    parser.add_argument("--lr-embedding", type=float, default=4e-4)
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--final-lr-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    # Loss weights
    parser.add_argument("--value-loss-weight", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    # Model
    parser.add_argument("--preset", type=str, default="base", choices=list(PRESETS),
                        help="Model size. See PRESETS for per-GPU DDP memory. Individual "
                             "--embed-dim/--num-layers/--num-heads override the preset.")
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.05)    
    parser.add_argument("--layer-drop", type=float, default=0.1,
                        help="Stochastic depth rate. Linearly scales from 0 (first layer) to this value (last layer). Try 0.1.")    # Checkpointing
    parser.add_argument("--max-steps", type=int, default=50_000,
                        help="Total optimizer steps to train for. This is THE training-length "
                             "knob: it bounds the loop and sets the LR decay horizon, and it "
                             "means the same thing on both data paths and at any world size.")
    parser.add_argument("--save-every", type=int, default=None,
                        help="DEPRECATED and ignored (was epoch-based checkpointing). "
                             "Use --save-steps.")
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--eval-steps", type=int, default=1000,
                        help="Validate every N optimizer steps. Val loss selects the best "
                             "checkpoint, so this sets the resolution of that choice. 0 "
                             "disables periodic validation (a final one still runs).")
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--resume-from", type=str, default=None)
    # EMA
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    # Muon
    parser.add_argument("--distributed-muon", action=argparse.BooleanOptionalAction, default=True,
                        help="Partition Newton-Schulz across ranks by whole parameter, so the "
                             "NS work and momentum state are divided instead of every rank "
                             "redundantly computing the same thing. No-op at world size 1.")
    parser.add_argument("--muon-schedules", action=argparse.BooleanOptionalAction, default=True,
                        help="Schedule Muon momentum (0.85->0.97, down to 0.90 during warmdown) "
                             "and cosine-decay its weight decay to zero, as nanochat does.")
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num-workers", type=int, default=12)
    # store_true with default=True could never be turned off. BooleanOptionalAction
    # gives a real --no-compile, which CPU/gloo testing and debugging both need.
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--compile-order", type=str, default="auto",
                        choices=["auto", "before-prepare", "after-prepare"],
                        help="Whether torch.compile runs before or after accelerator.prepare(). "
                             "'auto' = before on a single process (the ordering the 2.2x "
                             "single-GPU speedup was measured with), after under DDP so "
                             "Dynamo's DDPOptimizer engages. See the note at the call site.")
    args = parser.parse_args()

    # Epochs are gone as a unit of training length. They were never comparable between the
    # two data paths — the HDF5 dataset is indexed by *game* (738 optimizer steps/epoch) and
    # the shard dataset by *sample* (11,495 for elite_k16, 50,505 for full_k4) — so the same
    # --epochs N bought a 15-68x different amount of training, LR decay horizon and
    # validation cadence depending only on which --data/--shards flag you passed.
    for dead, replacement in (("epochs", "--max-steps"), ("save_every", "--save-steps")):
        if getattr(args, dead) is not None:
            rank0_print(f"WARNING: --{dead.replace('_', '-')} is deprecated and ignored; "
                  f"use {replacement}.")
    if args.max_steps <= 0:
        parser.error("--max-steps must be > 0")

    # Preset supplies the geometry; an explicit flag always wins over it.
    preset = PRESETS[args.preset]
    for flag, key in (("embed_dim", "embed_dim"), ("num_layers", "num_layers"),
                      ("num_heads", "num_heads")):
        if getattr(args, flag) is None:
            setattr(args, flag, preset[key])

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ── Logging ──────────────────────────────────────────────────────────
    # The run directory has to be decided by ONE rank and told to the others. Every rank
    # scanning logs/ for the next free run number is a race: they can pick the same number
    # (two ranks writing one run) or different ones (N runs, N-1 of them silently empty),
    # and mkdir(exist_ok=False) then crashes whichever rank loses. The timestamp in the
    # name makes independent derivation impossible too -- ranks start milliseconds apart.
    #
    # Ordering note: the Accelerator must exist before the broadcast (it is what sets up
    # the process group), but init_trackers needs the run dir. Hence build it against the
    # parent, then point it at the resolved run dir with set_directories().
    log_dir = Path("logs") / "pos2move_v2"
    log_dir.mkdir(parents=True, exist_ok=True)

    precision = resolve_precision(args.precision)
    accelerator = accelerate.Accelerator(
        log_with="tensorboard",
        project_dir=str(log_dir),
        mixed_precision=precision,
        gradient_accumulation_steps=args.grad_accum,
    )

    if accelerator.is_main_process:
        run_number = get_next_run_number(str(log_dir))
        run_name = f"run_{run_number:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_name = None
    run_name = broadcast_object_list([run_name], from_process=0)[0]

    log_path = log_dir / run_name
    checkpoint_dir = log_path / "checkpoints"
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    # Every rank must see the directory before anyone writes a checkpoint into it.
    accelerator.wait_for_everyone()
    accelerator.project_configuration.set_directories(str(log_path))

    # ── Dataset ──────────────────────────────────────────────────────────
    use_shards = args.shards is not None
    if use_shards:
        # Flat memmap shards: the replay, tokenize and legal-move enumeration were all
        # paid once at build time, and the val/test split is by whole shard (== whole
        # game range) rather than by sample, so no game straddles the split.
        train_set = FlatShardDataset(args.shards, "train")
        val_set = FlatShardDataset(args.shards, "val")
        test_set = FlatShardDataset(args.shards, "test")

        # Whole reserved shards hold far more than validation needs (478k samples at
        # K=16). Cap them the way the HDF5 path does, with a fixed seed so val loss stays
        # comparable across runs — it is what selects the best checkpoint.
        def cap(ds, name):
            if len(ds) <= args.max_val_samples:
                return ds
            g = np.random.default_rng(0xC0FFEE)
            keep = np.sort(g.choice(len(ds), size=args.max_val_samples, replace=False))
            rank0_print(f"  {name}: capped {len(ds):,} -> {args.max_val_samples:,} samples")
            return Subset(ds, keep.tolist())

        val_set = cap(val_set, "val")
        test_set = cap(test_set, "test")

        # The shard loaders bypass accelerator.prepare() (see the note at prepare()), so
        # rank splitting is NOT automatic here the way it is for the HDF5 path -- it has to
        # be asked for. Without this every rank draws the identical batches from the same
        # seeded RandomSampler, and DDP degenerates into N ranks computing one gradient N
        # times: the loss curve looks healthy, the effective batch never grows, and the
        # extra GPUs buy nothing.
        # This is exactly why the Accelerator is constructed above the dataset section:
        # DistributedSampler requires an initialised process group, and the Accelerator is
        # what initialises it. Building the loaders first raises "Default process group has
        # not been initialized".
        world_size = accelerator.num_processes
        train_loader = make_shard_dataloader(train_set, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers, drop_last=True,
                                             seed=args.seed, distributed=world_size > 1)
        # Val/test are deliberately NOT sharded. Each rank evaluates the whole (capped)
        # set and therefore computes an identical val loss, so best-model selection agrees
        # on every rank by construction and needs no gather or padding-trim. The cost is
        # N x redundant work on <=10k samples -- about two seconds -- against a class of
        # bug where ranks disagree about which checkpoint is best and race to write it.
        val_loader = make_shard_dataloader(val_set, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers)
        test_loader = make_shard_dataloader(test_set, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers)
        vocab_size = PostionTokenizer().vocab_size
        rank0_print(f"Shards: {args.shards} (K={train_set.meta['k']}, "
              f"mode={train_set.meta['sample_mode']}, seed={train_set.meta['seed']})")
    else:
        dataset = HDF5ChessDataset(
            hdf5_path=args.data,
            min_elo=args.min_elo,
            max_elo=args.max_elo,
            sample_weighting=args.sample_weighting,
            skip_opening_plies=args.skip_opening_plies,
        )

        val_size = min(int(0.1 * len(dataset)), args.max_val_samples)
        test_size = min(int(0.1 * len(dataset)), args.max_val_samples)
        train_size = len(dataset) - val_size - test_size
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

        # pin_memory lets the H2D copy run on the DMA engine and overlap with compute;
        # without it `.to(device, non_blocking=True)` is silently synchronous.
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True,
                                  persistent_workers=True, prefetch_factor=4, pin_memory=True,
                                  worker_init_fn=shard_worker_init_fn)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True,
                                worker_init_fn=shard_worker_init_fn)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True,
                                 worker_init_fn=shard_worker_init_fn)
        vocab_size = dataset.position_tokenizer.vocab_size

    rank0_print(f"Train: {len(train_set):,} | Val: {len(val_set):,} | Test: {len(test_set):,}")

    # ── Model ────────────────────────────────────────────────────────────
    model_config = {
        "vocab_size": vocab_size,
        "embed_dim": args.embed_dim,
        "nb_transformer_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "kvq_bias": False,
        "layer_drop": args.layer_drop,
    }
    if accelerator.is_main_process:
        with (log_path / "model_config.json").open("w") as f:
            json.dump(model_config, f, indent=2)

    # Effective batch scales with world size: every rank contributes its own micro-batch
    # to the same all-reduced gradient. Reporting it without num_processes understates the
    # real batch by N, which is exactly the number linear LR scaling is derived from.
    effective_bs = args.batch_size * args.grad_accum * accelerator.num_processes
    rank0_print(f"Effective batch size: {effective_bs} (micro={args.batch_size} × "
                f"accum={args.grad_accum} × ranks={accelerator.num_processes})")
    accelerator.init_trackers(
        project_name="pos2move_v2",
        config={
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": effective_bs,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "value_loss_weight": args.value_loss_weight,
            **model_config,
        },
    )

    device = accelerator.device
    rank0_print(f"Device: {device} | Precision: {precision} | Logging to: {log_path}")

    model = Pos2MoveV2(**model_config)
    total_params = sum(p.numel() for p in model.parameters())
    rank0_print(f"Model parameters: {total_params:,}")
    model.to(device)

    # torch.compile vs accelerator.prepare() ordering is not cosmetic under DDP.
    #
    # Dynamo's DDPOptimizer (torch._dynamo.backends.distributed, on by default) splits the
    # graph at DDP's gradient all-reduce bucket boundaries so the collectives can overlap
    # with backward compute. Its own docs: "DDPOptimizer applies when dynamo compiles
    # models wrapped in DistributedDataParallel". Compiling first gives DDP(OptimizedModule)
    # -- the wrap happens outside the compiled graph, DDPOptimizer never sees a DDP module,
    # and the overlap is lost. Compiling after prepare() gives compile(DDP(model)), which is
    # the shape it wants.
    #
    # Single process has no buckets to align to, so 'before' is kept there: it is the
    # ordering the measured 2.2x speedup and the compile-time numbers come from.
    compile_order = args.compile_order
    if compile_order == "auto":
        compile_order = "after-prepare" if accelerator.num_processes > 1 else "before-prepare"

    def maybe_compile(m, when):
        if args.compile and compile_order == when:
            rank0_print(f"torch.compile (mode={args.compile_mode}, {when})")
            return torch.compile(m, mode=args.compile_mode, dynamic=False)  # shapes are static
        return m

    model = maybe_compile(model, "before-prepare")

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

    rank0_print(f"LR: emb={lr_emb:.2e} | muon={args.lr_muon:.2e} | head={lr_head:.2e}")
    rank0_print(
        f"Params: emb={sum(p.numel() for p in adamw_emb_params):,} | "
        f"muon={sum(p.numel() for p in muon_params):,} | "
        f"adamw_other={sum(p.numel() for p in adamw_other_params):,} | "
        f"head={sum(p.numel() for p in adamw_head_params):,}"
    )

    # DistributedMuon degenerates to exactly torch.optim.Muon at world size 1 (verified in
    # tests/test_distributed_muon.py), so there is one code path rather than two.
    muon_cls = DistributedMuon if args.distributed_muon else torch.optim.Muon
    muon_optimizer = muon_cls(
        muon_params,
        lr=args.lr_muon,
        momentum=0.95,
        weight_decay=args.weight_decay,
    )
    if args.distributed_muon and accelerator.num_processes > 1:
        rank0_print(muon_optimizer.sharding_summary().split(" | ")[0])
    adamw_optimizer = torch.optim.AdamW([
        {"params": adamw_emb_params, "lr": lr_emb, "weight_decay": args.weight_decay},
        {"params": adamw_other_params, "lr": args.lr, "weight_decay": 0.0},
        {"params": adamw_head_params, "lr": lr_head, "weight_decay": 0.0},
    ])

    if use_shards:
        # The shard loaders are deliberately kept out of prepare(). They run with
        # batch_size=None and a BatchSampler so that a whole index list reaches
        # __getitem__ in one call; Accelerate reads batch_size=None as "this iterable
        # already yields batches" and would re-shard at the wrong granularity. Rank
        # splitting is done by the DistributedSampler inside make_shard_dataloader, and
        # the H2D copy by unpack_batch.
        model, muon_optimizer, adamw_optimizer = accelerator.prepare(
            model, muon_optimizer, adamw_optimizer
        )
    else:
        model, muon_optimizer, adamw_optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
            model, muon_optimizer, adamw_optimizer, train_loader, val_loader, test_loader
        )

    model = maybe_compile(model, "after-prepare")

    # ── EMA ──────────────────────────────────────────────────────────────
    use_ema = args.ema_decay > 0
    ema_state = None
    if use_ema:
        ema_state = create_ema_state(get_raw_model(model, accelerator))
        rank0_print(f"EMA enabled (decay={args.ema_decay})")

    # ── Resume ───────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    start_pass = 0
    global_step = 0
    trainer_state = None

    if args.resume_from:
        trainer_state = load_trainer_state(args.resume_from)
        if trainer_state:
            # "epoch" is the pre-step-paradigm key; read it so old checkpoints still resume.
            start_pass = trainer_state.get("data_pass", trainer_state.get("epoch", 0)) + 1
            global_step = trainer_state["global_step"]
            best_val_loss = trainer_state["best_val_loss"]

    # ── Scheduler ────────────────────────────────────────────────────────
    total_steps = args.max_steps
    scheduler_config = {
        "warmup_steps": args.warmup_steps,
        "total_steps": total_steps,
        "final_lr_ratio": args.final_lr_ratio,
    }
    if trainer_state and "scheduler_config" in trainer_state:
        # A resume keeps the schedule it started with, so the LR curve stays continuous.
        # That means --max-steps still bounds the *loop* but no longer sets the LR horizon,
        # and the two can silently disagree. Say so rather than let it be discovered later.
        scheduler_config = trainer_state["scheduler_config"]
        resumed_total = scheduler_config.get("total_steps")
        if resumed_total != args.max_steps:
            rank0_print(f"WARNING: resuming with the checkpoint's LR schedule "
                  f"(total_steps={resumed_total:,}) while --max-steps={args.max_steps:,} "
                  f"bounds the loop. The LR decays over {resumed_total:,} steps regardless.")
        total_steps = resumed_total

    # An "epoch" is not comparable between the two data paths: the HDF5 dataset is indexed
    # by *game* (738 optimizer steps/epoch) and the shard dataset by *sample* (11.5k for
    # elite_k16, 50.5k for full_k4). So --epochs N silently buys a 15-68x longer LR decay
    # on shards, and epoch-end validation fires that much less often. Print both so the
    # schedule is never a surprise, and say so out loud when --max-steps is not pinning it.
    steps_per_pass = len(train_loader) // max(1, args.grad_accum)
    samples_seen = args.max_steps * effective_bs
    rank0_print(f"Schedule: {total_steps:,} optimizer steps | warmup {args.warmup_steps:,} | "
          f"eval every {args.eval_steps:,} | save every {args.save_steps:,}")
    rank0_print(f"  one pass over the training data = {steps_per_pass:,} steps/rank "
          f"({args.max_steps / max(1, steps_per_pass):.2f} passes, "
          f"{samples_seen:,} samples consumed)")

    muon_scheduler = create_lr_scheduler(muon_optimizer, **scheduler_config)
    adamw_scheduler = create_lr_scheduler(adamw_optimizer, **scheduler_config)
    accelerator.register_for_checkpointing(muon_scheduler, adamw_scheduler)

    if args.resume_from:
        accelerator.load_state(args.resume_from)
        if use_ema:
            ema_path = Path(args.resume_from) / "ema_state.pt"
            if ema_path.exists():
                ema_state = load_ema_state(ema_path, device)
                rank0_print(f"Restored EMA state from {ema_path}")
            else:
                ema_state = create_ema_state(get_raw_model(model, accelerator))
                rank0_print("EMA state not found in checkpoint, re-initialized from model")
        rank0_print(f"Resumed from {args.resume_from} (step {global_step:,}/{args.max_steps:,}, "
              f"data pass {start_pass})")

    # ── Validation ───────────────────────────────────────────────────────
    # Runs on the --eval-steps counter. It used to run only at epoch end, which was a fine
    # cadence on the HDF5 path (738 steps/epoch) and a bad one on shards (11,495 steps for
    # elite_k16, 50,505 for full_k4) — the dataset index is a sample there, not a game. Val
    # loss selects the best checkpoint, so that gap mattered. Hence the step paradigm.

    def run_validation():
        """Evaluate on val_loader (under EMA weights if enabled). Restores train mode."""
        model.eval()
        unwrapped = get_raw_model(model, accelerator)
        if use_ema:
            swap_ema_weights(unwrapped, ema_state)

        # No cross-rank reduction here on purpose: val_loader is replicated, not sharded,
        # so every rank walks the identical set and arrives at the identical number. That
        # is what keeps best-model selection consistent across ranks without a gather.
        sums = dict(loss=0.0, ce=0.0, legal_ce=0.0, value=0.0, correct=0, legal_correct=0, total=0)
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False,
                              disable=not accelerator.is_main_process):
                b = unpack_batch(batch, device)
                move_logits, value = model(b["board"], b["player"], b["castling"], b["en_passant"])
                loss, metrics = compute_loss(
                    move_logits, value,
                    b["from_sq"], b["action_plane"],
                    b["legal_planes"], b["result"], b["is_white"], b["move_number"],
                    args.value_loss_weight,
                )
                B = move_logits.size(0)
                sums["loss"] += loss.item() * B
                sums["ce"] += metrics["ce"].item() * B
                sums["legal_ce"] += metrics["legal_ce"].item() * B
                sums["value"] += metrics["value_loss"].item() * B
                sums["correct"] += int(metrics["acc"].item() * B)
                sums["legal_correct"] += int(metrics["legal_acc"].item() * B)
                sums["total"] += B

        if use_ema:
            swap_ema_weights(unwrapped, ema_state)
        model.train()

        n = max(1, sums["total"])
        return {
            "val/loss": sums["loss"] / n,
            "val/ce": sums["ce"] / n,
            "val/legal_ce": sums["legal_ce"] / n,
            "val/value_loss": sums["value"] / n,
            "val/accuracy": sums["correct"] / n,
            "val/legal_accuracy": sums["legal_correct"] / n,
        }

    def record_validation(vals, data_pass, extra_log=None, label=""):
        """Log, print, and update best-model selection. Returns True if it was a new best."""
        nonlocal best_val_loss
        log_dict = dict(extra_log or {})
        log_dict.update(vals)
        if use_ema:
            log_dict["val/ema_loss"] = vals["val/loss"]
            log_dict["val/ema_accuracy"] = vals["val/accuracy"]
        accelerator.log(log_dict, step=global_step)

        ema_tag = " (EMA)" if use_ema else ""
        rank0_print(f"{label}val_loss={vals['val/loss']:.4f}{ema_tag} "
              f"| val_acc={vals['val/accuracy']:.4f} "
              f"| val_legal_acc={vals['val/legal_accuracy']:.4f}")

        if vals["val/loss"] < best_val_loss:
            best_val_loss = vals["val/loss"]
            best_path = checkpoint_dir / "best_model"
            accelerator.save_state(str(best_path))
            save_trainer_state(best_path, data_pass, global_step, best_val_loss, scheduler_config)
            if use_ema and accelerator.is_main_process:
                save_ema_state(ema_state, best_path / "ema_state.pt")
            rank0_print(f"  -> New best model (val_loss={vals['val/loss']:.4f})")
            return True
        return False

    # ── Training loop (step-driven) ──────────────────────────────────────
    # The loop is bounded by optimizer steps and cycles the dataloader as many times as
    # that takes. Nothing downstream keys off "epoch" any more: LR decay, validation and
    # checkpointing are all on step counters, which mean the same thing whichever dataset
    # is mounted and at any world size.

    def start_pass_shuffle(loader, n):
        """Reshuffle for a fresh pass over the data.

        DistributedSampler needs set_epoch or every rank replays the identical order on
        every pass. RandomSampler carries its generator across iterators and reshuffles on
        its own, so this is a no-op there.
        """
        for obj in (loader,
                    getattr(loader, "sampler", None),
                    getattr(getattr(loader, "sampler", None), "sampler", None)):
            if obj is not None and hasattr(obj, "set_epoch"):
                obj.set_epoch(n)
                return

    def new_window():
        """Train metrics accumulated since the last validation, sample-weighted."""
        return dict(loss=0.0, ce=0.0, legal_ce=0.0, value=0.0, acc=0.0, legal_acc=0.0, n=0)

    WINDOW_KEYS = ("loss", "ce", "legal_ce", "value", "acc", "legal_acc")

    def flush_window(w):
        """Average the window across ranks, not just within one.

        Each rank trains on a different slice, so these sums are rank-local: rank 0's
        numbers alone are a 1/N sample of the step. Reducing sum-of-(metric x batch) and
        sum-of-batch separately, then dividing, is exact regardless of how unevenly the
        last batches fall -- which is why this does not need gather_for_metrics and its
        padding-trim bookkeeping (that machinery also would not work here, since the shard
        loaders never went through prepare()).
        """
        vals = torch.tensor([w[k] for k in WINDOW_KEYS] + [float(w["n"])],
                            dtype=torch.float64, device=accelerator.device)
        if accelerator.num_processes > 1:
            vals = accelerator.reduce(vals, reduction="sum")
        total = max(1.0, vals[-1].item())
        return {f"train/{'value_loss' if k == 'value' else k}": vals[i].item() / total
                for i, k in enumerate(WINDOW_KEYS)}

    last_val_step = -1

    def validate_now(window, data_pass, label_prefix):
        """Flush the train window, validate, log both against the same step.

        No-ops if this step has already been validated, which happens whenever max_steps is
        an exact multiple of eval_steps and the final validation lands on the same step as
        the last periodic one. Worth guarding rather than tolerating: on the HDF5 path the
        two calls do not even agree, because ``HDF5ChessDataset.__getitem__`` samples a
        fresh ply per call, so its val set re-rolls on every pass. (The shard val set is
        fixed, so shard val loss is comparable across evaluations and HDF5 val loss is not
        — worth remembering when reading either curve.)
        """
        nonlocal last_val_step
        if global_step == last_val_step:
            return
        last_val_step = global_step
        extra = flush_window(window) if window["n"] else {}
        extra["data_pass"] = data_pass
        train_tag = f"train_loss={extra['train/loss']:.4f} | " if window["n"] else ""
        record_validation(run_validation(), data_pass, extra_log=extra,
                          label=f"{label_prefix}{train_tag}")

    model.train()
    window = new_window()
    data_pass = start_pass
    pbar = tqdm(total=args.max_steps, initial=global_step, desc="train", unit="step",
                disable=not accelerator.is_main_process)

    while global_step < args.max_steps:
        start_pass_shuffle(train_loader, data_pass)
        for batch in train_loader:
            with accelerator.accumulate(model):
                b = unpack_batch(batch, device)
                board, player = b["board"], b["player"]
                castling, en_passant = b["castling"], b["en_passant"]
                from_sq, action_plane = b["from_sq"], b["action_plane"]
                legal_planes, result = b["legal_planes"], b["result"]
                is_white, move_number = b["is_white"], b["move_number"]

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
                if args.muon_schedules:
                    # Momentum is not a constant in a well-tuned Muon run: a long memory
                    # early averages over a fast-moving gradient, and a slightly shorter one
                    # late helps the run settle. Weight decay cosine-decays to zero.
                    mom = muon_momentum(global_step, args.max_steps)
                    wdk = muon_weight_decay(global_step, args.max_steps, args.weight_decay)
                    for g in muon_optimizer.param_groups:
                        g["momentum"], g["weight_decay"] = mom, wdk
                muon_optimizer.step()
                adamw_optimizer.step()
                if accelerator.sync_gradients:
                    muon_scheduler.step()
                    adamw_scheduler.step()
                muon_optimizer.zero_grad()
                adamw_optimizer.zero_grad()

                # EMA is redundant under DDP but not wrong, and that is worth stating.
                # Gradients are all-reduced before the step, so every rank holds identical
                # params and computes an identical EMA from an identical decay -- no drift.
                # swap_ema_weights in run_validation is likewise symmetric because every
                # rank validates (val_loader is replicated, not sharded). The failure mode
                # to avoid is any rank taking a different branch here; sync_gradients is
                # world-uniform, so none does. tests/test_distributed.py asserts it.
                if use_ema and accelerator.sync_gradients:
                    update_ema(get_raw_model(model, accelerator), ema_state, args.ema_decay)

            bs = board.size(0)
            window["loss"] += loss.item() * bs
            window["ce"] += metrics["ce"].item() * bs
            window["legal_ce"] += metrics["legal_ce"].item() * bs
            window["value"] += metrics["value_loss"].item() * bs
            window["acc"] += metrics["acc"].item() * bs
            window["legal_acc"] += metrics["legal_acc"].item() * bs
            window["n"] += bs

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
                    **({"train/muon_momentum": muon_optimizer.param_groups[0]["momentum"],
                        "train/muon_wd": muon_optimizer.param_groups[0]["weight_decay"]}
                       if args.muon_schedules else {}),
                },
                step=global_step,
            )

            global_step += 1
            pbar.update(1)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                ckpt = checkpoint_dir / f"checkpoint_step_{global_step:07d}"
                accelerator.save_state(str(ckpt))
                save_trainer_state(ckpt, data_pass, global_step, best_val_loss, scheduler_config)
                if use_ema and accelerator.is_main_process:
                    save_ema_state(ema_state, ckpt / "ema_state.pt")
                if accelerator.is_main_process:
                    rank0_print(f"\n  Saved step checkpoint at step {global_step}")
                    cleanup_old_checkpoints(checkpoint_dir, args.max_checkpoints)

            # Validation is on a step counter, and global_step is identical on every rank,
            # so all ranks enter the collective forward together.
            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                validate_now(window, data_pass,
                             f"\nStep {global_step:,}/{args.max_steps:,}: ")
                window = new_window()

            if global_step >= args.max_steps:
                break

        data_pass += 1

    pbar.close()

    # Always finish on a validation, so the final steps can still win best_model even when
    # max_steps is not a multiple of eval_steps.
    rank0_print(f"\nReached max-steps ({args.max_steps:,}) after {data_pass} pass(es) over the data.")
    validate_now(window, data_pass, f"Final (step {global_step:,}): ")

    # ── Final test (using EMA weights) ───────────────────────────────────
    rank0_print("\nFinal test evaluation...")
    model.eval()
    if use_ema:
        swap_ema_weights(get_raw_model(model, accelerator), ema_state)
    test_loss_sum = 0.0
    test_correct = 0
    test_legal_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False,
                          disable=not accelerator.is_main_process):
            b = unpack_batch(batch, device)
            board, player = b["board"], b["player"]
            castling, en_passant = b["castling"], b["en_passant"]
            from_sq, action_plane = b["from_sq"], b["action_plane"]
            legal_planes, result = b["legal_planes"], b["result"]
            is_white, move_number = b["is_white"], b["move_number"]

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
    rank0_print(f"Test: loss={test_loss:.4f} | accuracy={test_acc:.4f} | legal_accuracy={test_legal_acc:.4f}")

    accelerator.end_training()
    rank0_print(f"\nTraining complete. Logs: {log_path}")


if __name__ == "__main__":
    main()
