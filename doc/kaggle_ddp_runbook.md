# Kaggle 2×T4 — DDP testbed runbook

The free rung of the ladder in [`distributed_training.md`](distributed_training.md): real
multi-device NCCL, real inter-GPU collectives, ~30 h/week, €0. Everything here should work before
any money is spent on a rented box.

**What this session is for:** proving the DDP path is *correct* on genuinely separate devices. It is
not for throughput numbers — two T4s are slower than the local 5070 Ti and will tell you nothing
useful about scaling.

---

## The three constraints that shape everything

| constraint | consequence |
|---|---|
| **T4 is compute capability 7.5** | No bf16 (needs 8.0). The trainer's default `--precision bf16` does not apply. `resolve_precision()` detects this and falls back to fp16 + GradScaler, printing a warning — but pass `--precision fp16` explicitly so it is a decision, not a fallback. |
| **~20 GB working disk** | `full_k4` is 20 GB decompressed and will not fit alongside anything else. Use `elite_k16` (4.95 GB, 967 MB compressed). |
| **~4 vCPUs** | The HDF5 path wants 8–12 dataloader workers and cannot get them. The shard path wants **0** when the shards fit page cache, which at 4.95 GB against Kaggle's ~29 GB RAM they do. This is the single biggest reason to use shards here. |

T4 also has 16 GB per GPU, so `--preset large` (156M params, 2.82 GB/rank under DDP) fits
comfortably. `--preset xl` (833M, 14.99 GB/rank) does **not** — it needs FSDP and ≥40 GB cards.

---

## Setup

Enable **GPU T4 ×2** in the notebook settings (Accelerator → GPU T4 ×2), and internet access.

```bash
!pip -q install "accelerate>=1.10.1,<1.11" python-chess h5py tensorboard zstandard
!git clone https://github.com/tchauffi/ChessTransformer /kaggle/working/ct
%cd /kaggle/working/ct
```

Do **not** `pip install -e .` here: the project pins a cu128 torch index, and resolving it would
fight Kaggle's preinstalled CUDA build. The trainer and scripts put `src/` on `sys.path`
themselves, so a bare clone runs as-is. If you ever see `ModuleNotFoundError: No module named
'chesstransformer'`, that bootstrap is missing on whatever entry point you invoked — prepend
`PYTHONPATH=/kaggle/working/ct/src` as a stopgap.

The accelerate pin matters: 1.13 and 1.14 regressed the CPU multi-rank path this project's tests
depend on. NCCL on Kaggle is unaffected, but keeping one version everywhere avoids surprises.

```python
# ~967 MB down, 4.95 GB on disk after decompression.
# snapshot_download rather than the `hf` CLI: the CLI entry point is named `hf` in recent
# huggingface_hub and `huggingface-cli` in older ones, and Kaggle's image pins its own.
from huggingface_hub import snapshot_download
snapshot_download("tchauffi/chesstransformer-shards", repo_type="dataset",
                  allow_patterns="elite_k16/*", local_dir="/kaggle/working/shards")
```

```python
# Decompressed in Python, not with the zstd CLI, which is not guaranteed to be on the image.
# Each .zst is removed as it is expanded so peak disk stays near the 4.95 GB final size.
import zstandard, pathlib, concurrent.futures
d = pathlib.Path("/kaggle/working/shards/elite_k16")
def dec(p):
    with open(p, "rb") as fi, open(p.with_suffix(""), "wb") as fo:
        zstandard.ZstdDecompressor().copy_stream(fi, fo)
    p.unlink()
with concurrent.futures.ThreadPoolExecutor(4) as ex:
    list(ex.map(dec, sorted(d.glob("*.zst"))))
print(len(list(d.glob("*.npy"))), "shards ready")
```

```bash
!df -h /kaggle/working | tail -1
```

---

## 1. Smoke test first — always

Never start a distributed session by launching the real run. Two minutes here saves an hour of
confused debugging.

```bash
!cd /kaggle/working/ct && NPROC=2 bash scripts/launch_train.sh \
    --shards /kaggle/working/shards/elite_k16 \
    --preset base --precision fp16 --no-compile \
    --batch-size 64 --grad-accum 1 --max-steps 20 --eval-steps 10 \
    --num-workers 0 --warmup-steps 5 --max-val-samples 1024 --save-steps 100000
```

Check, in order:

- `Device: cuda:0` / `cuda:1` — two *different* devices, not two ranks on one.
- `Effective batch size: 128 (micro=64 × accum=1 × ranks=2)` — the `× ranks=2` is the proof that DDP
  is actually engaged and that the batch grew.
- `one pass over the training data = N steps/rank` — must be **half** the single-rank number. If it
  is not, the loaders are not sharded and every rank is training on identical data.
- Exactly **one** run directory under `logs/pos2move_v2/`.

---

## 2. The test that actually matters — resume across world sizes

**This is unverified and is the reason to run Kaggle before renting anything.** It could not be
tested locally: one GPU, and the CPU/gloo path is blocked by an upstream accelerate/torch
serialization bug (`don't know how to restore data location ... tagged with cpu:0`). Single-rank GPU
resume *is* verified; changing world size on resume is not.

It matters because spot/interruptible instances are 2–4× cheaper, and "survives preemption at a
different node count" is the property that makes them usable.

```bash
# a) save at world_size=2
!cd /kaggle/working/ct && NPROC=2 bash scripts/launch_train.sh \
    --shards /kaggle/working/shards/elite_k16 --preset base --precision fp16 --no-compile \
    --batch-size 64 --max-steps 40 --save-steps 20 --eval-steps 0 --num-workers 0 --warmup-steps 5

# b) resume the step-20 checkpoint on ONE rank
!cd /kaggle/working/ct && NPROC=1 bash scripts/launch_train.sh \
    --shards /kaggle/working/shards/elite_k16 --preset base --precision fp16 --no-compile \
    --batch-size 64 --max-steps 40 --save-steps 100000 --eval-steps 0 --num-workers 0 \
    --warmup-steps 5 --resume-from logs/pos2move_v2/<RUN>/checkpoints/checkpoint_step_0000020
```

Expected: `Resumed from ... (step 20/40, data pass N)` and training continues to 40 without error.
Then repeat in reverse (save at 1, resume at 2). Record the outcome in
[`../docs/distributed-log.md`](../docs/distributed-log.md) either way — a negative result here is
worth more than a throughput number.

Note the trainer warns if the checkpoint's LR schedule disagrees with `--max-steps`: a resume keeps
the schedule it started with, deliberately, so the LR curve stays continuous.

---

## 3. Correctness harness on real NCCL

The gloo/CPU suite passes locally and in CI, but gloo and NCCL are different collective
implementations. Run it once against real devices:

```bash
!cd /kaggle/working/ct && CT_FORCE_SYNTHETIC_SHARDS=1 python -m pytest tests/ -v
```

The load-bearing test is `test_grad_equivalence_vs_single_rank`: N ranks at batch B must produce the
gradient one rank at batch N·B would.

---

## 4. Only then, a real run

```bash
!cd /kaggle/working/ct && NPROC=2 bash scripts/launch_train.sh \
    --shards /kaggle/working/shards/elite_k16 \
    --preset large --precision fp16 \
    --batch-size 96 --grad-accum 4 --max-steps 20000 \
    --eval-steps 500 --save-steps 2000 --num-workers 0 \
    --warmup-steps 500 --lr 8e-4 --max-checkpoints 3
```

Notes on the numbers:

- **`--num-workers 0`** is not a typo. The shards are in page cache; a batch fetch is a memmap read,
  which is cheaper than the IPC to hand it to a worker. With only ~4 vCPUs, workers would also be
  competing with the two training processes.
- **Effective batch is 96 × 4 × 2 = 768.** If you scale it further, scale the LR with it (linear
  scaling + longer warmup, per Goyal et al.) rather than leaving `--lr` where a smaller batch put it.
- **`--preset large`** is 156M params. It is a scaling exercise, not a strength bid: 46M already lost
  its head-to-head against v2.1 at 44.8%.
- Kaggle sessions are capped at ~9 h and can be interrupted, so `--save-steps 2000` and
  `--max-checkpoints 3` matter. Copy checkpoints out of `/kaggle/working` before the session ends.

---

## What to bring back

Append to [`../docs/distributed-log.md`](../docs/distributed-log.md):

- Whether world-size-change resume works (§2) — the open question.
- samples/s at 1 vs 2 ranks, and the scaling efficiency, with the caveat that two PCIe T4s are a
  weak scaling signal.
- Whether `--compile-order after-prepare` (the DDP default, which lets Dynamo's DDPOptimizer align
  graph breaks to gradient-bucket boundaries) actually beats `before-prepare` here. It is a knob
  precisely because it should be measured rather than assumed.
- Any place a rank printed something rank-0 was supposed to own.
