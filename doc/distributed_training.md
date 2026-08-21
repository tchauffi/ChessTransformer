# Distributed Training for ChessTransformer — Learning Roadmap

## Context

You've been passed over for two roles for lack of distributed-training experience. This repo is a good
vehicle: it has a real training loop, a real dataset, a custom optimizer, and a stated need to scale.
The goal is to build genuine distributed-training capability *and* make this project train faster —
on a ≤ €200 total cloud budget.

You implement everything yourself. This document is the TODO list, the gotchas to expect, and the
acceptance criteria to know when each piece is actually correct.

### Where the code stands today

| Fact | Where | Consequence |
|---|---|---|
| Already uses `accelerate.Accelerator` + `prepare()` | `src/chesstransformer/trainers/pos2move_v2_trainer.py:322,395` | DDP is *nominally* one `accelerate launch` away — but see the 6 landmines below |
| Single GPU only, no `torch.distributed` anywhere | grep-confirmed across `src/`, `scripts/` | Nothing has ever run multi-rank |
| `__getitem__` replays the game with python-chess + enumerates legal moves | `datasets/h5_lichess_dataset.py:119-244` | ~800 samples/s per core. **Measured: this is the bottleneck once `--compile` is on** — see `docs/distributed-log.md` |
| `RepeatingDataloader(repeat_factor=2)` exists "to avoid underloaded GPU" | trainer `:205-217,400` | Direct evidence the GPU is data-starved today |
| **Phase 0 is done.** End-to-end 2,272 samples/s; GPU busy 40%; `torch.compile` buys 2.2× on compute and **0% end-to-end** | `docs/distributed-log.md` | Phase 1 confirmed as a hard prerequisite for Phase 3 |
| 1,533,352 games / 148M plies in a 356 MB HDF5 | `data/elite_db.h5` | Fits in page cache; IO is *not* the problem, CPU decode is |
| Custom `torch.optim.Muon` + `AdamW` split | trainer `:383-393`, `src/chesstransformer/optimizer.py` | Zero distributed awareness. Muon is the most interesting distributed problem in this repo |
| Model is 11.7M params (46M variant exists) | `models/transformer/pos2move_v2.py:178` | Too small for FSDP/TP to *help*. See "The honest framing" below |
| Local hardware: 1× RTX 5070 Ti 16 GB, 16 cores, 30 GB RAM | `nvidia-smi` | Real multi-GPU needs rented boxes |
| No tests for training, no CI beyond an HF-Space deploy | `src/tests/` (1 file), `.github/workflows/` | You'll build the correctness harness from scratch — that's part of the value |

### The honest framing (read this before Phase 4)

You picked **data parallel**, **memory/sharding**, and **model/pipeline parallel**. Two of those three
do not help an 11.7M-param model — FSDP and tensor parallel would make it *slower*. That's fine, but
it changes how you should stage the work:

- **Phases 0–3 (DDP core) are real engineering wins** on this project. They make every experiment
  faster and they're where the interview questions actually land.
- **Phases 4–5 (FSDP/TP/PP) need a deliberately oversized model** to be meaningful. Build a
  `--preset xl` config (~400M–1B params) purely as a *scaling testbed*. Do not claim it improves
  playing strength — your own logs already show 46M didn't (`doc/selfplay_rl.md`, and the 46M vs v2.1
  head-to-head at 44.8%). Frame it as an infrastructure artifact. That framing is itself a good
  interview answer: "I built the sharded-training path and measured where it starts paying off."

---

## Budget plan — €200 hard cap

The discipline that keeps this cheap: **never debug on a rented multi-GPU box.** All correctness work
happens locally for free. Rented time is only for *measurement* and one *production run*.

### Free tiers you should exploit first
- **Local, 2–4 ranks on one GPU.** An 11.7M model at batch 256 uses < 3 GB. You can run
  `torchrun --nproc_per_node=4` with all ranks on `cuda:0`. NCCL works fine with multiple ranks per
  device for correctness testing. This catches ~90% of DDP bugs.
- **Local, `gloo` on CPU.** 4 ranks, tiny model, tiny dataset — runs in seconds, works in CI.
- **Kaggle: 2× T4, free, ~30 h/week.** Real multi-device NCCL, real inter-GPU comms. Use this to
  validate that your DDP path works on genuinely separate devices before paying for anything.

### Paid allocation (≈ $215)

| Phase | Instance | Hours | Est. cost |
|---|---|---|---|
| 3 — scaling curves | 8× RTX 4090 (vast.ai interruptible, ~$0.25–0.40/GPU-h) | 4 | **€12** |
| 3 — scaling curves, NVLink comparison | 8× A100 80 GB (RunPod/Lambda, ~$1.2–1.8/GPU-h) | 3 | **€35** |
| 4 — FSDP2 on the XL model | 4× A100 80 GB | 5 | **€30** |
| 5 — TP/PP on the XL model | 4× A100 80 GB | 4 | **€25** |
| 6 — one real production run | 8× A100 80 GB | 8 | **€80** |
| reserve | — | — | **€18** |

Prices move constantly — check vast.ai / RunPod / Lambda the week you book and re-plan. 4090s are
the value pick for throughput work; A100/H100 only when you need 80 GB or NVLink to make a point.

### Cost-discipline TODOs (do these *before* your first rental)
- [ ] Build a **`--smoke` config**: 200 steps, tiny slice of the dataset, runs end-to-end in < 2 min.
      Every rented session starts by running it. If it fails, you fix it locally, not at $10/hr.
- [ ] **Pre-bake a Docker image** with the venv resolved (`uv sync` baked in, torch 2.11 + cu128).
      Cold-starting `uv sync` on a rented box burns 10–15 min of paid time, every time.
- [ ] Get `data/elite_db.h5` (356 MB) onto **cheap object storage** (Cloudflare R2 free tier, or an
      HF dataset repo) so the box pulls it in seconds. Do the same for the preprocessed shards from Phase 1.
- [ ] **Use interruptible/spot instances**, and make that safe with elastic checkpointing (Phase 2).
      Spot is 2–4× cheaper and "survives preemption" is a legitimate distributed-training skill.
- [ ] Wrap every rented run in `timeout 4h ...` and a teardown hook. A forgotten 8×A100 box overnight
      is your entire budget.
- [ ] Keep a `docs/distributed-log.md` with cost-per-session. Interviewers love a candidate who can
      say "8×A100 for 3h cost $38 and bought me these three numbers."

---

## Phase 0 — Measure before you parallelize (local, free) — ✅ DONE 2026-08-05

**Results: [`docs/distributed-log.md`](../docs/distributed-log.md). Tools: `scripts/profile_training.py`,
`scripts/profile_dataloader.py`.**

Headline: compiled compute ceiling 5,664 samples/s, dataloader 5,132 samples/s, **actual end-to-end
2,272 samples/s with the GPU busy only 40% of the step.** `torch.compile` is a 2.2× speedup on
compute that delivers zero end-to-end gain — the textbook signature of a data-bound pipeline.

Two things the measurement corrected in this plan:
- In **eager** mode the trainer is genuinely *compute*-bound (data time is 0.44× compute). It only
  flips to data-bound **with `--compile`** — which is what `resume_training.sh` uses. The original
  claim here was half right.
- Isolated dataloader benchmarks *overestimate* delivered throughput. Alone the loader does 5,132
  samples/s; running alongside training it effectively delivers ~2,272, because the 12 workers
  contend with the main process for the same 16 cores. Always cross-check against the combined number.

Also measured, and directly relevant later: **Muon's step is a fixed ~12.8 ms independent of batch
size** (Newton–Schulz runs on weights, not activations) — 13% of a micro-batch step, ~3.5% of an
optimizer step at `--grad-accum 4`. Under DDP every rank duplicates it. That's the concrete payoff
for the distributed-Muon item in Phase 2.

Original brief, kept for reference:

You cannot honestly claim distributed-training skill without being able to answer *"how do you know
what the bottleneck is?"*

- [ ] Add `scripts/profile_training.py` (or a `--profile` flag) that reports, for N steps:
      samples/s, GPU util (`torch.cuda.utilization()`), time in dataloading vs. forward vs. backward
      vs. optimizer, and peak memory. Model it on the existing `scripts/bench_inference.py`.
- [ ] Run the PyTorch profiler with `ProfilerActivity.CPU|CUDA` and export a Chrome trace. Look at it.
      Find the gaps between kernels — those gaps are your data pipeline.
- [ ] Compute **MFU** (model FLOPs utilization) for the current run. For a 11.7M model at 67 tokens
      of context this will be embarrassingly low — that's the point. Write the number down.
- [ ] Answer definitively: **with `--num-workers 12`, is `data/elite_db.h5` decoding fast enough to
      saturate the 5070 Ti?** Run once with the real dataset, once with a synthetic in-memory dataset
      that returns pre-made tensors. The gap is your data pipeline tax.
- [ ] Delete `RepeatingDataloader` (`:205-217,400`) once you know the real number — it's a symptom-masker
      that duplicates gradients and breaks step accounting.

**Learn:** PyTorch Profiler, `torch.cuda.Event` timing, MFU as defined in the PaLM paper §Appendix B.

**Acceptance:** a table in `docs/distributed-log.md` giving samples/s, GPU util %, and MFU for the
current single-GPU trainer, plus the synthetic-data ceiling.

---

## Phase 1 — Fix the input pipeline (local, free)

**Do this before DDP — Phase 0 confirmed it's a hard prerequisite, not an optimisation.**
Feeding 8 GPUs of the 5070 Ti's class needs ~45,000 samples/s. At the best observed per-worker rate
(~800/s uncontended) that's ~57 dedicated cores, and A100/H100 compute is faster still. No realistic
8-GPU node has the CPU to feed this dataset in its current form. Scaling a CPU-bound pipeline across
GPUs is the classic beginner mistake.

Phase 0 also found three pieces of pure waste to delete on the way (details in `docs/distributed-log.md`):
`legal_moves_grid` and `legal_moves_mask` are built per sample and **never read by the trainer**;
`h5py.File` is opened **twice per sample**, the second time only to fetch three scalars. Targets:
end-to-end > 5,400 samples/s, GPU busy > 90%, ≤ 4 workers per GPU, < 1.5 MB per collated batch
(currently 5.8 MB).

- [ ] Write `scripts/build_shards.py`: a one-shot preprocessing pass that converts `elite_db.h5` into
      **flat, fixed-width, memory-mappable shards** of already-tokenized samples. Per sample you need:
      the 67-token position, `from_square`, `action_plane`, the `(64,73)` legal mask, `result`,
      `is_white`, `move_number`. Reuse the encoding logic already in
      `datasets/h5_lichess_dataset.py:__getitem__` and `models/tokenizer/alphazero_move_encoder.py`.
      - The legal mask is the size problem: 64×73 bools = 4672 bytes/sample raw. Store it **packed**
        (`np.packbits` → 584 bytes) or as a variable-length index list (typically ~30 legal moves →
        ~60 bytes as int16). Pick one, measure both.
      - Sizing: 148M plies is too many to materialize. Sample K positions per game (K = 8–16) → 12–25M
        samples. At ~200 bytes/sample packed that's 2.4–5 GB. Fits on disk, mostly fits in page cache.
- [ ] Write the matching `FlatShardDataset` in `src/chesstransformer/datasets/`: `np.memmap` per shard,
      O(1) `__getitem__`, no python-chess in the hot path, no `h5py.File` open per item.
- [ ] Make sharding **rank-and-worker aware**: shard *k* is owned by rank *r* worker *w* where
      `k % (world_size * num_workers) == r * num_workers + w`. This is the pattern you'll be asked to
      describe in interviews.
- [ ] Fix the **numpy RNG-per-worker bug**. `datasets/h5_lichess_dataset.py:_sample_move_idx` uses
      `np.random`, and PyTorch does *not* reseed numpy per worker — all 12 workers currently draw the
      same sequence, and under DDP all ranks would too. Add a `worker_init_fn` that seeds from
      `torch.initial_seed()` + `rank`. Verify by asserting distinct samples across ranks.
- [ ] Re-run Phase 0's profiler. Expected: GPU util from (probably) 40–60% to > 90%, and a
      single-GPU speedup with *no* extra hardware.

**Learn:** `np.memmap`, `torch.utils.data.get_worker_info()`, why map-style + `DistributedSampler`
beats `IterableDataset` for shuffling quality, and the tradeoff the other way for streaming.

**Acceptance:** same loss curve as the HDF5 path for the first 2000 steps (within noise), at
significantly higher samples/s, with `--num-workers 4` instead of 12.

---

## Phase 2 — DDP, done correctly (local multi-rank + free Kaggle 2×T4)

The trainer *looks* DDP-ready but has six landmines. Fixing each one is a specific lesson.

- [ ] **Launcher.** Add `scripts/launch_train.sh` wrapping `torchrun` (prefer it over `accelerate
      launch` here — you'll learn more from the explicit rendezvous args). Add an
      `accelerate/ddp.yaml` config too so you understand what Accelerate generates.
- [ ] **Rank-0 guards.** `get_next_run_number()` + `log_path.mkdir(exist_ok=False)` (trainer `:311-316`)
      will race and crash on rank ≥ 1. Compute the run dir on rank 0, then broadcast it
      (`accelerator.broadcast_object_list` / `dist.broadcast_object_list`). Guard `tqdm`
      (`:460,576,671`) and all `print`s on `is_main_process`.
- [ ] **Metric reduction.** Every `.item()` in the train/val/test loops (`:498-503`, `:597-603`,
      `:692-695`) is rank-local. Val loss currently drives best-model selection — under DDP you'd be
      selecting on 1/N of the data. Use `accelerator.gather_for_metrics()` (it handles the padding of
      the last uneven batch, which is exactly the subtle bug to understand).
- [ ] **Step accounting.** `total_steps = (len(train_loader) * epochs) // grad_accum` (`:423`) is
      computed on a *prepared* (already per-rank) loader. Under DDP this silently changes your LR
      schedule — the exact class of bug that already burned you once (`bigger-model-training` memory:
      the scheduler used the wrong `total_steps` and the run never converged). Derive `total_steps`
      from `--max-steps`, and assert it's world-size-independent.
- [ ] **Gradient accumulation without redundant all-reduce.** `accelerator.accumulate()` handles this,
      but implement `model.no_sync()` manually once in a scratch script so you understand what it
      does — this is a top-5 interview question.
- [ ] **`torch.compile` ordering.** Currently applied *before* `prepare()` (`:353`), giving
      `DDP(OptimizedModule)`. Try both orderings, measure, and learn why the recommended order is
      compile-then-wrap for `torch.compile` + DDP (and what `ddp_optimizer`/graph-break bucketing does).
- [ ] **EMA under DDP.** `update_ema` runs on every rank (`:494`) — redundant but consistent.
      `swap_ema_weights` mutates live params before validation (`:565`); confirm every rank does it
      identically or you desync. Consider keeping EMA on rank 0 only and broadcasting — measure the cost.
- [ ] **Checkpoint/resume under DDP.** `accelerator.save_state` writes `random_states_{0..N}.pkl`.
      Verify: save at step 500 with world_size=4, resume, and confirm bitwise-identical loss at step
      501 vs. the uninterrupted run. Also verify **world-size change on resume** (save at N=4, resume
      at N=2) — this is what makes spot instances usable.
- [ ] **Elastic / preemption.** Add `--max-restarts` to torchrun and a SIGTERM handler that checkpoints.
      Test by `kill`ing a rank mid-run.
- [ ] **Distributed Muon.** This is your differentiator. `torch.optim.Muon` orthogonalizes 2D weights
      via Newton–Schulz; under DDP *every rank computes the identical NS iteration on the identical
      all-reduced gradient* — pure redundant compute. Implement the sharded version: partition the
      2D params round-robin across ranks, each rank runs NS only on its slice, then `all_gather` the
      updates. Measure the step-time saving on the XL model in Phase 4. Reference: the Moonshot AI
      "Muon is Scalable" distributed-Muon writeup and Keller Jordan's original Muon post.
      Note the FSDP trap: NS needs the *full* matrix, so naive Muon on FSDP-sharded params is wrong.

### The correctness harness (build this — it's the thing that proves you know what you're doing)
- [ ] `tests/test_distributed.py`, runnable on CPU/gloo with 2–4 ranks in CI:
      - **Equivalence:** N=1 with batch B×4 produces the same loss trajectory as N=4 with batch B
        (same effective batch, same seed) to within fp tolerance. This single test catches almost every
        DDP bug.
      - **Gradient sync:** after `backward()`, all ranks have identical grads.
      - **Sampler coverage:** across one epoch, the union of indices seen by all ranks == the dataset,
        with no duplicates (except the documented `drop_last` tail).
      - **`set_epoch`:** shuffle order differs between epochs (Accelerate's `DataLoaderShard` handles
        this; verify it actually does with your loader).
      - **Determinism:** same seed + same world size → same loss.
- [ ] Add pytest to the dev group (it isn't there today) and a `.github/workflows/test.yml` that runs
      the gloo tests on CPU. Free CI, real distributed coverage.

**Learn:** "PyTorch Distributed: Experiences on Accelerating Data Parallel Training" (VLDB 2020) —
gradient bucketing and the backward/all-reduce overlap. `torchrun` rendezvous. NCCL vs gloo. Ring
vs. tree all-reduce.

**Acceptance:** equivalence test green at world sizes 1/2/4 on CPU and on Kaggle's 2× T4, resume test
green including a world-size change, `docs/distributed-log.md` updated.

---

## Phase 3 — Measure scaling, don't assume it (first rental, ~€47)

- [ ] **Strong scaling:** fixed global batch, N ∈ {1,2,4,8}. Plot speedup and efficiency. Expect a
      knee — find it and explain it.
- [ ] **Weak scaling:** fixed per-GPU batch, N ∈ {1,2,4,8}. Plot samples/s and step time.
- [ ] **Break down where the time goes** at N=8: compute vs. all-reduce vs. dataloader stall. Use
      `torch.profiler` with distributed trace, or `nvidia-smi dmon` + NCCL timing.
- [ ] **Tune the comms:** `DDP(bucket_cap_mb=...)`, `gradient_as_bucket_view=True`,
      `static_graph=True`, and the **bf16 compression comm hook**
      (`torch.distributed.algorithms.ddp_comm_hooks`). Measure each. An 11.7M model has a tiny gradient
      payload, so expect these to matter *less* here — knowing *why* is the lesson.
- [ ] **Compare interconnects:** the same strong-scaling curve on 8× PCIe 4090 vs. 8× NVLink A100.
      This is a genuinely impressive plot to bring to an interview.
- [ ] **Large-batch stability:** at N=8 your effective batch is 8× larger. Apply linear LR scaling +
      longer warmup and check convergence isn't hurt. Reference: "Accurate, Large Minibatch SGD"
      (Goyal et al.).
- [ ] Write `docs/distributed-training.md` with the plots and the numbers. Add a figure to
      `paper/main.tex` §Infrastructure.

**Acceptance:** scaling plots checked in, with a written explanation of the efficiency loss at N=8.

---

## Phase 4 — Memory & sharding: FSDP2 (~€30)

Needs a big model to be meaningful.

- [ ] Add a `--preset {base,large,xl}` to the trainer. `xl` ≈ 1536 dim / 32 layers ≈ 900M params.
      Nothing about `Pos2MoveV2` blocks this — it's just config (`trainer:298-306`). **Label it a
      scaling testbed in the code comment**, not a strength attempt.
- [ ] Establish the OOM baseline: at what size does DDP OOM on 4× A100 80 GB? Write it down.
- [ ] **Activation checkpointing** on the transformer blocks first (cheapest memory win). Measure the
      memory saved and the throughput cost — the classic ~30% compute for ~60% activation memory trade.
- [ ] **FSDP2** (`torch.distributed.fsdp.fully_shard`, the DTensor-based API in torch 2.11 — not the
      deprecated `FullyShardedDataParallel`). Shard the transformer blocks. Compare against DDP on
      peak memory, step time, and max trainable model size.
- [ ] Map FSDP's knobs onto the **ZeRO stages** (1 = optimizer state, 2 = + gradients, 3 = + params)
      so you can talk in the vocabulary the job ads use. Read the ZeRO paper.
- [ ] **Mixed-precision policy:** param/reduce/buffer dtypes independently. Understand why gradient
      reduction in fp32 while params are bf16 is often the right default.
- [ ] **CPU offload:** measure the throughput cliff. Know when it's worth it (almost never for
      training, sometimes for fine-tuning).
- [ ] **Sharded checkpointing** (`torch.distributed.checkpoint`) — a 900M model's optimizer state
      through `accelerator.save_state` is slow and rank-0-memory-bound. Verify a DCP checkpoint
      reloads at a *different* world size.
- [ ] Connect back to Muon: FSDP shards the 2D weight matrices, so Newton–Schulz on a shard is
      mathematically wrong. Either exclude Muon params from sharding, or all-gather before NS.
      Document the choice — this is a sharp, specific thing to have an opinion about.

**Acceptance:** a table of {DDP, DDP+ckpt, FSDP2-full-shard, FSDP2+offload} × {peak mem, samples/s,
max params that fit} on 4× A100.

---

## Phase 5 — Model & pipeline parallel (~€25)

At 900M params on 80 GB cards, TP/PP are *not* needed. Do them anyway, deliberately, and measure the
overhead so you can say exactly when they start paying.

- [ ] **Tensor parallel** with `torch.distributed.tensor.parallel` on `Pos2MoveV2`'s attention and MLP:
      `ColwiseParallel` on qkv/up-projections, `RowwiseParallel` on out/down-projections. Build a 2D
      `DeviceMesh` (TP × DP). Gotcha specific to this model: the **chess-geometry relative attention
      bias** (`pos2move_v2.py:11-52`) is indexed per head — you must shard it consistently with the
      head split or you'll get silently wrong attention.
- [ ] **Sequence parallel** on the norms. Note the joke here: context is 67 tokens, so this is pure
      overhead — that's a legitimate finding to report.
- [ ] Measure TP=2 vs TP=1 at fixed global batch. Expect it to be *slower*. Explain why in terms of
      the all-reduce-per-layer cost vs. the model's arithmetic intensity.
- [ ] **Pipeline parallel** with `torch.distributed.pipelining`: split the 32 blocks into 2–4 stages,
      run GPipe and 1F1B schedules, and measure the bubble. Compute the theoretical bubble fraction
      `(S-1)/(M+S-1)` and check your measurement against it.
- [ ] Optional stretch: 3D parallelism (DP × TP × PP) on 8 GPUs. Read `torchtitan` as the reference
      implementation — it's the cleanest modern example of exactly this.

**Learn:** Megatron-LM paper (TP), GPipe + PipeDream (PP), `torchtitan` source.

**Acceptance:** `docs/distributed-training.md` gains a section "when does model parallelism pay?"
with your own measured crossover point.

---

## Phase 6 — One real production run (~€80)

Cash it in on something the project actually wants.

- [ ] Pick the run that's genuinely untested. Per `doc/selfplay_rl.md` and your own results, more
      params did *not* help — so the interesting axis is **more and more diverse data**. Extend
      `scripts/build_db.py` beyond elite games (broader Elo band, more months) to ~10M games, build
      shards with Phase 1's tooling, and train the 11.7M or 46M model on 8 GPUs.
- [ ] Gate the result the way this repo already does: `scripts/engine_match.py` head-to-head vs. v2.1
      and `scripts/tune_vs_stockfish.py`. A negative result is fine and publishable — this repo already
      has four documented negative results and they're a strength.

---

## Files you'll touch

**Modify**
- `src/chesstransformer/trainers/pos2move_v2_trainer.py` — the bulk of Phase 2. Consider extracting
  the loop out of the 490-line `main()` first; it'll be unmaintainable otherwise.
- `src/chesstransformer/datasets/h5_lichess_dataset.py` — worker seeding fix; keep as the reference
  path for the equivalence test.
- `src/chesstransformer/optimizer.py` — distributed Muon.
- `src/chesstransformer/models/transformer/pos2move_v2.py` — `--preset` sizes, TP shard plan, activation checkpointing hooks.
- `pyproject.toml` — add pytest to the dev group.

**Create**
- `scripts/build_shards.py`, `src/chesstransformer/datasets/flat_shard_dataset.py`
- `scripts/profile_training.py`, `scripts/launch_train.sh`, `accelerate/ddp.yaml`
- `tests/test_distributed.py`, `.github/workflows/test.yml`
- `docker/train.Dockerfile`
- `docs/distributed-training.md`, `docs/distributed-log.md` (results + cost log)

---

## Verification

Run at each phase boundary — cheap, local, and it's what stops you from paying to discover a bug:

```bash
# Phase 0/1 — single-GPU throughput regression
uv run scripts/profile_training.py --steps 200 --data data/shards/

# Phase 2 — distributed correctness, CPU/gloo, no GPU needed
uv run pytest tests/test_distributed.py -v
torchrun --nproc_per_node=4 scripts/launch_train.sh --smoke   # 4 ranks on cuda:0

# Phase 2 — the equivalence check that matters
#   N=1 batch 1024 vs N=4 batch 256, same seed -> same loss curve
torchrun --nproc_per_node=1 ... --batch-size 1024 --max-steps 200
torchrun --nproc_per_node=4 ... --batch-size 256  --max-steps 200
# diff the two TensorBoard scalar streams; they must match within fp tolerance

# Phase 2 — resume + world-size change
# save at step 500 with N=4, resume with N=2, confirm loss continuity

# Every rented session — first command, always
./scripts/launch_train.sh --smoke   # must pass in <2 min or you go back to local
```

Strength gates stay unchanged and remain the final word on any model you produce:
`scripts/engine_match.py`, `scripts/tune_vs_stockfish.py`, `scripts/bench_inference.py --check`.

---

## Suggested order and rough effort

| Phase | Effort | Cost | Skip-able? |
|---|---|---|---|
| 0 — Measure | 1 day | €0 | No — everything else depends on it |
| 1 — Input pipeline | 2–3 days | €0 | No — DDP on a CPU-bound loader is pointless |
| 2 — DDP + test harness | 4–5 days | €0 | No — this is the core deliverable |
| 3 — Scaling measurement | 1–2 days | €47 | No — the numbers are the proof |
| 4 — FSDP2 / sharding | 3–4 days | €30 | Yes, if budget tightens |
| 5 — TP / PP | 3–4 days | €25 | Yes — lowest ROI for this model size |
| 6 — Production run | 1 day + wall-clock | €80 | Yes — drop if 4+5 overrun |

If the budget gets tight, protect Phases 0–3. A candidate who can show a measured scaling curve, an
equivalence test, and a written explanation of where efficiency was lost is more convincing than one
who name-drops FSDP and Megatron without numbers.
