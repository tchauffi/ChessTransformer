# Distributed training — measurement log

Running record of every measurement and every euro spent. See
[`doc/distributed_training.md`](../doc/distributed_training.md) for the roadmap.

---

## Phase 0 — Single-GPU baseline (2026-08-05)

**Hardware:** RTX 5070 Ti (16 GB, ~89 TFLOPS achievable bf16 dense), 16 CPU cores, 30 GB RAM.
**Software:** torch 2.11.0+cu128, bf16 autocast.
**Model:** Pos2MoveV2, 11,651,530 params (embed 256 / 16 layers / 8 heads), 64-token input.
**Data:** `data/elite_db.h5`, 1,533,352 games / 148M plies.
**Reproduce:** `uv run scripts/profile_training.py --batch-size 512 --compile`

### Headline

> ⚠️ **The end-to-end row of the original Phase 0 table was wrong.** It reported 2,276 (eager) and
> 2,272 (compiled) and concluded that `torch.compile` bought *zero* end-to-end throughput. That was
> a bug in `scripts/profile_training.py`: `count_flops` ran before bench C, and `FlopCounterMode` is
> a `TorchDispatchMode` whose fallback de-optimises the compiled model for every subsequent call. So
> bench C measured the **eager** model in both runs — which is exactly why the two numbers came out
> identical. Fixed 2026-08-05 by moving `count_flops` after every timing bench. Corrected numbers
> below; everything else in this section (component breakdown, dataloader scaling, root causes) was
> measured independently and stands.

Re-measured with the fixed harness, 8 workers:

| Measurement | eager | `--compile` |
|---|---:|---:|
| Compute-only ceiling (synthetic batch on GPU) | 2,551 samples/s | **5,643 samples/s** |
| MFU | 13.8 % | **30.8 %** |
| Dataloader-only, sustained, 8 workers | 3,797 samples/s | 3,801 samples/s |
| **Actual end-to-end** | **2,307 samples/s** | **4,422 samples/s** |
| verdict | compute-bound (0.67×) | **data-bound (1.48×)** |

`torch.compile` is worth **1.92× end-to-end**, not 0×. The pipeline *is* data-bound once compiled —
the loader ceiling of 3,801 sits below the 5,643 compute ceiling — so the Phase 1 conclusion holds.
But it was costing ~22 % of the step, not 60 %, and the original "paying for compile and getting
nothing" claim was false.

**Lesson worth keeping:** the artifact was invisible because the numbers were plausible. Two
benchmarks agreeing to within 0.2 % should have read as suspicious, not as a clean result.

### Step breakdown (compiled, batch 512, per micro-batch)

| Stage | ms | share |
|---|---:|---:|
| forward | 28.4 | 28.9 % |
| backward | 55.3 | 56.3 % |
| clip_grad_norm | 0.8 | 0.9 % |
| **Muon step** | **13.0** | **13.2 %** |
| AdamW step | 0.7 | 0.7 % |

Muon's cost is **fixed at ~12.8 ms regardless of batch size** (identical at 512 and 1024) —
it's Newton–Schulz on the weight matrices, not the activations. Two consequences:

- With the trainer's default `--grad-accum 4` it amortises over 4 micro-batches, so it's
  ~3.5 % of an optimizer step, not 13 %.
- Under DDP **every rank recomputes the identical 12.8 ms** on the identical all-reduced
  gradient. That is the concrete payoff for the distributed-Muon work in Phase 2.

### Dataloader scaling (random access across all 1.53M games)

| workers | samples/s | ms/batch | per-worker |
|---:|---:|---:|---:|
| 1 | 799 | 641.1 | 799 |
| 2 | 1,544 | 331.6 | 772 |
| 4 | 2,860 | 179.0 | 715 |
| 8 | 4,040 | 126.7 | 505 |
| 12 | 5,132 | 99.8 | 428 |
| 16 | 6,708 | 76.3 | 419 |

Scaling is sub-linear past 4 workers — per-worker rate falls from ~800 to ~420 samples/s
as the 16 cores saturate and workers contend with the main process.

### The contention effect (worth internalising)

Measured in isolation compute does 5,643 and the loader 3,801 (8 workers); the naive prediction for
an overlapped pipeline is the slower of the two, 3,801. Actual end-to-end is 4,422 — better than the
prediction (prefetch absorbs variance) but well short of the compute ceiling. Isolated dataloader
benchmarks
**overestimate** what the loader delivers during training, because the 12 workers compete
with the main process for the same cores while it's busy issuing GPU work. Always confirm
against the combined number.

### Root causes identified

1. **`__getitem__` replays the whole game with python-chess** on every sample
   (`datasets/h5_lichess_dataset.py:142-154`) and enumerates all legal moves. ~800 samples/s
   per core is the ceiling that follows from that.
2. **Two large tensors are built per sample and never used.** `legal_moves_grid` (64×64 bool)
   and `legal_moves_mask` (full move vocab) are constructed at `:161-171` and returned at
   `:242,236`; the trainer only reads `legal_moves_planes`. Pure waste, both CPU and the
   5.8 MB/batch H2D transfer.
3. **Two `h5py.File` opens per sample** — `:108` for moves and `:219` for elo/result. The
   second one exists only to fetch three scalars that could live in RAM.
4. ~~**SDPA runs off the flash path.**~~ `pos2move_v2.py:119-124` passes `attn_mask=attn_bias`,
   which does disqualify the flash backend — but **this is not recoverable and not the reason MFU
   is 31 %.** Every alternative measured slower. See the Phase 1b negative result below.
5. ~~**RMSNorm falls off the fused kernel**~~ — the `Mismatch dtype between input and weight`
   warning is real in eager but **irrelevant under `torch.compile`**, which emits its own fused
   Triton norm kernel. Fixing it is worth 0.3 %. See Phase 1b below.

*(4 and 5 were wrong. They were inferred from ATen warnings rather than measured on the compiled
model; both were tested and refuted on 2026-08-05. Items 1-3 were measured and stand.)*

### What this means for multi-GPU

Feeding 8 GPUs of *this* class needs ~45,000 samples/s. At the best observed per-worker rate
(~800/s, uncontended) that's ~57 dedicated cores — and A100/H100 compute is faster still, so
the real requirement is higher. **No realistic 8-GPU node has enough CPU to feed this
dataset in its current form.** Phase 1 is a hard prerequisite for Phase 3, not an optimisation.

### Correction to the original plan

The roadmap asserted the trainer was already CPU-bound and cited `RepeatingDataloader` as
proof. Half right. In **eager** mode it is genuinely compute-bound (dataloader time is 0.44×
compute). It only becomes data-bound **with `--compile`**, which is what `resume_training.sh`
actually uses. The conclusion (do Phase 1 first) survives; the reasoning needed fixing.

### Memory

Batch 1024 peaks at 14.4 GB of 15.5 GB — effectively the ceiling on this card. Batch 512 is
the safe working point and loses nothing (5,643 vs 5,838 samples/s compute-only).

### Targets for Phase 1 (set here, results below)

| Metric | baseline | target | **achieved (Phase 1a)** |
|---|---:|---:|---:|
| End-to-end samples/s (compiled, bs 512, 8w) | 4,422 | > 5,400 | 4,903 |
| Loader ceiling, 8 workers | 3,801 | > compute | **7,572** ✅ |
| Verdict | data-bound | compute-bound | **compute-bound** ✅ |
| Bytes per collated batch | 5.81 MB | < 1.5 MB | 2.69 MB |

End-to-end fell short of 5,400 because the target was derived from the bogus 2,272 figure and
assumed 3,100 samples/s of recoverable starvation. Only ~1,200 existed. The pipeline is now
compute-bound, which is the condition that actually mattered.

---

## Phase 1 — Input pipeline: where the per-sample cost lives (2026-08-05)

Design spec: [`doc/dataset_shards.md`](../doc/dataset_shards.md).

### `__getitem__` breakdown (300 random samples, single core)

| Component | ms | share |
|---|---:|---:|
| `h5py.File` open + read `moves[g]` (gzip) | 0.352 | 30 % |
| **2nd `h5py.File` open for elo/result** (`:219-222`) | **0.300** | **25 %** |
| legal moves → 3 tensors (`:160-171`) | 0.169 | 14 % |
| python-chess replay | 0.143 | 12 % |
| position tokenize | 0.011 | 1 % |
| tensor/dict construction, remainder | ~0.21 | 18 % |
| **total** | **1.186** | → **843 samples/s/core** |

The replay is *not* the dominant cost — I/O is, and half of that I/O is fetching data nothing reads.
The trainer consumes only 9 keys (`:463-472`); `white_elo`/`black_elo` are never read, and `result`
is 1.5 MB of `int8` that could live in RAM.

### HDF5 access pattern

| pattern | ms per `moves[g]` |
|---|---:|
| random, reopen each time (current) | 0.352 |
| random, persistent handle | 0.220 |
| **sequential, persistent handle** | **0.017** |

Gzip chunks of 10,000 games: a random read decompresses a whole chunk. **The shard build script must
iterate games sequentially** — 26 s vs 5.6 min for a full pass.

### Batch bytes (B = 512)

| | MB |
|---|---:|
| collated today | 5.81 |
| of which the model reads | 2.69 |
| legal moves as indices instead of a dense mask | 0.10 |

`legal_moves_grid` (2.10 MB) and `legal_moves_mask` (1.01 MB) are built per sample and read by
nothing. Legal `(from_square, plane)` pairs measured over 400 positions: mean 33.9, p95 48,
p99.9 53, max 53 — so a fixed cap of 64 uint16 indices (129 B) replaces the 4,672 B dense mask.

### Chunk cache

| `moves[g]` random read | ms |
|---|---:|
| reopen each time (current) | 0.352 |
| persistent handle, default `rdcc_nbytes` = 1 MB | 0.221 |
| persistent handle, `rdcc_nbytes` = 32 MB | **0.121** |
| persistent handle, 256 MB | 0.112 |

The default 1 MB chunk cache cannot hold a 10,000-game gzip chunk. 32 MB is the knee.

### Phase 1a complete — quick wins landed (2026-08-05)

Final state: metadata (`white_elo`/`black_elo`/`result`) held in RAM and indexed by actual game
index; `_get_game_moves` reads only `moves` through a lazy per-process handle with a 32 MB chunk
cache; `legal_moves_grid` and `legal_moves_tokens` no longer built; 12 keys returned.

| | single core | 1w | 2w | 4w | 8w | 12w | MB/batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (41d405c) | 1.123 ms | 869 | 1,707 | 3,034 | 5,252 | 6,090 | 5.81 |
| **now** | **0.568 ms** | **1,703** | **3,334** | **5,874** | **10,306** | **11,513** | **2.69** |

**1.98× single-core, 1.89× at 12 workers, batch bytes −54 %.** Matches the `lean` prototype
(0.564 ms) — nothing left in the quick-win set.

Verification: equivalence vs baseline over 2,000 samples × all 12 returned keys with the ply forced
identical, PASS. ELO-filtered construction (`min_elo=2400`) cross-checked against raw HDF5 — the
RAM arrays are unfiltered and indexed by `actual_game_idx`, which is the easy thing to get wrong
here. Fork safety re-verified after the change.

**4 workers now clear the compiled compute ceiling of 5,675 samples/s in isolation** (5,874); 8
workers give 1.3× headroom over it, which is the margin that survives the contention discount.

### End-to-end, fixed harness, 8 workers, `--compile`

| | compute ceiling | loader (8w) | **end-to-end** | MFU | verdict |
|---|---:|---:|---:|---:|---|
| baseline, eager | 2,551 | 3,797 | 2,307 | 13.8 % | compute-bound |
| baseline, compiled | 5,643 | 3,801 | **4,422** | 30.8 % | data-bound (1.48×) |
| dataset rewrite, compiled | 5,675 | 7,572 | **4,903** | 31.6 % | compute-bound (0.75×) |
| **+ `pin_memory=True`** | 5,645 | 10,194 | **5,070** | 31.1 % | **compute-bound (0.55×)** |

**End-to-end 4,422 → 5,070 (+15 %), at 90 % of the compute-only ceiling.**

### `RepeatingDataloader` removed — the largest real win, and invisible to the profiler

`RepeatingDataloader(repeat_factor=2)` yielded every batch **twice**. At the default `--grad-accum 4`
an optimizer step consumed `[A, A, B, B]`, so the accumulated gradient carried 2 × 512 unique samples
instead of 4 × 512 — **half of all training compute produced no new information**, and `__len__`
reported double the real batch count to the LR schedule.

`scripts/profile_training.py` never wrapped its loader, so this never appeared in any throughput
number here. In the actual trainer, removing it **doubles unique samples per optimizer step** at
identical wall-clock. That dwarfs everything else measured in this section.

Consequence for run configs: an epoch is now 2,955 batches instead of 5,910 (1,513,352 / 512), so
`--epochs N` yields half the optimizer steps it used to. Derive schedules from `--max-steps`.
Resuming an existing checkpoint is unaffected — `scheduler_config` is restored from
`trainer_state.json` when present (`:423-431`).

Verified: 40-step smoke run before and after `pin_memory` gives identical `train_loss=10.2789` and
`val_loss=11.1801`, with validation, test and checkpointing all exercised.

**+11 % end-to-end, and the verdict flips from data-bound to compute-bound.** End-to-end is now at
86 % of the compute-only ceiling — the loader has stopped being the wall on this GPU.

The modest end-to-end gain despite a 1.99× loader speedup is the correct outcome, not a
disappointment: the pipeline was only ~22 % starved, so ~22 % was all there was to recover. The
value of the change is the headroom (loader at 1.3× the compute ceiling instead of 0.67×), which is
what Phase 3 spends.

Remaining ceiling on this GPU is now compute, and the two known items are unchanged from Phase 0:
SDPA off the flash path (`attn_mask=attn_bias`) and RMSNorm off its fused kernel (bf16/fp32
mismatch). MFU is 31.6 %.

### Intermediate steps — commits f04d20d, c323456, 0132e7e (2026-08-05)

Single h5 open per game (metadata folded into `_get_game_moves`, cached alongside `moves`); lazy
per-process handle with `rdcc_nbytes=32<<20` and a PID guard; 7 unused keys removed from the
returned dict (v1 trainer deprecated).

| | single core | 1 worker | 4 | 8 | 12 | MB/batch |
|---|---:|---:|---:|---:|---:|---:|
| baseline (41d405c) | 1.127 ms | 867 | 3,040 | 5,209 | 6,145 | 5.81 |
| **now** | **0.931 ms** | **1,040** | **3,639** | **6,350** | **7,374** | **2.69** |
| all quick wins (prototype) | 0.558 ms | 1,724 | 5,935 | 10,367 | 11,909 | 2.68 |

**1.20×** at 12 workers, batch bytes halved. Equivalence vs baseline verified over 1,500 samples ×
all keys with the ply forced identical. Fork safety verified: parent reads a sample, then 4 workers
fork — each reopens via the PID guard, all 4,096 samples return. Picklable after use (`spawn`).

**Removing keys from the returned dict saves transfer, not CPU.** Single-core is flat (0.928 →
0.931 ms) because `legal_moves_grid` and `legal_moves_tokens` are still allocated and filled at
`:211-218`; only the collate and H2D copy got cheaper. The remaining 1.67× to the prototype is
`result` from RAM (`results` is loaded at `:68` but never stored) and not building the dead tensors.

**Correction:** an earlier entry here claimed `legal_moves_mask` was read by no trainer. It was read
by `position2move_trainer.py`; dropping it retires the v1 trainer rather than being free. That was
the intended trade (commit 0132e7e), but `position2move_trainer.py` now raises `KeyError` on
`legal_moves_mask` and `halfmove_clock` rather than failing with a deprecation message.

### Quick wins, measured together (2026-08-05)

Persistent lazy per-worker handle + 32 MB rdcc + `result` preloaded to RAM + Elos dropped +
`legal_moves_grid`/`legal_moves_mask` removed + only the 9 consumed keys returned + `pin_memory`.

| DataLoader samples/s | 1 worker | 4 | 8 | 12 | MB/batch |
|---|---:|---:|---:|---:|---:|
| current | 891 | 3,238 | 5,608 | 6,139 | 5.81 |
| **quick wins** | **1,670** | **6,130** | **10,520** | **11,323** | **2.68** |

Single core 1.150 → 0.557 ms/sample (**2.06×**). Equivalence verified over 2,000 samples × 9 keys
with the sampled ply forced identical: byte-identical, `legal_moves_planes` included.

Prototype: `scratchpad/lean.py` (not committed — implementation is the user's).

**Sum-of-parts was wrong again.** Estimating from the isolated component timings gave ~0.67 ms;
the persistent handle alone delivered 1.140 → 1.005 ms, ~a third of its isolated saving. The
isolated metadata benchmark reopened the file with nothing else live, whereas in `__getitem__` the
preceding `_get_game_moves` leaves the file warm. Same failure mode as the isolated dataloader
benchmark above — measure the whole, not the sum.

### End-to-end (with `--compile`)

**Not yet measured.** Run `scripts/profile_training.py --batch-size 512 --compile` after applying.

---

## Phase 1b — Compute: a negative result (2026-08-05)

With the loader fixed, the ceiling is compute at MFU 31 %. Phase 0 named two root causes. **Both are
false under `torch.compile`.**

### Kernel-level breakdown (compiled, bs 512, 94.0 ms GPU time/step)

| category | ms/step | % |
|---|---:|---:|
| gemm/matmul | 38.15 | 40.6 % |
| **attention (fmha)** | **25.17** | **26.8 %** |
| elementwise / copy | 17.79 | 18.9 % |
| norm | 10.47 | 11.1 % |

The single largest kernel is `fmha_cutlassB_bf16_aligned_64x64_k32_dropout_sm80` at 18.61 ms —
attention **backward**, 19.8 % of the step, for a layer that is only ~5 % of the model's FLOPs.

### Everything tried

| lever | samples/s | vs baseline | verdict |
|---|---:|---:|---|
| baseline (real cfg) | 5,641 | 1.000× | |
| RMSNorm weight cast to bf16 | 5,684 | 1.003× | **no effect** — inductor emits its own fused `triton_per_fused__fused_rms_norm` kernel; the ATen dispatch warning never fires under compile |
| GQA broadcast (drop `repeat_interleave`) | 5,184 | 0.915× | **worse**, and +1.25 GB peak |
| FlexAttention + `score_mod` | — | 0.13× | 7.5× slower: its 128-wide blocks waste ~half the work at T=67 |
| SDPA forced cuDNN | — | — | no kernel for this shape |
| SDPA MATH backend | — | 0.45× | worse |
| manual matmul attention, compiled | — | 0.28× | worse |
| `layer_drop` 0.0 vs 0.1 | 5,630 / 5,641 | 1.000× | irrelevant |
| **`--compile-mode max-autotune`** | **5,841** | **1.033×** | **the only free win** |
| attention `dropout_p=0` | 5,886 | 1.040× | model change — needs a strength gate |
| all dropout 0 | 5,973 | 1.059× | model change, −0.5 GB |
| batch 1024 | 5,816 | 1.028× | 12.6 GB peak, little headroom |
| *no chess-geometry bias (diagnostic)* | *6,520* | *1.155×* | not viable — it is the model |

### Why the flash path doesn't matter

Per layer, fwd+bwd at bs 512: no mask 1.01 ms, bias without grad 1.59 ms, **current 1.92 ms**. So the
bias costs 0.58 ms and its *gradient* another 0.33 — `bias_table` is a `Parameter`, so the backward
must reduce a `(1, H, 67, 67)` bias gradient over B=512. Getting onto the flash path would be worth
~15 %, but flash accepts no `attn_mask` at all, and every alternative that does accept one is slower
than what we already have.

### Conclusion

**There is no compute bug to fix.** MFU 31 % is what an `embed_dim=256`, `T=67`, 11.7M-param model
gets on this card: no hotspot, just many small kernels that are launch- and bandwidth-bound rather
than FLOP-bound. The lever for MFU is a *bigger model* — which is exactly what the Phase 4
`--preset xl` scaling testbed exists to measure.

Recommended: `--compile-mode max-autotune` for real runs (+3.3 %, costs a few minutes of first
compile). Dropout changes are a modelling decision, not a perf fix; gate them with
`scripts/engine_match.py` if you want the extra 4-6 %.

**Third correction in this session, same pattern.** Phase 0's root causes came from reading ATen
warnings and reasoning about kernel dispatch, not from measuring the compiled model. The warning was
real and the conclusion drawn from it was wrong. Measure the thing you actually run.

Harness fix: `MODEL_CFG` in `scripts/profile_training.py` had `layer_drop=0.0` while the shipped
model uses `0.1` — corrected, though it turns out to cost nothing.

---

## Convergence check after the Phase 1a changes (2026-08-05)

Sample content was already proven byte-identical, so the only change that alters training dynamics is
removing `RepeatingDataloader`. A/B against a copy of the *current* trainer with only that class
restored — same seed, same config, 1,200 optimizer steps each.

`--data data/elite_db.h5 --batch-size 512 --grad-accum 4 --max-steps 1200 --warmup-steps 100
--num-workers 8 --compile --seed 42 --epochs 5`

| at equal optimizer steps (= equal wall-clock) | A: current | B: with repeat | delta |
|---|---:|---:|---:|
| train loss @600 | 4.783 | 4.942 | **−0.159** |
| train loss @1200 | 4.477 | 4.618 | **−0.141** |
| train acc @1200 | 0.3008 | 0.2773 | **+2.3 pp** |
| **test loss** | **6.0192** | 6.0673 | **−0.048** |
| **test top-1 acc** | **0.2246** | 0.2156 | **+0.9 pp** |
| test legal acc | 0.2539 | 0.2447 | +0.9 pp |
| unique samples consumed | 2.46 M | 1.23 M | 2× |

**Converges cleanly and strictly better at equal wall-clock.** No divergence, no NaN, smooth descent,
val and test consistent with train. Runs kept at `logs/pos2move_v2/run_001_20260805_232614` (A) and
`run_002_...233348` (B).

Note the mechanism precisely: with `--grad-accum 4` the repeat made a step accumulate `[A, A, B, B]`,
so the gradient was `(gA + gB)/2` — a correct gradient over an *effective batch of 1024 unique
samples* instead of 2048. It was not a corrupted gradient, just a halved effective batch for the same
compute. That is why the gain is a real but modest ~0.9 pp rather than dramatic.

**One axis I cannot cleanly compare:** "equal unique samples" (A@k vs B@2k) makes B look better, but
both runs share `total_steps=1200`, so at step 2k B is twice as far through its LR decay. The
comparison is confounded by the schedule and should not be read as a result. Equal-wall-clock is the
clean axis.

Two harness bugs of mine on the way, both caught before they produced a wrong conclusion:
`--epochs 1` capped run A at 738 steps (one epoch = 1,513,352/512/4) so `--max-steps 1200` never
bound, while B's doubled `__len__` let it reach 1200; and the standalone B script resolved
`Path(__file__).parents[3]` to the wrong data path.

---

---

## Phase 1b — flat shards built and verified (2026-08-16)

Spec and rationale: [`../doc/dataset_shards.md`](../doc/dataset_shards.md) §3-§7. Tools:
`scripts/build_shards.py`, `scripts/verify_shards.py`, `datasets/flat_shard_dataset.py`.

`data/shards/elite_k16`: 24,498,538 samples from 1,533,352 games, K=16, 202 B/record, 4.95 GB,
built in **5.8 min on 14 workers** (70,290 samples/s). Zero games dropped as unreplayable.

### Loader, measured on the real build

| workers | samples/s | MB/batch (B=512) |
|---:|---:|---:|
| 0 | **144,272** | 0.10 |
| 1 | 68,796 | 0.10 |
| 4 | 115,585 | 0.10 |
| 8 | 138,911 | 0.10 |

Against the Phase 1a numbers: loader ceiling 7,572 → **138,911** (18×), per-sample CPU 0.568 ms →
**0.0069 ms** (82×), bytes per batch 2.69 MB → **0.10 MB** (27×). Every §6 target hit; the loader
one by 3.5×.

**`num_workers=0` is the fastest setting, and the 1-worker dip is the tell.** A batch fetch is now a
memmap read plus a fancy index — cheaper than the IPC round-trip to hand the work to another
process. One worker pays full IPC with nothing to amortise it (68,796, *worse than in-process*); it
takes ~4 workers to climb back. The practical consequence for Phase 2 is better than the plan
assumed: the budget was "≤ 4 workers per GPU", and 8 ranks × 0 workers leaves all 16 cores to the
ranks.

The input pipeline is no longer a constraint on this machine by any margin that matters: 144k
samples/s against a compiled compute ceiling of 5,643. The 8-GPU arithmetic that justified this work
(8 × 5,643 ≈ 45,000 samples/s, previously needing ~57 dedicated cores) is now 31 % of what a single
process serves.

### Correctness

Four checks green on the real build (`verify_shards.py`):

| check | result |
|---|---|
| encoding equivalence | 5,747 samples across 3 shards, **every field byte-identical** to an untouched `HDF5ChessDataset` replay |
| ply distribution | TVD **0.024** vs the online sampler, mean ply 37.59 vs 37.17 |
| split disjointness | 1,473,452 / 29,950 / 29,950 games, no overlap |
| padding safety | scattered bits == `legal_cnt` for all 4,096 sampled rows |

Legal-move cap C=64 was hit by **807 of 24.5 M samples (0.0033 %)**, under the 0.01 % rebuild
threshold, and the target-index-first rule meant no capped row lost its label.

### Two measurement mistakes, both mine, both caught by the harness

**The verifier was wrong before the build was.** Check 2 first drew its reference games from the
whole 1.53 M-game file while the shard under test covered only the first 4,000. Games sit in the
HDF5 in roughly chronological order and their length distribution drifts across the corpus, so it
reported TVD 0.032 with the shard visibly under-representing 100+ ply positions — corpus drift
misread as a sampler bug. Restricting the reference draw to the split's own game ranges moved mean
ply from 33.81-vs-36.87 to 33.73-vs-33.80. *Same class of error as the sum-of-parts estimates in
Phase 0 and 1a: the control has to match the thing being measured.*

**Sum-of-parts, a third time, and this one I avoided by measuring both.** The plan said "sample
without replacement" and I nearly shipped it unquestioned. Building both modes and histogramming
them showed the real trade: `unique` costs TVD 0.024 of marginal drift, `iid` costs 13.7 % duplicate
rows. Neither is free and neither was predictable from the spec. `unique` won on distinct positions
per byte, but it is now a `--sample-mode` flag with the numbers written down rather than an
assumption.

### Latent bug the port surfaced

`compute_loss` uses `is_white` as a boolean *mask* (`target_value[white_win & is_white]`). The shard
path returns it as `uint8`; passing that through unchanged silently converts the mask to fancy
indexing and scatters value targets onto the wrong rows, with a loss curve that still looks
plausible. `unpack_batch` now casts it explicitly and both paths are asserted to hand `compute_loss`
identical dtypes and shapes.

### Full corpus: `data/shards/full_k4` (same session)

`elite_db_full.h5`, 26,269,077 games, K=4 → **105,076,308 samples, 21.23 GB, 44.9 min on 14
workers** (38,990 samples/s). Zero games dropped, legal-cap hits 3,390 (0.0032 %). All four
correctness checks green.

Two things this build taught that the smaller one could not.

**The sampling drift scales with K, and the mechanism is now confirmed.** `full_k4`'s ply marginal
matches the online sampler to **TVD 0.0035** against `elite_k16`'s 0.024 — same code, same sampler.
That is exactly the without-replacement flattening predicted in `dataset_shards.md` §3: drift tracks
the sampling fraction `K / plies_available`, 17 % at K=16 and 4.8 % at K=4. The `unique` vs `iid`
trade is therefore only a real decision at high K.

**The optimal `num_workers` inverts with page-cache residency.**

| workers | `elite_k16` (4.95 GB, cached) | `full_k4` (21 GB, NVMe) |
|---:|---:|---:|
| 0 | **144,272** | 22,424 |
| 1 | 68,796 | 165,791 |
| 4 | 115,585 | 379,329 |
| 8 | 138,911 | **439,824** |

Cached, a fetch is a memory read, the only variable cost is IPC, and `workers=0` wins — with one
worker *worse than none*. Uncached, every fetch is a 4 KB page fault and the variable cost is queue
depth. The arithmetic checks out both ways: 22,424 samples/s × 4 KB ≈ 92 MB/s ≈ 22k IOPS is
single-threaded random read at QD1, and 439,824 ≈ 1.8 GB/s is what the drive does at depth 8. The
workers stopped being CPU parallelism and became I/O parallelism.

Rule for Phase 2: **size `num_workers` by whether the shard set fits RAM, not by core count.**
`elite_k16` wants 0 per rank, `full_k4` wants ~8 — and a rented box with more RAM may flip
`full_k4` back into the cached regime, which is a thing to re-measure rather than assume.

Note this also re-frames the "0 workers frees all 16 cores for the ranks" claim above: true for the
cached corpus, false for the disk-backed one, where 8 ranks × 8 workers would oversubscribe. Which
corpus a run uses now changes the CPU budget.

### Acceptance gate: loss equivalence — GREEN

2,000 steps, seed 42, `--batch-size 512 --grad-accum 4 --warmup-steps 100`. A = HDF5
`--min-elo 0 --num-workers 8`; B = `elite_k16 --num-workers 0`. `--max-steps` already overrides
`total_steps`, so both runs share an identical LR schedule despite `len(train_loader)` differing
15×. Runs at `logs/pos2move_v2/run_010_20260816_160119` (A) and `run_011_...161339` (B).

| steps | A ce | B ce | Δ | A legal_acc | B legal_acc |
|---|---:|---:|---:|---:|---:|
| 100-300 | 3.3853 | 3.3879 | +0.0026 | 0.2027 | 0.1997 |
| 600-1000 | 2.6695 | 2.6722 | +0.0026 | 0.3056 | 0.3053 |
| 1000-1400 | 2.4938 | 2.4917 | −0.0021 | 0.3348 | 0.3348 |
| 1700-2000 | 2.3010 | 2.2933 | −0.0077 | 0.3689 | 0.3696 |

**Batch-to-batch std of `train/step_ce` over steps 1700-2000 is 0.0725 (A) and 0.0643 (B).** Every
delta from step 100 on is ≤ 0.0077 — roughly 10 % of one batch's noise. The only larger window is
0-100, i.e. warmup, where the two paths' different first batches dominate. Green by any reading.

Val/test are **not** comparable between these runs and should not be quoted as an A/B: A's val is a
`random_split` over games with online ply sampling, B's is 10k rows from reserved shards. Different
evaluation sets. (Most of the apparent val gap is the value head — `val/value_loss` 0.5374 vs
0.5870, multiplied by `--value-loss-weight 5.0`.) Only the training curves are the gate.

### End-to-end: +2.7 %, which is the predicted result

Same two runs, steady state after step 200:

| | samples/s | ms/optimizer step | dataloader workers |
|---|---:|---:|---:|
| A — HDF5 | 5,724 | 357.8 | 8 |
| B — shards | **5,876** | **348.6** | **0** |

Phase 1a had already taken the pipeline compute-bound at 86 % of the compiled ceiling, so there was
no single-GPU throughput left for the shards to win, and `dataset_shards.md` §6 said so before this
was measured. Recording it plainly: **on one GPU these shards are worth ~3 %.**

The win is in the resource column. B matches A's throughput on **zero** dataloader workers against
A's eight — eight cores returned, on a 16-core box that has to feed 8 ranks in Phase 3. That, plus
the 26× loader headroom, is the whole case for this work, and it was the case the plan made.

### Gap found and closed: ELO filtering

`HDF5ChessDataset` filters by average ELO and the trainer's HDF5 path defaults to `--min-elo 2400`,
dropping 17 % of `elite_db.h5` and 33 % of `elite_db_full.h5`. `build_shards.py` had no such filter,
so **both shipped shard sets hold the unfiltered population.** The builder now takes
`--min-elo`/`--max-elo`, applied to games exactly as `HDF5ChessDataset.__init__` does and recorded
in `meta.json`; the gate above ran with `--min-elo 0` on the HDF5 side so both paths saw the same
games. The shards need a rebuild before they are a drop-in replacement for the default config.

Sanity check on that filter surfaced the corpus drift a third time: 2,750 of the *first* 4,000 games
are sub-2400, against 17 % across the whole file. **Any statistic taken from a prefix of this HDF5
is unrepresentative** — it has now caused one wrong verifier result and two moments of confusion.

### Open

Nothing in Phase 1. The shards need an ELO-filtered rebuild before they replace the default HDF5
config, but that is a build re-run, not open engineering. Phase 2 (DDP) is unblocked.

---

## Trainer switched from epochs to steps (2026-08-16)

Prompted by a simple question — *when does eval happen on shards?* — with a bad answer: only at
epoch end, and an epoch on shards is enormous.

| | samples in "1 epoch" | optimizer steps/epoch | validations per 70k-step run |
|---|---:|---:|---:|
| HDF5 (index = game) | 1,513,352 | 738 | 95 |
| `elite_k16` (index = sample) | 23,541,776 | 11,495 | 6 |
| `full_k4` | 103,434,492 | 50,505 | **1** |

Confirmed on the §5 A/B runs: A validated 3× (steps 739, 1478, 2000), B validated **once**, and only
because `--max-steps` forced a stop. Val loss is what selects `best_model`, so on `full_k4` the
"best" checkpoint would have been chosen from a single measurement.

The same `len(train_loader)` drives `total_steps_from_epochs`, so `--epochs N` also bought a 15-68×
different **LR decay horizon** depending only on which data flag was passed. That is the failure
mode that already cost one run (`bigger-model-training`: wrong `total_steps`, never converged), and
here it was armed by a flag that looks unrelated to the schedule.

**An epoch was never a comparable unit across the two data paths**, because the two datasets are
indexed by different things. Rather than paper over it with an `--eval-steps` flag, the loop is now
step-driven:

- `--max-steps` is the training-length knob. It bounds the loop *and* sets the LR horizon, and means
  the same thing on either dataset and at any world size.
- `--eval-steps` (default 1000) is the validation cadence. A final validation always runs, deduped
  when it lands on the same step as a periodic one.
- `--save-steps` is the only checkpoint cadence. Epoch checkpointing is gone.
- The loop cycles the dataloader as many passes as `--max-steps` requires, calling `set_epoch` on
  each pass so `DistributedSampler` actually reshuffles across ranks.
- Train metrics are a rolling window flushed at each validation, not per-epoch sums.
- `--epochs` and `--save-every` are accepted, warn, and do nothing, so existing invocations do not
  hard-fail. `resume_training.sh` had `--max-steps 70000 --epochs 300`, where the 300 existed purely
  to keep the epoch loop alive long enough — exactly the hack this removes. Updated to
  `--max-steps 70000 --eval-steps 500` (that config ran ~373 steps/pass, so the cadence is close to
  what it was).
- Old checkpoints resume: `trainer_state` now writes `data_pass` but still reads a legacy `epoch`
  key, verified against `run_033`'s state (`epoch: 9, global_step: 120068`).

### A pre-existing thing this surfaced

Two validations on identical weights disagreed (11.3260 then 11.4101). Not a bug in the new loop —
**`HDF5ChessDataset.__getitem__` samples a fresh ply per call, so the HDF5 val set re-rolls on every
pass.** HDF5 val loss carries resampling noise by construction and is not comparable across
evaluations; the shard val set is frozen and is. Worth knowing before reading either curve, and one
more small argument for the shard path. The redundant final validation that exposed it is now
deduped.

### Also closed: resume vs. `--max-steps`

A resumed run reuses the checkpoint's `scheduler_config`, so the LR keeps its original horizon while
`--max-steps` bounds the loop. Correct, but the two can silently disagree, so the trainer now says so
out loud when they do.

## Cost ledger

| Date | What | Provider | Hours | Cost |
|---|---|---|---:|---:|
| 2026-08-05 | Phase 0 baseline | local | — | €0 |
| | | | **total** | **€0 / €200** |
