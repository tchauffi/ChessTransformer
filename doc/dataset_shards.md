# Phase 1 — Input pipeline: design spec

Companion to [`distributed_training.md`](distributed_training.md). Measurements live in
[`../docs/distributed-log.md`](../docs/distributed-log.md).

**Goal:** stop the input pipeline being the wall, so a multi-GPU node is feedable at all.
Baseline 4,422 samples/s end-to-end, data-bound; §2 took the pipeline compute-bound at 4,903.
**§3-§5 (shards) are now built and verified** — loader ceiling 138,911 samples/s at 8 workers,
0.10 MB per batch, all four correctness checks green. See §6 for the numbers and §7 for what
shipped. One item remains open: the 2,000-step loss-equivalence gate in §5.

---

## 1. Where the 1.19 ms per sample actually goes

Measured on 300 randomly-drawn samples, single core (`data/elite_db.h5`):

| Component | ms | share | verdict |
|---|---:|---:|---|
| `h5py.File` open + read `moves[g]` (gzip) | 0.352 | 30 % | 0.220 ms with a persistent handle |
| **2nd `h5py.File` open for elo/result** (`:219-222`) | **0.300** | **25 %** | pure waste — see below |
| legal moves → 3 tensors (`:160-171`) | 0.169 | 14 % | 0.078 ms if only `legal_moves_planes` is built |
| python-chess replay to the sampled ply | 0.143 | 12 % | irreducible without preprocessing |
| position tokenize | 0.011 | 1 % | free |
| tensor/dict construction, remainder | ~0.21 | 18 % | shrinks with the field count |
| **total** | **1.186** | | **843 samples/s/core** |

Two things fall out that I did not expect:

**The metadata read is 25 % of the cost.** `__getitem__:219-222` opens the HDF5 file a second time
to fetch `white_elo`, `black_elo`, `result`. The trainer reads **none** of the elos (confirmed: the
only keys consumed at `:463-472` are `position`, `is_white`, `castling_rights`, `en_passant_file`,
`from_square`, `action_plane`, `legal_moves_planes`, `result`, `move_number`). And `result` is
`int8 × 1,533,352` = **1.5 MB** — it can simply live in RAM next to `num_moves_per_game`, which
`__init__` already loads.

**Random access to a gzip HDF5 is 13× more expensive than sequential.**

| access pattern | ms per `moves[g]` |
|---|---:|
| random, reopen file each time (current) | 0.352 |
| random, persistent handle | 0.220 |
| **sequential, persistent handle** | **0.017** |

`moves` is chunked at 10,000 games with gzip, so a random read decompresses a whole chunk to
extract one game. This is a hard constraint on the build script: **iterate games in order.** A
sequential pass over 1.53M games costs ~26 s of h5 reads; a shuffled pass costs ~5.6 min.

---

## 2. Quick wins — ✅ DONE 2026-08-05

~30 lines in `h5_lichess_dataset.py`. They compose with the shard work (the build script reuses the
same encoding path) and they are worth **1.9×** on their own. **These numbers are measured, not
projected** — see the caveat at the end of this section.

- [x] **Hold one `h5py.File` handle per worker** instead of reopening in `_get_game_moves`, with
      `rdcc_nbytes=32<<20`. The chunk cache is the bigger half of this: 0.352 ms → 0.221 ms from the
      persistent handle, → **0.121 ms** with a 32 MB cache (default is 1 MB, too small to hold a
      10,000-game gzip chunk).
      **Open it lazily on first `__getitem__`, never in `__init__`.** A handle created before the
      fork is inherited by all workers; HDF5 is not fork-safe and the documented failure modes are
      silent data corruption and deadlock. I tested the eager version and it ran 60 batches without
      complaint — which is exactly what makes it dangerous. The correct version costs one
      `if self._f is None`.
- [x] **Delete the second `h5py.File` open** (`:219-222`). `__init__` already reads the full
      `white_elo`/`black_elo`/`num_moves` columns at `:52-54` and then throws two of them away —
      just keep `result` (1.5 MB of `int8`). Drop the Elos from the returned dict; nothing reads them.
- [x] **Return only the 9 keys the v2 trainer consumes.** Every extra key is collated, pinned and
      copied. Full audit of the 19 returned keys against every consumer in the repo:

      | | v2 trainer | v1 trainer | bytes/sample |
      |---|:---:|:---:|---:|
      | `position` `is_white` `castling_rights` `en_passant_file` | X | X | 536 |
      | `from_square` `action_plane` `result` `move_number` | X | · | 32 |
      | `legal_moves_planes` | X | · | 4672 |
      | `move` `halfmove_clock` | · | X | 16 |
      | `legal_moves_mask` | · | X | 1968 |
      | `legal_moves_grid` | · | · | 4096 |
      | `game_id` `to_square` `is_promotion` `promotion_type` | · | · | 32 |
      | `white_elo` `black_elo` (dataset `__main__` only) | · | · | 16 |

      `legal_moves_mask` **is** read by `position2move_trainer.py`, so dropping it retires the v1
      trainer rather than being free. `legal_moves_grid` is genuinely dead everywhere.
- [x] **Stop *building* the dead tensors, not just returning them** (`:211-218`). Removing a key from
      the returned dict saves the collate and the H2D copy but not the CPU — the `torch.zeros` calls
      and the per-move vocab lookup still run for every sample.
- [x] **`pin_memory=True`** on all three DataLoaders (`:289-293`). Currently absent; without it every
      H2D copy stages through pageable memory and cannot overlap with compute.
- [x] Re-run `uv run scripts/profile_training.py --batch-size 512 --compile` and record the number.

Measured, all of the above applied together:

| | 1 worker | 4 | 8 | 12 | MB/batch |
|---|---:|---:|---:|---:|---:|
| current | 891 | 3,238 | 5,608 | 6,139 | 5.81 |
| **quick wins** | **1,670** | **6,130** | **10,520** | **11,323** | **2.68** |

Single core: 1.150 → 0.557 ms/sample. Equivalence checked over 2,000 samples × all 9 keys with the
ply forced identical — byte-identical, including `legal_moves_planes`.

**4 workers now beat the compiled compute ceiling of 5,643 samples/s in isolation.** Apply the
contention discount from Phase 0 before believing that; 8 workers is where the real headroom is.

> **Caveat, and it is the same mistake twice.** I first estimated these savings by timing each
> component in isolation and adding them up: 0.30 ms for the metadata open + 0.13 ms for the handle
> ⇒ ~0.67 ms/sample. The persistent handle alone actually delivered 1.140 → 1.005 ms, about a third
> of that. The isolated metadata benchmark opened the file 300 times with nothing else live; inside
> `__getitem__`, `_get_game_moves` has *just* opened and closed the same file, so the OS and HDF5
> leave it warm and the second open is far cheaper in context than on its own. Sum-of-parts
> overestimates for exactly the same reason the isolated dataloader benchmark did in Phase 0.
> Always re-measure the whole.

- [x] Delete `RepeatingDataloader` (`:203-217`, applied at `:400`). It yields the *same batch* twice,
      so every step trains on a duplicated gradient and `len()` lies to the LR schedule. It was a
      band-aid for GPU starvation; once the loader is fast it is a correctness bug with no upside.

---

## 3. Shard format

### What one sample must contain

Only the nine keys the trainer reads. Everything else is dropped.

| field | dtype | bytes | note |
|---|---|---:|---|
| `position` | uint8 × 64 | 64 | vocab is 13; int64 in the current path is 8× oversized |
| `legal_idx` | uint16 × C | 2C | flat `from_square * 73 + plane` indices |
| `legal_cnt` | uint8 | 1 | |
| `from_square` | uint8 | 1 | |
| `action_plane` | uint8 | 1 | 0-72 |
| `castling_rights` | uint8 | 1 | 0-15 |
| `en_passant_file` | uint8 | 1 | 0-8 |
| `is_white` | uint8 | 1 | |
| `result` | int8 | 1 | |
| `move_number` | uint16 | 2 | max observed 601 |

### The legal-mask decision (this is the one real design choice)

`legal_moves_planes` is `(64, 73)` bool = 4,672 bytes raw, and it dominates everything else. Three
options, measured over 400 real positions (**mean 33.9 set entries, p95 = 48, p99.9 = 53, max = 53**):

| encoding | bytes/sample | notes |
|---|---:|---|
| raw bool | 4,672 | what happens today |
| `np.packbits` | 584 | still 4× the rest of the sample combined |
| **fixed-width uint16 indices, cap C=64** | **129** | one memmap row per sample, O(1), trivially shardable |
| CSR (flat indices + uint32 offsets) | ~72 | no cap, 28 % smaller, needs a custom collate |

Recommendation: **fixed-width, C = 64.** The cap sits comfortably above the observed max of 53, and
fixed width is what makes a single `np.memmap` row per sample work — which in turn is what makes
rank/worker sharding a slicing exercise rather than an indexing exercise.

**The cap trick:** write the *target* move's `(from_square, plane)` index **first**, then the rest.
Clipping at C can then never drop the label, only a few alternative legal moves. Count overflows at
build time and print the rate; if it is above ~0.01 % raise C to 96 and rebuild.

CSR is the better format in the abstract (it is what Megatron's `IndexedDataset` does) and is worth
building if you want the exercise, but the extra collate machinery buys you 1.4 GB on a dataset that
already fits in page cache.

- [x] **Decided: fixed-width, C = 64.** CSR is the better format in the abstract and ~28 % smaller,
      but it needs a ragged collate on both sides and the 1.4 GB it saves is on a dataset that
      already fits in page cache. Fixed width is what lets one memmap row be one sample, which is
      what makes the batch-level gather in §4 a two-line `np.searchsorted` + fancy index.
      **Measured on the real 24.5 M-sample build: 807 samples hit the cap, 0.0033 %** — under the
      0.01 % rebuild threshold, and every one of them kept its label because the target index is
      written first (`verify_shards.py` check 1 asserts exactly that on capped rows).

### Sizing

C = 64 → **201 bytes/sample**.

| K (positions kept per game) | samples | on disk | fits in page cache? |
|---:|---:|---:|---|
| 8 | 12.3 M | 2.47 GB | yes, easily |
| **16** | **24.5 M** | **4.93 GB** | yes (~20 GB free) |
| 32 | 49.1 M | 9.86 GB | tight alongside 12 workers |

Recommendation: **K = 16**, ~17 % of the 148M available plies.

- [x] Sample those K plies **with the existing triangular middlegame weights**
      (`_sample_move_idx:78-100`), without replacement, at build time. The shard then *is* the
      sampling distribution and the loader shuffles uniformly — no weighting logic in the hot path.
      `build_shards.py:sample_plies` mirrors the online sampler's bounds and weights exactly.

      Without-replacement is not free and the cost was measured rather than assumed. Both modes
      are implemented (`--sample-mode`):

      | mode | TVD vs online marginal | duplicate rows | mean ply (shard / online) |
      |---|---:|---:|---|
      | **`unique`** (default) | **0.024** | **0 %** | 37.59 / 37.17 |
      | `iid` | 0.014 | 13.7 % | 33.55 / 33.66 |

      `unique` flattens the middlegame peak slightly into both tails — the marginal drift is real
      but it moves the mean ply by 0.4 plies. `iid` reproduces the marginal to within the 0.005
      noise floor of a 30k-sample histogram, but spends 13.7 % of the page-cache budget on exact
      repeats. Distinct positions per byte won.
- [x] For games with `num_moves - 1 < K`, keep all plies. Record the true sample count per shard;
      do not assume `K × num_games`. Real build: 24,498,538 samples, not `16 × 1,533,352 =
      24,533,632` — 35k short, all from games with fewer than K usable plies.

**The honest cost of this:** the HDF5 path redraws a fresh position per game on every pass, so positions are
effectively unlimited. A shard freezes 16 per game forever. At K=16 a run consuming 40M samples
sees each position ~1.6×. If that shows up as overfitting (watch the train/val gap against the
HDF5 baseline), rebuild with a different `--seed` mid-run rather than raising K — cheaper in RAM.

### File layout

```
data/shards/
  meta.json          # K, C, seed, dtypes, per-shard sample counts, game_id ranges, git sha
  shard_000.npy      # (n_i, 201) uint8, one row per sample
  ...
  shard_255.npy
```

- [x] 256 shards. It divides every `world_size × num_workers` you will plausibly use
      (1,2,4,8,16,32,64). Note the Phase 2 `k % (world_size * num_workers)` rule was **not** taken —
      see §4 for why map-style + `DistributedSampler` beat it.
- [x] Assign **games**, not samples, to shards — contiguous game ranges, so the build script's
      sequential h5 read order is preserved (see §1).

### The split gotcha

`pos2move_v2_trainer.py:287` does `random_split` over a dataset whose index **is a game**. That is
correct today: no game appears in both train and val.

Shard samples are positions, so a naive `random_split` over samples puts positions from the *same
game* on both sides of the split — the model sees the same opening it trained on and val loss goes
optimistically wrong. Val loss drives best-model selection here.

- [x] Reserve whole shards for val/test. **Not shards 0-9, though:** games sit in the HDF5 in
      roughly chronological ingestion order, so the first ten shards are an unrepresentative slice.
      `choose_split_shards` spreads the reserved ids evenly across `[0, num_shards)` instead.
      This also bit the *verifier* before it bit the build — check 2 compared the shard's ply
      histogram against games drawn from the whole file and reported a spurious TVD of 0.037 that
      was pure game-length drift across the corpus, not sampler error. It now draws its reference
      games from the same ranges the split covers.
- [x] Assert at load time that the `game_id` sets are disjoint (`verify_shards.py` check 3).
      Structurally guaranteed by contiguous ranges, but asserted anyway.

---

## 4. `FlatShardDataset`

`src/chesstransformer/datasets/flat_shard_dataset.py`

- [x] `np.memmap` each shard read-only in `__init__`; build a cumulative-offset array so a global
      index resolves to `(shard, row)` with one `np.searchsorted`. Never `np.load` a shard into RAM —
      let the page cache do it, so the 12 workers share one copy instead of 12. Mapped lazily per
      process behind a PID check, same reasoning as the HDF5 handle in §2.
- [x] **Batch-level fetch.** At 202 bytes/sample the per-item Python call overhead now dominates the
      actual work. `make_dataloader` wires `sampler=BatchSampler(...)` with `batch_size=None` so
      `__getitem__` receives a *list* of indices and does one vectorised numpy gather per batch.
      Within each shard the rows are gathered in ascending order and scattered back into batch
      order afterwards, so page touches stay sequential even though the batch is shuffled.
- [x] Return `legal_idx`/`legal_cnt` as-is. **Do not build the `(64, 73)` mask on CPU** — that would
      reintroduce the 2.4 MB/batch transfer you just removed. Every field also stays at its on-disk
      width on the wire (uint8 / int16) and is widened to int64/bool on device in `unpack_batch`.
      Returning int32 instead cost 0.28 MB/batch against 0.10 MB — 2.7× the H2D traffic for a
      cast that is free on the GPU.
- [x] Rebuild the mask on GPU in the training loop (`scatter_legal_planes`), one line:
      ```
      # idx: (B, C) int64 with pad slot 64*73, so padding lands in a scratch column
      mask = torch.zeros(B, 64 * 73 + 1, dtype=torch.bool, device=dev)
      mask.scatter_(1, idx, True)
      legal_planes = mask[:, :64 * 73].view(B, 64, 73)
      ```
      Pad entries must be written to index `64*73` at build time (or clamped in the loop), otherwise
      they set square a1's plane 0 legal on every sample.
- [x] Keep `HDF5ChessDataset` untouched as the reference implementation for the equivalence test.
      It is still the default path; `--shards` selects the new one.

- [x] **Rank/worker assignment: rejected, deliberately.** Phase 1's original bullet called for an
      `IterableDataset` owning shard *k* where `k % (world_size * num_workers) == rank * num_workers
      + w`. That rule shuffles only *within* a worker's own shards, so game-level correlation
      survives into the batch, and it silently unbalances whenever `world_size * num_workers` does
      not divide the shard count. A map-style dataset + `DistributedSampler` gives global shuffling
      and exact rank balance for free, and the memmap makes locality a non-issue — the shards stay
      a build-parallelism and split unit, not a reader constraint. The `k % (...)` pattern is the
      right answer for *streaming* data that does not fit on local disk; that is the tradeoff to
      describe, not the rule itself.

- [x] **Kept out of `accelerator.prepare()`.** `batch_size=None` reads to Accelerate as "this
      iterable already yields batches", so it would re-shard at the wrong granularity. Rank
      splitting is the `DistributedSampler` inside `make_dataloader`; the H2D copy is `unpack_batch`.

Batch bytes at B=512: **5.81 MB → 0.10 MB.**

---

## 5. Verification

The whole point is that this must be provably the same data distribution, not just faster.

- [x] **Encoding equivalence.** `verify_shards.py` check 1. **5,747 samples across 3 shards,
      every field byte-identical.** The shard stores no `game_id`, so the pairing is recovered
      structurally — rows are written game-by-game in ascending game and ply order, and the ply set
      is reproducible from `(seed, shard_id)`. Re-deriving the plies and pushing them through the
      *untouched* `HDF5ChessDataset.__getitem__` (with `_sample_move_idx` forced to the chosen ply)
      is what makes this two independent encoders rather than the builder checking itself.
      Capped rows are checked differently: the stored set must be a subset of the true legal set
      **and** must still contain the label.
- [x] **Distribution equivalence.** Check 2, 100k vs 100k. TVD **0.024**, mean ply 37.59 vs 37.17.
      The residual is the without-replacement flattening quantified in §3, not a lost weighting —
      a build that reverted to uniform would show mean ply ~48 and TVD an order of magnitude worse.
- [x] **Loss equivalence — GREEN.** 2,000 steps, seed 42, `--batch-size 512 --grad-accum 4
      --warmup-steps 100`, A = HDF5 `--min-elo 0 --num-workers 8`, B = `elite_k16 --num-workers 0`.
      `--min-elo 0` because the shards are unfiltered; see the ELO note below.
      Runs kept at `logs/pos2move_v2/run_010_20260816_160119` (A) and `run_011_...161339` (B).

      | steps | A ce | B ce | Δ | A legal_ce | B legal_ce | Δ | A legal_acc | B legal_acc |
      |---|---:|---:|---:|---:|---:|---:|---:|---:|
      | 0-100 | 6.1728 | 6.1880 | +0.0151 | 3.5631 | 3.5224 | −0.0407 | 0.0747 | 0.0707 |
      | 100-300 | 3.3853 | 3.3879 | +0.0026 | 2.9772 | 2.9822 | +0.0050 | 0.2027 | 0.1997 |
      | 300-600 | 2.9183 | 2.9111 | −0.0072 | 2.6667 | 2.6573 | −0.0093 | 0.2677 | 0.2668 |
      | 600-1000 | 2.6695 | 2.6722 | +0.0026 | 2.4827 | 2.4779 | −0.0048 | 0.3056 | 0.3053 |
      | 1000-1400 | 2.4938 | 2.4917 | −0.0021 | 2.3495 | 2.3436 | −0.0060 | 0.3348 | 0.3348 |
      | 1400-1700 | 2.3776 | 2.3783 | +0.0006 | 2.2576 | 2.2543 | −0.0034 | 0.3537 | 0.3542 |
      | 1700-2000 | 2.3010 | 2.2933 | −0.0077 | 2.1951 | 2.1855 | −0.0096 | 0.3689 | 0.3696 |

      **The yardstick matters more than the deltas.** Batch-to-batch std of `train/step_ce` over
      steps 1700-2000 is **0.0725** (A) and **0.0643** (B). Every delta from step 100 on is ≤ 0.0077
      — about **10 % of one batch's noise**. The only window above that is 0-100, which is warmup,
      where the two paths' different first batches dominate. Train `legal_acc` lands at 0.3689 vs
      0.3696. Within noise by any reading.

      **What this comparison does *not* license.** Val and test are *different evaluation sets*
      between the two runs — A's val is a `random_split` over games with online ply sampling, B's is
      10k rows drawn from reserved shards — so `val/loss` 6.1212 vs 6.4243 and `test/loss` 6.1735 vs
      5.9993 are not an A/B of anything. (Most of the val gap is the value head: `val/value_loss`
      0.5374 vs 0.5870, and `--value-loss-weight 5.0` turns that 0.0496 into 0.248 of the 0.303.)
      Only the training curves are a valid equivalence check here.
- [x] **Split disjointness.** Check 3. train 1,473,452 games / val 29,950 / test 29,950, no overlap.
- [x] **Worker RNG.** `flat_shard_dataset.worker_init_fn` seeds numpy from
      `torch.initial_seed() + 977 * rank`, and is applied to **both** paths — offline sampling
      removes `np.random` from the shard hot path, but the bug was still live in `HDF5ChessDataset`.

- [x] **Padding safety.** Check 4, which did not exist in the original plan and should have. Pad
      slots are written as `64*73 = 4672`, one past the action space, and scattered into a scratch
      column that is then sliced off. Writing pad as `0` instead would mark a1/plane-0 legal on
      every sample — a bug that degrades the legal-move mask subtly enough to survive a loss curve.
      The check asserts `set bits == legal_cnt` per sample.
- [x] Loader throughput measured by `verify_shards.py` check 5; see §6. Re-running
      `scripts/profile_training.py` end-to-end is pending the same quiet CPU as the loss gate.

## 6. Targets

Measured with the fixed profiler (`--batch-size 512 --compile --workers 8`); see the harness-bug
note in `docs/distributed-log.md` before comparing against anything recorded earlier.

| Metric | baseline | after quick wins | shard target | **shards, measured** |
|---|---:|---:|---:|---:|
| End-to-end samples/s | 4,422 | 4,903 | > 5,600 | **5,876** (HDF5 rerun: 5,724) |
| Loader ceiling, 8 workers | 3,801 | 7,572 | > 40,000 | **138,911** |
| Loader ceiling, 0 workers | — | — | — | **144,272** |
| Verdict | data-bound (1.48×) | compute-bound (0.75×) | compute-bound | **compute-bound (0.04×)** |
| Per-sample CPU cost | 1.123 ms | 0.568 ms | < 0.02 ms | **0.0069 ms** |
| Bytes per collated batch | 5.81 MB | 2.69 MB | 0.10 MB | **0.10 MB** |

Every target hit, the loader one by 3.5×.

**But end-to-end the shards buy +2.7 %, and that is the expected result, not a disappointment.**
Measured on the same 2,000-step runs as the §5 gate: 5,724 samples/s (HDF5, 8 workers) vs 5,876
(shards, 0 workers), 357.8 vs 348.6 ms per optimizer step. §6 said this in advance — Phase 1a had
already taken the pipeline compute-bound at 86 % of the compiled ceiling, so there was no
single-GPU throughput left for the shards to win. They were justified by the 8-GPU arithmetic and
they should be judged on it.

The real single-GPU win is in the resource column, not the throughput column: **B matches A's
throughput while using 0 dataloader workers instead of 8.** Eight cores came back. That is what
turns "can this feed 8 ranks" from a no into a yes.

Two more things worth pulling out:

**`num_workers=0` is now the fastest setting** (144,272 vs 138,911 at 8 workers, and a *dip* to
68,796 at 1 worker). The gather is a memmap read and a fancy index; at that cost the IPC round-trip
to a worker process is more expensive than the work it does. This is a better result than it looks
for Phase 2 — the plan budgeted "≤ 4 workers per GPU", and 8 GPUs × 0 workers leaves all 16 cores
for the ranks themselves. The dip at 1 worker is the shape to expect: one worker pays full IPC with
no parallelism to amortise it, and it takes ~4 workers just to climb back to the in-process rate.

**26× the loader headroom of the compiled compute ceiling** (144k vs 5,643 samples/s). The input
pipeline has stopped being a constraint on this machine by a wide margin, which is exactly the
condition §6 said the 8-GPU arithmetic needed: 8 × 5,643 ≈ 45,000 samples/s is now 31 % of what one
process can serve, instead of needing ~57 dedicated cores.

**Phase 1a already achieved the single-GPU goal.** End-to-end sits at 86 % of the compute-only
ceiling, and the loader now runs 1.3× *faster* than compute rather than 0.67×. There is no more
single-GPU throughput to win from the input pipeline — the remaining ceiling is compute at MFU 31.6 %,
and that ceiling turned out to be structural rather than a bug (see the Phase 1b negative result in
`docs/distributed-log.md`: neither the SDPA flash path nor the RMSNorm fused kernel is recoverable,
and every alternative measured slower).

So the shards are **not** justified by single-GPU speed any more. They are justified by the 8-GPU
arithmetic: 8 × 5,643 ≈ 45,000 samples/s, versus 7,572 today at 8 workers. Scaling the current
loader to that would need roughly 57 dedicated cores at the best per-worker rate, which no
affordable 8-GPU node has. That, and only that, is the case for Phase 1b — which means it can be
deferred until Phase 3 actually needs it, and the DDP work (Phase 2) can start now.


---

## 7. What was built (2026-08-16)

| file | role |
|---|---|
| `src/chesstransformer/datasets/shard_format.py` | the record layout, defined once, shared by writer/reader/verifier |
| `scripts/build_shards.py` | one-shot HDF5 → shards, `fork` pool, one contiguous game range per shard |
| `src/chesstransformer/datasets/flat_shard_dataset.py` | `FlatShardDataset`, `make_dataloader`, `scatter_legal_planes`, `worker_init_fn` |
| `scripts/verify_shards.py` | the five checks in §5 |
| `pos2move_v2_trainer.py --shards DIR` | swaps the data path; `unpack_batch` normalises both |

### Built artifacts

| | `data/shards/elite_k16` | `data/shards/full_k4` |
|---|---|---|
| source | `elite_db.h5`, 1,533,352 games | `elite_db_full.h5`, 26,269,077 games |
| K | 16 | 4 |
| samples | 24,498,538 | **105,076,308** |
| on disk | 4.95 GB | **21.23 GB** |
| build | 5.8 min on 14 workers (70,290 samples/s) | **44.9 min on 14 workers** (38,990 samples/s) |
| legal cap hits | 807 (0.0033 %) | 3,390 (0.0032 %) |
| games dropped | 0 | 0 |
| ply-marginal TVD | 0.024 | **0.0035** |
| page cache | fits | does not fit |

Both pass all four correctness checks. `elite_k16` is the one the loss-equivalence gate runs
against, because every other number in this document was measured on `elite_db.h5`.

### K is what drives the sampling drift, and the full build proves the mechanism

`full_k4`'s ply marginal matches the online sampler to **TVD 0.0035** — an order of magnitude
tighter than `elite_k16`'s 0.024, from the identical code path. That is the without-replacement
flattening behaving exactly as predicted in §3: the drift scales with the sampling fraction
`K / plies_available`. At K=16 over ~95 plies that is 17 % and the flattening is measurable; at
K=4 over ~84 plies it is 4.8 % and it disappears into the noise floor.

So the `unique` vs `iid` trade-off in §3 is **only** a real decision at high K. At K ≤ 4 there is
nothing to trade — take `unique` and the 13.7 % duplicate rows never come up.

### The optimal `num_workers` inverts with page-cache residency

| workers | `elite_k16` (4.95 GB, cached) | `full_k4` (21 GB, NVMe) |
|---:|---:|---:|
| 0 | **144,272** | 22,424 |
| 1 | 68,796 | 165,791 |
| 4 | 115,585 | 379,329 |
| 8 | 138,911 | **439,824** |

Same code, same record, opposite recommendation — and the mechanism is legible in the numbers.

When the shards fit page cache, a fetch is a memory read and the only variable cost is IPC, so
`num_workers=0` wins and one worker is *worse than none* (it pays full IPC with nothing to
amortise it).

When they do not, every fetch is a 4 KB page fault to NVMe and the variable cost is **queue depth**,
not CPU. At `workers=0`, 22,424 samples/s × 4 KB ≈ 92 MB/s ≈ 22k IOPS — textbook single-threaded
random read at queue depth 1. Eight workers issue eight concurrent faults and reach 439,824
samples/s ≈ 1.8 GB/s, which is what the drive does at depth 8. The workers are not there to do CPU
work any more; they exist purely to keep the NVMe queue full.

Practical rule: **size `num_workers` by whether the shard set fits RAM, not by core count.** For
Phase 2 that means `elite_k16` runs 0 workers per rank and `full_k4` wants ~8 — and on a rented
8-GPU box with more RAM than this machine, `full_k4` may well flip back to the cached regime.

### The record, as built

202 bytes, not the 201 estimated in §3 — one pad byte at offset 199 keeps `move_number` on an even
offset. `legal_idx` sits at offset 64, also even. Both matter: an unaligned memmap view of a `uint16`
field drops numpy out of its vectorised path and into a per-element copy on every batch gather.

### Things that were nearly bugs

* **`is_white` must stay `bool`.** `compute_loss` uses it as a *mask* (`target_value[white_win &
  is_white]`). The shard path returns it as `uint8`; handing that through unchanged turns the mask
  into fancy indexing and scatters value targets onto the wrong rows — silently, with a plausible
  loss curve.
* **`result` is `0/1/2`, not `-1/0/1`** (draw / white / black), matching the HDF5 column. The shard
  stores the raw value; only the docstring was ever wrong.
* **The reference sampler must be restricted to the split's own game ranges** when comparing
  histograms, or corpus-level game-length drift shows up as sampler error. See §3.


### ELO filtering (added after the first two builds)

`HDF5ChessDataset` filters games by average ELO in `__init__`, and the trainer's HDF5 path defaults
to `--min-elo 2400` — which drops **17 % of `elite_db.h5` and 33 % of `elite_db_full.h5`**. The
first two shard builds had no such filter, so they hold the *unfiltered* population.

`build_shards.py` now takes `--min-elo` / `--max-elo`, applied to games exactly as
`HDF5ChessDataset.__init__` does and recorded in `meta.json`. The filter has to live at build time
for the same reason the ply sampling does: the shard *is* the population.

Both shipped shard sets remain unfiltered — a superset, not wrong data, but not the same
distribution the HDF5 default trains on. The §5 gate therefore ran with `--min-elo 0` on the HDF5
side so both paths saw the same games. **Rebuild with `--min-elo 2400` before treating these shards
as a drop-in replacement for the default HDF5 config.**

One thing the filter smoke test re-confirmed: 2,750 of the *first* 4,000 games are sub-2400, against
17 % across the corpus. Games are in chronological order and the ELO floor drifts upward through the
file — the same corpus drift that made the verifier's first distribution check wrong. Any statistic
taken from a prefix of this HDF5 is unrepresentative.
