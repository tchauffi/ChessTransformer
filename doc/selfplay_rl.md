# Self-play RL: beyond the human-data ceiling

ChessTransformer v2.1 reaches ~2100 Elo trained **only on human games**. This document
explains why supervised learning on human data is hitting a ceiling, what we tried,
and the self-play research plan we are now executing — all under a single
RTX 5070 Ti (16 GB) compute budget.

## 1. Why we are data-limited

Two experiments established the ceiling:

- **Stockfish policy distillation (failed).** We distilled Stockfish best moves into
  the policy head with cross-entropy, scaling up to 200k labeled positions. The
  policy did not improve: the 11.7M model's **policy head is at capacity** for its
  size. More/better move labels don't help.
- **GRPO on Lichess puzzles** (`scripts/grpo_puzzles.py`). RL with a verifiable
  reward (puzzle solution match) on the policy head, with a KL anchor to a frozen
  reference. Same conclusion: policy-only updates on the small model have little
  headroom.

So the policy axis is saturated. But the model has a second output: the **value
head**, and the scaling curve says it matters enormously — the same network gains
**~+850 Elo** going from 25 to 800 MCTS simulations, and every one of those
simulations is scored by the value head at the leaves.

The value head's training labels are the weak link: it was trained on **human game
outcomes**. Humans blunder won positions, flag in winning endgames, and agree to
draws in decisive positions — so the label "this position led to a win" is noisy.
Self-play games *with search* produce far cleaner outcome labels for the same
positions.

## 2. The research plan

Classic AlphaZero from scratch needed a datacenter. We skip the expensive part —
the cold start — because we already have a 2100-Elo network to warm-start from.
The plan is staged so each step is cheap and falsifiable before the next:

| Step | Experiment | Cost | Question it answers |
|------|-----------|------|---------------------|
| 1 | **Value-head-only retraining on self-play outcomes** (current) | hours | How much Elo is hiding in better leaf evaluations? |
| 2 | Full expert iteration (policy targets = MCTS visit distributions) | days | Does search-amplified policy data beat the human-data policy ceiling? |
| 3 | Repeat on the 46M model after its human pretraining finishes | days | Does extra capacity unlock the policy axis? |

Step 1 is deliberately conservative: the policy stays **frozen**, so there is no
risk of policy collapse, no KL tuning, no gating loop — and it isolates exactly one
variable (value label quality).

## 3. Step 1 pipeline

Two scripts, reusing the existing MCTS bot and evaluation tooling.

### 3.1 Game generation — `scripts/selfplay_value_games.py`

Plays self-play games with the frozen v2.1 policy via `Pos2MoveV2MctsBot` and
records every position with the eventual game outcome:

```bash
uv run python scripts/selfplay_value_games.py \
    --model data/models/pos2move_v2.1 --out data/selfplay/v2.1-128sims \
    --games 2000 --sims 128
```

Design choices (mostly borrowed from the KataGo / AlphaZero playbooks):

- **Moderate search budget (128 sims/move).** Label quality scales with playing
  strength, but games/hour scales inversely with sims. For value training, more
  games at moderate strength beats few games at full strength.
- **Diversity** from the `engine_match.py` opening suite + sampling moves
  proportional to visit counts (`--move-temp 1.0`) for the first 30 plies, then
  argmax.
- **Resignation** when the white-POV root value stays beyond ±0.93 for 6
  consecutive plies. Game tails are a large fraction of plies and teach almost
  nothing. 10% of games (`--no-resign-frac`) play out fully so false-resignation
  rates stay measurable.
- **Hard cap** at 300 plies, adjudicated as a draw.
- **Throughput**: ~950–1050 games/h at 128 sims on the 5070 Ti (tree reuse + the
  shared transposition cache do a lot of work here). ~100 positions/game.

Output: `positions_*.npz` shards (encoded board tokens, side-to-move-POV outcome
`z ∈ {-1, 0, +1}`, root MCTS value, halfmove number, game id) plus a `games.jsonl`
audit log. Re-running with the same `--out` appends, so the dataset grows
incrementally.

### 3.2 Value-head training — `scripts/train_value_head.py`

```bash
uv run python scripts/train_value_head.py \
    --data data/selfplay/v2.1-128sims --base data/models/pos2move_v2.1 \
    --out data/models/pos2move_v2.1-spvalue
```

The key efficiency trick: the trunk is frozen, so the 256-dim state features are
computed **once** for the entire dataset (a single batched bf16 pass), and only the
~20k-parameter value-head MLP trains on the cached features. Full-dataset epochs
take seconds, so proper early stopping is free.

- **Loss mirrors pretraining exactly**: MSE against the stm-POV outcome, with the
  same halfmove ramp (early plies down-weighted, `--ramp-halfmoves 40`).
- **Optional soft targets** (`--soft-mix λ`): target = (1−λ)·z + λ·root_MCTS_value,
  KataGo-style, to reduce outcome-label variance from long games.
- **Validation splits by game**, not by position — consecutive positions of one
  game are near-duplicates and would leak.
- The script reports the **pretrained head's val MSE as the baseline** before
  training, plus sign-accuracy on decisive positions.
- Export is a standard model dir (base weights with only `value_head.*` swapped),
  loadable by every existing bot/script.

### 3.3 Evaluation

Strength is the only metric that counts. Deterministic A/B with the existing
opening-suite match (much lower variance than a Stockfish gauntlet):

```bash
uv run python scripts/engine_match.py --a-mcts --b-mcts --a-sims 400 --b-sims 400 \
    --a-model-dir data/models/pos2move_v2.1-spvalue \
    --b-model-dir data/models/pos2move_v2.1
```

If the new head wins clearly, confirm with the Stockfish gauntlet
(`scripts/tune_vs_stockfish.py`) for an absolute Elo estimate.

## 4. Step 2 preview: expert iteration

If step 1 pays off, the loop closes: alternate **generate** (self-play with the
*current* net, recording MCTS visit distributions as policy targets) and **train**
(policy CE on visit distributions + value MSE on outcomes, KL-anchored to the
previous net), gating each new net via `engine_match.py` before promoting it.

Planned efficiency upgrades for that stage, in order of expected impact:

1. **Gumbel AlphaZero search** for training games (Danihelka et al., 2022) — a
   valid policy-improvement target with 16–32 sims/move instead of hundreds.
2. **Playout cap randomization** (KataGo) — cheap budget for most moves, full
   budget for a random ~25%, and only those become policy targets.
3. **Cross-game leaf batching** — the current MCTS batches leaves within one tree
   (`sim_batch=16`); batching across N concurrent games is what would actually
   saturate the GPU.

Cheap non-RL supplements worth folding in at any stage: Syzygy tablebase positions
as perfect endgame value labels, and auxiliary prediction targets (final material,
opponent reply) to extract more signal per game.

## 4b. Rust generation core (step 2 infrastructure)

Profiling showed the Python generator is CPU-bound (~38% python-chess, ~30%
Python MCTS loop). `rust/selfplay-core` (PyO3 + shakmaty) moves boards, trees
and game lifecycle into Rust and batches leaf evaluations **across all
concurrent games**; Python keeps only the batched bf16/compiled NN forward
(`scripts/selfplay_rust.py`). The transposition cache stores legal-move-aligned
priors instead of full 64×73 logits (~200 B vs ~19 KB per entry).

Encoding parity with the Python tokenizers is enforced by
`scripts/check_rust_parity.py` (board tokens, castling/ep/player, action
planes, legal move sets — 38k+ random positions).

**Benchmark @ 128 sims:** 11,025 games/h single process (64 parallel games) vs
840 (Python single) and ~1,900 (3 Python workers) — **13× / 5.8×**. The new
generator also records MCTS visit distributions (CSR: visit_idx/visit_cnt/
visit_ptr) — the policy targets for expert iteration. Output stays loadable by
the existing training scripts.

## 5. Status log

- **2026-06-10** — Pipeline built and smoke-tested end to end. First real dataset
  generating: 2,000 games @ 128 sims → `data/selfplay/v2.1-128sims` (~200k
  positions, ~2 h). Next: train head, A/B vs base.
- **2026-06-10 (later)** — Generation exposed and fixed a latent MCTS bug
  (transposition-table eviction race in `_run_batch` once the TT fills — never
  reached in shorter-lived processes). Single-process generation is CPU-bound
  (GPU ~16%): profiling shows ~38% python-chess, ~30% Python MCTS loop, ~30%
  model+sync. Running 3 parallel workers (`--tt-size 100000` each — the 500k
  default costs ~9.5 GB RAM/worker) raised throughput from ~840 to ~1,900
  games/h. Final dataset: **4,865 games / 518k positions**.
- **2026-06-10 (result)** — **Step 1 is a null result, and an informative one.**
  Head-only retraining on 493k self-play positions: val MSE 0.2406 → 0.2369
  (−1.5%), decisive sign-acc 84.8% → 85.0%, best epoch = 1 then overfitting.
  Given the frozen trunk's features, the pretrained head was already
  near-optimal — the binding constraint is the **trunk representation**, not the
  value labels. This mirrors the policy-distillation finding: the 11.7M model is
  at capacity on *both* heads. Consequences: (a) value gains require unfreezing
  the trunk (value MSE + KL-anchored policy preservation — half-way to expert
  iteration), or (b) more capacity (the 46M model). The 518k-position dataset is
  directly reusable for both. A/B match vs base @ 400 sims: **+9 =27 −12
  (46.9%)** over 48 games — statistically even (σ ≈ 7%), confirming the null on
  the board as well.
- **2026-06-17** — Step 2 started. Fixed a dead import that left
  `scripts/selfplay_rust.py` (the visit-distribution generator) unrunnable on
  `main` — it imported `load_model` from the deleted `train_value_head.py`; now
  pulls it from `grpo_puzzles.py`. Built the expert-iteration trainer
  `scripts/train_expert_iter.py`: unfreezes the whole net, policy CE on MCTS
  visit distributions + ramped value MSE on outcomes + KL anchor to the frozen
  base policy, game-split validation, EMA, standard model-dir export. Smoke test
  on the 100k-position `v2.1-400sims-exit` shard (1 epoch) **moves both heads
  past the frozen base on held-out games**: val policy CE 1.3136 → 1.2718, value
  MSE 0.1787 → 0.1706 — the very signal the frozen-trunk Step 1 could not
  produce. Strength unverified pending the `engine_match.py` gate. Next: (a)
  generate a larger visit-distribution dataset with the fixed Rust generator
  (the 518k `v2.1-128sims` set predates visit recording and lacks policy
  targets), then (b) full train + gate vs base.
- **2026-06-17 (first gate)** — Trained `pos2move_v2.1-exit1` on the 100k
  `v2.1-400sims-exit` shard (12 epochs, early-stopped at 7, best epoch 4;
  kl-coef 0.1, value-weight 1.0). Held-out policy CE improved 1.3075 → 1.2371
  but the **value head overfit** the small set (train value MSE 0.016 vs val
  ~0.20, sign-acc drifted *down* 0.887 → 0.866). Gate vs base @ 400 sims:
  **+8 =28 −12 (45.8%)** over 48 games — statistically even, point estimate
  slightly negative → **not promoted**. Diagnosis: method works (the policy
  moved, unlike frozen-trunk Step 1), but 100k positions / 818 games is far too
  little to fine-tune the value head without overfitting, and the degraded value
  head cancels the policy gain at the MCTS leaves. The fix is **data volume**,
  not the approach. Next: generate a large visit-distribution dataset with the
  (now-fixed) Rust generator and retrain.
- **2026-06-17 (second gate — informative negative).** Generated 878k positions
  / 8,000 games @ 128 sims (`v2.1-exit-128`, 12k games/h) and retrained
  `exit2` with identical hyperparameters (only data volume changed). Held-out
  metrics improved *more* than exit1 — best epoch 1: policy CE 1.2400 → 1.1452,
  value MSE 0.2453 → 0.2333, sign-acc 0.813 → 0.830 (value head still overfits
  from epoch 2 on; early-stop kept epoch 1). **But the gate got worse, not
  better: +9 =23 −16 (42.7%) @ 400 sims** — below both 50% and exit1's 45.8%.
  Better supervised fit to the self-play targets **anti-correlates** with MCTS
  strength here. Leading hypothesis: the policy is distilled toward the **128-sim
  visit distribution of the base net**, a *weaker* teacher than base's own
  400-sim search; the more the policy moves toward it (exit2 moved more than
  exit1), the more the 400-sim prior is degraded. I.e. a generate/eval search-
  budget mismatch, not necessarily a dead method. Diagnostic to disentangle:
  re-gate exit2 vs base **at 128 sims** (matching the teacher). If exit2 ≥ base
  at 128 but < base at 400, the mismatch explains it and the fix is to generate
  visits at the eval budget (or higher). If exit2 < base at 128 too, expert
  iteration genuinely doesn't help the 11.7M net → Step 3 (46M capacity).
- **2026-06-17 (diagnostic).** Re-gated exit2 vs base **at 128 sims** (the
  teacher's budget): **+11 =24 −13 (47.9%)**, vs 42.7% at 400 sims. So the
  mismatch is real — the policy was pulled toward a 128-sim teacher and degrades
  more the harder you search past it — **but even at the matched budget exit2 is
  only even with base (≈50%, marginally negative), not better.** Expert
  iteration as set up here yields no strength on the 11.7M net and hurts above
  the teacher budget. This is the **4th consistent non-gain** (after Stockfish
  distillation, GRPO, frozen-trunk value retraining). Key methodological lesson
  surfaced: **supervised val loss (policy CE / value MSE) anti-correlates with
  MCTS strength** — exit2 had the best held-out metrics and the worst gate — so
  checkpoint selection must be done by MCTS games, not loss. Open confounds not
  yet removed: (a) checkpoint selected by loss, not MCTS (only the loss-best
  epoch was saved/gated); (b) policy vs value contributions not isolated (value
  head overfits from epoch 2); (c) teacher generated below eval budget.
- **2026-08-19 (the gates could not have detected any of this).** Before running
  a fifth experiment, we asked what the previous four could have measured. The
  answer reframes every null above. Simulating the sequential test below against
  a 66%-draw match gives the games needed to decide H0 = 0 vs H1 = +15 Elo:

  | true effect | median games | p90 |
  |---|---:|---:|
  | +80 Elo | 96 | 142 |
  | +40 Elo | 220 | 378 |
  | +25 Elo | 388 | 738 |
  | +15 Elo | 850 | 1876 |
  | 0 Elo | 834 | 2102 |

  Every gate in this document ran at 48–162 games. That resolves roughly +80 Elo
  and nothing finer — which is exactly the pattern in the record: the −89 Elo
  blitz regression came back clean and unambiguous, while every +20…+50 Elo
  question came back "not significant". **Those were not weak results, they were
  unmeasurable ones**, and "the 11.7M net is at capacity" is therefore not yet a
  measured conclusion.

  The binding constraint was the **opening book**, not the statistics. Both
  engines are deterministic at fixed nodes (`move_temp=0`, no root noise), so
  replaying an opening reproduces the same game move for move. The hand-written
  books cap `head_to_head.py` at 132 games and `engine_match.py` at 184, and no
  amount of re-running adds information. Fixed by
  `scripts/build_opening_book.py`: 2000 Stockfish-balanced lines (|cp| ≤ 80,
  deduped by reached position) → 4000 games of capacity.

  Also landed: **pentanomial scoring** in
  `src/chesstransformer/evaluation/sprt.py`. The two colour-swapped games of an
  opening are one sample, so opening difficulty cancels instead of inflating the
  interval. The self-test makes the mechanism visible — v2.1 gated against
  itself over 24 pairs returns pentanomial `[0, 0, 24, 0, 0]`, score exactly
  0.5000, CI ±0.00pp, **despite 16 of those 48 games being decisive**. And an
  **SPRT** that stops when the answer is clear and says *undecided* rather than
  offering a point estimate to over-read. Measured calibration: 4.0% type-I
  error at 0 Elo, 96.7% power at the +15 Elo boundary (α = β = 0.05).

  One consequence for the record above: the sharpened cp value head (S=3.0) was
  promoted to "worth ~+22 Elo" off 48 games at 400 sims (53.1%). The 162-game
  confirmation reads **46.9% at 400 sims and 53.4% at 800** — the original result
  did not replicate at its own budget, and the surviving effect is +24 ± 27 Elo
  (1σ). Re-gating it under SPRT at the production budget is the first use of the
  new harness.

- **2026-08-19 (a mechanism for the expert-iteration nulls).** `grep` for
  `dirichlet|gumbel|root_noise|exploration_noise` across every `.rs` and `.py`
  in the repo returns **zero matches**, and `rust/selfplay-core/src/lib.rs`
  (`finish_move`) builds the policy target directly from raw `root.n[i]`. **There
  is no root exploration in self-play at all** — the only variety comes from the
  opening book and `move_temp` sampling, neither of which perturbs the priors
  PUCT descends.

  Without root noise PUCT can only visit what the prior already ranks highly,
  and `scripts/eval_search_coverage.py` measures the floor: moves under
  `fpu/(c_puct*sqrt(1+sims))` (~0.47% prior at production settings) are never
  searched. The stored data agrees — `v2.1-400sims-exit` has 700,595 visit
  entries over 100,060 positions, a **mean support of 7.0 moves** out of ~30
  legal. So the visit distribution is a monotone sharpening of the prior over
  moves the prior already liked, and training the prior on it is
  self-distillation: it lowers policy entropy and adds no information.

  This predicts the observed signature exactly. `exit2` fit the targets *better*
  than `exit1` (val CE 1.2400→1.1452 vs 1.3075→1.2371) and gated *worse* (42.7%
  vs 45.8%). AlphaZero's policy-improvement operator requires search to discover
  moves the prior undervalues; that mechanism was simply absent. Fixing it
  (Dirichlet, then Gumbel with completed-Q targets) is queued as step 3 — but
  only after the two cheaper axes below, and only against a gate that can now
  see the answer.

- **2026-08-19 (step 4: dense Stockfish reward, in progress).** Distinct from all
  four attempts above by construction. `scripts/gen_sf_move_rewards.py` scores a
  candidate move set per self-play position with Stockfish at fixed depth;
  `scripts/grpo_selfplay.py` trains on it. What makes it a different bet:

  - **Not distillation.** `distill_policy.py` minimised CE against Stockfish's
    distribution over all 4672 actions — asking an 11.7M head to *reproduce* a
    stronger player. This only asks it to **re-rank the moves it already
    considers**, which is a far smaller demand on capacity.
  - **Not puzzle GRPO.** The reward is dense cp rather than a binary
    solution match, and every position comes from the engine's own games rather
    than a tactics set it never plays from.
  - **Not expert iteration.** No visit counts, so the sharpened-prior failure
    above cannot arise.

  Candidates are the union of the policy's top-K **and Stockfish's top-M**. The
  second half is deliberate: a table holding only the policy's own preferences
  could never teach it a good move it currently ranks low, which would rebuild
  the same blind spot the visibility floor already imposes. Verified on 40
  positions — SF's best move is in the candidate set 40/40 times, and the median
  cp gap between the stored argmax and a fresh SF search is 0.

  Two implementation traps, both now covered by `tests/test_grpo_selfplay.py`
  because both fail *silently* as a null result:
  1. The advantage baseline must be the **policy-weighted** mean. The update
     weights each candidate by π(c), so a uniform-mean baseline leaves
     `Σ π̃(c)·A_c ≠ 0` — a net push on the whole candidate block that says nothing
     about which move is better. End to end, the uniform baseline drove the
     expected cp of the policy's own moves **down 68cp**; the policy-weighted one
     drives it **up 93cp**.
  2. A group whose candidates are all equally good has σ ≈ 0, and dividing its
     rounding-level deviations by that σ amplifies float32 noise into a ~0.03
     advantage. Gated explicitly — dead-drawn and won endgames hit this often.

  Selection is by match play only (`scripts/gate_candidate.sh`), never by any
  number the trainer prints, per the lesson recorded above.
