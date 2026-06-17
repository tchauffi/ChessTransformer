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
