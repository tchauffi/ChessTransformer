# ChessTransformer

A transformer-based chess engine trained on elite Lichess games. The model learns to predict moves directly from board positions, then plays via an AlphaZero-style MCTS (policy priors + value head), with a compiled alpha-beta engine as an alternative.

**Current strength: ~2100 Elo** (MCTS @ 800 sims with FPU + tuned c_puct + tree reuse, model v2.1, vs Stockfish — MLE estimate over a 140-game gauntlet, skills 0–12, June 2026)

## Demo

ChessTransformer v2.1 (White, MCTS @ 800 sims) checkmating Stockfish Level 8 — the finishing sequence:

![ChessTransformer checkmates Stockfish Level 8](docs/demo.gif)

▶ [Watch the full game (MP4)](docs/demo.mp4)

## Improvements (June 2026)

| Change | Effect |
|---|---|
| **MCTS / PUCT engine** (new default) | beat the alpha-beta engine ~82% head-to-head |
| **`torch.compile` + CUDA graphs** | forward ~2.3× faster (lossless) |
| **Batched-leaf MCTS** (virtual loss) | ~8× faster per sim — amortizes the GPU→CPU sync |
| **Search tuning** — FPU (`fpu=0.2`), `c_puct=1.0`, 800 sims | +~280 Elo over untuned baseline |
| **Tree reuse** across moves | re-roots the retained subtree — deeper search at the same per-move cost |
| **Model v2.1** | promoted from `run_021`, now the default weights |
| **MLE Elo estimator** | per-level averaging was biased low; fit a single Elo over all games |

Tried and rejected: **Stockfish policy distillation** — no gain even at 200k labels (the policy is near the 11.7M model's capacity ceiling).

## Architecture — Pos2MoveV2

| Component | Details |
|---|---|
| Parameters | 11.7M |
| Attention | Grouped Query Attention (8 heads, 4 KV groups) + QK-norm |
| Position bias | Learnable chess-geometry relative bias (8 relation categories: file, rank, diagonal, knight-reach, king-adjacent, nearby, far, global) |
| Policy head | AlphaZero-style 64×73 action planes |
| Value head | Board state → scalar in (−1, 1) |
| Training | Muon + AdamW mixed optimizer, BF16, stochastic depth |
| Search (default) | **MCTS / PUCT** — policy priors + value head, batched-leaf evaluation with virtual loss |
| Search (alt) | Iterative-deepening alpha-beta with quiescence, nucleus filtering (top-p), transposition table |
| Inference | `torch.compile` + CUDA graphs, bucketed batches, BF16 |

## Search engines

Two search engines share the same network (`Pos2MoveV2`):

- **MCTS / PUCT** (`Pos2MoveV2MctsBot`, default) — AlphaZero-style. Policy head → priors, value head → leaf scores, most-visited move chosen. Batched-leaf evaluation with virtual loss amortizes the GPU→CPU sync (~8× faster than single-leaf). Tuned for **exploitation**: first-play-urgency (`fpu=0.2`) and `c_puct=1.0` (flatter priors / more exploration both lost). With FPU the search scales with simulations, so the default is **800 sims**. **Tree reuse** re-roots the retained subtree under the moves played (when the caller keeps `board.move_stack`), giving deeper search at the same per-move cost.
- **Alpha-beta** (`Pos2MoveV2Bot`) — iterative-deepening negamax with quiescence search, policy-prior move ordering, and a Zobrist transposition table.

Both run a `torch.compile`/CUDA-graph forward (~2.3× faster, lossless).

## Elo Evaluation

MCTS @ 800 sims (`c_puct=1.0`, `fpu=0.2`, tree reuse), model v2.1, 20 games/level vs Stockfish (`scripts/tune_vs_stockfish.py`):

| Stockfish skill | Approx. Elo | Score |
|---|---|---|
| 0–6 | ≤ 1500 | 100% |
| 8 | ~1700 | 90% |
| 10 | ~1900 | 85% |
| 12 | ~2100 | 35% |

**MLE estimate: ~2100 Elo.** Elo is fit by maximum likelihood over all games (averaging per-level estimates is biased low — saturated easy levels cap at a low value and drag the mean). Search tuning (FPU + `c_puct` + 800 sims) added ~+280 Elo over the untuned MCTS@400 baseline (~1793), and tree reuse a further small gain — all with no retraining. (Per-level scores carry ~20-game noise; the MLE smooths it.)

## Quick Start

### Docker (recommended)

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- API: http://localhost:5001

Model weights are baked into the backend image — no volume mounts needed.

**With GPU (NVIDIA):**
```bash
docker compose up --build   # deploy.resources.reservations already set in docker-compose.yml
```
Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

### Local Development

1. Install dependencies:
```bash
uv sync
```

2. Start the backend:
```bash
uv run python backend/api.py
```

3. Start the frontend (separate terminal):
```bash
cd frontend
npm install   # first time only
npm run dev
```

4. Open http://localhost:3000 and start playing.

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `data/models/pos2move_v2.1` | Path to a checkpoint directory |
| `ENGINE` | `mcts` | Search engine: `mcts` or `alphabeta` |
| `MCTS_SIMS` | `800` | MCTS simulations per move (when `ENGINE=mcts`) |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |

## Training

### Building the database

`scripts/build_db.py` downloads elite games from [database.nikonoel.fr](https://database.nikonoel.fr) and converts them to HDF5 in one step. Bullet and blitz games are excluded by default (keeps rapid, classical, correspondence).

```bash
# Download last 12 months and build database (default)
uv run scripts/build_db.py

# Specific date range
uv run scripts/build_db.py --from 2024-01 --to 2024-12

# Last 6 months
uv run scripts/build_db.py --last 6

# Everything available
uv run scripts/build_db.py --all

# Re-convert already-downloaded PGN files without re-downloading
uv run scripts/build_db.py --skip-download

# Include all game types (no time control filter)
uv run scripts/build_db.py --no-filter
```

Output goes to `data/elite_db.h5` and raw PGN files are cleaned up automatically unless `--keep-raw` is passed.

```bash
# Inspect position distribution after building
uv run scripts/dataset_sanity_check.py --data data/elite_db.h5 --games 5000
```

### Train Pos2MoveV2

```bash
uv run src/chesstransformer/trainers/pos2move_v2_trainer.py
```

### Evaluate

```bash
# Tune & benchmark search budget vs Stockfish (alpha-beta depths + MCTS sims)
uv run scripts/tune_vs_stockfish.py data/models/pos2move_v2.1 \
    --games 8 --skills 0 2 4 6 8

# Alpha-beta-only Elo gauntlet vs Stockfish
uv run scripts/elo_gauntlet.py data/models/pos2move_v2.1 \
    --depth 3 --games 10 --skills 0 2 4 6 7 8

# Deterministic engine-vs-engine A/B (MCTS vs alpha-beta, model A vs model B, ...)
uv run scripts/engine_match.py --a-mcts --a-sims 400 --b-quiescence 4 --b-depth 3

# Inference speed + lossless-regression guard
uv run scripts/bench_inference.py --depth 3 --save-golden golden.json
uv run scripts/bench_inference.py --depth 3 --check golden.json

# Export to ONNX (for TensorRT)
uv run scripts/export_onnx.py data/models/pos2move_v2.1
```

## Project Structure

```
ChessTransformer/
├── backend/
│   ├── api.py                        # FastAPI server (move, evaluate, validate endpoints)
│   └── Dockerfile
├── frontend/                         # Next.js web app (human vs bot)
│   └── app/components/ChessGame.tsx  # Main game component
├── data/
│   └── models/
│       ├── pos2move_v2.1/            # Bundled model weights (default)
│       └── pos2move_v2/              # Previous weights (fallback)
├── scripts/
│   ├── tune_vs_stockfish.py          # Sweep alpha-beta depth / MCTS sims vs Stockfish
│   ├── elo_gauntlet.py               # Elo estimation vs Stockfish (alpha-beta)
│   ├── engine_match.py               # Deterministic engine-vs-engine A/B
│   ├── bench_inference.py            # Inference speed + lossless-regression guard
│   ├── build_db.py                   # Download elite games and build HDF5 database
│   ├── export_onnx.py                # ONNX export for TensorRT
│   ├── quantize_onnx.py              # INT8 quantization
│   ├── dataset_sanity_check.py       # Dataset distribution analysis
│   └── compress_pgn_to_zst.py        # PGN compression utility
├── src/chesstransformer/
│   ├── bots/
│   │   ├── pos2move_v2_mcts_bot.py   # MCTS / PUCT bot (default)
│   │   ├── pos2move_v2_bot.py        # Alpha-beta bot (with quiescence + compile)
│   │   └── random_bot.py
│   ├── models/
│   │   ├── transformer/pos2move_v2.py  # Model architecture
│   │   └── tokenizer/
│   │       ├── alphazero_move_encoder.py  # 64×73 action planes
│   │       ├── position_tokenizer.py
│   │       └── move_tokenizer.py
│   ├── datasets/
│   │   ├── h5_lichess_dataset.py     # HDF5 dataset with phase-weighted sampling
│   │   └── dataset_h5_convertor.py
│   ├── optimizer.py                  # AdamW + Muon combined optimizer
│   └── trainers/
│       └── pos2move_v2_trainer.py
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

## Installation (development extras)

```bash
# Linting / formatting
uv sync --group dev

# ONNX / TensorRT export
uv sync --group optimized
```
