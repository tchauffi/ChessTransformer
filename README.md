# ChessTransformer

A transformer-based chess engine trained on elite Lichess games. The model learns to predict moves directly from board positions, then plays via an AlphaZero-style MCTS (policy priors + value head), with a compiled alpha-beta engine as an alternative.

**Current strength: ~1775 Elo** (MCTS @ 400 sims, model v2.1, vs Stockfish — 140-game gauntlet over skills 0–12, June 2026)

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

- **MCTS / PUCT** (`Pos2MoveV2MctsBot`, default) — AlphaZero-style. Uses the policy head as priors and the value head at leaves; picks the most-visited move. Batched-leaf evaluation with virtual loss collects several leaves per network call, amortizing the GPU→CPU sync (~8× faster than single-leaf). Beat the alpha-beta engine ~82% head-to-head.
- **Alpha-beta** (`Pos2MoveV2Bot`) — iterative-deepening negamax with quiescence search, policy-prior move ordering, and a Zobrist transposition table.

Both run a `torch.compile`/CUDA-graph forward (~2.3× faster, lossless).

## Elo Evaluation

MCTS @ 400 sims, model v2.1, 20 games/level vs Stockfish (`scripts/tune_vs_stockfish.py`):

| Stockfish skill | Approx. Elo | Score |
|---|---|---|
| 0–4 | ≤ 1200 | 95–100% |
| 6 | ~1500 | 70% |
| 8 | ~1700 | 62.5% |
| 10 | ~1900 | 40% |
| 12 | ~2100 | 27.5% |

**Weighted estimate: ~1775 Elo.** Tuning showed MCTS @ 400 sims is the sweet spot (~1725 in the quick scan, ~+175 Elo over the best alpha-beta config; 800 sims gave no gain).

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
| `MCTS_SIMS` | `400` | MCTS simulations per move (when `ENGINE=mcts`) |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |

## Training

### Data preparation

```bash
# Convert PGN/PGN.zst files to HDF5
uv run src/chesstransformer/datasets/dataset_h5_convertor.py \
    --input data/games/ --output data/elite_db.h5

# Inspect position distribution
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
