# ChessTransformer

A transformer-based chess engine trained on elite Lichess games. The model learns to predict moves directly from board positions, then plays via alpha-beta search guided by its policy and value heads.

**Current strength: ~1550 Elo** (depth-3 alpha-beta vs Stockfish, June 2026)

## Architecture — Pos2MoveV2

| Component | Details |
|---|---|
| Parameters | 11.7M |
| Attention | Grouped Query Attention (8 heads, 4 KV groups) + QK-norm |
| Position bias | Learnable chess-geometry relative bias (8 relation categories: file, rank, diagonal, knight-reach, king-adjacent, nearby, far, global) |
| Policy head | AlphaZero-style 64×73 action planes |
| Value head | Board state → scalar in (−1, 1) |
| Training | Muon + AdamW mixed optimizer, BF16, stochastic depth |
| Search | Iterative-deepening alpha-beta, nucleus filtering (top-p), transposition table |

## Elo Evaluation

| Stockfish skill | Approx. Elo | Result |
|---|---|---|
| ≤ 2 | ≤ 1000 | 100% win rate |
| 7 | ~1600 | ~50 / 50 |
| 8 | ~1700 | Heavy losses |

Evaluated with `scripts/elo_gauntlet.py` over 100 games across skills 0–10.

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
| `MODEL_PATH` | `data/models/pos2move_v2` | Path to a checkpoint directory |
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
# Elo gauntlet vs Stockfish
uv run scripts/elo_gauntlet.py data/models/pos2move_v2 \
    --depth 3 --games 10 --skills 0 2 4 6 7 8

# Export to ONNX (for TensorRT)
uv run scripts/export_onnx.py data/models/pos2move_v2
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
│   └── models/pos2move_v2/           # Bundled model weights
├── scripts/
│   ├── build_db.py                   # Download elite games and build HDF5 database
│   ├── elo_gauntlet.py               # Elo estimation vs Stockfish
│   ├── export_onnx.py                # ONNX export for TensorRT
│   ├── quantize_onnx.py              # INT8 quantization
│   ├── dataset_sanity_check.py       # Dataset distribution analysis
│   └── compress_pgn_to_zst.py        # PGN compression utility
├── src/chesstransformer/
│   ├── bots/
│   │   ├── pos2move_v2_bot.py        # Alpha-beta bot (main)
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
