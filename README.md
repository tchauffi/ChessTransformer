# ChessTransformer

This project aims to test the hypothesis that transformer is a suitable architecture for chess engines. After learning more about transformers, and it's capacity to model long text sequences, I want to see if it can be applied to chess, which is also a game with long sequences of moves. My main intuition is that a squence of chess moves can be treated similarly to a sequence of words in a sentence, where each move is dependent on the previous moves.

The final goal of the project is to build a transformer-based chess engine that can play chess at a high level, and evaluate its performance against existing chess engines such as Stockfish.

## Features

### ✅ Implemented
- ✅ Chess move tokenization (UCI format)
- ✅ Position-based transformer model (Position2Move)
- ✅ Training on Lichess game database
- ✅ **Puzzle-based training and evaluation** (NEW!)

### 🚧 To Do
- [ ] Fine tune model to play chess using reinforcement learning
- [ ] Add the possibility to choose the elo level of the engine
- [ ] Evaluate model performance against existing chess engines
- [ ] Optimize model for inference speed and memory usage
- [ ] Explore potential applications of the model in chess analysis and training tools
- [x] Create Next.js frontend for human-vs-bot testing

### 💡 Future Ideas
- [ ] Multimodal approach: vision transformer for chessboard images + move sequences
- [ ] LLM integration for natural language move explanations

## Quick Start

### Train on Chess Puzzles

Train a model on tactical puzzles from Lichess:

```bash
# Quick test with limited puzzles
uv run scripts/test_puzzle_dataset.py

# Full training on medium difficulty puzzles
./scripts/train_puzzle.sh --min-rating 1200 --max-rating 1800 --epochs 10

# Or use the trainer directly
uv run src/chesstransformer/trainers/puzzle_trainer.py --help
```

### Evaluate Puzzle Performance

```bash
uv run src/chesstransformer/utils/evaluate_puzzles.py \
    --model data/models/puzzle_training/run_001/best_model.pth \
    --num-puzzles 1000
```

See [doc/puzzle_training.md](doc/puzzle_training.md) for detailed documentation.

## Project installation
The project uses UV for Python dependency management. To install the project:

```bash
uv sync
```

This will install all required dependencies including the web frontend API dependencies (FastAPI, uvicorn).

For development with additional tools:

```bash
uv sync --group dev
```

## Web Frontend for Human vs Bot Testing

A Next.js web application is available for testing the chess bot interactively. See [frontend/README.md](frontend/README.md) for detailed setup and usage instructions.

**Quick Start (Development):**

1. Start the backend API server:
```bash
cd backend
uv run python api.py
```

2. In a separate terminal, start the frontend:
```bash
cd frontend
npm install  # First time only
npm run dev
```

3. Open http://localhost:3000 in your browser and start playing!

## Docker Deployment

Deploy the entire application using Docker Compose:

**Development mode:**
```bash
docker compose up --build
```

The application will be available at:
- Development: Frontend at http://localhost:3000, API at http://localhost:5001
- Production: Everything at http://localhost (port 80)

## Project structure
The project is structured as follows:
```
ChessTransformer/
├── backend/                # FastAPI backend server for bot integration
│   ├── api.py             # API endpoints for chess bot
│   └── Dockerfile         # Backend container definition
├── frontend/               # Next.js web application
│   ├── app/               # Next.js app directory
│   ├── Dockerfile         # Frontend container definition
│   └── README.md          # Frontend documentation
├── docker-compose.yml      # Development Docker orchestration
├── docker-compose.prod.yml # Production Docker with Nginx
├── nginx.conf              # Nginx reverse proxy configuration
├── data/                   # Data processing scripts and datasets
├── doc/                    # Documentation files
├── src/                    # Source code
│   └── chesstransformer/   # Main package
│       ├── bots/          # Chess bot implementations
│       ├── data/           # Data loading and preprocessing
│       ├── models/         # Model architectures
│       ├── trainers/       # Training scripts
│       └── utils/          # Utility functions
├── tests/                  # Unit tests
├── .gitignore              # Git ignore file
├── LICENSE                 # License file
├── README.md               # Project overview and setup instructions
├── pyproject.toml          # Project metadata and dependencies
└── uv.lock                 # UV lock file
```
