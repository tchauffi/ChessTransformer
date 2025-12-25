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
The project is using uv. To install the project, you can use pip:

```bash
pip install .
```
Or you can install the project in editable mode for development:

```bash 
pip install -e .
```

## Project structure
The project is structured as follows:
```
ChessTransformer/
├── data/                   # Data processing scripts and datasets
├── doc/                    # Documentation files
├── src/                    # Source code
│   └── chesstransformer/   # Main package
│       ├── data/           # Data loading and preprocessing
│       ├── models/         # Model architectures
│       ├── training/       # Training and evaluation scripts
│       └── utils/          # Utility functions
├── tests/                  # Unit tests
├── .gitignore              # Git ignore file
├── LICENSE                 # License file
├── README.md               # Project overview and setup instructions
├── pyproject.toml          # Project metadata and dependencies
```
