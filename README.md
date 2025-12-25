# ChessTransformer

This project aims to test the hypothesis that transformer is a suitable architecture for chess engines. After learning more about transformers, and it's capacity to model long text sequences, I want to see if it can be applied to chess, which is also a game with long sequences of moves. My main intuition is that a squence of chess moves can be treated similarly to a sequence of words in a sentence, where each move is dependent on the previous moves.

The final goal of the project is to build a transformer-based chess engine that can play chess at a high level, and evaluate its performance against existing chess engines such as Stockfish.

## To do list
- [ ] Implement tokenization for chess moves
- [ ] Build and train pre trained transformer model using Lichess database
- [ ] Fine tune model to play chess using reinforcement learning
- [ ] Add the possibility to choose the elo level of the engine
- [ ] Evaluate model performance against existing chess engines
- [ ] Optimize model for inference speed and memory usage
- [ ] Explore potential applications of the model in chess analysis and training tools
- [x] Create Next.js frontend for human-vs-bot testing

*Bonus* 
- [ ] Try to explore a multimodal approach by coopling a LLM with a vision transformer to process chessboard images along with move sequences.
- [ ] Experiment the possibility to connect this ches engine with a LLM to provide natural language explanations of the moves and strategies used by the engine.

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

**Quick Start:**

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

## Project structure
The project is structured as follows:
```
ChessTransformer/
├── backend/                # FastAPI backend server for bot integration
│   └── api.py             # API endpoints for chess bot
├── frontend/               # Next.js web application
│   ├── app/               # Next.js app directory
│   └── README.md          # Frontend documentation
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
