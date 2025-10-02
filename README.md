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

*Bonus* 
- [ ] Try to explore a multimodal approach by coopling a LLM with a vision transformer to process chessboard images along with move sequences.
- [ ] Experiment the possibility to connect this ches engine with a LLM to provide natural language explanations of the moves and strategies used by the engine.

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
