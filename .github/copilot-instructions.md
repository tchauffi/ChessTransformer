# ChessTransformer AI Agent Instructions

## Project Overview
This project explores using transformer architectures for chess engines, treating chess moves as sequences similar to NLP. The hypothesis: transformers can learn chess patterns like they learn language patterns.

## Architecture: Two Parallel Approaches

### 1. **Move Sequence → Move Sequence** (`src/chesstransformer/models/transformer/gpt.py`)
- GPT-style autoregressive model predicting next move from move history
- Dataset: `LichessSimpleUciDataset` (processes .pgn.zst files on-the-fly)
- Status: Initial implementation showed promise but struggled with legal move consistency

### 2. **Position → Move** (`src/chesstransformer/models/transformer/position2move.py`)
- Current active approach (see branch `position_based_model`)
- Input: 64-token board representation (one token per square) + turn indicator
- Output: Next move prediction from 1968-token move vocabulary
- Uses triple embedding: token (13 pieces) + position (64 squares) + player (2 colors)
- Dataset: HDF5 format for efficiency (`HDF5ChessDataset` + `PGNtoHDF5Converter`)

## Key Tokenization Strategy

**Position Tokenizer** (`position_tokenizer.py`):
- 13-token vocabulary: empty (0), 6 white pieces (1-6), 6 black pieces (7-12)
- Encodes board as flat 64-element array (a1→h1, a2→h2, ..., a8→h8)
- Board representation is flipped from chess.Board's internal ordering

**Move Tokenizer** (`move_tokenizer.py`):
- Vocabulary of 1968 legal chess moves in UCI format
- Pre-computed vocab stored in `data/tokenizer_models/chess_moves_vocab.json`
- All moves follow UCI notation (e.g., `e2e4`, `e7e8q` for promotion)

## Data Pipeline

1. **Source**: Lichess rated games (`.pgn.zst` compressed format in `data/`)
2. **Conversion**: `dataset_h5_convertor.py` → HDF5 with gzip compression
   - Run as: `python src/chesstransformer/datasets/dataset_h5_convertor.py`
3. **Training Dataset**: `h5_lichess_dataset.py` with LRU caching
   - Structure: `{'position': tensor(64), 'move': tensor(1), 'is_white': bool, 'game_id': int}`

## Training Workflow

**Trainer**: `moveseq2moveseq_trainer.py` (adapts for both model types)
- Uses Hugging Face Accelerate for multi-GPU/mixed precision
- Custom loss: `llm_loss + 10 * l2_illegal_moves_penalty`
- Illegal move masking via `legal_moves_mask` tensor
- TensorBoard logging with auto-incrementing run numbers (`logs/run_XXX_YYYYMMDD_HHMMSS/`)
- Learning rate: warmup (1000 steps) → linear decay
- Gradient accumulation for effective large batch sizes

**Key Training Command Pattern**:
```python
python src/chesstransformer/trainers/moveseq2moveseq_trainer.py
```

## Project Conventions

- **Package management**: Uses `hatchling` build system (NOT Poetry/setuptools)
- **Virtual env**: Expects `.venv` in project root (created via `pip install -e .`)
- **Model checkpoints**: Stored in both `data/models/` and `src/chesstransformer/models/`
- **Notebooks**: Used for experimentation (`notebooks/`), not production code
- **Tests**: unittest framework, run via `python -m unittest` from project root

## Critical Development Details

**When working with datasets**:
- Always use `zstandard` decompression for .pgn.zst files
- Skip non-Standard variants (`variant != "Standard"`)
- HDF5 files use resizable datasets with chunked compression
- Default chunk size: 10,000 positions

**When working with transformers**:
- Config dict pattern: `{"embed_dim": 512, "num_heads": 8, "context_size": 64, ...}`
- All models use custom `LayerNorm` and `GELU` implementations (see `utils.py`)
- Attention uses causal masking via `mask_future` parameter
- Position embeddings differ: GPT uses sequence positions, Position2Move uses board square positions

**Chess-specific considerations**:
- Legal move enforcement is soft (L2 penalty), not hard filtering
- Game outcomes encoded as special tokens for supervised learning signal
- python-chess library is the canonical interface for board logic
- UCI format is standard throughout (no SAN/algebraic notation)

## Documentation References

- **Design rationale**: `doc/design_stategy.md` (explains move-seq vs position-based approaches)
- **Tokenization details**: `doc/tokenization.md` (comprehensive UCI strategy breakdown)
- **Transformer theory**: `doc/transformer.md` (project-specific architecture decisions)

## Common Pitfalls

1. **Board orientation**: `PostionTokenizer` flips board order - verify array indexing matches chess squares
2. **Vocab paths**: Tokenizers default to `data/tokenizer_models/` - use absolute paths when unsure
3. **TensorBoard conflicts**: Multiple runs can interfere - always increment run numbers
4. **H5 file locking**: Close HDF5 files properly or risk corruption
5. **Gradient accumulation**: `accelerator.sync_gradients` check is required for proper metric logging
