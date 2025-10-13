import json
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data"

class MoveTokenizer:
    def __init__(self, vocab_path=DATA_DIR / "tokenizer_models/chess_moves_vocab.json"):

        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, move: str) -> int:
        """Encode a chess move string into a token ID.
        Args:
            move: A string representing a chess move in UCI format (e.g., "e2e4", "g1f3").
        Returns:
            An integer token ID representing the move.
        """
        if move not in self.vocab:
            raise ValueError(f"Move '{move}' not in vocabulary.")
        return self.vocab[move]
    
    def decode(self, token_id: int) -> str:
        """Decode a token ID back into a chess move string.
        Args:
            token_id: An integer token ID representing the move.
        Returns:
            A string representing a chess move in UCI format.
        """
        if token_id not in self.inv_vocab:
            raise ValueError(f"Token ID '{token_id}' not in vocabulary.")
        return self.inv_vocab[token_id]
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)
