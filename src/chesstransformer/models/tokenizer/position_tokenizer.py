import chess


class PostionTokenizer:
    """A simple tokenizer for chess board positions.
    Each square on the board is represented by a token ID based on the piece occupying it.
    Empty squares are represented by the token ID 0.
    The board is represented as an 8x8 grid, flattened into a list of 64 tokens.
    """

    def __init__(self):
        self.vocab = {
            "P": 1,
            "N": 2,
            "B": 3,
            "R": 4,
            "Q": 5,
            "K": 6,
            "p": 7,
            "n": 8,
            "b": 9,
            "r": 10,
            "q": 11,
            "k": 12,
            ".": 0,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, board: chess.Board) -> list[int]:
        """Encode a chess.Board object into a list of token IDs.
        The board is represented as an 8x8 grid, flattened into a list of 64 tokens.

        Args:
            board: A chess.Board object

        Returns:
            List of token IDs representing the board position
        """
        reordered_board = str(board).split("\n")[::-1]
        flattened_board = [
            char for row in reordered_board for char in row if char != " "
        ]
        token_ids = [self.vocab[char] for char in flattened_board]
        return token_ids

    def decode(self, token_ids: list[int]) -> chess.Board:
        """Decode a list of token IDs back into a chess board string.
        Args:
            token_ids: List of token IDs representing the board position
            Returns:
            A chess.Board object
        """
        chars = [self.inv_vocab[token_id] for token_id in token_ids]
        rows = ["".join(chars[i * 8 : (i + 1) * 8]) for i in range(8)][
            ::-1
        ]  # Reverse to get the correct order
        board_str = "\n".join(rows)
        board = self._ascii2board(board_str)
        return board

    def _ascii2board(self, ascii_board: str) -> chess.Board:
        """Convert an ASCII board representation back to a chess.Board object.
        Args:
            ascii_board: String representation of the board (8 lines of 8 characters)
        Returns:
            A chess.Board object
        """
        board = chess.Board.empty()
        rows = ascii_board.split("\n")
        for rank in range(8):
            file = 0
            for char in rows[7 - rank]:  # Reverse the order of ranks
                if char in self.vocab and char != ".":
                    square = chess.square(file, rank)
                    piece = chess.Piece.from_symbol(char)
                    board.set_piece_at(square, piece)
                file += 1
        return board
