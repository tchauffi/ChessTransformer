import unittest
import chess
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer


class TestPositionTokenizer(unittest.TestCase):
    """Unit tests for the PostionTokenizer class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tokenizer = PostionTokenizer()

    def test_vocab_initialization(self):
        """Test that vocabulary is correctly initialized."""
        expected_vocab = {
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
        self.assertEqual(self.tokenizer.vocab, expected_vocab)

    def test_inverse_vocab(self):
        """Test that inverse vocabulary is correctly created."""
        for symbol, token_id in self.tokenizer.vocab.items():
            self.assertEqual(self.tokenizer.inv_vocab[token_id], symbol)

    def test_encode_starting_position(self):
        """Test encoding the standard chess starting position."""
        board = chess.Board()
        token_ids = self.tokenizer.encode(board)

        # Check that we have 64 tokens (8x8 board)
        self.assertEqual(len(token_ids), 64)

        # Check first rank (white pieces from a1 to h1)
        expected_first_rank = [4, 2, 3, 5, 6, 3, 2, 4]  # R N B Q K B N R
        self.assertEqual(token_ids[:8], expected_first_rank)

        # Check second rank (white pawns from a2 to h2)
        expected_second_rank = [1] * 8  # P P P P P P P P
        self.assertEqual(token_ids[8:16], expected_second_rank)

        # Check middle ranks (empty squares)
        expected_empty_ranks = [0] * 32  # 4 ranks of empty squares
        self.assertEqual(token_ids[16:48], expected_empty_ranks)

        # Check seventh rank (black pawns from a7 to h7)
        expected_seventh_rank = [7] * 8  # p p p p p p p p
        self.assertEqual(token_ids[48:56], expected_seventh_rank)

        # Check eighth rank (black pieces from a8 to h8)
        expected_eighth_rank = [10, 8, 9, 11, 12, 9, 8, 10]  # r n b q k b n r
        self.assertEqual(token_ids[56:64], expected_eighth_rank)

    def test_encode_empty_board(self):
        """Test encoding an empty board."""
        board = chess.Board.empty()
        token_ids = self.tokenizer.encode(board)

        # All squares should be empty (token ID 0)
        self.assertEqual(len(token_ids), 64)
        self.assertEqual(token_ids, [0] * 64)

    def test_decode_starting_position(self):
        """Test decoding back to the starting position."""
        original_board = chess.Board()
        token_ids = self.tokenizer.encode(original_board)
        decoded_board = self.tokenizer.decode(token_ids)

        # Compare board FEN positions
        self.assertEqual(original_board.board_fen(), decoded_board.board_fen())

    def test_decode_empty_board(self):
        """Test decoding an empty board."""
        original_board = chess.Board.empty()
        token_ids = self.tokenizer.encode(original_board)
        decoded_board = self.tokenizer.decode(token_ids)

        # Compare board FEN positions
        self.assertEqual(original_board.board_fen(), decoded_board.board_fen())

    def test_encode_decode_roundtrip(self):
        """Test that encoding and then decoding returns the same position."""
        # Test with various positions
        test_positions = [
            chess.Board(),  # Starting position
            chess.Board.empty(),  # Empty board
            chess.Board(
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
            ),  # After e4
            chess.Board(
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
            ),  # After e4 e5
            chess.Board(
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
            ),  # After e4 e5 Nf3 Nc6
        ]

        for original_board in test_positions:
            with self.subTest(fen=original_board.board_fen()):
                token_ids = self.tokenizer.encode(original_board)
                decoded_board = self.tokenizer.decode(token_ids)
                self.assertEqual(original_board.board_fen(), decoded_board.board_fen())

    def test_encode_custom_position(self):
        """Test encoding a custom position."""
        # Create a simple endgame position: King + Rook vs King
        board = chess.Board.empty()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))

        token_ids = self.tokenizer.encode(board)

        # Check that we have 64 tokens
        self.assertEqual(len(token_ids), 64)

        # Check specific pieces are in the right positions
        # a1 = index 0, should be white Rook (4)
        self.assertEqual(token_ids[0], 4)
        # e1 = index 4, should be white King (6)
        self.assertEqual(token_ids[4], 6)
        # e8 = index 60, should be black King (12)
        self.assertEqual(token_ids[60], 12)

        # All other squares should be empty (0)
        non_piece_indices = [i for i in range(64) if i not in [0, 4, 60]]
        for idx in non_piece_indices:
            self.assertEqual(token_ids[idx], 0)

    def test_decode_custom_position(self):
        """Test decoding a custom position."""
        # Create token IDs for a simple position
        token_ids = [0] * 64
        token_ids[0] = 4  # White Rook on a1
        token_ids[4] = 6  # White King on e1
        token_ids[60] = 12  # Black King on e8

        board = self.tokenizer.decode(token_ids)

        # Verify the pieces are in the correct positions
        self.assertEqual(board.piece_at(chess.A1), chess.Piece(chess.ROOK, chess.WHITE))
        self.assertEqual(board.piece_at(chess.E1), chess.Piece(chess.KING, chess.WHITE))
        self.assertEqual(board.piece_at(chess.E8), chess.Piece(chess.KING, chess.BLACK))

        # Verify empty squares
        self.assertIsNone(board.piece_at(chess.D4))
        self.assertIsNone(board.piece_at(chess.H8))

    def test_encode_after_moves(self):
        """Test encoding after making some moves."""
        board = chess.Board()

        # Play Italian Game opening: e4 e5 Nf3 Nc6 Bc4 Nf6
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
        for move in moves:
            board.push_uci(move)

        token_ids = self.tokenizer.encode(board)

        # Should still have 64 tokens
        self.assertEqual(len(token_ids), 64)

        # Verify encoding and decoding roundtrip
        decoded_board = self.tokenizer.decode(token_ids)
        self.assertEqual(board.board_fen(), decoded_board.board_fen())

    def test_all_piece_types(self):
        """Test that all piece types are correctly encoded and decoded."""
        board = chess.Board.empty()

        # Place one of each piece type for each color
        white_pieces = [
            (chess.A1, chess.PAWN),
            (chess.B1, chess.KNIGHT),
            (chess.C1, chess.BISHOP),
            (chess.D1, chess.ROOK),
            (chess.E1, chess.QUEEN),
            (chess.F1, chess.KING),
        ]

        black_pieces = [
            (chess.A8, chess.PAWN),
            (chess.B8, chess.KNIGHT),
            (chess.C8, chess.BISHOP),
            (chess.D8, chess.ROOK),
            (chess.E8, chess.QUEEN),
            (chess.F8, chess.KING),
        ]

        for square, piece_type in white_pieces:
            board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))

        for square, piece_type in black_pieces:
            board.set_piece_at(square, chess.Piece(piece_type, chess.BLACK))

        # Test roundtrip
        token_ids = self.tokenizer.encode(board)
        decoded_board = self.tokenizer.decode(token_ids)

        self.assertEqual(board.board_fen(), decoded_board.board_fen())

    def test_token_id_ranges(self):
        """Test that all token IDs are within expected range."""
        board = chess.Board()
        token_ids = self.tokenizer.encode(board)

        # All token IDs should be between 0 and 12
        for token_id in token_ids:
            self.assertGreaterEqual(token_id, 0)
            self.assertLessEqual(token_id, 12)

    def test_ascii2board_helper(self):
        """Test the _ascii2board helper method."""
        ascii_board = """rnbqkbnr
pppppppp
........
........
........
........
PPPPPPPP
RNBQKBNR"""

        board = self.tokenizer._ascii2board(ascii_board)
        expected_board = chess.Board()

        self.assertEqual(board.board_fen(), expected_board.board_fen())


if __name__ == "__main__":
    unittest.main()
