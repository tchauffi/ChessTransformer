import csv
import zstandard as zstd
import torch
from torch.utils.data import Dataset
import chess
from typing import Optional
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer
from chesstransformer.models.tokenizer.move_tokenizer import MoveTokenizer


class LichessPuzzleDataset(Dataset):
    """Dataset for Lichess chess puzzles.
    
    Each puzzle consists of:
    - A FEN position (the position where the puzzle starts)
    - A sequence of moves (the solution to the puzzle)
    - Rating and themes
    
    The dataset returns position-move pairs where:
    - position: The current board state encoded as 64 tokens
    - move: The next move in the puzzle solution
    - is_white: Whether white is to move
    """
    
    def __init__(
        self, 
        puzzle_path: str,
        min_rating: int = None,
        max_rating: int = None,
        themes: list[str] = None,
        max_puzzles: int = None
    ):
        """
        Args:
            puzzle_path: Path to the .csv.zst puzzle file
            min_rating: Minimum puzzle rating to include
            max_rating: Maximum puzzle rating to include
            themes: List of themes to filter by (e.g., ['mate', 'mateIn2'])
            max_puzzles: Maximum number of puzzles to load (for testing)
        """
        self.position_tokenizer = PostionTokenizer()
        self.move_tokenizer = MoveTokenizer()
        self.puzzles = []
        
        print(f"Loading puzzles from {puzzle_path}...")
        
        # Load and filter puzzles
        with zstd.open(puzzle_path, 'rt', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_puzzles and i >= max_puzzles:
                    break
                
                # Parse puzzle data
                puzzle_rating = int(row['Rating'])
                puzzle_themes = row['Themes'].split()
                
                # Apply filters
                if min_rating and puzzle_rating < min_rating:
                    continue
                if max_rating and puzzle_rating > max_rating:
                    continue
                if themes and not any(theme in puzzle_themes for theme in themes):
                    continue
                
                # Parse moves (space-separated UCI moves)
                moves = row['Moves'].split()
                if len(moves) < 2:  # Need at least opponent move + our response
                    continue
                
                self.puzzles.append({
                    'puzzle_id': row['PuzzleId'],
                    'fen': row['FEN'],
                    'moves': moves,
                    'rating': puzzle_rating,
                    'themes': puzzle_themes,
                    'game_url': row['GameUrl']
                })
                
                if (i + 1) % 10000 == 0:
                    print(f"Loaded {i + 1} puzzles...")
        
        print(f"Total puzzles loaded: {len(self.puzzles)}")
        
        # Calculate total position-move pairs
        self.total_pairs = sum(len(p['moves']) - 1 for p in self.puzzles)
        print(f"Total position-move pairs: {self.total_pairs}")
    
    def __len__(self):
        """Return total number of position-move pairs across all puzzles."""
        return self.total_pairs
    
    def __getitem__(self, idx):
        """
        Returns a position-move pair from the puzzle dataset.
        
        The dataset is indexed as a flat sequence of all position-move pairs
        across all puzzles. This means we need to:
        1. Find which puzzle this index belongs to
        2. Find which move within that puzzle
        3. Replay the game to get the board state
        4. Return the encoded position + next move
        """
        # Find the puzzle and move index
        cumulative = 0
        for puzzle in self.puzzles:
            puzzle_pairs = len(puzzle['moves']) - 1
            if idx < cumulative + puzzle_pairs:
                # Found the right puzzle
                move_idx = idx - cumulative
                break
            cumulative += puzzle_pairs
        else:
            raise IndexError(f"Index {idx} out of range")
        
        # Set up the board from FEN
        board = chess.Board(puzzle['fen'])
        
        # Apply moves up to (but not including) the target move
        for i in range(move_idx + 1):  # +1 because first move is opponent's
            move = chess.Move.from_uci(puzzle['moves'][i])
            board.push(move)
        
        # Get the target move
        target_move = puzzle['moves'][move_idx + 1]
        
        # Encode the position
        position_tokens = self.position_tokenizer.encode(board)
        
        # Encode the move
        try:
            move_token = self.move_tokenizer.encode(target_move)
        except ValueError:
            # If move not in vocabulary, skip this sample
            # This shouldn't happen often with proper vocab
            print(f"Warning: Move {target_move} not in vocabulary for puzzle {puzzle['puzzle_id']}")
            move_token = 0  # Use a default
        
        # Get legal moves mask for loss calculation
        legal_moves = list(board.legal_moves)
        legal_moves_mask = torch.zeros(self.move_tokenizer.vocab_size, dtype=torch.float32)
        for move in legal_moves:
            try:
                move_id = self.move_tokenizer.encode(move.uci())
                legal_moves_mask[move_id] = 1.0
            except ValueError:
                continue
        
        return {
            'position': torch.tensor(position_tokens, dtype=torch.long),
            'move': torch.tensor(move_token, dtype=torch.long),
            'is_white': board.turn == chess.WHITE,
            'puzzle_id': puzzle['puzzle_id'],
            'puzzle_rating': puzzle['rating'],
            'legal_moves_mask': legal_moves_mask,
            'themes': puzzle['themes']
        }
    
    def get_puzzle_by_id(self, puzzle_id: str):
        """Get a complete puzzle by its ID."""
        for puzzle in self.puzzles:
            if puzzle['puzzle_id'] == puzzle_id:
                return puzzle
        return None


class LichessPuzzleFullSolutionDataset(Dataset):
    """Alternative puzzle dataset that returns complete puzzles (not individual moves).
    
    Useful for evaluation where you want to test solving the entire puzzle.
    """
    
    def __init__(
        self, 
        puzzle_path: str,
        min_rating: int = None,
        max_rating: int = None,
        themes: list[str] = None,
        max_puzzles: int = None
    ):
        """
        Args:
            puzzle_path: Path to the .csv.zst puzzle file
            min_rating: Minimum puzzle rating to include
            max_rating: Maximum puzzle rating to include
            themes: List of themes to filter by
            max_puzzles: Maximum number of puzzles to load
        """
        self.position_tokenizer = PostionTokenizer()
        self.move_tokenizer = MoveTokenizer()
        self.puzzles = []
        
        print(f"Loading puzzles from {puzzle_path}...")
        
        with zstd.open(puzzle_path, 'rt', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_puzzles and i >= max_puzzles:
                    break
                
                puzzle_rating = int(row['Rating'])
                puzzle_themes = row['Themes'].split()
                
                if min_rating and puzzle_rating < min_rating:
                    continue
                if max_rating and puzzle_rating > max_rating:
                    continue
                if themes and not any(theme in puzzle_themes for theme in themes):
                    continue
                
                moves = row['Moves'].split()
                if len(moves) < 2:
                    continue
                
                self.puzzles.append({
                    'puzzle_id': row['PuzzleId'],
                    'fen': row['FEN'],
                    'moves': moves,
                    'rating': puzzle_rating,
                    'themes': puzzle_themes,
                    'game_url': row['GameUrl']
                })
        
        print(f"Total puzzles loaded: {len(self.puzzles)}")
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        """Returns a complete puzzle with starting position and full solution."""
        puzzle = self.puzzles[idx]
        
        # Starting position
        board = chess.Board(puzzle['fen'])
        position_tokens = self.position_tokenizer.encode(board)
        
        # Encode solution moves
        solution_tokens = []
        for move_uci in puzzle['moves']:
            try:
                move_token = self.move_tokenizer.encode(move_uci)
                solution_tokens.append(move_token)
            except ValueError:
                solution_tokens.append(0)
        
        return {
            'puzzle_id': puzzle['puzzle_id'],
            'position': torch.tensor(position_tokens, dtype=torch.long),
            'solution': torch.tensor(solution_tokens, dtype=torch.long),
            'fen': puzzle['fen'],
            'moves_uci': puzzle['moves'],
            'rating': puzzle['rating'],
            'themes': puzzle['themes'],
            'is_white': board.turn == chess.WHITE,
        }
