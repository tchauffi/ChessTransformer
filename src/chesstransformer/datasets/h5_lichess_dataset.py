import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import chess
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer
from chesstransformer.models.tokenizer.move_tokenizer import MoveTokenizer

class HDF5ChessDataset(Dataset):
    def __init__(self, hdf5_path: str, cache_size: int = 500, min_elo: int = None, max_elo: int = None):
        """
        Efficient dataset that loads games as UCI sequences and samples random positions.
        
        Each dataset index corresponds to one game. When a sample is requested, it:
        1. Uses the index to select a game
        2. Randomly samples a position uniformly within that game
        3. Replays the game to that position to get the board state
        4. Returns the encoded board state + the next move
        
        Args:
            hdf5_path: Path to HDF5 file containing games
            cache_size: Number of games to cache in memory
            min_elo: Minimum average ELO for filtering games (optional)
            max_elo: Maximum average ELO for filtering games (optional)
        """
        self.hdf5_path = hdf5_path
        self.cache_size = cache_size
        self.game_cache = {}  # Cache for game move sequences
        self.position_tokenizer = PostionTokenizer()
        self.move_tokenizer = MoveTokenizer()
        
        # Load metadata and filter games by ELO
        with h5py.File(hdf5_path, 'r') as f:
            self.num_games = f['moves'].shape[0]
            num_moves = f['num_moves'][:]
            white_elos = f['white_elo'][:]
            black_elos = f['black_elo'][:]
            
            # Filter games by ELO if specified
            valid_games = np.ones(self.num_games, dtype=bool)
            if min_elo is not None or max_elo is not None:
                avg_elos = (white_elos + black_elos) / 2
                if min_elo is not None:
                    valid_games &= (avg_elos >= min_elo)
                if max_elo is not None:
                    valid_games &= (avg_elos <= max_elo)
            
            # Store valid game indices and their move counts
            self.valid_game_indices = np.where(valid_games)[0]
            self.num_moves_per_game = num_moves[valid_games]
            
            total_positions = int(self.num_moves_per_game.sum())
            print(f"Loaded dataset: {len(self.valid_game_indices)} games with {total_positions} total positions")
            if min_elo or max_elo:
                print(f"Filtered by ELO: {min_elo or 0} - {max_elo or 'inf'}")
    
    def __len__(self):
        """Return the number of games in the dataset."""
        return len(self.valid_game_indices)
    
    def _get_game_moves(self, game_idx: int) -> np.ndarray:
        """Load and cache a game's move sequence."""
        if game_idx in self.game_cache:
            return self.game_cache[game_idx]
        
        # Read from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            moves = f['moves'][game_idx]
        
        # Update cache (simple LRU-like behavior)
        if len(self.game_cache) >= self.cache_size:
            # Remove oldest item
            self.game_cache.pop(next(iter(self.game_cache)))
        
        self.game_cache[game_idx] = moves
        return moves
    
    def __getitem__(self, idx):
        """
        Returns a position-move pair by:
        1. Using idx to select a game
        2. Uniformly sampling a random position within that game
        3. Replaying the game to get the board state
        4. Returning encoded board + next move
        """
        # Map dataset index to actual game index
        actual_game_idx = self.valid_game_indices[idx]
        
        # Load game moves
        game_moves = self._get_game_moves(actual_game_idx)
        num_moves = self.num_moves_per_game[idx]
        
        # Uniformly sample a position within the game (exclude last move - no next move)
        max_move_idx = num_moves - 1
        if max_move_idx <= 0:
            # Edge case: game with only one move, use position 0
            move_idx = 0
        else:
            # Use random sampling for uniform distribution
            move_idx = np.random.randint(0, max_move_idx)
        
        # Replay game to position
        board = chess.Board()
        for i in range(move_idx):
            # Decode move token to UCI string
            move_token = int(game_moves[i])
            move_uci = self._decode_move_token(move_token)
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
            except (ValueError, AssertionError):
                # If move is invalid, skip to next position
                # This shouldn't happen with properly encoded data
                pass
        
        # Encode current position
        position = self.position_tokenizer.encode(board)
        position_tensor = torch.tensor(position, dtype=torch.long)

        legal_moves = list(board.legal_moves)
        legal_moves_tokens = torch.zeros(len(self.move_tokenizer.vocab), dtype=torch.bool)
        for move in legal_moves:
            if move.uci() in self.move_tokenizer.vocab:
                token_id = self.move_tokenizer.vocab[move.uci()]
                legal_moves_tokens[token_id] = True
        legal_moves_tensor = legal_moves_tokens
        
        # Get next move
        next_move_token = torch.tensor(int(game_moves[move_idx]), dtype=torch.long)
        
        # Get turn indicator
        is_white = board.turn  # True for white, False for black
        
        # Get castling rights as a single integer (0-15)
        # Bit 0: white kingside, Bit 1: white queenside, Bit 2: black kingside, Bit 3: black queenside
        castling_rights = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_rights |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_rights |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_rights |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_rights |= 8
        
        # Get en passant file (0-7 for a-h, 8 for none)
        if board.has_legal_en_passant():
            en_passant_file = chess.square_file(board.ep_square)
        else:
            en_passant_file = 8  # No en passant
        
        # Get halfmove clock (for 50-move rule)
        halfmove_clock = board.halfmove_clock
        
        # Get game metadata
        with h5py.File(self.hdf5_path, 'r') as f:
            white_elo = int(f['white_elo'][actual_game_idx])
            black_elo = int(f['black_elo'][actual_game_idx])
            result = int(f['result'][actual_game_idx])
        
        return {
            'position': position_tensor,
            'move': next_move_token,
            'is_white': is_white,
            'castling_rights': castling_rights,
            'en_passant_file': en_passant_file,
            'halfmove_clock': halfmove_clock,
            'game_id': actual_game_idx,
            'move_number': move_idx,
            'white_elo': white_elo,
            'black_elo': black_elo,
            'result': result,
            'legal_moves_mask': legal_moves_tensor
        }
    
    def _decode_move_token(self, token: int) -> str:
        """Decode a move token back to UCI string."""
        return self.move_tokenizer.decode(token)
    
if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python h5_lichess_dataset.py <path_to_h5_file>")
        sys.exit(1)
    
    dataset = HDF5ChessDataset(sys.argv[1], cache_size=100)
    print(f"Dataset length: {len(dataset)}")
    
    # Test a few samples
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Position shape: {sample['position'].shape}")
        print(f"  Move token: {sample['move'].item()}")
        print(f"  Turn: {'White' if sample['is_white'] else 'Black'}")
        print(f"  ELOs: {sample['white_elo']} vs {sample['black_elo']}")
        print(f"  Move number: {sample['move_number']} over {len(dataset._get_game_moves(sample['game_id']))}")
        print(f" Nb legal moves: {sample['legal_moves_mask'].sum().item()}")