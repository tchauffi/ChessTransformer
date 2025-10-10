from pathlib import Path
import json
from torch.utils.data import Dataset
import zstandard as zstd
import chess.pgn
import io
import torch
import chess


root_path = Path(__file__).parents[1]
print(f"Root path: {root_path}")


class LichessSimpleUciDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        vocab_path=root_path / "data/tokenizer_models/uci_vocab.json",
        context_length=512,
        num_samples=100_000,
    ):
        """
        Lichess dataset in the simple UCI format."""

        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.vocab_path = vocab_path
        self.context_length = context_length
        self._load_vocab()
        self.games = self._extract_games()

    def _extract_games(self, only_check_mate=True):
        games = []
        dctx = zstd.ZstdDecompressor()
        with open(self.dataset_path, "rb") as f:
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                game_count = 0

                while game_count < self.num_samples:
                    pgn = chess.pgn.read_game(text_stream)
                    if pgn is None:
                        break
                    
                    # Check if the game ended by a win/loss/draw
                    if pgn.headers.get("Result") not in ["1-0", "0-1", "1/2-1/2"]:
                        continue
                    
                    # Skip Chess960 games - only use classical chess
                    variant = pgn.headers.get("Variant", "Standard")
                    if variant != "Standard":
                        continue
                    
                    games.append(pgn)
                    game_count += 1
                    
                    if game_count % 10000 == 0:
                        print(f"Extracted {game_count}/{self.num_samples} games...", end="\r")

            print(f"\nFinished extracting {len(games)} games.")
            return games
        
    def _load_vocab(self):
        with open(self.vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}

    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, idx):
        game = self.games[idx]
        token_ids = self.encode_game(game)
        legals_mask = self.get_legal_moves_mask(game)
        target_ids = token_ids[1:] + [-100] # Shifted targets with -100 for padding
        return {"input_ids": torch.tensor(token_ids, dtype=torch.long),
                "target_ids": torch.tensor(target_ids, dtype=torch.long),
                "legal_moves_mask": legals_mask
                }
    
    
    def encode_game(self, game):
        move_list = [move.uci() for move in game.mainline_moves()]
        move_list = ["<START>"] + move_list + ["<END>"]
        token_ids = self.encode_moves(move_list)
        return token_ids
    
    def encode_moves(self, moves):
        token_ids = []
        for move in moves:
            if move in self.token_to_id:
                token_ids.append(self.token_to_id[move])
            else:
                token_ids.append(self.token_to_id["<UNK>"])
        
        return token_ids
    
    def get_legal_moves_mask(self, game):
        """
        For each move in the game, generate a mask of legal moves at that position.
        Returns a list of masks, where each mask is a tensor of shape (vocab_size,)
        with 1.0 for legal moves and 0.0 for illegal moves.
        """
        board = game.board()
        masks = []
        
        for move in game.mainline_moves():
            # Get all legal moves at current position
            legal_moves = [m.uci() for m in board.legal_moves]
            
            # Create mask for all tokens in vocabulary
            mask = torch.zeros(len(self.vocab), dtype=torch.float32)
            
            # Set 1.0 for legal moves
            for legal_move in legal_moves:
                if legal_move in self.token_to_id:
                    mask[self.token_to_id[legal_move]] = 1.0
            
            masks.append(mask)
            
            # Apply the move to the board for next iteration
            board.push(move)

        # Add mask for <START> and <END> tokens
        start_end_mask = torch.zeros(len(self.vocab), dtype=torch.float32)
        start_end_mask[self.token_to_id["<START>"]] = 1.0
        start_end_mask[self.token_to_id["<END>"]] = 1.0
        masks = [start_end_mask] + masks + [start_end_mask]
        
        return torch.stack(masks)
    
    def decode_game(self, token_ids):
        id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        moves = [id_to_token.get(tid, "<UNK>") for tid in token_ids if tid in id_to_token]
        return " ".join(moves)
