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
        num_samples=100_000,
    ):
        """
        Lichess dataset in the simple UCI format."""

        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.vocab_path = vocab_path
        self._load_vocab()
        self.games = self._extract_games()

    def _extract_games(self, only_check_mate=True):
        games = []
        dctx = zstd.ZstdDecompressor()
        with open(self.dataset_path, "rb") as f:
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                pgn = chess.pgn.read_game(text_stream)
                game_count = 0

                while pgn is not None and game_count < self.num_samples:
                    # check if the game ended by a win/loss/draw
                    if pgn.headers.get("Result") not in ["1-0", "0-1", "1/2-1/2"]:
                        pgn = chess.pgn.read_game(text_stream)
                        continue
                    games.append(pgn)
                    game_count += 1

            print(f"Finished extracting {len(games)} games.")
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
        return token_ids, legals_mask
    
    def encode_game(self, game):
        move_list = [move.uci() for move in game.mainline_moves()]
        token_ids = self.encode_moves(move_list)
        return torch.tensor(token_ids, dtype=torch.long)
    
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
        
        return torch.stack(masks)
    
    def decode_game(self, token_ids):
        id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        moves = [id_to_token.get(tid, "<UNK>") for tid in token_ids if tid in id_to_token]
        return " ".join(moves)
    

if __name__ == "__main__":
    import cairosvg
    from PIL import Image


    dataset = LichessSimpleUciDataset(
        dataset_path=root_path.parents[1] / "data/Lichess Rated Games 2017.pgn.zst",
        num_samples=10_000,
    )
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample token IDs: {sample}")

    # Visualize the board and legal moves
    import chess.svg

    # Get a game from the dataset
    game_idx = 0
    token_ids, legals_mask = dataset[game_idx]
    game = dataset.games[game_idx]

    # Get the board at a specific move position
    move_position = 5  # Change this to see different positions
    board = game.board()

    # Apply moves up to the desired position
    moves = list(game.mainline_moves())
    for i in range(min(move_position, len(moves))):
        board.push(moves[i])

    # Get legal moves from the mask instead of recalculating
    mask_at_position = legals_mask[move_position] if move_position < len(legals_mask) else None
    
    if mask_at_position is not None:
        # Create inverse vocabulary
        idx_to_token = {v: k for k, v in dataset.token_to_id.items()}
        
        # Get legal move tokens from the mask
        legal_move_tokens = [idx_to_token[i] for i in range(len(mask_at_position)) if mask_at_position[i] == 1.0]
        
        # Convert tokens to chess.Move objects for visualization
        legal_moves = [chess.Move.from_uci(token) for token in legal_move_tokens if token not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']]
        legal_squares = [move.to_square for move in legal_moves]

        # Create SVG visualization with legal move squares highlighted
        svg = chess.svg.board(
            board,
            squares=chess.SquareSet(legal_squares),
            size=400
        )

        # Save to file
        output_path = root_path.parents[1] / "board_visualization.svg"
        with open(output_path, "w") as f:
            f.write(svg)

        print(f"\nBoard visualization saved to: {output_path}")
        print(f"Position after {move_position} moves")
        print(f"Number of legal moves: {len(legal_moves)}")
        print(f"Legal moves from mask: {legal_move_tokens}")
        
        # Display the board in terminal
        print(f"\nBoard position after {move_position} moves:")
        print(board)
        print(f"\nFEN: {board.fen()}")

    # Create a GIF of 5 moves with legal moves highlighted using dataset masks
    num_moves_gif = 5
    board_gif = game.board()
    moves_list = list(game.mainline_moves())
    
    frames = []
    idx_to_token = {v: k for k, v in dataset.token_to_id.items()}
    
    for i in range(min(num_moves_gif + 1, len(moves_list) + 1)):
        # Get legal moves from the precomputed mask
        if i < len(legals_mask):
            mask_current = legals_mask[i]
            legal_move_tokens_current = [idx_to_token[j] for j in range(len(mask_current)) if mask_current[j] == 1.0]
            legal_moves_current = [chess.Move.from_uci(token) for token in legal_move_tokens_current if token not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']]
            legal_squares_current = [move.to_square for move in legal_moves_current]
        else:
            legal_squares_current = []
        
        # Create SVG
        svg_content = chess.svg.board(
            board_gif,
            squares=chess.SquareSet(legal_squares_current),
            size=400
        )
        
        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
        
        # Open as PIL Image
        img = Image.open(io.BytesIO(png_data))
        frames.append(img)
        
        # Apply next move if available
        if i < len(moves_list):
            board_gif.push(moves_list[i])
    
    # Save as GIF
    gif_path = root_path.parents[1] / "board_animation.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000,  # 1 second per frame
        loop=0
    )
    
    print(f"\nGIF animation saved to: {gif_path}")
print(f"Created {len(frames)} frames showing positions 0-{num_moves_gif}")
