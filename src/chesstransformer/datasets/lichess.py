from torch.utils.data import Dataset
import zstandard as zstd
import chess.pgn
import io
import torch


def extract_games_optimized(filepath, num_games=1000000, batch_size=1000):
    """
    Optimized game extraction with progress tracking.
    Process in batches and show estimated time remaining.

    Args:
        filepath: Path to the compressed PGN file
        num_games: Number of games to extract (default: 1,000,000)
        batch_size: Number of games to process in each batch (default: 1,000)

    Returns:
        List of extracted games
    """
    games = []
    dctx = zstd.ZstdDecompressor()
    with open(filepath, "rb") as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            pgn = chess.pgn.read_game(text_stream)
            game_count = 0
            batch_count = 0

            while pgn is not None and game_count < num_games:
                # check if the game ended by a win/loss/draw
                if pgn.headers.get("Result") not in ["1-0", "0-1", "1/2-1/2"]:
                    pgn = chess.pgn.read_game(text_stream)
                    continue
                games.append(pgn)
                game_count += 1
                batch_count += 1

                if batch_count >= batch_size:
                    batch_count = 0
                    print(f"Extracted {game_count}/{num_games} games...")

                pgn = chess.pgn.read_game(text_stream)

    print(f"Finished extracting {len(games)} games.")
    return games


def encode_game_moves(game, notation="uci", seprator=" <STEP> "):
    """
    Encode the moves of a single game using the provided tokenizer.

    Args:
        game: A chess.pgn.Game object
        notation: Notation format for moves ("uci" or "san", default: "uci")
        seprator: String to separate moves (default: " <STEP> ")

    Returns:
        List of moves as a single formatted string
    """
    if notation == "uci":
        move_sequence = seprator.join([move.uci() for move in game.mainline_moves()])
    elif notation == "san":
        board = game.board()
        move_sequence = seprator.join(
            [
                board.san(move)
                for move in game.mainline_moves()
                if not board.push(move)  # Push the move to update the board state
            ]
        )
    else:
        raise ValueError("Unsupported notation format. Use 'uci' or 'san'.")

    ending_condition = seprator + {
        "1-0": "<1-0>",
        "0-1": "<0-1>",
        "1/2-1/2": "<1/2-1/2>",
    }.get(game.headers.get("Result", ""), "")
    if ending_condition:
        move_sequence += ending_condition

    return move_sequence


class LichessDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        tokenizer,
        notation="uci",
        context_length=512,
        nb_games=100000,
    ):
        """
        Lichess dataset for training chess move prediction models.

        Args:
            dataset_path: Path to the lichess dataset file (PGN format)
            tokenizer: Tokenizer object to convert moves to token IDs
            notation: Notation format for moves ("uci" or "san", default: "uci")
            context_length: Maximum sequence length for model input (default: 512)
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.notation = notation

        print(f"Loading dataset from {dataset_path}...")
        self.games = extract_games_optimized(dataset_path, num_games=nb_games)
        print(f"Total games loaded: {len(self.games)}")

        self.encoded_games = []
        # encode and stack games together
        for idx, game in enumerate(self.games):
            encoded_moves = encode_game_moves(game, notation=self.notation)
            token_ids = self.tokenizer.encode(encoded_moves).ids

            self.encoded_games.append(token_ids)

        # flatten the list of token ids
        self.encoded_games = [
            token_id for game in self.encoded_games for token_id in game
        ]

    def __len__(self):
        return len(self.encoded_games) // self.context_length

    def __getitem__(self, idx):
        """
        Get a sequence of token IDs for training.

        Args:
            idx: Index of the game to retrieve
        Returns:
            input, target: Tuple of input and target token ID lists
        """
        start_idx = idx * self.context_length
        end_idx = start_idx + self.context_length + 1  # +1 for target shift

        if end_idx > len(self.encoded_games):
            raise IndexError("Index out of range for the dataset.")

        input_ids = self.encoded_games[start_idx : end_idx - 1]
        target_ids = self.encoded_games[start_idx + 1 : end_idx]

        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)

        return input_tensor, target_tensor


if __name__ == "__main__":
    from pathlib import Path
    import argparse
    from chesstransformer.models.tokenizer import create_bpe_tokenizer

    def arg_parser():
        parser = argparse.ArgumentParser(description="Test LichessDataset")
        parser.add_argument(
            "--dataset_path",
            type=str,
            default="data/Lichess Rated Games 2017.pgn.zst",
            help="Path to the lichess dataset file (PGN format)",
        )
        parser.add_argument(
            "--num_games",
            type=int,
            default=10000,
            help="Number of games to load for testing (default: 1000)",
        )
        parser.add_argument(
            "--vocab_path",
            type=str,
            default="src/chesstransformer/data/tokenizer_models/bpe_tokenizer_vocab2000.json",
            help="Path to the trained tokenizer file",
        )
        parser.add_argument(
            "--context_length",
            type=int,
            default=512,
            help="Context length for sequences (default: 512)",
        )
        parser.add_argument(
            "--notation",
            type=str,
            choices=["uci", "san"],
            default="uci",
            help="Notation format for moves (default: uci)",
        )
        args = parser.parse_args()
        return args

    args = arg_parser()
    dataset_path = Path(args.dataset_path)
    num_games = args.num_games
    vocab_path = Path(args.vocab_path)
    context_length = args.context_length
    notation = args.notation

    import tokenizers

    tokenizer = tokenizers.Tokenizer.from_file(str(vocab_path))
    print(f"Loading Lichess dataset from {dataset_path}...")
    dataset = LichessDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        notation=notation,
        context_length=context_length,
        nb_games=num_games,
    )

    print(f"Dataset loaded with {len(dataset)} games.")
    input_ids, target_ids = dataset[0]
    print(f"Sample input IDs: {input_ids}")
    print(f"Sample target IDs: {target_ids}")
    print(f"Encoded and stacked {len(dataset.encoded_games)} token IDs.")

    print(f"Decoded sample moves: {tokenizer.decode(input_ids)}")
