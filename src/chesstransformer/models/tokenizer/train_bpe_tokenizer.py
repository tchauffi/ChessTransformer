from pathlib import Path
import argparse

from chesstransformer.models.tokenizer import create_bpe_tokenizer


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer for chess moves."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=2000,
        help="Size of the vocabulary (default: 2000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the trained tokenizer",
        required=True,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the lichess dataset file (PGN format)",
        required=True,
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100000,
        help="Number of games to sample for training (default: 100000)",
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


def main():
    args = arg_parser()
    vocab_size = args.vocab_size
    output_dir = Path(args.output_dir)
    dataset_path = Path(args.dataset_path)
    num_samples = args.num_samples
    notation = args.notation

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating BPE tokenizer with vocab size {vocab_size}...")
    tokenizer, trainer = create_bpe_tokenizer(vocab_size=vocab_size)

    print(f"Loading dataset from {dataset_path}...")
    from chesstransformer.datasets.lichess import extract_games_optimized

    games = extract_games_optimized(dataset_path, num_games=num_samples)

    print(f"Extracting moves in {notation} notation...")
    print("Using <STEP> token to separate moves...")
    if notation == "uci":
        move_sequences = (
            "<STEP>".join([move.uci() for move in game.mainline_moves()])
            for game in games
        )
    elif notation == "san":
        move_sequences = (
            "<STEP>".join(
                [
                    (lambda board, move: (board.san(move), board.push(move))[0])(
                        board, move
                    )
                    for board in [game.board()]
                    for move in game.mainline_moves()
                ]
            )
            for game in games
        )
    else:
        raise ValueError(f"Unsupported notation: {notation}")

    print(f"Training tokenizer on {num_samples} games...")
    tokenizer.train_from_iterator(move_sequences, trainer=trainer)

    tokenizer_path = output_dir / f"bpe_tokenizer_vocab{vocab_size}.json"
    print(f"Saving tokenizer to {tokenizer_path}...")
    tokenizer.save(str(tokenizer_path))

    print("Tokenizer training complete.")


if __name__ == "__main__":
    main()
