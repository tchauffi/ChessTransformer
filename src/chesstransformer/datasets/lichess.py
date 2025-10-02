import zstandard as zstd
import chess.pgn
import io

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
                games.append(pgn)
                game_count += 1
                batch_count += 1

                if batch_count >= batch_size:
                    batch_count = 0
                    print(f"Extracted {game_count}/{num_games} games...")

                pgn = chess.pgn.read_game(text_stream)

    print(f"Finished extracting {len(games)} games.")
    return games

if __name__ == "__main__":
    from pathlib import Path
    filepath = Path("./data/Lichess Rated Games 2017.pgn.zst")
    print(f"Extracting games from {filepath}...")
    games = extract_games_optimized(filepath, num_games=10000)
    print(f"Total games extracted: {len(games)}")

    # print size in MB
    total_size = sum(len(str(game)) for game in games)
    print(f"Total size of extracted games: {total_size / (1024 * 1024):.2f} MB")
    print(f"Average size per game: {total_size / len(games):.2f} bytes")