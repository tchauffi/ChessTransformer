import h5py
import chess
import chess.pgn
import zstandard as zstd
import numpy as np
from tqdm import tqdm
from chesstransformer.models.tokenizer.move_tokenizer import MoveTokenizer
import io

class PGNtoHDF5Converter:
    def __init__(self, output_path: str, chunk_size: int = 1000):
        """
        Convert PGN files to HDF5 format storing complete games as UCI move sequences.
        
        This new approach stores games as tokenized move sequences instead of pre-computing
        all board positions, resulting in much more efficient storage.
        
        Args:
            output_path: Path to save the HDF5 file
            chunk_size: Number of games to accumulate before writing
        """
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.move_tokenizer = MoveTokenizer()
        
    def convert_pgn_zst(self, pgn_path: str, max_games: int = None):
        """Convert a .pgn.zst file to HDF5 format storing game sequences."""
        
        # Temporary storage for batching
        games_batch = []
        white_elo_batch = []
        black_elo_batch = []
        result_batch = []
        num_moves_batch = []
        
        game_count = 0
        total_positions = 0
        
        with open(pgn_path, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                with h5py.File(self.output_path, 'w') as hdf5_file:
                    # Create variable-length dataset for move sequences
                    dt = h5py.vlen_dtype(np.dtype('int16'))
                    moves_ds = hdf5_file.create_dataset(
                        'moves',
                        shape=(0,),
                        maxshape=(None,),
                        dtype=dt,
                        compression='gzip',
                        compression_opts=9,
                        chunks=(self.chunk_size,)
                    )
                    
                    # Create datasets for ELO ratings
                    white_elo_ds = hdf5_file.create_dataset(
                        'white_elo',
                        shape=(0,),
                        maxshape=(None,),
                        dtype='int16',
                        compression='gzip',
                        compression_opts=9,
                        chunks=(self.chunk_size,)
                    )
                    
                    black_elo_ds = hdf5_file.create_dataset(
                        'black_elo',
                        shape=(0,),
                        maxshape=(None,),
                        dtype='int16',
                        compression='gzip',
                        compression_opts=9,
                        chunks=(self.chunk_size,)
                    )
                    
                    # Create dataset for game results (0=draw, 1=white win, 2=black win)
                    result_ds = hdf5_file.create_dataset(
                        'result',
                        shape=(0,),
                        maxshape=(None,),
                        dtype='int8',
                        compression='gzip',
                        compression_opts=9,
                        chunks=(self.chunk_size,)
                    )
                    
                    # Create dataset for number of moves per game (for quick lookup)
                    num_moves_ds = hdf5_file.create_dataset(
                        'num_moves',
                        shape=(0,),
                        maxshape=(None,),
                        dtype='int16',
                        compression='gzip',
                        compression_opts=9,
                        chunks=(self.chunk_size,)
                    )
                    
                    pbar = tqdm(desc="Processing games", unit="games")
                    
                    while True:
                        pgn = chess.pgn.read_game(text_stream)
                        if pgn is None:
                            break
                        
                        # Skip non-standard variants
                        variant = pgn.headers.get("Variant", "Standard")
                        if variant != "Standard":
                            continue
                        
                        # Extract ELO ratings (default to 1500 if missing)
                        try:
                            white_elo = int(pgn.headers.get("WhiteElo", "1500"))
                        except ValueError:
                            white_elo = 1500
                            
                        try:
                            black_elo = int(pgn.headers.get("BlackElo", "1500"))
                        except ValueError:
                            black_elo = 1500
                        
                        # Extract result (0=draw, 1=white win, 2=black win)
                        result_str = pgn.headers.get("Result", "1/2-1/2")
                        if result_str == "1/2-1/2":
                            result = 0
                        elif result_str == "1-0":
                            result = 1
                        elif result_str == "0-1":
                            result = 2
                        else:
                            continue  # Skip games with unknown results
                        
                        # Tokenize moves
                        move_tokens = []
                        for move in pgn.mainline_moves():
                            try:
                                token = self.move_tokenizer.encode(move.uci())
                                move_tokens.append(token)
                            except ValueError:
                                # Skip game if any move cannot be encoded
                                break
                        
                        # Only store games with at least 10 moves (20 plies)
                        if len(move_tokens) < 10:
                            continue
                        
                        # Store game data
                        games_batch.append(np.array(move_tokens, dtype=np.int16))
                        white_elo_batch.append(white_elo)
                        black_elo_batch.append(black_elo)
                        result_batch.append(result)
                        num_moves_batch.append(len(move_tokens))
                        total_positions += len(move_tokens)
                        
                        # Write batch if full
                        if len(games_batch) >= self.chunk_size:
                            self._write_batch(
                                hdf5_file,
                                games_batch,
                                white_elo_batch,
                                black_elo_batch,
                                result_batch,
                                num_moves_batch
                            )
                            games_batch = []
                            white_elo_batch = []
                            black_elo_batch = []
                            result_batch = []
                            num_moves_batch = []
                        
                        game_count += 1
                        pbar.update(1)
                        
                        if max_games and game_count >= max_games:
                            break
                    
                    # Write remaining batch
                    if games_batch:
                        self._write_batch(
                            hdf5_file,
                            games_batch,
                            white_elo_batch,
                            black_elo_batch,
                            result_batch,
                            num_moves_batch
                        )
                    
                    # Store metadata
                    hdf5_file.attrs['num_games'] = game_count
                    hdf5_file.attrs['total_positions'] = total_positions
                    hdf5_file.attrs['move_vocab_size'] = 1968
                    
                    pbar.close()
        
        print(f"Conversion complete!")
        print(f"Games: {game_count}")
        print(f"Total positions: {total_positions}")
        print(f"Average moves per game: {total_positions / game_count:.1f}")
        print(f"Output: {self.output_path}")
    
    def _write_batch(self, hdf5_file, games, white_elos, black_elos, results, num_moves):
        """Write a batch of games to HDF5 file."""
        batch_size = len(games)
        
        # Resize datasets
        old_size = hdf5_file['moves'].shape[0]
        new_size = old_size + batch_size
        
        hdf5_file['moves'].resize((new_size,))
        hdf5_file['white_elo'].resize((new_size,))
        hdf5_file['black_elo'].resize((new_size,))
        hdf5_file['result'].resize((new_size,))
        hdf5_file['num_moves'].resize((new_size,))
        
        # Write data
        for i, game_moves in enumerate(games):
            hdf5_file['moves'][old_size + i] = game_moves
        
        hdf5_file['white_elo'][old_size:new_size] = np.array(white_elos, dtype=np.int16)
        hdf5_file['black_elo'][old_size:new_size] = np.array(black_elos, dtype=np.int16)
        hdf5_file['result'][old_size:new_size] = np.array(results, dtype=np.int8)
        hdf5_file['num_moves'][old_size:new_size] = np.array(num_moves, dtype=np.int16)


# Usage script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PGN.zst to HDF5')
    parser.add_argument('--input', type=str, help='Input .pgn.zst file')
    parser.add_argument('--output', type=str, help='Output .h5 file')
    parser.add_argument('--max-games', type=int, default=None, help='Maximum games to process')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Batch size for writing')
    
    args = parser.parse_args()
    
    converter = PGNtoHDF5Converter(args.output, chunk_size=args.chunk_size)
    converter.convert_pgn_zst(args.input, max_games=args.max_games)