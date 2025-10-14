import h5py
import chess
import chess.pgn
import zstandard as zstd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from chesstransformer.models.tokenizer.move_tokenizer import MoveTokenizer
import io

_worker_tokenizer = None

def _worker_init():
    global _worker_tokenizer
    _worker_tokenizer = MoveTokenizer()

def _process_game_entry(entry):
    global _worker_tokenizer
    tokens = []
    for uci in entry["moves"]:
        try:
            tokens.append(_worker_tokenizer.encode(uci))
        except ValueError:
            return None
    if len(tokens) < 10:
        return None
    token_array = np.array(tokens, dtype=np.int16)
    return token_array, entry["white_elo"], entry["black_elo"], entry["result"], len(tokens)

class PGNtoHDF5Converter:
    def __init__(self, output_path: str, chunk_size: int = 1000, num_workers: int = None):
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
        default_workers = (cpu_count() or 1) - 1
        self.num_workers = max(1, num_workers if num_workers is not None else max(1, default_workers))
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
                    use_parallel = self.num_workers > 1
                    pool = Pool(processes=self.num_workers, initializer=_worker_init) if use_parallel else None
                    pending_entries = []
                    games_batch, white_elo_batch, black_elo_batch, result_batch, num_moves_batch = [], [], [], [], []
                    reached_limit = False
                    try:
                        while True:
                            pgn = chess.pgn.read_game(text_stream)
                            if pgn is None or reached_limit:
                                break
                            entry = self._prepare_game_entry(pgn)
                            if entry is None:
                                continue
                            if use_parallel:
                                pending_entries.append(entry)
                                if len(pending_entries) >= self.chunk_size:
                                    game_count, total_positions, reached_limit = self._process_entries_parallel(
                                        pool, pending_entries, hdf5_file, pbar, game_count, total_positions, max_games
                                    )
                                    pending_entries = []
                            else:
                                move_tokens = self._encode_moves_sequential(entry["moves"])
                                if move_tokens is None:
                                    continue
                                games_batch.append(np.array(move_tokens, dtype=np.int16))
                                white_elo_batch.append(entry["white_elo"])
                                black_elo_batch.append(entry["black_elo"])
                                result_batch.append(entry["result"])
                                num_moves_batch.append(len(move_tokens))
                                total_positions += len(move_tokens)
                                game_count += 1
                                pbar.update(1)
                                if len(games_batch) >= self.chunk_size:
                                    self._write_batch(
                                        hdf5_file,
                                        games_batch,
                                        white_elo_batch,
                                        black_elo_batch,
                                        result_batch,
                                        num_moves_batch
                                    )
                                    games_batch, white_elo_batch, black_elo_batch, result_batch, num_moves_batch = [], [], [], [], []
                                if max_games and game_count >= max_games:
                                    reached_limit = True
                        if use_parallel and pending_entries and not (max_games and game_count >= max_games):
                            game_count, total_positions, _ = self._process_entries_parallel(
                                pool, pending_entries, hdf5_file, pbar, game_count, total_positions, max_games
                            )
                        if not use_parallel and games_batch:
                            self._write_batch(
                                hdf5_file,
                                games_batch,
                                white_elo_batch,
                                black_elo_batch,
                                result_batch,
                                num_moves_batch
                            )
                    finally:
                        if pool is not None:
                            pool.close()
                            pool.join()
                    pbar.close()
        
        print(f"Conversion complete!")
        print(f"Games: {game_count}")
        print(f"Total positions: {total_positions}")
        average_moves = (total_positions / game_count) if game_count else 0.0
        print(f"Average moves per game: {average_moves:.1f}")
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

    def _prepare_game_entry(self, pgn):
        variant = pgn.headers.get("Variant", "Standard")
        if variant != "Standard":
            return None
        try:
            white_elo = int(pgn.headers.get("WhiteElo", "1500"))
        except ValueError:
            white_elo = 1500
        try:
            black_elo = int(pgn.headers.get("BlackElo", "1500"))
        except ValueError:
            black_elo = 1500
        result_str = pgn.headers.get("Result", "1/2-1/2")
        if result_str == "1/2-1/2":
            result = 0
        elif result_str == "1-0":
            result = 1
        elif result_str == "0-1":
            result = 2
        else:
            return None
        moves = [move.uci() for move in pgn.mainline_moves()]
        if len(moves) < 10:
            return None
        return {"moves": moves, "white_elo": white_elo, "black_elo": black_elo, "result": result}

    def _encode_moves_sequential(self, moves):
        move_tokens = []
        for uci in moves:
            try:
                move_tokens.append(self.move_tokenizer.encode(uci))
            except ValueError:
                return None
        return move_tokens if len(move_tokens) >= 10 else None

    def _process_entries_parallel(self, pool, entries, hdf5_file, pbar, game_count, total_positions, max_games):
        results = pool.map(_process_game_entry, entries, chunksize=max(1, len(entries) // (self.num_workers * 2 or 1)))
        valid = [res for res in results if res is not None]
        if not valid:
            return game_count, total_positions, False
        reached_limit = False
        if max_games is not None:
            remaining = max_games - game_count
            if remaining <= 0:
                return game_count, total_positions, True
            if len(valid) > remaining:
                valid = valid[:remaining]
                reached_limit = True
        games = [item[0] for item in valid]
        white_elos = [item[1] for item in valid]
        black_elos = [item[2] for item in valid]
        results_meta = [item[3] for item in valid]
        num_moves = [item[4] for item in valid]
        self._write_batch(hdf5_file, games, white_elos, black_elos, results_meta, num_moves)
        pbar.update(len(valid))
        game_count += len(valid)
        total_positions += sum(num_moves)
        return game_count, total_positions, reached_limit


# Usage script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PGN.zst to HDF5')
    parser.add_argument('--input', type=str, help='Input .pgn.zst file')
    parser.add_argument('--output', type=str, help='Output .h5 file')
    parser.add_argument('--max-games', type=int, default=None, help='Maximum games to process')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Batch size for writing')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of worker processes (default: cpu count - 1)')
    
    args = parser.parse_args()
    
    converter = PGNtoHDF5Converter(args.output, chunk_size=args.chunk_size, num_workers=args.num_workers)
    converter.convert_pgn_zst(args.input, max_games=args.max_games)