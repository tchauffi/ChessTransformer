import h5py
import chess
import chess.pgn
import zstandard as zstd
import numpy as np
from tqdm import tqdm
from chesstransformer.models.tokenizer.move_tokenizer import MoveTokenizer
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer
import io

class PGNtoHDF5Converter:
    def __init__(self, output_path: str, chunk_size: int = 10000):
        """
        Convert PGN files to HDF5 format for efficient deep learning.
        
        Args:
            output_path: Path to save the HDF5 file
            chunk_size: Number of positions to accumulate before writing
        """
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.position_tokenizer = PostionTokenizer()
        self.move_tokenizer = MoveTokenizer()
        
    def convert_pgn_zst(self, pgn_path: str, max_games: int = None):
        """Convert a .pgn.zst file to HDF5 format."""
        
        # Temporary storage for batching
        positions_batch = []
        moves_batch = []
        is_white_batch = []
        game_ids_batch = []
        
        game_count = 0
        position_count = 0
        
        with open(pgn_path, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                with h5py.File(self.output_path, 'w') as hdf5_file:
                    # Create resizable datasets
                    positions_ds = hdf5_file.create_dataset(
                        'positions',
                        shape=(0, 64),
                        maxshape=(None, 64),
                        dtype='int8',
                        compression='gzip',
                        compression_opts=9,
                        chunks=(self.chunk_size, 64)
                    )
                    
                    moves_ds = hdf5_file.create_dataset(
                        'moves',
                        shape=(0,),
                        maxshape=(None,),
                        dtype='int16',
                        compression='gzip',
                        compression_opts=9,
                        chunks=(self.chunk_size,)
                    )
                    
                    is_white_ds = hdf5_file.create_dataset(
                        'is_white',
                        shape=(0,),
                        maxshape=(None,),
                        dtype='bool',
                        compression='gzip',
                        compression_opts=9,
                        chunks=(self.chunk_size,)
                    )
                    
                    game_ids_ds = hdf5_file.create_dataset(
                        'game_ids',
                        shape=(0,),
                        maxshape=(None,),
                        dtype='int32',
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
                        
                        # Process game
                        board = pgn.board()
                        for move in pgn.mainline_moves():
                            # Encode current position
                            position_tokens = self.position_tokenizer.encode(board)
                            
                            # Encode move
                            move_token = self.move_tokenizer.encode(move.uci())
                            
                            # Store data
                            positions_batch.append(position_tokens)
                            moves_batch.append(move_token)
                            is_white_batch.append(board.turn)  # True = white, False = black
                            game_ids_batch.append(game_count)
                            
                            position_count += 1
                            
                            # Write batch if full
                            if len(positions_batch) >= self.chunk_size:
                                self._write_batch(
                                    hdf5_file,
                                    positions_batch,
                                    moves_batch,
                                    is_white_batch,
                                    game_ids_batch
                                )
                                positions_batch = []
                                moves_batch = []
                                is_white_batch = []
                                game_ids_batch = []
                            
                            # Make move
                            board.push(move)
                        
                        game_count += 1
                        pbar.update(1)
                        
                        if max_games and game_count >= max_games:
                            break
                    
                    # Write remaining batch
                    if positions_batch:
                        self._write_batch(
                            hdf5_file,
                            positions_batch,
                            moves_batch,
                            is_white_batch,
                            game_ids_batch
                        )
                    
                    # Store metadata
                    hdf5_file.attrs['num_games'] = game_count
                    hdf5_file.attrs['num_positions'] = position_count
                    hdf5_file.attrs['vocab_size'] = 13
                    hdf5_file.attrs['move_vocab_size'] = 1968
                    
                    pbar.close()
        
        print(f"Conversion complete!")
        print(f"Games: {game_count}")
        print(f"Positions: {position_count}")
        print(f"Output: {self.output_path}")
    
    def _write_batch(self, hdf5_file, positions, moves, is_white, game_ids):
        """Write a batch of data to HDF5 file."""
        batch_size = len(positions)
        
        # Resize datasets
        old_size = hdf5_file['positions'].shape[0]
        new_size = old_size + batch_size
        
        hdf5_file['positions'].resize((new_size, 64))
        hdf5_file['moves'].resize((new_size,))
        hdf5_file['is_white'].resize((new_size,))
        hdf5_file['game_ids'].resize((new_size,))
        
        # Write data
        hdf5_file['positions'][old_size:new_size] = np.array(positions, dtype=np.int8)
        hdf5_file['moves'][old_size:new_size] = np.array(moves, dtype=np.int16)
        hdf5_file['is_white'][old_size:new_size] = np.array(is_white, dtype=bool)
        hdf5_file['game_ids'][old_size:new_size] = np.array(game_ids, dtype=np.int32)


# Usage script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PGN.zst to HDF5')
    parser.add_argument('input', type=str, help='Input .pgn.zst file')
    parser.add_argument('output', type=str, help='Output .h5 file')
    parser.add_argument('--max-games', type=int, default=None, help='Maximum games to process')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Batch size for writing')
    
    args = parser.parse_args()
    
    converter = PGNtoHDF5Converter(args.output, chunk_size=args.chunk_size)
    converter.convert_pgn_zst(args.input, max_games=args.max_games)