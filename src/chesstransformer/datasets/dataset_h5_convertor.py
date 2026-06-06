import h5py
import chess.pgn
import zstandard as zstd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from chesstransformer.models.tokenizer.move_tokenizer import MoveTokenizer
import io
from pathlib import Path

# Per-worker globals initialised once per process
_worker_tokenizer = None
_worker_exclude_tcs = []


def _worker_init(exclude_tcs):
    global _worker_tokenizer, _worker_exclude_tcs
    _worker_tokenizer = MoveTokenizer()
    _worker_exclude_tcs = exclude_tcs


def _process_raw_game(raw_pgn: str):
    """Full pipeline in each worker: text-filter → parse → tokenize.

    Moving parse into workers parallelises the most expensive step.
    The fast text-level filter avoids even calling chess.pgn.read_game
    for excluded time controls.
    """
    global _worker_tokenizer, _worker_exclude_tcs

    # Fast text-level filter before expensive PGN parsing
    if _worker_exclude_tcs:
        idx = raw_pgn.find('[Event "')
        if idx >= 0:
            end = raw_pgn.find('"]', idx + 8)
            if end >= 0:
                event_lower = raw_pgn[idx + 8 : end].lower()
                if any(tc in event_lower for tc in _worker_exclude_tcs):
                    return None

    game = chess.pgn.read_game(io.StringIO(raw_pgn))
    if game is None:
        return None

    if game.headers.get("Variant", "Standard") != "Standard":
        return None

    result_str = game.headers.get("Result", "*")
    if result_str == "1/2-1/2":
        result = 0
    elif result_str == "1-0":
        result = 1
    elif result_str == "0-1":
        result = 2
    else:
        return None

    try:
        white_elo = int(game.headers.get("WhiteElo", "1500"))
    except ValueError:
        white_elo = 1500
    try:
        black_elo = int(game.headers.get("BlackElo", "1500"))
    except ValueError:
        black_elo = 1500

    tokens = []
    for move in game.mainline_moves():
        try:
            tokens.append(_worker_tokenizer.encode(move.uci()))
        except ValueError:
            return None

    if len(tokens) < 10:
        return None

    return np.array(tokens, dtype=np.int16), white_elo, black_elo, result, len(tokens)


def _iter_raw_games(stream):
    """Yield individual PGN game strings from a text stream.

    Splits on '[Event' lines, which always start each game in standard PGN.
    Line-by-line reading keeps memory usage constant regardless of file size.
    """
    buf = []
    for line in stream:
        if line.startswith("[Event ") and buf:
            yield "".join(buf)
            buf = [line]
        else:
            buf.append(line)
    if buf:
        text = "".join(buf).strip()
        if text:
            yield text


class PGNtoHDF5Converter:
    def __init__(
        self,
        output_path: str,
        chunk_size: int = 10000,
        num_workers: int = None,
        exclude_time_controls: list = None,
    ):
        """
        Convert PGN files to HDF5 format storing complete games as UCI move sequences.

        Args:
            output_path: Path to save the HDF5 file
            chunk_size: Number of games to accumulate before writing (default: 10000)
            exclude_time_controls: List of time control categories to skip, matched against
                the Lichess Event header (e.g. ["Bullet", "Blitz", "UltraBullet"])
        """
        self.output_path = output_path
        self.chunk_size = chunk_size
        default_workers = (cpu_count() or 1) - 1
        self.num_workers = max(1, num_workers if num_workers is not None else max(1, default_workers))
        self.exclude_time_controls = [tc.lower() for tc in (exclude_time_controls or [])]

    def convert(self, input_path: str, max_games: int = None):
        """Convert PGN files to HDF5. Input can be a file or folder."""
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {input_path}")
        if path.is_file():
            self._convert_single_file(path, max_games)
        elif path.is_dir():
            self._convert_folder(path, max_games)
        else:
            raise ValueError(f"Invalid input path: {input_path}")

    def _convert_folder(self, folder_path: Path, max_games: int = None):
        pgn_files = sorted(list(folder_path.glob("*.pgn")) + list(folder_path.glob("*.pgn.zst")))
        if not pgn_files:
            raise ValueError(f"No PGN files found in {folder_path}")

        print(f"Found {len(pgn_files)} PGN file(s) in {folder_path}")
        for f in pgn_files:
            print(f"  - {f.name}")
        print()

        total_games = 0
        for pgn_file in pgn_files:
            remaining = None if max_games is None else max(0, max_games - total_games)
            if remaining == 0:
                break
            print(f"\nProcessing: {pgn_file.name}")
            total_games += self._convert_single_file(pgn_file, remaining, append=total_games > 0)
            if max_games and total_games >= max_games:
                print(f"\nReached maximum game limit ({max_games})")
                break

        print(f"\n{'=' * 60}")
        print(f"All files processed! Total games: {total_games}")
        print(f"Output: {self.output_path}")

    def _convert_single_file(self, pgn_path: Path, max_games: int = None, append: bool = False):
        if pgn_path.suffix == ".zst" or pgn_path.name.endswith(".pgn.zst"):
            return self._convert_compressed_file(str(pgn_path), max_games, append)
        else:
            return self._convert_uncompressed_file(str(pgn_path), max_games, append)

    def convert_pgn_zst(self, pgn_path: str, max_games: int = None):
        """Deprecated: Use convert() instead."""
        return self._convert_compressed_file(pgn_path, max_games, append=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_datasets(self, hdf5_file):
        """Create all HDF5 datasets (called only when not appending)."""
        dt = h5py.vlen_dtype(np.dtype("int16"))
        for name, dtype in [
            ("moves", dt),
            ("white_elo", "int16"),
            ("black_elo", "int16"),
            ("result", "int8"),
            ("num_moves", "int16"),
        ]:
            hdf5_file.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                compression="gzip",
                compression_opts=4,
                chunks=(self.chunk_size,),
            )

    def _write_batch(self, hdf5_file, games, white_elos, black_elos, results, num_moves):
        old = hdf5_file["moves"].shape[0]
        new = old + len(games)
        for ds in ("moves", "white_elo", "black_elo", "result", "num_moves"):
            hdf5_file[ds].resize((new,))
        for i, g in enumerate(games):
            hdf5_file["moves"][old + i] = g
        hdf5_file["white_elo"][old:new] = np.array(white_elos, dtype=np.int16)
        hdf5_file["black_elo"][old:new] = np.array(black_elos, dtype=np.int16)
        hdf5_file["result"][old:new] = np.array(results, dtype=np.int8)
        hdf5_file["num_moves"][old:new] = np.array(num_moves, dtype=np.int16)

    def _run_conversion(self, stream, hdf5_file, max_games) -> tuple[int, int]:
        """Core loop: stream raw PGN text → workers parse+tokenize → write HDF5.

        Returns (game_count, total_positions).
        """
        game_count = 0
        total_positions = 0
        games_buf, we_buf, be_buf, res_buf, nm_buf = [], [], [], [], []

        pbar = tqdm(desc="Processing games", unit="games")

        def flush():
            if games_buf:
                self._write_batch(hdf5_file, games_buf, we_buf, be_buf, res_buf, nm_buf)
                games_buf.clear(); we_buf.clear(); be_buf.clear()
                res_buf.clear(); nm_buf.clear()

        def accumulate(result):
            nonlocal game_count, total_positions
            token_array, we, be, res, nm = result
            games_buf.append(token_array)
            we_buf.append(we); be_buf.append(be)
            res_buf.append(res); nm_buf.append(nm)
            total_positions += nm
            game_count += 1
            pbar.update(1)
            if len(games_buf) >= self.chunk_size:
                flush()

        use_parallel = self.num_workers > 1
        raw_iter = _iter_raw_games(stream)

        if use_parallel:
            pool = Pool(
                processes=self.num_workers,
                initializer=_worker_init,
                initargs=(self.exclude_time_controls,),
            )
            early_stop = False
            try:
                # imap_unordered streams results as soon as each worker finishes —
                # no need to wait for a full chunk_size batch before seeing progress.
                for result in pool.imap_unordered(_process_raw_game, raw_iter, chunksize=200):
                    if result is None:
                        continue
                    accumulate(result)
                    if max_games and game_count >= max_games:
                        early_stop = True
                        break
            finally:
                if early_stop:
                    pool.terminate()
                else:
                    pool.close()
                pool.join()
        else:
            # Sequential fallback (single worker)
            _worker_init(self.exclude_time_controls)
            for raw_pgn in raw_iter:
                result = _process_raw_game(raw_pgn)
                if result is None:
                    continue
                accumulate(result)
                if max_games and game_count >= max_games:
                    break

        flush()
        pbar.close()
        return game_count, total_positions

    def _convert_compressed_file(self, pgn_path: str, max_games: int = None, append: bool = False):
        with open(pgn_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                stream = io.TextIOWrapper(reader, encoding="utf-8")
                mode = "a" if append else "w"
                with h5py.File(self.output_path, mode) as hdf5_file:
                    if not (append and "moves" in hdf5_file):
                        self._create_datasets(hdf5_file)
                    game_count, total_positions = self._run_conversion(stream, hdf5_file, max_games)

        self._print_summary(game_count, total_positions)
        return game_count

    def _convert_uncompressed_file(self, pgn_path: str, max_games: int = None, append: bool = False):
        with open(pgn_path, "r", encoding="utf-8") as pgn_file:
            mode = "a" if append else "w"
            with h5py.File(self.output_path, mode) as hdf5_file:
                if not (append and "moves" in hdf5_file):
                    self._create_datasets(hdf5_file)
                game_count, total_positions = self._run_conversion(pgn_file, hdf5_file, max_games)

        self._print_summary(game_count, total_positions)
        return game_count

    def _print_summary(self, game_count, total_positions):
        print("File conversion complete!")
        print(f"Games from this file: {game_count}")
        print(f"Positions from this file: {total_positions}")
        avg = (total_positions / game_count) if game_count else 0.0
        print(f"Average moves per game: {avg:.1f}")


# Usage script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PGN/PGN.zst file(s) or folder to HDF5")
    parser.add_argument("--input", type=str, required=True, help="Input .pgn/.pgn.zst file or folder")
    parser.add_argument("--output", type=str, required=True, help="Output .h5 file")
    parser.add_argument("--max-games", type=int, default=None, help="Maximum games to process")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Batch size for writing")
    parser.add_argument("--num-workers", type=int, default=None, help="Worker processes (default: cpu count - 1)")
    parser.add_argument(
        "--exclude-time-controls",
        nargs="+",
        default=[],
        metavar="TC",
        help="Time control categories to skip (e.g. Bullet Blitz UltraBullet)",
    )

    args = parser.parse_args()
    converter = PGNtoHDF5Converter(
        args.output,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        exclude_time_controls=args.exclude_time_controls,
    )
    converter.convert(args.input, max_games=args.max_games)
