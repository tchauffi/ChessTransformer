"""
Download elite games from database.nikonoel.fr and build an HDF5 database.

By default, bullet and blitz games are excluded (keeping rapid, classical, correspondence).

Usage examples:
    # Download last 12 months and build database
    uv run scripts/build_db.py

    # Download specific range
    uv run scripts/build_db.py --from 2024-01 --to 2024-12

    # Download last 6 months
    uv run scripts/build_db.py --last 6

    # Download everything
    uv run scripts/build_db.py --all

    # Convert already-downloaded files without re-downloading
    uv run scripts/build_db.py --skip-download

    # Include all game types (no filtering)
    uv run scripts/build_db.py --no-filter
"""

import argparse
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

NIKONOEL_BASE_URL = "https://database.nikonoel.fr"
DEFAULT_EXCLUDE = ["Bullet", "Blitz", "UltraBullet"]


def fetch_available_files() -> list[str]:
    """Return sorted list of .zip filenames available on database.nikonoel.fr."""
    req = urllib.request.Request(
        NIKONOEL_BASE_URL,
        headers={"User-Agent": "ChessTransformer/1.0 (database builder)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    # Extract full URLs for lichess_elite*.zip then keep just the filename
    urls = re.findall(rf'href=["\']({re.escape(NIKONOEL_BASE_URL)}/lichess[^"\'<>]+\.zip)["\']', html)
    filenames = sorted(set(Path(u).name for u in urls))
    return filenames


def filter_by_date(filenames: list[str], from_month: str = None, to_month: str = None, last_n: int = None) -> list[str]:
    """Filter filenames by date range or last-N selection."""

    def extract_month(name: str) -> str:
        m = re.search(r"(\d{4}-\d{2})", name)
        return m.group(1) if m else ""

    dated = [(extract_month(f), f) for f in filenames]
    dated = [(ym, f) for ym, f in dated if ym]
    dated.sort(key=lambda x: x[0])

    if from_month:
        dated = [(ym, f) for ym, f in dated if ym >= from_month]
    if to_month:
        dated = [(ym, f) for ym, f in dated if ym <= to_month]
    if last_n is not None:
        dated = dated[-last_n:]

    return [f for _, f in dated]


def download_file(filename: str, dest_dir: Path) -> Path:
    """Download a single zip file from nikonoel.fr into dest_dir with a progress bar."""
    url = f"{NIKONOEL_BASE_URL}/{filename}"
    dest = dest_dir / filename

    if dest.exists():
        print(f"  Already downloaded: {filename}")
        return dest

    print(f"  Downloading: {filename}")
    req = urllib.request.Request(url, headers={"User-Agent": "ChessTransformer/1.0"})

    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with open(dest, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=filename, leave=False) as pbar:
                while True:
                    chunk = resp.read(1 << 16)  # 64 KB
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

    return dest


def extract_zip(zip_path: Path, dest_dir: Path, keep_zip: bool) -> list[Path]:
    """Extract a zip archive into dest_dir with progress, returning paths of extracted files."""
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            out = dest_dir / info.filename
            if out.exists():
                extracted.append(out)
                continue
            with zf.open(info) as src, open(out, "wb") as dst:
                with tqdm(
                    total=info.file_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Extracting {info.filename}",
                    leave=False,
                ) as pbar:
                    while True:
                        chunk = src.read(1 << 16)  # 64 KB
                        if not chunk:
                            break
                        dst.write(chunk)
                        pbar.update(len(chunk))
            extracted.append(out)

    if not keep_zip:
        zip_path.unlink()

    return extracted


def build_database(raw_dir: Path, output: Path, exclude: list[str], max_games: int, num_workers: int):
    """Convert all PGN files in raw_dir into an HDF5 database."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from chesstransformer.datasets.dataset_h5_convertor import PGNtoHDF5Converter

    converter = PGNtoHDF5Converter(
        str(output),
        exclude_time_controls=exclude,
        num_workers=num_workers,
    )
    converter.convert(str(raw_dir), max_games=max_games)


def main():
    parser = argparse.ArgumentParser(
        description="Download elite chess games from database.nikonoel.fr and build an HDF5 database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Selection
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true", help="Download all available months")
    sel.add_argument("--last", type=int, metavar="N", default=12, help="Download last N months (default: 12)")

    parser.add_argument("--from", dest="from_month", metavar="YYYY-MM", help="Start month (inclusive)")
    parser.add_argument("--to", dest="to_month", metavar="YYYY-MM", help="End month (inclusive)")

    # Paths
    parser.add_argument("--output", default="data/elite_db.h5", help="Output HDF5 file (default: data/elite_db.h5)")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory for downloaded/extracted files (default: data/raw)")

    # Behaviour
    parser.add_argument("--skip-download", action="store_true", help="Skip download, convert existing PGN files in raw-dir")
    parser.add_argument("--keep-raw", action="store_true", help="Keep downloaded zip and extracted PGN files after conversion")
    parser.add_argument(
        "--no-filter", action="store_true", help="Include all game types (disable bullet/blitz exclusion)"
    )
    parser.add_argument("--max-games", type=int, default=None, help="Cap total number of games")
    parser.add_argument("--num-workers", type=int, default=None, help="Worker processes for conversion")

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output = Path(args.output)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Resolve which time controls to exclude
    exclude = [] if args.no_filter else DEFAULT_EXCLUDE
    if exclude:
        print(f"Excluding time controls: {', '.join(exclude)}")
    else:
        print("Including all time controls (no filter)")

    # Download + extract phase
    if not args.skip_download:
        print(f"\nFetching file index from {NIKONOEL_BASE_URL} ...")
        try:
            all_files = fetch_available_files()
        except Exception as e:
            print(f"ERROR: Could not fetch file index: {e}", file=sys.stderr)
            sys.exit(1)

        if not all_files:
            print("No .zip files found on the index page.", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(all_files)} available file(s)")

        # Filter by date
        if args.all:
            selected = all_files
        else:
            last_n = None if (args.from_month or args.to_month) else args.last
            selected = filter_by_date(all_files, from_month=args.from_month, to_month=args.to_month, last_n=last_n)

        if not selected:
            print("No files match the selected date range.", file=sys.stderr)
            sys.exit(1)

        print(f"Selected {len(selected)} file(s) to download:")
        for f in selected:
            print(f"  {f}")
        print()

        for filename in selected:
            try:
                zip_path = download_file(filename, raw_dir)
                print(f"  Extracting: {filename}")
                extract_zip(zip_path, raw_dir, keep_zip=args.keep_raw)
            except Exception as e:
                print(f"WARNING: Failed to process {filename}: {e}", file=sys.stderr)

    # Conversion phase
    pgn_files = sorted(raw_dir.glob("*.pgn"))
    if not pgn_files:
        print(f"\nNo .pgn files found in {raw_dir}. Nothing to convert.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(pgn_files)} PGN file(s) to convert → {output}")
    build_database(
        raw_dir=raw_dir,
        output=output,
        exclude=exclude,
        max_games=args.max_games,
        num_workers=args.num_workers,
    )

    # Clean up extracted PGNs unless --keep-raw
    if not args.keep_raw and not args.skip_download:
        for pgn in pgn_files:
            pgn.unlink(missing_ok=True)
        print("Cleaned up extracted PGN files.")


if __name__ == "__main__":
    main()
