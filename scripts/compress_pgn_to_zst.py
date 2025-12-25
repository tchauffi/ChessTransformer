#!/usr/bin/env python3
"""Convert PGN files to PGN.ZST (Zstandard compressed) format.

Usage:
    python scripts/compress_pgn_to_zst.py <input_folder> [output_folder]
    
    If output_folder is not specified, compressed files will be created
    in the same directory as the input files with .zst extension.
    
    Use --merge to combine all PGN files into a single compressed file.
"""

import argparse
import zstandard as zstd
from pathlib import Path
import sys


def compress_pgn_to_zst(input_path: Path, output_path: Path = None, compression_level: int = 3):
    """
    Compress a PGN file to PGN.ZST format.
    
    Args:
        input_path: Path to input PGN file
        output_path: Path to output PGN.ZST file (if None, uses input_path with .zst extension)
        compression_level: Zstandard compression level (1-22, default 3 for fast compression)
    """
    if output_path is None:
        output_path = input_path.with_suffix(input_path.suffix + '.zst')
    
    print(f"Compressing {input_path.name} -> {output_path.name}")
    
    # Create compressor
    cctx = zstd.ZstdCompressor(level=compression_level)
    
    # Read input file and compress
    with open(input_path, 'rb') as input_file:
        with open(output_path, 'wb') as output_file:
            cctx.copy_stream(input_file, output_file)
    
    # Show compression stats
    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    ratio = (1 - output_size / input_size) * 100
    
    print(f"  Original:   {input_size:,} bytes")
    print(f"  Compressed: {output_size:,} bytes")
    print(f"  Ratio:      {ratio:.1f}% reduction")
    print()


def merge_pgn_files_to_zst(pgn_files: list, output_path: Path, compression_level: int = 3):
    """
    Merge multiple PGN files into a single PGN.ZST file.
    
    Args:
        pgn_files: List of Path objects to PGN files
        output_path: Path to output merged PGN.ZST file
        compression_level: Zstandard compression level (1-22, default 3)
    """
    print(f"Merging {len(pgn_files)} files into {output_path.name}")
    
    # Create compressor
    cctx = zstd.ZstdCompressor(level=compression_level)
    
    total_input_size = 0
    
    with open(output_path, 'wb') as output_file:
        with cctx.stream_writer(output_file) as compressor:
            for i, pgn_file in enumerate(pgn_files, 1):
                print(f"  Adding {pgn_file.name} ({i}/{len(pgn_files)})")
                
                with open(pgn_file, 'rb') as input_file:
                    # Copy file contents
                    while True:
                        chunk = input_file.read(65536)  # 64KB chunks
                        if not chunk:
                            break
                        compressor.write(chunk)
                    
                    # Add newline separator between files if not already present
                    compressor.write(b'\n')
                
                total_input_size += pgn_file.stat().st_size
    
    # Show compression stats
    output_size = output_path.stat().st_size
    ratio = (1 - output_size / total_input_size) * 100
    
    print()
    print(f"  Total original:   {total_input_size:,} bytes")
    print(f"  Compressed:       {output_size:,} bytes")
    print(f"  Ratio:            {ratio:.1f}% reduction")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Convert PGN files to PGN.ZST (Zstandard compressed) format'
    )
    parser.add_argument(
        'input_folder',
        type=str,
        help='Folder containing PGN files to compress'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        nargs='?',
        default=None,
        help='Output folder for compressed files (default: same as input folder)'
    )
    parser.add_argument(
        '--level',
        type=int,
        default=3,
        help='Compression level (1-22, default: 3)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.pgn',
        help='File pattern to match (default: *.pgn)'
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='Merge all PGN files into a single compressed file'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='merged.pgn.zst',
        help='Output filename when using --merge (default: merged.pgn.zst)'
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        sys.exit(1)
    
    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory")
        sys.exit(1)
    
    # Setup output folder
    if args.output_folder:
        output_folder = Path(args.output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    else:
        output_folder = input_folder
    
    # Find all PGN files
    pgn_files = list(input_folder.glob(args.pattern))
    
    if not pgn_files:
        print(f"No files matching pattern '{args.pattern}' found in {input_folder}")
        sys.exit(1)
    
    print(f"Found {len(pgn_files)} file(s) to compress")
    print(f"Compression level: {args.level}")
    print()
    
    # Merge mode: combine all files into one
    if args.merge:
        output_path = output_folder / args.output_name
        try:
            merge_pgn_files_to_zst(pgn_files, output_path, compression_level=args.level)
            print(f"Merge complete! All files combined into {output_path}")
        except Exception as e:
            print(f"Error merging files: {e}")
            sys.exit(1)
        return
    
    # Compress each file separately
    for pgn_file in pgn_files:
        output_path = output_folder / (pgn_file.name + '.zst')
        try:
            compress_pgn_to_zst(pgn_file, output_path, compression_level=args.level)
        except Exception as e:
            print(f"Error compressing {pgn_file.name}: {e}")
            continue
    
    print(f"Compression complete! {len(pgn_files)} file(s) processed.")


if __name__ == '__main__':
    main()
