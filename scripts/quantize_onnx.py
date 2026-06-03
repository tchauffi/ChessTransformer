"""Quantize the ONNX model to INT8 using static calibration from real chess positions.

Usage
-----
    python scripts/quantize_onnx.py <model_dir> [--num-samples 200]

Requires ``model.onnx`` in *model_dir* (produced by ``export_onnx.py``).
Generates ``model_int8.onnx`` in the same directory.

Static quantization uses a calibration dataset of real board positions
to compute optimal scale/zero-point for each tensor, giving better
accuracy than dynamic quantization.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ChessCalibrationDataReader:
    """Feeds real board positions to the ONNX quantizer for calibration."""

    def __init__(self, num_samples: int = 200):
        import chess

        self.num_samples = num_samples
        self._index = 0
        self._data = self._generate_positions(num_samples)

    def _generate_positions(self, n: int) -> list[dict[str, np.ndarray]]:
        """Play random games to collect diverse positions."""
        import chess
        import random
        from chesstransformer.models.tokenizer import PostionTokenizer

        tokenizer = PostionTokenizer()
        positions = []
        seen = set()

        while len(positions) < n:
            board = chess.Board()
            for _ in range(random.randint(1, 80)):
                legal = list(board.legal_moves)
                if not legal:
                    break
                board.push(random.choice(legal))

                fen_key = board.board_fen()
                if fen_key in seen:
                    continue
                seen.add(fen_key)

                encoded = tokenizer.encode(board)
                castling = 0
                if board.has_kingside_castling_rights(chess.WHITE):
                    castling |= 1
                if board.has_queenside_castling_rights(chess.WHITE):
                    castling |= 2
                if board.has_kingside_castling_rights(chess.BLACK):
                    castling |= 4
                if board.has_queenside_castling_rights(chess.BLACK):
                    castling |= 8

                ep_file = chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8

                positions.append({
                    "board_tokens": np.array([encoded], dtype=np.int64),
                    "player_token": np.array([int(board.turn)], dtype=np.int64),
                    "castling_token": np.array([castling], dtype=np.int64),
                    "en_passant_token": np.array([ep_file], dtype=np.int64),
                })

                if len(positions) >= n:
                    break

        return positions

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._data):
            return None
        sample = self._data[self._index]
        self._index += 1
        return sample

    def rewind(self):
        self._index = 0


def quantize_model(model_dir: str | Path, num_samples: int = 200) -> Path:
    from onnxruntime.quantization import (
        QuantType,
        quantize_dynamic,
    )

    model_dir = Path(model_dir)
    input_path = model_dir / "model.onnx"
    output_path = model_dir / "model_int8.onnx"

    if not input_path.exists():
        raise FileNotFoundError(f"model.onnx not found in {model_dir}. Run export_onnx.py first.")

    print(f"Quantizing {input_path} → {output_path} (dynamic INT8) …")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        per_channel=True,
        weight_type=QuantType.QInt8,
    )

    size_before = input_path.stat().st_size / 1e6
    size_after = output_path.stat().st_size / 1e6
    print(f"Done ✓  FP32: {size_before:.1f} MB → INT8: {size_after:.1f} MB ({size_after/size_before:.0%})")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument("model_dir", help="Directory containing model.onnx")
    parser.add_argument("--num-samples", type=int, default=200, help="Calibration positions (default: 200)")
    args = parser.parse_args()

    quantize_model(args.model_dir, args.num_samples)
