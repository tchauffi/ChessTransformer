"""Export Pos2MoveV2 model to ONNX format for TensorRT conversion.

Usage
-----
    python scripts/export_onnx.py <model_dir> [output_path]

Example
-------
    python scripts/export_onnx.py logs/pos2move_v2/run_033_20260501_150556/checkpoints/best_model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2


def export_onnx(model_dir: str | Path, output_path: str | Path | None = None) -> Path:
    model_dir = Path(model_dir)

    # Locate config
    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        config_path = model_dir.parent.parent / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.json not found in {model_dir} or parent")

    with open(config_path) as f:
        config = json.load(f)

    print(f"Exporting Pos2MoveV2 from {model_dir}")

    # Load model
    model = Pos2MoveV2(**config)
    safetensors_path = model_dir / "model.safetensors"
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy inputs (batch=1)
    board_tokens = torch.randint(0, 13, (1, 64))
    player_token = torch.randint(0, 2, (1,))
    castling_token = torch.randint(0, 16, (1,))
    en_passant_token = torch.randint(0, 9, (1,))

    if output_path is None:
        output_path = model_dir / "model.onnx"
    output_path = Path(output_path)

    print(f"Exporting to {output_path} …")
    torch.onnx.export(
        model,
        (board_tokens, player_token, castling_token, en_passant_token),
        str(output_path),
        input_names=["board_tokens", "player_token", "castling_token", "en_passant_token"],
        output_names=["move_logits", "value"],
        dynamic_axes={
            "board_tokens": {0: "batch"},
            "player_token": {0: "batch"},
            "castling_token": {0: "batch"},
            "en_passant_token": {0: "batch"},
            "move_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    # Verify
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified ✓  ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir> [output_path]")
        sys.exit(1)
    model_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    export_onnx(model_dir, output_path)
