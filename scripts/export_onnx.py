"""Export Pos2MoveV2 model to ONNX format (TensorRT conversion, Rust ct-bot).

Usage
-----
    uv run python scripts/export_onnx.py <model_dir> [-o output.onnx] [--ema]

--ema overlays ema_state.pt on the safetensors weights (same recipe as
Pos2MoveV2Bot) and defaults the output name to model_ema.onnx.

Example
-------
    uv run python scripts/export_onnx.py data/models/pos2move_v2.1 --ema
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


def export_onnx(
    model_dir: str | Path, output_path: str | Path | None = None, use_ema: bool = False
) -> Path:
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

    if use_ema:
        ema_path = model_dir / "ema_state.pt"
        if not ema_path.exists():
            raise FileNotFoundError(f"No ema_state.pt in {model_dir}")
        ema_state = torch.load(str(ema_path), map_location="cpu", weights_only=True)
        ema_state = {k.replace("_orig_mod.", ""): v for k, v in ema_state.items()}
        missing, unexpected = model.load_state_dict(ema_state, strict=False)
        if unexpected:
            print(f"WARNING: EMA contains unexpected keys: {unexpected}")
        print(f"EMA weights loaded ({len(ema_state)} tensors; {len(missing)} buffers kept).")

    model.eval()

    # Dummy inputs. Batch=2, not 1: torch.export specializes size-1 dims to
    # constants, which bakes batch=1 reshapes into the graph.
    board_tokens = torch.randint(0, 13, (2, 64))
    player_token = torch.randint(0, 2, (2,))
    castling_token = torch.randint(0, 16, (2,))
    en_passant_token = torch.randint(0, 9, (2,))

    if output_path is None:
        output_path = model_dir / ("model_ema.onnx" if use_ema else "model.onnx")
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
        # The legacy TorchScript exporter cannot export aten::rms_norm; the
        # dynamo exporter decomposes it (lands on opset 18).
        opset_version=18,
        dynamo=True,
        external_data=False,  # single self-contained .onnx (no sidecar .data)
    )

    # Verify
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified ✓  ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--ema", action="store_true", help="overlay ema_state.pt weights")
    args = parser.parse_args()
    export_onnx(args.model_dir, args.output, use_ema=args.ema)
