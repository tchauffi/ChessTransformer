"""TensorRT FP16 engine of the self-play network, as a drop-in for Forward's torch model.

    uv run python scripts/trt_forward.py build data/rl/scratch/best /tmp/best.engine

TensorRT FP16 is ~1.12x the torch.compile bf16 forward at 128 rows (RTX 5070 Ti, 2026-10-30).
Plain FP16 is NOT usable: RMSNorm squares the residual stream, which overflows FP16 (65504) on
some positions -- rsqrt(inf) = 0 zeroes the token and value errors reach ~1.0. The pow and mean
of every RMSNorm therefore stay FP32 (OBEY_PRECISION_CONSTRAINTS); with that the engine is more
accurate than bf16 (value err 0.0006 vs 0.0016 against fp32, top-1 100% vs 99.6%). Keeping relu^2
in FP32 changes nothing -- its activations peak far below the FP16 range.

The engine is built from the ONNX export (scripts/export_onnx.py: scalar value, dynamic batch) and
is tied to the weights it was built from: selfplay_rust.Forward rebuilds it in a subprocess
(`build`) when --reload-from publishes new weights, and keeps serving the old one meanwhile.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUTS = ("board_tokens", "player_token", "castling_token", "en_passant_token")


def _rms_core(net) -> set[int]:
    """Indices of each RMSNorm's x^2 and mean: the ops that overflow FP16."""
    n = net.num_layers
    layers = [net.get_layer(i) for i in range(n)]
    pinned = set()
    for i, layer in enumerate(layers[:-1]):
        if layer.name.startswith("node_pow") and layers[i + 1].type.name == "REDUCE":
            pinned.update((i, i + 1))
    return pinned


def build_engine(model_dir: Path, engine_path: Path, max_batch: int = 1024, opt_batch: int = 128) -> int:
    """Export model_dir to ONNX and build a FP16 engine at engine_path; returns the layers pinned to FP32."""
    import tensorrt as trt
    from export_onnx import export_onnx

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = export_onnx(model_dir, Path(tmp) / "model.onnx")
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        pinned = _rms_core(network)
        if not pinned:
            raise RuntimeError("no RMSNorm pow/mean layers found: the ONNX layer names changed, so "
                               "the FP16 overflow guard would silently be missing")
        for i in pinned:
            layer = network.get_layer(i)
            layer.precision = trt.float32
            layer.set_output_type(0, trt.float32)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        profile = builder.create_optimization_profile()
        for name in INPUTS:
            tail = (64,) if name == "board_tokens" else ()
            profile.set_shape(name, (1, *tail), (opt_batch, *tail), (max_batch, *tail))
        config.add_optimization_profile(profile)
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError("TensorRT build failed")
    tmp_engine = engine_path.with_suffix(".part")
    tmp_engine.write_bytes(bytes(blob))
    tmp_engine.replace(engine_path)  # a reader never sees a half-written engine
    print(f"engine built in {time.time() - t0:.0f} s ({len(pinned)} RMSNorm layers in FP32) -> {engine_path}",
          flush=True)
    return len(pinned)


class TrtModel:
    """Callable like Pos2MoveV2's forward: (tokens...) -> (move logits (n, 64, 73), scalar value (n, 1)),
    both float32 on the current CUDA device. The value is already the scalar search uses."""

    def __init__(self, engine_path: Path):
        import tensorrt as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.out = {}  # outputs per batch size, allocated once; stream order makes reuse safe

    def __call__(self, bt, pl, ca, ep):
        n = bt.shape[0]
        if n not in self.out:
            self.out[n] = (torch.empty(n, 64, 73, dtype=torch.float32, device=bt.device),
                           torch.empty(n, 1, dtype=torch.float32, device=bt.device))
        logits, value = self.out[n]
        for name, t in zip(INPUTS, (bt, pl, ca, ep)):
            self.ctx.set_input_shape(name, tuple(t.shape))
            self.ctx.set_tensor_address(name, t.data_ptr())
        self.ctx.set_tensor_address("move_logits", logits.data_ptr())
        self.ctx.set_tensor_address("value", value.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return logits, value


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "build":
        sys.exit(__doc__)
    build_engine(Path(sys.argv[2]), Path(sys.argv[3]))
