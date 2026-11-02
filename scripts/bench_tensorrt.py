"""TensorRT FP16 against self-play's Forward (torch.compile, bf16): model-core evaluations per
second at self-play batch sizes, and numerics on real self-play positions against fp32.

Braun 2026 (arXiv 2609.37447) measured TensorRT FP16 at 1.86x TorchScript BF16 for its CNN;
this measures it for our transformer against what self-play runs today.

    uv run python scripts/bench_tensorrt.py --model data/rl/models/gen15 \
        --positions data/rl/selfplay/gen16 --batches 64,128,256

The engine is built from <model>/model.onnx (dynamic batch) and cached next to it as
model.fp16.engine; --rebuild forces a new build. Run it on an otherwise idle GPU.
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selfplay_rust import Forward, load_model  # noqa: E402

INPUTS = ("board_tokens", "player_token", "castling_token", "en_passant_token")


def positions(gen_dir: str, n: int):
    rng = np.random.default_rng(3)
    B, P, C, E, L = [], [], [], [], []
    for fn in sorted(glob.glob(f"{gen_dir}/positions*.npz"))[::5]:
        d = np.load(fn)
        for i in rng.choice(len(d["player"]), size=min(len(d["player"]), 32), replace=False):
            a, b = d["legal_ptr"][i], d["legal_ptr"][i + 1]
            B.append(d["boards"][i]); P.append(d["player"][i]); C.append(d["castling"][i]); E.append(d["ep"][i])
            L.append(d["legal_idx"][a:b].astype(np.int64))
        if len(B) >= n:
            break
    t = lambda x: torch.from_numpy(np.array(x[:n]).astype(np.int64)).cuda()
    return (t(B), t(P), t(C), t(E)), L[:n]


def build_engine(onnx_path: Path, engine_path: Path, max_batch: int, opt_batch: int, rebuild: bool):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    if engine_path.exists() and not rebuild:
        return runtime.deserialize_cuda_engine(engine_path.read_bytes())
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for name in INPUTS:
        tail = (64,) if name == "board_tokens" else ()
        profile.set_shape(name, (1, *tail), (opt_batch, *tail), (max_batch, *tail))
    config.add_optimization_profile(profile)
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    engine_path.write_bytes(bytes(blob))
    print(f"engine built in {time.time() - t0:.0f} s -> {engine_path}")
    return runtime.deserialize_cuda_engine(blob)


class TrtRunner:
    def __init__(self, engine):
        self.engine = engine
        self.ctx = engine.create_execution_context()
        self.out = {}  # outputs per batch size, allocated once as Forward's buffers are

    def __call__(self, bt, pl, ca, ep):
        n = bt.shape[0]
        if n not in self.out:
            self.out[n] = (torch.empty(n, 64, 73, dtype=torch.float32, device="cuda"),
                           torch.empty(n, 1, dtype=torch.float32, device="cuda"))
        logits, value = self.out[n]
        for name, t in zip(INPUTS, (bt, pl, ca, ep)):
            self.ctx.set_input_shape(name, tuple(t.shape))
            self.ctx.set_tensor_address(name, t.data_ptr())
        self.ctx.set_tensor_address("move_logits", logits.data_ptr())
        self.ctx.set_tensor_address("value", value.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return logits, value


def evals_per_s(fn, inputs, iters: int) -> float:
    for _ in range(20):
        fn(*inputs)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn(*inputs)
    b.record()
    torch.cuda.synchronize()
    return iters * inputs[0].shape[0] / (a.elapsed_time(b) / 1000)


def agreement(logits, value, ref_logits, ref_value, legal):
    top, dv, dl = [], [], []
    for i, leg in enumerate(legal):
        x, r = logits[i].reshape(-1)[leg], ref_logits[i].reshape(-1)[leg]
        top.append(int(x.argmax() == r.argmax()))
        dl.append(float((x - x.max() - (r - r.max())).abs().max()))
        dv.append(abs(float(value[i]) - float(ref_value[i])))
    return np.mean(top), np.mean(dl), np.max(dl), np.mean(dv), np.max(dv)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--positions", required=True, help="self-play generation dir for real positions")
    p.add_argument("--batches", default="64,128,256")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    model_dir = Path(args.model)

    (bt, pl, ca, ep), legal = positions(args.positions, 512)
    ref = load_model(model_dir, "cuda").float().eval()
    rl, rv = ref(bt, pl, ca, ep)
    rv = ref.value_scalar(rv.float()).squeeze(-1)
    del ref
    fw = Forward(str(model_dir), "cuda", False, True)
    engine = build_engine(model_dir / "model.onnx", model_dir / "model.fp16.engine", max(batches + [512]),
                          128, args.rebuild)
    trt_run = TrtRunner(engine)

    tl, tv = fw.model(bt, pl, ca, ep)
    tv = fw.value_scalar(tv.float()).squeeze(-1)
    el, ev = trt_run(bt, pl, ca, ep)
    for name, (l, v) in (("torch bf16 (self-play now)", (tl.float(), tv)), ("TensorRT fp16", (el, ev.squeeze(-1)))):
        a, dlm, dlx, dvm, dvx = agreement(l.cpu(), v.cpu(), rl.cpu(), rv.cpu(), legal)
        print(f"{name:27s} vs fp32 on {len(legal)} positions: top-1 {a:.1%} | legal logit err mean {dlm:.3f} "
              f"max {dlx:.3f} | value err mean {dvm:.4f} max {dvx:.4f}")
    for b in batches:
        ins = tuple(x[:b].contiguous() for x in (bt, pl, ca, ep))
        bufs = fw._buffers(b)
        for dst, src in zip(bufs, ins):
            dst.copy_(src)
        t_torch = evals_per_s(lambda *x: fw.model(*x), bufs, args.iters)
        t_trt = evals_per_s(trt_run, ins, args.iters)
        print(f"batch {b:4d}: torch.compile bf16 {t_torch:9,.0f} evals/s | TensorRT fp16 {t_trt:9,.0f} evals/s "
              f"| x{t_trt / t_torch:.2f}", flush=True)


if __name__ == "__main__":
    main()
