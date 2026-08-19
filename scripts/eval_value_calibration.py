"""Compare value heads by their output distribution and calibration.

Companion to `scripts/train_value_head.py`. That script reports val MSE, which
has repeatedly failed to predict playing strength: the `pos2move_v2.1-sfvalue`
head fit the Stockfish cp target better than any other (MSE -46.6%) and then
gated WORST of any head tried (33.3% over 48 games).

The diagnosed reason was not the average fit but the tail. 11.2% of the cp target
sits exactly at +-1 (mates), and an unbounded head fitted with MSE learns to
extrapolate past it -- emitting near-certain values for positions that are merely good. MCTS
is a maximiser over leaf values, so it seeks those spurious certainties out and
amplifies them (the same mechanism as the h4 artifact at high sim counts).

So this script measures what MSE averages away:

  over    share of |v| > 0.99, and max |v| -- the overconfidence tail. A head
          that predicts outside [-1, 1] at all is extrapolating past its target.
  std     spread of the outputs. Lower than the target's own std means a head
          that is systematically hedging.
  ECE     expected calibration error: bin by predicted value, compare the bin
          mean against the mean target in that bin. Answers "when this head
          says +0.8, is the position really worth +0.8?" -- which is what a
          leaf evaluator has to get right, and which MSE conflates with
          resolution.

Gate by games regardless (`scripts/head_to_head.py`); this is the mechanistic
read that says *why* a head plays the way it does, on far less compute than a
match. Uses the same seed/--val-frac split as train_value_head.py, so the
positions scored here are the ones that head held out.

Usage
-----
    uv run python scripts/eval_value_calibration.py \
        --data data/eval/lichess-sf \
        --models data/models/pos2move_v2.1 \
                 data/models/pos2move_v2.1-sfvalue \
                 data/models/pos2move_v2.1-sfvalue-bounded
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2


def load_model(base: Path, device: str) -> Pos2MoveV2:
    cfg = json.loads((base / "model_config.json").read_text())
    m = Pos2MoveV2(**cfg)
    with safe_open(str(base / "model.safetensors"), framework="pt", device="cpu") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.eval().to(device)


@torch.no_grad()
def predict(model: Pos2MoveV2, data: dict, idx: np.ndarray, device: str, batch: int) -> np.ndarray:
    """Value output over the selected positions, in fp32."""
    tensors = [torch.from_numpy(data[k][idx].astype(np.int64))
               for k in ("boards", "player", "castling", "ep")]
    out = np.empty(len(idx), dtype=np.float32)
    for i in range(0, len(idx), batch):
        sl = slice(i, i + batch)
        b, p, c, e = (t[sl].to(device) for t in tensors)
        _, v = model(b, p, c, e)
        out[sl] = v.squeeze(-1).float().cpu().numpy()
    return out


def ece(pred: np.ndarray, target: np.ndarray, bins: int = 20) -> tuple[float, list]:
    """Expected calibration error over equal-width bins of the prediction.

    Bins are weighted by occupancy, so near-empty extreme bins cannot dominate.
    """
    edges = np.linspace(-1.0, 1.0, bins + 1)
    # Predictions can fall outside [-1, 1]; clip only the bin assignment so the
    # overconfident tail lands in the end bins rather than being dropped.
    which = np.clip(np.digitize(pred, edges[1:-1]), 0, bins - 1)
    total, rows = 0.0, []
    for b in range(bins):
        m = which == b
        if not m.any():
            continue
        p_mean, t_mean, n = pred[m].mean(), target[m].mean(), int(m.sum())
        total += n * abs(p_mean - t_mean)
        rows.append((float(edges[b]), float(edges[b + 1]), n, float(p_mean), float(t_mean)))
    return total / len(pred), rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--positions", type=int, default=150_000,
                   help="subsample of the held-out split to score (0 = all)")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="must match the train_value_head.py run being audited")
    p.add_argument("--seed", type=int, default=0, help="likewise")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--detail", action="store_true", help="print per-bin calibration tables")
    args = p.parse_args()

    npz = Path(args.data) / "positions_eval.npz"
    with np.load(npz) as f:
        data = {k: f[k] for k in ("boards", "player", "castling", "ep", "value", "decisive")}
    n = len(data["value"])

    # Same split as train_value_head.py, so these positions are genuinely held out.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    va_idx = perm[:max(1, int(args.val_frac * n))]
    if args.positions and args.positions < len(va_idx):
        va_idx = va_idx[:args.positions]
    va_idx = np.sort(va_idx)

    target = data["value"][va_idx].astype(np.float32)
    decisive = data["decisive"][va_idx].astype(bool)
    print(f"Scoring {len(va_idx):,} held-out positions "
          f"(of {n:,}) | target std {target.std():.3f}, "
          f"|target|>0.99 {np.mean(np.abs(target) > 0.99):.1%}\n")

    hdr = (f"{'model':<34}{'MSE':>8}{'sign':>8}{'std':>8}"
           f"{'|v|>.99':>9}{'max|v|':>9}{'ECE':>8}")
    print(hdr)
    print("-" * len(hdr))
    detail = []
    for spec in args.models:
        base = Path(spec)
        model = load_model(base, args.device).bfloat16()
        pred = predict(model, data, va_idx, args.device, args.batch)
        mse = float(((pred - target) ** 2).mean())
        sign = float(((pred[decisive] > 0) == (target[decisive] > 0)).mean())
        over = float(np.mean(np.abs(pred) > 0.99))
        e, rows = ece(pred, target, args.bins)
        print(f"{base.name:<34}{mse:>8.4f}{sign:>7.1%}{pred.std():>8.3f}"
              f"{over:>8.2%}{np.abs(pred).max():>9.3f}{e:>8.4f}")
        detail.append((base.name, rows))
        del model
        torch.cuda.empty_cache()

    print("\n|v|>.99 is the overconfidence tail MCTS amplifies; max|v| above 1.0 "
          "means the head\nextrapolates past its own target range. ECE is mean "
          "|predicted - actual| over prediction bins.")

    if args.detail:
        for name, rows in detail:
            print(f"\n{name}")
            print(f"  {'bin':>14}{'n':>10}{'pred':>9}{'actual':>9}{'gap':>9}")
            for lo, hi, cnt, pm, tm in rows:
                print(f"  [{lo:+.2f},{hi:+.2f}]{cnt:>10,}{pm:>9.3f}{tm:>9.3f}{pm - tm:>+9.3f}")


if __name__ == "__main__":
    main()
