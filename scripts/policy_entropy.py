"""Policy entropy over the legal moves, per network, on the same full-search positions of a
self-play generation, next to the entropy of the search targets those positions carry.

The RL candidates narrow the policy: one generation at T=1 took v2.1's 1.75 nats to 1.45
(gen03), and gen14 sat at 1.39, the moves rated >= 0.5% falling from 12 to 8, which costs
~40 Elo at the bot's 4000 sims. This shows a candidate's width before its gate.

    uv run python scripts/policy_entropy.py data/rl/selfplay/gen16 1500 \
        gen15=data/rl/models/gen15 cand16=data/rl/models/cand16

Runs on the CPU (the GPU is usually busy with self-play); ~1 min per network.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selfplay_rust import Forward  # noqa: E402


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gen_dir", help="self-play generation whose full searches to read")
    p.add_argument("positions", type=int)
    p.add_argument("models", nargs="+", help="label=model_dir")
    p.add_argument("--temps", default="1.0,1.3", help="prior temperatures to report")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    temps = [float(t) for t in args.temps.split(",")]

    rng = np.random.default_rng(5)
    B, P, C, E, LEG, TGT = [], [], [], [], [], []
    for fn in sorted(glob.glob(f"{args.gen_dir}/positions*.npz"))[::3]:
        d = np.load(fn)
        full = np.flatnonzero(d["full"])
        for i in rng.choice(full, size=min(len(full), 40), replace=False):
            a, b = d["legal_ptr"][i], d["legal_ptr"][i + 1]
            if b - a < 2:
                continue
            va, vb = d["visit_ptr"][i], d["visit_ptr"][i + 1]
            B.append(d["boards"][i]); P.append(d["player"][i]); C.append(d["castling"][i]); E.append(d["ep"][i])
            LEG.append(d["legal_idx"][a:b].astype(np.int64))
            c = d["visit_cnt"][va:vb].astype(float)
            TGT.append(c / c.sum())
        if len(B) >= args.positions:
            break
    n = min(len(B), args.positions)
    B, P, C, E = map(np.array, (B[:n], P[:n], C[:n], E[:n]))
    LEG, TGT = LEG[:n], TGT[:n]
    print(f"{args.gen_dir}: {n} full-search positions, {np.mean([len(l) for l in LEG]):.1f} legal moves on average")
    print(f"  search targets (pruned visits): entropy {np.mean([entropy(t) for t in TGT]):.3f} nats")
    for label, path in (m.split("=", 1) for m in args.models):
        fw = Forward(path, "cpu", False, False)
        chunks = []
        for s in range(0, n, 256):
            lg = np.asarray(fw(B[s:s + 256], P[s:s + 256], C[s:s + 256], E[s:s + 256])[0], dtype=np.float32)
            chunks.append(lg.reshape(len(lg), -1))
        logits = np.concatenate(chunks)
        out = []
        for t in temps:
            hs, wide = [], []
            for row, leg in zip(logits, LEG):
                x = row[leg] / t
                q = np.exp(x - x.max())
                q /= q.sum()
                hs.append(entropy(q))
                wide.append(float((q >= 0.005).sum()))
            out.append(f"T={t}: entropy {np.mean(hs):.3f}, moves >= 0.5% {np.mean(wide):.1f}")
        print(f"  {label:10s} " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
