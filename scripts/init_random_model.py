"""A randomly initialised network with another one's architecture: the start of a run from scratch.

    .venv/bin/python scripts/init_random_model.py --like data/rl/models/gen15 \
        --out data/rl/scratch/best --seed 0

Writes model.safetensors, the source's model_config.json and SOURCE ("random_init_seed<N>"),
the files a model directory of the loop holds.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--like", required=True, help="model directory whose model_config.json to use")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    like, out = Path(args.like), Path(args.out)
    if (out / "model.safetensors").exists():
        sys.exit(f"{out} already holds a model")
    torch.manual_seed(args.seed)
    model = Pos2MoveV2(**json.loads((like / "model_config.json").read_text()))
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()},
              str(out / "model.safetensors"))
    shutil.copy(like / "model_config.json", out / "model_config.json")
    (out / "SOURCE").write_text(f"random_init_seed{args.seed}\n")
    n = sum(t.numel() for t in model.parameters())
    print(f"{out}: random {like.name} architecture, {n:,} parameters, seed {args.seed}")


if __name__ == "__main__":
    main()
