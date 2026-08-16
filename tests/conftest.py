"""Shared fixtures.

The distributed tests need a shard directory. The real ones are gitignored (5-21 GB) and
absent in CI, so rather than let the whole suite skip there -- a green tick that proves
nothing -- this synthesizes a small, format-valid shard set on demand.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chesstransformer.datasets.shard_format import (  # noqa: E402
    FORMAT_VERSION,
    LEGAL_CAP,
    NUM_ACTION_PLANES,
    NUM_SQUARES,
    PAD_ACTION,
    RECORD_DTYPE,
    dtype_to_meta,
)

REAL_SHARDS = Path(__file__).resolve().parent.parent / "data" / "shards" / "elite_k16"


def _synthesize(out_dir: Path, n_shards: int = 8, rows_per_shard: int = 640, seed: int = 0):
    """Write a shard set that is structurally identical to a real one.

    Values are random but respect every invariant the loader and the GPU scatter rely on:
    tokens within vocab, planes in range, `legal_cnt` matching the number of non-pad
    entries, and padding written as PAD_ACTION rather than 0. A generator that got the
    padding wrong would make the tests pass against data the real pipeline never produces.
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards, total = [], 0

    for sid in range(n_shards):
        rows = np.zeros(rows_per_shard, dtype=RECORD_DTYPE)
        rows["position"] = rng.integers(0, 13, size=(rows_per_shard, NUM_SQUARES), dtype=np.uint8)
        rows["from_square"] = rng.integers(0, NUM_SQUARES, size=rows_per_shard, dtype=np.uint8)
        rows["action_plane"] = rng.integers(0, NUM_ACTION_PLANES, size=rows_per_shard, dtype=np.uint8)
        rows["castling_rights"] = rng.integers(0, 16, size=rows_per_shard, dtype=np.uint8)
        rows["en_passant_file"] = rng.integers(0, 9, size=rows_per_shard, dtype=np.uint8)
        rows["is_white"] = rng.integers(0, 2, size=rows_per_shard, dtype=np.uint8)
        rows["result"] = rng.integers(0, 3, size=rows_per_shard).astype(np.int8)
        rows["move_number"] = rng.integers(0, 200, size=rows_per_shard, dtype=np.uint16)

        counts = rng.integers(4, 40, size=rows_per_shard)
        legal = np.full((rows_per_shard, LEGAL_CAP), PAD_ACTION, dtype=np.uint16)
        for i, c in enumerate(counts):
            # Target index first, exactly as build_shards.py writes it, then distinct others.
            target = int(rows["from_square"][i]) * NUM_ACTION_PLANES + int(rows["action_plane"][i])
            others = rng.choice(PAD_ACTION, size=c * 2, replace=False)
            others = [int(o) for o in others if int(o) != target][: c - 1]
            vals = [target] + others
            legal[i, : len(vals)] = vals
            counts[i] = len(vals)
        rows["legal_idx"] = legal
        rows["legal_cnt"] = counts.astype(np.uint8)

        np.save(out_dir / f"shard_{sid:04d}.npy", rows)
        shards.append({
            "shard_id": sid, "path": f"shard_{sid:04d}.npy",
            "game_start": sid * 1000, "game_end": (sid + 1) * 1000,
            "num_samples": rows_per_shard, "num_overflow": 0,
            "num_bad_push": 0, "num_skipped_games": 0, "num_filtered_games": 0,
        })
        total += rows_per_shard

    meta = {
        "format_version": FORMAT_VERSION, "record_dtype": dtype_to_meta(),
        "legal_cap": LEGAL_CAP, "pad_action": PAD_ACTION,
        "source_h5": "<synthetic>", "source_games": n_shards * 1000,
        "k": 4, "seed": seed, "sample_mode": "unique", "sample_weighting": "middlegame",
        "skip_opening_plies": 0, "min_elo": None, "max_elo": None,
        "num_samples": total,
        "splits": {"train": list(range(2, n_shards)), "val": [0], "test": [1]},
        "shards": shards, "git_sha": "synthetic", "build_seconds": 0.0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


@pytest.fixture(scope="session")
def shards(tmp_path_factory) -> str:
    """Real shards when present (they exercise real data), synthetic otherwise."""
    # CT_FORCE_SYNTHETIC_SHARDS lets a dev reproduce the CI path (no real shards) without
    # deleting 5 GB of data. CI itself just has no shards and takes the same branch.
    import os
    if not os.environ.get("CT_FORCE_SYNTHETIC_SHARDS") and (REAL_SHARDS / "meta.json").exists():
        return str(REAL_SHARDS)
    return str(_synthesize(tmp_path_factory.mktemp("shards")))


@pytest.fixture(scope="session")
def synthetic_shards(tmp_path_factory) -> str:
    """Always synthetic -- used by the test that checks the synthesizer itself is valid."""
    return str(_synthesize(tmp_path_factory.mktemp("synthetic"), seed=1))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


def has_git_lfs() -> bool:  # pragma: no cover - helper for local debugging
    return subprocess.run(["git", "lfs", "version"], capture_output=True).returncode == 0
