"""The overlapped GPU pipeline of selfplay_rust.py / gpu_match.py, on the CPU.

On CUDA the same code runs asynchronously (pinned buffers, events); there every
`verify_every`-th batch is re-checked against the synchronous forward at run time.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("selfplay_core")
MODEL = ROOT / "data/rl/models/gen00"
FLAT = (np.arange(64)[:, None] * 73 + np.arange(73)[None, :]).astype(np.float32)


@pytest.mark.skipif(not (MODEL / "model.safetensors").exists(), reason="no gen00 model")
def test_launch_wait_equals_the_synchronous_forward():
    from selfplay_rust import Forward

    fwd = Forward(str(MODEL), "cpu", False, False, buckets=(8, 16))
    fwd.verify_every = 1  # wait() re-checks each batch against __call__ itself
    rng = np.random.default_rng(0)
    for n in (5, 16, 21):  # one bucket, exactly a bucket, split over two
        boards = rng.integers(0, 13, (n, 64)).astype(np.uint8)
        player, castling, ep = (rng.integers(0, k, n).astype(np.uint8) for k in (2, 16, 9))
        legal = rng.integers(0, 64 * 73, (n, 7)).astype(np.int16)
        legal[:, 5:] = 64 * 73  # padding
        logits, values = fwd.wait(fwd.launch(0, boards, player, castling, ep, legal))
        full, ref_v = fwd(boards, player, castling, ep)
        ref = np.take_along_axis(full.reshape(n, -1), legal[:, :5].astype(np.int64), 1)
        assert np.array_equal(logits[:, :5], ref) and (logits[:, 5:] == 0).all()
        assert np.array_equal(values, ref_v)


class FakeForward:
    """Row-deterministic stand-in with Forward's interface (the second network differs)."""

    instances = 0

    def __init__(self, model_dir, device, use_ema, compile_model, buckets=None, compile_mode=None):
        self.salt = FakeForward.instances * 1.7
        FakeForward.instances += 1
        self.verify_every = 0
        self.pending = {}

    def __call__(self, boards, player, castling, ep):
        h = (boards.astype(np.float32) * np.arange(1, 65, dtype=np.float32)).sum(1) + self.salt
        logits = np.sin(h[:, None, None] * 0.013 + FLAT[None] * 0.37) * 3
        return logits.astype(np.float32), np.tanh(np.sin(h * 0.029)).astype(np.float32)

    def launch(self, slot, boards, player, castling, ep, legal):
        full, values = self(boards, player, castling, ep)
        flat = np.concatenate([full.reshape(len(full), 64 * 73), np.zeros((len(full), 1), np.float32)], 1)
        self.pending[slot] = (np.take_along_axis(flat, legal.astype(np.int64), 1), values)
        return slot

    def wait(self, slot):
        return self.pending.pop(slot)


def _match(monkeypatch, tmp_path, *extra):
    import gpu_match

    FakeForward.instances = 0
    monkeypatch.setattr(gpu_match, "Forward", FakeForward)
    out = tmp_path / f"games{len(list(tmp_path.iterdir()))}.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "gpu_match.py", "--model-a", "a", "--model-b", "b", "--openings", "6", "--max-sims", "32",
        "--ponder-max-sims", "16", "--max-plies", "16", "--threads", "2", "--device", "cpu",
        "--games-out", str(out), *extra])
    gpu_match.main()
    return sorted((r["idx"], r["a_white"], r["result"], r["plies"])
                  for r in map(json.loads, out.read_text().splitlines()))


def test_two_halves_play_the_same_match_as_one(monkeypatch, tmp_path):
    one = _match(monkeypatch, tmp_path, "--engines", "1")
    assert len(one) == 12
    assert _match(monkeypatch, tmp_path, "--engines", "2") == one
    assert _match(monkeypatch, tmp_path, "--engines", "3", "--parallel-games", "4") == one


def test_sprt_stops_the_match_early(monkeypatch, tmp_path):
    import gpu_match

    monkeypatch.setattr(gpu_match, "sprt_decision",
                        lambda llr, a, b, min_pairs_met: "H1" if min_pairs_met else None)
    games = _match(monkeypatch, tmp_path, "--sprt", "--min-pairs", "2", "--parallel-games", "4")
    assert 4 <= len(games) < 12
