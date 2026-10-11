"""Expert-iteration targets: WDL value mapping, full-search policy mask, playout caps."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_expert_iter import ACTION_SPACE, compute_loss, value_loss_per_sample  # noqa: E402

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_wdl_target_is_one_hot_outcome_and_keeps_expectation():
    t = torch.tensor([1.0, 0.0, -1.0, 0.6, -0.3])
    # Logits that put almost all mass on one class: the loss is ~0 only on that class.
    for cls, outcome in ((0, 1.0), (1, 0.0), (2, -1.0)):
        logits = torch.full((1, 3), -20.0)
        logits[0, cls] = 20.0
        loss, q = value_loss_per_sample(logits, torch.tensor([outcome]))
        assert loss.item() < 1e-6 and q.item() == pytest.approx(outcome)
    # The soft target's expectation p(win) - p(loss) is the scalar target itself, so a
    # head fitting it exactly backs up the same Q the search would.
    tgt = torch.stack([t.clamp(min=0), 1 - t.abs(), (-t).clamp(min=0)], -1)
    loss, q = value_loss_per_sample(tgt.clamp(min=1e-12).log(), t)
    assert torch.allclose(q, t, atol=1e-6)
    entropy = -(tgt * tgt.clamp(min=1e-12).log()).sum(-1)
    assert torch.allclose(loss, entropy, atol=1e-5)  # CE at the optimum = target entropy


def test_scalar_head_is_mse():
    loss, q = value_loss_per_sample(torch.tensor([[0.5], [-1.0]]), torch.tensor([1.0, -1.0]))
    assert torch.allclose(loss, torch.tensor([0.25, 0.0])) and torch.allclose(q, torch.tensor([0.5, -1.0]))


def test_policy_loss_ignores_fast_searches():
    torch.manual_seed(0)
    B = 4
    logits = torch.randn(B, 64, 73)
    value = torch.randn(B, 3)
    policy = torch.zeros(B, ACTION_SPACE)
    policy[torch.arange(B), torch.randint(0, ACTION_SPACE, (B,))] = 1.0
    z, root_v, hm = torch.zeros(B), torch.zeros(B), torch.full((B,), 80.0)
    full = torch.tensor([1.0, 0.0, 1.0, 0.0])
    _, m = compute_loss(logits, value, policy, full, z, root_v, hm, logits, 1.0, 0.0, 0.0, 40.0)
    scrambled = policy.clone()
    scrambled[1] = scrambled[1].roll(7)  # change only fast-search targets
    scrambled[3] = scrambled[3].roll(11)
    _, m2 = compute_loss(logits, value, scrambled, full, z, root_v, hm, logits, 1.0, 0.0, 0.0, 40.0)
    assert m["policy"].item() == pytest.approx(m2["policy"].item())


def _run_engine(cfg, games=6, seed=0):
    """Drive the Rust engine with a uniform-prior, zero-value fake network."""
    selfplay_core = pytest.importorskip("selfplay_core")
    base = dict(num_parallel_games=games, leaves_per_wave=4, max_plies=24, resign_threshold=2.0)
    eng = selfplay_core.Engine(json.dumps({**base, **cfg}), [START], seed, games)
    out = []
    while not eng.done():
        _, player, _, _ = eng.collect()
        n = len(player)
        eng.apply(np.zeros((n, 64, 73), np.float32), np.zeros(n, np.float32))
        out += eng.drain_finished()
    return out


def test_playout_caps_record_full_flag_and_budget():
    games = _run_engine(dict(num_sims=40, fast_sims=8, full_frac=0.3))
    full = np.concatenate([g["full"] for g in games]).astype(bool)
    visits = np.concatenate([np.add.reduceat(g["visit_cnt"].astype(int), g["visit_ptr"][:-1])
                             for g in games])
    assert 0.1 < full.mean() < 0.5
    # The root expansion is the first simulation and leaves no child visit.
    assert set(visits[full]) == {39} and set(visits[~full]) == {7}


def _first_root_visits(cfg, seed):
    g = _run_engine(cfg, games=1, seed=seed)[0]
    return tuple(g["visit_cnt"][g["visit_ptr"][0]:g["visit_ptr"][1]])


def test_defaults_are_all_full_and_noise_only_on_full_searches():
    assert all(g["full"].all() for g in _run_engine(dict(num_sims=40)))
    # With a fixed fake network the first root search is deterministic: the seed only
    # reaches it through the noise.
    plain = {_first_root_visits(dict(num_sims=40), s) for s in range(4)}
    noisy = {_first_root_visits(dict(num_sims=40, dirichlet_alpha=0.3), s) for s in range(4)}
    assert len(plain) == 1 and len(noisy) > 1
    # Fast searches are never noised (full_frac 0 makes every move fast).
    fast = dict(num_sims=40, fast_sims=8, full_frac=0.0)
    assert {_first_root_visits({**fast, "dirichlet_alpha": 0.3}, s) for s in range(4)} == \
        {_first_root_visits(fast, 0)}


def _tiny_wdl_model():
    from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2
    torch.manual_seed(0)
    return Pos2MoveV2(embed_dim=32, nb_transformer_layers=2, num_heads=4, dropout=0.0,
                      value_head_type="wdl", value_elo_offset=True)


def _human_batch(b=16, exact=False):
    g = torch.Generator().manual_seed(0)
    return {
        "position": torch.randint(0, 13, (b, 64), generator=g),
        "is_white": torch.randint(0, 2, (b,), generator=g),
        "castling_rights": torch.randint(0, 16, (b,), generator=g),
        "en_passant_file": torch.full((b,), 8),
        "from_square": torch.randint(0, 64, (b,), generator=g),
        "action_plane": torch.randint(0, 73, (b,), generator=g),
        "result": torch.randint(0, 3, (b,), generator=g),
        "result_tb": torch.randint(0, 3, (b,), generator=g),
        "value_exact": torch.full((b,), exact),
        "elo_gap": torch.full((b,), 300),
    }


def test_human_replay_feeds_the_rating_gap():
    # Pretraining reads human results through the Elo offset; the RL replay must too,
    # or it re-teaches the value head that the stronger side's positions are winning.
    from train_expert_iter import human_loss
    m = _tiny_wdl_model().eval()
    with torch.no_grad():
        with_gap = human_loss(m, _human_batch(), "cpu", 1.0, elo_gap=True)
        no_gap = human_loss(m, _human_batch(), "cpu", 1.0, elo_gap=False)
        exact = human_loss(m, _human_batch(exact=True), "cpu", 1.0, elo_gap=True)
        exact_no_gap = human_loss(m, _human_batch(exact=True), "cpu", 1.0, elo_gap=False)
    assert not torch.isclose(with_gap, no_gap)
    assert torch.isclose(exact, exact_no_gap)  # tablebase results ignore who was stronger
