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


# ── Gumbel completed-Q policy targets ───────────────────────────────────────
def test_completed_q_target_matches_the_paper_on_a_hand_example():
    from train_expert_iter import completed_q_targets
    # One position: prior .6/.3/.1, visits 10/5/0, Q .1/.3/-, network value 0.
    prior, n, q = np.array([.6, .3, .1]), np.array([10, 5, 0]), np.array([.1, .3, 0.])
    pi = completed_q_targets(prior, n, q, np.array([0, 3]), np.array([0.0]), np.array([0.0]))
    v_mix = (0.0 + 15 * (.6 * .1 + .3 * .3) / .9) / 16           # 0.15625
    cq = np.array([.1, .3, v_mix])
    want = prior * np.exp((50 + 10) * 0.1 * (cq + 1) / 2)
    np.testing.assert_allclose(pi, want / want.sum(), rtol=1e-5)
    assert pi[1] > prior[1] and pi[0] < prior[0]  # the better-valued move gains, by Q not visits


def test_completed_q_target_properties():
    from train_expert_iter import completed_q_targets
    rng = np.random.default_rng(0)
    lens = rng.integers(2, 30, size=50)
    ptr = np.concatenate([[0], np.cumsum(lens)])
    prior = rng.dirichlet(np.ones(lens.sum()))  # any positive weights; normalised per position below
    for i in range(50):
        prior[ptr[i]:ptr[i + 1]] /= prior[ptr[i]:ptr[i + 1]].sum()
    n = rng.integers(0, 40, size=lens.sum()) * (rng.random(lens.sum()) < 0.5)
    q = np.where(n > 0, rng.uniform(-1, 1, lens.sum()), 0.0)
    net_v, fb = rng.uniform(-1, 1, 50), np.zeros(50)
    pi = completed_q_targets(prior, n, q, ptr, net_v, fb)
    seg = np.repeat(np.arange(50), lens)
    np.testing.assert_allclose(np.bincount(seg, weights=pi), 1.0, rtol=1e-5)
    # No value scale: the prior itself.
    np.testing.assert_allclose(completed_q_targets(prior, n, q, ptr, net_v, fb, c_scale=0.0), prior, rtol=1e-5)
    # Unvisited moves share one completed value: their prior ratios survive.
    for i in range(50):
        sl = slice(ptr[i], ptr[i + 1])
        un = n[sl] == 0
        if un.sum() >= 2:
            np.testing.assert_allclose((pi[sl] / prior[sl])[un], (pi[sl] / prior[sl])[un][0], rtol=1e-4)
    # A NaN network value falls back to the given one.
    nan_v = net_v.copy(); nan_v[3] = np.nan
    fb2 = net_v.copy()
    np.testing.assert_allclose(completed_q_targets(prior, n, q, ptr, nan_v, fb2), pi, rtol=1e-5)


def test_dataset_takes_gumbel_targets_from_the_root_search_columns(tmp_path):
    from train_expert_iter import SelfplayDataset, completed_q_targets
    P = 4
    lens = np.array([3, 2, 4, 3]); ptr = np.concatenate([[0], np.cumsum(lens)])
    legal_idx = np.arange(lens.sum(), dtype=np.uint16) * 7
    prior = np.concatenate([np.full(k, 1 / k) for k in lens]).astype(np.float16)
    legal_n = np.array([5, 3, 0, 8, 0, 2, 2, 2, 2, 0, 6, 1], dtype=np.uint16)
    legal_q = np.where(legal_n > 0, 0.1, 0.0).astype(np.float16)
    vis = legal_n > 0
    shard = dict(boards=np.zeros((P, 64), np.uint8), player=np.ones(P, np.uint8), castling=np.zeros(P, np.uint8),
                 ep=np.zeros(P, np.uint8), halfmove=np.arange(P, dtype=np.uint16), z=np.zeros(P, np.float32),
                 root_v=np.zeros(P, np.float32), full=np.array([1, 0, 1, 0], np.uint8),
                 game_id=np.zeros(P, np.uint32), visit_idx=legal_idx[vis], visit_cnt=legal_n[vis],
                 visit_ptr=np.concatenate([[0], np.cumsum([vis[ptr[i]:ptr[i + 1]].sum() for i in range(P)])]),
                 legal_idx=legal_idx, legal_prior=prior, legal_n=legal_n, legal_q=legal_q, legal_ptr=ptr,
                 root_net_v=np.zeros(P, np.float32))
    np.savez(tmp_path / "positions_0000.npz", **shard)
    plain = SelfplayDataset([tmp_path])
    assert plain.full.tolist() == [1, 0, 1, 0] and plain.v_cnt.dtype == np.uint16
    g = SelfplayDataset([tmp_path], policy_target="gumbel")
    np.testing.assert_array_equal(g.v_idx, legal_idx)
    np.testing.assert_allclose(g.v_cnt, completed_q_targets(prior, legal_n, legal_q, ptr, shard["root_net_v"],
                                                            shard["root_v"]), rtol=1e-6)
    assert g.full.tolist() == [1, 0, 1, 0]
    assert SelfplayDataset([tmp_path], policy_target="gumbel", gumbel_fast=True).full.tolist() == [1, 1, 1, 1]
    for k in ("legal_idx", "legal_prior", "legal_n", "legal_q", "legal_ptr", "root_net_v"):
        del shard[k]
    np.savez(tmp_path / "positions_0000.npz", **shard)
    with pytest.raises(ValueError, match="root search columns"):
        SelfplayDataset([tmp_path], policy_target="gumbel")
