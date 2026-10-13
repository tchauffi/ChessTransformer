"""SelfPlayEngine: self-play on the deployed bot's search.

Move-for-move equality with the bot's own go() flow is tested in Rust
(chess-core selfplay::tests); these check the Python-facing protocol.
"""

import json

import numpy as np
import pytest

selfplay_core = pytest.importorskip("selfplay_core")

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ITALIAN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
FLAT = (np.arange(64)[:, None] * 73 + np.arange(73)[None, :]).astype(np.float32)


def _net(boards):
    h = (boards.astype(np.float32) * np.arange(1, 65, dtype=np.float32)).sum(1)
    logits = np.sin(h[:, None, None] * 0.013 + FLAT[None] * 0.37) * 3
    return logits.astype(np.float32), np.tanh(np.sin(h * 0.029)).astype(np.float32)


def _run(cfg, games=6, seed=0, cls=None, gathered=True, evals=None):
    """Play `games` games; `evals`, a list, collects every batch's size."""
    base = dict(num_parallel_games=4, max_plies=24, resign_threshold=2.0, threads=2)
    legacy = cls is selfplay_core.Engine
    eng = (cls or selfplay_core.SelfPlayEngine)(json.dumps({**base, **cfg}), [START, ITALIAN],
                                                seed, games)
    out = []
    while not eng.done():
        batch = eng.collect()
        boards, player = batch[0], batch[1]
        if evals is not None:
            evals.append(len(player))
        if len(player):
            logits, values = _net(boards)
        else:
            logits, values = np.zeros((0, 64, 73), np.float32), np.zeros(0, np.float32)
        if gathered and not legacy:
            legal = batch[4].astype(np.int64)
            flat = np.concatenate([logits.reshape(len(logits), 64 * 73),
                                   np.zeros((len(logits), 1), np.float32)], 1)
            eng.apply_legal(np.take_along_axis(flat, legal, 1), values)
        else:
            eng.apply(logits, values)
        out += eng.drain_finished()
    return sorted(out, key=lambda g: g["id"])


def _visit_sums(g):
    return np.add.reduceat(g["visit_cnt"].astype(int), g["visit_ptr"][:-1])


def test_same_records_as_the_legacy_engine():
    new, old = _run(dict(num_sims=16))[0], _run(dict(num_sims=16), cls=selfplay_core.Engine)[0]
    assert set(new) == set(old) | {"rule50", "repetition"}  # the legacy engine predates them
    for k in ("boards", "player", "castling", "ep", "halfmove", "root_v", "full", "z",
              "visit_idx", "visit_cnt", "visit_ptr"):
        assert new[k].dtype == old[k].dtype, k


def test_playout_caps_budgets_and_tree_reuse():
    games = _run(dict(num_sims=48, fast_sims=16, full_frac=0.3, move_temp=0.0))
    full = np.concatenate([g["full"] for g in games]).astype(bool)
    sums = np.concatenate([_visit_sums(g) for g in games])
    assert 0.1 < full.mean() < 0.6
    # Fresh sims are the budget; a reused tree adds its warm visits on top.
    assert (sums[full] >= 48).all() and (sums[~full] >= 16).all()
    assert (sums > np.where(full, 48, 16)).any()
    for g in games:  # the first search of a game has no tree to reuse
        assert _visit_sums(g)[0] == (48 if g["full"][0] else 16)


def test_capped_budgets_count_warm_visits_and_search_less():
    cfg = dict(num_sims=48, fast_sims=16, full_frac=0.3, move_temp=0.0)
    evals = {}
    for cap in ("none", "fast", "all"):
        n = []
        games = _run({**cfg, "cap_total_full": cap == "all", "cap_total_fast": cap != "none"},
                     evals=n)
        evals[cap] = sum(n)
        full = np.concatenate([g["full"] for g in games]).astype(bool)
        sums = np.concatenate([_visit_sums(g) for g in games])
        budget = np.where(full, 48, 16)
        # A capped search tops the reused tree up to the budget, or skips when it holds it.
        assert (sums >= budget).all()
        capped = np.ones_like(full) if cap == "all" else ~full if cap == "fast" else ~np.ones_like(full)
        if cap != "none":
            assert (sums[capped] == budget[capped]).mean() > 0.3, cap
    assert evals["all"] < evals["fast"] < evals["none"], evals


def test_defaults_all_full_seeded_and_thread_independent():
    cfg = dict(num_sims=24, fast_sims=8, full_frac=0.25, dirichlet_alpha=0.3)
    a = _run(cfg, seed=4)
    assert all(g["full"].all() for g in _run(dict(num_sims=24)))
    same = _run({**cfg, "threads": 1}, seed=4)
    assert [g["moves"] for g in a] == [g["moves"] for g in same]
    assert [g["moves"] for g in a] != [g["moves"] for g in _run(cfg, seed=5)]


def test_noise_never_touches_fast_searches():
    fast = dict(num_sims=24, fast_sims=8, full_frac=0.0, move_temp=0.0)
    quiet = _run(fast, seed=1)
    noisy = _run({**fast, "dirichlet_alpha": 0.3}, seed=1)
    for q, n in zip(quiet, noisy):
        assert q["moves"] == n["moves"] and (q["visit_cnt"] == n["visit_cnt"]).all()
    full = dict(num_sims=24, move_temp=0.0)
    assert [g["visit_cnt"].tolist() for g in _run(full, seed=1)] != \
        [g["visit_cnt"].tolist() for g in _run({**full, "dirichlet_alpha": 0.3}, seed=1)]


def test_gathered_logits_play_the_same_games():
    cfg = dict(num_sims=24, fast_sims=8, full_frac=0.3, dirichlet_alpha=0.3)
    a, b = _run(cfg, seed=2), _run(cfg, seed=2, gathered=False)
    assert [g["moves"] for g in a] == [g["moves"] for g in b]
    assert all((x["visit_cnt"] == y["visit_cnt"]).all() for x, y in zip(a, b))


def test_forced_playouts_prune_full_search_targets_only():
    """KataGo forced playouts + policy target pruning: full searches record pruned visits
    (possibly under the budget), fast ones keep every visit; seeded like the rest."""
    cfg = dict(num_sims=48, fast_sims=16, full_frac=0.3, dirichlet_alpha=0.3, move_temp=0.0)
    forced = {**cfg, "forced_playouts": 2.0}
    a, b = _run(forced, seed=3), _run(forced, seed=3)
    assert [g["moves"] for g in a] == [g["moves"] for g in b]
    plain = _run(cfg, seed=3)
    full = np.concatenate([g["full"] for g in a]).astype(bool)
    sums = np.concatenate([_visit_sums(g) for g in a])
    assert (sums[~full] >= 16).all()
    assert (sums[full] < 48).any(), "pruning never took a full search under its fresh budget"
    assert [g["visit_cnt"].tolist() for g in a] != [g["visit_cnt"].tolist() for g in plain]
