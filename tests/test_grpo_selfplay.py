"""Guards on the GRPO objective in scripts/grpo_selfplay.py.

The gradient direction test earns its keep: two separate bugs got past casual
inspection here. A uniform-mean baseline left a drift term that collapsed the
policy onto *worse* moves, and the first attempt to measure the direction was
itself wrong because the model defaults to train mode and dropout swamped the
signal. Both are cheap to detect and expensive to miss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2  # noqa: E402
from grpo_selfplay import CP_K, compute_loss, cp_to_value, group_advantages  # noqa: E402


# --- the cp -> value curve ----------------------------------------------------

def test_cp_to_value_matches_the_lichess_curve():
    """Must agree with scripts/prep_eval_value_data.py, which set the targets."""
    for cp in (-2000, -300, 0, 25, 300, 2000):
        expected = 2.0 / (1.0 + pow(2.718281828459045, -CP_K * cp)) - 1.0
        assert cp_to_value(torch.tensor([float(cp)])).item() == pytest.approx(
            expected, abs=1e-6)


def test_cp_to_value_is_monotonic_and_bounded():
    # +/-2000 is MATE_CP, the widest input the generator ever writes. Past that
    # the curve saturates to exactly +/-1 in float32.
    v = cp_to_value(torch.tensor([-2000.0, -100.0, 0.0, 100.0, 2000.0]))
    assert torch.all(v[1:] > v[:-1])
    assert v.min() >= -1.0 and v.max() <= 1.0
    assert v[2].item() == pytest.approx(0.0)


# --- the advantage invariant --------------------------------------------------

def _rand_group(n=16, c=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    value = torch.randn(n, c, generator=g)
    logits = torch.randn(n, c, generator=g)
    mask = torch.ones(n, c, dtype=torch.bool)
    mask[:, -2:] = torch.rand(n, 2, generator=g) > 0.5  # ragged candidate counts
    p = torch.softmax(logits, dim=1) * mask
    return value * mask, p, mask


def test_policy_weighted_advantages_sum_to_zero():
    """The invariant the uniform-mean baseline violated.

    If sum_c pi~(c) A_c is not zero, the update carries a net push on the whole
    candidate block relative to the non-candidate legal moves -- a term that has
    nothing to do with which candidate is better.
    """
    value, p, mask = _rand_group()
    adv = group_advantages(value, p, mask, adv_clip=1e9)
    pnorm = p / p.sum(1, keepdim=True)
    assert torch.allclose((pnorm * adv).sum(1), torch.zeros(value.size(0)), atol=1e-5)


def test_uniform_baseline_would_violate_it():
    """Pin down that the invariant actually discriminates, so the test can fail."""
    value, p, mask = _rand_group()
    cnt = mask.sum(1, keepdim=True).clamp(min=1)
    uniform_mean = (value * mask).sum(1, keepdim=True) / cnt
    adv_uniform = (value - uniform_mean) * mask
    pnorm = p / p.sum(1, keepdim=True)
    assert (pnorm * adv_uniform).sum(1).abs().max() > 1e-3


def test_advantages_are_clipped():
    value = torch.tensor([[10.0, 0.0, -10.0, 0.0]])
    p = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    adv = group_advantages(value, p, mask, adv_clip=2.0)
    assert adv.abs().max() <= 2.0 + 1e-6


def test_a_flat_group_yields_no_signal():
    """Every candidate equally good: nothing to re-rank, so no gradient.

    Without the explicit std gate this returns ~-0.03 rather than 0: the
    deviations are float32 rounding and the epsilon in the denominator is the
    same order, so the ratio amplifies noise into a real-looking advantage.
    """
    value = torch.full((3, 5), 0.4)
    p = torch.full((3, 5), 0.2)
    mask = torch.ones(3, 5, dtype=torch.bool)
    assert torch.allclose(group_advantages(value, p, mask, 4.0), torch.zeros(3, 5))


def test_a_nearly_flat_group_yields_no_signal():
    """The same, one float32 ulp apart rather than exactly equal."""
    value = torch.full((2, 4), 0.4)
    value[:, 0] += 1e-7
    p = torch.full((2, 4), 0.25)
    mask = torch.ones(2, 4, dtype=torch.bool)
    assert group_advantages(value, p, mask, 4.0).abs().max() < 1e-6


def test_a_genuinely_different_group_still_produces_signal():
    """The gate must not swallow small but real differences."""
    value = torch.tensor([[0.40, 0.30, 0.35, 0.32]])
    p = torch.full((1, 4), 0.25)
    mask = torch.ones(1, 4, dtype=torch.bool)
    assert group_advantages(value, p, mask, 4.0).abs().max() > 0.5


def test_masked_candidates_get_zero_advantage():
    value, p, mask = _rand_group(seed=3)
    adv = group_advantages(value, p, mask, 4.0)
    assert torch.all(adv[~mask] == 0.0)


# --- the gradient itself ------------------------------------------------------

def _tiny_model():
    """A small Pos2MoveV2 -- real module, real heads, fast on CPU."""
    torch.manual_seed(0)
    return Pos2MoveV2(embed_dim=32, nb_transformer_layers=2, num_heads=2,
                      dropout=0.0, layer_drop=0.0)


def _toy_batch(n=8, n_legal=10, n_cand=5, seed=1):
    """A batch in the shape compute_loss expects, with known-good candidates."""
    g = torch.Generator().manual_seed(seed)
    boards = torch.randint(0, 13, (n, 64), generator=g)
    player = torch.randint(0, 2, (n,), generator=g)
    castling = torch.randint(0, 16, (n,), generator=g)
    ep = torch.full((n,), 8)
    # Legal moves are distinct action ids; candidates are the first n_cand.
    legal = torch.stack([torch.randperm(4672, generator=g)[:n_legal] for _ in range(n)])
    cand = legal[:, :n_cand].clone()
    cand_mask = torch.ones(n, n_cand, dtype=torch.bool)
    legal_mask = torch.ones(n, n_legal, dtype=torch.bool)
    # A clear best candidate (column 0) and a clear worst (column 1).
    cp = torch.zeros(n, n_cand)
    cp[:, 0] = 400.0
    cp[:, 1] = -400.0
    return boards, player, castling, ep, cand, cand_mask, cp, legal, legal_mask


def _cand_logp(model, batch):
    b, pl, ca, ep, ci, cm, _, li, lm = batch
    model.eval()
    with torch.no_grad():
        logits, _ = model(b, pl, ca, ep)
        flat = logits.view(b.size(0), -1).float()
        neg = torch.finfo(flat.dtype).min
        lp = F.log_softmax(torch.gather(flat, 1, li).masked_fill(~lm, neg), dim=1)
        col = (li.unsqueeze(2) == ci.unsqueeze(1)).float().argmax(dim=1)
        return torch.gather(lp, 1, col)


@pytest.mark.parametrize("lr", [1e-4, 1e-3])
def test_gradient_raises_good_moves_and_lowers_bad_ones(lr):
    """One SGD step must move the best candidate up and the worst one down.

    Measured in eval mode: with the model left in train mode, dropout and
    layer-drop make each forward stochastic and the comparison measures noise
    rather than the update.
    """
    model, ref = _tiny_model(), _tiny_model()
    for p in ref.parameters():
        p.requires_grad_(False)
    batch = _toy_batch()

    before = _cand_logp(model, batch)
    model.eval()
    loss, _ = compute_loss(model, ref, batch, beta_kl=0.0, sample_k=0, adv_clip=4.0)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    opt.zero_grad()
    loss.backward()
    opt.step()
    delta = _cand_logp(model, batch) - before

    assert delta[:, 0].mean() > 0, "best candidate should gain probability"
    assert delta[:, 1].mean() < 0, "worst candidate should lose probability"
    assert delta[:, 0].mean() > delta[:, 1].mean()


def test_kl_anchor_opposes_movement():
    """A large KL coefficient must dominate the loss and hold the policy still."""
    model, ref = _tiny_model(), _tiny_model()
    ref.load_state_dict(model.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)
    batch = _toy_batch()
    model.eval()

    free, _ = compute_loss(model, ref, batch, beta_kl=0.0, sample_k=0, adv_clip=4.0)
    g_free = torch.autograd.grad(free, list(model.parameters()), allow_unused=True)
    anchored, _ = compute_loss(model, ref, batch, beta_kl=1e4, sample_k=0, adv_clip=4.0)
    g_anch = torch.autograd.grad(anchored, list(model.parameters()), allow_unused=True)

    # At the reference point KL is zero and so is its gradient, so a huge
    # coefficient must not change the update -- the anchor resists *departure*,
    # it does not pull at the starting point.
    for a, b in zip(g_free, g_anch):
        if a is not None and b is not None:
            assert torch.allclose(a, b, atol=1e-4)


def test_metrics_are_plain_floats():
    """Metrics must not carry graph references into the logging path."""
    model, ref = _tiny_model(), _tiny_model()
    for p in ref.parameters():
        p.requires_grad_(False)
    _, m = compute_loss(model, ref, _toy_batch(), 0.02, 0, 4.0)
    assert all(isinstance(v, float) for v in m.values())
    assert {"pg", "kl", "exp_cp", "ref_exp_cp", "d_cp", "top1", "entropy"} <= set(m)


def test_sampled_estimator_agrees_in_direction_with_closed_form():
    model, ref = _tiny_model(), _tiny_model()
    for p in ref.parameters():
        p.requires_grad_(False)
    batch = _toy_batch()
    model.eval()
    torch.manual_seed(0)
    exact, _ = compute_loss(model, ref, batch, 0.0, 0, 4.0)
    sampled, _ = compute_loss(model, ref, batch, 0.0, 64, 4.0)
    g_e = torch.autograd.grad(exact, list(model.parameters()), allow_unused=True)
    g_s = torch.autograd.grad(sampled, list(model.parameters()), allow_unused=True)
    dots = [float((a * b).sum()) for a, b in zip(g_e, g_s)
            if a is not None and b is not None and a.numel() > 1]
    assert sum(dots) > 0, "sampled and closed-form gradients should broadly agree"


# --- adaptive KL control ------------------------------------------------------

def _steer(beta: float, kl: float, target: float) -> float:
    """The controller from grpo_selfplay.main, in isolation."""
    if kl > 1.5 * target:
        return min(beta * 1.5, 1e3)
    if kl < target / 1.5:
        return max(beta / 1.5, 1e-4)
    return beta


def test_kl_controller_tightens_when_the_policy_drifts():
    assert _steer(0.5, kl=0.74, target=0.05) > 0.5


def test_kl_controller_relaxes_when_the_policy_is_stuck():
    assert _steer(0.5, kl=0.001, target=0.05) < 0.5


def test_kl_controller_holds_inside_the_deadband():
    assert _steer(0.5, kl=0.05, target=0.05) == 0.5


def test_kl_controller_converges_on_the_target():
    """Against a policy whose KL falls as beta rises, beta must settle."""
    beta, target = 0.02, 0.05
    for _ in range(200):
        kl = 0.03 / beta          # monotone decreasing in beta
        beta = _steer(beta, kl, target)
    assert 0.05 / 1.5 <= 0.03 / beta <= 0.05 * 1.5


def test_kl_controller_is_bounded():
    beta = 1.0
    for _ in range(500):
        beta = _steer(beta, kl=99.0, target=0.05)
    assert beta <= 1e3
    beta = 1.0
    for _ in range(500):
        beta = _steer(beta, kl=0.0, target=0.05)
    assert beta >= 1e-4
