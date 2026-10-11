"""Per-head towers: the last K layers private to each head, the rest shared."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2  # noqa: E402

CFG = dict(embed_dim=64, nb_transformer_layers=4, num_heads=4, dropout=0.0, value_head_type="wdl")


def _model(**kw):
    torch.manual_seed(0)
    return Pos2MoveV2(**CFG, **kw)


def _inputs(b=4):
    g = torch.Generator().manual_seed(1)
    return (torch.randint(0, 13, (b, 64), generator=g), torch.randint(0, 2, (b,), generator=g),
            torch.randint(0, 16, (b,), generator=g), torch.randint(0, 9, (b,), generator=g))


def test_zero_towers_is_v21():
    # Same keys, same init, same outputs: existing checkpoints load and seeded runs replay.
    torch.manual_seed(0)
    a = Pos2MoveV2(**CFG).eval()
    b = _model(head_tower_layers=0).eval()
    assert a.state_dict().keys() == b.state_dict().keys()
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k
    with torch.no_grad():
        for x, y in zip(a(*_inputs()), b(*_inputs())):
            assert torch.equal(x, y)


@pytest.mark.parametrize("k", [-1, 4])
def test_tower_depth_bounds(k):
    with pytest.raises(ValueError):
        _model(head_tower_layers=k)


def test_tower_layout_and_params():
    base, split = _model(), _model(head_tower_layers=2)
    assert len(split.transformer_layers) == 2
    assert len(split.policy_tower) == len(split.value_tower) == 2
    layer = sum(p.numel() for p in base.transformer_layers[0].parameters())
    norm = sum(p.numel() for p in base.final_norm.parameters())
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(split) - n(base) == 2 * layer + norm  # one extra copy of K layers + value_norm
    with torch.no_grad():
        logits, value = split.eval()(*_inputs())
    assert logits.shape == (4, 64, 73) and value.shape == (4, 3)


@pytest.mark.parametrize("pool", ["state", "attn"])
def test_each_loss_stays_out_of_the_other_tower(pool):
    m = _model(head_tower_layers=2, value_pool=pool).train()
    logits, value = m(*_inputs())

    def grads(loss):
        m.zero_grad()
        loss.backward(retain_graph=True)
        return {n: p.grad is not None and p.grad.abs().sum().item() > 0
                for n, p in m.named_parameters()}

    pol = grads(logits.float().logsumexp(-1).mean())
    val = grads(value.float().logsumexp(-1).mean())
    for name in pol:
        if name.startswith(("value_tower.", "value_norm.", "value_pool.", "value_head.")):
            assert not pol[name], name
        if name.startswith(("policy_tower.", "final_norm.", "move_head.")):
            assert not val[name], name
    shared = [n for n in pol if n.startswith("transformer_layers.") and n.endswith("weight")]
    assert shared and all(pol[n] and val[n] for n in shared)
