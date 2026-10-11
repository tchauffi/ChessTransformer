"""Attention bias: relation buckets (v2.1), a full per-pair table, and smolgen."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chesstransformer.models.transformer.pos2move_v2 import (  # noqa: E402
    Pos2MoveV2,
    build_chess_relation_index,
)

CFG = dict(embed_dim=64, nb_transformer_layers=4, num_heads=4, dropout=0.0, layer_drop=0.0,
           value_head_type="wdl", head_tower_layers=1)
SMOL = dict(smolgen=True, smolgen_compress=4, smolgen_hidden=32, smolgen_gen=16)


def _model(**kw):
    torch.manual_seed(0)
    return Pos2MoveV2(**CFG, **kw)


def _inputs(b=8):
    g = torch.Generator().manual_seed(1)
    return (torch.randint(0, 13, (b, 64), generator=g), torch.randint(0, 2, (b,), generator=g),
            torch.randint(0, 16, (b,), generator=g), torch.randint(0, 9, (b,), generator=g))


def _trained_buckets():
    """A bucket model whose tables are not all zero, so the expansion is exercised."""
    m = _model()
    with torch.no_grad():
        for mod in m.modules():
            if getattr(mod, "bias_table", None) is not None:
                mod.bias_table.normal_(0.0, 0.5)
    return m.eval()


def test_buckets_default_has_no_new_parameters():
    keys = _model().state_dict().keys()
    assert not [k for k in keys if "smolgen" in k or "bias_full" in k]
    assert sum(k.endswith("attn.bias_table") for k in keys) == 5  # 3 shared + 2 tower layers


def test_two_buckets_match_no_square_pair():
    # Why the full table exists: the file/rank/diagonal/knight rules shadow these two.
    counts = torch.bincount(build_chess_relation_index()[:64, :64].flatten(), minlength=8)
    assert counts[5] == 0 and counts[6] == 0


@pytest.mark.parametrize("kw", [dict(attn_bias="full"), SMOL, dict(attn_bias="full", **SMOL)])
def test_bucket_checkpoint_loads_and_plays_the_same(kw):
    old = _trained_buckets()
    new = _model(**kw)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and all("smolgen" in k for k in missing)
    new.eval()
    with torch.no_grad():
        for a, b in zip(old(*_inputs()), new(*_inputs())):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


def test_full_table_is_the_bucket_gather():
    old = _trained_buckets()
    new = _model(attn_bias="full")
    new.load_state_dict(old.state_dict())
    a, b = old.transformer_layers[0].attn, new.transformer_layers[0].attn
    assert torch.equal(b.bias_full, a.bias_table[:, a.rel_idx])


def _grads(m):
    m.zero_grad()
    logits, value = m(*_inputs())
    (logits.float().pow(2).mean() + value.float().pow(2).mean()).backward()
    return m.smolgen_proj.weight.grad, m.transformer_layers[0].attn.smolgen.dense1.weight.grad


def test_smolgen_starts_silent_then_learns():
    # Zero projection: no effect on play, but it gets a gradient, and once it is
    # non-zero the rest of the module does too (the LoRA-style zero start).
    m = _model(**SMOL).train()
    assert torch.count_nonzero(m.smolgen_proj.weight) == 0
    g_proj, g_up = _grads(m)
    assert torch.count_nonzero(g_proj) > 0
    assert g_up is None or torch.count_nonzero(g_up) == 0
    with torch.no_grad():
        m.smolgen_proj.weight.normal_(0.0, 0.01)
    _, g_up = _grads(m)
    assert torch.count_nonzero(g_up) > 0


def test_smolgen_bias_depends_on_the_position():
    m = _model(**SMOL).eval()
    with torch.no_grad():
        m.smolgen_proj.weight.normal_(0.0, 0.1)
        attn = m.transformer_layers[0].attn
        x = torch.randn(2, 67, 64)
        bias = attn.smolgen(x, m.smolgen_proj)
    assert bias.shape == (2, 4, 67, 67)
    assert not torch.allclose(bias[0], bias[1])


def test_unknown_attn_bias_rejected():
    with pytest.raises(ValueError):
        _model(attn_bias="smol")
