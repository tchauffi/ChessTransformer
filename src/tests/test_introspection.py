"""The introspection maps are only useful if they match the real forward pass."""

import math

import chess
import pytest
import torch

from chesstransformer.introspection.analyze import PositionAnalyzer
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2

TOKENS = 67


@pytest.fixture(scope="module")
def analyzer():
    # A small randomly-initialised model: this checks the plumbing, which is
    # independent of the weights, without depending on a checkpoint on disk.
    torch.manual_seed(0)
    model = Pos2MoveV2(embed_dim=64, nb_transformer_layers=3, num_heads=4, dropout=0.0).eval()
    return PositionAnalyzer(model, device="cpu")


def test_recomputed_attention_reproduces_the_module_output(analyzer):
    """probs @ V, run back through the output projection, must equal what SDPA returned."""
    board = chess.Board()
    for layer_idx, layer in enumerate(analyzer.model.transformer_layers):
        attn = layer.attn
        captured = {}
        h1 = attn.register_forward_pre_hook(lambda _m, a: captured.__setitem__("x", a[0].detach()))
        h2 = attn.register_forward_hook(lambda _m, _a, o: captured.__setitem__("out", o.detach()))
        try:
            analyzer._forward_batch([board])
        finally:
            h1.remove()
            h2.remove()

        with torch.no_grad():
            x = captured["x"]
            probs = analyzer._attention_probs(layer_idx, x).unsqueeze(0)
            b, t, _ = x.shape
            v = attn.v_proj(x).view(b, t, attn.num_kv_groups, attn.head_dim).transpose(1, 2)
            v = v.repeat_interleave(attn.group_size, dim=1)
            ctx = (probs @ v).transpose(1, 2).contiguous().view(b, t, attn.d_out)
            rebuilt = attn.proj(ctx)

        assert torch.allclose(rebuilt, captured["out"], atol=1e-5), f"layer {layer_idx} attention drifted"
        assert torch.allclose(probs.sum(-1), torch.ones(1), atol=1e-5)


def test_analysis_shapes_and_policy_normalisation(analyzer):
    rec = analyzer.analyze(chess.Board(), include_attention=True)

    assert rec["legal_count"] == 20
    probs = [m["prob"] for m in rec["policy"]["moves"]]
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-5)
    assert probs == sorted(probs, reverse=True)
    assert 0.0 <= rec["policy"]["legal_mass"] <= 1.0
    assert rec["policy"]["entropy"] <= rec["policy"]["max_entropy"] + 1e-6

    depth = rec["depth"]
    n = analyzer.n_layers
    assert len(depth["value_by_layer"]) == n
    assert len(depth["top_by_layer"]) == n
    # The last layer's lens sees the same residual the real heads do.
    assert depth["top_by_layer"][-1]["uci"] == rec["policy"]["moves"][0]["uci"]

    assert rec["attention"]["shape"] == [n, analyzer.n_heads, TOKENS, TOKENS]
    assert len(rec["attention"]["rollout"]) == TOKENS
    assert math.isclose(sum(rec["attention"]["rollout"]), 1.0, rel_tol=1e-3)


def test_child_values_are_from_the_movers_point_of_view(analyzer):
    # Mate in one: the child of Qxf7# is checkmate, worth +1 to the mover.
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    moves = analyzer.analyze(board, include_attention=False)["policy"]["moves"]
    analyzer.child_values(board, moves)

    mate = next(m for m in moves if m["uci"] == "d8h4")
    assert mate["child_value"] == 1.0
    assert all(-1.0 <= m["child_value"] <= 1.0 for m in moves)


def test_occlusion_saliency_leaves_kings_alone(analyzer):
    rec = analyzer.analyze(chess.Board(), include_attention=False)
    sal = rec["saliency"]
    assert chess.E1 not in sal["squares"] and chess.E8 not in sal["squares"]
    assert len(sal["squares"]) == 30  # 32 pieces, minus the two kings
    assert sal["value_delta"][chess.E1] == 0.0
