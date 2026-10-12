"""Rule-state inputs: the fifty-move counter and the repetition count."""

import json
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2  # noqa: E402

CFG = dict(embed_dim=32, nb_transformer_layers=2, num_heads=4, dropout=0.0, layer_drop=0.0,
           value_head_type="wdl")


def _inputs(b=6):
    g = torch.Generator().manual_seed(1)
    return (torch.randint(0, 13, (b, 64), generator=g), torch.randint(0, 2, (b,), generator=g),
            torch.randint(0, 16, (b,), generator=g), torch.randint(0, 9, (b,), generator=g))


def _rules(b=6):
    return dict(rule50=torch.tensor([0, 7, 40, 99, 100, 3])[:b], repetition=torch.tensor([0, 1, 2, 0, 1, 2])[:b])


def test_converted_model_plays_the_same_until_trained():
    torch.manual_seed(0)
    old = Pos2MoveV2(**CFG).eval()
    new = Pos2MoveV2(**CFG, rule_state=True)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and sorted(missing) == ["repetition_embedding.weight", "rule50_embedding.weight"]
    new.eval()
    with torch.no_grad():
        for a, b in zip(old(*_inputs()), new(*_inputs(), **_rules())):
            assert torch.equal(a, b)


def test_rule_state_reaches_the_heads_only_when_given():
    torch.manual_seed(0)
    m = Pos2MoveV2(**CFG, rule_state=True).eval()
    with torch.no_grad():
        m.rule50_embedding.weight.normal_(0.0, 0.5)
        m.repetition_embedding.weight.normal_(0.0, 0.5)
        base = m(*_inputs())
        ruled = m(*_inputs(), **_rules())
        fresh = m(*_inputs(), rule50=torch.zeros(6, dtype=torch.long), repetition=torch.zeros(6, dtype=torch.long))
    assert not torch.allclose(base[1], ruled[1])
    assert not torch.allclose(ruled[1], fresh[1])
    # rule50 / repetition of None = the input is absent (human shards), not "clock 0"
    assert not torch.allclose(base[1], fresh[1])


def test_old_model_ignores_rule_inputs():
    torch.manual_seed(0)
    m = Pos2MoveV2(**CFG).eval()
    with torch.no_grad():
        assert torch.equal(m(*_inputs())[0], m(*_inputs(), **_rules())[0])


def test_backfill_counts_repetitions_and_the_clock():
    from backfill_rule_state import game_rule_state
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8", "e2e4", "e7e5"]
    toks, r50, rep = game_rule_state(chess.STARTING_FEN, moves)
    assert len(toks) == len(moves)
    # the start position recurs before moves 5 and 9; knight positions recur in between
    assert rep.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 0]
    assert r50.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]  # e2e4 resets the clock


def _shard(path: Path, n: int, rules: bool):
    rng = np.random.default_rng(0)
    cols = dict(boards=rng.integers(0, 13, (n, 64), dtype=np.uint8), player=np.zeros(n, np.uint8),
                castling=np.zeros(n, np.uint8), ep=np.full(n, 8, np.uint8), halfmove=np.arange(n, dtype=np.uint16),
                z=np.zeros(n, np.float32), root_v=np.zeros(n, np.float32), full=np.ones(n, np.uint8),
                game_id=np.repeat(np.arange(n // 2), 2).astype(np.uint32),
                visit_idx=np.zeros(n, np.uint16), visit_cnt=np.ones(n, np.uint16), visit_ptr=np.arange(n + 1))
    if rules:
        cols.update(rule50=np.full(n, 9, np.uint8), repetition=np.ones(n, np.uint8))
    np.savez(path, **cols)


def test_dataset_reads_recorded_or_backfilled_rule_state(tmp_path):
    from train_expert_iter import SelfplayDataset, collate
    _shard(tmp_path / "positions_0000.npz", 4, rules=True)
    _shard(tmp_path / "positions_0001.npz", 4, rules=False)
    ds = SelfplayDataset([tmp_path])
    assert not ds.has_rule_state  # shard 1 has neither columns nor a side file
    np.savez(tmp_path / "rule_state_0001.npz", rule50=np.full(4, 3, np.uint8), repetition=np.zeros(4, np.uint8))
    ds = SelfplayDataset([tmp_path])
    assert ds.has_rule_state
    out = collate([ds[i] for i in range(8)])
    assert out[-2].tolist() == [9] * 4 + [3] * 4 and out[-1].tolist() == [1] * 4 + [0] * 4


def test_selfplay_forward_feeds_the_rule_state(tmp_path):
    from selfplay_rust import Forward
    torch.manual_seed(0)
    m = Pos2MoveV2(**CFG, rule_state=True)
    with torch.no_grad():
        m.rule50_embedding.weight.normal_(0.0, 0.5)
        m.repetition_embedding.weight.normal_(0.0, 0.5)
    save_file(m.state_dict(), str(tmp_path / "model.safetensors"))
    (tmp_path / "model_config.json").write_text(json.dumps({**CFG, "rule_state": True}))
    fwd = Forward(str(tmp_path), "cpu", False, False, buckets=(8, 16))
    assert fwd.rule_state
    x = [t.numpy().astype(np.uint8) for t in _inputs()]
    r = {k: v.numpy().astype(np.uint8) for k, v in _rules().items()}
    logits, values = fwd(*x, **r)
    with torch.no_grad():
        ref_logits, ref_v = m.bfloat16().eval()(*_inputs(), **_rules())
    np.testing.assert_allclose(values, m.value_scalar(ref_v.float()).squeeze(-1).numpy(), atol=2e-2)
    legal = np.tile(np.arange(5, dtype=np.int16), (6, 1))
    got, got_v = fwd.wait(fwd.launch(0, *x, legal, r["rule50"], r["repetition"]))
    np.testing.assert_allclose(got, logits.reshape(6, -1)[:, :5], atol=1e-2)
    np.testing.assert_allclose(got_v, values, atol=1e-3)


def test_selfplay_engine_records_the_rule_state():
    selfplay_core = pytest.importorskip("selfplay_core")
    if not hasattr(selfplay_core.SelfPlayEngine, "rule_state"):
        pytest.skip("selfplay_core wheel predates rule-state records")
    cfg = {"num_sims": 8, "num_parallel_games": 2, "max_plies": 12, "temp_plies": 12, "threads": 1,
           "rule_state": True}
    eng = selfplay_core.SelfPlayEngine(json.dumps(cfg), [chess.STARTING_FEN], 0, 2)
    games = []
    while not eng.done():
        boards, player, castling, ep, legal = eng.collect()
        r50, rep = eng.rule_state()
        assert len(r50) == len(rep) == len(player)
        eng.apply_legal(np.zeros(legal.shape, np.float32), np.zeros(len(player), np.float32))
        games += eng.drain_finished()
    from backfill_rule_state import game_rule_state
    for g in games:
        _, r50, rep = game_rule_state(g["fen"], g["moves"])
        assert g["rule50"].tolist() == r50.tolist() and g["repetition"].tolist() == rep.tolist()
