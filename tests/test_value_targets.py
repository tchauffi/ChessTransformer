"""Value-target corrections: the rating-gap offset, shard format v2, tablebase labels."""

import json
import math
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chesstransformer.datasets.flat_shard_dataset import FlatShardDataset  # noqa: E402
from chesstransformer.datasets.shard_format import RECORD_DTYPE, dtype_to_meta  # noqa: E402
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2  # noqa: E402
from chesstransformer.trainers.pos2move_v2_trainer import unpack_batch  # noqa: E402

SYZYGY = ROOT / "data" / "syzygy"


def _model(**kw):
    torch.manual_seed(0)
    return Pos2MoveV2(embed_dim=64, nb_transformer_layers=2, num_heads=4, dropout=0.0,
                      value_head_type="wdl", **kw).eval()


def _inputs(b=4):
    g = torch.Generator().manual_seed(1)
    return (torch.randint(0, 13, (b, 64), generator=g), torch.randint(0, 2, (b,), generator=g),
            torch.randint(0, 16, (b,), generator=g), torch.randint(0, 9, (b,), generator=g))


def test_offset_needs_wdl_head():
    with pytest.raises(ValueError):
        Pos2MoveV2(embed_dim=64, nb_transformer_layers=1, num_heads=4, value_elo_offset=True)


def test_no_gap_is_the_plain_head():
    """Inference passes no gap; the offset model must then equal the same weights without it."""
    with_offset, plain = _model(value_elo_offset=True), _model()
    plain.load_state_dict({k: v for k, v in with_offset.state_dict().items() if k != "elo_beta"})
    x = _inputs()
    with torch.no_grad():
        assert torch.equal(with_offset(*x)[1], plain(*x)[1])
        # And a model without the offset ignores a gap entirely.
        assert torch.equal(plain(*x, elo_gap=torch.full((4,), 300.0))[1], plain(*x)[1])


def test_offset_matches_the_elo_model_at_init():
    """At init the win/loss log-odds move by exactly the Elo logistic slope, ln(10)/400 per
    point, and the draw logit does not move."""
    m = _model(value_elo_offset=True)
    x = _inputs()
    gap = torch.tensor([400.0, -400.0, 0.0, 123.0])
    with torch.no_grad():
        base, shifted = m(*x)[1], m(*x, elo_gap=gap)[1]
    d = shifted - base
    log_odds_shift = d[:, 0] - d[:, 2]
    assert torch.allclose(log_odds_shift, gap * math.log(10) / 400, atol=1e-5)
    assert torch.equal(d[:, 1], torch.zeros(4))


def _write_shard_dir(out: Path, dtype: np.dtype, version: int, rows: np.ndarray):
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "shard_0000.npy", rows)
    meta_dtype = dtype_to_meta() if dtype == RECORD_DTYPE else {
        "names": list(dtype.names),
        "formats": [[dtype[n].subdtype[0].str, dtype[n].subdtype[1][0]] if dtype[n].subdtype
                    else dtype[n].str for n in dtype.names],
        "offsets": [dtype.fields[n][1] for n in dtype.names], "itemsize": dtype.itemsize}
    (out / "meta.json").write_text(json.dumps({
        "format_version": version, "record_dtype": meta_dtype, "pad_action": 4672,
        "splits": {"train": [0]},
        "shards": [{"shard_id": 0, "path": "shard_0000.npy", "game_start": 0, "game_end": 1,
                    "num_samples": len(rows)}]}))


def test_v1_shards_still_load(tmp_path):
    """v2 only appended fields, so a v1 directory must keep loading -- without them."""
    v1 = np.dtype({"names": RECORD_DTYPE.names[:10],
                   "formats": [RECORD_DTYPE.fields[n][0] for n in RECORD_DTYPE.names[:10]],
                   "offsets": [RECORD_DTYPE.fields[n][1] for n in RECORD_DTYPE.names[:10]],
                   "itemsize": 202})
    rows = np.zeros(3, dtype=v1)
    rows["result"] = [0, 1, 2]
    _write_shard_dir(tmp_path / "v1", v1, 1, rows)
    ds = FlatShardDataset(tmp_path / "v1")
    batch = ds[[0, 1, 2]]
    assert batch["result"].tolist() == [0, 1, 2]
    assert "elo_gap" not in batch and ds.optional_fields == ()


def test_v2_fields_and_label_selection(tmp_path):
    rows = np.zeros(3, dtype=RECORD_DTYPE)
    rows["result"] = [1, 1, 0]
    rows["result_tb"] = [1, 0, 2]          # sample 1 adjudicated, sample 2 exact
    rows["value_exact"] = [0, 0, 1]
    rows["elo_gap"] = [150, -80, 200]
    rows["legal_idx"] = 4672
    _write_shard_dir(tmp_path / "v2", RECORD_DTYPE, 2, rows)
    batch = FlatShardDataset(tmp_path / "v2")[[0, 1, 2]]

    human = unpack_batch(batch, "cpu")
    assert human["result"].tolist() == [1, 1, 0] and human["elo_gap"] is None

    tb = unpack_batch(batch, "cpu", value_labels="tablebase", elo_offset=True)
    assert tb["result"].tolist() == [1, 0, 2]
    # The exact sample is perfect play: who was playing must not shift its target.
    assert tb["elo_gap"].tolist() == [150.0, -80.0, 0.0]

    human_gap = unpack_batch(batch, "cpu", value_labels="human", elo_offset=True)
    assert human_gap["elo_gap"].tolist() == [150.0, -80.0, 200.0]


@pytest.fixture(scope="module")
def tb_builder():
    if not (SYZYGY / "KQvKR.rtbw").exists():
        pytest.skip("Syzygy tables not in data/syzygy")
    import build_shards
    build_shards._worker_init({"syzygy": str(SYZYGY)})
    return build_shards


@pytest.mark.parametrize("fen, expected", [
    ("8/8/8/4k3/8/8/8/KQ6 w - - 0 1", 1),         # KQ v K, white to move: white wins
    ("8/8/8/4k3/8/8/8/KQ6 b - - 0 1", 1),         # same, black to move: still white wins
    ("8/8/8/4k3/8/8/2r5/K7 w - - 0 1", 2),        # K v KR: black wins
    ("8/8/8/4k3/8/2r5/8/KR6 w - - 0 1", 0),       # KR v KR: draw
    ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 1),        # black is checkmated
    ("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", None),     # castling rights: tables cannot answer
])
def test_tb_result(tb_builder, fen, expected):
    assert tb_builder._tb_result(chess.Board(fen)) == expected


def test_tb_result_respects_the_fifty_move_rule(tb_builder):
    """This KQ v KR win takes 47 plies to the next capture or pawn move (DTZ 47): still a
    win with 50 plies already on the clock, a draw with 60."""
    fen = "8/8/3k4/8/8/2r5/8/4K2Q w - - {} 80"
    assert tb_builder._tb_result(chess.Board(fen.format(0))) == 1
    assert tb_builder._tb_result(chess.Board(fen.format(50))) == 1
    assert tb_builder._tb_result(chess.Board(fen.format(60))) == 0
