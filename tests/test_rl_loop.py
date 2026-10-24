"""The RL loop's self-play opening mix."""

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_selfplay_mixes_book_openings_past_the_gates_ones(tmp_path):
    selfplay_rust = pytest.importorskip("selfplay_rust")
    book = tmp_path / "book.json"
    book.write_text(json.dumps({"openings": [{"fen": f"book {i}"} for i in range(2000)]}))

    def openings(**kw):
        args = dict(book=str(book), book_skip=500, book_share=0.75, worker=None)
        return selfplay_rust.selfplay_openings(argparse.Namespace(**{**args, **kw}))

    mix = openings()
    from_book = [o for o in mix if o.startswith("book ")]
    assert {int(o.split()[1]) for o in from_book} == set(range(500, 2000))
    assert len(from_book) / len(mix) == pytest.approx(0.75, abs=0.02)
    assert mix[0].startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w")  # the start position
    assert not any(o.startswith("book ") for o in openings(book=None))
    assert all(o.startswith("book ") for o in openings(book_share=1.0))
