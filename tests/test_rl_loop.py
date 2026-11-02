"""The RL loop's decisions (scripts/match_verdict.py) and self-play's opening mix."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from match_verdict import verdict  # noqa: E402

PAIR = {0: (0, 0), 0.5: (0.5, 0), 1: (0.5, 0.5), 1.5: (1, 0.5), 2: (1, 1)}


def _games(path: Path, penta: list[int]) -> Path:
    """A games file whose pairs follow the pentanomial [LL, LD, LW/DD, DW, WW] for side A."""
    i = 0
    with open(path, "w") as f:
        for n, pair in zip(penta, (0, 0.5, 1, 1.5, 2)):
            for _ in range(n):
                for a_white, s in zip((True, False), PAIR[pair]):
                    f.write(json.dumps({"idx": i, "a_white": a_white, "score_a": s}) + "\n")
                i += 1
    return path


def test_the_gate_accepts_a_stronger_candidate_and_rejects_a_weaker_one(tmp_path):
    strong = verdict(_games(tmp_path / "s.jsonl", [1, 10, 67, 25, 4]), 0, 15)  # cand14's gate, +34
    assert strong["decision"] == "H1" and strong["lo"] > 0
    weak = verdict(_games(tmp_path / "w.jsonl", [4, 25, 67, 10, 1]), 0, 15)
    assert weak["decision"] == "H0" and weak["elo"] < 0
    even = verdict(_games(tmp_path / "e.jsonl", [2, 31, 126, 37, 4]), 0, 15)  # cand04's +9
    assert even["decision"] is None


def test_anchor_non_regression_against_the_champions_elo(tmp_path):
    f = _games(tmp_path / "a.jsonl", [7, 46, 59, 14, 2])  # gen00 vs v2.1: -58 Elo
    assert verdict(f, 0, 15)["elo"] == pytest.approx(-57.5, abs=0.5)
    assert verdict(f, -78, -58)["llr"] > 0     # champion at -58: holds
    assert verdict(f, -137, -117)["llr"] > 0   # champion at -117: holds
    assert verdict(f, -20, 0)["llr"] < 0       # champion at 0: lost ground


def test_the_command_line_takes_negative_bounds(tmp_path):
    f = _games(tmp_path / "a.jsonl", [7, 46, 59, 14, 2])
    out = subprocess.run([sys.executable, str(ROOT / "scripts/match_verdict.py"), str(f),
                          "--elo0=-77.5", "--elo1=-57.5"], capture_output=True, text=True, check=True)
    score, games, pairs, elo, lo, hi, llr, decision = out.stdout.split("\t")
    assert (games, pairs, decision.strip()) == ("256", "128", "-") and float(llr) > 0


def test_the_verdict_judges_the_openings_done_from_the_first_one_on(tmp_path):
    f = _games(tmp_path / "g.jsonl", [1, 10, 67, 25, 4])
    rows = f.read_text().splitlines()
    first = [r for r in rows if json.loads(r)["idx"] == 0]
    f.write_text("\n".join(r for r in rows if r not in first[1:]) + "\n")  # opening 0 unfinished
    v = verdict(f, 0, 15)
    assert (v["games"], v["pairs"], v["decision"]) == (0, 0, None)


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
