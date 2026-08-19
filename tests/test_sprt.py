"""Statistical properties of the pentanomial / SPRT match reporting.

These are the guards that matter: the whole point of the module is that the
numbers it prints are trustworthy, so the tests check calibration and error
rates rather than exercising the API.
"""

from __future__ import annotations

import math
import random

import pytest

from chesstransformer.evaluation.sprt import (
    MatchStats,
    elo_to_score,
    format_report,
    pair_up,
    score_to_elo,
    sprt_bounds,
    sprt_decision,
    sprt_llr,
    summarize,
)


# --- Elo <-> score round trip -------------------------------------------------

@pytest.mark.parametrize("elo", [-400, -100, -15, 0, 15, 100, 400])
def test_elo_score_round_trip(elo):
    assert score_to_elo(elo_to_score(elo)) == pytest.approx(elo, abs=1e-6)


def test_even_score_is_zero_elo():
    assert elo_to_score(0.0) == pytest.approx(0.5)
    assert score_to_elo(0.5) == pytest.approx(0.0)


# --- Pairing ------------------------------------------------------------------

def test_pair_up_matches_colours():
    results = {(0, True): 1.0, (0, False): 0.5, (1, True): 0.0, (1, False): 0.0}
    assert pair_up(results) == [1.5, 0.0]


def test_pair_up_drops_half_finished_pairs():
    """An interrupted match must not contribute a one-sided opening."""
    results = {(0, True): 1.0, (0, False): 0.5, (1, True): 1.0}
    assert pair_up(results) == [1.5]


# --- Pentanomial bookkeeping --------------------------------------------------

def test_counts_and_score():
    pairs = [0.0, 0.5, 1.0, 1.0, 1.5, 2.0]
    st = summarize(pairs)
    assert st.counts == (1, 1, 2, 1, 1)
    assert st.n_pairs == 6
    assert st.score == pytest.approx(sum(pairs) / 2 / 6)


def test_pairing_beats_per_game_when_openings_are_lopsided():
    """The case pairing exists for: colour-symmetric openings of varied difficulty.

    Every pair is a 1-0 / 0-1 split, so A and B are exactly even and the paired
    estimator sees zero variance, while the per-game estimator sees the full
    swing of every decisive game.
    """
    pairs = [1.0] * 20
    games = [1.0, 0.0] * 20
    st = summarize(pairs, games)
    assert st.score == pytest.approx(0.5)
    assert st.stderr == pytest.approx(0.0, abs=1e-12)
    assert st.stderr_unpaired > 0.05


def test_summarize_without_games_reports_no_pairing_gain():
    st = summarize([1.0, 0.5, 1.5])
    assert st.stderr_unpaired == 0.0
    assert "pairing gain" not in format_report(st, "A", "B")


def test_empty_match_is_inert():
    st = summarize([])
    assert st.n_pairs == 0 and st.score == pytest.approx(0.5)
    assert sprt_llr([]) == 0.0


# --- SPRT ---------------------------------------------------------------------

def test_llr_is_zero_without_variance():
    """All pairs identical: no information about the mean, and no divide by zero."""
    assert sprt_llr([1.0] * 50, 0.0, 15.0) == 0.0


def test_llr_sign_follows_the_evidence():
    strong = [2.0, 1.5, 2.0, 1.5, 1.5, 2.0, 1.0, 1.5] * 5
    weak = [0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 1.0, 0.5] * 5
    assert sprt_llr(strong, 0.0, 15.0) > 0
    assert sprt_llr(weak, 0.0, 15.0) < 0


def test_bounds_are_symmetric_for_equal_error_rates():
    lo, hi = sprt_bounds(0.05, 0.05)
    assert lo == pytest.approx(-hi)
    assert hi == pytest.approx(math.log(0.95 / 0.05))


def test_decision_respects_the_minimum_pair_floor():
    _, hi = sprt_bounds()
    assert sprt_decision(hi + 1, min_pairs_met=False) is None
    assert sprt_decision(hi + 1, min_pairs_met=True) == "H1"


def _simulate(elo_true: float, n_pairs: int, rng: random.Random,
              draw_rate: float = 0.66) -> list[float]:
    """A match between engines separated by `elo_true`, as pair scores.

    Draw rate is held fixed and the decisive mass is placed so the *mean* game
    score comes out at `elo_to_score(elo_true)`. Splitting the decisive mass
    proportionally instead would silently generate a much smaller edge than
    requested, since the draws pull the mean back toward 0.5.
    """
    target = elo_to_score(elo_true)
    p_a = target - 0.5 * draw_rate
    p_b = (1.0 - draw_rate) - p_a
    assert p_a >= 0 and p_b >= 0, "draw_rate too high for this Elo gap"
    pairs = []
    for _ in range(n_pairs):
        total = 0.0
        for _ in range(2):
            r = rng.random()
            total += 1.0 if r < p_a else (0.0 if r < p_a + p_b else 0.5)
        pairs.append(total)
    return pairs


def _sequential_run(elo_true: float, seed: int, max_pairs: int = 4000,
                    elo1: float = 15.0) -> str | None:
    rng = random.Random(seed)
    pairs: list[float] = []
    for i in range(max_pairs):
        pairs.extend(_simulate(elo_true, 1, rng))
        verdict = sprt_decision(sprt_llr(pairs, 0.0, elo1), min_pairs_met=i >= 20)
        if verdict:
            return verdict
    return None


def test_type_i_error_is_near_alpha_under_the_null():
    """Two identical engines must rarely be declared different."""
    runs = [_sequential_run(0.0, seed) for seed in range(120)]
    false_positives = sum(1 for v in runs if v == "H1")
    # alpha = 0.05; allow slack for 120 trials and the normal approximation.
    assert false_positives / len(runs) < 0.15


def test_power_against_a_clearly_stronger_engine():
    """A 40 Elo edge should be caught almost every time, and never mis-signed."""
    runs = [_sequential_run(40.0, seed + 500) for seed in range(60)]
    assert sum(1 for v in runs if v == "H1") / len(runs) > 0.80
    assert not any(v == "H0" for v in runs if v is not None and v == "H1")


def test_a_weaker_engine_is_rejected():
    runs = [_sequential_run(-30.0, seed + 900) for seed in range(60)]
    assert sum(1 for v in runs if v == "H0") / len(runs) > 0.85


def test_confidence_interval_covers_the_truth():
    """The 95% interval should contain the true Elo about 95% of the time."""
    rng = random.Random(7)
    covered = 0
    trials = 200
    for _ in range(trials):
        pairs = _simulate(25.0, 300, rng)
        lo, hi = summarize(pairs).elo_interval()
        covered += lo <= 25.0 <= hi
    assert covered / trials > 0.88


# --- Reporting ----------------------------------------------------------------

def test_report_flags_an_undecided_match():
    text = format_report(summarize([1.0, 0.5, 1.5, 1.0]), "A", "B", llr=0.3)
    assert "UNDECIDED" in text
    assert "pentanomial" in text


def test_report_handles_a_shutout_without_dividing_by_zero():
    text = format_report(summarize([2.0] * 5), "A", "B", llr=0.0)
    assert "decisive" in text
