"""Pentanomial scoring and sequential testing for engine matches.

Both match harnesses in this repo play every opening twice with the colours
swapped, then throw that structure away and treat the games as 2N independent
samples. Two consequences, both visible in the logs:

**The pairing is wasted.** Write the two games of one opening as ``g1, g2`` and
the pair average as ``x = (g1+g2)/2``. Then ``Var(x) = sigma_g^2 (1+rho)/2``
where ``rho`` is the within-pair correlation, so the standard error of a
pair-scored match is ``(1+rho)`` times the per-game one. When an opening is
simply good for White, A wins it with White and loses it with Black: ``rho`` is
negative and the opening's difficulty cancels out of the estimate instead of
inflating it. That is the whole of the pentanomial trick -- it is a re-grouping
of data the harness already collects, not extra games. The size of the gain
depends on the book, so :func:`format_report` prints the per-game interval
alongside the paired one and lets you see it rather than asserting a number.

**Fixed-N matches at these sizes cannot resolve the effects being chased.**
Measured on this repo: 128 games gave -19 Elo (CI -54..+16) and -24 Elo
(CI -61..+11). A sequential test spends games only until the answer is clear,
and -- more importantly -- reports *undecided* instead of inviting a point
estimate to be read as a result.

One caveat the caller must respect: at a fixed node budget this repo's search is
deterministic, so replaying the same opening reproduces the same game exactly.
Games are only independent samples if the openings differ. Growing the book is
what buys statistical power here; SPRT only decides when to stop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Pentanomial buckets: a pair of games scores 0, 0.5, 1, 1.5 or 2 for side A.
PAIR_BUCKETS = (0.0, 0.5, 1.0, 1.5, 2.0)
_BUCKET_LABELS = ("LL", "LD", "LW/DD", "DW", "WW")


def elo_to_score(elo: float) -> float:
    """Expected per-game score for an Elo advantage."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    """Elo difference implied by a per-game score. Undefined at 0 and 1."""
    score = min(max(score, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


@dataclass
class MatchStats:
    """Everything a caller needs to report a match, paired and unpaired."""

    n_pairs: int
    n_games: int
    counts: tuple[int, int, int, int, int]  # LL, LD, LW/DD, DW, WW
    score: float                            # per-game score for side A
    stderr: float                           # standard error of `score`, paired
    stderr_unpaired: float                  # same, scoring games independently
    variance: float                         # variance of the pair average

    @property
    def elo(self) -> float:
        return score_to_elo(self.score)

    def interval(self, z: float = 1.96) -> tuple[float, float]:
        """Confidence interval on the per-game score (paired)."""
        return (max(1e-9, self.score - z * self.stderr),
                min(1 - 1e-9, self.score + z * self.stderr))

    def elo_interval(self, z: float = 1.96) -> tuple[float, float]:
        lo, hi = self.interval(z)
        return score_to_elo(lo), score_to_elo(hi)


def pair_up(results: dict[tuple[int, bool], float]) -> list[float]:
    """Collapse per-game scores into pair scores in [0, 2].

    ``results`` maps ``(opening_index, a_played_white) -> score for A``. Only
    openings with both colours played contribute; a half-finished pair is
    dropped rather than counted as a partial game, which keeps the estimator
    unbiased if a match is interrupted mid-pair.
    """
    pairs = []
    indices = {idx for idx, _ in results}
    for idx in sorted(indices):
        white = results.get((idx, True))
        black = results.get((idx, False))
        if white is not None and black is not None:
            pairs.append(white + black)
    return pairs


def summarize(pairs: list[float], games: list[float] | None = None) -> MatchStats:
    """Pentanomial summary of a list of pair scores.

    ``games`` is the flat list of per-game scores behind those pairs. It is used
    only for the unpaired standard error printed as a comparison, and cannot be
    reconstructed from ``pairs`` -- a pair scoring 1.0 is either two draws or a
    win and a loss, and those have different per-game variance. Omit it and the
    comparison is simply not reported.
    """
    n = len(pairs)
    if n == 0:
        return MatchStats(0, 0, (0, 0, 0, 0, 0), 0.5, 0.0, 0.0, 0.0)

    counts = tuple(sum(1 for p in pairs if abs(p - b) < 1e-9) for b in PAIR_BUCKETS)

    # Work in per-game units so `score` is directly comparable to the old
    # reporting: x_i is the pair average, already in [0, 1].
    xs = [p / 2.0 for p in pairs]
    score = sum(xs) / n
    var = sum((x - score) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    stderr = math.sqrt(var / n)

    stderr_unpaired = 0.0
    m = 2 * n
    if games:
        m = len(games)
        mean_g = sum(games) / m
        var_g = sum((g - mean_g) ** 2 for g in games) / (m - 1) if m > 1 else 0.0
        stderr_unpaired = math.sqrt(var_g / m)

    return MatchStats(n, m, counts, score, stderr, stderr_unpaired, var)


def sprt_llr(pairs: list[float], elo0: float = 0.0, elo1: float = 15.0) -> float:
    """Generalized-SPRT log-likelihood ratio for H0: elo0 vs H1: elo1.

    The normal approximation used by Fishtest: for a variable with sample mean
    ``x`` and variance ``s2``, ``LLR = n (mu1-mu0) (x - (mu0+mu1)/2) / s2``.
    Applied to the *pair average* so the pentanomial variance reduction feeds
    straight into the test.

    Returns 0.0 while the estimate carries no information (fewer than two pairs,
    or zero observed variance -- every pair identical, which happens early in an
    all-draw match and would otherwise divide by zero).
    """
    n = len(pairs)
    if n < 2:
        return 0.0
    xs = [p / 2.0 for p in pairs]
    x = sum(xs) / n
    s2 = sum((v - x) ** 2 for v in xs) / (n - 1)
    if s2 <= 1e-12:
        return 0.0
    mu0, mu1 = elo_to_score(elo0), elo_to_score(elo1)
    return n * (mu1 - mu0) * (x - (mu0 + mu1) / 2.0) / s2


def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> tuple[float, float]:
    """Wald boundaries (lower, upper) on the LLR."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def sprt_decision(llr: float, alpha: float = 0.05, beta: float = 0.05,
                  min_pairs_met: bool = True) -> str | None:
    """``"H0"``, ``"H1"`` or ``None`` (keep playing).

    ``min_pairs_met`` lets the caller refuse to stop before a floor number of
    pairs. The normal approximation behind :func:`sprt_llr` is poor on a handful
    of samples, and an early crossing on 3 pairs is an artefact, not a result.
    """
    if not min_pairs_met:
        return None
    lo, hi = sprt_bounds(alpha, beta)
    if llr >= hi:
        return "H1"
    if llr <= lo:
        return "H0"
    return None


def format_report(stats: MatchStats, label_a: str, label_b: str,
                  llr: float | None = None, elo0: float = 0.0, elo1: float = 15.0,
                  alpha: float = 0.05, beta: float = 0.05) -> str:
    """The block both harnesses print at the end of a match."""
    lines = []
    lines.append(
        f"{label_a} vs {label_b}: {stats.n_games} games in {stats.n_pairs} pairs"
    )
    lines.append(
        "  pentanomial [LL, LD, LW/DD, DW, WW] = "
        f"[{', '.join(str(c) for c in stats.counts)}]"
    )
    lines.append(f"  score {stats.score:.4f}")

    if not 0.0 < stats.score < 1.0:
        lines.append(f"  {label_a} - {label_b} = decisive")
        return "\n".join(lines)

    lo, hi = stats.elo_interval()
    lines.append(
        f"  {label_a} - {label_b} ~ {stats.elo:+.0f} Elo "
        f"(95% CI {lo:+.0f} .. {hi:+.0f}, +/-{1.96 * stats.stderr * 100:.2f}pp)"
    )
    if stats.stderr > 0 and stats.stderr_unpaired > 0:
        ratio = stats.stderr_unpaired / stats.stderr
        lines.append(
            f"  pairing gain: unpaired SE would be "
            f"+/-{1.96 * stats.stderr_unpaired * 100:.2f}pp ({ratio:.2f}x wider)"
        )
    s_lo, s_hi = stats.interval()
    if s_lo < 0.5 < s_hi:
        lines.append(
            f"  NOT SIGNIFICANT: the interval spans 0 Elo. "
            f"{stats.n_games} games cannot separate these two."
        )

    if llr is not None:
        b_lo, b_hi = sprt_bounds(alpha, beta)
        verdict = sprt_decision(llr, alpha, beta)
        lines.append(
            f"  SPRT H0={elo0:+.0f} vs H1={elo1:+.0f} Elo "
            f"(alpha={alpha}, beta={beta}): LLR {llr:+.2f} "
            f"in [{b_lo:+.2f}, {b_hi:+.2f}]"
        )
        if verdict == "H1":
            lines.append(f"  ACCEPT H1: {label_a} is stronger than {elo0:+.0f} Elo.")
        elif verdict == "H0":
            lines.append(f"  ACCEPT H0: {label_a} does not reach {elo1:+.0f} Elo.")
        else:
            lines.append("  UNDECIDED: neither boundary reached; play more openings.")
    return "\n".join(lines)
