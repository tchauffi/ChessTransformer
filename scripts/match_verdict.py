"""A match file's verdict for the RL loop: score, paired Elo with its 95% interval, the SPRT.

    python scripts/match_verdict.py data/rl/gates/cand15_vs_gen14.jsonl [--elo0 0 --elo1 15]

Prints one tab-separated line on stdout,

    score  games  pairs  elo  lo  hi  llr  decision

with the scores of side A (the file's score_a), the paired Elo and its 95% interval, the
SPRT's log-likelihood ratio for H0 = elo0 vs H1 = elo1, and the decision: H1 or H0 once the
LLR is past a Wald bound (alpha = beta = 0.05), else "-". The match report goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chesstransformer.evaluation.sprt import (  # noqa: E402
    format_report, pair_up, sprt_decision, sprt_llr, summarize)


def verdict(path: Path, elo0: float, elo1: float) -> dict:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    res = {(r["idx"], r["a_white"]): r["score_a"] for r in rows}
    pairs = pair_up(res)
    stats = summarize(pairs, [res[k] for k in res if (k[0], not k[1]) in res])
    llr = sprt_llr(pairs, elo0, elo1)
    lo, hi = stats.elo_interval() if pairs else (0.0, 0.0)
    return {"score": sum(r["score_a"] for r in rows) / max(1, len(rows)), "games": len(rows),
            "pairs": len(pairs), "elo": stats.elo if pairs else 0.0, "lo": lo, "hi": hi,
            "llr": llr, "decision": sprt_decision(llr) if pairs else None, "stats": stats}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("games", type=Path)
    p.add_argument("--elo0", type=float, default=0.0)
    p.add_argument("--elo1", type=float, default=15.0)
    p.add_argument("--labels", nargs=2, default=("A", "B"))
    args = p.parse_args()
    v = verdict(args.games, args.elo0, args.elo1)
    print(format_report(v["stats"], *args.labels, llr=v["llr"], elo0=args.elo0, elo1=args.elo1),
          file=sys.stderr)
    print(f"{v['score']:.4f}\t{v['games']}\t{v['pairs']}\t{v['elo']:.1f}\t{v['lo']:.1f}\t"
          f"{v['hi']:.1f}\t{v['llr']:.3f}\t{v['decision'] or '-'}")


if __name__ == "__main__":
    main()
