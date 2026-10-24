"""Upload every champion of the RL loop to a private Hugging Face model repo.

A champion is a network that held data/rl/champion: gen00 (the seed), every promoted
candidate (history.tsv "yes") and every network installed without a gate ("forced").
Each one goes to its own folder, genNN/: weights, config, ONNX graph, training metadata,
and champion.json with how it got there (its gate, what it trained on, its strength
chained through the gates, as on the status page). The repo's README lists them all and
champion.json at the root names the current one. Rejected candidates stay local.

Generations already on the hub are skipped, so it runs on a timer (ct-hf-sync.timer,
hourly) and catches up after a failed upload. It only reads data/rl/: the loop never
waits on the network.

    .venv/bin/python scripts/hf_sync_champions.py              # tchauffi/chesstransformer-rl
    .venv/bin/python scripts/hf_sync_champions.py --dry-run    # list what would be uploaded

Download one:  hf download tchauffi/chesstransformer-rl --include "gen14/*" --local-dir .

It needs a Hugging Face login with write access (`hf auth login`). The user units:

    # ~/.config/systemd/user/ct-hf-sync.service
    [Service]
    Type=oneshot
    WorkingDirectory=%h/Documents/dev/ChessTransformer
    ExecStart=%h/Documents/dev/ChessTransformer/.venv/bin/python scripts/hf_sync_champions.py
    Nice=10

    # ~/.config/systemd/user/ct-hf-sync.timer
    [Timer]
    OnCalendar=hourly
    Persistent=true
    [Install]
    WantedBy=timers.target
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_status import RL, history, ladder, signed, verdict  # noqa: E402

FILES = ("model.safetensors", "model_config.json", "model.onnx", "expert_iter_meta.json", "SOURCE")


def champions() -> list[dict]:
    """gen00 and every promoted or installed generation, oldest first."""
    rows = history()
    lad = ladder(rows, max((r["n"] for r in rows), default=0)) or {"champion": [], "origin": None}
    chained = {c["gen"]: c for c in lad["champion"]}
    out = [{"gen": "gen00", "n": 0, "promoted": "seed", "base": None, "gate": None,
            "trained_on": [], "finished": None, "hours": None, "games": None}]
    out += [r for r in rows if r["promoted"] in ("yes", "forced")]
    for c in out:
        ch = chained.get(c["gen"])
        c["chained"] = ({"elo": ch["elo"], "ci95": ch["ci"], "origin": lad["origin"]}
                        if ch else None)
    return out


def card(c: dict) -> dict:
    """champion.json for one generation."""
    g = c["gate"]
    return {
        "generation": c["gen"],
        "how": {"seed": "seed of the loop", "forced": "installed without a gate",
                "yes": "promoted by its gate"}[c["promoted"]],
        "champion_since": str(c["finished"]) if c["finished"] else None,
        "candidate_of": c["base"],
        "trained_on": c["trained_on"],
        "gate": None if not g else {
            "vs": c["base"], "games": g["games"], "pairs": g["pairs"],
            "elo": g["elo"], "elo_ci95": [g["lo"], g["hi"]], "pentanomial": g["penta"],
            "score_paired": g["score"], "score_all_games": g["score_all"],
            "sprt_llr": g["llr"], "sprt": g["decision"] or "undecided",
        },
        "chained_elo": c["chained"],
    }


def readme(champs: list[dict], current: str | None) -> str:
    rows = []
    for c in reversed(champs):
        g, ch = c["gate"], c["chained"]
        gate = (f"vs {c['base']}: {signed(g['elo'])} ({signed(g['lo'])} to {signed(g['hi'])}), "
                f"{g['games']} games, {verdict(g)}" if g and g["elo"] is not None
                else {"seed": "seed", "forced": "installed without a gate"}.get(c["promoted"], ""))
        since = f"{c['finished']:%Y-%m-%d}" if c["finished"] else ""
        strength = ("" if not ch else "0 (reference)" if c["gen"] == ch["origin"]
                    else f"{signed(ch['elo'])} ± {ch['ci95']:.0f}")
        mark = " (current)" if c["gen"] == current else ""
        rows.append(f"| `{c['gen']}`{mark} | {since} | {gate} | {' + '.join(c['trained_on'])} | {strength} |")
    origin = next((c["chained"]["origin"] for c in champs if c["chained"]), "gen02")
    return f"""---
tags:
- chess
- reinforcement-learning
- self-play
- alphazero
library_name: pytorch
---

# ChessTransformer RL champions

Every network that became champion of the ChessTransformer expert-iteration loop
(`scripts/rl_generation.sh`): it plays a generation of self-play games, the next candidate
trains on them, and the candidate takes over when it wins its gate match. Each folder holds
one champion: `model.safetensors` + `model_config.json` (Pos2MoveV2, load with
`chesstransformer`), `model.onnx` (the Rust bot's graph), `expert_iter_meta.json` (its
training run) and `champion.json` (its gate). Uploaded by `scripts/hf_sync_champions.py`.

Current champion: **`{current}`** (also in `champion.json`).

| Generation | Champion since | Gate (paired Elo, 95%) | Trained on | Chained Elo over {origin} |
|---|---|---|---|---|
{chr(10).join(rows)}

Chained Elo adds up the promoted candidates' gate results from {origin}, the last network
installed without a gate; it is measured against the loop's own earlier networks, so other
opponents will see less.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", default="tchauffi/chesstransformer-rl")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    try:
        current = Path(os.readlink(RL / "champion")).name
    except OSError:
        current = None
    champs = champions()
    api = HfApi()
    if not args.dry_run:
        api.create_repo(args.repo, private=True, exist_ok=True)
        if not api.repo_info(args.repo).private:
            sys.exit(f"{args.repo} is public: refusing to upload the checkpoints there")
        present = {f.split("/")[0] for f in api.list_repo_files(args.repo)
                   if f.endswith("/model.safetensors")}
    else:
        present = set()

    for c in champs:
        if c["gen"] in present:
            continue
        src = RL / "models" / c["gen"]
        files = [f for f in FILES if (src / f).exists()]
        if "model.safetensors" not in files:
            print(f"{c['gen']}: no model.safetensors in {src}, skipped", flush=True)
            continue
        size = sum((src / f).stat().st_size for f in files) / 1e6
        print(f"{c['gen']}: uploading {', '.join(files)} ({size:.0f} MB)", flush=True)
        if args.dry_run:
            continue
        ops = [CommitOperationAdd(f"{c['gen']}/{f}", str(src / f)) for f in files]
        ops.append(CommitOperationAdd(f"{c['gen']}/champion.json",
                                      json.dumps(card(c), indent=1).encode()))
        api.create_commit(args.repo, ops, commit_message=f"Add {c['gen']} ({card(c)['how']})")

    index = {"champion": current, "generations": [c["gen"] for c in champs]}
    if args.dry_run:
        print(readme(champs, current))
        return
    ops = [CommitOperationAdd("README.md", readme(champs, current).encode()),
           CommitOperationAdd("champion.json", json.dumps(index, indent=1).encode())]
    # create_commit skips files whose content is unchanged; an empty commit is not made.
    info = api.create_commit(args.repo, ops, commit_message=f"Index: champion {current}")
    print(f"index: champion {current}, {len(champs)} champions -> {info.commit_url}", flush=True)


if __name__ == "__main__":
    main()
