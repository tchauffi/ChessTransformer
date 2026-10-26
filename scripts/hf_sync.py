"""Back up the RL loop to private Hugging Face repos: the champions and the games.

Champions, to the model repo tchauffi/chesstransformer-rl: gen00 (the seed), every
promoted candidate (history.tsv "yes") and every network installed without a gate
("forced"), each in its own folder genNN/ with its weights, config, ONNX graph, training
metadata and champion.json (its gate, what it trained on, its strength chained through
the gates, as on the status page). The README lists them and champion.json at the root
names the current one. Rejected candidates stay local. The anchors the gates hold
candidates to (data/rl/anchors/<label>/, e.g. v2.1) go to anchors/<label>/.

Games, to the dataset repo tchauffi/chesstransformer-rl-selfplay: every finished
generation's self-play as selfplay/genNN/ (games.jsonl, the positions_*.npz training
shards, heldout_games.json, generation_config.json), every finished match as
gates/*.jsonl, and at the root the loop's state (history.tsv, anchors.tsv, selfplay_hours,
forced_playouts, policy_target, policy_temp, fresh_full, full_frac, draw_contempt, sim_batch) and
logs/rl_generation.log,
so the repo mirrors data/rl/ and `make data` rebuilds a machine from it. The generation in progress and a gate still
being played are left for the next run.

Inputs (`--only inputs`, once), to the dataset repo tchauffi/chesstransformer-inputs,
which may be public: CC0, like the Lichess games they come from. The gates' opening book
data/openings/ and the human games every candidate replays, data/shards/elite_k32_v2/.
The champions and the games stay private: their repos are refused if they are ever public.
The script creates repos private and never changes a repo's visibility.

rl_generation.sh starts it in the background after each gate, so the next generation
never waits on the network; ct-hf-sync.timer (hourly) catches up after a failed upload.
Runs queue on a lock, and whatever is already on the hub is skipped.

    .venv/bin/python scripts/hf_sync.py                    # champions and games
    .venv/bin/python scripts/hf_sync.py --only games
    .venv/bin/python scripts/hf_sync.py --dry-run          # list what would be uploaded
    .venv/bin/python scripts/hf_sync.py --only inputs      # the book and the human games

Download:  hf download tchauffi/chesstransformer-rl --include "gen14/*" --local-dir .
           hf download tchauffi/chesstransformer-rl-selfplay --repo-type dataset \\
               --include "selfplay/gen14/*" --local-dir data/rl

It needs a Hugging Face login with write access (`hf auth login`). The user units:

    # ~/.config/systemd/user/ct-hf-sync.service
    [Service]
    Type=oneshot
    WorkingDirectory=%h/Documents/dev/ChessTransformer
    ExecStart=%h/Documents/dev/ChessTransformer/.venv/bin/python scripts/hf_sync.py
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
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_status import LOG, RL, ROOT, history, ladder, signed, verdict  # noqa: E402

MODEL_REPO = "tchauffi/chesstransformer-rl"
GAMES_REPO = "tchauffi/chesstransformer-rl-selfplay"
INPUTS_REPO = "tchauffi/chesstransformer-inputs"
# The loop's state; data/rl/selfplay_workers, _gpus and _parallel belong to the machine.
STATE_FILES = ("history.tsv", "anchors.tsv", "selfplay_hours", "forced_playouts", "policy_target",
               "policy_temp", "fresh_full", "full_frac", "draw_contempt", "sim_batch", "temp_cutoff",
               "selfplay_games")
INPUTS = ("openings", "shards/elite_k32_v2")
CHAMPION_FILES = ("model.safetensors", "model_config.json", "model.onnx", "expert_iter_meta.json",
                  "SOURCE")
SELFPLAY_FILES = ["games.jsonl", "heldout_games.json", "generation_config.json", "*.npz"]
FRESH_SECONDS = 3600  # a match file written to this recently may still be in progress


def private_repo(api: HfApi, repo: str, repo_type: str, public_ok: bool = False) -> set[str]:
    """Create the repo (private) if needed, refuse a public one unless `public_ok`, and list
    its files."""
    api.create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)
    if not public_ok and not api.repo_info(repo, repo_type=repo_type).private:
        sys.exit(f"{repo} is public: refusing to upload there")
    return set(api.list_repo_files(repo, repo_type=repo_type))


def hub_ahead(api: HfApi) -> list[str]:
    """The lines of the hub's history.tsv that this machine's lacks: when there are any,
    another machine runs the loop, and this one's state and index would overwrite its own."""
    try:
        hub = api.hf_hub_download(GAMES_REPO, "history.tsv", repo_type="dataset")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return []
    path = RL / "history.tsv"
    ours = set(path.read_text().splitlines()) if path.exists() else set()
    return [line for line in Path(hub).read_text().splitlines() if line not in ours]


# ── champions ─────────────────────────────────────────────────────────────────

def champions(rows: list[dict]) -> list[dict]:
    """gen00 and every promoted or installed generation, oldest first."""
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


def champions_readme(champs: list[dict], current: str | None) -> str:
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
training run) and `champion.json` (its gate). The games are in the dataset
[{GAMES_REPO}](https://huggingface.co/datasets/{GAMES_REPO}). Uploaded by `scripts/hf_sync.py`.

Current champion: **`{current}`** (also in `champion.json`).

| Generation | Champion since | Gate (paired Elo, 95%) | Trained on | Chained Elo over {origin} |
|---|---|---|---|---|
{chr(10).join(rows)}

Chained Elo adds up the promoted candidates' gate results from {origin}, the last network
installed without a gate; it is measured against the loop's own earlier networks, so other
opponents will see less.
"""


def sync_champions(api: HfApi, rows: list[dict], dry_run: bool) -> None:
    try:
        current = Path(os.readlink(RL / "champion")).name
    except OSError:
        current = None
    champs = champions(rows)
    present = set() if dry_run else private_repo(api, MODEL_REPO, "model")
    for c in champs:
        if f"{c['gen']}/model.safetensors" in present:
            continue
        src = RL / "models" / c["gen"]
        files = [f for f in CHAMPION_FILES if (src / f).exists()]
        if "model.safetensors" not in files:
            print(f"{c['gen']}: no model.safetensors in {src}, skipped", flush=True)
            continue
        size = sum((src / f).stat().st_size for f in files) / 1e6
        print(f"champion {c['gen']}: uploading {', '.join(files)} ({size:.0f} MB)", flush=True)
        if dry_run:
            continue
        ops = [CommitOperationAdd(f"{c['gen']}/{f}", str(src / f)) for f in files]
        ops.append(CommitOperationAdd(f"{c['gen']}/champion.json",
                                      json.dumps(card(c), indent=1).encode()))
        api.create_commit(MODEL_REPO, ops, commit_message=f"Add {c['gen']} ({card(c)['how']})")
    # The anchors the gates hold candidates to (data/rl/anchors/<label>, e.g. v2.1), so a new
    # machine's `make data` has them.
    for src in sorted(p for p in (RL / "anchors").glob("*") if p.is_dir()):
        if f"anchors/{src.name}/model.safetensors" in present:
            continue
        files = [f for f in ("model.safetensors", "model_config.json") if (src / f).exists()]
        if len(files) < 2:
            print(f"anchor {src.name}: no model.safetensors + model_config.json in {src}, skipped",
                  flush=True)
            continue
        print(f"anchor {src.name}: uploading {', '.join(files)}", flush=True)
        if not dry_run:
            api.create_commit(MODEL_REPO, [CommitOperationAdd(f"anchors/{src.name}/{f}", str(src / f))
                                           for f in files], commit_message=f"Add anchor {src.name}")
    if dry_run:
        return
    index = {"champion": current, "generations": [c["gen"] for c in champs]}
    ops = [CommitOperationAdd("README.md", champions_readme(champs, current).encode()),
           CommitOperationAdd("champion.json", json.dumps(index, indent=1).encode())]
    # create_commit drops unchanged files and makes no empty commit.
    api.create_commit(MODEL_REPO, ops, commit_message=f"Index: champion {current}")
    print(f"champions: {len(champs)} on the hub, current {current}", flush=True)


# ── games ─────────────────────────────────────────────────────────────────────

def finished_matches() -> list[Path]:
    """Every match file but a gate still being played (or any file written just now)."""
    out = []
    for f in sorted((RL / "gates").glob("*.jsonl")):
        m = re.match(r"cand(\d\d)_vs_gen\d\d\.jsonl$", f.name)
        if m:
            if not (RL / f"selfplay/gen{m.group(1)}/.gate_done").exists():
                continue
        elif time.time() - f.stat().st_mtime < FRESH_SECONDS and not Path(f"{f}.done").exists():
            continue
        out.append(f)
    return out


def games_readme(rows: list[dict], gens: list[str]) -> str:
    outcome = {"yes": "promoted", "no": "rejected", "forced": "not used (installed)"}
    table = []
    for r in reversed(rows):
        if r["gen"] not in gens:
            continue
        hours = f"{r['hours']:g} h" if r["hours"] else ""
        table.append(f"| `{r['gen']}` | {r['base'] or ''} | {hours} | {r['games']:,} | "
                     f"{r['positions']:,} | {' + '.join(r['trained_on'])} | "
                     f"{outcome.get(r['promoted'], r['promoted'])} |")
    return f"""---
tags:
- chess
- self-play
- reinforcement-learning
pretty_name: ChessTransformer RL self-play
---

# ChessTransformer RL self-play games

The self-play games of the ChessTransformer expert-iteration loop (`scripts/rl_generation.sh`)
and every match it played. Each generation is played by the champion of the time; the next
candidate trains on it (with its permanent held-out games set aside) and must win a gate
match to take over. The networks are in [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}).
Uploaded by `scripts/hf_sync.py` after each gate.

## Layout

- `selfplay/genNN/games.jsonl`: one game per line, `id`, `fen` (the start, a position from
  the opening list or the initial position), `moves` (UCI), `result`, `end`, `plies`,
  `resign_allowed`.
- `selfplay/genNN/positions_NNNN.npz`: the training positions, one per move, written by
  `scripts/selfplay_rust.py` (`ShardWriter`): `boards` (64 square tokens), `player`,
  `castling`, `ep`, `halfmove`, `z` (game outcome), `root_v` (search root value), `full`
  (1 = a full search, whose visits are a policy target), `game_id`, and the search's visit
  distribution as CSR rows `visit_ptr` / `visit_idx` (move index from_square * 73 + plane) /
  `visit_cnt`; from the bot's search also `legal_ptr` / `legal_idx` / `legal_prior` /
  `legal_n` / `legal_q` and `root_net_v`.
- `selfplay/genNN/heldout_games.json`: the game ids no candidate ever trains on.
- `selfplay/genNN/generation_config.json`: the self-play settings.
- `gates/candNN_vs_genMM.jsonl`: gate games, `idx` (opening), `start` (FEN), `a_white`,
  `result`, `score_a`, `plies`, `end`; each opening is played twice with colours swapped.
  The other files are earlier experiments' matches.

## Generations

| Generation | Played by | Self-play | Games | Positions | Its candidate trained on | Candidate |
|---|---|---|---|---|---|---|
{chr(10).join(table)}
"""


def sync_games(api: HfApi, rows: list[dict], dry_run: bool) -> None:
    present = set() if dry_run else private_repo(api, GAMES_REPO, "dataset")
    gens = sorted(p.name for p in (RL / "selfplay").glob("gen[0-9][0-9]")
                  if (p / ".generation_done").exists())
    for gen in gens:
        if f"selfplay/{gen}/games.jsonl" in present:  # a generation is one commit
            continue
        src = RL / "selfplay" / gen
        files = [f for pat in SELFPLAY_FILES for f in src.glob(pat)]
        size = sum(f.stat().st_size for f in files) / 1e6
        print(f"games {gen}: uploading {len(files)} files ({size:.0f} MB)", flush=True)
        if dry_run:
            continue
        n = sum(1 for _ in open(src / "games.jsonl"))
        api.upload_folder(repo_id=GAMES_REPO, repo_type="dataset", folder_path=src,
                          path_in_repo=f"selfplay/{gen}", allow_patterns=SELFPLAY_FILES,
                          commit_message=f"Add {gen} self-play ({n:,} games)")
    matches = finished_matches()
    if dry_run:
        print(f"matches: {len(matches)} files", flush=True)
        return
    ops = [CommitOperationAdd(f"gates/{f.name}", str(f)) for f in matches]
    ops += [CommitOperationAdd(f, str(RL / f)) for f in STATE_FILES if (RL / f).exists()]
    if LOG.exists():
        ops.append(CommitOperationAdd("rl_generation.log", str(LOG)))
    ops.append(CommitOperationAdd("README.md", games_readme(rows, gens).encode()))
    api.create_commit(GAMES_REPO, ops, repo_type="dataset", commit_message="Matches, state and index")
    print(f"games: {len(gens)} generations, {len(matches)} match files and the state on the hub",
          flush=True)


INPUTS_README = f"""---
license: cc0-1.0
pretty_name: ChessTransformer human games and opening book
tags:
- chess
- lichess
size_categories:
- 10M<n<100M
---

# ChessTransformer human games and opening book

The inputs of [ChessTransformer](https://github.com/tchauffi/ChessTransformer)'s self-play
loop besides its own games: **48.4 million positions from 1.53 million games of strong
human players**, and the **opening book** its matches start from. Every candidate network
replays these human positions (20% of each training step) so it keeps what it learned from
people, and every gate match between two networks plays the book.

## Source and license

The games come from the [Lichess Elite Database](https://database.nikonoel.fr), a selection
of games between strong players from the [Lichess open database](https://database.lichess.org),
which Lichess releases under CC0. This processed version is released under CC0 as well.
Results of games that reach an endgame the Syzygy tablebases cover are corrected with them.

## Files

### `shards/elite_k32_v2/`: human positions

256 files `shard_NNNN.npy` (10 GB), each a NumPy structured array of 206-byte records that
loads with `np.load(path, mmap_mode="r")`. Moves use AlphaZero's 73 move planes per
from-square. 32 positions were drawn from each game,
middlegame-weighted and without repeats. `meta.json` holds the record layout, the build
settings and the split: 246 shards for training, 5 for validation, 5 for test.

| Field | Type | Meaning |
|---|---|---|
| `position` | uint8[64] | square a1 = 0 ... h8 = 63, not flipped: 0 empty, 1-6 White PNBRQK, 7-12 Black pnbrqk |
| `legal_idx` | uint16[64] | legal moves as `from_square * 73 + plane`, the move played first, padded with 4672 |
| `legal_cnt` | uint8 | how many legal moves are stored |
| `from_square`, `action_plane` | uint8 | the move played |
| `castling_rights` | uint8 | bits: 1 White kingside, 2 White queenside, 4 Black kingside, 8 Black queenside |
| `en_passant_file` | uint8 | file of a capturable en-passant square, 8 = none |
| `is_white` | uint8 | 1 when White is to move |
| `result` | int8 | the game's result: 0 draw, 1 White won, 2 Black won |
| `move_number` | uint16 | the position's ply in the game |
| `elo_gap` | int16 | the side to move's rating minus its opponent's |
| `result_tb` | int8 | `result` corrected by the tablebases when the game reached a position they cover (same coding) |
| `value_exact` | uint8 | 1 when `result_tb` is the tablebases' own verdict on this position |

Of the 1.53 million games, 253,822 reached a tablebase position and 26,062 of those ended
with a result the tables contradict; 782,207 positions were relabelled and 474,425 carry an
exact value.

### `openings/`: match starts

- `book2k.json`: 2,000 openings for engine matches, 8 plies each from games with both
  players rated 2200 or more, kept when Stockfish at depth 12 scores them within 80
  centipawns; each entry has `uci` (the moves), `fen` and `cp`. A match plays every opening
  twice with colours swapped, so its difficulty cancels out.
- `endgame7.json`: 400 starts from the same games at their first position with 7 men or
  fewer, within 600 centipawns.

## Use

```python
import numpy as np
from huggingface_hub import hf_hub_download

path = hf_hub_download("{INPUTS_REPO}", "shards/elite_k32_v2/shard_0000.npy", repo_type="dataset")
rows = np.load(path, mmap_mode="r")
row = rows[0]
print(len(rows), row["position"], row["result"], row["elo_gap"])
```

Built with `scripts/build_db.py`, `scripts/build_shards.py` (`-k 32 --sample-weighting
middlegame --syzygy`) and `scripts/build_opening_book.py` in the ChessTransformer repo, and
uploaded by `scripts/hf_sync.py --only inputs`.
"""


def scrubbed(path: Path) -> bytes:
    """A JSON input without this machine's paths: those under the repo become relative
    (meta.json records where its HDF5 and tablebases were), and any other home path stops
    the upload, since the inputs may be public."""
    data = path.read_bytes().replace(f"{ROOT}/".encode(), b"")
    if re.search(rb"/home/|/Users/|" + re.escape(Path.home().name.encode()), data):
        sys.exit(f"{path} still holds a local path or the user name: not uploading it")
    return data


def sync_inputs(api: HfApi, dry_run: bool) -> None:
    present = set() if dry_run else private_repo(api, INPUTS_REPO, "dataset", public_ok=True)
    ops = [CommitOperationAdd("README.md", INPUTS_README.encode())]
    for sub in INPUTS:
        src = ROOT / "data" / sub
        files = sorted(f for f in src.iterdir() if f.is_file())
        data = [f for f in files if f.suffix != ".json"]
        if data and not all(f"{sub}/{f.name}" in present for f in data):
            size = sum(f.stat().st_size for f in data) / 1e6
            print(f"inputs {sub}: uploading {len(data)} files ({size:.0f} MB)", flush=True)
            if not dry_run:
                api.upload_folder(repo_id=INPUTS_REPO, repo_type="dataset", folder_path=src,
                                  path_in_repo=sub, ignore_patterns=["*.json"],
                                  commit_message=f"Add {sub}")
        # The JSON files (small) go every run, scrubbed; unchanged ones are dropped.
        ops += [CommitOperationAdd(f"{sub}/{f.name}", scrubbed(f)) for f in files if f.suffix == ".json"]
    if not dry_run:
        api.create_commit(INPUTS_REPO, ops, repo_type="dataset", commit_message="Index and metadata")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", choices=("champions", "games", "inputs"),
                   help="one part; inputs (static) only ever run this way")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Runs queue rather than race: one started after a gate waits for the hourly one.
    lock = open(RL / ".hf_sync.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    api, rows = HfApi(), history()
    if args.only == "inputs":
        sync_inputs(api, args.dry_run)
        return
    if ahead := hub_ahead(api):
        gens = ", ".join(line.split("\t")[0] for line in ahead)
        sys.exit(f"the hub has {gens}, which this machine's history.tsv lacks: another machine "
                 "runs the loop, so this one does not overwrite its backup")
    if args.only != "games":
        sync_champions(api, rows, args.dry_run)
    if args.only != "champions":
        sync_games(api, rows, args.dry_run)


if __name__ == "__main__":
    main()
