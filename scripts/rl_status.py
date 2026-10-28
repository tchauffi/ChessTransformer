"""Live status page for the RL loop (scripts/rl_generation.sh), so its progress can be
followed without asking.

It shows the generation in progress and its stage -- self-play, training, gate -- with
estimated end times, the champion's strength chained through the gates, and one row per
finished generation. It only reads data/rl/ and logs/rl_generation.log: standard library
only (sprt.py is plain Python), no GPU, no uv environment. It runs as its own user
service, outside ct-rl-loop's cgroup, so it keeps answering while the loop is frozen.

    python3 scripts/rl_status.py                   # http://localhost:8765
    python3 scripts/rl_status.py --host 0.0.0.0    # also from the local network
    python3 scripts/rl_status.py --json            # print the status once and exit

~/.config/systemd/user/ct-rl-status.service runs it with --host 0.0.0.0, so the page is
also at http://<this machine's address>:8765 on the local network:

    [Service]
    WorkingDirectory=%h/Documents/dev/ChessTransformer
    ExecStart=/usr/bin/python3 scripts/rl_status.py --host 0.0.0.0
    Nice=10
    Restart=on-failure

    [Install]
    WantedBy=default.target

End times count running time only: self-play's hour budget and the training steps stand
still while ct-rl-pause.timer has the loop frozen (22:00-08:00), so every estimate skips
those hours; `--pause none` is a machine that runs around the clock. Strength is chained head to head: each promoted candidate's paired gate Elo
against the champion that played its generation, summed from the last network installed
without a gate (history.tsv "forced"); the band adds up the gates' variances. Gains
against the loop's own earlier networks overstate what other opponents will see.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chesstransformer.evaluation.sprt import (  # noqa: E402
    complete_prefix,
    pair_up,
    sprt_bounds,
    sprt_decision,
    sprt_llr,
    summarize,
)

RL = ROOT / "data/rl"
LOG = ROOT / "logs/rl_generation.log"
UNIT = "ct-rl-loop.service"
PAUSE_HOUR, RESUME_HOUR = 22, 8  # ct-rl-pause.timer / ct-rl-resume.timer
PAUSED_NIGHTLY = True  # --pause none: the loop runs around the clock
LOG_TAIL = 4 << 20  # bytes; a 24 h generation logs ~250 KB
CACHE_SECONDS = 15
GATE_GAMES = 400
NO_GAMES = {"games": 0, "pairs": 0, "score": 0.5, "penta": [0] * 5, "score_all": None,
            "llr": 0.0, "decision": None, "elo": None, "lo": None, "hi": None}

STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)")
PROGRESS = re.compile(r"^(\d+)/\d+ games \| (\d+) pos \| (\d+) games/h")
STEP = re.compile(r"^\s+step (\d+)/(\d+) \|")
TRAINING = re.compile(r"^training \S*/cand(\d+) from (\S+) on ")
POSITIONS = re.compile(r"^Positions: ([\d,]+) over ([\d,]+) games")

SCRATCH = RL / "scratch"
SCRATCH_LOGS = ROOT / "logs/scratch"
SCRATCH_STEP = re.compile(
    r"^step (\d+) \| \S+ \| trained ([\d,]+) of budget [\d,]+ \| window ([\d,]+) rows, (\d+) shards"
    r" \| admitted ([\d,]+) \| policy ([\d.]+) value ([\d.]+) kl [\d.]+ \| ([\d,]+) samples/s")
SCRATCH_HELD = re.compile(r"^\s+held out \(EMA\): policy ([\d.]+) value ([\d.]+) sign_acc ([\d.]+)")
SCRATCH_EVERY = re.compile(r"evaluations every ([\d.]+) h, (\d+) games, accept >= ([\d.]+)")
SCRATCH_POINTS = 240  # chart points kept


# ── running time (the loop is frozen 22:00-08:00) ─────────────────────────────

def _paused(t: datetime) -> bool:
    return PAUSED_NIGHTLY and (t.hour >= PAUSE_HOUR or t.hour < RESUME_HOUR)


def _next(t: datetime, hour: int) -> datetime:
    """The first `hour`:00 after t."""
    n = t.replace(hour=hour, minute=0, second=0, microsecond=0)
    return n if n > t else n + timedelta(days=1)


def add_running(t: datetime, seconds: float) -> datetime:
    """Wall-clock time once `seconds` of running time have passed after t."""
    if not PAUSED_NIGHTLY:
        return t + timedelta(seconds=seconds)
    while True:
        if _paused(t):
            t = _next(t, RESUME_HOUR)
        stop = _next(t, PAUSE_HOUR)
        if seconds <= (stop - t).total_seconds():
            return t + timedelta(seconds=seconds)
        seconds -= (stop - t).total_seconds()
        t = stop


def running_between(a: datetime, b: datetime) -> float:
    """Seconds of running time between a and b."""
    if not PAUSED_NIGHTLY:
        return max(0.0, (b - a).total_seconds())
    total, t = 0.0, a
    while t < b:
        if _paused(t):
            t = _next(t, RESUME_HOUR)
            continue
        stop = min(_next(t, PAUSE_HOUR), b)
        total += (stop - t).total_seconds()
        t = stop
    return total


# ── reading the loop's state ──────────────────────────────────────────────────

def read_log(tail: int | None = LOG_TAIL) -> list[str]:
    try:
        with open(LOG, "rb") as f:
            if tail is not None:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - tail))
            data = f.read()
    except OSError:
        return []
    return data.decode("utf-8", "replace").splitlines()


def stamp(line: str) -> tuple[datetime | None, str]:
    m = STAMP.match(line)
    if not m:
        return None, line
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), m.group(2)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


_gate_cache: dict[Path, tuple[tuple[int, int], dict]] = {}


def gate_stats(path: Path) -> dict | None:
    """The gate report's numbers, from the games file (paired, as the SPRT reads them)."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    hit = _gate_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass  # a game still being written
    # Paired figures on the openings done from the first one on, which the gate's SPRT judges.
    res = complete_prefix({(r["idx"], r["a_white"]): r["score_a"] for r in rows})
    pairs = pair_up(res)
    s = summarize(pairs, list(res.values()))
    llr = sprt_llr(pairs)
    out = {
        "games": len(rows), "pairs": s.n_pairs, "score": s.score, "penta": list(s.counts),
        "score_all": sum(r["score_a"] for r in rows) / len(rows) if rows else None,
        "llr": llr, "decision": sprt_decision(llr), "elo": None, "lo": None, "hi": None,
    }
    if s.n_pairs >= 2 and 0.0 < s.score < 1.0:
        lo, hi = s.elo_interval()
        out.update(elo=s.elo, lo=lo, hi=hi)
    _gate_cache[path] = (key, out)
    return out


_trained_cache: tuple[float, dict] = (0.0, {})


def trained_on() -> dict[int, list[str]]:
    """Candidate number -> the generations it trained on (the last attempt in the log)."""
    global _trained_cache
    stamp_ = (RL / "history.tsv").stat().st_mtime if (RL / "history.tsv").exists() else 0.0
    if _trained_cache[0] != stamp_:
        out = {}
        for line in read_log(tail=None):
            _, msg = stamp(line)
            m = TRAINING.match(msg)
            if m:
                out[int(m.group(1))] = re.findall(r"selfplay/(gen\d\d)", msg)
        _trained_cache = (stamp_, out)
    return _trained_cache[1]


def history() -> list[dict]:
    rows = []
    try:
        with open(RL / "history.tsv") as f:
            table = list(csv.DictReader(f, delimiter="\t"))
    except OSError:
        return rows
    data = trained_on()
    for r in table:
        gen, n = r["gen"], int(r["gen"][3:])
        sp = RL / "selfplay" / gen
        cfg = read_json(sp / "generation_config.json")
        base = Path(cfg["model"]).name if cfg.get("model") else None
        gate = None
        if r["promoted"] != "forced":
            files = sorted((RL / "gates").glob(f"cand{n:02d}_vs_gen[0-9][0-9].jsonl"))
            if files:
                gate = gate_stats(files[-1])
                base = files[-1].stem.split("_vs_")[1]
        rows.append({
            "gen": gen, "n": n, "base": base, "hours": cfg.get("max_hours"),
            "games": int(r["games"]), "positions": int(r["positions"]),
            "promoted": r["promoted"], "gate": gate,
            "trained_on": [] if r["promoted"] == "forced" else data.get(n, []),
            "finished": mtime(sp / ".generation_done"),
        })
    return rows


def ladder(rows: list[dict], current: int) -> dict | None:
    """Champion strength chained through the gates, from the last ungated network."""
    elo = {"gen00": (0.0, 0.0)}  # champion -> (Elo, variance)
    origin, champ, cands = "gen00", [{"n": 0, "gen": "gen00", "elo": 0.0, "ci": 0.0}], []
    for r in rows:
        if r["promoted"] == "forced":
            elo, origin = {r["gen"]: (0.0, 0.0)}, r["gen"]
            champ, cands = [{"n": r["n"], "gen": r["gen"], "elo": 0.0, "ci": 0.0}], []
            continue
        gate, base = r["gate"], r["base"]
        if not gate or gate["elo"] is None or base not in elo:
            continue
        b, var = elo[base]
        sd = (gate["hi"] - gate["lo"]) / (2 * 1.96)
        promoted = r["promoted"] == "yes"
        cands.append({
            "n": r["n"], "elo": b + gate["elo"], "lo": b + gate["lo"], "hi": b + gate["hi"],
            "promoted": promoted,
            "tip": f"cand{r['n']:02d} vs {base}: {signed(gate['elo'])} Elo "
                   f"({signed(gate['lo'])} to {signed(gate['hi'])}), {gate['games']} games, "
                   f"{verdict(gate)}, {'promoted' if promoted else 'rejected'}",
        })
        if promoted:
            elo[r["gen"]] = (b + gate["elo"], var + sd * sd)
            champ.append({"n": r["n"], "gen": r["gen"], "elo": b + gate["elo"],
                          "ci": 1.96 * math.sqrt(var + sd * sd)})
    if len(champ) < 2:
        return None
    origin_row = next((r for r in rows if r["gen"] == origin), None)
    return {"origin": origin, "since": origin_row["finished"] if origin_row else None,
            "x0": champ[0]["n"], "x1": max(current, champ[-1]["n"]),
            "champion": champ, "candidates": cands}


def loop_state() -> dict:
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", UNIT, "-p", "ActiveState", "-p", "FreezerState"],
            capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    props = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return {"active": props.get("ActiveState", "unknown"),
            "frozen": props.get("FreezerState", "running") in ("frozen", "freezing"),
            "stop_file": (RL / "STOP").exists()}


def gpu() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        util, used, total, temp = (float(x) for x in out.splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return {"util": util, "used_gb": used / 1024, "total_gb": total / 1024, "temp": temp}


def last_stage_durations(lines: list[str]) -> tuple[float | None, float | None]:
    """Running seconds of the last finished training and gate in the log."""
    train = gate = None
    t_train = t_gate = None
    for line in lines:
        t, msg = stamp(line)
        if t is None:
            continue
        if TRAINING.match(msg):
            t_train, t_gate = t, None
        elif msg.startswith("gating ") and t_train:
            train, t_gate = running_between(t_train, t), t
        elif msg.startswith("cand vs champion") and t_gate:
            gate, t_gate = running_between(t_gate, t), None
    return train, gate


def read_tail(path: Path, tail: int = LOG_TAIL) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail))
            return f.read().decode(errors="replace").splitlines()
    except OSError:
        return []


def scratch() -> dict | None:
    """The continuous run from a random network (make scratch): trainer, self-play, evaluations."""
    if not SCRATCH.exists():
        return None
    num = lambda x: int(x.replace(",", ""))
    steps: list[dict] = []
    for line in read_tail(SCRATCH_LOGS / "rl_continuous_train.log"):
        t, msg = stamp(line)
        m = SCRATCH_STEP.match(msg) if t else None
        if m:
            steps.append({"time": t, "step": int(m[1]), "trained": num(m[2]), "window": num(m[3]),
                          "shards": int(m[4]), "admitted": num(m[5]), "policy": float(m[6]),
                          "value": float(m[7]), "rate": num(m[8])})
            continue
        m = SCRATCH_HELD.match(msg) if t else None
        if m and steps and "held_policy" not in steps[-1]:
            steps[-1].update(held_policy=float(m[1]), held_value=float(m[2]), sign_acc=float(m[3]))
    progress = None
    for line in reversed(read_tail(SCRATCH_LOGS / "rl_continuous_selfplay.log", 1 << 16)):
        m = PROGRESS.match(line)
        if m:
            progress = dict(zip(("games", "positions", "rate"), map(int, m.groups())))
            break
    loop = [stamp(x) for x in read_tail(SCRATCH_LOGS / "rl_continuous.log", 1 << 20)]
    loop = [(t, msg) for t, msg in loop if t]
    started = every = None
    evaluating = None
    anchor = None
    for t, msg in loop:
        if msg.startswith("continuous loop:"):
            started = anchor = t
            m = SCRATCH_EVERY.search(msg)
            if m:
                every = {"hours": float(m[1]), "games": int(m[2]), "accept": float(m[3])}
            evaluating = None
        elif msg.startswith("evaluating "):
            evaluating = t
        elif " vs accepted:" in msg or msg.startswith("no new publication"):
            anchor, evaluating = t, None
        elif msg.startswith("STOP found"):
            anchor, started = None, None
    evals = []
    try:
        with open(SCRATCH / "history.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                evals.append({"time": r["time"], "candidate": r["candidate"],
                              "score": float(r["score"]), "games": int(r["games"]),
                              "elo": float(r["elo"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
                              "decision": r["decision"]})
    except (OSError, KeyError, ValueError):
        pass
    now = datetime.now().replace(microsecond=0)
    next_eval = None
    if anchor and every and not evaluating:
        need = every["hours"] * 3600
        resume = anchor.replace(hour=RESUME_HOUR, minute=0, second=0)
        if PAUSED_NIGHTLY and _paused(anchor) and anchor < resume and not loop_state()["frozen"]:
            # started by hand before the resume timer: that early stretch ran too
            early = (resume - anchor).total_seconds()
            next_eval = anchor + timedelta(seconds=need) if early >= need else add_running(resume, need - early)
        else:
            next_eval = add_running(anchor, need)
    last = steps[-1] if steps else None
    keep = steps[:: max(1, len(steps) // SCRATCH_POINTS + 1)] + ([last] if last else [])
    accepted = SCRATCH / "best" / "SOURCE"
    try:
        accepted = accepted.read_text().strip()
    except OSError:
        accepted = None
    return {"last": last, "curve": [x for x in keep if "held_policy" in x],
            "selfplay": progress, "every": every, "started": started, "evaluating": evaluating,
            "next_eval": next_eval, "evals": evals, "accepted": accepted}


def status() -> dict:
    now = datetime.now().replace(microsecond=0)
    lines = read_log()
    try:
        champion = Path(os.readlink(RL / "champion")).name
    except OSError:
        champion = None
    g = 1
    while (RL / f"selfplay/gen{g:02d}/.generation_done").exists():
        g += 1
    gen, cand = f"gen{g:02d}", f"cand{g:02d}"
    sp = RL / "selfplay" / gen

    marker = max((i for i, line in enumerate(lines) if f"=== {gen}: champion" in line),
                 default=None)
    cur = lines[marker + 1:] if marker is not None else []
    cfg = read_json(sp / "generation_config.json")
    try:
        hours = float(cfg.get("max_hours") or (RL / "selfplay_hours").read_text())
    except (OSError, ValueError):
        hours = 8.0
    train_s, gate_s = last_stage_durations(lines)

    if not sp.exists():
        stage = "idle"
    elif not (sp / ".selfplay_done").exists():
        stage = "selfplay"
    elif not (RL / "models" / cand / "model.safetensors").exists():
        stage = "training"
    elif not (sp / ".gate_done").exists():
        stage = "gate" if any(stamp(x)[1].startswith("gating ") for x in cur) else "export"
    else:
        stage = "recording"

    now_s: dict = {"gen": gen, "cand": cand, "base": champion, "stage": stage, "hours": hours}
    selfplay_end = None
    if stage == "selfplay":
        games = pos = rate = 0
        for line in reversed(cur):
            m = PROGRESS.match(line)
            if m:
                games, pos, rate = map(int, m.groups())
                break
        done = games / rate if rate else 0.0
        selfplay_end = add_running(now, max(0.0, hours - done) * 3600)
        now_s.update(games=games, positions=pos, rate=rate, hours_done=done,
                     selfplay_end=selfplay_end)
    if stage == "training":
        start, step, total, data, positions = None, 0, 0, [], None
        for line in cur:
            t, msg = stamp(line)
            if t and TRAINING.match(msg):
                start, step, total = t, 0, 0
                data = re.findall(r"selfplay/(gen\d\d)", msg)
            m = STEP.match(line)
            if m:
                step, total = int(m.group(1)), int(m.group(2))
            m = POSITIONS.match(line)
            if m:
                positions = int(m.group(1).replace(",", ""))
        end = None
        if start and step:
            per = running_between(start, now) / step
            end = add_running(now, per * (total - step))
        now_s.update(step=step, total=total, trained_on=data, positions=positions,
                     training_end=end)
    gate_file = RL / "gates" / f"{cand}_vs_{champion}.jsonl"
    if stage == "gate":
        gs = gate_stats(gate_file) or NO_GAMES
        start = max((t for t, msg in map(stamp, cur) if t and msg.startswith("gating ")),
                    default=None)
        latest = None
        if gs["games"] and start:
            per = running_between(start, now) / gs["games"]
            latest = add_running(now, per * max(0, GATE_GAMES - gs["games"]))
        now_s.update(gate=gs, gate_latest=latest, bounds=sprt_bounds())
    # When the result should land, if training and the gate take as long as last time.
    if train_s and gate_s:
        if stage == "selfplay":
            now_s["result_eta"] = add_running(selfplay_end, train_s + gate_s)
        elif stage == "training":
            end = now_s["training_end"]
            now_s["result_eta"] = add_running(end, gate_s) if end else add_running(now, train_s + gate_s)
        elif stage == "export":
            now_s["result_eta"] = add_running(now, gate_s)

    rows = history()
    last_entry = next((x for x in reversed(lines) if STAMP.match(x)), None)
    try:
        du = shutil.disk_usage(RL)
        disk = {"free_gb": du.free / 1e9, "used_pct": 100 * du.used / (du.used + du.free)}
    except OSError:
        disk = None
    return {
        "now": now, "loop": loop_state(), "champion": champion, "current": now_s,
        "history": rows, "ladder": ladder(rows, g), "gpu": gpu(), "disk": disk,
        "selfplay": {k: cfg.get(k) for k in ("sims", "fast_sims", "full_frac", "parallel_games")},
        "last_entry": last_entry, "scratch": scratch(),
    }


_lock = threading.Lock()
_cached: tuple[float, dict | None] = (0.0, None)


def cached_status() -> dict:
    global _cached
    with _lock:
        if _cached[1] is None or time.monotonic() - _cached[0] > CACHE_SECONDS:
            _cached = (time.monotonic(), status())
        return _cached[1]


# ── rendering ─────────────────────────────────────────────────────────────────

def signed(x: float, digits: int = 0) -> str:
    return f"{x:+.{digits}f}".replace("-", "\u2212")


def verdict(gate: dict) -> str:
    return {"H1": "test passed", "H0": "test failed"}.get(gate["decision"], "test undecided")


def when(t: datetime | None, now: datetime) -> str:
    if t is None:
        return "unknown"
    days = (t.date() - now.date()).days
    day = {0: "today", 1: "tomorrow", -1: "yesterday"}.get(days, t.strftime("%a %-d %b"))
    return f"{day} {t:%H:%M}"


def esc(x) -> str:
    return html.escape(str(x))


def meter(frac: float) -> str:
    pct = max(0.0, min(1.0, frac)) * 100
    return f'<div class="meter"><span style="width:{pct:.1f}%"></span></div>'


def render_state(s: dict) -> str:
    loop, now = s["loop"], s["now"]
    if loop["active"] == "active" and loop["frozen"]:
        cls, text = "paused", ("Paused for the night, resumes at 08:00" if _paused(now)
                               else "Paused (frozen)")
    elif loop["active"] == "active":
        cls, text = "running", f"Running, pauses at {PAUSE_HOUR}:00" if PAUSED_NIGHTLY else "Running"
    else:
        cls, text = "stopped", f"Stopped ({loop['active']})"
    if loop["stop_file"]:
        text += ", stops after this generation"
    icon = {"running": "\u25cf", "paused": "\u275a\u275a", "stopped": "\u25a0"}[cls]
    return f'<p class="state {cls}"><span class="icon" aria-hidden="true">{icon}</span>{esc(text)}</p>'


def render_now(s: dict) -> str:
    c, now = s["current"], s["now"]
    stage = c["stage"]
    if stage == "idle":
        return '<section class="card"><h2>No generation running</h2></section>'
    order = ["selfplay", "training", "gate"]
    at = {"export": "training", "recording": "gate"}.get(stage, stage)
    steps = []
    for i, (key, label) in enumerate(zip(order, ["Self-play", "Training", "Gate"])):
        cls = "done" if order.index(at) > i else "active" if key == at else "todo"
        mark = "\u2713" if cls == "done" else str(i + 1)
        steps.append(f'<li class="{cls}"><span class="n">{mark}</span>{label}</li>')
    body, facts = "", []
    if stage == "selfplay":
        body = (f'<p class="lead">{c["games"]:,} games, {c["positions"]:,} positions</p>'
                + meter(c["hours_done"] / c["hours"])
                + f'<p class="sub">{c["hours_done"]:.1f} of {c["hours"]:g} h of running time, '
                  f'{c["rate"]:,} games/h</p>')
        facts.append(("Self-play ends", f'about {when(c["selfplay_end"], now)}'))
    elif stage == "training":
        data = " + ".join(c["trained_on"]) or "?"
        size = f', {c["positions"] / 1e6:.1f}M positions' if c["positions"] else ""
        if c["total"]:
            body = (f'<p class="lead">Step {c["step"]:,} of {c["total"]:,}</p>'
                    + meter(c["step"] / c["total"]))
        else:
            body = '<p class="lead">Loading the games</p>'
        body += f'<p class="sub">{esc(c["cand"])} from {esc(c["base"])} on {esc(data)}{size}</p>'
        facts.append(("Training ends", f'about {when(c["training_end"], now)}'))
    elif stage == "export":
        body = '<p class="lead">Exporting the candidate and checking its ONNX graph</p>'
    elif stage == "gate":
        g = c["gate"]
        lo, hi = c["bounds"]
        body = f'<p class="lead">{esc(c["cand"])} vs {esc(c["base"])}: {g["games"]} of up to {GATE_GAMES} games</p>'
        if g["elo"] is not None:
            body += (f'<p class="sub">{signed(g["elo"])} Elo ({signed(g["lo"])} to {signed(g["hi"])}, '
                     f'95%) over {g["pairs"]} pairs</p>')
        pos = (min(max(g["llr"], lo), hi) - lo) / (hi - lo) * 100
        body += (f'<div class="llr"><div class="track"><span class="zero"></span>'
                 f'<span class="mark" style="left:{pos:.1f}%"></span></div>'
                 f'<div class="ends"><span>rejects at {signed(lo, 2)}</span>'
                 f'<span>LLR {signed(g["llr"], 2)}</span><span>accepts at {signed(hi, 2)}</span></div></div>')
        if g["score_all"] is not None:
            facts.append(("Score over all games", f'{g["score_all"]:.4f} (promoted at 0.5 or more)'))
        facts.append(("All 400 games done", f'about {when(c["gate_latest"], now)}, '
                                            'unless the test stops it sooner'))
    elif stage == "recording":
        body = '<p class="lead">Recording the gate result</p>'
    if c.get("result_eta"):
        facts.append(("Gate result", f'about {when(c["result_eta"], now)}, if training and the '
                                     'gate take as long as last time'))
    dl = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in facts)
    return (f'<section class="card now"><h2>{esc(c["gen"])} <span class="muted">played by '
            f'{esc(c["base"])}</span></h2><ol class="stepper">{"".join(steps)}</ol>'
            f'{body}<dl class="facts">{dl}</dl></section>')


def render_tiles(s: dict) -> str:
    tiles = []
    rows = s["history"]
    champ_row = next((r for r in rows if r["gen"] == s["champion"]), None)
    since = f'since {when(champ_row["finished"], s["now"])}' if champ_row and champ_row["finished"] else ""
    tiles.append(("Champion", esc(s["champion"]), since))
    last = next((r for r in reversed(rows) if r["gate"] and r["gate"]["elo"] is not None), None)
    if last:
        g = last["gate"]
        tiles.append(("Last gate", f'{signed(g["elo"])} Elo',
                      f'cand{last["n"]:02d} vs {esc(last["base"])}, '
                      f'{"promoted" if last["promoted"] == "yes" else "rejected"}'))
    if s["gpu"]:
        g = s["gpu"]
        tiles.append(("GPU", f'{g["util"]:.0f}%', f'{g["used_gb"]:.1f} of {g["total_gb"]:.1f} GB, {g["temp"]:.0f} \u00b0C'))
    if s["disk"]:
        d = s["disk"]
        warn = ' <span class="warn">\u26a0 low</span>' if d["free_gb"] < 20 else ""
        tiles.append(("Disk free", f'{d["free_gb"]:.0f} GB', f'{d["used_pct"]:.0f}% used{warn}'))
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><p class="label">{k}</p><p class="value">{v}</p><p class="sub">{sub}</p></div>'
        for k, v, sub in tiles) + "</div>"


def render_strength(s: dict) -> str:
    lad = s["ladder"]
    if not lad:
        return ""
    last = lad["champion"][-1]
    since = f' ({lad["since"]:%-d %b})' if lad["since"] else ""
    data = {"x0": lad["x0"], "x1": lad["x1"], "current": s["current"]["gen"],
            "champion": lad["champion"], "candidates": lad["candidates"],
            "label": f'Champion strength, {signed(last["elo"])} Elo over {lad["origin"]}'}
    blob = json.dumps(data).replace("</", "<\\/")
    return (
        '<section class="card"><h2>Strength</h2>'
        f'<p class="hero">{signed(last["elo"])}<span class="unit"> Elo</span></p>'
        f'<p class="sub">{esc(last["gen"])} over {esc(lad["origin"])}{esc(since)}, the last network '
        f'installed without a gate (95%: {signed(last["elo"] - last["ci"])} to '
        f'{signed(last["elo"] + last["ci"])})</p>'
        '<ul class="legend"><li><span class="key line"></span>Champion, 95% band</li>'
        '<li><span class="key dot s1"></span>Promoted candidate</li>'
        '<li><span class="key dot s2"></span>Rejected candidate</li>'
        '<li><span class="key whisker"></span>Its gate\'s 95% interval</li></ul>'
        f'<div id="ladder"></div><script type="application/json" id="ladder-data">{blob}</script>'
        '<p class="caption">By generation. Each candidate\'s paired gate Elo against the champion '
        'that played its generation, added up over the promotions; the band adds up the gates\' '
        'uncertainty. Gains against the loop\'s own earlier networks overstate what other '
        'opponents will see.</p></section>')


def spark(points: list[tuple[float, float]], label: str, unit: str, digits: int) -> str:
    """A small line chart as inline SVG, with a hover label per point."""
    if len(points) < 2:
        return f'<div class="chart"><p class="label">{esc(label)}</p><p class="sub">Not enough steps yet</p></div>'
    W, H, l, r, t, b = 320, 120, 40, 8, 8, 20
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = (y1 - y0) * 0.1 or 0.01
    y0, y1 = y0 - pad, y1 + pad
    px = lambda x: l + (x - x0) / ((x1 - x0) or 1) * (W - l - r)
    py = lambda y: t + (1 - (y - y0) / (y1 - y0)) * (H - t - b)
    path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
    grid = "".join(
        f'<line x1="{l}" x2="{W - r}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="var(--grid)"/>'
        f'<text x="{l - 4}" y="{py(v) + 4:.1f}" text-anchor="end" fill="var(--muted)" font-size="10">{v:.{digits}f}</text>'
        for v in (y0 + (y1 - y0) * k / 3 for k in range(4)))
    dots = "".join(
        f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="6" fill="transparent" '
        f'data-tip="step {x:,.0f}: {y:.{digits + 1}f}{esc(unit)}"/>'
        for x, y in points[:: max(1, len(points) // 60)])
    ends = (f'<text x="{l}" y="{H - 5}" fill="var(--muted)" font-size="10">{x0:,.0f}</text>'
            f'<text x="{W - r}" y="{H - 5}" text-anchor="end" fill="var(--muted)" font-size="10">'
            f'step {x1:,.0f}</text>')
    return (f'<div class="chart"><p class="label">{esc(label)} <span class="muted">{ys[-1]:.{digits + 1}f}{esc(unit)}</span></p>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label)}">{grid}{ends}'
            f'<polyline points="{path}" fill="none" stroke="var(--s1)" stroke-width="2" '
            f'stroke-linejoin="round"/>{dots}</svg></div>')


def render_scratch(s: dict) -> str:
    sc, now = s.get("scratch"), s["now"]
    if not sc or not sc["last"]:
        return ""
    last, sp, every = sc["last"], sc["selfplay"], sc["every"]
    tiles = [("Training step", f'{last["step"]:,}', f'{last["trained"] / 1e6:.1f}M samples seen, '
              f'{last["rate"]:,} samples/s'),
             ("Window", f'{last["window"]:,}', f'rows in {last["shards"]} shards, {last["admitted"]:,} admitted')]
    if sp:
        tiles.append(("Self-play", f'{sp["games"]:,} games',
                      f'{sp["positions"]:,} positions, {sp["rate"]:,} games/h'))
    nxt = f'running since {sc["evaluating"]:%H:%M}' if sc["evaluating"] else when(sc["next_eval"], now)
    tiles.append(("Next evaluation", esc(nxt),
                  f'every {every["hours"]:g} h of running time, {every["games"]} games, '
                  f'accept at {every["accept"]:.2f}' if every else ""))
    if "held_policy" in last:
        tiles.append(("Held-out policy", f'{last["held_policy"]:.3f}',
                      f'cross-entropy; value sign accuracy {last["sign_acc"]:.1%}'))
    curve = sc["curve"]
    charts = (spark([(x["step"], x["held_policy"]) for x in curve], "Held-out policy cross-entropy", "", 2)
              + spark([(x["step"], x["sign_acc"] * 100) for x in curve], "Held-out value sign accuracy", "%", 1)
              + spark([(x["step"], x["value"]) for x in curve], "Training value loss", "", 2))
    rows = "".join(
        f'<tr class="{"promoted" if e["decision"] == "accepted" else "rejected"}"><th>{esc(e["candidate"])}</th>'
        f'<td>{esc(e["time"][:16])}</td><td class="num">{signed(e["elo"])} <span class="muted">('
        f'{signed(e["lo"])} to {signed(e["hi"])})</span></td><td class="num">{e["games"]}</td>'
        f'<td>{esc(e["decision"])}</td></tr>' for e in reversed(sc["evals"]))
    table = ('<div class="scroll"><table><thead><tr><th>Candidate</th><th>Played</th>'
             '<th class="num">Elo vs accepted (95%)</th><th class="num">Games</th><th>Outcome</th></tr></thead>'
             f'<tbody>{rows}</tbody></table></div>' if rows else
             '<p class="sub">No evaluation has finished yet.</p>')
    acc = f' Accepted network: {esc(sc["accepted"])}.' if sc["accepted"] else ""
    cards = "".join(f'<div class="tile"><p class="label">{k}</p><p class="value">{v}</p><p class="sub">{sub}</p></div>'
                    for k, v, sub in tiles)
    return ('<section class="card"><h2>Scratch run <span class="muted">random network, continuous '
            f'training</span></h2><div class="tiles">{cards}</div><div class="charts">{charts}</div>'
            f'<h3>Evaluations</h3>{table}<p class="caption">Each evaluation plays the latest published '
            f'network against the accepted one.{acc}</p></section>')


def render_table(s: dict) -> str:
    c, now = s["current"], s["now"]
    out = []
    if c["stage"] != "idle":
        games = f'{c["games"]:,}' if "games" in c else ""
        label = {"selfplay": "self-play", "export": "exporting", "recording": "recording"}.get(
            c["stage"], c["stage"])
        out.append(f'<tr class="live"><th>{esc(c["gen"])}</th><td>{esc(c["base"])}</td>'
                   f'<td class="num">{c["hours"]:g} h</td><td class="num">{games}</td>'
                   f'<td>{esc(" + ".join(c.get("trained_on", [])))}</td>'
                   f'<td colspan="2">{label} in progress</td><td></td></tr>')
    for r in reversed(s["history"]):
        g = r["gate"]
        if g and g["elo"] is not None:
            gate = (f'<td class="num">{signed(g["elo"])} <span class="muted">({signed(g["lo"])} to '
                    f'{signed(g["hi"])})</span></td><td>{g["games"]} games, {verdict(g)[5:]}</td>')
        else:
            gate = '<td class="muted" colspan="2">installed without a gate</td>'
        outcome = {"yes": "promoted", "no": "rejected"}.get(r["promoted"], r["promoted"])
        hours = f'{r["hours"]:g} h' if r["hours"] else ""
        fin = f'{r["finished"]:%-d %b}' if r["finished"] else ""
        out.append(f'<tr class="{outcome}"><th>{esc(r["gen"])}</th><td>{esc(r["base"] or "")}</td>'
                   f'<td class="num">{hours}</td><td class="num">{r["games"]:,}</td>'
                   f'<td>{esc(" + ".join(r["trained_on"]))}</td>{gate}'
                   f'<td>{outcome} <span class="muted">{fin}</span></td></tr>')
    return ('<section class="card"><h2>Generations</h2><div class="scroll"><table>'
            '<thead><tr><th>Gen</th><th>Played by</th><th class="num">Self-play</th>'
            '<th class="num">Games</th><th>Candidate trained on</th><th class="num">Gate Elo (95%)</th>'
            '<th>Test (+0 vs +15 Elo)</th><th>Outcome</th></tr></thead><tbody>'
            + "".join(out) + "</tbody></table></div></section>")


def render(s: dict) -> str:
    sp = s["selfplay"]
    setup = []
    if sp.get("sims"):
        setup.append(f'{s["current"]["hours"]:g} h of self-play per generation, {sp["parallel_games"]} '
                     f'games at once; {sp["sims"]} sims on {sp["full_frac"]:.0%} of moves, '
                     f'{sp["fast_sims"]} on the rest')
    if PAUSED_NIGHTLY:
        setup.append(f"GPU work paused {PAUSE_HOUR}:00 to 0{RESUME_HOUR}:00")
    last = esc(s["last_entry"] or "")
    page = (
        f'<main id="page"><header><div><h1>RL training</h1>'
        f'<p class="sub">Updated {s["now"]:%H:%M:%S}, refreshes every minute'
        f'<span id="stale" hidden> (not reachable)</span></p></div>{render_state(s)}</header>'
        f'{render_scratch(s)}{render_tiles(s)}{render_now(s)}{render_strength(s)}{render_table(s)}'
        f'<footer><p>{esc("; ".join(setup))}.</p><p class="mono">Last log entry: {last}</p></footer></main>')
    return HEAD + page + '<div id="tip" role="tooltip" hidden></div><script>' + SCRIPT + "</script></body></html>"


HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RL training status</title><style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;
--s1-track:#cde2fb;--s2:#eb6834;--band:.1;--good:#0ca30c;--warning:#fab219;--critical:#d03b3b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--s1:#3987e5;--s1-track:#184f95;--s2:#d95926;--band:.2}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;
--s1-track:#184f95;--s2:#d95926;--band:.2}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:22px;margin:0}h2{font-size:16px;margin:0 0 12px}
header{display:flex;flex-wrap:wrap;gap:8px 24px;justify-content:space-between;align-items:center;margin-bottom:16px}
.sub,.muted{color:var(--ink-2)}.sub{margin:2px 0 0;font-size:13px}.muted{font-weight:400}
.state{margin:0;padding:6px 12px;border-radius:999px;background:var(--surface);border:1px solid var(--border);font-weight:600;font-size:14px}
.state .icon{margin-right:8px;font-size:12px}.state.running .icon{color:var(--good)}
.state.paused .icon{color:var(--muted)}.state.stopped .icon{color:var(--critical)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-bottom:8px}
.chart svg{width:100%;height:auto;display:block}.chart .label{margin:0 0 2px;font-size:13px;color:var(--ink-2)}
h3{font-size:15px;margin:12px 0 6px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:12px}
.tile,.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.card{margin-bottom:12px}
.tile .label{margin:0;font-size:13px;color:var(--ink-2)}.tile .value{margin:2px 0 0;font-size:24px;font-weight:600}
.warn{color:var(--ink);font-weight:600}
.stepper{display:flex;gap:8px;list-style:none;padding:0;margin:0 0 14px}
.stepper li{flex:1;display:flex;align-items:center;gap:8px;font-size:14px;color:var(--muted);padding-bottom:8px;border-bottom:3px solid var(--grid)}
.stepper .n{display:inline-grid;place-items:center;width:22px;height:22px;border-radius:50%;border:1.5px solid currentColor;font-size:12px;font-weight:600}
.stepper .done{color:var(--ink-2);border-color:var(--s1)}
.stepper .active{color:var(--ink);font-weight:600;border-color:var(--s1)}
.stepper .active .n{background:var(--s1);border-color:var(--s1);color:#fff}
.lead{margin:0 0 8px;font-size:18px;font-weight:600}
.meter{height:8px;border-radius:4px;background:var(--s1-track);overflow:hidden}
.meter span{display:block;height:100%;background:var(--s1);border-radius:4px}
.facts{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:12px 0 0;font-size:14px}
.facts dt{color:var(--ink-2)}.facts dd{margin:0}
.llr{margin:10px 0 0}.llr .track{position:relative;height:8px;border-radius:4px;background:var(--grid)}
.llr .zero{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--axis)}
.llr .mark{position:absolute;top:-4px;width:16px;height:16px;margin-left:-8px;border-radius:50%;background:var(--ink);border:2px solid var(--surface)}
.llr .ends{display:flex;justify-content:space-between;font-size:12px;color:var(--ink-2);margin-top:6px}
.hero{margin:0;font-size:48px;font-weight:600;line-height:1.1}.hero .unit{font-size:20px;color:var(--ink-2)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;list-style:none;padding:0;margin:12px 0 4px;font-size:13px;color:var(--ink-2)}
.legend li{display:flex;align-items:center;gap:6px}
.key{display:inline-block}.key.line{width:16px;height:2px;background:var(--s1)}
.key.dot{width:10px;height:10px;border-radius:50%}.key.s1{background:var(--s1)}.key.s2{background:var(--s2)}
.key.whisker{width:2px;height:12px;background:var(--muted)}
#ladder{width:100%;min-height:220px}
#ladder svg{display:block}
#ladder .grid{stroke:var(--grid);stroke-width:1}#ladder .base{stroke:var(--axis)}
#ladder .tick{fill:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
#ladder .tick.now{fill:var(--ink);font-weight:600}
#ladder .band{fill:var(--s1);opacity:var(--band)}
#ladder .step{fill:none;stroke:var(--s1);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
#ladder .whisker{stroke:var(--muted);stroke-width:1.5}
#ladder .pt{stroke:var(--surface);stroke-width:2}#ladder .pt.s1{fill:var(--s1)}#ladder .pt.s2{fill:var(--s2)}
#ladder .hit{fill:transparent;cursor:pointer}
#ladder .end{fill:var(--ink);font-size:13px;font-weight:600}
.caption{font-size:13px;color:var(--ink-2);margin:6px 0 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid)}
thead th{font-size:12px;color:var(--ink-2);font-weight:600}
tbody th{font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}
tr.live td,tr.live th{background:color-mix(in srgb,var(--s1) 8%,transparent)}
tr.rejected th,tr.rejected td{color:var(--ink-2)}
footer{font-size:13px;color:var(--ink-2)}footer p{margin:4px 0}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;overflow-wrap:anywhere}
#tip{position:fixed;z-index:10;max-width:280px;padding:6px 10px;border-radius:6px;background:var(--ink);color:var(--surface);font-size:13px;pointer-events:none}
</style></head><body>"""

SCRIPT = r"""
const tip = document.getElementById('tip');
function showTip(el, ev) {
  tip.textContent = el.dataset.tip; tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 12, y = ev.clientY + 12;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 12;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 12;
  tip.style.left = Math.max(8, x) + 'px'; tip.style.top = Math.max(8, y) + 'px';
}
for (const type of ['pointermove', 'pointerdown']) document.addEventListener(type, ev => {
  const el = ev.target.closest && ev.target.closest('[data-tip]');
  if (el) showTip(el, ev); else tip.hidden = true;
});
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const fmt = v => (v > 0 ? '+' : v < 0 ? '\u2212' : '') + Math.abs(Math.round(v));
function niceStep(raw) {
  for (const s of [5, 10, 20, 25, 50, 100, 200]) if (s >= raw) return s;
  return 500;
}
function drawLadder() {
  const box = document.getElementById('ladder'), src = document.getElementById('ladder-data');
  if (!box || !src) return;
  const d = JSON.parse(src.textContent);
  const W = box.clientWidth, H = W < 480 ? 230 : 290;
  const m = {l: 44, r: 52, t: 12, b: 28};
  const vals = [0];
  d.champion.forEach(c => vals.push(c.elo - c.ci, c.elo + c.ci));
  d.candidates.forEach(c => vals.push(c.lo, c.hi));
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const step = niceStep((hi - lo) / 5);
  lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const X = n => m.l + (n - d.x0) / Math.max(1, d.x1 - d.x0) * (W - m.l - m.r);
  const Y = v => m.t + (hi - v) / (hi - lo) * (H - m.t - m.b);
  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(d.label)}">`;
  for (let v = lo; v <= hi + 1e-9; v += step) {
    s += `<line class="grid${v === 0 ? ' base' : ''}" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/>`;
    s += `<text class="tick" x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${fmt(v)}</text>`;
  }
  const every = (W - m.l - m.r) / Math.max(1, d.x1 - d.x0) < 24 ? 2 : 1;
  for (let n = d.x0; n <= d.x1; n++) {
    if ((d.x1 - n) % every) continue;
    const now = n === d.x1 ? ' now' : '';
    s += `<text class="tick${now}" x="${X(n)}" y="${H - 8}" text-anchor="middle">${n}</text>`;
  }
  const ch = d.champion;
  ch.forEach((c, i) => {
    const next = i + 1 < ch.length ? ch[i + 1].n : d.x1;
    if (c.ci > 0) s += `<rect class="band" x="${X(c.n)}" width="${Math.max(0, X(next) - X(c.n))}" y="${Y(c.elo + c.ci)}" height="${Y(c.elo - c.ci) - Y(c.elo + c.ci)}"/>`;
  });
  let path = `M${X(ch[0].n)},${Y(ch[0].elo)}`;
  for (let i = 1; i < ch.length; i++) path += `H${X(ch[i].n)}V${Y(ch[i].elo)}`;
  path += `H${X(d.x1)}`;
  s += `<path class="step" d="${path}"/>`;
  s += `<circle class="pt s1" cx="${X(ch[0].n)}" cy="${Y(0)}" r="4.5"/>`;
  s += `<circle class="hit" cx="${X(ch[0].n)}" cy="${Y(0)}" r="12" data-tip="${esc(ch[0].gen + ': the starting point, 0 by definition')}"/>`;
  for (const c of d.candidates) {
    s += `<line class="whisker" x1="${X(c.n)}" x2="${X(c.n)}" y1="${Y(c.hi)}" y2="${Y(c.lo)}"/>`;
    s += `<circle class="pt ${c.promoted ? 's1' : 's2'}" cx="${X(c.n)}" cy="${Y(c.elo)}" r="4.5"/>`;
  }
  for (const c of d.candidates) s += `<circle class="hit" cx="${X(c.n)}" cy="${Y(c.elo)}" r="12" data-tip="${esc(c.tip)}"/>`;
  const last = ch[ch.length - 1];
  s += `<text class="end" x="${X(d.x1) + 8}" y="${Y(last.elo) + 4}">${fmt(last.elo)}</text>`;
  box.innerHTML = s + '</svg>';
}
async function refresh() {
  try {
    const r = await fetch(location.pathname, {cache: 'no-store'});
    if (!r.ok) throw new Error(r.status);
    const doc = new DOMParser().parseFromString(await r.text(), 'text/html');
    document.getElementById('page').replaceWith(doc.getElementById('page'));
    drawLadder();
  } catch (e) {
    const el = document.getElementById('stale'); if (el) el.hidden = false;
  }
}
setInterval(refresh, 60000);
addEventListener('resize', drawLadder);
drawLadder();
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/":
                body, ctype = render(cached_status()).encode(), "text/html; charset=utf-8"
            elif path == "/status.json":
                body = json.dumps(cached_status(), default=str, indent=1).encode()
                ctype = "application/json"
            else:
                self.send_error(404)
                return
        except Exception as e:  # keep serving; show what broke
            body, ctype = f"status failed: {type(e).__name__}: {e}".encode(), "text/plain"
            self.send_response(500)
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--json", action="store_true", help="print the status once and exit")
    p.add_argument("--pause", default="22-8",
                   help="nightly pause the estimates skip, START-END hours, or none (default 22-8)")
    args = p.parse_args()
    global PAUSED_NIGHTLY, PAUSE_HOUR, RESUME_HOUR
    if args.pause == "none":
        PAUSED_NIGHTLY = False
    else:
        PAUSE_HOUR, RESUME_HOUR = (int(h) for h in args.pause.split("-"))
    if args.json:
        print(json.dumps(status(), default=str, indent=1))
        return
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"RL status on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
