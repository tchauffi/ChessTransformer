"""Browse this machine's self-play games on the web demo's board: the games viewer's server.

It serves the frontend's static export (frontend/out, `make viewer` builds it) and a JSON API
over the self-play runs here: the from-scratch run (data/rl/scratch), the continuous rig
(data/rl/continuous) and every generation (data/rl/selfplay/genNN). A game comes with what
its self-play searched: for each position the positions_*.npz row the trainer reads, i.e. the
search's root value, the network's own value and the visited moves with their visits, priors
and values. Games still waiting in an unwritten shard come without it.

    .venv/bin/python scripts/games_viewer.py --host 0.0.0.0      # http://<host>:8766/games

API (JSON): /api/runs; /api/games?run=&net=&result=&order=new|old|short|long&offset=&limit=;
/api/game?run=&id=. Values are for White, in (-1, 1), as the demo's evaluation bar reads them;
a move's q is for the side playing it, as its search view reads it.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "data/rl"
OUT = ROOT / "frontend/out"
sys.path.insert(0, str(ROOT / "src"))
from chesstransformer.models.tokenizer.alphazero_move_encoder import move_to_action_plane  # noqa: E402

NPL = 73  # action planes per from-square: a move's index is from_square * 73 + plane
TOP = 12  # visited moves sent per position


def run_dirs() -> list[tuple[str, str, Path]]:
    """(id, label, self-play directory) of every run with games here, the live ones first."""
    out = [(rid, label, d) for rid, label, d in (
        ("scratch", "From scratch", RL / "scratch/selfplay"),
        ("continuous", "Continuous rig", RL / "continuous/selfplay"))
        if (d / "games.jsonl").exists()]
    out += [(d.name, f"{d.name} (generational)", d) for d in sorted((RL / "selfplay").glob("gen[0-9][0-9]"), reverse=True)
            if (d / "games.jsonl").exists()]
    return out


def net_step(net: str | None) -> int | None:
    """The training step of a continuous run's publication (step_0012000 -> 12000)."""
    if not net or not net.startswith("step_"):
        return None
    return int(net.split("_")[1])


class Run:
    """A run's games (summaries, read as games.jsonl grows) and where each game's rows lie in
    its shards (a game's positions are written together, in one shard, in order)."""

    def __init__(self, rid: str, label: str, path: Path):
        self.id, self.label, self.path = rid, label, path
        self.lock = threading.Lock()
        self.games: list[dict] = []
        self.by_id: dict[int, int] = {}
        self.read = 0  # bytes of games.jsonl parsed
        self.shards: set[str] = set()
        self.rows: dict[int, tuple[Path, int, int]] = {}  # game id -> shard, first row, rows

    def refresh(self) -> None:
        with self.lock:
            log = self.path / "games.jsonl"
            size = log.stat().st_size
            if size > self.read:
                with open(log, "rb") as f:
                    f.seek(self.read)
                    data = f.read(size - self.read)
                data = data[:data.rfind(b"\n") + 1]  # the last line may still be being written
                at = self.read
                for line in data.splitlines(keepends=True):
                    g = json.loads(line)
                    self.by_id[g["id"]] = len(self.games)
                    self.games.append({"id": g["id"], "net": net_step(g.get("net")), "result": g["result"],
                                       "end": g["end"], "plies": g["plies"], "fen": g["fen"],
                                       "_at": at, "_len": len(line)})
                    at += len(line)
                self.read += len(data)
            for p in sorted(self.path.glob("positions_*.npz")):  # complete: written by rename
                if p.name in self.shards:
                    continue
                ids = np.load(p)["game_id"]
                self.shards.add(p.name)
                starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
                for s, e in zip(starts, np.r_[starts[1:], len(ids)]):
                    self.rows[int(ids[s])] = (p, int(s), int(e - s))

    def info(self) -> dict:
        nets: dict[int, int] = {}
        for g in self.games:
            if g["net"] is not None:
                nets[g["net"]] = nets.get(g["net"], 0) + 1
        return {"id": self.id, "label": self.label, "games": len(self.games),
                "nets": [{"net": n, "games": c} for n, c in sorted(nets.items(), reverse=True)]}

    def query(self, q: dict[str, str]) -> dict:
        games = self.games
        if q.get("net", "all") != "all":
            games = [g for g in games if g["net"] == int(q["net"])]
        if q.get("result", "all") != "all":
            games = [g for g in games if g["result"] == q["result"]]
        order = q.get("order", "new")
        if order == "new":
            games = games[::-1]
        elif order in ("short", "long"):
            games = sorted(games, key=lambda g: g["plies"], reverse=order == "long")
        offset, limit = int(q.get("offset", 0)), min(200, int(q.get("limit", 50)))
        page = [{k: v for k, v in g.items() if not k.startswith("_")} for g in games[offset:offset + limit]]
        return {"total": len(games), "games": page}

    def game(self, gid: int) -> dict:
        s = self.games[self.by_id[gid]]
        with open(self.path / "games.jsonl", "rb") as f:
            f.seek(s["_at"])
            g = json.loads(f.read(s["_len"]))
        out = {k: v for k, v in s.items() if not k.startswith("_")} | {"moves": g["moves"], "positions": None}
        if gid in self.rows:
            p, first, n = self.rows[gid]
            if n == len(g["moves"]):
                out["positions"] = positions(p, first, g["fen"], g["moves"])
        return out


@lru_cache(maxsize=4)
def shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as d:
        return {k: d[k] for k in ("player", "root_v", "root_net_v", "full", "legal_ptr", "legal_idx",
                                  "legal_prior", "legal_n", "legal_q") if k in d.files}


def positions(path: Path, first: int, fen: str, moves: list[str]) -> list[dict] | None:
    """Per position of the game, its search: values for White and the visited moves."""
    d = shard(path)
    if "legal_ptr" not in d:  # an engine that did not record the root search
        return None
    board, out = chess.Board(fen), []
    for i, uci in enumerate(moves):
        r = first + i
        white = 1.0 if d["player"][r] == 1 else -1.0
        legal = {m.from_square * NPL + move_to_action_plane(m.from_square, m.to_square, m.promotion): m
                 for m in board.legal_moves}
        a, b = int(d["legal_ptr"][r]), int(d["legal_ptr"][r + 1])
        n = d["legal_n"][a:b].astype(int)
        top = []
        for k in np.argsort(-n, kind="stable")[:TOP]:
            m = legal.get(int(d["legal_idx"][a + k]))
            if m is None or n[k] == 0:
                continue
            top.append({"move": m.uci(), "visits": int(n[k]), "prior": round(float(d["legal_prior"][a + k]), 4),
                        "q": round(float(d["legal_q"][a + k]), 4)})
        out.append({"v": round(white * float(d["root_v"][r]), 4),
                    "net_v": round(white * float(d["root_net_v"][r]), 4) if "root_net_v" in d else None,
                    "full": bool(d["full"][r]), "visits": int(n.sum()), "top": top})
        board.push_uci(uci)
    return out


RUNS: dict[str, Run] = {}


def get_run(rid: str) -> Run:
    for i, label, path in run_dirs():
        if i == rid:
            if rid not in RUNS or RUNS[rid].path != path:
                RUNS[rid] = Run(i, label, path)
            RUNS[rid].refresh()
            return RUNS[rid]
    raise KeyError(rid)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/runs":
                return self.json([get_run(rid).info() for rid, _, _ in run_dirs()])
            if url.path == "/api/games":
                return self.json(get_run(q["run"]).query(q))
            if url.path == "/api/game":
                return self.json(get_run(q["run"]).game(int(q["id"])))
        except (KeyError, ValueError) as e:
            return self.json({"error": f"not found: {e}"}, 404)
        if url.path == "/":
            self.send_response(302)
            self.send_header("Location", "/games")
            self.end_headers()
            return
        self.static(url.path)

    def static(self, path: str) -> None:
        if not OUT.exists():
            return self.json({"error": "build the page first: make viewer"}, 503)
        base = (OUT / path.lstrip("/")).resolve()
        if not base.is_relative_to(OUT.resolve()):
            return self.json({"error": "not found"}, 404)
        for f in (base, base.with_name(base.name + ".html"), base / "index.html"):
            if f.is_file():
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        self.json({"error": "not found"}, 404)

    def json(self, data, code: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")  # `next dev` on another port
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # quiet
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"games viewer on http://{args.host}:{args.port}/games ({len(run_dirs())} runs)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
