"""Match two networks on the GPU with the deployed bot's search, all games at once.

Each side of each game is ct-bot's own search (``chess_core::search``, the code
the lichess bot runs) and plays the way the bot plays a lichess game
(``chess_core::matchplay``): the tree is kept between moves, a move gets the
per-speed cap of fresh simulations clipped by the warm visits already on the
root, and after moving the side ponders the position after its move until the
opponent replies. The defaults are the deployed blitz/bullet settings
(--max-sims 4000, --ponder-max-sims 2000, --sim-batch 16, c_puct 1.0, fpu 0.2,
no tablebases). The Rust driver advances every search one wave at a time and
returns all pending positions; here each network evaluates its share in one
forward. What differs from lichess: no clock (every move is cap-bound, as
81-100 % of live moves are), no opening temperature (book starts are past ply
2), and bf16 torch numerics instead of the int8 ONNX graph.

Games: openings from the book, each played twice with colours swapped; draw by
threefold / 50-move / insufficient material, or at --max-plies; no resignation.
The games file has head_to_head.py's columns (idx, start, a_white, result,
score_a, plies), so every scorer of gate files reads it unchanged.

Usage
-----
    uv run python scripts/gpu_match.py --model-a data/rl/models/cand02 \\
        --model-b data/rl/models/gen01 --openings 200 --games-out gate.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import selfplay_core  # noqa: E402
from selfplay_rust import Forward  # noqa: E402

from chesstransformer.evaluation.sprt import (  # noqa: E402
    format_report,
    pair_up,
    sprt_decision,
    sprt_llr,
    summarize,
)


def schedule(fens: list[str]) -> tuple[list[str], list[bool]]:
    """Game 2i is opening i with A white, game 2i+1 the same opening with B white."""
    return [f for f in fens for _ in (0, 1)], [a for _ in fens for a in (True, False)]


class Half:
    """One MatchEngine over every `k`-th opening (both colours of an opening stay in
    one engine). Local game j is opening `openings[j // 2]`."""

    def __init__(self, cfg: dict, lines: list[str], openings: list[int]):
        self.openings = openings
        fens, a_white = schedule([lines[o] for o in openings])
        self.engine = selfplay_core.MatchEngine(json.dumps(cfg), fens, a_white)
        self.handles = None  # [(rows, handle)] of the batch in flight, one per network

    def launch(self, forwards, slot: int):
        boards, player, castling, ep, nets, legal = self.engine.collect()
        self.n, self.width = len(player), legal.shape[1]
        self.handles = []
        for net, fwd in enumerate(forwards):
            rows = np.flatnonzero(nets == net)
            if len(rows):
                self.handles.append((rows, fwd, fwd.launch(2 * slot + net, boards[rows], player[rows],
                                                           castling[rows], ep[rows], legal[rows])))

    def apply(self):
        logits = np.empty((self.n, self.width), dtype=np.float32)
        values = np.empty(self.n, dtype=np.float32)
        for rows, fwd, handle in self.handles:
            logits[rows], values[rows] = fwd.wait(handle)
        self.engine.apply_legal(logits, values)
        self.handles = None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-a", required=True, help="model dir (model.safetensors + config)")
    p.add_argument("--model-b", required=True)
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--book", type=Path, default=Path("data/openings/book2k.json"))
    p.add_argument("--openings", type=int, default=100,
                   help="opening lines from the front of the book; each is played twice")
    p.add_argument("--max-sims", type=int, default=4000,
                   help="fresh simulations per move (ct-bot --max-sims; 5000 = deployed rapid)")
    p.add_argument("--ponder-max-sims", type=int, default=2000,
                   help="ponder budget after each move (ct-bot --ponder-max-sims); 0 = no ponder")
    p.add_argument("--sim-batch", type=int, default=16, help="leaves per search wave (ct-bot)")
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--fpu", type=float, default=0.2)
    p.add_argument("--prior-temp", type=float, default=1.0,
                   help="softmax temperature of the policy priors (ct-bot --prior-temp)")
    for side in ("a", "b"):  # per-net overrides of the three search settings
        p.add_argument(f"--c-puct-{side}", type=float, default=None)
        p.add_argument(f"--fpu-{side}", type=float, default=None)
        p.add_argument(f"--prior-temp-{side}", type=float, default=None)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--parallel-games", type=int, default=128, help="games in flight, all halves")
    p.add_argument("--engines", type=int, default=2,
                   help="engine halves taking turns on the GPU (2 = the CPU searches one while "
                        "the GPU evaluates the other); 1 = no overlap")
    p.add_argument("--threads", type=int, default=4,
                   help="CPU threads for the searches (the lichess bot shares the CPU by day)")
    p.add_argument("--tt-cap", type=int, default=150_000,
                   help="TT entries per search; it only caches evaluations, so it changes the "
                        "GPU work, not the moves")
    p.add_argument("--sprt", action="store_true",
                   help="stop once the paired SPRT decides H0 vs H1 (games in flight are dropped)")
    p.add_argument("--elo0", type=float, default=0.0)
    p.add_argument("--elo1", type=float, default=15.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--min-pairs", type=int, default=20, help="never stop before this many pairs")
    p.add_argument("--verify-every", type=int, default=50,
                   help="recompute every N-th async batch synchronously and compare (0 = never)")
    p.add_argument("--games-out", type=Path, default=None, help="append one JSON line per game")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    la = args.label_a or Path(args.model_a).name
    lb = args.label_b or Path(args.model_b).name

    lines = [o["fen"] for o in json.loads(args.book.read_text())["openings"]]
    if not 1 <= args.openings <= len(lines):
        p.error(f"--openings must be in 1..{len(lines)}")
    k_eng = max(1, min(args.engines, args.openings))
    cfg = {
        "sims": args.max_sims,
        "ponder_sims": args.ponder_max_sims,
        "sim_batch": args.sim_batch,
        "c_puct": args.c_puct,
        "fpu": args.fpu,
        "prior_temp": args.prior_temp,
        "max_plies": args.max_plies,
        "num_parallel_games": max(1, args.parallel_games // k_eng),
        "threads": args.threads,
        "tt_cap": args.tt_cap,
    }
    search = {}
    for side in ("a", "b"):
        for key in ("c_puct", "fpu", "prior_temp"):
            v = getattr(args, f"{key}_{side}")
            cfg[f"{key}_{side}"] = v if v is not None else cfg[key]
        search[side] = (f"c_puct {cfg[f'c_puct_{side}']}, fpu {cfg[f'fpu_{side}']}, "
                        f"prior_temp {cfg[f'prior_temp_{side}']}")
    halves = [Half(cfg, lines, list(range(k, args.openings, k_eng))) for k in range(k_eng)]
    # Two compiled networks share the card: capture graphs up to 512 rows only (larger
    # batches are split), half of self-play's set.
    forwards = [Forward(m, args.device, False, not args.no_compile, buckets=(64, 128, 256, 512))
                for m in (args.model_a, args.model_b)]
    for fwd in forwards:
        fwd.verify_every = args.verify_every
    n_games = 2 * args.openings
    print(f"{la}: {args.model_a} ({search['a']})\n{lb}: {args.model_b} ({search['b']})\n"
          f"{args.openings} openings x 2 colours = {n_games} games, {args.max_sims} sims/move "
          f"+ {args.ponder_max_sims} ponder, {min(args.parallel_games, n_games)} at once"
          f"{', SPRT' if args.sprt else ''}\n", flush=True)

    games_out = args.games_out.open("a") if args.games_out else None
    results: dict[tuple[int, bool], float] = {}
    decided = None
    t0 = time.time()
    while decided is None and not all(h.engine.done() and h.handles is None for h in halves):
        for k, half in enumerate(halves):
            if half.handles is not None:
                half.apply()
            for g in half.engine.drain_finished():
                idx, aw = half.openings[g["id"] // 2], g["a_white"]
                z = g["z_white"]
                result = "1-0" if z > 0 else "0-1" if z < 0 else "1/2-1/2"
                s = 0.5 if z == 0 else (1.0 if (z > 0) == aw else 0.0)
                results[(idx, aw)] = s
                tag = "=" if s == 0.5 else ("+" if s == 1.0 else "-")
                print(f"[{len(results):3}/{n_games}] {la if aw else lb:10}=W  opening {idx:4} "
                      f"-> {result:7} ({la} {tag}) {len(g['moves'])} plies, {g['end']} "
                      f"| {time.time() - t0:.0f}s", flush=True)
                if games_out:
                    games_out.write(json.dumps({
                        "idx": idx, "start": g["fen"], "a_white": aw, "result": result,
                        "score_a": s, "plies": len(g["moves"]), "end": g["end"]}) + "\n")
                    games_out.flush()
                if args.sprt and decided is None:
                    pairs = pair_up(results)
                    llr = sprt_llr(pairs, args.elo0, args.elo1)
                    verdict = sprt_decision(llr, args.alpha, args.beta,
                                            min_pairs_met=len(pairs) >= args.min_pairs)
                    if verdict:
                        decided = (verdict, llr, len(pairs))
                        print(f"  SPRT decided {verdict} after {len(pairs)} pairs (LLR {llr:+.2f}); "
                              f"games in flight are dropped", flush=True)
            if decided is not None:
                break
            if not half.engine.done():
                half.launch(forwards, k)
    if games_out:
        games_out.close()

    pairs = pair_up(results)
    scores = list(results.values())
    llr = decided[1] if decided else sprt_llr(pairs, args.elo0, args.elo1)
    print("\n" + format_report(summarize(pairs, scores), la, lb, llr, args.elo0, args.elo1,
                               args.alpha, args.beta))
    print(f"\n{la} score {sum(scores) / len(scores):.3f} over {len(scores)} games "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
