"""MatchEngine: two networks, a fixed schedule, ct-bot's search per side.

The move-for-move equivalence with the bot's own flow (reuse + pondering through
go()) is tested in Rust: chess-core matchplay::tests.
"""

import json

import chess
import numpy as np
import pytest

selfplay_core = pytest.importorskip("selfplay_core")

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ITALIAN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
FLAT = (np.arange(64)[:, None] * 73 + np.arange(73)[None, :]).astype(np.float32)


def _play(eng, forward, gathered=True):
    """Drive a MatchEngine; `gathered` feeds only the legal moves' logits (apply_legal)."""
    out = []
    while not eng.done():
        boards, player, castling, ep, nets, legal = eng.collect()
        if len(player):
            logits, values = forward(nets, boards)
        else:
            logits, values = np.zeros((0, 64, 73), np.float32), np.zeros(0, np.float32)
        if gathered:
            flat = np.concatenate([logits.reshape(len(logits), 64 * 73),
                                   np.zeros((len(logits), 1), np.float32)], 1)
            eng.apply_legal(np.take_along_axis(flat, legal.astype(np.int64), 1), values)
        else:
            eng.apply(np.ascontiguousarray(logits), values)
        out += eng.drain_finished()
    return out


def _opposed_nets(nets, boards):
    """Net A wants the legal move with the lowest from*73+plane index, net B the highest."""
    sign = np.where(nets == 0, -1.0, 1.0).astype(np.float32)
    # 5 logits per index step: even two moves from one square are ~99/1 in the prior.
    return sign[:, None, None] * FLAT[None] * 5.0, np.zeros(len(nets), np.float32)


def test_each_side_plays_its_own_network():
    # One leaf per wave and no ponder, so the most-visited move is the prior's argmax.
    cfg = dict(sims=8, ponder_sims=0, sim_batch=1, max_plies=30, threads=2)
    fens, a_white = [START, START, ITALIAN, ITALIAN], [True, False, True, False]
    games = _play(selfplay_core.MatchEngine(json.dumps(cfg), fens, a_white), _opposed_nets)
    assert sorted(g["id"] for g in games) == [0, 1, 2, 3]
    checked = 0
    for g in games:
        assert g["fen"] == fens[g["id"]] and g["a_white"] == a_white[g["id"]]
        board = chess.Board(g["fen"])
        for uci in g["moves"]:
            a_to_move = (board.turn == chess.WHITE) == g["a_white"]
            flat = {u: f * 73 + p for u, f, p in selfplay_core.debug_move_planes(board.fen())}
            want = min(flat, key=flat.get) if a_to_move else max(flat, key=flat.get)
            assert uci == want, (g["id"], board.fen(), uci, want)
            board.push_uci(uci)
            checked += 1
    assert checked > 50


def _position_net(nets, boards):
    h = (boards.astype(np.float32) * np.arange(1, 65, dtype=np.float32)).sum(1) + nets * 1.7
    logits = np.sin(h[:, None, None] * 0.013 + FLAT[None] * 0.37) * 3
    return logits.astype(np.float32), np.tanh(np.sin(h * 0.029)).astype(np.float32)


def test_results_do_not_depend_on_threads_or_games_in_flight():
    # Every search is independent, so neither the thread split nor how many games
    # share a batch may change a single move.
    fens, a_white = [START, START, ITALIAN, ITALIAN, START], [True, False, True, False, True]

    def run(threads, parallel):
        cfg = dict(sims=48, ponder_sims=24, max_plies=24, threads=threads,
                   num_parallel_games=parallel)
        games = _play(selfplay_core.MatchEngine(json.dumps(cfg), fens, a_white), _position_net)
        return sorted((g["id"], g["moves"], g["z_white"], g["end"]) for g in games)

    ref = run(1, 5)
    assert len(ref) == 5 and all(len(m) == 24 or e == "terminal" for _, m, _, e in ref)
    assert run(3, 5) == ref
    assert run(2, 2) == ref


def test_gathered_logits_play_the_same_games():
    fens, a_white = [START, ITALIAN], [True, False]
    cfg = json.dumps(dict(sims=48, ponder_sims=24, max_plies=24, threads=2))

    def run(gathered):
        games = _play(selfplay_core.MatchEngine(cfg, fens, a_white), _position_net, gathered)
        return sorted((g["id"], g["moves"]) for g in games)

    assert run(True) == run(False)


@pytest.mark.parametrize("key,value", [("c_puct", 3.0), ("fpu", 0.6), ("prior_temp", 2.0)])
def test_per_net_search_settings_change_only_that_net(key, value):
    fens, a_white = [START, ITALIAN, START], [True, False, False]
    base = dict(sims=48, ponder_sims=24, max_plies=24, threads=2)

    def run(**extra):
        games = _play(selfplay_core.MatchEngine(json.dumps({**base, **extra}), fens, a_white),
                      _position_net)
        return {g["id"]: g["moves"] for g in games}

    ref = run()
    assert run(**{f"{key}_a": base.get(key, {"c_puct": 1.0, "fpu": 0.2, "prior_temp": 1.0}[key])}) == ref
    for side in ("a", "b"):
        got = run(**{f"{key}_{side}": value})
        assert got != ref
        for gid, moves in got.items():
            first = next((i for i, (x, y) in enumerate(zip(moves, ref[gid])) if x != y), None)
            if first is None:
                continue
            # the first deviation is a move of the net whose setting changed
            a_moves_first = chess.Board(fens[gid]).turn == chess.WHITE
            a_to_move = (first % 2 == 0) == (a_moves_first == a_white[gid])
            assert a_to_move == (side == "a"), (gid, first)
