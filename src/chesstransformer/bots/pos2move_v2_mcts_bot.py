"""AlphaZero-style PUCT MCTS engine for Pos2MoveV2.

Subclasses :class:`Pos2MoveV2Bot` to reuse its (optionally torch.compiled and
cached) ``_evaluate`` — the policy head supplies move priors and the value head
scores leaves. Move selection is by visit count, the standard AlphaZero choice.

This is a correctness-first single-leaf implementation: one network evaluation
per simulation (the compiled batch-1 path is fast). Collecting several leaves
per network call with virtual loss — to better fill the GPU — is a natural
follow-up that this structure leaves room for.
"""

from __future__ import annotations

import math

import chess
import numpy as np

from chesstransformer.bots.pos2move_v2_bot import _BATCH_BUCKETS, _DEFAULT_RUN, Pos2MoveV2Bot
from chesstransformer.models.tokenizer.alphazero_move_encoder import move_to_action_plane


class _Node:
    __slots__ = ("prior", "N", "W", "children")

    def __init__(self, prior: float):
        self.prior = prior
        self.N = 0          # visit count
        self.W = 0.0        # total value, from this node's side-to-move POV
        self.children: dict[chess.Move, _Node] | None = None

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0

    @property
    def expanded(self) -> bool:
        return self.children is not None


class Pos2MoveV2MctsBot(Pos2MoveV2Bot):
    def __init__(
        self,
        *args,
        num_simulations: int = 400,
        c_puct: float = 1.5,
        sim_batch: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        # Leaves collected per network call. >1 amortizes the GPU->CPU sync that
        # otherwise dominates (one stall per simulation). 1 = legacy single-leaf.
        # Capped at the largest batch bucket so _batch_run's preallocated buffers
        # (and the captured CUDA graphs) are never overrun.
        self.sim_batch = max(1, min(sim_batch, _BATCH_BUCKETS[-1]))

    def _priors_and_value(self, board: chess.Board):
        """Legal moves, their softmax policy priors, and the value-head score."""
        logits, value = self._evaluate(board)
        moves = list(board.legal_moves)
        scores = np.array(
            [
                logits[m.from_square, move_to_action_plane(m.from_square, m.to_square, m.promotion)]
                for m in moves
            ],
            dtype=np.float64,
        )
        scores -= scores.max()
        exp = np.exp(scores)
        probs = exp / exp.sum()
        return moves, probs, value

    def _expand(self, node: _Node, board: chess.Board) -> float:
        self.nodes_searched += 1
        moves, probs, value = self._priors_and_value(board)
        node.children = {m: _Node(float(p)) for m, p in zip(moves, probs)}
        return value

    @staticmethod
    def _softmax_priors(moves: list[chess.Move], logits: np.ndarray) -> np.ndarray:
        scores = np.array(
            [
                logits[m.from_square, move_to_action_plane(m.from_square, m.to_square, m.promotion)]
                for m in moves
            ],
            dtype=np.float64,
        )
        scores -= scores.max()
        exp = np.exp(scores)
        return exp / exp.sum()

    def _expand_from_logits(self, node: _Node, moves: list[chess.Move], logits: np.ndarray):
        self.nodes_searched += 1
        probs = self._softmax_priors(moves, logits)
        node.children = {m: _Node(float(p)) for m, p in zip(moves, probs)}

    @staticmethod
    def _backup_vl(path: list[_Node], leaf_value: float):
        """Back up a leaf value, undoing the virtual loss applied during descent.

        Descent did ``N += 1; W += 1`` on each path node (a virtual *win* in the
        node's own POV, which looks like a loss to its parent under the negamax
        ``-child.Q`` selection, so concurrent descents avoid the path). Here we
        subtract that +1 back out and apply the real, sign-alternating value. N
        is left as-is since the virtual-loss pass already counted the visit.
        """
        v = leaf_value
        for node in reversed(path):
            node.W += v - 1.0
            v = -v

    def _select_child(self, node: _Node) -> tuple[chess.Move, _Node]:
        sqrt_total = math.sqrt(node.N)
        best_score = -float("inf")
        best_move = None
        best_child = None
        for move, child in node.children.items():
            # child.Q is from the child's side-to-move POV; negate for the
            # parent's POV (negamax). Unvisited children get Q=0.
            u = self.c_puct * child.prior * sqrt_total / (1 + child.N)
            score = -child.Q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def _simulate(self, root: _Node, board: chess.Board):
        node = root
        path = [node]
        pushed = 0
        while node.expanded:
            move, child = self._select_child(node)
            board.push(move)
            pushed += 1
            node = child
            path.append(node)

        term = self._terminal_value(board)
        value = term if term is not None else self._expand(node, board)

        # Negamax backup: value is from the leaf's side-to-move POV; flip sign
        # at each step toward the root.
        v = value
        for n in reversed(path):
            n.N += 1
            n.W += v
            v = -v

        for _ in range(pushed):
            board.pop()

    def _run_batch(self, root: _Node, board: chess.Board, batch_n: int):
        """Collect up to ``batch_n`` leaves (virtual loss applied during descent),
        evaluate the unique non-terminal ones in a single batched forward, then
        expand and back them up. This turns ~batch_n synchronous GPU->CPU syncs
        into one."""
        leaves = []          # (path, node, key, moves) awaiting network eval
        items_by_key = {}    # key -> (pos, player, castling, ep, key) for _batch_run

        for _ in range(batch_n):
            node = root
            path = [node]
            pushed = 0
            while node.expanded:
                move, child = self._select_child(node)
                board.push(move)
                pushed += 1
                node = child
                path.append(node)

            # Virtual loss: a virtual *win* in each node's own POV (looks like a
            # loss to its parent under negamax -child.Q selection) so other
            # descents in this batch avoid re-exploring the same path.
            for n in path:
                n.N += 1
                n.W += 1.0

            term = self._terminal_value(board)
            if term is not None:
                self._backup_vl(path, term)
            else:
                key = chess.polyglot.zobrist_hash(board)
                moves = list(board.legal_moves)
                leaves.append((path, node, key, moves))
                if key not in items_by_key:
                    pos, player, castling, ep = self._encode_position(board)
                    items_by_key[key] = (pos, player, castling, ep, key)

            for _ in range(pushed):
                board.pop()

        # One batched forward (+ one sync) populates the TT for every position.
        if items_by_key:
            self._batch_run(list(items_by_key.values()))

        for path, node, key, moves in leaves:
            logits, value = self._tt[key]
            if not node.expanded:
                self._expand_from_logits(node, moves, logits)
            self._backup_vl(path, value)

    def predict(self, board: chess.Board) -> tuple[str, float | None]:
        self.nodes_searched = 0
        self.tt_hits = 0
        self.completed_depth = self.num_simulations

        root = _Node(prior=1.0)
        root_value = self._expand(root, board)
        if not root.children:
            return next(iter(board.legal_moves)).uci(), 0.0
        # Seed the root with one visit so the first PUCT term is well-defined.
        root.N = 1
        root.W = root_value

        if self.sim_batch <= 1:
            for _ in range(self.num_simulations):
                self._simulate(root, board)
        else:
            done = 0
            while done < self.num_simulations:
                n = min(self.sim_batch, self.num_simulations - done)
                self._run_batch(root, board, n)
                done += n

        best_move, best_child = max(root.children.items(), key=lambda kv: kv[1].N)
        # Root value from side-to-move POV (negate child's POV).
        best_value = -best_child.Q
        return best_move.uci(), best_value


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default=str(_DEFAULT_RUN))
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()

    bot = Pos2MoveV2MctsBot(
        model_dir=args.model_dir, num_simulations=args.sims, c_puct=args.c_puct,
        time_limit=0.0, use_ema=args.ema,
    )
    board = chess.Board()
    for _ in range(8):
        t0 = time.time()
        mv, val = bot.predict(board)
        print(f"{'W' if board.turn else 'B'} {mv} val={val:+.3f} "
              f"sims={bot.num_simulations} nodes={bot.nodes_searched} time={time.time()-t0:.2f}s")
        board.push_uci(mv)
        if board.is_game_over():
            break
