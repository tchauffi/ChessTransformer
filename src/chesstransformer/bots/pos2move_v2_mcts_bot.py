"""AlphaZero-style PUCT MCTS engine for Pos2MoveV2.

Subclasses :class:`Pos2MoveV2Bot` to reuse its (torch.compiled, transposition-
cached) ``_evaluate`` — the policy head supplies move priors and the value head
scores leaves. Move selection is by visit count (the standard AlphaZero choice).

Node representation is **edge-centric**: a node stores its children's visit
counts / values / priors as numpy arrays, and child ``_Node`` objects are created
lazily only when a child is actually descended into. This avoids allocating a
node per legal move at every expansion (which dominated latency) and lets PUCT
selection run vectorized over the arrays. Leaves are collected in batches with
virtual loss so the network sees one batched forward per wave.
"""

from __future__ import annotations

import math

import chess
import numpy as np

from chesstransformer.bots.pos2move_v2_bot import _BATCH_BUCKETS, _DEFAULT_RUN, Pos2MoveV2Bot
from chesstransformer.models.tokenizer.alphazero_move_encoder import move_to_action_plane


class _Node:
    """A search node. Child visit stats live in arrays here (edge-centric); child
    ``_Node`` objects are created lazily on first descent. A node is *expanded*
    once ``moves`` is set."""

    __slots__ = ("moves", "priors", "child_N", "child_W", "children")

    def __init__(self):
        self.moves: list[chess.Move] | None = None
        self.priors: np.ndarray | None = None
        self.child_N: np.ndarray | None = None  # int64, per child
        self.child_W: np.ndarray | None = None  # float64, child-POV total value
        self.children: list[_Node | None] | None = None

    @property
    def expanded(self) -> bool:
        return self.moves is not None


class Pos2MoveV2MctsBot(Pos2MoveV2Bot):
    def __init__(
        self,
        *args,
        num_simulations: int = 800,
        c_puct: float = 1.0,
        prior_temp: float = 1.0,
        fpu: float | None = 0.2,
        sim_batch: int = 16,
        tree_reuse: bool = True,
        move_temp: float = 0.0,
        move_temp_plies: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        # Move selection. By default the move is argmax(visit count) — fully
        # deterministic (what the gauntlet / API rely on). For variety (e.g. so
        # the opening isn't identical every game), set move_temp > 0 to *sample*
        # the move ∝ visit_count**(1/move_temp) for the first move_temp_plies
        # half-moves, then fall back to argmax. move_temp=1 samples proportional
        # to visits; higher is more random.
        self.move_temp = move_temp
        self.move_temp_plies = move_temp_plies
        # Tree reuse: keep the searched subtree under the moves actually played
        # and re-root it next move, so prior visits carry over (effectively
        # deeper search for free). Requires the caller to keep board.move_stack
        # across calls; falls back to a fresh tree otherwise (e.g. boards rebuilt
        # from FEN with no history).
        self.tree_reuse = tree_reuse
        self._reuse_root: _Node | None = None
        self._reuse_moves: list | None = None
        # prior_temp > 1 flattens the policy priors so good-but-low-prior moves
        # still get explored (the value head can then promote them). fpu (first-
        # play-urgency): value unvisited children as parent_Q - fpu; None keeps
        # the legacy behavior (unvisited contribute 0 to selection).
        self.prior_temp = prior_temp
        self.fpu = fpu
        # Leaves collected per network call. >1 amortizes the GPU->CPU sync that
        # otherwise dominates (one stall per simulation). 1 = legacy single-leaf.
        # Capped at the largest batch bucket so _batch_run's preallocated buffers
        # (and the captured CUDA graphs) are never overrun.
        self.sim_batch = max(1, min(sim_batch, _BATCH_BUCKETS[-1]))

    # ── expansion ────────────────────────────────────────────────────────
    def _compute_priors(self, moves: list[chess.Move], logits: np.ndarray) -> np.ndarray:
        """Softmax policy priors over legal moves, with temperature flattening."""
        scores = np.array(
            [
                logits[m.from_square, move_to_action_plane(m.from_square, m.to_square, m.promotion)]
                for m in moves
            ],
            dtype=np.float64,
        )
        scores /= self.prior_temp
        scores -= scores.max()
        exp = np.exp(scores)
        return exp / exp.sum()

    def _expand_node(self, node: _Node, moves: list[chess.Move], priors: np.ndarray):
        self.nodes_searched += 1
        n = len(moves)
        node.moves = moves
        node.priors = priors
        node.child_N = np.zeros(n, dtype=np.int64)
        node.child_W = np.zeros(n, dtype=np.float64)
        node.children = [None] * n

    def _expand_leaf(self, node: _Node, board: chess.Board) -> float:
        """Evaluate ``board`` (policy + value) and expand ``node``; return value."""
        logits, value = self._evaluate(board)
        moves = list(board.legal_moves)
        self._expand_node(node, moves, self._compute_priors(moves, logits))
        return value

    # ── selection / descent / backup ─────────────────────────────────────
    def _select_idx(self, node: _Node) -> int:
        """Vectorized PUCT: pick the child index maximizing -Q + U."""
        N = node.child_N
        total = int(N.sum())
        sqrt_total = math.sqrt(1.0 + total)  # +1 ≈ the node's own eval visit
        # value-to-parent = -child_Q (child stores value from its own POV).
        q_parent = np.zeros_like(node.child_W)
        np.divide(-node.child_W, N, out=q_parent, where=N > 0)
        if self.fpu is not None:
            parent_q = (-node.child_W.sum() / total) if total > 0 else 0.0
            q_parent[N == 0] = parent_q - self.fpu
        u = (self.c_puct * sqrt_total) * node.priors / (1.0 + N)
        return int(np.argmax(q_parent + u))

    def _descend(self, node: _Node, board: chess.Board, apply_vl: bool):
        """Walk to an unexpanded leaf, pushing moves. Returns (leaf_node, path),
        where path is the list of (node, child_idx) edges traversed."""
        path: list[tuple[_Node, int]] = []
        while node.expanded:
            idx = self._select_idx(node)
            if apply_vl:
                # Virtual loss: a virtual *win* in the child's POV looks like a
                # loss to this node under -Q selection, so concurrent descents in
                # the batch avoid the path.
                node.child_N[idx] += 1
                node.child_W[idx] += 1.0
            path.append((node, idx))
            board.push(node.moves[idx])
            child = node.children[idx]
            if child is None:
                child = _Node()
                node.children[idx] = child
            node = child
        return node, path

    @staticmethod
    def _backup(path: list[tuple[_Node, int]], value: float):
        """Negamax backup over edges (no virtual loss)."""
        v = value
        for node, idx in reversed(path):
            node.child_N[idx] += 1
            node.child_W[idx] += v
            v = -v

    @staticmethod
    def _backup_vl(path: list[tuple[_Node, int]], value: float):
        """Backup that undoes the per-edge virtual loss (+1 added during descent)
        and applies the real, sign-alternating value. N already counted."""
        v = value
        for node, idx in reversed(path):
            node.child_W[idx] += v - 1.0
            v = -v

    # ── search drivers ───────────────────────────────────────────────────
    def _simulate(self, root: _Node, board: chess.Board):
        node, path = self._descend(root, board, apply_vl=False)
        pushed = len(path)
        term = self._terminal_value(board)
        value = term if term is not None else self._expand_leaf(node, board)
        self._backup(path, value)
        for _ in range(pushed):
            board.pop()

    def _run_batch(self, root: _Node, board: chess.Board, batch_n: int):
        """Collect up to ``batch_n`` leaves (virtual loss during descent),
        evaluate the unique uncached non-terminal ones in one batched forward,
        then expand and back them up."""
        leaves = []          # (path, leaf_node, key, moves)
        items_by_key = {}    # key -> (pos, player, castling, ep, key)

        for _ in range(batch_n):
            node, path = self._descend(root, board, apply_vl=True)
            term = self._terminal_value(board)
            if term is not None:
                self._backup_vl(path, term)
            else:
                key = chess.polyglot.zobrist_hash(board)
                moves = list(board.legal_moves)
                leaves.append((path, node, key, moves))
                # Reuse cached policy+value (transpositions / reused subtree /
                # prior moves); only batch positions we haven't evaluated yet.
                if key in self._tt:
                    self.tt_hits += 1
                elif key not in items_by_key:
                    pos, player, castling, ep = self._encode_position(board)
                    items_by_key[key] = (pos, player, castling, ep, key)
            for _ in range(len(path)):
                board.pop()

        if items_by_key:
            self._batch_run(list(items_by_key.values()))  # populates self._tt

        for path, node, key, moves in leaves:
            logits, value = self._tt[key]
            if not node.expanded:
                self._expand_node(node, moves, self._compute_priors(moves, logits))
            self._backup_vl(path, value)

    # ── tree reuse ───────────────────────────────────────────────────────
    def _take_reuse_root(self, board: chess.Board) -> _Node | None:
        """Re-root the retained tree to ``board`` if it follows the previous
        search position by a few plies. Consumes the stored tree."""
        root = self._reuse_root
        self._reuse_root = None
        if root is None or self._reuse_moves is None:
            return None
        ms = board.move_stack
        rp = len(self._reuse_moves)
        gap = len(ms) - rp
        if gap < 1 or gap > 4 or list(ms[:rp]) != self._reuse_moves:
            return None
        node = root
        for mv in ms[rp:]:
            if not node.expanded:
                return None
            try:
                idx = node.moves.index(mv)
            except ValueError:
                return None
            child = node.children[idx]
            if child is None or not child.expanded:
                return None
            node = child
        return node if node.expanded else None

    def predict(self, board: chess.Board) -> tuple[str, float | None]:
        self.nodes_searched = 0
        self.tt_hits = 0
        self.completed_depth = self.num_simulations

        if not any(board.legal_moves):
            return chess.Move.null().uci(), 0.0

        root = self._take_reuse_root(board) if self.tree_reuse else None
        if root is None:
            root = _Node()
            self._expand_leaf(root, board)

        if self.sim_batch <= 1:
            for _ in range(self.num_simulations):
                self._simulate(root, board)
        else:
            done = 0
            while done < self.num_simulations:
                n = min(self.sim_batch, self.num_simulations - done)
                self._run_batch(root, board, n)
                done += n

        ply = 2 * (board.fullmove_number - 1) + (0 if board.turn == chess.WHITE else 1)
        if self.move_temp > 0 and ply < self.move_temp_plies and root.child_N.sum() > 0:
            # Sample ∝ visit_count**(1/temp) over the opening plies for variety.
            weights = root.child_N.astype(np.float64) ** (1.0 / self.move_temp)
            total = weights.sum()
            best_idx = int(np.random.choice(len(weights), p=weights / total)) if total > 0 \
                else int(np.argmax(root.child_N))
        else:
            best_idx = int(np.argmax(root.child_N))
        best_move = root.moves[best_idx]
        n = int(root.child_N[best_idx])
        # Root value from side-to-move POV (negate child's POV).
        best_value = float(-root.child_W[best_idx] / n) if n > 0 else 0.0

        if self.tree_reuse:
            self._reuse_root = root
            self._reuse_moves = list(board.move_stack)

        return best_move.uci(), best_value


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default=str(_DEFAULT_RUN))
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--c-puct", type=float, default=1.0)
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
