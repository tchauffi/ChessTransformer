"""Bot for Pos2MoveV2 with alpha-beta search + BF16."""

from __future__ import annotations

import json
import time as _time
from pathlib import Path

import chess
import numpy as np
import torch
from safetensors import safe_open

from chesstransformer.models.tokenizer import PostionTokenizer
from chesstransformer.models.tokenizer.alphazero_move_encoder import move_to_action_plane
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2

_DEFAULT_RUN = Path(__file__).parents[3] / "data" / "models" / "pos2move_v2"


class _SearchTimeout(Exception):
    pass


class Pos2MoveV2Bot:
    def __init__(
        self,
        model_dir: str | Path = _DEFAULT_RUN,
        depth: int = 3,
        top_p: float = 0.90,
        time_limit: float = 30.0,
        tt_size: int = 500_000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_ema: bool = False,
    ):
        self.depth = depth
        self.top_p = top_p
        self.time_limit = time_limit
        self.nodes_searched = 0
        self.tt_hits = 0
        self.completed_depth = 0
        self.device = device
        self.position_tokenizer = PostionTokenizer()

        model_dir = Path(model_dir)

        config_path = model_dir / "model_config.json"
        if not config_path.exists():
            config_path = model_dir.parent.parent / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No model_config.json found for {model_dir}")

        with open(config_path) as f:
            config = json.load(f)

        model = Pos2MoveV2(**config)

        safetensors_path = model_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"No model.safetensors in {model_dir}")
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}

        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        if use_ema:
            ema_path = model_dir / "ema_state.pt"
            if not ema_path.exists():
                raise FileNotFoundError(f"No ema_state.pt in {model_dir}")
            ema_state = torch.load(str(ema_path), map_location="cpu", weights_only=True)
            ema_state = {k.replace("_orig_mod.", ""): v for k, v in ema_state.items()}
            missing, unexpected = model.load_state_dict(ema_state, strict=False)
            if unexpected:
                print(f"WARNING: EMA contains unexpected keys: {unexpected}")
            print(f"EMA weights loaded ({len(ema_state)} tensors; {len(missing)} buffers kept).")

        model.eval()
        self.model = model.bfloat16().to(device)

        self._buf_board = torch.zeros(1, 64, dtype=torch.long, device=device)
        self._buf_player = torch.zeros(1, dtype=torch.long, device=device)
        self._buf_castling = torch.zeros(1, dtype=torch.long, device=device)
        self._buf_ep = torch.zeros(1, dtype=torch.long, device=device)

        self._tt_size = tt_size
        self._tt: dict[str, tuple[np.ndarray, float]] = {}
        self._deadline: float = 0.0

        self._warmup()

        print(
            f"Pos2MoveV2Bot loaded (device={device}, depth={depth}, "
            f"top_p={top_p:.0%}, params={sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M)"
        )

    @torch.no_grad()
    def _warmup(self):
        for _ in range(3):
            self.model(self._buf_board, self._buf_player, self._buf_castling, self._buf_ep)
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _adaptive_time_limit(self, board: chess.Board) -> float:
        if self.time_limit <= 0:
            return 0.0
        ply = len(board.move_stack)
        move_number = ply // 2 + 1
        if move_number <= 10:
            scale = 0.10
        elif move_number <= 20:
            scale = 0.40
        elif move_number <= 40:
            scale = 1.0
        else:
            scale = 0.60
        return self.time_limit * scale

    def predict(self, board: chess.Board) -> tuple[str, float | None]:
        self.nodes_searched = 0
        self.tt_hits = 0
        self.completed_depth = 0

        effective_limit = self._adaptive_time_limit(board)
        self._deadline = _time.monotonic() + effective_limit if effective_limit > 0 else float("inf")

        best_move = None
        best_value = 0.0

        for d in range(1, self.depth + 1):
            stack_len = len(board.move_stack)
            try:
                move, value = self._search_root(board, d)
                best_move = move
                best_value = value
                self.completed_depth = d
            except _SearchTimeout:
                while len(board.move_stack) > stack_len:
                    board.pop()
                break

        if best_move is None:
            best_move = next(iter(board.legal_moves))

        return best_move.uci(), best_value

    def get_value(self, board: chess.Board) -> float:
        _, value = self._evaluate(board)
        return value

    def _check_time(self):
        if _time.monotonic() >= self._deadline:
            raise _SearchTimeout()

    @torch.no_grad()
    def _evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        fen = board.board_fen() + (" w" if board.turn else " b")
        cached = self._tt.get(fen)
        if cached is not None:
            self.tt_hits += 1
            return cached

        position = self.position_tokenizer.encode(board)
        self._buf_board[0] = torch.tensor(position, dtype=torch.long, device=self.device)
        self._buf_player[0] = int(board.turn)

        castling = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            castling |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            castling |= 8
        self._buf_castling[0] = castling

        ep_file = chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8
        self._buf_ep[0] = ep_file

        move_logits, value_t = self.model(
            self._buf_board, self._buf_player, self._buf_castling, self._buf_ep
        )

        logits = move_logits[0].float().cpu().numpy()
        value = float(value_t.item())

        if self._tt_size > 0 and len(self._tt) >= self._tt_size:
            del self._tt[next(iter(self._tt))]
        self._tt[fen] = (logits, value)

        return logits, value

    @torch.no_grad()
    def _batch_run(self, items: list[tuple[np.ndarray, int, int, int, str]]):
        n = len(items)
        board_arr = torch.zeros(n, 64, dtype=torch.long, device=self.device)
        player_arr = torch.zeros(n, dtype=torch.long, device=self.device)
        castling_arr = torch.zeros(n, dtype=torch.long, device=self.device)
        ep_arr = torch.zeros(n, dtype=torch.long, device=self.device)

        for i, (pos, player, castling, ep, _) in enumerate(items):
            board_arr[i] = torch.tensor(pos, dtype=torch.long)
            player_arr[i] = player
            castling_arr[i] = castling
            ep_arr[i] = ep

        move_logits, value_t = self.model(board_arr, player_arr, castling_arr, ep_arr)
        logits_np = move_logits.float().cpu().numpy()
        values_np = value_t.float().cpu().numpy()

        for i, (_, _, _, _, fen) in enumerate(items):
            if self._tt_size > 0 and len(self._tt) >= self._tt_size:
                del self._tt[next(iter(self._tt))]
            self._tt[fen] = (logits_np[i], float(values_np[i]))

    def _encode_position(self, board: chess.Board) -> tuple[np.ndarray, int, int, int]:
        position = self.position_tokenizer.encode(board)
        player = int(board.turn)
        castling = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            castling |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            castling |= 8
        ep_file = chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8
        return position, player, castling, ep_file

    def _prefetch_children(self, board: chess.Board, moves: list[chess.Move]):
        uncached = []
        for move in moves:
            board.push(move)
            fen = board.board_fen() + (" w" if board.turn else " b")
            if fen not in self._tt and not board.is_game_over(claim_draw=True):
                pos, player, castling, ep = self._encode_position(board)
                uncached.append((pos, player, castling, ep, fen))
            board.pop()
        if len(uncached) >= 2:
            self._batch_run(uncached)

    def _top_p_for_depth(self, depth: int, max_depth: int) -> float:
        plies_from_root = max_depth - depth
        if plies_from_root == 0:
            return self.top_p
        elif plies_from_root <= 2:
            return self.top_p * 0.7
        else:
            return self.top_p * 0.5

    def _search_root(self, board: chess.Board, depth: int) -> tuple[chess.Move, float]:
        ordered_moves = self._order_moves(board, self.top_p)
        if not ordered_moves:
            return chess.Move.null(), 0.0
        self._prefetch_children(board, [m for m, _ in ordered_moves])
        best_move = ordered_moves[0][0]
        alpha = float("-inf")
        for move, _ in ordered_moves:
            self._check_time()
            board.push(move)
            value = -self._alpha_beta(board, depth - 1, -float("inf"), -alpha, depth)
            board.pop()
            if value > alpha:
                alpha = value
                best_move = move
        return best_move, alpha

    def _alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float, max_depth: int) -> float:
        self.nodes_searched += 1
        self._check_time()

        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            return 0.0 if result == "1/2-1/2" else -1.0

        if depth <= 0:
            _, value = self._evaluate(board)
            return value

        p = self._top_p_for_depth(depth, max_depth)
        ordered_moves = self._order_moves(board, p)
        self._prefetch_children(board, [m for m, _ in ordered_moves])

        for move, _ in ordered_moves:
            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, max_depth)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _order_moves(self, board: chess.Board, p: float, min_moves: int = 2, max_moves: int = 30):
        logits, _ = self._evaluate(board)
        moves = []
        raw_scores = []
        for move in board.legal_moves:
            plane = move_to_action_plane(move.from_square, move.to_square, move.promotion)
            raw_scores.append(float(logits[move.from_square, plane]))
            moves.append(move)
        if not moves:
            return []

        scores = np.array(raw_scores, dtype=np.float64)
        scores -= scores.max()
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()

        order = np.argsort(-probs)
        cumsum = 0.0
        selected = []
        for idx in order:
            selected.append((moves[idx], float(probs[idx])))
            cumsum += probs[idx]
            if cumsum >= p and len(selected) >= min_moves:
                break
            if len(selected) >= max_moves:
                break

        while len(selected) < min_moves and len(selected) < len(moves):
            selected.append((moves[order[len(selected)]], float(probs[order[len(selected)]])))

        return selected


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default=str(_DEFAULT_RUN))
    parser.add_argument("depth", nargs="?", type=int, default=3)
    parser.add_argument("top_p", nargs="?", type=float, default=0.90)
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()

    bot = Pos2MoveV2Bot(
        model_dir=args.model_dir, depth=args.depth, top_p=args.top_p,
        time_limit=30.0, use_ema=args.ema,
    )

    board = chess.Board()
    print(f"\nStarting position:\n{board}\n")

    for i in range(10):
        t0 = time.time()
        move_uci, value = bot.predict(board)
        elapsed = time.time() - t0
        print(
            f"{'White' if board.turn else 'Black'} plays {move_uci}  "
            f"(depth={bot.completed_depth}/{args.depth}, value={value:+.3f}, "
            f"nodes={bot.nodes_searched}, tt_hits={bot.tt_hits}, time={elapsed:.2f}s)"
        )
        board.push_uci(move_uci)
        print(board)
        print()

        if board.is_game_over():
            print(f"Game over: {board.result()}")
            break
