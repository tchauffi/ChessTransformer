"""AlphaZero-style action encoding for chess moves.

Encodes moves as (from_square, action_plane) pairs using the 8×8×73
representation from the AlphaZero paper (Silver et al., 2018).

The 73 action planes decompose as:
  0–55:  Queen-type moves — 8 directions × 7 distances
  56–63: Knight moves     — 8 L-shaped jumps
  64–72: Underpromotions  — 3 directions × 3 piece types (knight/bishop/rook)

Queen promotions are encoded as regular queen-type moves (distance-1 to the
last rank).  Underpromotions get their own dedicated planes.
"""

import chess
import torch

NUM_ACTION_PLANES = 73

# Direction vectors (file_delta, rank_delta) for queen-type moves
QUEEN_DIRECTIONS = [
    (0, 1),    # N
    (1, 1),    # NE
    (1, 0),    # E
    (1, -1),   # SE
    (0, -1),   # S
    (-1, -1),  # SW
    (-1, 0),   # W
    (-1, 1),   # NW
]

# Knight jump offsets (file_delta, rank_delta)
KNIGHT_JUMPS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# Underpromotion piece types (queen promotion is a regular queen-type move)
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# Pre-compute knight jump lookup for O(1) detection
_KNIGHT_JUMP_INDEX = {(df, dr): i for i, (df, dr) in enumerate(KNIGHT_JUMPS)}


def move_to_action_plane(from_sq: int, to_sq: int, promotion: int | None = None) -> int:
    """Convert a chess move to its action plane index (0–72).

    Args:
        from_sq: Source square (0–63, python-chess ordering).
        to_sq:   Destination square (0–63).
        promotion: Promotion piece type (chess.QUEEN, chess.KNIGHT, etc.) or None.

    Returns:
        Action plane index in [0, 72].
    """
    df = (to_sq % 8) - (from_sq % 8)
    dr = (to_sq // 8) - (from_sq // 8)

    # ── Underpromotion ───────────────────────────────────────────────────
    if promotion is not None and promotion != chess.QUEEN:
        # Direction: file delta ∈ {-1, 0, +1} → index 0, 1, 2
        dir_idx = df + 1
        piece_idx = UNDERPROMO_PIECES.index(promotion)
        return 64 + dir_idx * 3 + piece_idx

    # ── Knight move ──────────────────────────────────────────────────────
    knight_idx = _KNIGHT_JUMP_INDEX.get((df, dr))
    if knight_idx is not None:
        return 56 + knight_idx

    # ── Queen-type move (includes queen promotions) ──────────────────────
    distance = max(abs(df), abs(dr))
    dir_f = (df // distance) if df != 0 else 0
    dir_r = (dr // distance) if dr != 0 else 0
    dir_idx = QUEEN_DIRECTIONS.index((dir_f, dir_r))
    return dir_idx * 7 + (distance - 1)


def action_to_move(from_sq: int, plane: int) -> tuple[int, int | None]:
    """Decode an action plane back to (to_square, promotion_piece | None).

    For queen-type moves landing on promotion ranks, the caller must check
    whether the source piece is a pawn and add ``chess.QUEEN`` promotion.

    Args:
        from_sq: Source square (0–63).
        plane:   Action plane index (0–72).

    Returns:
        (to_sq, promotion_piece) — promotion_piece is set only for underpromotions.
    """
    from_file = from_sq % 8
    from_rank = from_sq // 8

    if plane < 56:
        # Queen-type move
        dir_idx = plane // 7
        distance = plane % 7 + 1
        df, dr = QUEEN_DIRECTIONS[dir_idx]
        to_file = from_file + df * distance
        to_rank = from_rank + dr * distance
        return to_rank * 8 + to_file, None

    if plane < 64:
        # Knight move
        df, dr = KNIGHT_JUMPS[plane - 56]
        to_file = from_file + df
        to_rank = from_rank + dr
        return to_rank * 8 + to_file, None

    # Underpromotion
    promo_idx = plane - 64
    dir_idx = promo_idx // 3
    piece_idx = promo_idx % 3
    df = dir_idx - 1  # 0→-1, 1→0, 2→+1
    to_file = from_file + df
    to_rank = 7 if from_rank == 6 else 0  # white promotes to rank 8, black to rank 1
    return to_rank * 8 + to_file, UNDERPROMO_PIECES[piece_idx]


def build_legal_moves_planes(board: chess.Board) -> torch.Tensor:
    """Build a (64, 73) boolean mask of legal moves in action-plane format."""
    mask = torch.zeros(64, NUM_ACTION_PLANES, dtype=torch.bool)
    for move in board.legal_moves:
        plane = move_to_action_plane(move.from_square, move.to_square, move.promotion)
        mask[move.from_square, plane] = True
    return mask
