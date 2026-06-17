//! Position/move encodings — must match the Python tokenizers exactly
//! (`PostionTokenizer`, `alphazero_move_encoder.move_to_action_plane`).

use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{CastlingSide, Chess, Color, EnPassantMode, Move, Position, Role};

pub const NUM_PLANES: usize = 73;

pub fn piece_token(role: Role, color: Color) -> u8 {
    let base = match role {
        Role::Pawn => 1,
        Role::Knight => 2,
        Role::Bishop => 3,
        Role::Rook => 4,
        Role::Queen => 5,
        Role::King => 6,
    };
    if color == Color::White {
        base
    } else {
        base + 6
    }
}

/// (from_sq, to_sq, promotion) in python-chess conventions: castling is the
/// king's two-square move (e1g1), squares indexed a1=0..h8=63.
pub fn move_coords(m: &Move) -> (usize, usize, Option<Role>) {
    match m {
        Move::Castle { king, rook } => {
            let k = u32::from(*king) as usize;
            let r = u32::from(*rook) as usize;
            let to = if r > k { k + 2 } else { k - 2 };
            (k, to, None)
        }
        _ => (
            u32::from(m.from().expect("no drops in chess")) as usize,
            u32::from(m.to()) as usize,
            m.promotion(),
        ),
    }
}

/// AlphaZero 8x8x73 action plane (mirror of move_to_action_plane).
pub fn action_plane(from: usize, to: usize, promo: Option<Role>) -> usize {
    let df = (to % 8) as i32 - (from % 8) as i32;
    let dr = (to / 8) as i32 - (from / 8) as i32;

    if let Some(p) = promo {
        if p != Role::Queen {
            let dir_idx = (df + 1) as usize;
            let piece_idx = match p {
                Role::Knight => 0,
                Role::Bishop => 1,
                Role::Rook => 2,
                _ => unreachable!("queen handled above"),
            };
            return 64 + dir_idx * 3 + piece_idx;
        }
    }

    const KNIGHT_JUMPS: [(i32, i32); 8] = [
        (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
    ];
    if let Some(i) = KNIGHT_JUMPS.iter().position(|&d| d == (df, dr)) {
        return 56 + i;
    }

    const QUEEN_DIRS: [(i32, i32); 8] = [
        (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
    ];
    let dist = df.abs().max(dr.abs());
    let dir_f = if df != 0 { df / dist } else { 0 };
    let dir_r = if dr != 0 { dr / dist } else { 0 };
    let dir_idx = QUEEN_DIRS.iter().position(|&d| d == (dir_f, dir_r)).unwrap();
    dir_idx * 7 + (dist as usize - 1)
}

pub fn encode_position(pos: &Chess) -> ([u8; 64], u8, u8, u8) {
    let mut tokens = [0u8; 64];
    let board = pos.board();
    for sq in board.occupied() {
        let piece = board.piece_at(sq).unwrap();
        tokens[u32::from(sq) as usize] = piece_token(piece.role, piece.color);
    }
    let player = if pos.turn() == Color::White { 1 } else { 0 };
    let castles = pos.castles();
    let mut castling = 0u8;
    if castles.has(Color::White, CastlingSide::KingSide) {
        castling |= 1;
    }
    if castles.has(Color::White, CastlingSide::QueenSide) {
        castling |= 2;
    }
    if castles.has(Color::Black, CastlingSide::KingSide) {
        castling |= 4;
    }
    if castles.has(Color::Black, CastlingSide::QueenSide) {
        castling |= 8;
    }
    let ep = pos
        .ep_square(EnPassantMode::Legal)
        .map(|s| (u32::from(s) % 8) as u8)
        .unwrap_or(8);
    (tokens, player, castling, ep)
}

pub fn zobrist(pos: &Chess) -> u64 {
    let h: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
    h.0
}

/// Terminal value from the side-to-move POV, or None. `keys` is the sequence
/// of zobrist keys of all positions seen (game history + search path),
/// including the current position — mirrors python-chess is_repetition(3)
/// seeing the search pushes on the shared board.
pub fn terminal_value(pos: &Chess, keys: &[u64], current: u64) -> Option<f32> {
    if pos.legal_moves().is_empty() {
        return Some(if pos.is_check() { -1.0 } else { 0.0 });
    }
    if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
        return Some(0.0);
    }
    if keys.iter().filter(|&&k| k == current).count() >= 3 {
        return Some(0.0);
    }
    None
}

/// Half-move index of the position (0 = white's first move).
pub fn halfmove(pos: &Chess) -> u32 {
    2 * (u32::from(pos.fullmoves()) - 1) + if pos.turn() == Color::White { 0 } else { 1 }
}
