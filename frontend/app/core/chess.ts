import type { Chess, Color, PieceSymbol, Square } from 'chess.js';
import type { PositionDataType } from 'react-chessboard';

export const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const FILES = 'abcdefgh';

export type Phase = 'setup' | 'playing' | 'over';

/** One half-move as played, with the position it produced. */
export interface Ply {
  san: string;
  from: Square;
  to: Square;
  color: Color;
  piece: PieceSymbol;
  captured?: PieceSymbol;
  promotion?: PieceSymbol;
  castle: boolean;
  check: boolean;
  fen: string;
}

export interface Premove {
  from: Square;
  to: Square;
  promotion?: PieceSymbol;
}

export type ResultReason =
  | 'checkmate'
  | 'resignation'
  | 'timeout'
  | 'timeout vs insufficient material'
  | 'stalemate'
  | 'insufficient material'
  | 'repetition'
  | '50-move rule'
  | 'agreement'
  | 'abort';

export interface GameResult {
  winner: Color | null;
  reason: ResultReason;
}

/** Seconds of base time and increment; `null` means an untimed game. */
export type TimeControl = { base: number; inc: number } | null;

export const TIME_CONTROLS: { id: string; label: string; value: TimeControl }[] = [
  { id: 'none', label: 'No clock', value: null },
  { id: '3+2', label: '3 | 2', value: { base: 180, inc: 2 } },
  { id: '5+0', label: '5 min', value: { base: 300, inc: 0 } },
  { id: '10+0', label: '10 min', value: { base: 600, inc: 0 } },
  { id: '15+10', label: '15 | 10', value: { base: 900, inc: 10 } },
  { id: '30+0', label: '30 min', value: { base: 1800, inc: 0 } },
];

export const timeControlById = (id: string): TimeControl =>
  TIME_CONTROLS.find((t) => t.id === id)?.value ?? null;

export const other = (c: Color): Color => (c === 'w' ? 'b' : 'w');

const coords = (sq: string) => ({ f: FILES.indexOf(sq[0]), r: Number(sq[1]) - 1 });
const square = (f: number, r: number) => `${FILES[f]}${r + 1}` as Square;
const onBoard = (f: number, r: number) => f >= 0 && f < 8 && r >= 0 && r < 8;

/** Piece placement of a FEN, in react-chessboard's position format. */
export function placement(fen: string): PositionDataType {
  const pos: PositionDataType = {};
  fen
    .split(' ')[0]
    .split('/')
    .forEach((row, i) => {
      let f = 0;
      for (const ch of row) {
        if (ch >= '1' && ch <= '8') {
          f += Number(ch);
        } else {
          const color = ch === ch.toUpperCase() ? 'w' : 'b';
          pos[`${FILES[f]}${8 - i}`] = { pieceType: `${color}${ch.toUpperCase()}` };
          f++;
        }
      }
    });
  return pos;
}

/** Automatic game endings that chess.js can see from the move history. */
export function detectEnd(g: Chess): GameResult | null {
  if (g.isCheckmate()) return { winner: other(g.turn()), reason: 'checkmate' };
  if (g.isStalemate()) return { winner: null, reason: 'stalemate' };
  if (g.isInsufficientMaterial()) return { winner: null, reason: 'insufficient material' };
  if (g.isThreefoldRepetition()) return { winner: null, reason: 'repetition' };
  if (g.isDrawByFiftyMoves()) return { winner: null, reason: '50-move rule' };
  return null;
}

/** Whether `color` still has enough material to ever deliver mate (decides timeout vs draw). */
export function canMate(pos: PositionDataType, color: Color): boolean {
  let minors = 0;
  for (const { pieceType } of Object.values(pos)) {
    if (pieceType[0] !== color) continue;
    const t = pieceType[1];
    if (t === 'P' || t === 'R' || t === 'Q') return true;
    if (t === 'N' || t === 'B') minors++;
  }
  return minors >= 2;
}

export const RESULT_SCORE = (r: GameResult | null): string =>
  !r || r.reason === 'abort' ? '*' : r.winner === 'w' ? '1-0' : r.winner === 'b' ? '0-1' : '1/2-1/2';

const REASON_TEXT: Record<ResultReason, string> = {
  checkmate: 'by checkmate',
  resignation: 'by resignation',
  timeout: 'on time',
  'timeout vs insufficient material': 'timeout vs insufficient material',
  stalemate: 'by stalemate',
  'insufficient material': 'by insufficient material',
  repetition: 'by repetition',
  '50-move rule': 'by the 50-move rule',
  agreement: 'by agreement',
  abort: 'before it began',
};

export const reasonText = (r: GameResult) => REASON_TEXT[r.reason];

// ---------------------------------------------------------------- material

const VALUE: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
const START_COUNT: Record<string, number> = { p: 8, n: 2, b: 2, r: 2, q: 1 };
const CAPTURE_ORDER: PieceSymbol[] = ['p', 'n', 'b', 'r', 'q'];

/** Pieces each side has taken (chess.com style: missing from the start set) and the material balance. */
export function material(pos: PositionDataType) {
  const count = { w: { p: 0, n: 0, b: 0, r: 0, q: 0 }, b: { p: 0, n: 0, b: 0, r: 0, q: 0 } } as Record<
    Color,
    Record<string, number>
  >;
  let balance = 0;
  for (const { pieceType } of Object.values(pos)) {
    const color = pieceType[0] as Color;
    const t = pieceType[1].toLowerCase();
    if (t === 'k') continue;
    count[color][t]++;
    balance += (color === 'w' ? 1 : -1) * VALUE[t];
  }
  const taken = (victim: Color) =>
    CAPTURE_ORDER.flatMap((t) => Array(Math.max(0, START_COUNT[t] - count[victim][t])).fill(t) as PieceSymbol[]);
  return { takenBy: { w: taken('b'), b: taken('w') }, balance };
}

// ---------------------------------------------------------------- premoves

const KNIGHT = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
const KING = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
const ROOK_DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const BISHOP_DIRS = [[1, 1], [1, -1], [-1, 1], [-1, -1]];

/**
 * Squares a piece could reach on some future turn, by geometry alone — the
 * occupancy will have changed by the time a premove fires, so blockers are ignored.
 */
export function premoveDests(pos: PositionDataType, from: string, color: Color): Square[] {
  const piece = pos[from]?.pieceType;
  if (!piece || piece[0] !== color) return [];
  const { f, r } = coords(from);
  const out: Square[] = [];
  const step = (df: number, dr: number) => onBoard(f + df, r + dr) && out.push(square(f + df, r + dr));
  const slide = (dirs: number[][]) =>
    dirs.forEach(([df, dr]) => {
      for (let i = 1; onBoard(f + df * i, r + dr * i); i++) out.push(square(f + df * i, r + dr * i));
    });
  switch (piece[1]) {
    case 'P': {
      const dir = color === 'w' ? 1 : -1;
      step(0, dir);
      if (r === (color === 'w' ? 1 : 6)) step(0, 2 * dir);
      step(-1, dir);
      step(1, dir);
      break;
    }
    case 'N':
      KNIGHT.forEach(([df, dr]) => step(df, dr));
      break;
    case 'B':
      slide(BISHOP_DIRS);
      break;
    case 'R':
      slide(ROOK_DIRS);
      break;
    case 'Q':
      slide([...ROOK_DIRS, ...BISHOP_DIRS]);
      break;
    case 'K': {
      KING.forEach(([df, dr]) => step(df, dr));
      const home = color === 'w' ? 0 : 7;
      const rook = `${color}R`;
      if (f === 4 && r === home) {
        if (pos[square(7, home)]?.pieceType === rook) out.push(square(6, home));
        if (pos[square(0, home)]?.pieceType === rook) out.push(square(2, home));
      }
      break;
    }
  }
  return out;
}

export const isPromotionSquare = (pos: PositionDataType, from: string, to: string) => {
  const piece = pos[from]?.pieceType;
  return piece?.[1] === 'P' && (to[1] === '8' || to[1] === '1');
};

/** Board placement after a premove, so queued premoves show where pieces will be. */
export function applyPremove(pos: PositionDataType, pm: Premove): PositionDataType {
  const piece = pos[pm.from];
  if (!piece) return pos;
  const next = { ...pos };
  delete next[pm.from];
  const color = piece.pieceType[0];
  const { f: ff, r } = coords(pm.from);
  const { f: tf } = coords(pm.to);
  if (piece.pieceType[1] === 'K' && Math.abs(tf - ff) === 2) {
    const [rookFrom, rookTo] = tf > ff ? [square(7, r), square(5, r)] : [square(0, r), square(3, r)];
    if (next[rookFrom]) {
      next[rookTo] = next[rookFrom];
      delete next[rookFrom];
    }
  }
  next[pm.to] = isPromotionSquare(pos, pm.from, pm.to)
    ? { pieceType: `${color}${(pm.promotion ?? 'q').toUpperCase()}` }
    : piece;
  return next;
}

/** Board placement with one piece lifted to another square (the pawn waiting on the promotion picker). */
export function movePiece(pos: PositionDataType, from: string, to: string): PositionDataType {
  const next = { ...pos };
  next[to] = pos[from];
  delete next[from];
  return next;
}

export function kingSquare(pos: PositionDataType, color: Color): string | undefined {
  return Object.keys(pos).find((sq) => pos[sq].pieceType === `${color}K`);
}

// ---------------------------------------------------------------- evaluation

/**
 * The value head outputs an expected score in (-1, 1). Shown as pawns with the
 * Leela convention (cp = 90·tan(1.5637·q)), which reads like a familiar engine eval.
 */
export function evalToPawns(v: number): number {
  const q = Math.max(-0.995, Math.min(0.995, v));
  return 0.9 * Math.tan(1.5637541897 * q);
}

export function formatEval(v: number): string {
  const p = evalToPawns(v);
  const abs = Math.abs(p);
  const txt = abs >= 10 ? abs.toFixed(0) : abs.toFixed(1);
  return txt === '0.0' ? '0.0' : `${p > 0 ? '+' : '−'}${txt}`;
}

// ---------------------------------------------------------------- clocks

export function formatClock(ms: number): string {
  const t = Math.max(0, ms);
  if (t < 20_000) {
    const tenths = Math.floor(t / 100);
    return `0:${String(Math.floor(tenths / 10)).padStart(2, '0')}.${tenths % 10}`;
  }
  const s = Math.ceil(t / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`;
}
