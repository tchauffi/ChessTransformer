// The board's look, shared by the game (ChessGame) and the games viewer (GameViewer).

import { defaultArrowOptions, type ChessboardOptions } from 'react-chessboard';

// Board palette: sand and deep teal, with brass for move highlights and coral for
// premoves and annotations.
export const BOARD = { light: '#e9e1cc', dark: '#5d8a84' };
export const TINT = {
  last: 'rgba(247, 206, 84, 0.5)',
  selected: 'rgba(247, 206, 84, 0.75)',
  premove: 'rgba(214, 86, 74, 0.6)',
  mark: 'rgba(226, 92, 68, 0.85)',
};
export const DOT = 'radial-gradient(circle, rgba(18, 38, 33, 0.22) 21%, transparent 22.5%)';
export const RING = 'radial-gradient(circle, transparent 71%, rgba(18, 38, 33, 0.24) 72.5%)';
export const CHECK =
  'radial-gradient(circle, rgba(236, 62, 46, 0.95) 0%, rgba(236, 62, 46, 0.6) 34%, rgba(236, 62, 46, 0) 72%)';
const NOTATION_SIZE = 'clamp(9px, 1.55vmin, 13px)';

/** Squares, coordinates, frame, animation and arrow colours of every board. */
export const BOARD_LOOK: ChessboardOptions = {
  darkSquareStyle: { backgroundColor: BOARD.dark },
  lightSquareStyle: { backgroundColor: BOARD.light },
  darkSquareNotationStyle: { color: BOARD.light, fontWeight: 600 },
  lightSquareNotationStyle: { color: BOARD.dark, fontWeight: 600 },
  alphaNotationStyle: { fontSize: NOTATION_SIZE, position: 'absolute', bottom: 1, right: 3, userSelect: 'none' },
  numericNotationStyle: { fontSize: NOTATION_SIZE, position: 'absolute', top: 2, left: 3, userSelect: 'none' },
  boardStyle: { borderRadius: 4, boxShadow: '0 6px 24px rgba(0, 0, 0, 0.35)' },
  animationDurationInMs: 170,
  arrowOptions: {
    ...defaultArrowOptions,
    color: '#f0a830',
    secondaryColor: '#3fbf8f',
    tertiaryColor: '#e0574b',
    opacity: 0.82,
    activeOpacity: 0.6,
  },
};
