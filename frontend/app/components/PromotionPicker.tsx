'use client';

import type { Color, PieceSymbol } from 'chess.js';
import { defaultPieces } from 'react-chessboard';
import { IconX } from './icons';

const CHOICES: PieceSymbol[] = ['q', 'n', 'r', 'b'];
const NAMES: Partial<Record<PieceSymbol, string>> = { q: 'Queen', n: 'Knight', r: 'Rook', b: 'Bishop' };

interface Props {
  square: string;
  color: Color;
  orientation: 'white' | 'black';
  onPick: (piece: PieceSymbol) => void;
  onCancel: () => void;
}

/** chess.com-style column of promotion choices, dropping from the promotion square toward the centre. */
export default function PromotionPicker({ square, color, orientation, onPick, onCancel }: Props) {
  const file = 'abcdefgh'.indexOf(square[0]);
  const rank = Number(square[1]);
  const col = orientation === 'white' ? file : 7 - file;
  const row = orientation === 'white' ? 8 - rank : rank - 1;
  const fromTop = row === 0;

  return (
    <div className="absolute inset-0 z-30" onClick={onCancel} onContextMenu={(e) => e.preventDefault()}>
      <div className="absolute inset-0 bg-black/35" />
      <div
        role="dialog"
        aria-label="Choose a promotion piece"
        className={`absolute flex w-[12.5%] overflow-hidden rounded-md bg-[#f4f1ea] shadow-[0_8px_28px_rgba(0,0,0,0.45)] ${
          fromTop ? 'top-0 flex-col' : 'bottom-0 flex-col-reverse'
        }`}
        style={{ left: `${col * 12.5}%` }}
        onClick={(e) => e.stopPropagation()}
      >
        {CHOICES.map((p) => {
          const Piece = defaultPieces[`${color}${p.toUpperCase()}`];
          return (
            <button
              key={p}
              type="button"
              autoFocus={p === 'q'}
              onClick={() => onPick(p)}
              className="aspect-square w-full p-[6%] transition-colors hover:bg-[#e3dcc8] focus-visible:bg-[#e3dcc8]"
              aria-label={NAMES[p]}
            >
              <Piece />
            </button>
          );
        })}
        <button
          type="button"
          onClick={onCancel}
          className="grid h-8 place-items-center bg-[#e3dcc8] text-[#5b5547] hover:text-black"
          aria-label="Cancel promotion"
        >
          <IconX width={16} height={16} />
        </button>
      </div>
    </div>
  );
}
