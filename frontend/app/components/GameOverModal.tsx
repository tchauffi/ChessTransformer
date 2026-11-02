'use client';

import type { Color } from 'chess.js';
import { BotAvatar, HumanAvatar } from './Avatars';
import { IconDownload, IconX } from './icons';
import { RESULT_SCORE, reasonText, type GameResult } from '../core/chess';

interface Props {
  result: GameResult;
  playerColor: Color;
  /** The engine played both sides: name the winner by color. */
  selfPlay: boolean;
  onRematch: () => void;
  onNewGame: () => void;
  onDownload: () => void;
  onClose: () => void;
}

export default function GameOverModal({ result, playerColor, selfPlay, onRematch, onNewGame, onDownload, onClose }: Props) {
  const aborted = result.reason === 'abort';
  const title = aborted
    ? 'Game aborted'
    : result.winner === null
      ? 'Draw'
      : selfPlay
        ? `${result.winner === 'w' ? 'White' : 'Black'} won`
        : result.winner === playerColor
          ? 'You won'
          : 'ChessTransformer won';
  const score = RESULT_SCORE(result).replace('1/2', '½').replace('1/2', '½');
  const [left, right] = playerColor === 'w' ? score.split('-') : score.split('-').reverse();

  return (
    <div className="modal-in absolute inset-0 z-40 grid place-items-center bg-black/30 p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="result-title"
        className="relative w-full max-w-[21rem] rounded-xl bg-panel p-6 pt-7 text-center shadow-[0_20px_60px_rgba(0,0,0,0.55)]"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={onClose}
          className="absolute right-2.5 top-2.5 rounded-md p-1.5 text-muted hover:bg-raised hover:text-fg"
          aria-label="Close and review the game"
        >
          <IconX width={18} height={18} />
        </button>
        <h2 id="result-title" className="font-display text-[24px] font-bold leading-tight text-fg">
          {title}
        </h2>
        <p className="mt-1 text-[15px] text-muted">{reasonText(result)}</p>

        {!aborted ? (
          <div className="mt-5 flex items-center justify-center gap-4">
            {selfPlay ? <BotAvatar size={56} /> : <HumanAvatar size={56} color={playerColor} />}
            <span className="font-display text-2xl font-bold tabular-nums text-fg">
              {left}
              <span className="px-1.5 text-faint">–</span>
              {right}
            </span>
            <BotAvatar size={56} />
          </div>
        ) : null}

        <div className="mt-6 grid gap-2">
          <button type="button" onClick={onRematch} autoFocus className="btn-primary h-12 text-[17px]">
            Rematch
          </button>
          <button type="button" onClick={onNewGame} className="btn-secondary h-11">
            New game
          </button>
        </div>
        {!aborted ? (
          <button
            type="button"
            onClick={onDownload}
            className="mx-auto mt-4 inline-flex items-center gap-1.5 text-sm text-muted hover:text-fg"
          >
            <IconDownload width={16} height={16} />
            Download PGN
          </button>
        ) : null}
      </div>
    </div>
  );
}
