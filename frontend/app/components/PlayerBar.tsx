'use client';

import { useEffect, useState } from 'react';
import type { Color, PieceSymbol } from 'chess.js';
import { defaultPieces } from 'react-chessboard';
import { BotAvatar, HumanAvatar } from './Avatars';
import { formatClock, other } from '../core/chess';

export interface ClockView {
  ms: number; // remaining when `since` was stamped
  since: number;
  running: boolean;
}

interface Props {
  kind: 'bot' | 'human';
  name: string;
  rating?: number;
  color: Color;
  taken: PieceSymbol[];
  advantage: number;
  clock: ClockView | null;
  toMove: boolean;
  thinking?: boolean;
}

export default function PlayerBar({ kind, name, rating, color, taken, advantage, clock, toMove, thinking }: Props) {
  return (
    <div className="flex h-12 items-center gap-3 px-0.5">
      {kind === 'bot' ? <BotAvatar size={40} thinking={thinking} /> : <HumanAvatar size={40} color={color} />}
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-1.5 truncate leading-tight">
          <span className="truncate text-[15px] font-semibold text-fg">{name}</span>
          {rating ? <span className="text-sm text-muted">({rating})</span> : null}
          {thinking ? <ThinkingDots /> : null}
        </div>
        <Captured pieces={taken} pieceColor={other(color)} advantage={advantage} />
      </div>
      {clock ? <Clock {...clock} toMove={toMove} /> : null}
    </div>
  );
}

function ThinkingDots() {
  return (
    <span className="ml-1 inline-flex gap-[3px] self-center" aria-label="thinking">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="thinking-dot h-[5px] w-[5px] rounded-full bg-jade"
          style={{ animationDelay: `${i * 160}ms` }}
        />
      ))}
    </span>
  );
}

function Captured({ pieces, pieceColor, advantage }: { pieces: PieceSymbol[]; pieceColor: Color; advantage: number }) {
  const groups = pieces.reduce<PieceSymbol[][]>((acc, p) => {
    const last = acc[acc.length - 1];
    if (last && last[0] === p) last.push(p);
    else acc.push([p]);
    return acc;
  }, []);
  return (
    <div className="mt-0.5 flex h-[18px] items-center gap-1.5">
      {groups.map((group) => (
        <span key={group[0]} className="flex">
          {group.map((p, i) => {
            const Piece = defaultPieces[`${pieceColor}${p.toUpperCase()}`];
            return (
              <span
                key={i}
                className={`block h-[18px] w-[18px] ${i ? '-ml-[11px]' : ''} ${pieceColor === 'b' ? 'captured-dark' : ''}`}
              >
                <Piece />
              </span>
            );
          })}
        </span>
      ))}
      {advantage > 0 ? <span className="text-[13px] font-medium text-muted">+{advantage}</span> : null}
    </div>
  );
}

function Clock({ ms, since, running, toMove }: ClockView & { toMove: boolean }) {
  const [now, setNow] = useState(since);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(id);
  }, [running]);

  const left = running ? ms - Math.max(0, now - since) : ms;
  const low = left < 20_000;
  const active = running || toMove;
  const tone = active
    ? low
      ? 'bg-coral text-white'
      : 'bg-sand text-ink'
    : low
      ? 'bg-raised text-coral'
      : 'bg-raised text-muted';
  return (
    <div
      className={`min-w-[5.25rem] rounded-md px-2.5 py-1.5 text-right text-[20px] font-semibold leading-none tabular-nums transition-colors lg:min-w-[6.5rem] lg:px-3 lg:text-[22px] ${tone}`}
      role="timer"
      aria-live="off"
    >
      {formatClock(left)}
    </div>
  );
}
