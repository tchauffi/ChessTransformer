'use client';

import type { JSX } from 'react';
import { defaultPieces } from 'react-chessboard';
import { BotAvatar } from './Avatars';
import { TIME_CONTROLS } from '../core/chess';

export type ColorChoice = 'w' | 'random' | 'b';

export interface EngineInfo {
  sims?: number;
  elo?: number;
  model?: string;
}

interface Props {
  engine: EngineInfo | null;
  offline: boolean;
  color: ColorChoice;
  onColor: (c: ColorChoice) => void;
  timeControl: string;
  onTimeControl: (id: string) => void;
  onPlay: () => void;
  onWatch: () => void;
}

const WhiteKing = defaultPieces.wK;
const BlackKing = defaultPieces.bK;

function RandomKing() {
  return (
    <span className="relative block h-full w-full">
      <span className="absolute inset-0 [clip-path:inset(0_50%_0_0)]">
        <WhiteKing />
      </span>
      <span className="absolute inset-0 [clip-path:inset(0_0_0_50%)]">
        <BlackKing />
      </span>
    </span>
  );
}

const COLORS: { id: ColorChoice; label: string; Icon: () => JSX.Element }[] = [
  { id: 'w', label: 'White', Icon: WhiteKing },
  { id: 'random', label: 'Random', Icon: RandomKing },
  { id: 'b', label: 'Black', Icon: BlackKing },
];

export default function SetupPanel({ engine, offline, color, onColor, timeControl, onTimeControl, onPlay, onWatch }: Props) {
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
      <section className="flex gap-4 px-5 pb-5 pt-2">
        <BotAvatar size={76} thinking={!engine && !offline} />
        <div className="min-w-0">
          <h2 className="font-display text-[22px] font-bold leading-tight text-fg">ChessTransformer</h2>
          <p className="mt-0.5 text-sm text-muted">
            {engine?.elo ? `Rated about ${engine.elo}` : offline ? 'Engine offline' : 'Connecting to the engine'}
          </p>
          <p className="mt-2.5 text-sm leading-relaxed text-fg/80">
            A transformer that learned chess from human games only, then searches{' '}
            {engine?.sims ? `${engine.sims} positions` : 'ahead'} before each move.
          </p>
        </div>
      </section>

      <section className="border-t border-line px-5 py-4">
        <h3 className="mb-2.5 text-sm font-medium text-muted">I play as</h3>
        <div className="grid grid-cols-3 gap-2" role="radiogroup" aria-label="Your color">
          {COLORS.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              role="radio"
              aria-checked={color === id}
              onClick={() => onColor(id)}
              className={`choice flex flex-col items-center gap-1 rounded-lg py-2.5 text-sm font-medium ${
                color === id ? 'choice-on' : ''
              }`}
            >
              <span className="block h-9 w-9">
                <Icon />
              </span>
              {label}
            </button>
          ))}
        </div>
      </section>

      <section className="border-t border-line px-5 py-4">
        <h3 className="mb-2.5 text-sm font-medium text-muted">Time control</h3>
        <div className="grid grid-cols-3 gap-2" role="radiogroup" aria-label="Time control">
          {TIME_CONTROLS.map((tc) => (
            <button
              key={tc.id}
              type="button"
              role="radio"
              aria-checked={timeControl === tc.id}
              onClick={() => onTimeControl(tc.id)}
              className={`choice h-11 rounded-lg text-[15px] font-semibold tabular-nums ${
                timeControl === tc.id ? 'choice-on' : ''
              }`}
            >
              {tc.label}
            </button>
          ))}
        </div>
        <p className="mt-2.5 text-[13px] leading-snug text-faint">
          The engine&apos;s clock runs while it searches, so faster time controls press it too.
        </p>
      </section>

      <div className="mt-auto px-5 pb-5 pt-2">
        <button type="button" onClick={onPlay} disabled={offline} className="btn-primary h-14 w-full text-xl">
          Play
        </button>
        <button type="button" onClick={onWatch} disabled={offline} className="btn-secondary mt-2 h-11 w-full disabled:opacity-45">
          Watch it play itself
        </button>
        {offline ? (
          <p className="mt-3 text-[13px] leading-snug text-coral">
            Can&apos;t reach the engine. Start the backend (see the README) and this page will connect on its own.
          </p>
        ) : null}
        <nav className="mt-4 flex justify-center gap-5 text-[13px] text-faint">
          <a href="https://github.com/tchauffi/ChessTransformer" target="_blank" rel="noopener noreferrer" className="hover:text-fg">
            Source on GitHub
          </a>
          <a href="https://lichess.org/@/ChessTransformerBot" target="_blank" rel="noopener noreferrer" className="hover:text-fg">
            Challenge it on Lichess
          </a>
        </nav>
      </div>
    </div>
  );
}
