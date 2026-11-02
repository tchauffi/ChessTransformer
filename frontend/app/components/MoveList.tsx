'use client';

import { useEffect, useRef } from 'react';
import type { Ply } from '../core/chess';

interface Props {
  plies: Ply[];
  /** Number of plies applied in the shown position (0 = start). */
  current: number;
  onSelect: (index: number) => void;
  variant: 'table' | 'strip';
  emptyText?: string;
  /** For a game from a set-up position: its first move number, and Black to move first. */
  firstNumber?: number;
  blackFirst?: boolean;
}

export default function MoveList({ plies, current, onSelect, variant, emptyText, firstNumber = 1, blackFirst = false }: Props) {
  const shift = blackFirst ? 1 : 0; // an empty White half-move before Black's first
  const box = useRef<HTMLDivElement>(null);

  // Keep the shown move in view without scrolling the page itself.
  useEffect(() => {
    const el = box.current;
    const active = el?.querySelector<HTMLElement>('[data-active="true"]');
    if (!el) return;
    if (variant === 'strip') {
      el.scrollLeft = active
        ? active.offsetLeft - el.clientWidth / 2 + active.clientWidth / 2
        : current === 0
          ? 0
          : el.scrollWidth;
    } else if (active) {
      const top = active.offsetTop - el.offsetTop;
      if (top < el.scrollTop || top + active.clientHeight > el.scrollTop + el.clientHeight) {
        el.scrollTop = top - el.clientHeight / 2;
      }
    } else if (current === plies.length) {
      el.scrollTop = el.scrollHeight;
    }
  }, [current, plies.length, variant]);

  const rows = Array.from({ length: Math.ceil((plies.length + shift) / 2) }, (_, i) => i);

  const move = (index: number) => {
    const ply = plies[index - 1];
    if (!ply) return null;
    const active = index === current;
    return (
      <button
        type="button"
        data-active={active}
        onClick={() => onSelect(index)}
        className={`move-btn rounded px-1.5 py-0.5 text-left tabular-nums ${
          active ? 'bg-line font-bold text-fg' : 'text-fg/85 hover:bg-raised'
        }`}
      >
        {ply.san}
      </button>
    );
  };

  if (variant === 'strip') {
    return (
      <div ref={box} className="no-scrollbar flex h-9 items-center gap-1 overflow-x-auto whitespace-nowrap px-2 text-sm">
        {plies.length === 0 ? <span className="text-muted">{emptyText}</span> : null}
        {plies.map((_, i) => (
          <span key={i} className="flex items-center gap-1">
            {(i + shift) % 2 === 0 ? (
              <span className="text-faint">{firstNumber + (i + shift) / 2}.</span>
            ) : i === 0 ? (
              <span className="text-faint">{firstNumber}…</span>
            ) : null}
            {move(i + 1)}
          </span>
        ))}
      </div>
    );
  }

  return (
    <div ref={box} className="min-h-0 flex-1 overflow-y-auto text-[15px]">
      {plies.length === 0 ? <p className="px-4 py-3 text-sm text-muted">{emptyText}</p> : null}
      <ol>
        {rows.map((r) => (
          <li
            key={r}
            className={`grid grid-cols-[3rem_1fr_1fr] items-center px-2 py-[3px] ${r % 2 ? 'bg-white/[0.025]' : ''}`}
          >
            <span className="pl-2 text-faint tabular-nums">{firstNumber + r}.</span>
            <span>{r === 0 && shift ? <span className="px-1.5 text-faint">…</span> : move(2 * r + 1 - shift)}</span>
            <span>{move(2 * r + 2 - shift)}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
