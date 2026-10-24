'use client';

import { useEffect, useState } from 'react';
import type { Color } from 'chess.js';
import { defaultPieces } from 'react-chessboard';

/**
 * How strongly a query square attends to another, using the chess-geometry
 * relations the model's attention bias is built on: king-adjacent, knight
 * reach, shared file/rank, shared diagonal, everything else.
 */
function attention(q: number, s: number): number {
  if (q === s) return 1;
  const df = Math.abs((q % 8) - (s % 8));
  const dr = Math.abs((q >> 3) - (s >> 3));
  const d = Math.max(df, dr);
  if (d === 1) return 0.7;
  if (df * dr === 2) return 0.56;
  if (df === 0 || dr === 0) return 0.48 - d * 0.045;
  if (df === dr) return 0.42 - d * 0.04;
  return 0.05;
}

const IDLE_QUERY = 28; // e4 from the avatar's point of view

/**
 * The engine's avatar: an 8×8 attention map. While the engine searches, the
 * query square wanders the board and the map follows it.
 */
export function BotAvatar({ size = 40, thinking = false }: { size?: number; thinking?: boolean }) {
  const [query, setQuery] = useState(IDLE_QUERY);

  useEffect(() => {
    if (!thinking || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    let seed = 17;
    const id = window.setInterval(() => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      setQuery((q) => {
        // a knight-ish hop, wrapped onto the board, keeps the motion legible
        const hops = [17, 15, 10, 6, -6, -10, -15, -17, 9, -9, 7, -7];
        return (q + hops[seed % hops.length] + 64) % 64;
      });
    }, 240);
    return () => {
      window.clearInterval(id);
      setQuery(IDLE_QUERY);
    };
  }, [thinking]);

  const gap = size >= 64 ? 1 : 0;
  return (
    <div
      className="grid shrink-0 overflow-hidden rounded-[22%] bg-[#0f1a17]"
      style={{
        width: size,
        height: size,
        padding: Math.round(size * 0.08),
        gridTemplateColumns: 'repeat(8, 1fr)',
        gridTemplateRows: 'repeat(8, 1fr)',
        gap,
      }}
      aria-hidden
    >
      {Array.from({ length: 64 }, (_, s) => {
        const a = attention(query, s);
        return (
          <span
            key={s}
            className="rounded-[1px] transition-[background-color] duration-200"
            style={{ backgroundColor: `rgba(63, 191, 143, ${0.08 + 0.92 * a})` }}
          />
        );
      })}
    </div>
  );
}

export function HumanAvatar({ size = 40, color }: { size?: number; color: Color }) {
  const Pawn = defaultPieces[color === 'w' ? 'wP' : 'bP'];
  return (
    <div
      className="grid shrink-0 place-items-center rounded-[22%] bg-raised"
      style={{ width: size, height: size }}
      aria-hidden
    >
      <div style={{ width: size * 0.78, height: size * 0.78 }}>
        <Pawn />
      </div>
    </div>
  );
}
