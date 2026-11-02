'use client';

import { useMemo } from 'react';
import { Chess, type Color } from 'chess.js';
import { formatEval } from '../core/chess';
import type { SearchInfo, SearchNode } from '../core/game';

/** A searched move with its SAN and the UCI path that leads to it from the searched position. */
interface Branch {
  id: string;
  san: string;
  color: Color;
  visits: number;
  prior: number;
  q: number;
  path: string[];
  children: Branch[];
}

function annotate(fen: string, nodes: SearchNode[], path: string[] = []): Branch[] {
  const out: Branch[] = [];
  for (const n of nodes) {
    const g = new Chess(fen);
    try {
      const m = g.move({ from: n.move.slice(0, 2), to: n.move.slice(2, 4), promotion: n.move[4] });
      const p = [...path, n.move];
      out.push({
        id: p.join(' '),
        san: m.san,
        color: m.color,
        visits: n.visits,
        prior: n.prior,
        q: n.q,
        path: p,
        children: annotate(g.fen(), n.children, p),
      });
    } catch {
      // a move that doesn't parse here would mean a stale tree; skip it
    }
  }
  return out;
}

/** The branch followed by always taking the most-visited reply. */
const mainLine = (b: Branch): Branch[] => (b.children[0] ? [b, ...mainLine(b.children[0])] : [b]);

// ---------------------------------------------------------------- tree layout

const MAX_DEPTH = 6;
const ROW = 20;
const COL = 56;
const ROOT_X = 8;
const FIRST_X = 26;
const LABEL_W = 40;

interface Placed {
  branch: Branch;
  depth: number;
  x: number;
  y: number;
  parent: Placed | null;
}

/** Tidy tree, left to right: leaves take consecutive rows, parents centre on their children. */
function layout(roots: Branch[]) {
  const placed: Placed[] = [];
  let row = 0;
  const place = (branch: Branch, depth: number, parent: Placed | null): Placed => {
    const me: Placed = { branch, depth, x: FIRST_X + (depth - 1) * COL, y: 0, parent };
    placed.push(me);
    const kids = depth < MAX_DEPTH ? branch.children : [];
    if (!kids.length) {
      me.y = row++ * ROW + ROW / 2;
    } else {
      const ys = kids.map((k) => place(k, depth + 1, me).y);
      me.y = (ys[0] + ys[ys.length - 1]) / 2;
    }
    return me;
  };
  const tops = roots.map((r) => place(r, 1, null));
  const rootY = tops.length ? (tops[0].y + tops[tops.length - 1].y) / 2 : ROW / 2;
  return { placed, rootY, height: Math.max(row, 1) * ROW };
}

// ---------------------------------------------------------------- view

interface Props {
  search: SearchInfo;
  /** The searched position. */
  fen: string;
  /** SAN of the move the engine played from it, once known. */
  played?: string;
  /** Path being previewed on the board, if any. */
  active: string[] | null;
  onPreview: (path: string[] | null, pin?: boolean) => void;
  /** Also show each move's prior from the network (the games viewer). */
  showPrior?: boolean;
}

export default function SearchView({ search, fen, played, active, onPreview, showPrior }: Props) {
  const roots = useMemo(() => annotate(fen, search.tree), [fen, search.tree]);
  const { placed, rootY, height } = useMemo(() => layout(roots), [roots]);
  const total = Math.max(1, search.visits);
  const activeId = active?.join(' ') ?? null;
  const chosen = roots.find((r) => r.san === played) ?? roots[0];
  const mainIds = useMemo(() => new Set(chosen ? mainLine(chosen).map((b) => b.id) : []), [chosen]);
  const onActive = (id: string) => !!activeId && (activeId === id || activeId.startsWith(`${id} `));

  const hover = (path: string[]) => ({
    onMouseEnter: () => onPreview(path),
    onMouseLeave: () => onPreview(null),
    onFocus: () => onPreview(path),
    onBlur: () => onPreview(null),
    onClick: () => onPreview(path, true),
  });

  return (
    <div className="min-h-0 flex-1 overflow-y-auto pb-3">
      <p className="px-4 pt-3 text-sm leading-snug text-muted">
        {played ? (
          <>
            Chose <span className="font-semibold text-fg">{played}</span> after
          </>
        ) : (
          'Weighed'
        )}{' '}
        {search.visits.toLocaleString()} positions
        {search.elapsedMs != null ? ` in ${(search.elapsedMs / 1000).toFixed(1)} s` : ''}. Bars show each move&apos;s
        share of the search{showPrior ? ', the faint figure the share the network gave it before searching' : ''};
        hover or tap one to replay its line on the board.
      </p>
      {search.depth ? (
        <p className="px-4 pt-1.5 text-sm leading-snug text-muted">
          Its main line runs <span className="font-semibold text-fg">{search.depth.mainLine}</span> half-moves deep and
          the deepest branch {search.depth.max}; a typical simulation stops after {search.depth.mean.toFixed(0)}. The
          diagram below shows only the first {MAX_DEPTH}.
        </p>
      ) : null}

      <ol className="mt-2 px-2">
        {roots.map((r) => {
          const share = r.visits / total;
          const line = mainLine(r).slice(1, 8);
          const white = r.color === 'w' ? r.q : -r.q;
          const isChosen = r === chosen;
          return (
            <li key={r.id}>
              <button
                type="button"
                {...hover(mainLine(r).at(-1)!.path)}
                className={`w-full rounded-md px-2 py-1.5 text-left transition-colors hover:bg-raised ${
                  onActive(r.id) ? 'bg-raised' : ''
                }`}
              >
                <span className="flex items-center gap-2.5 text-sm">
                  <span className={`w-11 shrink-0 font-semibold ${isChosen ? 'text-jade' : 'text-fg'}`}>{r.san}</span>
                  <span className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-raised">
                    <span
                      className={`absolute inset-y-0 left-0 rounded-full ${isChosen ? 'bg-jade' : 'bg-muted/60'}`}
                      style={{ width: `${Math.max(2, share * 100)}%` }}
                    />
                  </span>
                  <span className="w-9 shrink-0 text-right tabular-nums text-muted">{Math.round(share * 100)}%</span>
                  {showPrior ? (
                    <span className="w-9 shrink-0 text-right tabular-nums text-faint">{Math.round(r.prior * 100)}%</span>
                  ) : null}
                  <span className="w-11 shrink-0 text-right font-medium tabular-nums text-fg">{formatEval(white)}</span>
                </span>
                {line.length ? (
                  <span className="mt-0.5 block truncate pl-[3.375rem] text-xs text-faint">
                    {line.map((b) => b.san).join(' ')}
                  </span>
                ) : null}
              </button>
            </li>
          );
        })}
      </ol>

      <h3 className="mt-3 px-4 text-sm font-medium text-muted">Branches explored</h3>
      <svg
        className="mt-1.5 block w-full"
        viewBox={`0 0 ${FIRST_X + (MAX_DEPTH - 1) * COL + LABEL_W + 4} ${height}`}
        role="img"
        aria-label="The engine's search tree: each column is one more move ahead, line thickness is search effort"
      >
        <circle cx={ROOT_X} cy={rootY} r={4} className="fill-sand" />
        {placed.map((p) => {
          const x1 = p.parent ? p.parent.x + LABEL_W : ROOT_X + 4;
          const y1 = p.parent ? p.parent.y : rootY;
          const mid = (x1 + p.x) / 2;
          const share = p.branch.visits / total;
          const lit = onActive(p.branch.id);
          const main = mainIds.has(p.branch.id);
          return (
            <path
              key={`e-${p.branch.id}`}
              d={`M${x1},${y1} C${mid},${y1} ${mid},${p.y} ${p.x - 2},${p.y}`}
              fill="none"
              strokeLinecap="round"
              strokeWidth={1 + 6 * Math.sqrt(share)}
              className={lit ? 'stroke-[#f0a830]' : main ? 'stroke-jade' : 'stroke-muted'}
              strokeOpacity={lit || main ? 0.9 : 0.25 + 0.6 * Math.sqrt(share)}
            />
          );
        })}
        {placed.map((p) => {
          const lit = onActive(p.branch.id);
          const main = mainIds.has(p.branch.id);
          return (
            <g key={`n-${p.branch.id}`} {...hover(p.branch.path)} className="cursor-pointer" tabIndex={0}>
              <rect
                x={p.x - 2}
                y={p.y - 8}
                width={LABEL_W}
                height={16}
                rx={4}
                className={lit ? 'fill-[#f0a830]' : 'fill-panel'}
              />
              <text
                x={p.x + 2}
                y={p.y + 4}
                fontSize={11.5}
                fontWeight={main || lit ? 700 : 500}
                className={lit ? 'fill-ink' : main ? 'fill-jade' : 'fill-fg'}
                fillOpacity={lit || main ? 1 : 0.55 + 0.45 * Math.sqrt(p.branch.visits / total)}
              >
                {p.branch.san}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
