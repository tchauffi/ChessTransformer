'use client';

import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { Chess, type Color } from 'chess.js';
import { Chessboard, type Arrow, type ChessboardOptions } from 'react-chessboard';

import { BotAvatar } from './Avatars';
import { NavButton } from './ChessGame';
import EvalBar from './EvalBar';
import MoveList from './MoveList';
import PlayerBar from './PlayerBar';
import SearchView from './SearchView';
import { IconCopy, IconFirst, IconFlip, IconLast, IconNext, IconPause, IconPlay, IconPrev, IconRepeat } from './icons';
import { BOARD_LOOK, CHECK, TINT } from '../core/board';
import { formatEval, kingSquare, material, other, placement, START_FEN, type Ply } from '../core/chess';
import type { SearchInfo } from '../core/game';
import { useStoredState } from '../core/hooks';

// Same origin by default: scripts/games_viewer.py serves this page and its API together.
const API = process.env.NEXT_PUBLIC_GAMES_API ?? '';
const PAGE = 50;
const STEP_MS = 650;

interface Run {
  id: string;
  label: string;
  games: number;
  /** The publications of a continuous run that played its games (training steps), newest first. */
  nets: { net: number; games: number }[];
}

interface Summary {
  id: number;
  net: number | null;
  result: string;
  end: string;
  plies: number;
  fen: string;
}

/** What self-play's search saw from one position; values for White, a move's q for its player. */
interface Position {
  v: number;
  net_v: number | null;
  full: boolean;
  visits: number;
  top: { move: string; visits: number; prior: number; q: number }[];
}

interface Game extends Summary {
  moves: string[];
  positions: Position[] | null;
}

interface Filters {
  run: string;
  net: string;
  result: string;
  order: string;
}

const RESULTS = [
  { value: 'all', label: 'All results' },
  { value: '1-0', label: 'White won' },
  { value: '0-1', label: 'Black won' },
  { value: '1/2-1/2', label: 'Draws' },
];
const ORDERS = [
  { value: 'new', label: 'Newest first' },
  { value: 'old', label: 'Oldest first' },
  { value: 'short', label: 'Shortest first' },
  { value: 'long', label: 'Longest first' },
];

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(String(res.status));
  return res.json();
}

/** The game as the demo's plies, and how the rules ended it, if they did. */
function replay(fen: string, moves: string[]): { plies: Ply[]; reason: string | null } {
  const g = new Chess(fen);
  const plies: Ply[] = [];
  for (const uci of moves) {
    try {
      const m = g.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
      plies.push({
        san: m.san,
        from: m.from,
        to: m.to,
        color: m.color,
        piece: m.piece,
        captured: m.captured,
        promotion: m.promotion,
        castle: m.isKingsideCastle() || m.isQueensideCastle(),
        check: g.inCheck(),
        fen: g.fen(),
      });
    } catch {
      break; // a move chess.js refuses: the game is shown up to there
    }
  }
  const reason = g.isCheckmate()
    ? 'by checkmate'
    : g.isStalemate()
      ? 'by stalemate'
      : g.isInsufficientMaterial()
        ? 'by insufficient material'
        : g.isThreefoldRepetition()
          ? 'by repetition'
          : g.isDrawByFiftyMoves()
            ? 'by the 50-move rule'
            : null;
  return { plies, reason };
}

function outcome(g: Summary, reason: string | null): string {
  const who = g.result === '1-0' ? 'White won' : g.result === '0-1' ? 'Black won' : 'Draw';
  const how = g.end === 'resign' ? 'by resignation' : g.end === 'max_plies' ? 'at the move limit' : reason ?? '';
  return `${who} ${how}`.trim();
}

const netName = (run: Run | undefined, g: Summary | null) =>
  !g ? '' : g.net != null ? `Network at step ${g.net.toLocaleString('en')}` : `Champion, ${run?.id ?? ''}`;

function pgn(g: Game, plies: Ply[], player: string): string {
  let n = Number(g.fen.split(' ')[5]) || 1;
  const moves: string[] = [];
  plies.forEach((p, i) => {
    if (p.color === 'w') moves.push(`${n}.`);
    else if (i === 0) moves.push(`${n}...`);
    moves.push(p.san);
    if (p.color === 'b') n += 1;
  });
  const tags = { Event: 'ChessTransformer self-play', Round: String(g.id), White: player, Black: player,
    Result: g.result, SetUp: '1', FEN: g.fen };
  return `${Object.entries(tags).map(([k, v]) => `[${k} "${v}"]`).join('\n')}\n\n${moves.join(' ')} ${g.result}\n`;
}

export default function GameViewer() {
  const [filters, setFilters] = useStoredState<Filters>('ct-viewer', { run: '', net: 'all', result: 'all', order: 'new' });
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [list, setList] = useState<{ total: number; games: Summary[] } | null>(null);
  const [offline, setOffline] = useState(false);
  const [reload, setReload] = useState(0);
  const [gameId, setGameId] = useState<number | null>(null);
  const [game, setGame] = useState<Game | null>(null);
  const [shown, setShown] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [tab, setTab] = useState<'games' | 'moves' | 'search'>('games');
  const [hoverPath, setHoverPath] = useState<{ path: string[]; key: string } | null>(null);
  const [pinnedPath, setPinnedPath] = useState<{ path: string[]; key: string } | null>(null);
  const [copied, setCopied] = useState(false);

  // A link names a position: #run=scratch&game=188&ply=2&tab=search. Read once; the ply waits
  // for its game to load.
  const pendingPly = useRef<number | null>(null);
  useEffect(() => {
    const h = new URLSearchParams(window.location.hash.slice(1));
    const t = h.get('tab');
    if (h.get('run')) setFilters((f) => ({ ...f, run: h.get('run')!, net: 'all' }));
    if (h.get('game')) setGameId(Number(h.get('game')));
    if (h.get('ply')) pendingPly.current = Number(h.get('ply'));
    if (t === 'games' || t === 'moves' || t === 'search') setTab(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- once, on load
  }, []);

  const run = runs?.find((r) => r.id === filters.run) ?? runs?.[0];
  const query = run
    ? `run=${encodeURIComponent(run.id)}&net=${filters.net}&result=${encodeURIComponent(filters.result)}&order=${filters.order}`
    : null;

  // ------------------------------------------------------------ data

  useEffect(() => {
    let cancelled = false;
    getJson<Run[]>('/api/runs')
      .then((r) => {
        if (cancelled) return;
        setRuns(r);
        setOffline(false);
      })
      .catch(() => !cancelled && setOffline(true));
    return () => {
      cancelled = true;
    };
  }, [reload]);

  useEffect(() => {
    if (!query) return;
    let cancelled = false;
    getJson<{ total: number; games: Summary[] }>(`/api/games?${query}&offset=0&limit=${PAGE}`)
      .then((l) => !cancelled && setList(l))
      .catch(() => !cancelled && setOffline(true));
    return () => {
      cancelled = true;
    };
  }, [query, reload]);

  // A fresh list opens its first game.
  useEffect(() => {
    if (gameId == null && list?.games.length) setGameId(list.games[0].id);
  }, [list, gameId]);

  const runId = run?.id;
  useEffect(() => {
    if (gameId == null || !runId) return;
    let cancelled = false;
    getJson<Game>(`/api/game?run=${encodeURIComponent(runId)}&id=${gameId}`)
      .then((g) => {
        if (cancelled) return;
        setGame(g);
        setShown(Math.max(0, Math.min(g.moves.length, pendingPly.current ?? 0)));
        pendingPly.current = null;
        setPlaying(false);
        setHoverPath(null);
        setPinnedPath(null);
      })
      .catch(() => !cancelled && setOffline(true));
    return () => {
      cancelled = true;
    };
  }, [gameId, runId]);

  const more = () => {
    if (!list || !query) return;
    getJson<{ total: number; games: Summary[] }>(`/api/games?${query}&offset=${list.games.length}&limit=${PAGE}`)
      .then((l) => setList({ total: l.total, games: [...list.games, ...l.games] }))
      .catch(() => setOffline(true));
  };

  const setFilter = (patch: Partial<Filters>) => {
    setFilters({ ...filters, ...patch });
    setList(null);
    setGameId(null);
  };

  // ------------------------------------------------------------ derived position state

  const { plies, reason } = useMemo(() => (game ? replay(game.fen, game.moves) : { plies: [], reason: null }), [game]);
  const startFen = game?.fen ?? START_FEN;
  const shownFen = shown === 0 ? startFen : plies[shown - 1].fen;
  const atEnd = shown === plies.length;
  const toMove: Color = shownFen.split(' ')[1] === 'b' ? 'b' : 'w';
  const orientation: 'white' | 'black' = flipped ? 'black' : 'white';
  const bottomColor: Color = flipped ? 'b' : 'w';
  const position = useMemo(() => placement(shownFen), [shownFen]);
  const { takenBy, balance } = useMemo(() => material(position), [position]);

  // The search shown is the one made from this position, else (at the end) the one that led to it.
  const searchIndex = game?.positions ? (atEnd ? shown - 1 : shown) : -1;
  const pos = searchIndex >= 0 ? (game?.positions?.[searchIndex] ?? null) : null;
  const search: SearchInfo | null = pos
    ? { sims: pos.visits, visits: pos.visits, tree: pos.top.map((m) => ({ ...m, children: [] })) }
    : null;
  const searchRootFen = searchIndex < 0 ? null : searchIndex === 0 ? startFen : plies[searchIndex - 1].fen;
  const evalValue = !atEnd && game?.positions ? game.positions[shown].v : null;

  const previewKey = `${game?.id}:${shown}:${tab}`;
  const activePath =
    (hoverPath?.key === previewKey ? hoverPath.path : null) ?? (pinnedPath?.key === previewKey ? pinnedPath.path : null);
  const preview = useMemo(() => {
    if (!activePath || !searchRootFen) return null;
    const g = new Chess(searchRootFen);
    const arrows: Arrow[] = [];
    let last: { from: string; to: string } | null = null;
    for (const uci of activePath) {
      try {
        const m = g.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
        arrows.push({ startSquare: m.from, endSquare: m.to, color: 'rgba(63, 191, 143, 0.95)' });
        last = { from: m.from, to: m.to };
      } catch {
        break;
      }
    }
    return { position: placement(g.fen()), arrows, last };
  }, [activePath, searchRootFen]);

  const onPreview = (path: string[] | null, pin = false) => {
    if (path && pin) {
      const same = pinnedPath?.key === previewKey && pinnedPath.path.join() === path.join();
      setPinnedPath(same ? null : { path, key: previewKey });
    } else {
      setHoverPath(path ? { path, key: previewKey } : null);
    }
  };

  const squareStyles = useMemo(() => {
    const s: Record<string, CSSProperties> = {};
    const last = shown > 0 ? plies[shown - 1] : null;
    if (last) {
      s[last.from] = { backgroundColor: TINT.last };
      s[last.to] = { backgroundColor: TINT.last };
      if (last.check) {
        const king = kingSquare(position, other(last.color));
        if (king) s[king] = { ...s[king], backgroundImage: CHECK };
      }
    }
    return s;
  }, [shown, plies, position]);

  useEffect(() => {
    if (game && runId) window.history.replaceState(null, '', `#run=${runId}&game=${game.id}&ply=${shown}&tab=${tab}`);
  }, [game, runId, shown, tab]);

  // ------------------------------------------------------------ navigation

  const goTo = (ply: number) => {
    setShown(Math.max(0, Math.min(plies.length, ply)));
    setPlaying(false);
  };

  useEffect(() => {
    if (!playing) return;
    if (shown >= plies.length) {
      setPlaying(false);
      return;
    }
    const t = window.setTimeout(() => setShown((s) => s + 1), STEP_MS);
    return () => window.clearTimeout(t);
  }, [playing, shown, plies.length]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.target instanceof HTMLElement && e.target.closest('input, textarea, select')) return;
      const step = (to: number) => {
        e.preventDefault();
        goTo(to);
      };
      switch (e.key) {
        case 'ArrowLeft':
          return step(shown - 1);
        case 'ArrowRight':
          return step(shown + 1);
        case 'ArrowUp':
        case 'Home':
          return step(0);
        case 'ArrowDown':
        case 'End':
          return step(plies.length);
        case ' ':
          e.preventDefault();
          return setPlaying((p) => !p && !atEnd);
        case 'f':
          return setFlipped((f) => !f);
        case 'Escape':
          setHoverPath(null);
          setPinnedPath(null);
          return;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const copyPgn = async () => {
    if (!game) return;
    try {
      await navigator.clipboard.writeText(pgn(game, plies, netName(run, game)));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      // clipboard blocked
    }
  };

  // ------------------------------------------------------------ view

  const boardOptions: ChessboardOptions = {
    ...BOARD_LOOK,
    id: 'ct-viewer',
    position: preview?.position ?? position,
    boardOrientation: orientation,
    squareStyles: preview
      ? preview.last
        ? { [preview.last.from]: { backgroundColor: TINT.last }, [preview.last.to]: { backgroundColor: TINT.last } }
        : {}
      : squareStyles,
    arrows: preview?.arrows ?? [],
    allowDragging: false,
    onSquareClick: () => {
      setHoverPath(null);
      setPinnedPath(null);
    },
  };

  const bar = (c: Color) => (
    <PlayerBar
      kind="bot"
      name={netName(run, game)}
      color={c}
      taken={takenBy[c]}
      advantage={c === 'w' ? balance : -balance}
      clock={null}
      toMove={!!game && !atEnd && toMove === c}
    />
  );
  const blackFirst = startFen.split(' ')[1] === 'b';
  const firstNumber = Number(startFen.split(' ')[5]) || 1;

  return (
    <main className="app">
      <div className="play-area">
        <div className="board-grid" data-eval="on">
          <div className="strip-slot lg:hidden">
            <MoveList variant="strip" plies={plies} current={shown} onSelect={goTo} firstNumber={firstNumber} blackFirst={blackFirst} />
          </div>
          <div className="top-slot">{bar(other(bottomColor))}</div>
          <div className="eval-slot">
            <EvalBar value={evalValue} final={game && atEnd ? game.result : undefined} orientation={orientation} />
          </div>
          <div className="board-slot relative" onContextMenu={(e) => e.preventDefault()}>
            <Chessboard options={boardOptions} />
          </div>
          <div className="bottom-slot">{bar(bottomColor)}</div>
        </div>

        <aside className="panel">
          <header className="flex items-center justify-between border-b border-line py-2.5 pl-4 pr-2">
            <div className="flex items-center gap-2.5">
              <BotAvatar size={26} />
              <span className="font-display text-[19px] font-bold tracking-[-0.01em] text-fg">Self-play games</span>
            </div>
            <button
              type="button"
              onClick={() => setReload((n) => n + 1)}
              title="Reload the runs and games"
              aria-label="Reload the runs and games"
              className="grid h-9 w-9 place-items-center rounded-md text-muted transition-colors hover:bg-raised hover:text-fg"
            >
              <IconRepeat />
            </button>
          </header>

          <div className="flex items-center justify-between gap-3 px-4 py-2.5 text-sm" aria-live="polite">
            <span className={game ? 'font-semibold text-fg' : 'text-muted'}>
              {offline
                ? "The games server didn't answer."
                : game
                  ? `#${game.id}: ${outcome(game, reason)}`
                  : 'Loading games…'}
            </span>
            {game ? (
              <button type="button" onClick={copyPgn} className="flex shrink-0 items-center gap-1.5 text-faint hover:text-fg">
                <IconCopy width={15} height={15} />
                {copied ? 'Copied' : 'PGN'}
              </button>
            ) : null}
          </div>

          <div role="tablist" aria-label="Viewer panel" className="flex gap-1 border-t border-line px-2">
            {(['games', 'moves', 'search'] as const).map((t) => (
              <button
                key={t}
                type="button"
                role="tab"
                aria-selected={tab === t}
                onClick={() => setTab(t)}
                className={`relative px-3 py-2.5 text-sm font-semibold transition-colors ${
                  tab === t
                    ? 'text-fg after:absolute after:inset-x-2 after:bottom-0 after:h-0.5 after:rounded-full after:bg-jade'
                    : 'text-muted hover:text-fg'
                }`}
              >
                {t === 'games' ? 'Games' : t === 'moves' ? 'Moves' : 'Search'}
              </button>
            ))}
          </div>
          <div
            role="tabpanel"
            className="flex min-h-0 flex-1 flex-col border-y border-line bg-black/[0.06] max-lg:h-96 max-lg:flex-none"
            onMouseLeave={() => setHoverPath(null)}
          >
            {offline ? (
              <p className="px-4 py-3 text-sm leading-snug text-muted">
                Start the server on the training machine with <code className="text-fg">make viewer</code>, then reload.
              </p>
            ) : tab === 'games' ? (
              <GameList
                runs={runs}
                run={run}
                filters={filters}
                onFilter={setFilter}
                list={list}
                current={gameId}
                onOpen={setGameId}
                onMore={more}
              />
            ) : tab === 'moves' ? (
              <MoveList
                variant="table"
                plies={plies}
                current={shown}
                onSelect={goTo}
                firstNumber={firstNumber}
                blackFirst={blackFirst}
                emptyText="Pick a game in Games."
              />
            ) : search && searchRootFen && pos ? (
              <div className="flex min-h-0 flex-1 flex-col">
                <p className="px-4 pt-3 text-sm leading-snug text-muted">
                  {pos.full ? 'Full search' : 'Fast search'}: it valued the position{' '}
                  <span className="font-semibold text-fg">{formatEval(pos.v)}</span>
                  {pos.net_v != null ? (
                    <>
                      , the network alone <span className="font-semibold text-fg">{formatEval(pos.net_v)}</span>
                    </>
                  ) : null}{' '}
                  (for White).
                </p>
                <SearchView
                  search={search}
                  fen={searchRootFen}
                  played={plies[searchIndex]?.san}
                  active={activePath}
                  onPreview={onPreview}
                  showPrior
                />
              </div>
            ) : (
              <p className="px-4 py-3 text-sm leading-snug text-muted">
                {game && !game.positions
                  ? "This game's search isn't on disk yet: self-play writes it with the next batch of positions."
                  : 'Pick a game in Games.'}
              </p>
            )}
          </div>

          <div className="mt-auto flex items-center gap-1 px-3 pb-3 pt-3">
            <NavButton onClick={() => goTo(0)} disabled={shown === 0} label="First move (Home)">
              <IconFirst />
            </NavButton>
            <NavButton onClick={() => goTo(shown - 1)} disabled={shown === 0} label="Previous move (←)">
              <IconPrev />
            </NavButton>
            <NavButton
              onClick={() => setPlaying((p) => !p && !atEnd)}
              disabled={!game || (atEnd && !playing)}
              label={playing ? 'Pause (space)' : 'Play through the game (space)'}
            >
              {playing ? <IconPause /> : <IconPlay />}
            </NavButton>
            <NavButton onClick={() => goTo(shown + 1)} disabled={atEnd} label="Next move (→)">
              <IconNext />
            </NavButton>
            <NavButton onClick={() => goTo(plies.length)} disabled={atEnd} label="Last move (End)">
              <IconLast />
            </NavButton>
            <NavButton onClick={() => setFlipped((f) => !f)} label="Flip board (F)" narrow>
              <IconFlip />
            </NavButton>
          </div>
        </aside>
      </div>
    </main>
  );
}

function GameList({
  runs,
  run,
  filters,
  onFilter,
  list,
  current,
  onOpen,
  onMore,
}: {
  runs: Run[] | null;
  run: Run | undefined;
  filters: Filters;
  onFilter: (patch: Partial<Filters>) => void;
  list: { total: number; games: Summary[] } | null;
  current: number | null;
  onOpen: (id: number) => void;
  onMore: () => void;
}) {
  const runOptions = (runs ?? []).map((r) => ({ value: r.id, label: `${r.label} · ${r.games.toLocaleString('en')}` }));
  const netOptions = [
    { value: 'all', label: 'All networks' },
    ...(run?.nets ?? []).map((n) => ({ value: String(n.net), label: `Step ${n.net.toLocaleString('en')} · ${n.games}` })),
  ];
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid grid-cols-2 gap-2 px-3 py-3">
        <Select id="viewer-run" label="Run" value={run?.id ?? ''} options={runOptions} onChange={(v) => onFilter({ run: v, net: 'all' })} />
        <Select
          id="viewer-net"
          label="Network"
          value={filters.net}
          options={netOptions}
          onChange={(v) => onFilter({ net: v })}
          disabled={!run?.nets.length}
        />
        <Select id="viewer-result" label="Result" value={filters.result} options={RESULTS} onChange={(v) => onFilter({ result: v })} />
        <Select id="viewer-order" label="Order" value={filters.order} options={ORDERS} onChange={(v) => onFilter({ order: v })} />
      </div>
      <p className="px-4 pb-1.5 text-xs text-faint">
        {list ? `${list.total.toLocaleString('en')} games` : 'Loading…'}
      </p>
      <ol className="min-h-0 flex-1 overflow-y-auto border-t border-line">
        {list?.games.map((g) => (
          <li key={g.id}>
            <button
              type="button"
              onClick={() => onOpen(g.id)}
              className={`grid w-full grid-cols-[4.5rem_3.25rem_1fr_auto] items-center gap-2 px-4 py-1.5 text-left text-sm transition-colors ${
                g.id === current ? 'bg-line text-fg' : 'text-fg/85 hover:bg-raised'
              }`}
            >
              <span className="tabular-nums text-faint">#{g.id}</span>
              <ResultChip result={g.result} />
              <span className="tabular-nums text-muted">
                {g.plies} plies{g.end === 'resign' ? ', resigned' : g.end === 'max_plies' ? ', move limit' : ''}
              </span>
              <span className="tabular-nums text-faint">{g.net != null ? `step ${g.net.toLocaleString('en')}` : ''}</span>
            </button>
          </li>
        ))}
        {list && list.games.length < list.total ? (
          <li className="px-3 py-2">
            <button type="button" onClick={onMore} className="btn-secondary h-9 w-full text-sm">
              Show {Math.min(PAGE, list.total - list.games.length)} more
            </button>
          </li>
        ) : null}
      </ol>
    </div>
  );
}

function ResultChip({ result }: { result: string }) {
  const look =
    result === '1-0' ? 'bg-sand text-ink' : result === '0-1' ? 'bg-ink text-fg ring-1 ring-line' : 'bg-raised text-muted';
  return (
    <span className={`rounded px-1.5 py-0.5 text-center text-xs font-semibold tabular-nums ${look}`}>
      {result === '1/2-1/2' ? '½-½' : result}
    </span>
  );
}

function Select({
  id,
  label,
  value,
  options,
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
  disabled?: boolean;
}) {
  return (
    <label htmlFor={id} className="flex min-w-0 flex-col gap-1 text-xs text-faint">
      {label}
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="h-9 min-w-0 rounded-md border border-line bg-raised px-2 text-sm text-fg disabled:opacity-50"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
