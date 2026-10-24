'use client';

import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type CSSProperties, type ReactNode } from 'react';
import { Chess, type Color, type PieceSymbol, type Square } from 'chess.js';
import { Chessboard, defaultArrowOptions, type Arrow, type ChessboardOptions } from 'react-chessboard';

import { BotAvatar } from './Avatars';
import EvalBar from './EvalBar';
import GameOverModal from './GameOverModal';
import MoveList from './MoveList';
import PlayerBar, { type ClockView } from './PlayerBar';
import PromotionPicker from './PromotionPicker';
import SettingsMenu, { DEFAULT_SETTINGS, type Settings } from './SettingsMenu';
import SearchView from './SearchView';
import SetupPanel, { type ColorChoice, type EngineInfo } from './SetupPanel';
import {
  IconCopy,
  IconDownload,
  IconDraw,
  IconFirst,
  IconFlag,
  IconFlip,
  IconLast,
  IconNext,
  IconPause,
  IconPlay,
  IconPlus,
  IconPrev,
  IconRepeat,
  IconUndo,
} from './icons';
import {
  applyPremove,
  isPromotionSquare,
  kingSquare,
  material,
  movePiece,
  other,
  placement,
  premoveDests,
  reasonText,
  RESULT_SCORE,
  START_FEN,
  TIME_CONTROLS,
  timeControlById,
} from '../core/chess';
import { DRAW_OFFER_GAP, evalAt, GameController } from '../core/game';
import { useStoredState } from '../core/hooks';

// An empty NEXT_PUBLIC_API_URL means same origin (the Space serves the API and the page together).
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:5001';

// Board palette: sand and deep teal, with brass for move highlights and coral for
// premoves and annotations.
const BOARD = { light: '#e9e1cc', dark: '#5d8a84' };
const TINT = {
  last: 'rgba(247, 206, 84, 0.5)',
  selected: 'rgba(247, 206, 84, 0.75)',
  premove: 'rgba(214, 86, 74, 0.6)',
  mark: 'rgba(226, 92, 68, 0.85)',
};
const DOT = 'radial-gradient(circle, rgba(18, 38, 33, 0.22) 21%, transparent 22.5%)';
const RING = 'radial-gradient(circle, transparent 71%, rgba(18, 38, 33, 0.24) 72.5%)';
const CHECK =
  'radial-gradient(circle, rgba(236, 62, 46, 0.95) 0%, rgba(236, 62, 46, 0.6) 34%, rgba(236, 62, 46, 0) 72%)';
const NOTATION_SIZE = 'clamp(9px, 1.55vmin, 13px)';

interface PendingPromotion {
  from: Square;
  to: Square;
  premove: boolean;
}

interface SetupPrefs {
  color: ColorChoice;
  tc: string;
}

export default function ChessGame() {
  const [ctl] = useState(() => new GameController(API_BASE_URL));
  const snap = useSyncExternalStore(ctl.subscribe, ctl.getSnapshot, ctl.getSnapshot);
  const { phase, plies, playerColor, thinking, premoves, result, clock, autoplay } = snap;

  const [prefs, setPrefs] = useStoredState<SetupPrefs>('ct-setup', { color: 'w', tc: 'none' });
  const [settings, setSettings] = useStoredState<Settings>('ct-settings', DEFAULT_SETTINGS);
  const [engine, setEngine] = useState<EngineInfo | null>(null);
  const [offline, setOffline] = useState(false);

  // Board interaction state. `view` and `marks` remember the move count they were
  // set at, so a new move returns the board to the live position and wipes marks.
  const [flipped, setFlipped] = useState(false);
  const [view, setView] = useState<{ ply: number; len: number } | null>(null);
  const [selected, setSelected] = useState<Square | null>(null);
  const [marks, setMarks] = useState<{ squares: string[]; len: number; at: number } | null>(null);
  const [promotion, setPromotion] = useState<PendingPromotion | null>(null);
  const [confirmResign, setConfirmResign] = useState(false);
  const [dismissedGame, setDismissedGame] = useState(0);
  const [copied, setCopied] = useState(false);
  const [tab, setTab] = useState<'moves' | 'search'>('moves');
  // A search line replayed on the board: hovered, or pinned by a click. Both are
  // tied to the position they were set on, so any move or navigation drops them.
  const [hoverPath, setHoverPath] = useState<{ path: string[]; key: string } | null>(null);
  const [pinnedPath, setPinnedPath] = useState<{ path: string[]; key: string } | null>(null);
  const lastDrop = useRef(0);
  const rightDown = useRef<string | null>(null);

  useEffect(() => ctl.setSoundEnabled(settings.sound), [ctl, settings.sound]);
  useEffect(() => () => ctl.dispose(), [ctl]);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    const poll = (attempt: number) => {
      fetch(`${API_BASE_URL}/api/health`)
        .then((res) => res.json())
        .then((data) => {
          if (cancelled) return;
          setEngine({ sims: data.sims, elo: data.approx_elo, model: data.model });
          setOffline(false);
        })
        .catch(() => {
          if (cancelled) return;
          if (attempt >= 2) setOffline(true);
          timer = window.setTimeout(() => poll(attempt + 1), 2500);
        });
    };
    poll(0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  // ------------------------------------------------------------ derived position state

  const botColor = other(playerColor);
  const liveFen = plies.length ? plies[plies.length - 1].fen : START_FEN;
  const shown = view && view.len === plies.length ? view.ply : plies.length;
  const isLive = shown === plies.length;
  const shownFen = shown === 0 ? START_FEN : plies[shown - 1].fen;
  const orientation: 'white' | 'black' = (playerColor === 'w') !== flipped ? 'white' : 'black';
  const visibleMarks = useMemo(
    () => (marks && marks.len === plies.length && marks.at === shown ? marks.squares : []),
    [marks, plies.length, shown],
  );

  // The search shown is the one made from this position, else the one that led to it.
  const searchIndex = snap.searches[shown] ? shown : snap.searches[shown - 1] ? shown - 1 : null;
  const search = searchIndex == null ? null : snap.searches[searchIndex];
  const previewKey = `${plies.length}:${shown}:${tab}`;
  const activePath =
    (hoverPath?.key === previewKey ? hoverPath.path : null) ?? (pinnedPath?.key === previewKey ? pinnedPath.path : null);
  const searchRootFen = searchIndex == null ? null : searchIndex === 0 ? START_FEN : plies[searchIndex - 1].fen;

  const preview = useMemo(() => {
    if (!activePath || !searchRootFen) return null;
    const g = new Chess(searchRootFen);
    const side = g.turn();
    const arrows: Arrow[] = [];
    let last: { from: string; to: string } | null = null;
    for (const [i, uci] of activePath.entries()) {
      try {
        const m = g.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
        // the searching side's moves in jade, replies in brass; older moves fade
        const alpha = Math.max(0.3, 0.95 - (activePath.length - 1 - i) * 0.13).toFixed(2);
        const rgb = m.color === side ? '63, 191, 143' : '240, 168, 48';
        arrows.push({ startSquare: m.from, endSquare: m.to, color: `rgba(${rgb}, ${alpha})` });
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
  const clearPreview = () => {
    setHoverPath(null);
    setPinnedPath(null);
  };

  const liveGame = useMemo(() => new Chess(liveFen), [liveFen]);
  const livePosition = useMemo(() => placement(liveFen), [liveFen]);
  const shownPosition = useMemo(() => placement(shownFen), [shownFen]);
  const plannedPosition = useMemo(() => premoves.reduce(applyPremove, livePosition), [livePosition, premoves]);
  const boardPosition = useMemo(() => {
    if (!isLive) return shownPosition;
    return promotion ? movePiece(plannedPosition, promotion.from, promotion.to) : plannedPosition;
  }, [isLive, shownPosition, plannedPosition, promotion]);

  const myTurn = phase === 'playing' && !autoplay && liveGame.turn() === playerColor && !thinking;
  const activeSelected =
    selected && phase === 'playing' && isLive && plannedPosition[selected]?.pieceType[0] === playerColor
      ? selected
      : null;

  const dests = useMemo<Square[]>(() => {
    if (!activeSelected) return [];
    return myTurn
      ? liveGame.moves({ square: activeSelected, verbose: true }).map((m) => m.to)
      : premoveDests(plannedPosition, activeSelected, playerColor);
  }, [activeSelected, myTurn, liveGame, plannedPosition, playerColor]);

  const squareStyles = useMemo(() => {
    const s: Record<string, CSSProperties> = {};
    const tint = (sq: string, color: string) => (s[sq] = { ...s[sq], backgroundColor: color });
    const last = shown > 0 ? plies[shown - 1] : null;
    if (last) {
      tint(last.from, TINT.last);
      tint(last.to, TINT.last);
      if (last.check) {
        const king = kingSquare(shownPosition, other(last.color));
        if (king) s[king] = { ...s[king], backgroundImage: CHECK };
      }
    }
    if (isLive) {
      premoves.forEach((pm) => {
        tint(pm.from, TINT.premove);
        tint(pm.to, TINT.premove);
      });
      if (activeSelected) {
        tint(activeSelected, TINT.selected);
        if (settings.legalMoves) {
          dests.forEach((d) => (s[d] = { ...s[d], backgroundImage: boardPosition[d] ? RING : DOT }));
        }
      }
    }
    visibleMarks.forEach((sq) => tint(sq, TINT.mark));
    return s;
  }, [shown, plies, shownPosition, isLive, premoves, activeSelected, settings.legalMoves, dests, boardPosition, visibleMarks]);

  // ------------------------------------------------------------ actions

  const goTo = (ply: number) => {
    const clamped = Math.max(0, Math.min(plies.length, ply));
    setView(clamped === plies.length ? null : { ply: clamped, len: plies.length });
    setPromotion(null);
  };

  /** Start a game with the saved setup; `watch` hands both sides to the engine. */
  const begin = (watch = false) => {
    const color: Color = prefs.color === 'random' ? (Math.random() < 0.5 ? 'w' : 'b') : prefs.color;
    setFlipped(false);
    setView(null);
    setSelected(null);
    setMarks(null);
    setPromotion(null);
    setConfirmResign(false);
    clearPreview();
    if (watch) setTab('search');
    ctl.start(color, timeControlById(prefs.tc), watch);
  };

  const toSetup = () => {
    setView(null);
    setSelected(null);
    setPromotion(null);
    ctl.reset();
  };

  const resign = () => {
    if (!plies.some((p) => p.color === playerColor) || confirmResign) {
      setConfirmResign(false);
      ctl.resign();
      return;
    }
    setConfirmResign(true);
    window.setTimeout(() => setConfirmResign(false), 3000);
  };

  const downloadPgn = () => {
    const url = URL.createObjectURL(new Blob([ctl.pgn()], { type: 'application/x-chess-pgn' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `chesstransformer-${new Date().toISOString().slice(0, 10)}.pgn`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyPgn = async () => {
    try {
      await navigator.clipboard.writeText(ctl.pgn());
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      // clipboard blocked; the download button still works
    }
  };

  /** Queue a premove if the piece could ever make it; promotions ask for a piece first. */
  const planPremove = (from: Square, to: Square): boolean => {
    if (!premoveDests(plannedPosition, from, playerColor).includes(to)) return false;
    setSelected(null);
    if (isPromotionSquare(plannedPosition, from, to) && !settings.autoQueen) {
      setPromotion({ from, to, premove: true });
    } else {
      ctl.queuePremove({ from, to });
    }
    return true;
  };

  const tryMove = (from: Square, to: Square): boolean => {
    // Ask the controller rather than this render: a drag started while the
    // engine was thinking can be dropped after its reply, and is then a move.
    if (ctl.isPlayerTurn()) {
      if (ctl.needsPromotion(from, to) && !settings.autoQueen) {
        setSelected(null);
        setPromotion({ from, to, premove: false });
        return true;
      }
      const moved = ctl.playMove(from, to);
      if (moved) setSelected(null);
      return moved;
    }
    return planPremove(from, to);
  };

  const pickPromotion = (piece: PieceSymbol) => {
    if (!promotion) return;
    const { from, to, premove } = promotion;
    setPromotion(null);
    if (premove && !ctl.isPlayerTurn()) ctl.queuePremove({ from, to, promotion: piece });
    else ctl.playMove(from, to, piece);
  };

  const onDrop = (from: string, to: string | null): boolean => {
    lastDrop.current = performance.now();
    if (!to || from === to || !isLive || phase !== 'playing' || promotion || autoplay) return false;
    return tryMove(from as Square, to as Square);
  };
  // react-chessboard keeps the drop handler from when the drag began; route
  // through a ref so a drop always sees the current position and settings.
  const dropRef = useRef(onDrop);
  useEffect(() => {
    dropRef.current = onDrop;
  });

  const onSquareClick = (square: Square) => {
    // a drag that ends on its own square also fires a click; ignore that echo
    if (performance.now() - lastDrop.current < 150) return;
    if (preview) return clearPreview();
    if (!isLive) return goTo(plies.length);
    if (phase !== 'playing' || promotion || autoplay) return;
    if (activeSelected && activeSelected !== square && dests.includes(square)) {
      tryMove(activeSelected, square);
      setSelected(null);
      return;
    }
    const own = plannedPosition[square]?.pieceType[0] === playerColor;
    if (own && activeSelected !== square) return setSelected(square);
    setSelected(null);
    if (!own) ctl.clearPremoves();
  };

  const onSquareRightClick = (square: string) => {
    if (premoves.length || activeSelected) {
      ctl.clearPremoves();
      setSelected(null);
      return;
    }
    const squares = visibleMarks.includes(square)
      ? visibleMarks.filter((s) => s !== square)
      : [...visibleMarks, square];
    setMarks({ squares, len: plies.length, at: shown });
  };

  // ------------------------------------------------------------ keyboard

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.target instanceof HTMLElement && e.target.closest('input, textarea, select, [role="dialog"]')) {
        if (e.key !== 'Escape') return;
      }
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
        case 'f':
          return setFlipped((f) => !f);
        case 'Escape':
          clearPreview();
          setDismissedGame(snap.gameId);
          setPromotion(null);
          setSelected(null);
          ctl.clearPremoves();
          return;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  // ------------------------------------------------------------ view model

  const { takenBy, balance } = useMemo(() => material(shownPosition), [shownPosition]);
  const setupTc = timeControlById(prefs.tc);
  const clockFor = (c: Color): ClockView | null => {
    if (phase === 'setup') return setupTc ? { ms: setupTc.base * 1000, since: 0, running: false } : null;
    if (!clock) return null;
    return { ms: clock[c], since: clock.since, running: clock.running === c && phase === 'playing' };
  };

  const bar = (c: Color) => {
    // Under autoplay the engine is playing the player's side too, and the bar says so.
    const isBot = c === botColor || autoplay;
    return (
      <PlayerBar
        kind={isBot ? 'bot' : 'human'}
        name={isBot ? 'ChessTransformer' : 'You'}
        rating={isBot ? engine?.elo : undefined}
        color={c}
        taken={takenBy[c]}
        advantage={c === 'w' ? balance : -balance}
        clock={clockFor(c)}
        toMove={phase === 'playing' && liveGame.turn() === c}
        thinking={thinking && liveGame.turn() === c}
      />
    );
  };
  const bottomColor: Color = orientation === 'white' ? 'w' : 'b';

  const finalScore = result && !['resignation', 'timeout', 'abort'].includes(result.reason) ? RESULT_SCORE(result) : undefined;
  const evalValue = evalAt(snap.evals, shown);

  const playerMoves = plies.filter((p) => p.color === playerColor).length;
  const status = (() => {
    if (phase === 'over' && result) {
      if (result.reason === 'abort') return 'Game aborted';
      const who =
        result.winner === null
          ? 'Draw'
          : autoplay
            ? `${result.winner === 'w' ? 'White' : 'Black'} won`
            : result.winner === playerColor
              ? 'You won'
              : 'ChessTransformer won';
      return `${who} ${reasonText(result)}`;
    }
    if (snap.engineError) return "The engine didn't answer.";
    if (autoplay) return 'ChessTransformer is playing itself';
    if (thinking) return 'ChessTransformer is thinking…';
    if (phase === 'playing' && liveGame.turn() === playerColor) return liveGame.inCheck() ? 'Your move. You are in check.' : 'Your move';
    return '';
  })();
  const tcLabel = TIME_CONTROLS.find((t) => t.value === snap.timeControl)?.label ?? 'No clock';

  const boardOptions: ChessboardOptions = {
    id: 'ct-board',
    position: preview?.position ?? boardPosition,
    boardOrientation: orientation,
    squareStyles: preview
      ? preview.last
        ? { [preview.last.from]: { backgroundColor: TINT.last }, [preview.last.to]: { backgroundColor: TINT.last } }
        : {}
      : squareStyles,
    arrows: preview?.arrows ?? [],
    allowDrawingArrows: !preview,
    darkSquareStyle: { backgroundColor: BOARD.dark },
    lightSquareStyle: { backgroundColor: BOARD.light },
    darkSquareNotationStyle: { color: BOARD.light, fontWeight: 600 },
    lightSquareNotationStyle: { color: BOARD.dark, fontWeight: 600 },
    alphaNotationStyle: { fontSize: NOTATION_SIZE, position: 'absolute', bottom: 1, right: 3, userSelect: 'none' },
    numericNotationStyle: { fontSize: NOTATION_SIZE, position: 'absolute', top: 2, left: 3, userSelect: 'none' },
    dropSquareStyle: { boxShadow: 'inset 0 0 0 3px rgba(255, 255, 255, 0.7)' },
    draggingPieceStyle: { transform: 'scale(1.12)' },
    draggingPieceGhostStyle: { opacity: 0.3 },
    boardStyle: { borderRadius: 4, boxShadow: '0 6px 24px rgba(0, 0, 0, 0.35)' },
    animationDurationInMs: 170,
    allowDragging: phase === 'playing' && isLive && !preview,
    canDragPiece: ({ piece }) => piece.pieceType[0] === playerColor && !promotion && !autoplay,
    arrowOptions: {
      ...defaultArrowOptions,
      color: '#f0a830',
      secondaryColor: '#3fbf8f',
      tertiaryColor: '#e0574b',
      opacity: 0.82,
      activeOpacity: 0.6,
    },
    onPieceDrag: ({ square }) => square && setSelected(square as Square),
    onPieceDrop: ({ sourceSquare, targetSquare }) => dropRef.current(sourceSquare, targetSquare),
    onSquareClick: ({ square }) => onSquareClick(square as Square),
    // Right-click is read from mousedown/up on the same square rather than the
    // contextmenu event, which Linux fires on mousedown, before an arrow drag.
    onSquareMouseDown: ({ square }, e) => {
      if (e.button === 0 && visibleMarks.length) setMarks(null);
      if (e.button === 2) rightDown.current = square;
    },
    onSquareMouseUp: ({ square }, e) => {
      if (e.button === 2 && rightDown.current === square) onSquareRightClick(square);
      rightDown.current = null;
    },
  };

  const showModal = phase === 'over' && result && dismissedGame !== snap.gameId;

  return (
    <main className="app">
      <div className="play-area">
        <div className="board-grid" data-eval={settings.evalBar ? 'on' : 'off'}>
          <div className="strip-slot lg:hidden">
            <MoveList variant="strip" plies={plies} current={shown} onSelect={goTo} emptyText="Moves appear here" />
          </div>
          <div className="top-slot">{bar(other(bottomColor))}</div>
          {settings.evalBar ? (
            <div className="eval-slot">
              <EvalBar value={evalValue} final={isLive ? finalScore : undefined} orientation={orientation} />
            </div>
          ) : null}
          <div className="board-slot relative" onContextMenu={(e) => e.preventDefault()}>
            <Chessboard options={boardOptions} />
            {promotion ? (
              <PromotionPicker
                square={promotion.to}
                color={playerColor}
                orientation={orientation}
                onPick={pickPromotion}
                onCancel={() => setPromotion(null)}
              />
            ) : null}
            {showModal ? (
              <GameOverModal
                result={result}
                playerColor={playerColor}
                selfPlay={autoplay}
                onRematch={() => begin(autoplay)}
                onNewGame={toSetup}
                onDownload={downloadPgn}
                onClose={() => setDismissedGame(snap.gameId)}
              />
            ) : null}
          </div>
          <div className="bottom-slot">{bar(bottomColor)}</div>
        </div>

        <aside className="panel">
          <header className="flex items-center justify-between border-b border-line py-2.5 pl-4 pr-2">
            <div className="flex items-center gap-2.5">
              <BotAvatar size={26} />
              <span className="font-display text-[19px] font-bold tracking-[-0.01em] text-fg">ChessTransformer</span>
            </div>
            <SettingsMenu settings={settings} onChange={setSettings} />
          </header>

          {phase === 'setup' ? (
            <SetupPanel
              engine={engine}
              offline={offline}
              color={prefs.color}
              onColor={(color) => setPrefs({ ...prefs, color })}
              timeControl={prefs.tc}
              onTimeControl={(tc) => setPrefs({ ...prefs, tc })}
              onPlay={() => begin()}
              onWatch={() => begin(true)}
            />
          ) : (
            <div className="flex min-h-0 flex-1 flex-col">
              <div className="flex items-center justify-between gap-3 px-4 py-2.5 text-sm" aria-live="polite">
                <span className={phase === 'over' ? 'font-semibold text-fg' : snap.engineError ? 'text-coral' : 'text-muted'}>
                  {status}
                </span>
                <span className="shrink-0 text-faint">{tcLabel}</span>
              </div>

              <div role="tablist" aria-label="Game panel" className="flex gap-1 border-t border-line px-2">
                {(['moves', 'search'] as const).map((t) => (
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
                    {t === 'moves' ? 'Moves' : 'Engine search'}
                  </button>
                ))}
              </div>
              <div
                role="tabpanel"
                className="flex min-h-0 flex-1 flex-col border-y border-line bg-black/[0.06] max-lg:h-80 max-lg:flex-none"
                onMouseLeave={() => setHoverPath(null)}
              >
                {tab === 'moves' ? (
                  <MoveList
                    variant="table"
                    plies={plies}
                    current={shown}
                    onSelect={goTo}
                    emptyText={playerColor === 'w' ? 'Make your first move.' : 'ChessTransformer moves first.'}
                  />
                ) : search && searchIndex != null && searchRootFen ? (
                  <SearchView
                    search={search}
                    fen={searchRootFen}
                    played={plies[searchIndex]?.san}
                    active={activePath}
                    onPreview={onPreview}
                  />
                ) : (
                  <p className="px-4 py-3 text-sm leading-snug text-muted">
                    {thinking
                      ? 'ChessTransformer is searching. Its tree appears here once it moves.'
                      : plies.some((p) => p.color === botColor) && !Object.keys(snap.searches).length
                        ? "This backend doesn't report its search. The Rust backend (ct-bot serve) does."
                        : "The engine's search shows up here after its next move."}
                  </p>
                )}
              </div>

              {snap.engineError ? (
                <div className="mx-3 mt-3 flex items-center justify-between gap-3 rounded-lg bg-coral/15 px-3 py-2 text-sm text-fg">
                  Check that the backend is running, then retry.
                  <button type="button" onClick={() => ctl.retry()} className="btn-secondary h-8 px-3 text-sm">
                    Retry
                  </button>
                </div>
              ) : null}
              {snap.notice ? <p className="mx-4 mt-3 text-sm text-muted">{snap.notice}</p> : null}

              <div className="grid grid-cols-4 gap-2 px-3 pt-3">
                {phase === 'playing' ? (
                  <>
                    <PanelButton
                      onClick={() => ctl.offerDraw()}
                      disabled={autoplay || playerMoves === 0 || plies.length - snap.lastDrawOffer < DRAW_OFFER_GAP}
                      icon={<IconDraw />}
                      label="Draw"
                    />
                    <PanelButton
                      onClick={resign}
                      icon={<IconFlag />}
                      label={playerMoves === 0 ? 'Abort' : confirmResign ? 'Confirm' : 'Resign'}
                      danger={confirmResign}
                    />
                    <PanelButton
                      onClick={() => ctl.takeback()}
                      disabled={autoplay || playerMoves === 0}
                      icon={<IconUndo />}
                      label="Takeback"
                    />
                    <PanelButton
                      onClick={() => ctl.setAutoplay(!autoplay)}
                      icon={autoplay ? <IconPause /> : <IconPlay />}
                      label={autoplay ? 'Take over' : 'Autoplay'}
                      primary={autoplay}
                    />
                  </>
                ) : (
                  <>
                    <PanelButton onClick={() => begin(autoplay)} icon={<IconRepeat />} label="Rematch" primary />
                    <PanelButton onClick={toSetup} icon={<IconPlus />} label="New game" />
                    <PanelButton onClick={copyPgn} icon={<IconCopy />} label={copied ? 'Copied' : 'Copy PGN'} />
                    <PanelButton onClick={downloadPgn} icon={<IconDownload />} label="Download" />
                  </>
                )}
              </div>

              <div className="mt-auto flex items-center gap-1 px-3 pb-3 pt-3">
                <NavButton onClick={() => goTo(0)} disabled={shown === 0} label="First move (Home)">
                  <IconFirst />
                </NavButton>
                <NavButton onClick={() => goTo(shown - 1)} disabled={shown === 0} label="Previous move (←)">
                  <IconPrev />
                </NavButton>
                <NavButton onClick={() => goTo(shown + 1)} disabled={isLive} label="Next move (→)">
                  <IconNext />
                </NavButton>
                <NavButton onClick={() => goTo(plies.length)} disabled={isLive} label="Current position (End)">
                  <IconLast />
                </NavButton>
                <NavButton onClick={() => setFlipped((f) => !f)} label="Flip board (F)" narrow>
                  <IconFlip />
                </NavButton>
              </div>
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}

function PanelButton({
  onClick,
  icon,
  label,
  disabled,
  danger,
  primary,
}: {
  onClick: () => void;
  icon: ReactNode;
  label: string;
  disabled?: boolean;
  danger?: boolean;
  primary?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`flex h-14 flex-col items-center justify-center gap-0.5 rounded-lg text-[13px] font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        primary ? 'btn-primary' : danger ? 'bg-coral text-white' : 'bg-raised text-fg hover:bg-line'
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function NavButton({
  onClick,
  disabled,
  label,
  narrow,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  label: string;
  narrow?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      className={`grid h-10 place-items-center rounded-md bg-raised text-fg transition-colors hover:bg-line disabled:cursor-default disabled:text-faint disabled:hover:bg-raised ${
        narrow ? 'ml-1 w-12' : 'flex-1'
      }`}
    >
      {children}
    </button>
  );
}
