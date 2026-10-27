// The game itself, outside React: rules, clocks, the engine round-trip, the
// premove queue and autoplay (the engine playing the player's side too). Async
// work (engine replies, clock flags) mutates this object and publishes an
// immutable snapshot, so the UI never acts on stale state.

import { Chess, type Color, type Move, type PieceSymbol, type Square } from 'chess.js';
import { canMate, detectEnd, other, placement, RESULT_SCORE, type GameResult, type Phase, type Ply, type Premove, type TimeControl } from './chess';
import { playSound, type SoundName } from './sound';

export interface ClockState {
  w: number; // ms left, as of `since`
  b: number;
  running: Color | null;
  since: number;
}

/** One searched move as the engine reports it (UCI, from the searched position). */
export interface SearchNode {
  move: string;
  visits: number;
  prior: number;
  /** Mean value for the side playing `move`, in (-1, 1). */
  q: number;
  children: SearchNode[];
}

/** The engine's account of one search: its budget and its most-visited branches. */
export interface SearchInfo {
  sims: number;
  visits: number;
  /** Unknown for a recorded game's search (the games viewer). */
  elapsedMs?: number;
  /** Plies from the searched position: mean simulation, most-visited line, deepest path. */
  depth?: { mean: number; mainLine: number; max: number };
  tree: SearchNode[];
}

export interface GameSnapshot {
  gameId: number;
  phase: Phase;
  plies: Ply[];
  playerColor: Color;
  timeControl: TimeControl;
  clock: ClockState | null;
  thinking: boolean;
  /** The engine plays the player's side as well. */
  autoplay: boolean;
  engineError: boolean;
  premoves: Premove[];
  result: GameResult | null;
  /** Engine value for White, keyed by the number of plies in the evaluated position. */
  evals: Record<number, number>;
  /** The engine's search, keyed by the number of plies in the searched position. */
  searches: Record<number, SearchInfo>;
  notice: string | null;
  lastDrawOffer: number;
}

const INITIAL: GameSnapshot = {
  gameId: 0,
  phase: 'setup',
  plies: [],
  playerColor: 'w',
  timeControl: null,
  clock: null,
  thinking: false,
  autoplay: false,
  engineError: false,
  premoves: [],
  result: null,
  evals: {},
  searches: {},
  notice: null,
  lastDrawOffer: -Infinity,
};

/** Latest known evaluation at or before ply `index`. */
export function evalAt(evals: Record<number, number>, index: number): number | null {
  let best = -1;
  for (const k of Object.keys(evals)) {
    const n = Number(k);
    if (n <= index && n > best) best = n;
  }
  return best < 0 ? null : evals[best];
}

/** Plies the player must wait between draw offers. */
export const DRAW_OFFER_GAP = 6;

/** Breathing room between autoplay moves in untimed games, so each one can be followed. */
const AUTOPLAY_PAUSE_MS = 450;

export class GameController {
  private game = new Chess();
  private snap: GameSnapshot = INITIAL;
  private listeners = new Set<() => void>();
  private request = { id: 0, ctrl: null as AbortController | null };
  private timers: number[] = [];
  private autoplayTimer = 0;
  private soundEnabled = true;

  constructor(private api: string) {}

  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  };

  getSnapshot = () => this.snap;

  private set(patch: Partial<GameSnapshot>) {
    this.snap = { ...this.snap, ...patch };
    this.listeners.forEach((fn) => fn());
  }

  private sound(name: SoundName) {
    if (this.soundEnabled) playSound(name);
  }

  private get botColor() {
    return other(this.snap.playerColor);
  }

  dispose() {
    this.cancelRequest();
    this.clearTimers();
  }

  // ------------------------------------------------------------ lifecycle

  start(color: Color, timeControl: TimeControl, autoplay = false) {
    this.cancelRequest();
    this.clearTimers();
    this.game = new Chess();
    const gameId = this.snap.gameId + 1;
    const ms = timeControl ? timeControl.base * 1000 : 0;
    this.set({
      ...INITIAL,
      gameId,
      phase: 'playing',
      playerColor: color,
      timeControl,
      autoplay,
      clock: timeControl ? { w: ms, b: ms, running: null, since: Date.now() } : null,
    });
    this.sound('start');

    // Reset the server's session (search tree, repetition history) before its first move.
    const id = ++this.request.id;
    fetch(`${this.api}/api/new-game`, { method: 'POST' })
      .catch(() => undefined)
      .finally(() => {
        if (id !== this.request.id || this.snap.gameId !== gameId) return;
        if (color === 'b' || this.snap.autoplay) this.requestEngineMove();
      });
  }

  /** Leave the game for the setup screen. */
  reset() {
    this.cancelRequest();
    this.clearTimers();
    this.game = new Chess();
    this.set({ ...INITIAL, gameId: this.snap.gameId + 1 });
  }

  setSoundEnabled(on: boolean) {
    this.soundEnabled = on;
  }

  /**
   * Hand the player's side to the engine, or take it back. Taking over while the
   * engine is searching for the player's side cancels that search.
   */
  setAutoplay(on: boolean) {
    const { phase, autoplay, thinking, playerColor } = this.snap;
    if (phase !== 'playing' || autoplay === on) return;
    const playersTurn = this.game.turn() === playerColor;
    this.set({ autoplay: on, premoves: [] });
    if (on && playersTurn && !thinking) this.requestEngineMove();
    if (!on && playersTurn && thinking) this.cancelRequest();
    // Taken over during the pause before the engine's own move: that move still comes.
    if (!on && !playersTurn && !thinking) this.requestEngineMove();
  }

  resign() {
    if (this.snap.phase !== 'playing') return;
    if (!this.snap.plies.some((p) => p.color === this.snap.playerColor)) {
      this.finish({ winner: null, reason: 'abort' });
    } else {
      this.finish({ winner: this.botColor, reason: 'resignation' });
    }
  }

  offerDraw() {
    const { phase, plies, evals, lastDrawOffer } = this.snap;
    if (phase !== 'playing' || plies.length - lastDrawOffer < DRAW_OFFER_GAP) return;
    // The engine takes a draw when it judges itself worse, or when a long game has gone dead level.
    const white = evalAt(evals, plies.length);
    const mine = white == null ? null : this.botColor === 'w' ? white : -white;
    const accept = mine != null && (mine <= -0.3 || (plies.length >= 60 && Math.abs(mine) <= 0.08));
    if (accept) {
      this.finish({ winner: null, reason: 'agreement' });
    } else {
      this.set({ notice: 'ChessTransformer declined your draw offer.', lastDrawOffer: plies.length });
      this.sound('notify');
    }
  }

  /** Undo back to the player's previous turn: their last move and the engine's reply, if any. */
  takeback() {
    const { phase, plies, playerColor, clock, evals } = this.snap;
    if (phase !== 'playing' || this.snap.autoplay || !plies.some((p) => p.color === playerColor)) return;
    this.cancelRequest();
    const n = this.game.turn() === playerColor ? 2 : 1;
    for (let i = 0; i < n; i++) this.game.undo();
    const kept = plies.slice(0, plies.length - n);

    let nextClock = clock;
    if (clock) {
      const t = Date.now();
      nextClock = { ...clock, since: t, running: kept.length >= 2 ? playerColor : null };
      if (clock.running) nextClock[clock.running] = Math.max(0, clock[clock.running] - (t - clock.since));
    }
    const keep = <T>(byPly: Record<number, T>) =>
      Object.fromEntries(Object.entries(byPly).filter(([k]) => Number(k) <= kept.length));
    this.set({
      plies: kept,
      clock: nextClock,
      evals: keep(evals),
      searches: keep(this.snap.searches),
      thinking: false,
      engineError: false,
      premoves: [],
      notice: null,
    });
    this.sound('move');
    this.scheduleClock();
  }

  retry() {
    const { phase, thinking, autoplay } = this.snap;
    if (phase === 'playing' && !thinking && (autoplay || this.game.turn() === this.botColor)) this.requestEngineMove();
  }

  pgn(): string {
    const { playerColor, result, timeControl } = this.snap;
    const d = new Date();
    const pad = (n: number) => String(n).padStart(2, '0');
    const g = this.game;
    g.setHeader('Event', 'Casual game');
    g.setHeader('Site', 'ChessTransformer demo');
    g.setHeader('Date', `${d.getFullYear()}.${pad(d.getMonth() + 1)}.${pad(d.getDate())}`);
    g.setHeader('White', playerColor === 'w' ? 'You' : 'ChessTransformer');
    g.setHeader('Black', playerColor === 'b' ? 'You' : 'ChessTransformer');
    g.setHeader('TimeControl', timeControl ? `${timeControl.base}+${timeControl.inc}` : '-');
    g.setHeader('Result', RESULT_SCORE(result));
    if (result) g.setHeader('Termination', result.reason);
    return g.pgn();
  }

  // ------------------------------------------------------------ moves

  isPlayerTurn() {
    const { phase, playerColor, thinking, autoplay } = this.snap;
    return phase === 'playing' && !autoplay && this.game.turn() === playerColor && !thinking;
  }

  needsPromotion(from: Square, to: Square) {
    return this.game.moves({ square: from, verbose: true }).some((m) => m.to === to && m.isPromotion());
  }

  playMove(from: Square, to: Square, promotion?: PieceSymbol): boolean {
    if (!this.isPlayerTurn()) return false;
    const legal = this.game.moves({ square: from, verbose: true }).find((m) => m.to === to);
    if (!legal) return false;
    const move = this.game.move({ from, to, promotion: legal.isPromotion() ? (promotion ?? 'q') : undefined });
    if (!this.commit(move, true)) this.requestEngineMove();
    return true;
  }

  queuePremove(pm: Premove) {
    if (this.snap.phase !== 'playing' || this.snap.autoplay || this.isPlayerTurn()) return;
    this.set({ premoves: [...this.snap.premoves, pm] });
    this.sound('premove');
  }

  clearPremoves() {
    if (this.snap.premoves.length) this.set({ premoves: [] });
  }

  private commit(move: Move, byPlayer: boolean): boolean {
    const g = this.game;
    const ply: Ply = {
      san: move.san,
      from: move.from,
      to: move.to,
      color: move.color,
      piece: move.piece,
      captured: move.captured,
      promotion: move.promotion,
      castle: move.isKingsideCastle() || move.isQueensideCastle(),
      check: g.inCheck(),
      fen: g.fen(),
    };
    const plies = [...this.snap.plies, ply];

    // Clocks start once both sides have moved; increment is earned on moves made against the clock.
    let clock = this.snap.clock;
    const tc = this.snap.timeControl;
    if (clock && tc) {
      const t = Date.now();
      const next: ClockState = { ...clock, since: t, running: plies.length >= 2 ? other(move.color) : null };
      if (clock.running === move.color) {
        next[move.color] = Math.max(0, clock[move.color] - (t - clock.since)) + tc.inc * 1000;
      }
      clock = next;
    }
    this.set({ plies, clock, notice: null, engineError: false });
    this.sound(
      ply.check ? 'check' : ply.promotion ? 'promote' : ply.castle ? 'castle' : ply.captured ? 'capture' : byPlayer ? 'move' : 'opponent',
    );

    const end = detectEnd(g);
    if (end) {
      this.finish(end);
      return true;
    }
    this.scheduleClock();
    return false;
  }

  private async requestEngineMove() {
    const fen = this.game.fen();
    const index = this.snap.plies.length;
    const gameId = this.snap.gameId;
    this.request.ctrl?.abort();
    const ctrl = new AbortController();
    const id = ++this.request.id;
    this.request.ctrl = ctrl;
    this.set({ thinking: true, engineError: false });

    try {
      const res = await fetch(`${this.api}/api/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen }),
        signal: ctrl.signal,
      });
      if (!res.ok) throw new Error(`engine answered ${res.status}`);
      const data: {
        move: string;
        value?: number;
        search?: {
          sims: number;
          visits: number;
          elapsed_ms: number;
          depth?: { mean: number; max: number; main_line: number };
          tree: SearchNode[];
        };
      } = await res.json();
      if (id !== this.request.id || this.snap.gameId !== gameId || this.snap.phase !== 'playing') return;
      this.request.ctrl = null;

      const uci = data.move;
      const move = this.game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
      const evals =
        typeof data.value === 'number'
          ? { ...this.snap.evals, [index]: move.color === 'w' ? data.value : -data.value }
          : this.snap.evals;
      // Only the Rust backend reports its search tree; the Python one sends none.
      const searches = data.search
        ? {
            ...this.snap.searches,
            [index]: {
              sims: data.search.sims,
              visits: data.search.visits,
              elapsedMs: data.search.elapsed_ms,
              depth: data.search.depth && {
                mean: data.search.depth.mean,
                mainLine: data.search.depth.main_line,
                max: data.search.depth.max,
              },
              tree: data.search.tree,
            },
          }
        : this.snap.searches;
      this.set({ thinking: false, evals, searches });
      if (this.commit(move, move.color === this.snap.playerColor)) return;
      if (this.snap.autoplay) this.scheduleAutoplay();
      else this.firePremove();
    } catch {
      if (id !== this.request.id) return;
      this.request.ctrl = null;
      this.set({ thinking: false, engineError: true });
    }
  }

  /** Play the first queued premove if it is legal now; an illegal one cancels the whole queue. */
  private firePremove() {
    const [pm, ...rest] = this.snap.premoves;
    if (!pm) return;
    const legal = this.game.moves({ square: pm.from, verbose: true }).some((m) => m.to === pm.to);
    this.set({ premoves: legal ? rest : [] });
    if (legal) this.playMove(pm.from, pm.to, pm.promotion);
  }

  private scheduleAutoplay() {
    window.clearTimeout(this.autoplayTimer);
    const gameId = this.snap.gameId;
    this.autoplayTimer = window.setTimeout(
      () => {
        const { autoplay, phase, thinking } = this.snap;
        if (autoplay && phase === 'playing' && !thinking && this.snap.gameId === gameId) this.requestEngineMove();
      },
      this.snap.timeControl ? 0 : AUTOPLAY_PAUSE_MS,
    );
  }

  private cancelRequest() {
    window.clearTimeout(this.autoplayTimer);
    this.request.id++;
    this.request.ctrl?.abort();
    this.request.ctrl = null;
    if (this.snap.thinking) this.set({ thinking: false });
  }

  // ------------------------------------------------------------ clocks & endings

  private clearTimers() {
    this.timers.forEach((t) => window.clearTimeout(t));
    this.timers = [];
  }

  private scheduleClock() {
    this.clearTimers();
    const { clock, phase, playerColor, timeControl } = this.snap;
    if (!clock?.running || phase !== 'playing') return;
    const side = clock.running;
    const left = clock[side] - (Date.now() - clock.since);
    this.timers.push(window.setTimeout(() => this.flag(side), Math.max(0, left) + 5));
    if (side === playerColor && timeControl && timeControl.base >= 60 && left > 10_000) {
      this.timers.push(window.setTimeout(() => this.sound('lowtime'), left - 10_000));
    }
  }

  private flag(side: Color) {
    const { clock, phase } = this.snap;
    if (!clock || clock.running !== side || phase !== 'playing') return;
    if (clock[side] - (Date.now() - clock.since) > 0) return this.scheduleClock();
    const winner = other(side);
    const pos = placement(this.game.fen());
    this.finish(
      canMate(pos, winner) ? { winner, reason: 'timeout' } : { winner: null, reason: 'timeout vs insufficient material' },
    );
  }

  private finish(result: GameResult) {
    this.cancelRequest();
    this.clearTimers();
    let clock = this.snap.clock;
    if (clock?.running) {
      const t = Date.now();
      clock = { ...clock, [clock.running]: Math.max(0, clock[clock.running] - (t - clock.since)), running: null, since: t };
    }
    this.set({ phase: 'over', result, clock, thinking: false, premoves: [] });
    this.timers.push(window.setTimeout(() => this.sound('end'), result.reason === 'abort' ? 0 : 250));
  }
}
