//! One game between two networks, each side played the way the deployed bot
//! plays a lichess game, with its search driven one wave at a time so that a
//! batched runner (selfplay-core `MatchEngine`) can evaluate the waves of many
//! games together.
//!
//! The move flow is ct-bot's lichess loop (branch lichess-levers, the deployed
//! build), with the clock plan pinned at the per-speed cap -- 81-100 % of live
//! moves are cap-bound:
//!
//! * our turn: `set_position` re-roots the kept tree (ponder hit or reuse),
//!   then a search of `effective_sims(cap, warm visits, total_cap)` sims;
//! * after our move: ponder the position after it for
//!   `ponder_budget(ponder_sims, warm visits, total_cap)` sims, unless the
//!   game is over;
//! * the opponent's move stops our ponder after its current wave.
//!
//! Both sides advance one wave per round: two engines thinking and pondering
//! concurrently on equal hardware.

use crate::encoding::{terminal_value, zobrist};
use crate::search::{
    effective_sims, ponder_budget, EvalOutput, EvalRequest, Job, NoEval, Search, SearchParams, Step,
};
use shakmaty::{CastlingMode, Chess, Color, Move, Position};

/// Search settings of one side (ct-bot flags).
#[derive(Clone, Copy)]
pub struct SideParams {
    /// Per net (0 = A), so each network can play with its own search settings.
    pub c_puct: [f64; 2],
    pub fpu: [f64; 2],
    pub prior_temp: [f32; 2],
    pub sim_batch: usize,
    pub tt_cap: usize,
}

impl SideParams {
    fn search_params(&self, s: usize) -> SearchParams {
        SearchParams {
            c_puct: self.c_puct[s],
            fpu: self.fpu[s],
            prior_temp: self.prior_temp[s],
            sim_batch: self.sim_batch,
            tt_cap: self.tt_cap,
            legal_in_requests: true,
            ..SearchParams::default()
        }
    }
}

/// The budget rules both sides play under.
#[derive(Clone, Copy)]
pub struct MatchRules {
    /// Per-move fresh-sims cap (ct-bot `--max-sims`).
    pub sims: u32,
    /// Ponder budget (ct-bot `--ponder-max-sims`); 0 = no pondering.
    pub ponder_sims: u32,
    /// Total root-visit ceiling: `sims` + PONDER_HEADROOM.
    pub total_cap: u32,
    /// Draw after this many plies from the start position (0 = none).
    pub max_plies: u32,
}

struct Side {
    search: Search<NoEval>,
    job: Option<Job>,
    pondering: bool,
    /// The requests of the eval this side has out.
    out: Option<Vec<EvalRequest>>,
    /// The opponent has moved: stop pondering and search our move.
    must_move: bool,
}

pub struct MatchGame {
    pub id: usize,
    pub a_white: bool,
    start: Chess,
    moves: Vec<Move>,
    moves_uci: Vec<String>,
    pos: Chess,
    history: Vec<u64>,
    sides: [Side; 2], // 0 = network A, 1 = network B
    end: Option<(f32, &'static str)>,
}

impl MatchGame {
    pub fn new(id: usize, start: Chess, a_white: bool, params: SideParams) -> Self {
        let side = |s: usize| Side {
            search: Search::new(NoEval, params.search_params(s)),
            job: None,
            pondering: false,
            out: None,
            must_move: false,
        };
        let mut g = MatchGame {
            id,
            a_white,
            history: vec![zobrist(&start)],
            pos: start.clone(),
            start,
            moves: Vec::new(),
            moves_uci: Vec::new(),
            sides: [side(0), side(1)],
            end: None,
        };
        g.check_end(0);
        if g.end.is_none() {
            let s = g.to_move();
            g.sides[s].must_move = true;
        }
        g
    }

    /// (z from white's POV, reason) once the game is over.
    pub fn end(&self) -> Option<(f32, &'static str)> {
        self.end
    }

    pub fn moves_uci(&self) -> &[String] {
        &self.moves_uci
    }

    /// The eval side `s` (0 = A) is waiting for, if any.
    pub fn pending(&self, s: usize) -> Option<&[EvalRequest]> {
        self.sides[s].out.as_deref()
    }

    /// Outputs for side `s`'s pending eval, in its order.
    pub fn resolve(&mut self, s: usize, outputs: Vec<EvalOutput>) {
        let side = &mut self.sides[s];
        side.out = None;
        side.search.resolve(side.job.as_mut().expect("eval out without a job"), outputs);
    }

    fn to_move(&self) -> usize {
        if (self.pos.turn() == Color::White) == self.a_white { 0 } else { 1 }
    }

    fn check_end(&mut self, max_plies: u32) {
        let key = *self.history.last().unwrap();
        if let Some(t) = terminal_value(&self.pos, &self.history, key) {
            // Mate scores -1 for the side to move; the side that just moved wins.
            let z = if t < 0.0 {
                if self.pos.turn() == Color::Black { 1.0 } else { -1.0 }
            } else {
                0.0
            };
            self.end = Some((z, "terminal"));
        } else if max_plies > 0 && self.moves.len() as u32 >= max_plies {
            self.end = Some((0.0, "max_plies"));
        }
    }

    /// Step both sides until each waits on an eval or idles, or the game ends.
    /// Within a pass the side searching its move goes first and the pondering
    /// side second, so a ponder sees the opponent's move as soon as it is
    /// played -- whichever network is which.
    pub fn advance(&mut self, rules: &MatchRules) {
        while self.end.is_none() {
            let mut progressed = false;
            for ponder_phase in [false, true] {
                for s in 0..2 {
                    let side = &self.sides[s];
                    if self.end.is_some() || side.out.is_some() {
                        continue;
                    }
                    if (side.job.is_some() && side.pondering) != ponder_phase {
                        continue;
                    }
                    progressed |= self.step_side(s, rules);
                }
            }
            if !progressed {
                break;
            }
        }
    }

    /// Whether the game is live but neither side has anything to do -- a bug.
    pub fn stalled(&self) -> bool {
        self.end.is_none() && self.sides.iter().all(|s| s.out.is_none())
    }

    /// One step of side `s`. Returns whether anything moved on without an eval.
    fn step_side(&mut self, s: usize, rules: &MatchRules) -> bool {
        let side = &mut self.sides[s];
        let Some(job) = side.job.as_mut() else {
            if side.must_move {
                side.must_move = false;
                self.begin_move(s, rules);
                return true;
            }
            return false;
        };
        if side.pondering && side.must_move {
            job.stop(); // the opponent's move arrived
        }
        match side.search.step(job) {
            Step::Eval(reqs) => {
                side.out = Some(reqs);
                false
            }
            Step::Wave => true,
            Step::Done => {
                let job = side.job.take().unwrap();
                let result = side.search.finish(job);
                if side.pondering {
                    side.pondering = false;
                } else {
                    self.play(s, result.best, rules);
                }
                true
            }
        }
    }

    /// Our turn: re-root onto the game (ponder hit or plain reuse), then search
    /// the effective budget.
    fn begin_move(&mut self, s: usize, rules: &MatchRules) {
        let side = &mut self.sides[s];
        side.search.set_position(self.start.clone(), &self.moves);
        let sims = if rules.ponder_sims > 0 {
            effective_sims(rules.sims, side.search.root_visits(), rules.total_cap)
        } else {
            rules.sims
        };
        side.job = Some(side.search.start(sims));
        side.pondering = false;
    }

    fn play(&mut self, s: usize, best: Move, rules: &MatchRules) {
        self.moves_uci.push(best.to_uci(CastlingMode::Standard).to_string());
        self.pos.play_unchecked(&best);
        self.history.push(zobrist(&self.pos));
        self.moves.push(best);
        self.check_end(rules.max_plies);
        if self.end.is_some() {
            return;
        }
        if rules.ponder_sims > 0 && !self.pos.is_game_over() {
            let side = &mut self.sides[s];
            side.search.set_position(self.start.clone(), &self.moves);
            let budget = ponder_budget(rules.ponder_sims, side.search.root_visits(), rules.total_cap);
            if budget > 0 {
                side.job = Some(side.search.start(budget));
                side.pondering = true;
            }
        }
        self.sides[1 - s].must_move = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::NUM_PLANES;
    use crate::search::Evaluator;
    use shakmaty::fen::Fen;
    use std::sync::atomic::AtomicBool;

    /// A deterministic stand-in network; `salt` makes A and B different players.
    struct HashEval(f64);

    impl Evaluator for HashEval {
        fn eval_batch(&mut self, batch: &[EvalRequest]) -> anyhow::Result<Vec<EvalOutput>> {
            Ok(batch.iter().map(|r| hash_out(self.0, r)).collect())
        }
    }

    fn hash_out(salt: f64, r: &EvalRequest) -> EvalOutput {
        let mut h = salt + r.player as f64 * 0.7 + r.castling as f64 * 0.3 + r.ep as f64 * 0.11;
        for (i, &t) in r.tokens.iter().enumerate() {
            h += (t as f64) * ((i as f64) * 0.618).sin();
        }
        let logits = (0..64 * NUM_PLANES)
            .map(|k| ((h * 0.37 + k as f64 * 0.0131).sin() * 2.5) as f32)
            .collect();
        EvalOutput { logits, value: ((h * 0.173).sin() * 0.9) as f32 }
    }

    const SALTS: [f64; 2] = [0.0, 1.7];

    #[test]
    fn each_net_searches_with_its_own_settings() {
        let p = SideParams {
            c_puct: [1.0, 2.5], fpu: [0.2, 0.05], prior_temp: [1.0, 1.4], sim_batch: 16,
            tt_cap: 1000,
        };
        let (a, b) = (p.search_params(0), p.search_params(1));
        assert_eq!((a.c_puct, a.fpu, a.prior_temp), (1.0, 0.2, 1.0));
        assert_eq!((b.c_puct, b.fpu, b.prior_temp), (2.5, 0.05, 1.4));
    }

    fn pos(fen: &str) -> Chess {
        fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
    }

    fn params(sim_batch: usize) -> SideParams {
        SideParams {
            c_puct: [1.0; 2], fpu: [0.2; 2], prior_temp: [1.0; 2], sim_batch, tt_cap: 2_000_000,
        }
    }

    /// The batched driver, one game.
    fn batched(start: &Chess, a_white: bool, rules: MatchRules, sim_batch: usize) -> Vec<String> {
        let mut g = MatchGame::new(0, start.clone(), a_white, params(sim_batch));
        while g.end().is_none() {
            g.advance(&rules);
            assert!(g.end().is_some() || !g.stalled(), "stalled");
            for s in 0..2 {
                if let Some(reqs) = g.pending(s) {
                    let outs = reqs.iter().map(|r| hash_out(SALTS[s], r)).collect();
                    g.resolve(s, outs);
                }
            }
        }
        g.moves_uci().to_vec()
    }

    /// The bot's own flow, sequentially through `go()`: our search, then our
    /// ponder to completion, then the opponent's turn.
    fn sequential(start: &Chess, a_white: bool, rules: MatchRules, sim_batch: usize) -> Vec<String> {
        let mk = |s: usize| Search::new(HashEval(SALTS[s]), params(sim_batch).search_params(s));
        let mut sides = [mk(0), mk(1)];
        let mut moves: Vec<Move> = Vec::new();
        let mut out = Vec::new();
        let mut p = start.clone();
        let mut history = vec![zobrist(&p)];
        let stop = AtomicBool::new(false);
        loop {
            let s = if (p.turn() == Color::White) == a_white { 0 } else { 1 };
            let search = &mut sides[s];
            search.set_position(start.clone(), &moves);
            let sims = effective_sims(rules.sims, search.root_visits(), rules.total_cap);
            let best = search.go(sims, None, &stop).unwrap().best;
            out.push(best.to_uci(CastlingMode::Standard).to_string());
            p.play_unchecked(&best);
            history.push(zobrist(&p));
            moves.push(best);
            let key = *history.last().unwrap();
            if terminal_value(&p, &history, key).is_some() || moves.len() as u32 >= rules.max_plies {
                return out;
            }
            if !p.is_game_over() {
                search.set_position(start.clone(), &moves);
                let budget = ponder_budget(rules.ponder_sims, search.root_visits(), rules.total_cap);
                if budget > 0 {
                    search.go(budget, None, &stop).unwrap();
                }
            }
        }
    }

    #[test]
    fn batched_driver_plays_the_bots_own_moves() {
        // A ponder (<= 24 sims, <= 7 waves) always ends before the opponent's search
        // (96 sims, >= 7 waves of 16 ... with reuse clipping never below 64): no ponder is
        // stopped early, so the sequential flow is the exact reference.
        let rules = MatchRules { sims: 96, ponder_sims: 24, total_cap: 96 + 4000, max_plies: 40 };
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            "6k1/5ppp/R7/Q7/3q4/6PP/7K/8 b - - 0 1",
        ] {
            for a_white in [true, false] {
                let start = pos(fen);
                let b = batched(&start, a_white, rules, 4);
                let r = sequential(&start, a_white, rules, 4);
                assert!(b.len() > 4, "{fen}: game too short to test anything");
                assert_eq!(b, r, "{fen} a_white={a_white}");
            }
        }
    }

    #[test]
    fn early_ponder_stop_still_plays_legal_games() {
        // Ponder budget above the move budget: ponders get stopped mid-way.
        let rules = MatchRules { sims: 64, ponder_sims: 400, total_cap: 64 + 4000, max_plies: 30 };
        let start = pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let moves = batched(&start, true, rules, 16);
        let mut p = start;
        for u in &moves {
            let m = u.parse::<shakmaty::uci::UciMove>().unwrap().to_move(&p).unwrap();
            p.play_unchecked(&m);
        }
        assert_eq!(moves.len(), 30);
    }
}
