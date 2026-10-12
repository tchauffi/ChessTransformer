//! One self-play game on the deployed bot's search (`search::Search`), driven
//! one wave at a time so that a batched runner (selfplay-core
//! `SelfPlayEngine`) can evaluate the waves of many games together.
//!
//! The search is the bot's: virtual-loss waves of `sim_batch` leaves with TT
//! hits deferred to the end of the wave, the two-fold rule inside the tree,
//! and the tree kept and re-rooted from move to move (one tree per game: both
//! sides are the same network). On top of it, the expert-iteration options:
//!
//! * playout cap randomization (KataGo): each move is a full search of
//!   `num_sims` with probability `full_frac`, else `fast_sims`. Only full
//!   searches are policy targets (the `full` flag). A budget counts fresh sims,
//!   or, with `cap_total_full` / `cap_total_fast`, the root's total visits:
//!   the reused tree's warm visits count toward it and only the rest is
//!   searched (none when the tree already holds the budget);
//! * root Dirichlet noise on full searches (`search::RootNoise`);
//! * move temperature for the first `temp_plies` half-moves (the bot's own
//!   sampler, seeded);
//! * resignation once the root value stays past `resign_threshold` for
//!   `resign_plies` plies, except in a `no_resign_frac` share of games.
//!
//! A record is the position before the move, the search's root value and the
//! root visit counts -- warm visits from the reused tree included, as Lc0
//! trains on them.

use rand::prelude::*;
use rand::rngs::StdRng;
use shakmaty::{CastlingMode, Chess, Color, Move, Position};

use crate::encoding::{action_plane, encode_position, halfmove, move_coords, rule_state, terminal_value, zobrist,
                      NUM_PLANES};
use crate::search::{EvalOutput, EvalRequest, Job, NoEval, RootNoise, Search, SearchParams, Step};

#[derive(Clone, Copy)]
pub struct SelfPlayRules {
    pub num_sims: u32,
    pub fast_sims: u32,
    pub full_frac: f64,
    pub dirichlet_alpha: f64,
    pub dirichlet_eps: f64,
    pub move_temp: f64,
    pub temp_plies: u32,
    pub resign_threshold: f32,
    pub resign_plies: u32,
    pub no_resign_frac: f64,
    pub max_plies: u32,
    pub sim_batch: usize,
    pub c_puct: f64,
    pub fpu: f64,
    pub prior_temp: f32,
    pub tt_cap: usize,
    /// The full / fast budget caps the root's total visits, warm ones included,
    /// instead of counting fresh sims.
    pub cap_total_full: bool,
    pub cap_total_fast: bool,
    /// The network reads the rule state (`SearchParams::rule_state`).
    pub rule_state: bool,
}

pub struct Record {
    pub tokens: [u8; 64],
    pub player: u8,
    pub castling: u8,
    pub ep: u8,
    pub halfmove: u16,
    /// The root's rule state (`encoding::rule_state`) over the game so far.
    pub rule50: u8,
    pub repetition: u8,
    pub root_v: f32,
    pub full: u8,
    pub visit_idx: Vec<u16>, // from_sq * 73 + plane
    pub visit_cnt: Vec<u16>,
}

pub struct SelfPlayGame {
    pub id: usize,
    pub opening: usize,
    start: Chess,
    moves: Vec<Move>,
    moves_uci: Vec<String>,
    pos: Chess,
    history: Vec<u64>,
    search: Search<NoEval>,
    job: Option<Job>,
    full: bool,
    out: Option<Vec<EvalRequest>>,
    rng: StdRng,
    resign_allowed: bool,
    resign_streak: u32,
    resign_sign: i32,
    records: Vec<Record>,
    end: Option<(f32, &'static str)>,
}

impl SelfPlayGame {
    /// All of the game's randomness (budgets, noise, temperature, resignation)
    /// derives from `seed`.
    pub fn new(id: usize, opening: usize, start: Chess, rules: &SelfPlayRules, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let params = SearchParams {
            c_puct: rules.c_puct,
            fpu: rules.fpu,
            prior_temp: rules.prior_temp,
            sim_batch: rules.sim_batch,
            move_temp: rules.move_temp,
            move_temp_plies: rules.temp_plies,
            tt_cap: rules.tt_cap,
            legal_in_requests: true,
            rule_state: rules.rule_state,
            ..SearchParams::default()
        };
        let mut search = Search::new(NoEval, params);
        search.seed_rng(rng.gen());
        let resign_allowed = rng.gen::<f64>() >= rules.no_resign_frac;
        let mut g = SelfPlayGame {
            id,
            opening,
            history: vec![zobrist(&start)],
            pos: start.clone(),
            start,
            moves: Vec::new(),
            moves_uci: Vec::new(),
            search,
            job: None,
            full: true,
            out: None,
            rng,
            resign_allowed,
            resign_streak: 0,
            resign_sign: 0,
            records: Vec::new(),
            end: None,
        };
        g.check_end(rules);
        g
    }

    /// (z from white's POV, reason) once the game is over.
    pub fn end(&self) -> Option<(f32, &'static str)> {
        self.end
    }

    pub fn moves_uci(&self) -> &[String] {
        &self.moves_uci
    }

    pub fn records(&self) -> &[Record] {
        &self.records
    }

    pub fn resign_allowed(&self) -> bool {
        self.resign_allowed
    }

    pub fn tt_len(&self) -> usize {
        self.search.tt_len()
    }

    pub fn pending(&self) -> Option<&[EvalRequest]> {
        self.out.as_deref()
    }

    pub fn resolve(&mut self, outputs: Vec<EvalOutput>) {
        self.out = None;
        self.search.resolve(self.job.as_mut().expect("eval out without a job"), outputs);
    }

    /// Whether the game is live but has nothing to do -- a bug.
    pub fn stalled(&self) -> bool {
        self.end.is_none() && self.out.is_none()
    }

    /// (budget, full) for the next move.
    fn draw_budget(&mut self, rules: &SelfPlayRules) -> (u32, bool) {
        if rules.fast_sims == 0 || self.rng.gen::<f64>() < rules.full_frac {
            (rules.num_sims, true)
        } else {
            (rules.fast_sims, false)
        }
    }

    /// Step the search until it waits on an eval or the game ends.
    pub fn advance(&mut self, rules: &SelfPlayRules) {
        while self.end.is_none() && self.out.is_none() {
            if self.job.is_none() {
                self.search.set_position(self.start.clone(), &self.moves);
                let (budget, full) = self.draw_budget(rules);
                let cap_total = if full { rules.cap_total_full } else { rules.cap_total_fast };
                let sims = if cap_total { budget.saturating_sub(self.search.root_visits()) } else { budget };
                let noise = (full && rules.dirichlet_alpha > 0.0).then(|| RootNoise {
                    alpha: rules.dirichlet_alpha,
                    eps: rules.dirichlet_eps,
                    seed: self.rng.gen(),
                });
                self.full = full;
                self.job = Some(self.search.start_with_noise(sims, noise));
            }
            match self.search.step(self.job.as_mut().unwrap()) {
                Step::Eval(reqs) => self.out = Some(reqs),
                Step::Wave => {}
                Step::Done => {
                    let job = self.job.take().unwrap();
                    let result = self.search.finish(job);
                    self.record(result.value);
                    self.play(result.best, result.value, rules);
                }
            }
        }
    }

    fn record(&mut self, root_v: f32) {
        let (tokens, player, castling, ep) = encode_position(&self.pos);
        let (key, prior) = self.history.split_last().expect("history holds the current position");
        let (rule50, repetition) = rule_state(&self.pos, prior, *key);
        let (mut visit_idx, mut visit_cnt) = (Vec::new(), Vec::new());
        for (mv, n, _, _) in self.search.root_stats() {
            if n > 0 {
                let (f, t, p) = move_coords(&mv);
                visit_idx.push((f * NUM_PLANES + action_plane(f, t, p)) as u16);
                visit_cnt.push(n.min(u16::MAX as u32) as u16);
            }
        }
        self.records.push(Record {
            tokens,
            player,
            castling,
            ep,
            halfmove: halfmove(&self.pos).min(u16::MAX as u32) as u16,
            rule50,
            repetition,
            root_v,
            full: self.full as u8,
            visit_idx,
            visit_cnt,
        });
    }

    fn play(&mut self, best: Move, root_v: f32, rules: &SelfPlayRules) {
        let white_to_move = self.pos.turn() == Color::White;
        self.moves_uci.push(best.to_uci(CastlingMode::Standard).to_string());
        self.pos.play_unchecked(&best);
        self.history.push(zobrist(&self.pos));
        self.moves.push(best);

        // Resign bookkeeping on the white-POV root value.
        let v_white = if white_to_move { root_v } else { -root_v };
        if v_white.abs() >= rules.resign_threshold {
            let sign = if v_white > 0.0 { 1 } else { -1 };
            self.resign_streak = if sign == self.resign_sign { self.resign_streak + 1 } else { 1 };
            self.resign_sign = sign;
        } else {
            self.resign_streak = 0;
        }
        self.check_end(rules);
    }

    fn check_end(&mut self, rules: &SelfPlayRules) {
        let key = *self.history.last().unwrap();
        self.end = if let Some(t) = terminal_value(&self.pos, &self.history, key) {
            // Mate scores -1 for the side to move; the side that just moved wins.
            let z = if t < 0.0 {
                if self.pos.turn() == Color::Black { 1.0 } else { -1.0 }
            } else {
                0.0
            };
            Some((z, "terminal"))
        } else if self.resign_allowed && self.resign_streak >= rules.resign_plies {
            Some((self.resign_sign as f32, "resign"))
        } else if self.moves.len() as u32 >= rules.max_plies {
            Some((0.0, "max_plies"))
        } else {
            None
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::Evaluator;
    use shakmaty::fen::Fen;
    use std::sync::atomic::AtomicBool;

    struct HashEval;

    impl Evaluator for HashEval {
        fn eval_batch(&mut self, batch: &[EvalRequest]) -> anyhow::Result<Vec<EvalOutput>> {
            Ok(batch.iter().map(hash_out).collect())
        }
    }

    fn hash_out(r: &EvalRequest) -> EvalOutput {
        let mut h = r.player as f64 * 0.7 + r.castling as f64 * 0.3 + r.ep as f64 * 0.11;
        for (i, &t) in r.tokens.iter().enumerate() {
            h += (t as f64) * ((i as f64) * 0.618).sin();
        }
        let logits = (0..64 * NUM_PLANES)
            .map(|k| ((h * 0.37 + k as f64 * 0.0131).sin() * 2.5) as f32)
            .collect();
        EvalOutput { logits, value: ((h * 0.173).sin() * 0.9) as f32 }
    }

    fn rules() -> SelfPlayRules {
        SelfPlayRules {
            num_sims: 64, fast_sims: 16, full_frac: 0.3, dirichlet_alpha: 0.0, dirichlet_eps: 0.25,
            move_temp: 1.0, temp_plies: 10, resign_threshold: 2.0, resign_plies: 6,
            no_resign_frac: 0.1, max_plies: 40, sim_batch: 16, c_puct: 1.0, fpu: 0.2,
            prior_temp: 1.0, tt_cap: 2_000_000, cap_total_full: false, cap_total_fast: false,
            rule_state: false,
        }
    }

    fn start() -> Chess {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            .parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
    }

    fn batched(rules: &SelfPlayRules, seed: u64) -> SelfPlayGame {
        let mut g = SelfPlayGame::new(0, 0, start(), rules, seed);
        while g.end().is_none() {
            g.advance(rules);
            if let Some(reqs) = g.pending() {
                let outs = reqs.iter().map(hash_out).collect();
                g.resolve(outs);
            }
        }
        g
    }

    /// The bot's own flow through go() with the same draws as `SelfPlayGame`
    /// (search rng, resign flag, then one budget draw per move), tree reuse
    /// through set_position, temperature through the bot's own sampler:
    /// (moves, full flags, warm root visits before each search).
    fn reference(r: &SelfPlayRules, seed: u64, plies: usize) -> (Vec<String>, Vec<u8>, Vec<u32>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let params = SearchParams { sim_batch: r.sim_batch, move_temp: r.move_temp,
                                    move_temp_plies: r.temp_plies, ..SearchParams::default() };
        let mut s = Search::new(HashEval, params);
        s.seed_rng(rng.gen());
        let _resign_allowed = rng.gen::<f64>() >= r.no_resign_frac;
        let (mut moves, mut want, mut fulls, mut warms) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..plies {
            s.set_position(start(), &moves);
            let full = rng.gen::<f64>() < r.full_frac;
            let budget = if full { r.num_sims } else { r.fast_sims };
            let warm = s.root_visits();
            let capped = if full { r.cap_total_full } else { r.cap_total_fast };
            let sims = if capped { budget.saturating_sub(warm) } else { budget };
            let res = s.go(sims, None, &AtomicBool::new(false)).unwrap();
            want.push(res.best.to_uci(CastlingMode::Standard).to_string());
            fulls.push(full as u8);
            warms.push(warm);
            moves.push(res.best);
        }
        (want, fulls, warms)
    }

    #[test]
    fn batched_game_is_the_bots_search_move_for_move() {
        for (cap_total_full, cap_total_fast) in [(false, false), (false, true), (true, true)] {
            let r = SelfPlayRules { cap_total_full, cap_total_fast, ..rules() };
            let g = batched(&r, 11);
            let (want, fulls, _) = reference(&r, 11, g.moves_uci().len());
            assert!(want.len() > 10);
            assert_eq!(g.moves_uci(), &want[..], "caps {cap_total_full} {cap_total_fast}");
            assert_eq!(g.records().iter().map(|r| r.full).collect::<Vec<_>>(), fulls);
        }
    }

    #[test]
    fn capped_budgets_count_the_warm_visits() {
        let r = SelfPlayRules { move_temp: 0.0, cap_total_full: true, cap_total_fast: true, ..rules() };
        let g = batched(&r, 3);
        let (_, _, warms) = reference(&r, 3, g.moves_uci().len());
        let (mut topped_up, mut skipped) = (0, 0);
        for (rec, &warm) in g.records().iter().zip(&warms) {
            let budget = if rec.full == 1 { r.num_sims } else { r.fast_sims };
            let n: u32 = rec.visit_cnt.iter().map(|&c| c as u32).sum();
            // Searched up to the budget, or not at all when the reused tree held it.
            assert_eq!(n, budget.max(warm), "warm {warm} budget {budget}");
            if warm >= budget { skipped += 1 } else if warm > 0 { topped_up += 1 }
        }
        assert!(topped_up > 0 && skipped > 0, "topped up {topped_up}, skipped {skipped}");
    }

    #[test]
    fn visits_cover_the_fresh_budget_and_reuse_adds_warm_ones() {
        let r = SelfPlayRules { move_temp: 0.0, ..rules() };
        let g = batched(&r, 3);
        let sums: Vec<u32> = g.records().iter()
            .map(|x| x.visit_cnt.iter().map(|&c| c as u32).sum()).collect();
        let first = &g.records()[0];
        // A fresh tree: exactly the budget (the root expansion is not a visit).
        assert_eq!(sums[0], if first.full == 1 { r.num_sims } else { r.fast_sims });
        for (rec, &n) in g.records().iter().zip(&sums) {
            assert!(n >= if rec.full == 1 { r.num_sims } else { r.fast_sims });
        }
        assert!(g.records().iter().zip(&sums).any(|(rec, &n)| n > if rec.full == 1 { r.num_sims }
                                                                     else { r.fast_sims }),
                "tree reuse never carried a warm visit");
    }

    #[test]
    fn noise_changes_full_searches_only_and_is_seeded() {
        let quiet = SelfPlayRules { move_temp: 0.0, ..rules() };
        let noisy = SelfPlayRules { dirichlet_alpha: 0.3, ..quiet };
        let a = batched(&noisy, 5);
        assert_eq!(a.moves_uci(), batched(&noisy, 5).moves_uci(), "noise must follow the seed");
        let b = batched(&quiet, 5);
        // Same budget draws; up to the first full search the games must agree.
        let first_full = a.records().iter().position(|r| r.full == 1).unwrap();
        assert_eq!(a.moves_uci()[..first_full], b.moves_uci()[..first_full]);
        let differs = (0..a.records().len().min(b.records().len()))
            .any(|i| a.records()[i].visit_cnt != b.records()[i].visit_cnt);
        assert!(differs, "noise left every full search unchanged");
    }
}
