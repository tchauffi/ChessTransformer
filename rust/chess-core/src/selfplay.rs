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
//! * KataGo forced playouts on full searches (`forced_playouts` = k, 0 = off): every
//!   visited root move gets at least `sqrt(k * P * N)` visits, and the record keeps
//!   the visits left after policy target pruning (`tree::prune_forced`), which also
//!   choose the move;
//! * move temperature for the first `temp_plies` half-moves (the bot's own
//!   sampler, seeded), over the moves within `temp_cutoff` of the best Q when set;
//! * resignation once the root value stays past `resign_threshold` for
//!   `resign_plies` plies, except in a `no_resign_frac` share of games.
//!
//! A record is the position before the move, the search's root value and the
//! root visit counts -- warm visits from the reused tree included, as Lc0
//! trains on them -- plus the root's search over every legal move (clean prior,
//! raw visits, Q) and the network's raw value: what a completed-Q (Gumbel) policy
//! target needs, for fast searches as well as full ones.

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
    /// Sample only the moves within this of the most-visited move's Q
    /// (`SearchParams::move_temp_cutoff`; 0 = all). In gen14, T=1 over every move gave
    /// away ~0.3 Q a game before ply 30, and a 0.05 cutoff keeps 2/3 of the variety.
    pub temp_cutoff: f64,
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
    /// KataGo forced playouts + policy target pruning on full searches: k (2 in
    /// KataGo), 0 = off.
    pub forced_playouts: f64,
    /// Full searches start from a fresh tree. A reused subtree's warm visits were
    /// spent at depth 1, without root noise or forced playouts, and were a third of
    /// the policy target on average in gen14.
    pub fresh_full: bool,
    /// The two-fold rule over the game history (`SearchParams::twofold_history`).
    pub twofold_history: bool,
    /// Contempt for rule draws when ahead (`SearchParams::draw_contempt`).
    pub draw_contempt: f64,
    pub contempt_ramp: f64,
    /// Measure the tree's depth after every search (`SearchTally`); a walk of the
    /// whole tree per move, so off outside benchmarks.
    pub depth_stats: bool,
}

/// A game's searches summed, per kind (`SelfPlayGame::tally`): what the self-play
/// benchmark reports per move. Depths come from `Search::depth_stats` (warm visits
/// of the reused tree included) and stay 0 unless `SelfPlayRules::depth_stats`.
#[derive(Clone, Copy, Default)]
pub struct SearchTally {
    pub moves: u64,
    pub sims: u64,
    pub nn_evals: u64,
    pub tt_hits: u64,
    pub depth_mean: f64,
    pub depth_max: u64,
    pub main_line: u64,
}

impl SearchTally {
    pub fn add(&mut self, o: &SearchTally) {
        self.moves += o.moves;
        self.sims += o.sims;
        self.nn_evals += o.nn_evals;
        self.tt_hits += o.tt_hits;
        self.depth_mean += o.depth_mean;
        self.depth_max += o.depth_max;
        self.main_line += o.main_line;
    }
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
    /// Every legal root move, in the search's order: flat index, the network's prior
    /// before root noise, visits before any pruning, and Q from the side to move's
    /// POV (0 when unvisited).
    pub legal_idx: Vec<u16>,
    pub legal_prior: Vec<f32>,
    pub legal_n: Vec<u16>,
    pub legal_q: Vec<f32>,
    /// The network's raw value of the root (NaN once the cache has dropped it).
    pub root_net_v: f32,
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
    tally: [SearchTally; 2],
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
            move_temp_cutoff: rules.temp_cutoff,
            tt_cap: rules.tt_cap,
            legal_in_requests: true,
            rule_state: rules.rule_state,
            twofold_history: rules.twofold_history,
            draw_contempt: rules.draw_contempt,
            contempt_ramp: rules.contempt_ramp,
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
            tally: [SearchTally::default(); 2],
        };
        g.check_end(rules);
        g
    }

    /// The game's searches so far, [fast, full].
    pub fn tally(&self) -> &[SearchTally; 2] {
        &self.tally
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
                // The budget draw only uses the game's rng, so drawing it first keeps every
                // game the same when fresh_full is off.
                let (budget, full) = self.draw_budget(rules);
                if full && rules.fresh_full {
                    self.search.drop_tree();
                }
                self.search.set_position(self.start.clone(), &self.moves);
                let cap_total = if full { rules.cap_total_full } else { rules.cap_total_fast };
                let sims = if cap_total { budget.saturating_sub(self.search.root_visits()) } else { budget };
                let noise = (full && rules.dirichlet_alpha > 0.0).then(|| RootNoise {
                    alpha: rules.dirichlet_alpha,
                    eps: rules.dirichlet_eps,
                    seed: self.rng.gen(),
                });
                self.full = full;
                let mut job = self.search.start_with_noise(sims, noise);
                if full && rules.forced_playouts > 0.0 {
                    job.force_playouts(rules.forced_playouts);
                }
                self.job = Some(job);
            }
            match self.search.step(self.job.as_mut().unwrap()) {
                Step::Eval(reqs) => self.out = Some(reqs),
                Step::Wave => {}
                Step::Done => {
                    let job = self.job.take().unwrap();
                    let pruned = (self.full && rules.forced_playouts > 0.0)
                        .then(|| self.search.pruned_root_visits(rules.forced_playouts));
                    let result = match &pruned {
                        Some(visits) => self.search.finish_with_visits(job, visits),
                        None => self.search.finish(job),
                    };
                    let t = &mut self.tally[self.full as usize];
                    t.moves += 1;
                    t.sims += result.sims as u64;
                    t.nn_evals += result.nn_evals as u64;
                    t.tt_hits += result.tt_hits as u64;
                    if rules.depth_stats {
                        let d = self.search.depth_stats();
                        t.depth_mean += d.mean as f64;
                        t.depth_max += d.max as u64;
                        t.main_line += d.main_line as u64;
                    }
                    self.record(result.value, pruned.as_deref());
                    self.play(result.best, result.value, rules);
                }
            }
        }
    }

    /// `visits`: the root's counts to keep, in `Search::root_moves` order (pruned);
    /// `None`, its own.
    fn record(&mut self, root_v: f32, visits: Option<&[u32]>) {
        let (tokens, player, castling, ep) = encode_position(&self.pos);
        let (key, prior) = self.history.split_last().expect("history holds the current position");
        let (rule50, repetition) = rule_state(&self.pos, prior, *key);
        let (mut visit_idx, mut visit_cnt) = (Vec::new(), Vec::new());
        let counts: Vec<(Move, u32)> = match visits {
            Some(v) => self.search.root_moves().iter().cloned().zip(v.iter().copied()).collect(),
            None => self.search.root_stats().into_iter().map(|(mv, n, _, _)| (mv, n)).collect(),
        };
        for (mv, n) in counts {
            if n > 0 {
                let (f, t, p) = move_coords(&mv);
                visit_idx.push((f * NUM_PLANES + action_plane(f, t, p)) as u16);
                visit_cnt.push(n.min(u16::MAX as u32) as u16);
            }
        }
        let root = self.search.root();
        let (legal_prior, root_net_v) = match self.search.root_eval() {
            Some((priors, v)) if priors.len() == root.moves.len() => (priors, v),
            _ => (root.priors.clone(), f32::NAN),
        };
        let legal_idx = root.moves.iter()
            .map(|m| { let (f, t, p) = move_coords(m); (f * NUM_PLANES + action_plane(f, t, p)) as u16 })
            .collect();
        let legal_n = root.n.iter().map(|&n| n.min(u16::MAX as u32) as u16).collect();
        let legal_q = (0..root.n.len())
            .map(|i| if root.n[i] > 0 { (-root.w[i] / root.n[i] as f64) as f32 } else { 0.0 })
            .collect();
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
            legal_idx,
            legal_prior,
            legal_n,
            legal_q,
            root_net_v,
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
            move_temp: 1.0, temp_plies: 10, temp_cutoff: 0.0, resign_threshold: 2.0, resign_plies: 6,
            no_resign_frac: 0.1, max_plies: 40, sim_batch: 16, c_puct: 1.0, fpu: 0.2,
            prior_temp: 1.0, tt_cap: 2_000_000, cap_total_full: false, cap_total_fast: false,
            rule_state: false, forced_playouts: 0.0, fresh_full: false, twofold_history: false,
            draw_contempt: 0.0, contempt_ramp: 0.3, depth_stats: false,
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
    fn tally_sums_the_searches_by_kind() {
        let off = batched(&rules(), 11);
        let r = SelfPlayRules { depth_stats: true, ..rules() };
        let g = batched(&r, 11);
        assert_eq!(g.moves_uci(), off.moves_uci(), "measuring the depth changes no move");
        let [fast, full] = *g.tally();
        let n_full = g.records().iter().filter(|r| r.full == 1).count() as u64;
        assert_eq!((full.moves, fast.moves), (n_full, g.records().len() as u64 - n_full));
        assert_eq!((full.sims, fast.sims), (full.moves * 64, fast.moves * 16));
        assert!(full.nn_evals > 0 && full.nn_evals + full.tt_hits <= full.sims);
        assert!(full.depth_mean > fast.depth_mean / fast.moves as f64 * full.moves as f64);
        assert!(full.main_line >= full.moves && full.depth_max >= full.main_line);
        assert_eq!(off.tally()[1].depth_mean, 0.0);
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
    fn fresh_full_searches_start_from_an_empty_tree() {
        let reused = SelfPlayRules { move_temp: 0.0, ..rules() };
        let fresh = SelfPlayRules { fresh_full: true, ..reused };
        let total = |g: &SelfPlayGame, full: u8| -> Vec<u32> {
            g.records().iter().filter(|r| r.full == full)
                .map(|r| r.visit_cnt.iter().map(|&c| c as u32).sum()).collect()
        };
        let (a, b) = (batched(&reused, 7), batched(&fresh, 7));
        // Fresh sims are counted (no cap): a full search on a fresh tree holds the budget
        // exactly, one on a reused tree holds more.
        assert!(total(&b, 1).iter().all(|&n| n == fresh.num_sims), "{:?}", total(&b, 1));
        assert!(total(&a, 1).iter().any(|&n| n > reused.num_sims), "{:?}", total(&a, 1));
        // Fast searches still reuse the tree.
        assert!(total(&b, 0).iter().any(|&n| n > fresh.fast_sims), "{:?}", total(&b, 0));
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

    /// A search of `sims` from `start` + `moves` with forced playouts `k`, driven one
    /// wave at a time: (root priors, root visits, pruned visits).
    fn forced_search(moves: &[&str], sims: u32, k: f64) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
        let mut s = Search::new(HashEval, SearchParams::default());
        let mut pos = start();
        let played: Vec<Move> = moves.iter().map(|u| {
            let m = u.parse::<shakmaty::uci::UciMove>().unwrap().to_move(&pos).unwrap();
            pos.play_unchecked(&m);
            m
        }).collect();
        s.set_position(start(), &played);
        let mut job = s.start_with_noise(sims, None);
        job.force_playouts(k);
        loop {
            match s.step(&mut job) {
                Step::Eval(reqs) => s.resolve(&mut job, reqs.iter().map(hash_out).collect()),
                Step::Wave => {}
                Step::Done => break,
            }
        }
        let stats = s.root_stats();
        let priors = s.root_moves().iter().map(|m| stats.iter().find(|x| &x.0 == m).unwrap().3).collect();
        let n = s.root_moves().iter().map(|m| stats.iter().find(|x| &x.0 == m).unwrap().1).collect();
        (priors, n, s.pruned_root_visits(k))
    }

    #[test]
    fn forced_playouts_fill_quotas_and_pruning_takes_some_back() {
        let lines: [&[&str]; 4] = [&[], &["e2e4"], &["e2e4", "e7e5", "g1f3"], &["d2d4", "g8f6", "c2c4", "e7e6"]];
        let (mut cut, mut dropped) = (0, 0);
        for line in lines {
            let (_, plain, _) = forced_search(line, 400, 0.0);
            let (priors, n, pruned) = forced_search(line, 400, 2.0);
            let total: u32 = n.iter().sum();
            for i in 0..n.len() {
                // Every visited child got its quota, up to the last wave's growth of it.
                let quota = (2.0 * priors[i] as f64 * (total - 16) as f64).sqrt();
                assert!(n[i] == 0 || n[i] as f64 >= quota.floor(), "{line:?} child {i}: {} < {quota:.1}", n[i]);
                assert!(pruned[i] <= n[i]);
                dropped += (n[i] > 0 && pruned[i] == 0) as u32;
            }
            cut += total - pruned.iter().sum::<u32>();
            let best = |v: &[u32]| (0..v.len()).max_by_key(|&i| v[i]).unwrap();
            assert_eq!(best(&pruned), best(&n), "pruning never changes the most visited move");
            // Forcing widens: at least as many moves visited as by the plain search.
            assert!(n.iter().filter(|&&x| x > 0).count() >= plain.iter().filter(|&&x| x > 0).count());
        }
        assert!(cut > 40 && dropped > 3, "pruning cut {cut} visits, dropped {dropped} moves");
    }

    #[test]
    fn forced_playouts_change_full_searches_only_and_follow_the_seed() {
        let plain = SelfPlayRules { move_temp: 0.0, dirichlet_alpha: 0.3, ..rules() };
        let forced = SelfPlayRules { forced_playouts: 2.0, ..plain };
        let a = batched(&forced, 5);
        assert_eq!(a.moves_uci(), batched(&forced, 5).moves_uci());
        let b = batched(&plain, 5);
        let first_full = a.records().iter().position(|r| r.full == 1).unwrap();
        assert_eq!(a.moves_uci()[..first_full], b.moves_uci()[..first_full]);
        assert_ne!(a.records()[first_full].visit_cnt, b.records()[first_full].visit_cnt);
        for r in a.records().iter().filter(|r| r.full == 0) {
            assert!(r.visit_cnt.iter().all(|&c| c > 0));
        }
    }

    #[test]
    fn records_keep_the_root_search_for_completed_q_targets() {
        for forced in [0.0, 2.0] {
            let r = SelfPlayRules { move_temp: 0.0, dirichlet_alpha: 0.3, forced_playouts: forced, ..rules() };
            let g = batched(&r, 7);
            assert!(g.records().iter().any(|x| x.full == 0) && g.records().iter().any(|x| x.full == 1));
            for rec in g.records() {
                let req = EvalRequest { tokens: rec.tokens, player: rec.player, castling: rec.castling, ep: rec.ep,
                                        rule50: rec.rule50, repetition: rec.repetition, legal: Vec::new() };
                let out = hash_out(&req);
                // The network's own prior and value, not the noised root.
                let mut prior: Vec<f32> = rec.legal_idx.iter().map(|&i| out.logits[i as usize]).collect();
                crate::tree::softmax_priors(&mut prior, 1.0);
                for (a, b) in prior.iter().zip(&rec.legal_prior) {
                    assert!((a - b).abs() < 1e-6, "prior {a} vs recorded {b}");
                }
                assert_eq!(rec.root_net_v, out.value);
                assert_eq!(rec.legal_n.len(), rec.legal_idx.len());
                assert_eq!(rec.legal_q.len(), rec.legal_idx.len());
                // The target is the raw visits (pruned, with forced playouts).
                for (&i, &c) in rec.visit_idx.iter().zip(&rec.visit_cnt) {
                    let k = rec.legal_idx.iter().position(|&x| x == i).expect("target move is legal");
                    assert!(c <= rec.legal_n[k]);
                    if forced == 0.0 {
                        assert_eq!(c, rec.legal_n[k]);
                    }
                }
                let visited = rec.legal_n.iter().filter(|&&n| n > 0).count();
                assert!(visited >= rec.visit_idx.len());
                assert!(rec.legal_q.iter().zip(&rec.legal_n).all(|(&q, &n)| n > 0 || q == 0.0));
                assert!(rec.legal_q.iter().all(|q| (-1.0..=1.0).contains(q)));
            }
        }
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

    /// The sampled moves' worst shortfall from the most-visited move's Q, over the
    /// sampled plies (no forced playouts, so the raw visits chose the move).
    fn worst_sampled_shortfall(g: &SelfPlayGame, temp_plies: u32) -> f32 {
        let mut worst = 0.0f32;
        for (rec, mv) in g.records().iter().zip(&g.moves) {
            if u32::from(rec.halfmove) >= temp_plies {
                break;
            }
            let (f, t, p) = move_coords(mv);
            let played = (f * NUM_PLANES + action_plane(f, t, p)) as u16;
            let k = rec.legal_idx.iter().position(|&i| i == played).unwrap();
            let top = (0..rec.legal_n.len()).max_by_key(|&i| rec.legal_n[i]).unwrap();
            worst = worst.max(rec.legal_q[top] - rec.legal_q[k]);
        }
        worst
    }

    #[test]
    fn temp_cutoff_samples_only_moves_near_the_best_q() {
        let open = SelfPlayRules { temp_plies: 40, max_plies: 40, ..rules() };
        let cut = SelfPlayRules { temp_cutoff: 0.05, ..open };
        let (mut before, mut after) = (0.0f32, 0.0f32);
        for seed in 0..6 {
            before = before.max(worst_sampled_shortfall(&batched(&open, seed), 40));
            after = after.max(worst_sampled_shortfall(&batched(&cut, seed), 40));
        }
        assert!(before > 0.05, "no sampled move fell short without the cutoff: {before}");
        assert!(after <= 0.05 + 1e-5, "a sampled move fell {after} short of the best Q");
        assert_eq!(batched(&cut, 3).moves_uci(), batched(&cut, 3).moves_uci(), "must follow the seed");
    }
}
