//! Single-game batched MCTS over one position -- the deployed bot's search
//! (moved here from ct-bot so the batched match runner runs the same code).
//! Semantics mirror `Pos2MoveV2MctsBot`: PUCT + FPU, virtual-loss leaf waves
//! with one batched NN forward per wave, zobrist TT for priors + value,
//! threefold/50-move awareness over game history + search path (plus Lc0's
//! two-fold rule for cycles inside the search), and tree reuse across moves
//! (re-root when the new position extends the previous root by 1..=4
//! expanded plies).
//!
//! `go()` runs a whole search against the `Evaluator`. The same search can be
//! driven one wave at a time -- `start`, then `step` until `Step::Done`,
//! handing each `Step::Eval` batch's outputs to `resolve`, then `finish` --
//! which is what `go()` itself does, so the two cannot drift apart. That lets
//! a driver interleave hundreds of searches and evaluate all their waves in
//! one forward.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use crate::encoding::{
    action_plane, encode_position, halfmove, move_coords, rule_key, rule_state,
    search_terminal_value, zobrist,
    NUM_PLANES,
};
use crate::tree::{backup_vl, puct_select, softmax_priors, Node};
use rand::prelude::*;
use rand::rngs::StdRng;
use shakmaty::{Chess, Move, Position};
use shakmaty_syzygy::{AmbiguousWdl, Tablebase, Wdl};

pub use crate::eval::{EvalOutput, EvalRequest, Evaluator, NoEval};

pub struct SearchParams {
    pub c_puct: f64,
    pub fpu: f64,
    pub prior_temp: f32,
    pub sim_batch: usize,
    /// Sample the move ∝ visits^(1/temp) for the first `move_temp_plies`
    /// half-moves (0.0 = always argmax — the deterministic gauntlet setting).
    pub move_temp: f64,
    pub move_temp_plies: u32,
    pub tt_cap: usize,
    /// Fill `EvalRequest::legal` (batched drivers that gather legal logits on
    /// the GPU). The bot leaves it off and evaluates full logits.
    pub legal_in_requests: bool,
    /// The network reads the rule state (`EvalRequest::rule50` / `repetition`), so its
    /// evaluations are cached per (position, rule state) instead of per position.
    /// Requests carry the rule state either way; off, the search is unchanged.
    pub rule_state: bool,
    /// Syzygy tables. In-tree, positions reached by a capture or pawn move
    /// that the tables cover are scored exactly (WDL) instead of by the value
    /// head. At a covered root, a won position plays the DTZ-optimal move
    /// outright and a drawn/lost one restricts the search to result-preserving
    /// moves (root probing needs the DTZ tables too).
    pub syzygy: Option<Arc<Tablebase<Chess>>>,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            c_puct: 1.0,
            fpu: 0.2,
            prior_temp: 1.0,
            sim_batch: 16,
            move_temp: 0.0,
            move_temp_plies: 0,
            tt_cap: 2_000_000,
            legal_in_requests: false,
            rule_state: false,
            syzygy: None,
        }
    }
}

/// Root Dirichlet noise for one search (self-play exploration, AlphaZero): once
/// the root is expanded its priors become `(1 - eps) * p + eps * Dir(alpha)`.
/// The TT keeps the clean priors. The bot never sets it.
#[derive(Clone, Copy)]
pub struct RootNoise {
    pub alpha: f64,
    pub eps: f64,
    pub seed: u64,
}

/// Standard normal via Box-Muller (rand 0.8 has no distributions beyond uniform
/// without rand_distr).
pub fn std_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Gamma(alpha, 1) by Marsaglia-Tsang, boosted for alpha < 1.
pub fn sample_gamma(rng: &mut StdRng, alpha: f64) -> f64 {
    if alpha < 1.0 {
        let u = rng.gen::<f64>();
        return sample_gamma(rng, alpha + 1.0) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (x, v) = loop {
            let x = std_normal(rng);
            let v = 1.0 + c * x;
            if v > 0.0 {
                break (x, v * v * v);
            }
        };
        let u = rng.gen::<f64>();
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Extra TOTAL-root-visit headroom above the per-move sims cap when pondering.
/// Deployed lichess bot semantics (ct-bot `timeman`, branch lichess-levers): a
/// visit-cap A/B showed capping totals near the fresh-sims cap loses strength,
/// so the headroom only bounds chained-ponder tails.
pub const PONDER_HEADROOM: u32 = 4000;

/// Fresh-sim budget for a move given `plan` sims from the clock, `warm`
/// visits already on the root and the total-visits ceiling `total_cap`
/// (per-speed max sims + PONDER_HEADROOM). Min-cap form, not `plan - warm`:
/// warm visits cost no clock, so only clip when totals near the edge.
pub fn effective_sims(plan: u32, warm: u32, total_cap: u32) -> u32 {
    plan.min(total_cap.saturating_sub(warm)).max(64)
}

/// Ponder-search budget: like `effective_sims` with no floor -- a ponder root
/// already carrying `total_cap` visits gets 0 sims.
pub fn ponder_budget(max_sims: u32, warm: u32, total_cap: u32) -> u32 {
    max_sims.min(total_cap.saturating_sub(warm))
}

pub struct SearchResult {
    pub best: Move,
    /// Root value from the side-to-move POV.
    pub value: f32,
    pub sims: u32,
    pub nn_evals: u32,
    pub tt_hits: u32,
    /// Tablebase-scored leaves (or 1 when the root move came from the tables).
    pub tb_hits: u32,
    pub elapsed: Duration,
}

/// What the tablebases say about the root.
enum RootTb {
    /// Not covered (or no tables, or a probe failed): search normally.
    Off,
    /// Won: play the DTZ-optimal move without searching.
    Play(Move),
    /// Drawn or lost: search only among the moves that keep the best result.
    Restrict(Vec<Move>),
}

enum LeafEval {
    /// Index into the deduped NN batch (priors computed after the forward).
    Batch(usize),
    /// Transposition / reused-subtree hit: priors (legal-move order) + value.
    Cached(Vec<f32>, f32),
}

struct PendingLeaf {
    node: usize,
    path: Vec<(usize, usize)>,
    key: u64,
    moves: Vec<Move>,
    eval: LeafEval,
}

enum Stage {
    /// Expand the root (one eval unless the TT has it), then waves.
    Root,
    /// The root's eval is out: (zobrist key, legal moves).
    RootPending(u64, Vec<Move>),
    /// Leaf waves until the budget is spent or the job is stopped.
    Waves,
    /// A wave's NN batch is out.
    WavePending(Vec<PendingLeaf>),
    Done,
}

/// One search in progress, from `start` to `finish`.
pub struct Job {
    sims: u32,
    t0: Instant,
    sims_done: u32,
    nn_evals: u32,
    tt_hits: u32,
    tb_hits: u32,
    restrict: Option<Vec<Move>>,
    tb_move: Option<Move>,
    stage: Stage,
    stopped: bool,
    noise: Option<RootNoise>,
}

impl Job {
    /// End the search after the wave in progress (go()'s stop flag / deadline).
    pub fn stop(&mut self) {
        self.stopped = true;
    }

    pub fn is_done(&self) -> bool {
        matches!(self.stage, Stage::Done)
    }
}

/// What a search needs next.
pub enum Step {
    /// Evaluate these positions and pass the outputs, in order, to `resolve`.
    Eval(Vec<EvalRequest>),
    /// A wave completed without any NN work (terminal / TB / TT leaves only).
    Wave,
    /// Budget spent (or stopped, or a tablebase move): call `finish`.
    Done,
}

pub struct Search<E: Evaluator> {
    pub params: SearchParams,
    eval: E,
    tree: Vec<Node>,
    pos: Chess,
    history: Vec<u64>, // zobrist of every game position incl. current
    tt: HashMap<u64, (Vec<f32>, f32)>, // priors (legal-move order) + stm value
    prev_start: Option<Chess>,
    prev_moves: Vec<Move>,
    rng: StdRng,
}

impl<E: Evaluator> Search<E> {
    pub fn new(eval: E, mut params: SearchParams) -> Self {
        params.rule_state |= eval.reads_rule_state();
        let pos = Chess::default();
        let history = vec![zobrist(&pos)];
        Search {
            params,
            eval,
            tree: vec![Node::default()],
            pos,
            history,
            tt: HashMap::new(),
            prev_start: None,
            prev_moves: Vec::new(),
            rng: StdRng::from_entropy(),
        }
    }

    pub fn new_game(&mut self) {
        self.tree = vec![Node::default()];
        self.tt.clear();
        self.prev_start = None;
        self.prev_moves.clear();
        self.set_position(Chess::default(), &[]);
    }

    pub fn position(&self) -> &Chess {
        &self.pos
    }

    /// Visits already on the root (warm visits from tree reuse / pondering).
    pub fn root_visits(&self) -> u32 {
        if !self.tree[0].expanded {
            return 0;
        }
        self.tree[0].n.iter().sum()
    }

    /// Set the root to `start` + `moves`, rebuilding the zobrist history and
    /// re-rooting the previous search tree when the new position extends the
    /// previous one by 1..=4 plies through expanded nodes.
    pub fn set_position(&mut self, start: Chess, moves: &[Move]) {
        let mut pos = start.clone();
        let mut history = vec![zobrist(&pos)];
        for m in moves {
            pos.play_unchecked(m);
            history.push(zobrist(&pos));
        }

        let reused = self.try_reroot(&start, moves);
        if !reused {
            self.tree = vec![Node::default()];
        }
        self.pos = pos;
        self.history = history;
        self.prev_start = Some(start);
        self.prev_moves = moves.to_vec();
    }

    /// Mirror of `Pos2MoveV2MctsBot._take_reuse_root`: accept if the new move
    /// list extends the previous root's by 1..=4 plies and every step descends
    /// into an expanded child. On success, compacts the subtree into a fresh
    /// arena rooted at index 0.
    fn try_reroot(&mut self, start: &Chess, moves: &[Move]) -> bool {
        let Some(prev_start) = &self.prev_start else {
            return false;
        };
        if prev_start != start || !self.tree[0].expanded {
            return false;
        }
        let rp = self.prev_moves.len();
        let gap = moves.len() as i64 - rp as i64;
        if !(1..=4).contains(&gap) || moves[..rp] != self.prev_moves[..] {
            return false;
        }
        let mut node = 0usize;
        for mv in &moves[rp..] {
            if !self.tree[node].expanded {
                return false;
            }
            let Some(idx) = self.tree[node].moves.iter().position(|m| m == mv) else {
                return false;
            };
            let child = self.tree[node].children[idx];
            if child < 0 || !self.tree[child as usize].expanded {
                return false;
            }
            node = child as usize;
        }
        self.tree = reroot(&self.tree, node);
        true
    }

    /// Run up to `sims` simulations (or until `deadline` / `stop`, checked
    /// between waves; at least one wave always completes). Panics if the root
    /// position has no legal moves — callers check game-over first.
    pub fn go(&mut self, sims: u32, deadline: Option<Instant>, stop: &AtomicBool) -> Result<SearchResult> {
        let mut job = self.start(sims);
        loop {
            match self.step(&mut job) {
                Step::Done => break,
                Step::Eval(batch) => {
                    let root = matches!(job.stage, Stage::RootPending(..));
                    let outputs = self.eval.eval_batch(&batch)?;
                    self.resolve(&mut job, outputs);
                    if root {
                        continue; // the stop/deadline checks run between waves only
                    }
                }
                Step::Wave => {}
            }
            if stop.load(Ordering::Relaxed) {
                job.stop();
            } else if let Some(d) = deadline {
                if Instant::now() >= d {
                    job.stop();
                }
            }
        }
        Ok(self.finish(job))
    }

    /// Begin a search of up to `sims` simulations from the current position.
    pub fn start(&mut self, sims: u32) -> Job {
        let mut job = Job {
            sims,
            t0: Instant::now(),
            sims_done: 0,
            nn_evals: 0,
            tt_hits: 0,
            tb_hits: 0,
            restrict: None,
            tb_move: None,
            stage: Stage::Root,
            stopped: false,
            noise: None,
        };
        match self.root_tablebase() {
            RootTb::Play(best) => {
                job.tb_move = Some(best);
                job.tb_hits = 1;
                job.stage = Stage::Done;
            }
            RootTb::Restrict(keep) => {
                self.tree = vec![Node::default()]; // a reused root has every move
                job.restrict = Some(keep);
            }
            RootTb::Off => {}
        }
        job
    }

    /// `start`, plus root noise: applied now to a reused root, else as soon as
    /// the root is expanded -- before the first wave either way.
    pub fn start_with_noise(&mut self, sims: u32, noise: Option<RootNoise>) -> Job {
        let mut job = self.start(sims);
        job.noise = noise;
        if self.tree[0].expanded && !job.is_done() {
            self.apply_root_noise(&mut job);
        }
        job
    }

    fn legal_of(&self, moves: &[Move]) -> Vec<u16> {
        if self.params.legal_in_requests { legal_indices(moves) } else { Vec::new() }
    }

    /// Seed the move-temperature sampler (the bot keeps its entropy seed).
    pub fn seed_rng(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    /// TT entries held (evaluation cache size).
    pub fn tt_len(&self) -> usize {
        self.tt.len()
    }

    fn apply_root_noise(&mut self, job: &mut Job) {
        let Some(n) = job.noise.take() else { return };
        if n.alpha <= 0.0 {
            return;
        }
        let mut rng = StdRng::seed_from_u64(n.seed);
        let root = &mut self.tree[0];
        let noise: Vec<f64> = (0..root.priors.len()).map(|_| sample_gamma(&mut rng, n.alpha)).collect();
        let total: f64 = noise.iter().sum();
        if total > 0.0 {
            for (p, x) in root.priors.iter_mut().zip(&noise) {
                *p = ((1.0 - n.eps) * (*p as f64) + n.eps * x / total) as f32;
            }
        }
    }

    /// Advance `job` to its next NN batch (or the end of a wave, or the end).
    pub fn step(&mut self, job: &mut Job) -> Step {
        loop {
            match std::mem::replace(&mut job.stage, Stage::Done) {
                Stage::Done => return Step::Done,
                Stage::RootPending(..) | Stage::WavePending(_) => {
                    panic!("step() while an eval is out: call resolve() first")
                }
                Stage::Root => {
                    job.stage = Stage::Waves;
                    if self.tree[0].expanded {
                        continue;
                    }
                    let pos_key = *self.history.last().unwrap();
                    let prior = &self.history[..self.history.len() - 1];
                    let (rule50, repetition) = rule_state(&self.pos, prior, pos_key);
                    let key = self.eval_key(pos_key, rule50, repetition);
                    let moves: Vec<Move> = self.pos.legal_moves().to_vec();
                    assert!(!moves.is_empty(), "go() on a terminal position");
                    if let Some((priors, _)) = self.tt.get(&key).cloned() {
                        job.tt_hits += 1;
                        self.expand_root(job, moves, priors);
                        continue;
                    }
                    let (tokens, player, castling, ep) = encode_position(&self.pos);
                    let legal = self.legal_of(&moves);
                    job.stage = Stage::RootPending(key, moves);
                    return Step::Eval(vec![EvalRequest {
                        tokens, player, castling, ep, rule50, repetition, legal,
                    }]);
                }
                Stage::Waves => {
                    if job.stopped || job.sims_done >= job.sims {
                        return Step::Done;
                    }
                    let (pending, batch) = self.collect_wave(job);
                    if batch.is_empty() {
                        self.resolve_wave(job, pending, Vec::new());
                        job.stage = Stage::Waves;
                        return Step::Wave;
                    }
                    job.stage = Stage::WavePending(pending);
                    return Step::Eval(batch);
                }
            }
        }
    }

    /// Hand back the outputs for the batch `step` returned, in its order.
    pub fn resolve(&mut self, job: &mut Job, outputs: Vec<EvalOutput>) {
        match std::mem::replace(&mut job.stage, Stage::Waves) {
            Stage::RootPending(key, moves) => {
                let out = outputs.into_iter().next().expect("one output for the root");
                job.nn_evals += 1;
                let priors = priors_for(&moves, &out.logits, self.params.prior_temp);
                self.insert_tt(key, priors.clone(), out.value);
                self.expand_root(job, moves, priors);
            }
            Stage::WavePending(pending) => {
                job.nn_evals += outputs.len() as u32;
                self.resolve_wave(job, pending, outputs);
            }
            _ => panic!("resolve() with no eval out"),
        }
    }

    /// The move and stats of a finished (or stopped) search.
    pub fn finish(&mut self, job: Job) -> SearchResult {
        if let Some(best) = job.tb_move {
            return SearchResult {
                best,
                value: 1.0,
                sims: 0,
                nn_evals: 0,
                tt_hits: 0,
                tb_hits: 1,
                elapsed: job.t0.elapsed(),
            };
        }
        let root = &self.tree[0];
        let best_idx = if self.params.move_temp > 0.0
            && halfmove(&self.pos) < self.params.move_temp_plies
        {
            let weights: Vec<f64> = root
                .n
                .iter()
                .map(|&n| (n as f64).powf(1.0 / self.params.move_temp))
                .collect();
            let total: f64 = weights.iter().sum();
            if total > 0.0 {
                let mut r = self.rng.gen::<f64>() * total;
                let mut pick = root.n.len() - 1;
                for (i, w) in weights.iter().enumerate() {
                    if r < *w {
                        pick = i;
                        break;
                    }
                    r -= w;
                }
                pick
            } else {
                0
            }
        } else {
            (0..root.n.len()).max_by_key(|&i| root.n[i]).unwrap_or(0)
        };
        let value = if root.n[best_idx] > 0 {
            (-root.w[best_idx] / root.n[best_idx] as f64) as f32
        } else {
            0.0
        };

        SearchResult {
            best: root.moves[best_idx].clone(),
            value,
            sims: job.sims_done,
            nn_evals: job.nn_evals,
            tt_hits: job.tt_hits,
            tb_hits: job.tb_hits,
            elapsed: job.t0.elapsed(),
        }
    }

    fn expand_root(&mut self, job: &mut Job, moves: Vec<Move>, priors: Vec<f32>) {
        let (moves, priors) = match &job.restrict {
            Some(keep) => restrict_moves(moves, priors, keep),
            None => (moves, priors),
        };
        self.tree[0].expand(moves, priors);
        self.apply_root_noise(job);
    }

    /// Descend one wave of leaves. Terminal and tablebase leaves are backed up
    /// at once; the rest wait for the wave's NN batch.
    fn collect_wave(&mut self, job: &mut Job) -> (Vec<PendingLeaf>, Vec<EvalRequest>) {
        let wave = (self.params.sim_batch as u32).min(job.sims - job.sims_done) as usize;
        let mut pending: Vec<PendingLeaf> = Vec::with_capacity(wave);
        let mut batch: Vec<EvalRequest> = Vec::new();
        let mut batch_index: HashMap<u64, usize> = HashMap::new();

        for _ in 0..wave {
            let (node_idx, path, pos, path_keys) = self.descend();
            let key = zobrist(&pos);
            let mut all_keys = self.history.clone();
            all_keys.extend_from_slice(&path_keys);
            let root_idx = self.history.len() - 1;
            if let Some(t) = search_terminal_value(&pos, &all_keys, key, root_idx) {
                backup_vl(&mut self.tree, &path, t);
                job.sims_done += 1;
                continue;
            }
            if let Some(t) = self.leaf_tablebase(&pos) {
                backup_vl(&mut self.tree, &path, t);
                job.sims_done += 1;
                job.tb_hits += 1;
                continue;
            }
            let moves: Vec<Move> = pos.legal_moves().to_vec();
            let (rule50, repetition) = rule_state(&pos, &all_keys[..all_keys.len() - 1], key);
            let key = self.eval_key(key, rule50, repetition);
            // Defer cached (TT) hits and NN leaves alike: their virtual loss
            // stays applied through the wave so concurrent descents diversify
            // (mirrors Pos2MoveV2MctsBot._run_batch). Backing TT hits up
            // immediately collapses search breadth once the TT fills, which
            // wrecks play at high sim counts.
            let eval = if let Some((priors, value)) = self.tt.get(&key).cloned() {
                job.tt_hits += 1;
                LeafEval::Cached(priors, value)
            } else {
                let legal_in_requests = self.params.legal_in_requests;
                let batch_idx = *batch_index.entry(key).or_insert_with(|| {
                    let (tokens, player, castling, ep) = encode_position(&pos);
                    let legal = if legal_in_requests { legal_indices(&moves) } else { Vec::new() };
                    batch.push(EvalRequest { tokens, player, castling, ep, rule50, repetition, legal });
                    batch.len() - 1
                });
                LeafEval::Batch(batch_idx)
            };
            pending.push(PendingLeaf { node: node_idx, path, key, moves, eval });
        }
        (pending, batch)
    }

    fn resolve_wave(&mut self, job: &mut Job, pending: Vec<PendingLeaf>, outputs: Vec<EvalOutput>) {
        for leaf in pending {
            let prior_temp = self.params.prior_temp;
            let (priors, value) = match leaf.eval {
                LeafEval::Cached(priors, value) => (priors, value),
                // A later leaf in this wave may have inserted the key; prefer
                // the TT entry so transpositions share one eval.
                LeafEval::Batch(bi) => match self.tt.get(&leaf.key) {
                    Some(hit) => hit.clone(),
                    None => {
                        let priors = priors_for(&leaf.moves, &outputs[bi].logits, prior_temp);
                        self.insert_tt(leaf.key, priors.clone(), outputs[bi].value);
                        (priors, outputs[bi].value)
                    }
                },
            };
            if !self.tree[leaf.node].expanded {
                self.tree[leaf.node].expand(leaf.moves, priors);
            }
            backup_vl(&mut self.tree, &leaf.path, value);
            job.sims_done += 1;
        }
    }

    /// Exact WDL for a leaf the tables cover, from its side-to-move POV. Only
    /// probed straight after a capture or pawn move (halfmove clock 0), where
    /// WDL is exact under the 50-move rule; cursed wins and blessed losses
    /// score as the draws they are.
    fn leaf_tablebase(&self, pos: &Chess) -> Option<f32> {
        let tb = self.params.syzygy.as_ref()?;
        if pos.halfmoves() != 0 || pos.board().occupied().count() > tb.max_pieces() {
            return None;
        }
        Some(match tb.probe_wdl_after_zeroing(pos).ok()? {
            Wdl::Win => 1.0,
            Wdl::Loss => -1.0,
            Wdl::CursedWin | Wdl::Draw | Wdl::BlessedLoss => 0.0,
        })
    }

    fn root_tablebase(&self) -> RootTb {
        let Some(tb) = &self.params.syzygy else {
            return RootTb::Off;
        };
        if self.pos.board().occupied().count() > tb.max_pieces() {
            return RootTb::Off;
        }
        // Each move's result for us under the 50-move rule from the current
        // clock: +1 win, 0 draw, -1 loss. Ambiguous (DTZ-rounded) verdicts are
        // read pessimistically for us.
        let moves = self.pos.legal_moves();
        let mut verdicts = Vec::with_capacity(moves.len());
        for m in &moves {
            let mut after = self.pos.clone();
            after.play_unchecked(m);
            let v = match tb.probe_wdl(&after) {
                Ok(AmbiguousWdl::Loss) => 1,
                Ok(AmbiguousWdl::Win | AmbiguousWdl::MaybeWin) => -1,
                Ok(_) => 0,
                Err(_) => return RootTb::Off,
            };
            verdicts.push(v);
        }
        let Some(&best) = verdicts.iter().max() else {
            return RootTb::Off;
        };
        if best == 1 {
            return match tb.best_move(&self.pos) {
                Ok(Some((m, _))) => RootTb::Play(m),
                _ => RootTb::Off,
            };
        }
        RootTb::Restrict(
            moves
                .iter()
                .zip(&verdicts)
                .filter(|&(_, &v)| v == best)
                .map(|(m, _)| m.clone())
                .collect(),
        )
    }

    /// Virtual-loss descent to an unexpanded node; returns (leaf node index,
    /// path edges, leaf position, zobrist keys pushed en route).
    fn descend(&mut self) -> (usize, Vec<(usize, usize)>, Chess, Vec<u64>) {
        let mut node_idx = 0usize;
        let mut pos = self.pos.clone();
        let mut path = Vec::new();
        let mut keys = Vec::new();
        loop {
            if !self.tree[node_idx].expanded {
                return (node_idx, path, pos, keys);
            }
            let idx = puct_select(&self.tree[node_idx], self.params.c_puct, self.params.fpu);
            {
                let node = &mut self.tree[node_idx];
                node.n[idx] += 1; // virtual loss
                node.w[idx] += 1.0;
            }
            path.push((node_idx, idx));
            let mv = self.tree[node_idx].moves[idx].clone();
            pos.play_unchecked(&mv);
            keys.push(zobrist(&pos));
            let child = self.tree[node_idx].children[idx];
            node_idx = if child >= 0 {
                child as usize
            } else {
                self.tree.push(Node::default());
                let new_idx = self.tree.len() - 1;
                self.tree[node_idx].children[idx] = new_idx as i32;
                new_idx
            };
        }
    }

    /// Root children as (move, visits, Q from the root side-to-move POV, prior),
    /// most-visited first. Diagnostic only.
    pub fn root_stats(&self) -> Vec<(Move, u32, f64, f32)> {
        let root = &self.tree[0];
        let mut out: Vec<_> = (0..root.moves.len())
            .map(|i| {
                let q = if root.n[i] > 0 { -root.w[i] / root.n[i] as f64 } else { f64::NAN };
                (root.moves[i].clone(), root.n[i], q, root.priors[i])
            })
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1));
        out
    }

    /// Most-visited line from the root as (move, visits, Q from the mover's
    /// POV). Diagnostic only.
    pub fn pv(&self, max_len: usize) -> Vec<(Move, u32, f64)> {
        let mut out = Vec::new();
        let mut node = 0usize;
        while out.len() < max_len && self.tree[node].expanded {
            let nd = &self.tree[node];
            let Some(i) = (0..nd.moves.len()).max_by_key(|&i| nd.n[i]) else { break };
            if nd.n[i] == 0 {
                break;
            }
            out.push((nd.moves[i].clone(), nd.n[i], -nd.w[i] / nd.n[i] as f64));
            if nd.children[i] < 0 {
                break;
            }
            node = nd.children[i] as usize;
        }
        out
    }

    /// Raw value-head estimate for a position (side-to-move POV), no search and no
    /// game history (repetition count 0).
    pub fn raw_value(&mut self, pos: &Chess) -> Result<f32> {
        let (tokens, player, castling, ep) = encode_position(pos);
        let (rule50, repetition) = rule_state(pos, &[], 0);
        Ok(self
            .eval
            .eval_batch(&[EvalRequest {
                tokens, player, castling, ep, rule50, repetition, legal: Vec::new(),
            }])?
            .remove(0)
            .value)
    }

    /// TT / batch key of an evaluation: the position, plus its rule state when the
    /// network reads it.
    fn eval_key(&self, pos_key: u64, rule50: u8, repetition: u8) -> u64 {
        if self.params.rule_state { rule_key(pos_key, rule50, repetition) } else { pos_key }
    }

    fn insert_tt(&mut self, key: u64, priors: Vec<f32>, value: f32) {
        if self.tt.len() >= self.params.tt_cap {
            self.tt.clear(); // crude but rare; avoids unbounded growth
        }
        self.tt.insert(key, (priors, value));
    }
}

/// Flat logit index of each move, in order.
pub fn legal_indices(moves: &[Move]) -> Vec<u16> {
    moves
        .iter()
        .map(|m| {
            let (f, t, p) = move_coords(m);
            (f * NUM_PLANES + action_plane(f, t, p)) as u16
        })
        .collect()
}

/// Softmax priors over `moves` from full logits, or from logits already
/// gathered in move order (see `EvalOutput::logits`) -- the same floats either way.
fn priors_for(moves: &[Move], logits: &[f32], prior_temp: f32) -> Vec<f32> {
    let mut scores: Vec<f32> = if logits.len() == 64 * NUM_PLANES {
        moves
            .iter()
            .map(|m| {
                let (f, t, p) = move_coords(m);
                logits[f * NUM_PLANES + action_plane(f, t, p)]
            })
            .collect()
    } else {
        assert_eq!(logits.len(), moves.len(), "gathered logits do not match the legal moves");
        logits.to_vec()
    };
    softmax_priors(&mut scores, prior_temp);
    scores
}

/// Keep only `keep` (with their priors, renormalized) from a root expansion.
fn restrict_moves(moves: Vec<Move>, priors: Vec<f32>, keep: &[Move]) -> (Vec<Move>, Vec<f32>) {
    let (moves, mut priors): (Vec<Move>, Vec<f32>) =
        moves.into_iter().zip(priors).filter(|(m, _)| keep.contains(m)).unzip();
    let total: f32 = priors.iter().sum();
    let k = priors.len() as f32;
    for p in priors.iter_mut() {
        *p = if total > 0.0 { *p / total } else { 1.0 / k };
    }
    (moves, priors)
}

/// Copy the subtree under `root` into a fresh arena with the root at index 0.
fn reroot(tree: &[Node], root: usize) -> Vec<Node> {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut order = vec![root];
    map.insert(root, 0);
    let mut i = 0;
    while i < order.len() {
        let old = order[i];
        i += 1;
        for &c in &tree[old].children {
            if c >= 0 && !map.contains_key(&(c as usize)) {
                map.insert(c as usize, order.len());
                order.push(c as usize);
            }
        }
    }
    order
        .iter()
        .map(|&old| {
            let node = &tree[old];
            Node {
                expanded: node.expanded,
                moves: node.moves.clone(),
                priors: node.priors.clone(),
                n: node.n.clone(),
                w: node.w.clone(),
                children: node
                    .children
                    .iter()
                    .map(|&c| if c >= 0 { map[&(c as usize)] as i32 } else { -1 })
                    .collect(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;
    use shakmaty::uci::UciMove;
    use shakmaty::CastlingMode;

    /// Uniform priors, constant value — search behavior then depends only on
    /// the game-theoretic terminal values.
    struct UniformEval;

    impl Evaluator for UniformEval {
        fn eval_batch(&mut self, batch: &[EvalRequest]) -> Result<Vec<EvalOutput>> {
            Ok(batch
                .iter()
                .map(|_| EvalOutput { logits: vec![0.0; 64 * NUM_PLANES], value: 0.0 })
                .collect())
        }
    }

    fn pos(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .unwrap()
            .into_position(CastlingMode::Standard)
            .unwrap()
    }

    fn mv(p: &Chess, uci: &str) -> Move {
        UciMove::from_ascii(uci.as_bytes())
            .unwrap()
            .to_move(p)
            .unwrap()
    }

    /// Uniform outputs; records every request's (tokens, rule50, repetition).
    struct RecordingEval {
        seen: Arc<std::sync::Mutex<Vec<([u8; 64], u8, u8)>>>,
        rule_state: bool,
    }

    impl Evaluator for RecordingEval {
        fn eval_batch(&mut self, batch: &[EvalRequest]) -> Result<Vec<EvalOutput>> {
            self.seen.lock().unwrap().extend(batch.iter().map(|r| (r.tokens, r.rule50, r.repetition)));
            Ok(batch
                .iter()
                .map(|_| EvalOutput { logits: vec![0.0; 64 * NUM_PLANES], value: 0.0 })
                .collect())
        }

        fn reads_rule_state(&self) -> bool {
            self.rule_state
        }
    }

    #[test]
    fn rule_state_counts_earlier_occurrences_and_caps() {
        let p = pos("8/8/8/4k3/8/8/4K3/7R w - - 37 60");
        assert_eq!(crate::encoding::rule_state(&p, &[], 7), (37, 0));
        assert_eq!(crate::encoding::rule_state(&p, &[7, 1, 7], 7), (37, 2));
        assert_eq!(crate::encoding::rule_state(&p, &[7, 7, 7, 7], 7), (37, 2));
        let p = pos("8/8/8/4k3/8/8/4K3/7R w - - 140 90");
        assert_eq!(crate::encoding::rule_state(&p, &[], 7).0, 100);
        let k = crate::encoding::rule_key(7, 4, 1);
        assert!(k != 7 && k != crate::encoding::rule_key(7, 4, 0) && k != crate::encoding::rule_key(7, 5, 1));
    }

    #[test]
    fn requests_carry_the_rule_state_and_key_the_cache_by_it() {
        // The start position again after the knights go out and back: same board,
        // halfmove clock 4, one earlier occurrence.
        let start = Chess::default();
        let start_tokens = encode_position(&start).0;
        let (mut p, mut moves) = (start.clone(), Vec::new());
        for u in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            let m = mv(&p, u);
            p.play_unchecked(&m);
            moves.push(m);
        }
        for rule_state in [false, true] {
            let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
            let eval = RecordingEval { seen: seen.clone(), rule_state };
            let mut s = Search::new(eval, SearchParams::default());
            s.set_position(start.clone(), &[]);
            s.go(64, None, &AtomicBool::new(false)).unwrap();
            assert_eq!(seen.lock().unwrap()[0], (start_tokens, 0, 0), "the root: fresh clock, no repetition");
            let before = seen.lock().unwrap().len();
            s.set_position(start.clone(), &moves);
            s.go(64, None, &AtomicBool::new(false)).unwrap();
            let again: Vec<_> =
                seen.lock().unwrap()[before..].iter().filter(|r| r.0 == start_tokens).cloned().collect();
            if rule_state {
                assert_eq!(again.first(), Some(&(start_tokens, 4, 1)), "a new evaluation under the new rule state");
            } else {
                assert!(again.is_empty(), "without rule-state inputs the cached evaluation is reused: {again:?}");
            }
        }
    }

    #[test]
    fn finds_mate_in_one() {
        // Back-rank: Ra1-a8 is mate.
        let p = pos("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1");
        let mut s = Search::new(UniformEval, SearchParams::default());
        s.set_position(p, &[]);
        let r = s.go(256, None, &AtomicBool::new(false)).unwrap();
        assert_eq!(
            r.best.to_uci(CastlingMode::Standard).to_string(),
            "a1a8",
            "should find the back-rank mate"
        );
        assert!(r.value > 0.9, "root value should be near +1, got {}", r.value);
    }

    #[test]
    fn detects_threefold_in_search() {
        // K+R vs K: after the same position has occurred twice in the game
        // history, the search must see that one more repetition is a draw
        // (value 0), not a win.
        let start = pos("8/8/8/4k3/8/8/4K3/7R w - - 0 1");
        // Rh1-g1-h1 with king shuffles e5-d5: reaches the start position
        // (w to move) a 2nd time; a 3rd occurrence inside the search is a draw.
        let mut moves = Vec::new();
        let mut p = start.clone();
        for u in ["h1g1", "e5d5", "g1h1", "d5e5"] {
            let m = mv(&p, u);
            p.play_unchecked(&m);
            moves.push(m);
        }
        let mut s = Search::new(UniformEval, SearchParams::default());
        s.set_position(start, &moves);
        let r = s.go(128, None, &AtomicBool::new(false)).unwrap();
        // With a uniform NN the absolute move doesn't matter; the search must
        // simply terminate (repetition leaves are terminal, not NN-evaluated)
        // and the root value must be drawish, not a fantasy win.
        assert!(r.value.abs() < 0.5, "root value should be drawish, got {}", r.value);
        assert_eq!(r.sims, 128);
    }

    /// Value = material balance from the side to move's POV; policy = uniform
    /// plus a bonus for checking moves. A crude stand-in for a real net, but
    /// one that can tell winning from losing and looks at forcing moves first.
    struct CheckingMaterialEval;

    fn request_to_pos(r: &EvalRequest) -> Chess {
        const CH: &[u8; 13] = b".PNBRQKpnbrqk";
        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                match r.tokens[rank * 8 + file] {
                    0 => empty += 1,
                    t => {
                        if empty > 0 {
                            fen.push_str(&empty.to_string());
                            empty = 0;
                        }
                        fen.push(CH[t as usize] as char);
                    }
                }
            }
            if empty > 0 {
                fen.push_str(&empty.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }
        fen.push_str(if r.player == 1 { " w - - 0 1" } else { " b - - 0 1" });
        pos(&fen)
    }

    impl Evaluator for CheckingMaterialEval {
        fn eval_batch(&mut self, batch: &[EvalRequest]) -> Result<Vec<EvalOutput>> {
            const VAL: [f32; 7] = [0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 0.0];
            Ok(batch
                .iter()
                .map(|r| {
                    let white: f32 = r
                        .tokens
                        .iter()
                        .map(|&t| match t {
                            1..=6 => VAL[t as usize],
                            7..=12 => -VAL[t as usize - 6],
                            _ => 0.0,
                        })
                        .sum();
                    let stm = if r.player == 1 { white } else { -white };
                    let p = request_to_pos(r);
                    let mut logits = vec![0.0; 64 * NUM_PLANES];
                    for m in p.legal_moves() {
                        let mut after = p.clone();
                        after.play_unchecked(&m);
                        if after.is_check() {
                            let (f, t, pr) = move_coords(&m);
                            logits[f * NUM_PLANES + action_plane(f, t, pr)] = 3.0;
                        }
                    }
                    EvalOutput { logits, value: (stm / 3.0).tanh() }
                })
                .collect())
        }
    }

    #[test]
    fn finds_perpetual_via_twofold() {
        // Black is a queen and rook down; per Stockfish (d26) every move loses
        // (-6.5 or worse) except Qf2+ Kh1 Qf1+ Kh2 Qf2+, a perpetual. With the
        // two-fold rule that cycle scores as a draw 5 plies in; threefold-only
        // needs 9, which this budget does not reach.
        let p = pos("6k1/5ppp/R7/Q7/3q4/6PP/7K/8 b - - 0 1");
        let mut s = Search::new(CheckingMaterialEval, SearchParams::default());
        s.set_position(p, &[]);
        let r = s.go(3000, None, &AtomicBool::new(false)).unwrap();
        let best = r.best.to_uci(CastlingMode::Standard).to_string();
        assert_eq!(best, "d4f2", "should take the perpetual (value {})", r.value);
        assert!(r.value > -0.2, "perpetual should score near 0, got {}", r.value);
    }

    #[test]
    fn twofold_only_counts_repeats_inside_the_search() {
        use crate::encoding::search_terminal_value;
        let p = Chess::default(); // non-terminal; only the key sequence matters
        let (a, b, c, d) = (1u64, 2, 3, 4);
        // Cycle entirely after the root (root at index 0): draw.
        assert_eq!(search_terminal_value(&p, &[a, b, c, d, a], a, 0), Some(0.0));
        // The earlier occurrence predates the root: only one repeat, play on.
        assert_eq!(search_terminal_value(&p, &[a, b, c, d, a], a, 1), None);
        // ... unless it is the third occurrence overall.
        assert_eq!(search_terminal_value(&p, &[a, b, a, c, d, a], a, 3), Some(0.0));
        // No repeat at all.
        assert_eq!(search_terminal_value(&p, &[a, b, c, d], d, 0), None);
    }

    /// Tables from $SYZYGY_PATH (3-4-piece is enough for these tests).
    fn syzygy() -> Arc<Tablebase<Chess>> {
        let path = std::env::var("SYZYGY_PATH").expect("set SYZYGY_PATH to run tablebase tests");
        let mut tb = Tablebase::new();
        for dir in path.split(':') {
            tb.add_directory(dir).unwrap();
        }
        Arc::new(tb)
    }

    fn tb_params() -> SearchParams {
        SearchParams { syzygy: Some(syzygy()), ..SearchParams::default() }
    }

    /// The opponent's WDL after `m` is played from `p`.
    fn wdl_after(tb: &Tablebase<Chess>, p: &Chess, m: &Move) -> AmbiguousWdl {
        let mut after = p.clone();
        after.play_unchecked(m);
        tb.probe_wdl(&after).unwrap()
    }

    #[test]
    #[ignore = "needs SYZYGY_PATH"]
    fn tablebase_win_at_root_plays_without_search() {
        let p = pos("8/2K5/8/8/4k3/8/8/6R1 w - - 0 1"); // KRvK, DTZ 25
        let mut s = Search::new(UniformEval, tb_params());
        s.set_position(p.clone(), &[]);
        let r = s.go(800, None, &AtomicBool::new(false)).unwrap();
        assert_eq!((r.sims, r.tb_hits), (0, 1), "should not search a won TB root");
        assert_eq!(wdl_after(&syzygy(), &p, &r.best), AmbiguousWdl::Loss);
    }

    #[test]
    #[ignore = "needs SYZYGY_PATH"]
    fn tablebase_draw_at_root_keeps_the_draw() {
        // Drawn KRvKR where 11 of 19 moves lose, all of them quiet: in-tree
        // probes (zeroing moves only) cannot see these, root filtering must.
        let p = pos("8/7r/8/8/3K4/8/3k4/4R3 w - - 0 1");
        let tb = syzygy();
        let mut s = Search::new(UniformEval, tb_params());
        s.set_position(p.clone(), &[]);
        let r = s.go(400, None, &AtomicBool::new(false)).unwrap();
        assert_eq!(s.root_stats().len(), 8, "root should hold only the 8 drawing moves");
        assert!(
            !matches!(wdl_after(&tb, &p, &r.best), AmbiguousWdl::Win | AmbiguousWdl::MaybeWin),
            "played a losing move"
        );
    }

    #[test]
    #[ignore = "needs SYZYGY_PATH"]
    fn tablebase_scores_captures_into_covered_endgames() {
        // 5 pieces at the root (beyond 3-4-piece tables): searched normally,
        // but Kxd2/Qxd2 land in KQvKR and must be scored from the tables.
        let p = pos("8/8/8/3k4/8/2r5/3p4/3QK3 w - - 0 1");
        let mut s = Search::new(UniformEval, tb_params());
        s.set_position(p, &[]);
        let r = s.go(400, None, &AtomicBool::new(false)).unwrap();
        assert!(r.sims == 400 && r.tb_hits > 0, "sims {} tb hits {}", r.sims, r.tb_hits);
    }

    #[test]
    fn tree_reuse_reroots() {
        let start = Chess::default();
        let mut s = Search::new(UniformEval, SearchParams::default());
        s.set_position(start.clone(), &[]);
        s.go(128, None, &AtomicBool::new(false)).unwrap();
        let nodes_before = s.tree.len();
        assert!(nodes_before > 1);

        // Advance by one ply; the tree should re-root into the e2e4 subtree
        // (visited >5 times at 128 sims over 20 root moves), not reset.
        let m1 = mv(s.position(), "e2e4");
        s.set_position(start, &[m1]);
        assert!(
            s.tree[0].expanded,
            "re-rooted tree should keep the searched subtree"
        );
        assert!(s.tree.len() < nodes_before, "subtree must be a strict subset");
    }
}
