//! Self-play on the deployed bot's search: many `chess_core::selfplay` games in
//! parallel threads, their waves handed to Python as one batch. Same protocol
//! and game records as `Engine`, which keeps its own older search.

use std::collections::{HashMap, VecDeque};

use chess_core::search::EvalOutput;
use chess_core::selfplay::{SearchTally, SelfPlayGame, SelfPlayRules};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
            PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::Deserialize;
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess};

const POLICY: usize = 64 * 73;

/// Pad index of the legal-move array: one past the last logit, so a GPU gather
/// over logits with one appended column reads a dummy there.
const PAD: i16 = POLICY as i16;

/// The legal-move index rows of a batch, padded to the longest: (flat, width, lengths).
fn legal_rows<'a>(reqs: impl Iterator<Item = &'a chess_core::search::EvalRequest>) -> (Vec<i16>, usize, Vec<u16>) {
    let reqs: Vec<_> = reqs.collect();
    let width = reqs.iter().map(|r| r.legal.len()).max().unwrap_or(0).max(1);
    let mut flat = vec![PAD; reqs.len() * width];
    let mut lens = Vec::with_capacity(reqs.len());
    for (i, r) in reqs.iter().enumerate() {
        for (j, &m) in r.legal.iter().enumerate() {
            flat[i * width + j] = m as i16;
        }
        lens.push(r.legal.len() as u16);
    }
    (flat, width, lens)
}

// Unknown settings are an error: an engine built without a feature would otherwise
// ignore its setting silently (a build without forced playouts dropped them).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    num_sims: u32,
    #[serde(default = "d_games")]
    num_parallel_games: usize,
    #[serde(default = "d_batch")]
    sim_batch: usize,
    #[serde(default = "d_cpuct")]
    c_puct: f64,
    #[serde(default = "d_fpu")]
    fpu: f64,
    #[serde(default = "d_one")]
    prior_temp: f32,
    #[serde(default = "d_one_f64")]
    move_temp: f64,
    #[serde(default = "d_temp_plies")]
    temp_plies: u32,
    #[serde(default = "d_resign")]
    resign_threshold: f32,
    #[serde(default = "d_resign_plies")]
    resign_plies: u32,
    #[serde(default = "d_noresign")]
    no_resign_frac: f64,
    #[serde(default = "d_max_plies")]
    max_plies: u32,
    /// TT entries per game (it only caches evaluations).
    #[serde(default = "d_tt")]
    tt_cap: usize,
    #[serde(default)]
    fast_sims: u32,
    #[serde(default = "d_one_f64")]
    full_frac: f64,
    #[serde(default)]
    dirichlet_alpha: f64,
    #[serde(default = "d_dir_eps")]
    dirichlet_eps: f64,
    #[serde(default = "d_threads")]
    threads: usize,
    /// Count the reused tree's warm visits toward the full / fast budget.
    #[serde(default)]
    cap_total_full: bool,
    #[serde(default)]
    cap_total_fast: bool,
    /// The network reads the rule state: cache its evaluations per rule state.
    #[serde(default)]
    rule_state: bool,
    /// KataGo forced playouts + policy target pruning on full searches (k; 0 = off).
    #[serde(default)]
    forced_playouts: f64,
    /// Full searches start from a fresh tree (`SelfPlayRules::fresh_full`).
    #[serde(default)]
    fresh_full: bool,
    /// The two-fold rule over the game history (`SearchParams::twofold_history`).
    #[serde(default)]
    twofold_history: bool,
    /// Contempt for rule draws when ahead (`SearchParams::draw_contempt`; 0 = off).
    #[serde(default)]
    draw_contempt: f64,
    #[serde(default = "d_contempt_ramp")]
    contempt_ramp: f64,
    /// Measure the tree's depth after every search (`SelfPlayRules::depth_stats`).
    #[serde(default)]
    depth_stats: bool,
}
fn d_games() -> usize { 128 }
fn d_batch() -> usize { 16 }
fn d_cpuct() -> f64 { 1.0 }
fn d_fpu() -> f64 { 0.2 }
fn d_one() -> f32 { 1.0 }
fn d_one_f64() -> f64 { 1.0 }
fn d_temp_plies() -> u32 { 30 }
fn d_resign() -> f32 { 0.93 }
fn d_resign_plies() -> u32 { 6 }
fn d_noresign() -> f64 { 0.1 }
fn d_max_plies() -> u32 { 300 }
fn d_tt() -> usize { 100_000 }
fn d_dir_eps() -> f64 { 0.25 }
fn d_threads() -> usize { 6 }
fn d_contempt_ramp() -> f64 { 0.3 }

#[pyclass]
pub struct SelfPlayEngine {
    rules: SelfPlayRules,
    num_parallel_games: usize,
    threads: usize,
    openings: Vec<Chess>,
    opening_fens: Vec<String>,
    /// Game slots. `collect` closes the gap a finished game leaves; `collect_queued` empties
    /// its slot instead, as batches in flight hold slot numbers.
    games: Vec<Option<SelfPlayGame>>,
    finished: Vec<SelfPlayGame>,
    rng: StdRng,
    started: usize,
    completed: usize,
    target: usize,
    positions_recorded: usize,
    /// The finished games' searches, [fast, full].
    tally_done: [SearchTally; 2],
    /// (game slot, positions) of the last collect, in batch order.
    owners: Vec<(usize, usize)>,
    /// Legal-move count per position of the last collect, and the padded width.
    legal_lens: Vec<u16>,
    legal_width: usize,
    /// Rule state (halfmove clock, repetitions) per position of the last collect.
    rule50: Vec<u8>,
    repetition: Vec<u8>,
    /// Queued protocol: the slots whose positions wait for a batch, in arrival order, and
    /// per slot whether it waits there or in a batch on the GPU.
    queue: VecDeque<usize>,
    queued: Vec<bool>,
    in_flight: Vec<bool>,
    /// The batches handed out by `collect_queued` and not applied yet, by id.
    batches: HashMap<u64, Batch>,
    next_batch: u64,
}

/// One batch of the queued protocol: (slot, positions) in batch order, each position's
/// legal-move count and the padded width.
struct Batch {
    owners: Vec<(usize, usize)>,
    lens: Vec<u16>,
    width: usize,
}

/// The NN inputs of a batch, in its order.
struct Rows {
    boards: Vec<u8>,
    player: Vec<u8>,
    castling: Vec<u8>,
    ep: Vec<u8>,
    rule50: Vec<u8>,
    repetition: Vec<u8>,
    legal: Vec<i16>,
    width: usize,
    lens: Vec<u16>,
}

impl SelfPlayEngine {
    /// Start games until `num_parallel_games` play, into emptied slots first. Every new game
    /// takes the next (opening, seed) draw, so game k is the same game whatever the protocol.
    fn fill(&mut self) {
        for i in 0..self.games.len() {
            if self.games[i].is_none() && self.started < self.target {
                self.games[i] = Some(self.spawn());
            }
        }
        while self.games.len() < self.num_parallel_games && self.started < self.target {
            let g = self.spawn();
            self.games.push(Some(g));
        }
    }

    fn spawn(&mut self) -> SelfPlayGame {
        let oi = self.rng.gen_range(0..self.openings.len());
        let seed = self.rng.gen();
        let g = SelfPlayGame::new(self.started, oi, self.openings[oi].clone(), &self.rules, seed);
        self.started += 1;
        g
    }

    fn game(&self, slot: usize) -> &SelfPlayGame {
        self.games[slot].as_ref().expect("live game slot")
    }

    /// Advance every game until it waits on an eval or ends, retire the finished ones and
    /// start their successors, until no game finishes. `in_place`: empty a finished game's
    /// slot instead of closing the gap.
    fn advance_all(&mut self, in_place: bool) -> PyResult<()> {
        loop {
            let rules = self.rules;
            self.par_games(|_, g| g.advance(&rules));
            if let Some(g) = self.games.iter().flatten().find(|g| g.stalled()) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "self-play engine stalled in game {}", g.id)));
            }
            let mut retired = false;
            let mut i = 0;
            while i < self.games.len() {
                if self.games[i].as_ref().is_some_and(|g| g.end().is_some()) {
                    let g = if in_place { self.games[i].take() } else { self.games.swap_remove(i) }
                        .expect("finished game");
                    self.positions_recorded += g.records().len();
                    for (done, t) in self.tally_done.iter_mut().zip(g.tally()) {
                        done.add(t);
                    }
                    self.completed += 1;
                    self.finished.push(g);
                    retired = true;
                    if in_place {
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }
            self.fill();
            if !retired || self.games.iter().all(Option::is_none) {
                return Ok(());
            }
        }
    }

    /// The pending positions of `owners`' games, in order.
    fn rows(&self, owners: &[(usize, usize)]) -> Rows {
        let (mut boards, mut player, mut castling, mut ep) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut rule50, mut repetition) = (Vec::new(), Vec::new());
        let reqs = || owners.iter().flat_map(|&(gi, _)| self.game(gi).pending().unwrap().iter());
        for r in reqs() {
            boards.extend_from_slice(&r.tokens);
            player.push(r.player);
            castling.push(r.castling);
            ep.push(r.ep);
            rule50.push(r.rule50);
            repetition.push(r.repetition);
        }
        let (legal, width, lens) = legal_rows(reqs());
        Rows { boards, player, castling, ep, rule50, repetition, legal, width, lens }
    }

    /// Resolve the games of `owners` with the gathered legal-move logits (rows of `width`)
    /// and values of their positions, in order.
    fn resolve(&mut self, owners: &[(usize, usize)], lens: &[u16], width: usize, flat: &[f32], values: &[f32]) {
        let mut offset = vec![None; self.games.len()];
        let mut at = 0;
        for &(gi, k) in owners {
            offset[gi] = Some((at, k));
            at += k;
        }
        let offset = &offset;
        self.par_games(|gi, g| {
            let Some((at, k)) = offset[gi] else { return };
            let outputs: Vec<EvalOutput> = (at..at + k)
                .map(|i| EvalOutput { logits: flat[i * width..i * width + lens[i] as usize].to_vec(),
                                      value: values[i] })
                .collect();
            g.resolve(outputs);
        });
    }

    /// Run `f` over every game on `threads` threads, which pull small chunks as they finish:
    /// a game's wave costs very different time (a move boundary records the move and re-roots
    /// the tree), and with one fixed slice per thread the wave waited for the slowest slice.
    fn par_games<F>(&mut self, f: F)
    where
        F: Fn(usize, &mut SelfPlayGame) + Sync,
    {
        const GRAIN: usize = 2;
        let n = self.games.len();
        let threads = self.threads.max(1).min(n.div_ceil(GRAIN)).max(1);
        if threads == 1 {
            for (i, g) in self.games.iter_mut().enumerate() {
                if let Some(g) = g {
                    f(i, g);
                }
            }
            return;
        }
        let work = std::sync::Mutex::new(self.games.chunks_mut(GRAIN).enumerate());
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let (f, work) = (&f, &work);
                scope.spawn(move || loop {
                    let next = work.lock().unwrap().next();
                    let Some((ci, games)) = next else { break };
                    for (j, g) in games.iter_mut().enumerate() {
                        if let Some(g) = g {
                            f(ci * GRAIN + j, g);
                        }
                    }
                });
            }
        });
    }
}

#[pymethods]
impl SelfPlayEngine {
    #[new]
    fn new(config_json: &str, openings: Vec<String>, seed: u64, target: usize) -> PyResult<Self> {
        let c: Config = serde_json::from_str(config_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("config: {e}")))?;
        let mut parsed = Vec::new();
        for fen in &openings {
            let f: Fen = fen.parse()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fen {fen}: {e}")))?;
            let pos: Chess = f.into_position(CastlingMode::Standard)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fen {fen}: {e}")))?;
            parsed.push(pos);
        }
        let rules = SelfPlayRules {
            num_sims: c.num_sims, fast_sims: c.fast_sims, full_frac: c.full_frac,
            dirichlet_alpha: c.dirichlet_alpha, dirichlet_eps: c.dirichlet_eps,
            move_temp: c.move_temp, temp_plies: c.temp_plies,
            resign_threshold: c.resign_threshold, resign_plies: c.resign_plies,
            no_resign_frac: c.no_resign_frac, max_plies: c.max_plies, sim_batch: c.sim_batch,
            c_puct: c.c_puct, fpu: c.fpu, prior_temp: c.prior_temp, tt_cap: c.tt_cap,
            cap_total_full: c.cap_total_full, cap_total_fast: c.cap_total_fast,
            rule_state: c.rule_state, forced_playouts: c.forced_playouts, fresh_full: c.fresh_full,
            twofold_history: c.twofold_history, draw_contempt: c.draw_contempt,
            contempt_ramp: c.contempt_ramp, depth_stats: c.depth_stats,
        };
        let mut e = SelfPlayEngine {
            rules,
            num_parallel_games: c.num_parallel_games.min(target.max(1)),
            threads: c.threads,
            openings: parsed,
            opening_fens: openings,
            games: Vec::new(),
            finished: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            started: 0,
            completed: 0,
            target,
            positions_recorded: 0,
            tally_done: [SearchTally::default(); 2],
            owners: Vec::new(),
            legal_lens: Vec::new(),
            legal_width: 1,
            rule50: Vec::new(),
            repetition: Vec::new(),
            queue: VecDeque::new(),
            queued: Vec::new(),
            in_flight: Vec::new(),
            batches: HashMap::new(),
            next_batch: 0,
        };
        e.fill();
        Ok(e)
    }

    /// Advance every game to its next NN wave; returns (boards, player,
    /// castling, ep, legal) of all pending positions, `legal` being the legal
    /// moves' flat logit indices per position, padded with 4672.
    #[allow(clippy::type_complexity)]
    fn collect<'py>(&mut self, py: Python<'py>) -> PyResult<(
        Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>, Bound<'py, PyArray2<i16>>,
    )> {
        self.advance_all(false)?;
        self.owners = self.games.iter().enumerate()
            .filter_map(|(gi, g)| Some((gi, g.as_ref()?.pending()?.len())))
            .collect();
        let r = self.rows(&self.owners);
        let n = r.player.len();
        self.legal_lens = r.lens;
        self.legal_width = r.width;
        self.rule50 = r.rule50;
        self.repetition = r.repetition;
        Ok((
            PyArray1::from_vec(py, r.boards).reshape([n, 64]).expect("reshape"),
            r.player.into_pyarray(py),
            r.castling.into_pyarray(py),
            r.ep.into_pyarray(py),
            PyArray1::from_vec(py, r.legal).reshape([n, r.width]).expect("reshape"),
        ))
    }

    /// The queued protocol, where up to `batches in flight` batches of one bucket each keep
    /// the GPU fed without lockstep waves: advance every game whose evaluation is back, queue
    /// its positions, and hand out the oldest queued ones as batch `id`, whole games only, at
    /// most `max_rows` positions (a single game bigger than that goes alone). Returns (id,
    /// boards, player, castling, ep, legal), as `collect` plus the id `apply_queued` takes; an
    /// empty batch (no id taken, nothing to apply) means every game waits on a batch in
    /// flight. Games are the same games as under `collect`; one engine uses one protocol.
    #[allow(clippy::type_complexity)]
    fn collect_queued<'py>(&mut self, py: Python<'py>, max_rows: usize) -> PyResult<(
        u64, Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>, Bound<'py, PyArray2<i16>>,
    )> {
        self.advance_all(true)?;
        let n_slots = self.games.len();
        self.queued.resize(n_slots, false);
        self.in_flight.resize(n_slots, false);
        for (gi, g) in self.games.iter().enumerate() {
            if g.as_ref().is_some_and(|g| g.pending().is_some()) && !self.queued[gi] && !self.in_flight[gi] {
                self.queue.push_back(gi);
                self.queued[gi] = true;
            }
        }
        let mut owners = Vec::new();
        let mut rows = 0;
        while let Some(&gi) = self.queue.front() {
            let k = self.game(gi).pending().expect("queued game waits on an eval").len();
            if rows > 0 && rows + k > max_rows {
                break;
            }
            self.queue.pop_front();
            self.queued[gi] = false;
            self.in_flight[gi] = true;
            owners.push((gi, k));
            rows += k;
        }
        let r = self.rows(&owners);
        let n = r.player.len();
        let id = self.next_batch;
        if n > 0 {
            self.next_batch += 1;
            self.batches.insert(id, Batch { owners, lens: r.lens, width: r.width });
        }
        self.rule50 = r.rule50;
        self.repetition = r.repetition;
        Ok((
            id,
            PyArray1::from_vec(py, r.boards).reshape([n, 64]).expect("reshape"),
            r.player.into_pyarray(py),
            r.castling.into_pyarray(py),
            r.ep.into_pyarray(py),
            PyArray1::from_vec(py, r.legal).reshape([n, r.width]).expect("reshape"),
        ))
    }

    /// `apply_legal` for batch `id` of `collect_queued`, in any order of the batches.
    fn apply_queued(&mut self, id: u64, logits: PyReadonlyArray2<f32>, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let Some(b) = self.batches.remove(&id) else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "apply_queued: no batch {id} in flight")));
        };
        let logits = logits.as_array();
        let values = values.as_slice()?;
        let total = b.lens.len();
        if values.len() != total || logits.shape() != [total, b.width] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "apply_queued: got {} values / logits {:?} for batch {id} of {} positions of width {}",
                values.len(), logits.shape(), total, b.width)));
        }
        let flat = logits.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("apply_queued: logits must be C-contiguous")
        })?;
        self.resolve(&b.owners, &b.lens, b.width, flat, values);
        for &(gi, _) in &b.owners {
            self.in_flight[gi] = false;
        }
        Ok(())
    }

    /// Batches handed out by `collect_queued` and not applied yet.
    fn batches_in_flight(&self) -> usize {
        self.batches.len()
    }

    /// (rule50, repetition) of the last collect's positions, in its order -- the extra
    /// inputs of a network with rule-state inputs.
    #[allow(clippy::type_complexity)]
    fn rule_state<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>) {
        (PyArray1::from_slice(py, &self.rule50), PyArray1::from_slice(py, &self.repetition))
    }

    /// Like `apply`, with the legal moves' logits already gathered: row i holds
    /// the logits at the indices of the last collect's `legal` row i.
    fn apply_legal(&mut self, logits: PyReadonlyArray2<f32>, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let logits = logits.as_array();
        let values = values.as_slice()?;
        let total = self.legal_lens.len();
        let width = self.legal_width;
        if values.len() != total || logits.shape() != [total, width] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "apply_legal: got {} values / logits {:?} for {} pending positions of width {}",
                values.len(), logits.shape(), total, width)));
        }
        let flat = logits.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("apply_legal: logits must be C-contiguous")
        })?;
        let lens = std::mem::take(&mut self.legal_lens);
        let owners = std::mem::take(&mut self.owners);
        self.resolve(&owners, &lens, width, flat, values);
        Ok(())
    }

    /// Outputs for the batch of the last collect(), in its order.
    fn apply(&mut self, logits: PyReadonlyArray3<f32>, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let logits = logits.as_array();
        let values = values.as_slice()?;
        let total: usize = self.owners.iter().map(|o| o.1).sum();
        if values.len() != total || logits.shape() != [total, 64, 73] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "apply: got {} values / logits {:?} for {} pending positions",
                values.len(), logits.shape(), total)));
        }
        let flat = logits.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("apply: logits must be C-contiguous")
        })?;
        let mut offset = vec![None; self.games.len()];
        let mut at = 0;
        for &(gi, k) in &self.owners {
            offset[gi] = Some((at, k));
            at += k;
        }
        let offset = &offset;
        self.par_games(|gi, g| {
            let Some((at, k)) = offset[gi] else { return };
            let outputs: Vec<EvalOutput> = (at..at + k)
                .map(|i| EvalOutput { logits: flat[i * POLICY..(i + 1) * POLICY].to_vec(),
                                      value: values[i] })
                .collect();
            g.resolve(outputs);
        });
        self.owners.clear();
        Ok(())
    }

    /// Completed games since the last call, as dicts of numpy arrays (the
    /// `Engine.drain_finished` layout).
    fn drain_finished<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut out = Vec::new();
        for g in std::mem::take(&mut self.finished) {
            let (z_white, end) = g.end().expect("finished game has a result");
            let recs = g.records();
            let p = recs.len();
            let mut boards = Vec::with_capacity(p * 64);
            let (mut player, mut castling, mut ep) = (Vec::new(), Vec::new(), Vec::new());
            let (mut halfmove, mut root_v, mut z, mut full) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let (mut rule50, mut repetition) = (Vec::new(), Vec::new());
            let (mut visit_idx, mut visit_cnt) = (Vec::new(), Vec::new());
            let mut visit_ptr = vec![0i64];
            let (mut legal_idx, mut legal_prior, mut legal_n, mut legal_q) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let (mut legal_ptr, mut root_net_v) = (vec![0i64], Vec::new());
            for r in recs {
                boards.extend_from_slice(&r.tokens);
                player.push(r.player);
                castling.push(r.castling);
                ep.push(r.ep);
                halfmove.push(r.halfmove);
                rule50.push(r.rule50);
                repetition.push(r.repetition);
                root_v.push(r.root_v);
                full.push(r.full);
                z.push(if r.player == 1 { z_white } else { -z_white });
                visit_idx.extend_from_slice(&r.visit_idx);
                visit_cnt.extend_from_slice(&r.visit_cnt);
                visit_ptr.push(visit_idx.len() as i64);
                legal_idx.extend_from_slice(&r.legal_idx);
                legal_prior.extend_from_slice(&r.legal_prior);
                legal_n.extend_from_slice(&r.legal_n);
                legal_q.extend_from_slice(&r.legal_q);
                legal_ptr.push(legal_idx.len() as i64);
                root_net_v.push(r.root_net_v);
            }
            let d = PyDict::new(py);
            d.set_item("boards", PyArray1::from_vec(py, boards).reshape([p, 64]).expect("reshape"))?;
            d.set_item("player", player.into_pyarray(py))?;
            d.set_item("castling", castling.into_pyarray(py))?;
            d.set_item("ep", ep.into_pyarray(py))?;
            d.set_item("halfmove", halfmove.into_pyarray(py))?;
            d.set_item("rule50", rule50.into_pyarray(py))?;
            d.set_item("repetition", repetition.into_pyarray(py))?;
            d.set_item("root_v", root_v.into_pyarray(py))?;
            d.set_item("full", full.into_pyarray(py))?;
            d.set_item("z", z.into_pyarray(py))?;
            d.set_item("visit_idx", visit_idx.into_pyarray(py))?;
            d.set_item("visit_cnt", visit_cnt.into_pyarray(py))?;
            d.set_item("visit_ptr", visit_ptr.into_pyarray(py))?;
            d.set_item("legal_idx", legal_idx.into_pyarray(py))?;
            d.set_item("legal_prior", legal_prior.into_pyarray(py))?;
            d.set_item("legal_n", legal_n.into_pyarray(py))?;
            d.set_item("legal_q", legal_q.into_pyarray(py))?;
            d.set_item("legal_ptr", legal_ptr.into_pyarray(py))?;
            d.set_item("root_net_v", root_net_v.into_pyarray(py))?;
            d.set_item("moves", g.moves_uci().to_vec())?;
            d.set_item("fen", &self.opening_fens[g.opening])?;
            d.set_item("z_white", z_white)?;
            d.set_item("end", end)?;
            d.set_item("resign_allowed", g.resign_allowed())?;
            d.set_item("id", g.id)?;
            out.push(d);
        }
        Ok(out)
    }

    fn done(&self) -> bool {
        self.completed >= self.target
    }

    /// (games started, games completed, positions recorded, TT entries, positions searched):
    /// the last counts the games in flight too, a throughput that needs no game to finish.
    fn stats(&self) -> (usize, usize, usize, usize, usize) {
        let tt = self.games.iter().flatten().map(|g| g.tt_len()).sum();
        let in_flight: usize = self.games.iter().flatten().map(|g| g.records().len()).sum();
        (self.started, self.completed, self.positions_recorded, tt, self.positions_recorded + in_flight)
    }

    /// The searches so far, games in flight included, as [fast, full] dicts of sums: moves,
    /// sims, nn_evals, tt_hits, depth_mean, depth_max, main_line (the depths 0 unless
    /// `depth_stats`).
    fn search_stats<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut sum = self.tally_done;
        for g in self.games.iter().flatten() {
            for (s, t) in sum.iter_mut().zip(g.tally()) {
                s.add(t);
            }
        }
        sum.iter().map(|t| {
            let d = PyDict::new(py);
            d.set_item("moves", t.moves)?;
            d.set_item("sims", t.sims)?;
            d.set_item("nn_evals", t.nn_evals)?;
            d.set_item("tt_hits", t.tt_hits)?;
            d.set_item("depth_mean", t.depth_mean)?;
            d.set_item("depth_max", t.depth_max)?;
            d.set_item("main_line", t.main_line)?;
            Ok(d)
        }).collect()
    }
}
