//! Batched matches between two networks, played with the deployed bot's search
//! and move flow (`chess_core::matchplay`: tree reuse, the per-move budget
//! rules and pondering of ct-bot's lichess loop). Every side of every game is
//! a `chess_core::search::Search` driven one wave at a time; this wrapper runs
//! the games in parallel and hands their waves to Python, one forward per
//! network for all games, instead of ct-bot's 16 positions per call.
//!
//! Not modelled: the clock (moves play at the per-speed cap, as 81-100 % of
//! live moves do), the opening temperature (a book start is past ply 2), int8
//! numerics (the forward runs in Python).

use std::collections::HashMap;

use chess_core::matchplay::{MatchGame, MatchRules, SideParams};
use chess_core::search::EvalOutput;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
            PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)] // as SelfPlayEngine's Config: a setting this build lacks is an error
struct MatchConfig {
    /// Per-move fresh-sims cap (ct-bot `--max-sims`, 4000 for bullet/blitz).
    #[serde(default = "d_sims")]
    sims: u32,
    /// Ponder budget (ct-bot `--ponder-max-sims`); 0 = no pondering.
    #[serde(default = "d_ponder")]
    ponder_sims: u32,
    /// Total root visits allowed above `sims` (ct-bot PONDER_HEADROOM).
    #[serde(default = "d_headroom")]
    headroom: u32,
    #[serde(default = "d_batch")]
    sim_batch: usize,
    #[serde(default = "d_cpuct")]
    c_puct: f64,
    #[serde(default = "d_fpu")]
    fpu: f64,
    #[serde(default = "d_one")]
    prior_temp: f32,
    /// Per-net overrides of the three search settings above (net A / net B).
    #[serde(default)]
    c_puct_a: Option<f64>,
    #[serde(default)]
    c_puct_b: Option<f64>,
    #[serde(default)]
    fpu_a: Option<f64>,
    #[serde(default)]
    fpu_b: Option<f64>,
    #[serde(default)]
    prior_temp_a: Option<f32>,
    #[serde(default)]
    prior_temp_b: Option<f32>,
    /// TT entries per search. ct-bot's 2M would need tens of GB across hundreds
    /// of searches; the TT only caches evaluations, so a smaller one changes the
    /// GPU work, not the search.
    #[serde(default = "d_tt")]
    tt_cap: usize,
    #[serde(default = "d_games")]
    num_parallel_games: usize,
    #[serde(default = "d_max_plies")]
    max_plies: u32,
    #[serde(default = "d_threads")]
    threads: usize,
    /// Net A / net B reads the rule state: cache its evaluations per rule state.
    #[serde(default)]
    rule_state_a: bool,
    #[serde(default)]
    rule_state_b: bool,
}
fn d_sims() -> u32 { 4000 }
fn d_ponder() -> u32 { 2000 }
fn d_headroom() -> u32 { chess_core::search::PONDER_HEADROOM }
fn d_batch() -> usize { 16 }
fn d_cpuct() -> f64 { 1.0 }
fn d_fpu() -> f64 { 0.2 }
fn d_one() -> f32 { 1.0 }
fn d_tt() -> usize { 150_000 }
fn d_games() -> usize { 128 }
fn d_max_plies() -> u32 { 200 }
fn d_threads() -> usize { 8 }

#[pyclass]
pub struct MatchEngine {
    cfg: MatchConfig,
    rules: MatchRules,
    params: SideParams,
    fens: Vec<String>,
    starts: Vec<Chess>,
    a_white: Vec<bool>,
    games: Vec<MatchGame>,
    started: usize,
    completed: usize,
    finished: Vec<MatchGame>,
    /// (game slot, side, positions) of the last collect, in batch order.
    owners: Vec<(usize, usize, usize)>,
    /// Legal-move count per position of the last collect, and the padded width.
    legal_lens: Vec<u16>,
    legal_width: usize,
    /// Rule state (halfmove clock, repetitions) per position of the last collect.
    rule50: Vec<u8>,
    repetition: Vec<u8>,
}

impl MatchEngine {
    fn fill(&mut self) {
        while self.games.len() < self.cfg.num_parallel_games && self.started < self.fens.len() {
            let i = self.started;
            self.started += 1;
            self.games.push(MatchGame::new(i, self.starts[i].clone(), self.a_white[i], self.params));
        }
    }

    /// Run `f` over every game on `threads` threads, which pull small chunks as they finish:
    /// a game's wave costs very different time (a move boundary records the move and re-roots
    /// the tree), and with one fixed slice per thread the wave waited for the slowest slice.
    fn par_games<F>(&mut self, f: F)
    where
        F: Fn(usize, &mut MatchGame) + Sync,
    {
        const GRAIN: usize = 2;
        let n = self.games.len();
        let threads = self.cfg.threads.max(1).min(n.div_ceil(GRAIN)).max(1);
        if threads == 1 {
            for (i, g) in self.games.iter_mut().enumerate() {
                f(i, g);
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
                        f(ci * GRAIN + j, g);
                    }
                });
            }
        });
    }
}

#[pymethods]
impl MatchEngine {
    /// Game i starts from `fens[i]` with network A white iff `a_white[i]`.
    #[new]
    fn new(config_json: &str, fens: Vec<String>, a_white: Vec<bool>) -> PyResult<Self> {
        let cfg: MatchConfig = serde_json::from_str(config_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("config: {e}")))?;
        if fens.len() != a_white.len() || fens.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "need one a_white flag per opening, and at least one game"));
        }
        let mut starts = Vec::with_capacity(fens.len());
        for fen in &fens {
            let f: Fen = fen.parse()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fen {fen}: {e}")))?;
            let pos: Chess = f.into_position(CastlingMode::Standard)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fen {fen}: {e}")))?;
            starts.push(pos);
        }
        let rules = MatchRules {
            sims: cfg.sims,
            ponder_sims: cfg.ponder_sims,
            total_cap: cfg.sims + cfg.headroom,
            max_plies: cfg.max_plies,
        };
        let params = SideParams {
            c_puct: [cfg.c_puct_a.unwrap_or(cfg.c_puct), cfg.c_puct_b.unwrap_or(cfg.c_puct)],
            fpu: [cfg.fpu_a.unwrap_or(cfg.fpu), cfg.fpu_b.unwrap_or(cfg.fpu)],
            prior_temp: [cfg.prior_temp_a.unwrap_or(cfg.prior_temp),
                         cfg.prior_temp_b.unwrap_or(cfg.prior_temp)],
            sim_batch: cfg.sim_batch,
            tt_cap: cfg.tt_cap,
            rule_state: [cfg.rule_state_a, cfg.rule_state_b],
        };
        let mut m = MatchEngine {
            cfg, rules, params, fens, starts, a_white, games: Vec::new(), started: 0,
            completed: 0, finished: Vec::new(), owners: Vec::new(), legal_lens: Vec::new(),
            legal_width: 1, rule50: Vec::new(), repetition: Vec::new(),
        };
        m.fill();
        Ok(m)
    }

    /// Advance every search to its next NN wave; returns the batch (boards,
    /// player, castling, ep, net, legal) with net 0 = A, 1 = B and `legal` the
    /// legal moves' flat logit indices per position, padded with 4672.
    #[allow(clippy::type_complexity)]
    fn collect<'py>(&mut self, py: Python<'py>) -> PyResult<(
        Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray2<i16>>,
    )> {
        loop {
            let rules = self.rules;
            self.par_games(|_, g| g.advance(&rules));
            if let Some(g) = self.games.iter().find(|g| g.stalled()) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "match engine stalled in game {}", g.id)));
            }
            // Retire finished games and refill from the schedule; new games need
            // advancing before the batch goes out.
            let before = self.games.len();
            let mut i = 0;
            while i < self.games.len() {
                if self.games[i].end().is_some() {
                    self.finished.push(self.games.swap_remove(i));
                    self.completed += 1;
                } else {
                    i += 1;
                }
            }
            let retired = before != self.games.len();
            self.fill();
            if !retired || self.games.is_empty() {
                break;
            }
        }

        self.owners.clear();
        self.rule50.clear();
        self.repetition.clear();
        let (mut boards, mut player, mut castling, mut ep, mut net) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (gi, g) in self.games.iter().enumerate() {
            for s in 0..2 {
                if let Some(reqs) = g.pending(s) {
                    for r in reqs {
                        boards.extend_from_slice(&r.tokens);
                        player.push(r.player);
                        castling.push(r.castling);
                        ep.push(r.ep);
                        net.push(s as u8);
                        self.rule50.push(r.rule50);
                        self.repetition.push(r.repetition);
                    }
                    self.owners.push((gi, s, reqs.len()));
                }
            }
        }
        let n = player.len();
        let (legal, width, lens) = legal_rows(
            self.owners.iter().flat_map(|&(gi, s, _)| self.games[gi].pending(s).unwrap().iter()));
        self.legal_lens = lens;
        self.legal_width = width;
        Ok((
            PyArray1::from_vec(py, boards).reshape([n, 64]).expect("reshape"),
            player.into_pyarray(py),
            castling.into_pyarray(py),
            ep.into_pyarray(py),
            net.into_pyarray(py),
            PyArray1::from_vec(py, legal).reshape([n, width]).expect("reshape"),
        ))
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
        let lens = &lens;
        let mut offsets: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        let mut at = 0;
        for &(gi, s, k) in &self.owners {
            offsets.insert((gi, s), (at, k));
            at += k;
        }
        let offsets = &offsets;
        self.par_games(|gi, g| {
            for s in 0..2 {
                let Some(&(at, k)) = offsets.get(&(gi, s)) else { continue };
                let outputs: Vec<EvalOutput> = (at..at + k)
                    .map(|i| EvalOutput {
                        logits: flat[i * width..i * width + lens[i] as usize].to_vec(),
                        value: values[i],
                    })
                    .collect();
                g.resolve(s, outputs);
            }
        });
        self.owners.clear();
        Ok(())
    }

    /// Outputs for the batch of the last collect(), in its order.
    fn apply(&mut self, logits: PyReadonlyArray3<f32>, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let logits = logits.as_array();
        let values = values.as_slice()?;
        let total: usize = self.owners.iter().map(|o| o.2).sum();
        if values.len() != total || logits.shape() != [total, 64, 73] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "apply: got {} values / logits {:?} for {} pending positions",
                values.len(), logits.shape(), total)));
        }
        let flat = logits.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("apply: logits must be C-contiguous")
        })?;
        let mut offsets: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        let mut at = 0;
        for &(gi, s, k) in &self.owners {
            offsets.insert((gi, s), (at, k));
            at += k;
        }
        let offsets = &offsets;
        self.par_games(|gi, g| {
            for s in 0..2 {
                let Some(&(at, k)) = offsets.get(&(gi, s)) else { continue };
                let outputs: Vec<EvalOutput> = (at..at + k)
                    .map(|i| EvalOutput {
                        logits: flat[i * POLICY..(i + 1) * POLICY].to_vec(),
                        value: values[i],
                    })
                    .collect();
                g.resolve(s, outputs);
            }
        });
        self.owners.clear();
        Ok(())
    }

    /// Completed games since the last call.
    fn drain_finished<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut out = Vec::new();
        for g in std::mem::take(&mut self.finished) {
            let (z, end) = g.end().expect("finished game has a result");
            let d = PyDict::new(py);
            d.set_item("id", g.id)?;
            d.set_item("fen", &self.fens[g.id])?;
            d.set_item("a_white", g.a_white)?;
            d.set_item("moves", g.moves_uci().to_vec())?;
            d.set_item("z_white", z)?;
            d.set_item("end", end)?;
            out.push(d);
        }
        Ok(out)
    }

    fn done(&self) -> bool {
        self.completed >= self.fens.len()
    }

    /// (games started, games completed)
    fn stats(&self) -> (usize, usize) {
        (self.started, self.completed)
    }
}
