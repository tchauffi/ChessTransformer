//! Self-play generation core: cross-game batched MCTS over shakmaty boards.
//!
//! Python keeps only the batched NN forward; this crate owns boards, search
//! trees and game lifecycle for many concurrent games. Protocol per round:
//!
//!   boards, player, castling, ep = engine.collect()   # deduped NN-pending positions
//!   logits, values = nn_forward(...)                  # (n, 64, 73) f32, (n,) f32
//!   engine.apply(logits, values)                      # expand/backup, play ready moves
//!   for game in engine.drain_finished(): ...          # completed game records
//!
//! Search semantics mirror `Pos2MoveV2MctsBot` (PUCT + FPU, virtual loss,
//! negamax backup, terminal rules incl. in-search threefold repetition).
//! Encodings and tree primitives live in the shared `chess-core` crate —
//! see the parity test.

use std::collections::HashMap;

use chess_core::encoding::{
    action_plane, encode_position, halfmove, move_coords, terminal_value, zobrist, NUM_PLANES,
};
use chess_core::tree::{backup_vl, puct_select, softmax_priors, Node};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::Deserialize;
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Color, Move, Position};

// ── per-game state ───────────────────────────────────────────────────────

struct Rec {
    tokens: [u8; 64],
    player: u8,
    castling: u8,
    ep: u8,
    halfmove: u16,
    root_v: f32,
    visit_idx: Vec<u16>, // from_sq * 73 + plane
    visit_cnt: Vec<u16>,
}

struct Game {
    id: usize,
    start_fen: String,
    pos: Chess,
    history: Vec<u64>, // zobrist of every position incl. start
    tree: Vec<Node>,
    sims_done: u32,
    inflight: u32,
    records: Vec<Rec>,
    moves_uci: Vec<String>,
    resign_allowed: bool,
    resign_streak: u32,
    resign_sign: i32,
    plies: u32,
    active: bool,
}

struct Finished {
    records: Vec<Rec>,
    moves_uci: Vec<String>,
    start_fen: String,
    z_white: f32,
    end: &'static str,
    resign_allowed: bool,
    id: usize,
}

struct NnItem {
    key: u64,
    tokens: [u8; 64],
    player: u8,
    castling: u8,
    ep: u8,
    move_fp: Vec<(usize, usize)>, // (from_sq, plane) per legal move, in move-list order
}

struct PendingLeaf {
    game: usize,
    node: usize,
    path: Vec<(usize, usize)>, // (node_idx, child_idx)
    key: u64,
    moves: Vec<Move>,
}

#[derive(Deserialize)]
struct Config {
    num_sims: u32,
    #[serde(default = "d_games")] num_parallel_games: usize,
    #[serde(default = "d_leaves")] leaves_per_wave: u32,
    #[serde(default = "d_cpuct")] c_puct: f64,
    #[serde(default = "d_fpu")] fpu: f64,
    #[serde(default = "d_one")] prior_temp: f32,
    #[serde(default = "d_one_f64")] move_temp: f64,
    #[serde(default = "d_temp_plies")] temp_plies: u32,
    #[serde(default = "d_resign")] resign_threshold: f32,
    #[serde(default = "d_resign_plies")] resign_plies: u32,
    #[serde(default = "d_noresign")] no_resign_frac: f64,
    #[serde(default = "d_max_plies")] max_plies: u32,
    #[serde(default = "d_tt_cap")] tt_cap: usize,
}
fn d_games() -> usize { 64 }
fn d_leaves() -> u32 { 4 }
fn d_cpuct() -> f64 { 1.0 }
fn d_fpu() -> f64 { 0.2 }
fn d_one() -> f32 { 1.0 }
fn d_one_f64() -> f64 { 1.0 }
fn d_temp_plies() -> u32 { 30 }
fn d_resign() -> f32 { 0.93 }
fn d_resign_plies() -> u32 { 6 }
fn d_noresign() -> f64 { 0.1 }
fn d_max_plies() -> u32 { 300 }
fn d_tt_cap() -> usize { 2_000_000 }

// move_temp is deserialized as f64 for sampling math convenience.

#[pyclass]
struct Engine {
    cfg: Config,
    openings: Vec<Chess>,
    opening_fens: Vec<String>,
    games: Vec<Game>,
    pending: Vec<PendingLeaf>,
    nn_items: Vec<NnItem>,
    nn_index: HashMap<u64, usize>,
    tt: HashMap<u64, (Vec<f32>, f32)>, // priors (legal-move order) + stm value
    finished: Vec<Finished>,
    rng: StdRng,
    started: usize,
    completed: usize,
    target: usize,
    positions_recorded: usize,
}

impl Engine {
    fn fresh_tree() -> Vec<Node> {
        vec![Node::default()]
    }

    fn new_game(&mut self) -> Game {
        let oi = self.rng.gen_range(0..self.openings.len());
        let pos = self.openings[oi].clone();
        let id = self.started;
        self.started += 1;
        Game {
            id,
            start_fen: self.opening_fens[oi].clone(),
            history: vec![zobrist(&pos)],
            pos,
            tree: Self::fresh_tree(),
            sims_done: 0,
            inflight: 0,
            records: Vec::new(),
            moves_uci: Vec::new(),
            resign_allowed: self.rng.gen::<f64>() >= self.cfg.no_resign_frac,
            resign_streak: 0,
            resign_sign: 0,
            plies: 0,
            active: true,
        }
    }

    /// Descend with virtual loss to an unexpanded node. Returns the leaf node
    /// index, the path, the leaf position and the zobrist keys pushed en route.
    fn descend(&mut self, gi: usize) -> (usize, Vec<(usize, usize)>, Chess, Vec<u64>) {
        let mut node_idx = 0usize;
        let mut pos = self.games[gi].pos.clone();
        let mut path = Vec::new();
        let mut keys = Vec::new();
        loop {
            if !self.games[gi].tree[node_idx].expanded {
                return (node_idx, path, pos, keys);
            }
            let idx = puct_select(&self.games[gi].tree[node_idx], self.cfg.c_puct, self.cfg.fpu);
            {
                let node = &mut self.games[gi].tree[node_idx];
                node.n[idx] += 1; // virtual loss
                node.w[idx] += 1.0;
            }
            path.push((node_idx, idx));
            let mv = self.games[gi].tree[node_idx].moves[idx].clone();
            pos.play_unchecked(&mv);
            keys.push(zobrist(&pos));
            let child = self.games[gi].tree[node_idx].children[idx];
            node_idx = if child >= 0 {
                child as usize
            } else {
                self.games[gi].tree.push(Node::default());
                let new_idx = self.games[gi].tree.len() - 1;
                self.games[gi].tree[node_idx].children[idx] = new_idx as i32;
                new_idx
            };
        }
    }

    fn resolve_leaf(game: &mut Game, node_idx: usize, moves: Vec<Move>, priors: &[f32],
                    value: f32, path: &[(usize, usize)]) {
        if !game.tree[node_idx].expanded {
            game.tree[node_idx].expand(moves, priors.to_vec());
        }
        backup_vl(&mut game.tree, path, value);
        game.sims_done += 1;
    }

    /// Search budget reached: record the position, play the move, handle game
    /// end / next-move reset.
    fn finish_move(&mut self, gi: usize) {
        let cfg_move_temp = self.cfg.move_temp;
        let cfg_temp_plies = self.cfg.temp_plies;

        let (tokens, player, castling, ep) = encode_position(&self.games[gi].pos);
        let halfmove = halfmove(&self.games[gi].pos);

        // Visit distribution + move selection from the root.
        let (best_idx, root_v, visit_idx, visit_cnt) = {
            let root = &self.games[gi].tree[0];
            let mut visit_idx = Vec::new();
            let mut visit_cnt = Vec::new();
            for i in 0..root.moves.len() {
                if root.n[i] > 0 {
                    let (f, t, p) = move_coords(&root.moves[i]);
                    visit_idx.push((f * NUM_PLANES + action_plane(f, t, p)) as u16);
                    visit_cnt.push(root.n[i].min(u16::MAX as u32) as u16);
                }
            }
            let best = if cfg_move_temp > 0.0 && halfmove < cfg_temp_plies {
                let weights: Vec<f64> = root
                    .n
                    .iter()
                    .map(|&n| (n as f64).powf(1.0 / cfg_move_temp))
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
            let rv = if root.n[best] > 0 {
                (-root.w[best] / root.n[best] as f64) as f32
            } else {
                0.0
            };
            (best, rv, visit_idx, visit_cnt)
        };

        self.games[gi].records.push(Rec {
            tokens, player, castling, ep,
            halfmove: halfmove.min(u16::MAX as u32) as u16,
            root_v,
            visit_idx, visit_cnt,
        });
        self.positions_recorded += 1;

        let mv = self.games[gi].tree[0].moves[best_idx].clone();
        let uci = mv.to_uci(CastlingMode::Standard).to_string();
        let game = &mut self.games[gi];
        game.moves_uci.push(uci);
        game.pos.play_unchecked(&mv);
        game.history.push(zobrist(&game.pos));
        game.plies += 1;

        // Resign bookkeeping (white-POV root value).
        let v_white = if player == 1 { root_v } else { -root_v };
        if v_white.abs() >= self.cfg.resign_threshold {
            let sign = if v_white > 0.0 { 1 } else { -1 };
            game.resign_streak = if sign == game.resign_sign {
                game.resign_streak + 1
            } else {
                1
            };
            game.resign_sign = sign;
        } else {
            game.resign_streak = 0;
        }

        // Game over?
        let cur_key = *game.history.last().unwrap();
        let term = terminal_value(&game.pos, &game.history, cur_key);
        let outcome: Option<(f32, &'static str)> = if let Some(t) = term {
            let z = if t < 0.0 {
                // side to move is mated; the side that just moved wins
                if game.pos.turn() == Color::Black { 1.0 } else { -1.0 }
            } else {
                0.0
            };
            Some((z, "terminal"))
        } else if game.resign_allowed && game.resign_streak >= self.cfg.resign_plies {
            Some((game.resign_sign as f32, "resign"))
        } else if game.plies >= self.cfg.max_plies {
            Some((0.0, "max_plies"))
        } else {
            None
        };

        match outcome {
            Some((z_white, end)) => {
                let game = &mut self.games[gi];
                let fin = Finished {
                    records: std::mem::take(&mut game.records),
                    moves_uci: std::mem::take(&mut game.moves_uci),
                    start_fen: game.start_fen.clone(),
                    z_white,
                    end,
                    resign_allowed: game.resign_allowed,
                    id: game.id,
                };
                self.finished.push(fin);
                self.completed += 1;
                if self.started < self.target {
                    self.games[gi] = self.new_game();
                } else {
                    self.games[gi].active = false;
                }
            }
            None => {
                let game = &mut self.games[gi];
                game.tree = Self::fresh_tree();
                game.sims_done = 0;
            }
        }
    }
}

#[pymethods]
impl Engine {
    #[new]
    fn new(config_json: &str, openings: Vec<String>, seed: u64, target: usize) -> PyResult<Self> {
        let cfg: Config = serde_json::from_str(config_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("config: {e}")))?;
        let mut parsed = Vec::new();
        for fen in &openings {
            let f: Fen = fen
                .parse()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fen {fen}: {e}")))?;
            let pos: Chess = f
                .into_position(CastlingMode::Standard)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fen {fen}: {e}")))?;
            parsed.push(pos);
        }
        let n_games = cfg.num_parallel_games.min(target.max(1));
        let mut engine = Engine {
            cfg,
            openings: parsed,
            opening_fens: openings,
            games: Vec::new(),
            pending: Vec::new(),
            nn_items: Vec::new(),
            nn_index: HashMap::new(),
            tt: HashMap::new(),
            finished: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            started: 0,
            completed: 0,
            target,
            positions_recorded: 0,
        };
        for _ in 0..n_games {
            let g = engine.new_game();
            engine.games.push(g);
        }
        Ok(engine)
    }

    /// Run descent waves over all active games; immediately resolve terminal
    /// and TT-cached leaves. Returns encoded positions needing an NN forward
    /// (deduped), as numpy arrays.
    fn collect<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>,
    ) {
        for gi in 0..self.games.len() {
            if !self.games[gi].active {
                continue;
            }
            for _ in 0..self.cfg.leaves_per_wave {
                let g = &self.games[gi];
                if g.sims_done + g.inflight >= self.cfg.num_sims {
                    break;
                }
                let root_unexpanded = !self.games[gi].tree[0].expanded;
                if root_unexpanded && self.games[gi].inflight > 0 {
                    break; // root eval already queued
                }
                let (node_idx, path, pos, path_keys) = self.descend(gi);
                let key = zobrist(&pos);
                let mut all_keys = self.games[gi].history.clone();
                all_keys.extend_from_slice(&path_keys);
                if let Some(t) = terminal_value(&pos, &all_keys, key) {
                    let game = &mut self.games[gi];
                    backup_vl(&mut game.tree, &path, t);
                    game.sims_done += 1;
                    continue;
                }
                let moves: Vec<Move> = pos.legal_moves().to_vec();
                if let Some((priors, value)) = self.tt.get(&key).cloned() {
                    let game = &mut self.games[gi];
                    Self::resolve_leaf(game, node_idx, moves, &priors, value, &path);
                    continue;
                }
                self.nn_index.entry(key).or_insert_with(|| {
                    let (tokens, player, castling, ep) = encode_position(&pos);
                    let move_fp = moves
                        .iter()
                        .map(|m| {
                            let (f, t, p) = move_coords(m);
                            (f, action_plane(f, t, p))
                        })
                        .collect();
                    self.nn_items.push(NnItem { key, tokens, player, castling, ep, move_fp });
                    self.nn_items.len() - 1
                });
                self.pending.push(PendingLeaf { game: gi, node: node_idx, path, key, moves });
                self.games[gi].inflight += 1;
                if root_unexpanded {
                    break; // nothing else to search until the root is expanded
                }
            }
        }

        let n = self.nn_items.len();
        let mut boards = Vec::with_capacity(n * 64);
        let mut player = Vec::with_capacity(n);
        let mut castling = Vec::with_capacity(n);
        let mut ep = Vec::with_capacity(n);
        for item in &self.nn_items {
            boards.extend_from_slice(&item.tokens);
            player.push(item.player);
            castling.push(item.castling);
            ep.push(item.ep);
        }
        let boards = PyArray1::from_vec(py, boards)
            .reshape([n, 64])
            .expect("reshape");
        (
            boards,
            player.into_pyarray(py),
            castling.into_pyarray(py),
            ep.into_pyarray(py),
        )
    }

    /// Consume NN results for the batch returned by the last collect():
    /// expand + back up all pending leaves, then play moves for games whose
    /// search budget is complete (recording positions / finishing games).
    fn apply(
        &mut self,
        logits: PyReadonlyArray3<f32>,
        values: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        let logits = logits.as_array();
        let values = values.as_slice()?;
        if values.len() != self.nn_items.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "apply: got {} values for {} pending NN items",
                values.len(),
                self.nn_items.len()
            )));
        }

        if self.tt.len() + self.nn_items.len() > self.cfg.tt_cap {
            self.tt.clear(); // crude but rare; avoids unbounded growth
        }
        for (i, item) in self.nn_items.iter().enumerate() {
            let mut scores: Vec<f32> = item
                .move_fp
                .iter()
                .map(|&(f, p)| logits[[i, f, p]])
                .collect();
            softmax_priors(&mut scores, self.cfg.prior_temp);
            self.tt.insert(item.key, (scores, values[i]));
        }

        let pending = std::mem::take(&mut self.pending);
        for leaf in pending {
            let (priors, value) = self.tt.get(&leaf.key).cloned().expect("just inserted");
            let game = &mut self.games[leaf.game];
            Self::resolve_leaf(game, leaf.node, leaf.moves, &priors, value, &leaf.path);
            game.inflight -= 1;
        }
        self.nn_items.clear();
        self.nn_index.clear();

        for gi in 0..self.games.len() {
            if self.games[gi].active
                && self.games[gi].inflight == 0
                && self.games[gi].sims_done >= self.cfg.num_sims
            {
                self.finish_move(gi);
            }
        }
        Ok(())
    }

    /// Completed games since the last call, as dicts of numpy arrays.
    fn drain_finished<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let finished = std::mem::take(&mut self.finished);
        let mut out = Vec::with_capacity(finished.len());
        for f in finished {
            let p = f.records.len();
            let mut boards = Vec::with_capacity(p * 64);
            let (mut player, mut castling, mut ep) =
                (Vec::with_capacity(p), Vec::with_capacity(p), Vec::with_capacity(p));
            let mut halfmove = Vec::with_capacity(p);
            let (mut root_v, mut z) = (Vec::with_capacity(p), Vec::with_capacity(p));
            let (mut visit_idx, mut visit_cnt) = (Vec::new(), Vec::new());
            let mut visit_ptr = Vec::with_capacity(p + 1);
            visit_ptr.push(0i64);
            for r in &f.records {
                boards.extend_from_slice(&r.tokens);
                player.push(r.player);
                castling.push(r.castling);
                ep.push(r.ep);
                halfmove.push(r.halfmove);
                root_v.push(r.root_v);
                z.push(if r.player == 1 { f.z_white } else { -f.z_white });
                visit_idx.extend_from_slice(&r.visit_idx);
                visit_cnt.extend_from_slice(&r.visit_cnt);
                visit_ptr.push(visit_idx.len() as i64);
            }
            let d = PyDict::new(py);
            d.set_item("boards", PyArray1::from_vec(py, boards).reshape([p, 64]).expect("reshape"))?;
            d.set_item("player", player.into_pyarray(py))?;
            d.set_item("castling", castling.into_pyarray(py))?;
            d.set_item("ep", ep.into_pyarray(py))?;
            d.set_item("halfmove", halfmove.into_pyarray(py))?;
            d.set_item("root_v", root_v.into_pyarray(py))?;
            d.set_item("z", z.into_pyarray(py))?;
            d.set_item("visit_idx", visit_idx.into_pyarray(py))?;
            d.set_item("visit_cnt", visit_cnt.into_pyarray(py))?;
            d.set_item("visit_ptr", visit_ptr.into_pyarray(py))?;
            d.set_item("moves", f.moves_uci)?;
            d.set_item("fen", f.start_fen)?;
            d.set_item("z_white", f.z_white)?;
            d.set_item("end", f.end)?;
            d.set_item("resign_allowed", f.resign_allowed)?;
            d.set_item("id", f.id)?;
            out.push(d);
        }
        Ok(out)
    }

    fn done(&self) -> bool {
        self.completed >= self.target
    }

    /// (games started, games completed, positions recorded, TT entries)
    fn stats(&self) -> (usize, usize, usize, usize) {
        (self.started, self.completed, self.positions_recorded, self.tt.len())
    }
}

// ── debug helpers for the Python parity test ────────────────────────────

#[pyfunction]
fn debug_encode(fen: &str) -> PyResult<(Vec<u8>, u8, u8, u8)> {
    let f: Fen = fen
        .parse()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    let pos: Chess = f
        .into_position(CastlingMode::Standard)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    let (tokens, player, castling, ep) = encode_position(&pos);
    Ok((tokens.to_vec(), player, castling, ep))
}

#[pyfunction]
fn debug_move_planes(fen: &str) -> PyResult<Vec<(String, u8, u8)>> {
    let f: Fen = fen
        .parse()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    let pos: Chess = f
        .into_position(CastlingMode::Standard)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    let mut out = Vec::new();
    for m in pos.legal_moves() {
        let (from, to, promo) = move_coords(&m);
        out.push((
            m.to_uci(CastlingMode::Standard).to_string(),
            from as u8,
            action_plane(from, to, promo) as u8,
        ));
    }
    Ok(out)
}

#[pyfunction]
fn debug_zobrist(fen: &str) -> PyResult<u64> {
    let f: Fen = fen
        .parse()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    let pos: Chess = f
        .into_position(CastlingMode::Standard)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    Ok(zobrist(&pos))
}

#[pymodule]
fn selfplay_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_function(wrap_pyfunction!(debug_encode, m)?)?;
    m.add_function(wrap_pyfunction!(debug_move_planes, m)?)?;
    m.add_function(wrap_pyfunction!(debug_zobrist, m)?)?;
    Ok(())
}
