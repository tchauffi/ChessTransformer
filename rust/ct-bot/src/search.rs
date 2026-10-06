//! Single-game batched MCTS over one position — semantics mirror
//! `Pos2MoveV2MctsBot` / `selfplay-core::Engine`: PUCT + FPU, virtual-loss
//! leaf waves with one batched NN forward per wave, zobrist TT for priors +
//! value, threefold/50-move awareness over game history + search path
//! (plus Lc0's two-fold rule for cycles inside the search), and
//! tree reuse across moves (re-root when the new position extends the
//! previous root by 1..=4 expanded plies).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use chess_core::encoding::{
    action_plane, encode_position, halfmove, move_coords, search_terminal_value, zobrist,
    NUM_PLANES,
};
use chess_core::tree::{backup_vl, puct_select, softmax_priors, Node};
use rand::prelude::*;
use rand::rngs::StdRng;
use shakmaty::{Chess, Move, Position};
use shakmaty_syzygy::{AmbiguousWdl, Tablebase, Wdl};

use crate::nn::{EvalRequest, Evaluator};

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
            syzygy: None,
        }
    }
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
    pub fn new(eval: E, params: SearchParams) -> Self {
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
        let t0 = Instant::now();
        let mut nn_evals = 0u32;
        let mut tt_hits = 0u32;
        let mut tb_hits = 0u32;
        let mut sims_done = 0u32;

        let restrict = match self.root_tablebase() {
            RootTb::Play(best) => {
                return Ok(SearchResult {
                    best,
                    value: 1.0,
                    sims: 0,
                    nn_evals: 0,
                    tt_hits: 0,
                    tb_hits: 1,
                    elapsed: t0.elapsed(),
                });
            }
            RootTb::Restrict(keep) => {
                self.tree = vec![Node::default()]; // a reused root has every move
                Some(keep)
            }
            RootTb::Off => None,
        };

        if !self.tree[0].expanded {
            let key = *self.history.last().unwrap();
            let moves: Vec<Move> = self.pos.legal_moves().to_vec();
            assert!(!moves.is_empty(), "go() on a terminal position");
            let priors = if let Some((priors, _)) = self.tt.get(&key).cloned() {
                tt_hits += 1;
                priors
            } else {
                let (tokens, player, castling, ep) = encode_position(&self.pos);
                let out = self
                    .eval
                    .eval_batch(&[EvalRequest { tokens, player, castling, ep }])?
                    .remove(0);
                nn_evals += 1;
                let priors = priors_for(&moves, &out.logits, self.params.prior_temp);
                self.insert_tt(key, priors.clone(), out.value);
                priors
            };
            let (moves, priors) = match &restrict {
                Some(keep) => restrict_moves(moves, priors, keep),
                None => (moves, priors),
            };
            self.tree[0].expand(moves, priors);
        }

        while sims_done < sims {
            let wave = (self.params.sim_batch as u32).min(sims - sims_done) as usize;
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
                    sims_done += 1;
                    continue;
                }
                if let Some(t) = self.leaf_tablebase(&pos) {
                    backup_vl(&mut self.tree, &path, t);
                    sims_done += 1;
                    tb_hits += 1;
                    continue;
                }
                let moves: Vec<Move> = pos.legal_moves().to_vec();
                // Defer cached (TT) hits and NN leaves alike: their virtual loss
                // stays applied through the wave so concurrent descents diversify
                // (mirrors Pos2MoveV2MctsBot._run_batch). Backing TT hits up
                // immediately collapses search breadth once the TT fills, which
                // wrecks play at high sim counts.
                let eval = if let Some((priors, value)) = self.tt.get(&key).cloned() {
                    tt_hits += 1;
                    LeafEval::Cached(priors, value)
                } else {
                    let batch_idx = *batch_index.entry(key).or_insert_with(|| {
                        let (tokens, player, castling, ep) = encode_position(&pos);
                        batch.push(EvalRequest { tokens, player, castling, ep });
                        batch.len() - 1
                    });
                    LeafEval::Batch(batch_idx)
                };
                pending.push(PendingLeaf { node: node_idx, path, key, moves, eval });
            }

            let outputs = if batch.is_empty() {
                Vec::new()
            } else {
                let o = self.eval.eval_batch(&batch)?;
                nn_evals += o.len() as u32;
                o
            };
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
                sims_done += 1;
            }

            if stop.load(Ordering::Relaxed) {
                break;
            }
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    break;
                }
            }
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

        Ok(SearchResult {
            best: root.moves[best_idx].clone(),
            value,
            sims: sims_done,
            nn_evals,
            tt_hits,
            tb_hits,
            elapsed: t0.elapsed(),
        })
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

    /// Raw value-head estimate for a position (side-to-move POV), no search.
    pub fn raw_value(&mut self, pos: &Chess) -> Result<f32> {
        let (tokens, player, castling, ep) = encode_position(pos);
        Ok(self
            .eval
            .eval_batch(&[EvalRequest { tokens, player, castling, ep }])?
            .remove(0)
            .value)
    }

    fn insert_tt(&mut self, key: u64, priors: Vec<f32>, value: f32) {
        if self.tt.len() >= self.params.tt_cap {
            self.tt.clear(); // crude but rare; avoids unbounded growth
        }
        self.tt.insert(key, (priors, value));
    }
}

fn priors_for(moves: &[Move], logits: &[f32], prior_temp: f32) -> Vec<f32> {
    let mut scores: Vec<f32> = moves
        .iter()
        .map(|m| {
            let (f, t, p) = move_coords(m);
            logits[f * NUM_PLANES + action_plane(f, t, p)]
        })
        .collect();
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
    use crate::nn::EvalOutput;
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
        use chess_core::encoding::search_terminal_value;
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
        crate::load_syzygy(&path).unwrap().expect("no tables found")
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
