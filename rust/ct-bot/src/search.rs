//! Single-game batched MCTS over one position — semantics mirror
//! `Pos2MoveV2MctsBot` / `selfplay-core::Engine`: PUCT + FPU, virtual-loss
//! leaf waves with one batched NN forward per wave, zobrist TT for priors +
//! value, threefold/50-move awareness over game history + search path, and
//! tree reuse across moves (re-root when the new position extends the
//! previous root by 1..=4 expanded plies).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use chess_core::encoding::{
    action_plane, encode_position, halfmove, move_coords, terminal_value, zobrist, NUM_PLANES,
};
use chess_core::tree::{backup_vl, puct_select, softmax_priors, Node};
use rand::prelude::*;
use rand::rngs::StdRng;
use shakmaty::{Chess, Move, Position};

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
    pub elapsed: Duration,
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
        let mut sims_done = 0u32;

        if !self.tree[0].expanded {
            let key = *self.history.last().unwrap();
            let moves: Vec<Move> = self.pos.legal_moves().to_vec();
            assert!(!moves.is_empty(), "go() on a terminal position");
            if let Some((priors, _)) = self.tt.get(&key).cloned() {
                tt_hits += 1;
                self.tree[0].expand(moves, priors);
            } else {
                let (tokens, player, castling, ep) = encode_position(&self.pos);
                let out = self
                    .eval
                    .eval_batch(&[EvalRequest { tokens, player, castling, ep }])?
                    .remove(0);
                nn_evals += 1;
                let priors = priors_for(&moves, &out.logits, self.params.prior_temp);
                self.insert_tt(key, priors.clone(), out.value);
                self.tree[0].expand(moves, priors);
            }
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
                if let Some(t) = terminal_value(&pos, &all_keys, key) {
                    backup_vl(&mut self.tree, &path, t);
                    sims_done += 1;
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
            elapsed: t0.elapsed(),
        })
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
