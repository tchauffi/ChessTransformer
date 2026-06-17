//! Edge-centric MCTS tree primitives: node storage, PUCT selection with FPU,
//! virtual-loss backup, and prior softmax. Semantics mirror
//! `Pos2MoveV2MctsBot` exactly.

use shakmaty::Move;

#[derive(Default)]
pub struct Node {
    pub expanded: bool,
    pub moves: Vec<Move>,
    pub priors: Vec<f32>,
    pub n: Vec<u32>,
    pub w: Vec<f64>,
    pub children: Vec<i32>, // -1 = not materialized
}

impl Node {
    pub fn expand(&mut self, moves: Vec<Move>, priors: Vec<f32>) {
        let k = moves.len();
        self.moves = moves;
        self.priors = priors;
        self.n = vec![0; k];
        self.w = vec![0.0; k];
        self.children = vec![-1; k];
        self.expanded = true;
    }
}

pub fn softmax_priors(scores: &mut [f32], temp: f32) {
    let mut max = f32::NEG_INFINITY;
    for s in scores.iter_mut() {
        *s /= temp;
        max = max.max(*s);
    }
    let mut sum = 0.0;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }
}

/// PUCT child selection: maximize Q + c_puct * sqrt(1+N) * P / (1+n), with
/// unvisited children valued at parent_Q - fpu (first-play urgency).
pub fn puct_select(node: &Node, c_puct: f64, fpu: f64) -> usize {
    let total: u32 = node.n.iter().sum();
    let sqrt_total = (1.0 + total as f64).sqrt();
    let parent_q = if total > 0 {
        -node.w.iter().sum::<f64>() / total as f64
    } else {
        0.0
    };
    let mut best = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for i in 0..node.moves.len() {
        let q = if node.n[i] > 0 {
            -node.w[i] / node.n[i] as f64
        } else {
            parent_q - fpu
        };
        let u = c_puct * sqrt_total * node.priors[i] as f64 / (1.0 + node.n[i] as f64);
        let s = q + u;
        if s > best_score {
            best_score = s;
            best = i;
        }
    }
    best
}

/// Negamax backup along a virtual-lossed path: N was already counted by the
/// virtual loss during descent; undo the +1.0 on W and apply the real value.
pub fn backup_vl(tree: &mut [Node], path: &[(usize, usize)], value: f32) {
    let mut v = value as f64;
    for &(node_idx, idx) in path.iter().rev() {
        tree[node_idx].w[idx] += v - 1.0;
        v = -v;
    }
}
