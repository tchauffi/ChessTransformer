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

/// KataGo's forced playouts (Wu 2019, "Accelerating Self-Play Learning in Go"): a
/// root child that has any visits but fewer than `sqrt(k * P * N)` -- P its (noised)
/// prior, N the root's visits -- is searched next, ahead of PUCT. The most short of
/// its quota goes first. `None` when every child has its quota (PUCT decides).
pub fn forced_select(node: &Node, k: f64) -> Option<usize> {
    let total: u32 = node.n.iter().sum();
    let mut best = None;
    let mut best_short = 0.0;
    for i in 0..node.moves.len() {
        if node.n[i] == 0 {
            continue;
        }
        let short = (k * node.priors[i] as f64 * total as f64).sqrt() - node.n[i] as f64;
        if short > best_short {
            best_short = short;
            best = Some(i);
        }
    }
    best
}

/// KataGo's policy target pruning, the other half of forced playouts: the root's
/// visit counts with each child other than the most visited one cut by up to its
/// forced quota `sqrt(k * P * N)`, as long as the cut leaves its PUCT score below the
/// most visited child's -- the visits that PUCT alone would not have given it at this
/// total. A child left with a single visit is dropped. With no forced playouts in the
/// search the cuts only remove visits PUCT would not have spent either.
pub fn prune_forced(node: &Node, c_puct: f64, k: f64) -> Vec<u32> {
    let mut out = node.n.clone();
    let total: u32 = node.n.iter().sum();
    let Some(best) = (0..node.n.len()).max_by_key(|&i| node.n[i]) else { return out };
    if total == 0 {
        return out;
    }
    let sqrt_total = (1.0 + total as f64).sqrt();
    let puct = |i: usize, n: u32| {
        -node.w[i] / node.n[i] as f64 + c_puct * sqrt_total * node.priors[i] as f64 / (1.0 + n as f64)
    };
    let target = puct(best, node.n[best]);
    for i in 0..node.n.len() {
        if i == best || node.n[i] == 0 {
            continue;
        }
        let quota = (k * node.priors[i] as f64 * total as f64).sqrt();
        let mut m = node.n[i];
        while m > 0 && ((node.n[i] - m + 1) as f64) <= quota && puct(i, m - 1) < target {
            m -= 1;
        }
        out[i] = if m <= 1 { 0 } else { m };
    }
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{Role, Square};

    /// A root with these priors, visits and Q (root side-to-move POV).
    fn root(priors: &[f32], n: &[u32], q: &[f64]) -> Node {
        let mv = Move::Normal { role: Role::Pawn, from: Square::A2, capture: None, to: Square::A3, promotion: None };
        let mut node = Node::default();
        node.expand(vec![mv; priors.len()], priors.to_vec());
        node.n = n.to_vec();
        node.w = q.iter().zip(n).map(|(q, &n)| -q * n as f64).collect();
        node
    }

    #[test]
    fn forced_playouts_go_to_the_visited_child_most_short_of_its_quota() {
        // N = 52: quotas sqrt(2 P N) = 7.9 / 5.6 / 3.2 / 2.3
        let node = root(&[0.6, 0.3, 0.1, 0.05], &[47, 2, 3, 0], &[0.0; 4]);
        assert_eq!(forced_select(&node, 2.0), Some(1)); // 3.6 short; the unvisited child never
        let filled = root(&[0.6, 0.3, 0.1, 0.05], &[43, 6, 3, 0], &[0.0; 4]);
        assert_eq!(forced_select(&filled, 2.0), Some(2)); // 0.2 short
        assert_eq!(forced_select(&root(&[0.6, 0.3, 0.1, 0.05], &[42, 6, 4, 0], &[0.0; 4]), 2.0), None);
    }

    #[test]
    fn pruning_cuts_only_visits_puct_would_not_have_given() {
        // N = 113, c_puct 1, target = PUCT of the most visited child = 0.1527.
        let node = root(&[0.40, 0.3, 0.2, 0.05, 0.05], &[80, 12, 8, 10, 3], &[0.1, 0.0, -0.2, -0.4, -0.4]);
        assert_eq!(prune_forced(&node, 1.0, 2.0), vec![
            80, // the most visited child keeps every visit
            12, // already above the target: nothing to cut
            6,  // cut 2: one more would lift its PUCT past the target
            7,  // cut 3: its whole quota sqrt(2 * 0.05 * 113) = 3.4
            0,  // cut to a single visit: dropped
        ]);
        assert_eq!(prune_forced(&node, 1.0, 0.0), node.n, "no quota, no cut");
    }
}
