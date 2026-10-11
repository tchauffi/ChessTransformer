//! The network interface the search is written against: encoded positions in,
//! flat policy logits and a side-to-move value out. Implemented by ct-bot's
//! ONNX Runtime evaluator, by test doubles, and -- for the batched match runner
//! in selfplay-core -- by nothing at all: that driver steps the search itself
//! (`Search::step` / `Search::resolve`) and runs the forward in Python.

use anyhow::Result;

pub struct EvalRequest {
    pub tokens: [u8; 64],
    pub player: u8,
    pub castling: u8,
    pub ep: u8,
    /// The legal moves' flat logit indices (from_sq * 73 + plane), in the
    /// search's move order -- filled only with `SearchParams::legal_in_requests`,
    /// so a batched driver can gather those logits on the GPU and send back a
    /// few dozen floats per position instead of 4672.
    pub legal: Vec<u16>,
}

pub struct EvalOutput {
    /// Either the flat [64*73] logits for one position (from_sq * 73 + plane),
    /// or exactly one logit per legal move in the order of the request's
    /// `legal` (a position never has 4672 legal moves, so the length says which).
    pub logits: Vec<f32>,
    pub value: f32,
}

/// Batched position evaluation.
pub trait Evaluator {
    fn eval_batch(&mut self, batch: &[EvalRequest]) -> Result<Vec<EvalOutput>>;
}

/// For searches driven step by step, whose evaluations arrive through `resolve`.
pub struct NoEval;

impl Evaluator for NoEval {
    fn eval_batch(&mut self, _batch: &[EvalRequest]) -> Result<Vec<EvalOutput>> {
        anyhow::bail!("NoEval: this search is driven with step()/resolve(), not go()")
    }
}
