//! ONNX Runtime evaluator for the Pos2MoveV2 model.
//!
//! Inputs (all i64): board_tokens [n,64], player_token [n], castling_token [n],
//! en_passant_token [n]. Outputs: move_logits [n,64,73] f32, value [n,1] f32.
//! Parity with python-onnxruntime is guarded by scripts/check_onnx_parity.py
//! --rust-bin.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3};
use ort::session::{builder::GraphOptimizationLevel, Session};

pub struct EvalRequest {
    pub tokens: [u8; 64],
    pub player: u8,
    pub castling: u8,
    pub ep: u8,
}

pub struct EvalOutput {
    /// Flat [64*73] logits for one position (from_sq * 73 + plane).
    pub logits: Vec<f32>,
    pub value: f32,
}

/// Batched position evaluation. Implemented by `OnnxEvaluator` (and by test
/// doubles in the search unit tests).
pub trait Evaluator {
    fn eval_batch(&mut self, batch: &[EvalRequest]) -> Result<Vec<EvalOutput>>;
}

pub struct OnnxEvaluator {
    session: Session,
}

/// ort builder errors carry a non-Send recovery payload; strip it for anyhow.
fn ort_ok<T, R>(r: std::result::Result<T, ort::Error<R>>) -> Result<T> {
    r.map_err(|e| anyhow::anyhow!("ort: {e}"))
}

impl OnnxEvaluator {
    pub fn new(model: &Path, intra_threads: usize, use_cuda: bool) -> Result<Self> {
        let threads = if intra_threads == 0 {
            (std::thread::available_parallelism().map_or(4, |n| n.get()) / 2).clamp(1, 8)
        } else {
            intra_threads
        };
        #[allow(unused_mut)]
        let mut builder = ort_ok(
            ort_ok(ort_ok(Session::builder())?.with_optimization_level(GraphOptimizationLevel::Level3))?
                .with_intra_threads(threads),
        )?;
        if use_cuda {
            #[cfg(feature = "cuda")]
            {
                use ort::execution_providers::CUDAExecutionProvider;
                builder = ort_ok(
                    builder.with_execution_providers([CUDAExecutionProvider::default().build()]),
                )?;
                log::info!("CUDA execution provider requested (falls back to CPU if unavailable)");
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("--device cuda requires building with --features cuda");
            }
        }
        let session = ort_ok(builder.commit_from_file(model))
            .with_context(|| format!("loading ONNX model {}", model.display()))?;
        log::info!("ONNX model loaded: {} ({} intra threads)", model.display(), threads);
        Ok(Self { session })
    }

    fn eval_batch_impl(&mut self, batch: &[EvalRequest]) -> Result<Vec<EvalOutput>> {
        let n = batch.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut boards = Array2::<i64>::zeros((n, 64));
        let mut player = Array1::<i64>::zeros(n);
        let mut castling = Array1::<i64>::zeros(n);
        let mut ep = Array1::<i64>::zeros(n);
        for (i, r) in batch.iter().enumerate() {
            for (j, &t) in r.tokens.iter().enumerate() {
                boards[[i, j]] = t as i64;
            }
            player[i] = r.player as i64;
            castling[i] = r.castling as i64;
            ep[i] = r.ep as i64;
        }

        let outputs = self.session.run(ort::inputs![
            "board_tokens" => ort::value::Tensor::from_array(boards)?,
            "player_token" => ort::value::Tensor::from_array(player)?,
            "castling_token" => ort::value::Tensor::from_array(castling)?,
            "en_passant_token" => ort::value::Tensor::from_array(ep)?,
        ])?;

        let logits: Array3<f32> = outputs["move_logits"]
            .try_extract_array::<f32>()?
            .into_dimensionality()?
            .to_owned();
        let value: Array2<f32> = outputs["value"]
            .try_extract_array::<f32>()?
            .into_dimensionality()?
            .to_owned();

        Ok((0..n)
            .map(|i| EvalOutput {
                logits: logits
                    .index_axis(ndarray::Axis(0), i)
                    .iter()
                    .copied()
                    .collect(),
                value: value[[i, 0]],
            })
            .collect())
    }
}

impl Evaluator for OnnxEvaluator {
    fn eval_batch(&mut self, batch: &[EvalRequest]) -> Result<Vec<EvalOutput>> {
        self.eval_batch_impl(batch)
    }
}
