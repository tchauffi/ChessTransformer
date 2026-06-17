//! ct-bot: standalone Rust bot for the ChessTransformer Pos2MoveV2 model.
//!
//! Subcommands: `uci` (engine for gauntlets/cutechess), `lichess` (Bot API
//! client), `bench` (NN + search throughput), `eval` (single-position parity
//! probe used by scripts/check_onnx_parity.py).

mod bench;
mod lichess;
mod nn;
mod search;
mod serve;
mod timeman;
mod uci;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

const DEFAULT_MODEL: &str = "data/models/pos2move_v2.1/model.int8.onnx";

#[derive(Parser)]
#[command(name = "ct-bot", about = "ChessTransformer Rust bot")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct ModelArgs {
    /// Path to the ONNX model
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: PathBuf,
    /// Intra-op threads for ONNX Runtime (0 = auto)
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Inference device
    #[arg(long, default_value = "cpu", value_parser = ["cpu", "cuda"])]
    device: String,
}

impl ModelArgs {
    fn evaluator(&self) -> Result<nn::OnnxEvaluator> {
        nn::OnnxEvaluator::new(&self.model, self.threads, self.device == "cuda")
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// UCI engine mode
    Uci(UciArgs),
    /// Lichess bot client
    Lichess(LichessArgs),
    /// NN + search throughput benchmark
    Bench(BenchArgs),
    /// HTTP backend for the web frontend (replaces backend/api.py)
    Serve(ServeArgs),
    /// Evaluate one FEN; print value + per-move priors as JSON (parity probe)
    #[command(hide = true)]
    Eval(EvalArgs),
}

#[derive(Args)]
struct UciArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Default simulations per move (when `go` has no nodes/time limits)
    #[arg(long, default_value_t = 800)]
    sims: u32,
    /// Leaves per NN wave
    #[arg(long, default_value_t = 16)]
    sim_batch: usize,
    #[arg(long, default_value_t = 1.0)]
    c_puct: f64,
    #[arg(long, default_value_t = 0.2)]
    fpu: f64,
    #[arg(long, default_value_t = 1.0)]
    prior_temp: f32,
    /// Seed for the clock-mode sims/sec estimate (refined after every move)
    #[arg(long, default_value_t = 400.0)]
    initial_sims_per_sec: f64,
}

#[derive(Args, Clone)]
struct LichessArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// API token of the BOT account (falls back to $LICHESS_BOT_TOKEN)
    #[arg(long, env = "LICHESS_BOT_TOKEN", hide_env_values = true)]
    token: String,
    /// Accept rated challenges (casual is always accepted)
    #[arg(long)]
    allow_rated: bool,
    /// Accepted challenge speeds
    #[arg(long, value_delimiter = ',', default_values_t =
        ["bullet".to_string(), "blitz".to_string(), "rapid".to_string(), "classical".to_string()])]
    speeds: Vec<String>,
    /// Maximum concurrent games
    #[arg(long, default_value_t = 1)]
    max_games: usize,
    /// Leaves per NN wave
    #[arg(long, default_value_t = 16)]
    sim_batch: usize,
    /// Opening variety: sample moves ∝ visits^(1/temp) for the first
    /// `--move-temp-plies` half-moves (0.0 = deterministic argmax)
    #[arg(long, default_value_t = 0.0)]
    move_temp: f64,
    #[arg(long, default_value_t = 16)]
    move_temp_plies: u32,
    /// Seed for the sims/sec estimate (refined after every move)
    #[arg(long, default_value_t = 400.0)]
    initial_sims_per_sec: f64,
    /// Lichess server (override for testing)
    #[arg(long, default_value = "https://lichess.org")]
    base_url: String,
}

#[derive(Args)]
struct BenchArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Simulation budgets for the search benchmark
    #[arg(long, num_args = 1.., default_values_t = [800u32, 3200, 12800])]
    sims: Vec<u32>,
    /// Leaves per NN wave
    #[arg(long, default_value_t = 16)]
    sim_batch: usize,
}

#[derive(Args, Clone)]
struct ServeArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long, default_value_t = 5001, env = "PORT")]
    port: u16,
    /// MCTS simulations per /api/move request
    #[arg(long, default_value_t = 800, env = "MCTS_SIMS")]
    sims: u32,
    /// Leaves per NN wave
    #[arg(long, default_value_t = 16)]
    sim_batch: usize,
    /// Reported in /api/health for the frontend display
    #[arg(long, default_value_t = 2100)]
    approx_elo: u32,
}

#[derive(Args)]
struct EvalArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long)]
    fen: String,
}

fn cmd_eval(args: &EvalArgs) -> Result<()> {
    use chess_core::encoding::{action_plane, encode_position, move_coords, NUM_PLANES};
    use nn::Evaluator;
    use shakmaty::{fen::Fen, CastlingMode, Chess, Position};

    let fen: Fen = args.fen.parse()?;
    let pos: Chess = fen.into_position(CastlingMode::Standard)?;
    let (tokens, player, castling, ep) = encode_position(&pos);

    let mut eval = args.model.evaluator()?;
    let out = eval
        .eval_batch(&[nn::EvalRequest { tokens, player, castling, ep }])?
        .remove(0);

    let moves = pos.legal_moves();
    let mut scores: Vec<f32> = moves
        .iter()
        .map(|m| {
            let (f, t, p) = move_coords(m);
            out.logits[f * NUM_PLANES + action_plane(f, t, p)]
        })
        .collect();
    chess_core::tree::softmax_priors(&mut scores, 1.0);

    let moves_json: Vec<serde_json::Value> = moves
        .iter()
        .zip(scores.iter())
        .map(|(m, p)| {
            serde_json::json!({
                "uci": m.to_uci(CastlingMode::Standard).to_string(),
                "prior": p,
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::json!({ "value": out.value, "moves": moves_json })
    );
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let cli = Cli::parse();
    match &cli.cmd {
        Cmd::Eval(args) => cmd_eval(args),
        Cmd::Bench(args) => bench::run(args),
        Cmd::Uci(args) => uci::run(args),
        Cmd::Lichess(args) => lichess::run(args),
        Cmd::Serve(args) => serve::run(args),
    }
}
