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
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use shakmaty::Chess;
use shakmaty_syzygy::Tablebase;

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

/// Load Syzygy tables from a ':'-separated list of directories (the UCI
/// `SyzygyPath` convention). Empty means none.
pub fn load_syzygy(paths: &str) -> Result<Option<Arc<Tablebase<Chess>>>> {
    let dirs: Vec<&str> = paths.split(':').filter(|d| !d.trim().is_empty()).collect();
    if dirs.is_empty() {
        return Ok(None);
    }
    let mut tb = Tablebase::new();
    let mut n = 0;
    for dir in &dirs {
        n += tb.add_directory(dir).with_context(|| format!("syzygy path {dir}"))?;
    }
    log::info!("syzygy: {n} table files from {paths} (up to {} pieces)", tb.max_pieces());
    Ok(Some(Arc::new(tb)))
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
    /// Search one FEN from scratch at several budgets; print the top root
    /// moves (visits, Q, prior) per budget (search-behavior probe)
    #[command(hide = true)]
    Probe(ProbeArgs),
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
    /// Syzygy tablebase directories, ':'-separated (UCI option SyzygyPath)
    #[arg(long, default_value = "")]
    syzygy_path: String,
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
    /// Syzygy tablebase directories, ':'-separated
    #[arg(long, default_value = "")]
    syzygy_path: String,
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
    /// Syzygy tablebase directories, ':'-separated
    #[arg(long, default_value = "", env = "SYZYGY_PATH")]
    syzygy_path: String,
    /// Also serve the web frontend's static build (Next.js `out/`) from this
    /// directory, so one process hosts the whole demo (Hugging Face Space).
    #[arg(long, env = "STATIC_DIR")]
    static_dir: Option<std::path::PathBuf>,
}

#[derive(Args)]
struct EvalArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long)]
    fen: String,
}

#[derive(Args)]
struct ProbeArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long, default_value = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")]
    fen: String,
    #[arg(long, num_args = 1.., default_values_t = [800u32, 3200, 12800])]
    sims: Vec<u32>,
    #[arg(long, default_value_t = 16)]
    sim_batch: usize,
    #[arg(long, default_value_t = 1.0)]
    c_puct: f64,
    #[arg(long, default_value_t = 0.2)]
    fpu: f64,
    #[arg(long, default_value_t = 1.0)]
    prior_temp: f32,
    /// Root moves to print per budget
    #[arg(long, default_value_t = 6)]
    top: usize,
    /// Syzygy tablebase directories, ':'-separated
    #[arg(long, default_value = "")]
    syzygy_path: String,
}

fn cmd_probe(args: &ProbeArgs) -> Result<()> {
    use shakmaty::{fen::Fen, CastlingMode, Chess};
    use std::sync::atomic::AtomicBool;

    let fen: Fen = args.fen.parse()?;
    let pos: Chess = fen.into_position(CastlingMode::Standard)?;
    let params = search::SearchParams {
        c_puct: args.c_puct,
        fpu: args.fpu,
        prior_temp: args.prior_temp,
        sim_batch: args.sim_batch,
        syzygy: load_syzygy(&args.syzygy_path)?,
        ..search::SearchParams::default()
    };
    let mut s = search::Search::new(args.model.evaluator()?, params);
    for &sims in &args.sims {
        s.new_game();
        s.set_position(pos.clone(), &[]);
        let r = s.go(sims, None, &AtomicBool::new(false))?;
        println!(
            "sims {:>6}  best {}  value {:+.3}  tb hits {}  ({:.1}s)",
            r.sims,
            r.best.to_uci(CastlingMode::Standard),
            r.value,
            r.tb_hits,
            r.elapsed.as_secs_f64()
        );
        let pv: Vec<String> = s
            .pv(16)
            .iter()
            .map(|(m, n, q)| format!("{}({n},{q:+.2})", m.to_uci(CastlingMode::Standard)))
            .collect();
        println!("    pv {}", pv.join(" "));
        for (m, n, q, p) in s.root_stats().into_iter().take(args.top) {
            println!(
                "    {:<6} n {:>6}  q {:+.3}  prior {:.3}",
                m.to_uci(CastlingMode::Standard).to_string(),
                n,
                q,
                p
            );
        }
    }
    Ok(())
}

fn cmd_eval(args: &EvalArgs) -> Result<()> {
    use chess_core::encoding::{action_plane, encode_position, move_coords, NUM_PLANES};
    use nn::Evaluator;
    use shakmaty::{fen::Fen, CastlingMode, Chess, Position};

    let fen: Fen = args.fen.parse()?;
    let pos: Chess = fen.into_position(CastlingMode::Standard)?;
    let (tokens, player, castling, ep) = encode_position(&pos);
    let (rule50, repetition) = chess_core::encoding::rule_state(&pos, &[], 0);

    let mut eval = args.model.evaluator()?;
    let out = eval
        .eval_batch(&[nn::EvalRequest {
            tokens, player, castling, ep, rule50, repetition, legal: Vec::new(),
        }])?
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
        Cmd::Probe(args) => cmd_probe(args),
        Cmd::Bench(args) => bench::run(args),
        Cmd::Uci(args) => uci::run(args),
        Cmd::Lichess(args) => lichess::run(args),
        Cmd::Serve(args) => serve::run(args),
    }
}
