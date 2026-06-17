//! HTTP backend for the web frontend — a drop-in replacement for the FastAPI
//! server in backend/api.py (same routes, same JSON shapes), serving moves
//! from the Rust MCTS instead of the Python bot.
//!
//! Like the Python version, one session board is retained so the search keeps
//! its move history (threefold awareness) and tree reuse across requests; a
//! FEN that doesn't extend the session simply starts a fresh history.

use std::sync::Arc;

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use shakmaty::{fen::Fen, CastlingMode, Chess, Color, EnPassantMode, Move, Position};
use tokio::sync::Mutex;

use crate::nn::OnnxEvaluator;
use crate::search::{Search, SearchParams};
use crate::ServeArgs;

struct Session {
    search: Search<OnnxEvaluator>,
    start: Chess,
    moves: Vec<Move>,
    pos: Chess, // start with moves applied
}

struct App {
    session: Mutex<Session>,
    sims: u32,
    model_name: String,
    approx_elo: u32,
}

fn fen_of(pos: &Chess) -> String {
    Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string()
}

fn game_over(pos: &Chess) -> bool {
    pos.is_game_over() || pos.halfmoves() >= 150
}

fn result_str(pos: &Chess) -> Option<String> {
    if !game_over(pos) {
        return None;
    }
    Some(match pos.outcome() {
        Some(shakmaty::Outcome::Decisive { winner: Color::White }) => "1-0".into(),
        Some(shakmaty::Outcome::Decisive { winner: Color::Black }) => "0-1".into(),
        _ => "1/2-1/2".into(),
    })
}

fn parse_fen(fen: &str) -> Result<Chess, (StatusCode, String)> {
    fen.parse::<Fen>()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid FEN string: {e}")))?
        .into_position(CastlingMode::Standard)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid FEN string: {e}")))
}

impl Session {
    /// Mirror of api.py `_resolve_board`: keep the session history when the
    /// requested FEN equals the session position or extends it by one legal
    /// move (the human's reply); otherwise restart history from the FEN.
    fn resolve(&mut self, req: &Chess) {
        use chess_core::encoding::zobrist;
        let want = zobrist(req);
        if zobrist(&self.pos) == want {
            return;
        }
        for m in self.pos.legal_moves() {
            let mut next = self.pos.clone();
            next.play_unchecked(&m);
            if zobrist(&next) == want {
                self.moves.push(m);
                self.pos = next;
                return;
            }
        }
        self.start = req.clone();
        self.moves.clear();
        self.pos = req.clone();
    }

    fn push(&mut self, m: Move) {
        self.pos.play_unchecked(&m);
        self.moves.push(m);
    }
}

#[derive(Deserialize)]
struct FenReq {
    fen: String,
}

#[derive(Deserialize)]
struct ValidateReq {
    fen: String,
    #[serde(rename = "move")]
    mv: String,
}

type Resp = Result<Json<Value>, (StatusCode, String)>;

async fn health(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "bot_type": "CtBot (Rust MCTS)",
        "engine": "mcts",
        "model": app.model_name,
        "sims": app.sims,
        "approx_elo": app.approx_elo,
    }))
}

async fn bot_move(State(app): State<Arc<App>>, Json(req): Json<FenReq>) -> Resp {
    let req_pos = parse_fen(&req.fen)?;
    if game_over(&req_pos) {
        return Err((StatusCode::BAD_REQUEST, "Game is already over".into()));
    }
    let sims = app.sims;
    let result = tokio::task::spawn_blocking(move || {
        let app = app;
        let mut session = app.session.blocking_lock();
        session.resolve(&req_pos);
        let (start, moves) = (session.start.clone(), session.moves.clone());
        session.search.set_position(start, &moves);
        let r = session
            .search
            .go(sims, None, &std::sync::atomic::AtomicBool::new(false))
            .map_err(|e| e.to_string())?;
        session.push(r.best.clone());
        let pos = session.pos.clone();
        Ok::<_, String>(json!({
            "move": r.best.to_uci(CastlingMode::Standard).to_string(),
            "probability": r.value,
            "fen": fen_of(&pos),
            "game_over": game_over(&pos),
            "result": result_str(&pos),
            "value": r.value,
        }))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn new_game(State(app): State<Arc<App>>) -> Json<Value> {
    let mut session = app.session.lock().await;
    session.search.new_game();
    session.start = Chess::default();
    session.moves.clear();
    session.pos = Chess::default();
    Json(json!({ "fen": fen_of(&session.pos), "game_over": false }))
}

async fn validate_move(Json(req): Json<ValidateReq>) -> Resp {
    let pos = parse_fen(&req.fen)?;
    let Ok(uci) = shakmaty::uci::UciMove::from_ascii(req.mv.as_bytes()) else {
        return Ok(Json(json!({ "valid": false, "error": "Invalid move format" })));
    };
    match uci.to_move(&pos) {
        Ok(m) => {
            let mut next = pos.clone();
            next.play_unchecked(&m);
            Ok(Json(json!({
                "valid": true,
                "fen": fen_of(&next),
                "game_over": game_over(&next),
                "result": result_str(&next),
            })))
        }
        Err(_) => Ok(Json(json!({ "valid": false, "error": "Illegal move" }))),
    }
}

async fn evaluate(State(app): State<Arc<App>>, Json(req): Json<FenReq>) -> Resp {
    let pos = parse_fen(&req.fen)?;
    let value = tokio::task::spawn_blocking(move || {
        let app = app;
        let mut session = app.session.blocking_lock();
        session.search.raw_value(&pos).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({ "value": value })))
}

pub fn run(args: &ServeArgs) -> Result<()> {
    let eval = OnnxEvaluator::new(
        &args.model.model,
        args.model.threads,
        args.model.device == "cuda",
    )?;
    let search = Search::new(
        eval,
        SearchParams { sim_batch: args.sim_batch, ..SearchParams::default() },
    );
    let app = Arc::new(App {
        session: Mutex::new(Session {
            search,
            start: Chess::default(),
            moves: Vec::new(),
            pos: Chess::default(),
        }),
        sims: args.sims,
        model_name: args
            .model
            .model
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".into()),
        approx_elo: args.approx_elo,
    });

    let router = axum::Router::new()
        .route("/api/health", get(health))
        .route("/api/move", post(bot_move))
        .route("/api/new-game", post(new_game))
        .route("/api/validate-move", post(validate_move))
        .route("/api/evaluate", post(evaluate))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(app);

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let addr = format!("0.0.0.0:{}", args.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        log::info!("serving on http://{addr} (sims={})", args.sims);
        axum::serve(listener, router).await?;
        Ok(())
    })
}
