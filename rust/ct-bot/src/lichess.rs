//! Lichess Bot API client: streams account events, accepts challenges by
//! policy, and plays each game over its own ndjson game stream with the
//! batched MCTS search (one Search + ONNX session per concurrent game).
//!
//! Docs: https://lichess.org/api#tag/Bot

use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::Value;
use shakmaty::uci::UciMove;
use shakmaty::{fen::Fen, CastlingMode, Chess, Color, Move, Position};
use tokio::sync::Mutex;

use crate::nn::OnnxEvaluator;
use crate::search::{Search, SearchParams};
use crate::timeman::TimeManager;
use crate::LichessArgs;

#[derive(Clone)]
struct Client {
    http: reqwest::Client,
    base: String,
}

impl Client {
    fn new(base: &str, token: &str) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        let mut auth = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))?;
        auth.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, auth);
        Ok(Client {
            http: reqwest::Client::builder()
                .default_headers(headers)
                .user_agent("ct-bot (ChessTransformer)")
                .build()?,
            base: base.trim_end_matches('/').to_string(),
        })
    }

    async fn get_json(&self, path: &str) -> Result<Value> {
        let resp = self.http.get(format!("{}{path}", self.base)).send().await?;
        anyhow::ensure!(resp.status().is_success(), "GET {path}: {}", resp.status());
        Ok(resp.json().await?)
    }

    async fn post(&self, path: &str, form: &[(&str, &str)]) -> Result<()> {
        let resp = self
            .http
            .post(format!("{}{path}", self.base))
            .form(form)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("POST {path}: {status} {body}");
        }
        Ok(())
    }

    async fn stream(&self, path: &str) -> Result<NdjsonStream> {
        let resp = self.http.get(format!("{}{path}", self.base)).send().await?;
        anyhow::ensure!(resp.status().is_success(), "GET {path}: {}", resp.status());
        Ok(NdjsonStream { resp, buf: Vec::new() })
    }
}

/// Newline-delimited JSON stream; empty lines are lichess keep-alives.
struct NdjsonStream {
    resp: reqwest::Response,
    buf: Vec<u8>,
}

impl NdjsonStream {
    async fn next(&mut self) -> Result<Option<Value>> {
        loop {
            if let Some(i) = self.buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = self.buf.drain(..=i).collect();
                let line = &line[..line.len() - 1];
                if line.is_empty() {
                    continue;
                }
                return Ok(Some(serde_json::from_slice(line)?));
            }
            match self.resp.chunk().await? {
                Some(chunk) => self.buf.extend_from_slice(&chunk),
                None => return Ok(None),
            }
        }
    }
}

fn s<'a>(v: &'a Value, path: &[&str]) -> &'a str {
    let mut cur = v;
    for p in path {
        cur = &cur[p];
    }
    cur.as_str().unwrap_or("")
}

struct Bot {
    client: Client,
    args: LichessArgs,
    our_id: String,
    active: Arc<Mutex<HashSet<String>>>,
}

pub fn run(args: &LichessArgs) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_async(args.clone()))
}

async fn run_async(args: LichessArgs) -> Result<()> {
    let client = Client::new(&args.base_url, &args.token)?;
    let account = client.get_json("/api/account").await.context("auth check")?;
    let our_id = s(&account, &["id"]).to_string();
    let title = s(&account, &["title"]);
    anyhow::ensure!(!our_id.is_empty(), "could not read account id");
    if title != "BOT" {
        log::warn!("account '{our_id}' has no BOT title — upgrade it first (see README)");
    }
    log::info!("connected as {our_id} ({title})");

    let bot = Arc::new(Bot {
        client,
        args,
        our_id,
        active: Arc::new(Mutex::new(HashSet::new())),
    });

    loop {
        match bot.client.stream("/api/stream/event").await {
            Ok(mut events) => loop {
                match events.next().await {
                    Ok(Some(ev)) => bot.clone().handle_event(ev).await,
                    Ok(None) => break,
                    Err(e) => {
                        log::warn!("event stream error: {e:#}");
                        break;
                    }
                }
            },
            Err(e) => log::warn!("event stream connect failed: {e:#}"),
        }
        log::info!("event stream closed; reconnecting in 5s");
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

impl Bot {
    async fn handle_event(self: Arc<Self>, ev: Value) {
        match s(&ev, &["type"]) {
            "challenge" => {
                let c = &ev["challenge"];
                let id = s(c, &["id"]).to_string();
                if s(c, &["challenger", "id"]) == self.our_id {
                    return; // our own outgoing challenge
                }
                match self.challenge_verdict(c).await {
                    None => {
                        log::info!(
                            "accepting challenge {id} from {} ({} {})",
                            s(c, &["challenger", "id"]),
                            s(c, &["speed"]),
                            if c["rated"].as_bool().unwrap_or(false) { "rated" } else { "casual" },
                        );
                        if let Err(e) = self
                            .client
                            .post(&format!("/api/challenge/{id}/accept"), &[])
                            .await
                        {
                            log::warn!("accept {id}: {e:#}");
                        }
                    }
                    Some(reason) => {
                        log::info!("declining challenge {id}: {reason}");
                        let _ = self
                            .client
                            .post(&format!("/api/challenge/{id}/decline"), &[("reason", reason)])
                            .await;
                    }
                }
            }
            "gameStart" => {
                let game_id = s(&ev, &["game", "gameId"]).to_string();
                let game_id = if game_id.is_empty() {
                    s(&ev, &["game", "id"]).to_string()
                } else {
                    game_id
                };
                let mut active = self.active.lock().await;
                if !active.insert(game_id.clone()) {
                    return; // already playing it
                }
                drop(active);
                let bot = self.clone();
                tokio::spawn(async move {
                    if let Err(e) = bot.play_game(&game_id).await {
                        log::error!("game {game_id}: {e:#}");
                    }
                    bot.active.lock().await.remove(&game_id);
                    log::info!("game {game_id} finished");
                });
            }
            _ => {}
        }
    }

    /// None = accept; Some(reason) = decline reason for the API.
    async fn challenge_verdict(&self, c: &Value) -> Option<&'static str> {
        if s(c, &["variant", "key"]) != "standard" {
            return Some("variant");
        }
        if !self.args.speeds.iter().any(|sp| sp == s(c, &["speed"])) {
            return Some("timeControl");
        }
        if c["rated"].as_bool().unwrap_or(false) && !self.args.allow_rated {
            return Some("casual");
        }
        if self.active.lock().await.len() >= self.args.max_games {
            return Some("later");
        }
        None
    }

    async fn play_game(&self, game_id: &str) -> Result<()> {
        let eval = OnnxEvaluator::new(
            &self.args.model.model,
            self.args.model.threads,
            self.args.model.device == "cuda",
        )?;
        let params = SearchParams {
            sim_batch: self.args.sim_batch,
            move_temp: self.args.move_temp,
            move_temp_plies: self.args.move_temp_plies,
            ..SearchParams::default()
        };
        let mut search = Some(Search::new(eval, params));
        let mut tm = TimeManager::new(self.args.initial_sims_per_sec);

        let mut start = Chess::default();
        let mut our_color = Color::White;
        // Cumulative `moves` makes mid-game stream reconnects stateless.
        let mut attempts = 0u32;
        loop {
            let mut stream = match self
                .client
                .stream(&format!("/api/bot/game/stream/{game_id}"))
                .await
            {
                Ok(s) => s,
                Err(e) => {
                    attempts += 1;
                    anyhow::ensure!(attempts < 5, "game stream connect failed repeatedly: {e:#}");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
            };

            while let Some(msg) = stream.next().await? {
                let state = match s(&msg, &["type"]) {
                    "gameFull" => {
                        our_color = if s(&msg, &["white", "id"]) == self.our_id {
                            Color::White
                        } else {
                            Color::Black
                        };
                        let initial = s(&msg, &["initialFen"]);
                        start = if initial.is_empty() || initial == "startpos" {
                            Chess::default()
                        } else {
                            initial
                                .parse::<Fen>()?
                                .into_position(CastlingMode::Standard)?
                        };
                        log::info!("game {game_id}: playing {our_color:?}");
                        msg["state"].clone()
                    }
                    "gameState" => msg.clone(),
                    "opponentGone" => {
                        if msg["claimWinInSeconds"].as_u64() == Some(0) {
                            let _ = self
                                .client
                                .post(&format!("/api/bot/game/{game_id}/claim-victory"), &[])
                                .await;
                        }
                        continue;
                    }
                    _ => continue, // chatLine etc.
                };

                let status = s(&state, &["status"]);
                if status != "started" && !status.is_empty() {
                    log::info!("game {game_id}: over ({status})");
                    return Ok(());
                }

                let moves = parse_moves(&start, s(&state, &["moves"]))?;
                let pos_after = {
                    let mut p = start.clone();
                    for m in &moves {
                        p.play_unchecked(m);
                    }
                    p
                };
                if pos_after.turn() != our_color || pos_after.is_game_over() {
                    continue;
                }

                let (our_ms, our_inc) = if our_color == Color::White {
                    (state["wtime"].as_u64(), state["winc"].as_u64())
                } else {
                    (state["btime"].as_u64(), state["binc"].as_u64())
                };
                let (sims, deadline) = tm.plan(
                    Duration::from_millis(our_ms.unwrap_or(60_000)),
                    Duration::from_millis(our_inc.unwrap_or(0)),
                );

                // Search on a blocking thread; Search moves in and out.
                let mut sr = search.take().expect("search in use");
                let start_clone = start.clone();
                let (sr, result) = tokio::task::spawn_blocking(move || {
                    sr.set_position(start_clone, &moves);
                    let r = sr.go(sims, Some(deadline), &AtomicBool::new(false));
                    (sr, r)
                })
                .await?;
                search = Some(sr);
                let result = result?;
                tm.update(result.sims, result.elapsed);

                let uci = result.best.to_uci(CastlingMode::Standard).to_string();
                log::info!(
                    "game {game_id}: {uci} value={:+.3} sims={} ({} nn evals, {:.2}s, ~{:.0} sims/s)",
                    result.value,
                    result.sims,
                    result.nn_evals,
                    result.elapsed.as_secs_f64(),
                    tm.sims_per_sec(),
                );
                let path = format!("/api/bot/game/{game_id}/move/{uci}");
                if let Err(e) = self.client.post(&path, &[]).await {
                    log::warn!("game {game_id}: move post failed ({e:#}); retrying once");
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    if let Err(e) = self.client.post(&path, &[]).await {
                        // Typically the game ended (resign/abort) under us.
                        log::warn!("game {game_id}: move post failed again: {e:#}");
                    }
                }
            }

            attempts += 1;
            anyhow::ensure!(attempts < 20, "game stream kept dropping");
            log::info!("game {game_id}: stream dropped; reconnecting");
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }
}

fn parse_moves(start: &Chess, moves_str: &str) -> Result<Vec<Move>> {
    let mut pos = start.clone();
    let mut moves = Vec::new();
    for t in moves_str.split_whitespace() {
        let m = UciMove::from_ascii(t.as_bytes())?.to_move(&pos)?;
        pos.play_unchecked(&m);
        moves.push(m);
    }
    Ok(moves)
}
