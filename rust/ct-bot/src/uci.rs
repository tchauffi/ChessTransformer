//! Minimal-but-correct UCI engine front-end: enough for python-chess
//! `SimpleEngine` (the Stockfish gauntlet / sims-scaling driver) and
//! cutechess-cli. `position` carries the cumulative move list, so the search's
//! threefold history and tree reuse work exactly as in the Python bot.

use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, Move, Position};

use crate::nn::OnnxEvaluator;
use crate::search::{Search, SearchParams};
use crate::timeman::TimeManager;
use crate::UciArgs;

fn value_to_cp(v: f32) -> i32 {
    (v.clamp(-0.999, 0.999).atanh() * 290.0) as i32
}

struct Options {
    model: std::path::PathBuf,
    threads: usize,
    device_cuda: bool,
    sims: u32,
    params: SearchParams,
}

struct Uci {
    opts: Options,
    search: Option<Search<OnnxEvaluator>>,
    start: Chess,
    moves: Vec<Move>,
    tm: TimeManager,
    stop: Arc<AtomicBool>,
}

impl Uci {
    fn ensure_search(&mut self) -> Result<&mut Search<OnnxEvaluator>> {
        if self.search.is_none() {
            let eval = OnnxEvaluator::new(&self.opts.model, self.opts.threads, self.opts.device_cuda)?;
            let params = SearchParams {
                c_puct: self.opts.params.c_puct,
                fpu: self.opts.params.fpu,
                prior_temp: self.opts.params.prior_temp,
                sim_batch: self.opts.params.sim_batch,
                ..SearchParams::default()
            };
            let mut search = Search::new(eval, params);
            search.set_position(self.start.clone(), &self.moves);
            self.search = Some(search);
        }
        Ok(self.search.as_mut().unwrap())
    }

    fn cmd_uci(&self) {
        println!("id name ChessTransformer-MCTS");
        println!("id author tchauffi");
        println!(
            "option name Model type string default {}",
            self.opts.model.display()
        );
        println!("option name Sims type spin default {} min 1 max 10000000", self.opts.sims);
        println!(
            "option name SimBatch type spin default {} min 1 max 256",
            self.opts.params.sim_batch
        );
        println!("option name Threads type spin default {} min 0 max 64", self.opts.threads);
        println!("option name CPuct type string default {}", self.opts.params.c_puct);
        println!("option name FPU type string default {}", self.opts.params.fpu);
        println!("option name PriorTemp type string default {}", self.opts.params.prior_temp);
        println!("uciok");
    }

    fn cmd_setoption(&mut self, line: &str) {
        // "setoption name <Name> [value <Value>]"
        let rest = line.trim_start_matches("setoption").trim();
        let rest = rest.strip_prefix("name").map(str::trim).unwrap_or(rest);
        let (name, value) = match rest.split_once(" value ") {
            Some((n, v)) => (n.trim(), v.trim()),
            None => (rest, ""),
        };
        match name.to_ascii_lowercase().as_str() {
            "model" => {
                self.opts.model = value.into();
                self.search = None; // rebuilt on next isready/go
            }
            "threads" => {
                if let Ok(v) = value.parse() {
                    self.opts.threads = v;
                    self.search = None;
                }
            }
            "sims" => {
                if let Ok(v) = value.parse() {
                    self.opts.sims = v;
                }
            }
            "simbatch" => {
                if let Ok(v) = value.parse() {
                    self.opts.params.sim_batch = v;
                    if let Some(s) = &mut self.search {
                        s.params.sim_batch = v;
                    }
                }
            }
            "cpuct" => {
                if let Ok(v) = value.parse() {
                    self.opts.params.c_puct = v;
                    if let Some(s) = &mut self.search {
                        s.params.c_puct = v;
                    }
                }
            }
            "fpu" => {
                if let Ok(v) = value.parse() {
                    self.opts.params.fpu = v;
                    if let Some(s) = &mut self.search {
                        s.params.fpu = v;
                    }
                }
            }
            "priortemp" => {
                if let Ok(v) = value.parse() {
                    self.opts.params.prior_temp = v;
                    if let Some(s) = &mut self.search {
                        s.params.prior_temp = v;
                    }
                }
            }
            _ => log::warn!("unknown option: {name}"),
        }
    }

    fn cmd_position(&mut self, line: &str) -> Result<()> {
        let mut tokens = line.split_whitespace().skip(1).peekable(); // skip "position"
        let start: Chess = match tokens.next() {
            Some("startpos") => Chess::default(),
            Some("fen") => {
                let mut fen_parts = Vec::new();
                while let Some(&t) = tokens.peek() {
                    if t == "moves" {
                        break;
                    }
                    fen_parts.push(tokens.next().unwrap());
                }
                let fen: Fen = fen_parts.join(" ").parse()?;
                fen.into_position(CastlingMode::Standard)?
            }
            other => anyhow::bail!("bad position command: {other:?}"),
        };
        let mut moves = Vec::new();
        if tokens.peek() == Some(&"moves") {
            tokens.next();
            let mut pos = start.clone();
            for t in tokens {
                let m = UciMove::from_ascii(t.as_bytes())?.to_move(&pos)?;
                pos.play_unchecked(&m);
                moves.push(m);
            }
        }
        self.start = start.clone();
        self.moves = moves;
        if self.search.is_some() {
            let mvs = self.moves.clone();
            self.ensure_search()?.set_position(start, &mvs);
        }
        Ok(())
    }

    fn cmd_go(&mut self, line: &str) -> Result<()> {
        let mut nodes: Option<u32> = None;
        let mut movetime: Option<u64> = None;
        let mut wtime: Option<u64> = None;
        let mut btime: Option<u64> = None;
        let mut winc: u64 = 0;
        let mut binc: u64 = 0;
        let mut infinite = false;
        let mut tokens = line.split_whitespace().skip(1);
        while let Some(t) = tokens.next() {
            let mut next_u64 = || tokens.next().and_then(|v| v.parse::<u64>().ok());
            match t {
                "nodes" => nodes = next_u64().map(|v| v as u32),
                "movetime" => movetime = next_u64(),
                "wtime" => wtime = next_u64(),
                "btime" => btime = next_u64(),
                "winc" => winc = next_u64().unwrap_or(0),
                "binc" => binc = next_u64().unwrap_or(0),
                "infinite" => infinite = true,
                _ => {}
            }
        }

        let default_sims = self.opts.sims;
        let turn = {
            let search = self.ensure_search()?;
            if search.position().legal_moves().is_empty() {
                println!("bestmove 0000");
                return Ok(());
            }
            search.position().turn()
        };
        let (sims, deadline) = if let Some(n) = nodes {
            (n, None)
        } else if let Some(ms) = movetime {
            let margin = Duration::from_millis(30.min(ms / 10));
            (u32::MAX, Some(Instant::now() + Duration::from_millis(ms) - margin))
        } else if wtime.is_some() || btime.is_some() {
            let (our_time, our_inc) = if turn == Color::White {
                (wtime.unwrap_or(0), winc)
            } else {
                (btime.unwrap_or(0), binc)
            };
            let (sims, deadline) = self
                .tm
                .plan(Duration::from_millis(our_time), Duration::from_millis(our_inc));
            (sims, Some(deadline))
        } else if infinite {
            (u32::MAX, None)
        } else {
            (default_sims, None)
        };

        self.stop.store(false, Ordering::Relaxed);
        let stop = self.stop.clone();
        let result = self.ensure_search()?.go(sims, deadline, &stop)?;
        self.tm.update(result.sims, result.elapsed);

        let ms = result.elapsed.as_millis().max(1);
        println!(
            "info depth 1 nodes {} nps {} time {} score cp {} pv {}",
            result.sims,
            (result.sims as u128 * 1000) / ms,
            ms,
            value_to_cp(result.value),
            result.best.to_uci(CastlingMode::Standard)
        );
        println!("bestmove {}", result.best.to_uci(CastlingMode::Standard));
        Ok(())
    }
}

pub fn run(args: &UciArgs) -> Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<String>();

    // Reader thread: `stop`/`quit` abort an in-flight search immediately (the
    // flag is checked between waves); all lines but `stop` queue in order for
    // the main loop, which exits when it reaches `quit`.
    {
        let stop = stop.clone();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            for line in stdin.lock().lines() {
                let Ok(line) = line else { break };
                let cmd = line.trim();
                if cmd == "stop" || cmd == "quit" {
                    stop.store(true, Ordering::Relaxed);
                    if cmd == "stop" {
                        continue;
                    }
                }
                if tx.send(line).is_err() {
                    break;
                }
            }
            // EOF: behave like quit so the engine exits cleanly.
            stop.store(true, Ordering::Relaxed);
            let _ = tx.send("quit".to_string());
        });
    }

    let mut uci = Uci {
        opts: Options {
            model: args.model.model.clone(),
            threads: args.model.threads,
            device_cuda: args.model.device == "cuda",
            sims: args.sims,
            params: SearchParams {
                sim_batch: args.sim_batch,
                c_puct: args.c_puct,
                fpu: args.fpu,
                prior_temp: args.prior_temp,
                ..SearchParams::default()
            },
        },
        search: None,
        start: Chess::default(),
        moves: Vec::new(),
        tm: TimeManager::new(args.initial_sims_per_sec),
        stop: stop.clone(),
    };

    for line in rx {
        let cmd = line.trim();
        let head = cmd.split_whitespace().next().unwrap_or("");
        let outcome = match head {
            "uci" => {
                uci.cmd_uci();
                Ok(())
            }
            "setoption" => {
                uci.cmd_setoption(cmd);
                Ok(())
            }
            "isready" => uci.ensure_search().map(|_| println!("readyok")),
            "ucinewgame" => {
                if let Some(s) = &mut uci.search {
                    s.new_game();
                }
                uci.start = Chess::default();
                uci.moves.clear();
                Ok(())
            }
            "position" => uci.cmd_position(cmd),
            "go" => uci.cmd_go(cmd),
            "quit" => break,
            "" | "stop" => Ok(()),
            _ => {
                log::warn!("unknown command: {cmd}");
                Ok(())
            }
        };
        if let Err(e) = outcome {
            log::error!("{head}: {e:#}");
            println!("info string error: {e}");
        }
    }
    Ok(())
}
