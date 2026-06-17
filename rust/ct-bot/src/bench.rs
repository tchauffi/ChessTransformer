//! Throughput benchmark: raw NN evals/sec at several batch sizes, and (once
//! the search exists) per-move search timings at several sim budgets.

use std::time::Instant;

use anyhow::Result;
use rand::prelude::*;
use rand::rngs::StdRng;
use shakmaty::{Chess, Position};

use crate::nn::{EvalRequest, Evaluator};
use crate::BenchArgs;

/// Random reachable positions (random playouts from the start position).
fn random_positions(n: usize, seed: u64) -> Vec<Chess> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut pos = Chess::default();
        for _ in 0..rng.gen_range(10..80) {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            pos.play_unchecked(moves.choose(&mut rng).unwrap());
            if pos.is_game_over() {
                break;
            }
        }
        out.push(pos);
    }
    out
}

fn requests(positions: &[Chess]) -> Vec<EvalRequest> {
    positions
        .iter()
        .map(|p| {
            let (tokens, player, castling, ep) = chess_core::encoding::encode_position(p);
            EvalRequest { tokens, player, castling, ep }
        })
        .collect()
}

pub fn run(args: &BenchArgs) -> Result<()> {
    let mut eval = args.model.evaluator()?;
    let positions = random_positions(256, 0);
    let reqs = requests(&positions);

    println!("── NN throughput ({}) ──", args.model.model.display());
    println!("{:>6} {:>12} {:>12}", "batch", "ms/batch", "evals/s");
    for &batch in &[1usize, 8, 16, 32, 64, 128] {
        // warmup
        for chunk in reqs.chunks(batch).take(3) {
            eval.eval_batch(chunk)?;
        }
        let t0 = Instant::now();
        let mut evals = 0usize;
        while t0.elapsed().as_secs_f64() < 2.0 {
            for chunk in reqs.chunks(batch) {
                if chunk.len() < batch {
                    break;
                }
                eval.eval_batch(chunk)?;
                evals += batch;
                if t0.elapsed().as_secs_f64() >= 2.0 {
                    break;
                }
            }
        }
        let el = t0.elapsed().as_secs_f64();
        println!(
            "{:>6} {:>12.2} {:>12.0}",
            batch,
            el / (evals as f64 / batch as f64) * 1e3,
            evals as f64 / el
        );
    }

    bench_search(args)
}

/// Per-move search timings at each sim budget on startpos + 3 middlegame
/// positions.
fn bench_search(args: &BenchArgs) -> Result<()> {
    use crate::search::{Search, SearchParams};
    use shakmaty::fen::Fen;
    use shakmaty::CastlingMode;
    use std::sync::atomic::AtomicBool;

    const FENS: [&str; 3] = [
        // Italian, Najdorf and QGD middlegames.
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b kq - 5 5",
        "rn1qkb1r/1p2pppp/p2p1n2/8/3NP1b1/2N1B3/PPP2PPP/R2QKB1R w KQkq - 2 7",
        "r1bq1rk1/pp2bppp/2n1pn2/3p2B1/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 6 8",
    ];

    println!("\n── Search throughput (sim_batch={}) ──", args.sim_batch);
    println!(
        "{:>7} {:>10} {:>10} {:>10} {:>9}",
        "sims", "s/move", "sims/s", "nn evals", "tt hits"
    );
    let stop = AtomicBool::new(false);
    for &sims in &args.sims {
        let eval = args.model.evaluator()?;
        let mut search = Search::new(
            eval,
            SearchParams { sim_batch: args.sim_batch, ..SearchParams::default() },
        );
        let (mut secs, mut nn, mut hits) = (0.0f64, 0u64, 0u64);
        for fen in [None, Some(FENS[0]), Some(FENS[1]), Some(FENS[2])] {
            let pos = match fen {
                None => Chess::default(),
                Some(f) => f
                    .parse::<Fen>()
                    .unwrap()
                    .into_position(CastlingMode::Standard)
                    .unwrap(),
            };
            search.set_position(pos, &[]);
            let r = search.go(sims, None, &stop)?;
            secs += r.elapsed.as_secs_f64();
            nn += r.nn_evals as u64;
            hits += r.tt_hits as u64;
        }
        println!(
            "{:>7} {:>10.2} {:>10.0} {:>10} {:>9}",
            sims,
            secs / 4.0,
            sims as f64 * 4.0 / secs,
            nn / 4,
            hits / 4
        );
    }
    Ok(())
}
