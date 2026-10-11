//! Shared chess primitives for the ChessTransformer Rust crates: model
//! position/move encodings (exact mirrors of the Python tokenizers), the
//! edge-centric MCTS tree with PUCT selection / virtual-loss backup, and the
//! bot's search itself (`search`).
//!
//! Used by `selfplay-core` (PyO3 generation engine and batched match runner)
//! and `ct-bot` (standalone bot binary). The match runner and the deployed bot
//! run the same `search::Search`, so a gate measures the search that plays on
//! lichess. Encoding parity with Python is guarded by
//! `scripts/check_rust_parity.py`.

pub mod encoding;
pub mod eval;
pub mod matchplay;
pub mod search;
pub mod selfplay;
pub mod tree;
