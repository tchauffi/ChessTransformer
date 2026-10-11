//! Shared chess primitives for the ChessTransformer Rust crates: model
//! position/move encodings (exact mirrors of the Python tokenizers), the
//! edge-centric MCTS tree with PUCT selection / virtual-loss backup, and the
//! bot's search itself (`search`).
//!
//! Used by `selfplay-core` (PyO3 generation engine) and `ct-bot` (standalone
//! bot binary). Encoding parity with Python is guarded by
//! `scripts/check_rust_parity.py`.

pub mod encoding;
pub mod eval;
pub mod search;
pub mod tree;
