//! Shared chess primitives for the ChessTransformer Rust crates: model
//! position/move encodings (exact mirrors of the Python tokenizers) and the
//! edge-centric MCTS tree with PUCT selection / virtual-loss backup.
//!
//! Used by `selfplay-core` (PyO3 generation engine) and `ct-bot` (standalone
//! bot binary). Encoding parity with Python is guarded by
//! `scripts/check_rust_parity.py`.

pub mod encoding;
pub mod tree;
