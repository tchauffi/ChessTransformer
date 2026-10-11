//! The bot's MCTS lives in `chess_core::search`, shared with selfplay-core's
//! batched match runner so that gates play exactly the search deployed here.
pub use chess_core::search::*;
