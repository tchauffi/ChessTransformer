"""ChessTransformer — Hugging Face Space (Gradio).

Play the transformer chess engine in the browser. Runs on CPU: the model is
only 11.7M params, so MCTS at a modest sim count returns a move in ~1-2s.

Weights are resolved in this order:
  1. ``MODEL_PATH`` env var (a checkpoint dir with model.safetensors + config)
  2. a ``model/`` dir next to this file (how ``prepare.sh`` ships the Space)
  3. the repo's ``data/models/pos2move_v2.1`` (for local dev runs)
  4. download from GitHub (LFS media) as a last resort
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chess
import gradio as gr
from gradio_chessboard import Chessboard

# ── make `chesstransformer` importable ──────────────────────────────────────
# On the Space the package is vendored next to this file (cwd is on sys.path).
# For local dev runs from the repo, add the repo's src/ to the path.
_repo_src = Path(__file__).resolve().parents[1] / "src"
if (_repo_src / "chesstransformer").exists():
    sys.path.insert(0, str(_repo_src))

from chesstransformer.bots import Pos2MoveV2MctsBot  # noqa: E402

START_FEN = chess.STARTING_FEN
DEFAULT_SIMS = int(os.environ.get("MCTS_SIMS", "400"))

# GitHub locations for the last-resort weight download.
_GH = "tchauffi/ChessTransformer"
_BRANCH = os.environ.get("MODEL_BRANCH", "main")
_RAW = f"https://raw.githubusercontent.com/{_GH}/{_BRANCH}/data/models/pos2move_v2.1"
_LFS = f"https://media.githubusercontent.com/media/{_GH}/{_BRANCH}/data/models/pos2move_v2.1"


def _download_weights() -> str:
    import urllib.request

    dest = Path("/tmp/pos2move_v2.1")
    dest.mkdir(parents=True, exist_ok=True)
    for fname, url in [("model_config.json", f"{_RAW}/model_config.json"),
                       ("model.safetensors", f"{_LFS}/model.safetensors")]:
        out = dest / fname
        if not out.exists():
            print(f"Downloading {fname} …")
            urllib.request.urlretrieve(url, out)
    return str(dest)


def resolve_model_dir() -> str:
    env = os.environ.get("MODEL_PATH")
    if env and (Path(env) / "model.safetensors").exists():
        return env
    for cand in (
        Path(__file__).resolve().parent / "model",
        Path("model"),
        Path(__file__).resolve().parents[1] / "data" / "models" / "pos2move_v2.1",
    ):
        if (cand / "model.safetensors").exists():
            return str(cand)
    return _download_weights()


# ── load the bot once (CPU, no compile) ─────────────────────────────────────
MODEL_DIR = resolve_model_dir()
bot = Pos2MoveV2MctsBot(
    model_dir=MODEL_DIR,
    num_simulations=DEFAULT_SIMS,
    device="cpu",
    compile=False,
    time_limit=0.0,
)


def _result_text(board: chess.Board) -> str:
    outcome = board.outcome()
    if outcome is None:
        return ""
    if outcome.winner is None:
        return f"Draw ({outcome.termination.name.lower().replace('_', ' ')}). New game?"
    who = "You win! 🎉" if outcome.winner == chess.WHITE else "ChessTransformer wins."
    return f"Checkmate — {who} New game?"


def on_move(fen: str, sims: int):
    """Human just moved (White). Reply with the bot's move (Black)."""
    board = chess.Board(fen)
    if board.is_game_over():
        return fen, _result_text(board)

    bot.num_simulations = int(sims)
    uci, value = bot.predict(board)
    board.push_uci(uci)

    move_san = "—"
    tmp = chess.Board(fen)
    try:
        move_san = tmp.san(chess.Move.from_uci(uci))
    except Exception:
        move_san = uci

    if board.is_game_over():
        status = f"Bot played **{move_san}**.  {_result_text(board)}"
    else:
        eval_str = f"  (eval {value:+.2f})" if value is not None else ""
        status = f"Bot played **{move_san}**{eval_str}.  Your move."
    return board.fen(), status


def new_game():
    return START_FEN, "New game — you are **White**. Make your move."


with gr.Blocks(title="ChessTransformer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# ♟️ ChessTransformer\n"
        "An **11.7M-parameter transformer trained only on human games** "
        "(no self-play, no RL), playing via AlphaZero-style MCTS. "
        "It reaches **~2100 Elo** vs Stockfish at full strength; this CPU demo "
        "runs at a lower sim count so moves come back quickly.\n\n"
        "**You play White.** Drag a piece to move — the bot replies automatically.  "
        "[GitHub repo →](https://github.com/tchauffi/ChessTransformer)"
    )
    with gr.Row():
        with gr.Column(scale=3):
            board = Chessboard(value=START_FEN, game_mode=True, orientation="white",
                               label="Drag to move (you are White)")
        with gr.Column(scale=1):
            status = gr.Markdown("You are **White**. Make your move.")
            sims = gr.Slider(32, 1200, value=DEFAULT_SIMS, step=8, label="Engine strength (MCTS sims/move)",
                             info="Higher = stronger but slower on CPU")
            new_btn = gr.Button("New game", variant="primary")

    board.move(on_move, inputs=[board, sims], outputs=[board, status])
    new_btn.click(new_game, outputs=[board, status])


if __name__ == "__main__":
    demo.launch()
