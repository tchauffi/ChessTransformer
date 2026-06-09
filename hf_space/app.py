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


def _human_color(human_color: str) -> bool:
    return chess.BLACK if human_color == "black" else chess.WHITE


def _result_text(board: chess.Board, human_color: str) -> str:
    outcome = board.outcome()
    if outcome is None:
        return ""
    if outcome.winner is None:
        return f"Draw ({outcome.termination.name.lower().replace('_', ' ')}). New game?"
    who = "You win! 🎉" if outcome.winner == _human_color(human_color) else "ChessTransformer wins."
    return f"Checkmate — {who} New game?"


def _bot_move(board: chess.Board, sims: int):
    """Play the side to move with the bot; return its SAN and value."""
    bot.num_simulations = int(sims)
    uci, value = bot.predict(board)
    san = board.san(chess.Move.from_uci(uci))
    board.push_uci(uci)
    return san, value


def on_move(fen: str, sims: int, human_color: str):
    """Human just moved; reply with the bot's move (whichever side is to move)."""
    board = chess.Board(fen)
    if board.is_game_over():
        return board.fen(), _result_text(board, human_color)

    san, value = _bot_move(board, sims)

    if board.is_game_over():
        status = f"Bot played **{san}**.  {_result_text(board, human_color)}"
    else:
        eval_str = f"  (eval {value:+.2f})" if value is not None else ""
        status = f"Bot played **{san}**{eval_str}.  Your move."
    return board.fen(), status


def new_game(play_as: str, sims: int):
    """Start a fresh game. If the human plays Black, the bot (White) opens."""
    human_color = "black" if play_as == "Black" else "white"
    orientation = human_color
    board = chess.Board()
    if human_color == "black":
        san, _ = _bot_move(board, sims)
        status = f"New game — you are **Black**. Bot opened with **{san}**. Your move."
    else:
        status = "New game — you are **White**. Make your move."
    return gr.update(value=board.fen(), orientation=orientation), status, human_color


with gr.Blocks(title="ChessTransformer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# ♟️ ChessTransformer\n"
        "An **11.7M-parameter transformer trained only on human games** "
        "(no self-play, no RL), playing via AlphaZero-style MCTS. "
        "It reaches **~2100 Elo** vs Stockfish at full strength; this CPU demo "
        "runs at a lower sim count so moves come back quickly.\n\n"
        "**Pick your color and hit New game.** Drag a piece to move — the bot "
        "replies automatically.  "
        "[GitHub repo →](https://github.com/tchauffi/ChessTransformer)"
    )
    human_color = gr.State("white")
    with gr.Row():
        with gr.Column(scale=3):
            board = Chessboard(value=START_FEN, game_mode=True, orientation="white",
                               label="Drag to move")
        with gr.Column(scale=1):
            status = gr.Markdown("You are **White**. Make your move.")
            play_as = gr.Radio(["White", "Black"], value="White", label="Play as")
            sims = gr.Slider(32, 1200, value=DEFAULT_SIMS, step=8, label="Engine strength (MCTS sims/move)",
                             info="Higher = stronger but slower on CPU")
            new_btn = gr.Button("New game", variant="primary")

    board.move(on_move, inputs=[board, sims, human_color], outputs=[board, status])
    new_btn.click(new_game, inputs=[play_as, sims], outputs=[board, status, human_color])


if __name__ == "__main__":
    demo.launch()
