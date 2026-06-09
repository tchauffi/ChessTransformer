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
    # Sample the opening moves (∝ visit count) so games aren't identical every
    # time; switches to deterministic best-move play after the opening.
    move_temp=1.0,
    move_temp_plies=12,
)


def _human_color(human_color: str) -> bool:
    return chess.BLACK if human_color == "black" else chess.WHITE


def _white_pov_value(board: chess.Board) -> float:
    """Value-head eval in [-1, 1] from White's perspective (+ = White better)."""
    if board.is_game_over():
        if board.is_checkmate():
            return -1.0 if board.turn == chess.WHITE else 1.0
        return 0.0  # stalemate / draw
    v = bot.get_value(board)
    if v is None:
        return 0.0
    return v if board.turn == chess.WHITE else -v


def eval_bar_html(board: chess.Board, orientation: str) -> str:
    """A vertical eval bar (White share fills from the bottom of White's side).
    Flipped to match the board orientation so the viewer's side is at the bottom."""
    wv = _white_pov_value(board)
    white_pct = round(max(0.0, min(1.0, (wv + 1) / 2)) * 100, 1)
    white_div = f'<div style="height:{white_pct}%;background:#f5f5f5;"></div>'
    black_div = f'<div style="height:{round(100 - white_pct, 1)}%;background:#3a3a3a;"></div>'
    # orientation "white": Black on top, White on bottom; flipped when "black".
    segments = black_div + white_div if orientation == "white" else white_div + black_div
    return (
        f'<div title="Model eval (White POV): {wv:+.2f}" '
        'style="display:flex;flex-direction:column;width:26px;height:460px;'
        'border:1px solid #999;border-radius:4px;overflow:hidden;">'
        f'{segments}</div>'
    )


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
    """Human just moved; reply with the bot's move (whichever side is to move).

    A generator: first yields a "thinking" status (so the player sees the bot is
    working during the ~1-3s CPU search), then yields the bot's reply. The board
    is shown from White's side, so the eval bar is White-oriented."""
    board = chess.Board(fen)
    if board.is_game_over():
        yield board.fen(), _result_text(board, human_color), eval_bar_html(board, "white")
        return

    # Immediate feedback while the search runs (board/eval unchanged for now).
    yield gr.update(), "⏳ ChessTransformer is thinking…", gr.update()

    san, value = _bot_move(board, sims)

    if board.is_game_over():
        status = f"Bot played **{san}**.  {_result_text(board, human_color)}"
    else:
        eval_str = f"  (eval {value:+.2f})" if value is not None else ""
        status = f"Bot played **{san}**{eval_str}.  Your move."
    yield board.fen(), status, eval_bar_html(board, "white")


def new_game(play_as: str, sims: int):
    """Start a fresh game. If the human plays Black, the bot (White) opens.

    Returns (fen, status, eval_bar_html, human_color) — written straight to the
    statically-mounted board so a same-colour replay still resets it."""
    human_color = "black" if play_as == "Black" else "white"
    board = chess.Board()
    if human_color == "black":
        san, _ = _bot_move(board, sims)
        status = f"New game — you are **Black**. Bot opened with **{san}**. Your move."
    else:
        status = "New game — you are **White**. Make your move."
    return board.fen(), status, eval_bar_html(board, "white"), human_color


with gr.Blocks(title="ChessTransformer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# ♟️ ChessTransformer\n"
        "An **11.7M-parameter transformer trained only on human games** "
        "(no self-play, no RL), playing via AlphaZero-style MCTS. "
        "It reaches **~2100 Elo** vs Stockfish at full strength; this CPU demo "
        "runs at a lower sim count so moves come back quickly.\n\n"
        "**Pick your color and hit New game.** Drag a piece to move — the bot "
        "replies automatically. (The board is shown from White's side.)  "
        "[GitHub repo →](https://github.com/tchauffi/ChessTransformer)"
    )
    human_color = gr.State("white")

    with gr.Row():
        # Solid min_widths so the board container always has a size at mount —
        # the chessboard.js component reloads the page if it inits at 0 width.
        eval_col = gr.Column(scale=0, min_width=44)
        board_col = gr.Column(scale=3, min_width=320)
        ctrl_col = gr.Column(scale=1, min_width=200)

    with eval_col:
        eval_bar = gr.HTML(eval_bar_html(chess.Board(), "white"), visible=False)

    # Board mounted statically (not in a gr.render) so it initializes once on the
    # laid-out page — the dynamic remount was failing on mobile and triggering the
    # component's window.location.reload() loop.
    with board_col:
        board = Chessboard(value=START_FEN, game_mode=True, orientation="white",
                           label="Drag to move", min_width=320)
        status = gr.Markdown("You are **White**. Make your move.")

    with ctrl_col:
        play_as = gr.Radio(["White", "Black"], value="White", label="Play as")
        sims = gr.Slider(32, 1200, value=DEFAULT_SIMS, step=8,
                         label="Engine strength (MCTS sims/move)",
                         info="Higher = stronger but slower on CPU")
        show_eval = gr.Checkbox(False, label="Show evaluation bar")
        new_btn = gr.Button("New game", variant="primary")

    # on_move is a generator → streams the "thinking" status; show_progress hidden
    # so Gradio's overlay doesn't sit over the board.
    board.move(on_move, inputs=[board, sims, human_color],
               outputs=[board, status, eval_bar], show_progress="hidden")
    show_eval.change(lambda show: gr.update(visible=show), inputs=show_eval, outputs=eval_bar)
    new_btn.click(new_game, inputs=[play_as, sims],
                  outputs=[board, status, eval_bar, human_color])


if __name__ == "__main__":
    demo.launch()
