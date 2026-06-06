"""
FastAPI server for ChessTransformer bot integration.
Provides endpoints for the Next.js frontend to interact with the chess bot.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import os

from chesstransformer.bots import Pos2MoveV2Bot, Pos2MoveV2MctsBot, RandomBot

app = FastAPI(title="ChessTransformer API")

_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_allow_origins = [o.strip() for o in _origins.split(",")] if _origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set MODEL_PATH env var to point to a specific checkpoint directory.
# ENGINE selects the search: "mcts" (default, strongest — won 82% head-to-head
# vs the alpha-beta engine) or "alphabeta". MCTS_SIMS tunes MCTS strength/speed.
model_path = os.environ.get("MODEL_PATH")
engine = os.environ.get("ENGINE", "mcts").lower()
engine_info = {"engine": "none", "model": None, "sims": None, "approx_elo": None}
try:
    kwargs = {}
    if model_path:
        kwargs["model_dir"] = model_path
    if engine == "alphabeta":
        bot = Pos2MoveV2Bot(**kwargs)
        bot_type = "Pos2MoveV2Bot (alpha-beta)"
        engine_info.update(engine="alpha-beta", approx_elo=1550)
    else:
        sims = int(os.environ.get("MCTS_SIMS", "800"))
        kwargs["num_simulations"] = sims
        bot = Pos2MoveV2MctsBot(**kwargs)
        bot_type = "Pos2MoveV2MctsBot (MCTS)"
        engine_info.update(engine="mcts", sims=sims, approx_elo=2100)
    engine_info["model"] = os.path.basename(os.path.normpath(model_path)) if model_path else "pos2move_v2.1"
except Exception as e:
    print(f"Warning: Could not load Pos2MoveV2 bot: {e}")
    print("Falling back to RandomBot")
    bot = RandomBot()
    bot_type = "RandomBot"


# The API is otherwise stateless (each request carries a FEN, which has no move
# history). We keep one persistent game board so the MCTS engine receives full
# move history and its tree-reuse stays active across moves. A single-game demo:
# concurrent games simply fall back to a fresh (history-less) tree, which is safe.
_session_board: chess.Board | None = None


def _resolve_board(fen: str) -> chess.Board:
    """Board at ``fen`` carrying its move history when possible (so MCTS can reuse
    its search tree). Infers the opponent's last move from the retained session
    board; falls back to a history-less board on mismatch / new game."""
    global _session_board
    key = fen.split(" ")[:4]  # piece placement, turn, castling, ep (ignore clocks)
    prev = _session_board
    if prev is not None:
        if prev.fen().split(" ")[:4] == key:
            return prev
        for mv in list(prev.legal_moves):
            prev.push(mv)
            if prev.fen().split(" ")[:4] == key:
                return prev  # found the move that reached `fen`; history intact
            prev.pop()
    return chess.Board(fen)  # new game / out of sync -> fresh, no reuse this move


# Pydantic models for request/response validation
class MoveRequest(BaseModel):
    fen: str


class ValidateMoveRequest(BaseModel):
    fen: str
    move: str


class HealthResponse(BaseModel):
    status: str
    bot_type: str
    engine: str | None = None
    model: str | None = None
    sims: int | None = None
    approx_elo: int | None = None


class EvalRequest(BaseModel):
    fen: str


class EvalResponse(BaseModel):
    value: float | None


class MoveResponse(BaseModel):
    move: str
    probability: float | None
    fen: str
    game_over: bool
    result: str | None
    value: float | None = None


class NewGameResponse(BaseModel):
    fen: str
    game_over: bool


class ValidateMoveResponse(BaseModel):
    valid: bool
    fen: str | None = None
    game_over: bool | None = None
    result: str | None = None
    error: str | None = None


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint (also reports the active engine for display)."""
    return HealthResponse(status="ok", bot_type=bot_type, **engine_info)


@app.post("/api/move", response_model=MoveResponse)
async def get_bot_move(request: MoveRequest):
    """
    Get bot's move for the current board state.

    Expected JSON body:
    {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    }

    Returns:
    {
        "move": "e2e4",
        "probability": 0.15,
        "fen": "updated_fen_after_move",
        "game_over": false,
        "result": null
    }
    """
    try:
        board = _resolve_board(request.fen)  # carries move history -> MCTS tree reuse
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    try:
        if board.is_game_over():
            raise HTTPException(status_code=400, detail="Game is already over")

        move_uci, probability = bot.predict(board)

        value = None
        if hasattr(bot, 'get_value'):
            value = bot.get_value(board)

        board.push_uci(move_uci)
        # Persist with the bot's move applied so the next request can extend it.
        global _session_board
        _session_board = board

        return MoveResponse(
            move=move_uci,
            probability=probability,
            fen=board.fen(),
            game_over=board.is_game_over(),
            result=board.result() if board.is_game_over() else None,
            value=value,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/new-game", response_model=NewGameResponse)
async def new_game():
    """
    Start a new game.

    Returns the initial board state.
    """
    global _session_board
    _session_board = None  # drop any retained search tree / history from prior game
    board = chess.Board()
    return NewGameResponse(fen=board.fen(), game_over=False)


@app.post("/api/validate-move", response_model=ValidateMoveResponse)
async def validate_move(request: ValidateMoveRequest):
    """
    Validate if a move is legal.

    Expected JSON body:
    {
        "fen": "current_board_fen",
        "move": "e2e4"
    }
    """
    try:
        board = chess.Board(request.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    try:
        move = chess.Move.from_uci(request.move)
        if move in board.legal_moves:
            board.push(move)
            return ValidateMoveResponse(
                valid=True,
                fen=board.fen(),
                game_over=board.is_game_over(),
                result=board.result() if board.is_game_over() else None,
            )
        return ValidateMoveResponse(valid=False, error="Illegal move")
    except ValueError:
        return ValidateMoveResponse(valid=False, error="Invalid move format")


@app.post("/api/evaluate", response_model=EvalResponse)
async def evaluate_position(request: EvalRequest):
    """Get the bot's value estimate for a position."""
    try:
        board = chess.Board(request.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    value = None
    if hasattr(bot, 'get_value'):
        value = bot.get_value(board)
    return EvalResponse(value=value)


if __name__ == "__main__":
    import uvicorn

    print(f"Starting ChessTransformer API server with {bot_type}")
    uvicorn.run(app, host="0.0.0.0", port=5001)
