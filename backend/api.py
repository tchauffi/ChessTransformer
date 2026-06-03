"""
FastAPI server for ChessTransformer bot integration.
Provides endpoints for the Next.js frontend to interact with the chess bot.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import os

from chesstransformer.bots import Pos2MoveV2Bot, RandomBot

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

# Set MODEL_PATH env var to point to a specific checkpoint directory
model_path = os.environ.get("MODEL_PATH")
try:
    kwargs = {}
    if model_path:
        kwargs["model_dir"] = model_path
    bot = Pos2MoveV2Bot(**kwargs)
    bot_type = "Pos2MoveV2Bot"
except Exception as e:
    print(f"Warning: Could not load Pos2MoveV2Bot: {e}")
    print("Falling back to RandomBot")
    bot = RandomBot()
    bot_type = "RandomBot"


# Pydantic models for request/response validation
class MoveRequest(BaseModel):
    fen: str


class ValidateMoveRequest(BaseModel):
    fen: str
    move: str


class HealthResponse(BaseModel):
    status: str
    bot_type: str


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
    """Health check endpoint."""
    return HealthResponse(status="ok", bot_type=bot_type)


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
        board = chess.Board(request.fen)
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
