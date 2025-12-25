"""
FastAPI server for ChessTransformer bot integration.
Provides endpoints for the Next.js frontend to interact with the chess bot.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from chesstransformer.bots import Position2MoveBot, RandomBot

app = FastAPI(title="ChessTransformer API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize bot (use RandomBot if model is not available)
try:
    bot = Position2MoveBot()
    bot_type = "Position2MoveBot"
except Exception as e:
    print(f"Warning: Could not load Position2MoveBot: {e}")
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

class MoveResponse(BaseModel):
    move: str
    probability: float | None
    fen: str
    game_over: bool
    result: str | None

class NewGameResponse(BaseModel):
    fen: str
    game_over: bool

class ValidateMoveResponse(BaseModel):
    valid: bool
    fen: str | None = None
    game_over: bool | None = None
    result: str | None = None
    error: str | None = None

@app.get('/api/health', response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status='ok',
        bot_type=bot_type
    )

@app.post('/api/move', response_model=MoveResponse)
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
        # Create board from FEN
        board = chess.Board(request.fen)
        
        # Check if game is over
        if board.is_game_over():
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'Game is over',
                    'game_over': True,
                    'result': board.result()
                }
            )
        
        # Get bot's move
        move_uci, probability = bot.predict(board)
        
        # Apply move
        board.push_uci(move_uci)
        
        # Return move and new board state
        return MoveResponse(
            move=move_uci,
            probability=probability,
            fen=board.fen(),
            game_over=board.is_game_over(),
            result=board.result() if board.is_game_over() else None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/new-game', response_model=NewGameResponse)
async def new_game():
    """
    Start a new game.
    
    Returns the initial board state.
    """
    board = chess.Board()
    return NewGameResponse(
        fen=board.fen(),
        game_over=False
    )

@app.post('/api/validate-move', response_model=ValidateMoveResponse)
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
        
        # Check if move is legal
        try:
            move = chess.Move.from_uci(request.move)
            if move in board.legal_moves:
                board.push(move)
                return ValidateMoveResponse(
                    valid=True,
                    fen=board.fen(),
                    game_over=board.is_game_over(),
                    result=board.result() if board.is_game_over() else None
                )
            else:
                return ValidateMoveResponse(
                    valid=False,
                    error='Illegal move'
                )
        except:
            return ValidateMoveResponse(
                valid=False,
                error='Invalid move format'
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    print(f"Starting ChessTransformer API server with {bot_type}")
    uvicorn.run(app, host='0.0.0.0', port=5001)
