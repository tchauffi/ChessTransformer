# ChessTransformer Backend API

FastAPI server that provides REST API endpoints for the ChessTransformer bot integration.

## Overview

This backend server exposes the ChessTransformer chess bot through a REST API, allowing the Next.js frontend (or any other client) to interact with the bot and play chess games.

## Features

- FastAPI-based REST API
- CORS enabled for cross-origin requests
- Automatic fallback to RandomBot if model is unavailable
- Type-safe request/response validation with Pydantic
- Health check endpoint
- Move validation

## Prerequisites

- Python 3.12+
- UV package manager

## Installation

From the project root:

```bash
uv sync
```

This installs all dependencies including FastAPI and uvicorn.

## Running the Server

From the backend directory:

```bash
cd backend
uv run python api.py
```

The server will start on `http://localhost:5000`.

## API Endpoints

### Health Check

Check if the API is running and get the bot type.

```
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "bot_type": "Position2MoveBot"
}
```

### Get Bot Move

Get the bot's next move for a given board position.

```
POST /api/move
```

**Request Body:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
}
```

**Response:**
```json
{
  "move": "b8c6",
  "probability": 0.014155865646898746,
  "fen": "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
  "game_over": false,
  "result": null
}
```

### New Game

Start a new game and get the initial board position.

```
POST /api/new-game
```

**Response:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "game_over": false
}
```

### Validate Move

Validate if a move is legal for a given board position.

```
POST /api/validate-move
```

**Request Body:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "move": "e2e4"
}
```

**Response (valid move):**
```json
{
  "valid": true,
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "game_over": false,
  "result": null
}
```

**Response (invalid move):**
```json
{
  "valid": false,
  "error": "Illegal move"
}
```

## Bot Types

The API supports multiple bot implementations:

- **Position2MoveBot**: The main transformer-based chess bot (requires model files)
- **RandomBot**: Fallback bot that makes random legal moves (used if model is unavailable)

The active bot type is returned in the `/api/health` endpoint.

## Development

### Testing with curl

Health check:
```bash
curl http://localhost:5000/api/health
```

Get bot move:
```bash
curl -X POST http://localhost:5000/api/move \
  -H "Content-Type: application/json" \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"}'
```

### Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

## Configuration

### Port

By default, the server runs on port 5000. To change this, modify the `uvicorn.run()` call in `api.py`:

```python
uvicorn.run(app, host='0.0.0.0', port=YOUR_PORT)
```

### CORS

CORS is configured to allow all origins for development. For production, update the `allow_origins` list in `api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid input or game is over
- `500 Internal Server Error`: Server error

Error responses include a detail message:

```json
{
  "detail": "Error message here"
}
```

## Production Deployment

For production deployment:

1. Update CORS configuration to allow only your frontend domain
2. Use a production ASGI server like uvicorn with multiple workers:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 5000 --workers 4
   ```
3. Set up a reverse proxy (nginx, Apache) in front of the API
4. Consider using a process manager like systemd or supervisor

## License

MIT
