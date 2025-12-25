# ChessTransformer Frontend

A Next.js web application for testing the ChessTransformer bot interactively. Play chess as a human against the AI-powered chess bot.

## Features

- 🎮 Interactive chessboard UI
- 🤖 Real-time bot integration
- 📊 Move history display
- ✅ Game status tracking (check, checkmate, draw)
- 🎯 Intuitive and beginner-friendly interface

## Prerequisites

- Node.js 18+ and npm
- Python 3.12+
- UV package manager (for backend dependencies)

## Setup

### 1. Install Backend Dependencies

From the project root directory:

```bash
cd /path/to/ChessTransformer
uv sync
```

This will install all Python dependencies including FastAPI and uvicorn.

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

## Running the Application

You need to run both the backend API server and the frontend development server.

### Start the Backend API Server

From the project root directory:

```bash
cd backend
uv run python api.py
```

The API server will start on `http://localhost:5000`.

**Note:** The backend will automatically fall back to using RandomBot if the Position2MoveBot model is not available.

### Start the Frontend Development Server

In a separate terminal, from the project root:

```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:3000`.

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. You'll see a chessboard with you playing as White
3. Make your move by dragging and dropping pieces
4. The bot (playing as Black) will automatically respond
5. The move history is displayed on the right side
6. Click "New Game" to start over at any time

## Configuration

The frontend API URL can be configured via environment variable:

Create a `.env.local` file in the `frontend` directory:

```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## Building for Production

### Build the Frontend

```bash
cd frontend
npm run build
npm start
```

This will create an optimized production build.

### Deploy

The frontend can be deployed to platforms like Vercel, Netlify, or any Node.js hosting service.

The backend API should be deployed separately and the `NEXT_PUBLIC_API_URL` environment variable should be updated to point to the production API URL.

## Technology Stack

- **Frontend:**
  - Next.js 15 (React framework)
  - TypeScript
  - Tailwind CSS
  - chess.js (Chess logic)
  - react-chessboard (Chessboard UI)

- **Backend:**
  - FastAPI (Python web framework)
  - python-chess (Chess engine)
  - ChessTransformer models

## Troubleshooting

### Backend Connection Error

If you see "Failed to connect to the backend" error:
- Ensure the backend API server is running on port 5000
- Check that CORS is enabled in the API
- Verify the API URL in your environment variables

### Bot Not Loading

If the bot type shows "RandomBot" instead of "Position2MoveBot":
- This is expected if the model files are not available
- The RandomBot will work fine for testing the interface
- To use the full Position2MoveBot, ensure model files are in the correct location

## API Endpoints

The backend provides the following endpoints:

- `GET /api/health` - Health check and bot type
- `POST /api/move` - Get bot's next move
- `POST /api/new-game` - Start a new game
- `POST /api/validate-move` - Validate a move

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs)
- [Learn Next.js](https://nextjs.org/learn)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT

