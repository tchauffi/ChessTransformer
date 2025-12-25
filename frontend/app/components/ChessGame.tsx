'use client';

import { useState, useEffect } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

interface MoveHistoryEntry {
  moveNumber: number;
  white?: string;
  black?: string;
}

export default function ChessGame() {
  const [fen, setFen] = useState('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const [moveHistory, setMoveHistory] = useState<string[]>([]);
  const [gameStatus, setGameStatus] = useState<string>('');
  const [isThinking, setIsThinking] = useState(false);
  const [botType, setBotType] = useState<string>('Loading...');
  const [error, setError] = useState<string>('');
  const [playerColor, setPlayerColor] = useState<'w' | 'b'>('w'); // 'w' for white, 'b' for black
  const [gameStarted, setGameStarted] = useState(false);

  useEffect(() => {
    // Check API health on mount
    fetch(`${API_BASE_URL}/api/health`)
      .then(res => res.json())
      .then(data => {
        setBotType(data.bot_type);
      })
      .catch(err => {
        setError('Failed to connect to the backend. Make sure the API server is running.');
        console.error('API health check failed:', err);
      });
  }, []);

  const updateGameStatus = (chess: Chess) => {
    if (chess.isCheckmate()) {
      const winner = chess.turn() === 'w' ? 'Black' : 'White';
      setGameStatus(`Checkmate! ${winner} wins!`);
    } else if (chess.isDraw()) {
      setGameStatus('Game drawn!');
    } else if (chess.isStalemate()) {
      setGameStatus('Stalemate!');
    } else if (chess.isCheck()) {
      setGameStatus('Check!');
    } else {
      const turn = chess.turn() === 'w' ? 'White' : 'Black';
      setGameStatus(`${turn} to move`);
    }
  };

  const makeMove = (sourceSquare: string, targetSquare: string) => {
    const game = new Chess(fen);
    
    // Check if it's the player's turn
    if (game.turn() !== playerColor) {
      return false;
    }
    
    try {
      const move = game.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q', // Always promote to queen for simplicity
      });

      if (move === null) {
        return false;
      }

      setFen(game.fen());
      setMoveHistory(prev => [...prev, move.san]);
      updateGameStatus(game);

      // If game is not over, get bot's move
      if (!game.isGameOver()) {
        getBotMove(game.fen());
      }

      return true;
    } catch (error) {
      console.error('Invalid move:', error);
      return false;
    }
  };

  const getBotMove = async (currentFen: string) => {
    setIsThinking(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/api/move`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fen: currentFen,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get bot move');
      }

      const data = await response.json();
      
      // Convert UCI move to SAN notation before applying
      // Create a temporary Chess instance to get the SAN notation
      const tempGame = new Chess(currentFen);
      const from = data.move.substring(0, 2);
      const to = data.move.substring(2, 4);
      const promotionPiece = data.move.length > 4 ? data.move.substring(4) as 'q' | 'r' | 'b' | 'n' : 'q';
      const moveObj = tempGame.move({ from, to, promotion: promotionPiece });
      
      if (!moveObj) {
        throw new Error('Invalid bot move');
      }
      
      // Update state with the FEN from the API response
      setFen(data.fen);
      setMoveHistory(prev => [...prev, moveObj.san]);
      
      // Update game status with the new position
      const newGame = new Chess(data.fen);
      updateGameStatus(newGame);
    } catch (err) {
      setError('Failed to get bot move. Please try again.');
      console.error('Error getting bot move:', err);
    } finally {
      setIsThinking(false);
    }
  };

  const startNewGame = async (color: 'w' | 'b') => {
    const initialFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    setFen(initialFen);
    setMoveHistory([]);
    setError('');
    setPlayerColor(color);
    setGameStarted(true);
    
    const newGame = new Chess(initialFen);
    updateGameStatus(newGame);
    
    // If bot plays white (player chose black), get bot's first move
    if (color === 'b') {
      getBotMove(initialFen);
    }
  };

  const resetToColorSelection = () => {
    setGameStarted(false);
    const initialFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    setFen(initialFen);
    setMoveHistory([]);
    setError('');
    const newGame = new Chess(initialFen);
    updateGameStatus(newGame);
  };

  const formatMoveHistory = (): MoveHistoryEntry[] => {
    const formatted: MoveHistoryEntry[] = [];
    for (let i = 0; i < moveHistory.length; i += 2) {
      formatted.push({
        moveNumber: Math.floor(i / 2) + 1,
        white: moveHistory[i],
        black: moveHistory[i + 1],
      });
    }
    return formatted;
  };

  useEffect(() => {
    const chess = new Chess(fen);
    updateGameStatus(chess);
  }, [fen]);

  // Show color selection screen if game hasn't started
  if (!gameStarted) {
    return (
      <div className="w-full max-w-4xl mx-auto p-4">
        <div className="mb-6 text-center">
          <h1 className="text-4xl font-bold mb-2">ChessTransformer</h1>
          <p className="text-gray-600">Human vs Bot Testing Interface</p>
          <p className="text-sm text-gray-500 mt-2">Bot: {botType}</p>
        </div>

        {error && (
          <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}

        <div className="bg-white rounded-lg shadow-lg p-8">
          <h2 className="text-2xl font-semibold mb-6 text-center">Choose Your Color</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <button
              onClick={() => startNewGame('w')}
              className="flex flex-col items-center justify-center p-8 bg-white border-4 border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all group"
            >
              <div className="text-6xl mb-4">♔</div>
              <h3 className="text-xl font-semibold mb-2 group-hover:text-blue-600">Play as White</h3>
              <p className="text-gray-600 text-sm">You move first</p>
            </button>
            <button
              onClick={() => startNewGame('b')}
              className="flex flex-col items-center justify-center p-8 bg-white border-4 border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all group"
            >
              <div className="text-6xl mb-4">♚</div>
              <h3 className="text-xl font-semibold mb-2 group-hover:text-blue-600">Play as Black</h3>
              <p className="text-gray-600 text-sm">Bot moves first</p>
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-7xl mx-auto p-4">
      <div className="mb-6 text-center">
        <h1 className="text-4xl font-bold mb-2">ChessTransformer</h1>
        <p className="text-gray-600">Human vs Bot Testing Interface</p>
        <p className="text-sm text-gray-500 mt-2">Bot: {botType}</p>
        <p className="text-sm text-gray-600 mt-1">
          You are playing as {playerColor === 'w' ? 'White' : 'Black'}
        </p>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chessboard */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-lg p-4">
            <div className="mb-4 flex justify-between items-center">
              <div className="text-lg font-semibold">
                {gameStatus}
              </div>
              {isThinking && (
                <div className="text-blue-600 animate-pulse">
                  Bot is thinking...
                </div>
              )}
            </div>
            <div className="w-full max-w-[600px] mx-auto">
              <Chessboard
                position={fen}
                boardOrientation={playerColor === 'w' ? 'white' : 'black'}
                onPieceDrop={(sourceSquare, targetSquare) => {
                  const game = new Chess(fen);
                  if (game.isGameOver() || isThinking) {
                    return false;
                  }
                  return makeMove(sourceSquare, targetSquare);
                }}
                boardWidth={Math.min(600, typeof window !== 'undefined' ? window.innerWidth - 40 : 600)}
              />
            </div>
            <div className="mt-4 flex gap-3">
              <button
                onClick={() => startNewGame(playerColor)}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                New Game (Same Color)
              </button>
              <button
                onClick={resetToColorSelection}
                className="flex-1 bg-gray-600 hover:bg-gray-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                Change Color
              </button>
            </div>
          </div>
        </div>

        {/* Move History */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-lg p-4">
            <h2 className="text-xl font-semibold mb-4">Move History</h2>
            <div className="max-h-[500px] overflow-y-auto">
              {formatMoveHistory().length === 0 ? (
                <p className="text-gray-500 text-center py-4">No moves yet</p>
              ) : (
                <table className="w-full text-sm">
                  <thead className="border-b">
                    <tr>
                      <th className="text-left py-2 px-2">#</th>
                      <th className="text-left py-2 px-2">White</th>
                      <th className="text-left py-2 px-2">Black</th>
                    </tr>
                  </thead>
                  <tbody>
                    {formatMoveHistory().map((entry, idx) => (
                      <tr key={idx} className="border-b hover:bg-gray-50">
                        <td className="py-2 px-2 text-gray-600">{entry.moveNumber}</td>
                        <td className="py-2 px-2 font-mono">{entry.white || ''}</td>
                        <td className="py-2 px-2 font-mono">{entry.black || ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          {/* Game Info */}
          <div className="bg-white rounded-lg shadow-lg p-4 mt-4">
            <h2 className="text-xl font-semibold mb-4">Game Info</h2>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Your Color:</span>
                <span className="font-semibold">{playerColor === 'w' ? 'White ♔' : 'Black ♚'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Turn:</span>
                <span className="font-semibold">
                  {(() => {
                    const game = new Chess(fen);
                    return game.turn() === playerColor 
                      ? `${game.turn() === 'w' ? 'White' : 'Black'} (You)` 
                      : `${game.turn() === 'w' ? 'White' : 'Black'} (Bot)`;
                  })()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Moves:</span>
                <span className="font-semibold">{moveHistory.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Status:</span>
                <span className="font-semibold">
                  {(() => {
                    const game = new Chess(fen);
                    return game.isGameOver() ? 'Game Over' : 'In Progress';
                  })()}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
