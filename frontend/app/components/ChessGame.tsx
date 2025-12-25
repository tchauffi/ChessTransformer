'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Chess, Square } from 'chess.js';
import { Chessboard } from 'react-chessboard';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';
const INITIAL_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

interface MoveHistoryEntry {
  moveNumber: number;
  white?: string;
  black?: string;
}

// Piece values for material calculation
const PIECE_VALUES: Record<string, number> = {
  p: 1,
  n: 3,
  b: 3,
  r: 5,
  q: 9,
};

// Unicode pieces for display
const PIECE_SYMBOLS: Record<string, { white: string; black: string }> = {
  p: { white: '♙', black: '♟' },
  n: { white: '♘', black: '♞' },
  b: { white: '♗', black: '♝' },
  r: { white: '♖', black: '♜' },
  q: { white: '♕', black: '♛' },
};

// SVG piece images (chess.com neo theme)
const PIECE_SVGS: Record<string, { white: string; black: string }> = {
  p: { 
    white: 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wp.png',
    black: 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bp.png'
  },
  n: { 
    white: 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wn.png',
    black: 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bn.png'
  },
  b: { 
    white: 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wb.png',
    black: 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bb.png'
  },
  r: { 
    white: 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wr.png',
    black: 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/br.png'
  },
  q: { 
    white: 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wq.png',
    black: 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bq.png'
  },
};

// Starting piece counts
const STARTING_PIECES: Record<string, number> = {
  p: 8,
  n: 2,
  b: 2,
  r: 2,
  q: 1,
};

export default function ChessGame() {
  const [game, setGame] = useState(new Chess());
  const [moveHistory, setMoveHistory] = useState<string[]>([]);
  const [gameStatus, setGameStatus] = useState<string>('White to move');
  const [isThinking, setIsThinking] = useState(false);
  const [botType, setBotType] = useState<string>('Loading...');
  const [error, setError] = useState<string>('');
  const [playerColor, setPlayerColor] = useState<'w' | 'b'>('w');
  const [gameStarted, setGameStarted] = useState(false);
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [lastMove, setLastMove] = useState<{ from: Square; to: Square } | null>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/health`)
      .then(res => res.json())
      .then(data => setBotType(data.bot_type))
      .catch(err => {
        setError('Failed to connect to the backend. Make sure the API server is running.');
        console.error('API health check failed:', err);
      });
  }, []);

  // Calculate captured pieces for each side
  const capturedPieces = useMemo(() => {
    const board = game.board().flat();
    
    // Count current pieces on board
    const whitePieces: Record<string, number> = { p: 0, n: 0, b: 0, r: 0, q: 0 };
    const blackPieces: Record<string, number> = { p: 0, n: 0, b: 0, r: 0, q: 0 };
    
    board.forEach(square => {
      if (square && square.type !== 'k') {
        if (square.color === 'w') {
          whitePieces[square.type]++;
        } else {
          blackPieces[square.type]++;
        }
      }
    });
    
    // Calculate captured (starting - current)
    const capturedByWhite: string[] = []; // Black pieces captured by white
    const capturedByBlack: string[] = []; // White pieces captured by black
    
    let whiteMaterial = 0;
    let blackMaterial = 0;
    
    Object.keys(STARTING_PIECES).forEach(piece => {
      const blackCaptured = STARTING_PIECES[piece] - blackPieces[piece];
      const whiteCaptured = STARTING_PIECES[piece] - whitePieces[piece];
      
      // White captured these black pieces
      for (let i = 0; i < blackCaptured; i++) {
        capturedByWhite.push(piece);
        whiteMaterial += PIECE_VALUES[piece];
      }
      
      // Black captured these white pieces
      for (let i = 0; i < whiteCaptured; i++) {
        capturedByBlack.push(piece);
        blackMaterial += PIECE_VALUES[piece];
      }
    });
    
    // Sort by value (queen, rook, bishop, knight, pawn)
    const pieceOrder = ['q', 'r', 'b', 'n', 'p'];
    capturedByWhite.sort((a, b) => pieceOrder.indexOf(a) - pieceOrder.indexOf(b));
    capturedByBlack.sort((a, b) => pieceOrder.indexOf(a) - pieceOrder.indexOf(b));
    
    return {
      byWhite: capturedByWhite,
      byBlack: capturedByBlack,
      whiteMaterial,
      blackMaterial,
      advantage: whiteMaterial - blackMaterial,
    };
  }, [game]);

  const updateGameStatus = useCallback((chess: Chess) => {
    if (chess.isCheckmate()) {
      const winner = chess.turn() === 'w' ? 'Black' : 'White';
      setGameStatus(`🏆 Checkmate! ${winner} wins!`);
    } else if (chess.isDraw()) {
      setGameStatus('🤝 Game drawn!');
    } else if (chess.isStalemate()) {
      setGameStatus('🤝 Stalemate!');
    } else if (chess.isCheck()) {
      setGameStatus('⚠️ Check!');
    } else {
      const turn = chess.turn() === 'w' ? 'White' : 'Black';
      setGameStatus(`${turn} to move`);
    }
  }, []);

  // Calculate legal moves for selected piece
  const legalMoves = useMemo(() => {
    if (!selectedSquare) return [];
    const moves = game.moves({ square: selectedSquare, verbose: true });
    return moves.map(m => m.to);
  }, [game, selectedSquare]);

  // Generate square styles for highlighting
  const squareStyles = useMemo(() => {
    const styles: Record<string, React.CSSProperties> = {};
    
    // Highlight selected square
    if (selectedSquare) {
      styles[selectedSquare] = {
        backgroundColor: 'rgba(255, 255, 0, 0.5)',
      };
    }
    
    // Highlight legal move squares
    legalMoves.forEach(square => {
      const piece = game.get(square as Square);
      if (piece) {
        // Capture move - red circle
        styles[square] = {
          background: 'radial-gradient(circle, transparent 60%, rgba(220, 38, 38, 0.5) 60%)',
        };
      } else {
        // Regular move - darker green dot
        styles[square] = {
          background: 'radial-gradient(circle, rgba(34, 197, 94, 0.6) 25%, transparent 25%)',
        };
      }
    });

    // Highlight last move
    if (lastMove) {
      styles[lastMove.from] = {
        ...styles[lastMove.from],
        backgroundColor: 'rgba(250, 204, 21, 0.5)',
      };
      styles[lastMove.to] = {
        ...styles[lastMove.to],
        backgroundColor: 'rgba(250, 204, 21, 0.5)',
      };
    }
    
    return styles;
  }, [selectedSquare, legalMoves, lastMove, game]);

  const onDrop = (sourceSquare: Square, targetSquare: Square): boolean => {
    if (game.turn() !== playerColor) return false;
    
    try {
      const move = game.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q',
      });

      if (move === null) return false;

      const newGame = new Chess(game.fen());
      setGame(newGame);
      setMoveHistory(prev => [...prev, move.san]);
      setLastMove({ from: sourceSquare, to: targetSquare });
      setSelectedSquare(null);
      updateGameStatus(newGame);

      if (!newGame.isGameOver()) {
        getBotMove(newGame.fen());
      }

      return true;
    } catch (err) {
      console.error('Invalid move:', err);
      return false;
    }
  };

  const onSquareClick = (square: Square) => {
    if (isThinking || game.isGameOver()) return;
    if (game.turn() !== playerColor) return;

    const piece = game.get(square);
    
    // If clicking on own piece, select it
    if (piece && piece.color === playerColor) {
      setSelectedSquare(square);
      return;
    }
    
    // If a piece is selected and clicking on a legal move square, make the move
    if (selectedSquare && legalMoves.includes(square)) {
      onDrop(selectedSquare, square);
      return;
    }
    
    // Otherwise deselect
    setSelectedSquare(null);
  };

  const getBotMove = async (currentFen: string) => {
    setIsThinking(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/api/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: currentFen }),
      });

      if (!response.ok) throw new Error('Failed to get bot move');

      const data = await response.json();
      const tempGame = new Chess(currentFen);
      const from = data.move.substring(0, 2) as Square;
      const to = data.move.substring(2, 4) as Square;
      const promotionPiece = data.move.length > 4 ? data.move.substring(4) as 'q' | 'r' | 'b' | 'n' : 'q';
      const moveObj = tempGame.move({ from, to, promotion: promotionPiece });
      
      if (!moveObj) throw new Error('Invalid bot move');
      
      const newGame = new Chess(data.fen);
      setGame(newGame);
      setMoveHistory(prev => [...prev, moveObj.san]);
      setLastMove({ from, to });
      updateGameStatus(newGame);
    } catch (err) {
      setError('Failed to get bot move. Please try again.');
      console.error('Error getting bot move:', err);
    } finally {
      setIsThinking(false);
    }
  };

  const startNewGame = async (color: 'w' | 'b') => {
    const newGame = new Chess();
    setGame(newGame);
    setMoveHistory([]);
    setError('');
    setPlayerColor(color);
    setGameStarted(true);
    setSelectedSquare(null);
    setLastMove(null);
    updateGameStatus(newGame);
    
    if (color === 'b') {
      getBotMove(INITIAL_FEN);
    }
  };

  const resetToColorSelection = () => {
    setGameStarted(false);
    const newGame = new Chess();
    setGame(newGame);
    setMoveHistory([]);
    setError('');
    setSelectedSquare(null);
    setLastMove(null);
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

  // Color selection screen
  if (!gameStarted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-emerald-900 via-green-800 to-emerald-900 relative overflow-hidden">
        {/* Decorative chess pattern background */}
        <div className="absolute inset-0 opacity-5">
          <div className="grid grid-cols-8 h-full">
            {Array.from({ length: 64 }).map((_, i) => (
              <div
                key={i}
                className={`aspect-square ${
                  (Math.floor(i / 8) + i) % 2 === 0 ? 'bg-white' : 'bg-transparent'
                }`}
              />
            ))}
          </div>
        </div>

        <div className="relative z-10 min-h-screen flex flex-col items-center justify-center p-4 sm:p-6">
          {/* Hero Section */}
          <div className="text-center mb-8 sm:mb-12">
            <div className="inline-flex items-center justify-center w-20 h-20 sm:w-24 sm:h-24 bg-white/10 backdrop-blur rounded-2xl mb-6 shadow-2xl border border-white/20">
              <span className="text-5xl sm:text-6xl">♟️</span>
            </div>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold text-white mb-4 tracking-tight">
              ChessTransformer
            </h1>
            <p className="text-emerald-200 text-lg sm:text-xl max-w-md mx-auto leading-relaxed">
              Challenge an AI powered by transformer neural networks
            </p>
            
            {/* Bot Status Badge */}
            <div className="mt-6 inline-flex items-center gap-3 bg-black/20 backdrop-blur-sm px-5 py-2.5 rounded-full border border-white/10">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
              </span>
              <span className="text-emerald-100 font-medium">{botType}</span>
            </div>
          </div>

          {error && (
            <div className="mb-8 p-4 bg-red-500/20 border border-red-400/50 text-red-100 rounded-xl backdrop-blur-sm max-w-md w-full text-center">
              {error}
            </div>
          )}

          {/* Color Selection Cards */}
          <div className="w-full max-w-2xl">
            <h2 className="text-xl sm:text-2xl font-semibold text-white mb-6 text-center">
              Choose Your Side
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6 px-4">
              {/* Play as White */}
              <button
                onClick={() => startNewGame('w')}
                className="group relative bg-gradient-to-br from-white via-gray-50 to-gray-100 rounded-2xl p-6 sm:p-8 shadow-2xl hover:shadow-white/20 transition-all duration-300 hover:scale-[1.02] hover:-translate-y-1 border-4 border-transparent hover:border-yellow-400/50"
              >
                <div className="text-center">
                  <div className="text-6xl sm:text-7xl mb-4 filter drop-shadow-lg transform group-hover:scale-110 transition-transform">
                    ♔
                  </div>
                  <h3 className="text-xl sm:text-2xl font-bold text-gray-800 mb-2">Play White</h3>
                  <p className="text-gray-500 text-sm sm:text-base">You make the first move</p>
                </div>
                <div className="absolute top-3 right-3 bg-yellow-400 text-yellow-900 text-xs font-bold px-2 py-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity">
                  1st
                </div>
              </button>

              {/* Play as Black */}
              <button
                onClick={() => startNewGame('b')}
                className="group relative bg-gradient-to-br from-gray-800 via-gray-900 to-black rounded-2xl p-6 sm:p-8 shadow-2xl hover:shadow-emerald-500/20 transition-all duration-300 hover:scale-[1.02] hover:-translate-y-1 border-4 border-transparent hover:border-emerald-400/50"
              >
                <div className="text-center">
                  <div className="text-6xl sm:text-7xl mb-4 filter drop-shadow-lg transform group-hover:scale-110 transition-transform">
                    ♚
                  </div>
                  <h3 className="text-xl sm:text-2xl font-bold text-white mb-2">Play Black</h3>
                  <p className="text-gray-400 text-sm sm:text-base">Bot makes the first move</p>
                </div>
                <div className="absolute top-3 right-3 bg-emerald-400 text-emerald-900 text-xs font-bold px-2 py-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity">
                  2nd
                </div>
              </button>
            </div>
          </div>

          {/* Features */}
          <div className="mt-12 flex flex-wrap justify-center gap-6 text-emerald-200/70 text-sm">
            <div className="flex items-center gap-2">
              <span>🧠</span>
              <span>Transformer AI</span>
            </div>
            <div className="flex items-center gap-2">
              <span>⚡</span>
              <span>Real-time moves</span>
            </div>
            <div className="flex items-center gap-2">
              <span>🎯</span>
              <span>Legal move hints</span>
            </div>
          </div>

          {/* Footer with Links */}
          <div className="mt-10 text-center space-y-4">
            <p className="text-emerald-300/50 text-xs sm:text-sm">
              Trained on millions of chess games from{' '}
              <a 
                href="https://database.lichess.org/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-emerald-400 hover:text-emerald-300 underline underline-offset-2 transition-colors"
              >
                Lichess Database
              </a>
            </p>
            <div className="flex items-center justify-center gap-4">
              <a
                href="https://github.com/tchauffi/ChessTransformer"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-emerald-300/60 hover:text-white transition-colors text-sm"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                </svg>
                <span>View on GitHub</span>
              </a>
              <a
                href="https://lichess.org/"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-emerald-300/60 hover:text-white transition-colors text-sm"
              >
                <span className="text-lg">♞</span>
                <span>Lichess</span>
              </a>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Game screen
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto p-3 sm:p-4 md:p-6">
        {/* Header */}
        <header className="text-center mb-4 sm:mb-6">
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-white mb-1 sm:mb-2">
            ♟️ ChessTransformer
          </h1>
          <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-4 text-xs sm:text-sm">
            <span className="text-slate-400">Bot: <span className="text-emerald-400 font-medium">{botType}</span></span>
            <span className="text-slate-600 hidden sm:inline">|</span>
            <span className="text-slate-400">
              Playing as <span className="font-medium text-white">{playerColor === 'w' ? '⬜ White' : '⬛ Black'}</span>
            </span>
          </div>
        </header>

        {error && (
          <div className="mb-4 p-3 sm:p-4 bg-red-500/20 border border-red-500/50 text-red-200 rounded-xl max-w-xl mx-auto text-sm sm:text-base">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-[1fr,300px] xl:grid-cols-[1fr,340px] gap-4 sm:gap-6">
          {/* Board Section */}
          <div className="bg-slate-800/50 backdrop-blur rounded-xl sm:rounded-2xl p-3 sm:p-4 md:p-6 border border-slate-700/50 shadow-2xl">
            {/* Status Bar */}
            <div className="flex items-center justify-between mb-3 sm:mb-4 px-1 sm:px-2">
              <div className={`text-base sm:text-lg font-semibold ${
                game.isGameOver() 
                  ? 'text-yellow-400' 
                  : game.isCheck() 
                    ? 'text-red-400' 
                    : 'text-white'
              }`}>
                {gameStatus}
              </div>
              {isThinking && (
                <div className="flex items-center gap-2 text-emerald-400">
                  <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
                  <span className="text-xs sm:text-sm">Thinking...</span>
                </div>
              )}
            </div>

            {/* Chess Board - Responsive */}
            <div className="w-full max-w-[min(100%,600px)] mx-auto">
              {/* Captured pieces - Opponent (top) */}
              <div className="flex items-center justify-between mb-2 px-1 min-h-[36px] bg-slate-800/60 rounded-lg py-1.5">
                <div className="flex items-center gap-0.5 flex-wrap px-2">
                  {(playerColor === 'w' ? capturedPieces.byBlack : capturedPieces.byWhite).map((piece, idx) => (
                    <img 
                      key={`top-${piece}-${idx}`} 
                      src={PIECE_SVGS[piece][playerColor === 'w' ? 'white' : 'black']}
                      alt={piece}
                      className="w-6 h-6 sm:w-7 sm:h-7 drop-shadow-md"
                    />
                  ))}
                  {(playerColor === 'w' ? capturedPieces.byBlack : capturedPieces.byWhite).length === 0 && (
                    <span className="text-slate-600 text-xs">No captures</span>
                  )}
                </div>
                {capturedPieces.advantage !== 0 && (
                  <span className={`text-xs sm:text-sm font-bold px-2 py-0.5 rounded mr-2 ${
                    (playerColor === 'w' && capturedPieces.advantage < 0) || 
                    (playerColor === 'b' && capturedPieces.advantage > 0)
                      ? 'bg-emerald-500/30 text-emerald-300'
                      : 'text-slate-600'
                  }`}>
                    {playerColor === 'w' 
                      ? (capturedPieces.advantage < 0 ? `+${Math.abs(capturedPieces.advantage)}` : '')
                      : (capturedPieces.advantage > 0 ? `+${capturedPieces.advantage}` : '')
                    }
                  </span>
                )}
              </div>
              
              <div className="relative w-full" style={{ paddingBottom: '100%' }}>
                <div className="absolute inset-0">
                  <Chessboard
                    options={{
                      id: "main-board",
                      position: game.fen(),
                      boardOrientation: playerColor === 'w' ? 'white' : 'black',
                      allowDragging: !isThinking && !game.isGameOver() && game.turn() === playerColor,
                      squareStyles: squareStyles,
                      showNotation: true,
                      animationDurationInMs: 200,
                      darkSquareStyle: { backgroundColor: '#15803d' },
                      lightSquareStyle: { backgroundColor: '#f0fdf4' },
                      canDragPiece: ({ piece }) => {
                        if (isThinking || game.isGameOver()) return false;
                        if (game.turn() !== playerColor) return false;
                        return piece.pieceType[0] === playerColor;
                      },
                      onPieceDrag: ({ square }) => {
                        setSelectedSquare(square as Square);
                      },
                      onPieceDrop: ({ sourceSquare, targetSquare }) => {
                        return onDrop(sourceSquare as Square, targetSquare as Square);
                      },
                      onSquareClick: ({ square }) => {
                        onSquareClick(square as Square);
                      },
                    }}
                  />
                </div>
              </div>
              
              {/* Captured pieces - Player (bottom) */}
              <div className="flex items-center justify-between mt-2 px-1 min-h-[36px] bg-slate-800/60 rounded-lg py-1.5">
                <div className="flex items-center gap-0.5 flex-wrap px-2">
                  {(playerColor === 'w' ? capturedPieces.byWhite : capturedPieces.byBlack).map((piece, idx) => (
                    <img 
                      key={`bottom-${piece}-${idx}`} 
                      src={PIECE_SVGS[piece][playerColor === 'w' ? 'black' : 'white']}
                      alt={piece}
                      className="w-6 h-6 sm:w-7 sm:h-7 drop-shadow-md"
                    />
                  ))}
                  {(playerColor === 'w' ? capturedPieces.byWhite : capturedPieces.byBlack).length === 0 && (
                    <span className="text-slate-600 text-xs">No captures</span>
                  )}
                </div>
                {capturedPieces.advantage !== 0 && (
                  <span className={`text-xs sm:text-sm font-bold px-2 py-0.5 rounded mr-2 ${
                    (playerColor === 'w' && capturedPieces.advantage > 0) || 
                    (playerColor === 'b' && capturedPieces.advantage < 0)
                      ? 'bg-emerald-500/30 text-emerald-300'
                      : 'text-slate-600'
                  }`}>
                    {playerColor === 'w' 
                      ? (capturedPieces.advantage > 0 ? `+${capturedPieces.advantage}` : '')
                      : (capturedPieces.advantage < 0 ? `+${Math.abs(capturedPieces.advantage)}` : '')
                    }
                  </span>
                )}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="mt-4 sm:mt-6 flex gap-2 sm:gap-3">
              <button
                onClick={() => startNewGame(playerColor)}
                className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold py-2.5 sm:py-3 px-4 sm:px-6 rounded-lg sm:rounded-xl transition-all text-sm sm:text-base shadow-lg hover:shadow-emerald-500/25"
              >
                🔄 New Game
              </button>
              <button
                onClick={resetToColorSelection}
                className="flex-1 bg-slate-700 hover:bg-slate-600 text-white font-semibold py-2.5 sm:py-3 px-4 sm:px-6 rounded-lg sm:rounded-xl transition-all text-sm sm:text-base border border-slate-600"
              >
                🎨 Change Side
              </button>
            </div>
          </div>

          {/* Side Panel */}
          <div className="space-y-4">
            {/* Move History */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl sm:rounded-2xl p-3 sm:p-4 border border-slate-700/50">
              <h2 className="text-base sm:text-lg font-semibold text-white mb-3 flex items-center gap-2">
                📜 Moves
              </h2>
              <div className="max-h-[200px] sm:max-h-[280px] overflow-y-auto">
                {formatMoveHistory().length === 0 ? (
                  <p className="text-slate-500 text-center py-4 text-sm">No moves yet</p>
                ) : (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-slate-400 border-b border-slate-700">
                        <th className="text-left py-2 px-2 w-8">#</th>
                        <th className="text-left py-2 px-2">White</th>
                        <th className="text-left py-2 px-2">Black</th>
                      </tr>
                    </thead>
                    <tbody>
                      {formatMoveHistory().map((entry, idx) => (
                        <tr key={idx} className="text-slate-200 border-b border-slate-700/50 hover:bg-slate-700/30">
                          <td className="py-1.5 sm:py-2 px-2 text-slate-500">{entry.moveNumber}</td>
                          <td className="py-1.5 sm:py-2 px-2 font-mono text-xs sm:text-sm">{entry.white || ''}</td>
                          <td className="py-1.5 sm:py-2 px-2 font-mono text-xs sm:text-sm">{entry.black || ''}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>

            {/* Game Info */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl sm:rounded-2xl p-3 sm:p-4 border border-slate-700/50">
              <h2 className="text-base sm:text-lg font-semibold text-white mb-3">ℹ️ Info</h2>
              <div className="space-y-2.5 text-sm">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Your Side</span>
                  <span className="text-white font-medium">
                    {playerColor === 'w' ? '⬜ White' : '⬛ Black'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Turn</span>
                  <span className={`font-medium ${game.turn() === playerColor ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {game.turn() === playerColor ? '👤 You' : '🤖 Bot'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Moves</span>
                  <span className="text-white font-medium">{moveHistory.length}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Status</span>
                  <span className={`font-medium text-xs px-2 py-0.5 rounded-full ${
                    game.isGameOver() 
                      ? 'bg-amber-500/20 text-amber-300' 
                      : 'bg-emerald-500/20 text-emerald-300'
                  }`}>
                    {game.isGameOver() ? 'Game Over' : 'Playing'}
                  </span>
                </div>
              </div>
            </div>

            {/* Help Tip */}
            <div className="bg-emerald-900/30 rounded-xl p-3 border border-emerald-700/30">
              <p className="text-emerald-300/80 text-xs text-center">
                💡 Drag pieces or click to see legal moves
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
