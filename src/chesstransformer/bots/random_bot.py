import chess
import random

class RandomBot:
    def __init__(self):
        pass

    def predict(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available.")
        # pick a random move
        return random.choice(legal_moves)
    

if __name__ == "__main__":
    bot = RandomBot()
    board = chess.Board()
    print("Initial board:")
    print(board)
    move = bot.predict(board)
    print(f"RandomBot suggests move: {move}")
    board.push(move)
    print("Board after move:")
    print(board)

    move = bot.predict(board)
    print(f"RandomBot suggests move: {move}")
    board.push(move)
    print("Board after move:")
    print(board)