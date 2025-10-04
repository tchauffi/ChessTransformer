import chess
from PIL import Image
from io import BytesIO
import cairosvg


def _create_board_from_moves(moves: list[str]) -> chess.Board:
    """
    Create a chess board and apply a sequence of moves.

    Args:
        moves: List of moves in UCI format (e.g., ['e2e4', 'd7d5'])

    Returns:
        Chess board with moves applied
    """
    board = chess.Board()
    for move_str in moves:
        try:
            move = chess.Move.from_uci(move_str)
            board.push(move)
        except ValueError:
            pass  # Skip invalid moves
    return board


def generate_board_image(moves: list[str], size: int = 400) -> Image.Image:
    """
    Generate a PIL image of a chess board after applying a sequence of moves.

    Args:
        moves: List of moves in UCI format (e.g., ['e2e4', 'd7d5'])
        size: Size of the output image in pixels (default: 400)

    Returns:
        PIL Image of the chess board
    """
    import chess.svg

    board = _create_board_from_moves(moves)

    # Generate SVG using chess library
    svg_data = chess.svg.board(board, size=size)

    # Convert SVG to PNG using cairosvg
    png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))

    # Load PNG into PIL Image
    img = Image.open(BytesIO(png_data))

    return img


def generate_board_ascii(moves: list[str]) -> str:
    """
    Generate an ASCII representation of a chess board after applying a sequence of moves.

    Args:
        moves: List of moves in UCI format (e.g., ['e2e4', 'd7d5'])

    Returns:
        String representation of the chess board
    """
    board = _create_board_from_moves(moves)
    return str(board)


def generate_game_gif(moves: list[str], size: int = 400, duration: int = 500) -> bytes:
    """
    Generate an animated GIF of a chess game from a sequence of moves.

    Args:
        moves: List of moves in UCI format (e.g., ['e2e4', 'd7d5'])
        size: Size of each frame in pixels (default: 400)
        duration: Duration of each frame in milliseconds (default: 500)

    Returns:
        Bytes of the animated GIF
    """
    frames = []

    # Initial position
    img = generate_board_image([], size)
    frames.append(img)

    # Apply moves and capture frames
    moves_so_far = []
    for move_str in moves:
        try:
            move = chess.Move.from_uci(move_str)
            moves_so_far.append(move_str)
            img = generate_board_image(moves_so_far, size)
            frames.append(img)
        except ValueError:
            pass  # Skip invalid moves

    # Save frames as GIF in memory
    gif_bytes = BytesIO()
    frames[0].save(
        gif_bytes,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    gif_bytes.seek(0)

    return gif_bytes.getvalue()


if __name__ == "__main__":
    moves = ["e2e4"]
    img = generate_board_image(moves)
    img.show()

    ascii_board = generate_board_ascii(moves)
    print(ascii_board)

    coup_du_berget = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
    gif = generate_game_gif(coup_du_berget)
    with open("coup_du_berget.gif", "wb") as f:
        f.write(gif)
