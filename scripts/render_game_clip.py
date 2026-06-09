"""Render a bot-vs-Stockfish game to an MP4 clip for social media.

Plays several games headless, keeps the most compelling decisive bot win
(checkmate preferred, shorter preferred), then renders that game to MP4.
Rendering replays the saved move list, so it is fully deterministic.

Usage
-----
    uv run python scripts/render_game_clip.py \
        --model data/models/pos2move_v2.1 \
        --skills 8 10 --sims 800 \
        --search-games 8 --out clip.mp4
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.svg
import cairosvg
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chesstransformer.bots.pos2move_v2_mcts_bot import Pos2MoveV2MctsBot

SF_PATH = "/usr/games/stockfish"


# ── headless game search ───────────────────────────────────────────────────
@dataclass
class GameRecord:
    moves: list[str]               # UCI moves
    evals: list[float | None]      # bot eval after its own moves (None for SF)
    skill: int
    bot_color: chess.Color
    winner_is_bot: bool
    is_checkmate: bool
    plies: int

    def score(self) -> tuple:
        # rank: bot win first, checkmate first, then shorter, then higher skill
        return (self.winner_is_bot, self.is_checkmate, -self.plies, self.skill)

    def to_dict(self) -> dict:
        return {
            "moves": self.moves, "evals": self.evals, "skill": self.skill,
            "bot_color": bool(self.bot_color), "winner_is_bot": self.winner_is_bot,
            "is_checkmate": self.is_checkmate, "plies": self.plies,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GameRecord":
        return cls(d["moves"], d["evals"], d["skill"], d["bot_color"],
                   d["winner_is_bot"], d["is_checkmate"], d["plies"])


def play_headless(bot: Pos2MoveV2MctsBot, sf_skill: int, bot_color: chess.Color,
                  max_moves: int) -> GameRecord:
    engine = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    engine.configure({"Skill Level": sf_skill})
    board = chess.Board()
    moves: list[str] = []
    evals: list[float | None] = []

    n = 0
    while not board.is_game_over() and n < max_moves:
        if board.turn == bot_color:
            uci, val = bot.predict(board)
            move = chess.Move.from_uci(uci)
            evals.append(val)
        else:
            move = engine.play(board, chess.engine.Limit(time=0.1)).move
            evals.append(None)
        moves.append(move.uci())
        board.push(move)
        n += 1
    engine.quit()

    outcome = board.outcome()
    winner_is_bot = bool(outcome and outcome.winner == bot_color)
    is_mate = bool(outcome and outcome.termination == chess.Termination.CHECKMATE)
    return GameRecord(moves, evals, sf_skill, bot_color,
                      winner_is_bot, is_mate and winner_is_bot, len(moves))


# ── rendering ───────────────────────────────────────────────────────────────
def board_to_image(board: chess.Board, last_move: chess.Move | None, size: int,
                   flip: bool) -> Image.Image:
    svg = chess.svg.board(
        board, lastmove=last_move, size=size, flipped=flip,
        colors={"square light": "#f0d9b5", "square dark": "#b58863",
                "lastmove": "#cdd16e88"},
    )
    png = cairosvg.svg2png(bytestring=svg.encode())
    return Image.open(io.BytesIO(png)).convert("RGB")


def _font(bold: bool, sz: int):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", sz)
    except Exception:
        return ImageFont.load_default()


def add_caption(img: Image.Image, text: str, sub: str, banner: str) -> Image.Image:
    """Compose a persistent top banner (the matchup) + the board + a bottom
    caption strip. Nothing overlaps the board."""
    W, H = img.size
    top_h, bot_h = 48, 56
    out = Image.new("RGB", (W, top_h + H + bot_h), (24, 24, 28))
    out.paste(img, (0, top_h))
    draw = ImageDraw.Draw(out)

    # ── persistent top banner: "<bot>  vs  Stockfish · Level N" ──
    # auto-fit: shrink the font until the whole matchup fits the width.
    bot_name, sf_name = banner.split("||")
    pad, gap = 14, 9
    sz = 21
    while sz > 12:
        fb, fr = _font(True, sz), _font(False, sz - 3)
        total = (draw.textlength(bot_name, font=fb) + gap
                 + draw.textlength("vs", font=fr) + gap
                 + draw.textlength(sf_name, font=fb))
        if total <= W - 2 * pad:
            break
        sz -= 1
    fb, fr = _font(True, sz), _font(False, sz - 3)
    y = (top_h - sz) // 2
    x = pad
    draw.text((x, y), bot_name, fill=(120, 190, 255), font=fb)       # bot (blue)
    x += draw.textlength(bot_name, font=fb) + gap
    draw.text((x, y + 2), "vs", fill=(170, 170, 175), font=fr)
    x += draw.textlength("vs", font=fr) + gap
    draw.text((x, y), sf_name, fill=(255, 170, 90), font=fb)         # stockfish (orange)

    # ── bottom caption: move label + sub ──
    draw.text((14, top_h + H + 7), text, fill=(255, 255, 255), font=_font(True, 22))
    if sub:
        draw.text((14, top_h + H + 33), sub, fill=(190, 190, 195), font=_font(False, 15))
    return out


def render_frames(rec: GameRecord, sims: int, size: int,
                  highlight_last: int = 0) -> list[Image.Image]:
    flip = rec.bot_color == chess.BLACK
    side = "White" if rec.bot_color == chess.WHITE else "Black"
    banner = f"ChessTransformer v2.1 ({side})||Stockfish · Level {rec.skill}"
    board = chess.Board()
    frames: list[Image.Image] = []

    img = board_to_image(board, None, size, flip)
    frames.append(add_caption(
        img, f"MCTS · {sims} sims · ~2100 Elo · 11.7M params",
        "Trained on human games only — no self-play, no RL", banner))

    n_moves = rec.plies
    for i, uci in enumerate(rec.moves):
        move = chess.Move.from_uci(uci)
        is_bot = board.turn == rec.bot_color
        san = board.san(move)
        board.push(move)
        if is_bot:
            label = f"Bot: {san}"
            if rec.evals[i] is not None:
                label += f"   (eval {rec.evals[i]:+.2f})"
        else:
            label = f"Stockfish: {san}"
        mn = (i + 2) // 2
        if highlight_last and i >= n_moves - highlight_last:
            sub = f"Move {mn}  ·  ⚡ finishing sequence"
        else:
            sub = f"Move {mn}"
        frames.append(add_caption(board_to_image(board, move, size, flip), label, sub, banner))

    # result frame
    if rec.is_checkmate:
        result_text = "Checkmate — ChessTransformer wins!"
    elif rec.winner_is_bot:
        result_text = "ChessTransformer wins!"
    else:
        result_text = board.result()
    frames.append(add_caption(board_to_image(board, None, size, flip),
                              result_text, board.result(), banner))
    return frames


def frames_to_mp4(frames: list[Image.Image], durations: list[float], out_path: str):
    assert len(frames) == len(durations)
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, img in enumerate(frames):
            p = Path(tmp) / f"frame_{i:04d}.png"
            img.save(str(p))
            paths.append(str(p))
        concat = Path(tmp) / "concat.txt"
        with concat.open("w") as f:
            for p, dur in zip(paths, durations):
                f.write(f"file '{p}'\nduration {dur}\n")
            f.write(f"file '{paths[-1]}'\n")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
            # ensure even dimensions for yuv420p without forcing a square
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos",
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", out_path,
        ], check=True, capture_output=True)
    print(f"Saved: {out_path} ({sum(durations):.1f}s)")


def build_durations(rec: GameRecord, fast_sec: float, highlight_sec: float,
                    highlight_last: int, intro_sec: float, end_sec: float) -> list[float]:
    """One duration per frame: intro, then `fast_sec` per move, with the last
    `highlight_last` *plies* slowed to `highlight_sec`, and a long hold on the
    final mate frame. Frame layout: [intro, move_1, ..., move_P, result]."""
    n_moves = rec.plies
    durs = [intro_sec]
    for i in range(n_moves):
        in_highlight = i >= n_moves - highlight_last
        durs.append(highlight_sec if in_highlight else fast_sec)
    durs.append(end_sec)
    return durs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="data/models/pos2move_v2.1")
    p.add_argument("--skills", type=int, nargs="+", default=[8, 10])
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--cpuct", type=float, default=1.0)
    p.add_argument("--search-games", type=int, default=8,
                   help="Max games to play looking for a decisive bot win")
    p.add_argument("--max-moves", type=int, default=90)
    # playback pacing
    p.add_argument("--fast-sec", type=float, default=0.45,
                   help="Seconds per move through the bulk of the game")
    p.add_argument("--highlight-sec", type=float, default=1.3,
                   help="Seconds per move during the finishing sequence")
    p.add_argument("--highlight-last", type=int, default=10,
                   help="Number of final plies to slow down (the highlight)")
    p.add_argument("--intro-sec", type=float, default=2.0)
    p.add_argument("--sec-end", type=float, default=4.0)
    p.add_argument("--size", type=int, default=600)
    p.add_argument("--out", default="clip.mp4")
    # save / replay (skip the search)
    p.add_argument("--save-game", default=None,
                   help="Write the chosen game's moves to this JSON")
    p.add_argument("--from-json", default=None,
                   help="Replay a saved game JSON instead of searching (instant)")
    args = p.parse_args()

    if args.from_json:
        import json
        best = GameRecord.from_dict(json.loads(Path(args.from_json).read_text()))
        print(f"Replaying saved game from {args.from_json} "
              f"(skill {best.skill}, {best.plies} plies)")
    else:
        bot = Pos2MoveV2MctsBot(model_dir=args.model, num_simulations=args.sims,
                                c_puct=args.cpuct, time_limit=0.0)
        best = None
        for g in range(args.search_games):
            skill = args.skills[g % len(args.skills)]
            color = chess.WHITE if g % 2 == 0 else chess.BLACK
            rec = play_headless(bot, skill, color, args.max_moves)
            tag = ("MATE" if rec.is_checkmate else "WIN" if rec.winner_is_bot
                   else "draw/loss")
            print(f"  game {g + 1}/{args.search_games}: skill {skill}, "
                  f"bot {'W' if color else 'B'} → {tag} in {rec.plies} plies")
            if best is None or rec.score() > best.score():
                best = rec
            if best.is_checkmate and best.plies <= 60:
                print("  found a clean checkmate win — stopping search.")
                break
        assert best is not None

    # always persist the chosen game so re-renders never need a re-search
    save_path = args.save_game or (str(Path(args.out).with_suffix("")) + "_game.json")
    import json
    Path(save_path).write_text(json.dumps(best.to_dict()))
    print(f"Saved game to {save_path}")

    print(f"\nRendering best game (skill {best.skill}, "
          f"{'mate' if best.is_checkmate else 'win' if best.winner_is_bot else 'non-win'}, "
          f"{best.plies} plies)...")
    hl = min(args.highlight_last, best.plies)
    frames = render_frames(best, args.sims, args.size, highlight_last=hl)
    durations = build_durations(best, args.fast_sec, args.highlight_sec,
                                hl, args.intro_sec, args.sec_end)
    frames_to_mp4(frames, durations, args.out)


if __name__ == "__main__":
    main()
