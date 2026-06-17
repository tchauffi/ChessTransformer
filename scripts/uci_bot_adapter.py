"""Drop-in `predict()` bot backed by a UCI engine (the Rust ct-bot).

Lets elo_gauntlet.run_gauntlet / sims_scaling.py drive `ct-bot uci` exactly
like the Python bots. python-chess sends the cumulative move list with every
`position` command, so the engine's tree reuse and threefold history work
unchanged.

Usage
-----
    from uci_bot_adapter import UciBotAdapter
    bot = UciBotAdapter(["rust/target/release/ct-bot", "uci"], sims=800)
    move_uci, value = bot.predict(board)
    bot.close()
"""

from __future__ import annotations

import chess
import chess.engine


class UciBotAdapter:
    def __init__(self, cmd: list[str], sims: int, options: dict | None = None):
        self.engine = chess.engine.SimpleEngine.popen_uci(cmd)
        if options:
            self.engine.configure(options)
        self.sims = sims
        self.num_simulations = sims  # used by elo_gauntlet for PGN naming
        self.depth = None

    def predict(self, board: chess.Board) -> tuple[str, float | None]:
        result = self.engine.play(board, chess.engine.Limit(nodes=self.sims))
        return result.move.uci(), None

    def close(self):
        self.engine.quit()
