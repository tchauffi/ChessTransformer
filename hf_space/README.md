---
title: ChessTransformer
emoji: ♟️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Play a chess transformer trained only on human games
---

# ♟️ ChessTransformer

Play an **11.7M-parameter transformer trained only on human games** (no self-play,
no reinforcement learning), searching with AlphaZero-style MCTS. At full strength it
reaches **~2100 Elo** against Stockfish.

- Pick a color and a clock, then play: drag or click to move, premove while it
  thinks, right-click to draw arrows and mark squares.
- **Watch it play itself**, or press **Autoplay** mid-game to hand it your side.
- The **Engine search** tab shows the branches it explored for each move; hover one
  to replay that line on the board.

The engine is the Rust `ct-bot` (ONNX Runtime, int8, CPU). On this free 2-vCPU
Space it searches 800 positions a move, which takes a few seconds.

Code, weights, and the training pipeline: **https://github.com/tchauffi/ChessTransformer**

> This `README.md` is the Space config. The deployable Space is assembled by
> `prepare.sh` (see `DEPLOY.md`): this file, the Dockerfile, the `rust/` and
> `frontend/` sources, and the int8 model.
