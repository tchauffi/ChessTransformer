---
title: ChessTransformer
emoji: ♟️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.0
app_file: app.py
pinned: false
license: mit
short_description: Play a chess transformer trained only on human games
---

# ♟️ ChessTransformer

Play an **11.7M-parameter transformer trained only on human games** (no self-play,
no reinforcement learning), playing via AlphaZero-style MCTS. At full strength it
reaches **~2100 Elo** against Stockfish; this Space runs on CPU at a lower
simulation count so moves come back in ~1–2s.

You play **White** — drag a piece and the bot replies automatically. Use the
slider to trade strength for speed.

Code, weights, and the training pipeline: **https://github.com/tchauffi/ChessTransformer**

> This `README.md` is the Space config. The deployable Space is assembled by
> `prepare.sh` (see `DEPLOY.md`) — it bundles `app.py`, this file, the slim
> `requirements.txt`, the `chesstransformer` package source, and the model weights.
