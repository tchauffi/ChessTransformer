# Deploying the ChessTransformer Space

A self-contained Gradio Space that lets anyone play the bot in the browser, on
CPU — no GPU, no signup. This removes the "can't try it out" barrier that gets
Show HN posts flagged.

## Test locally first

```bash
# from the repo root (uses local weights + installed package)
uv run python hf_space/app.py
# open the printed http://127.0.0.1:7860
```

## Deploy to Hugging Face Spaces

1. **Create a Space**: https://huggingface.co/new-space → SDK **Gradio**, hardware
   **CPU basic** (free). Note its git URL, e.g.
   `https://huggingface.co/spaces/<you>/chesstransformer`.

2. **Assemble the deployable bundle** (vendors the package + weights):

   ```bash
   bash hf_space/prepare.sh        # writes hf_space/space_build/
   ```

3. **Push it to the Space:**

   ```bash
   cd hf_space/space_build
   git init && git lfs install
   git add -A && git commit -m "ChessTransformer playable demo"
   git remote add space https://huggingface.co/spaces/<you>/chesstransformer
   git push --force space main
   ```

   (You'll need a HF token with write access; `huggingface-cli login` once.)

4. The Space builds and goes live at
   `https://huggingface.co/spaces/<you>/chesstransformer`. First build takes a
   few minutes (installing torch CPU + gradio).

## Notes

- **Strength vs speed**: the slider controls MCTS sims/move. Default 200
  (~1.5s/move on the free 2-vCPU tier, ~1725 Elo). Full strength is 800 sims
  (~2100 Elo) but slower on CPU.
- **Weights** are bundled into the Space (≈45 MB, via LFS). If you'd rather not
  bundle them, set the `MODEL_PATH` env var on the Space, or rely on the
  app's GitHub-LFS download fallback (slower cold start).
- Once live, update the GitHub README's Demo section with the Space link and
  it's ready for a (re-)Show HN.
