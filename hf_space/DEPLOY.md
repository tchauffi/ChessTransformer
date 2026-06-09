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

## Deploy via GitHub Actions (recommended — no stored token)

Uses HF [Trusted Publishers](https://huggingface.co/docs/hub/trusted-publishers):
the CI job proves its identity with a short-lived GitHub OIDC token and exchanges
it for a 1-hour HF token. Nothing to store or rotate. Workflow:
`.github/workflows/deploy-hf-space.yml`.

1. **Create the Space**: https://huggingface.co/new-space → SDK **Gradio**,
   hardware **CPU basic** (free), name it `chesstransformer`.

2. **Register this repo as a trusted publisher** on the Space →
   *Settings → Trusted Publishers* → provider **GitHub Actions**:
   - `repository` = `tchauffi/ChessTransformer`
   - `ref` = `refs/heads/main`
   - `workflow` = `deploy-hf-space.yml`

3. If your HF username isn't `tchauffi`, set the repo variable **`HF_SPACE_REPO`**
   (GitHub → Settings → Secrets and variables → Actions → Variables) to
   `<you>/chesstransformer`.

4. **Merge to `main`.** The workflow runs on any push touching `hf_space/**`,
   the package, or the v2.1 weights — or trigger it manually from the Actions tab
   (*Run workflow*). It assembles the bundle (`prepare.sh`) and uploads it.

The Space builds and goes live at `https://huggingface.co/spaces/<you>/chesstransformer`
(first build takes a few minutes: torch CPU + gradio).

## Deploy manually (alternative — needs a write token)

```bash
bash hf_space/prepare.sh                       # writes hf_space/space_build/
cd hf_space/space_build
git init && git lfs install
git add -A && git commit -m "ChessTransformer playable demo"
git remote add space https://huggingface.co/spaces/<you>/chesstransformer
git push --force space main                    # huggingface-cli login first
```

## Notes

- **Strength vs speed**: the slider controls MCTS sims/move. Default 200
  (~1.5s/move on the free 2-vCPU tier, ~1725 Elo). Full strength is 800 sims
  (~2100 Elo) but slower on CPU.
- **Weights** are bundled into the Space (≈45 MB, via LFS). If you'd rather not
  bundle them, set the `MODEL_PATH` env var on the Space, or rely on the
  app's GitHub-LFS download fallback (slower cold start).
- Once live, update the GitHub README's Demo section with the Space link and
  it's ready for a (re-)Show HN.
