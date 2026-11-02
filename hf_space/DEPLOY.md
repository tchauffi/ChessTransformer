# Deploying the ChessTransformer Space

A Docker Space that lets anyone play the bot in the browser, on CPU — no GPU, no
signup. One process does everything: the Rust engine (`ct-bot serve`, ONNX
Runtime, int8 model) answers the API and serves the static Next.js frontend on
port 7860.

## Test locally first

```bash
bash hf_space/prepare.sh                      # writes hf_space/space_build/
docker build -t ct-hf-space hf_space/space_build
docker run --rm -p 7860:7860 --cpus 2 ct-hf-space
# open http://localhost:7860
```

`--cpus 2` mimics the free tier. The first build takes several minutes (the
Rust release build uses LTO); HF's builders take about as long.

## Deploy via GitHub Actions (recommended — no stored token)

Uses HF [Trusted Publishers](https://huggingface.co/docs/hub/trusted-publishers):
the CI job proves its identity with a short-lived GitHub OIDC token and exchanges
it for a 1-hour HF token. Nothing to store or rotate. Workflow:
`.github/workflows/deploy-hf-space.yml`.

1. **Create the Space**: https://huggingface.co/new-space → SDK **Docker**,
   hardware **CPU basic** (free). An existing Gradio Space needs no change: the
   uploaded `README.md` declares `sdk: docker` and the Space rebuilds as Docker.

2. **Register this repo as a trusted publisher** on the Space →
   *Settings → Trusted Publishers* → provider **GitHub Actions**:
   - `repository` = `tchauffi/ChessTransformer`
   - `ref` = `refs/heads/main`
   - `workflow` = `deploy-hf-space.yml`

3. If the Space isn't `tchauffi/ChessTransformer`, set the repo variable
   **`HF_SPACE_REPO`** (GitHub → Settings → Secrets and variables → Actions →
   Variables) to `<you>/<space>`.

4. **Merge to `main`.** The workflow runs on any push touching `hf_space/**`,
   `frontend/**`, `rust/**` or the int8 model — or trigger it manually from the
   Actions tab (*Run workflow*). It assembles the bundle (`prepare.sh`) and
   uploads it, deleting Space files the bundle no longer has.

## Deploy manually (alternative — needs a write token)

```bash
bash hf_space/prepare.sh
hf auth login
hf upload <you>/<space> hf_space/space_build . --repo-type space --delete "*"
```

## Notes

- **Strength vs speed**: `MCTS_SIMS` sets the search per move. The default 800
  measured 2.7–5.7 s a move on the free tier (2026-10-11), roughly twice a local
  2-thread estimate; time scales about linearly with sims. Change it under the
  Space's *Settings → Variables*; it applies on restart, no rebuild. On paid
  CPU hardware raise `ORT_THREADS` to the vCPU count as well.
- **One shared engine session**: every visitor's requests go through the same
  search, one at a time. Moves stay correct for everyone (each request carries
  its position), but simultaneous players wait for each other, and a game
  interleaved with another loses tree reuse and repetition history.
- **Engine search tab** works here: it needs the Rust backend, which is what the
  Space runs.
