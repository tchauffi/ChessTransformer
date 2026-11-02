#!/usr/bin/env bash
# Unattended A/B for the dynamic attention bias and the 2026-10-11 trainer fixes, started at
# 08:00 when the GPU's night pause ends:
#
#   systemd-run --user --unit ct-smolgen-ab --on-calendar "2026-10-12 08:00:00" ... \
#     /bin/bash -c 'scripts/queue_smolgen_ab.sh >> logs/queue_smolgen_ab.log 2>&1'
#
#   A  cand02_sg  gen01 converted by convert_attn_bias.py --full --smolgen (plays exactly as
#                 gen01 until trained), trained on gen01+gen02 with cand02's recipe plus
#                 --new-lr-mult 10 (the zero-started smolgen must move within one generation),
#                 --compile (eager does not fit this model at batch 512: >15.4 GiB; compiled 9.4),
#                 --no-train-dropout (with dropout the KL anchor is mostly noise: 0.050 for
#                 identical weights) and the human replay's rating gap (now the default)
#   B  cand02     same start, same data, old architecture and recipe (trained 2026-10-11;
#                 +6 Elo over gen01 after 387 gate games)
#
# A bundle, not an ablation: it answers "is the new recipe better than the current best".
# Then up to 400 games A vs B at the deployed budget, stopping early on the SPRT (+0/+15).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

RL=data/rl
BASE=$RL/models/gen01_sg
A=$RL/models/cand02_sg
B=$RL/models/cand02
GAMES_OUT=$RL/gates/cand02_sg_vs_cand02.jsonl
log() { echo "[$(date '+%F %T')] $*"; }

if systemctl --user is-active --quiet ct-rl-loop.service; then
  log "the RL loop is running -- one GPU job at a time, not starting"; exit 1
fi

log "forward speed on the GPU, old vs new architecture (bf16, batch 1024, uncompiled)"
uv run python - <<'EOF'
import sys, time, torch
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from grpo_puzzles import load_model
x = (torch.randint(0, 13, (1024, 64)), torch.randint(0, 2, (1024,)), torch.randint(0, 16, (1024,)),
     torch.full((1024,), 8))
x = [t.cuda() for t in x]
for d in ("data/rl/models/gen01", "data/rl/models/gen01_sg"):
    m = load_model(Path(d), "cuda").bfloat16().eval()
    with torch.inference_mode():
        for _ in range(5): m(*x)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(50): m(*x)
        torch.cuda.synchronize()
    print(f"  {d}: {50 * 1024 / (time.perf_counter() - t):,.0f} positions/s")
EOF

if [[ ! -f "$A/model.safetensors" ]]; then
  log "training $A from $BASE on gen01 + gen02, 2 epochs"
  uv run python scripts/train_expert_iter.py --data "$RL/selfplay/gen01" "$RL/selfplay/gen02" \
    --base "$BASE" --out "$A" --epochs 2 --batch-size 512 --lr 1e-4 --kl-coef 0.1 \
    --soft-mix 0.5 --value-weight 1.0 --human-shards data/shards/elite_k32_v2 --human-frac 0.2 \
    --select last --new-lr-mult 10 --no-train-dropout --compile \
    || { log "training failed"; exit 1; }
  uv run --with onnxscript python scripts/export_onnx.py "$A" -o "$A/model.onnx" \
    || { log "export failed"; exit 1; }
  uv run python scripts/check_onnx_parity.py --model "$A" --onnx "$A/model.onnx" \
    || { log "ONNX parity failed"; exit 1; }
fi

log "match $A vs $B, up to 400 games"
ok=0
for extra in "" "--no-compile --engines 1"; do
  rm -f "$GAMES_OUT"
  uv run python scripts/gpu_match.py --model-a "$A" --model-b "$B" --label-a cand02_sg \
    --label-b cand02 --openings 200 --max-sims 4000 --ponder-max-sims 2000 --sprt \
    --games-out "$GAMES_OUT" $extra | tail -14 && { ok=1; break; }
  log "match failed${extra:+ ($extra)}"
done
((ok)) || exit 1
log "done"
