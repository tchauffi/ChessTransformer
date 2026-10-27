# Set up a machine for ChessTransformer's RL loop (scripts/rl_generation.sh) and run it.
#
#   make setup              uv, Rust, the Python env, the self-play engine and the data
#   make check              GPU and engine, then a short self-play smoke test on the champion
#   make bench              self-play games/h at WORKERS x PARALLEL (try a few before `make tune`)
#   make bench-forward      the GPU's evals/s by batch size, the ceiling `make bench` compares to
#   make tune               this machine's self-play size, written to data/rl/ for the loop
#   make validate           check a generation's games, shards and training targets
#   make loop               generations back to back until `make stop` (run it inside tmux)
#   make continuous         continuous training instead, on every GPU, until `make stop`
#   make scratch            the same for a new network from random weights (data/rl/scratch)
#   make backup             upload what is new; CHECKPOINT=1 after `make stop` to move the
#                           continuous rig: `make setup` + `make continuous` resume it elsewhere
#   make status             the live status page (localhost:8765; reach it with ssh -L 8765:localhost:8765)
#
# The data comes from the private Hugging Face repos scripts/hf_sync.py fills after each gate:
# the champions (MODELS), the games and the loop's state (GAMES) and the inputs (INPUTS). Log in
# first with `uv run hf auth login`, or export HF_TOKEN. Variables override on the command line:
#
#   make data GENS="gen14 gen15"      those self-play generations' games, or GENS=all for every one
#                                     (default: the ones after the champion, all the loop trains
#                                     on; each generation is hundreds of files, and HF allows
#                                     1000 requests per 5 minutes)
#   make bench WORKERS=4 PARALLEL=1024 MINUTES=15
#   make tune WORKERS=4 PARALLEL=1024 HOURS=6

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
export PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)
# Numbers awk prints feed other tools: no locale's decimal comma (0,25 under fr_FR).
export LC_ALL := C

RL       ?= data/rl
DATA     ?= data
MODELS   ?= tchauffi/chesstransformer-rl
GAMES    ?= tchauffi/chesstransformer-rl-selfplay
INPUTS   ?= tchauffi/chesstransformer-inputs
# Self-play generations to download (default: those after the champion; all), and the input
# folders under data/.
GENS       ?=
INPUT_DIRS ?= openings shards/elite_k32_v2
# Self-play size: workers sharing the GPU(s), games in flight between them, GPU ids.
WORKERS  ?= 4
PARALLEL ?= 1024
# Search threads per worker (THREADS), the gate's games in flight and threads, and the most
# games a gate and an anchor match play (up to 1000: the book's first 500 openings).
THREADS  ?= 6
MATCH_PARALLEL ?=
MATCH_THREADS  ?=
GATE_GAMES     ?=
ANCHOR_GAMES   ?=
GPUS     ?=
HOURS    ?=
MINUTES  ?= 15
# The status page: its address, and the nightly pause its estimates skip (none, or e.g. 22-8).
HOST     ?= 127.0.0.1
PAUSE    ?= none
# Never rust/target: on the home machine the deployed lichess bot's binary lives there.
WHEEL_TARGET ?= rust/target-wheel

HF := uv run hf
STATE := README.md history.tsv anchors.tsv selfplay_hours forced_playouts policy_target policy_temp \
	fresh_full full_frac draw_contempt sim_batch temp_cutoff selfplay_games soft_mix sims c_puct \
	cont_window_rows cont_reuse cont_lr cont_eval_hours cont_eval_games rl_generation.log gates/*
# The loop's self-play settings (rl_generation.sh), for the benchmark.
SELFPLAY_ARGS = --sims $$(cat $(RL)/sims 2>/dev/null || echo 800) --fast-sims 160 \
	--c-puct $$(cat $(RL)/c_puct 2>/dev/null || echo 1.0) --full-frac $$(cat $(RL)/full_frac 2>/dev/null || echo 0.25) --cap-total all \
	--dirichlet-alpha 0.3 --dirichlet-eps 0.25 --forced-playouts 2 --fresh-full \
	--draw-contempt $$(cat $(RL)/draw_contempt 2>/dev/null || echo 0.2) \
	--temp-cutoff $$(cat $(RL)/temp_cutoff 2>/dev/null || echo 0.05) \
	--sim-batch $$(cat $(RL)/sim_batch 2>/dev/null || echo 1) \
	--book data/openings/book2k.json

.PHONY: help setup tools env engine data check bench bench-forward validate tune mps loop continuous scratch viewer stop status backup push-inputs

help:  ## this list
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:[^#]*## /\t/' | column -t -s $$'\t'

setup: engine data  ## everything a new machine needs before `make check`

tools:  ## uv and a Rust toolchain, if missing (a C compiler must already be there)
	@command -v cc >/dev/null || { echo "no C compiler: sudo apt install build-essential"; exit 1; }
	command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
	command -v cargo >/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

env: tools  ## the Python env, with the optimized group (ONNX export and parity check)
	@if pgrep -f "[s]cripts/rl_generation.sh" >/dev/null; then  # [s]: not this recipe itself
		echo "the RL loop is running here: uv sync would swap its packages under it"; exit 1
	fi
	uv sync --group optimized

engine: env  ## build and install the Rust self-play engine (uv sync removes it)
	uvx maturin build --release -m rust/selfplay-core/Cargo.toml -o dist \
		--target-dir $(WHEEL_TARGET) --interpreter .venv/bin/python
	uv pip install --python .venv/bin/python --force-reinstall --no-deps \
		"$$(ls -t dist/selfplay_core-*.whl | head -1)"  # --python: not an activated VIRTUAL_ENV

data:  ## download the champion, the games, the loop's state and the inputs; rebuild data/rl
	@if [[ -e $(RL)/champion && -z "$(FORCE)" ]]; then
		echo "$(RL)/champion exists: this machine already has a loop (FORCE=1 to download over it)"; exit 1
	fi
	$(HF) auth whoami >/dev/null 2>&1 || { echo "log in first: uv run hf auth login (or export HF_TOKEN)"; exit 1; }
	$(HF) download $(MODELS) champion.json README.md --local-dir $(RL)/models --quiet >/dev/null
	champ=$$(uv run python -c 'import json, sys; print(json.load(open(sys.argv[1]))["champion"])' $(RL)/models/champion.json)
	echo "champion $$champ: downloading it, then the games and the loop's state, then the inputs"
	$(HF) download $(MODELS) --include "$$champ/*" --local-dir $(RL)/models --quiet >/dev/null
	# The anchors the gates hold candidates to (data/rl/anchors/v2.1, ...).
	$(HF) download $(MODELS) --include "anchors/*" --local-dir $(RL) --quiet >/dev/null
	$(HF) download $(GAMES) --repo-type dataset --local-dir $(RL) \
		--include $(foreach f,$(STATE),"$(f)") --quiet >/dev/null
	gens="$(strip $(GENS))"
	if [[ -z $$gens ]]; then  # FRESH_ONLY: the loop trains only on the generations after the champion
		gens=$$(awk -v c=$${champ#gen} 'NR > 1 && substr($$1, 4) + 0 > c + 0 { print $$1 }' $(RL)/history.tsv)
	fi
	if [[ $$gens == all ]]; then
		$(HF) download $(GAMES) --repo-type dataset --local-dir $(RL) --quiet >/dev/null
	elif [[ -n $$gens ]]; then
		pats=(); for g in $$gens; do pats+=("selfplay/$$g/*"); done
		$(HF) download $(GAMES) --repo-type dataset --local-dir $(RL) --include "$${pats[@]}" --quiet >/dev/null
	fi
	# The continuous rig, when one was backed up: its accepted network, the trainer checkpoint
	# (`make backup CHECKPOINT=1`), its games, evaluations and state; `make continuous` resumes it.
	$(HF) download $(MODELS) --include "continuous/*" --local-dir $(RL) --quiet >/dev/null
	$(HF) download $(GAMES) --repo-type dataset --include "continuous/*" --local-dir $(RL) --quiet >/dev/null
	$(HF) download $(INPUTS) --repo-type dataset --local-dir $(DATA) \
		--include $(foreach d,$(INPUT_DIRS),"$(d)/*") --quiet >/dev/null
	# Every generation history.tsv records is finished: mark it so, as the loop left it.
	tail -n +2 $(RL)/history.tsv | cut -f1 | while read -r g; do
		mkdir -p $(RL)/selfplay/$$g
		touch $(RL)/selfplay/$$g/{.selfplay_done,.gate_done,.generation_done}
	done
	ln -sfn models/$$champ $(RL)/champion
	mkdir -p logs
	[[ -e logs/rl_generation.log ]] || cp $(RL)/rl_generation.log logs/rl_generation.log
	echo "champion $$champ; $$(($$(wc -l < $(RL)/history.tsv) - 1)) generations recorded;" \
		"$$(find $(RL)/selfplay -name games.jsonl | wc -l) with their games"

check:  ## the GPU and the engine, then 2 self-play workers on the champion for about a minute
	uv run python -c "import torch, selfplay_core; assert torch.cuda.is_available(), 'no CUDA GPU'; \
		print('GPU:', ', '.join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())), \
		'| torch', torch.__version__)"
	rm -rf /tmp/ct-smoke
	uv run python scripts/selfplay_rust.py --model $(RL)/champion --out /tmp/ct-smoke --games 64 \
		--sims 64 --fast-sims 16 --max-plies 60 --parallel-games 64 --workers 2 --max-hours 0.02 \
		--forced-playouts 2 --dirichlet-alpha 0.3 --fresh-full --draw-contempt 0.2 --sim-batch 1 | grep -vE "wrote positions"
	# The loop's engine records the root search over every legal move (gen03 onwards).
	uv run python -c "import glob, numpy as np; \
		d = np.load(sorted(glob.glob('/tmp/ct-smoke/positions_*.npz'))[0]); \
		assert 'legal_q' in d, 'this engine does not record the root search: not the loop engine'"
	echo "smoke test: $$(wc -l < /tmp/ct-smoke/games.jsonl) games written, forced playouts, draw contempt and one leaf per wave on"

bench:  ## self-play games/h at WORKERS x PARALLEL for MINUTES, the loop's settings (data/rl untouched)
	out=/tmp/ct-bench-$(WORKERS)x$(PARALLEL); rm -rf $$out
	uv run python scripts/selfplay_rust.py --model $(RL)/champion --out $$out --games 1000000 \
		--max-hours $$(awk 'BEGIN { print $(MINUTES) / 60 }') --parallel-games $(PARALLEL) \
		--workers $(WORKERS) --threads $(THREADS) $(if $(GPUS),--gpus $(GPUS)) --seed 999 $(SELFPLAY_ARGS) \
		| awk '/ games \| / { if (++n % 25 == 0) print; next } !/wrote positions/'
	nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

bench-forward:  ## the GPU's evals/s by batch size: self-play's ceiling (TOP=2048 tries a bigger top bucket)
	uv run python scripts/bench_forward.py --model $(RL)/champion $(if $(TOP),--top $(TOP))

validate:  ## check a generation's games, shards and training targets (GEN=gen15; default: latest done)
	gen=$${GEN:-$$(ls -d $(RL)/selfplay/gen[0-9][0-9]/.selfplay_done | tail -1 | xargs dirname | xargs basename)}
	CUDA_VISIBLE_DEVICES= uv run python scripts/validate_selfplay.py $(RL)/selfplay/$$gen

tune:  ## write WORKERS, PARALLEL, THREADS, GPUS (MATCH_*, GATE_GAMES, ANCHOR_GAMES, HOURS if given)
	echo $(WORKERS) > $(RL)/selfplay_workers
	echo $(PARALLEL) > $(RL)/selfplay_parallel
	echo $(THREADS) > $(RL)/selfplay_threads
	$(if $(MATCH_PARALLEL),echo $(MATCH_PARALLEL) > $(RL)/match_parallel)
	$(if $(MATCH_THREADS),echo $(MATCH_THREADS) > $(RL)/match_threads)
	$(if $(GATE_GAMES),echo $(GATE_GAMES) > $(RL)/gate_games)
	$(if $(ANCHOR_GAMES),echo $(ANCHOR_GAMES) > $(RL)/anchor_games)
	$(if $(GPUS),echo $(GPUS) > $(RL)/selfplay_gpus,rm -f $(RL)/selfplay_gpus)
	$(if $(HOURS),echo $(HOURS) > $(RL)/selfplay_hours)
	echo "self-play: $(WORKERS) workers, $(PARALLEL) games in flight$(if $(GPUS), on GPUs $(GPUS)), $$(cat $(RL)/selfplay_hours 2>/dev/null || echo 8) h per generation"

mps:  ## start NVIDIA MPS, so several workers' kernels run on a GPU at once
	nvidia-cuda-mps-control -d

continuous:  ## continuous training until `make stop` (several GPUs: trainer on the first, self-play on the rest); logs/rl_continuous*.log
	mkdir -p logs
	rm -f $(RL)/STOP
	RL=$(RL) /bin/bash scripts/rl_continuous.sh >> logs/rl_continuous.log 2>&1

# From scratch, as in the paper: a random network with the champion's architecture, no seed games,
# no human replay, a window grown from 600k positions and training from 600k (from 200k, the first
# games' results were memorised), every move searched (SCRATCH_SIMS), the value on the game
# result (soft mix 0.1), lr 5e-4; state in $(RL)/scratch, logs in logs/scratch/. One GPU
# runs one rig: `make stop`, then `make continuous` resumes the other.
SCRATCH_SIMS ?= 300
SCRATCH_START ?= 600000
scratch:  ## continuous training of a new network from scratch (data/rl/scratch) until `make stop`
	mkdir -p logs/scratch
	rm -f $(RL)/STOP
	[[ -f $(RL)/scratch/best/model.safetensors ]] || \
		.venv/bin/python scripts/init_random_model.py --like $(RL)/champion --out $(RL)/scratch/best
	RL=$(RL) CONT=$(RL)/scratch LOGS=logs/scratch SEED_DATA= HUMAN_SHARDS= ANCHOR= \
		SIMS=$(SCRATCH_SIMS) FULL_FRAC=1.0 SOFT_MIX=0.1 POLICY_TEMP=1.0 LR=5e-4 LR_EMBEDDING=5e-4 \
		WINDOW_GROWTH=1 WINDOW_START=$(SCRATCH_START) MIN_ROWS=$(SCRATCH_START) WINDOW_ROWS=20000000 \
		/bin/bash scripts/rl_continuous.sh >> logs/scratch/rl_continuous.log 2>&1

# The games viewer: the web demo's board over this machine's self-play games and their searches
# (scripts/games_viewer.py); HOST=0.0.0.0 to open it from the network.
VIEWER_PORT ?= 8766
viewer:  ## build the self-play games viewer and serve it on HOST:VIEWER_PORT/games
	(cd frontend && { [[ -d node_modules ]] || npm ci --no-audit --no-fund; } \
		&& NEXT_OUTPUT=export NEXT_PUBLIC_GAMES_VIEWER=1 NEXT_PUBLIC_API_URL= npx next build)
	.venv/bin/python scripts/games_viewer.py --host $(HOST) --port $(VIEWER_PORT)

loop:  ## generations back to back until `make stop` (inside tmux; logs/rl_generation.log)
	mkdir -p logs
	rm -f $(RL)/STOP
	while [[ ! -e $(RL)/STOP ]]; do
		RL=$(RL) /bin/bash scripts/rl_generation.sh >> logs/rl_generation.log 2>&1 \
			|| { echo "generation failed: see logs/rl_generation.log"; exit 1; }
	done
	echo "STOP found: loop ended"

stop:  ## let the loop finish the current generation (its gate and backup included), then end
	touch $(RL)/STOP

status:  ## the live status page on HOST:8765 (PAUSE=22-8 for a machine paused at night)
	uv run python scripts/rl_status.py --host $(HOST) --pause $(PAUSE)

backup:  ## upload new champions, games, the loops' state now (CHECKPOINT=1: the continuous trainer's too, to move)
	uv run python scripts/hf_sync.py $(if $(CHECKPOINT),--checkpoint)

push-inputs:  ## once, from a machine that has them: the opening book and the human games
	uv run python scripts/hf_sync.py --only inputs
