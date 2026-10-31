#!/usr/bin/env bash
# Restart the scratch rig (`make scratch`, run as the user unit ct-rl-loop) on the current code
# and settings: `make stop`, wait for the loop to end, then start the unit again with the
# environment it had. Extra VAR=value arguments reach make, e.g. a learning-rate drop:
#
#   scripts/restart_scratch.sh SCRATCH_LR=1.5e-4
#
# A frozen loop (the night pause, 22:00-08:00) never sees STOP, so this refuses one; queue the
# restart past the thaw instead:
#
#   systemd-run --user --unit ct-scratch-restart --on-calendar='2026-11-01 08:02' \
#       -p WorkingDirectory=$PWD scripts/restart_scratch.sh
#
# Cancel a queued one with `systemctl --user stop ct-scratch-restart.timer`.
set -euo pipefail
cd "$(dirname "$0")/.."
UNIT=ct-rl-loop
LOG=logs/scratch/rl_continuous.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
state() { systemctl --user show -p "$1" --value "$UNIT"; }

env_args=()
for kv in $(state Environment); do env_args+=(-E "$kv"); done  # the unit's own, e.g. SP_EXTRA=--trt
(( ${#env_args[@]} )) || env_args=(-E UV_NO_SYNC=1 -E LC_ALL=C)
for kv in "$@"; do env_args+=(-E "$kv"); done

if [[ $(state ActiveState) == active ]]; then
  [[ $(state FreezerState) == running ]] || { log "restart: $UNIT is frozen, not stopping it"; exit 1; }
  log "restart: make stop, then make scratch again at $(git rev-parse --short HEAD)${*:+ with $*}"
  touch data/rl/STOP
  while [[ $(state ActiveState) =~ ^(active|activating|deactivating|reloading)$ ]]; do sleep 15; done
fi
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
systemd-run --user --unit "$UNIT" -p WorkingDirectory="$PWD" "${env_args[@]}" /usr/bin/make scratch
log "restart: $UNIT started"
