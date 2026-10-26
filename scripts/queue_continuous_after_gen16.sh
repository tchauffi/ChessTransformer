#!/usr/bin/env bash
# One-off (2026-10-26): switch from the generational loop to continuous training once gen16 is
# settled. Waits for gen16's gate (and anchor) to end -- the loop then starts gen17's
# self-play --, stops the loop, moves gen17's first games aside and runs `make continuous`
# as ct-rl-loop (so the night freeze applies), from the champion the gate left, with gen16's
# games in the replay window. Cancel before it fires: systemctl --user stop ct-continuous-wait.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
RL=data/rl
log() { echo "[$(date '+%F %T')] switch: $*"; }

until [[ -f $RL/selfplay/gen16/.generation_done ]]; do
  systemctl --user is-active -q ct-rl-loop.service || { log "the loop stopped before gen16 ended"; exit 1; }
  sleep 20
done
log "gen16 done (champion $(readlink $RL/champion)): stopping the generational loop"
systemctl --user stop ct-rl-loop.service
systemctl --user reset-failed ct-rl-loop.service 2>/dev/null
if [[ -d $RL/selfplay/gen17 && ! -f $RL/selfplay/gen17/.selfplay_done ]]; then
  mv $RL/selfplay/gen17 "$RL/selfplay/gen17_before_continuous_$(date +%Y%m%d%H%M)"
  log "moved gen17's first games aside"
fi
exec systemd-run --user --unit ct-rl-loop -p WorkingDirectory="$PWD" -E UV_NO_SYNC=1 -E LC_ALL=C \
  -E PATH="$PATH" -E SEED_DATA=$RL/selfplay/gen16 /usr/bin/make continuous
