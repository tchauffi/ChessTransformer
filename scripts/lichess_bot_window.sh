#!/usr/bin/env bash
# Stop the lichess bot service between games. Run daily at 19:00 by the systemd user timer
# ct-bot-lichess-stop.timer; ct-bot-lichess-start.timer starts the service again at 08:00.
#
# ct-bot has no SIGTERM handling, so stopping it mid-game abandons the game and loses it
# on time. This polls the account's games in progress and stops the service in the first
# gap. A challenge accepted between the poll and the stop is the only race, and the gap
# is one request. After MAX_WAIT_MIN the service is stopped regardless.
#
#   LICHESS_BOT_TOKEN=... scripts/lichess_bot_window.sh
set -uo pipefail

SERVICE=ct-bot-lichess.service
POLL_SEC="${POLL_SEC:-10}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-120}"

deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
while systemctl --user is-active --quiet "$SERVICE"; do
  # An unreachable API means the bot cannot be playing either: stop.
  playing="$(curl -sf --max-time 15 -H "Authorization: Bearer $LICHESS_BOT_TOKEN" \
    https://lichess.org/api/account/playing | jq -r '.nowPlaying | length' 2>/dev/null)"
  if [[ "$playing" == "0" || -z "$playing" ]]; then
    echo "no game in progress (api: ${playing:-unreachable}), stopping $SERVICE"
    exec systemctl --user stop "$SERVICE"
  fi
  if (( $(date +%s) >= deadline )); then
    echo "still in a game after ${MAX_WAIT_MIN} min, stopping $SERVICE anyway"
    exec systemctl --user stop "$SERVICE"
  fi
  sleep "$POLL_SEC"
done
echo "$SERVICE is not running"
