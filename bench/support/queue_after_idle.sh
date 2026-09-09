#!/usr/bin/env bash
# Wait for the LAN to go quiet, then run lan_queue.sh for the big checkpoints.
#
# A file-based script rather than an inline `bash -c`: the wait condition has to
# identify the copies in flight, and a `pgrep -f 'tar -C .../models'` written
# inline matches the waiter's OWN command line. That self-match has already cost
# this fleet a killed ssh session once. Polling aggregate SIZE STABILITY needs no
# process inspection at all.
set -uo pipefail

DEST=${DEST:-$HOME/models}
STABLE_POLLS=${STABLE_POLLS:-4}
INTERVAL=${INTERVAL:-30}

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

log "waiting for $DEST to stop growing (${STABLE_POLLS} polls x ${INTERVAL}s)"
stable=0
last=-1
while [ "$stable" -lt "$STABLE_POLLS" ]; do
  now=$(du -sk "$DEST" 2>/dev/null | cut -f1 || echo 0)
  if [ "$now" = "$last" ]; then
    stable=$((stable + 1))
  else
    stable=0
    log "  still moving: $((now / 1024 / 1024))GB"
  fi
  last=$now
  sleep "$INTERVAL"
done
log "quiet at $((last / 1024 / 1024))GB; starting the big queue"

exec bash "$(dirname "$0")/lan_queue.sh"
