#!/bin/sh
# Run nas_fetch.sh over a list of models, ONE AT A TIME, on the QNAP.
#
# Sequential because the destination pool is HDD and shares spindles with Plex
# and the arr stack, and because the house WAN is the real bottleneck -- three
# parallel pulls of a 50 GB model finish no sooner in aggregate and make the
# media stack stutter. The K3 fetch learned the same thing.
#
# Each model gets its own directory named the way this store already names
# things (the HF id's last segment), so nas_inventory.py and lan_queue.sh find
# them without special cases.
set -u

FETCH=/share/models/_fetch.sh
ROOT=/share/models
LOG=$ROOT/_fetch_driver.log

echo "=== driver start $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$ ===" >> "$LOG"

for m in "$@"; do
  name=$(printf '%s' "$m" | sed 's|.*/||')
  dest=$ROOT/$name
  # Already complete? The marker count against the file list is the check --
  # a directory existing says nothing, which is exactly how 93 minutes and
  # 1.45 TB were spent re-baking a K3 arena that was already there.
  if [ -s "$dest/.filelist" ]; then
    want=$(wc -l < "$dest/.filelist")
    have=$(find "$dest" -name '*.done' 2>/dev/null | wc -l)
    if [ "$want" -gt 0 ] && [ "$have" -ge "$want" ]; then
      echo "SKIP $m -- $have/$want files already complete" >> "$LOG"
      continue
    fi
  fi
  echo "--- $m -> $dest $(date -u +%H:%M:%SZ)" >> "$LOG"
  sh "$FETCH" "$m" "$dest" >> "$LOG" 2>&1
  echo "--- $m exit=$? $(date -u +%H:%M:%SZ)" >> "$LOG"
done

echo "=== driver end $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
