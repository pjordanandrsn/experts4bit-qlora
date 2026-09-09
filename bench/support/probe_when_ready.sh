#!/usr/bin/env bash
# Wait for a LAN copy to settle, then probe it. Poll SIZE STABILITY rather than
# the copying process: a `pgrep -f` for the tar pipeline matches this script's
# own command line, which is a mistake already paid for twice in this repo's
# history (kill/pgrep self-match). Size unchanged across three polls means the
# writer is done or wedged, and either way probing tells us which.
set -euo pipefail

DIR=$1            # local model directory being filled
MIN_GB=$2         # refuse to probe a copy that is obviously short
OUT=$3            # row path
shift 3           # remaining args go to the probe

REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-$HOME/.venvs/e4b/bin/python}

stable=0
last=-1
while [ "$stable" -lt 3 ]; do
  now=$(du -sk "$DIR" 2>/dev/null | cut -f1 || echo 0)
  if [ "$now" = "$last" ]; then stable=$((stable + 1)); else stable=0; fi
  last=$now
  sleep 20
done

gb=$((last / 1024 / 1024))
if [ "$gb" -lt "$MIN_GB" ]; then
  echo "REFUSING to probe $DIR: settled at ${gb}GB, expected >= ${MIN_GB}GB -- the copy is short, so a"
  echo "load failure here would be an artefact of a truncated checkpoint and would land in the rows as"
  echo "if it were a fact about the family." >&2
  exit 1
fi

# Size stability is a HINT, not completeness. A stalled transfer looks identical
# to a finished one -- an OLMoE copy went flat for over a minute and this script
# happily started a probe against a checkpoint still being written. The probe's
# integrity stage is the real check (it compares each shard against the length
# its own header declares, exit 5), so treat exit 5 as "not done yet" and come
# back, rather than believing the poll.
cd "$REPO"
attempt=0
while [ "$attempt" -lt 20 ]; do
  attempt=$((attempt + 1))
  echo "$DIR looks settled at ${gb}GB; probe attempt $attempt"
  PYTHONPATH=$REPO "$PY" bench/support/support_probe.py --model "$DIR" --out "$OUT" "$@"
  rc=$?
  if [ "$rc" -ne 5 ]; then
    echo "probe exit $rc (not a broken copy) -- done"
    exit "$rc"
  fi
  echo "exit 5: a shard is short of its declared length, so the copy is still in"
  echo "  flight. Waiting 60s and re-probing rather than recording a transfer"
  echo "  fault as a fact about the family."
  sleep 60
  gb=$(( $(du -sk "$DIR" 2>/dev/null | cut -f1) / 1024 / 1024 ))
done
echo "giving up after $attempt attempts: $DIR never passed the integrity check" >&2
exit 1
