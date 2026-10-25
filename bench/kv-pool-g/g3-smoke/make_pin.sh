#!/bin/bash
# bench/kv-pool-g/g3-smoke/make_pin.sh -- regenerate staged.sha256 (every file the G3 smoke box stages, named as the box
# sees it); --check compares instead of writing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
pin(){ for f in g3_smoke_run.sh g3_smoke.py; do echo "$(sha_of "$HERE/$f")  $f"; done
       for f in k8_bake.py calib.json; do echo "$(sha_of "$REPO/bench/p39/$f")  $f"; done; }
if [ "${1:-}" = --check ]; then
  diff <(pin) "$HERE/staged.sha256" > /dev/null || { echo "staged.sha256 is stale: run bench/kv-pool-g/g3-smoke/make_pin.sh"; exit 1; }
  echo "staged.sha256 matches"; exit 0
fi
pin > "$HERE/staged.sha256"
echo "pinned $(wc -l < "$HERE/staged.sha256" | tr -d ' ') files -> $HERE/staged.sha256"
