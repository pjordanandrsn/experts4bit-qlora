#!/bin/bash
# Re-pin bench/p67/staged.sha256 from this tree (run from anywhere). The names are as the BOX sees them after
# tp4_drive.sh stages them flat into /root/tp4; p67_drive.sh resolves each back to its source here.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
sha(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
{
  echo "# Names as the BOX sees them after staging (flat in /root/tp4). p67_run.sh runs \`sha256sum -c\` on this file ON"
  echo "# the box before anything else; p67_drive.sh resolves each name to its source in this tree and compares; CI"
  echo "# (tests/test_p67_staged_pin.py) does the same. Regenerate with bench/p67/pin.sh."
  for f in "$HERE/p67_run.sh" "$HERE/registered.knobs" "$REPO/bench/tp4/tp4_run.sh" "$REPO/bench/tp4/tp4_arm.py" \
           "$REPO/bench/tp4/tp4_reduce.py" "$REPO/bench/tp4/tp4_alpaca.py" "$REPO/bench/flagship-matrix/drivers/n9_datasets.py" \
           "$REPO/bench/flagship-matrix/ds_manifest.json"; do
    echo "$(sha "$f")  $(basename "$f")"
  done
} > "$HERE/staged.sha256"
cat "$HERE/staged.sha256"
