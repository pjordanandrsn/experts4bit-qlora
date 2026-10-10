#!/bin/bash
# Q235's controller (bench/q235/PREREG-q235.md), run by the rental launcher as --command; it creates no compute. Merged, clean
# source and the sealed closure (q235_stage.py), then bench/tc1/tc1_drive.sh with q235_run.sh as its runner: the nonce
# handshake, heartbeat, stall report, partial and final fetch of the receipts, and the error line on a failure are tc1_drive's.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
HEAD=$(git -C "$REPO" rev-parse HEAD)
if [ "${TC1_DRIVE_DRYRUN:-0}" != 1 ]; then
  [ -z "$(git -C "$REPO" status --porcelain)" ] || { echo 'Q235 refuses dirty source'; exit 78; }
  git -C "$REPO" merge-base --is-ancestor "$HEAD" origin/main || { echo 'Q235 refuses an unmerged instrument'; exit 78; }
fi
export E4B_SHA=${E4B_SHA:-$HEAD}
[ "$E4B_SHA" = "$HEAD" ] || { echo 'Q235: E4B_SHA differs from the launching checkout'; exit 78; }
export TC1_BOX=A TC1_RUNNER=q235_run.sh
TC1_EXTRA_STAGE=$(python3 "$HERE/q235_stage.py" --assets "${Q235_ASSET_DIR:?the directory holding the locked input assets}")
export TC1_EXTRA_STAGE
export GNF4_SHA=724ccc454f006c1a46836e434e997f31f293747f       # grouped-nf4-gemm 0.45.0, the commit gate 1 ran
exec bash "$REPO/bench/tc1/tc1_drive.sh"
