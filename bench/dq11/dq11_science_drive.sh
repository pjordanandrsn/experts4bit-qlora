#!/bin/bash
# Real instrument controller. No compute creation. Merged clean source and sealed inputs before transport.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DQ11_HEAD=$(git -C "$REPO" rev-parse HEAD)
if [ "${TC1_DRIVE_DRYRUN:-0}" != 1 ]; then
  [ -z "$(git -C "$REPO" status --porcelain)" ] || { echo 'DQ11 refuses dirty source'; exit 78; }
  git -C "$REPO" merge-base --is-ancestor "$DQ11_HEAD" origin/main || { echo 'DQ11 refuses unmerged instrument'; exit 78; }
fi
export E4B_SHA=${E4B_SHA:-$DQ11_HEAD}
[ "$E4B_SHA" = "$DQ11_HEAD" ] || { echo 'DQ11 source differs from launcher checkout'; exit 78; }
export TC1_BOX=A TC1_RUNNER=dq11_science_run.sh
export TC1_EXTRA_STAGE
TC1_EXTRA_STAGE=$(python3 "$HERE/dq11_science_stage.py" --assets "${DQ11_ASSET_DIR:?canonical asset directory required}")
export GNF4_SHA=724ccc454f006c1a46836e434e997f31f293747f
export HF_TOKEN_FILE="$HERE/no-token-for-dq11"
exec bash "$REPO/bench/tc1/tc1_drive.sh"
