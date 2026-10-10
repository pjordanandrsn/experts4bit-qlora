#!/bin/bash
# Draft controller: only TC1's local stage-plan branch is admitted. No transport.
set -euo pipefail
[ "${TC1_DRIVE_DRYRUN:-0}" = 1 ] || { echo 'DQ11 DRAFT_NOT_LAUNCHABLE: no SSH or compute'; exit 78; }
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export TC1_BOX=A TC1_RUNNER=dq11_run.sh
export TC1_EXTRA_STAGE
TC1_EXTRA_STAGE=$(python3 "$HERE/dq11_stage.py")
export GNF4_SHA=724ccc454f006c1a46836e434e997f31f293747f
export HF_TOKEN_FILE="$HERE/no-token-for-dq11-draft"
exec bash "$REPO/bench/tc1/tc1_drive.sh"
