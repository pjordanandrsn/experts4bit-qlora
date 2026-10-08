#!/bin/bash
# CONTROLLER side, delegates lifecycle/receipt transport to existing tc1_drive. Creates no compute itself.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export TC1_BOX=A TC1_RUNNER=dq10_run.sh
export GNF4_SHA=6ee2e10408161a9d3c874975c9191a7f2957e6f4
export TC1_EXTRA_STAGE
TC1_EXTRA_STAGE=$(python3 "$HERE/dq10_stage.py")
# The synthetic lane needs no HF token. The driver sees an explicitly absent token file and stages none.
export HF_TOKEN_FILE="$HERE/no-token-for-synthetic-lane"
exec bash "$REPO/bench/tc1/tc1_drive.sh"
