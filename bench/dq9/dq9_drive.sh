#!/bin/bash
# CONTROLLER side, delegates lifecycle/receipt transport to existing tc1_drive. Creates no compute itself.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export TC1_BOX=A TC1_RUNNER=dq9_run.sh
export GNF4_SHA=6ee2e10408161a9d3c874975c9191a7f2957e6f4
DQ7="$REPO/bench/dq7"
export TC1_EXTRA_STAGE="$HERE/dq9_run.sh $HERE/dq9_arm.py $HERE/dq9_proof.py $HERE/dq9_phase.py $HERE/dq9_reduce.py $HERE/dq9_admission.py $HERE/instrument.sha256 $HERE/runtime.json $DQ7/dq7_arm.py $DQ7/dq7_subject.py $DQ7/configs/qwen3_14b.json $DQ7/configs/qwen3_32b.json $DQ7/configs/llama31_8b.json $REPO/bench/dq3/dq3_vram_probe.py $REPO/bench/dq3/dq3_egress_probe.py"
# The synthetic lane needs no HF token. The driver sees an explicitly absent token file and stages none.
export HF_TOKEN_FILE="$HERE/no-token-for-synthetic-lane"
exec bash "$REPO/bench/tc1/tc1_drive.sh"
