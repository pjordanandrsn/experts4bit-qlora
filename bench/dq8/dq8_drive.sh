#!/bin/bash
# Existing controller/guard/receipt transport; creates no compute itself.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DQ7="$REPO/bench/dq7"
export TC1_BOX=A TC1_RUNNER=dq8_run.sh
export GNF4_SHA=6ee2e10408161a9d3c874975c9191a7f2957e6f4
export TC1_EXTRA_STAGE="$HERE/dq8_run.sh $HERE/dq8_refusal.py $HERE/dq8_reduce.py $DQ7/dq7_arm.py $DQ7/dq7_subject.py $DQ7/runtime.json $DQ7/configs/qwen3_14b.json $DQ7/configs/qwen3_32b.json $DQ7/configs/llama31_8b.json $REPO/bench/dq6/dq6_vram_probe.py $REPO/bench/dq3/dq3_vram_probe.py $REPO/bench/dq3/dq3_egress_probe.py"
export HF_TOKEN_FILE="$HERE/no-token-for-synthetic-lane"
exec bash "$REPO/bench/tc1/tc1_drive.sh"
