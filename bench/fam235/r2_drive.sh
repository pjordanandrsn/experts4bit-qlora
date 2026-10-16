#!/bin/bash
# R2-Mixtral's controller (bench/fam235/PREREG-r2-mixtral.md), run by the rental launcher as --command; it creates no compute. A
# copy of Q235's (bench/q235/q235_drive.sh), checked line for line by bench/fam235/r2_drift.py: merged, clean source and R2's
# sealed closure (r2_stage.py), then bench/tc1/tc1_drive.sh with r2_run.sh as its runner.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
HEAD=$(git -C "$REPO" rev-parse HEAD)
if [ "${TC1_DRIVE_DRYRUN:-0}" != 1 ]; then
  [ -z "$(git -C "$REPO" status --porcelain)" ] || { echo 'R2 refuses dirty source'; exit 78; }
  git -C "$REPO" merge-base --is-ancestor "$HEAD" origin/main || { echo 'R2 refuses an unmerged instrument'; exit 78; }
fi
export E4B_SHA=${E4B_SHA:-$HEAD}
[ "$E4B_SHA" = "$HEAD" ] || { echo 'R2: E4B_SHA differs from the launching checkout'; exit 78; }
# Gates 3 and 5 (PREREG "Before any rental"): a passing fetch gate for this commit, under a day old, or no launch.
[ -n "${R2_FETCH_GATE:-}" ] || { echo 'R2 refuses: R2_FETCH_GATE unset -- run bench/fam235/r2_fetch_gate.py before quoting'; exit 78; }
python3 "$HERE/r2_fetch_gate.py" --e4b-sha "$E4B_SHA" --check-report "$R2_FETCH_GATE" \
  || { echo "R2 refuses: the fetch gate report $R2_FETCH_GATE does not let $E4B_SHA launch"; exit 78; }
cp "$R2_FETCH_GATE" "${E4B_RENT_RUN_DIR:?}/r2_fetch_gate.json"
export TC1_BOX=A TC1_RUNNER=r2_run.sh
TC1_EXTRA_STAGE=$(python3 "$HERE/r2_stage.py" --assets "${R2_ASSET_DIR:?the directory holding the locked input assets}")
export TC1_EXTRA_STAGE
export GNF4_SHA=724ccc454f006c1a46836e434e997f31f293747f       # grouped-nf4-gemm 0.45.0, the commit gate 1 ran
exec bash "$REPO/bench/tc1/tc1_drive.sh"
