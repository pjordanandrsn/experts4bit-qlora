#!/bin/bash
# DQ11 MLP-adapter BUILD Phase 2, CONTROLLER side (DQ11-MLP-BUILD-AMENDMENT-5.md section 8): the launch manifest's
# command under the shared launcher. Stages the sealed science closure (dq11_science_stage.py) and the separately hashed
# Phase 2 instrument, then runs bench/tc1/tc1_drive.sh with dq11_mlp_phase2_box.sh as the box's runner. The box installs
# e4b at this clean, merged HEAD. Nothing here creates, destroys or approves compute.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
HEAD=$(git -C "$REPO" rev-parse HEAD)
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo 'PHASE2_REFUSAL: dirty launch source'; exit 78; }
git -C "$REPO" merge-base --is-ancestor "$HEAD" origin/main || { echo 'PHASE2_REFUSAL: unmerged launch source'; exit 78; }
INSTRUMENT="dq11_mlp_phase1.py dq11_mlp_phase1_run.py dq11_mlp_phase2.py dq11_mlp_phase2_run.py dq11_mlp_phase2_box.sh dq11_mlp_phase2_drive.sh dq11_wheel_retry.py dq11_census_transport.py"
export E4B_SHA=$HEAD
export TC1_BOX=A TC1_RUNNER=dq11_mlp_phase2_box.sh
export DQ11_REQUIRED_DRIVER=${DQ11_REQUIRED_DRIVER:?exact manifest driver required}
export HF_TOKEN_FILE="$HERE/no-token-for-dq11"
STAGE=$(python3 "$HERE/dq11_science_stage.py" --assets "${DQ11_ASSET_DIR:?canonical asset directory required}")
for file in $INSTRUMENT; do
  [ -f "$HERE/$file" ] || { echo "PHASE2_REFUSAL: missing instrument $file"; exit 78; }
  STAGE="$STAGE $HERE/$file"
done
HASHES="${E4B_RENT_RUN_DIR:?controller run directory required}/dq11-mlp-phase2-stage.sha256"
# shellcheck disable=SC2086
( cd "$HERE" && shasum -a 256 $INSTRUMENT ) > "$HASHES"
STAGE="$STAGE $HASHES"
export TC1_EXTRA_STAGE="$STAGE"
exec python3 "$HERE/dq11_census_transport.py" drive bash "$REPO/bench/tc1/tc1_drive.sh"
