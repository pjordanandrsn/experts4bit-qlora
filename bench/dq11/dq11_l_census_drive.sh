#!/bin/bash
# Stage the sealed 13b science and separately hashed merged census instrument.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
HEAD=$(git -C "$REPO" rev-parse HEAD)
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo 'CENSUS_REFUSAL: dirty launch source'; exit 78; }
git -C "$REPO" merge-base --is-ancestor "$HEAD" origin/main || { echo 'CENSUS_REFUSAL: unmerged launch source'; exit 78; }
git -C "$REPO" merge-base --is-ancestor 6e197a3f42599c8d10667a4c133627496e40f28f "$HEAD" || {
  echo 'CENSUS_REFUSAL: merged instrument absent'; exit 78;
}
export E4B_SHA=13b55990433f357d63232f75926ce19f5bda5b42
export GNF4_SHA=724ccc454f006c1a46836e434e997f31f293747f
export TC1_BOX=A TC1_RUNNER=dq11_l_census_box.sh TC1_L_CENSUS=1
export DQ11_REQUIRED_DRIVER=${DQ11_REQUIRED_DRIVER:?exact manifest driver required}
export HF_TOKEN_FILE="$HERE/no-token-for-dq11"
STAGE=$(python3 "$HERE/dq11_science_stage.py" --assets "${DQ11_ASSET_DIR:?canonical asset directory required}")
for file in dq11_l_census_run.py dq11_l_census_capture.py dq11_l_census_reduce.py dq11_l_census_launch.py dq11_l_census_box.sh dq11_l_census_drive.sh dq11_wheel_retry.py dq11_census_transport.py; do
  [ -f "$HERE/$file" ] || { echo "CENSUS_REFUSAL: missing instrument $file"; exit 78; }
  STAGE="$STAGE $HERE/$file"
done
HASHES="${E4B_RENT_RUN_DIR:?controller run directory required}/dq11-l-census-stage.sha256"
( cd "$HERE" && shasum -a 256 dq11_l_census_run.py dq11_l_census_capture.py dq11_l_census_reduce.py \
    dq11_l_census_launch.py dq11_l_census_box.sh dq11_l_census_drive.sh dq11_wheel_retry.py dq11_census_transport.py ) > "$HASHES"
STAGE="$STAGE $HASHES"
export TC1_EXTRA_STAGE="$STAGE"
export TC1_CENSUS_STAGE_HELPER="$HERE/dq11_census_transport.py"
exec python3 "$HERE/dq11_census_transport.py" drive bash "$REPO/bench/tc1/tc1_drive.sh"
