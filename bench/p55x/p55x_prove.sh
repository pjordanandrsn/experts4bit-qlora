#!/bin/bash
# bench/p55x/p55x_prove.sh -- lane P55x, the PROVING run's --command (board motion 1, 2026-09-07: before any
# run whose guard exceeds an hour, rent one cheap box of the same provider class and image and prove the path
# end to end). It is deliberately trivial in cost and deliberately NOT trivial in what it records.
#
# Two jobs:
#   1. prove the launcher path on hardware -- attach, pre-flight, command handoff, receipt, ledger row,
#      teardown proof. The launcher owns all of that; this script only has to run and return.
#   2. MEASURE, and print, the quantity the registered run depends on and nothing else measures: how fast
#      this class of box can push bytes OUT. The rental pre-flight measures the box's DOWNLOAD (and, since
#      adertha#125, its download from the HF CDN specifically). P55x's product is 15.2 GiB that has to travel
#      the other way, and that direction has never been measured on a box of this class by anything.
#
# It does not refuse. A proving run that can only say "pass" is indistinguishable from one that can only say
# "fail" (P41, 2026-09-07): the number is the point, so the number is printed and recorded whatever it says.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p55x_prove] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; OUT="$E4B_RENT_RUN_DIR/p55x-prove"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -p $PORT root@$HOST"
RSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT"
PROBE_MB=${P55X_PROBE_MB:-256}; MIN_UP_MBS=${P55X_MIN_UP_MBS:-8}; MIN_DISK_GB=${P55X_MIN_DISK_GB:-200}
mkdir -p "$OUT" || { say "cannot create $OUT"; exit 20; }

say "box facts ($HOST:$PORT, instance $E4B_RENT_INSTANCE_ID)"
$SSH 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; echo "---"; df -BG --output=avail /root | tail -1; echo "---"; df -h /root | tail -1; echo "---"; free -g | head -2; echo "---"; lscpu | grep -E "^Model name"' \
  > "$OUT/box.txt" 2>&1 || { say "ssh to the box failed -- the path is NOT proven"; cat "$OUT/box.txt"; exit 21; }
cat "$OUT/box.txt"
GPU=$(head -1 "$OUT/box.txt"); AVAIL=$(sed -n '2p' "$OUT/box.txt" | tr -dc '0-9')

say "upload probe: ${PROBE_MB} MB off the box, the direction the pre-flight does not measure"
$SSH "dd if=/dev/urandom of=/root/.uprobe.bin bs=1M count=$PROBE_MB status=none" || { say "could not create the probe file"; exit 22; }
D=$(mktemp -d "${TMPDIR:-/tmp}/p55x-prove.XXXXXX"); T0=$(date +%s)
rsync -a --no-compress -e "$RSH" "root@$HOST:/root/.uprobe.bin" "$D/" >/dev/null 2>&1; PRC=$?
T1=$(date +%s); $SSH "rm -f /root/.uprobe.bin" >/dev/null 2>&1
GOT=$(wc -c < "$D/.uprobe.bin" 2>/dev/null || echo 0); rm -rf "$D"
UP=$(python3 -c "print(f'{$GOT / 1048576.0 / max(1.0, $T1 - $T0):.2f}')")
ART_MIN=$(python3 -c "print(round(15.2 * 1024 / max(float('$UP'), 0.01) / 60))")

python3 - <<PY | tee "$OUT/prove.json"
import json
print(json.dumps({
  "run_id": "$E4B_RENT_RUN_ID", "instance_id": "$E4B_RENT_INSTANCE_ID",
  "gpu": """$GPU""".strip(), "avail_disk_gb": ${AVAIL:-0}, "lane_disk_floor_gb": $MIN_DISK_GB,
  "upload_probe_mb": $PROBE_MB, "upload_bytes": $GOT, "upload_seconds": $((T1 - T0)),
  "upload_mb_s": float("$UP"), "lane_upload_floor_mb_s": $MIN_UP_MBS, "rsync_rc": $PRC,
  "artifact_15_2_GiB_minutes_at_this_rate": $ART_MIN,
  "would_this_box_pass_the_lane_K0": (${AVAIL:-0} >= $MIN_DISK_GB and float("$UP") >= $MIN_UP_MBS and $PRC == 0),
}, indent=1))
PY
say "MEASURED: upload ${UP} MB/s (lane floor ${MIN_UP_MBS}); the 15.2 GiB artifact would take ~${ART_MIN} min at this rate"
say "MEASURED: ${AVAIL:-0} GB free on /root (lane floor ${MIN_DISK_GB}) on ${GPU}"
say "path proven end to end; the launcher owns the receipt, the ledger row and the teardown proof"
exit 0
