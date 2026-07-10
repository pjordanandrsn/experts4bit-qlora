#!/usr/bin/env bash
# 02_fio_ladder.sh — measure S (SSD sequential read) single-drive then RAID-0 stripe, + the
# one-hour throttle curve with smartctl temps. This is the characterization nobody else publishes
# and a STANDING ASSET even if the training sweep never runs. Uses O_DIRECT (bypasses page cache).
# Requires: fio, smartctl (nvme-cli). Set TARGET_SINGLE and TARGET_STRIPE to mount dirs / raw devs.
set -u
OUT="${OUT:-receipts}"; mkdir -p "$OUT"
SINGLE="${TARGET_SINGLE:-/mnt/single}"     # a single raw drive's fs, or a raw /dev/nvmeXn1
STRIPE="${TARGET_STRIPE:-/mnt/stripe}"      # the RAID-0 md0 fs
run_fio() { # label path
  local label="$1" path="$2"
  echo "=== fio $label ($path) — 1.25GB seq read, QD1 & QD2, O_DIRECT ==="
  for qd in 1 2; do
    fio --name=seqread --rw=read --bs=1M --size=1250M --iodepth=$qd --direct=1 --ioengine=libaio \
        --filename="$path/fio_probe" --output-format=json 2>/dev/null \
      | python3 -c "import sys,json;d=json.load(sys.stdin);r=d['jobs'][0]['read'];print(f'  QD$qd  {r[\"bw_bytes\"]/1e9:.2f} GB/s')" 2>/dev/null \
      || echo "  QD$qd  (fio failed — check path/perms)"
  done
  rm -f "$path/fio_probe" 2>/dev/null
}
[ -e "$SINGLE" ] && run_fio "single-drive" "$SINGLE" || echo "(TARGET_SINGLE $SINGLE absent — skip)"
[ -e "$STRIPE" ] && run_fio "raid0-stripe" "$STRIPE" || echo "(TARGET_STRIPE $STRIPE absent — skip)"
echo "=== 1-hour sustained read + smartctl temp every 30s (throttle curve) ==="
if [ "${THROTTLE:-0}" = 1 ] && [ -e "$STRIPE" ]; then
  ( fio --name=sustain --rw=read --bs=1M --size=200G --loops=1000 --direct=1 --ioengine=libaio \
        --runtime=3600 --time_based --filename="$STRIPE/fio_sustain" >/dev/null 2>&1 & echo $! > /tmp/fio.pid )
  DRV=$(lsblk -dn -o NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print "/dev/"$1; exit}')
  : > "$OUT/throttle.csv"; echo "t_s,temp_c,bw_note" >> "$OUT/throttle.csv"
  for t in $(seq 0 30 3600); do
    temp=$(smartctl -A "$DRV" 2>/dev/null | awk '/Temperature:/{print $2; exit}')
    echo "$t,$temp," >> "$OUT/throttle.csv"; sleep 30
    kill -0 $(cat /tmp/fio.pid 2>/dev/null) 2>/dev/null || break
  done
  rm -f "$STRIPE/fio_sustain"; echo "throttle curve -> $OUT/throttle.csv"
else
  echo "(set THROTTLE=1 to run the 1-hour sustained test)"
fi
echo "Record S1 (single QD1/2), S2 (stripe QD1/2) into knee_predictions.json stage 2."
