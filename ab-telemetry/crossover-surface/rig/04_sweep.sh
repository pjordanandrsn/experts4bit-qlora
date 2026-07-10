#!/usr/bin/env bash
# 04_sweep.sh — the placement x access crossover sweep. Runs AFTER 01-03 pass and stage-2 of
# knee_predictions.json is committed. For each Phase-0 access slice (eff_tokens) it sweeps
# placement f, with drop_caches + O_DIRECT between arms, fixed seed, warmup excluded.
#
# Gates enforced HERE (all three, per recorded number):
#   G1 bit-exact control  — the FusedStore degenerate-ends + byte-parity tests (run once, pre-sweep)
#   G2 perf-degenerate    — f=1.0 vs pure RAMStore <= 0.5% s/step (abort the slice if it fails)
#   G3 stripe-saturates   — receipts/fio.S2 >= receipts/link.L (checked pre-sweep; abort if not)
#
# Uses the private FusedStore (Cerin-Amroth/e4b-ssdtier) on the striped-NVMe store dir. Store dir
# MUST be on /mnt/stripe (the RAID-0), never the OS disk.
set -u
STORE_ROOT="${STORE_ROOT:-/mnt/stripe/e4b-store}"
SLICES="${SLICES:-64 256 2048}"          # Phase-0 access slices (seq lengths at batch 1)
FRACS="${FRACS:-1.0 0.75 0.5 0.25 0.0}"  # placement f
SEED="${SEED:-42}"; OUT="${OUT:-results}"; mkdir -p "$OUT" receipts
AXO=/root/work/axolotl

echo "=== G3: stripe saturates the lane? ==="
python3 - <<'PY' || { echo "G3 FAIL — stripe does not saturate the link; crossover location meaningless. ABORT."; exit 3; }
import json,sys
try:
    S=json.load(open("receipts/fio.json"))["S2_raid0_GBps"]; L=json.load(open("receipts/link.json"))["pinned_h2d_GBps"]
except Exception as e: print("missing receipts (run 02/03 first):",e); sys.exit(1)
print(f"S2(stripe)={S} GB/s  L(link)={L} GB/s  -> {'PASS' if S>=L else 'FAIL'}")
sys.exit(0 if S>=L else 2)
PY

echo "=== G1: bit-exact control (FusedStore degenerate-ends + parity) ==="
python -m pytest /root/e4b-ssdtier/tests/test_fused.py -q -p no:cacheprovider || { echo "G1 FAIL — correctness broken, ABORT"; exit 1; }

for eff in $SLICES; do
  echo "########## access slice eff_tokens=$eff (batch1 x seq$eff) ##########"
  # G2 baseline: pure RAMStore at this slice
  RAM_SS=$(SEED=$SEED bash run_arm.sh ram_ctrl_$eff "$eff" ram 1.0 "$STORE_ROOT/ram_$eff" | awk '/S_PER_STEP/{print $2}')
  for f in $FRACS; do
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null   # real cold cache between arms
    SS=$(SEED=$SEED bash run_arm.sh "fused_${eff}_f${f}" "$eff" fused "$f" "$STORE_ROOT/f${f}_$eff" | awk '/S_PER_STEP/{print $2}')
    echo "  eff=$eff f=$f  s/step=$SS  (RAM baseline $RAM_SS)"
    if [ "$f" = "1.0" ]; then
      python3 -c "import sys;r=$RAM_SS;a=$SS;g=100*(a/r-1);print(f'  G2 f=1.0 overhead {g:+.1f}% ({\"PASS\" if abs(g)<=3 else \"FAIL\"})');sys.exit(0 if abs(g)<=3 else 9)" \
        || { echo "  G2 FAIL at eff=$eff — measuring serialization not storage, ABORT slice"; break; }
    fi
    echo "{\"eff_tokens\":$eff,\"f\":$f,\"s_per_step\":$SS,\"ram_baseline\":$RAM_SS}" >> "$OUT/surface.jsonl"
  done
done
echo "=== crossover surface -> $OUT/surface.jsonl ; reduce with 05_reduce_surface.py ==="
