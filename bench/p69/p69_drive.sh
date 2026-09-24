#!/bin/bash
# bench/p69/p69_drive.sh -- lane P69, CONTROLLER side: the launcher's --command (bench/p69/P69-PREREG.md).
# Runs grouped-nf4-gemm's bench/calibrate.py (at the pinned GNF4_SHA) twice on a rented RTX 5090, GPU benches only,
# and fetches both schema-/2 blobs back. Nothing is installed on the box; nothing here creates, destroys or approves
# compute -- the launcher owns attach, pre-flight, receipt, ledger row and teardown.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p69_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_INSTANCE_ID GNF4_SHA; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight, with GNF4_SHA pinned"; exit 78; }
done
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; OUT="$E4B_RENT_RUN_DIR/p69"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
RAW="https://raw.githubusercontent.com/pjordanandrsn/grouped-nf4-gemm/$GNF4_SHA/bench/calibrate.py"
mkdir -p "$OUT/run1" "$OUT/run2" || { say "cannot create $OUT"; exit 20; }

# 1. the pinned script, by commit, with its hash in the receipt
curl -fsSL --retry 3 -o "$OUT/calibrate.py" "$RAW" || { say "fetch of $RAW failed"; exit 20; }
grep -q "h2d_64mb_single" "$OUT/calibrate.py" || { say "refusing: the pinned calibrate.py has no single-copy probe (pre-#402 commit?)"; exit 20; }
(sha256sum "$OUT/calibrate.py" 2>/dev/null || shasum -a 256 "$OUT/calibrate.py") | cut -d" " -f1 > "$OUT/calibrate.sha256" || exit 20
say "calibrate.py @ $GNF4_SHA sha256 $(cat "$OUT/calibrate.sha256")"

# 2. box facts, and the card-class refusal (the registration is about a gen 4 x16 RTX 5090)
$SSH 'nvidia-smi --query-gpu=name,driver_version,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit --format=csv,noheader; echo "---CPU"; lscpu | grep -E "^Model name|^CPU\(s\):|^NUMA node\(s\)"; echo "---TORCH"; python3 -c "import torch;print(torch.__version__, torch.version.cuda)"' \
  > "$OUT/box.txt" 2>&1 || { say "ssh to the box failed -- nothing ran"; cat "$OUT/box.txt"; exit 21; }
cat "$OUT/box.txt"
grep -q "RTX 5090" "$OUT/box.txt" || { say "refusing: the card is not an RTX 5090"; exit 15; }

# 3. stage and run twice, GPU benches only
$SSH 'mkdir -p /root/p69/run1 /root/p69/run2' && $SCP "$OUT/calibrate.py" "root@$HOST:/root/p69/calibrate.py" || { say "stage failed"; exit 20; }
for r in 1 2; do
  say "run $r"
  $SSH "cd /root/p69/run$r && python3 /root/p69/calibrate.py --skip-cpu --tag 5090-p69-run$r --out /root/p69/run$r/calib.json" > "$OUT/run$r/stdout.txt" 2>&1 \
    || { say "run $r failed"; tail -5 "$OUT/run$r/stdout.txt"; exit 22; }
  tail -3 "$OUT/run$r/stdout.txt"
  $SCP "root@$HOST:/root/p69/run$r/calib.json" "$OUT/run$r/calib.json" || { say "fetch-back of run $r failed"; exit 22; }
done

# 4. the read: the three numbers per run, their ratio capped at 1, and the spread
python3 - "$OUT" <<'PY' | tee "$OUT/read.txt"
import json, sys
out = sys.argv[1]
rows = {}
for r in (1, 2):
    b = json.load(open(f"{out}/run{r}/calib.json"))
    l = b["gpu_bench"]["devices"][0]["b_link"]
    bb, single = l["h2d_64mb"]["gbs"], l["h2d_64mb_single"]["gbs"]
    rows[r] = {"schema": b["schema"], "h2d_64mb": bb, "h2d_64mb_single": single, "link_eff": min(1.0, single / bb)}
    print(f"run{r}: schema {b['schema']}  back-to-back {bb} GB/s  single {single} GB/s  link_eff {rows[r]['link_eff']:.3f}")
spread = {k: abs(rows[1][k] - rows[2][k]) / max(rows[1][k], rows[2][k]) for k in ("h2d_64mb", "h2d_64mb_single", "link_eff")}
print("spread run1 vs run2:", {k: f"{v:.1%}" for k, v in spread.items()})
json.dump({"runs": rows, "spread": spread}, open(f"{out}/read.json", "w"), indent=1)
PY
say "P69 complete: both blobs in $OUT (the read is P69-PREREG's; the claim and receipt land in grouped-nf4-gemm)"
exit 0
