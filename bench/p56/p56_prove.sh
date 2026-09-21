#!/bin/bash
# bench/p56/p56_prove.sh -- lane P56's PROVING run (board motion 1, 2026-09-07: before any run whose guard
# exceeds an hour, rent one cheap box of the same provider class and image and prove the path end to end).
# Trivial in cost, deliberately not trivial in what it records.
#
# Two jobs:
#   1. Prove the launcher path on hardware -- attach, pre-flight, command handoff, receipt, ledger row,
#      teardown proof. The launcher owns all of that; this script only has to run and return.
#   2. MEASURE the quantity that decides whether P56's registered draw can run AT ALL on the box it draws,
#      and that nothing in the pre-flight looks at: HOST RAM against Gemma-4's single 49.9 GiB shard.
#
# Why that quantity. `google/gemma-4-26B-A4B-it` ships its weights as 49.9 GiB + 1.7 GiB, where every other
# family in the corpus shards at ~5 GiB, and `load_moe_4bit_streaming` maps the whole shard with
# `safe_open(device="cuda")`. e4b#344 records it failing on 2 of 6 rented 5090s with `CUDA error: invalid
# argument`, driver and GPU both refuted. The sibling lane's host fingerprint (bench/p55/P55-PREREG.md)
# orders every one of those six outcomes by host RAM: 30 GiB refuses the map outright, 64 GiB maps it and
# dies opaquely, 96 / 125 / 188 GiB pass.
#
# So a P56 draw on a low-RAM box loses all four arms to a load fault that has nothing to do with the
# question, and the $2.42 with them. This prints the number before that can happen.
#
# It does NOT refuse on it. A proving run that can only say "pass" and one that can only say "fail" look
# identical on a green result (P41, 2026-09-07) -- the number is the point, so the number is printed and
# recorded whatever it says, and the go/no-go on the registered draw is a human reading it.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p56_prove] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; OUT="$E4B_RENT_RUN_DIR/p56-prove"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -p $PORT root@$HOST"
# The shard, in GiB, and the headroom the fingerprint says separates a pass from a fault.
SHARD_GIB=${P56_SHARD_GIB:-49.9}; RAM_PASS_GIB=${P56_RAM_PASS_GIB:-96}
mkdir -p "$OUT" || { say "cannot create $OUT"; exit 20; }

say "box facts ($HOST:$PORT, instance $E4B_RENT_INSTANCE_ID)"
$SSH 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; echo "---MEM"; free -g | head -2; echo "---MEMINFO"; grep -E "^MemTotal|^MemAvailable" /proc/meminfo; echo "---CGROUP"; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "none"; echo "---DISK"; df -BG /root | tail -1; echo "---CPU"; lscpu | grep -E "^Model name|^CPU\(s\):"' \
  > "$OUT/box.txt" 2>&1 || { say "ssh to the box failed -- the path is NOT proven"; cat "$OUT/box.txt"; exit 21; }
cat "$OUT/box.txt"

# One CUDA op, so "the box works" is measured rather than assumed.
say "cuda smoke"
$SSH 'python3 -c "import torch;a=torch.ones(256,256,device=\"cuda\");print(\"cuda_ok\", float((a@a).sum().item()))" 2>&1 | tail -2' \
  > "$OUT/cuda.txt" 2>&1; cat "$OUT/cuda.txt"

python3 - "$OUT" "$SHARD_GIB" "$RAM_PASS_GIB" <<'PY' | tee "$OUT/verdict.txt"
import json, re, sys
out, shard, ram_pass = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
txt = open(out + "/box.txt").read()
m = re.search(r"^MemTotal:\s+(\d+) kB", txt, re.M)
avail = re.search(r"^MemAvailable:\s+(\d+) kB", txt, re.M)
ram = (int(m.group(1)) / 1048576.0) if m else None          # kB -> GiB
av = (int(avail.group(1)) / 1048576.0) if avail else None
cg = re.search(r"---CGROUP\n(\S+)", txt)
cg_gib = None
if cg and cg.group(1).isdigit():
    cg_gib = int(cg.group(1)) / (1024 ** 3)
eff = min([x for x in (ram, cg_gib) if x is not None], default=None)
rec = {"host_ram_gib": ram, "mem_available_gib": av, "cgroup_limit_gib": cg_gib,
       "effective_ram_gib": eff, "gemma4_shard_gib": shard,
       "headroom_gib": (eff - shard) if eff else None,
       "fingerprint_pass_floor_gib": ram_pass,
       "meets_fingerprint_pass_class": (eff is not None and eff >= ram_pass),
       "cuda_ok": "cuda_ok" in open(out + "/cuda.txt").read()}
print(json.dumps(rec, indent=2))
if eff is None:
    print("\nRAM NOT READABLE -- the registered draw's go/no-go cannot be informed by this run.")
elif rec["meets_fingerprint_pass_class"]:
    print(f"\nGO-CLASS: {eff:.0f} GiB effective RAM against a {shard} GiB shard "
          f"({eff - shard:.0f} GiB headroom); at or above every host in the fingerprint that PASSED.")
else:
    print(f"\nBELOW THE PASS CLASS: {eff:.0f} GiB effective RAM against a {shard} GiB shard. "
          f"Every fingerprint host at or under 64 GiB either refused the map or died inside it "
          f"(e4b#344). A P56 draw on this class is likely to lose all four arms to a load fault, "
          f"which is not P56's question. Draw a higher-RAM box.")
PY
say "proving run complete (path proven; the RAM reading above informs the registered draw, it does not gate it)"
exit 0
