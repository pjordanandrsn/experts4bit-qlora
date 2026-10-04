#!/bin/bash
# bench/k29/k29_run.sh -- lane K29, BOX side. What a page-locked host byte costs this container's cgroup (grouped-nf4-gemm
# #71, kernel/PREREG-k29-pinned-charge-cgroup-v2.md). Forensics, the cgroup v2 check (STOP-1), then the probe:
# pinned and pageable allocations at the registered sizes, each in a fresh process, read as the cgroup's own charge.
# Nothing is installed -- the probe needs only the image's torch.
set -uo pipefail
: "${K29_RUN_ID:?}"; : "${K29_RUN_NONCE:?}"; : "${K29_DEADLINE_EPOCH:?}"
W=$(cd "$(dirname "$0")" && pwd); cd "$W" || exit 20
mkdir -p logs
NONCE=$K29_RUN_NONCE
echo "$NONCE" > K29_RUN_NONCE                  # the controller's start handshake
say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a summary.txt; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "K29_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "K29_SUCCESS.$NONCE"; say "TP_DONE rc=$rc"; : > "TP_DONE.$NONCE"; exit "$rc"; }
trap 'finish 130' INT TERM
SIZES=${K29_SIZES:-0,340,512,700,1024,1359,2048,2100,3000,4096}
REPS=${K29_REPS:-2}
say "K29 $K29_RUN_ID: sizes $SIZES MB x $REPS reps, pinned and pageable"

{
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
  echo "kernel: $(uname -r)"
  echo "cgroup mounts:"; grep -E " cgroup2? " /proc/mounts 2>/dev/null | sed 's/^/  /'
  echo "proc_self_cgroup: $(tr '\n' ' ' < /proc/self/cgroup 2>/dev/null)"
  for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    [ -r "$f" ] && echo "$f: $(cat "$f")"
  done
  grep -E '^(MemTotal|MemAvailable)' /proc/meminfo 2>/dev/null
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1
} > forensics.txt 2>&1
cat forensics.txt

# STOP-1: the class is cgroup v2. A v1 box is recorded and the probe still runs (its rows are free evidence and the
# rehearsal's VOID path), but the reducer gives no verdict.
if grep -qE " cgroup2 " /proc/mounts && [ -r /sys/fs/cgroup/memory.current ]; then
  echo v2 > cgroup_version.txt; say "cgroup v2: memory.current readable (class drawn)"
else
  echo v1-or-none > cgroup_version.txt; say "STOP-1: not cgroup v2 -- the probe runs, the verdict is VOID"
fi

if [ "$(date +%s)" -ge $(( K29_DEADLINE_EPOCH - 300 )) ]; then say "STOP: under 5 min of guard left"; finish 40; fi
PROBE_S=$(( K29_DEADLINE_EPOCH - $(date +%s) - 180 ))
perl -e "alarm $PROBE_S; exec @ARGV" python3 pinned_charge_probe.py --sizes "$SIZES" --reps "$REPS" --out k29.json > logs/probe.log 2>&1
rc=$?
tail -2 logs/probe.log | tee -a summary.txt
[ "$rc" = 0 ] && [ -s k29.json ] || { say "HARNESS: the probe exited $rc without k29.json -- no verdict"; finish 41; }
say "probe done"
finish 0
