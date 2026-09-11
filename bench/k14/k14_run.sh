#!/bin/bash
# bench/k14/k14_run.sh -- lane K14, BOX side. The prereg, the bench and the results live in
# grouped-nf4-gemm (kernel/PREREG-k14-smallm-int4-gemm.md); only the runner is here, because
# pod-launch.sh pins exactly `adertha` and `e4b` by design and that allowlist is what makes
# "which code ran" answerable.
#
# No model, no calibration, no bake: install gnf4 at GNF4_SHA, clone the SAME sha for the
# harness (bench harnesses are deliberately unpackaged there), prove the two agree, run.
set -uo pipefail
W=/root/k14; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] k14: $*"; }
NONCE=${K14_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/K14_RUN_NONCE.tmp && mv $W/K14_RUN_NONCE.tmp $W/K14_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > K14_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > K14_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in K14_RUN_ID K14_DEADLINE_EPOCH K14_INSTANCE_ID GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char sha"; finish 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > summary.txt; echo "$K14_INSTANCE_ID" > INSTANCE_ID

python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt

say "install gnf4 @$GNF4_SHA"
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 \
  || { tail -3 logs/pip_gnf4.log; say "PIP FAIL"; finish 9; }
say "clone the same sha for the harness"
perl -e 'alarm 900; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git src > logs/clone.log 2>&1 \
  && git -C src checkout -q "$GNF4_SHA" || { tail -3 logs/clone.log; say "CLONE FAIL"; finish 9; }
GOT=$(git -C src rev-parse HEAD)
[ "$GOT" = "$GNF4_SHA" ] || { say "TRIPWIRE: clone is $GOT, pin is $GNF4_SHA"; finish 9; }
# The harness must judge the library that is INSTALLED, not the clone beside it: the clone
# supplies k14_bench.py only, and int4_b32 resolves through the installed package. Prove the
# installed module is the pinned cut before any number is produced.
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import importlib, inspect, os, sys
import int4_b32, int4_pack_ref  # noqa: F401
src = os.path.abspath("/root/k14/src/kernel/int4_b32.py")
got = os.path.abspath(inspect.getsourcefile(int4_b32))
assert got != src, f"int4_b32 resolved to the CLONE ({got}); the installed package must win"
for n in ("gemm_int4_b32_grouped_captured", "gemv_int4_b32", "quant_x_rows"):
    assert hasattr(int4_b32, n), f"installed gnf4 lacks {n}"
import torch, triton
open("/root/k14/versions.txt", "w").write(
    f"gnf4 {importlib.metadata.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\n"
    f"int4_b32 from {got}\ntorch {torch.__version__}\ntriton {triton.__version__}\n")
print("tripwire OK: int4_b32 from", got)
PYT
cat versions.txt | tee -a summary.txt

left=$(( K14_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 600 ] && { say "STOP-2: no time left"; finish 30; }
[ "$left" -gt 3600 ] && left=3600
say "bench (alarm ${left}s)"
perl -e "alarm $left; exec @ARGV" python src/kernel/k14_bench.py --out $W/k14_rows.json \
  --ms "${K14_MS:-1,4,8,16,32}" > logs/k14_bench.log 2>&1
rc=$?
tail -40 logs/k14_bench.log | tee -a summary.txt
[ "$rc" = 0 ] && [ -s "$W/k14_rows.json" ] || { say "BENCH rc=$rc (rows.json $( [ -s $W/k14_rows.json ] && echo present || echo ABSENT))"; finish "${rc:-20}"; }
say "----- summary -----"; cat summary.txt
finish 0
