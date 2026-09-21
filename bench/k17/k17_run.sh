#!/bin/bash
# bench/k17/k17_run.sh -- lane K17, BOX side. The prereg, the kernel, the bench and the results live in
# grouped-nf4-gemm (kernel/PREREG-k17-fused-splitk-gemv.md, kernel/int4_b32.py, kernel/k17_bench.py); only
# the runner is here, because pod-launch.sh pins exactly `adertha` and `e4b` by design. Pattern: bench/k14/k14_run.sh.
#
# No model, no calibration, no bake: install gnf4 at GNF4_SHA (a branch commit -- the K17 kernel is not on a
# release), clone the SAME sha for the harness, prove the INSTALLED module is the pinned cut, copy the two bench
# files OUT of the clone and run them from the work dir, so `import int4_b32` cannot resolve to the clone.
set -uo pipefail
W=/root/k17; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] k17: $*"; }
NONCE=${K17_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/K17_RUN_NONCE.tmp && mv $W/K17_RUN_NONCE.tmp $W/K17_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > K17_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > K17_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in K17_RUN_ID K17_DEADLINE_EPOCH K17_INSTANCE_ID GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char sha"; finish 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > summary.txt; echo "$K17_INSTANCE_ID" > INSTANCE_ID

python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *5090*) ;; *) say "REFUSED: card is '$GPU', the lane registers the RTX 5090 class"; echo "refused: class $GPU" > REFUSAL; finish 15;; esac

say "install gnf4 @$GNF4_SHA"
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 \
  || { tail -3 logs/pip_gnf4.log; say "PIP FAIL"; finish 9; }
say "clone the same sha for the harness"
perl -e 'alarm 900; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git src > logs/clone.log 2>&1 \
  && git -C src checkout -q "$GNF4_SHA" || { tail -3 logs/clone.log; say "CLONE FAIL"; finish 9; }
GOT=$(git -C src rev-parse HEAD)
[ "$GOT" = "$GNF4_SHA" ] || { say "TRIPWIRE: clone is $GOT, pin is $GNF4_SHA"; finish 9; }
cp src/kernel/k17_bench.py src/kernel/k14_bench.py $W/ || { say "bench files missing in the clone"; finish 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import importlib.metadata, inspect, os
import int4_b32, int4_pack_ref  # noqa: F401
src = os.path.abspath("/root/k17/src/kernel/int4_b32.py")
got = os.path.abspath(inspect.getsourcefile(int4_b32))
assert got != src, f"int4_b32 resolved to the CLONE ({got}); the installed package must win"
for n in ("gemv_int4_b32", "gemv_counter_len", "gemv_fused_reduce_default"):
    assert hasattr(int4_b32, n), f"installed gnf4 lacks {n} (not the K17 cut)"
assert "fused_reduce" in inspect.signature(int4_b32.gemv_int4_b32).parameters, "gemv_int4_b32 has no fused_reduce: not the K17 cut"
import torch, triton
open("/root/k17/versions.txt", "w").write(
    f"gnf4 {importlib.metadata.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\n"
    f"int4_b32 from {got}\ntorch {torch.__version__}\ntriton {triton.__version__}\n")
print("tripwire OK: int4_b32 from", got)
PYT
cat versions.txt | tee -a summary.txt

left=$(( K17_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 600 ] && { say "STOP: no time left"; finish 30; }
[ "$left" -gt 2400 ] && left=2400
# The correctness suite is pytest; the vast image has none (k16-5090 run 1: `No module named pytest`,
# rc=21 with a healthy card -- a harness fault, not a kernel one). Install it explicitly, own log, own code.
python -c "import pytest" 2>/dev/null || perl -e 'alarm 300; exec @ARGV' python -m pip install -q --no-input pytest > logs/pip_pytest.log 2>&1 \
  || { tail -3 logs/pip_pytest.log; say "PYTEST INSTALL FAIL"; finish 9; }
say "correctness first: the interpreter contract on CPU (P1 on the small cases), alarm 900"
TRITON_INTERPRET=1 perl -e 'alarm 900; exec @ARGV' python -m pytest src/kernel/test_int4_b32_fused_reduce_interp.py src/kernel/test_int4_b32.py -q -x -p no:cacheprovider > logs/tests_interp.log 2>&1; irc=$?
tail -3 logs/tests_interp.log | tee -a summary.txt
[ "$irc" = 0 ] || { say "INTERPRETER CONTRACT FAIL rc=$irc -- no perf number is produced"; finish 21; }
say "then the compiled contract on this card (P1 on the six registered shapes x four row counts), alarm 900"
TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest src/kernel/test_int4_b32_fused_reduce_interp.py -q -x -p no:cacheprovider > logs/tests_compiled.log 2>&1; crc=$?
tail -3 logs/tests_compiled.log | tee -a summary.txt
[ "$crc" = 0 ] || { say "COMPILED CONTRACT FAIL rc=$crc -- no perf number is produced"; finish 22; }
say "bench (alarm ${left}s): two-launch vs fused, six shapes x four row counts, K14's instrument"
perl -e "alarm $left; exec @ARGV" python $W/k17_bench.py $W/k17_rows.json > logs/k17_bench.log 2>&1
rc=$?
tail -12 logs/k17_bench.log | tee -a summary.txt
[ "$rc" = 0 ] && [ -s "$W/k17_rows.json" ] || { say "BENCH rc=$rc (rows.json $( [ -s $W/k17_rows.json ] && echo present || echo ABSENT))"; finish "${rc:-20}"; }
say "----- summary -----"; cat summary.txt
finish 0
