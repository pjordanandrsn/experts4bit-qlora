#!/bin/bash
# bench/k18/k18_run.sh -- lane K18, BOX side. The prereg, the kernel, the bench and the results live in
# grouped-nf4-gemm (kernel/PREREG-k18-grouped-expert-gemv.md, kernel/int4_b32.py, kernel/k18_bench.py); the
# runner and the recorded routing it replays live here (bench/p60/receipts/eids_b16.int16.{bin,json}, staged by
# the driver), because pod-launch.sh pins exactly `adertha` and `e4b` by design. Pattern: bench/k17/k17_run.sh.
#
# No model, no calibration, no bake: install gnf4 at GNF4_SHA (a branch commit -- the K18 kernel is not on a
# release), clone the SAME sha for the harness, prove the INSTALLED module is the pinned cut, copy the two bench
# files OUT of the clone and run them from the work dir, so `import int4_b32` cannot resolve to the clone.
set -uo pipefail
W=/root/k18; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] k18: $*"; }
NONCE=${K18_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/K18_RUN_NONCE.tmp && mv $W/K18_RUN_NONCE.tmp $W/K18_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > K18_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > K18_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in K18_RUN_ID K18_DEADLINE_EPOCH K18_INSTANCE_ID GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char sha"; finish 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > summary.txt; echo "$K18_INSTANCE_ID" > INSTANCE_ID

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
cp src/kernel/k18_bench.py $W/ || { say "bench files missing in the clone"; finish 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import importlib.metadata, inspect, os
import int4_b32, int4_pack_ref  # noqa: F401
src = os.path.abspath("/root/k18/src/kernel/int4_b32.py")
got = os.path.abspath(inspect.getsourcefile(int4_b32))
assert got != src, f"int4_b32 resolved to the CLONE ({got}); the installed package must win"
for n in ("gemv_int4_b32", "gemv_int4_b32_grouped", "_gemv_int4_b32_grouped"):
    assert hasattr(int4_b32, n), f"installed gnf4 lacks {n} (not the K18 cut)"
assert int4_b32.gemv_fused_reduce_default() is False, "the fused reduce is defaulted on: the served arm would not be the served GEMV"
import torch, triton
open("/root/k18/versions.txt", "w").write(
    f"gnf4 {importlib.metadata.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\n"
    f"int4_b32 from {got}\ntorch {torch.__version__}\ntriton {triton.__version__}\n")
print("tripwire OK: int4_b32 from", got)
PYT
cat versions.txt | tee -a summary.txt

left=$(( K18_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 600 ] && { say "STOP: no time left"; finish 30; }
[ "$left" -gt 2400 ] && left=2400
# The correctness suite is pytest; the vast image has none (k16-5090 run 1: `No module named pytest`,
# rc=21 with a healthy card -- a harness fault, not a kernel one). Install it explicitly, own log, own code.
python -c "import pytest" 2>/dev/null || perl -e 'alarm 300; exec @ARGV' python -m pip install -q --no-input pytest > logs/pip_pytest.log 2>&1 \
  || { tail -3 logs/pip_pytest.log; say "PYTEST INSTALL FAIL"; finish 9; }
say "correctness first: the interpreter contract on CPU (P1 on the small cases), alarm 900"
# Two processes, not one: at the K18 sha conftest lists test_int4_b32_grouped_interp.py in _INTERP_FILES and
# test_int4_b32.py is not, so on a box WITH a device the pair in one pytest is refused as a mixed session
# (UsageError, rc 4) before a test runs. K17's sha predated its own file's listing, which is why its pair ran.
TRITON_INTERPRET=1 perl -e 'alarm 900; exec @ARGV' python -m pytest src/kernel/test_int4_b32_grouped_interp.py -q -x -p no:cacheprovider > logs/tests_interp.log 2>&1; irc=$?
tail -3 logs/tests_interp.log | tee -a summary.txt
if [ "$irc" = 0 ]; then
  TRITON_INTERPRET=1 perl -e 'alarm 900; exec @ARGV' python -m pytest src/kernel/test_int4_b32.py -q -x -p no:cacheprovider > logs/tests_interp_served.log 2>&1; irc=$?
  tail -3 logs/tests_interp_served.log | tee -a summary.txt
fi
[ "$irc" = 0 ] || { say "INTERPRETER CONTRACT FAIL rc=$irc -- no perf number is produced"; finish 21; }
say "then the compiled contract on this card (P1 incl. the real expert shapes at R=128 and a graph capture), alarm 900"
TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest src/kernel/test_int4_b32_grouped_interp.py -q -x -p no:cacheprovider > logs/tests_compiled.log 2>&1; crc=$?
tail -3 logs/tests_compiled.log | tee -a summary.txt
[ "$crc" = 0 ] || { say "COMPILED CONTRACT FAIL rc=$crc -- no perf number is produced"; finish 22; }
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/k18/staged.sha256"; finish 9; }
say "bench (alarm ${left}s): P2 replay of P60's recorded ids (served / grouped / dedup) + P3 small R"
perl -e "alarm $left; exec @ARGV" python $W/k18_bench.py $W/eids_b16.int16.bin $W/eids_b16.int16.json $W/k18_rows.json > logs/k18_bench.log 2>&1
rc=$?
tail -16 logs/k18_bench.log | tee -a summary.txt
[ "$rc" = 0 ] && [ -s "$W/k18_rows.json" ] || { say "BENCH rc=$rc (rows.json $( [ -s $W/k18_rows.json ] && echo present || echo ABSENT))"; finish "${rc:-20}"; }
say "----- summary -----"; cat summary.txt
finish 0
