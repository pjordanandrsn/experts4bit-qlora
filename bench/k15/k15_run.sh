#!/bin/bash
# bench/k15/k15_run.sh -- lane K14, BOX side. The prereg, the bench and the results live in
# grouped-nf4-gemm (kernel/PREREG-k15-marlin-comparator.md); only the runner is here, because
# pod-launch.sh pins exactly `adertha` and `e4b` by design and that allowlist is what makes
# "which code ran" answerable.
#
# No model, no calibration, no bake: install gnf4 at GNF4_SHA, clone the SAME sha for the
# harness (bench harnesses are deliberately unpackaged there), prove the two agree, run.
set -uo pipefail
W=/root/k15; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] k15: $*"; }
NONCE=${K15_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/K15_RUN_NONCE.tmp && mv $W/K15_RUN_NONCE.tmp $W/K15_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > K15_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > K15_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in K15_RUN_ID K15_DEADLINE_EPOCH K15_INSTANCE_ID GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char sha"; finish 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > summary.txt; echo "$K15_INSTANCE_ID" > INSTANCE_ID

python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
# The comparator is vLLM 0.28.0, whose wheel is built against torch 2.13.0+cu130 and
# REFUSES to initialise on a driver older than CUDA 13.0 ("The NVIDIA driver on your
# system is too old (found version 12090)"). P37 measured vLLM on driver 595.84; this
# lane drew a 12.9 box and discovered it only AFTER installing gnf4, running our whole
# side and pulling ~4 minutes of vLLM wheels (k15-marlin-3, rc=21, $0.16). Checked here
# instead, before anything is installed, so a wrong draw costs seconds.
CUDA_MAJOR=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\)\..*/\1/p' | head -1)
echo "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1), CUDA ${CUDA_MAJOR:-?}" | tee -a summary.txt
case "$CUDA_MAJOR" in
  ''|*[!0-9]*) say "cannot read the driver CUDA version -- refusing rather than guessing"; finish 30;; 
esac
if [ "$CUDA_MAJOR" -lt 13 ]; then
  say "HOST-LIMITED: driver exposes CUDA $CUDA_MAJOR, vLLM 0.28.0 needs 13.0 -- this box cannot run the comparator; draw another"
  echo "SKIPPED host-limited driver CUDA $CUDA_MAJOR < 13" >> summary.txt
  finish 30
fi
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
src = os.path.abspath("/root/k15/src/kernel/int4_b32.py")
got = os.path.abspath(inspect.getsourcefile(int4_b32))
assert got != src, f"int4_b32 resolved to the CLONE ({got}); the installed package must win"
for n in ("gemm_int4_b32_grouped_captured", "gemv_int4_b32", "quant_x_rows"):
    assert hasattr(int4_b32, n), f"installed gnf4 lacks {n}"
import torch, triton
open("/root/k15/versions.txt", "w").write(
    f"gnf4 {importlib.metadata.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\n"
    f"int4_b32 from {got}\ntorch {torch.__version__}\ntriton {triton.__version__}\n")
print("tripwire OK: int4_b32 from", got)
PYT
cat versions.txt | tee -a summary.txt

left=$(( K15_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 600 ] && { say "STOP-2: no time left"; finish 30; }
[ "$left" -gt 3600 ] && left=3600
# --- side A: OUR stack, in the image python (torch 2.8.0+cu128)
say "ours: k14_bench (alarm ${left}s)"
perl -e "alarm $left; exec @ARGV" python src/kernel/k14_bench.py --out $W/k14_rows_same_box.json \
  --ms "${K15_MS:-1,4,8,16,32}" > logs/k14_bench.log 2>&1
rc_ours=$?
tail -25 logs/k14_bench.log | tee -a summary.txt
[ "$rc_ours" = 0 ] && [ -s "$W/k14_rows_same_box.json" ] || say "OURS rc=$rc_ours"

# --- side B: vLLM 0.28.0 in its OWN venv. It brings torch 2.13.0+cu130 and would
# otherwise replace the torch our kernels were measured on -- P37's two-venv shape.
say "theirs: vllm venv"
# `--system-site-packages=false` is not a thing: it is a store_true flag, so argparse
# refused with "ignored explicit argument 'false'" and the venv was never created. A
# plain `python -m venv` already excludes system site-packages, which is the whole
# point here -- the vLLM side must not see the torch our side was measured on.
python -m venv $W/venv_vllm > logs/venv.log 2>&1 || {
  tail -5 logs/venv.log; say "VENV FAIL (see logs/venv.log)"; finish 9; }
[ -x $W/venv_vllm/bin/pip ] || { say "VENV FAIL: no pip in $W/venv_vllm"; finish 9; }
$W/venv_vllm/bin/python -c "import sys; assert sys.prefix != sys.base_prefix, 'not isolated'" \
  || { say "VENV FAIL: not isolated from the image python"; finish 9; }
perl -e 'alarm 2400; exec @ARGV' $W/venv_vllm/bin/pip install -q --no-input "vllm==0.28.0" > logs/pip_vllm.log 2>&1 \
  || { tail -5 logs/pip_vllm.log; say "PIP FAIL (vllm)"; finish 9; }
$W/venv_vllm/bin/python -c "
import torch, vllm, json
open('$W/versions_vllm.txt','w').write(f'vllm {vllm.__version__}\ntorch {torch.__version__}\n')
print('vllm', vllm.__version__, 'torch', torch.__version__)" | tee -a summary.txt \
  || { say "VLLM IMPORT FAIL"; finish 9; }
left2=$(( K15_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left2" -lt 300 ] && { say "STOP: no time for side B"; finish 30; }
[ "$left2" -gt 3600 ] && left2=3600
# Two passes. vLLM's own kernel selector logs that use_atomic_add helps for small
# size_n -- exactly k_proj/v_proj at N=512 -- so measuring only the default would
# understate the comparator on half the shapes.
rc_theirs=0
for pass_ in default atomic; do
  [ "$pass_" = atomic ] && export VLLM_MARLIN_USE_ATOMIC_ADD=1 || unset VLLM_MARLIN_USE_ATOMIC_ADD
  left2=$(( K15_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left2" -lt 240 ] && { say "STOP: no time for the $pass_ pass"; break; }
  [ "$left2" -gt 1800 ] && left2=1800
  say "theirs: marlin ($pass_ pass, alarm ${left2}s)"
  perl -e "alarm $left2; exec @ARGV" $W/venv_vllm/bin/python src/kernel/k15_marlin_bench.py \
    --out $W/k15_marlin_rows_$pass_.json --ms "${K15_MS:-1,4,8,16,32}" --groups "${K15_GROUPS:-128,32}" \
    > logs/k15_marlin_bench_$pass_.log 2>&1
  prc=$?
  tail -24 logs/k15_marlin_bench_$pass_.log | tee -a summary.txt
  [ "$prc" = 0 ] && [ -s "$W/k15_marlin_rows_$pass_.json" ] || { say "marlin $pass_ pass rc=$prc"; rc_theirs=1; }
done
unset VLLM_MARLIN_USE_ATOMIC_ADD

rc=0
[ "$rc_ours" = 0 ] && [ -s "$W/k14_rows_same_box.json" ] || rc=20
[ "$rc_theirs" = 0 ] && [ -s "$W/k15_marlin_rows_default.json" ] || rc=21
[ "$rc" = 0 ] || { say "BENCH ours=$rc_ours theirs=$rc_theirs"; finish "$rc"; }
say "----- summary -----"; cat summary.txt
finish 0
