#!/bin/bash
# bench/k30/k30_run.sh -- lane K30, BOX side. The prereg, the sweep, the correctness check and the rule live in
# grouped-nf4-gemm (kernel/PREREG-k30-splitk-r-term-l4.md, bench/int4/sk_sweep.py, bench/int4/k30_sk_check.py,
# bench/int4/k30_reduce.py); only this runner lives here, because pod-launch.sh pins exactly `adertha` and `e4b` by
# design. Pattern: bench/k20/k20_run.sh.
#
# No model, no checkpoint, no download: install gnf4 at GNF4_SHA (the K30 registration's commit), clone the SAME sha
# for the harness, prove the INSTALLED module is the pinned cut, copy the harness OUT of the clone and run it from the
# work dir so `import int4_b32` cannot resolve to the clone. Correctness first -- the compiled int4_b32 suite, the
# per-sk agreement check, the rule's self-test, the plan cross-check -- and no timing if any of them fails. Then the
# 48-cell sweep twice (k30p1, k30p2), then the reducer's verdict.
#
# K30_REHEARSAL=1 (the $0 A2000 rehearsal only): lifts the card check and sweeps two R values per shape. The reducer
# then reads VOID by construction (wrong card, missing cells), which is the rehearsal's expected verdict.
# K30_PROVE=1 (the proving rental): the card check and forensics, then exit 0 -- no install, no gate, no sweep. It
# proves attach, pre-flight, command handoff, receipt and teardown on the provider class before the reading.
set -uo pipefail
W=/root/k30; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] k30: $*"; }
NONCE=${K30_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/K30_RUN_NONCE.tmp && mv $W/K30_RUN_NONCE.tmp $W/K30_RUN_NONCE
SMI_PID=""
finish(){ local rc=$1; [ -n "$SMI_PID" ] && kill "$SMI_PID" 2>/dev/null; printf '%s\n' "$rc" > K30_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > K30_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in K30_RUN_ID K30_DEADLINE_EPOCH K30_INSTANCE_ID GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char sha"; finish 78; }
REHEARSAL=${K30_REHEARSAL:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > summary.txt; echo "$K30_INSTANCE_ID" > INSTANCE_ID

python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm,pcie.gen.max,pcie.width.max /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
SMS=$(python -c "import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)")
echo "card $GPU, $SMS SMs, rehearsal=$REHEARSAL" | tee -a summary.txt
# An exact match: "NVIDIA L40S" contains "L4", and a substring test would accept it.
if [ "$REHEARSAL" != 1 ]; then
  [ "$GPU" = "NVIDIA L4" ] || { say "REFUSED: card is '$GPU', the lane registers the NVIDIA L4"; echo "refused: class $GPU" > REFUSAL; finish 15; }
  [ "$SMS" -le 64 ] || { say "REFUSED: $SMS SMs is above the 64-SM class the R term is gated to"; echo "refused: $SMS SMs" > REFUSAL; finish 15; }
fi
[ "${K30_PROVE:-0}" = 1 ] && { say "PROVE: card and forensics recorded; no install, no timing"; echo "prove ok" >> summary.txt; finish 0; }

command -v git >/dev/null || { perl -e 'alarm 300; exec @ARGV' apt-get install -y -q git > logs/apt_git.log 2>&1 || { say "NO GIT"; finish 9; }; }
command -v gcc >/dev/null || { say "NO C COMPILER (Triton needs one): the image must be a -devel image"; finish 9; }
say "install gnf4 @$GNF4_SHA"
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 \
  || { tail -3 logs/pip_gnf4.log; say "PIP FAIL"; finish 9; }
say "clone the same sha for the harness"
perl -e 'alarm 900; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git src > logs/clone.log 2>&1 \
  && git -C src checkout -q "$GNF4_SHA" || { tail -3 logs/clone.log; say "CLONE FAIL"; finish 9; }
GOT=$(git -C src rev-parse HEAD)
[ "$GOT" = "$GNF4_SHA" ] || { say "TRIPWIRE: clone is $GOT, pin is $GNF4_SHA"; finish 9; }
for f in sk_sweep.py k30_sk_check.py k30_reduce.py; do cp "src/bench/int4/$f" $W/ || { say "harness file $f missing in the clone (not the K30 cut)"; finish 9; }; done
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import importlib.metadata, inspect, os
import int4_b32, int4_pack_ref  # noqa: F401
src = os.path.abspath("/root/k30/src/kernel/int4_b32.py")
got = os.path.abspath(inspect.getsourcefile(int4_b32))
assert got != src, f"int4_b32 resolved to the CLONE ({got}); the installed package must win"
for n in ("_plan", "_gemv_int4_b32", "reduce_partials", "SPLITK_R_TERM_MAX_SMS", "SPLITK_R_FLOOR", "SPLITK_TARGET_BLOCKS_PER_SM"):
    assert hasattr(int4_b32, n), f"installed gnf4 lacks {n} (not the K30 cut)"
assert "R" in inspect.signature(int4_b32._plan).parameters, "the installed _plan takes no R: not the cut K30 measures"
import torch, triton
open("/root/k30/versions.txt", "w").write(
    f"gnf4 {importlib.metadata.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\n"
    f"int4_b32 from {got}\ntorch {torch.__version__}\ntriton {triton.__version__}\n"
    f"SPLITK_R_TERM_MAX_SMS {int4_b32.SPLITK_R_TERM_MAX_SMS} SPLITK_R_FLOOR {int4_b32.SPLITK_R_FLOOR} "
    f"SPLITK_TARGET_BLOCKS_PER_SM {int4_b32.SPLITK_TARGET_BLOCKS_PER_SM}\n")
print("tripwire OK: int4_b32 from", got)
PYT
cat versions.txt | tee -a summary.txt
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/k30/staged.sha256"; finish 9; }

left(){ echo $(( K30_DEADLINE_EPOCH - $(date +%s) - 600 )); }
[ "$(left)" -ge 600 ] || { say "STOP: no time left"; finish 30; }
# The vast image has no pytest (k16-5090 run 1); install it explicitly, own log, own code.
python -c "import pytest" 2>/dev/null || perl -e 'alarm 300; exec @ARGV' python -m pip install -q --no-input pytest > logs/pip_pytest.log 2>&1 \
  || { tail -3 logs/pip_pytest.log; say "PYTEST INSTALL FAIL"; finish 9; }
say "correctness 1/4: the int4_b32 GEMV, reduce and plan tests compiled on this card, alarm 900"
TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest src/kernel/test_int4_b32.py -q -x -p no:cacheprovider -k "gemv_matches_reference or reduce_partials or plan_" > logs/tests_compiled.log 2>&1; crc=$?
tail -3 logs/tests_compiled.log | tee -a summary.txt
[ "$crc" = 0 ] || { say "COMPILED CONTRACT FAIL rc=$crc -- no perf number is produced"; finish 21; }
say "correctness 2/4: every swept sk agrees with sk=1, alarm 900"
perl -e 'alarm 900; exec @ARGV' python $W/k30_sk_check.py > logs/sk_check.log 2>&1; krc=$?
tail -1 logs/sk_check.log | tee -a summary.txt
[ "$krc" = 0 ] || { say "SK CHECK FAIL rc=$krc -- no perf number is produced"; finish 22; }
say "correctness 3/4: the rule's self-test"
python $W/k30_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "RULE SELF-TEST FAILED"; finish 21; }
say "correctness 4/4: the installed _plan picks what the reducer prices"
(cd $W && python -c "import k30_reduce; k30_reduce.check_installed_plan()") | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "PLAN CROSS-CHECK FAILED"; finish 21; }

# the card's clocks and power through the sweep: a power-capped 72 W part is a forensics column, not a filter
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 5 > logs/smi.csv 2>&1 &
SMI_PID=$!
[ "$REHEARSAL" = 1 ] && export RS=16,128
mkdir -p $W/rows
for pass in k30p1 k30p2; do
  need=${K30_PASS_NEED_S:-1500}
  [ "$(left)" -ge "$need" ] || { say "STOP-2: pass $pass needs ${need}s, only $(left)s left -- skipped (host-limited)"; echo "SKIPPED $pass host-limited deadline" >> summary.txt; break; }
  for fam in qwen3_moe granitemoe olmoe; do
    for proj in gate_up down; do
      a=$(left); [ "$a" -gt 900 ] && a=900; [ "$a" -lt 60 ] && { say "STOP-2: no time for $pass $fam $proj"; break 3; }
      say "sweep $pass $fam $proj (alarm ${a}s)"
      OUT_PREFIX=$pass perl -e "alarm $a; exec @ARGV" python $W/sk_sweep.py $fam $proj > logs/sweep_${pass}_${fam}_${proj}.log 2>&1; src=$?
      [ "$src" = 0 ] && [ -s "$W/rows/${pass}_${fam}_${proj}.json" ] || { say "SWEEP FAIL $pass $fam $proj rc=$src"; echo "SWEEP FAIL $pass $fam $proj rc=$src" >> summary.txt; finish 20; }
      echo "swept $pass $fam $proj" >> summary.txt
    done
  done
done
kill "$SMI_PID" 2>/dev/null; SMI_PID=""
say "reduce"
python $W/k30_reduce.py $W/rows $W/k30_verdict.json --check-installed-plan > logs/reduce.log 2>&1; rrc=$?
grep -aE '^K30' logs/reduce.log | tee -a summary.txt
say "----- summary -----"; cat summary.txt
[ "$REHEARSAL" = 1 ] && { grep -q 'K30_VERDICT VOID' logs/reduce.log && { say "rehearsal: VOID as it must be off the target card"; finish 0; } || { say "rehearsal: reducer did not say VOID"; finish 31; }; }
[ "$rrc" = 0 ] || { say "REDUCER rc=$rrc (VOID or error)"; finish "$rrc"; }
finish 0
