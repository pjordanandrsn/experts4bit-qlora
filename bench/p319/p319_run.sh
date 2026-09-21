#!/bin/bash
# bench/p319/p319_run.sh -- lane P319, BOX side. Answers one question:
# does grouped-nf4-gemm#319 (the f32 paged-decode compute modes missing the
# reference) reproduce on sm_120 at v0.32.1, and if it does, which dot
# precision fixes it and what does that cost. Runs the suite at the RELEASE
# commit first, so "already fixed" and "fixed by the branch" cannot be
# confused. Creates, destroys and approves nothing.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p319_run] $*"; }
for v in P319_RUN_ID P319_RUN_NONCE P319_DEADLINE_EPOCH GNF4_BASE_SHA GNF4_HEAD_SHA; do
  [ -n "${!v:-}" ] || { say "refusing: $v unset"; exit 78; }
done
W=$(pwd); mkdir -p logs
# bind the nonce first: the controller's handshake proves THIS process started
printf '%s' "$P319_RUN_NONCE" > P319_RUN_NONCE

finish(){ rc=$1; say "finish rc=$rc"; printf '%s' "$rc" > P319_RC; touch "TP_DONE.$P319_RUN_NONCE"; exit "$rc"; }
deadline_left(){ echo $(( P319_DEADLINE_EPOCH - $(date -u +%s) )); }
[ "$(deadline_left)" -gt 300 ] || { say "refusing: under 5 min of clock left"; finish 78; }

{
  python -c 'import torch,triton
p=torch.cuda.get_device_properties(0)
print("device", p.name)
print("cap sm_%d%d" % torch.cuda.get_device_capability())
print("torch", torch.__version__, "triton", triton.__version__)
print("smem_optin", getattr(p, "shared_memory_per_block_optin", "?"))'
} > summary.txt 2>&1 || finish 9
say "$(tr '\n' ' ' < summary.txt)"

python -m pip install -q --no-input pytest > logs/pip.log 2>&1 || { tail -3 logs/pip.log; finish 9; }
git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git gnf4 > logs/clone.log 2>&1 \
  || { tail -3 logs/clone.log; finish 9; }

# ---- arm A: the RELEASE commit, untouched. This is the #319 claim's subject.
git -C gnf4 checkout -q "$GNF4_BASE_SHA" || finish 9
( cd gnf4/kernel && python -m pytest test_fp8_paged_attn.py -q --no-header \
    -p no:cacheprovider ) > logs/pytest_base.log 2>&1
rc_base=$?
{ echo; echo "== arm A: $GNF4_BASE_SHA (release, unmodified) =="; echo "pytest rc=$rc_base";
  grep -E "^(FAILED|ERROR)" logs/pytest_base.log | head -40; tail -3 logs/pytest_base.log; } >> summary.txt
say "arm A pytest rc=$rc_base"

# ---- arm B: the branch. Default precision is tf32, i.e. arm A's behaviour
# stated rather than inherited -- so B differing from A is itself a finding.
git -C gnf4 checkout -q "$GNF4_HEAD_SHA" || finish 9
( cd gnf4/kernel && python -m pytest test_fp8_paged_attn.py -q --no-header \
    -p no:cacheprovider ) > logs/pytest_head.log 2>&1
rc_head=$?
{ echo; echo "== arm B: $GNF4_HEAD_SHA (branch, default tf32) =="; echo "pytest rc=$rc_head";
  grep -E "^(FAILED|ERROR)" logs/pytest_head.log | head -40; tail -3 logs/pytest_head.log; } >> summary.txt
say "arm B pytest rc=$rc_head"

# ---- arm C: accuracy x precision, throughput x precision, and the PTX the
# compiler actually emits for each. The mechanism, not an inference from it.
cp "$W/p319_probe.py" gnf4/kernel/ || finish 9
( cd gnf4/kernel && python p319_probe.py ) > logs/probe.log 2>&1
rc_probe=$?
{ echo; echo "== arm C: precision sweep (rc=$rc_probe) =="; cat logs/probe.log; } >> summary.txt
say "arm C probe rc=$rc_probe"

# ---- arm D: the suite at each precision, so a precision that FIXES the
# failures is recorded as passing the real gate, not just a smaller error.
for prec in tf32x3 ieee; do
  ( cd gnf4/kernel && GNF4_ATTN_F32_PRECISION=$prec python -m pytest \
      test_fp8_paged_attn.py -q --no-header -p no:cacheprovider \
      -k "split or packed" ) > "logs/pytest_$prec.log" 2>&1
  rc=$?
  { echo; echo "== arm D: GNF4_ATTN_F32_PRECISION=$prec (f32 modes only) =="; echo "pytest rc=$rc";
    grep -E "^(FAILED|ERROR)" "logs/pytest_$prec.log" | head -20; tail -3 "logs/pytest_$prec.log"; } >> summary.txt
  say "arm D $prec pytest rc=$rc"
done

finish 0
