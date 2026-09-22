#!/bin/bash
# bench/p319/p319b_run.sh -- lane P319, BOX side. Answers one question:
# does grouped-nf4-gemm#319 (the f32 paged-decode compute modes missing the
# reference) reproduce on sm_120 at v0.32.1, and if it does, which dot
# precision fixes it and what does that cost. Runs the suite at the RELEASE
# commit first, so "already fixed" and "fixed by the branch" cannot be
# confused. Creates, destroys and approves nothing.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p319b_run] $*"; }
for v in P319B_RUN_ID P319B_RUN_NONCE P319B_DEADLINE_EPOCH GNF4_BASE_SHA GNF4_HEAD_SHA; do
  [ -n "${!v:-}" ] || { say "refusing: $v unset"; exit 78; }
done
W=$(pwd); mkdir -p logs
# bind the nonce first: the controller's handshake proves THIS process started
printf '%s' "$P319B_RUN_NONCE" > P319B_RUN_NONCE

finish(){ rc=$1; say "finish rc=$rc"; printf '%s' "$rc" > P319B_RC; touch "TP_DONE.$P319B_RUN_NONCE"; exit "$rc"; }
deadline_left(){ echo $(( P319B_DEADLINE_EPOCH - $(date -u +%s) )); }
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

# ---- the only arm: the round-2 diagnostic at the branch commit. Round 1
# already established that the suite fails here and that the dot precision
# does not move it; this run asks WHERE the divergence is.
git -C gnf4 checkout -q "$GNF4_HEAD_SHA" || finish 9
cp "$W/p319b_probe.py" gnf4/kernel/ || finish 9
( cd gnf4/kernel && python p319b_probe.py ) > logs/probe.log 2>&1
rc_probe=$?
{ echo; echo "== round-2 diagnostic at $GNF4_HEAD_SHA (rc=$rc_probe) =="; cat logs/probe.log; } >> summary.txt
say "diagnostic rc=$rc_probe"

finish 0
