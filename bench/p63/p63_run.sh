#!/bin/bash
# bench/p63/p63_run.sh -- lane P63, BOX side (bench/p63/P63-PREREG.md; experts4bit-qlora#708). Started detached by
# p63_drive.sh with the run's nonce; P63_RUN_NONCE first, then P63_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit
# and P63_SUCCESS.<nonce> only on clean completion.
#
# Does a token's output depend on how many rows share its forward? Three stacks (nf4, int4, int4nf), one process each,
# built the way step_decomp builds its shipped stack; every sub-arm is a route chosen by configuration only. Per token:
# T = 1 incremental decode (the control) against the same token inside a 17- and a 16-row verify and a 160-row
# prefill, plus a module replay and a kernel census. p63_reduce.py applies the pre-registration on the box. Nothing
# here changes a default. Pattern: bench/p59/p59_run.sh; the fma side diagnostic is grouped-nf4-gemm #397's.
set -uo pipefail
W=/root/p63; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p63: $*"; }
NONCE=${P63_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P63_RUN_NONCE.tmp && mv $W/P63_RUN_NONCE.tmp $W/P63_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P63_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P63_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P63_RUN_ID P63_DEADLINE_EPOCH P63_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char sha"; finish 78; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# the P42 lane hook must NOT be on the path: the probe applies the int4 lanes itself (it refuses if the hook is active)
unset PYTHONPATH
MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
: > summary.txt; echo "$P63_INSTANCE_ID" > INSTANCE_ID
# ---- staged pieces, byte-for-byte
for f in p63_probe.py p63_compare.py p63_reduce.py fixture.txt serve_stack.py k8_bake.py calib.json staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p63/staged.sha256"; finish 9; }
# ---- the box: a dud or a different class is refused before anything is installed
python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; nvcc --version 2>/dev/null | tail -1 | tee -a forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *5090*) ;; *) say "REFUSED: card is '$GPU', the lane registers the RTX 5090 class"; echo "refused: class $GPU" > REFUSAL; finish 15;; esac
df -h /root | tail -1 | tee -a forensics.txt
# ---- deadline guard + per-stack alarm (P59's): never start a stack that cannot finish 10 min before teardown
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P63_DEADLINE_EPOCH" ] || { say "STOP: $2 needs ${need}s, only $((P63_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P63_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
# ---- install: e4b pinned (image python; P59's toolchain pins), gnf4 at the lane's cut, and the same gnf4 sha cloned
# for the side diagnostic (a kernel/ script, not in the wheel)
export DEBIAN_FRONTEND=noninteractive
say "install e4b @$E4B_SHA, gnf4 @$GNF4_SHA"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git src > logs/clone.log 2>&1 && git -C src checkout -q "$GNF4_SHA" \
  || say "gnf4 clone failed (only the side diagnostic needs it; continuing)"
python - <<'PYT' || { say "TRIPWIRE FAIL (e4b)"; finish 9; }
import os, importlib.metadata as md
import experts4bit_qlora as e, torch, triton, transformers
import experts4bit_qlora.engines.hybrid as hy
from experts4bit_qlora.engines import hot_residency as hr
from experts4bit_qlora.engines.int4_attn import Int4Linear, _smallm_kernels
import int4_b32, nf4_grouped
assert not getattr(hy.enable_hybrid_tier, "_gen_hooked", False), "the P42 hook is active: the lanes would apply twice"
assert "site-packages" in int4_b32.__file__, int4_b32.__file__
assert getattr(Int4Linear, "SMALLM_ROWS_MAX", None) == 16 and Int4Linear.GEMV_ROWS_MAX == 1
_smallm_kernels()                                           # K16 installed: the 2..16-row bucket exists on this box
assert hasattr(int4_b32, "combine_rows") and hasattr(int4_b32, "gemv_int4_b32")
assert int4_b32.gemv_fused_reduce_default() is False, "the lane reads the shipped two-launch GEMV default"
fix = hasattr(hr, "_int4_part_or_none")
open("/root/p63/versions.txt", "a").write(
    f"e4b {e.__version__} @{os.environ['E4B_SHA']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\n"
    f"torch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\n"
    f"bitsandbytes {md.version('bitsandbytes')}\nint4_part_or_none_fix {fix}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "fix", fix)
PYT
cat versions.txt | tee -a summary.txt
# ---- fetch (pinned) and bake the NF4 arena (P39's k8_bake.py, from the pinned LOCAL snapshot)
say "fetch $MID @ $REV"
perl -e 'alarm 4800; exec @ARGV' python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -n 1 logs/fetch.log); [ -d "$SNAP" ] || { say "DL FAIL (no snapshot dir: $SNAP)"; finish 11; }
echo "snapshot $SNAP" | tee -a summary.txt
say "bake NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
QA=$W/work/nf4.arena; [ -e "$QA" ] || { say "BAKE FAIL (no arena)"; python -c "import json; r=json.load(open('$W/work/bake.json')); print(r.get('err')); print(r.get('tb'))" 2>/dev/null | tail -n 20; finish 12; }
rm -rf $W/work/nf4snap                                      # the relocation source; the arena is what the stacks load
# ---- stacks: one process each (P63-PREREG "Arms")
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
stack(){ local S=$1 NEED=$2; can_run "$NEED" "$S" || { rec 30; return; }
  local AL; AL=$(arm_alarm 2400); say "stack $S (alarm ${AL}s)"
  perl -e "alarm $AL; exec @ARGV" python -u $W/p63_probe.py --stack "$S" --model "$SNAP" --arena "$QA" --calib $W/calib.json \
    --fixture $W/fixture.txt --out $W/out/$S > logs/run_$S.log 2>&1
  local r=$?
  grep -aE "^P63 |P63ARM|REFUSED|Traceback|Error" logs/run_$S.log | tail -12 | cut -c1-400 | sed "s/^/    /"
  { echo -n "stack $S rc=$r "; grep -a "P63ARM" logs/run_$S.log | tail -1 | cut -c1-600; echo; } >> summary.txt
  [ -s "$W/out/$S/p63_arm.json" ] || { echo "RECEIPT MISSING $S" >> summary.txt; r=${r:-42}; [ "$r" = 0 ] && r=42; }
  rec $r; }
stack int4 1500
stack nf4 1200
stack int4nf 1200
# ---- reduce on the box (the verdict is the reducer's JSON, read against the prereg; never this script's exit code)
say "reduce"; python $W/p63_reduce.py $W/out --md $W/RESULTS-p63-generated.md --json $W/p63_rep.json > RESULTS.txt 2>&1; red=$?
head -c 6000 RESULTS.txt | tee -a summary.txt >/dev/null
[ "$red" = 14 ] && rec 14
# ---- side diagnostic (disclosed, NOT part of the decision; never changes the exit code): what sm_120's combine_rows
# computes, grouped-nf4-gemm #397's script from the clone at GNF4_SHA, only where that cut carries it
FMA=src/kernel/receipts-b393/a2000-fma-attribution/fma_attribution.py
if [ -f "$FMA" ]; then
  say "side diagnostic: fma_attribution"
  (perl -e 'alarm 600; exec @ARGV' python "$FMA" "$W/fma_attribution_5090.json" > logs/fma_attribution.log 2>&1 \
    && echo "fma_attribution: written" || echo "fma_attribution: rc=$? (non-fatal)") | tee -a summary.txt
else
  echo "fma_attribution: not in this gnf4 cut (non-fatal; the script lands with grouped-nf4-gemm #397)" | tee -a summary.txt
fi
say "----- summary -----"; cat summary.txt; echo "----- versions -----"; cat versions.txt
finish "$rc_any"
