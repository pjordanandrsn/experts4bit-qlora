#!/bin/bash
# bench/p44/p44a_run.sh -- lane P44-a, BOX side (P44-PREREG.md). Started detached by p44a_drive.sh with the run's
# nonce; writes P44A_RUN_NONCE first, then P44A_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P44A_SUCCESS.<nonce> only when every registered piece ran.
#
# One RTX 5090. (1) OLMoE two-text K8: nf4 / int4all / calibexp_all on wikitext AND c4val1 -- the K8 machinery as
# bo5/bo6/bo7/p37/p42 ran it (step_decomp.py --ppl-steps 2048 --b1d-loop eager, k8_bake.py, k8_gate.verdict applied
# by p44_reduce.py, never here). (2) the per-expert residual census for Granite (calibexp recipe) and Mixtral (RTN)
# -- expert_residuals.py; data, no gate. Nothing here changes a kernel or a default.
set -uo pipefail
W=/root/p44a; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] p44a: $*"; }
NONCE=${P44A_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P44A_RUN_NONCE.tmp && mv $W/P44A_RUN_NONCE.tmp $W/P44A_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P44A_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P44A_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P44A_RUN_ID P44A_DEADLINE_EPOCH P44A_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in E4B_SHA GNF4_SHA; do case "${!v}" in *[!0-9a-f]*|"") say "refusing: $v is not hex"; finish 78;; esac; [ ${#v} -ge 0 ] && [ ${#E4B_SHA} -eq 40 ] && [ ${#GNF4_SHA} -eq 40 ] || { say "refusing: $v is not a 40-char sha"; finish 78; }; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=24
MIN_MBPS=${P44_MIN_MBPS:-20}
: > summary.txt; echo "$P44A_INSTANCE_ID" > INSTANCE_ID
# ---- staged pieces, byte-for-byte (names as the box sees them)
for f in p44a_run.sh serve_stack.py expert_residuals.py step_decomp.py k8_bake.py calib.json hook/usercustomize.py staged-a.sha256; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-a.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p44/staged-a.sha256"; finish 9; }
# ---- card class, CUDA, egress: refusal rows, not stalls
python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h /root | tail -1 | tee -a forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *5090*) ;; *) say "REFUSED: card is '$GPU', the lane registers the RTX 5090 class"; echo "refused: class $GPU" > REFUSAL; finish 15;; esac
say "egress pre-flight: HF CDN, 50 MB range, 20 s cap (floor ${MIN_MBPS} MB/s)"
BPS=$(curl -sSL --max-time 20 -r 0-52428800 -o /dev/null -w '%{speed_download}' https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors 2>/dev/null || echo 0)
MBPS=$(python3 -c "print(round(float('${BPS:-0}')/1e6,1))"); say "HF CDN ${MBPS} MB/s"; echo "hf_cdn_mbps=$MBPS" >> forensics.txt
if python3 -c "import sys; sys.exit(0 if float('${MBPS:-0}') < float('$MIN_MBPS') else 1)"; then say "REFUSED: egress ${MBPS} MB/s < ${MIN_MBPS} -- host-limited, not a result"; echo "refused: egress $MBPS MB/s" > REFUSAL; finish 14; fi
# ---- install: e4b pinned + P37's toolchain pins, then gnf4 at the released pin (last, so the kernel pin wins)
say "install e4b @$E4B_SHA (image python; P37 pins) + gnf4 @$GNF4_SHA"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import site, os, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
import usercustomize  # noqa: F401
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated, calibrate_expert_hessians  # noqa: F401
from experts4bit_qlora.k8_gate import verdict  # noqa: F401
from gptq_pack import gptq_pack_int4_b32, HessianAccumulator  # noqa: F401
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p44a/versions.txt", "w").write(f"e4b {e.__version__} @{os.environ['E4B_SHA']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
print("tripwire OK: e4b", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
# ---- helpers
left(){ echo $(( P44A_DEADLINE_EPOCH - $(date +%s) )); }
can_run(){ local need=$1; [ $(( $(date +%s) + need + 600 )) -le "$P44A_DEADLINE_EPOCH" ] || { say "STOP: $2 needs ${need}s, only $(left)s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local l=$(( P44A_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$l" -lt 900 ] && l=900; [ "$l" -gt 14400 ] && l=14400; echo "$l"; }
fetch(){ local MID=$1 REV=$2; say "fetch $MID @ $REV"
  perl -e "alarm $(arm_alarm); exec @ARGV" python - "$MID" "$REV" <<'PYF' > logs/fetch_${MID//\//--}.log 2>&1
import os, sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "merges.txt", "vocab.json", "*.tiktoken", "*.jinja"], max_workers=4)
print(f"staged in {(time.time()-t0)/60:.1f} min at {p}"); print("STAGED", os.path.basename(os.path.realpath(p)))
PYF
  local rc=$?; [ $rc -ne 0 ] && { tail -3 logs/fetch_${MID//\//--}.log; say "FETCH FAIL rc=$rc"; return 11; }
  local GOT; GOT=$(grep -a "^STAGED " logs/fetch_${MID//\//--}.log | tail -1 | awk '{print $2}')
  [ "$GOT" = "$REV" ] || { say "PIN MISMATCH staged=$GOT != pin=$REV -- not coerced"; return 12; }
  local RDIR=/root/.cache/huggingface/hub/models--${MID//\//--}; mkdir -p "$RDIR/refs" && printf '%s' "$REV" > "$RDIR/refs/main"; say "PIN OK $GOT"; }
bake(){ local MID=$1 TAG=$2; [ -e "$W/work_$TAG/nf4.arena" ] && return 0; say "bake $TAG"; mkdir -p $W/work_$TAG
  K8_MODEL="$MID" K8_WORK="$W/work_$TAG" perl -e "alarm $(arm_alarm); exec @ARGV" python $W/k8_bake.py > logs/bake_$TAG.log 2>&1 || { tail -3 logs/bake_$TAG.log; say "BAKE FAIL $TAG"; return 12; }
  [ -e "$W/work_$TAG/nf4.arena" ] || { say "BAKE FAIL $TAG (no arena)"; return 12; }; }
free_family(){ rm -rf $W/work_$1; rm -rf /root/.cache/huggingface/hub/models--${2//\//--}; say "freed $1 ($(df -h /root | tail -1 | awk '{print $4}') free)"; }
first_chunk_watchdog(){ local pid=$1 name=$2 budget=${P44_FIRST_CHUNK_S:-1500} t0; t0=$(date +%s)   # p42's: bound a latency-bound host
  case "$name" in *calibexp*) ;; *) return 0;; esac
  while kill -0 "$pid" 2>/dev/null; do
    grep -qa "INT4EXP calibrated experts" "logs/run_$name.log" 2>/dev/null && { say "$name: calibration chunk 1 in $(( $(date +%s) - t0 ))s"; return 0; }
    if [ $(( $(date +%s) - t0 )) -ge "$budget" ]; then say "HOST-LIMITED: $name produced no calibration chunk in ${budget}s -- killing the arm"; echo "HOSTLIMITED $name no calibration chunk in ${budget}s" >> summary.txt; kill -TERM "$pid" 2>/dev/null; sleep 10; kill -KILL "$pid" 2>/dev/null; return 30; fi
    sleep 20
  done; return 0; }
# k8_arm FAMILY TAG ARM SRC ARENA  -- p42's k8_arm in flags (2048 steps, eager loop, no-fuse-qkv); env from serve_stack.py (ONE table)
k8_arm(){ local FAM=$1 TAG=$2 ARM=$3 SRC=$4 AR=$5 MID; MID=$(python $W/serve_stack.py model $FAM | awk '{print $1}')
  local ENV; ENV=$(python $W/serve_stack.py env $FAM $ARM) || { say "no env for $FAM/$ARM"; return 78; }
  local name=${ARM}_$SRC; { echo "P44a k8 fam=$FAM arm=$ARM src=$SRC env=[$ENV] at=$(date -u +%FT%TZ)"; } > logs/run_$name.log
  env $ENV perl -e "alarm $(arm_alarm); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source $SRC --out $W/${TAG}_ppl_${ARM}_$SRC.json >> logs/run_$name.log 2>&1 &
  local pid=$! rc=0; first_chunk_watchdog "$pid" "$name" || rc=$?
  local wrc=0; wait "$pid" 2>/dev/null || wrc=$?; [ "$rc" = 0 ] && rc=$wrc
  grep -aE "K8_PPL|INT4EXP|ATTNINT4|REFUSED|Error" logs/run_$name.log | tail -4 | sed "s/^/    /"
  { echo -n "k8 $TAG $ARM src=$SRC rc=$rc "; grep -aE "K8_PPL" logs/run_$name.log | tail -1 | cut -c1-240; echo; } >> summary.txt; return $rc; }
rc_any=0
# ================= (1) OLMoE two-text K8
read OL OLREV <<<"$(python $W/serve_stack.py model olmoe)"
if can_run 1200 olmoe_fetch_bake && fetch "$OL" "$OLREV" && bake "$OL" olmoe; then OA=$W/work_olmoe/nf4.arena
  for ARM in nf4 int4all calibexp_all; do for SRC in wikitext c4val1; do
    need=900; case "$ARM" in calibexp_all) need=3000;; esac
    can_run $need "olmoe/$ARM/$SRC" && { k8_arm olmoe olmoe $ARM $SRC $OA || rc_any=$?; }
  done; done
else say "OLMoE fetch/bake failed"; echo "OLMOE fetch/bake FAILED" >> summary.txt; rc_any=11; fi
free_family olmoe "$OL"
# ================= (2) census: Granite (calibexp recipe, hook default 32 x 512 C4) then Mixtral (RTN; same activations)
census(){ local FAM=$1; read MID REV <<<"$(python $W/serve_stack.py model $FAM)"
  can_run "$2" "${FAM}_census" || return 40
  fetch "$MID" "$REV" && bake "$MID" $FAM || { echo "CENSUS $FAM fetch/bake FAILED" >> summary.txt; return 11; }
  say "census $FAM"; E4B_MODEL_ID=$MID perl -e "alarm $(arm_alarm); exec @ARGV" python $W/expert_residuals.py --family $FAM --arena $W/work_$FAM/nf4.arena --calib $W/calib.json \
      --source c4 --nseq 32 --out $W/census_$FAM.json > logs/census_$FAM.log 2>&1; local rc=$?
  tail -3 logs/census_$FAM.log | sed "s/^/    /"; { echo -n "census $FAM rc=$rc "; tail -1 logs/census_$FAM.log | cut -c1-200; } >> summary.txt
  free_family $FAM "$MID"; return $rc; }
census granite 1800 || rc_any=${rc_any:-$?}; [ "$rc_any" = 0 ] && rc_any=$?
census mixtral 5400 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
for f in olmoe_ppl_nf4_wikitext olmoe_ppl_nf4_c4val1 olmoe_ppl_int4all_wikitext olmoe_ppl_int4all_c4val1 olmoe_ppl_calibexp_all_wikitext olmoe_ppl_calibexp_all_c4val1 census_granite census_mixtral; do
  [ -s "$W/$f.json" ] && echo "ROW $f present" >> summary.txt || { echo "ROW $f MISSING" >> summary.txt; [ "$rc_any" = 0 ] && rc_any=41; }
done
say "----- summary -----"; cat summary.txt
finish "$rc_any"
