#!/bin/bash
# bench/p44/p44b_run.sh -- lane P44-b, BOX side (P44-PREREG.md + amendment 1). Started detached by p44b_drive.sh
# with the run's nonce; P44B_RUN_NONCE first, then P44B_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P44B_SUCCESS.<nonce> only when every registered arm produced a row.
#
# One >= 80 GB card. K0 controls ON THIS HOST first (no KL row without them), then per family: pinned fetch, NF4
# arena bake (k8_bake.py), kl_serve.py -- the reference scored once (decode-shaped teacher forcing) and freed, then
# the arms bo7 timed, one at a time. Gemma-4-26B-A4B-it: nf4 / r1epi / int4_r1epi / calattn_r1epi against the bf16
# checkpoint. gpt-oss-20b: nf4_r12 / store_r12 against dequant-to-bf16 of the shipped MXFP4 bytes. Mixtral is
# NOT_RUN by registration. Nothing here changes a kernel or a default; the reading rule is applied by p44_reduce.py.
set -uo pipefail
W=/root/p44b; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] p44b: $*"; }
NONCE=${P44B_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P44B_RUN_NONCE.tmp && mv $W/P44B_RUN_NONCE.tmp $W/P44B_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P44B_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P44B_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P44B_RUN_ID P44B_DEADLINE_EPOCH P44B_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in E4B_SHA GNF4_SHA; do case "${!v}" in *[!0-9a-f]*|"") say "refusing: $v is not hex"; finish 78;; esac; done
[ ${#E4B_SHA} -eq 40 ] && [ ${#GNF4_SHA} -eq 40 ] || { say "refusing: a pin is not a 40-char sha"; finish 78; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=24
MIN_MBPS=${P44_MIN_MBPS:-20}; MIN_VRAM_GB=${P44_MIN_VRAM_GB:-80}
: > summary.txt; echo "$P44B_INSTANCE_ID" > INSTANCE_ID
for f in p44b_run.sh serve_stack.py kl_serve.py kl_fidelity.py kl_paths.py kl_prompts.py k8_bake.py calib.json hook/usercustomize.py staged-b.sha256; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-b.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p44/staged-b.sha256"; finish 9; }
python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h /root | tail -1 | tee -a forensics.txt
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${VRAM_MB:-0}" -lt $(( MIN_VRAM_GB * 1000 )) ]; then say "REFUSED: card holds ${VRAM_MB} MiB < ${MIN_VRAM_GB} GB -- not the registered class"; echo "refused: vram class" > REFUSAL; finish 15; fi
say "egress pre-flight: HF CDN, 50 MB range, 20 s cap (floor ${MIN_MBPS} MB/s)"
BPS=$(curl -sSL --max-time 20 -r 0-52428800 -o /dev/null -w '%{speed_download}' https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors 2>/dev/null || echo 0)
MBPS=$(python3 -c "print(round(float('${BPS:-0}')/1e6,1))"); say "HF CDN ${MBPS} MB/s"; echo "hf_cdn_mbps=$MBPS" >> forensics.txt
if python3 -c "import sys; sys.exit(0 if float('${MBPS:-0}') < float('$MIN_MBPS') else 1)"; then say "REFUSED: egress ${MBPS} MB/s < ${MIN_MBPS} -- host-limited, not a result"; echo "refused: egress $MBPS MB/s" > REFUSAL; finish 14; fi
[ -s /root/.cache/huggingface/token ] || [ -n "${HF_TOKEN:-}" ] || say "WARNING: no HF token on the box -- gated repos (Gemma) will refuse at fetch"
say "install e4b @$E4B_SHA (image python; P37 pins) + gnf4 @$GNF4_SHA"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import site, os, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
import usercustomize  # noqa: F401
from experts4bit_qlora.engines.int4_attn import Int4Linear  # noqa: F401
from experts4bit_qlora.engines.glue_fuse import fuse_t1_glue  # noqa: F401
import experts4bit_qlora as e, torch, triton, transformers
from transformers import Mxfp4Config  # noqa: F401  (the gpt-oss dequant reference)
open("/root/p44b/versions.txt", "w").write(f"e4b {e.__version__} @{os.environ['E4B_SHA']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
print("tripwire OK: e4b", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
# ---- K0 controls on THIS host: the instrument's own gate
perl -e 'alarm 600; exec @ARGV' python $W/kl_fidelity.py --controls --out $W/k0.json > logs/k0.log 2>&1 || { tail -5 logs/k0.log; say "K0 CONTROLS FAILED -- no KL row is produced (the instrument's rule)"; echo "K0 FAILED" >> summary.txt; finish 16; }
python -c "import json; r=json.load(open('$W/k0.json')); assert r['all_passed']; print('K0 all_passed', [c['name'] if 'name' in c else c.get('control') for c in r['controls']])" | tee -a summary.txt || { say "K0 receipt not passing"; finish 16; }
left(){ echo $(( P44B_DEADLINE_EPOCH - $(date +%s) )); }
can_run(){ local need=$1; [ $(( $(date +%s) + need + 600 )) -le "$P44B_DEADLINE_EPOCH" ] || { say "STOP: $2 needs ${need}s, only $(left)s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local l=$(( P44B_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$l" -lt 900 ] && l=900; [ "$l" -gt 14400 ] && l=14400; echo "$l"; }
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
free_family(){ rm -rf $W/work_$1 $W/refcache_$1; rm -rf /root/.cache/huggingface/hub/models--${2//\//--}; say "freed $1 ($(df -h /root | tail -1 | awk '{print $4}') free)"; }
# kl FAMILY NEED_S  -- fetch, bake, kl_serve over the family's registered arms; receipt <family>_kl.json (rewritten per arm)
kl(){ local FAM=$1; read MID REV <<<"$(python $W/serve_stack.py model $FAM)"
  can_run "$2" "${FAM}_kl" || return 40
  fetch "$MID" "$REV" && bake "$MID" $FAM || { echo "KL $FAM fetch/bake FAILED" >> summary.txt; return 11; }
  say "kl_serve $FAM (scorer by control (i): decode if the reference agrees with itself, else prefill; reference cached to refcache_$FAM.<scorer> then freed)"
  E4B_MODEL_ID=$MID perl -e "alarm $(arm_alarm); exec @ARGV" python -u $W/kl_serve.py --family $FAM --arena $W/work_$FAM/nf4.arena --calib $W/calib.json \
      --k0-receipt $W/k0.json --ref-cache $W/refcache_$FAM --scorer auto --out $W/${FAM}_kl.json > logs/kl_$FAM.log 2>&1; local rc=$?
  grep -aE "^== |NOT MEASURED|receipt ->|Error" logs/kl_$FAM.log | tail -8 | sed "s/^/    /"
  { echo -n "kl $FAM rc=$rc "; grep -aE "receipt ->" logs/kl_$FAM.log | tail -1 | cut -c1-200; echo; } >> summary.txt
  free_family $FAM "$MID"; return $rc; }
rc_any=0
kl gemma4 5400 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
kl gptoss 3000 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
for f in gemma4_kl gptoss_kl; do [ -s "$W/$f.json" ] && echo "ROW $f present" >> summary.txt || { echo "ROW $f MISSING" >> summary.txt; [ "$rc_any" = 0 ] && rc_any=41; }; done
say "----- summary -----"; cat summary.txt
finish "$rc_any"
