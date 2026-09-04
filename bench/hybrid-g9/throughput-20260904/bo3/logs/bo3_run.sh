#!/bin/bash
# Validation lane bo3: the build-out under review, one arm PER FUSION (a refusal is a row, never the end of a family).
# Per model: NF4 bake; K8 quality (2048 steps, wikitext) for nf4 / int4 experts / + calibrated int4 attention;
# decode speed at B=1 (nf4, int4exp, calib, calib+fusions) and B=16 (nf4, int4exp), graph loop, timed.
# A refusal is a RESULT: the summary records it, and the gap list is the point.
set -uo pipefail
LANE=bo3; W=/root/$LANE; mkdir -p $W; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook
say(){ echo "[$(date -u +%FT%TZ)] $*"; }
say "install: gnf4 @claude/kernel-integration (#327 + #328) + e4b @claude/buildout-integration-2 (#369 #370 #371 #372; #367 NOT included -- refused pending A/B)"
perl -e 'alarm 1200; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@claude/kernel-integration" "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@claude/buildout-integration-2" "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken > pip.log 2>&1
rc=$?; echo "pip rc=$rc"; [ $rc -ne 0 ] && { tail -4 pip.log; echo "PIP FAIL"; touch TP_DONE; exit 9; }
python - <<'PYT' || { echo "TRIPWIRE FAIL"; touch TP_DONE; exit 9; }
import inspect
from int4_b32 import rmsnorm_resid_rows, rope_norm_heads, router_epilogue, gemv_int4_b32, scaled_resid_add_rows
assert "scale" in inspect.signature(rmsnorm_resid_rows).parameters, "kernel lacks the scaled residual fold (#328)"
assert "select_on_logits" in inspect.signature(router_epilogue).parameters, "kernel lacks select_on_logits (#327)"
from gptq_pack import gptq_pack_int4_b32, HessianAccumulator
from experts4bit_qlora.engines.int4_attn_calib import enable_serve_attn_int4_calib
from experts4bit_qlora.engines.int4_attn import Int4Linear
from experts4bit_qlora.k8_gate import verdict
import os; assert os.path.exists("/root/bo3/hook/usercustomize.py"), "int4 hook missing"
import usercustomize  # noqa: F401
from experts4bit_qlora.arch.moe_conventions import convention_for
assert convention_for("gemma4_text").fused_prefix == "experts", "gemma4 convention missing (#369)"
import experts4bit_qlora.engines.fp8_paged_kv as _kv; assert _kv._auto_k_groups(64) == 4, "head_dim-64 must stay on 4 groups (f32 path) in this lane"
from experts4bit_qlora.engines.router_epilogue import _kernel_supports_select_on_logits, _structural
from experts4bit_qlora.engines.glue_r2 import _layer_scale, _kernel_has_scaled_fold
import int4_b32; assert _kernel_has_scaled_fold(int4_b32), "e4b does not see the scaled fold"
from experts4bit_qlora.engines.int4_experts import _gptoss_packer_layout, _check_gptoss_wrapper
import experts4bit_qlora as e; print("tripwire OK: kernels(#327,#328), gemma4 conv(#369), router kinds(#370), scaled fold(#371), gptoss int4(#372), hook; e4b", e.__version__)
PYT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')" || { echo "DUD BOX"; touch TP_DONE; exit 10; }
[ -s /root/.cache/huggingface/token ] || { echo "TOKEN MISSING"; touch TP_DONE; exit 8; }

: > summary.txt
fenv(){ case "$1" in 0) echo "0 0 0";; r1) echo "1 0 0";; r12) echo "1 1 0";; epi) echo "0 0 1";; r1epi) echo "1 0 1";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
k8(){ # $1 tag $2 arm $3 exp $4 calib $5 fuse-token $6 model $7 arena
  read G R E <<<"$(fenv $5)"
  say "K8 $1/$2 (exp=$3 calib=$4 fuse=$5 steps=2048)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$6" E4B_SERVE_EXP_INT4=$3 E4B_SERVE_ATTN_INT4_CALIB=$4 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 3600; exec @ARGV' python $W/step_decomp.py --model "$6" --arena "$7" \
      --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv \
      --ppl-source wikitext --out $W/$1_ppl_$2.json > run_$1_ppl_$2.log 2>&1
  rc=$?; grep -aE "K8_PPL|INT4EXP|ATTNINT4|REFUSED|Error" run_$1_ppl_$2.log | tail -2 | sed "s/^/    /"; return $rc
}
arm(){ # $1 tag $2 arm $3 batch $4 exp $5 calib $6 fuse-token $7 model $8 arena
  read G R E <<<"$(fenv $6)"
  say "arm $1/$2 (B=$3 exp=$4 calib=$5 fuse=$6)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$7" E4B_SERVE_EXP_INT4=$4 E4B_SERVE_ATTN_INT4_CALIB=$5 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$7" --arena "$8" \
      --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $3 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv \
      --out $W/$1_$2.json > run_$1_$2.log 2>&1
  rc=$?; grep -aE "B1D_TIMED|BV3_|INT4EXP|ATTNINT4|REFUSED|Error" run_$1_$2.log | tail -2 | sed "s/^/    /"; return $rc
}
while read -r MID TAG; do
  [ -z "${MID:-}" ] && continue
  say "=============== $TAG ($MID) ==============="
  python - "$MID" <<'PYD' > dl_$TAG.log 2>&1
import os, sys, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
for a in range(6):
    try:
        print("OK", snapshot_download(sys.argv[1], allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.tiktoken"], ignore_patterns=["original/*","consolidated*"], max_workers=4)); break
    except Exception as e:
        print("retry", a, repr(e)[:120], flush=True); time.sleep(10)
PYD
  grep -q "^OK " dl_$TAG.log || { echo "$TAG: DOWNLOAD FAILED" >> summary.txt; continue; }
  say "bake $TAG"
  K8_MODEL="$MID" K8_WORK="$W/work_$TAG" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > bake_$TAG.log 2>&1
  python -c "import json,sys;d=json.load(open('$W/work_$TAG/bake.json'));sys.exit(0 if d.get('status')=='OK' else 1)" 2>/dev/null || { echo "$TAG: BAKE FAILED" >> summary.txt; grep -aiE 'error|cuda' bake_$TAG.log | tail -2; continue; }
  ARENA=$W/work_$TAG/nf4.arena
  if [ "$TAG" = "qwen3" ]; then
    # regression only: the reference family must not move under the build-out
    k8  $TAG nf4     0 0 0   "$MID" $ARENA; k8 $TAG all 1 1 all "$MID" $ARENA
    arm $TAG b1_nf4  1 0 0 0 "$MID" $ARENA; arm $TAG b1_all 1 1 1 all "$MID" $ARENA; arm $TAG b16_all 16 1 1 all "$MID" $ARENA
  else
    for spec in "nf4 0 0 0" "int4exp 1 0 0" "calib 1 1 0" "r1 0 0 r1" "r12 0 0 r12" "epi 0 0 epi" "stack 1 0 r1epi" "all 1 1 all"; do
      set -- $spec; k8 $TAG $1 $2 $3 $4 "$MID" $ARENA || echo "$TAG/ppl_$1 FAIL" >> summary.txt
    done
    for spec in "nf4 0 0 0" "int4exp 1 0 0" "calib 1 1 0" "r1 0 0 r1" "r12 0 0 r12" "epi 0 0 epi" "stack 1 0 r1epi" "all 1 1 all"; do
      set -- $spec; arm $TAG b1_$1 1 $2 $3 $4 "$MID" $ARENA || echo "$TAG/b1_$1 FAIL" >> summary.txt
    done
    for spec in "nf4 0 0 0" "int4exp 1 0 0" "stack 1 0 r1epi" "all 1 1 all"; do
      set -- $spec; arm $TAG b16_$1 16 $2 $3 $4 "$MID" $ARENA || echo "$TAG/b16_$1 FAIL" >> summary.txt
    done
  fi
  python - "$TAG" <<'PYS' | tee -a summary.txt
import json, sys, glob, os
t = sys.argv[1]
row = [t]
for f in sorted(glob.glob(f"/root/bo3/{t}_*.json")):
    n = os.path.basename(f)[len(t)+1:-5]
    if n.startswith("route") or n == "calib": continue
    try: r = json.load(open(f))
    except Exception: row.append(f"{n}=unreadable"); continue
    if "mean_nll" in r: row.append(f"{n}={r['mean_nll']:.4f}")
    elif "aggregate_tok_s" in r: row.append(f"{n}={float(r['aggregate_tok_s']):.1f}")
    elif r.get("step_ms_clean"): row.append(f"{n}={1000.0/float(r['step_ms_clean']):.1f}")
print("TP " + " ".join(row))
PYS
  grep -ahE "B1D_TIMED|BV3_GRAPH|patched|enabled|calibrated|RuntimeError|TypeError|folded" run_${TAG}_*.log 2>/dev/null | sort | uniq -c | sed "s/^/TPLINE $TAG /" | tee -a summary.txt
done < $W/models.txt
say "TP_DONE"; touch TP_DONE
