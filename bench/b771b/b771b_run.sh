#!/bin/bash
# bench/b771b/b771b_run.sh -- lane B771b, BOX side (bench/b771b/PREREG-b771b.md; e4b#771, e4b#777). Two stages on one
# RTX 5090. The install and the test arms follow bench/b511/b511_run.sh (clone e4b at the launch commit, install it
# editable under a constraints file that pins the image's torch/triton, gnf4 at its pin); the decode stage follows
# bench/b771/b771_run.sh (fetch, bake, one process per arm through P81's step_decomp.py).
#   T  -- the GPU tests on sm_89+: e4b's bucketed-graph tests (the replay test, the #777 invariant and routing tests)
#         and gnf4's fused-append byte gates. armT.xml.
#   M  -- the registered mutation (mut_b771b.py restores the old single-row routing); the invariant test alone. armM.xml.
#   D  -- NF4 Qwen3-30B-A3B, P80's trace: A1, B1, B2, A2, P (P80's arms, A with the default grouping) and Ad (the eager
#         runner with the device grouping). b771b_{ARM}.json.
# Started detached by b771b_drive.sh with the run's nonce; B771B_RUN_NONCE first, then B771B_EXIT_CODE.<nonce> +
# TP_DONE.<nonce> on every exit and B771B_SUCCESS.<nonce> only when the reducer ran.
#
# Knobs (recorded; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# B771B_MODEL B771B_REVISION B771B_GPU_CLASS B771B_MIN_DISK_GB B771B_SEG B771B_REHEARSAL.
# B771B_PROVE=1: the install, the tripwires and the reducer self-test only -- no test arm, no model; exit 0.
set -uo pipefail
W=/root/b771b; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] b771b: $*"; }
NONCE=${B771B_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/B771B_RUN_NONCE.tmp && mv $W/B771B_RUN_NONCE.tmp $W/B771B_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > B771B_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > B771B_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in B771B_RUN_ID B771B_DEADLINE_EPOCH B771B_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in E4B_SHA GNF4_SHA; do val=${!v}; case "$val" in *[!0-9a-f]*|"") say "refusing: $v is not hex"; finish 78;; esac
  [ ${#val} -eq 40 ] || { say "refusing: $v is not a 40-char sha"; finish 78; }; done
D_MODEL=Qwen/Qwen3-30B-A3B; D_REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
MID=${B771B_MODEL:-$D_MODEL}; REV=${B771B_REVISION:-$D_REV}; GPU_CLASS=${B771B_GPU_CLASS:-5090}
MIN_DISK_GB=${B771B_MIN_DISK_GB:-130}; SEG=${B771B_SEG:-32}; REHEARSAL=${B771B_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false E4B_MODEL_ID=$MID
unset E4B_SERVE_EXP_INT4 E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_LMHEAD_INT4_CALIB E4B_SERVE_DENSE_INT4_CALIB E4B_SERVE_ATTN_INT4 \
      E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 E4B_FUSE_ROUTER_EPI E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST
: > summary.txt; echo "$B771B_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS model=$MID rev=$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB seg=$SEG" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$MID" != "$D_MODEL" ] || [ "$REV" != "$D_REV" ] || [ "$GPU_CLASS" != 5090 ] || [ "$SEG" != 32 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
for f in b771b_reduce.py mut_b771b.py step_decomp.py k8_bake.py calib.json staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/b771b/staged.sha256"; finish 9; }
python $W/b771b_reduce.py --self-test | tee -a summary.txt || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed "s/^/compute-app /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
command -v gcc >/dev/null || { say "no gcc in the image (the lane registers the -devel image)"; finish 9; }
arm_alarm(){ local left=$(( B771B_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 600 ] && left=600; echo "$left"; }
# ---- install (B511's): gnf4 at its pin, e4b cloned at the launch commit and installed editable, torch/triton held
TV=$(python -c "import torch; print(torch.__version__)"); XV=$(python -c "import triton; print(triton.__version__)")
printf 'torch==%s\ntriton==%s\n' "$TV" "$XV" > constraints.txt
say "install gnf4 @$GNF4_SHA; clone e4b @$E4B_SHA and gnf4 @$GNF4_SHA (for its tests)"
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' git clone -q https://github.com/pjordanandrsn/experts4bit-qlora.git e4b > logs/clone.log 2>&1 && git -C e4b checkout -q "$E4B_SHA" || { tail -3 logs/clone.log; say "CLONE FAIL (e4b)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git gnf4 > logs/clone_gnf4.log 2>&1 && git -C gnf4 checkout -q "$GNF4_SHA" || { tail -3 logs/clone_gnf4.log; say "CLONE FAIL (gnf4)"; finish 9; }
[ "$(git -C e4b rev-parse HEAD)" = "$E4B_SHA" ] && [ "$(git -C gnf4 rev-parse HEAD)" = "$GNF4_SHA" ] || { say "TRIPWIRE: a clone is not at its pin"; finish 9; }
perl -e 'alarm 1200; exec @ARGV' python -m pip install -q --no-input -c constraints.txt -e "./e4b[test]" "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece > logs/pip_e4b.log 2>&1 || { tail -5 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
python - <<'PYT' > versions.txt 2>&1 || { cat versions.txt; say "TRIPWIRE FAIL"; finish 9; }
import importlib.metadata as md, torch, triton, transformers
import experts4bit_qlora as e, fp8_kv
assert e.__file__.startswith("/root/b771b/e4b/"), f"experts4bit_qlora resolves to {e.__file__}, not the clone"
src = open("/root/b771b/e4b/experts4bit_qlora/engines/paged_attention.py").read()
assert 'getattr(ctx.kv, "_g_sel", None) is not None' in src, "the clone lacks the #777 routing"
assert hasattr(fp8_kv, "_e4m3_group"), "gnf4 lacks the #413 fix"
print("e4b", e.__version__, e.__file__); print("gnf4", md.version("grouped-nf4-gemm"))
print("torch", torch.__version__, "triton", triton.__version__, "transformers", transformers.__version__)
print("bitsandbytes", md.version("bitsandbytes")); print("cc", torch.cuda.get_device_capability())
PYT
grep -q "^torch $TV " versions.txt || { cat versions.txt; say "TRIPWIRE: torch changed during install"; finish 9; }
cat versions.txt | tee -a summary.txt
if [ "${B771B_PROVE:-0}" = 1 ]; then echo "PROVE -- install, tripwires and reducer self-test only" | tee -a summary.txt; : > PROVED; finish 0; fi
# ---- T: the GPU tests (the registered reading is PASS for every one, none skipped)
say "arm T"
(cd $W/e4b && perl -e "alarm 1500; exec @ARGV" python -m pytest tests/test_decode_graph_buckets.py tests/test_bucket1_append_routing.py -v -rs -p no:cacheprovider --junitxml=$W/armT_e4b.xml) > armT_e4b.txt 2>&1
(cd $W/gnf4/kernel && perl -e "alarm 900; exec @ARGV" python -m pytest test_fp8_kv_append.py -v -rs -p no:cacheprovider --junitxml=$W/armT_gnf4.xml) > armT_gnf4.txt 2>&1
python - <<'PYM' || { say "could not merge the arm-T junit"; finish 9; }
import xml.etree.ElementTree as ET
root = ET.Element("testsuites")
for f in ("/root/b771b/armT_e4b.xml", "/root/b771b/armT_gnf4.xml"):
    t = ET.parse(f).getroot()
    for s in ([t] if t.tag == "testsuite" else list(t)):
        root.append(s)
ET.ElementTree(root).write("/root/b771b/armT.xml")
PYM
{ echo -n "arm T e4b: "; tail -n 1 armT_e4b.txt; echo -n "arm T gnf4: "; tail -n 1 armT_gnf4.txt; } | tee -a summary.txt
# ---- M: the registered mutation, then the invariant test alone (the registered reading is FAIL)
say "arm M"
cp mut_b771b.py e4b/ && (cd $W/e4b && python mut_b771b.py) >> summary.txt 2>&1 || { say "mutation did not apply"; finish 9; }
(cd $W/e4b && perl -e "alarm 900; exec @ARGV" python -m pytest tests/test_decode_graph_buckets.py -v -rs -p no:cacheprovider -k every_bucket_step --junitxml=$W/armM.xml) > armM.txt 2>&1
{ echo -n "arm M: "; tail -n 1 armM.txt; } | tee -a summary.txt
git -C e4b checkout -q -- . && rm -f e4b/mut_b771b.py
git -C e4b diff --quiet || { say "the clone did not restore after the mutation"; finish 9; }
# ---- D: fetch (pinned), bake (P39's), the arms
say "fetch $MID @ $REV"
perl -e "alarm $(arm_alarm); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
QA=$W/work/nf4.arena; mkdir -p $W/work
K8_MODEL="$MID" K8_WORK="$W/work" perl -e "alarm $(arm_alarm); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e "$QA" ] || { say "BAKE FAIL (no arena)"; finish 12; }
grep -aE "arena|BAKE" logs/bake.log | tail -1 | tee -a summary.txt
rm -rf $W/work/nf4snap 2>/dev/null
arm(){ local ARM=$1 MODE=$2 GROUP=$3
  say "arm $ARM (mode=$MODE grouping=$GROUP seg=$SEG)"
  perl -e "alarm $(arm_alarm); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json \
      --placement-override all-vram --amort off --batch 16 --prompt-len 512 --no-fuse-qkv \
      --dynb-mode "$MODE" --dynb-grouping "$GROUP" --dynb-arm "$ARM" --dynb-seg "$SEG" --out $W/b771b_$ARM.json > logs/arm_$ARM.log 2>&1
  local rc=$?
  [ -s $W/b771b_$ARM.json ] || python -c "import json,sys; json.dump({'arm':'$ARM','mode':'$MODE','status':'error','rc':$rc,'tail':sys.argv[1]}, open('$W/b771b_$ARM.json','w'))" "$(tail -3 logs/arm_$ARM.log | tr '\n' ' ' | cut -c1-600)"
  { echo -n "arm $ARM rc=$rc "; grep -a "P81_DYNB\|TRACE MISMATCH\|Error" logs/arm_$ARM.log | tail -1 | cut -c1-400; echo; } | tee -a summary.txt
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; }
arm A1 eager default
arm B1 graph default
arm B2 graph default
arm A2 eager default
arm P padded default
arm Ad eager device
python $W/b771b_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish 0
