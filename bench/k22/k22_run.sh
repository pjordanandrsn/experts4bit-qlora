#!/bin/bash
# bench/k22/k22_run.sh -- lane K22, BOX side. The prereg, the kernel and the bench live in grouped-nf4-gemm
# (kernel/PREREG-k22-gptoss-mxfp4-b16.md, kernel/mxfp4_grouped.py, kernel/k22_bench.py); this runner drives the model
# phases, which need this repo's harness. Started detached by k22_drive.sh with the run's nonce; K22_RUN_NONCE first,
# then K22_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and K22_SUCCESS.<nonce> only when the bench ran.
#
# gpt-oss-20b at B=16 on an RTX 5090, three phases on one box:
#   1. census   bo7's store_r12 configuration (MXFP4 store, NF4 kept for batched rows, folds r1/r2), B=16, graph-window
#               timing + P42 replay census                                         -> census_b16.json, logs/census_b16.txt
#   2. routing  16 wikitext rows by gpt-oss's own tokenizer (step_decomp._k8_window, as P37), then
#               bench/families/record_eids.py on the served model: 128 teacher-forced B=16 steps -> eids_b16.pt
#   3. bench    k22_bench.py on that routing, its instrument the census's _gemm_nf4_grouped ms/step -> k22_rows.json
# K22_PROVE=1 is the proving run: refusals, install + tripwire, K21's and K16's contracts compiled on this card, the
# bench self-test; no model. Lane failures use rc 31-33 (never 13/14/17/18, which exclude the machine). Knobs (any off its default marks a REHEARSAL): K22_GPU_CLASS K22_MIN_DISK_GB K22_REHEARSAL.
set -uo pipefail
W=/root/k22; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] k22: $*"; }
NONCE=${K22_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/K22_RUN_NONCE.tmp && mv $W/K22_RUN_NONCE.tmp $W/K22_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > K22_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > K22_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in K22_RUN_ID K22_DEADLINE_EPOCH K22_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA$GNF4_SHA" in *[!0-9a-f]*) say "refusing: a sha is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] && [ ${#GNF4_SHA} -eq 40 ] || { say "refusing: E4B_SHA / GNF4_SHA must be 40-char shas"; finish 78; }
MID=openai/gpt-oss-20b; REV=6cee5e81ee83917806bbde320786a8fb61efebee
GPU_CLASS=${K22_GPU_CLASS:-5090}; MIN_DISK_GB=${K22_MIN_DISK_GB:-100}; REHEARSAL=${K22_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false E4B_MODEL_ID=$MID
export PYTHONPATH=$W/hook E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST E4B_FUSED_KV_APPEND
# bo7's store_r12 arm, verbatim (bench/hybrid-g9/throughput-20260904/bo7/logs/bo7_run.sh:64): the MXFP4 store for single
# rows, NF4 kept for batched rows, folds r1/r2, no router epilogue (the recorder needs the router module called)
STOREENV="E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=0"
: > summary.txt; echo "$K22_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MID rev=$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 100 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py p42_reduce.py record_eids.py serve_stack.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/k22/staged.sha256"; finish 9; }
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
arm_alarm(){ local cap=$1 left=$(( K22_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; P37 pins)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" pytest || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
perl -e 'alarm 600; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/src > logs/clone.log 2>&1 && git -C $W/src checkout -q $GNF4_SHA \
  || { say "CLONE FAIL"; finish 9; }
cp $W/src/kernel/k22_bench.py $W/ || { say "k22_bench.py missing at $GNF4_SHA"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import mxfp4_grouped, nf4_grouped, int4_b32
for mod, n in ((mxfp4_grouped, "gemm_mxfp4_grouped_smallm"), (mxfp4_grouped, "gemv_mxfp4_b32"),
               (nf4_grouped, "gemm_4bit_grouped_captured"), (int4_b32, "build_group_tiles_fused")):
    assert hasattr(mod, n), f"installed gnf4 lacks {n}"
src = os.path.abspath(inspect.getsourcefile(mxfp4_grouped))
assert not src.startswith("/root/k22/src/"), f"mxfp4_grouped resolved to the clone ({src})"
from experts4bit_qlora.engines import hot_residency as hr
assert hr._MXFP4_GEMV_ROWS == 16, "the MXFP4 store's GEMV row limit moved: the census would not be the route bo7 measured"
import usercustomize  # noqa: F401
assert usercustomize.__file__.startswith("/root/k22/hook/"), usercustomize.__file__
import experts4bit_qlora as e, torch, triton, transformers
open("/root/k22/versions.txt", "w").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\n"
                                          f"torch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\n"
                                          f"cc {torch.cuda.get_device_capability()}\n")
print("tripwire OK")
PYT
cat versions.txt | tee -a summary.txt
python $W/k22_bench.py --self-test | tee -a summary.txt || { say "RULE SELF-TEST FAILED"; finish 21; }
if [ "${K22_PROVE:-0}" = 1 ]; then
  (cd $W/src/kernel && PYTHONPATH= TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_mxfp4_grouped_smallm_interp.py test_int4_smallm_interp.py -q -p no:cacheprovider) > logs/k21_contract.log 2>&1
  rc=$?; { echo -n "PROVE K21+K16 contract compiled rc=$rc: "; tail -1 logs/k21_contract.log; } | tee -a summary.txt
  [ "$rc" = 0 ] || { say "PROVE: the K21 contract does not hold on this card"; finish 23; }
  : > PROVED; finish 0
fi
say "fetch $MID @ $REV"
perl -e "alarm $(arm_alarm 2400); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', ignore_patterns=['original/*', 'metal/*'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
say "bake NF4 arena"; mkdir -p $W/work_gptoss
K8_MODEL="$MID" K8_WORK="$W/work_gptoss" perl -e "alarm $(arm_alarm 2400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
QA=$W/work_gptoss/nf4.arena; [ -e "$QA" ] || { say "BAKE FAIL: no arena"; finish 12; }
# ---- phase 1: census (descriptive; its _gemm_nf4_grouped row is the bench's instrument)
say "phase 1: census B=16 (store_r12)"
env $STOREENV perl -e "alarm $(arm_alarm 1800); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json \
  --placement-override all-vram --amort off --batch 16 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv \
  --replay-profile-out $W/logs/census_b16.txt --out $W/census_b16.json > logs/run_census_b16.log 2>&1
rc=$?; { echo -n "census B=16 rc=$rc "; grep -aE "BV3_|REFUSED|Error" logs/run_census_b16.log | tail -1 | cut -c1-200; echo; } | tee -a summary.txt
[ "$rc" = 0 ] && [ -s logs/census_b16.txt ] || { say "CENSUS FAIL"; finish 31; }
NF4MS=$(python - <<'PYC'
import importlib.util
spec = importlib.util.spec_from_file_location("p42", "/root/k22/p42_reduce.py"); p42 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p42)
rep, rows = p42.parse_census(open("/root/k22/logs/census_b16.txt").read())
print(sum(r["self_ms"] for r in rows if r["name"].strip() == "_gemm_nf4_grouped") / rep)
PYC
)
echo "CENSUS _gemm_nf4_grouped ms/step $NF4MS" | tee -a summary.txt
# ---- phase 2: prompts by gpt-oss's own tokenizer, then the routing record on the served model
say "phase 2: prompts + routing record"
python - "$MID" "$REV" <<'PYP' > logs/prompts.log 2>&1 || { tail -5 logs/prompts.log; say "PROMPT FAIL"; finish 32; }
import sys, json, hashlib, types
sys.path.insert(0, "/root/k22")
import step_decomp
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], revision=sys.argv[2])
a = types.SimpleNamespace(ppl_source="wikitext", ppl_chat=False, ppl_chat_suffix="", prompt_offset=0, prompt_span=0,
                          prompt_len=512, batch=16, ppl_steps=0)
ids, step, prompts, _, _ = step_decomp._k8_window(a, tok)
assert len(prompts) == 16 and all(len(p) == 512 for p in prompts) and len(set(map(tuple, prompts))) == 16
sha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
json.dump({"model": sys.argv[1], "revision": sys.argv[2], "source": "wikitext-2-raw-v1 test via step_decomp._k8_window",
           "batch": 16, "prompt_len": 512, "row_step": int(step), "prompts_sha256": sha, "prompts": prompts}, open("/root/k22/prompts_b16.json", "w"))
print(f"PROMPTS B=16 sha={sha}")
PYP
grep -a PROMPTS logs/prompts.log | tee -a summary.txt
env $STOREENV perl -e "alarm $(arm_alarm 2400); exec @ARGV" python $W/record_eids.py --model "$MID" --arena "$QA" --calib $W/calib.json \
  --prompts $W/prompts_b16.json --out $W/eids_b16.pt > logs/record.log 2>&1
rc=$?; { echo -n "record rc=$rc "; grep -aE "P60EIDS|Error" logs/record.log | tail -1 | cut -c1-200; echo; } | tee -a summary.txt
[ "$rc" = 0 ] && [ -s eids_b16.pt ] || { say "RECORD FAIL"; finish 33; }
# ---- phase 3: the kernel bench on the recorded routing
say "phase 3: k22_bench (alarm $(arm_alarm 2400))"
PYTHONPATH= perl -e "alarm $(arm_alarm 2400); exec @ARGV" python $W/k22_bench.py $W/eids_b16.pt $W/k22_rows.json --census-nf4-ms "$NF4MS" > logs/k22_bench.log 2>&1
rc=$?; grep -aE '^copy |K22 EVAL|K22_VERDICT' logs/k22_bench.log | tee -a summary.txt
[ "$rc" = 0 ] && [ -s k22_rows.json ] || { say "BENCH rc=$rc"; finish 22; }
finish 0
