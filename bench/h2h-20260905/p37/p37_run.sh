#!/bin/bash
# Lane p37 (vLLM head-to-head, same box, current vs current) -- pre-registered in P37-VLLM-PREREG.md.
# Cut = the shipped code: e4b main @f4b639fd2640 (= v0.35.0 + #396 docs) + gnf4 main @ddcb850e05c3 (= v0.30.0 + #341 docs) --
# bo7's cut, bo7's hook v6, bo7's step_decomp.py / k8_bake.py / calib.json staged VERBATIM from bench/hybrid-g9/throughput-20260904/bo7/.
# vLLM = the latest PyPI release at launch (P37_VLLM_VERSION overrides; the tripwire prints and pins it in every receipt).
# Pattern: bo7_run.sh / bo7b.sh (per-arm `perl -e 'alarm N'`, summary.txt, TP_DONE, `say`) + h2h_run.sh (e4b FIRST, vLLM in its own venv).
# e4b runs in the IMAGE python: CPython skips usercustomize inside a venv (ENABLE_USER_SITE=False), which would run the BASE path
# under the licensed label with a vacuous delta. The tripwire asserts site.ENABLE_USER_SITE; the arms' banners prove engagement.
set -uo pipefail
LANE=p37; W=/root/$LANE; mkdir -p $W $W/logs $W/hook; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
E4B_SHA=${E4B_SHA:-f4b639fd2640bb603b6b7b63ea010312b0bb351d}; GNF4_SHA=${GNF4_SHA:-ddcb850e05c3595e2bb87813df70427f7e4bafce}
BF16_MID=Qwen/Qwen3-30B-A3B;            BF16_REV=${P37_BF16_REV:-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}
GPTQ_MID=Qwen/Qwen3-30B-A3B-GPTQ-Int4;  GPTQ_REV=${P37_GPTQ_REV:-9b534e4318b7ebc3c961a839f13eb18b1833f441}
E4B_TORCH_CU128=${P37_E4B_TORCH_CU128:-0}   # 1 = force-reinstall torch 2.8.0+cu128 into the image python (bo7's exact toolchain); default = the image's torch
SKIP=${P37_SKIP:-}                          # space-separated arm tags to skip (e.g. "lic_eager")
export P37_INSTANCE_ID=${P37_INSTANCE_ID:-$(cat /root/p37/INSTANCE_ID 2>/dev/null)}   # written by the launcher; joins every receipt to the bill
say(){ echo "[$(date -u +%FT%TZ)] $*"; }
: > summary.txt
# ---------------------------------------------------------------- staged files: the bo7 lane pieces, byte-for-byte (staged.sha256 comes from the receipt dir; see README-lanes.md)
for f in hook/usercustomize.py step_decomp.py k8_bake.py calib.json p37_vllm.py p37_reduce.py; do [ -s $W/$f ] || { echo "STAGE MISSING: $f"; touch TP_DONE; exit 9; }; done
(cd $W && sha256sum hook/usercustomize.py step_decomp.py k8_bake.py calib.json) | tee logs/staged.actual.sha256
if [ -s $W/staged.sha256 ]; then (cd $W && sha256sum -c staged.sha256) || { echo "STAGED FILES DIFFER FROM THE bo7 RECEIPT COPIES"; touch TP_DONE; exit 9; }; else echo "WARNING: no staged.sha256 -- the staged pieces are not proven identical to bo7's (record above)"; fi
# ---------------------------------------------------------------- e4b install (image python) + tripwire (the cut, by symbol AND version; the hook loadable; triton 3.4)
say "install e4b (image python): gnf4 @$GNF4_SHA + e4b @$E4B_SHA"
if [ "$E4B_TORCH_CU128" = "1" ]; then perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128 > logs/pip_torch.log 2>&1; echo "torch cu128 reinstall rc=$?"; fi
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1
rc=$?; echo "pip(e4b) rc=$rc"; [ $rc -ne 0 ] && { tail -4 logs/pip_e4b.log; echo "PIP FAIL (e4b)"; touch TP_DONE; exit 9; }
PYTHONPATH=$W/hook python - <<'PYT' || { echo "TRIPWIRE FAIL (e4b)"; touch TP_DONE; exit 9; }
import site, os, inspect, importlib.metadata as md
assert site.ENABLE_USER_SITE, "ENABLE_USER_SITE is False: usercustomize would NOT load (venv?) -- the licensed arm would run the base path"
from int4_b32 import rope_heads, reduce_partials, combine_rows, swiglu_rows
from gptq_pack import gptq_pack_int4_b32, HessianAccumulator
from experts4bit_qlora.engines.int4_experts import calibrate_expert_hessians, _ExpertHessianSink, enable_serve_experts_int4_calibrated
from experts4bit_qlora.engines import int4_experts, hot_residency
assert "E4B_INT4_GPTQ_DEVICE" in inspect.getsource(int4_experts), "gpu-solve knob missing"
assert hasattr(hot_residency, "_CALIB_SINK") and hasattr(hot_residency, "_swiglu_or"), "tap or swiglu path missing"
assert os.path.exists("/root/p37/hook/usercustomize.py"), "hook missing"
import usercustomize  # noqa: F401  (loadable; ENGAGEMENT is proven per arm by the INT4EXP / ATTNINT4 banners)
import experts4bit_qlora as e, torch, triton, transformers
assert e.__version__ == "0.35.0", e.__version__
assert md.version("grouped-nf4-gemm") == "0.30.0", md.version("grouped-nf4-gemm")
assert triton.__version__.startswith("3.4"), f"triton {triton.__version__} != 3.4.x (bo7's JIT toolchain)"
assert transformers.__version__ == "5.16.1", transformers.__version__
print("p37 tripwire OK (e4b): e4b", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__, "triton", triton.__version__, "hook v6 loadable, ENABLE_USER_SITE", site.ENABLE_USER_SITE)
open("/root/p37/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ.get('E4B_SHA','')}\ngnf4 {md.version('grouped-nf4-gemm')}\ntorch(e4b) {torch.__version__}\ntriton(e4b) {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
PYT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|NUMA node\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; cat /sys/fs/cgroup/memory.max 2>/dev/null | sed "s/^/cgroup memory.max /" | tee -a forensics.txt
nvcc --version 2>/dev/null | tail -1 | tee -a forensics.txt
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')" || { echo "DUD BOX"; touch TP_DONE; exit 10; }
[ -s /root/.cache/huggingface/token ] || echo "NOTE: no HF token staged (Qwen3 is not gated; rate limits apply)"
df -h /root | tail -1
# ---------------------------------------------------------------- vLLM venv (its own torch/triton; version = latest PyPI at launch unless P37_VLLM_VERSION)
VLLM_VER=${P37_VLLM_VERSION:-$(python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('https://pypi.org/pypi/vllm/json',timeout=60))['info']['version'])" 2>/dev/null)}
[ -z "$VLLM_VER" ] && { echo "VLLM VERSION UNRESOLVED (set P37_VLLM_VERSION)"; VLLM_OK=0; } || VLLM_OK=1
if [ "$VLLM_OK" = "1" ]; then
  say "install vllm==$VLLM_VER into $W/venv-vllm"
  python -m venv $W/venv-vllm && perl -e 'alarm 2700; exec @ARGV' $W/venv-vllm/bin/pip install -q --no-input --no-cache-dir "vllm==$VLLM_VER" "huggingface_hub>=0.23" > logs/pip_vllm.log 2>&1
  rc=$?; echo "pip(vllm) rc=$rc"; [ $rc -ne 0 ] && { tail -6 logs/pip_vllm.log; echo "PIP FAIL (vllm) -- e4b arms still run; NO COMPARATOR"; VLLM_OK=0; }
fi
if [ "$VLLM_OK" = "1" ]; then
  $W/venv-vllm/bin/python - <<'PYV' >> logs/tripwire_vllm.log 2>&1 || { echo "VLLM IMPORT FAIL -- e4b arms still run; NO COMPARATOR"; tail -5 logs/tripwire_vllm.log; VLLM_OK=0; }
import vllm, torch, importlib.metadata as md
tri = None
try:
    import triton; tri = triton.__version__
except Exception: pass
print("p37 tripwire OK (vllm):", vllm.__version__, "torch", torch.__version__, "triton", tri)
open("/root/p37/versions.txt", "a").write(f"vllm {vllm.__version__}\ntorch(vllm) {torch.__version__}\ntriton(vllm) {tri}\n")
PYV
  tail -1 logs/tripwire_vllm.log
fi
# ---------------------------------------------------------------- downloads (revisions PINNED), bake (bo7's k8_bake.py), prompt dump (step_decomp's OWN _k8_window)
fetch(){ local MID=$1 REV=$2 TAG=$3 AL=$4; say "fetch $TAG ($MID @ $REV)"
  perl -e "alarm $AL; exec @ARGV" python - "$MID" "$REV" <<'PYF' > logs/fetch_$TAG.log 2>&1
import sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "merges.txt", "vocab.json"], max_workers=4)
print(f"staged in {(time.time()-t0)/60:.1f} min at {p}")
PYF
  rc=$?; tail -1 logs/fetch_$TAG.log; [ $rc -ne 0 ] && { echo "$TAG: DL FAIL rc=$rc" | tee -a summary.txt; return 1; }; df -h /root | tail -1; return 0; }
fetch $BF16_MID $BF16_REV bf16 4800 || { echo "DL FAIL (bf16)"; touch TP_DONE; exit 11; }
say "bake qwen3 (k8_bake.py, bo7's)"; mkdir -p $W/work_qwen3
K8_MODEL="$BF16_MID" K8_WORK="$W/work_qwen3" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > logs/bake_qwen3.log 2>&1 || { echo "BAKE FAIL"; tail -3 logs/bake_qwen3.log; touch TP_DONE; exit 12; }
grep -aE "arena|BAKE" logs/bake_qwen3.log | tail -1; QA=$W/work_qwen3/nf4.arena; [ -e "$QA" ] || { echo "BAKE FAIL (no arena)"; touch TP_DONE; exit 12; }
say "prompt dump: step_decomp._k8_window at B=1 and B=16 -> prompts_b{1,16}.json (identical ids for both engines)"
python - "$BF16_MID" "$BF16_REV" <<'PYP' > logs/prompts.log 2>&1 || { echo "PROMPT DUMP FAIL"; tail -5 logs/prompts.log; touch TP_DONE; exit 13; }
import sys, json, hashlib, types
sys.path.insert(0, "/root/p37")
import step_decomp                                   # the staged harness; its own window rule, not a re-implementation
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], revision=sys.argv[2])
for B in (1, 16):
    a = types.SimpleNamespace(ppl_source="wikitext", ppl_chat=False, ppl_chat_suffix="", prompt_offset=0, prompt_span=0,
                              prompt_len=512, batch=B, ppl_steps=0)
    ids, step, prompts, _, ppl_sha = step_decomp._k8_window(a, tok)
    assert all(len(p) == 512 for p in prompts) and len(prompts) == B
    assert len(set(tuple(p) for p in prompts)) == B, "rows must be distinct"
    rows_sha = [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]
    file_sha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    rec = {"model": sys.argv[1], "revision": sys.argv[2], "source": "wikitext-2-raw-v1 test via step_decomp._k8_window", "batch": B,
           "prompt_len": 512, "corpus_tokens": int(ids.numel()), "row_step": int(step), "rows_sha256": rows_sha, "prompts_sha256": file_sha,
           "prompts": prompts}
    json.dump(rec, open(f"/root/p37/prompts_b{B}.json", "w"))
    print(f"PROMPTS B={B} corpus={ids.numel()} step={step} sha={file_sha}")
PYP
grep -a PROMPTS logs/prompts.log | tee -a summary.txt
PSHA1=$(python -c "import json; print(json.load(open('/root/p37/prompts_b1.json'))['prompts_sha256'])"); PSHA16=$(python -c "import json; print(json.load(open('/root/p37/prompts_b16.json'))['prompts_sha256'])")
[ "$VLLM_OK" = "1" ] && { fetch $GPTQ_MID $GPTQ_REV gptq 1800 || { echo "DL FAIL (gptq) -- NO COMPARATOR"; VLLM_OK=0; }; }
# ---------------------------------------------------------------- helpers
vram_start(){ ( while :; do echo "$(date -u +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits)"; sleep 1; done ) > $W/vram_$1.txt 2>/dev/null & echo $!; }
vram_stop(){ kill $1 2>/dev/null; wait $1 2>/dev/null; }
skip(){ case " $SKIP " in *" $1 "*) return 0;; *) return 1;; esac; }
fenv(){ case "$1" in 0) echo "0 0 0";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
# e4b ARM B EXP CALIBATTN FUSE LOOP [extra env...]  -- bo7b's arm() with the loop selectable; receipt e4b_b${B}_${ARM}.json
e4b_arm(){ local ARM=$1 B=$2 EXP=$3 CA=$4 FU=$5 LOOP=$6; shift 6; read G R E <<<"$(fenv $FU)"
  skip $ARM && { say "skip e4b/$ARM B=$B"; return 0; }
  local AL=1800; case "$*" in *E4B_SERVE_EXP_INT4_CALIB=1*) AL=5400;; esac
  say "arm e4b/$ARM (B=$B exp=$EXP calib=$CA fuse=$FU loop=$LOOP $* alarm=$AL prompts_sha=$([ $B = 1 ] && echo $PSHA1 || echo $PSHA16))"
  local sp; sp=$(vram_start e4b_b${B}_$ARM)
  env "$@" PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$BF16_MID" \
      E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e "alarm $AL; exec @ARGV" python $W/step_decomp.py --model "$BF16_MID" --arena "$QA" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop $LOOP --b1d-timed --no-fuse-qkv --out $W/e4b_b${B}_$ARM.json > logs/run_e4b_b${B}_$ARM.log 2>&1
  local rc=$?; vram_stop $sp
  if [ $rc -eq 142 ] && [ ! -s $W/e4b_b${B}_$ARM.json ]; then python -c "import json; json.dump({'engine':'e4b','arm':'$ARM','batch':$B,'status':'alarm','reason':'arm alarm $AL s'}, open('$W/e4b_b${B}_$ARM.json','w'))"; fi
  grep -aE "B1D_TIMED|BV3_|INT4EXP|ATTNINT4|gptq /|REFUSED|Error" logs/run_e4b_b${B}_$ARM.log | tail -3 | sed "s/^/    /"
  { echo -n "e4b/$ARM B=$B rc=$rc "; grep -aE "B1D_TIMED|BV3_" logs/run_e4b_b${B}_$ARM.log | tail -1 | cut -c1-300; echo; } >> summary.txt
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader; }
# vllm ARM B  -- p37_vllm.py in the venv; identical prompt ids from prompts_b${B}.json; receipt vllm_b${B}_${ARM}.json
vllm_arm(){ local ARM=$1 B=$2; [ "$VLLM_OK" = "1" ] || { echo "vllm/$ARM B=$B SKIPPED (no comparator)" >> summary.txt; return 0; }
  skip $ARM && { say "skip vllm/$ARM B=$B"; return 0; }
  say "arm vllm/$ARM (B=$B vllm==$VLLM_VER alarm=2400 prompts_sha=$([ $B = 1 ] && echo $PSHA1 || echo $PSHA16))"
  local sp; sp=$(vram_start vllm_b${B}_$ARM)
  env P37_ARM=$ARM P37_BATCH=$B P37_PROMPTS=$W/prompts_b$B.json P37_MODEL=$GPTQ_MID P37_REV=$GPTQ_REV P37_OUT=$W/vllm_b${B}_$ARM.json VLLM_LOGGING_LEVEL=INFO \
    perl -e 'alarm 2400; exec @ARGV' $W/venv-vllm/bin/python $W/p37_vllm.py > logs/run_vllm_b${B}_$ARM.log 2>&1
  local rc=$?; vram_stop $sp
  if [ ! -s $W/vllm_b${B}_$ARM.json ]; then local why; why=$(grep -aE "Engine core|CUDA|Error|error" logs/run_vllm_b${B}_$ARM.log | tail -2 | tr '\n' ' ' | cut -c1-400)
    python -c "import json,sys; json.dump({'engine':'vllm','arm':'$ARM','batch':$B,'status':('alarm' if $rc==142 else 'vllm_init_failed'),'rc':$rc,'reason':sys.argv[1]}, open('$W/vllm_b${B}_$ARM.json','w'))" "$why"; fi
  grep -aE "P37VLLM|Marlin|MARLIN|Capturing CUDA graphs|Actual usage|Available KV cache|Engine core" logs/run_vllm_b${B}_$ARM.log | grep -v "^\s*$" | tail -4 | cut -c1-240 | sed "s/^/    /"
  { echo -n "vllm/$ARM B=$B rc=$rc "; grep -a "P37VLLM" logs/run_vllm_b${B}_$ARM.log | tail -1 | cut -c1-300; echo; } >> summary.txt; }
# ---------------------------------------------------------------- arms, in the pre-registered order (e4b control -> vLLM block -> e4b licensed -> repeats -> e4b eager)
CAL="E4B_SERVE_EXP_INT4_CALIB=1"
for B in 1 16; do e4b_arm nf4_r1 $B 0 0 0 graph; done
for B in 1 16; do vllm_arm graph_r1 $B; vllm_arm eager $B; vllm_arm fp8kv $B; done
for B in 1 16; do e4b_arm lic_r1 $B 1 1 all graph $CAL E4B_CALIB_NSEQ=128; done
for B in 1 16; do e4b_arm nf4_r2 $B 0 0 0 graph; done
for B in 1 16; do vllm_arm graph_r2 $B; done
e4b_arm lic_r2 1 1 1 all graph $CAL E4B_CALIB_NSEQ=128
e4b_arm lic_eager 1 1 1 all eager $CAL E4B_CALIB_NSEQ=128
# ---------------------------------------------------------------- reduce, summarise, mark
say "reduce"
python $W/p37_reduce.py $W --md $W/RESULTS-p37.md | tee RESULTS.txt
echo "----- summary.txt -----"; cat summary.txt; echo "----- versions.txt -----"; cat versions.txt
[ "$VLLM_OK" = "1" ] || echo "NO COMPARATOR: the vLLM side did not install/import; e4b arms only -- not an H2H" | tee -a summary.txt
say "TP_DONE"; touch TP_DONE
