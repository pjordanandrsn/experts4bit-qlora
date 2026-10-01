#!/bin/bash
# bench/p86/p86_run.sh -- lane P86, BOX side (bench/p86/PREREG-p86.md; e4b#564). Started detached by p86_drive.sh with
# the run's nonce; P86_RUN_NONCE first, then P86_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P86_SUCCESS.<nonce> only when the reducer ran.
#
# Where do the two engines' decode steps go, kernel by kernel, on ONE box with the same prompt token ids? P58 measured
# vLLM 0.30.0 at 1.087x (B=1) and 1.396x (B=16) e4b's int4 stack, end to end. This lane keeps P58's protocol and adds
# a per-kernel census on BOTH sides:
#   e4b   -- the current release's int4 stack (P58's env: RTN int4 experts, uncalibrated int4 attention, glue r1/r2,
#            router epilogue; fused q/k/v at B=1 and B=16, the default since P59), through P58's harness
#            (bench/p39/step_decomp.py + bench/p42 hook); timed by the graph-replay window, censused by
#            --replay-profile-out (P42's protocol: 8 profiled replays after the window);
#   vLLM  -- 0.30.0 (P58's comparator), Qwen/Qwen3-30B-A3B-GPTQ-Int4 (Marlin), timed by P37's slope arm
#            (bench/h2h-20260905/p37/p37_vllm.py, unchanged) and censused by p86_vllm_census.py (the same slope
#            applied to the GPU's kernel record, the engine in-process).
# Each timed arm is drawn twice (A/A); each census once. The reducer applies the pre-registered reading.
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P86_GPU_CLASS P86_MIN_DRIVER P86_VLLM P86_MIN_DISK_GB P86_REHEARSAL.
# P86_PROVE=1 is the PROVING RUN: the refusals, both installs with their tripwires, the reducer self-test, an egress
# probe, and the vLLM census arm itself on vLLM 0.30.0 with a small model (Qwen/Qwen3-0.6B, B=4, an 8 -> 24-token
# slope), so the in-process profiler is proven on this build, driver and card before the reading; no 30B model.
set -uo pipefail
W=/root/p86; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p86: $*"; }
NONCE=${P86_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P86_RUN_NONCE.tmp && mv $W/P86_RUN_NONCE.tmp $W/P86_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P86_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P86_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P86_RUN_ID P86_DEADLINE_EPOCH P86_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=9407d499a4d1e0fe8c22b050878a9f869b385b45     # grouped-nf4-gemm v0.33.7, e4b 0.37.8's floor: a registered constant
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=24 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39; export E4B_MODEL_ID=$MID
GPTQ_MID=Qwen/Qwen3-30B-A3B-GPTQ-Int4; GPTQ_REV=9b534e4318b7ebc3c961a839f13eb18b1833f441   # P37's / P58's comparator checkpoint
GPU_CLASS=${P86_GPU_CLASS:-5090}; MIN_DRIVER=${P86_MIN_DRIVER:-580}; VLLM_VER=${P86_VLLM:-0.30.0}
MIN_DISK_GB=${P86_MIN_DISK_GB:-200}; REHEARSAL=${P86_REHEARSAL:-0}
export P37_INSTANCE_ID=$P86_INSTANCE_ID    # p37_vllm.py (staged byte-identical) records the instance under P37's env name
: > summary.txt; echo "$P86_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA vllm=$VLLM_VER gpu_class=$GPU_CLASS min_driver=$MIN_DRIVER min_disk_gb=$MIN_DISK_GB" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DRIVER" != 580 ] || [ "$VLLM_VER" != 0.30.0 ] || [ "$MIN_DISK_GB" != 200 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py p37_vllm.py p86_vllm_census.py p86_reduce.py p42_reduce.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p86/staged.sha256"; finish 9; }
python $W/p86_reduce.py --self-test | tee -a summary.txt || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- refusals before anything is installed or fetched: the card class, the driver vLLM's wheels need, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); DRV_MAJOR=${DRV%%.*}
[ "${DRV_MAJOR:-0}" -ge "$MIN_DRIVER" ] 2>/dev/null || { say "REFUSED: driver $DRV < $MIN_DRIVER (vLLM $VLLM_VER's wheels are CUDA 13.0)"; echo "refused: driver $DRV" > REFUSAL; finish 18; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
# ---- deadline guard + per-arm alarm (P54's): never start an arm that cannot finish 10 min before teardown
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P86_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P86_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P86_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
# ---- install: e4b at the launch commit + P37's toolchain pins (image python); gnf4 at e4b's floor
export DEBIAN_FRONTEND=noninteractive
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; P37 pins)"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
PYTHONPATH=$W/hook WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL (e4b)"; finish 9; }
import site, os, json, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
from experts4bit_qlora.engines.int4_attn import Int4Linear, _smallm_kernels
assert getattr(Int4Linear, "SMALLM_ROWS_MAX", None) == 16 and hasattr(Int4Linear, "fuse"), "e4b cut lacks the K16 route or Int4Linear.fuse"
_smallm_kernels()
import int4_b32
assert int4_b32.gemv_fused_reduce_default() is False, "the fused reduce is defaulted on (the census reads the two-launch default)"
import usercustomize  # noqa: F401
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p86/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch(e4b) {torch.__version__}\ntriton(e4b) {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
print("tripwire OK (e4b):", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
VOK=1
say "install vllm==$VLLM_VER into $W/venv-vllm"
python -m venv $W/venv-vllm && perl -e 'alarm 2700; exec @ARGV' $W/venv-vllm/bin/pip install -q --no-input --no-cache-dir "vllm==$VLLM_VER" "huggingface_hub>=0.23" > logs/pip_vllm.log 2>&1
rc=$?; echo "pip(vllm $VLLM_VER) rc=$rc" | tee -a summary.txt
if [ $rc -ne 0 ]; then tail -6 logs/pip_vllm.log; say "PIP FAIL (vllm) -- no comparator"; VOK=0; fi
if [ $VOK = 1 ]; then
  $W/venv-vllm/bin/python -c "import vllm, torch; open('/root/p86/versions.txt','a').write(f'vllm {vllm.__version__}\ntorch(vllm) {torch.__version__}\n'); print('tripwire OK (vllm):', vllm.__version__, torch.__version__)" > logs/tripwire_vllm.log 2>&1 \
    || { say "VLLM IMPORT FAIL -- no comparator"; tail -5 logs/tripwire_vllm.log; VOK=0; }
  [ $VOK = 1 ] && tail -1 logs/tripwire_vllm.log | tee -a summary.txt
fi
[ $VOK = 1 ] || { say "no vLLM comparator: the lane cannot answer its question"; echo "NO COMPARATOR" >> summary.txt; finish 19; }
if [ "${P86_PROVE:-0}" = 1 ]; then
  echo "PROVE -- the proving run: both installs tripwired; the census arm on vLLM $VLLM_VER with a small model" | tee -a summary.txt
  python - <<'PYP'
import json, hashlib, random
random.seed(0)
prompts = [[random.randrange(1000, 30000) for _ in range(64)] for _ in range(4)]
json.dump({"batch": 4, "prompts": prompts, "prompts_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest()},
          open("/root/p86/prompts_prove.json", "w"))
PYP
  env P86_BATCH=4 P86_PROMPTS=$W/prompts_prove.json P86_MODEL=Qwen/Qwen3-0.6B P86_REV=main P86_OUT=$W/vllm_census_prove.json \
      P86_SHORT=8 P86_LONG=24 P86_MAX_LEN=512 perl -e 'alarm 1200; exec @ARGV' $W/venv-vllm/bin/python $W/p86_vllm_census.py > logs/run_vllm_census_prove.log 2>&1
  rc=$?; { echo -n "PROVE census rc=$rc "; grep -a "P86VLLMCENSUS" logs/run_vllm_census_prove.log | tail -1 | cut -c1-240; echo; } | tee -a summary.txt
  python - <<'PYC' 2>&1 | tee -a summary.txt
import json
d = json.load(open("/root/p86/vllm_census_prove.json"))
dec = [k for k in d["kernels"] if k["calls_long"] > k["calls_short"]]
per_layer = [k for k in dec if abs(k["calls_per_step"] - 28) < 0.5]     # Qwen3-0.6B has 28 layers
ok = d["in_process"] and d["decode_steps"] == 16 and len(dec) >= 10 and len(per_layer) >= 3 and sum(k["us_per_step"] for k in dec) > 0
print(f"PROVE census {'OK' if ok else 'FAILED'}: in_process={d['in_process']} decode kernels={len(dec)} per-layer={len(per_layer)} "
      f"decode ms/step={sum(k['us_per_step'] for k in dec) / 1e3:.3f} cudagraph={d.get('cudagraph_mode')}")
raise SystemExit(0 if ok else 1)
PYC
  [ "${PIPESTATUS[0]}" = 0 ] && [ "$rc" = 0 ] || { say "PROVE: the census arm did not prove"; finish 23; }
  say "PROVE: HF CDN egress probe (50 MB range, 20 s cap; recorded, not a refusal)"
  python - <<'PYE' 2>&1 | tail -1 | tee -a summary.txt forensics.txt
import time, urllib.request
t = time.time()
try:
    r = urllib.request.urlopen(urllib.request.Request("https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors",
                                                      headers={"Range": "bytes=0-52428799"}), timeout=20)
    n = len(r.read())
    print(f"PROVE hf_cdn_mbps={n / (time.time() - t) / 1e6:.1f} bytes={n}")
except Exception as e:
    print(f"PROVE hf_cdn_probe_failed {type(e).__name__}: {str(e)[:120]}")
PYE
  : > PROVED; finish 0
fi
# ---- fetch (pinned), bake (P39's k8_bake.py), prompts (step_decomp's own window, identical ids for both engines) -- P58's
say "fetch $MID @ $REV"
perl -e 'alarm 4800; exec @ARGV' python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL (bf16)"; finish 11; }
say "fetch $GPTQ_MID @ $GPTQ_REV"
perl -e 'alarm 1800; exec @ARGV' python -c "from huggingface_hub import snapshot_download as s; print(s('$GPTQ_MID', revision='$GPTQ_REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch_gptq.log 2>&1 || { tail -2 logs/fetch_gptq.log; say "DL FAIL (gptq)"; finish 11; }
say "bake NF4 arena"; mkdir -p $W/work_qwen3
K8_MODEL="$MID" K8_WORK="$W/work_qwen3" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
QA=$W/work_qwen3/nf4.arena; [ -e "$QA" ] || { say "BAKE FAIL: no arena"; finish 12; }
say "prompt dump: step_decomp._k8_window at B=1 and B=16 -> prompts_b{1,16}.json"
python - "$MID" "$REV" <<'PYP' > logs/prompts.log 2>&1 || { tail -5 logs/prompts.log; say "PROMPT DUMP FAIL"; finish 13; }
import sys, json, hashlib, types
sys.path.insert(0, "/root/p86")
import step_decomp
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], revision=sys.argv[2])
for B in (1, 16):
    a = types.SimpleNamespace(ppl_source="wikitext", ppl_chat=False, ppl_chat_suffix="", prompt_offset=0, prompt_span=0,
                              prompt_len=512, batch=B, ppl_steps=0)
    ids, step, prompts, _, ppl_sha = step_decomp._k8_window(a, tok)
    assert all(len(p) == 512 for p in prompts) and len(prompts) == B and len(set(tuple(p) for p in prompts)) == B
    rows_sha = [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]
    file_sha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    json.dump({"model": sys.argv[1], "revision": sys.argv[2], "batch": B, "prompt_len": 512, "row_step": int(step),
               "rows_sha256": rows_sha, "prompts_sha256": file_sha, "prompts": prompts}, open(f"/root/p86/prompts_b{B}.json", "w"))
    print(f"PROMPTS B={B} corpus={ids.numel()} step={step} sha={file_sha}")
PYP
grep -a PROMPTS logs/prompts.log | tee -a summary.txt
# ---- arms
# e4b_arm SUFFIX B CENSUS  -- P58's int4 env and command line, fused q/k/v at both batches (P54 B=1, P59 B=16); the
# first draw censuses (--replay-profile-out: 8 replays after the timed window, outside it)
e4b_arm(){ local SFX=$1 B=$2 CEN=$3; local AL; AL=$(arm_alarm 1800)
  say "arm e4b/int4$SFX (B=$B census=$CEN alarm=$AL)"
  { echo "P86 arm=e4b_b${B}_int4$SFX at=$(date -u +%FT%TZ)"; } > logs/run_e4b_b${B}_int4$SFX.log
  env E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1 \
    perl -e "alarm $AL; exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --fuse-qkv \
      ${CEN:+--replay-profile-out $W/logs/census_e4b_b$B.txt} --out $W/e4b_b${B}_int4$SFX.json >> logs/run_e4b_b${B}_int4$SFX.log 2>&1
  local rc=$?
  { echo -n "e4b/int4$SFX B=$B rc=$rc "; grep -aE "B1D_TIMED|BV3_" logs/run_e4b_b${B}_int4$SFX.log | tail -1 | cut -c1-240; echo; } >> summary.txt
  grep -aqE "fused q/k/v projections on 48 attention modules" logs/run_e4b_b${B}_int4$SFX.log || { echo "FQKV NOT ENGAGED in e4b_b${B}_int4$SFX" >> summary.txt; }
  return $rc; }
# vllm_time B SUFFIX -- P37's slope arm (graph_r1 settings), receipt vllm_graph_b${B}$SFX.json
vllm_time(){ local B=$1 SFX=$2; local AL; AL=$(arm_alarm 2400)
  say "arm vllm/graph$SFX (B=$B alarm=$AL)"
  env P37_ARM=graph_r1 P37_BATCH=$B P37_PROMPTS=$W/prompts_b$B.json P37_MODEL=$GPTQ_MID P37_REV=$GPTQ_REV P37_OUT=$W/vllm_graph_b$B$SFX.json VLLM_LOGGING_LEVEL=INFO \
    perl -e "alarm $AL; exec @ARGV" $W/venv-vllm/bin/python $W/p37_vllm.py > logs/run_vllm_graph_b$B$SFX.log 2>&1
  local rc=$?
  { echo -n "vllm/graph$SFX B=$B rc=$rc "; grep -a "P37VLLM" logs/run_vllm_graph_b$B$SFX.log | tail -1 | cut -c1-260; echo; } >> summary.txt; return $rc; }
# vllm_census B -- p86_vllm_census.py, receipt vllm_census_b$B.json
vllm_census(){ local B=$1; local AL; AL=$(arm_alarm 2400)
  say "arm vllm/census (B=$B alarm=$AL)"
  env P86_BATCH=$B P86_PROMPTS=$W/prompts_b$B.json P86_MODEL=$GPTQ_MID P86_REV=$GPTQ_REV P86_OUT=$W/vllm_census_b$B.json VLLM_LOGGING_LEVEL=INFO \
    perl -e "alarm $AL; exec @ARGV" $W/venv-vllm/bin/python $W/p86_vllm_census.py > logs/run_vllm_census_b$B.log 2>&1
  local rc=$?
  { echo -n "vllm/census B=$B rc=$rc "; grep -a "P86VLLMCENSUS" logs/run_vllm_census_b$B.log | tail -1 | cut -c1-260; echo; } >> summary.txt; return $rc; }
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
# the registered order: first draws (e4b with census, then vLLM timing), the vLLM censuses, then the second draws
for B in 16 1; do can_run 900 e4b_b$B && { e4b_arm "" $B 1; rec $?; }; done
for B in 16 1; do can_run 900 vllm_graph_b$B && { vllm_time $B ""; rec $?; }; done
for B in 16 1; do can_run 900 vllm_census_b$B && { vllm_census $B; rec $?; }; done
for B in 16 1; do can_run 900 e4b_b${B}_r2 && { e4b_arm _r2 $B ""; rec $?; }; done
for B in 16 1; do can_run 900 vllm_graph_b${B}_r2 && { vllm_time $B _r2; rec $?; }; done
say "reduce"; python $W/p86_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish "$rc_any"
