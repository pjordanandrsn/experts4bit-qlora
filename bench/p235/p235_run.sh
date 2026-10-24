#!/bin/bash
# bench/p235/p235_run.sh -- lane P235, BOX side (bench/p235/PREREG-p235.md): P58's runner (bench/p58/p58_run.sh) by
# named edits. Started detached by p235_drive.sh with the run's nonce; P235_RUN_NONCE first, then
# P235_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and P235_SUCCESS.<nonce> only on clean completion.
#
# P58's same-box vLLM head-to-head re-read at the release pins, each a PyPI release whose version AND provenance
# provenance.py checks before any arm: experts4bit-qlora 0.58.0 and grouped-nf4-gemm 0.51.0 (a failure is rc 9; the
# launch commit E4B_SHA is the harness's, recorded and never installed), transformers 5.17.0, bitsandbytes 0.50.1, and
# TWO vLLM builds in their own venvs, both constants: the PRIMARY 0.32.0 and the SECONDARY 0.30.0 (P58's primary). A
# vLLM build that fails its install, its provenance or its release proof loses its rows; the e4b arms continue. P58's arms, alarms, STOP
# rules and reducer call, unchanged; nothing here changes a default. The runner does not collect the GPU's UUID.
set -uo pipefail
W=/root/p235; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p235: $*"; }
NONCE=${P235_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P235_RUN_NONCE.tmp && mv $W/P235_RUN_NONCE.tmp $W/P235_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P235_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P235_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P235_RUN_ID P235_DEADLINE_EPOCH P235_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=24 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
# every python this runner starts writes and reads bytecode under a fresh, empty prefix: no in-tree .pyc is ever loaded
[ -e "$W/pycache-$NONCE" ] && { say "refusing: $W/pycache-$NONCE exists"; finish 78; }
mkdir "$W/pycache-$NONCE" && export PYTHONPYCACHEPREFIX=$W/pycache-$NONCE RELEASE_PROOF_FRESH_PYCACHE=$W/pycache-$NONCE || { say "refusing: no pycache prefix"; finish 78; }
echo "PYTHONPYCACHEPREFIX $PYTHONPYCACHEPREFIX" >> $W/versions.txt
MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39; export E4B_MODEL_ID=$MID
GPTQ_MID=Qwen/Qwen3-30B-A3B-GPTQ-Int4; GPTQ_REV=9b534e4318b7ebc3c961a839f13eb18b1833f441   # P37's comparator checkpoint, unchanged
VLLM_PRIMARY=0.32.0; VLLM_SECONDARY=0.30.0   # provenance.py's VLLM pins, constants: a newer vLLM is a new reading
export P37_INSTANCE_ID=$P235_INSTANCE_ID    # p37_vllm.py (staged byte-identical) records the instance under P37's env name
: > summary.txt; echo "$P235_INSTANCE_ID" > INSTANCE_ID
# ---- staged pieces, byte-for-byte
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py p37_vllm.py p235_reduce.py provenance.py release_proof.py staged.sha256; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p235/staged.sha256"; finish 9; }
# ---- deadline guard + per-arm alarm (P54's): never start an arm that cannot finish 10 min before teardown
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P235_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P235_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P235_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
# ---- install (image python): experts4bit-qlora 0.58.0 and grouped-nf4-gemm 0.51.0 as PyPI wheels + P235's toolchain
# pins, --no-compile (install-time .pyc would land under the prefix, outside the wheel's RECORD); then provenance.py,
# unconditionally, before anything is timed (a version string is not a release)
export DEBIAN_FRONTEND=noninteractive
say "install experts4bit-qlora==0.58.0 + grouped-nf4-gemm==0.51.0 from PyPI (image python; P235 pins); harness commit $E4B_SHA, recorded, never installed"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --no-compile --index-url https://pypi.org/simple --prefer-binary --only-binary experts4bit-qlora "experts4bit-qlora==0.58.0" \
  "transformers==5.17.0" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --no-compile --index-url https://pypi.org/simple --only-binary grouped-nf4-gemm --force-reinstall --no-deps "grouped-nf4-gemm==0.51.0" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
python -I $W/provenance.py check experts4bit-qlora=0.58.0 grouped-nf4-gemm=0.51.0 --out $W/provenance.json > logs/provenance_e4b.log 2>&1 || { cat logs/provenance_e4b.log; say "PROVENANCE FAIL (e4b/gnf4)"; finish 9; }
cat logs/provenance_e4b.log
PYTHONPATH=$W/hook python - <<'PYT' || { say "TRIPWIRE FAIL (e4b)"; finish 9; }
import site, os, inspect, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
import experts4bit_qlora
assert experts4bit_qlora.__version__ == "0.58.0", f"e4b {experts4bit_qlora.__version__} is not the 0.58.0 release P235 reads"
assert md.version("grouped-nf4-gemm") == "0.51.0", f"grouped-nf4-gemm {md.version('grouped-nf4-gemm')} is not the 0.51.0 release P235 reads"
from experts4bit_qlora.engines.int4_attn import Int4Linear, _smallm_kernels
assert getattr(Int4Linear, "SMALLM_ROWS_MAX", None) == 16, "e4b cut lacks the K16 route (#578)"
assert hasattr(Int4Linear, "fuse") and hasattr(Int4Linear, "from_packed"), "e4b cut lacks Int4Linear.fuse (#651): the B=1 arm would refuse"
_smallm_kernels()   # the installed gnf4 must carry int4_smallm, or the route would refuse at enable time
import int4_b32
assert hasattr(int4_b32, "gemv_counter_len") and int4_b32.gemv_fused_reduce_default() is False, "gnf4 cut is not K17's, or the fused reduce is defaulted on (P235 runs the two-launch default, as P58)"
import usercustomize  # noqa: F401
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p235/versions.txt", "a").write(f"e4b {e.__version__} (PyPI; provenance.json) harness@{os.environ['E4B_SHA']}\ngnf4 {md.version('grouped-nf4-gemm')} (PyPI; provenance.json)\ntorch(e4b) {torch.__version__}\ntriton(e4b) {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
print("tripwire OK (e4b):", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__, "triton", triton.__version__)
PYT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt; lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt
cat /sys/fs/cgroup/memory.max 2>/dev/null | sed "s/^/cgroup memory.max /" | tee -a forensics.txt; nvcc --version 2>/dev/null | tail -1 | tee -a forensics.txt
python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
df -h /root | tail -1
# ---- vLLM venvs: PRIMARY 0.32.0, SECONDARY 0.30.0 (the constants above), each from PyPI with its provenance checked
# here and its release proof right before its arms.
# Each install or provenance failure is recorded and the lane continues (P37: a vLLM install failure cannot void the
# e4b arms); that build's rows read SKIPPED (no comparator).
declare -A VOK VVER; VVER[primary]=$VLLM_PRIMARY; VVER[secondary]=$VLLM_SECONDARY
[ "$VLLM_PRIMARY" = "$VLLM_SECONDARY" ] && { say "primary == secondary ($VLLM_PRIMARY): one venv, the secondary rows are skipped"; VOK[secondary]=0; }
vllm_install(){ local TAG=$1 VER=$2; [ -z "$VER" ] && { VOK[$TAG]=0; return; }
  say "install vllm==$VER into $W/venv-$TAG"
  python -m venv $W/venv-$TAG && perl -e 'alarm 2700; exec @ARGV' $W/venv-$TAG/bin/pip install -q --no-input --no-compile --no-cache-dir --index-url https://pypi.org/simple --only-binary vllm "vllm==$VER" "huggingface_hub>=0.23" > logs/pip_vllm_$TAG.log 2>&1
  local rc=$?; echo "pip(vllm $TAG $VER) rc=$rc" | tee -a summary.txt; [ $rc -ne 0 ] && { tail -6 logs/pip_vllm_$TAG.log; say "PIP FAIL (vllm $TAG) -- e4b arms still run; NO $TAG COMPARATOR"; VOK[$TAG]=0; return; }
  $W/venv-$TAG/bin/python -I $W/provenance.py check "vllm=$VER" --out $W/provenance_vllm_$TAG.json > logs/provenance_vllm_$TAG.log 2>&1 \
    || { cat logs/provenance_vllm_$TAG.log; say "PROVENANCE FAIL (vllm $TAG) -- e4b arms still run; NO $TAG COMPARATOR"; echo "PROVENANCE FAIL (vllm $TAG $VER): its rows are refused" >> summary.txt; VOK[$TAG]=0; return; }
  $W/venv-$TAG/bin/python - "$TAG" <<'PYV' >> logs/tripwire_vllm_$TAG.log 2>&1 || { say "VLLM IMPORT FAIL ($TAG) -- NO $TAG COMPARATOR"; tail -5 logs/tripwire_vllm_$TAG.log; VOK[$TAG]=0; return; }
import sys, vllm, torch
tri = None
try:
    import triton; tri = triton.__version__
except Exception: pass
print("tripwire OK (vllm", sys.argv[1] + "):", vllm.__version__, "torch", torch.__version__, "triton", tri)
open("/root/p235/versions.txt", "a").write(f"vllm({sys.argv[1]}) {vllm.__version__}\ntorch(vllm {sys.argv[1]}) {torch.__version__}\ntriton(vllm {sys.argv[1]}) {tri}\n")
PYV
  tail -1 logs/tripwire_vllm_$TAG.log; VOK[$TAG]=1; }
vllm_install primary "$VLLM_PRIMARY"
[ "${VOK[secondary]:-1}" = 0 ] || vllm_install secondary "$VLLM_SECONDARY"
# ---- fetch (pinned), bake (bo7's k8_bake.py), prompts (step_decomp's own window, identical ids for both engines) -- as P37/P54
say "fetch $MID @ $REV"
perl -e 'alarm 4800; exec @ARGV' python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL (bf16)"; finish 11; }
say "bake NF4 arena"; mkdir -p $W/work_qwen3
bake_fail(){ say "BAKE FAIL ($1) -- k8_bake.py's record (err + traceback), so the reason travels with the receipt:"
  python -c "import json,sys; r=json.load(open('$W/work_qwen3/bake.json')); print('BAKE_ERR', r.get('status'), r.get('step')); print(r.get('err')); print(r.get('tb'))" 2>/dev/null | tee -a logs/bake.log | tail -n 25 | cut -c1-400
  finish 12; }
K8_MODEL="$MID" K8_WORK="$W/work_qwen3" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; bake_fail "rc"; }
QA=$W/work_qwen3/nf4.arena; [ -e "$QA" ] || bake_fail "no arena"
say "prompt dump: step_decomp._k8_window at B=1 and B=16 -> prompts_b{1,16}.json"
python - "$MID" "$REV" <<'PYP' > logs/prompts.log 2>&1 || { tail -5 logs/prompts.log; say "PROMPT DUMP FAIL"; finish 13; }
import sys, json, hashlib, types
sys.path.insert(0, "/root/p235")
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
    json.dump(rec, open(f"/root/p235/prompts_b{B}.json", "w"))
    print(f"PROMPTS B={B} corpus={ids.numel()} step={step} sha={file_sha}")
PYP
grep -a PROMPTS logs/prompts.log | tee -a summary.txt
if [ "${VOK[primary]:-0}" = 1 ] || [ "${VOK[secondary]:-0}" = 1 ]; then
  say "fetch $GPTQ_MID @ $GPTQ_REV"
  perl -e 'alarm 1800; exec @ARGV' python -c "from huggingface_hub import snapshot_download as s; print(s('$GPTQ_MID', revision='$GPTQ_REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch_gptq.log 2>&1 \
    || { tail -2 logs/fetch_gptq.log; say "DL FAIL (gptq) -- NO COMPARATOR"; echo "GPTQ DL FAIL: no vLLM arms" >> summary.txt; VOK[primary]=0; VOK[secondary]=0; }
fi
df -h /root | tail -1
# ---- helpers (P54's speed_arm shape: receipt e4b_b${B}_${TAG}.json, log logs/run_e4b_b${B}_${TAG}.log)
vram_start(){ ( while :; do echo "$(date -u +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits)"; sleep 1; done ) > $W/vram_$1.txt 2>/dev/null & echo $!; }
vram_stop(){ kill $1 2>/dev/null; wait $1 2>/dev/null; }
# e4b_arm TAG B EXP FUSE LOOP   -- EXP=0: NF4 control (no folds); EXP=1: P54's int4 stack (RTN experts, uncalibrated int4
# attention, glue r1/r2 + router epilogue); FUSE=--fuse-qkv or --no-fuse-qkv (the LAST flag wins on step_decomp's argparse line)
e4b_arm(){ local TAG=$1 B=$2 EXP=$3 FUSE=$4 LOOP=$5; local G=1 R=1 E=1 AI=1; [ "$EXP" = 0 ] && { G=0; R=0; E=0; AI=0; }
  local AL; AL=$(arm_alarm 1800); local sp; sp=$(vram_start e4b_b${B}_$TAG)
  say "arm e4b/$TAG (B=$B exp=$EXP attn_int4=$AI fuse=$FUSE loop=$LOOP alarm=$AL)"
  { echo "P235 arm=e4b_b${B}_$TAG gnf4=0.51.0 at=$(date -u +%FT%TZ)"; } > logs/run_e4b_b${B}_$TAG.log
  env E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4=$AI E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e "alarm $AL; exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop $LOOP --b1d-timed --no-fuse-qkv $FUSE \
      --out $W/e4b_b${B}_$TAG.json >> logs/run_e4b_b${B}_$TAG.log 2>&1
  local rc=$?; vram_stop $sp
  if [ $rc -eq 142 ] && [ ! -s $W/e4b_b${B}_$TAG.json ]; then python -c "import json; json.dump({'engine':'e4b','arm':'$TAG','batch':$B,'status':'alarm','reason':'arm alarm $AL s'}, open('$W/e4b_b${B}_$TAG.json','w'))"; fi
  grep -aE "B1D_TIMED|BV3_|INT4EXP|ATTNINT4|fused q/k/v|REFUSED|Error" logs/run_e4b_b${B}_$TAG.log | tail -4 | sed "s/^/    /"
  { echo -n "e4b/$TAG B=$B rc=$rc "; grep -aE "B1D_TIMED|BV3_" logs/run_e4b_b${B}_$TAG.log | tail -1 | cut -c1-240; echo; } >> summary.txt
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; return $rc; }
# vllm_arm VTAG ARM B  -- p37_vllm.py (staged byte-identical; P37_* env names) in venv-$VTAG; receipt vllm_${VTAG}_b${B}_${ARM}.json
vllm_arm(){ local VT=$1 ARM=$2 B=$3; [ "${VOK[$VT]:-0}" = 1 ] || { echo "vllm-$VT/$ARM B=$B SKIPPED (no $VT comparator)" >> summary.txt; return 0; }
  local AL; AL=$(arm_alarm 2400); local sp; sp=$(vram_start vllm_${VT}_b${B}_$ARM)
  say "arm vllm-$VT/$ARM (B=$B vllm==${VVER[$VT]} alarm=$AL)"
  env P37_ARM=$ARM P37_BATCH=$B P37_PROMPTS=$W/prompts_b$B.json P37_MODEL=$GPTQ_MID P37_REV=$GPTQ_REV P37_OUT=$W/vllm_${VT}_b${B}_$ARM.json VLLM_LOGGING_LEVEL=INFO \
    perl -e "alarm $AL; exec @ARGV" $W/venv-$VT/bin/python $W/p37_vllm.py > logs/run_vllm_${VT}_b${B}_$ARM.log 2>&1
  local rc=$?; vram_stop $sp
  if [ ! -s $W/vllm_${VT}_b${B}_$ARM.json ]; then local why; why=$(grep -aE "Engine core|CUDA|Error|error" logs/run_vllm_${VT}_b${B}_$ARM.log | tail -2 | tr '\n' ' ' | cut -c1-400)
    python -c "import json,sys; json.dump({'engine':'vllm','arm':'$ARM','batch':$B,'vllm_version':'${VVER[$VT]}','status':('alarm' if $rc==142 else 'vllm_init_failed'),'rc':$rc,'reason':sys.argv[1]}, open('$W/vllm_${VT}_b${B}_$ARM.json','w'))" "$why"; fi
  grep -aE "P37VLLM|Marlin|MARLIN|Capturing CUDA graphs|Actual usage|Available KV cache|Engine core" logs/run_vllm_${VT}_b${B}_$ARM.log | grep -v "^\s*$" | tail -4 | cut -c1-240 | sed "s/^/    /"
  { echo -n "vllm-$VT/$ARM B=$B rc=$rc "; grep -a "P37VLLM" logs/run_vllm_${VT}_b${B}_$ARM.log | tail -1 | cut -c1-300; echo; } >> summary.txt; return $rc; }
# ---- arms, in the pre-registered order: e4b control -> e4b int4 (first draws) -> both vLLM primaries -> repeats -> eager pairing
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
# ---- release proofs: the installed bytes ARE the published wheels', read as the arms import them (cwd $W,
# PYTHONPATH=$W/hook, the arms' python, no -I), after every install and right before that build's first arm; a proof
# file whose lines are not all pycache_fresh true (or not one per pin) refuses like the proof itself
proof_fresh(){ python -c 'import json, sys; rows = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]; sys.exit(0 if len(rows) == int(sys.argv[2]) and all(r.get("pycache_fresh") is True for r in rows) else 1)' "$1" "$2"; }
(cd $W && PYTHONPATH=$W/hook python $W/release_proof.py experts4bit-qlora==0.58.0 grouped-nf4-gemm==0.51.0) > $W/release_proof.jsonl 2> logs/release_proof_e4b.log || { cat logs/release_proof_e4b.log; say "RELEASE PROOF FAIL (e4b/gnf4)"; finish 9; }
proof_fresh $W/release_proof.jsonl 2 || { cat $W/release_proof.jsonl; say "RELEASE PROOF FAIL (e4b/gnf4): pycache_fresh is not true"; finish 9; }
can_run 900 e4b_nf4_r1_b1     && { e4b_arm nf4_r1  1  0 --no-fuse-qkv graph; rec $?; }
can_run 900 e4b_nf4_r1_b16    && { e4b_arm nf4_r1  16 0 --no-fuse-qkv graph; rec $?; }
can_run 900 e4b_int4_r1_b1    && { e4b_arm int4_r1 1  1 --fuse-qkv    graph; rec $?; }   # P54's licensed B=1 lever: fused q/k/v
can_run 900 e4b_int4_r1_b16   && { e4b_arm int4_r1 16 1 --no-fuse-qkv graph; rec $?; }   # P54: no B=16 default for the fusion
for VT in primary secondary; do [ "${VOK[$VT]:-0}" = 1 ] || continue   # no venv: its rows already read SKIPPED
  (cd $W && PYTHONPATH=$W/hook $W/venv-$VT/bin/python $W/release_proof.py "vllm==${VVER[$VT]}") > $W/release_proof_vllm_$VT.jsonl 2> logs/release_proof_vllm_$VT.log \
    || { cat logs/release_proof_vllm_$VT.log; say "RELEASE PROOF FAIL (vllm $VT) -- e4b arms still run; NO $VT COMPARATOR"; echo "RELEASE PROOF FAIL (vllm $VT ${VVER[$VT]}): its rows are refused" >> summary.txt; VOK[$VT]=0; }
  [ "${VOK[$VT]}" = 0 ] || proof_fresh $W/release_proof_vllm_$VT.jsonl 1 || { say "RELEASE PROOF FAIL (vllm $VT): pycache_fresh is not true -- NO $VT COMPARATOR"; echo "RELEASE PROOF FAIL (vllm $VT ${VVER[$VT]}): pycache_fresh not true, its rows are refused" >> summary.txt; VOK[$VT]=0; }
done
for VT in primary secondary; do for B in 1 16; do can_run 900 vllm_${VT}_graph_r1_b$B && { vllm_arm $VT graph_r1 $B; rec $?; }; done; done
can_run 900 e4b_nf4_r2_b1     && { e4b_arm nf4_r2  1  0 --no-fuse-qkv graph; rec $?; }
can_run 900 e4b_nf4_r2_b16    && { e4b_arm nf4_r2  16 0 --no-fuse-qkv graph; rec $?; }
can_run 900 e4b_int4_r2_b1    && { e4b_arm int4_r2 1  1 --fuse-qkv    graph; rec $?; }
can_run 900 e4b_int4_r2_b16   && { e4b_arm int4_r2 16 1 --no-fuse-qkv graph; rec $?; }
for VT in primary secondary; do for B in 1 16; do can_run 900 vllm_${VT}_graph_r2_b$B && { vllm_arm $VT graph_r2 $B; rec $?; }; done; done
can_run 900 vllm_primary_eager_b1 && { vllm_arm primary eager 1; rec $?; }            # secondary rows: the eager pairing at B=1
can_run 900 e4b_int4_eager_b1     && { e4b_arm int4_eager 1 1 --fuse-qkv eager; rec $?; }
# ---- engagement checks the reducer also applies (recorded here so summary.txt carries them)
grep -aqE "fused q/k/v projections on 48 attention modules" logs/run_e4b_b1_int4_r1.log 2>/dev/null || { echo "FQKV NOT ENGAGED in e4b_b1_int4_r1" >> summary.txt; rec 43; }
grep -aqE "fused q/k/v projections on" logs/run_e4b_b16_int4_r1.log 2>/dev/null && { echo "FQKV LEAKED into e4b_b16_int4_r1" >> summary.txt; rec 44; }
grep -aqE "INT4EXP|ATTNINT4" logs/run_e4b_b1_nf4_r1.log 2>/dev/null && { echo "INT4 LEAKED into the NF4 control" >> summary.txt; rec 44; }
# ---- reduce, summarise, mark
say "reduce"; python $W/p235_reduce.py $W --md $W/RESULTS-p235-generated.md | tee RESULTS.txt
say "----- summary -----"; cat summary.txt; echo "----- versions -----"; cat versions.txt
finish "$rc_any"
