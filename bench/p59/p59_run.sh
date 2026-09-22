#!/bin/bash
# bench/p59/p59_run.sh -- lane P59, BOX side (bench/p59/P59-PREREG.md). Started detached by p59_drive.sh with the
# run's nonce; P59_RUN_NONCE first, then P59_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P59_SUCCESS.<nonce> only on clean completion.
#
# The gate on the B=16 fused-q/k/v default: KL at B=16 (teacher-forced, one token per forward, sixteen rows at
# once -- the K16 small-M GEMM route P57 traced the token divergence to) between P54's int4 stack unfused and fused,
# with the register's NF4 control as the anchor and a fresh-process rebuild of the unfused stack as the determinism
# control. Four arms, one process each (the staged P42 hook applies the env lanes at load; kl_b16.py builds the arm
# via P44's serve_stack and refuses a half-fused model), logits saved per arm, KL read by p59_reduce.py on the box.
# Nothing here changes a default; the pre-registration's decision rule reads the verdicts.
set -uo pipefail
W=/root/p59; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p59: $*"; }
NONCE=${P59_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P59_RUN_NONCE.tmp && mv $W/P59_RUN_NONCE.tmp $W/P59_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P59_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P59_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P59_RUN_ID P59_DEADLINE_EPOCH P59_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=24 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39; export E4B_MODEL_ID=$MID
: > summary.txt; echo "$P59_INSTANCE_ID" > INSTANCE_ID
# ---- staged pieces, byte-for-byte
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py serve_stack.py kl_fidelity.py kl_b16.py p59_reduce.py staged.sha256; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p59/staged.sha256"; finish 9; }
# ---- deadline guard + per-arm alarm (P54's): never start an arm that cannot finish 10 min before teardown
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P59_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P59_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P59_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
# ---- install: e4b pinned + P37's toolchain pins (image python); gnf4 at the K17 cut
export DEBIAN_FRONTEND=noninteractive
say "install e4b @$E4B_SHA (image python; P37 pins)"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
PYTHONPATH=$W/hook python - <<'PYT' || { say "TRIPWIRE FAIL (e4b)"; finish 9; }
import site, os, inspect, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
from experts4bit_qlora.engines.int4_attn import Int4Linear, _smallm_kernels
assert getattr(Int4Linear, "SMALLM_ROWS_MAX", None) == 16, "e4b cut lacks the K16 route (#578)"
assert hasattr(Int4Linear, "fuse") and hasattr(Int4Linear, "from_packed"), "e4b cut lacks Int4Linear.fuse (#651): the B=1 arm would refuse"
_smallm_kernels()   # the installed gnf4 must carry int4_smallm, or the route would refuse at enable time
import int4_b32
assert hasattr(int4_b32, "gemv_counter_len") and int4_b32.gemv_fused_reduce_default() is False, "gnf4 cut is not K17's, or the fused reduce is defaulted on (P59 runs the two-launch default)"
from experts4bit_qlora.engines.qkv_fuse import fuse_qkv  # noqa: F401  (the lever under test must import)
import usercustomize  # noqa: F401
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p59/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['E4B_SHA']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\ntorch(e4b) {torch.__version__}\ntriton(e4b) {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
print("tripwire OK (e4b):", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__, "triton", triton.__version__)
PYT
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt; lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt
cat /sys/fs/cgroup/memory.max 2>/dev/null | sed "s/^/cgroup memory.max /" | tee -a forensics.txt; nvcc --version 2>/dev/null | tail -1 | tee -a forensics.txt
python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
df -h /root | tail -1
# ---- K0 controls on THIS host: kl_fidelity's own gate (no KL row without them)
perl -e 'alarm 600; exec @ARGV' python $W/kl_fidelity.py --controls --out $W/k0.json > logs/k0.log 2>&1 || { tail -5 logs/k0.log; say "K0 CONTROLS FAILED -- no KL row is produced"; echo "K0 FAILED" >> summary.txt; finish 14; }
python -c "import json; r=json.load(open('$W/k0.json')); assert r['all_passed']; print('K0 all_passed', r.get('metric_version'))" | tee -a summary.txt || { say "K0 receipt not passing"; finish 14; }
# ---- fetch (pinned), bake (bo7's k8_bake.py), prompts (step_decomp's own B=16 window -- P54/P57/P58's rows) -- as P58
say "fetch $MID @ $REV"
perl -e 'alarm 4800; exec @ARGV' python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL (bf16)"; finish 11; }
bake_fail(){ say "BAKE FAIL ($1) -- k8_bake.py's record (err + traceback), so the reason travels with the receipt:"
  python -c "import json,sys; r=json.load(open('$W/work_qwen3/bake.json')); print('BAKE_ERR', r.get('status'), r.get('step')); print(r.get('err')); print(r.get('tb'))" 2>/dev/null | tee -a logs/bake.log | tail -n 25 | cut -c1-400
  finish 12; }
say "bake NF4 arena"; mkdir -p $W/work_qwen3
K8_MODEL="$MID" K8_WORK="$W/work_qwen3" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; bake_fail "rc"; }
QA=$W/work_qwen3/nf4.arena; [ -e "$QA" ] || bake_fail "no arena"
say "prompt dump: step_decomp._k8_window at B=16 -> prompts_b16.json (the timed arms' rows)"
python - "$MID" "$REV" <<'PYP' > logs/prompts.log 2>&1 || { tail -5 logs/prompts.log; say "PROMPT DUMP FAIL"; finish 13; }
import sys, json, hashlib, types
sys.path.insert(0, "/root/p59")
import step_decomp
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], revision=sys.argv[2])
B = 16
a = types.SimpleNamespace(ppl_source="wikitext", ppl_chat=False, ppl_chat_suffix="", prompt_offset=0, prompt_span=0, prompt_len=512, batch=B, ppl_steps=0)
ids, step, prompts, _, ppl_sha = step_decomp._k8_window(a, tok)
assert all(len(p) == 512 for p in prompts) and len(prompts) == B and len(set(tuple(p) for p in prompts)) == B
rec = {"model": sys.argv[1], "revision": sys.argv[2], "source": "wikitext-2-raw-v1 test via step_decomp._k8_window", "batch": B, "prompt_len": 512,
       "corpus_tokens": int(ids.numel()), "row_step": int(step), "rows_sha256": [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts],
       "prompts_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(), "prompts": prompts}
json.dump(rec, open("/root/p59/prompts_b16.json", "w"))
print(f"PROMPTS B={B} corpus={ids.numel()} step={step} sha={rec['prompts_sha256']}")
PYP
grep -a PROMPTS logs/prompts.log | tee -a summary.txt
# ---- arms: one process each; the hook applies the env lanes at load. EXP=0 -> NF4 control (no folds, no int4).
# kl_arm TAG EXP FUSE_FLAG   (receipt dir out/$TAG: logits.pt + census.json; log logs/run_$TAG.log)
kl_arm(){ local TAG=$1 EXP=$2 FUSE=$3; local G=1 R=1 E=1 AI=1; [ "$EXP" = 0 ] && { G=0; R=0; E=0; AI=0; }
  local AL; AL=$(arm_alarm 1800); say "arm $TAG (exp=$EXP attn_int4=$AI fuse=$FUSE alarm=$AL)"
  { echo "P59 arm=$TAG gnf4=$GNF4_SHA at=$(date -u +%FT%TZ)"; } > logs/run_$TAG.log
  env E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4=$AI E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e "alarm $AL; exec @ARGV" python -u $W/kl_b16.py --arm $TAG --model "$MID" --arena "$QA" --calib $W/calib.json --prompts $W/prompts_b16.json --prefix 384 --out $W/out/$TAG $FUSE >> logs/run_$TAG.log 2>&1
  local rc=$?
  grep -aE "P59ARM|P59 fused|INT4EXP|ATTNINT4|fused q/k/v|REFUSED|Error|Traceback" logs/run_$TAG.log | tail -5 | sed "s/^/    /"
  { echo -n "arm $TAG rc=$rc "; grep -a "P59ARM" logs/run_$TAG.log | tail -1 | cut -c1-300; echo; } >> summary.txt
  [ -s "$W/out/$TAG/logits.pt" ] || { echo "LOGITS MISSING $TAG" >> summary.txt; return 42; }
  return $rc; }
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
can_run 900 nf4       && { kl_arm nf4       0 "";           rec $?; }
can_run 900 int4      && { kl_arm int4      1 "";           rec $?; }
can_run 900 int4_fqkv && { kl_arm int4_fqkv 1 "--fuse-qkv"; rec $?; }
can_run 900 int4_aa   && { kl_arm int4_aa   1 "";           rec $?; }
# engagement checks the reducer also applies (recorded here so summary.txt carries them)
grep -aqE "P59 fused q/k/v projections on 48 attention modules" logs/run_int4_fqkv.log 2>/dev/null || { echo "FQKV NOT ENGAGED in int4_fqkv" >> summary.txt; rec 41; }
grep -aqE "INT4EXP|ATTNINT4|fused q/k/v" logs/run_nf4.log 2>/dev/null && { echo "INT4 LEAKED into the NF4 control" >> summary.txt; rec 44; }
# ---- reduce on the box (the logits stay here: ~1.2 GB per arm; the drive fetches summaries, censuses and logs)
say "reduce"; python $W/p59_reduce.py $W/out --md $W/RESULTS-p59-generated.md --json $W/p59_rep.json | tee RESULTS.txt
say "----- summary -----"; cat summary.txt; echo "----- versions -----"; cat versions.txt
finish "$rc_any"
