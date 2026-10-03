#!/bin/bash
# bench/p99/p99_run.sh -- lane P99, BOX side (bench/p99/PREREG-p99.md; e4b#913, e4b#564). Started detached by
# p99_drive.sh with the run's nonce; P99_RUN_NONCE first, then P99_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P99_SUCCESS.<nonce> only when the reducer ran.
#
# Localise P98's replay fault (#913): on ONE RTX 5090, P98's harness (bench/p98/p98_box.py, run through p99_box.py,
# which logs a P99_STEP line before every decode call), seven arms, each a fresh serve_paged.build_engine in its own
# process:
#   d0g          P98's arm g repeated: Qwen3.6, buckets 1-16, K25 at its default  -- the fault must reproduce
#   d1g / d1e    OLMoE-1B-7B (no linear-attention state), graphs / padded-eager oracle
#   d2g / d2e    Qwen3.6 with E4B_NF4_GROUPED_SMALLM=0 (K25 off), graphs / oracle
#   d3g / d3e    Qwen3.6 with the single bucket 16, graphs / oracle
#   premise      the three hybrid GPU test files PASS, none skipped (rc 25), before anything is fetched
#   order        fetch Qwen3.6 and OLMoE; bake both arenas (bench/p98/p98_bake.py); the arms in the order above; reduce
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P99_GPU_CLASS P99_MIN_DISK_GB P99_REHEARSAL P99_ARMS. The guard is 1 h, so there is no proving run.
set -uo pipefail
W=/root/p99; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p99: $*"; }
NONCE=${P99_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P99_RUN_NONCE.tmp && mv $W/P99_RUN_NONCE.tmp $W/P99_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P99_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P99_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P99_RUN_ID P99_DEADLINE_EPOCH P99_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=34da93d6fe8d2a401b7001705658ce00b2b18213   # grouped-nf4-gemm v0.34.1 (fp8_paged_attn, fused batch KV append, nvme_arena), e4b CI's pin
MODEL=Qwen/Qwen3.6-35B-A3B; REV=995ad96eacd98c81ed38be0c5b274b04031597b0
OLMOE=allenai/OLMoE-1B-7B-0924-Instruct; OREV=7f1c97f440f06ce36705e4f2b843edb5925f4498
GPU_CLASS=${P99_GPU_CLASS:-5090}; MIN_DISK_GB=${P99_MIN_DISK_GB:-170}; REHEARSAL=${P99_REHEARSAL:-0}
ARMS=${P99_ARMS:-d0g d1g d1e d2g d2e d3g d3e}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever starts unset: the engine at e4b's defaults
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_PLACEMENT
: > summary.txt; echo "$P99_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV olmoe=$OLMOE@$OREV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB arms='$ARMS'" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 170 ] || [ "$ARMS" != "d0g d1g d1e d2g d2e d3g d3e" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p99_box.py p98_box.py p98_bake.py p99_reduce.py calib.json test_linear_state_gpu.py test_hybrid_decode_graphs_gpu.py test_linear_state_graph_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p99/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (two checkpoints, 72 + 14 GB, their NF4 snapshots and arenas)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P99_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P99_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P99_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest fastapi || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md, importlib.util
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora.engines import linear_state, paged_runner
assert hasattr(linear_state, "_bucket_selector") and hasattr(linear_state.LinearStatePool, "_index"), "needs #907 + #908"
src = inspect.getsource(paged_runner.PagedModelRunner.enable_decode_graphs)
assert "_warm_linear_state" in src and "not captured yet" not in src, "enable_decode_graphs still refuses hybrids"
from experts4bit_qlora.serve_paged import build_engine  # noqa: F401
import fp8_paged_attn, fp8_kv, nvme_arena  # noqa: F401  (the kernel, the fused batch append, the arena)
assert hasattr(fp8_kv, "fp8_kv_append_bt1"), "gnf4 lacks the fused batch KV append the buckets need"
import experts4bit_qlora as e, torch, triton
mods = {m: importlib.util.find_spec(m) is not None for m in ("fla", "causal_conv1d", "kernels")}
open("/root/p99/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\ngated-deltanet kernel modules {mods}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p99_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched: the hybrid paged path through the real kernel, hybrid
# decode graphs under the scheduler against the padded eager step, and the linear state under real capture. All four
# tests must PASS: a skip (no sm_89+, a missing kernel) is a failure.
(cd $W && PYTHONPATH='' perl -e 'alarm 1200; exec @ARGV' python -m pytest test_linear_state_gpu.py test_hybrid_decode_graphs_gpu.py test_linear_state_graph_gpu.py -q -rs -s -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "4 passed" && ! echo "$LASTL" | grep -q skipped; then
  say "premise held: the hybrid path, its bucket graphs and its real capture on this card"
else
  say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
# ---- the checkpoints
fetch(){ local TAG=$1 MID=$2 R=$3
  can_run 2400 "fetch_$TAG" || finish 40
  say "fetch $MID @ $R"
  perl -e "alarm $(arm_alarm 2400); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$R', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json','*.jinja'], max_workers=8))" > logs/fetch_$TAG.log 2>&1 || { tail -2 logs/fetch_$TAG.log; say "DL FAIL $TAG"; finish 11; }
  tail -1 logs/fetch_$TAG.log | tee -a summary.txt; }
fetch qwen "$MODEL" "$REV"
fetch olmoe "$OLMOE" "$OREV"
# ---- the NF4 arenas the engines serve from (bench/p98/p98_bake.py, each checkpoint pinned)
bake(){ local TAG=$1 MID=$2 R=$3
  can_run 1800 "bake_$TAG" || finish 40
  say "bake $TAG"; mkdir -p $W/work_$TAG
  perl -e "alarm $(arm_alarm 1800); exec @ARGV" python $W/p98_bake.py --model "$MID" --revision "$R" --work $W/work_$TAG > logs/bake_$TAG.log 2>&1
  local rc=$?
  { echo -n "bake $TAG rc=$rc "; python -c "import json; r=json.load(open('$W/work_$TAG/bake.json')); print({k: r.get(k) for k in ('status','layers','experts','snapshot_gib','loaded_commit','err')})" 2>&1 | cut -c1-300; } | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -5 logs/bake_$TAG.log; say "BAKE FAIL $TAG"; finish 12; }; }
bake qwen "$MODEL" "$REV"
bake olmoe "$OLMOE" "$OREV"
# ---- the arms, each a fresh engine in its own process (a fault aborts the process: it writes no record)
arm(){ local NAME=$1 A AL MID R ARENA ENVS="" XB=""
  case "$NAME" in
    d0g) A=g;;
    d1g|d1e) A=${NAME:2:1}; MID=$OLMOE; R=$OREV; ARENA=$W/work_olmoe/nf4.arena;;
    d2g|d2e) A=${NAME:2:1}; ENVS="E4B_NF4_GROUPED_SMALLM=0";;
    d3g|d3e) A=${NAME:2:1}; XB="--buckets 16";;
    *) say "unknown arm $NAME"; return 26;;
  esac
  MID=${MID:-$MODEL}; R=${R:-$REV}; ARENA=${ARENA:-$W/work_qwen/nf4.arena}
  AL=$(arm_alarm 900)
  can_run 600 "arm_$NAME" || return 40
  say "arm $NAME: --arm $A $MID ${ENVS:-} ${XB:-} (alarm=$AL)"
  # shellcheck disable=SC2086  # XB is a flag list by design
  env $ENVS perl -e "alarm $AL; exec @ARGV" python $W/p99_box.py --arm $A --model "$MID" --revision "$R" --arena "$ARENA" \
    --calib $W/calib.json --out $W/arm_$NAME.json $XB > logs/arm_$NAME.log 2>&1
  local rc=$?
  { echo -n "arm $NAME rc=$rc "; grep -aE "^P98_ARM" logs/arm_$NAME.log | tail -1 | cut -c1-400; grep -ac "device-side assert triggered" logs/arm_$NAME.log | sed 's/^/ asserts=/'; grep -aE "^P99_STEP" logs/arm_$NAME.log | tail -1; } | tr '\n' ' ' >> summary.txt; echo >> summary.txt
  return 0; }
for N in $ARMS; do arm $N; rec $?; done
say "reduce"; python $W/p99_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish "$rc_any"
