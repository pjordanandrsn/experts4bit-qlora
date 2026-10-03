#!/bin/bash
# bench/p107/p107_run.sh -- lane P107, BOX side (bench/p107/PREREG-p107.md; e4b#960). Started detached by p107_drive.sh
# with the run's nonce; P107_RUN_NONCE first, then P107_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P107_SUCCESS.<nonce> only when the reducer ran.
#
# Paged prefill attention's route A/B (e4b#960), on ONE RTX 5090 at the launch commit:
#   premise  tests/test_paged_prefill_attn_route_gpu.py on this card: flash reaches the flash kernel at Qwen3's
#            attention shapes over three 512-token chunks, math never does, and they agree within 1 % (1 test; a skip
#            is a failure; rc 25)
#   ab       p107_box.py ab: ONE engine (SC1's int4_sched stack, max_seqs 1, chunk 512); E4B_PAGED_PREFILL_ATTN
#            math / flash switched between requests -- three interleaved rounds of TTFT at 512 and 4096 tokens, a
#            kernel census per route, the SERVED-PREFILL NLL over 12 fresh windows x 2 routes (the paged prefill path
#            itself, 512-token chunks), and a device-time profile of one 4096-token request per route
#   order    install; premise; fetch; bake; prompts (SC1's digests checked); ab; reduce
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P107_GPU_CLASS P107_MIN_DISK_GB P107_REHEARSAL P107_ARMS. The guard is 1 h, so there is no proving run.
set -uo pipefail
W=/root/p107; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p107: $*"; }
NONCE=${P107_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P107_RUN_NONCE.tmp && mv $W/P107_RUN_NONCE.tmp $W/P107_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P107_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P107_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P107_RUN_ID P107_DEADLINE_EPOCH P107_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=34da93d6fe8d2a401b7001705658ce00b2b18213   # grouped-nf4-gemm v0.34.1, e4b CI's pin and SC1 box B's
MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
SHA_B1=a8e6ea1d7d140dbe726c94f0e6325eb11457a93f196419483de79507272f95e3        # SC1 box B prompts_b1.json
SHA_B1_4096=cd70a142d533eb88a3d2bec79bc4b11539f3a08b30c20ba5fbba248fb9dbafdd   # SC1 box B prompts_b1_4096.json
GPU_CLASS=${P107_GPU_CLASS:-5090}; MIN_DISK_GB=${P107_MIN_DISK_GB:-110}; REHEARSAL=${P107_REHEARSAL:-0}
ARMS=${P107_ARMS:-ab}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever starts unset; each arm sets SC1's stack explicitly
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_PLACEMENT E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_INT4_ARTIFACT_DIR E4B_INT4_EXPECTED_FINGERPRINT E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN PYTHONPATH
# SC1's stack, byte for byte (bench/sc1/sc1_run.sh SPEEDENV / FOLDS / ROUTEENV; the int4_sched arms SC1 box B's TTFT ran)
FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
SPEEDENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 $FOLDS"
ROUTEENV="E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto"
: > summary.txt; echo "$P107_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB arms='$ARMS'" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 110 ] || [ "$ARMS" != "ab" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p107_box.py p107_reduce.py sc1_e4b_sched.py sc1_prompts.py step_decomp.py p98_bake.py calib.json test_paged_prefill_attn_route_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p107/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the 61 GB checkpoint, its NF4 snapshot and arena)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P107_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P107_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P107_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
# The tripwire: the commits, and the code under test present AS the trace read it -- so a reading is about this path.
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
import pytest  # noqa: F401  (the premise runs under it; P102 A1: p102-5090-1 stopped there without it)
from experts4bit_qlora.engines import paged_attention as pa
assert pa.PREFILL_ATTN_ROUTES == ("math", "flash"), pa.PREFILL_ATTN_ROUTES
assert pa._prefill_attn_mode_env() == "math", "E4B_PAGED_PREFILL_ATTN's default is not math at the launch commit"
from torch.nn.attention.bias import causal_lower_right  # noqa: F401  (the flash route's mask)
from experts4bit_qlora.engines import hot_residency as hr
assert hr._int4_prefill_mode_env() == "k19", "the int4 prefill route is not k19 here (both attention routes share it)"
import fp8_paged_attn  # noqa: F401
import experts4bit_qlora as e, torch, triton
open("/root/p107/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p107_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p107_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
# ---- the premise, on THIS card, before anything is fetched: both attention routes at Qwen3's attention shapes --
# flash reaches the flash kernel, math never does, and they agree (1 test; a skip is a failure)
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_paged_prefill_attn_route_gpu.py -q -rs -s -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; grep -a "rel flash" logs/premise.log | tr '\n' ' '; echo; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "1 passed" && ! echo "$LASTL" | grep -q skipped; then
  say "premise held: both attention routes on this card"
else
  say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# ---- the checkpoint (pinned), the NF4 arena (bench/p98/p98_bake.py, revision-pinned), SC1's prompt files
can_run 2400 fetch_qwen3 || finish 40
say "fetch $MODEL @ $REV"
perl -e "alarm $(arm_alarm 2400); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=8))" > logs/fetch_qwen3.log 2>&1 || { tail -2 logs/fetch_qwen3.log; say "DL FAIL"; finish 11; }
tail -1 logs/fetch_qwen3.log | tee -a summary.txt
can_run 1800 bake_qwen3 || finish 40
say "bake qwen3"; mkdir -p $W/work_qwen3
perl -e "alarm $(arm_alarm 1800); exec @ARGV" python $W/p98_bake.py --model "$MODEL" --revision "$REV" --work $W/work_qwen3 > logs/bake_qwen3.log 2>&1
rc=$?
{ echo -n "bake qwen3 rc=$rc "; python -c "import json; r=json.load(open('$W/work_qwen3/bake.json')); print({k: r.get(k) for k in ('status','layers','experts','snapshot_gib','loaded_commit','err')})" 2>&1 | cut -c1-300; } | tee -a summary.txt
[ "$rc" = 0 ] && [ -e $W/work_qwen3/nf4.arena ] || { tail -5 logs/bake_qwen3.log; say "BAKE FAIL"; finish 12; }
say "prompt dump: sc1_prompts.py (step_decomp._k8_window) -> prompts_b1.json + prompts_b1_4096.json (and SC1's other files)"
SC1_HARNESS_DIR=$W perl -e "alarm $(arm_alarm 1200); exec @ARGV" python $W/sc1_prompts.py --model "$MODEL" --rev "$REV" --out $W > logs/prompts.log 2>&1 || { tail -5 logs/prompts.log; say "PROMPT DUMP FAIL"; finish 19; }
grep -aE "^PROMPTS|^WINDOW" logs/prompts.log | tee -a summary.txt
for pair in "prompts_b1.json:$SHA_B1" "prompts_b1_4096.json:$SHA_B1_4096"; do
  f=${pair%%:*}; want=${pair#*:}
  got=$(python -c "import json; print(json.load(open('$W/$f'))['prompts_sha256'])" 2>/dev/null)
  [ "$got" = "$want" ] || { say "REFUSED: $f digest $got is not SC1 box B's $want -- not the token ids SC1 read"; echo "refused: prompts $f" > REFUSAL; finish 19; }
  echo "PROMPTS_MATCH_SC1 $f $got" | tee -a summary.txt
done
# ---- the arm: ONE engine for TTFT, the kernel census, the served-prefill NLL and the profile
ab_arm(){ local AL; AL=$(arm_alarm 2700); can_run 1500 arm_ab || return 40
  say "arm ab: one engine, math / flash interleaved over three rounds at 512 and 4096 tokens, then the served-prefill NLL (alarm=$AL)"
  env PYTHONPATH= $ROUTEENV $SPEEDENV \
    E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work_qwen3/nf4.arena E4B_PAGED_CALIB=$W/calib.json \
    E4B_PAGED_PLACEMENT=all-vram E4B_PAGED_MAX_SEQS=1 E4B_PAGED_MAX_TOKENS_PER_SEQ=4104 E4B_PAGED_CHUNK_TOKENS=512 \
    E4B_PAGED_GRAPHS=1 E4B_PAGED_BUCKETS=1 E4B_PAGED_FUSE_QKV=1 E4B_PAGED_TORCH_THREADS=8 \
    P107_PROMPTS_512=$W/prompts_b1.json P107_PROMPTS_4096=$W/prompts_b1_4096.json P107_OUT=$W/ab_routes.json \
    perl -e "alarm $AL; exec @ARGV" python $W/p107_box.py ab > logs/arm_ab.log 2>&1
  local rc=$?
  { echo -n "arm ab rc=$rc windows=$(grep -ac '^P107_NLL ' logs/arm_ab.log) "; grep -aE "^P107_TTFT " logs/arm_ab.log | cut -c1-300; } >> summary.txt; echo >> summary.txt
  [ "$rc" = 0 ] || return 26; }
for N in $ARMS; do
  case "$N" in ab) ab_arm; rec $?;; *) say "unknown arm $N"; rec 26;; esac
done
say "reduce"; python $W/p107_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish "$rc_any"
