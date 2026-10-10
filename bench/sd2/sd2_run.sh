#!/bin/bash
# bench/sd2/sd2_run.sh -- lane SD2, BOX side of the CUDA proof sd2-prove-N (bench/sd2/PREREG-sd2.md, Amendment 1;
# e4b#1313). Started detached by sd2_drive.sh with the run's nonce; SD2_RUN_NONCE first, then SD2_EXIT_CODE.<nonce> +
# TP_DONE.<nonce> on every exit and SD2_SUCCESS.<nonce> only when the reducer ran.
#
# Correctness only, ONE RTX 5090, nothing timed. The target is the stacked integration commit (main + SD2's build
# PRs 1-5 at their reviewed heads) + grouped-nf4-gemm v0.45.0 (CI's pin), served as the shipped default with
# E4B_PAGED_SPEC=eagle3, k = 3 and the pinned head. The target's own GPU tests run on the card, then sd2_box.py --prove
# (census, capture bitwise, V0 addressing at k = 1..3 with its three mutants, the draft gate, the transition), then the
# reducer. The harness is the launch commit (E4B_SHA), checked out by SHA as its own worktree; it is never installed.
#   order    refusals; install + clones + tripwire; self-tests; fetch (model, head); bake; prompts; GPU tests; prove; reduce
set -uo pipefail
W=/root/sd2; mkdir -p $W/logs $W/src; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] sd2: $*"; }
NONCE=${SD2_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/SD2_RUN_NONCE.tmp && mv $W/SD2_RUN_NONCE.tmp $W/SD2_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > SD2_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > SD2_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in SD2_RUN_ID SD2_DEADLINE_EPOCH SD2_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
# ---- the target (Amendment 1): the integration commit and its build heads, the kernel package, the model, the head
E4B_T=539a2d2694096a81c6b272b335e26d818647cd95     # sd2/serve-wiring: main aa47af8b + builds 1-5 (#1553 #1554 #1556 #1558 #1559)
GNF4_T=724ccc454f006c1a46836e434e997f31f293747f    # grouped-nf4-gemm v0.45.0, the CI pin
E4B_H=$E4B_SHA                                     # the harness: the launch commit, which stages this kit
MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
HEAD_REPO=RedHatAI/Qwen3-30B-A3B-speculator.eagle3; HEAD_REV=6afc5aa2477b923467fb9a8d906782b984a9a6ba
HEAD_SHA256=d2d6e2e63e09dc755053ae5c98cdececae3611ae5e202d4fa5411126dd3b1dfa; HEAD_BYTES=1044539336
GPU_CLASS=${SD2_GPU_CLASS:-5090}; MIN_DISK_GB=${SD2_MIN_DISK_GB:-150}; MIN_RAM_GB=${SD2_MIN_RAM_GB:-60}
NEED_FETCH=2100; NEED_BAKE=600; NEED_TESTS=900; NEED_PROVE=1500   # seconds
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset: the default server is the subject (fixed knobs below)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS E4B_FUSE_COMBINE E4B_INT4_TILE_PROGRAMS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS E4B_PAGED_DECODE_LOOKAHEAD \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS GNF4_GEMV_BW GNF4_PDL \
      E4B_PAGED_PREFILL_GRAPH E4B_KV_STEP_SELECT E4B_PAGED_SPEC E4B_PAGED_SPEC_K E4B_PAGED_SPEC_HEAD
: > summary.txt; echo "$SD2_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b_target=$E4B_T gnf4_target=$GNF4_T e4b_harness=$E4B_H model=$MODEL@$REV head=$HEAD_REPO@$HEAD_REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB" | tee -a summary.txt
if [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 150 ] || [ "$MIN_RAM_GB" != 60 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in sd2_box.py sd2_reduce.py sd1_box.py sd1_eagle3.py p109_box.py k8_bake.py calib.json chat_prompts.json staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/sd2/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk, the host RAM
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$SD2_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((SD2_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( SD2_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_PIN=$(python -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install deps (transformers 5.17.0, bitsandbytes 0.50.2; torch held at $TORCH_PIN)"
pipx logs/pip_deps.log 1800 --prefer-binary "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece \
  safetensors "huggingface_hub>=1.31,<2" pytest numpy || { tail -4 logs/pip_deps.log; say "PIP FAIL (deps)"; finish 9; }
clone(){ local url=$1 dir=$2
  [ -d $dir/.git ] || perl -e 'alarm 900; exec @ARGV' git clone -q --filter=blob:none "$url" $dir > logs/clone_$(basename $dir).log 2>&1; }
clone https://github.com/pjordanandrsn/experts4bit-qlora.git $W/src/e4b || { say "CLONE FAIL (e4b)"; finish 9; }
clone https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/src/gnf4 || { say "CLONE FAIL (gnf4)"; finish 9; }
git -C $W/src/e4b worktree add -q --detach $W/src/e4b_T $E4B_T > /dev/null 2>&1 || { say "CHECKOUT FAIL e4b_T $E4B_T"; finish 9; }
git -C $W/src/gnf4 worktree add -q --detach $W/src/gnf4_T $GNF4_T > /dev/null 2>&1 || { say "CHECKOUT FAIL gnf4_T $GNF4_T"; finish 9; }
git -C $W/src/e4b worktree add -q --detach $W/src/e4b_H $E4B_H > /dev/null 2>&1 || { say "CHECKOUT FAIL e4b_H $E4B_H"; finish 9; }
for pair in "e4b_T $E4B_T" "gnf4_T $GNF4_T" "e4b_H $E4B_H"; do
  set -- $pair; got=$(git -C $W/src/$1 rev-parse HEAD)
  [ "$got" = "$2" ] || { say "CHECKOUT FAIL $1 is $got, not $2"; finish 9; }
done
WATCHDOG=$W/src/e4b_H/bench/common/hf_fetch_watchdog.py
[ -s $WATCHDOG ] || { say "HARNESS MISSING: $WATCHDOG"; finish 9; }
pipx logs/pip_target.log 600 --no-deps -e $W/src/e4b_T -e $W/src/gnf4_T || { tail -3 logs/pip_target.log; say "PIP FAIL (target)"; finish 9; }
E4B_T=$E4B_T E4B_H=$E4B_H python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, importlib.metadata as md
import experts4bit_qlora as e, int4_b32, transformers, torch, triton
assert "/src/e4b_T/" in e.__file__ and "/src/gnf4_T/" in int4_b32.__file__, (e.__file__, int4_b32.__file__)
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora.engines import spec_decode, eagle3_draft
from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
from experts4bit_qlora import serve_paged
assert hasattr(Fp8PagedKV, "alias_bind") and hasattr(serve_paged, "_enable_spec"), "the target lacks SD2's build"
assert eagle3_draft.HEAD_SHA256 == "d2d6e2e63e09dc755053ae5c98cdececae3611ae5e202d4fa5411126dd3b1dfa"
hub = md.version("huggingface_hub")
assert tuple(int(x) for x in hub.split(".")[:2]) >= (1, 31), f"huggingface_hub {hub} < 1.31"
open("/root/sd2/versions.txt", "a").write(f"e4b target {e.__version__} sha {os.environ['E4B_T']}\nharness sha {os.environ['E4B_H']}\ngnf4 target {md.version('grouped-nf4-gemm')}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\nhuggingface_hub {hub}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, md.version("grouped-nf4-gemm"))
PYT
cat versions.txt | tee -a summary.txt
python $W/sd2_reduce.py --self-test | tail -1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/sd2_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
python $W/sd1_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SD1 BOX SELF-TEST FAILED"; finish 21; }
python $WATCHDOG --self-test 2>/dev/null | tail -1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "WATCHDOG SELF-TEST FAILED"; finish 21; }
# ---- the checkpoint (under the watchdog), the head (pinned by revision and sha256, with its config), the NF4 arena
can_run $NEED_FETCH fetch || finish 40
say "fetch $MODEL @ $REV"
FA=$(step_alarm 2700)
HF_HUB_VERBOSITY=info perl -e "alarm $((FA + 90)); exec @ARGV" python $WATCHDOG --repo "$MODEL" --revision "$REV" \
  --allow '*.safetensors' --allow '*.json' --allow 'tokenizer*' --allow '*.model' --allow '*.txt' --allow merges.txt \
  --allow vocab.json --max-workers 8 --poll-s 30 --stall-s 180 --max-restarts 3 --budget-s $FA \
  > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
say "fetch the head $HEAD_REPO @ $HEAD_REV"
perl -e "alarm 1200; exec @ARGV" python -c "from huggingface_hub import hf_hub_download as d; d('$HEAD_REPO', 'config.json', revision='$HEAD_REV'); print(d('$HEAD_REPO', 'model.safetensors', revision='$HEAD_REV'))" \
  > logs/fetch_head.log 2>&1 || { tail -2 logs/fetch_head.log; say "DL FAIL (head)"; finish 11; }
HEAD=$(tail -1 logs/fetch_head.log); HEAD_DIR=$(dirname "$HEAD")
[ "$(stat -L -c %s "$HEAD")" = "$HEAD_BYTES" ] && [ "$(sha256sum "$HEAD" | cut -d' ' -f1)" = "$HEAD_SHA256" ] \
  || { say "HEAD MISMATCH: $(stat -L -c %s "$HEAD") B, $(sha256sum "$HEAD" | cut -c1-16)"; finish 11; }
[ -s "$HEAD_DIR/config.json" ] || { say "HEAD CONFIG MISSING in $HEAD_DIR"; finish 11; }
echo "HEAD $HEAD_REPO@$HEAD_REV sha256 $HEAD_SHA256 dir $HEAD_DIR" | tee -a summary.txt
can_run $NEED_BAKE bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt
# ---- the prompts: R = P109's wikitext rows; C = the pinned UltraChat rows, chat-templated (SD1's bytes)
perl -e "alarm 1200; exec @ARGV" python $W/p109_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts_R.json > logs/prompts_R.log 2>&1 \
  || { tail -4 logs/prompts_R.log; say "PROMPTS FAIL (R)"; finish 19; }
perl -e "alarm 600; exec @ARGV" python $W/sd1_box.py --chat-prompts $W/chat_prompts.json --snapshot "$SNAP" --outdir $W > logs/prompts_C.log 2>&1 \
  || { tail -4 logs/prompts_C.log; say "PROMPTS FAIL (C)"; finish 19; }
grep -a "PROMPTS" logs/prompts_R.log logs/prompts_C.log | tee -a summary.txt
# ---- the target's own GPU tests, on the card (CI cannot run them: sm_89+)
can_run $NEED_TESTS "gpu tests" || finish 40
GPU_TESTS="tests/test_spec_decode_gpu.py tests/test_spec_decode.py tests/test_kv_alias_verify.py tests/test_eagle3_draft.py tests/test_scheduler_speculative.py tests/test_serve_paged_spec.py tests/test_decode_graph_buckets.py tests/test_kv_step_select.py"
say "gpu tests"
(cd $W/src/e4b_T && perl -e "alarm $(step_alarm 1500); exec @ARGV" python -m pytest -q -rs -p no:cacheprovider $GPU_TESTS) > logs/gpu_tests.log 2>&1
GT_RC=$?
python - "$GT_RC" <<'PYG' | tee -a summary.txt
import json, re, sys
log = open("/root/sd2/logs/gpu_tests.log", encoding="utf-8", errors="replace").read()
last = [ln for ln in log.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
tail = last[-1] if last else ""
num = lambda w: int((re.search(rf"(\d+) {w}", tail) or [0, 0])[1])
card = sum(1 for ln in log.splitlines() if ln.startswith("SKIPPED") and "e4m3" in ln)
rec = {"rc": int(sys.argv[1]), "passed": num("passed"), "failed": num("failed"), "skipped": num("skipped"),
       "skipped_for_card": card, "summary": tail.strip()}
json.dump(rec, open("/root/sd2/gpu_tests.json", "w"))
print("SD2_GPU_TESTS " + json.dumps(rec))
PYG
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/gpu_tests.json ] || { say "GPU TESTS RECORD FAIL"; finish 22; }
# ---- the proof: the shipped default with speculation on, k = 3, graphs on
can_run $NEED_PROVE prove || finish 40
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16 E4B_INT4_TILE_PROGRAMS=1 E4B_PAGED_SPEC=eagle3 E4B_PAGED_SPEC_K=3 E4B_PAGED_SPEC_HEAD=$HEAD_DIR"
AL=$(step_alarm 2400); say "prove (alarm=$AL)"
# shellcheck disable=SC2086  # ENGINE_ENV is an assignment list by design
env PYTHONPATH= $ENGINE_ENV E4B_SHA=$E4B_T GNF4_SHA=$GNF4_T \
  perl -e "alarm $AL; exec @ARGV" python $W/sd2_box.py --prove --prompts-r $W/prompts_R.json --prompts-c $W/prompts_C-think.json \
    --head-dir "$HEAD_DIR" --out $W/prove.json > logs/prove.log 2>&1
rc=$?
{ echo "prove rc=$rc"; grep -a "^SD2_PROVE" logs/prove.log; } | tee -a summary.txt
[ "$rc" = 0 ] && [ -s $W/prove.json ] || { tail -8 logs/prove.log | cut -c1-300 | tee -a summary.txt; say "PROVE FAILED (rc=$rc)"; finish 23; }
say "reduce"; python $W/sd2_reduce.py --prove $W/prove.json --gpu-tests $W/gpu_tests.json --out $W/verdict_prove.json 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict_prove.json ] || { say "REDUCER FAILED"; finish 22; }
finish 0
