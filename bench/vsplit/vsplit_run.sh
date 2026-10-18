#!/bin/bash
# bench/vsplit/vsplit_run.sh -- the verify split, BOX side of the read vsplit-5090-N (bench/vsplit/PREREG-vsplit.md; e4b#1313).
# Started detached by vsplit_drive.sh with the run's nonce; VSPLIT_RUN_NONCE first, then VSPLIT_EXIT_CODE.<nonce> +
# TP_DONE.<nonce> on every exit and VSPLIT_SUCCESS.<nonce> only when the reducer ran. P132's runner (bench/p132/
# p132_run.sh at its Amendment 2) with three changes: no card-uuid refusal (any RTX 5090; the uuid is recorded), a
# second unprofiled timing pass after the census (the self-pair), and the verify split's reducer.
#
# The subject is SD2's read server, at SD2's read target (e4b 88ae1cfd, re-proven on the card by SD2's package-diff audit
# and its GPU tests) + grouped-nf4-gemm v0.45.0, E4B_PAGED_SPEC=eagle3 at k = 3 with the pinned head, 16 slots, graphs on.
#   order    refusals (incl. the guard); install + clones + audit + tripwire; self-tests; GPU tests; fetch (model, head);
#            bake; prompts; nsys; CANARY (buckets 1..4 captured, one replay per condition, a short census read by SC1b's
#            classifier: a failure costs minutes, rc 26); timing pass A (SD2's V1, unprofiled); the census (n = 1..4,
#            graph mode then node mode, each its own process); timing pass B (the self-pair); reduce
set -uo pipefail
W=/root/vsplit; mkdir -p $W/logs $W/src $W/census; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] vsplit: $*"; }
NONCE=${VSPLIT_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/VSPLIT_RUN_NONCE.tmp && mv $W/VSPLIT_RUN_NONCE.tmp $W/VSPLIT_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > VSPLIT_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > VSPLIT_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in VSPLIT_RUN_ID VSPLIT_DEADLINE_EPOCH VSPLIT_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
# ---- the subject: SD2's read target and its proven build, the kernel package, the model, the head
E4B_T=88ae1cfd8fbe3477189f5a42d8448f395e974345      # SD2's read target (E4B_T_READ): the cost the desk prediction targets
E4B_T_PROVE=539a2d2694096a81c6b272b335e26d818647cd95  # SD2's proven build: the package-diff audit's base
GNF4_T=724ccc454f006c1a46836e434e997f31f293747f       # grouped-nf4-gemm v0.45.0, the CI pin
E4B_H=$E4B_SHA                                        # the harness: the launch commit, which stages this kit
MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
HEAD_REPO=RedHatAI/Qwen3-30B-A3B-speculator.eagle3; HEAD_REV=6afc5aa2477b923467fb9a8d906782b984a9a6ba
HEAD_SHA256=d2d6e2e63e09dc755053ae5c98cdececae3611ae5e202d4fa5411126dd3b1dfa; HEAD_BYTES=1044539336
GPU_CLASS=${VSPLIT_GPU_CLASS:-5090}; MIN_DISK_GB=${VSPLIT_MIN_DISK_GB:-150}; MIN_RAM_GB=${VSPLIT_MIN_RAM_GB:-60}
GUARD_S=4500; NEED_TESTS=600; NEED_FETCH=2100; NEED_BAKE=600; NEED_CANARY=900; NEED_TIMING=600; NEED_CAPTURE=600   # --wallclock-h 1.25
SKIP=32; STEPS=64; CANARY_STEPS=12; MOE_LAYERS=48
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset (SD2's list): the SD2 read server is the subject (fixed knobs below)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS E4B_FUSE_COMBINE E4B_INT4_TILE_PROGRAMS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS E4B_PAGED_DECODE_LOOKAHEAD \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS GNF4_GEMV_BW GNF4_PDL \
      E4B_PAGED_PREFILL_GRAPH E4B_KV_STEP_SELECT E4B_PAGED_SPEC E4B_PAGED_SPEC_K E4B_PAGED_SPEC_HEAD
: > summary.txt; echo "$VSPLIT_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b_target=$E4B_T gnf4_target=$GNF4_T e4b_harness=$E4B_H model=$MODEL@$REV head=$HEAD_REPO@$HEAD_REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB" | tee -a summary.txt
if [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 150 ] || [ "$MIN_RAM_GB" != 60 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p132_box.py vsplit_reduce.py sd2_box.py sd2_reduce.py sd2_audit.py audit_read.tsv sd1_box.py sd1_eagle3.py p109_box.py \
         p115_quality.py p110_box.py p108_box.py p97_box.py k8_bake.py calib.json chat_prompts.json expect_w1.json \
         sc1b_census.py kernel_classes_nf4_v1_2.json staged-vsplit.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-vsplit.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/vsplit/staged-vsplit.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk, the host RAM
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap,power.limit --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
# any RTX 5090: every number the rule compares is this card's own (PREREG-vsplit.md). The card is recorded, not refused:
# the uuid is field 4 of forensics.txt's first line.
card_uuid(){ awk -F', ' 'NR == 1 {print $4}' "$1"; }
GPU_UUID=$(card_uuid forensics.txt)
printf '{"uuid": "%s"}\n' "$GPU_UUID" > card.json
echo "VSPLIT_CARD uuid=$GPU_UUID" | tee -a summary.txt
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$VSPLIT_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((VSPLIT_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( VSPLIT_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
LEFT=$(( VSPLIT_DEADLINE_EPOCH - $(date +%s) ))
[ "$LEFT" -ge $(( GUARD_S - 900 )) ] || { say "REFUSED: the deadline leaves ${LEFT}s; the registered guard is ${GUARD_S}s"; echo "refused: guard ${LEFT}s" > REFUSAL; finish 17; }
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
# SD2's package-diff audit, unchanged: every package file that differs from the proven build is listed in audit_read.tsv
git -C $W/src/e4b cat-file -e "$E4B_T_PROVE^{commit}" 2>/dev/null \
  || perl -e 'alarm 300; exec @ARGV' git -C $W/src/e4b fetch -q origin refs/pull/1559/head > /dev/null 2>&1
python $W/sd2_audit.py --repo $W/src/e4b --from $E4B_T_PROVE --to $E4B_T --list $W/audit_read.tsv --out $W/audit.json \
  | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "AUDIT FAIL: a package file differs from the proven build unlisted"; finish 31; }
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
open("/root/vsplit/versions.txt", "a").write(f"e4b target {e.__version__} sha {os.environ['E4B_T']}\nharness sha {os.environ['E4B_H']}\ngnf4 target {md.version('grouped-nf4-gemm')}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\nhuggingface_hub {hub}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, md.version("grouped-nf4-gemm"))
PYT
cat versions.txt | tee -a summary.txt
python $W/vsplit_reduce.py --self-test | tail -1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p132_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
python $W/sd2_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SD2 BOX SELF-TEST FAILED"; finish 21; }
python $W/sd2_audit.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "AUDIT SELF-TEST FAILED"; finish 21; }
python $W/sc1b_census.py --self-test | tail -1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "CENSUS SELF-TEST FAILED"; finish 21; }
python $WATCHDOG --self-test 2>/dev/null | tail -1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "WATCHDOG SELF-TEST FAILED"; finish 21; }
# ---- the target's own GPU tests, on the card, before the 62 GB fetch (SD2's list)
can_run $NEED_TESTS "gpu tests" || finish 40
GPU_TESTS="tests/test_spec_decode_gpu.py tests/test_spec_decode.py tests/test_kv_alias_verify.py tests/test_eagle3_draft.py tests/test_scheduler_speculative.py tests/test_serve_paged_spec.py tests/test_decode_graph_buckets.py tests/test_kv_step_select.py"
say "gpu tests"
(cd $W/src/e4b_T && perl -e "alarm $(step_alarm 1500); exec @ARGV" python -m pytest -q -rs -p no:cacheprovider $GPU_TESTS) > logs/gpu_tests.log 2>&1
GT_RC=$?
python - "$GT_RC" <<'PYG' | tee -a summary.txt
import json, re, sys
log = open("/root/vsplit/logs/gpu_tests.log", encoding="utf-8", errors="replace").read()
last = [ln for ln in log.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
tail = last[-1] if last else ""
num = lambda w: int((re.search(rf"(\d+) {w}", tail) or [0, 0])[1])
rec = {"rc": int(sys.argv[1]), "passed": num("passed"), "failed": num("failed"), "skipped": num("skipped"), "summary": tail.strip()}
json.dump(rec, open("/root/vsplit/gpu_tests.json", "w"))
print("VSPLIT_GPU_TESTS " + json.dumps(rec))
PYG
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/gpu_tests.json ] || { say "GPU TESTS RECORD FAIL"; finish 22; }
[ "$GT_RC" = 0 ] || { say "GPU TESTS FAILED (rc=$GT_RC) -- stopping before the fetch"; finish 24; }
# ---- the checkpoint (under the watchdog), the head (pinned by revision and sha256), the NF4 arena, the prompts
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
echo "HEAD $HEAD_REPO@$HEAD_REV sha256 $HEAD_SHA256 dir $HEAD_DIR" | tee -a summary.txt
can_run $NEED_BAKE bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt
perl -e "alarm 1200; exec @ARGV" python $W/p109_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts_R.json > logs/prompts_R.log 2>&1 \
  || { tail -4 logs/prompts_R.log; say "PROMPTS FAIL (R)"; finish 19; }
grep -a "PROMPTS" logs/prompts_R.log | tee -a summary.txt
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16 E4B_INT4_TILE_PROGRAMS=1 E4B_PAGED_SPEC=eagle3 E4B_PAGED_SPEC_K=3 E4B_PAGED_SPEC_HEAD=$HEAD_DIR"
# ---- Nsight Systems (P123's install, version and trace flags)
install_nsys(){ say "install nsight-systems-cli-$NSYS_VER (NVIDIA devtools apt repo; the image's own nsys is Nsight Compute's, never used)"
  { apt-get update -qq && apt-get install -y -qq --no-install-recommends gnupg2 wget ca-certificates \
    && wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | gpg --dearmor > /usr/share/keyrings/nvidia-devtools-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu$(. /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /" > /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-get update -qq && apt-get install -y -qq --no-install-recommends "nsight-systems-cli-$NSYS_VER"; } > logs/install_nsys.log 2>&1
  NSYS=$(dpkg -L "nsight-systems-cli-$NSYS_VER" 2>/dev/null | grep -E '/nsys$' | head -1)
  if [ -n "$NSYS" ] && [ -x "$NSYS" ]; then
    echo "nsys $("$NSYS" --version 2>&1 | tail -1) at $NSYS" | tee -a versions.txt summary.txt
  else say "NSYS INSTALL FAILED: $(tail -1 logs/install_nsys.log | cut -c1-160)"; finish 26; fi; }
nsys_export(){ local rep=$1; [ -s "$rep.nsys-rep" ] || { say "no report $rep.nsys-rep"; return 1; }
  "$NSYS" export --type sqlite --force-overwrite true -o "$rep.sqlite" "$rep.nsys-rep" > "logs/export_$(basename "$rep").log" 2>&1; }
gpu_free(){ local i; for i in $(seq 1 "${1:-120}"); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ] && return 0; sleep 1; done
  say "GPU still held after ${1:-120}s"; return 1; }
NSYS_VER=2025.6.1; NSYS=""
NSYS_TRACE="-t cuda-sw,nvtx --sample=none --cpuctxsw=none"
install_nsys
# capture N MODE STEPS SKIP OUT_PREFIX: one condition in its own process under nsys; returns the capture's rc
capture(){ local n=$1 mode=$2 steps=$3 skip=$4 pre=$5 al rc
  al=$(step_alarm 1800); say "capture ${pre}_n${n}_$mode (alarm=$al)"
  # shellcheck disable=SC2086  # ENGINE_ENV is an assignment list by design
  env PYTHONPATH= $ENGINE_ENV E4B_SHA=$E4B_T GNF4_SHA=$GNF4_T \
    perl -e "alarm $al; exec @ARGV" "$NSYS" profile $NSYS_TRACE --cuda-graph-trace=$mode --capture-range=cudaProfilerApi \
      --capture-range-end=stop --force-overwrite true -o "$W/census/${pre}_n${n}_$mode" python $W/p132_box.py --mode census \
      --n $n --capture $mode --skip $skip --steps $steps --prompts-r $W/prompts_R.json --out $W/${pre}_drv_n${n}_$mode.json \
      > logs/run_${pre}_n${n}_$mode.log 2>&1
  rc=$?; pkill -TERM -f "p132_box.py --mode census" 2>/dev/null; gpu_free 120 || rc=${rc/#0/32}
  [ "$rc" = 0 ] && { nsys_export "$W/census/${pre}_n${n}_$mode" || rc=30; }
  { echo -n "capture ${pre}_n${n}_$mode rc=$rc "; grep -a "^P132_CENSUS" logs/run_${pre}_n${n}_$mode.log | tail -1 | cut -c1-300; echo; } | tee -a summary.txt
  return $rc; }
# ---- the CANARY, first: buckets 1..4 captured and one replay per condition, then a short census read by the classifier
can_run $NEED_CANARY canary || finish 40
AL=$(step_alarm 900); say "canary (alarm=$AL)"
# shellcheck disable=SC2086
env PYTHONPATH= $ENGINE_ENV E4B_SHA=$E4B_T GNF4_SHA=$GNF4_T \
  perl -e "alarm $AL; exec @ARGV" python $W/p132_box.py --mode canary --prompts-r $W/prompts_R.json --out $W/canary.json > logs/canary.log 2>&1
rc=$?; grep -a "^P132_CANARY" logs/canary.log | tail -1 | cut -c1-400 | tee -a summary.txt
[ "$rc" = 0 ] || { tail -6 logs/canary.log | cut -c1-300 | tee -a summary.txt; say "CANARY FAILED (rc=$rc): buckets or replays -- stopping before the census"; finish 26; }
capture 2 graph $CANARY_STEPS 2 canary || { say "CANARY FAILED: the short graph-mode capture"; finish 26; }
capture 2 node $CANARY_STEPS 2 canary || { say "CANARY FAILED: the short node-mode capture"; finish 26; }
# Amendment 1: SC1b reads one row per PERIOD between consecutive replays (N - 1) and trims both edges, so N bracketed
# replays keep at most N - 3. The floor allows one long period on top: N - 4 (12 -> 8). Registration's N - 2 never passed.
python $W/sc1b_census.py arm --graph $W/census/canary_n2_graph.sqlite --node $W/census/canary_n2_node.sqlite --engine e4b --batch 2 \
  --classes $W/kernel_classes_nf4_v1_2.json --delim graph --min-steps $((CANARY_STEPS - 4)) --moe-layers $MOE_LAYERS \
  --out $W/canary_census.json 2>&1 | tail -1 | tee -a summary.txt
python - <<'PYC' || { say "CANARY FAILED: the classifier did not read the short census cleanly"; finish 26; }
import json, sys
c = json.load(open("/root/vsplit/canary_census.json"))
blocking = [x for x in (c.get("labels") or []) if x in ("CLASS_MAP_INCOMPLETE", "CLASS_MAP_SEGMENT_BROKEN", "NSYS_DIAGNOSTIC_ERRORS", "CLOCK_MISMATCH")]
rec = json.load(open("/root/vsplit/canary.json"))
rec["census"] = {"status": c.get("status"), "labels": c.get("labels"), "blocking": blocking}
ok = c.get("status") == "ok" and not blocking
rec["status"] = "ok" if (rec.get("status") == "ok" and ok) else "failed"
json.dump(rec, open("/root/vsplit/canary.json", "w"), indent=1)
print("VSPLIT_CANARY_CENSUS " + json.dumps(rec["census"]))
sys.exit(0 if ok else 1)
PYC
# ---- timing pass A: SD2's V1 terms, unprofiled, in this box (the anchors, verify_ms(n), full_ms(k): the rule's ms)
can_run $NEED_TIMING timing || finish 40
AL=$(step_alarm 1200); say "timing pass A (alarm=$AL)"
# shellcheck disable=SC2086
env PYTHONPATH= $ENGINE_ENV E4B_SHA=$E4B_T GNF4_SHA=$GNF4_T \
  perl -e "alarm $AL; exec @ARGV" python $W/p132_box.py --mode timing --prompts-r $W/prompts_R.json --out $W/timing.json > logs/timing.log 2>&1
rc=$?; grep -a "^P132_TIMING" logs/timing.log | tail -1 | tee -a summary.txt
[ "$rc" = 0 ] && [ -s $W/timing.json ] || { tail -6 logs/timing.log | cut -c1-300 | tee -a summary.txt; say "TIMING FAILED (rc=$rc) -- the reducer will VOID"; }
# ---- the census: n = 1 (decode bucket 1) and n = 2..4 (verify), graph mode then node mode, each its own process
for n in 1 2 3 4; do
  for MODE in graph node; do
    can_run $NEED_CAPTURE "capture n$n $MODE" || finish 40
    capture $n $MODE $STEPS $SKIP census || say "CAPTURE n$n $MODE FAILED -- the reducer will VOID"
  done
  U=$(python -c "import json; t=json.load(open('$W/timing.json')); a=(t['anchor_a_ms']+t['anchor_b_ms'])/2; print(round(a if $n==1 else t['k$((n - 1))']['verify_ms'], 4))" 2>/dev/null)
  G=$W/census/census_n${n}_graph.sqlite; N=$W/census/census_n${n}_node.sqlite
  if [ -s "$G" ] && [ -s "$N" ]; then
    python $W/sc1b_census.py arm --graph "$G" --node "$N" --engine e4b --batch $n --classes $W/kernel_classes_nf4_v1_2.json \
      --delim graph ${U:+--unprofiled-ms $U} --moe-layers $MOE_LAYERS --out $W/census_n$n.json 2>&1 | tail -1 | tee -a summary.txt
  else
    echo "census n=$n UNREAD (missing export) -- the reducer will VOID" | tee -a summary.txt
  fi
done
# ---- timing pass B: the same terms in a fresh process after the census (the self-pair: within 3 % of pass A)
can_run $NEED_TIMING "timing pass B" || finish 40
AL=$(step_alarm 1200); say "timing pass B (alarm=$AL)"
# shellcheck disable=SC2086
env PYTHONPATH= $ENGINE_ENV E4B_SHA=$E4B_T GNF4_SHA=$GNF4_T \
  perl -e "alarm $AL; exec @ARGV" python $W/p132_box.py --mode timing --prompts-r $W/prompts_R.json --out $W/timing_b.json > logs/timing_b.log 2>&1
rc=$?; grep -a "^P132_TIMING" logs/timing_b.log | tail -1 | sed 's/^/pass B /' | tee -a summary.txt
[ "$rc" = 0 ] && [ -s $W/timing_b.json ] || { tail -6 logs/timing_b.log | cut -c1-300 | tee -a summary.txt; say "TIMING B FAILED (rc=$rc) -- the reducer will VOID"; }
say "reduce"; python $W/vsplit_reduce.py --dir $W --out $W/verdict_vsplit.json 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict_vsplit.json ] || { say "REDUCER FAILED"; finish 22; }
finish 0
