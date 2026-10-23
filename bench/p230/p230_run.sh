#!/bin/bash
# P230's box runner (bench/p230/PREREG-p230.md): P113's runner (bench/p113/p113_run.sh) on today's harness (P226's
# skeleton: the refusals, then the pre-install fit, before any install). Does grouped-nf4-gemm's programmatic dependent
# launch, as it ships (GNF4_PDL on, capped at 8 rows), decode the DEFAULT NF4 server's tokens exactly, and faster, at one
# request, under decode-only timing? On ONE RTX 5090: the default serve_paged server (graphs on, every serving lever
# unset) on Qwen3-30B-A3B's NF4 store, in four arms of their own processes, a palindrome -- OFF1 DEF1 DEF2 OFF2 -- each
# through W1 (one request: where the cap lets PDL engage) and W16 (16 at once: the control, where it cannot);
# p230_reduce.py applies the registered rule.
#   premise  on THIS card, before anything is fetched (rc 25): tests/test_decode_graph_buckets.py, 7 passed, and from a
#            grouped-nf4-gemm clone at GNF4_SHA, kernel/test_pdl.py, 26 passed, and K33's two PDL tests, 2 passed; none
#            skipped
#   order    refusals; the pre-install fit; STARTED; install + tripwire; self-tests; premise; fetch; bake; prompts;
#            four arms; reduce
# No power floor: the read is a ratio inside one box (the power limit is recorded). No rehearsal knobs: every reading
# knob is frozen. This script never creates compute.
set -uo pipefail
W=/root/p230; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p230: $*"; }
NONCE=${P230_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P230_RUN_NONCE.tmp && mv $W/P230_RUN_NONCE.tmp $W/P230_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P230_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P230_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P230_RUN_ID P230_DEADLINE_EPOCH P230_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949   # grouped-nf4-gemm v0.50.0: the released default (PDL on, cap 8; K33 with PDL; K39's prmt)
MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39                     # SC1's pin
SHORT=32; LONG=160; REPS=3                                                                # P113's lengths and reps
# the needs (PREREG "Budget"): the slowest measured fetch (806 s) and bake (362 s) x 1.5; an arm is P113's 600 s
NEED_FETCH=1210; NEED_BAKE=550; NEED_ARM=600
INSTALL_TO_GATE_S=450   # the two installs, the clone, the self-tests and the premise
RESERVE=600             # can_run's margin after a step: the reduce, the fetch-back and the teardown
GPU_CLASS=5090; MIN_DISK_GB=150; MIN_RAM_GB=60
case "$P230_RUN_ID" in
  p230-5090-1|p230-5090-2|p230-5090-r1|p230-5090-r2|p230-5090-r3|p230-5090-r4|p230-5090-r5|p230-5090-r6|p230-5090-r7|p230-5090-r8) :;;
  *) say "unregistered run ID: $P230_RUN_ID"; finish 78;;
esac
[ -z "${P230_GPU_CLASS:-}${P230_MIN_DISK_GB:-}${P230_MIN_RAM_GB:-}${P230_SHORT:-}${P230_LONG:-}${P230_REPS:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever, engine knob and kernel switch starts unset: the default server is the subject, and GNF4_PDL is
# set only in OFF's own process
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_PAGED_DECODE_LOOKAHEAD E4B_PAGED_STEP_TRACE E4B_PAGED_TRACE \
      E4B_PAGED_LAST_LOGITS E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN \
      GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET E4B_INT8_HEAD_BW E4B_K25_PLAN E4B_K25_LUT
: > summary.txt; echo "$P230_INSTANCE_ID" > INSTANCE_ID; echo "$P230_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB short=$SHORT long=$LONG reps=$REPS" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p230_box.py p230_reduce.py p109_box.py k8_bake.py calib.json test_decode_graph_buckets.py staged-p230.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p230.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p230/staged-p230.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: CUDA, the card class, the disk, the host RAM
CUDA_PROBE=$(python - <<'PYC' 2>/dev/null | tail -1
import sys
try:
    import torch
except Exception as e:
    print(f"no-torch {type(e).__name__}"); sys.exit(0)
try:
    ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
except Exception:
    ok = False
print("ok" if ok else f"no-cuda torch {torch.__version__}")
PYC
)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in
  ok) ;;
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE})"
            echo "refused: cuda unusable (${CUDA_PROBE})" > REFUSAL; finish 18;;
  *) say "IMAGE FAULT: torch does not import (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
[ "$CC" = 12.0 ] || { say "REFUSED: 5090 compute capability must be 12.0, got $CC"; echo "refused: cc $CC" > REFUSAL; finish 15; }
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge 30000 ] || { say "REFUSED: 5090 VRAM ${VRAM_MIB:-?} MiB below 30000 MiB floor"; echo "refused: vram ${VRAM_MIB:-?}" > REFUSAL; finish 15; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the checkpoint, the NF4 snapshot and arena)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
# ---- the pre-install fit check (bench/common/README.md, "Guard fit"): before any install and any staged script
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_FETCH + NEED_BAKE + 4 * NEED_ARM + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P230_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a start; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
# ---- the start: every pre-install check has passed; from here the box counts toward the box-started cap
: > $W/STARTED; echo "STARTED $(date -u +%FT%TZ)" | tee -a summary.txt
# the time-left check: FIT (the step's need plus the 600 s reserve) is computed once, compared, and printed as compared
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + 600)); left=$((P230_DEADLINE_EPOCH - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s (${need}s + the 600 s reserve), only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P230_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_FULL" = 2.8.0+cu128 ] || { say "REFUSED: torch image $TORCH_FULL is not 2.8.0+cu128"; finish 28; }
TORCH_PIN=2.8.0; echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0; torch held at $TORCH_PIN)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest numpy || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
assert md.version("grouped-nf4-gemm") == "0.50.0", md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
import torch
from experts4bit_qlora import serve_paged
assert serve_paged._graphs_env("", "cuda", "all-vram") is True, "the served default captures graphs"
import int4_b32, nf4_grouped
assert int4_b32.PDL_DEFAULT is True and int4_b32.PDL_MAX_ROWS_DEFAULT == 8, "the shipped switch: on, capped at 8 rows"
assert "griddepcontrol.wait" in inspect.getsource(int4_b32._pdl_enter.fn), "the preamble"
assert "_pdl_kw(dev, T)" in inspect.getsource(nf4_grouped._gemv_nf4_bw_launch), "K33 takes the switch"
assert nf4_grouped._bw_engages(1536, 2048, "cuda") and nf4_grouped._bw_engages(2048, 768, "cuda"), \
    "K33 must engage at Qwen3-30B-A3B's two expert shapes on this card (GNF4_GEMV_BW=auto, >= 160 SMs)"
from triton.backends.nvidia.compiler import CUDAOptions
assert "launch_pdl" in CUDAOptions.__dataclass_fields__, "this triton cannot launch with PDL"
assert torch.cuda.get_device_capability() >= (9, 0), "the card is below sm_90: PDL cannot engage"
import experts4bit_qlora as e, triton
open("/root/p230/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p230_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p230_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched: e4b's graph buckets, then gnf4's switch contract and K33's
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_decode_graph_buckets.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise run rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
perl -e 'alarm 600; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/gnf4src > logs/clone.log 2>&1 && git -C $W/gnf4src checkout -q $GNF4_SHA \
  || { say "CLONE FAIL (gnf4)"; finish 9; }
(cd $W/gnf4src/kernel && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_pdl.py -q -rs -p no:cacheprovider) > logs/premise_pdl.log 2>&1
rc2=$?; LASTL2=$(tail -1 logs/premise_pdl.log)
{ echo -n "premise GNF4_PDL contract rc=$rc2: "; echo "$LASTL2"; } | tee -a summary.txt
(cd $W/gnf4src/kernel && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest -q -rs -p no:cacheprovider \
  "test_nf4_gemv_bw.py::test_the_pdl_preamble_precedes_every_global_load" "test_nf4_gemv_bw.py::test_pdl_on_is_bitwise_pdl_off") > logs/premise_k33.log 2>&1
rc3=$?; LASTL3=$(tail -1 logs/premise_k33.log)
{ echo -n "premise K33 PDL rc=$rc3: "; echo "$LASTL3"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -qE '^7 passed' && ! echo "$LASTL" | grep -q skipped \
   && [ "$rc2" = 0 ] && echo "$LASTL2" | grep -qE '^26 passed' && ! echo "$LASTL2" | grep -q skipped \
   && [ "$rc3" = 0 ] && echo "$LASTL3" | grep -qE '^2 passed' && ! echo "$LASTL3" | grep -q skipped; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc rc2=$rc2 rc3=$rc3" > REFUSAL; finish 25
fi
# ---- the checkpoint, its NF4 arena (P39's k8_bake.py, as SC1 bakes it) and the prompts
FREE_GB=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED before fetch: disk ${FREE_GB:-?} GB"; finish 13; }
can_run $NEED_FETCH fetch || finish 40
say "fetch $MODEL @ $REV"
perl -e "alarm $(step_alarm 2700); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
can_run $NEED_BAKE bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt
perl -e "alarm 1200; exec @ARGV" python $W/p230_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts.json > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P109_PROMPTS" logs/prompts.log | tee -a summary.txt
# ---- the four arms: OFF1 DEF1 DEF2 OFF2 (a palindrome), each a fresh process on the default server (only the switch
# differs: OFF GNF4_PDL=0; DEF the shipped default, the switch and its cap unset)
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json"
for TAG in OFF1 DEF1 DEF2 OFF2; do
  ARM=${TAG:0:3}
  case "$ARM" in OFF) SS="GNF4_PDL=0";; DEF) SS="";; esac
  can_run $NEED_ARM "arm $TAG" || finish 40
  AL=$(step_alarm 1800); say "arm $TAG (alarm=$AL)"
  # shellcheck disable=SC2086  # ENGINE_ENV and SS are assignment lists by design
  env PYTHONPATH= $ENGINE_ENV $SS P230_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p230_box.py --prompts $W/prompts.json --out $W/arm_$TAG.json --tag $TAG \
      --short $SHORT --long $LONG --reps $REPS > logs/arm_$TAG.log 2>&1
  rc=$?
  { echo -n "arm $TAG rc=$rc "; grep -a "^P230_ARM" logs/arm_$TAG.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -6 logs/arm_$TAG.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG FAILED (rc=$rc) -- the reducer will VOID"; }
done
say "reduce"; python $W/p230_reduce.py --dir $W --out $W/verdict.json --e4b-sha $E4B_SHA 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict.json ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/verdict.json'))['verdict'])")
case "$V" in KEEP|NULL|SLOWER|FUNCTION_FAIL) finish 0;; *) say "P230 $V -- no reading"; finish 27;; esac
