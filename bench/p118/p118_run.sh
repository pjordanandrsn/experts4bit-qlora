#!/bin/bash
# bench/p118/p118_run.sh -- lane P118, BOX side (bench/p118/PREREG-p118.md; e4b#1313). Started detached by p118_drive.sh
# with the run's nonce; P118_RUN_NONCE first, then P118_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P118_SUCCESS.<nonce> only when the reducer ran (or, under P118_PROVE=1, when the proof passed).
#
# Does the decode lookahead (E4B_PAGED_DECODE_LOOKAHEAD=1: each decode step issued before the previous one is read back,
# its input ids taken on the device) decode the default serve_paged server faster at one request, with identical
# tokens? On ONE RTX 5090:
#   speed    the shipped default server (graphs on, max_seqs 16 named) in four arms of their own processes -- L0a L1a
#            L1b L0b (L0 = the switch unset, the shipped default; L1 = E4B_PAGED_DECODE_LOOKAHEAD=1) -- each through W16
#            (16 requests at once) and W1 (one request: the subject), with the lookahead's issue / collect / overlap
#            counts; then, outside the timed passes, one traced pass per workload (engines.step_trace) that prices the
#            host gap between decode steps;
#   rule     p118_reduce.py.
#   premise  on THIS card, before anything is fetched: tests/test_decode_graph_buckets.py, tests/test_kv_step_select.py
#            and tests/test_decode_lookahead_gpu.py, 16 passed, none skipped (the bucket graphs replay as the padded eager
#            step; the lookahead decodes the synchronous tokens, buckets and KV lengths through captured graphs, with
#            the overlap engaged); tests/test_decode_lookahead.py without its HTTP case, 18 passed, none skipped (the
#            protocol on the CPU and on this card's pinned staging and events) (rc 25)
#   order    install + tripwire; reducer and box self-tests; premise; fetch; NF4 arena bake; prompts; four arms; reduce
#
# Refusals: CUDA unusable 18 (the host floor; a torch that will not import stays 10), card class 15, disk 13, host RAM
# 16. Only 13 and 18 name the machine for the launcher's next draw.
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P118_GPU_CLASS P118_MIN_DISK_GB P118_MIN_RAM_GB P118_REHEARSAL P118_SHORT P118_LONG P118_REPS.
# P118_PROVE=1 is the PROVING RUN: everything above, end to end, on ibm-granite/granite-3.1-3b-a800m-instruct at 8 / 24
# tokens and 1 rep.
set -uo pipefail
W=/root/p118; mkdir -p $W/logs $W/traces; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p118: $*"; }
NONCE=${P118_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P118_RUN_NONCE.tmp && mv $W/P118_RUN_NONCE.tmp $W/P118_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P118_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P118_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P118_RUN_ID P118_DEADLINE_EPOCH P118_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=b4f93f1c62d1e3436ed45bec8ccd608c90433737   # grouped-nf4-gemm 0.42.0: e4b CI's pin at registration; a registered constant
PROVE=${P118_PROVE:-0}
if [ "$PROVE" = 1 ]; then
  MODEL=ibm-granite/granite-3.1-3b-a800m-instruct; REV=a02780686e08a03fe0d2679a293b5c74a90efa89   # P94's pin (SC1's proof model)
  SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1
  NEED_FETCH=300; NEED_BAKE=300; NEED_ARM=300                                                    # the proof's own time-left checks
else
  MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39                            # SC1's pin
  SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3
  NEED_FETCH=1800; NEED_BAKE=1200; NEED_ARM=600
fi
GPU_CLASS=${P118_GPU_CLASS:-5090}; MIN_DISK_GB=${P118_MIN_DISK_GB:-150}; MIN_RAM_GB=${P118_MIN_RAM_GB:-60}
REHEARSAL=${P118_REHEARSAL:-0}; SHORT=${P118_SHORT:-$SHORT_DEF}; LONG=${P118_LONG:-$LONG_DEF}; REPS=${P118_REPS:-$REPS_DEF}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset: the default server is the subject
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_PAGED_DECODE_LOOKAHEAD E4B_PAGED_STEP_TRACE E4B_PAGED_TRACE \
      E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN GNF4_GEMV_SPLITK \
      GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET
: > summary.txt; echo "$P118_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB prove=$PROVE short=$SHORT long=$LONG reps=$REPS" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 150 ] || [ "$MIN_RAM_GB" != 60 ] \
   || [ "$SHORT" != "$SHORT_DEF" ] || [ "$LONG" != "$LONG_DEF" ] || [ "$REPS" != "$REPS_DEF" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p118_box.py p118_reduce.py p109_box.py k8_bake.py calib.json test_decode_graph_buckets.py test_kv_step_select.py \
         test_decode_lookahead.py test_decode_lookahead_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p118/staged.sha256"; finish 9; }
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
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE}) -- registered host floor"
            echo "refused: cuda unusable (${CUDA_PROBE})" > REFUSAL; echo "BOX_REFUSED cuda=unusable" >> summary.txt; finish 18;;
  *) say "DUD BOX (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 61 GB checkpoint, its NF4 snapshot and arena)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P118_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P118_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P118_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_PIN=$(python -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==$TORCH_PIN" > $W/constraints.txt
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
assert md.version("grouped-nf4-gemm") == "0.42.0", md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora import serve_paged
from experts4bit_qlora.engines.paged_runner import PagedModelRunner
from experts4bit_qlora.engines.scheduler import ContinuousScheduler
from experts4bit_qlora.engines.step_trace import StepTrace  # noqa: F401  (the traced pass)
assert serve_paged._graphs_env("", "cuda", "all-vram") is True, "the subject is the graph default"
# the switch P118 reads: opt-in at this commit, wired into the engine's scheduler, on the runner's two entry points
assert serve_paged._lookahead_env("") is False and serve_paged._lookahead_env("1") is True, "the lookahead is opt-in"
assert "lookahead=cfg.decode_lookahead" in inspect.getsource(serve_paged.build_engine)
assert "lookahead" in inspect.signature(ContinuousScheduler).parameters
assert callable(getattr(PagedModelRunner, "issue_decode", None)) and callable(getattr(PagedModelRunner, "collect_decode", None))
import fp8_paged_attn, fp8_kv, nvme_arena  # noqa: F401  (decode kernel, KV append, arena bake)
import torch, triton
import experts4bit_qlora as e
open("/root/p118/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p118_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p118_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched: the graphs and the lookahead through them, then the protocol
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_decode_graph_buckets.py test_kv_step_select.py test_decode_lookahead_gpu.py -q -rs -p no:cacheprovider) > logs/premise_graphs.log 2>&1
rc=$?; LASTG=$(tail -1 logs/premise_graphs.log)
{ echo -n "premise graphs rc=$rc: "; echo "$LASTG"; } | tee -a summary.txt
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_decode_lookahead.py -k "not server_answers" -q -rs -p no:cacheprovider) > logs/premise_protocol.log 2>&1
rc2=$?; LASTP=$(tail -1 logs/premise_protocol.log)
{ echo -n "premise protocol rc=$rc2: "; echo "$LASTP"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTG" | grep -q "16 passed" && ! echo "$LASTG" | grep -q skipped \
   && [ "$rc2" = 0 ] && echo "$LASTP" | grep -q "18 passed" && ! echo "$LASTP" | grep -q skipped; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc/$rc2" > REFUSAL; finish 25
fi
# ---- the checkpoint, its NF4 arena (P39's k8_bake.py, as SC1 bakes it) and the prompts
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
perl -e "alarm 1200; exec @ARGV" python $W/p118_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts.json > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P109_PROMPTS" logs/prompts.log | tee -a summary.txt
# ---- the four arms: L0a L1a L1b L0b (ABBA), each a fresh process on the default graph server (only the switch differs)
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"
LA="E4B_PAGED_DECODE_LOOKAHEAD=1"
for TAG in L0a L1a L1b L0b; do
  ARM=${TAG:0:2}; KN=""; [ "$ARM" = L1 ] && KN=$LA
  can_run $NEED_ARM "arm $TAG" || finish 40
  AL=$(step_alarm 2400); say "arm $TAG (alarm=$AL)"
  # shellcheck disable=SC2086  # ENGINE_ENV and KN are assignment lists by design
  env PYTHONPATH= $ENGINE_ENV $KN P118_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p118_box.py --prompts $W/prompts.json --out $W/arm_$TAG.json --tag $TAG \
      --short $SHORT --long $LONG --reps $REPS --trace-dir $W/traces > logs/arm_$TAG.log 2>&1
  rc=$?
  { echo -n "arm $TAG rc=$rc "; grep -a "^P118_ARM" logs/arm_$TAG.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -6 logs/arm_$TAG.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG FAILED (rc=$rc) -- the reducer will VOID"; }
done
PF=""; [ "$PROVE" = 1 ] && PF="--proof"
say "reduce"; python $W/p118_reduce.py --dir $W --out $W/verdict.json --e4b-sha $E4B_SHA $PF 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict.json ] || { say "REDUCER FAILED"; finish 22; }
if [ "$PROVE" = 1 ]; then
  V=$(python -c "import json; print(json.load(open('$W/verdict.json'))['verdict'])")
  [ "$V" != VOID ] || { say "PROVE: the reducer VOIDed the proof -- not proved"; finish 27; }
  echo "PROVE -- the proving run: the whole box on $MODEL; reducer verdict $V (not a reading)" | tee -a summary.txt
  : > PROVED
fi
finish 0
