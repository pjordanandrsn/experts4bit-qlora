#!/bin/bash
# bench/p121/p121_run.sh -- lane P121, BOX side (bench/p121/PREREG-p121.md; e4b#846). Derived from
# bench/p115/p115d_run.sh by named substitutions. Started detached by p121_drive.sh with the run's nonce; P121_RUN_NONCE
# first, then P121_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and P121_SUCCESS.<nonce> only when the reducer ran
# (or, under P121_PROVE=1, when the proof passed).
#
# Does K25 (E4B_NF4_GROUPED_SMALLM=auto, the default since P96, licensed on Granite and OLMoE only) cost quality or
# speed against the served NF4 M-tile (0) on Qwen3-30B-A3B at the served W16 step (128 routed rows)? On ONE RTX 5090:
#   speed    the shipped default server (graphs on, max_seqs 16 named) in four arms of their own processes -- K0a K1a K1b
#            K0b (K0 = 0, K1 = auto) -- each through W16 and W1, with the K25 / M-tile route tally;
#   quality  P115 Phase B's instrument at 16 windows a pass (every decode step 128 routed rows), two processes on the
#            default server built eager: K0 scores R, its floor and the mutant on wikitext and c4val1 and saves R; K1
#            scores ON against it. Skipped when a speed arm failed (the reducer VOIDs);
#   continuity (reported, no bar): P96's arms through the same instrument at T == 1 (one window a pass) on 12 wikitext
#            windows -- Cm (0 + T == 1 device grouping, the M-tile) scores R, Ct (1: K25 at T == 1) scores ON;
#   rule     p121_reduce.py.
#   premise  on THIS card, before anything is fetched: tests/test_decode_graph_buckets.py and
#            tests/test_k25_row_exact_gpu.py, 11 passed, none skipped (the bucket graphs replay as the padded eager step;
#            K25's rows are bit-equal alone and in a 16-row step, match the dequant oracle and replay in a graph) (rc 25)
#   order    install + tripwire; reducer, box and quality self-tests; premise; fetch; NF4 arena bake; prompts; four arms;
#            quality OFF then ON; continuity OFF then ON; reduce
#
# Refusals: CUDA unusable 18 (the host floor; a torch that will not import stays 10), card class 15, disk 13, host RAM
# 16. Only 13 and 18 name the machine for the launcher's next draw.
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P121_GPU_CLASS P121_MIN_DISK_GB P121_MIN_RAM_GB P121_REHEARSAL P121_SHORT P121_LONG P121_REPS P121_WINDOWS P121_CONT.
# P121_PROVE=1 is the PROVING RUN: everything above, end to end, on ibm-granite/granite-3.1-3b-a800m-instruct at 8 / 24
# tokens, 1 rep and 16 windows of 32 positions per text (Granite's W16 step is 128 routed rows too).
set -uo pipefail
W=/root/p121; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p121: $*"; }
NONCE=${P121_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P121_RUN_NONCE.tmp && mv $W/P121_RUN_NONCE.tmp $W/P121_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P121_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P121_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P121_RUN_ID P121_DEADLINE_EPOCH P121_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=56f90e3555b089cdc8fb508e42ea0d2c069ce8e5   # grouped-nf4-gemm 0.43.0 main at registration (K25 since #429); a registered constant
PROVE=${P121_PROVE:-0}
if [ "$PROVE" = 1 ]; then
  MODEL=ibm-granite/granite-3.1-3b-a800m-instruct; REV=a02780686e08a03fe0d2679a293b5c74a90efa89   # P94's pin (SC1's proof model)
  SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1; WINDOWS_DEF=16; CONT_DEF=32
  NEED_FETCH=300; NEED_BAKE=300; NEED_ARM=240; NEED_QOFF=600; NEED_QON=300; NEED_CONT=300                                     # the proof's own time-left checks
else
  MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39                            # SC1's pin
  SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3; WINDOWS_DEF=48; CONT_DEF=128
  NEED_FETCH=1800; NEED_BAKE=1200; NEED_ARM=600; NEED_QOFF=1800; NEED_QON=900; NEED_CONT=600
fi
GPU_CLASS=${P121_GPU_CLASS:-5090}; MIN_DISK_GB=${P121_MIN_DISK_GB:-150}; MIN_RAM_GB=${P121_MIN_RAM_GB:-60}
REHEARSAL=${P121_REHEARSAL:-0}; SHORT=${P121_SHORT:-$SHORT_DEF}; LONG=${P121_LONG:-$LONG_DEF}; REPS=${P121_REPS:-$REPS_DEF}
WINDOWS=${P121_WINDOWS:-$WINDOWS_DEF}; CONT=${P121_CONT:-$CONT_DEF}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset: the default server is the subject
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_PAGED_DECODE_LOOKAHEAD E4B_PAGED_STEP_TRACE E4B_PAGED_TRACE \
      E4B_PAGED_LAST_LOGITS E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN \
      GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET
: > summary.txt; echo "$P121_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB prove=$PROVE short=$SHORT long=$LONG reps=$REPS windows=$WINDOWS cont=$CONT" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 150 ] || [ "$MIN_RAM_GB" != 60 ] \
   || [ "$SHORT" != "$SHORT_DEF" ] || [ "$LONG" != "$LONG_DEF" ] || [ "$REPS" != "$REPS_DEF" ] || [ "$WINDOWS" != "$WINDOWS_DEF" ] || [ "$CONT" != "$CONT_DEF" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p121_box.py p121_reduce.py p115_quality.py p109_box.py p110_box.py p108_box.py p97_box.py k8_bake.py calib.json test_decode_graph_buckets.py test_k25_row_exact_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p115/staged.sha256"; finish 9; }
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
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 61 GB checkpoint, its NF4 snapshot and arena, ~8 GB of reference log-probs)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P121_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P121_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P121_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
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
import os, json, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
assert md.version("grouped-nf4-gemm") == "0.43.0", md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora import serve_paged
from experts4bit_qlora.engines import hot_residency as hr
assert serve_paged._graphs_env("", "cuda", "all-vram") is True, "the subject is the graph default"
# the fusion knobs: opt-in at this commit (the arms hold them unset; the default server's census is zero)
assert all(serve_paged._fusion_env(k, "") == "0" for k in serve_paged.FUSION_KNOBS), "the knobs are opt-in at this commit"
# K25: auto by default, at <= 256 routed rows; T == 1 stays on the decode GEMV under auto
assert hr._k25_mode_env() == "auto", hr._k25_mode_env()
import inspect, nf4_smallm
assert hasattr(nf4_smallm, "gemm_nf4_grouped_smallm"), "grouped-nf4-gemm without K25"
src = inspect.getsource(hr)
assert 'R_rows <= 256 and _k25_mode != "0"' in src, "K25's row cap"
assert 'k25_t1 = T == 1 and int4_stores is None and _k25_mode_env() == "1"' in src, "T == 1 takes K25 only under 1"
import nf4_grouped as ng
assert hasattr(ng, "gemm_4bit_grouped_captured"), "the M-tile K0 runs"
assert ng._bw() == "auto" and {(1536, 2048), (2048, 768)} <= set(ng._BW_SHAPES), (ng._bw(), ng._BW_SHAPES)
hr.DEVICE_GROUPING[0] = True
assert hr._collapsed_grouping(1, None) == (True, False) and hr._collapsed_grouping(4, None) == (False, True)
hr.DEVICE_GROUPING[0] = False
import fp8_paged_attn, fp8_kv, nvme_arena  # noqa: F401  (decode kernel, KV append, arena bake)
import torch, triton
import experts4bit_qlora as e
open("/root/p121/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p121_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p121_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
python $W/p115_quality.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "QUALITY SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_decode_graph_buckets.py test_k25_row_exact_gpu.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "11 passed" && ! echo "$LASTL" | grep -q skipped; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
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
perl -e "alarm 1200; exec @ARGV" python $W/p121_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts.json > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P109_PROMPTS" logs/prompts.log | tee -a summary.txt
# ---- the four speed arms: K0a K1a K1b K0b (ABBA), each a fresh process on the default graph server (only the knob differs)
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"
KN0="E4B_NF4_GROUPED_SMALLM=0"
KN1="E4B_NF4_GROUPED_SMALLM=auto"
ARMS_OK=1
for TAG in K0a K1a K1b K0b; do
  ARM=${TAG:0:2}; KN=$KN0; [ "$ARM" = K1 ] && KN=$KN1
  can_run $NEED_ARM "arm $TAG" || finish 40
  AL=$(step_alarm 2400); say "arm $TAG (alarm=$AL)"
  # shellcheck disable=SC2086  # ENGINE_ENV and KN are assignment lists by design
  env PYTHONPATH= $ENGINE_ENV $KN P121_ARM=$ARM P121_PROVE=$PROVE E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA perl -e "alarm $AL; exec @ARGV" python $W/p121_box.py --mode speed --prompts $W/prompts.json --out $W/arm_$TAG.json --tag $TAG --short $SHORT --long $LONG --reps $REPS > logs/arm_$TAG.log 2>&1
  rc=$?
  { echo -n "arm $TAG rc=$rc "; grep -a "^P121_ARM" logs/arm_$TAG.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { ARMS_OK=0; tail -6 logs/arm_$TAG.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG FAILED (rc=$rc) -- the reducer will VOID"; }
done
# ---- the quality phases (not when a speed arm failed): K0 then K1, each a fresh process on the default server built
# eager, 16 windows a pass; K0 writes R's log-probs to work/ref (they stay on the box)
if [ "$ARMS_OK" = 1 ]; then
  mkdir -p $W/work/ref
  for PH in off on; do
    ARM=K0; KN=$KN0; NEED=$NEED_QOFF; [ "$PH" = on ] && { ARM=K1; KN=$KN1; NEED=$NEED_QON; }
    can_run $NEED "quality $PH" || finish 40
    AL=$(step_alarm 3600); say "quality $PH (alarm=$AL)"
    # shellcheck disable=SC2086
    env PYTHONPATH= $ENGINE_ENV $KN E4B_PAGED_GRAPHS=0 P121_ARM=$ARM P121_PROVE=$PROVE E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA perl -e "alarm $AL; exec @ARGV" python $W/p121_box.py --mode quality --out $W/quality_$PH.json --ref-dir $W/work/ref --windows $WINDOWS --cont $CONT > logs/quality_$PH.log 2>&1
    rc=$?
    { echo -n "quality $PH rc=$rc "; grep -a "^P121_QUALITY" logs/quality_$PH.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
    [ "$rc" = 0 ] || { tail -6 logs/quality_$PH.log | cut -c1-300 | tee -a summary.txt; say "QUALITY $PH FAILED (rc=$rc) -- the reducer will VOID"; break; }
  done
  # ---- the continuity row (reported, no bar): P96's m and t arms, one window a pass, 12 wikitext windows; a failure here
  # is recorded and the lane goes on (the reducer reads the row as not run)
  mkdir -p $W/work/ref_c
  for PH in off on; do
    ARM=Cm; KN="E4B_NF4_GROUPED_SMALLM=0 E4B_NF4_T1_DEVICE_GROUPING=1"
    [ "$PH" = on ] && { ARM=Ct; KN="E4B_NF4_GROUPED_SMALLM=1 E4B_NF4_T1_DEVICE_GROUPING=0"; }
    can_run $NEED_CONT "continuity $PH" || break
    AL=$(step_alarm 1800); say "continuity $PH (alarm=$AL)"
    # shellcheck disable=SC2086
    env PYTHONPATH= $ENGINE_ENV $KN E4B_PAGED_GRAPHS=0 P121_ARM=$ARM P121_PROVE=$PROVE E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA perl -e "alarm $AL; exec @ARGV" python $W/p121_box.py --mode quality --out $W/continuity_$PH.json --ref-dir $W/work/ref_c --texts wikitext --windows 12 --group 1 --cont $CONT > logs/continuity_$PH.log 2>&1
    rc=$?
    { echo -n "continuity $PH rc=$rc "; grep -a "^P121_QUALITY" logs/continuity_$PH.log | tail -1 | cut -c1-400; echo; } | tee -a summary.txt
    [ "$rc" = 0 ] || { tail -4 logs/continuity_$PH.log | cut -c1-300 | tee -a summary.txt; say "CONTINUITY $PH FAILED (rc=$rc) -- reported as not run"; rm -f $W/continuity_$PH.json; break; }
  done
else
  echo "a speed arm failed -- the quality phases are not run (the reducer VOIDs)" | tee -a summary.txt
fi
say "reduce"; python $W/p121_reduce.py --dir $W --out $W/verdict.json --e4b-sha $E4B_SHA 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict.json ] || { say "REDUCER FAILED"; finish 22; }
if [ "$PROVE" = 1 ]; then
  V=$(python -c "import json; print(json.load(open('$W/verdict.json'))['verdict'])")
  [ "$V" != VOID ] || { say "PROVE: the reducer VOIDed the proof -- not proved"; finish 27; }
  echo "PROVE -- the proving run: the whole box on $MODEL; reducer verdict $V (not a reading)" | tee -a summary.txt
  : > PROVED
fi
finish 0
