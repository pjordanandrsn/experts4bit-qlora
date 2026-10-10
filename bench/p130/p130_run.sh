#!/bin/bash
# bench/p130/p130_run.sh -- lane P130, BOX side (bench/p130/PREREG-p130.md; e4b#846). Derived from
# bench/p117/p117_run.sh by named substitutions, with P115's registered host floor (Amendment 1: torch that imports but
# cannot use the GPU exits 18) and P127's fetch watchdog (Amendment 2). Started detached by p130_drive.sh with the run's
# nonce; P130_RUN_NONCE first, then P130_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and P130_SUCCESS.<nonce> only
# when the reducer ran (or, under P130_PROVE=1, when the proof passed).
#
# Do #1583's prefill knobs make the served 512-token prefill faster, and does P1's arithmetic cost quality? On ONE RTX
# 5090: SC2e's int4 stack built by build_engine in each of the registered processes, with E4B_FUSE_PREFILL_GLUE (P1) set
# per process in the order 0 1 1 0 0 1 1 0. Each process captures the served first-chunk prefill graph with
# E4B_PREFILL_LEAN_DISPATCH (P2) off and on and times the two interleaved (Phase A); process 1 also runs Phase B's R, rep,
# floor and mutant, process 2 the subject P1 (P117's teacher-forced instrument). p130_reduce.py applies the rule.
#   premise  on THIS card, before anything is fetched: tests/test_prefill_graph_gpu.py, 9 passed, none skipped (rc 25)
#   order    refusals; install + tripwire; self-tests; premise; fetch (watchdog); bake; the processes; reduce
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P130_GPU_CLASS P130_MIN_DISK_GB P130_MIN_RAM_GB P130_REHEARSAL.
# P130_PROVE=1 is the PROVING RUN: everything above on Qwen3-30B-A3B itself at the proof's sizes (four processes, 16
# windows of 32 positions, 4 speed windows, 3 rounds): no other family engages the int4 K19 prefill route P2 changes.
set -uo pipefail
W=/root/p130; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p130: $*"; }
NONCE=${P130_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P130_RUN_NONCE.tmp && mv $W/P130_RUN_NONCE.tmp $W/P130_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P130_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P130_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P130_RUN_ID P130_DEADLINE_EPOCH P130_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=724ccc454f006c1a46836e434e997f31f293747f   # grouped-nf4-gemm v0.45.0, e4b CI's pin; a registered constant
PROVE=${P130_PROVE:-0}
MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39                            # SC1's pin, SC2e's and P117's model
ORDER="0 1 1 0 0 1 1 0"                                                                          # P1 per process: ABBA, twice
if [ "$PROVE" = 1 ]; then
  PROCS=4; WINDOWS=16; CONT=32; SPEED_WINDOWS=4; ROUNDS=3; WARM=1; EAGER_ROUNDS=1; EAGER_WINDOWS=2
  NEED_FETCH=2400; NEED_BAKE=1800; NEED_PROC=420                                                # the proof's own time-left checks
else
  PROCS=8; WINDOWS=64; CONT=128; SPEED_WINDOWS=16; ROUNDS=12; WARM=2; EAGER_ROUNDS=2; EAGER_WINDOWS=4
  NEED_FETCH=3600; NEED_BAKE=2700; NEED_PROC=600
fi
GPU_CLASS=${P130_GPU_CLASS:-5090}; MIN_DISK_GB=${P130_MIN_DISK_GB:-150}; MIN_RAM_GB=${P130_MIN_RAM_GB:-60}
REHEARSAL=${P130_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset; the reading then sets SC2e's int4 stack (below) and P1 per process
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS E4B_FUSE_COMBINE E4B_INT4_TILE_PROGRAMS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS E4B_PAGED_DECODE_LOOKAHEAD \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_PAGED_PREFILL_GRAPH E4B_PAGED_BULK_KV E4B_PAGED_LAST_LOGITS E4B_FUSE_PREFILL_GLUE E4B_PREFILL_LEAN_DISPATCH GNF4_GEMV_BW GNF4_PDL
# SC2e's served stack (bench/sc1/sc1_run.sh's SPEEDENV, FOLDS and ROUTEENV, byte for byte, + fused q/k/v), as P117's
FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
SPEEDENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 $FOLDS"
ROUTEENV="E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto"
LEVERS="$ROUTEENV $SPEEDENV E4B_PAGED_FUSE_QKV=1"
# the engine is built eager with ONE small slot and no prefill graph: the box makes its own runners (P117's build)
ENGINE_KNOBS="E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=1 E4B_PAGED_MAX_TOKENS_PER_SEQ=768 E4B_PAGED_PREFILL_GRAPH=0"
: > summary.txt; echo "$P130_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB prove=$PROVE procs=$PROCS windows=$WINDOWS cont=$CONT speed_windows=$SPEED_WINDOWS rounds=$ROUNDS" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 150 ] || [ "$MIN_RAM_GB" != 60 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p130_box.py p130_reduce.py p117_box.py p108_box.py p97_box.py k8_bake.py calib.json test_prefill_graph_gpu.py hf_fetch_watchdog.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p130/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the host floor (P115 Amendment 1), the card class, disk, RAM
CUDA_PROBE=$(python - 2>/dev/null <<'PYC'
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
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 61 GB checkpoint, its NF4 snapshot and arena, R's log-probs)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P130_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P130_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P130_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_PIN=$(python -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0; torch held at $TORCH_PIN)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=1.31,<2" pytest || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
assert md.version("grouped-nf4-gemm") == "0.45.0", md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
hub = md.version("huggingface_hub")
assert tuple(int(x) for x in hub.split(".")[:2]) >= (1, 31), f"huggingface_hub {hub} < 1.31 (the fetch watchdog)"
from experts4bit_qlora.engines import glue_fuse, hot_residency
from experts4bit_qlora.engines.paged_runner import PagedModelRunner, PrefillGraphRefused  # noqa: F401
assert glue_fuse.PREFILL_GLUE_ENV == "E4B_FUSE_PREFILL_GLUE" and isinstance(glue_fuse.PREFILL_FOLD_SEEN, dict)
assert glue_fuse.rows_cap("1") is None and glue_fuse.rows_cap("0") == 64, "P1 (#1583): the knob under test"
assert hot_residency.PREFILL_LEAN_DISPATCH_ENV == "E4B_PREFILL_LEAN_DISPATCH" and isinstance(hot_residency.K19_DISPATCH_SEEN, dict)
assert isinstance(hot_residency.ROUTE_SEEN, dict)
import int4_smallm
k19 = inspect.signature(int4_smallm.gemm_int4_b32_grouped_smallm).parameters
assert "scatter" in k19 and "gather_div" in k19, "P2 needs K19's lean dispatch (K23)"
for name in ("enable_prefill_graph", "disable_prefill_graph", "_prefill_forward", "_prefill_scope"):
    assert hasattr(PagedModelRunner, name), name
assert PagedModelRunner._PG_KEY == -1 and "last_logits" in inspect.signature(PagedModelRunner.__init__).parameters
import sys; sys.path[:0] = ["/root/p130"]
import p108_box, p117_box  # noqa: F401  (P108's and P117's helpers at their registered bytes)
assert hasattr(p117_box, "paged_pass") and hasattr(p117_box, "windows") and hasattr(p108_box, "_release")
import fp8_paged_attn, fp8_kv, nvme_arena  # noqa: F401  (the decode kernel, the KV append, the arena bake)
import experts4bit_qlora as e, torch, triton
open("/root/p130/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\nhuggingface_hub {hub}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p130_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/hf_fetch_watchdog.py --self-test 2>/dev/null | tail -1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "WATCHDOG SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_prefill_graph_gpu.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise run rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "9 passed" && ! echo "$LASTL" | grep -q skipped; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# ---- the checkpoint (under P127's byte-growth watchdog) and its NF4 arena (P39's k8_bake.py, as SC1 and P117 bake it)
can_run $NEED_FETCH fetch || finish 40
say "fetch $MODEL @ $REV"
FA=$(step_alarm 2700)
HF_HUB_VERBOSITY=info perl -e "alarm $((FA + 90)); exec @ARGV" python $W/hf_fetch_watchdog.py --repo "$MODEL" --revision "$REV" \
  --allow '*.safetensors' --allow '*.json' --allow 'tokenizer*' --allow '*.model' --allow '*.txt' --allow merges.txt \
  --allow vocab.json --max-workers 8 --poll-s 30 --stall-s 180 --max-restarts 3 --budget-s $FA \
  > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
can_run $NEED_BAKE bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt
# ---- the processes: P1 per process in the registered order; process 1 runs Phase B off, process 2 Phase B on
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json $ENGINE_KNOBS $LEVERS"
n=0
for P1 in $ORDER; do
  n=$((n + 1)); [ "$n" -le "$PROCS" ] || break
  PB=none; [ "$n" = 1 ] && PB=off; [ "$n" = 2 ] && PB=on
  can_run $NEED_PROC "process $n" || finish 40
  P1ENV=""; [ "$P1" = 1 ] && P1ENV="E4B_FUSE_PREFILL_GLUE=1"
  AL=$(step_alarm 2400); say "process $n: P1=$P1, Phase B $PB (alarm=$AL)"
  # shellcheck disable=SC2086  # ENGINE_ENV and P1ENV are assignment lists by design
  env PYTHONPATH= $ENGINE_ENV $P1ENV E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p130_box.py --out $W/proc_$n.json --proc $n --p1 $P1 --phase-b $PB \
      --ref-dir $W/work/ref --windows $WINDOWS --cont $CONT --speed-windows $SPEED_WINDOWS --rounds $ROUNDS --warm $WARM \
      --eager-rounds $EAGER_ROUNDS --eager-windows $EAGER_WINDOWS > logs/proc_$n.log 2>&1
  rc=$?
  { echo -n "process $n rc=$rc "; grep -aE "^P130_SPEED" logs/proc_$n.log | tail -1 | cut -c1-700; echo; } | tee -a summary.txt
  grep -aE "^P130_(ARM|BOX|QUALITY_ERROR)" logs/proc_$n.log | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -8 logs/proc_$n.log | cut -c1-300 | tee -a summary.txt; say "PROCESS $n FAILED (rc=$rc) -- the reducer will VOID"; }
done
PROVEFLAG=""; [ "$PROVE" = 1 ] && PROVEFLAG="--prove"
say "reduce"; python $W/p130_reduce.py --dir $W --out $W/verdict.json --e4b-sha $E4B_SHA $PROVEFLAG 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict.json ] || { say "REDUCER FAILED"; finish 22; }
if [ "$PROVE" = 1 ]; then
  V=$(python -c "import json; v = json.load(open('$W/verdict.json')); print(v['verdict'], (v.get('quality') or {}).get('verdict'))")
  case "$V" in VOID*|NO_READING*|*QUALITY_VOID) say "PROVE: the reducer read '$V' on the proof -- not proved"; finish 27;; esac
  echo "PROVE -- the proving run: the whole box on $MODEL at the proof's sizes; reducer verdict $V (not a reading)" | tee -a summary.txt
  : > PROVED
fi
finish 0
