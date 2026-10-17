#!/bin/bash
# bench/p134/p134_run.sh -- lane P134, BOX side (bench/p134/PREREG-p134.md). Started detached by p134_drive.sh with the
# run's nonce; P134_RUN_NONCE first, then P134_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and P134_SUCCESS.<nonce>
# only when the reducer ran for every family (or, under P134_PROVE=1, when the proof passed for both).
#
# Does grouped-nf4-gemm K33's bandwidth-targeted NF4 decode GEMV (GNF4_GEMV_BW=1, the shipped plan table) decode the
# default serve_paged server faster at one request on Granite-3.1-3b-a800m and OLMoE-1B-7B on one RTX 5090, where auto
# engages only at Qwen3-30B-A3B's shapes, at no measurable quality cost? Derived from bench/p133/p133_run.sh by named
# substitutions:
#   families granite then olmoe, one after the other on one box, each with its own fetch, bake, prompts, fit, four arms,
#            two quality phases and verdict (records suffixed _<family>); a family's fit failure VOIDs that family
#            (no rerun) and the next family still runs; the run then exits 31;
#   card     the RTX 5090 class, compute capability 12.0 and >= 160 SMs;
#   premise  the 5090's counts (p134_box.PREMISE: 35 + 18, none skipped);
#   proof    P134_PROVE=1 runs BOTH families at 8 / 24 tokens, 1 rep, 12 windows x 32 positions, and the reducer reports
#            each family's g1 band from its own B0 W1 step (posted before the reading);
#   dry      P134_DRY=1: two tiny families (p133_tiny.py's fixture twice) on any CUDA card. Never a reading.
# Pins, arms, control, quality instrument, rule and refusal codes are P134's.
#
# Refusals: CUDA unusable 18 (the host floor; a torch that will not import stays 10), torch not the pin 28, card class 15,
# disk 13, host RAM 16. Only 13 and 18 name the machine for the launcher's next draw.
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P134_GPU_CLASS P134_MIN_DISK_GB P134_MIN_RAM_GB P134_REHEARSAL P134_SHORT P134_LONG P134_REPS P134_WINDOWS P134_CONT.
set -uo pipefail
W=/root/p134; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p134: $*"; }
NONCE=${P134_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P134_RUN_NONCE.tmp && mv $W/P134_RUN_NONCE.tmp $W/P134_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P134_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P134_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P134_RUN_ID P134_DEADLINE_EPOCH P134_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
E4B_PIN=c8d7e629e4c8431fc86970cf7195d8e67f72e22a      # the e4b package P134 measures (P133's pin); a registered constant
GNF4_SHA=6b7c2e7474d20c8843814dd46dad6cfc3d94b440      # grouped-nf4-gemm 0.46.0, exact; a registered constant
TORCH_WANT=2.8.0+cu128
[ "$E4B_SHA" = "$E4B_PIN" ] || { say "refusing: E4B_SHA $E4B_SHA is not the registered e4b pin $E4B_PIN"; finish 78; }
PROVE=${P134_PROVE:-0}; DRY=${P134_DRY:-0}
[ "$PROVE" = 1 ] && [ "$DRY" = 1 ] && { say "refusing: P134_PROVE and P134_DRY"; finish 78; }
GRAN=ibm-granite/granite-3.1-3b-a800m-instruct; GRAN_REV=a02780686e08a03fe0d2679a293b5c74a90efa89   # P94's pin
OLMOE=allenai/OLMoE-1B-7B-0924-Instruct; OLMOE_REV=7f1c97f440f06ce36705e4f2b843edb5925f4498         # P99's served pin
if [ "$DRY" = 1 ]; then
  FAMS="tinya tinyb"                                                                              # p133_tiny.py's fixture, twice
  SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3; WINDOWS_DEF=4; CONT_DEF=32
  NEED_FETCH=300; NEED_BAKE=300; NEED_FIT=240; NEED_ARM=240; NEED_QOFF=600; NEED_QON=300
elif [ "$PROVE" = 1 ]; then
  FAMS="granite olmoe"
  SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1; WINDOWS_DEF=12; CONT_DEF=32
  NEED_FETCH=300; NEED_BAKE=300; NEED_FIT=240; NEED_ARM=240; NEED_QOFF=600; NEED_QON=300           # the proof's own time-left checks
else
  FAMS="granite olmoe"
  SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3; WINDOWS_DEF=24; CONT_DEF=128
  NEED_FETCH=900; NEED_BAKE=900; NEED_FIT=300; NEED_ARM=400; NEED_QOFF=1500; NEED_QON=600
fi
model_of(){ case "$1" in granite) echo "$GRAN";; olmoe) echo "$OLMOE";; tiny*) echo "$W/fam/$1/p133-tiny";; esac; }
rev_of(){ case "$1" in granite) echo "$GRAN_REV";; olmoe) echo "$OLMOE_REV";; *) echo "";; esac; }
GPU_CLASS=${P134_GPU_CLASS:-5090}; MIN_DISK_GB=${P134_MIN_DISK_GB:-100}; MIN_RAM_GB=${P134_MIN_RAM_GB:-60}
REHEARSAL=${P134_REHEARSAL:-0}; SHORT=${P134_SHORT:-$SHORT_DEF}; LONG=${P134_LONG:-$LONG_DEF}; REPS=${P134_REPS:-$REPS_DEF}
WINDOWS=${P134_WINDOWS:-$WINDOWS_DEF}; CONT=${P134_CONT:-$CONT_DEF}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false P134_DRY=$DRY
# every serving lever and engine knob starts unset: the default server is the subject
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS \
      GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET
: > summary.txt; echo "$P134_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS run=$P134_RUN_ID e4b=$E4B_SHA gnf4=$GNF4_SHA families=${FAMS// /,} granite=$GRAN@$GRAN_REV olmoe=$OLMOE@$OLMOE_REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB prove=$PROVE dry=$DRY short=$SHORT long=$LONG reps=$REPS windows=$WINDOWS cont=$CONT" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 100 ] || [ "$MIN_RAM_GB" != 60 ] \
   || [ "$SHORT" != "$SHORT_DEF" ] || [ "$LONG" != "$LONG_DEF" ] || [ "$REPS" != "$REPS_DEF" ] \
   || [ "$WINDOWS" != "$WINDOWS_DEF" ] || [ "$CONT" != "$CONT_DEF" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
[ "$DRY" = 1 ] && { echo "DRY -- the \$0 dry run on two tiny families: NEVER a reading" | tee -a summary.txt; : > DRY; }
# ---- staged pieces, byte-for-byte
for f in p134_box.py p134_reduce.py p133_tiny.py p115_reduce.py p115_quality.py p109_box.py p110_box.py p108_box.py p97_box.py \
         k8_bake.py calib.json test_decode_graph_buckets.py test_kv_step_select.py test_gemv_bw_served_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p134/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: CUDA, the torch pin, the card, the disk, the host RAM
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
if not ok:
    print(f"no-cuda torch {torch.__version__}"); sys.exit(0)
cc, p = torch.cuda.get_device_capability(0), torch.cuda.get_device_properties(0)
print(f"ok torch={torch.__version__} cc={cc[0]}.{cc[1]} sms={p.multi_processor_count}")
PYC
)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in
  ok*) ;;
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE}) -- registered host floor"
            echo "refused: cuda unusable (${CUDA_PROBE})" > REFUSAL; echo "BOX_REFUSED cuda=unusable" >> summary.txt; finish 18;;
  *) say "DUD BOX (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
case "$CUDA_PROBE" in *" torch=$TORCH_WANT "*) ;; *) say "REFUSED: the image's torch is not the registered $TORCH_WANT ($CUDA_PROBE)"; echo "refused: torch ($CUDA_PROBE)" > REFUSAL; finish 28;; esac
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [ "$DRY" = 1 ]; then
  echo "DRY: card '$GPU_NAME' ($CUDA_PROBE) -- the class is not checked" | tee -a summary.txt
else
  case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
  SMS=${CUDA_PROBE##*sms=}
  case "$CUDA_PROBE" in *" cc=12.0 sms="*) ;; *) SMS=0;; esac
  [ "${SMS:-0}" -ge 160 ] 2>/dev/null || { say "REFUSED: $CUDA_PROBE -- the lane registers sm_120 with >= 160 SMs"; echo "refused: config $CUDA_PROBE" > REFUSAL; finish 15; }
fi
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (two checkpoints of 20.4 GB, their NF4 snapshots and arenas, the reference log-probs)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P134_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P134_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P134_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
echo "torch==${TORCH_WANT%%+*}" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0; bitsandbytes 0.50.2; torch held at $TORCH_WANT)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest numpy || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
perl -e 'alarm 600; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/gnf4src > logs/clone.log 2>&1 && git -C $W/gnf4src checkout -q $GNF4_SHA \
  || { say "CLONE FAIL (gnf4, for its kernel contract)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA WANT_TORCH=$TORCH_WANT python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the pin: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pin: {dg}"
assert md.version("grouped-nf4-gemm") == "0.46.0", md.version("grouped-nf4-gemm")
assert md.version("bitsandbytes") == "0.50.2", md.version("bitsandbytes")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
import torch, triton
assert torch.__version__ == os.environ["WANT_TORCH"], torch.__version__
from experts4bit_qlora import serve_paged
from experts4bit_qlora.engines import hot_residency as hr
from experts4bit_qlora.engines.paged_runner import PagedModelRunner
assert serve_paged._graphs_env("", "cuda", "all-vram") is True, "the subject is the graph default"
assert "DEVICE_GROUPING[0] = True" in inspect.getsource(serve_paged._batched_graph_grouping)
assert hasattr(PagedModelRunner, "enable_decode_graphs") and isinstance(hr.DEVICE_GROUPING, list)
# the engagement boundary the lane reads: T == 1 singleton groups (the decode GEMV), T > 1 device grouping (the M-tile)
hr.DEVICE_GROUPING[0] = True
assert hr._collapsed_grouping(1, None) == (True, False) and hr._collapsed_grouping(4, None) == (False, True)
hr.DEVICE_GROUPING[0] = False
import nf4_grouped as ng
for n in ("_gemv_nf4_bw", "_nf4_prmt32", "_bw", "_bw_engages", "_bw_plan", "_bw_decode", "_sm_count", "_BW_SHAPES",
          "_BW_PLANS", "_BW_PLAN_DEFAULT", "_DOTPAD_CONFIGS", "dispatch_counts"):
    assert hasattr(ng, n), f"installed gnf4 lacks {n} (the GNF4_GEMV_BW route P134 reads)"
assert {"bw_tree", "bw_prmt32", "bw_splitk", "dotpad", "scalar"} <= set(ng.dispatch_counts()), ng.dispatch_counts()
assert not any(os.environ.get(k) for k in ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE", "GNF4_GEMV_DOTPAD"))
assert ng._BW_SHAPES == frozenset({(1536, 2048), (2048, 768)}) and ng._bw() == "auto", "0.46.0's auto, at Qwen3's shapes"
dev = torch.device("cuda")
cc, sm = torch.cuda.get_device_capability(), torch.cuda.get_device_properties(0).multi_processor_count
assert ng._sm_count(dev) == sm
if os.environ.get("P134_DRY") != "1":
    assert cc == (12, 0) and sm >= 160, (cc, sm)
# the gate the lane reads: auto engages only at Qwen3's two shapes (on >= 160 SMs), never at Granite's or OLMoE's,
# and dot-pad's table has neither family, so B0 is the scalar GEMV at their shapes on every part
assert ng._bw_engages(1536, 2048, dev) is (sm >= 160) and ng._bw_engages(2048, 768, dev) is (sm >= 160)
FAMILY_SHAPES = ((1024, 1536), (1536, 512), (2048, 2048), (2048, 1024))
assert not any(ng._bw_engages(n, k, dev) for n, k in FAMILY_SHAPES), "auto must not engage at the families' shapes"
assert not any(s in ng._DOTPAD_CONFIGS for s in FAMILY_SHAPES), "dot-pad must not serve the families' shapes"
assert ng._bw_decode(dev) == "prmt32", "prmt32 must be the decode on this compiled CUDA target"
assert ng._bw_plan(1536, 2048) == (16, 1024, 4, 1) and ng._bw_plan(2048, 768) == (16, 256, 4, 1), "K33's Qwen3 plans, shipped"
assert ng._bw_plan(1024, 1536) == (16, 512, 8, 1) and ng._bw_plan(1536, 512) == (16, 256, 8, 1), "K33's Granite plans, shipped"
assert ng._bw_plan(2048, 2048) == (16, 1024, 4, 1) and ng._bw_plan(2048, 1024) == (16, 256, 4, 1), "K33's OLMoE plans, shipped"
assert ng._bw_plan(512, 512) == ng._BW_PLAN_DEFAULT == (16, 256, 4, 1), "the default plan (the dry fixture's shapes)"
os.environ["GNF4_GEMV_BW"] = "1"
assert ng._bw() == "1" and all(ng._bw_engages(n, k, dev) for n, k in FAMILY_SHAPES + ((512, 256),)), "B1 engages everywhere"
del os.environ["GNF4_GEMV_BW"]
import fp8_paged_attn, fp8_kv, nvme_arena  # noqa: F401  (decode kernel, KV append, arena bake)
import experts4bit_qlora as e
open("/root/p134/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {cc}\nsm_count {sm}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p134_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p134_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
python $W/p115_quality.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "QUALITY SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched: the kernel's own contract, then the served route's; each
# read from its junit report against p134_box.PREMISE (pass count and exact skipped ids)
(cd $W/gnf4src/kernel && PYTHONPATH= TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_nf4_gemv_bw.py -q -rs -p no:cacheprovider --junitxml=$W/logs/premise_kernel.xml) > logs/premise_bw_contract.log 2>&1
rc=$?; { echo -n "premise kernel contract rc=$rc: "; tail -1 logs/premise_bw_contract.log; } | tee -a summary.txt
python $W/p134_box.py --premise $W/logs/premise_kernel.xml --expect kernel | tee -a summary.txt; pk=${PIPESTATUS[0]}
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_decode_graph_buckets.py test_kv_step_select.py test_gemv_bw_served_gpu.py -q -rs -p no:cacheprovider --junitxml=$W/logs/premise_served.xml) > logs/premise.log 2>&1
rc2=$?; { echo -n "premise served route rc=$rc2: "; tail -1 logs/premise.log; } | tee -a summary.txt
python $W/p134_box.py --premise $W/logs/premise_served.xml --expect served | tee -a summary.txt; ps=${PIPESTATUS[0]}
if [ "$rc" = 0 ] && [ "$pk" = 0 ] && [ "$rc2" = 0 ] && [ "$ps" = 0 ]; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc/$pk/$rc2/$ps" > REFUSAL; finish 25
fi
PF=""; [ "$PROVE" = 1 ] && PF="--proof"; [ "$DRY" = 1 ] && PF="--dry"
BW="GNF4_GEMV_BW=1"
FIT_VOID=0; REDUCED=""
# ---- per family, in order: the checkpoint (or the dry fixture), its NF4 arena (P39's k8_bake.py), the prompts, the fit
# check, the four speed arms, the two quality phases and the family's verdict
for FAM in $FAMS; do
  MODEL=$(model_of $FAM); REV=$(rev_of $FAM); FW=$W/work/$FAM; mkdir -p $FW
  echo "FAMILY $FAM $MODEL@$REV" | tee -a summary.txt
  can_run $NEED_FETCH "fetch $FAM" || finish 40
  if [ "$DRY" = 1 ]; then
    say "build the dry fixture for $FAM"
    perl -e "alarm $(step_alarm 1200); exec @ARGV" python $W/p133_tiny.py --out "$MODEL" > logs/fetch_$FAM.log 2>&1 || { tail -3 logs/fetch_$FAM.log; say "FIXTURE FAIL"; finish 11; }
    grep -a "^P133_TINY" logs/fetch_$FAM.log | tee -a summary.txt; SNAP=$MODEL
  else
    say "fetch $MODEL @ $REV"
    perl -e "alarm $(step_alarm 2700); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=8))" > logs/fetch_$FAM.log 2>&1 || { tail -2 logs/fetch_$FAM.log; say "DL FAIL"; finish 11; }
    SNAP=$(tail -1 logs/fetch_$FAM.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt
  fi
  [ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
  can_run $NEED_BAKE "bake $FAM" || finish 40
  say "bake the NF4 arena ($FAM)"
  K8_MODEL="$SNAP" K8_WORK="$FW" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake_$FAM.log 2>&1 \
    || { tail -3 logs/bake_$FAM.log; say "BAKE FAIL"; finish 12; }
  [ -e $FW/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
  grep -a "BAKE" logs/bake_$FAM.log | tail -1 | tee -a summary.txt
  PR=$W/prompts_$FAM.json
  perl -e "alarm 1200; exec @ARGV" python $W/p134_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $PR > logs/prompts_$FAM.log 2>&1 \
    || { tail -4 logs/prompts_$FAM.log; say "PROMPTS FAIL"; finish 19; }
  grep -a "^P109_PROMPTS" logs/prompts_$FAM.log | tee -a summary.txt
  ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$FW/nf4.arena E4B_PAGED_CALIB=$W/calib.json"
  # the fit check, before this family's arms: the shipped default (B0) builds and decodes W1. Not fitting VOIDs the family
  can_run $NEED_FIT "fit $FAM" || finish 40
  AL=$(step_alarm 1800); say "fit $FAM (alarm=$AL)"
  # shellcheck disable=SC2086  # ENGINE_ENV is an assignment list by design
  env PYTHONPATH= $ENGINE_ENV P134_ARM=B0 E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p134_box.py --mode fit --prompts $PR --out $W/fit_$FAM.json --short $SHORT > logs/fit_$FAM.log 2>&1
  rc=$?
  { echo -n "fit $FAM rc=$rc "; grep -a "^P134_FIT" logs/fit_$FAM.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
  if [ "$rc" != 0 ]; then
    tail -6 logs/fit_$FAM.log | cut -c1-300 | tee -a summary.txt
    echo "FIT VOID ($FAM): W1 does not fit the default server -- no arm runs for $FAM and no rerun is registered" | tee -a summary.txt
    python $W/p134_reduce.py --dir $W --family $FAM --out $W/verdict_$FAM.json --e4b-sha $E4B_SHA $PF 2>&1 | tee -a summary.txt
    FIT_VOID=1; REDUCED="$REDUCED $FAM"; continue
  fi
  # the four speed arms: B0a B1a B1b B0b (ABBA), each a fresh process on the default server (only the switch differs)
  ARMS_OK=1
  for TAG in B0a B1a B1b B0b; do
    ARM=${TAG:0:2}; KN=""; [ "$ARM" = B1 ] && KN=$BW
    can_run $NEED_ARM "arm $TAG $FAM" || finish 40
    AL=$(step_alarm 2400); say "arm $TAG $FAM (alarm=$AL)"
    # shellcheck disable=SC2086  # ENGINE_ENV and KN are assignment lists by design
    env PYTHONPATH= $ENGINE_ENV $KN P134_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
      perl -e "alarm $AL; exec @ARGV" python $W/p134_box.py --mode speed --prompts $PR --out $W/arm_${TAG}_$FAM.json --tag $TAG \
        --short $SHORT --long $LONG --reps $REPS > logs/arm_${TAG}_$FAM.log 2>&1
    rc=$?
    { echo -n "arm $TAG $FAM rc=$rc "; grep -a "^P134_ARM" logs/arm_${TAG}_$FAM.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
    [ "$rc" = 0 ] || { ARMS_OK=0; tail -6 logs/arm_${TAG}_$FAM.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG $FAM FAILED (rc=$rc) -- the reducer will VOID $FAM"; }
  done
  # the quality phase (STOP-5: not when one of this family's speed arms failed): OFF then ON, each a fresh process on the
  # default server built eager, one window per pass; OFF writes the reference's log-probs to work/<family>/ref
  if [ "$ARMS_OK" = 1 ]; then
    mkdir -p $FW/ref
    for PH in off on; do
      NEED=$NEED_QOFF; ARM=B0; KN=""; [ "$PH" = on ] && { NEED=$NEED_QON; ARM=B1; KN=$BW; }
      can_run $NEED "quality $PH $FAM" || finish 40
      AL=$(step_alarm 4200); say "quality $PH $FAM (alarm=$AL)"
      # shellcheck disable=SC2086
      env PYTHONPATH= $ENGINE_ENV $KN E4B_PAGED_GRAPHS=0 P134_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
        perl -e "alarm $AL; exec @ARGV" python $W/p134_box.py --mode quality --out $W/quality_${PH}_$FAM.json --ref-dir $FW/ref \
          --windows $WINDOWS --cont $CONT --group 1 > logs/quality_${PH}_$FAM.log 2>&1
      rc=$?
      { echo -n "quality $PH $FAM rc=$rc "; grep -a "^P134_QUALITY" logs/quality_${PH}_$FAM.log | tail -1 | cut -c1-900; echo; } | tee -a summary.txt
      [ "$rc" = 0 ] || { tail -6 logs/quality_${PH}_$FAM.log | cut -c1-300 | tee -a summary.txt; say "QUALITY $PH $FAM FAILED (rc=$rc) -- the reducer will VOID $FAM"; break; }
    done
  else
    echo "STOP-5 ($FAM): a speed arm failed -- the quality phase is not run (the reducer VOIDs $FAM)" | tee -a summary.txt
  fi
  say "reduce $FAM"; python $W/p134_reduce.py --dir $W --family $FAM --out $W/verdict_$FAM.json --e4b-sha $E4B_SHA $PF 2>&1 | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict_$FAM.json ] || { say "REDUCER FAILED ($FAM)"; finish 22; }
  REDUCED="$REDUCED $FAM"
done
verdict_of(){ python -c "import json; print(json.load(open('$W/verdict_$1.json'))['verdict'])"; }
for FAM in $FAMS; do echo "VERDICT $FAM $(verdict_of $FAM)$([ "$DRY" = 1 ] && echo ' (DRY: never a reading)')" | tee -a summary.txt; done
[ "$FIT_VOID" = 1 ] && { say "a family's fit check failed: VOID for it, no rerun (31)"; finish 31; }
if [ "$PROVE" = 1 ]; then
  for FAM in $FAMS; do
    [ "$(verdict_of $FAM)" != VOID ] || { say "PROVE: the reducer VOIDed $FAM -- not proved"; finish 27; }
  done
  echo "PROVE -- the proving run: the whole box on both families; reducer verdicts above (not a reading)" | tee -a summary.txt
  : > PROVED
fi
finish 0
