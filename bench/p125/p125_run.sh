#!/bin/bash
# bench/p125/p125_run.sh -- lane P125, BOX side (bench/p125/PREREG-p125.md; e4b#1313). Derived from bench/p123/p123_run.sh
# (the refusals, install, tripwire, fetch, bake and prompts; the per-arm processes) and bench/p115/p115d_run.sh (the
# quality phases). Started detached by p125_drive.sh with the run's nonce; P125_RUN_NONCE first, then
# P125_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and P125_SUCCESS.<nonce> only when the reducer ran (or, under
# P125_PROVE=1, when the proof passed).
#
# Does calibrated int4 attention (B), and the calibrated int4 lm_head on top (C), license on the shipped default's
# single-stream decode, and what do they buy? On ONE RTX 5090, every arm its own process, every lever calibrated on the
# box exactly as a user's build runs it:
#   speed    A1 B1 C1 C2 B2 A2 (palindromic): the default graph server at 16 slots through P109's W16 and W1, with the
#            int4 census, each Int4Linear's digest, memory (free before load, peaks after the build, the first prefill
#            and the runs) and each arm's auto slot count at 2048 and 4096 tokens a slot;
#   quality  A B C M K, the server built eager: wikitext at one window per pass (t1, the one-row gemv path), wikitext
#            in 16-row pieces (k16, K16), c4val1 in 16-row pieces (the K8-style ppl, reported). A writes R first; M is
#            RTN attention (reported); K is B with its scales rolled one block (the sure-fail mutant);
#   rule     p125_reduce.py: VOID / READ, B's and C's licences (K8's budget in nats, argmax 0.95, UNDERPOWERED).
#   premise  on THIS card, before anything is fetched: P115 Phase D's four tests and the int4 attention tests
#            (39 passed, none skipped) (rc 25)
#   order    install + tripwire; self-tests; premise; fetch (the checkpoint, then C4 validation shards 0 and 1: the
#            calibration's text and c4val1's); NF4 arena bake; prompts; six speed arms; five quality arms; reduce
#
# Refusals: CUDA unusable 18 (the host floor; a torch that will not import stays 10), card class 15, disk 13, host RAM
# 16. Only 13 and 18 name the machine for the launcher's next draw.
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P125_GPU_CLASS P125_MIN_DISK_GB P125_MIN_RAM_GB P125_REHEARSAL P125_SHORT P125_LONG P125_REPS.
# P125_PROVE=1 is the PROVING RUN: everything above, end to end, on ibm-granite/granite-3.1-3b-a800m-instruct at 8 / 24
# tokens and 1 rep, the quality gates capped at 16 windows (its default leaves the fusions off: four int4 modules a
# layer; the calibration build time is measured here first).
set -uo pipefail
W=/root/p125; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p125: $*"; }
NONCE=${P125_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P125_RUN_NONCE.tmp && mv $W/P125_RUN_NONCE.tmp $W/P125_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P125_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P125_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P125_RUN_ID P125_DEADLINE_EPOCH P125_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=6ee2e10408161a9d3c874975c9191a7f2957e6f4   # grouped-nf4-gemm 0.43.0: P116's GEMV is its default; a registered constant
PROVE=${P125_PROVE:-0}
if [ "$PROVE" = 1 ]; then
  MODEL=ibm-granite/granite-3.1-3b-a800m-instruct; REV=a02780686e08a03fe0d2679a293b5c74a90efa89   # P94's pin (SC1's proof model)
  SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1; CAP=16
  NEED_FETCH=300; NEED_BAKE=300; NEED_ARM=300; NEED_QUALITY=600                                  # the proof's own time-left checks
else
  MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39                            # SC1's pin
  SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3; CAP=0
  NEED_FETCH=1800; NEED_BAKE=1200; NEED_ARM=600; NEED_QUALITY=1800
fi
GPU_CLASS=${P125_GPU_CLASS:-5090}; MIN_DISK_GB=${P125_MIN_DISK_GB:-150}; MIN_RAM_GB=${P125_MIN_RAM_GB:-60}
REHEARSAL=${P125_REHEARSAL:-0}; SHORT=${P125_SHORT:-$SHORT_DEF}; LONG=${P125_LONG:-$LONG_DEF}; REPS=${P125_REPS:-$REPS_DEF}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset: the default server is the subject; each arm sets only its own lever
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_SERVE_LMHEAD_INT4_CALIB \
      E4B_SERVE_DENSE_INT4_CALIB E4B_ATTN_INT4_WIDE E4B_ATTN_INT4_SMALLM E4B_CALIB_LAYERS_PER_PASS E4B_INT4_ARTIFACT_DIR \
      E4B_INT4_EXPECTED_FINGERPRINT E4B_INT4_DUMP_ARTIFACT_DIR E4B_INT4_ASSIGNMENT E4B_INT4_WIDE_TILES E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_PAGED_DECODE_LOOKAHEAD E4B_PAGED_STEP_TRACE E4B_PAGED_TRACE \
      E4B_PAGED_LAST_LOGITS E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN \
      GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET
: > summary.txt; echo "$P125_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB prove=$PROVE short=$SHORT long=$LONG reps=$REPS cap=$CAP" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 150 ] || [ "$MIN_RAM_GB" != 60 ] \
   || [ "$SHORT" != "$SHORT_DEF" ] || [ "$LONG" != "$LONG_DEF" ] || [ "$REPS" != "$REPS_DEF" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p125_box.py p125_reduce.py p109_box.py p115_quality.py p110_box.py p108_box.py p97_box.py k8_bake.py calib.json \
         test_decode_graph_buckets.py test_kv_step_select.py test_fused_glue_decode_graphs_gpu.py test_gemv_bw_served_gpu.py \
         test_int4_attn.py test_int4_attn_calib.py staged-p125.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p125.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p125/staged-p125.sha256"; finish 9; }
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
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 61 GB checkpoint, its NF4 snapshot and arena, the references)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P125_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P125_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P125_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_PIN=$(python -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s, no cache"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input --no-cache-dir -c $W/constraints.txt "$@" >> $log 2>&1; }
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
from experts4bit_qlora import serve_paged as sp
assert sp._graphs_env("", "cuda", "all-vram") is True, "the subject is the graph default"
# the shipped default (#1361): an unset knob resolves per family -- auto on Qwen3-MoE, 0 on Granite-MoE
unset = {k: sp.FUSION_UNSET for k in sp.FUSION_KNOBS}
m, s = sp.resolve_fusion_modes(dict(unset), "qwen3_moe")
assert set(m.values()) == {"auto"} and set(s.values()) == {"default-allowlisted"}, (m, s)
m, s = sp.resolve_fusion_modes(dict(unset), "granitemoe")
assert set(m.values()) == {"0"} and set(s.values()) == {"default-off"}, (m, s)
# the int4 attention paths the gates read: one row on the activation-quantised GEMV, 2..16 on K16, more on the bf16
# copy; fused q/k/v on the int4 store; the calibration on the box from C4
from experts4bit_qlora.engines.int4_attn import Int4Linear
assert (Int4Linear.GEMV_ROWS_MAX, Int4Linear.SMALLM_ROWS_MAX) == (1, 16), (Int4Linear.GEMV_ROWS_MAX, Int4Linear.SMALLM_ROWS_MAX)
assert callable(Int4Linear.fuse)
from experts4bit_qlora.engines.int4_attn_calib import calibrate_attention_hessians, enable_serve_attn_int4_calib  # noqa: F401
from int4_b32 import gemv_int4_b32, quant_x_rows  # noqa: F401
assert os.environ.get("E4B_CALIB_SOURCE") is None and os.environ.get("E4B_CALIB_NSEQ") is None, "the calibration at its defaults"
import nf4_grouped as ng
assert ng._bw() == "auto" and {(1536, 2048), (2048, 768)} <= set(ng._BW_SHAPES), (ng._bw(), ng._BW_SHAPES)
import fp8_paged_attn, fp8_kv, nvme_arena, datasets  # noqa: F401,E401
import torch, triton  # noqa: E401
import experts4bit_qlora as e
open("/root/p125/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ndatasets {md.version('datasets')}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p125_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p125_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_decode_graph_buckets.py test_kv_step_select.py test_fused_glue_decode_graphs_gpu.py test_gemv_bw_served_gpu.py test_int4_attn.py test_int4_attn_calib.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "39 passed" && ! echo "$LASTL" | grep -q skipped; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# ---- the checkpoint, C4 validation (shard 0: the calibration's text; shard 1: c4val1's windows), the NF4 arena, prompts
can_run $NEED_FETCH fetch || finish 40
say "fetch $MODEL @ $REV"
perl -e "alarm $(step_alarm 2700); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
perl -e "alarm 900; exec @ARGV" python - > logs/fetch_c4.log 2>&1 <<'PYF' || { tail -3 logs/fetch_c4.log; say "DL FAIL (C4)"; finish 11; }
import time
from datasets import load_dataset
for shard in ("00000", "00001"):
    t = time.time()
    ds = load_dataset("allenai/c4", data_files={"v": f"en/c4-validation.{shard}-of-00008.json.gz"}, split="v")
    print(f"C4_FETCH shard {shard}: {len(ds)} docs in {time.time() - t:.1f} s", flush=True)
PYF
grep -a "^C4_FETCH" logs/fetch_c4.log | tee -a summary.txt
can_run $NEED_BAKE bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt
perl -e "alarm 1200; exec @ARGV" python $W/p109_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts.json > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P109_PROMPTS" logs/prompts.log | tee -a summary.txt
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"
lever(){ case "$1" in A) echo "";; B|K) echo "E4B_SERVE_ATTN_INT4_CALIB=1";;
                      C) echo "E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_LMHEAD_INT4_CALIB=1";; M) echo "E4B_SERVE_ATTN_INT4=1";; esac; }
# ---- the six speed arms, palindromic: each a fresh process on the default graph server, the arm's lever alone
for TAG in A1 B1 C1 C2 B2 A2; do
  ARM=${TAG:0:1}
  can_run $NEED_ARM "arm $TAG" || finish 40
  AL=$(step_alarm 2400); say "speed arm $TAG (alarm=$AL)"
  # shellcheck disable=SC2086  # ENGINE_ENV and the lever are assignment lists by design
  env PYTHONPATH= $ENGINE_ENV $(lever $ARM) P125_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p125_box.py --mode speed --prompts $W/prompts.json --out $W/arm_$TAG.json \
      --tag $TAG --short $SHORT --long $LONG --reps $REPS > logs/arm_$TAG.log 2>&1
  rc=$?
  { echo -n "arm $TAG rc=$rc "; grep -a "^P125_ARM" logs/arm_$TAG.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -6 logs/arm_$TAG.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG FAILED (rc=$rc) -- the reducer will VOID"; }
done
# ---- the five quality arms: the server built eager; A first (it writes R's log-probs, which stay on the box)
CAPF=""; [ "$CAP" != 0 ] && CAPF="--max-windows $CAP"
for ARM in A B C M K; do
  can_run $NEED_QUALITY "quality $ARM" || finish 40
  AL=$(step_alarm 3600); say "quality arm $ARM (alarm=$AL)"
  # shellcheck disable=SC2086
  env PYTHONPATH= $ENGINE_ENV E4B_PAGED_GRAPHS=0 $(lever $ARM) P125_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p125_box.py --mode quality --out $W/quality_$ARM.json --ref-dir $W/work/ref \
      $CAPF > logs/quality_$ARM.log 2>&1
  rc=$?
  { echo -n "quality $ARM rc=$rc "; grep -a "^P125_GATE\|^P125_QUALITY" logs/quality_$ARM.log | cut -c1-400; echo; } | tee -a summary.txt
  if [ "$rc" != 0 ]; then
    tail -6 logs/quality_$ARM.log | cut -c1-300 | tee -a summary.txt; say "QUALITY $ARM FAILED (rc=$rc) -- the reducer will VOID"
    [ "$ARM" = A ] && { say "no reference: the other quality arms cannot score"; break; }
  fi
done
PF=""; [ "$PROVE" = 1 ] && PF="--proof --cap $CAP"
say "reduce"; python $W/p125_reduce.py --dir $W --out $W/verdict_p125.json --e4b-sha $E4B_SHA $PF 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict_p125.json ] || { say "REDUCER FAILED"; finish 22; }
if [ "$PROVE" = 1 ]; then
  V=$(python -c "import json; print(json.load(open('$W/verdict_p125.json'))['verdict'])")
  [ "$V" != VOID ] || { say "PROVE: the reducer VOIDed the proof -- not proved"; finish 27; }
  echo "PROVE -- the proving run: the whole box on $MODEL; reducer verdict $V (not a reading)" | tee -a summary.txt
  : > PROVED
fi
finish 0
