#!/bin/bash
# bench/i8dense/i8_run.sh -- the W8A16 quality PREMISE, BOX side (bench/i8dense/PREMISE-w8a16.md; e4b#1313). Generated from
# bench/p125/p125_run.sh by named edits: P125's refusals, install, tripwire, self-tests, premise, fetch, bake and window
# check byte for byte; the six speed arms and the prompts dropped; two quality arms, A then I8; this premise's reducer.
# Started detached by i8_drive.sh with the run's nonce; I8_RUN_NONCE first, then I8_EXIT_CODE.<nonce> + TP_DONE.<nonce>
# on every exit and I8_SUCCESS.<nonce> only when the reducer ran.
#
# The SPLIT premise: does int8 weight-only rounding of the attention projections alone (the head bf16) keep P125's
# quality gate with margin? On ONE RTX 5090, P125's subject exactly (e4b 4934aba9 + grouped-nf4-gemm 0.43.0,
# Qwen3-30B-A3B at ad44e777, the default server built eager at all-VRAM), each arm its own process, in this order:
#   A       the default: writes R's log-probs per gate (t1 196 windows, k16 123, c4 16; wikitext every 1024 tokens);
#   I8ATTN  the candidate: the attention projections rounded to their per-output-row int8 round trip, the head bf16;
#   I8HEAD  reported only: the lm_head rounded, the attention bf16 (a deadline skips it gracefully; it rules nothing);
#   rule    i8_reduce.py: I8ATTN alone, GO iff argmax >= 0.95 and d + 1.645 SE <= P125's bound at t1 AND k16, else
#           PARK; VOID on a record or window fault. A PREMISE, never a reading.
#
# Refusals: CUDA unusable 18, card class 15, disk 13, host RAM 16; premise tests 25; deadline skip 40.
set -uo pipefail
W=/root/i8dense; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] i8dense: $*"; }
NONCE=${I8_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/I8_RUN_NONCE.tmp && mv $W/I8_RUN_NONCE.tmp $W/I8_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > I8_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > I8_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in I8_RUN_ID I8_DEADLINE_EPOCH I8_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
E4B_H=$E4B_SHA                                    # the harness commit this kit shipped from (recorded, not installed)
E4B_T=4934aba9c88da7ff49e12e4911354538ae0fdf48   # P125's subject: e4b 0.50.0, the commit P125's box installed (p125-5090-2)
GNF4_SHA=6ee2e10408161a9d3c874975c9191a7f2957e6f4   # grouped-nf4-gemm 0.43.0: P125's pin
MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
NEED_FETCH=1800; NEED_BAKE=1200; NEED_QUALITY=1100
GPU_CLASS=${I8_GPU_CLASS:-5090}; MIN_DISK_GB=${I8_MIN_DISK_GB:-150}; MIN_RAM_GB=${I8_MIN_RAM_GB:-60}
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
echo "$I8_INSTANCE_ID" > INSTANCE_ID; : > summary.txt
echo "KNOBS e4b_target=$E4B_T e4b_harness=$E4B_H gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB" | tee -a summary.txt
if [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 150 ] || [ "$MIN_RAM_GB" != 60 ]; then
  echo "REHEARSAL -- NOT the premise: a knob is off its default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in i8_box.py i8_fakequant.py i8_reduce.py p125_box.py p125_reduce.py p109_box.py p115_quality.py p110_box.py p108_box.py \
         p97_box.py k8_bake.py calib.json test_decode_graph_buckets.py test_kv_step_select.py test_fused_glue_decode_graphs_gpu.py \
         test_gemv_bw_served_gpu.py test_int4_attn.py test_int4_attn_calib.py staged-i8dense.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-i8dense.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/i8dense/staged-i8dense.sha256"; finish 9; }
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
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$I8_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((I8_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( I8_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_PIN=$(python -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s, no cache"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input --no-cache-dir -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$E4B_T (P125's subject) + gnf4 @$GNF4_SHA (image python; transformers 5.17.0; torch held at $TORCH_PIN)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_T" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest numpy || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_T WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
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
# Amendment 1: the int4 head builds the graph server only with #1426's fix (the prefill graph read the head's .weight)
from experts4bit_qlora.engines.paged_runner import _output_width  # noqa: F401
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
open("/root/i8dense/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ndatasets {md.version('datasets')}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
for t in p125_reduce p125_box i8_fakequant i8_box i8_reduce; do
  python $W/$t.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SELF-TEST FAILED ($t)"; finish 21; }
done
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
# ---- Amendment 3: every gate's windows load, full, from the real corpora and tokenizer -- before any arm (rc 19)
CAPF=""
perl -e "alarm 1200; exec @ARGV" python $W/i8_box.py --windows-check --model "$MODEL" --revision "$REV" > logs/windows.log 2>&1 \
  || { tail -4 logs/windows.log | cut -c1-300 | tee -a summary.txt; say "WINDOWS FAIL: a gate's windows do not fit its corpus"; finish 19; }
grep -a "^I8_WINDOWS" logs/windows.log | cut -c1-400 | tee -a summary.txt
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"
# ---- the two quality arms: the server built eager; A first (it writes R's log-probs, which stay on the box)
for ARM in A I8ATTN I8HEAD; do
  if ! can_run $NEED_QUALITY "quality $ARM"; then
    [ "$ARM" = I8HEAD ] && { say "I8HEAD skipped on the deadline (reported only; the reducer still runs)"; break; }
    finish 40
  fi
  AL=$(step_alarm 3600); say "quality arm $ARM (alarm=$AL)"
  # shellcheck disable=SC2086
  env PYTHONPATH= $ENGINE_ENV E4B_PAGED_GRAPHS=0 I8_ARM=$ARM E4B_SHA=$E4B_T GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/i8_box.py --mode quality --out $W/quality_$ARM.json --ref-dir $W/work/ref \
      > logs/quality_$ARM.log 2>&1
  rc=$?
  { echo -n "quality $ARM rc=$rc "; grep -a "^I8_GATE\|^I8_QUALITY" logs/quality_$ARM.log | cut -c1-400; echo; } | tee -a summary.txt
  if [ "$rc" != 0 ]; then
    tail -6 logs/quality_$ARM.log | cut -c1-300 | tee -a summary.txt; say "QUALITY $ARM FAILED (rc=$rc) -- the reducer will VOID"
    [ "$ARM" = A ] && { say "no reference: the lever arms cannot score"; break; }
  fi
done
say "reduce"; python $W/i8_reduce.py --dir $W --out $W/verdict_i8.json 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict_i8.json ] || { say "REDUCER FAILED"; finish 22; }
finish 0
