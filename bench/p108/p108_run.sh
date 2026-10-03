#!/bin/bash
# bench/p108/p108_run.sh -- lane P108, BOX side (bench/p108/PREREG-p108.md; e4b#359). Started detached by p108_drive.sh
# with the run's nonce; P108_RUN_NONCE first, then P108_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P108_SUCCESS.<nonce> only when the reducer ran (or, under P108_PROVE=1, when the proof passed).
#
# Gemma-4's paged path against transformers' forward, judged against transformers' own chaos, on ONE RTX 5090:
# google/gemma-4-26B-A4B-it loaded once through e4b's streaming loader (NF4 experts); p108_box.py reads 32 wikitext
# windows (1280-token prompts, so the 1024 sliding window binds; 256-token continuations) through the reference, three
# floor draws, the paged subject and two mutants; p108_reduce.py applies the registered rule.
#   premise  on THIS card, before anything is fetched: tests/test_gemma4_paged_window_gpu.py, 2 passed, none skipped (a
#            tiny Gemma-4's binding window through the paged path, stand-in and REAL fp8 kernel, within 2x its control
#            on every seed) (rc 25)
#   order    install + tripwire (#964 and #966 present); reducer self-test; premise; fetch; box; reduce
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P108_GPU_CLASS P108_MIN_DISK_GB P108_MIN_RAM_GB P108_REHEARSAL P108_MODEL_DIR P108_BOX_EXTRA P108_PREMISE_ALLOW_SKIP.
# P108_PROVE=1 is the PROVING RUN: the refusals, the install with its tripwire, the reducer self-test, the premise and
# an HF CDN egress probe; no model.
set -uo pipefail
W=/root/p108; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p108: $*"; }
NONCE=${P108_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P108_RUN_NONCE.tmp && mv $W/P108_RUN_NONCE.tmp $W/P108_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P108_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P108_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P108_RUN_ID P108_DEADLINE_EPOCH P108_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=34da93d6fe8d2a401b7001705658ce00b2b18213   # grouped-nf4-gemm v0.34.1 (fp8_paged_attn), e4b CI's pin; a registered constant
MODEL=google/gemma-4-26B-A4B-it; REV=4d7ae4984b7db7de8f8457170b3f1a419ee76d52   # P67's revision
GPU_CLASS=${P108_GPU_CLASS:-5090}; MIN_DISK_GB=${P108_MIN_DISK_GB:-100}; MIN_RAM_GB=${P108_MIN_RAM_GB:-90}
REHEARSAL=${P108_REHEARSAL:-0}; PROVE=${P108_PROVE:-0}
MODEL_DIR=${P108_MODEL_DIR:-}; BOX_EXTRA=${P108_BOX_EXTRA:-}; ALLOW_SKIP=${P108_PREMISE_ALLOW_SKIP:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever starts unset: the paged runner at e4b's defaults
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS
: > summary.txt; echo "$P108_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB prove=$PROVE model_dir=${MODEL_DIR:-hub} box_extra='${BOX_EXTRA}' allow_skip=$ALLOW_SKIP" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 100 ] || [ "$MIN_RAM_GB" != 90 ] || [ -n "$MODEL_DIR" ] \
   || [ -n "$BOX_EXTRA" ] || [ "$ALLOW_SKIP" != 0 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p108_box.py p108_reduce.py p97_box.py test_gemma4_paged_window_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p108/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk, the host RAM (#344)
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 52 GB checkpoint)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB (#344: Gemma-4's load faulted on 30-64 GiB hosts)"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P108_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P108_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P108_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_PIN=$(python -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0; torch held at $TORCH_PIN)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from transformers import Gemma4TextConfig
from experts4bit_qlora.engines import paged_attention
from experts4bit_qlora.serve_paged import _kv_geometry
assert hasattr(paged_attention, "_fallback_mask"), "needs #966 (the unbound fallback's masks)"
cfg = Gemma4TextConfig(vocab_size=256, hidden_size=128, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=32, global_head_dim=64, num_global_key_value_heads=1,
                       layer_types=["sliding_attention", "full_attention"], sliding_window=16, enable_moe_block=False,
                       attention_k_eq_v=True,   # as the real config: the full layers then take the global KV heads
                       hidden_size_per_layer_input=0, vocab_size_per_layer_input=256)
assert _kv_geometry(cfg) == ([2, 1], [32, 64]), "needs #964 (a per-layer config's KV geometry)"
import fp8_paged_attn  # noqa: F401  (the decode kernel)
import experts4bit_qlora as e, torch, triton
open("/root/p108/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p108_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_gemma4_paged_window_gpu.py -q -rs -s -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise run rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
grep -aE "gemma4 window seed" logs/premise.log | sed 's/^/  /' | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "2 passed" && ! echo "$LASTL" | grep -q skipped; then
  echo "premise ok" | tee -a summary.txt
elif [ "$ALLOW_SKIP" = 1 ] && [ "$rc" = 0 ]; then
  echo "premise ok (not fully run on this card: REHEARSAL)" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
if [ "$PROVE" = 1 ]; then
  echo "PROVE -- the proving run: install tripwired; reducer self-test passed; premise held on this card" | tee -a summary.txt
  say "PROVE: HF CDN egress probe (50 MB range, 20 s cap; recorded, not a refusal)"
  python - <<'PYE' 2>&1 | tail -1 | tee -a summary.txt forensics.txt
import time, urllib.request
t = time.time()
try:
    r = urllib.request.urlopen(urllib.request.Request("https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors",
                                                      headers={"Range": "bytes=0-52428799"}), timeout=20)
    n = len(r.read())
    print(f"PROVE hf_cdn_mbps={n / (time.time() - t) / 1e6:.1f} bytes={n}")
except Exception as e:
    print(f"PROVE hf_cdn_probe_failed {type(e).__name__}: {str(e)[:120]}")
PYE
  : > PROVED; finish 0
fi
SRC=$MODEL; [ -n "$MODEL_DIR" ] && SRC="$MODEL_DIR/${MODEL#*/}"
# ---- the checkpoint (a local P108_MODEL_DIR is a rehearsal)
if [ -z "$MODEL_DIR" ]; then
  can_run 2400 fetch || finish 40
  say "fetch $MODEL @ $REV"
  perl -e "alarm $(step_alarm 2700); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','*.jinja'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
  tail -1 logs/fetch.log | tee -a summary.txt
fi
# ---- the box: one load, the reference, the floors, the subject and the mutants
can_run 2700 box || finish 40
AL=$(step_alarm 4500); say "the box (alarm=$AL)"
# shellcheck disable=SC2086  # BOX_EXTRA is a list of rehearsal flags by design
perl -e "alarm $AL; exec @ARGV" python $W/p108_box.py --model "$SRC" --revision "$REV" --out $W/box.json $BOX_EXTRA > logs/box.log 2>&1
rc=$?
{ echo -n "box rc=$rc "; grep -aE "^P108_BOX" logs/box.log | tail -1 | cut -c1-1200; echo; } | tee -a summary.txt
[ "$rc" = 0 ] || { tail -8 logs/box.log; say "BOX FAILED"; finish 26; }
say "reduce"; python $W/p108_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish 0
