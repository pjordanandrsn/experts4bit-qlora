#!/bin/bash
# bench/p97/p97_run.sh -- lane P97, BOX side (bench/p97/PREREG-p97.md; e4b#564). Started detached by p97_drive.sh with
# the run's nonce; P97_RUN_NONCE first, then P97_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P97_SUCCESS.<nonce> only when the reducer ran (or, under P97_PROVE=1, when the proof passed).
#
# The first GPU reading of hybrid paged serving (#889, #897): on ONE RTX 5090, the paged runner -- fp8 KV on a compact
# pool, chunked prefill, batched decode, per-slot Gated DeltaNet state -- against transformers' own forward on the same
# in-process weights, teacher-forced over 4 wikitext windows (512-token prompts, 256-token continuations).
#   subject  Qwen/Qwen3.6-35B-A3B   (qwen3_5_moe: 30 linear-attention + 10 attention layers)
#   control  OLMoE-1B-7B-Instruct   (16 attention layers; its only paged error is the fp8 KV)
#   premise  on THIS card, before anything is fetched: tests/test_linear_state_gpu.py PASSES (not skips) -- the hybrid
#            path through gnf4's real fp8 kernel on tiny models, within its control (rc 25)
#   order    both checkpoints fetched (control, subject); then p97_box.py on the control, then on the subject (which
#            also runs the mutant pass: the state write-back dropped)
# The reducer applies the registered rule (subject mean KL <= 2 x max(control's, 1e-3), agreement >= control's - 0.02).
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P97_GPU_CLASS P97_MIN_DISK_GB P97_REHEARSAL P97_MODEL_DIR (local checkpoints) P97_BOX_EXTRA (p97_box.py flags)
# P97_PREMISE_ALLOW_SKIP. P97_PROVE=1 is the PROVING RUN: the refusals, the install with its tripwire, the reducer
# self-test, the premise and an HF CDN egress probe; no model.
set -uo pipefail
W=/root/p97; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p97: $*"; }
NONCE=${P97_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P97_RUN_NONCE.tmp && mv $W/P97_RUN_NONCE.tmp $W/P97_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P97_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P97_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P97_RUN_ID P97_DEADLINE_EPOCH P97_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=34da93d6fe8d2a401b7001705658ce00b2b18213   # grouped-nf4-gemm v0.34.1 (fp8_paged_attn), e4b CI's pin; a registered constant
SUBJ=Qwen/Qwen3.6-35B-A3B; SUBJ_REV=995ad96eacd98c81ed38be0c5b274b04031597b0
CTRL=allenai/OLMoE-1B-7B-0924-Instruct; CTRL_REV=7f1c97f440f06ce36705e4f2b843edb5925f4498
GPU_CLASS=${P97_GPU_CLASS:-5090}; MIN_DISK_GB=${P97_MIN_DISK_GB:-130}; REHEARSAL=${P97_REHEARSAL:-0}; PROVE=${P97_PROVE:-0}
MODEL_DIR=${P97_MODEL_DIR:-}; BOX_EXTRA=${P97_BOX_EXTRA:-}; ALLOW_SKIP=${P97_PREMISE_ALLOW_SKIP:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever starts unset: the paged runner at e4b's defaults
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID
: > summary.txt; echo "$P97_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA subject=$SUBJ@$SUBJ_REV control=$CTRL@$CTRL_REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB prove=$PROVE model_dir=${MODEL_DIR:-hub} box_extra='${BOX_EXTRA}' allow_skip=$ALLOW_SKIP" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 130 ] || [ -n "$MODEL_DIR" ] || [ -n "$BOX_EXTRA" ] || [ "$ALLOW_SKIP" != 0 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p97_box.py p97_reduce.py test_linear_state_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p97/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 72 GB and a 14 GB checkpoint)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P97_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P97_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P97_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0, the release the hybrid path was built on)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, importlib.metadata as md, importlib.util
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from transformers.cache_utils import LinearAttentionLayer  # noqa: F401  (the state carrier the pool fills)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet  # noqa: F401
from experts4bit_qlora.engines.linear_state import LinearStatePool, install  # noqa: F401  (#889)
from experts4bit_qlora.engines.paged_runner import kv_layer_map, kv_layers  # noqa: F401  (#897)
import fp8_paged_attn  # noqa: F401  (the decode kernel the subject's 10 attention layers read)
import experts4bit_qlora as e, torch, triton
mods = {m: importlib.util.find_spec(m) is not None for m in ("fla", "causal_conv1d", "kernels")}
open("/root/p97/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\ngated-deltanet kernel modules {mods}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p97_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched: the hybrid path through the real fp8 kernel on tiny
# models stays within its all-attention control. It must PASS: a skip (no sm_89+, no fp8_paged_attn) is a failure.
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_linear_state_gpu.py -q -rs -s -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise hybrid-on-the-kernel rc=$rc: "; grep -aE "^control worst" logs/premise.log | tail -1 | tr '\n' ' '; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "1 passed" && ! echo "$LASTL" | grep -q skipped; then
  say "premise held: the hybrid paged path on the real kernel stays within its control"
elif [ "$ALLOW_SKIP" = 1 ] && [ "$rc" = 0 ] && echo "$LASTL" | grep -q "1 skipped"; then
  say "premise SKIPPED on this card (REHEARSAL: P97_PREMISE_ALLOW_SKIP=1)"
else
  say "PREMISE FAILED: the hybrid paged path did not pass on the real kernel on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
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
# ---- the checkpoints: control, then subject (a local P97_MODEL_DIR is a rehearsal)
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
fetch(){ local TAG=$1 MID=$2 REV=$3
  [ -n "$MODEL_DIR" ] && return 0
  can_run 1800 "${TAG}_fetch" || return 40
  say "fetch $MID @ $REV"
  perl -e "alarm $(arm_alarm 2400); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json','*.jinja'], max_workers=8))" > logs/fetch_$TAG.log 2>&1 || { tail -2 logs/fetch_$TAG.log; say "DL FAIL $TAG"; return 11; }
  tail -1 logs/fetch_$TAG.log | tee -a summary.txt; }
fetch control "$CTRL" "$CTRL_REV"; CTRL_OK=$?; rec $CTRL_OK
fetch subject "$SUBJ" "$SUBJ_REV"; SUBJ_OK=$?; rec $SUBJ_OK
# box TAG MID REV -- p97_box.py at the registered shape (its defaults); P97_BOX_EXTRA only in a rehearsal
box(){ local TAG=$1 MID=$2 REV=$3 SRC AL; AL=$(arm_alarm 2700)
  SRC=$MID; [ -n "$MODEL_DIR" ] && SRC="$MODEL_DIR/${MID#*/}"
  can_run 1200 "box_$TAG" || return 40
  say "box $TAG: $SRC @ $REV (alarm=$AL)"
  # shellcheck disable=SC2086  # P97_BOX_EXTRA is a flag list by design (rehearsal only)
  perl -e "alarm $AL; exec @ARGV" python $W/p97_box.py --model "$SRC" --revision "$REV" --tag $TAG --out $W/p97_$TAG.json $BOX_EXTRA > logs/box_$TAG.log 2>&1
  local rc=$?
  { echo -n "box $TAG rc=$rc "; grep -aE "^P97_MODEL|^P97_MUTANT" logs/box_$TAG.log | tr '\n' ' ' | cut -c1-600; echo; } >> summary.txt
  [ "$rc" = 0 ] || { tail -5 logs/box_$TAG.log; rc=26; }
  return $rc; }
[ "$CTRL_OK" = 0 ] && { box control "$CTRL" "$CTRL_REV"; rec $?; }
[ "$SUBJ_OK" = 0 ] && { box subject "$SUBJ" "$SUBJ_REV"; rec $?; }
say "reduce"; python $W/p97_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish "$rc_any"
