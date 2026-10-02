#!/bin/bash
# bench/p98/p98_run.sh -- lane P98, BOX side (bench/p98/PREREG-p98.md; e4b#564). Started detached by p98_drive.sh with
# the run's nonce; P98_RUN_NONCE first, then P98_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P98_SUCCESS.<nonce> only when the reducer ran (or, under P98_PROVE=1, when the proof passed).
#
# Hybrid decode under CUDA graphs, through the production serving stack (#907, #908): on ONE RTX 5090, Qwen3.6-35B-A3B
# served by serve_paged.build_engine (the NF4 arena, the hybrid expert tier, all-VRAM placement, the fp8 KV on a compact
# 10-layer pool, the per-slot Gated DeltaNet state), three arms on fresh engines, each in its own process:
#   g  bucketed decode graphs (1,2,4,8,16), captured
#   e  the same buckets and grouping, every step eager on the same padded layout (capture=False): the bitwise oracle
#   p  plain eager decode (the serving default)
# over two workloads (W16: 16 staggered requests; W1: one request).
#   premise  on THIS card, before anything is fetched: the three hybrid GPU test files PASS, none skipped (rc 25)
#   order    fetch; bake the NF4 arena (p98_bake.py, the checkpoint pinned); arms g, e, p; reduce
# The reducer applies the registered rule (every bucket captured and replayed, no eager step, g's tokens == e's).
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P98_GPU_CLASS P98_MIN_DISK_GB P98_REHEARSAL P98_MODEL_DIR P98_BOX_EXTRA P98_BAKE_EXTRA P98_PREMISE_ALLOW_SKIP
# P98_ARMS. P98_PROVE=1 is the PROVING RUN: the refusals, the install with its tripwire, the reducer self-test, the
# premise and an HF CDN egress probe; no model.
set -uo pipefail
W=/root/p98; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p98: $*"; }
NONCE=${P98_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P98_RUN_NONCE.tmp && mv $W/P98_RUN_NONCE.tmp $W/P98_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P98_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P98_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P98_RUN_ID P98_DEADLINE_EPOCH P98_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=34da93d6fe8d2a401b7001705658ce00b2b18213   # grouped-nf4-gemm v0.34.1 (fp8_paged_attn, fused batch KV append, nvme_arena), e4b CI's pin
MODEL=Qwen/Qwen3.6-35B-A3B; REV=995ad96eacd98c81ed38be0c5b274b04031597b0
GPU_CLASS=${P98_GPU_CLASS:-5090}; MIN_DISK_GB=${P98_MIN_DISK_GB:-170}; REHEARSAL=${P98_REHEARSAL:-0}; PROVE=${P98_PROVE:-0}
MODEL_DIR=${P98_MODEL_DIR:-}; BOX_EXTRA=${P98_BOX_EXTRA:-}; BAKE_EXTRA=${P98_BAKE_EXTRA:-}; ALLOW_SKIP=${P98_PREMISE_ALLOW_SKIP:-0}
ARMS=${P98_ARMS:-g e p}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever starts unset: the engine at e4b's defaults
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_PLACEMENT
: > summary.txt; echo "$P98_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB prove=$PROVE model_dir=${MODEL_DIR:-hub} box_extra='${BOX_EXTRA}' bake_extra='${BAKE_EXTRA}' allow_skip=$ALLOW_SKIP arms='$ARMS'" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 170 ] || [ -n "$MODEL_DIR" ] || [ -n "$BOX_EXTRA" ] \
   || [ -n "$BAKE_EXTRA" ] || [ "$ALLOW_SKIP" != 0 ] || [ "$ARMS" != "g e p" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p98_box.py p98_bake.py p98_reduce.py calib.json test_linear_state_gpu.py test_hybrid_decode_graphs_gpu.py test_linear_state_graph_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p98/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 72 GB checkpoint, a 17 GB NF4 snapshot and its arena)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P98_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P98_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P98_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest fastapi || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md, importlib.util
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora.engines import linear_state, paged_runner
assert hasattr(linear_state, "_bucket_selector") and hasattr(linear_state.LinearStatePool, "_index"), "needs #907 + #908"
src = inspect.getsource(paged_runner.PagedModelRunner.enable_decode_graphs)
assert "_warm_linear_state" in src and "not captured yet" not in src, "enable_decode_graphs still refuses hybrids"
from experts4bit_qlora.serve_paged import build_engine  # noqa: F401
import fp8_paged_attn, fp8_kv, nvme_arena  # noqa: F401  (the kernel, the fused batch append, the arena)
assert hasattr(fp8_kv, "fp8_kv_append_bt1"), "gnf4 lacks the fused batch KV append the buckets need"
import experts4bit_qlora as e, torch, triton
mods = {m: importlib.util.find_spec(m) is not None for m in ("fla", "causal_conv1d", "kernels")}
open("/root/p98/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\ngated-deltanet kernel modules {mods}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p98_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched: the hybrid paged path through the real kernel, hybrid
# decode graphs under the scheduler against the padded eager step, and the linear state under real capture. All four
# tests must PASS: a skip (no sm_89+, a missing kernel) is a failure.
(cd $W && PYTHONPATH='' perl -e 'alarm 1200; exec @ARGV' python -m pytest test_linear_state_gpu.py test_hybrid_decode_graphs_gpu.py test_linear_state_graph_gpu.py -q -rs -s -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "4 passed" && ! echo "$LASTL" | grep -q skipped; then
  say "premise held: the hybrid path, its bucket graphs and its real capture on this card"
elif [ "$ALLOW_SKIP" = 1 ] && [ "$rc" = 0 ]; then
  say "premise not fully run on this card (REHEARSAL: P98_PREMISE_ALLOW_SKIP=1): $LASTL"
else
  say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
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
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
SRC=$MODEL; [ -n "$MODEL_DIR" ] && SRC="$MODEL_DIR/${MODEL#*/}"
# ---- the checkpoint (a local P98_MODEL_DIR is a rehearsal)
if [ -z "$MODEL_DIR" ]; then
  can_run 2400 fetch || finish 40
  say "fetch $MODEL @ $REV"
  perl -e "alarm $(arm_alarm 2400); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json','*.jinja'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
  tail -1 logs/fetch.log | tee -a summary.txt
fi
# ---- the NF4 arena the engine serves from (the checkpoint pinned)
can_run 2400 bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
# shellcheck disable=SC2086  # P98_BAKE_EXTRA is a flag list by design (rehearsal only)
perl -e "alarm $(arm_alarm 2400); exec @ARGV" python $W/p98_bake.py --model "$SRC" --revision "$REV" --work $W/work $BAKE_EXTRA > logs/bake.log 2>&1
rc=$?; { echo -n "bake rc=$rc "; python -c "import json; r=json.load(open('$W/work/bake.json')); print({k: r.get(k) for k in ('status','layers','experts','snapshot_gib','load_s','snapshot_s','bake_s','loaded_commit','err')})" 2>&1 | cut -c1-400; } | tee -a summary.txt
[ "$rc" = 0 ] || { tail -5 logs/bake.log; say "BAKE FAIL"; finish 12; }
# ---- the arms, each a fresh engine in its own process
arm(){ local A=$1 AL; AL=$(arm_alarm 2700)
  can_run 900 "arm_$A" || return 40
  say "arm $A (alarm=$AL)"
  # shellcheck disable=SC2086  # P98_BOX_EXTRA is a flag list by design (rehearsal only)
  perl -e "alarm $AL; exec @ARGV" python $W/p98_box.py --arm $A --model "$SRC" --revision "$REV" --arena $W/work/nf4.arena \
    --calib $W/calib.json --out $W/arm_$A.json $BOX_EXTRA > logs/arm_$A.log 2>&1
  local rc=$?
  { echo -n "arm $A rc=$rc "; grep -aE "^P98_ARM" logs/arm_$A.log | tail -1 | cut -c1-600; echo; } >> summary.txt
  [ "$rc" = 0 ] || { tail -8 logs/arm_$A.log; rc=26; }
  return $rc; }
for A in $ARMS; do arm $A; rec $?; done
say "reduce"; python $W/p98_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish "$rc_any"
