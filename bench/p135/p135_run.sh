#!/bin/bash
# bench/p135/p135_run.sh -- P135 (Track B1's R-b), BOX side (bench/p135/PREREG-p135.md). Started detached by p135_drive.sh
# with the run's nonce; P135_RUN_NONCE first, then P135_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P135_SUCCESS.<nonce> only when the reducer ran.
#
# What do Mixtral's ON_auto folds buy at one row, on one RTX 5090? Derived from bench/fam/fam_run.sh's speed path
# (FAM Amendment 4) by named substitutions: Mixtral only (FAM Amendment 5's pin, disk floors and fetch), the server's own
# pool at FAM Amendment 7's 768 tokens a slot, p135_speed.py (all three folds swapped, bucket 1 only) and p135_reduce.py,
# and grouped-nf4-gemm at e4b CI's pin. One process times both settings on one model.
#   premise  on THIS card, before anything is fetched: tests/test_fusion_modes.py and
#            tests/test_fused_glue_decode_graphs_gpu.py, all passed, none skipped (rc 25)
#
# Refusals: CUDA unusable 18 (a torch that will not import stays 10), card class 15, disk 13, host RAM 16.
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P135_GPU_CLASS P135_MIN_DISK_GB P135_MIN_RAM_GB P135_REHEARSAL.
set -uo pipefail
W=/root/p135; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p135: $*"; }
NONCE=${P135_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P135_RUN_NONCE.tmp && mv $W/P135_RUN_NONCE.tmp $W/P135_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P135_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P135_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P135_RUN_ID P135_DEADLINE_EPOCH P135_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
# grouped-nf4-gemm at e4b CI's pin at registration (.github/workflows/ci.yml), force-installed after e4b; a registered constant
GNF4_SHA=724ccc454f006c1a46836e434e997f31f293747f
MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1; REV=eba92302a2861cdc0098cc54bc9f17cb2c47eb61   # FAM Amendment 5's pin
# Sized from FAM's Mixtral boxes (fetch 3000 s, bake 1500 s needs; Amendment 5) and Amendment 4's speed process
NEED_FETCH=3000; NEED_BAKE=1500; NEED_SPEED=1200; CAP_SPEED=1800
# FAM Amendment 5: Mixtral's 93.4 GB checkpoint, its NF4 snapshot and arena need about 145 GB; refuse below 220 GB free at
# the start and, after the installs, below 165 GB free before the fetch
DISK_DEF=220; FETCH_DISK_GB=165
GPU_CLASS=${P135_GPU_CLASS:-5090}; MIN_DISK_GB=${P135_MIN_DISK_GB:-$DISK_DEF}; MIN_RAM_GB=${P135_MIN_RAM_GB:-60}
REHEARSAL=${P135_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS \
      GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE GNF4_TRITON_PREBIND \
      E4B_INT4_WIDE_TILES E4B_FUSE_PREFILL_GLUE E4B_PREFILL_LEAN_DISPATCH
: > summary.txt; echo "$P135_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS run=$P135_RUN_ID e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != "$DISK_DEF" ] || [ "$MIN_RAM_GB" != 60 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p135_speed.py p135_reduce.py p115_quality.py p110_box.py p108_box.py p97_box.py k8_bake.py calib.json \
         test_fusion_modes.py test_fused_glue_decode_graphs_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p135/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched (P115 Amendment 1's host floor: 18 names the machine)
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
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the checkpoint, its NF4 snapshot and arena)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P135_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P135_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P135_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
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
import os, json, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora import serve_paged
from experts4bit_qlora.engines import glue_fuse
assert serve_paged.FUSION_KNOBS == ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
assert all(os.environ.get(k) is None for k in serve_paged.FUSION_KNOBS), "the knobs start unset; the speed process names them"
assert not glue_fuse.prefill_glue_on() and glue_fuse.rows_cap() == glue_fuse.DECODE_ROWS_MAX == 64, "decode-only folds"
import fp8_paged_attn, fp8_kv, nvme_arena, int4_b32, nf4_grouped  # noqa: F401
for k in ("rmsnorm_rows", "rmsnorm_resid_rows", "scaled_resid_add_rows", "rope_norm_heads", "rope_heads", "router_epilogue"):
    assert hasattr(int4_b32, k), f"int4_b32 has no {k}"
import experts4bit_qlora as e, torch, triton
open("/root/p135/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p135_speed.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SPEED BOX SELF-TEST FAILED"; finish 21; }
python $W/p135_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched
(cd $W && PYTHONPATH='' perl -e 'alarm 1200; exec @ARGV' python -m pytest test_fusion_modes.py test_fused_glue_decode_graphs_gpu.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise run rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q " passed" && ! echo "$LASTL" | grep -qE "skipped|failed|error"; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# ---- the checkpoint, its NF4 arena (P39's k8_bake.py), then the speed process and the reducer
can_run $NEED_FETCH "fetch" || finish 40
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$FETCH_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free before the fetch < ${FETCH_DISK_GB} GB"; echo "refused: disk before fetch ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
say "fetch $MODEL @ $REV"
perl -e "alarm $(step_alarm 3600); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', ignore_patterns=['original/*', 'metal/*', 'consolidated*'], max_workers=8))" > logs/fetch.log 2>&1 \
  || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
can_run $NEED_BAKE "bake" || finish 40
mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
python -c "import json,sys; sys.exit(0 if json.load(open('$W/work/bake.json')).get('status') == 'OK' else 1)" || { say "BAKE FAIL: bake.json status is not OK"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | cut -c1-200 | tee -a summary.txt
can_run $NEED_SPEED "speed" || finish 40
al=$(step_alarm $CAP_SPEED); say "speed (alarm=$al)"
# FAM's ON_auto knobs; the server built eager with 2 slots of 768 tokens (Amendment 7's size); the runners capture their own graphs
env PYTHONPATH= E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json \
  E4B_PAGED_FUSE_QKV=auto E4B_FUSE_T1_GLUE=auto E4B_FUSE_T1_GLUE_R2=auto E4B_FUSE_ROUTER_EPI=auto \
  E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=2 E4B_PAGED_MAX_TOKENS_PER_SEQ=768 E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
  perl -e "alarm $al; exec @ARGV" python $W/p135_speed.py --out $W/speed.json > logs/speed.log 2>&1
rc=$?
{ echo -n "speed rc=$rc "; grep -aE "^P135_SPEED" logs/speed.log | tail -1 | cut -c1-400; echo; } | tee -a summary.txt
[ "$rc" = 0 ] || tail -6 logs/speed.log | cut -c1-300 | tee -a summary.txt
say "reduce"
python $W/p135_reduce.py --rec $W/speed.json --out $W/verdict.json --e4b-sha $E4B_SHA 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict.json ] || { say "REDUCER FAILED"; finish 22; }
rm -rf $W/work/nf4snap                                      # the snapshot is not needed after the speed process (disk)
finish 0
