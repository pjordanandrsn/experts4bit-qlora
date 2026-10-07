#!/bin/bash
# bench/p115/p115c_run.sh -- lane P115 Phase C, BOX side (bench/p115/PREREG-p115.md, Amendment 2; e4b#1313). Started
# detached by p115c_drive.sh with the run's nonce; P115C_RUN_NONCE first, then P115C_EXIT_CODE.<nonce> +
# TP_DONE.<nonce> on every exit and P115C_SUCCESS.<nonce> only when the reducer ran (or, under P115C_PROVE=1, when the
# proof passed).
#
# Does `auto` on the four fusion knobs engage by structure on gpt-oss-20b and Qwen3.6-35B-A3B, refuse what `1` refuses,
# and compute nothing grossly wrong -- and does Granite-3.1-3b-a800m pass Phase B's full quality read under it? On ONE
# RTX 5090, per model, in order Granite (cheapest), gpt-oss, Qwen3.6 (a later model's fetch or bake failure still leaves
# the earlier reads): fetch, NF4 arena bake, P109's prompts in the model's tokenizer, then fresh processes of
# p115c_box.py -- serve off (twice through the prompts), serve on (auto), serve explicit (1; must refuse); then for
# gpt-oss and Qwen3.6 SANE off (R saved) and SANE on (scored against R), for Granite Phase B's quality off and on at
# Phase B's size (both texts, 48 windows each) -- and p115c_reduce.py.
#   premise  on THIS card, before anything is fetched: tests/test_decode_graph_buckets.py, test_kv_step_select.py,
#            test_fused_glue_decode_graphs_gpu.py and test_fusion_modes.py, 48 passed, none skipped (rc 25)
#
# Refusals: CUDA unusable 18 (Amendment 1's host floor; a torch that will not import stays 10), card class 15, disk 13,
# host RAM 16. Only 13 and 18 name the machine for the launcher's next draw.
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P115C_GPU_CLASS P115C_MIN_DISK_GB P115C_MIN_RAM_GB P115C_REHEARSAL P115C_NEW P115C_WINDOWS P115C_QWINDOWS P115C_CONT.
# P115C_PROVE=1 is the PROVING RUN: Granite alone, every process kind (serve, SANE and quality), at short lengths.
set -uo pipefail
W=/root/p115c; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p115c: $*"; }
NONCE=${P115C_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P115C_RUN_NONCE.tmp && mv $W/P115C_RUN_NONCE.tmp $W/P115C_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P115C_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P115C_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P115C_RUN_ID P115C_DEADLINE_EPOCH P115C_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=b4f93f1c62d1e3436ed45bec8ccd608c90433737   # grouped-nf4-gemm v0.42.0, Phase A/B's pin; a registered constant
PROVE=${P115C_PROVE:-0}
if [ "$PROVE" = 1 ]; then
  TAGS="granite"; NEW_DEF=8; WINDOWS_DEF=12; QWINDOWS_DEF=12; CONT_DEF=32
  NEED_FETCH=300; NEED_BAKE=300; NEED_PROC=240
else
  TAGS="granite gptoss qw36"; NEW_DEF=32; WINDOWS_DEF=12; QWINDOWS_DEF=48; CONT_DEF=128
  NEED_FETCH=1500; NEED_BAKE=900; NEED_PROC=600
fi
GPU_CLASS=${P115C_GPU_CLASS:-5090}; MIN_DISK_GB=${P115C_MIN_DISK_GB:-200}; MIN_RAM_GB=${P115C_MIN_RAM_GB:-60}
REHEARSAL=${P115C_REHEARSAL:-0}; NEW=${P115C_NEW:-$NEW_DEF}; WINDOWS=${P115C_WINDOWS:-$WINDOWS_DEF}; CONT=${P115C_CONT:-$CONT_DEF}
QWINDOWS=${P115C_QWINDOWS:-$QWINDOWS_DEF}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS \
      GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_TRITON_PREBIND
: > summary.txt; echo "$P115C_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA models=$TAGS gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB prove=$PROVE new=$NEW windows=$WINDOWS qwindows=$QWINDOWS cont=$CONT" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 200 ] || [ "$MIN_RAM_GB" != 60 ] \
   || [ "$NEW" != "$NEW_DEF" ] || [ "$WINDOWS" != "$WINDOWS_DEF" ] || [ "$QWINDOWS" != "$QWINDOWS_DEF" ] || [ "$CONT" != "$CONT_DEF" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p115c_box.py p115c_reduce.py p115_reduce.py p115_quality.py p109_box.py p110_box.py p108_box.py p97_box.py k8_bake.py p98_bake.py \
         calib.json test_decode_graph_buckets.py test_kv_step_select.py test_fused_glue_decode_graphs_gpu.py test_fusion_modes.py staged-c.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-c.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p115/staged-c.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk, the host RAM
# A GPU the image's torch cannot use is the REGISTERED HOST FLOOR (rent.py's class 18), as TC1 amendment 61 made it: exit 18 with
# a REFUSAL line, so the launcher names the machine and a relaunch cannot buy it again. Exit 10 named no machine:
# p115-5090-1 and tc1-5090-119 both drew Vast machine 34887 (CUDA error 803) and read HARNESS_ERROR. A torch that will not
# import is the image's fault, not the host's, so it stays 10.
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
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (two checkpoints, their NF4 snapshots and arenas)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P115C_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P115C_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P115C_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
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
assert md.version("grouped-nf4-gemm") == "0.42.0", md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora import serve_paged
from experts4bit_qlora.engines import glue_fuse
assert serve_paged.FUSION_KNOBS == ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
assert serve_paged._fusion_env("E4B_FUSE_T1_GLUE", "") == "0", "the knobs' default is 0 at this commit (Phase C reads auto, opt-in)"
assert serve_paged._fusion_env("E4B_FUSE_T1_GLUE", "auto") == "auto" and glue_fuse.fold_mode("E4B_FUSE_T1_GLUE", "1") == "1"
assert "fold_modes" in inspect.signature(__import__("experts4bit_qlora.engines.qkv_fuse", fromlist=["x"]).fuse_qkv).parameters
assert serve_paged._graphs_env("", "cuda", "all-vram") is True, "the subject is the graph default"
assert all(os.environ.get(k) is None for k in serve_paged.FUSION_KNOBS), "the knobs start unset; each process sets its arm"
import fp8_paged_attn, fp8_kv, nvme_arena, int4_b32  # noqa: F401  (decode kernel, KV append, arena bake, the glue kernels)
for k in ("rmsnorm_rows", "rmsnorm_resid_rows", "scaled_resid_add_rows", "rope_norm_heads", "rope_heads", "router_epilogue"):
    assert hasattr(int4_b32, k), f"int4_b32 has no {k}"
import experts4bit_qlora as e, torch, triton
open("/root/p115c/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p115c_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p115c_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
python $W/p115_quality.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "QUALITY SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched
(cd $W && PYTHONPATH='' perl -e 'alarm 1200; exec @ARGV' python -m pytest test_decode_graph_buckets.py test_kv_step_select.py test_fused_glue_decode_graphs_gpu.py test_fusion_modes.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise run rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "48 passed" && ! echo "$LASTL" | grep -q skipped; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
model_of(){ case $1 in gptoss) echo "openai/gpt-oss-20b 6cee5e81ee83917806bbde320786a8fb61efebee";;
  qw36) echo "Qwen/Qwen3.6-35B-A3B 995ad96eacd98c81ed38be0c5b274b04031597b0";;
  granite) echo "ibm-granite/granite-3.1-3b-a800m-instruct a02780686e08a03fe0d2679a293b5c74a90efa89";; esac; }
# gpt-oss is served on SC2g's e4b path (SC2G_E4B_ENV: native MXFP4 decode, NF4 prefill); the others on the server's defaults
model_env(){ case $1 in gptoss) echo "E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 GNF4_TRITON_PREBIND=1";; *) echo "";; esac; }
harness(){ python -c "import json,sys; json.dump({'model_tag': sys.argv[1], 'reason': sys.argv[2]}, open('$W/harness_' + sys.argv[1] + '.json', 'w'))" "$M" "$1"; say "HARNESS $M: $1"; }
for M in $TAGS; do
  read -r MODEL REV <<< "$(model_of $M)"
  can_run $NEED_FETCH "fetch $M" || { harness "fetch skipped: deadline"; continue; }
  say "fetch $M $MODEL @ $REV"
  perl -e "alarm $(step_alarm 3600); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', ignore_patterns=['original/*', 'metal/*', 'consolidated*'], max_workers=8))" > logs/fetch_$M.log 2>&1 \
    || { tail -2 logs/fetch_$M.log; harness "fetch failed"; continue; }
  SNAP=$(tail -1 logs/fetch_$M.log); echo "FETCH $M $MODEL@$REV $SNAP" | tee -a summary.txt
  can_run $NEED_BAKE "bake $M" || { harness "bake skipped: deadline"; continue; }
  mkdir -p $W/work_$M
  if [ "$M" = qw36 ]; then
    perl -e "alarm $(step_alarm 3600); exec @ARGV" python $W/p98_bake.py --model "$MODEL" --revision "$REV" --work $W/work_$M > logs/bake_$M.log 2>&1 \
      || { tail -3 logs/bake_$M.log; harness "bake failed"; continue; }
  else
    K8_MODEL="$SNAP" K8_WORK="$W/work_$M" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake_$M.log 2>&1 \
      || { tail -3 logs/bake_$M.log; harness "bake failed"; continue; }
  fi
  [ -e $W/work_$M/nf4.arena ] || { harness "bake failed: no arena"; continue; }
  python -c "import json,sys; sys.exit(0 if json.load(open('$W/work_$M/bake.json')).get('status') == 'OK' else 1)" || { harness "bake failed: bake.json status is not OK"; continue; }
  grep -a "BAKE" logs/bake_$M.log | tail -1 | cut -c1-200 | sed "s/^/$M /" | tee -a summary.txt
  perl -e "alarm 1200; exec @ARGV" python $W/p115c_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts_$M.json > logs/prompts_$M.log 2>&1 \
    || { tail -4 logs/prompts_$M.log; harness "prompts failed"; continue; }
  ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work_$M/nf4.arena E4B_PAGED_CALIB=$W/calib.json $(model_env $M)"
  STEPS="serve:off serve:on serve:explicit sane:off sane:on"             # gpt-oss, Qwen3.6
  [ "$M" = granite ] && STEPS="serve:off serve:on serve:explicit quality:off quality:on"   # Granite: Phase B's full read
  [ "$M" = granite ] && [ "$PROVE" = 1 ] && STEPS="serve:off serve:on serve:explicit sane:off sane:on quality:off quality:on"
  for STEP in $STEPS; do
    MODE=${STEP%%:*}; ARM=${STEP#*:}; VAL=0; [ "$ARM" = on ] && VAL=auto; [ "$ARM" = explicit ] && VAL=1
    KN="E4B_PAGED_FUSE_QKV=$VAL E4B_FUSE_T1_GLUE=$VAL E4B_FUSE_T1_GLUE_R2=$VAL E4B_FUSE_ROUTER_EPI=$VAL"
    GR=""; [ "$MODE" != serve ] && GR="E4B_PAGED_GRAPHS=0"            # SANE and quality build the default server eager
    WIN=$WINDOWS; REF=$W/work_$M/ref; [ "$MODE" = quality ] && { WIN=$QWINDOWS; REF=$W/work_$M/qref; }
    can_run $NEED_PROC "$M $MODE $ARM" || { say "skipped $M $MODE $ARM (deadline)"; continue; }
    AL=$(step_alarm 2400); say "$M $MODE $ARM (alarm=$AL)"
    # shellcheck disable=SC2086  # assignment lists by design
    env PYTHONPATH= $ENGINE_ENV $KN $GR P115C_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
      perl -e "alarm $AL; exec @ARGV" python $W/p115c_box.py --mode $MODE --model-tag $M --prompts $W/prompts_$M.json \
        --out $W/${MODE}_${M}_${ARM}.json --ref-dir $REF --new $NEW --windows $WIN --cont $CONT > logs/${MODE}_${M}_${ARM}.log 2>&1
    rc=$?
    { echo -n "$M $MODE $ARM rc=$rc "; grep -aE "^P115C_(SERVE|SANE|QUALITY)" logs/${MODE}_${M}_${ARM}.log | tail -1 | cut -c1-500; echo; } | tee -a summary.txt
    [ "$rc" = 0 ] || tail -4 logs/${MODE}_${M}_${ARM}.log | cut -c1-300 | tee -a summary.txt
  done
  rm -rf $W/work_$M/nf4snap                                # the snapshot is not needed after the model's processes (disk)
done
PF=""; [ "$PROVE" = 1 ] && PF="--proof"
say "reduce"; python $W/p115c_reduce.py --dir $W --out $W/verdict_c.json --e4b-sha $E4B_SHA --new $NEW $PF 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict_c.json ] || { say "REDUCER FAILED"; finish 22; }
if [ "$PROVE" = 1 ]; then
  V=$(python -c "import json; print(json.load(open('$W/verdict_c.json'))['verdict'])")
  [ "$V" != VOID ] || { say "PROVE: the reducer VOIDed the proof -- not proved"; finish 27; }
  echo "PROVE -- the proving run: the whole box on granite; reducer verdict $V (not a reading)" | tee -a summary.txt
  : > PROVED
fi
finish 0
