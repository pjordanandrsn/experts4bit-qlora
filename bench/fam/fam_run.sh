#!/bin/bash
# bench/fam/fam_run.sh -- lane FAM, BOX side (bench/fam/PREREG-fam.md; e4b#1362). Started detached by fam_drive.sh with
# the run's nonce; FAM_RUN_NONCE first, then FAM_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# FAM_SUCCESS.<nonce> only when the reducer ran (or, under FAM_PROVE=1, when the proof passed).
#
# Does the B=1 fused stack, at T == 1 and T == 12, stay inside each family's OWN neutral floor? On ONE RTX 5090, ONE
# family per box (FAM_FAMILY: granite, gptoss or qw36; Amendment 1): fetch, NF4 arena bake, then fresh processes of
# fam_box.py -- OFF first (it writes every cell's reference), then each ON config -- and for gpt-oss the anchor's two
# processes on SC2g's path; then fam_reduce.py over that family.
#   premise  on THIS card, before anything is fetched: tests/test_fam_split1_gpu.py and tests/test_fusion_modes.py,
#            all passed, none skipped (rc 25)
#
# Refusals: CUDA unusable 18 (a torch that will not import stays 10), card class 15, disk 13, host RAM 16.
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# FAM_GPU_CLASS FAM_MIN_DISK_GB FAM_MIN_RAM_GB FAM_REHEARSAL FAM_CONT.
# FAM_PROVE=1 is the PROVING RUN: Granite alone, every process kind (OFF, ON_epi, ON_auto, the anchor's two on SC2g's
# path), at 32 teacher-forced positions.
set -uo pipefail
W=/root/fam; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] fam: $*"; }
NONCE=${FAM_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/FAM_RUN_NONCE.tmp && mv $W/FAM_RUN_NONCE.tmp $W/FAM_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > FAM_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > FAM_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in FAM_RUN_ID FAM_DEADLINE_EPOCH FAM_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=6ee2e10408161a9d3c874975c9191a7f2957e6f4   # grouped-nf4-gemm v0.43.0, e4b CI's pin at registration; a registered constant
PROVE=${FAM_PROVE:-0}
# Amendment 1: one family per box (FAM_FAMILY), every time-left check sized from fam-prove-1's measured wall time per
# process (OFF 1024 s, an ON config about 170 s at 32 positions; x 127/31 at 128) with a 1.5x margin, Qwen3.6 at 1.5x
# Granite's step. Each NEED + 600 s of fetch-back fits inside the family's guard (2.5 h / 3.75 h / 4.0 h; proof 1.25 h).
FAMILY=${FAM_FAMILY:-}
if [ "$PROVE" = 1 ]; then
  [ -z "$FAMILY" ] || [ "$FAMILY" = granite ] || { say "refusing: the proof is Granite's (FAM_FAMILY=$FAMILY)"; finish 78; }
  TAGS="granite"; CONT_DEF=32; NEED_FETCH=300; NEED_BAKE=300; NEED_OFF=1800; NEED_ON=400; CAP_OFF=2400; CAP_ON=900
else
  case "$FAMILY" in
    granite) NEED_FETCH=900;  NEED_BAKE=600; NEED_OFF=6300; NEED_ON=1100; CAP_OFF=7200;  CAP_ON=1800;;
    gptoss)  NEED_FETCH=1200; NEED_BAKE=900; NEED_OFF=6300; NEED_ON=1100; CAP_OFF=7200;  CAP_ON=1800;;
    qw36)    NEED_FETCH=2400; NEED_BAKE=900; NEED_OFF=9500; NEED_ON=1600; CAP_OFF=10800; CAP_ON=2700;;
    *) say "refusing: FAM_FAMILY must be granite, gptoss or qw36 (got '${FAMILY}')"; finish 78;;
  esac
  TAGS="$FAMILY"; CONT_DEF=128
fi
GPU_CLASS=${FAM_GPU_CLASS:-5090}; MIN_DISK_GB=${FAM_MIN_DISK_GB:-200}; MIN_RAM_GB=${FAM_MIN_RAM_GB:-60}
REHEARSAL=${FAM_REHEARSAL:-0}; CONT=${FAM_CONT:-$CONT_DEF}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS \
      GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_TRITON_PREBIND E4B_INT4_WIDE_TILES
: > summary.txt; echo "$FAM_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA family=${FAMILY:-proof} models=$TAGS gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB prove=$PROVE cont=$CONT" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 200 ] || [ "$MIN_RAM_GB" != 60 ] || [ "$CONT" != "$CONT_DEF" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in fam_box.py fam_reduce.py p115_quality.py p110_box.py p108_box.py p97_box.py k8_bake.py p98_bake.py calib.json \
         test_fam_split1_gpu.py test_fusion_modes.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/fam/staged.sha256"; finish 9; }
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
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (three checkpoints, their NF4 snapshots and arenas)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$FAM_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((FAM_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( FAM_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
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
assert md.version("grouped-nf4-gemm") == "0.43.0", md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora import serve_paged
assert serve_paged.FUSION_KNOBS == ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
assert serve_paged._fusion_env("E4B_FUSE_T1_GLUE", "auto") == "auto" and serve_paged._fusion_env("E4B_FUSE_T1_GLUE", "0") == "0"
assert all(os.environ.get(k) is None for k in serve_paged.FUSION_KNOBS), "the knobs start unset; each process names its config"
import fp8_paged_attn, fp8_kv, nvme_arena, int4_b32, nf4_grouped  # noqa: F401
assert "n_split" in inspect.signature(fp8_paged_attn.fp8_paged_decode_attention).parameters, "split1 needs n_split"
for k in ("rmsnorm_rows", "rmsnorm_resid_rows", "scaled_resid_add_rows", "rope_norm_heads", "rope_heads", "router_epilogue"):
    assert hasattr(int4_b32, k), f"int4_b32 has no {k}"
import experts4bit_qlora as e, torch, triton
open("/root/fam/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/fam_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/fam_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
python $W/p115_quality.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "QUALITY SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched
(cd $W && PYTHONPATH='' perl -e 'alarm 1200; exec @ARGV' python -m pytest test_fam_split1_gpu.py test_fusion_modes.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise run rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q " passed" && ! echo "$LASTL" | grep -qE "skipped|failed|error"; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
model_of(){ case $1 in gptoss) echo "openai/gpt-oss-20b 6cee5e81ee83917806bbde320786a8fb61efebee";;
  qw36) echo "Qwen/Qwen3.6-35B-A3B 995ad96eacd98c81ed38be0c5b274b04031597b0";;
  granite) echo "ibm-granite/granite-3.1-3b-a800m-instruct a02780686e08a03fe0d2679a293b5c74a90efa89";; esac; }
configs_of(){ case $1 in gptoss) echo "OFF ON_glue ON_r2 ON_epi ON_auto";; *) echo "OFF ON_auto";; esac; }
[ "$PROVE" = 1 ] && configs_of(){ echo "OFF ON_epi ON_auto"; }
# each config's four knobs, named explicitly (fam_box.CONFIGS; the box refuses anything else)
knobs_of(){ local q=0 g=0 r=0 e=0
  case $1 in ON_glue) g=auto;; ON_r2) r=auto;; ON_epi) e=auto;; ON_auto) q=auto; g=auto; r=auto; e=auto;; esac
  echo "E4B_PAGED_FUSE_QKV=$q E4B_FUSE_T1_GLUE=$g E4B_FUSE_T1_GLUE_R2=$r E4B_FUSE_ROUTER_EPI=$e"; }
SC2G="E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 GNF4_TRITON_PREBIND=1"
harness(){ python -c "import json,sys; json.dump({'model_tag': sys.argv[1], 'reason': sys.argv[2]}, open('$W/harness_' + sys.argv[1] + '.json', 'w'))" "$M" "$1"; say "HARNESS $M: $1"; }
box(){ # tag config path out ref [extra args]
  local tag=$1 config=$2 path=$3 out=$4 ref=$5; shift 5
  local need=$NEED_ON cap=$CAP_ON env_path=""; [ "$config" = OFF ] && { need=$NEED_OFF; cap=$CAP_OFF; }
  [ "$path" = sc2g ] && env_path=$SC2G
  can_run $need "$tag $config $path" || return 1
  local al; al=$(step_alarm $cap); say "$tag $config $path (alarm=$al)"
  # shellcheck disable=SC2086  # assignment lists by design
  env PYTHONPATH= $ENGINE_ENV $(knobs_of $config) $env_path E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=16 E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $al; exec @ARGV" python $W/fam_box.py --config $config --path $path --out $W/$out --ref-root $ref --cont $CONT "$@" \
    > logs/${out%.json}.log 2>&1
  local rc=$?
  { echo -n "$tag $config $path rc=$rc "; grep -aE "^FAM_BOX" logs/${out%.json}.log | tail -1 | cut -c1-400; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || tail -4 logs/${out%.json}.log | cut -c1-300 | tee -a summary.txt
  return $rc
}
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
  ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work_$M/nf4.arena E4B_PAGED_CALIB=$W/calib.json"
  # OFF first: it writes every cell's reference; an ON config with no reference refuses.
  box $M OFF default fam_${M}_OFF.json $W/work_$M/ref || { harness "OFF failed"; continue; }
  for C in $(configs_of $M); do
    [ "$C" = OFF ] && continue
    box $M $C default fam_${M}_${C}.json $W/work_$M/ref
  done
  # The anchor (gpt-oss in the reading; Granite in the proof, which runs every process kind): Phase C's setting on
  # SC2g's path -- wikitext, shape 12, set A -- reported, never gated.
  if [ "$M" = gptoss ] || [ "$PROVE" = 1 ]; then
    box $M OFF sc2g fam_${M}_anchor_OFF.json $W/work_$M/aref --tag anchor --texts wikitext --shapes 12 --sets A \
      && box $M ON_auto sc2g fam_${M}_anchor_ON_auto.json $W/work_$M/aref --tag anchor --texts wikitext --shapes 12 --sets A
  fi
  rm -rf $W/work_$M/nf4snap                                # the snapshot is not needed after the model's processes (disk)
done
PF="--families $TAGS"; [ "$PROVE" = 1 ] && PF="--proof"
say "reduce"; python $W/fam_reduce.py --dir $W --out $W/verdict.json --e4b-sha $E4B_SHA $PF 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict.json ] || { say "REDUCER FAILED"; finish 22; }
if [ "$PROVE" = 1 ]; then
  python -c "import json,sys; v=json.load(open('$W/verdict.json')); sys.exit(1 if any(x == 'VOID' for f in v['families'].values() for x in f['verdict'].values()) else 0)" \
    || { say "PROVE: the reducer VOIDed the proof -- not proved"; finish 27; }
  echo "PROVE -- the proving run: the whole box on granite; reducer verdicts (not a reading) above" | tee -a summary.txt
  : > PROVED
fi
finish 0
