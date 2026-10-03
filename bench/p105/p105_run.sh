#!/bin/bash
# bench/p105/p105_run.sh -- lane P105, BOX side (bench/p105/PREREG-p105.md; e4b#928). Started detached by p105_drive.sh
# with the run's nonce; P105_RUN_NONCE first, then P105_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P105_SUCCESS.<nonce> only when the reducer ran (or, under P105_PROVE=1, when the proof passed).
#
# Lanes P103/P104's question re-asked with a premise that is a distribution: the Gated DeltaNet kernels under e4b's hybrid paged path,
# on ONE RTX 5090 and ONE host. Qwen3.6-35B-A3B served by serve_paged.build_engine (P98's measurement, p98_box.py at its
# registered bytes), in three PHASES on the same box:
#   t   transformers' torch path (no fla, no causal_conv1d)
#   f   after `pip install flash-linear-attention==0.5.2`
#   fc  after `pip install causal-conv1d==1.7.0` as well
# Each phase: an engagement record (kernels_<phase>.json), the PREMISE, then P98's three arms on fresh engines (g bucketed
# decode graphs 1,2,4,8,16; e the same padded steps eager, the bitwise oracle; p plain eager) over P98's W16 and W1.
#   premise t        the three pinned hybrid GPU files + test_linear_state_chunk_matched_gpu.py +
#                    test_linear_state_dense_parity_gpu.py: 11 passed, none skipped
#   premise f / fc   test_hybrid_decode_graphs_gpu.py + test_linear_state_graph_gpu.py + the chunk-matched ALL-LINEAR test
#                    (bit for bit, 4 seeds) + the dense parity file (8 seeds each): 9 passed, none skipped. The two
#                    single-seed tiny-MoE hybrid checks (a seed lottery, lane P104) also run, REPORTED, not gating.
# Phase t's premise gates the lane (rc 25); a kernel phase's install or premise failure is recorded for the reducer.
#   order    install; tripwire (no kernels present); premise t; fetch; bake; arms t; install fla; probe; premise f; arms f;
#            install causal-conv1d; probe; premise fc; arms fc; reduce
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P105_GPU_CLASS P105_MIN_DISK_GB P105_REHEARSAL P105_MODEL_DIR P105_PREMISE_ALLOW_SKIP P105_PHASES. P105_PROVE=1 is the
# PROVING RUN: the refusals, the install with its tripwire, the reducer self-test, premise t, both kernel installs with
# their probes and premises, and an HF CDN egress probe; no model.
set -uo pipefail
W=/root/p105; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p105: $*"; }
NONCE=${P105_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P105_RUN_NONCE.tmp && mv $W/P105_RUN_NONCE.tmp $W/P105_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P105_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P105_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P105_RUN_ID P105_DEADLINE_EPOCH P105_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=34da93d6fe8d2a401b7001705658ce00b2b18213   # grouped-nf4-gemm v0.34.1 (fp8_paged_attn, fused batch KV append, nvme_arena), e4b CI's pin
MODEL=Qwen/Qwen3.6-35B-A3B; REV=995ad96eacd98c81ed38be0c5b274b04031597b0
FLA_PIN="flash-linear-attention==0.5.2"; CC_PIN="causal-conv1d==1.7.0"
GPU_CLASS=${P105_GPU_CLASS:-5090}; MIN_DISK_GB=${P105_MIN_DISK_GB:-170}; REHEARSAL=${P105_REHEARSAL:-0}; PROVE=${P105_PROVE:-0}
MODEL_DIR=${P105_MODEL_DIR:-}; ALLOW_SKIP=${P105_PREMISE_ALLOW_SKIP:-0}; PHASES=${P105_PHASES:-t f fc}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever starts unset: the engine at e4b's defaults
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_PLACEMENT USE_HUB_KERNELS
: > summary.txt; echo "$P105_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV kernels='$FLA_PIN $CC_PIN' gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB prove=$PROVE model_dir=${MODEL_DIR:-hub} allow_skip=$ALLOW_SKIP phases='$PHASES'" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 170 ] || [ -n "$MODEL_DIR" ] \
   || [ "$ALLOW_SKIP" != 0 ] || [ "$PHASES" != "t f fc" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p105_reduce.py p98_box.py p98_bake.py p98_reduce.py calib.json test_linear_state_gpu.py test_hybrid_decode_graphs_gpu.py test_linear_state_graph_gpu.py test_linear_state_chunk_matched_gpu.py test_linear_state_dense_parity_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p105/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 72 GB checkpoint, a 17 GB NF4 snapshot and its arena)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P105_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P105_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P105_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_PIN=$(python -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0; torch held at $TORCH_PIN)"
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
from experts4bit_qlora.engines import linear_state, paged_runner, hot_residency
assert hasattr(linear_state, "_bucket_selector") and hasattr(linear_state.LinearStatePool, "_index"), "needs #907 + #908"
assert hasattr(hot_residency._HotResidency, "_row_index"), "needs #918 (the #913 fix)"
src = inspect.getsource(paged_runner.PagedModelRunner.enable_decode_graphs)
assert "_warm_linear_state" in src and "not captured yet" not in src, "enable_decode_graphs still refuses hybrids"
from experts4bit_qlora.serve_paged import build_engine  # noqa: F401
import fp8_paged_attn, fp8_kv, nvme_arena  # noqa: F401  (the kernel, the fused batch append, the arena)
assert hasattr(fp8_kv, "fp8_kv_append_bt1"), "gnf4 lacks the fused batch KV append the buckets need"
mods = {m: importlib.util.find_spec(m) is not None for m in ("fla", "causal_conv1d", "kernels")}
assert not any(mods.values()), f"phase t must run transformers' torch path, but kernel modules are present: {mods}"
import experts4bit_qlora as e, torch, triton
open("/root/p105/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\ngated-deltanet kernel modules (phase t) {mods}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p105_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- per phase: what transformers resolves (the engagement record), then the premise on THIS card
probe(){ local P=$1
  python - "$P" "$TORCH_PIN" <<'PYK' > logs/kernels_$P.log 2>&1
import importlib.metadata as md, inspect, json, sys
phase, torch_pin = sys.argv[1:3]
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as m
rec = {"phase": phase, "torch": torch.__version__, "torch_held": torch.__version__.split("+")[0] == torch_pin}
for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
    impl = inspect.getclosurevars(getattr(m, name)).nonlocals.get("implementation")
    rec[name] = getattr(impl, "__module__", None)
for dist in ("flash-linear-attention", "fla-core", "causal-conv1d", "kernels"):
    try:
        rec[dist] = md.version(dist)
    except md.PackageNotFoundError:
        rec[dist] = None
json.dump(rec, open(f"/root/p105/kernels_{phase}.json", "w"), indent=1)
print("KERNELS", json.dumps(rec))
PYK
  local rc=$?; { echo -n "kernels $P rc=$rc "; tail -1 logs/kernels_$P.log | cut -c1-500; } | tee -a summary.txt; return $rc; }
premise(){ local P=$1 FILES WANT
  local LINEAR="test_linear_state_chunk_matched_gpu.py::test_an_all_linear_pool_equals_transformers_prefilled_in_the_same_chunks"
  if [ "$P" = t ]; then
    FILES="test_linear_state_gpu.py test_hybrid_decode_graphs_gpu.py test_linear_state_graph_gpu.py test_linear_state_chunk_matched_gpu.py test_linear_state_dense_parity_gpu.py"; WANT="11 passed"
  else
    FILES="test_hybrid_decode_graphs_gpu.py test_linear_state_graph_gpu.py $LINEAR test_linear_state_dense_parity_gpu.py"; WANT="9 passed"
    # the two single-seed tiny-MoE hybrid checks (lanes P103 and P104 showed them a seed lottery): REPORTED, never gating
    (cd $W && PYTHONPATH='' perl -e 'alarm 600; exec @ARGV' python -m pytest test_linear_state_gpu.py test_linear_state_chunk_matched_gpu.py::test_the_hybrid_paged_path_stays_within_its_control_against_chunk_matched_references -q -rs -s -p no:cacheprovider) > logs/moe_single_seed_$P.log 2>&1
    { echo -n "single-seed MoE checks $P (reported): "; grep -oaE "(chunk-matched: )?control worst [0-9][^|]*[|] hybrid worst [0-9.e+-]+" logs/moe_single_seed_$P.log | tr '\n' ';'; echo -n "| "; tail -1 logs/moe_single_seed_$P.log; } | tee -a summary.txt
  fi
  # shellcheck disable=SC2086  # FILES is a list by design
  (cd $W && PYTHONPATH='' perl -e 'alarm 1500; exec @ARGV' python -m pytest $FILES -q -rs -s -p no:cacheprovider) > logs/premise_$P.log 2>&1
  local rc=$? LASTL; LASTL=$(tail -1 logs/premise_$P.log)
  { echo -n "premise $P rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
  grep -oaE "dense seed [0-9]+: .*" logs/premise_$P.log | sed "s/^/  $P /" | tee -a summary.txt
  if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "$WANT" && ! echo "$LASTL" | grep -q skipped; then return 0; fi
  [ "$ALLOW_SKIP" = 1 ] && [ "$rc" = 0 ] && { say "premise $P not fully run on this card (REHEARSAL)"; return 0; }
  return 1; }
install_kernels(){ local P=$1 pin=$2
  say "phase $P: install $pin (torch held at $TORCH_PIN)"
  pipx logs/pip_$P.log 1200 --prefer-binary --no-build-isolation "$pin"; local rc=$?
  echo "install $P rc=$rc ($pin)" | tee -a summary.txt; return $rc; }
probe t >/dev/null; premise t || { say "PREMISE t FAILED on this card"; echo "premise t failed" > REFUSAL; finish 25; }
KSTATE=""   # per kernel phase: ok | install_failed | premise_failed (the reducer reads summary.txt and kernels_<phase>.json)
kernel_phase(){ local P=$1 pin=$2
  if ! install_kernels $P "$pin"; then KSTATE="$KSTATE $P:install_failed"; echo "phase $P install_failed" >> summary.txt; return 1; fi
  probe $P || { KSTATE="$KSTATE $P:install_failed"; echo "phase $P install_failed (probe)" >> summary.txt; return 1; }
  if ! premise $P; then KSTATE="$KSTATE $P:premise_failed"; echo "phase $P premise_failed" >> summary.txt; return 1; fi
  KSTATE="$KSTATE $P:ok"; echo "phase $P ok" >> summary.txt; return 0; }
if [ "$PROVE" = 1 ]; then
  kernel_phase f "$FLA_PIN"; kernel_phase fc "$CC_PIN"
  echo "PROVE -- the proving run: install tripwired; reducer self-test passed; premise t held; kernel phases:$KSTATE" | tee -a summary.txt
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
# ---- the checkpoint (a local P105_MODEL_DIR is a rehearsal)
if [ -z "$MODEL_DIR" ]; then
  can_run 2400 fetch || finish 40
  say "fetch $MODEL @ $REV"
  perl -e "alarm $(arm_alarm 2400); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json','*.jinja'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
  tail -1 logs/fetch.log | tee -a summary.txt
fi
# ---- the NF4 arena the engine serves from (the checkpoint pinned)
can_run 2400 bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
perl -e "alarm $(arm_alarm 2400); exec @ARGV" python $W/p98_bake.py --model "$SRC" --revision "$REV" --work $W/work > logs/bake.log 2>&1
rc=$?; { echo -n "bake rc=$rc "; python -c "import json; r=json.load(open('$W/work/bake.json')); print({k: r.get(k) for k in ('status','layers','experts','snapshot_gib','load_s','snapshot_s','bake_s','loaded_commit','err')})" 2>&1 | cut -c1-400; } | tee -a summary.txt
[ "$rc" = 0 ] || { tail -5 logs/bake.log; say "BAKE FAIL"; finish 12; }
# ---- P98's arms, each a fresh engine in its own process (transformers resolves the kernels at import)
arm(){ local A=$1 P=$2 AL; AL=$(arm_alarm 2700)
  can_run 900 "arm_${A}_$P" || return 40
  say "phase $P arm $A (alarm=$AL)"
  perl -e "alarm $AL; exec @ARGV" python $W/p98_box.py --arm $A --model "$SRC" --revision "$REV" --arena $W/work/nf4.arena \
    --calib $W/calib.json --out $W/arm_${A}_$P.json > logs/arm_${A}_$P.log 2>&1
  local rc=$?
  { echo -n "phase $P arm $A rc=$rc "; grep -aE "^P98_ARM" logs/arm_${A}_$P.log | tail -1 | cut -c1-600; echo; } >> summary.txt
  [ "$rc" = 0 ] || { tail -8 logs/arm_${A}_$P.log; rc=26; }
  return $rc; }
for P in $PHASES; do
  case "$P" in
    t) ;;
    f) kernel_phase f "$FLA_PIN" || continue;;
    fc) case "$KSTATE" in *f:ok*) ;; *) echo "phase fc skipped: phase f did not hold" | tee -a summary.txt; continue;; esac
        kernel_phase fc "$CC_PIN" || continue;;
    *) say "unknown phase $P"; finish 78;;
  esac
  for A in g e p; do arm $A $P; rec $?; done
done
echo "kernel phases:$KSTATE" | tee -a summary.txt
say "reduce"; python $W/p105_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish "$rc_any"
