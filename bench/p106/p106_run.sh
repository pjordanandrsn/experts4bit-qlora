#!/bin/bash
# bench/p106/p106_run.sh -- lane P106, BOX side (bench/p106/PREREG-p106.md). Started detached by p106_drive.sh with the
# run's nonce; P106_RUN_NONCE first, then P106_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P106_SUCCESS.<nonce> only when the reducer ran (or, under P106_PROVE=1, when the proof passed).
#
# The Gated DeltaNet kernels (flash-linear-attention 0.5.2 + causal-conv1d 1.7.0) against transformers' torch path, in
# ONE process on ONE RTX 5090: their quality in nats and their prefill TTFT, on Qwen3.6-35B-A3B served by
# serve_paged.build_engine (P98's engine). p106_box.py switches transformers' four Gated DeltaNet functions in place
# (gdn_toggle.py); toggle_probe.py proves on THIS card that the switched-off path is a kernel-free process's arithmetic.
#   order    install (kernel-free) + tripwire; reducer self-test; toggle SAVE (tiny hybrids, kernel-free process);
#            install fla, then causal-conv1d; engagement probe (kernels_fc.json); PREMISE fc (P105's kernel-phase set:
#            the two graph files, the chunk-matched all-linear test, the dense parity file -- 9 passed, none skipped);
#            toggle COMPARE (toggle_check.json); fetch; bake; the box (box.json); reduce
# A failed kernel install (27), premise (25) or toggle check (28) ends the run before the fetch: no reading can follow.
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P106_GPU_CLASS P106_MIN_DISK_GB P106_REHEARSAL P106_MODEL_DIR P106_PREMISE_ALLOW_SKIP P106_BOX_EXTRA. P106_PROVE=1 is
# the PROVING RUN: the refusals, the install with its tripwire, the reducer self-test, the toggle save, both kernel
# installs, the engagement probe, the premise, the toggle compare and an HF CDN egress probe; no model.
set -uo pipefail
W=/root/p106; mkdir -p $W/logs $W/work; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p106: $*"; }
NONCE=${P106_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P106_RUN_NONCE.tmp && mv $W/P106_RUN_NONCE.tmp $W/P106_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P106_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P106_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P106_RUN_ID P106_DEADLINE_EPOCH P106_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=34da93d6fe8d2a401b7001705658ce00b2b18213   # grouped-nf4-gemm v0.34.1 (fp8_paged_attn, fused batch KV append, nvme_arena), e4b CI's pin
MODEL=Qwen/Qwen3.6-35B-A3B; REV=995ad96eacd98c81ed38be0c5b274b04031597b0
FLA_PIN="flash-linear-attention==0.5.2"; CC_PIN="causal-conv1d==1.7.0"
GPU_CLASS=${P106_GPU_CLASS:-5090}; MIN_DISK_GB=${P106_MIN_DISK_GB:-170}; REHEARSAL=${P106_REHEARSAL:-0}; PROVE=${P106_PROVE:-0}
MODEL_DIR=${P106_MODEL_DIR:-}; ALLOW_SKIP=${P106_PREMISE_ALLOW_SKIP:-0}; BOX_EXTRA=${P106_BOX_EXTRA:-}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever starts unset: the engine at e4b's defaults
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_PLACEMENT USE_HUB_KERNELS E4B_INT4_PREFILL
: > summary.txt; echo "$P106_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV kernels='$FLA_PIN $CC_PIN' gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB prove=$PROVE model_dir=${MODEL_DIR:-hub} allow_skip=$ALLOW_SKIP box_extra='$BOX_EXTRA'" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 170 ] || [ -n "$MODEL_DIR" ] \
   || [ "$ALLOW_SKIP" != 0 ] || [ -n "$BOX_EXTRA" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p106_box.py p106_reduce.py gdn_toggle.py toggle_probe.py p98_box.py p98_bake.py calib.json test_hybrid_decode_graphs_gpu.py test_linear_state_graph_gpu.py test_linear_state_chunk_matched_gpu.py test_linear_state_dense_parity_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p106/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 72 GB checkpoint, a 17 GB NF4 snapshot and its arena)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P106_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P106_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P106_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
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
assert hasattr(paged_runner.PagedModelRunner, "free_slot"), "the box frees its lockstep slots"
from experts4bit_qlora.serve_paged import build_engine  # noqa: F401
import fp8_paged_attn, fp8_kv, nvme_arena  # noqa: F401  (the kernel, the fused batch append, the arena)
mods = {m: importlib.util.find_spec(m) is not None for m in ("fla", "causal_conv1d", "kernels")}
assert not any(mods.values()), f"the toggle's reference must be a kernel-free process, but kernel modules are present: {mods}"
import experts4bit_qlora as e, torch, triton
open("/root/p106/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\ngated-deltanet kernel modules before the installs {mods}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p106_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the toggle's reference: tiny hybrids' logits in THIS kernel-free process
say "toggle SAVE (kernel-free process)"
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python $W/toggle_probe.py --save $W/work/toggle_ref.pt) > logs/toggle_save.log 2>&1
rc=$?; { echo -n "toggle save rc=$rc "; grep -aE "^SAVED" logs/toggle_save.log | tail -1; echo; } | tee -a summary.txt
[ "$rc" = 0 ] || { tail -5 logs/toggle_save.log; say "TOGGLE SAVE FAILED"; echo "toggle save failed" > REFUSAL; finish 28; }
# ---- the kernels, torch held; what transformers resolves; the premise on THIS card; the toggle against its reference
for pin in "$FLA_PIN" "$CC_PIN"; do
  say "install $pin (torch held at $TORCH_PIN)"
  pipx logs/pip_${pin%%=*}.log 1200 --prefer-binary --no-build-isolation "$pin"; rc=$?
  echo "install rc=$rc ($pin)" | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -3 logs/pip_${pin%%=*}.log; say "KERNEL INSTALL FAILED: $pin"; echo "install failed: $pin" > REFUSAL; finish 27; }
done
python - "$TORCH_PIN" <<'PYK' > logs/kernels_fc.log 2>&1
import importlib.metadata as md, inspect, json, sys
torch_pin = sys.argv[1]
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as m
rec = {"phase": "fc", "torch": torch.__version__, "torch_held": torch.__version__.split("+")[0] == torch_pin}
for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
    impl = inspect.getclosurevars(getattr(m, name)).nonlocals.get("implementation")
    rec[name] = getattr(impl, "__module__", None)
for dist in ("flash-linear-attention", "fla-core", "causal-conv1d", "kernels"):
    try:
        rec[dist] = md.version(dist)
    except md.PackageNotFoundError:
        rec[dist] = None
json.dump(rec, open("/root/p106/kernels_fc.json", "w"), indent=1)
print("KERNELS", json.dumps(rec))
PYK
rc=$?; { echo -n "kernels fc rc=$rc "; tail -1 logs/kernels_fc.log | cut -c1-500; } | tee -a summary.txt
[ "$rc" = 0 ] || { say "ENGAGEMENT PROBE FAILED"; echo "engagement probe failed" > REFUSAL; finish 27; }
LINEAR="test_linear_state_chunk_matched_gpu.py::test_an_all_linear_pool_equals_transformers_prefilled_in_the_same_chunks"
FILES="test_hybrid_decode_graphs_gpu.py test_linear_state_graph_gpu.py $LINEAR test_linear_state_dense_parity_gpu.py"; WANT="9 passed"
# shellcheck disable=SC2086  # FILES is a list by design
(cd $W && PYTHONPATH='' perl -e 'alarm 1500; exec @ARGV' python -m pytest $FILES -q -rs -s -p no:cacheprovider) > logs/premise_fc.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise_fc.log)
{ echo -n "premise run rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
grep -oaE "dense seed [0-9]+: .*" logs/premise_fc.log | sed "s/^/  fc /" | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "$WANT" && ! echo "$LASTL" | grep -q skipped; then
  echo "premise fc ok" | tee -a summary.txt
elif [ "$ALLOW_SKIP" = 1 ] && [ "$rc" = 0 ]; then
  echo "premise fc ok (not fully run on this card: REHEARSAL)" | tee -a summary.txt
else
  echo "premise fc failed" | tee -a summary.txt; say "PREMISE fc FAILED on this card"; echo "premise fc failed" > REFUSAL; finish 25
fi
say "toggle COMPARE (fla + causal-conv1d installed)"
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python $W/toggle_probe.py --compare $W/work/toggle_ref.pt --json-out $W/toggle_check.json) > logs/toggle_compare.log 2>&1
rc=$?; grep -aE "^(STEP|TOGGLE_PROBE)" logs/toggle_compare.log | cut -c1-300 | tee -a summary.txt
[ "$rc" = 0 ] || { tail -5 logs/toggle_compare.log; say "TOGGLE CHECK FAILED"; echo "toggle check failed" > REFUSAL; finish 28; }
if [ "$PROVE" = 1 ]; then
  echo "PROVE -- the proving run: install tripwired; reducer self-test passed; toggle saved kernel-free; both kernels installed and engaged; premise fc held; toggle check passed" | tee -a summary.txt
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
# ---- the checkpoint (a local P106_MODEL_DIR is a rehearsal)
if [ -z "$MODEL_DIR" ]; then
  can_run 2400 fetch || finish 40
  say "fetch $MODEL @ $REV"
  perl -e "alarm $(step_alarm 2400); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json','*.jinja'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
  tail -1 logs/fetch.log | tee -a summary.txt
fi
# ---- the NF4 arena the engine serves from (the checkpoint pinned)
can_run 2400 bake || finish 40
say "bake the NF4 arena"
perl -e "alarm $(step_alarm 2400); exec @ARGV" python $W/p98_bake.py --model "$SRC" --revision "$REV" --work $W/work > logs/bake.log 2>&1
rc=$?; { echo -n "bake rc=$rc "; python -c "import json; r=json.load(open('$W/work/bake.json')); print({k: r.get(k) for k in ('status','layers','experts','snapshot_gib','load_s','snapshot_s','bake_s','loaded_commit','err')})" 2>&1 | cut -c1-400; } | tee -a summary.txt
[ "$rc" = 0 ] || { tail -5 logs/bake.log; say "BAKE FAIL"; finish 12; }
# ---- the box: one engine, quality (Q), the mutant (M) and TTFT (T) with the kernels switched in place
can_run 1800 box || finish 40
AL=$(step_alarm 3000); say "the box (alarm=$AL)"
# shellcheck disable=SC2086  # BOX_EXTRA is a list of rehearsal flags by design
perl -e "alarm $AL; exec @ARGV" python $W/p106_box.py --model "$SRC" --revision "$REV" --arena $W/work/nf4.arena \
  --calib $W/calib.json --out $W/box.json $BOX_EXTRA > logs/box.log 2>&1
rc=$?
{ echo -n "box rc=$rc "; grep -aE "^P106_BOX" logs/box.log | tail -1 | cut -c1-900; echo; } | tee -a summary.txt
grep -aE "^P106_WINDOW" logs/box.log | cut -c1-400 >> summary.txt
[ "$rc" = 0 ] || { tail -8 logs/box.log; say "BOX FAILED"; finish 26; }
say "reduce"; python $W/p106_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish 0
