#!/bin/bash
# Lane P158, BOX side (bench/p158/PREREG-p158.md; e4b#1313): the served read of K39, the K25 lut="prmt" decode of
# grouped-nf4-gemm v0.50.0, as the batched-decode expert GEMM of Qwen3-30B-A3B (E4B_K25_LUT=prmt, e4b#1862),
# bit-identical to the select tree by contract. On ONE RTX 5090 at a power limit of at least 575 W, grouped-nf4-gemm
# v0.50.0 (41ef086e):
#   speed    the shipped default server (graphs on, max_seqs 16 named, 4096 tokens per sequence, the split cap at its
#            qwen3_moe default, the int8 head on by its qwen3_moe default, the dense attention GEMV off, K25 at auto):
#            first a WARM-UP process S0w (process 0 of P131 Amendment 2: an S0 build and speed pass, integrity-checked,
#            reported only), then EIGHT arms of their own processes, ABBA twice -- S0a S1a S1b S0b S0c S1c S1d S0d
#            (S0 = E4B_K25_LUT=tree, named; S1 = prmt) -- each through the W16 and W1 of P109 at FIVE timed reps, the
#            card telemetry read either side of every timed block and a clock sampler (its own process) during it;
#   rule     p158_reduce.py (VOID, the token null of the tree first / NOISY / FUNCTION_FAIL / FLIP / NO_FLIP). No SANE
#            phase: the arms are bitwise-identical by contract, so the token-identity gate replaces a quality read.
#   premise  on THIS card, before anything is fetched: the 19 premise tests of P136, the compiled contract of K39
#            (kernel/test_nf4_smallm_prmt_gpu.py of grouped-nf4-gemm v0.50.0, 23) and the served-route contract of
#            e4b (tests/test_k25_lut_gpu.py, 7) = 49 (rc 25).
#   order    install + tripwire; reducer and box self-tests; premise; fetch; NF4 arena bake; prompts; nine processes;
#            reduce.
# The runner of P153 (bench/p153/p153_run.sh) by counted edits: this lane: run ids, box, reducer, knob and pins; the
# contracts of K39 and e4b in place of K36 in the premise; the tripwire pins the kernel bytes of K39; no SANE phase.
# Refusals: CUDA unusable 18 (the host floor; a torch that will not import stays 10), card class 15, disk 13, host RAM
# 16, power limit below 575 W 17 (unreadable 15), a guard that does not fit 41. Only 13 and 18 name the machine for the next draw of the launcher. This
# script never creates compute.
set -uo pipefail
W=/root/p158; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p158: $*"; }
NONCE=${P158_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P158_RUN_NONCE.tmp && mv $W/P158_RUN_NONCE.tmp $W/P158_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P158_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P158_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P158_RUN_ID P158_DEADLINE_EPOCH P158_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949 # grouped-nf4-gemm v0.50.0 (gnf4#572): K39, the K25 lut="prmt"
SHORT=32; LONG=160; REPS=5
NEED_FETCH=1200; NEED_BAKE=600; NEED_ARM=420
INSTALL_TO_GATE_S=600   # install, tripwire, self-tests and premise: p151-5090-4s 102 s and p153-5090-1s 59 s, P155s allowance kept
NEED_BENCH=2100         # fetch, bake, prompts, nine processes, reduce: max over p151-5090-4 (1,347 s at nine arms) and p153-5090-1 (1,039 s), x1.5
RESERVE=600             # can_runs margin after the work: the reduce, the fetch-back and the teardown
GPU_CLASS=5090; MIN_RAM_GB=60; MIN_DISK_GB=150
MIN_POWER_W=575   # the 5090 served floor (P149, P151, P153; ruled by the maintainer): a capped board refuses before the install, rc 17
case "$P158_RUN_ID" in
  p158-5090-1|p158-5090-2|p158-5090-3) MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39;;   # the SC1 pin
  *) say "unregistered run ID: $P158_RUN_ID"; finish 78;;
esac
[ -z "${P158_GPU_CLASS:-}${P158_MIN_DISK_GB:-}${P158_MIN_RAM_GB:-}${P158_REHEARSAL:-}${P158_SHORT:-}${P158_LONG:-}${P158_REPS:-}${P158_MIN_POWER_W:-}${P158_NEED_BENCH:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset: the default server is the subject (the arms set E4B_K25_LUT alone;
# the split cap and the int8 head stay at their qwen3_moe defaults (16; on), the dense attention GEMV off, K25 at auto)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_PAGED_DECODE_LOOKAHEAD E4B_PAGED_STEP_TRACE E4B_PAGED_TRACE \
      E4B_PAGED_LAST_LOGITS E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN \
      GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET E4B_ATTN_SPLIT_CAP E4B_DENSE_GEMV_BW E4B_INT8_HEAD_BW E4B_K25_LUT
: > summary.txt; echo "$P158_INSTANCE_ID" > INSTANCE_ID; echo "$P158_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB min_power_w=$MIN_POWER_W short=$SHORT long=$LONG reps=$REPS" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p158_box.py p158_reduce.py p109_box.py k8_bake.py calib.json test_decode_graph_buckets.py test_kv_step_select.py \
         test_fused_glue_decode_graphs_gpu.py test_gemv_bw_served_gpu.py test_nf4_smallm_prmt_gpu.py test_k25_lut_gpu.py \
         staged-p158.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p158.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p158/staged-p158.sha256"; finish 9; }
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
nvidia-smi --query-gpu=power.limit,enforced.power.limit,clocks.max.sm,temperature.gpu,clocks.sm --format=csv,noheader | sed "s/^/power.limit,enforced.power.limit,clocks.max.sm,temperature.gpu,clocks.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
[ "$CC" = 12.0 ] || { say "REFUSED: 5090 compute capability must be 12.0, got $CC"; finish 15; }
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge 30000 ] || { say "REFUSED: 5090 VRAM ${VRAM_MIB:-?} MiB below 30000 MiB floor"; finish 15; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 61 GB checkpoint, its NF4 snapshot and arena)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
# ---- the POWER floor (P149s), before anything is installed: a capped board measures its cap
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -dc 0-9)
[ -n "$PL" ] || { say "REFUSED: power.limit unreadable"; echo "refused: power.limit unreadable" > REFUSAL; finish 15; }
echo "POWER_LIMIT_W $PL (floor $MIN_POWER_W)" | tee -a summary.txt
[ "$PL" -ge "$MIN_POWER_W" ] || { say "REFUSED: power.limit ${PL} W < ${MIN_POWER_W} W (host-limited power cap)"; echo "refused: power.limit ${PL} W" > REFUSAL; finish 17; }
# ---- the guard fit (e4b#1845s cross-lane rule), before the install: the install still ahead, the work and the reserve
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P158_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + RESERVE)) -le "$P158_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P158_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P158_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_FULL" = 2.8.0+cu128 ] || { say "REFUSED: torch image $TORCH_FULL is not 2.8.0+cu128"; finish 28; }
TORCH_PIN=2.8.0; echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
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
assert md.version("grouped-nf4-gemm") == "0.50.0", md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora import serve_paged as sp
assert sp._graphs_env("", "cuda", "all-vram") is True, "the subject is the graph default"
import nf4_grouped as ng
assert {"bw_prmt32", "dotpad", "scalar"} <= set(ng.dispatch_counts()), ng.dispatch_counts()
# the subject: Qwen3's expert projections on the bandwidth GEMV (P116, auto) and every fusion knob resolved per family
assert ng._bw() == "auto" and {(1536, 2048), (2048, 768)} <= set(ng._BW_SHAPES), (ng._bw(), ng._BW_SHAPES)
m, s = sp.resolve_fusion_modes({k: sp.FUSION_UNSET for k in sp.FUSION_KNOBS}, "qwen3_moe")
assert set(m.values()) == {"auto"} and set(s.values()) == {"default-allowlisted"}, (m, s)
# the split cap (e4b#1768 over grouped-nf4-gemm#558): the kernel takes split_cap and plans min(cap, plan); qwen3_moe
# takes 16 by default (e4b#1792), in both arms
import inspect
import fp8_paged_attn
assert "split_cap" in inspect.signature(fp8_paged_attn.fp8_paged_decode_attention).parameters, "the kernel's split_cap"
kw = dict(block_tokens=16, ktile=64, compute="fp8", pack_heads=False, sms=170)
assert [fp8_paged_attn.decode_split_plan(b, 4, 256, **kw) for b in (1, 2, 4, 8, 16)] == [32, 32, 23, 11, 5]
assert [fp8_paged_attn.decode_split_plan(b, 4, 256, split_cap=16, **kw) for b in (1, 2, 4, 8, 16)] == [16, 16, 16, 11, 5]
assert sp.ATTN_SPLIT_CAP_DEFAULT_FAMILIES == {"qwen3_moe": 16}, sp.ATTN_SPLIT_CAP_DEFAULT_FAMILIES
assert sp.resolve_attn_split_cap(sp._attn_split_cap_env(""), "qwen3_moe") == (16, "default-allowlisted")
assert sp.resolve_attn_split_cap(sp._attn_split_cap_env("0"), "qwen3_moe") == (None, "explicit")
# the lever (e4b#1862 over grouped-nf4-gemm v0.50.0's K39): K25's kernel BYTES (P156's pin, byte-identical in the
# release), prmt among its decodes, the knob's resolution, the served plan at the tree by default; the int8 head ON by
# qwen3_moe's default at Qwen3's head (P153's FLIP) and the dense attention GEMV off (both arms keep the defaults)
import hashlib
import nf4_smallm
assert hashlib.sha256(open(nf4_smallm.__file__, "rb").read()).hexdigest() == "ee46969a4f786c36abe26bfaf0a52e426db4465406a5e31c6085fd3270b26568", "K25/K39's kernel bytes"
assert "prmt" in nf4_smallm._LUT_MODES and "tree" in nf4_smallm._LUT_MODES, nf4_smallm._LUT_MODES
from experts4bit_qlora.engines import hot_residency as hr
assert hr._K25_PLAN == {"block_n": 32, "kc": 64, "warps": 4, "stages": 3, "lut": "tree", "dot_bf16": False}, hr._K25_PLAN
assert isinstance(hr.K25_LUT_SEEN, dict) and callable(hr.resolve_k25_lut)
for _v, _want in (("tree", "tree"), ("prmt", "prmt")):
    os.environ["E4B_K25_LUT"] = _v
    _r = hr.resolve_k25_lut()
    assert (_r["env"], _r["lut"], _r["why"]) == (_v, _want, None), _r
    assert hr._k25_plan(nf4_smallm)["lut"] == _want
del os.environ["E4B_K25_LUT"]
assert sp.INT8_HEAD_BW_DEFAULT_FAMILIES == {"qwen3_moe": ((151936, 2048),)}, sp.INT8_HEAD_BW_DEFAULT_FAMILIES
assert sp.resolve_int8_head_bw(sp._int8_head_bw_env(""), "qwen3_moe", (151936, 2048)) == ("1", "default-allowlisted")
assert sp.PagedServeConfig.__dataclass_fields__["dense_gemv_bw"].default == "0", "the dense attention GEMV is off"
from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
assert {"append", "rewind", "kernel_args", "reference_kv", "attention", "_record_split"} <= set(dir(Fp8PagedKV))
import int4_b32 as ib
assert hasattr(ib, "swiglu_rows") and hasattr(ib, "rope_norm_qk"), "the expert path's swiglu and the q/k norm+rope kernel"
import fp8_kv, nvme_arena, int4_b32  # noqa: F401  (KV append, arena bake, the glue folds)
import torch, triton
import experts4bit_qlora as e
open("/root/p158/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p158_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p158_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched: P136's 19, K39's compiled contract (23) and e4b's
# served-route contract (7)
(cd $W && PYTHONPATH='' perl -e 'alarm 1200; exec @ARGV' python -m pytest test_decode_graph_buckets.py test_kv_step_select.py test_fused_glue_decode_graphs_gpu.py test_gemv_bw_served_gpu.py test_nf4_smallm_prmt_gpu.py test_k25_lut_gpu.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -qE '^49 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$'; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# ---- the checkpoint, its NF4 arena (P39's k8_bake.py, as SC1 bakes it) and the prompts
can_run $NEED_FETCH fetch || finish 40
say "fetch $MODEL @ $REV"
perl -e "alarm $(step_alarm 2700); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', ignore_patterns=['original/*','metal/*','consolidated*'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
can_run $NEED_BAKE bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt
perl -e "alarm 1200; exec @ARGV" python $W/p158_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts.json > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P109_PROMPTS" logs/prompts.log | tee -a summary.txt
# ---- the eight speed arms: S0a S1a S1b S0b S0c S1c S1d S0d (ABBA twice), each a fresh process on the default graph
# server (only E4B_K25_LUT differs), five timed reps
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"
for TAG in S0w S0a S1a S1b S0b S0c S1c S1d S0d; do
  ARM=${TAG:0:2}; DV=tree; [ "$ARM" = S1 ] && DV=prmt
  can_run $NEED_ARM "arm $TAG" || finish 40
  AL=$(step_alarm 2400); say "arm $TAG (alarm=$AL)"
  # shellcheck disable=SC2086  # ENGINE_ENV is an assignment list by design
  env PYTHONPATH= $ENGINE_ENV E4B_K25_LUT=$DV P158_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p158_box.py --mode speed --prompts $W/prompts.json --out $W/arm_$TAG.json --tag $TAG \
      --short $SHORT --long $LONG --reps $REPS > logs/arm_$TAG.log 2>&1
  rc=$?
  { echo -n "arm $TAG rc=$rc "; grep -a "^P158_ARM" logs/arm_$TAG.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -6 logs/arm_$TAG.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG FAILED (rc=$rc) -- the reducer will VOID"; }
done
say "reduce"; python $W/p158_reduce.py --dir $W --out $W/verdict_p158.json --e4b-sha "$E4B_SHA" 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p158.json" ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/verdict_p158.json'))['verdict'])")
case "$V" in FLIP|NO_FLIP|FUNCTION_FAIL) ;; *) say "P158 $V -- no reading (NOISY and VOID are not readings)"; finish 27;; esac
finish 0
