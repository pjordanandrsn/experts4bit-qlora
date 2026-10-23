#!/bin/bash
# Lane P232, BOX side (bench/p232/PREREG-p232.md; lever K, K2b): the served read of the int8 lm_head at 2, 4 and 8 rows on
# Qwen3-30B-A3B's default server. E4B_INT8_HEAD_ROWS=8 routes the head's 2..8-row calls to grouped-nf4-gemm K37's
# dense_i8.linear_i8 over the head's own int8 codes and scales, in place of F.linear on their bf16 rounding (the head is
# on its int8 grid in both arms: E4B_INT8_HEAD_BW is on by its qwen3_moe default). On ONE RTX 5090 at a power limit of
# at least 575 W, grouped-nf4-gemm v0.50.0 (K36 and K37 as released):
#   speed    the shipped default server (graphs on, max_seqs 16 named, 4096 tokens per sequence, the split cap and the int8
#            head at their qwen3_moe defaults, the dense attention GEMV off): first a WARM-UP process Rw (P131 Amendment
#            2's process 0: an R build and speed pass, integrity-checked, reported only), then EIGHT arms of their own
#            processes, ABBA twice -- Ra Ka Kb Rb Rc Kc Kd Rd (R = E4B_INT8_HEAD_ROWS unset; K = 8) -- each through W2, W4
#            and W8 (2, 4 or 8 of P109's prompt rows at once) at FIVE timed reps, the card's telemetry read either side
#            of every timed block;
#   quality  three processes on the default server built eager, Phase B's instrument at group T = 2, 4 and 8 in turn
#            (24 wikitext windows x 128 positions, the pass KV at the server's 4096 tokens): R scores R (and its rep
#            floor) and saves its log-probs, K scores the knob against them, B16 (E4B_INT8_HEAD_BW=0, reported only)
#            scores the bf16 head against them too. Skipped when a speed arm failed;
#   rule     p232_reduce.py (VOID / NOISY / FUNCTION_FAIL / per-T PASS / FLIP_ROWS_8, _4, _2 / NO_FLIP; TRIPWIRE reported).
#   premise  on THIS card, before anything is fetched: P229's (the real K37 and K36 kernels compiled here, 11; the route's
#            CPU suite, 45), staged in the repo's layout under t/ (rc 25).
#   order    the refusals, the download probe and the guard fit; install + tripwire; reducer, box and quality self-tests;
#            premise; fetch; NF4 arena bake; prompts; the warm-up and eight arms; the three quality phases; reduce.
# P153's runner (bench/p153/p153_run.sh) by counted edits: this lane's run ids, box, reducer, knob, arms, workloads and
# pins; P229's premise in place of P153's; P231's download probe and pre-install guard fit; P229's shakedown ids.
# Refusals: CUDA unusable 18 (the host floor; a torch that will not import stays 10), torch image 28, card class 15, disk
# 13, host RAM 16, power limit below 575 W 17 (unreadable 15), download below 40 MB/s 19, guard fit 41. Only 13 and 18
# name the machine for the launcher's next draw. This script never creates compute.
set -uo pipefail
W=/root/p232; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p232: $*"; }
NONCE=${P232_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P232_RUN_NONCE.tmp && mv $W/P232_RUN_NONCE.tmp $W/P232_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P232_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P232_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P232_RUN_ID P232_DEADLINE_EPOCH P232_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949   # grouped-nf4-gemm v0.50.0, released: K36 and K37 as shipped (P229's)
TORCH_WANT=2.8.0+cu128
SHORT=32; LONG=160; REPS=5; CONT=128; WINDOWS=24
# the needs (PREREG "Budget and the fit"), every term conservative
INSTALL_TO_GATE_S=600   # the two installs, the self-tests and the two-part premise
NEED_FETCH=1500         # 57 GB at the probe's 40 MB/s floor
NEED_BAKE=400
NEED_ARM=240            # the warm-up and each of the eight speed arms: 9 x 240 = 2,160 s (Amendment 1)
NEED_Q=700              # each quality phase at its three group sizes: 3 x 700 = 2,100 s
RESERVE=600             # can_run's margin after a step: the reduce, the fetch-back and the teardown
GPU_CLASS=5090; MIN_RAM_GB=60; MIN_DISK_GB=140; MIN_DL_MB_S=40
MIN_POWER_W=575   # the 5090 served floor (P149's and P153's, the maintainer's ruling): a capped board refuses before the install, rc 17
SHAKE=0   # p232-shake-N: the shakedown, explore tier, on any card at sm_86 or later; never a reading
case "$P232_RUN_ID" in
  p232-5090-1|p232-5090-2|p232-5090-3) MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39;;   # SC1's pin
  p232-shake-1|p232-shake-2|p232-shake-3) SHAKE=1; REPS=3; MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39;;
  *) say "unregistered run ID: $P232_RUN_ID"; finish 78;;
esac
SHAKE_FLAG=""; [ "$SHAKE" = 1 ] && SHAKE_FLAG="--shakedown"
[ -z "${P232_GPU_CLASS:-}${P232_MIN_DISK_GB:-}${P232_MIN_RAM_GB:-}${P232_MIN_DL_MB_S:-}${P232_REHEARSAL:-}${P232_SHORT:-}${P232_LONG:-}${P232_REPS:-}${P232_CONT:-}${P232_WINDOWS:-}${P232_MIN_POWER_W:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever, engine knob and kernel knob starts unset: the default server is the subject (the arms set
# E4B_INT8_HEAD_ROWS, and B16 E4B_INT8_HEAD_BW, alone; the split cap and the int8 head stay at their qwen3_moe defaults,
# the dense attention GEMV off)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS E4B_INT4_WIDE_TILES E4B_INT4_TILE_PROGRAMS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_PAGED_DECODE_LOOKAHEAD E4B_PAGED_STEP_TRACE E4B_PAGED_TRACE \
      E4B_PAGED_LAST_LOGITS E4B_PAGED_KV_POOL E4B_PAGED_SPEC E4B_PAGED_SPEC_K E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL \
      GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE \
      TRITON_INTERPRET E4B_ATTN_SPLIT_CAP E4B_DENSE_GEMV_BW E4B_SERVE_ATTN_INT8 E4B_K25_LUT E4B_K25_PLAN E4B_NF4_SMALLM_BW \
      E4B_DENSE_LT_PIN E4B_INT8_HEAD_BW E4B_INT8_HEAD_ROWS
: > summary.txt; echo "$P232_INSTANCE_ID" > INSTANCE_ID; echo "$P232_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB min_dl_mb_s=$MIN_DL_MB_S min_power_w=$MIN_POWER_W short=$SHORT long=$LONG reps=$REPS windows=$WINDOWS cont=$CONT shake=$SHAKE" | tee -a summary.txt
# ---- staged pieces, byte-for-byte (P229's premise in the repo's layout under t/: the CPU suite reads bench/ beside tests/)
for f in p232_box.py p232_reduce.py p115_quality.py p109_box.py p110_box.py p108_box.py p97_box.py k8_bake.py calib.json \
         t/tests/test_int8_head_bw.py t/tests/test_int8_head_rows_interp.py t/tests/conftest.py t/bench/i8dense/i8_fakequant.py \
         t/bench/p148/p148_kernels.py staged-p232.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p232.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p232/staged-p232.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: CUDA, the torch pin, the card class, the disk, the host RAM
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
if not ok:
    print(f"no-cuda torch {torch.__version__}"); sys.exit(0)
cc, p = torch.cuda.get_device_capability(0), torch.cuda.get_device_properties(0)
print(f"ok torch={torch.__version__} cc={cc[0]}.{cc[1]} sms={p.multi_processor_count}")
PYC
)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in
  ok*) ;;
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE}) -- registered host floor"
            echo "refused: cuda unusable (${CUDA_PROBE})" > REFUSAL; echo "BOX_REFUSED cuda=unusable" >> summary.txt; finish 18;;
  *) say "DUD BOX (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
case "$CUDA_PROBE" in *" torch=$TORCH_WANT "*) ;; *) say "REFUSED: the image's torch is not the registered $TORCH_WANT ($CUDA_PROBE)"; echo "refused: torch ($CUDA_PROBE)" > REFUSAL; finish 28;; esac
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,enforced.power.limit,clocks.max.sm,temperature.gpu,clocks.sm --format=csv,noheader | sed "s/^/power.limit,enforced.power.limit,clocks.max.sm,temperature.gpu,clocks.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [ "$SHAKE" = 1 ]; then   # any NVIDIA card at sm_86 or later (P229's shakedown form): the structure, never the speed
  case "$CC" in 8.6|8.9|9.0|10.0|12.0) ;; *) say "REFUSED: a shakedown needs compute capability 8.6 or later, got $CC"; echo "refused: cc $CC" > REFUSAL; finish 15;; esac
else
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
[ "$CC" = 12.0 ] || { say "REFUSED: 5090 compute capability must be 12.0, got $CC"; echo "refused: cc $CC" > REFUSAL; finish 15; }
fi
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge 30000 ] || { say "REFUSED: VRAM ${VRAM_MIB:-?} MiB below the 30000 MiB floor (the default server at 16 x 4096 tokens)"; echo "refused: vram ${VRAM_MIB:-?}" > REFUSAL; finish 15; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 57 GB checkpoint, its NF4 snapshot and arena, the reference log-probs)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
# ---- the POWER floor (P149's), before anything is installed: a capped board measures its cap (a shakedown records it)
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -dc 0-9)
[ -n "$PL" ] || { say "REFUSED: power.limit unreadable"; echo "refused: power.limit unreadable" > REFUSAL; finish 15; }
echo "POWER_LIMIT_W $PL (floor $MIN_POWER_W)" | tee -a summary.txt
[ "$SHAKE" = 1 ] || [ "$PL" -ge "$MIN_POWER_W" ] || { say "REFUSED: power.limit ${PL} W < ${MIN_POWER_W} W (host-limited power cap)"; echo "refused: power.limit ${PL} W" > REFUSAL; finish 17; }
# the download probe (P231's form): one sustained GET of the checkpoint's first shard from the endpoint the fetch uses,
# until 256 MiB AND 20 s, or 60 s. A host below MIN_DL_MB_S, a non-2xx status or an error is refused (19, HOST_REFUSED,
# never a strike): 57 GB at 40 MB/s is 1,425 s, inside NEED_FETCH's 1,500 s, so the floor is what keeps the fetch
# inside its need. download_probe.json is written on EVERY box, refused or not: the bytes read (they count in the box's
# download allowance) and the measured MB/s travel with the receipt. The staged token goes to the resolve endpoint only
# (an unredirected header), never to the CDN and never on a command line.
rm -f $W/download_probe.json
DL=$(MIX=$MODEL MIX_REV=$REV FLOOR=$MIN_DL_MB_S OUT=$W/download_probe.json timeout 120 python - <<'PYD' 2>/dev/null | tail -1
import json, os, time, urllib.request
m, rev, tok = os.environ["MIX"], os.environ["MIX_REV"], ""
rec = {"model": m, "revision": rev, "floor_mb_s": int(os.environ["FLOOR"]), "status": None, "bytes": 0, "seconds": None,
       "mb_s": None, "shard": None, "error": None, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
try:
    tok = open("/root/.cache/huggingface/token").read().strip()
except OSError:
    pass
def get(url, t):
    req = urllib.request.Request(url)
    if tok:
        req.add_unredirected_header("Authorization", f"Bearer {tok}")
    return urllib.request.urlopen(req, timeout=t)
t0 = None
try:
    meta = json.load(get(f"https://huggingface.co/api/models/{m}/revision/{rev}", 30))
    rec["shard"] = sorted(s["rfilename"] for s in meta["siblings"] if s["rfilename"].endswith(".safetensors"))[0]
    t0 = time.time()
    with get(f"https://huggingface.co/{m}/resolve/{rev}/{rec['shard']}", 60) as r:
        rec["status"] = r.status
        while True:
            dt = time.time() - t0
            if (rec["bytes"] >= 256 << 20 and dt >= 20) or dt >= 60:
                break
            b = r.read(1 << 20)
            if not b:
                break
            rec["bytes"] += len(b)
except Exception as e:  # noqa: BLE001 -- a refusal reason, recorded by name
    rec["status"] = rec["status"] or getattr(e, "code", None)
    rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
finally:
    if t0 is not None:
        rec["seconds"] = round(time.time() - t0, 2)
        if rec["seconds"] and rec["error"] is None:
            rec["mb_s"] = round(rec["bytes"] / rec["seconds"] / 1e6, 1)
    json.dump(rec, open(os.environ["OUT"], "w"), indent=1)
print(f"status={rec['status']} bytes={rec['bytes']} seconds={rec['seconds']} mb_s={rec['mb_s'] if rec['mb_s'] is not None else 'none'}")
PYD
)
[ -s $W/download_probe.json ] || printf '{"status": null, "bytes": null, "mb_s": null, "error": "the probe wrote no record (killed at 120 s or did not run)"}\n' > $W/download_probe.json
echo "DOWNLOAD ${DL:-none}" | tee -a summary.txt
DL_MB_S=${DL##*mb_s=}; DL_MB_S=${DL_MB_S%%.*}
case "$DL" in status=2[0-9][0-9]\ *) ;; *) DL_MB_S=0;; esac
[ "${DL_MB_S:-0}" -ge "$MIN_DL_MB_S" ] 2>/dev/null || { say "REFUSED: download ${DL:-none} -- below ${MIN_DL_MB_S} MB/s"; echo "refused: download ${DL:-none}" > REFUSAL; finish 19; }
# ---- the pre-install fit check (bench/common/README.md, "Guard fit"): before any install and any staged script
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_FETCH + NEED_BAKE + 9 * NEED_ARM + 3 * NEED_Q + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P232_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
# ---- the start: every pre-install check has passed; from here the box counts toward the box-started cap
: > $W/STARTED; echo "STARTED $(date -u +%FT%TZ)" | tee -a summary.txt
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + RESERVE)); left=$((P232_DEADLINE_EPOCH - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s (${need}s + the ${RESERVE} s reserve), only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P232_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
echo "torch==${TORCH_WANT%%+*}" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0; torch held at $TORCH_WANT)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" "brotli>=1.2.0" pytest numpy || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA WANT_TORCH=$TORCH_WANT python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
assert md.version("grouped-nf4-gemm") == "0.50.0", md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
import torch, triton
assert torch.__version__ == os.environ["WANT_TORCH"], torch.__version__
assert triton.__version__ == "3.4.0", triton.__version__
from experts4bit_qlora import serve_paged as sp
assert sp._graphs_env("", "cuda", "all-vram") is True, "the subject is the graph default"
import nf4_grouped as ng
assert {"bw_prmt32", "dotpad", "scalar"} <= set(ng.dispatch_counts()), ng.dispatch_counts()
# the subject: Qwen3's expert projections on the bandwidth GEMV (P116, auto) and every fusion knob resolved per family
assert ng._bw() == "auto" and {(1536, 2048), (2048, 768)} <= set(ng._BW_SHAPES), (ng._bw(), ng._BW_SHAPES)
m, s = sp.resolve_fusion_modes({k: sp.FUSION_UNSET for k in sp.FUSION_KNOBS}, "qwen3_moe")
assert set(m.values()) == {"auto"} and set(s.values()) == {"default-allowlisted"}, (m, s)
# the split cap (e4b#1768 over grouped-nf4-gemm#558): qwen3_moe takes 16 by default (e4b#1792), in every arm
import inspect
import fp8_paged_attn
assert "split_cap" in inspect.signature(fp8_paged_attn.fp8_paged_decode_attention).parameters, "the kernel's split_cap"
kw = dict(block_tokens=16, ktile=64, compute="fp8", pack_heads=False, sms=170)
assert [fp8_paged_attn.decode_split_plan(b, 4, 256, **kw) for b in (1, 2, 4, 8, 16)] == [32, 32, 23, 11, 5]
assert [fp8_paged_attn.decode_split_plan(b, 4, 256, split_cap=16, **kw) for b in (1, 2, 4, 8, 16)] == [16, 16, 16, 11, 5]
assert sp.ATTN_SPLIT_CAP_DEFAULT_FAMILIES == {"qwen3_moe": 16}, sp.ATTN_SPLIT_CAP_DEFAULT_FAMILIES
assert sp.resolve_attn_split_cap(sp._attn_split_cap_env(""), "qwen3_moe") == (16, "default-allowlisted")
# the head (P153's default over K36): the grid's quantizer BYTES and K36's plan; on by default for qwen3_moe at the
# head's shape, in R and K alike; an explicit 0 is B16's opt-out
import ast, hashlib
import dense_gemv
assert dense_gemv.PLANS_I8 == {(151936, 2048): (4, 1024, 4)}, dense_gemv.PLANS_I8
_src = inspect.getsource(dense_gemv).replace("\r\n", "\n")
_fn = next(n for n in ast.parse(_src).body if isinstance(n, ast.FunctionDef) and n.name == "quantize_rows_int8")
assert hashlib.sha256(ast.get_source_segment(_src, _fn).encode()).hexdigest() == "8ea6dd9ee1b52594419f40339272059e413956aaed2a2f02d382450f854e8c6c", "K36's quantizer"
from experts4bit_qlora.engines import int8_head_bw as ih
assert ih.ENV == sp.INT8_HEAD_BW_ENV == "E4B_INT8_HEAD_BW"
assert sp.INT8_HEAD_BW_DEFAULT_FAMILIES == {"qwen3_moe": ((151936, 2048),)}, sp.INT8_HEAD_BW_DEFAULT_FAMILIES
assert sp.resolve_int8_head_bw(sp.FUSION_UNSET, "qwen3_moe", (151936, 2048)) == ("1", "default-allowlisted")
assert sp.resolve_int8_head_bw("0", "qwen3_moe", (151936, 2048)) == ("0", "explicit")
assert sp.PagedServeConfig.__dataclass_fields__["int8_head_bw"].default == sp.FUSION_UNSET
assert sp.PagedServeConfig.__dataclass_fields__["dense_gemv_bw"].default == "0", "the dense attention GEMV is off"
# the knob (K2b, e4b#2024 over K37): unset is 1 (today's route), 8 routes 2..8 rows to linear_i8 at its shipped plan
import dense_i8
assert ih.ROWS_ENV == sp.INT8_HEAD_ROWS_ENV == "E4B_INT8_HEAD_ROWS" and ih.MAX_ROWS == 8
assert sp.PagedServeConfig.__dataclass_fields__["int8_head_rows"].default == 1, "the knob is off by default"
assert ih.rows_value("") == 1 and ih.rows_value("8") == 8
assert all(tuple(dense_i8.plan(151936, 2048, m)) == (32, 256, 4) for m in range(2, 9)), "the head's shipped plan per M"
from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
assert {"append", "rewind", "kernel_args", "reference_kv", "attention", "_record_split"} <= set(dir(Fp8PagedKV))
import int4_b32 as ib
assert hasattr(ib, "swiglu_rows") and hasattr(ib, "rope_norm_qk"), "the expert path's swiglu and the q/k norm+rope kernel"
import fp8_kv, nvme_arena  # noqa: F401  (KV append, arena bake)
import experts4bit_qlora as e
cc, sm = torch.cuda.get_device_capability(), torch.cuda.get_device_properties(0).multi_processor_count
open("/root/p232/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {cc}\nsm_count {sm}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p232_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p232_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
python $W/p115_quality.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "QUALITY SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched (P229's): K2b's route through the REAL kernels compiled
# here (the interpreter file with TRITON_INTERPRET=0: 11, its four CUDA-graph replays included) and its CPU suite (45,
# whose wrapper runs the same file interpreted), staged copies in the repo's own layout under t/
(cd $W/t && PYTHONPATH='' TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest tests/test_int8_head_rows_interp.py -q -rs -p no:cacheprovider) > logs/premise_gpu.log 2>&1
rc1=$?; L1=$(tail -1 logs/premise_gpu.log)
(cd $W/t && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest tests/test_int8_head_bw.py -q -rs -p no:cacheprovider) > logs/premise_cpu.log 2>&1
rc2=$?; L2=$(tail -1 logs/premise_cpu.log)
{ echo "premise gpu rc=$rc1: $L1"; echo "premise cpu rc=$rc2: $L2"; } | tee -a summary.txt
if [ "$rc1" = 0 ] && echo "$L1" | grep -qE '^11 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$' \
   && [ "$rc2" = 0 ] && echo "$L2" | grep -qE '^45 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$'; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc1/$rc2" > REFUSAL; finish 25
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
perl -e "alarm 1200; exec @ARGV" python $W/p232_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts.json > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P109_PROMPTS" logs/prompts.log | tee -a summary.txt
# ---- the warm-up Rw, then the eight speed arms: Ra Ka Kb Rb Rc Kc Kd Rd (ABBA twice), each a fresh process on the
# default graph server (only E4B_INT8_HEAD_ROWS differs), five timed reps at W2, W4 and W8
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"
ARMS_OK=1
for TAG in Rw Ra Ka Kb Rb Rc Kc Kd Rd; do
  ARM=${TAG:0:1}; KN=""; [ "$ARM" = K ] && KN="E4B_INT8_HEAD_ROWS=8"
  can_run $NEED_ARM "arm $TAG" || finish 40
  AL=$(step_alarm 2400); say "arm $TAG (alarm=$AL)"
  # shellcheck disable=SC2086  # ENGINE_ENV and KN are assignment lists by design
  env PYTHONPATH= $ENGINE_ENV $KN P232_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p232_box.py --mode speed --prompts $W/prompts.json --out $W/arm_$TAG.json --tag $TAG \
      --short $SHORT --long $LONG --reps $REPS $SHAKE_FLAG > logs/arm_$TAG.log 2>&1
  rc=$?
  { echo -n "arm $TAG rc=$rc "; grep -a "^P232_ARM" logs/arm_$TAG.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { ARMS_OK=0; tail -6 logs/arm_$TAG.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG FAILED (rc=$rc) -- the reducer will VOID"; }
done
# ---- the quality phases (not when a speed arm failed): R, then K, then B16, each a fresh process on the default server
# built eager, at group T = 2, 4 and 8 in turn; R writes its log-probs per T to work/ref/T<T> (they stay on the box)
if [ "$ARMS_OK" = 1 ]; then
  mkdir -p $W/work/ref
  for PH in R K B16; do
    KN=""; [ "$PH" = K ] && KN="E4B_INT8_HEAD_ROWS=8"; [ "$PH" = B16 ] && KN="E4B_INT8_HEAD_BW=0"
    can_run $NEED_Q "quality $PH" || finish 40
    AL=$(step_alarm 3000); say "quality $PH (alarm=$AL)"
    # shellcheck disable=SC2086
    env PYTHONPATH= $ENGINE_ENV $KN E4B_PAGED_GRAPHS=0 P232_ARM=$PH E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
      perl -e "alarm $AL; exec @ARGV" python $W/p232_box.py --mode quality --out $W/quality_$PH.json --ref-dir $W/work/ref \
        --windows $WINDOWS --cont $CONT $SHAKE_FLAG > logs/quality_$PH.log 2>&1
    rc=$?
    { echo -n "quality $PH rc=$rc "; grep -a "^P232_QUALITY" logs/quality_$PH.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
    [ "$rc" = 0 ] || { tail -6 logs/quality_$PH.log | cut -c1-300 | tee -a summary.txt; say "QUALITY $PH FAILED (rc=$rc) -- the reducer will VOID"; break; }
  done
else
  echo "a speed arm failed -- the quality phases are not run (the reducer VOIDs)" | tee -a summary.txt
fi
say "reduce"; python $W/p232_reduce.py --dir $W --out $W/verdict_p232.json --e4b-sha "$E4B_SHA" 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p232.json" ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/verdict_p232.json'))['verdict'])")
# a reading finishes 0 only when it was READ: FLIP_ROWS_8/4/2, NO_FLIP or FUNCTION_FAIL. NOISY and VOID are not readings
# (rc 27). A shakedown finishes 0 only on SHAKEDOWN.
case "$SHAKE/$V" in
  1/SHAKEDOWN) finish 0;;
  1/*) say "P232 shakedown $V -- not a clean shakedown"; finish 27;;
  0/FLIP_ROWS_8|0/FLIP_ROWS_4|0/FLIP_ROWS_2|0/NO_FLIP|0/FUNCTION_FAIL) finish 0;;
  0/*) say "P232 $V -- no reading (NOISY and VOID are not readings)"; finish 27;;
esac
