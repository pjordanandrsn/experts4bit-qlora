#!/bin/bash
# P231's box runner, lever B3 (bench/p231/PREREG-p231.md): P141's runner (bench/p141/p141_run.sh) on today's harness
# (P230's skeleton: the refusals, then the pre-install fit, then STARTED, before any install). Does E4B_NF4_SMALLM_BW=1
# (every (token, slot) row its own group, so grouped-nf4-gemm's K33 GEMV, one row per program) decode the default
# serve_paged server faster at 2 and 4 requests on Mixtral-8x7B on one RTX 5090, at no measurable quality cost?
#   server   P136's Mixtral size, E4B_PAGED_MAX_SEQS=16 E4B_PAGED_MAX_TOKENS_PER_SEQ=768 (P141's)
#   arms     B0a B1a B1b B0b (ABBA, fresh processes): B0 the switch unset (the device tile table and K25 at 2..8 tokens),
#            B1 E4B_NF4_SMALLM_BW=1; W2 and W4 ruled, W1 and W16 inert controls, W8 reported only; decode-only timing
#   quality  P115's teacher-forced instrument at group 4 (T == 4, the route B1 changes): OFF (B0) then ON (B1)
#   premise  on THIS card, before anything is fetched (rc 25): e4b's graph-bucket tests, 7 passed; and the ROWS premise
#            (p231_box.py --mode rows): B1's expert call returns each (token, slot) row bitwise identically at T = 1,
#            2 and 4 at Mixtral's shapes, which is what lets the group-4 quality read cover group 2
# No power floor (ratios inside one box; the power limit is recorded). Every reading knob is frozen. This script never
# creates compute.
set -uo pipefail
W=/root/p231; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p231: $*"; }
NONCE=${P231_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P231_RUN_NONCE.tmp && mv $W/P231_RUN_NONCE.tmp $W/P231_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P231_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P231_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P231_RUN_ID P231_DEADLINE_EPOCH P231_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949   # grouped-nf4-gemm v0.50.0: Mixtral's K33 shapes and plans shipped (P141)
TORCH_WANT=2.8.0+cu128
MIX=mistralai/Mixtral-8x7B-Instruct-v0.1; MIX_REV=eba92302a2861cdc0098cc54bc9f17cb2c47eb61        # FAM Amendment 5's pin
SIZE_ENV="E4B_PAGED_MAX_SEQS=16 E4B_PAGED_MAX_TOKENS_PER_SEQ=768"                                # P136's Mixtral server
SHORT=32; LONG=160; REPS=3; WINDOWS=24; CONT=128; GROUP=4
# the needs (PREREG "Budget"): P141's fetch, bake, fit and quality needs; an arm 900 s (five workloads, not two)
NEED_FETCH=3000; NEED_BAKE=1500; NEED_FIT=600; NEED_ARM=900; NEED_QOFF=2400; NEED_QON=900
INSTALL_TO_GATE_S=600   # the two installs, the self-tests and both premises
RESERVE=600             # can_run's margin after a step: the reduce, the fetch-back and the teardown
GPU_CLASS=5090; MIN_DISK_GB=220; MIN_RAM_GB=60; MIN_DL_MB_S=40   # 93 GB at 40 MB/s is 2,325 s, inside NEED_FETCH
case "$P231_RUN_ID" in
  p231-5090-1|p231-5090-2|p231-5090-r1|p231-5090-r2|p231-5090-r3|p231-5090-r4|p231-5090-r5|p231-5090-r6|p231-5090-r7|p231-5090-r8) :;;
  *) say "unregistered run ID: $P231_RUN_ID"; finish 78;;
esac
[ -z "${P231_GPU_CLASS:-}${P231_MIN_DISK_GB:-}${P231_MIN_RAM_GB:-}${P231_MIN_DL_MB_S:-}${P231_SHORT:-}${P231_LONG:-}${P231_REPS:-}${P231_WINDOWS:-}${P231_CONT:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever, engine knob and kernel switch starts unset: the default server is the subject, and
# E4B_NF4_SMALLM_BW is set only in B1's own processes
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS E4B_NF4_SMALLM_BW \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_PAGED_DECODE_LOOKAHEAD E4B_PAGED_STEP_TRACE E4B_PAGED_TRACE \
      E4B_PAGED_LAST_LOGITS E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN \
      GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET E4B_INT8_HEAD_BW E4B_K25_PLAN E4B_K25_LUT
: > summary.txt; echo "$P231_INSTANCE_ID" > INSTANCE_ID; echo "$P231_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MIX@$MIX_REV server=$SIZE_ENV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB min_dl_mb_s=$MIN_DL_MB_S short=$SHORT long=$LONG reps=$REPS windows=$WINDOWS cont=$CONT group=$GROUP" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p231_box.py p231_reduce.py p115_quality.py p115_reduce.py p110_box.py p108_box.py p97_box.py p109_box.py k8_bake.py \
         calib.json test_decode_graph_buckets.py staged-p231.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p231.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p231/staged-p231.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: CUDA, the torch pin, the card, the disk, the host RAM
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
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE})"; echo "refused: cuda unusable (${CUDA_PROBE})" > REFUSAL; finish 18;;
  *) say "IMAGE FAULT: torch does not import (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
case "$CUDA_PROBE" in *" torch=$TORCH_WANT "*) ;; *) say "REFUSED: the image's torch is not the registered $TORCH_WANT ($CUDA_PROBE)"; echo "refused: torch ($CUDA_PROBE)" > REFUSAL; finish 28;; esac
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
SMS=${CUDA_PROBE##*sms=}
case "$CUDA_PROBE" in *" cc=12.0 sms="*) ;; *) SMS=0;; esac
[ "${SMS:-0}" -ge 160 ] 2>/dev/null || { say "REFUSED: $CUDA_PROBE -- the lane registers sm_120 with >= 160 SMs"; echo "refused: config $CUDA_PROBE" > REFUSAL; finish 15; }
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge 30000 ] || { say "REFUSED: 5090 VRAM ${VRAM_MIB:-?} MiB below 30000 MiB floor"; echo "refused: vram ${VRAM_MIB:-?}" > REFUSAL; finish 15; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (Mixtral's 93 GB checkpoint, its NF4 snapshot and arena, the reference log-probs)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
# the download probe (P225's form): one sustained GET of the checkpoint's first shard from the endpoint the fetch uses,
# until 256 MiB AND 20 s, or 60 s; a host below MIN_DL_MB_S, a non-2xx status or an error is refused (19), never a
# strike: the 93 GB fetch after the start would not fit NEED_FETCH. The staged token goes to the resolve endpoint only
# (an unredirected header), never to the CDN and never on a command line.
DL=$(MIX=$MIX MIX_REV=$MIX_REV timeout 120 python - <<'PYD' 2>/dev/null | tail -1
import json, os, time, urllib.request
m, rev, tok = os.environ["MIX"], os.environ["MIX_REV"], ""
try:
    tok = open("/root/.cache/huggingface/token").read().strip()
except OSError:
    pass
def get(url, t):
    req = urllib.request.Request(url)
    if tok:
        req.add_unredirected_header("Authorization", f"Bearer {tok}")
    return urllib.request.urlopen(req, timeout=t)
try:
    meta = json.load(get(f"https://huggingface.co/api/models/{m}/revision/{rev}", 30))
    shard = sorted(s["rfilename"] for s in meta["siblings"] if s["rfilename"].endswith(".safetensors"))[0]
    t0, n = time.time(), 0
    with get(f"https://huggingface.co/{m}/resolve/{rev}/{shard}", 60) as r:
        status = r.status
        while True:
            dt = time.time() - t0
            if (n >= 256 << 20 and dt >= 20) or dt >= 60:
                break
            b = r.read(1 << 20)
            if not b:
                break
            n += len(b)
    dt = time.time() - t0
    print(f"status={status} bytes={n} seconds={dt:.1f} mb_s={n / dt / 1e6:.1f}" if dt else f"status={status} mb_s=none")
except Exception as e:  # noqa: BLE001 -- a refusal reason, recorded by name
    print(f"status={getattr(e, 'code', None)} error={type(e).__name__} mb_s=none")
PYD
)
echo "DOWNLOAD ${DL:-none}" | tee -a summary.txt
DL_MB_S=${DL##*mb_s=}; DL_MB_S=${DL_MB_S%%.*}
case "$DL" in status=2[0-9][0-9]\ *) ;; *) DL_MB_S=0;; esac
[ "${DL_MB_S:-0}" -ge "$MIN_DL_MB_S" ] 2>/dev/null || { say "REFUSED: download ${DL:-none} -- below ${MIN_DL_MB_S} MB/s"; echo "refused: download ${DL:-none}" > REFUSAL; finish 19; }
# ---- the pre-install fit check (bench/common/README.md, "Guard fit"): before any install and any staged script
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_FETCH + NEED_BAKE + NEED_FIT + 4 * NEED_ARM + NEED_QOFF + NEED_QON + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P231_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a start; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
# ---- the start: every pre-install check has passed; from here the box counts toward the box-started cap
: > $W/STARTED; echo "STARTED $(date -u +%FT%TZ)" | tee -a summary.txt
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + 600)); left=$((P231_DEADLINE_EPOCH - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s (${need}s + the 600 s reserve), only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P231_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
echo "torch==${TORCH_WANT%%+*}" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0; bitsandbytes 0.50.2; torch held at $TORCH_WANT)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" "brotli>=1.2.0" pytest numpy || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA WANT_TORCH=$TORCH_WANT python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
assert md.version("grouped-nf4-gemm") == "0.50.0", md.version("grouped-nf4-gemm")
assert md.version("bitsandbytes") == "0.50.2", md.version("bitsandbytes")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
import torch, triton
assert torch.__version__ == os.environ["WANT_TORCH"], torch.__version__
from experts4bit_qlora import serve_paged
from experts4bit_qlora.engines import hot_residency as hr
from experts4bit_qlora.engines.paged_runner import PagedModelRunner
assert serve_paged._graphs_env("", "cuda", "all-vram") is True, "the subject is the graph default"
assert hasattr(PagedModelRunner, "_capture_bucket") and "def _capture_bucket(self, b" in inspect.getsource(PagedModelRunner)
assert isinstance(hr.ROUTE_SEEN, dict) and callable(hr._fused_over_stack)
# the switch the lane reads, at the boundary it moves: unset, 2..8 tokens take device grouping; set, singleton groups
hr.DEVICE_GROUPING[0] = True
os.environ.pop("E4B_NF4_SMALLM_BW", None)
assert hr._collapsed_grouping(1, None) == (True, False) and hr._collapsed_grouping(4, None) == (False, True)
os.environ["E4B_NF4_SMALLM_BW"] = "1"
assert all(hr._collapsed_grouping(t, None) == (True, False) for t in (2, 4, 8)), "B1: 2..8 tokens take singleton groups"
assert hr._collapsed_grouping(16, None) == (False, True), "B1 is inert above 8 tokens"
del os.environ["E4B_NF4_SMALLM_BW"]
hr.DEVICE_GROUPING[0] = False
import nf4_grouped as ng
dev = torch.device("cuda")
assert all(ng._bw_engages(n, k, dev) for n, k in ((28672, 4096), (4096, 14336))), "K33 at Mixtral's shapes (P141's licence)"
assert ng._bw_decode(dev) == "prmt32"
assert not any(os.environ.get(k) for k in ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE", "GNF4_GEMV_DOTPAD"))
import fp8_paged_attn, fp8_kv, nvme_arena  # noqa: F401  (decode kernel, KV append, arena bake)
import experts4bit_qlora as e
cc, sm = torch.cuda.get_device_capability(), torch.cuda.get_device_properties(0).multi_processor_count
open("/root/p231/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {cc}\nsm_count {sm}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p231_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p231_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
python $W/p115_quality.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "QUALITY SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched: e4b's graph buckets (read from the junit report), then the
# ROWS premise (B1's expert call bitwise per row at T = 1, 2 and 4); a failure HALTS the lane until it is amended (25)
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_decode_graph_buckets.py -q -rs -p no:cacheprovider --junitxml=$W/logs/premise_served.xml) > logs/premise.log 2>&1
rc=$?; { echo -n "premise served route rc=$rc: "; tail -1 logs/premise.log; } | tee -a summary.txt
python $W/p231_box.py --premise $W/logs/premise_served.xml --expect served | tee -a summary.txt; ps=${PIPESTATUS[0]}
AL=$(step_alarm 900)
env PYTHONPATH= E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA perl -e "alarm $AL; exec @ARGV" python $W/p231_box.py --mode rows --out $W/rows.json > logs/rows.log 2>&1
pr=$?; grep -a "^P231_ROWS" logs/rows.log | tail -1 | tee -a summary.txt
if [ "$rc" = 0 ] && [ "$ps" = 0 ] && [ "$pr" = 0 ]; then
  echo "premise ok" | tee -a summary.txt
else
  tail -4 logs/rows.log | cut -c1-300 | tee -a summary.txt
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card -- the lane halts until amended"; echo "premise failed rc=$rc/$ps/$pr" > REFUSAL; finish 25
fi
# ---- the checkpoint, its NF4 arena (P39's k8_bake.py) and the prompts
FREE_GB=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED before fetch: disk ${FREE_GB:-?} GB"; finish 13; }
FW=$W/work/mixtral; mkdir -p $FW
can_run $NEED_FETCH fetch || finish 40
say "fetch $MIX @ $MIX_REV"
perl -e "alarm $(step_alarm 3600); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MIX', revision='$MIX_REV', ignore_patterns=['original/*', 'metal/*', 'consolidated*'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MIX@$MIX_REV $SNAP" | tee -a summary.txt
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
can_run $NEED_BAKE bake || finish 40
say "bake the NF4 arena"
K8_MODEL="$SNAP" K8_WORK="$FW" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $FW/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt
PR=$W/prompts.json
perl -e "alarm 1200; exec @ARGV" python $W/p231_box.py --prompts-only --model "$MIX" --revision "$MIX_REV" --out $PR > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P109_PROMPTS" logs/prompts.log | tee -a summary.txt
ENGINE_ENV="E4B_PAGED_MODEL=$MIX E4B_PAGED_REVISION=$MIX_REV E4B_PAGED_ARENA=$FW/nf4.arena E4B_PAGED_CALIB=$W/calib.json $SIZE_ENV"
# ---- the fit check: the shipped default (B0) builds and decodes W16, the widest workload; not fitting halts (31)
can_run $NEED_FIT fit || finish 40
AL=$(step_alarm 1800); say "fit (alarm=$AL)"
# shellcheck disable=SC2086  # ENGINE_ENV is an assignment list by design
env PYTHONPATH= $ENGINE_ENV P231_ARM=B0 E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
  perl -e "alarm $AL; exec @ARGV" python $W/p231_box.py --mode fit --prompts $PR --out $W/fit.json --short $SHORT > logs/fit.log 2>&1
rc=$?
{ echo -n "fit rc=$rc "; grep -a "^P231_FIT" logs/fit.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
if [ "$rc" != 0 ]; then
  tail -6 logs/fit.log | cut -c1-300 | tee -a summary.txt
  python $W/p231_reduce.py --dir $W --out $W/verdict.json --e4b-sha $E4B_SHA 2>&1 | tee -a summary.txt
  say "the fit check failed: the lane halts until amended (31)"; finish 31
fi
# ---- the four speed arms: B0a B1a B1b B0b (ABBA), each a fresh process on the default server (only the switch differs)
ARMS_OK=1
for TAG in B0a B1a B1b B0b; do
  ARM=${TAG:0:2}; KN=""; [ "$ARM" = B1 ] && KN="E4B_NF4_SMALLM_BW=1"
  can_run $NEED_ARM "arm $TAG" || finish 40
  AL=$(step_alarm 2400); say "arm $TAG (alarm=$AL)"
  # shellcheck disable=SC2086  # ENGINE_ENV and KN are assignment lists by design
  env PYTHONPATH= $ENGINE_ENV $KN P231_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p231_box.py --mode speed --prompts $PR --out $W/arm_$TAG.json --tag $TAG \
      --short $SHORT --long $LONG --reps $REPS > logs/arm_$TAG.log 2>&1
  rc=$?
  { echo -n "arm $TAG rc=$rc "; grep -a "^P231_ARM" logs/arm_$TAG.log | tail -1 | cut -c1-700; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { ARMS_OK=0; tail -6 logs/arm_$TAG.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG FAILED (rc=$rc) -- the reducer will VOID"; }
done
# ---- the quality phase (not after a failed speed arm): OFF (B0) then ON (B1), each a fresh process on the default server
# built eager, four windows per pass (T == 4); OFF writes the reference's log-probs to work/mixtral/ref
if [ "$ARMS_OK" = 1 ]; then
  mkdir -p $FW/ref
  for PH in off on; do
    NEED=$NEED_QOFF; ARM=B0; KN=""; [ "$PH" = on ] && { NEED=$NEED_QON; ARM=B1; KN="E4B_NF4_SMALLM_BW=1"; }
    can_run $NEED "quality $PH" || finish 40
    AL=$(step_alarm 4200); say "quality $PH (alarm=$AL)"
    # shellcheck disable=SC2086
    env PYTHONPATH= $ENGINE_ENV $KN E4B_PAGED_GRAPHS=0 P231_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
      perl -e "alarm $AL; exec @ARGV" python $W/p231_box.py --mode quality --out $W/quality_$PH.json --ref-dir $FW/ref \
        --windows $WINDOWS --cont $CONT --group $GROUP > logs/quality_$PH.log 2>&1
    rc=$?
    { echo -n "quality $PH rc=$rc "; grep -a "^P231_QUALITY" logs/quality_$PH.log | tail -1 | cut -c1-900; echo; } | tee -a summary.txt
    [ "$rc" = 0 ] || { tail -6 logs/quality_$PH.log | cut -c1-300 | tee -a summary.txt; say "QUALITY $PH FAILED (rc=$rc) -- the reducer will VOID"; break; }
  done
else
  echo "a speed arm failed -- the quality phase is not run (the reducer VOIDs)" | tee -a summary.txt
fi
say "reduce"; python $W/p231_reduce.py --dir $W --out $W/verdict.json --e4b-sha $E4B_SHA 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict.json ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/verdict.json'))['verdict'])")
case "$V" in LICENSED|SLOWER|QUALITY_FAIL|FUNCTION_FAIL) finish 0;; *) say "P231 $V -- no reading"; finish 27;; esac
