#!/bin/bash
# P236's box runner (bench/p236/PREREG-p236.md): grouped-nf4-gemm 0.51.0's decode GEMV vs bitsandbytes 0.50.2's
# matmul_4bit on the same NF4 bytes, op level, one RTX 5090. P227's runner (bench/p227/p227_run.sh) by named edits:
# this lane's run ids, bench and reducer; no e4b install, no premise suite, no power floor (a ratio inside one box; the
# limit is recorded); the installs are grouped-nf4-gemm at its 0.51.0 commit and bitsandbytes 0.50.2 on the image's
# torch. No instance id and no host name are written here. This script never creates compute.
set -uo pipefail
W=/root/p236; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p236: $*"; }
NONCE=${P236_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P236_RUN_NONCE.tmp && mv $W/P236_RUN_NONCE.tmp $W/P236_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P236_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P236_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P236_RUN_ID P236_DEADLINE_EPOCH; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
GNF4_SHA=30adb1bfa76f95f9ce4a4c29734b8f36de907eca   # grouped-nf4-gemm v0.51.0: `git rev-list -n1 v0.51.0` on the canonical repo
GNF4_VERSION=0.51.0; BNB_VERSION=0.50.2; TORCH_WANT=2.8.0+cu128
INSTALL_TO_GATE_S=600   # the two installs, the tripwire and the self-tests
NEED_BENCH=600          # the bench: about 2 min by the PREREG arithmetic, x5
RESERVE=600             # can_run's margin after the bench: the reduce, the fetch-back and the teardown
GPU_CLASS=5090; MIN_RAM_GB=16; MIN_DISK_GB=20
case "$P236_RUN_ID" in
  p236-5090-1|p236-5090-2|p236-5090-r1|p236-5090-r2|p236-5090-r3) :;;
  *) say "unregistered run ID: $P236_RUN_ID"; finish 78;;
esac
[ -z "${P236_GPU_CLASS:-}${P236_MIN_DISK_GB:-}${P236_MIN_RAM_GB:-}${P236_NEED_BENCH:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# every python this box runs reads and writes bytecode only under a fresh, empty prefix, never an in-tree __pycache__, so
# no stale .pyc can stand in for a proven source (the release-proof rule's R2); recorded in summary.txt and the record
export PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/p236-pycache.XXXXXX)
export RELEASE_PROOF_FRESH_PYCACHE=$PYTHONPYCACHEPREFIX   # release_proof.py records pycache_fresh only under this marker
# every grouped-nf4-gemm and bitsandbytes knob starts unset: both libraries as they ship
unset GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN GNF4_GEMV_SPLITK GNF4_PDL \
      GNF4_PDL_MAX_ROWS TRITON_INTERPRET BNB_CUDA_VERSION
: > summary.txt; echo "$P236_RUN_ID" > RUN_ID
echo "PYTHONPYCACHEPREFIX $PYTHONPYCACHEPREFIX ($(find "$PYTHONPYCACHEPREFIX" -mindepth 1 | wc -l | tr -d ' ') entries at start)" | tee -a summary.txt
echo "KNOBS gnf4=$GNF4_SHA ($GNF4_VERSION) bnb=$BNB_VERSION torch=$TORCH_WANT gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB need_bench=$NEED_BENCH" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p236_bench.py p236_reduce.py p236_run.sh release_proof.py staged-p236.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p236.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p236/staged-p236.sha256"; finish 9; }
# ---- refusals before anything is installed: CUDA, the torch image, the card, the disk, the host RAM, the guard fit
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
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE})"; echo "refused: cuda unusable" > REFUSAL; finish 18;;
  *) say "IMAGE FAULT: torch does not import (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
case "$CUDA_PROBE" in *" torch=$TORCH_WANT "*) ;; *) say "REFUSED: the image's torch is not $TORCH_WANT ($CUDA_PROBE)"; finish 28;; esac
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
SMS=${CUDA_PROBE##*sms=}
case "$CUDA_PROBE" in *" cc=12.0 sms="*) ;; *) SMS=0;; esac
[ "${SMS:-0}" -ge 160 ] 2>/dev/null || { say "REFUSED: $CUDA_PROBE -- the lane registers sm_120 with >= 160 SMs"; echo "refused: config $CUDA_PROBE" > REFUSAL; finish 15; }
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge 30000 ] || { say "REFUSED: VRAM ${VRAM_MIB:-?} MiB below the 30000 MiB floor"; finish 15; }
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -dc 0-9)
echo "POWER_LIMIT_W ${PL:-unread} (recorded only)" | tee -a summary.txt
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P236_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
# ---- the start: every pre-install check has passed; from here the box counts toward the box-started cap
: > $W/STARTED; echo "STARTED $(date -u +%FT%TZ)" | tee -a summary.txt
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + RESERVE)); left=$((P236_DEADLINE_EPOCH - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s, only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P236_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 300 ] && left=300; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
echo "torch==${TORCH_WANT%%+*}" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install bitsandbytes $BNB_VERSION + gnf4 @$GNF4_SHA (image python; torch held at $TORCH_WANT)"
pipx logs/pip_bnb.log 900 --prefer-binary "bitsandbytes==$BNB_VERSION" numpy || { tail -4 logs/pip_bnb.log; say "PIP FAIL (bitsandbytes)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_GNF4=$GNF4_SHA WANT_GNF4_V=$GNF4_VERSION WANT_BNB=$BNB_VERSION WANT_TORCH=$TORCH_WANT python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, sys, importlib.metadata as md
# the release proof (the maintainer's rule, 2026-10-24): the reducer's own check, unconditional, before anything is timed
sys.path.insert(0, "/root/p236"); import p236_reduce as R
assert (R.GNF4_SHA, R.GNF4_VERSION, R.BNB_VERSION) == (os.environ["WANT_GNF4"], os.environ["WANT_GNF4_V"], os.environ["WANT_BNB"])
dists = {name: R.dist_record(name) for name in R.RELEASES}
faults = R.provenance_faults(dists)
assert not faults, f"not the registered releases: {faults} ({dists})"
import torch, triton
assert torch.__version__ == os.environ["WANT_TORCH"], torch.__version__
import nf4_grouped as ng
import bitsandbytes as bnb
import bitsandbytes.backends.cuda.ops as BO
dev = torch.device("cuda")
# the gnf4 arm: K33 engages at every registered shape on this card, with every knob at its default
shapes = [(28672, 4096), (4096, 14336), (1536, 2048), (2048, 768), (2048, 2048), (2048, 1024)]
assert all(ng._bw_engages(n, k, dev) for n, k in shapes), "K33 auto does not engage at a registered shape"
assert ng._bw_decode(dev) == "prmt32", ng._bw_decode(dev)
assert set(ng.dispatch_counts()) >= {"bw_prmt32", "bw_tree", "bw_splitk", "dotpad", "dotpad_splitk", "scalar", "scalar_splitk"}
# the bnb arm: matmul_4bit dispatches bitsandbytes::gemm_4bit, whose CUDA kernel picks one of these two routes
assert callable(BO._gemm_4bit_kernel_impl) and callable(BO._dequant_linear_fallback), "bitsandbytes' gemm_4bit routes"
assert hasattr(torch.ops.bitsandbytes, "gemm_4bit"), "bitsandbytes::gemm_4bit is not registered"
cc, sm = torch.cuda.get_device_capability(), torch.cuda.get_device_properties(0).multi_processor_count
open("/root/p236/versions.txt", "a").write(f"gnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\nbitsandbytes {md.version('bitsandbytes')}\ntorch {torch.__version__}\ntriton {triton.__version__}\ncc {cc}\nsm_count {sm}\n")
print("tripwire OK: gnf4", md.version("grouped-nf4-gemm"), "bnb", md.version("bitsandbytes"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p236_bench.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BENCH SELF-TEST FAILED"; finish 21; }
python $W/p236_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the bench: per projection and M, the premise, correctness, engagement, then the timed palindromes
can_run $NEED_BENCH bench || finish 40
# ---- the release proof (the rule for every head-to-head row, 2026-10-24): bitsandbytes' installed BYTES are the wheel
# PyPI publishes for 0.50.2 (release_proof.py, staged byte-identical; its sha256 is named in the PREREG). It is the LAST
# step before the bench, in the bench's exact import environment (this interpreter, cwd $W, sys.path[0] $W, PYTHONPATH
# empty, the same PYTHONPYCACHEPREFIX), with nothing installed in between. grouped-nf4-gemm is git at the v0.51.0 commit, which the tripwire's provenance check already pins.
RELEASE_PROOF_NETWORK=1 PYTHONPATH= perl -e 'alarm 600; exec @ARGV' python $W/release_proof.py "bitsandbytes==$BNB_VERSION" > $W/release_proof.json 2> logs/release_proof.err
rp=$?; { echo -n "release proof rc=$rp: "; head -c 300 $W/release_proof.json; head -c 300 logs/release_proof.err; echo; } | tee -a summary.txt
[ "$rp" = 0 ] || { say "RELEASE PROOF FAILED: bitsandbytes is not the published $BNB_VERSION"; finish 9; }
AL=$(step_alarm 1200); say "bench (alarm=$AL)"
PYTHONPATH= perl -e "alarm $AL; exec @ARGV" python $W/p236_bench.py --out $W/bench_p236.json --release-proof $W/release_proof.json > logs/bench.log 2>&1
rc=$?
{ echo -n "bench rc=$rc "; grep -a "^P236_BENCH" logs/bench.log | tail -1 | cut -c1-300; echo; } | tee -a summary.txt
[ "$rc" = 0 ] || { tail -4 logs/bench.log | cut -c1-300 | tee -a summary.txt; say "BENCH FAILED (rc=$rc) -- the reducer will VOID"; }
say "reduce"; python $W/p236_reduce.py --bench "$W/bench_p236.json" --out "$W/verdict_p236.json" 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p236.json" ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/verdict_p236.json'))['verdict'])")
case "$V" in READ) finish 0;; *) say "P236 $V -- no reading"; finish 27;; esac
