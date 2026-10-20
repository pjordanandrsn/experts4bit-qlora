#!/bin/bash
# bench/p159/p159_run.sh -- P159, BOX side (bench/p159/PREREG-p159.md). Started detached by p159_drive.sh with the run's
# nonce; P159_RUN_NONCE first, then P159_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and P159_SUCCESS.<nonce> only
# when the probe wrote its result and the reducer its verdict. Derived from bench/smt-fuse/smt_fuse_run.sh (the nonce
# handshake, Q235's pinned install, the tripwire).
#   order   env and stage checks; the guard fit (before anything else runs); refusals (CUDA with 16 GB, the host floors
#           from p159_box.py --plan, disk); install + tripwire; the self-tests; the probe under its cap; the reducer
# Refusals: CUDA unusable or under 16 GB 18, host floors (MemTotal, MemAvailable, pool K) 16, disk 13, guard fit 41.
set -uo pipefail
W=${P159_HOME:-/root/p159}; export P159_HOME=$W; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p159: $*"; }
NONCE=${P159_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P159_RUN_NONCE.tmp && mv $W/P159_RUN_NONCE.tmp $W/P159_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P159_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P159_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P159_RUN_ID P159_DEADLINE_EPOCH P159_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
E4B_PIN=d65793e4a106c7db6ecfc219668a2d6881d207e5      # Q235's e4b, as the SMT fuse lane ran the registered block
GNF4_SHA=724ccc454f006c1a46836e434e997f31f293747f      # grouped-nf4-gemm 0.45.0, exact
TORCH_WANT=2.11.0+cu128                                 # Q235's wheel lock (wheels.json), not the image's torch
LOCK_SHA=84f361ed35fff9a857f340b65488ad582dc4270aef7e67cbeb6b46b8370021f3   # Q235's wheels.json at d65793e4
[ "$E4B_SHA" = "$E4B_PIN" ] || { say "refusing: E4B_SHA $E4B_SHA is not the registered e4b pin $E4B_PIN"; finish 78; }
DRY=${P159_DRY:-0}
if [ "$DRY" = 1 ]; then EXPERTS=8; DISK_DEF=5                 # the $0 wiring run: correctness only, never a reading
else EXPERTS=128; DISK_DEF=40; fi                               # the registered slab: one Qwen3-235B layer's experts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True P159_DRY=$DRY
unset OMP_NUM_THREADS MKL_NUM_THREADS TRITON_INTERPRET GLIBC_TUNABLES   # only P's processes get the tunables
: > summary.txt; echo "$P159_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS run=$P159_RUN_ID e4b=$E4B_SHA gnf4=$GNF4_SHA experts=$EXPERTS dry=$DRY" | tee -a summary.txt
[ "$DRY" = 1 ] && { echo "DRY -- the \$0 wiring run on a few experts: NEVER a reading" | tee -a summary.txt; : > DRY; }
for f in p159_box.py p159_reduce.py smt_fuse_box.py staged-p159.sha256 q235_bootstrap.py wheels.json dq11_require_git.sh; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
[ "$(sha256sum $W/wheels.json | cut -d' ' -f1)" = "$LOCK_SHA" ] || { say "wheels.json is not Q235's lock"; finish 9; }
(cd $W && sha256sum -c staged-p159.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p159/staged-p159.sha256"; finish 9; }
# ---- the guard fit, before anything else runs (bench/common/README.md, "Guard fit"); p159_box.time_fit has the terms
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
INSTALL_TO_GATE_S=345  # the install and setup to the probe (270 s) plus the slab write (75 s): max over hosts + 25 %
NEED_PROBE=11530       # the probe: 25 processes x 442 s + the balloon's 120 s in each of 4 B (p159_box.py time_fit)
RESERVE=600            # the reducer, the fetch-back and the teardown
FIT=$((INSTALL_TO_GATE_S + NEED_PROBE + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P159_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: guard fit" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + RESERVE)); left=$((P159_DEADLINE_EPOCH - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s (${need}s + the ${RESERVE} s reserve), only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
# ---- refusals before anything is installed
CUDA_PROBE=$(python - <<'PYC' 2>/dev/null | tail -1
import sys
try:
    import torch
except Exception as e:
    print(f"no-torch {type(e).__name__}"); sys.exit(0)
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if ok else 0
print(f"ok torch={torch.__version__} gpu={torch.cuda.get_device_name(0).replace(' ', '_')} vram_gb={gb:.0f}" if ok else f"no-cuda torch {torch.__version__}")
PYC
)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in
  ok*) VRAM=$(echo "$CUDA_PROBE" | sed -n 's/.*vram_gb=\([0-9]*\).*/\1/p')
       [ "$DRY" = 1 ] || [ "${VRAM:-0}" -ge 16 ] || { echo "refused: vram ${VRAM}" > REFUSAL; echo "BOX_REFUSED vram=${VRAM}GB" >> summary.txt; finish 18; };;
  no-cuda*) echo "refused: cuda unusable" > REFUSAL; echo "BOX_REFUSED cuda=unusable" >> summary.txt; finish 18;;
  *) say "DUD BOX (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)" | tee forensics.txt
free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
# Amendment 2 (4): the GPU facts a bytes question needs (driver, VBIOS, ECC and its error counters, retired/remapped
# memory, clock-event reasons), each bounded at 30 s; forensics only, never a refusal
{ perl -e 'alarm 30; exec @ARGV' nvidia-smi --query-gpu=driver_version,vbios_version --format=csv,noheader 2>&1
  perl -e 'alarm 30; exec @ARGV' nvidia-smi -q -d ECC,PAGE_RETIREMENT,ROW_REMAPPER,PERFORMANCE 2>&1 | head -150; } >> forensics.txt
# the host floors (MemTotal >= 64 GiB, MemAvailable >= 32 GiB, pool K >= 8), stdlib only, on the image's python
python3 $W/p159_box.py --plan | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] || { say "REFUSED: the host floors or THP (p159_box.py --plan)"; { echo -n "refused: host floors: "; grep -a "^P159_PLAN_CHECK" summary.txt | tail -1 | sed -n 's/.*"why": "\([^"]*\)".*/\1/p'; } > REFUSAL; finish 16; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$DISK_DEF" ] || { say "REFUSED: ${FREE_GB:-?} GB free < $DISK_DEF GB"; echo "refused: disk" > REFUSAL; finish 13; }
# ---- install (the paid path) or the dry box's own tree
if [ "$DRY" = 1 ]; then
  [ -d $W/e4bsrc/experts4bit_qlora ] || { say "DRY: no e4b tree at $W/e4bsrc"; finish 9; }
  export PYTHONPATH=$W/e4bsrc
else
  perl -e 'alarm 300; exec @ARGV' bash $W/dq11_require_git.sh > logs/require_git.log 2>&1 || { tail -2 logs/require_git.log; say "GIT FAIL"; finish 9; }
  command -v python3.11 >/dev/null || { say "REFUSED: no python3.11 for Q235's lock"; echo "refused: python3.11" > REFUSAL; finish 10; }
  python3.11 -m venv $W/venv-q235 || { say "VENV FAIL"; finish 9; }
  export PATH=$W/venv-q235/bin:$PATH
  say "install: Q235's wheel lock ($LOCK_SHA) + e4b @$E4B_SHA + gnf4 @$GNF4_SHA (q235_bootstrap.py)"
  env E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA perl -e 'alarm 2400; exec @ARGV' python $W/q235_bootstrap.py --directory $W \
    --cache $W/.cache/wheels > logs/bootstrap.log 2>&1 || { tail -4 logs/bootstrap.log; say "BOOTSTRAP FAIL"; finish 9; }
fi
WANT_E4B=$E4B_SHA WANT_TORCH=$TORCH_WANT python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import json, os, importlib.metadata as md
import torch
if os.environ.get("P159_DRY") != "1":
    d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
    assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the pin: {d}"
    assert torch.__version__ == os.environ["WANT_TORCH"], torch.__version__
    assert md.version("transformers") == "5.19.0" and md.version("bitsandbytes") == "0.50.2"
from experts4bit_qlora.arch.moe_conventions import fuse_experts  # noqa: F401
from experts4bit_qlora._vendor.experts import Experts4bit  # noqa: F401
open(os.path.join(os.environ["P159_HOME"], "versions.txt"), "w").write(
    f"torch {torch.__version__}\nbitsandbytes {md.version('bitsandbytes')}\nsafetensors {md.version('safetensors')}\n"
    f"e4b @{os.environ['WANT_E4B']}\n")
print("tripwire OK", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
for s in p159_box.py p159_reduce.py; do
  python $W/$s --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SELF-TEST FAILED: $s"; finish 21; }
done
# ---- the probe, under its on-box cap (p159_box.py: per process min(time left, max(600 s, 2 x the first process's wall)))
can_run $NEED_PROBE probe || finish 40
LEFT_P=$(( P159_DEADLINE_EPOCH - $(date +%s) - RESERVE ))
say "probe: $EXPERTS experts, 25 processes, ${LEFT_P}s left for it"
env E4B_SHA=$E4B_SHA P159_TIME_LEFT_S=$LEFT_P perl -e "alarm $LEFT_P; exec @ARGV" python $W/p159_box.py --out $W/result.json --experts $EXPERTS > logs/probe.log 2>&1
rc=$?
grep -a "^P159_PLAN\|^P159_BALLOON\|^P159_RESULT\|^P159_REFUSED\|^P159_CAP_HIT\|^P159_CHILD_FAILED\|^P159_DONE\|^P159_BYTES_GATE" logs/probe.log | cut -c1-1500 | tee -a summary.txt
case "$rc" in
  0|3) ;;
  5) say "REFUSED by the probe's host check"; echo "refused: host floors (probe)" > REFUSAL; finish 16;;
  *) tail -5 logs/probe.log | tee -a summary.txt; say "PROBE FAILED (rc=$rc)"; finish 22;;
esac
[ -s $W/result.json ] || { say "PROBE WROTE NO RESULT"; finish 22; }
python $W/p159_reduce.py --result $W/result.json --out $W/verdict_p159.json | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict_p159.json ] || { say "REDUCER FAILED"; finish 23; }
[ "$rc" = 3 ] && { say "BYTES MISMATCH: a finding, not a timing"; finish 3; }
finish 0
