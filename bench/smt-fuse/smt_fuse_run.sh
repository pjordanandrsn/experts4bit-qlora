#!/bin/bash
# bench/smt-fuse/smt_fuse_run.sh -- the SMT fuse measurement, BOX side (bench/smt-fuse/PREREG-smt-fuse.md). Started
# detached by smt_fuse_drive.sh with the run's nonce; SMT_FUSE_RUN_NONCE first, then SMT_FUSE_EXIT_CODE.<nonce> +
# TP_DONE.<nonce> on every exit and SMT_FUSE_SUCCESS.<nonce> only when the probe wrote its result.
#
# Does torch's intra-op pool at 32 threads (logical CPUs) beat 16 (physical cores) for the streaming loader's host-side
# expert fuse, on a server-class SMT host, at one Qwen3-235B layer? A timing premise for Q235's simplification amendment,
# never a ruling. Pieces from bench/p146/p146_run.sh: the nonce handshake, the refusals, the pinned install.
#   order   refusals (CUDA, the torch pin, an SMT host with >= 32 logical CPUs in the affinity and a quota >= 16, RAM,
#           disk); install + tripwire (or the dry box's own stack); the probe's self-test; the probe under its on-box cap
# Refusals: CUDA unusable 18, torch not the pin 28, not an SMT host or too few CPUs 16, RAM 16, disk 13.
# Amendment 1 (ids smt-sib-*): the same install, then smt_sib_box.py's pinned PHYS / SIB / WIDE arms over the same block;
# its host refusal (rc 16) is K = min(16, floor(quota / 2.5)) >= 8 with 2K two-thread cores in the affinity.
set -uo pipefail
W=${SMT_FUSE_HOME:-/root/smt_fuse}; export SMT_FUSE_HOME=$W; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] smt_fuse: $*"; }
NONCE=${SMT_FUSE_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/SMT_FUSE_RUN_NONCE.tmp && mv $W/SMT_FUSE_RUN_NONCE.tmp $W/SMT_FUSE_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > SMT_FUSE_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > SMT_FUSE_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in SMT_FUSE_RUN_ID SMT_FUSE_DEADLINE_EPOCH SMT_FUSE_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
E4B_PIN=d65793e4a106c7db6ecfc219668a2d6881d207e5      # Q235's e4b (A15, the source q235-pro6000-7 ran); a registered constant
GNF4_SHA=724ccc454f006c1a46836e434e997f31f293747f      # grouped-nf4-gemm 0.45.0, exact: Q235's (bench/q235/q235_drive.sh)
TORCH_WANT=2.11.0+cu128                                 # Q235's wheel lock (wheels.json), not the image's torch
LOCK_SHA=84f361ed35fff9a857f340b65488ad582dc4270aef7e67cbeb6b46b8370021f3   # Q235's wheels.json at d65793e4
[ "$E4B_SHA" = "$E4B_PIN" ] || { say "refusing: E4B_SHA $E4B_SHA is not the registered e4b pin $E4B_PIN"; finish 78; }
DRY=${SMT_FUSE_DRY:-0}
# Amendment 1: the smt-sib-* ids run the pinned sibling probe (smt_sib_box.py) over the same block; smt-fuse-* the original
case "$SMT_FUSE_RUN_ID" in smt-sib-*) MODE=sib;; *) MODE=fuse;; esac
if [ "$DRY" = 1 ]; then EXPERTS=8; MIN_LOGICAL=2; MIN_P=3; DISK_DEF=5; RAM_DEF=8               # the dry box: correctness only
else EXPERTS=128; MIN_LOGICAL=32; MIN_P=24; DISK_DEF=40; RAM_DEF=32; fi                      # one full Qwen3-235B layer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS= SMT_FUSE_DRY=$DRY
unset OMP_NUM_THREADS MKL_NUM_THREADS TRITON_INTERPRET
: > summary.txt; echo "$SMT_FUSE_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS run=$SMT_FUSE_RUN_ID e4b=$E4B_SHA gnf4=$GNF4_SHA experts=$EXPERTS dry=$DRY mode=$MODE" | tee -a summary.txt
[ "$DRY" = 1 ] && { echo "DRY -- the \$0 dry run on a few experts: NEVER a reading" | tee -a summary.txt; : > DRY; }
for f in smt_fuse_box.py staged.sha256 smt_sib_box.py thp_window.py q235_bootstrap.py wheels.json dq11_require_git.sh; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
[ "$(sha256sum $W/wheels.json | cut -d' ' -f1)" = "$LOCK_SHA" ] || { say "wheels.json is not Q235's lock"; finish 9; }
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/smt-fuse/staged.sha256"; finish 9; }
# ---- refusals before anything is installed
CUDA_PROBE=$(python - <<'PYC' 2>/dev/null | tail -1
import sys
try:
    import torch
except Exception as e:
    print(f"no-torch {type(e).__name__}"); sys.exit(0)
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(f"ok torch={torch.__version__} gpu={torch.cuda.get_device_name(0).replace(' ', '_')}" if ok else f"no-cuda torch {torch.__version__}")
PYC
)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in
  ok*) ;;
  no-cuda*) echo "refused: cuda unusable" > REFUSAL; echo "BOX_REFUSED cuda=unusable" >> summary.txt; finish 18;;
  *) say "DUD BOX (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)" | tee forensics.txt
free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
TPC=$(lscpu | awk -F: '/Thread\(s\) per core/{gsub(/ /,"",$2); print $2}')
AFF=$(python -c "import os; print(len(os.sched_getaffinity(0)))")
QUOTA=$(python - <<'PYQ'
import os
try:
    q, p = open("/sys/fs/cgroup/cpu.max").read().split()
    print("none" if q == "max" else f"{int(q) / int(p):.2f}")
except Exception:
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read()); p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        print("none" if q <= 0 else f"{q / p:.2f}")
    except Exception:
        print("none")
PYQ
)
echo "HOST threads_per_core=$TPC affinity=$AFF quota=$QUOTA" | tee -a summary.txt
if [ "$DRY" != 1 ]; then
  [ "${TPC:-1}" -ge 2 ] || { say "REFUSED: not an SMT host (threads per core $TPC)"; echo "refused: not SMT" > REFUSAL; finish 16; }
fi
[ "${AFF:-0}" -ge "$MIN_LOGICAL" ] || { say "REFUSED: $AFF CPUs in the affinity < $MIN_LOGICAL"; echo "refused: affinity $AFF" > REFUSAL; finish 16; }
# the high arm the loader would run: P = min(32, ceil(quota), affinity); refused (NOT_RUN, $0) under MIN_P, since a
# 16-vs-under-24 ratio is too weak to read (the maintainer's ruling)
if [ "$MODE" = fuse ]; then
  P=$(python -c "import math; q='$QUOTA'; print(min(32, $AFF, math.ceil(float(q)) if q != 'none' else 32))")
  echo "HIGH_ARM P=$P (min 32, ceil quota $QUOTA, affinity $AFF)" | tee -a summary.txt
  [ "${P:-0}" -ge "$MIN_P" ] || { say "REFUSED: quota $QUOTA gives a high arm of $P < $MIN_P"; echo "refused: high arm $P" > REFUSAL; finish 16; }
elif [ "$DRY" != 1 ]; then
  # Amendment 1: K = min(16, floor(quota / 2.5)) >= 8 and 2K two-thread cores in the affinity (stdlib only, before the install)
  python $W/smt_sib_box.py --plan --quota "$QUOTA" | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "REFUSED: the host cannot carry the sibling arms"; echo "refused: sibling arms" > REFUSAL; finish 16; }
fi
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$RAM_DEF" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < $RAM_DEF GiB"; echo "refused: ram" > REFUSAL; finish 16; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$DISK_DEF" ] || { say "REFUSED: ${FREE_GB:-?} GB free < $DISK_DEF GB"; echo "refused: disk" > REFUSAL; finish 13; }
# ---- install (the paid path) or the dry box's own tree
if [ "$DRY" = 1 ]; then
  [ -d $W/e4bsrc/experts4bit_qlora ] || { say "DRY: no e4b tree at $W/e4bsrc"; finish 9; }
  export PYTHONPATH=$W/e4bsrc
else
  # Q235's own install, unchanged: git first (dq11_require_git.sh), a python3.11 venv, then q235_bootstrap.py, which fetches
  # every wheel in the hash-locked wheels.json and installs e4b and gnf4 at their pins, checking pip's recorded commits
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
if os.environ.get("SMT_FUSE_DRY") != "1":
    d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
    assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the pin: {d}"
    assert torch.__version__ == os.environ["WANT_TORCH"], torch.__version__          # Q235's lock, not the image's
    assert md.version("transformers") == "5.19.0" and md.version("bitsandbytes") == "0.50.2"
from experts4bit_qlora.arch.moe_conventions import fuse_experts  # noqa: F401
from experts4bit_qlora._vendor.experts import Experts4bit  # noqa: F401
open(os.path.join(os.environ["SMT_FUSE_HOME"], "versions.txt"), "w").write(
    f"torch {torch.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ne4b @{os.environ['WANT_E4B']}\n")
print("tripwire OK", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/smt_fuse_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SELF-TEST FAILED"; finish 21; }
if [ "$MODE" = sib ]; then
  python $W/smt_sib_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SIB SELF-TEST FAILED"; finish 21; }
fi
# ---- the probe, under its cap sized on the box: min(time left - 600 s, 1.5 x the reps x the warm rep's own wall),
# ---- enforced inside smt_fuse_box.py between reps (lane_record.phase_caps.probe = {"on_box": ...})
LEFT=$(( SMT_FUSE_DEADLINE_EPOCH - $(date +%s) - 600 ))
[ "$LEFT" -gt 120 ] || { say "STOP: only ${LEFT}s left for the probe"; echo "SKIPPED probe host-limited deadline" >> summary.txt; finish 40; }
say "probe: $EXPERTS experts, cap min(${LEFT}s, 1.5 x reps x warm)"
if [ "$MODE" = sib ]; then
  env E4B_SHA=$E4B_SHA SMT_FUSE_TIME_LEFT_S=$LEFT perl -e "alarm $LEFT; exec @ARGV" python $W/smt_sib_box.py --out $W/result.json --experts $EXPERTS > logs/probe.log 2>&1
else
  env E4B_SHA=$E4B_SHA SMT_FUSE_TIME_LEFT_S=$LEFT perl -e "alarm $LEFT; exec @ARGV" python $W/smt_fuse_box.py --out $W/result.json --experts $EXPERTS > logs/probe.log 2>&1
fi
rc=$?
grep -a "^SMT_FUSE_SUMMARY\|^SMT_FUSE_CAP\|^SMT_SIB_SUMMARY\|^SMT_SIB_CAP\|^SMT_SIB_PLAN\|^SMT_SIB_REFUSED" logs/probe.log | cut -c1-1500 | tee -a summary.txt
[ "$rc" = 0 ] && [ -s $W/result.json ] || { tail -5 logs/probe.log | tee -a summary.txt; say "PROBE FAILED (rc=$rc)"; finish 22; }
finish 0
