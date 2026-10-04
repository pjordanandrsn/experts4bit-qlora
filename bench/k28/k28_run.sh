#!/bin/bash
# bench/k28/k28_run.sh -- lane K28, BOX side. The prereg and the bench live in grouped-nf4-gemm
# (kernel/PREREG-k28-pdl-decode-chain.md, kernel/k28_bench.py). Started detached by k28_drive.sh with the run's nonce;
# K28_RUN_NONCE first, then K28_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and K28_SUCCESS.<nonce> only when the
# bench ran.
#
# Does programmatic dependent launch (GNF4_PDL=1) shorten a CUDA-graph replay of the served B=1 decode layer's gnf4
# kernels, bit-identically? One RTX 5090, no model:
#   premise  the switch's contract compiled on this card (grouped-nf4-gemm kernel/test_pdl.py: 23 passed, none
#            skipped -- its sm_90+ tests are the on-card bitwise and engagement checks; rc 23)
#   bench    k28_bench.py: 48 layers of the served B=1 layer's gnf4 kernels at Qwen3-30B-A3B shapes, off vs on -> k28.json
# Lane failures use rc 23 / 34 (never 13/14/17/18, which exclude the machine, except the disk floor's 13).
# Knobs (any off its default marks a REHEARSAL): K28_GPU_CLASS K28_MIN_DISK_GB K28_REHEARSAL.
set -uo pipefail
W=/root/k28; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] k28: $*"; }
NONCE=${K28_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/K28_RUN_NONCE.tmp && mv $W/K28_RUN_NONCE.tmp $W/K28_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > K28_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > K28_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in K28_RUN_ID K28_DEADLINE_EPOCH K28_INSTANCE_ID GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*) say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA must be a 40-char sha"; finish 78; }
GPU_CLASS=${K28_GPU_CLASS:-5090}; MIN_DISK_GB=${K28_MIN_DISK_GB:-20}; REHEARSAL=${K28_REHEARSAL:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > summary.txt; echo "$K28_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS gnf4=$GNF4_SHA gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 20 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
[ -s $W/staged.sha256 ] || { say "STAGE MISSING: staged.sha256"; finish 9; }
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/k28/staged.sha256"; finish 9; }
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
arm_alarm(){ local cap=$1 left=$(( K28_DEADLINE_EPOCH - $(date +%s) - 300 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 300 ] && left=300; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
say "install gnf4 @$GNF4_SHA (image python and torch) + pytest"
pipx logs/pip_gnf4.log 900 --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
pipx logs/pip_pytest.log 600 pytest || { tail -3 logs/pip_pytest.log; say "PIP FAIL (pytest)"; finish 9; }
perl -e 'alarm 600; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/src > logs/clone.log 2>&1 && git -C $W/src checkout -q $GNF4_SHA \
  || { say "CLONE FAIL"; finish 9; }
cp $W/src/kernel/k28_bench.py $W/ || { say "k28_bench.py missing at $GNF4_SHA"; finish 9; }
WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import int4_b32
for n in ("pdl_active", "pdl_default", "pdl_refresh", "_pdl_kw", "_pdl_enter", "PDL_ENV"):
    assert hasattr(int4_b32, n), f"installed gnf4 lacks {n} (the GNF4_PDL switch K28 measures)"
assert int4_b32.PDL_ENV == "GNF4_PDL" and int4_b32.PDL_DEFAULT is False
src = os.path.abspath(inspect.getsourcefile(int4_b32))
assert not src.startswith("/root/k28/src/"), f"int4_b32 resolved to the clone ({src})"
import torch, triton
from triton.backends.nvidia.compiler import CUDAOptions
assert "launch_pdl" in CUDAOptions.__dataclass_fields__, "this triton cannot launch with PDL"
assert "griddepcontrol.wait" in inspect.getsource(int4_b32._pdl_enter.fn), "installed gnf4's preamble is not the inline-PTX one"
assert torch.cuda.get_device_capability() >= (9, 0), "the card is below sm_90"
os.environ["GNF4_PDL"] = "1"
assert int4_b32.pdl_refresh() is True and int4_b32.pdl_active("cuda") is True, "gnf4 does not read the switch as active on this card"
del os.environ["GNF4_PDL"]
assert int4_b32.pdl_refresh() is False
open("/root/k28/versions.txt", "w").write(f"gnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\n"
                                          f"torch {torch.__version__}\ntriton {triton.__version__}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK")
PYT
cat versions.txt | tee -a summary.txt
python $W/k28_bench.py --self-test | tee -a summary.txt || { say "RULE SELF-TEST FAILED"; finish 21; }
# ---- the premise: the switch's contract compiled on this card, every test run (its sm_90+ tests skip nowhere here)
(cd $W/src/kernel && PYTHONPATH= TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_pdl.py -q -rs -p no:cacheprovider) > logs/pdl_contract.log 2>&1
rc=$?; LASTL=$(tail -1 logs/pdl_contract.log); echo "premise GNF4_PDL contract compiled rc=$rc: $LASTL" | tee -a summary.txt
{ [ "$rc" = 0 ] && echo "$LASTL" | grep -q "23 passed" && ! echo "$LASTL" | grep -q skipped; } \
  || { say "PREMISE FAILED: the switch's contract does not hold (or did not all run) on this card"; echo "pdl contract rc=$rc: $LASTL" > REFUSAL; finish 23; }
# ---- the bench
say "bench (alarm $(arm_alarm 1500))"
PYTHONPATH= perl -e "alarm $(arm_alarm 1500); exec @ARGV" python $W/k28_bench.py $W/k28.json > logs/k28_bench.log 2>&1
rc=$?; grep -aE '^copy |^K28 |K28_VERDICT' logs/k28_bench.log | tee -a summary.txt
[ "$rc" = 0 ] && [ -s k28.json ] || { say "BENCH rc=$rc"; finish 34; }
finish 0
