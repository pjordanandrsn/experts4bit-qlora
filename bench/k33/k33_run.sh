#!/bin/bash
# bench/k33/k33_run.sh -- lane K33, BOX side. The prereg and the bench live in grouped-nf4-gemm
# (kernel/PREREG-k33-nf4-decode-gemv-bw.md, kernel/k33_bench.py). Started detached by k33_drive.sh with the run's nonce;
# K33_RUN_NONCE first, then K33_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and K33_SUCCESS.<nonce> only when the
# bench ran.
#
# Does the bandwidth-targeted NF4 decode GEMV (GNF4_GEMV_BW=1, _gemv_nf4_bw) run the single-row expert projections near
# the streaming ceiling, faster than the served route? One RTX 5090, no model:
#   premise  the kernel's contract compiled on this card (grouped-nf4-gemm kernel/test_nf4_gemv_bw.py: 27 passed, none
#            skipped -- prmt32 bitwise the tree, one-hot readback, the tolerance contract, the PTX, PDL; rc 23)
#   bench    k33_bench.py: every layer of Qwen3-30B-A3B, Granite-3.1-3b-a800m and OLMoE-1B-7B, one projection per
#            graph, the incumbent route against _gemv_nf4_bw's two decodes -> k33.json
# Lane failures use rc 23 / 34 (never 13/14/17/18, which exclude the machine, except the disk floor's 13).
# Knobs (any off its default marks a REHEARSAL): K33_GPU_CLASS K33_MIN_DISK_GB K33_REHEARSAL.
set -uo pipefail
W=/root/k33; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] k33: $*"; }
NONCE=${K33_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/K33_RUN_NONCE.tmp && mv $W/K33_RUN_NONCE.tmp $W/K33_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > K33_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > K33_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in K33_RUN_ID K33_DEADLINE_EPOCH K33_INSTANCE_ID GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*) say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA must be a 40-char sha"; finish 78; }
GPU_CLASS=${K33_GPU_CLASS:-5090}; MIN_DISK_GB=${K33_MIN_DISK_GB:-20}; REHEARSAL=${K33_REHEARSAL:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset GNF4_GEMV_BW GNF4_GEMV_BW_DECODE GNF4_GEMV_BW_PLAN GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN GNF4_GEMV_SPLITK GNF4_PDL GNF4_PDL_MAX_ROWS \
      GNF4_GEMV_WIDE_LOADS GNF4_GEMV_VEC_LOADS TRITON_INTERPRET
: > summary.txt; echo "$K33_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS gnf4=$GNF4_SHA gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 20 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
[ -s $W/staged.sha256 ] || { say "STAGE MISSING: staged.sha256"; finish 9; }
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/k33/staged.sha256"; finish 9; }
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
arm_alarm(){ local cap=$1 left=$(( K33_DEADLINE_EPOCH - $(date +%s) - 300 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 300 ] && left=300; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
say "install gnf4 @$GNF4_SHA (image python and torch) + pytest + numpy"
pipx logs/pip_gnf4.log 900 --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
pipx logs/pip_pytest.log 600 pytest numpy || { tail -3 logs/pip_pytest.log; say "PIP FAIL (pytest)"; finish 9; }
perl -e 'alarm 600; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/src > logs/clone.log 2>&1 && git -C $W/src checkout -q $GNF4_SHA \
  || { say "CLONE FAIL"; finish 9; }
cp $W/src/kernel/k33_bench.py $W/ || { say "k33_bench.py missing at $GNF4_SHA"; finish 9; }
WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import nf4_grouped as ng
for n in ("_gemv_nf4_bw", "_nf4_prmt32", "_nf4_tree", "_bw", "_bw_engages", "_bw_plan", "_bw_decode", "_bw_tables",
          "_BW_SHAPES", "dispatch_counts"):
    assert hasattr(ng, n), f"installed gnf4 lacks {n} (the GNF4_GEMV_BW route K33 measures)"
assert {"bw_tree", "bw_prmt32", "bw_splitk"} <= set(ng.dispatch_counts()), ng.dispatch_counts()
assert ng._BW_SHAPES == frozenset(), "auto must engage nowhere before K33 reads"
assert ng._bw() == "0", "the switch is off by default"
src = os.path.abspath(inspect.getsourcefile(ng))
assert not src.startswith("/root/k33/src/"), f"nf4_grouped resolved to the clone ({src})"
import torch, triton
assert ng._bw_decode("cuda") == "prmt32", "prmt32 must be the decode on this compiled CUDA target"
assert "prmt.b32" in inspect.getsource(ng._nf4_prmt32.fn), "installed gnf4's decode is not the PTX byte-permute one"
open("/root/k33/versions.txt", "w").write(f"gnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\n"
                                          f"torch {torch.__version__}\ntriton {triton.__version__}\ncc {torch.cuda.get_device_capability()}\n"
                                          f"sm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK")
PYT
cat versions.txt | tee -a summary.txt
python $W/k33_bench.py --self-test | tee -a summary.txt || { say "RULE SELF-TEST FAILED"; finish 21; }
# ---- the premise: the kernel's contract compiled on this card, every test run (its sm_90+ PDL tests skip nowhere here)
(cd $W/src/kernel && PYTHONPATH= TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_nf4_gemv_bw.py -q -rs -s -p no:cacheprovider) > logs/bw_contract.log 2>&1
rc=$?; LASTL=$(tail -1 logs/bw_contract.log); echo "premise GNF4_GEMV_BW contract compiled rc=$rc: $LASTL" | tee -a summary.txt
grep -a "^K33 PTX" logs/bw_contract.log | tee -a summary.txt
{ [ "$rc" = 0 ] && echo "$LASTL" | grep -q "27 passed" && ! echo "$LASTL" | grep -q skipped; } \
  || { say "PREMISE FAILED: the kernel's contract does not hold (or did not all run) on this card"; echo "bw contract rc=$rc: $LASTL" > REFUSAL; finish 23; }
# ---- the bench
say "bench (alarm $(arm_alarm 1500))"
PYTHONPATH= perl -e "alarm $(arm_alarm 1500); exec @ARGV" python $W/k33_bench.py $W/k33.json > logs/k33_bench.log 2>&1
rc=$?; grep -aE '^K33 |K33_VERDICT' logs/k33_bench.log | cut -c1-600 | tee -a summary.txt
[ "$rc" = 0 ] && [ -s k33.json ] || { say "BENCH rc=$rc"; finish 34; }
finish 0
