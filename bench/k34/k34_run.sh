#!/bin/bash
# bench/k34/k34_run.sh -- lane K34, BOX side. The prereg and the bench live in grouped-nf4-gemm
# (kernel/PREREG-k34-k16-wide-plan-census.md, kernel/k34_bench.py). Started detached by k34_drive.sh with the run's nonce;
# K34_RUN_NONCE first, then K34_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and K34_SUCCESS.<nonce> only when the
# bench ran.
#
# Which K16 plan runs the 32- and 64-row tiles (grouped-nf4-gemm #522's block_m=) fastest at Qwen3-30B-A3B's fused qkv
# (5120x2048) and o (2048x4096) projections? EXPLORATORY: a CANDIDATE licenses nothing; a plan that would move a default
# gets its own confirmatory read in experts4bit-qlora. One RTX 5090, no model:
#   premise  the small-M GEMM's contract compiled on this card (grouped-nf4-gemm kernel/test_int4_smallm_interp.py with
#            TRITON_INTERPRET=0: 25 passed, none skipped -- the 16/32/64-row tiles, split-K, the workspace; rc 23)
#   bench    k34_bench.py: 48 layers of synthetic int4-b32 stores per projection, one graph per (projection, tile, arm or
#            plan); 48 plans selected under one bf16 ulp of the fp32 reference, the best re-timed against the shipped
#            plan twice and cuBLAS bf16 -> k34.json
# Lane failures use rc 23 / 34 (never 13/14/17/18, which exclude the machine, except the disk floor's 13).
# Knobs (any off its default marks a REHEARSAL): K34_GPU_CLASS K34_MIN_DISK_GB K34_REHEARSAL.
set -uo pipefail
W=/root/k34; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] k34: $*"; }
NONCE=${K34_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/K34_RUN_NONCE.tmp && mv $W/K34_RUN_NONCE.tmp $W/K34_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > K34_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > K34_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in K34_RUN_ID K34_DEADLINE_EPOCH K34_INSTANCE_ID GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*) say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA must be a 40-char sha"; finish 78; }
GPU_CLASS=${K34_GPU_CLASS:-5090}; MIN_DISK_GB=${K34_MIN_DISK_GB:-20}; REHEARSAL=${K34_REHEARSAL:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_FUSED_REDUCE GNF4_GEMV_BW GNF4_TILE_PROGRAMS TRITON_INTERPRET
: > summary.txt; echo "$K34_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS gnf4=$GNF4_SHA gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 20 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
[ -s $W/staged.sha256 ] || { say "STAGE MISSING: staged.sha256"; finish 9; }
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/k34/staged.sha256"; finish 9; }
# A GPU the image's torch cannot use is the REGISTERED HOST FLOOR (rent.py's class 18), as TC1 amendment 61 made it: exit 18 with
# a REFUSAL line, so the launcher names the machine and a relaunch cannot buy it again. Exit 10 named no machine:
# p115-5090-1 and tc1-5090-119 both drew Vast machine 34887 (CUDA error 803) and read HARNESS_ERROR. A torch that will not
# import is the image's fault, not the host's, so it stays 10.
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
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
arm_alarm(){ local cap=$1 left=$(( K34_DEADLINE_EPOCH - $(date +%s) - 300 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 300 ] && left=300; echo "$left"; }
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
cp $W/src/kernel/k34_bench.py $W/ || { say "k34_bench.py missing at $GNF4_SHA"; finish 9; }
WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import sys
import int4_smallm as sm
for n in ("gemm_int4_b32_smallm", "plan_smallm", "smallm_workspace", "smallm_block_m", "SMALLM_ROWS_MAX", "_SUPPORTED_BLOCK_M",
          "_SUPPORTED_KC"):
    assert hasattr(sm, n), f"installed gnf4 lacks {n} (the small-M GEMM K34 censuses)"
assert sm.SMALLM_ROWS_MAX == 64 and tuple(sm._SUPPORTED_BLOCK_M) == (16, 32, 64), "the 32- and 64-row tiles (#522)"
assert {128, 256} <= set(sm._SUPPORTED_KC), sm._SUPPORTED_KC
assert (sm.smallm_block_m(32), sm.smallm_block_m(64)) == (32, 64)
p = inspect.signature(sm.gemm_int4_b32_smallm).parameters
assert {"block_n", "kc", "sk", "warps", "stages", "workspace", "block_m"} <= set(p), list(p)
# The shipped plan is what experts4bit-qlora serves: Int4Linear passes plan_smallm(N, K) and leaves warps/stages default.
served = {k: sm.plan_smallm(N, K) + (p["warps"].default, p["stages"].default) for k, (N, K) in
          {"qkv": (5120, 2048), "o": (2048, 4096)}.items()}
sys.path.insert(0, "/root/k34")
import k34_bench as kb
assert all(v == tuple(kb.SHIPPED) for v in served.values()), f"k34_bench's SHIPPED {kb.SHIPPED} is not the served plan {served}"
assert kb.SHAPES == {"qkv": (5120, 2048), "o": (2048, 4096)} and tuple(kb.TILES) == (32, 64) and len(kb.PLANS) == 48
assert tuple(kb.SHIPPED) in [tuple(q) for q in kb.PLANS], "the shipped plan must be in the grid"
src = os.path.abspath(inspect.getsourcefile(sm))
assert not src.startswith("/root/k34/src/"), f"int4_smallm resolved to the clone ({src})"
import torch, triton
open("/root/k34/versions.txt", "w").write(f"gnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\n"
                                          f"torch {torch.__version__}\ntriton {triton.__version__}\ncc {torch.cuda.get_device_capability()}\n"
                                          f"sm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK")
PYT
cat versions.txt | tee -a summary.txt
python $W/k34_bench.py --self-test | tee -a summary.txt || { say "RULE SELF-TEST FAILED"; finish 21; }
# ---- the premise: the small-M GEMM's contract compiled on this card, every test run
(cd $W/src/kernel && PYTHONPATH= TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_int4_smallm_interp.py -q -rs -p no:cacheprovider) > logs/smallm_contract.log 2>&1
rc=$?; LASTL=$(tail -1 logs/smallm_contract.log); echo "premise int4 small-M contract compiled rc=$rc: $LASTL" | tee -a summary.txt
{ [ "$rc" = 0 ] && echo "$LASTL" | grep -q "25 passed" && ! echo "$LASTL" | grep -q skipped; } \
  || { say "PREMISE FAILED: the kernel's contract does not hold (or did not all run) on this card"; echo "smallm contract rc=$rc: $LASTL" > REFUSAL; finish 23; }
# ---- the bench
say "bench (alarm $(arm_alarm 1500))"
PYTHONPATH= perl -e "alarm $(arm_alarm 1500); exec @ARGV" python $W/k34_bench.py $W/k34.json > logs/k34_bench.log 2>&1
rc=$?; grep -aE '^K34 |K34_VERDICT' logs/k34_bench.log | cut -c1-600 | tee -a summary.txt
[ "$rc" = 0 ] && [ -s k34.json ] || { say "BENCH rc=$rc"; finish 34; }
finish 0
