#!/bin/bash
# bench/p66/p66_run.sh -- lane P66, BOX side (bench/p66/P66-PREREG.md). Started detached by p66_drive.sh with the
# run's nonce; P66_RUN_NONCE first, then P66_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P66_SUCCESS.<nonce> only on clean completion. Pattern: bench/p60/p60_run.sh (model lane) + bench/b374 (nonce).
#
# The residency launch census: what residency adds per token in CUDA launches and host syncs against the
# all-resident step, whether that moves with cold fraction, and cold_deadline's transfer prediction against the
# measured transfer -- pipelined (NF4) at five hot fractions including 0, the hybrid tier (VRAM + NVMe), and the
# MXFP4 engines (pinned and NVMe), each against its family's all-resident step, captured where capture works.
# Level L (p66_census.py) is the registered reading; Level M (step_decomp through p66_step.py) is context.
# Nothing here changes a default.
#
# P66_MODE (forwarded by p66_drive.sh): `full` (default) is the reading. `prove` is the PROVING RENTAL the compute
# rule requires before any rental whose guard exceeds 1 h (P66-PREREG.md "Box, cost, receipts"): the box checks
# (reading-only floors RECORDED, not enforced -- see floor()), the pinned install, the tripwire and the pin, then one
# timed shard of the pinned Qwen3 fetch if time allows, and it stops -- no calibration, no bake, no arm.
#
# REHEARSAL OVERRIDES (P66_REHEARSAL_*). The registered run uses none of them: p66_drive.sh forwards only the run's
# identity and the two pins (tests/test_p66_staged_pin.py holds that), and every override in effect is written to
# summary.txt as `OVERRIDE ...`, so a rehearsal's receipts can never pass for a reading. They exist so this script
# itself -- not only the census it runs -- is exercised on the free NAS GPU before a box is rented:
#   P66_REHEARSAL_CLASS            card-name pattern instead of 5090
#   P66_REHEARSAL_MIN_VRAM_MB / _MIN_DISK_GB / _MIN_RAM_GB / _PIN_GIB / _MIN_MBPS   the host minimums
#   P66_REHEARSAL_SKIP_INSTALL=1   use the packages already importable; P66_REHEARSAL_GNF4_SRC = a gnf4 tree
#   P66_REHEARSAL_QWEN_SNAPSHOT / _GPTOSS_SNAPSHOT   local checkpoint dirs instead of the pinned HF fetch
#   P66_REHEARSAL_NF4_FAMILY       the family label for that NF4 checkpoint
#   P66_REHEARSAL_NF4_ARENA        an NF4 arena already baked from it by the same k8_bake.py (skips the bake)
#   P66_REHEARSAL_MX_LAYERS        a-b: bake and census only these MXFP4 layers
#   P66_REHEARSAL_TOKENS / _WARM   tokens per pass
set -uo pipefail
W=/root/p66; mkdir -p $W/logs $W/L $W/M; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p66: $*"; }
NONCE=${P66_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P66_RUN_NONCE.tmp && mv $W/P66_RUN_NONCE.tmp $W/P66_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P66_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P66_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P66_RUN_ID P66_DEADLINE_EPOCH P66_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in E4B_SHA GNF4_SHA; do
  val=${!v}
  case "$val" in *[!0-9a-f]*|"") say "refusing: $v is not hex"; finish 78;; esac
  [ ${#val} -eq 40 ] || { say "refusing: $v is not a 40-char sha"; finish 78; }
done
MODE=${P66_MODE:-full}; case "$MODE" in full|prove) ;; *) say "refusing: P66_MODE must be full or prove, got '$MODE'"; finish 78;; esac
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
QMID=Qwen/Qwen3-30B-A3B; QREV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39      # P60's pin
GMID=openai/gpt-oss-20b; GREV=6cee5e81ee83917806bbde320786a8fb61efebee      # tp4's pin
: > summary.txt; echo "$P66_INSTANCE_ID" > INSTANCE_ID
CLASS=${P66_REHEARSAL_CLASS:-5090}; MIN_VRAM_MB=${P66_REHEARSAL_MIN_VRAM_MB:-30000}; MIN_DISK_GB=${P66_REHEARSAL_MIN_DISK_GB:-180}
MIN_RAM_GB=${P66_REHEARSAL_MIN_RAM_GB:-64}; MIN_MBPS=${P66_REHEARSAL_MIN_MBPS:-80}; FETCH_WORKERS=8; PIN_GIB=${P66_REHEARSAL_PIN_GIB:-16}; SKIP_INSTALL=${P66_REHEARSAL_SKIP_INSTALL:-0}
QS_LOCAL=${P66_REHEARSAL_QWEN_SNAPSHOT:-}; GS_LOCAL=${P66_REHEARSAL_GPTOSS_SNAPSHOT:-}; NF_FAM=${P66_REHEARSAL_NF4_FAMILY:-qwen3-30b-a3b}
MX_LAYERS=${P66_REHEARSAL_MX_LAYERS:-all}; TOKENS=${P66_REHEARSAL_TOKENS:-8}; WARM=${P66_REHEARSAL_WARM:-4}
echo "MODE $MODE" | tee -a summary.txt
for v in $(env | sed -n 's/^\(P66_REHEARSAL_[A-Z0-9_]*\)=.*/\1/p' | sort); do echo "OVERRIDE $v=${!v} -- this run is a REHEARSAL, not a reading" | tee -a summary.txt; done
# ---- staged pieces, byte-for-byte
for f in p66_census.py p66_reduce.py p66_step.py k8_bake.py step_decomp.py step_budget.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p66/staged.sha256"; finish 9; }
# ---- deadline guard + per-arm alarm (P54's): never start an arm that cannot finish 10 min before teardown.
# `need` is the arm's EXPECTED time, `cap` its alarm; they are separate on purpose. The first draft passed the cap as
# the need, so on a 2 h guard a late arm with a 40-minute cap was skipped as "no time left" while it needed three.
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P66_DEADLINE_EPOCH" ] || { say "STOP: $2 needs ${need}s, only $((P66_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P66_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 300 ] && left=300; echo "$left"; }

# floor(): a host minimum the READING needs (P66-PREREG "Box"). The reading refuses on it. A PROOF (P66_MODE=prove)
# only records it: the proving box is not the reading's box, so a reading-only floor that refuses a proof buys
# nothing and spends the proof budget -- P65 Amendment 1 (bench/p65/P65-PREREG.md) lost two proofs that way, each
# missing by ~2 %. The card class and a dud box refuse either way; so does disk below what the proof itself needs.
floor(){ local rc=$1 msg=$2
  if [ "$MODE" = prove ]; then say "PROOF: would refuse the READING -- $msg (recorded, not enforced in a proof)"
    echo "floor_would_refuse_reading rc=$rc $msg" | tee -a forensics.txt >> summary.txt; return 0; fi
  say "REFUSED: $msg -- host-limited, not a result"; echo "refused: $msg" > REFUSAL; finish "$rc"; }
PROOF_MIN_DISK_GB=20        # a proof's own need: the pip dependencies and one ~4 GB checkpoint shard

# ---- the box: class, then the host minimums the residency arms need (P66-PREREG "Box")
python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv,noheader | sed "s/^/power,clock,pcie gen cur,max,width cur,max: /" | tee -a forensics.txt
lscpu | grep -E "Model name|^CPU\(s\)|NUMA node\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt
cat /sys/fs/cgroup/memory.max 2>/dev/null | sed "s/^/cgroup memory.max /" | tee -a forensics.txt
cat /sys/fs/cgroup/cpu.max 2>/dev/null | sed "s/^/cgroup cpu.max /" | tee -a forensics.txt
df -h /root | tail -1 | tee -a forensics.txt; nvcc --version 2>/dev/null | tail -1 | tee -a forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *$CLASS*) ;; *) say "REFUSED: card is '$GPU', the lane registers the RTX $CLASS class"; echo "refused: class $GPU" > REFUSAL; finish 15;; esac
VFREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "${VFREE:-0}" -ge "$MIN_VRAM_MB" ] || floor 15 "${VFREE} MiB VRAM free, the all-resident arms need >= $MIN_VRAM_MB"
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "$MODE" = prove ] && [ "${FREE_GB:-0}" -lt "$PROOF_MIN_DISK_GB" ] && { say "REFUSED: ${FREE_GB} GB free on /root < $PROOF_MIN_DISK_GB GB, the proof's own need"; echo "refused: disk ${FREE_GB} GB (proof)" > REFUSAL; finish 13; }
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || floor 13 "${FREE_GB} GB free on /root < $MIN_DISK_GB GB (two checkpoints + two arenas + the NF4 snapshot)"
AVAIL_GB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
CG=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max); CG_GB=$([ "$CG" = max ] && echo 99999 || echo $(( CG / 1073741824 )))
[ "${AVAIL_GB:-0}" -ge "$MIN_RAM_GB" ] && [ "$CG_GB" -ge "$MIN_RAM_GB" ] || floor 13 "host RAM available ${AVAIL_GB} GB, cgroup ${CG_GB} GB; the pipelined arm pins a 15.2 GiB arena beside 15.2 GiB of materialized experts -- needs >= $MIN_RAM_GB GB"
# ---- egress, measured the way the fetch runs (P65 Amendment 2's lesson): the reading's snapshot_download pulls with
# FETCH_WORKERS (8) workers, so the probe is eight parallel 50 MB ranges of one HF CDN file and the rate is their total bytes over
# the wall time of all eight. In Python: the stock image ships no curl. A single stream under-read a 4-worker fetch
# 2.3x on P65's boxes. The reading needs ~75 GB (Qwen3 61 + gpt-oss 14): 16 min at the floor.
say "egress pre-flight: HF CDN, $FETCH_WORKERS parallel 50 MB ranges, 30 s cap (floor ${MIN_MBPS} MB/s aggregate)"
MBPS=$(P66_FETCH_WORKERS=$FETCH_WORKERS python3 - <<'PYEG' 2>/dev/null || echo 0
import os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL = "https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors"     # 440 MB: holds 8 x 50 MB
CH, N, CAP = 52428800, int(os.environ["P66_FETCH_WORKERS"]), 30.0
def get(i):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={i * CH}-{(i + 1) * CH - 1}"})
    got, t0 = 0, time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=CAP) as r:
            while time.perf_counter() - t0 < CAP:
                b = r.read(1 << 20)
                if not b:
                    break
                got += len(b)
    except Exception:
        pass
    return got
t0 = time.perf_counter()
with ThreadPoolExecutor(N) as ex:
    total = sum(ex.map(get, range(N)))
print(round(total / 1e6 / max(1e-3, time.perf_counter() - t0), 1))
PYEG
)
say "HF CDN ${MBPS} MB/s ($FETCH_WORKERS parallel ranges)"; echo "hf_cdn_mbps_${FETCH_WORKERS}x=$MBPS" | tee -a forensics.txt >> summary.txt
if python3 -c "import sys; sys.exit(0 if float('${MBPS:-0}') < float('$MIN_MBPS') else 1)"; then floor 14 "egress ${MBPS} MB/s < ${MIN_MBPS} MB/s over $FETCH_WORKERS streams (the reading fetches ~75 GB)"; fi

# ---- install: e4b pinned (P37's toolchain pins, image python); gnf4 at GNF4_SHA, plus a clone of the same sha for
# bench/calibrate.py (not packaged)
export DEBIAN_FRONTEND=noninteractive
if [ "$SKIP_INSTALL" = 1 ]; then
  GSRC=${P66_REHEARSAL_GNF4_SRC:?P66_REHEARSAL_SKIP_INSTALL needs P66_REHEARSAL_GNF4_SRC (a gnf4 tree for bench/calibrate.py)}
  say "REHEARSAL: no install; packages as importable, calibrate from $GSRC"
else
  # the stock pytorch devel image ships no git (seen on the A2000 rehearsal); every install below is a git URL
  command -v git >/dev/null 2>&1 || { say "no git in the image: apt-get install git"; perl -e 'alarm 600; exec @ARGV' sh -c "apt-get update -qq && apt-get install -y -qq git" > logs/apt_git.log 2>&1; }
  command -v git >/dev/null 2>&1 || { tail -3 logs/apt_git.log; say "NO GIT -- cannot install the pinned packages"; finish 9; }
  say "install e4b @$E4B_SHA"
  perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
    "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
  perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
  perl -e 'alarm 900; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git src > logs/clone.log 2>&1 && git -C src checkout -q "$GNF4_SHA" || { tail -3 logs/clone.log; say "CLONE FAIL"; finish 9; }
  [ "$(git -C src rev-parse HEAD)" = "$GNF4_SHA" ] || { say "TRIPWIRE: clone is not $GNF4_SHA"; finish 9; }
  GSRC=$W/src
fi
P66_SKIP_INSTALL=$SKIP_INSTALL python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import importlib.metadata as md, inspect, os
import experts4bit_qlora as e, nf4_grouped, mxfp4_residency, mxfp4_pipelined, cold_deadline, nvme_arena, cpu_grouped
for m in (nf4_grouped, mxfp4_residency, cold_deadline, nvme_arena):
    f = os.path.abspath(inspect.getsourcefile(m))
    assert "/root/p66/src/" not in f, f"{m.__name__} resolved to the CLONE ({f}), not the installed package"
    assert os.environ["P66_SKIP_INSTALL"] == "1" or "site-packages" in f, f"{m.__name__} resolved to {f}"
from experts4bit_qlora.engines.pipelined import pipelined_available
assert pipelined_available(), "pipelined engine unavailable (triton / nf4_grouped / CUDA)"
assert cpu_grouped.cpu_kernels_available(), "gnf4_native CPU kernels did not build: the hybrid tier refuses without them"
assert hasattr(mxfp4_residency.Mxfp4NvmeResidency, "_resolve_src"), "gnf4 cut lacks the MXFP4 NVMe engine"
import torch, triton, transformers
open("/root/p66/versions.txt", "w").write(
    f"e4b {e.__version__} @{os.environ['E4B_SHA']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\n"
    f"e4b from {e.__file__}\ngnf4 from {nf4_grouped.__file__}\n"
    f"torch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\n")
print("tripwire OK:", e.__version__, md.version("grouped-nf4-gemm"), torch.__version__, triton.__version__)
PYT
cat versions.txt | tee -a summary.txt
python -c "import torch; t=torch.empty($PIN_GIB<<30, dtype=torch.uint8).pin_memory(); assert t.is_pinned(); print('pinned $PIN_GIB GiB ok')" > logs/pin.log 2>&1 \
  || { tail -2 logs/pin.log; floor 13 "cannot pin $PIN_GIB GiB of host memory (the residency arenas are UVA-read)"; }

# ---- the proving rental stops here, after one timed shard of the pinned fetch (the reading's longest unknown)
if [ "$MODE" = prove ]; then
  if [ -n "$QS_LOCAL" ]; then
    echo "PROVE fetch: skipped (rehearsal: local snapshot)" | tee -a summary.txt
  else
    left=$(( P66_DEADLINE_EPOCH - $(date +%s) - 120 )); [ "$left" -gt 240 ] && left=240
    if [ "$left" -ge 30 ]; then
      t0=$(date +%s)
      perl -e "alarm $left; exec @ARGV" python -c "from huggingface_hub import hf_hub_download as d; import os; p=d('$QMID', 'model-00001-of-00016.safetensors', revision='$QREV'); print(p, os.path.getsize(p))" > logs/prove_fetch.out 2> logs/prove_fetch.log
      rc=$?; t1=$(date +%s); B=$(tail -1 logs/prove_fetch.out | awk '{print $2}')
      echo "PROVE fetch: rc=$rc bytes=${B:-0} seconds=$((t1 - t0)) MB/s=$(python -c "print(round(${B:-0}/1e6/max(1,$((t1 - t0))), 1))")" | tee -a summary.txt
    else
      echo "PROVE fetch: skipped (no time left in the guard)" | tee -a summary.txt
    fi
  fi
  : > P66_PROVED.$NONCE
  say "PROVED: class, install and tripwire pass on this box; reading floors it would miss: $(grep -c '^floor_would_refuse_reading' summary.txt)"
  finish 0
fi

# ---- calibration: cold_deadline's constants come from the box's OWN blob (Costs.from_blob field names)
say "calibrate (gnf4 bench/calibrate.py)"
perl -e "alarm 1500; exec @ARGV" python $GSRC/bench/calibrate.py --out $W/calib.json --nvme-dir $W --nvme-gib 8 --tag p66 > logs/calibrate.log 2>&1 \
  || { tail -5 logs/calibrate.log; say "CALIBRATE FAIL -- the transfer model has no measured constants"; finish 33; }
tail -4 logs/calibrate.log | tee -a summary.txt
rm -f $W/hybrid_calib_nvme.dat    # calibrate.py's 8 GiB NVMe test file (seen on the A2000 rehearsal): not a receipt

# ---- the NF4 family: fetch (pinned), bake the NF4 arena (k8_bake.py, P60's recipe)
if [ -n "$QS_LOCAL" ]; then
  QS=$QS_LOCAL
else
  say "fetch $QMID @ $QREV"
  perl -e 'alarm 4800; exec @ARGV' python -c "from huggingface_hub import snapshot_download as s; print(s('$QMID', revision='$QREV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=$FETCH_WORKERS))" > logs/fetch_qwen.out 2> logs/fetch_qwen.log \
    || { tail -2 logs/fetch_qwen.log; say "DL FAIL (qwen)"; finish 11; }
  QS=$(tail -1 logs/fetch_qwen.out)
fi
[ -s "$QS/config.json" ] || { say "DL FAIL (nf4 family): no config at '$QS'"; finish 11; }
echo "nf4 snapshot $QS" | tee -a summary.txt
if [ -n "${P66_REHEARSAL_NF4_ARENA:-}" ]; then
  QA=$P66_REHEARSAL_NF4_ARENA; [ -e "$QA" ] && [ -s "$QA.index.json" ] || { say "REHEARSAL arena '$QA' missing"; finish 12; }
else
  say "bake NF4 arena"; mkdir -p $W/work_qwen3
  K8_MODEL="$QS" K8_WORK="$W/work_qwen3" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > logs/bake_qwen.log 2>&1 || { tail -3 logs/bake_qwen.log; say "BAKE FAIL (rc)"; finish 12; }
  QA=$W/work_qwen3/nf4.arena; [ -e "$QA" ] || { cat $W/work_qwen3/bake.json 2>/dev/null | head -20; say "BAKE FAIL (no arena)"; finish 12; }
  rm -rf $W/work_qwen3/nf4snap
  python -c "import json; r=json.load(open('$W/work_qwen3/bake.json')); print('BAKE', r['status'], r.get('layers'), r.get('experts'), r.get('snapshot_gib'), 'GiB')" | tee -a summary.txt
fi

C="python -u $W/p66_census.py --calib $W/calib.json --out $W/L --warm $WARM --tokens $TOKENS"
NF="--family $NF_FAM --arena $QA --snapshot $QS"
L_FAILED=""   # a Level L arm that fails is recorded and the lane continues; the success marker is withheld (rc 32)
larm(){ local n=$1 need=$2 cap=$3; shift 3; can_run $need "$n" || { L_FAILED="$L_FAILED $n(time)"; return 0; }
  say "L $n"; perl -e "alarm $(arm_alarm $cap); exec @ARGV" "$@" > logs/$n.log 2>&1; local rc=$?
  echo "L $n rc=$rc" | tee -a summary.txt; grep -aE "have-skip|capture probe|REFUSE|Error" logs/$n.log | tail -6 | tee -a summary.txt
  [ "$rc" = 0 ] || L_FAILED="$L_FAILED $n(rc=$rc)"; return 0; }
# ---- Level L, NF4: the all-resident reference first, then the issue's subject (pipelined), then the hybrid tier
# need (expected s) and cap (alarm s): expected from the A2000 rehearsals scaled 3x (48 layers against 16), then
# doubled; caps are the ceilings a hung arm cannot exceed
larm L_ref   300 1200 $C $NF --path ref
# 0.828125 = 106 of 128 hot = 17.19 % cold, exactly vLLM #57794's 88 of 512 (P8)
larm L_pipe  900 2700 $C $NF --path pipe --hot-fracs 1.0,0.828125,0.5,0.25,0.0 --controlled 0,4,8
larm L_hyb   900 2400 $C $NF --path hyb --hot-fracs 0.5,0.25 --controlled 0,4,8 --hot-rows 64 --capture-probe

# ---- gpt-oss-20b: fetch (pinned), relocation-bake the native MXFP4 arena (all 24 layers), Level L MXFP4
GS=""
if [ -n "$GS_LOCAL" ]; then
  GS=$GS_LOCAL
else
  say "fetch $GMID @ $GREV"
  if perl -e 'alarm 2400; exec @ARGV' python -c "from huggingface_hub import snapshot_download as s; print(s('$GMID', revision='$GREV', allow_patterns=['model*.safetensors','*.json'], max_workers=$FETCH_WORKERS))" > logs/fetch_gptoss.out 2> logs/fetch_gptoss.log; then
    GS=$(tail -1 logs/fetch_gptoss.out)
  else
    tail -2 logs/fetch_gptoss.log; say "DL FAIL (gpt-oss) -- the MXFP4 arms are skipped, recorded"; echo "SKIPPED MXFP4 arms: download failed" >> summary.txt
    L_FAILED="$L_FAILED MXFP4(download)"
  fi
fi
if [ -n "$GS" ] && [ -s "$GS/config.json" ]; then
  say "bake MXFP4 arena (layers $MX_LAYERS)"
  perl -e 'alarm 1800; exec @ARGV' python -c "
import sys, nvme_arena
lay = None if sys.argv[3] == 'all' else list(range(int(sys.argv[3].split('-')[0]), int(sys.argv[3].split('-')[1]) + 1))
nvme_arena.bake(sys.argv[1], sys.argv[2], layers=lay)" "$GS" $W/gptoss.arena "$MX_LAYERS" > logs/bake_gptoss.log 2>&1 \
    || { tail -3 logs/bake_gptoss.log; say "BAKE FAIL (gpt-oss) -- the MXFP4 arms are skipped"; echo "SKIPPED MXFP4 arms: bake failed" >> summary.txt; GS=""; L_FAILED="$L_FAILED MXFP4(bake)"; }
elif [ -n "$GS" ]; then
  say "no gpt-oss config at '$GS' -- the MXFP4 arms are skipped"; GS=""; L_FAILED="$L_FAILED MXFP4(snapshot)"
fi
if [ -n "$GS" ]; then
  tail -1 logs/bake_gptoss.log | tee -a summary.txt
  MX="--family gpt-oss-20b --arena $W/gptoss.arena --snapshot $GS"
  larm L_mref  300 1200 $C $MX --path mref
  larm L_mpin  360 1800 $C $MX --path mpin --hot-fracs 0.5,0.0 --controlled 0,2,4
  larm L_mnvme 600 2400 $C $MX --path mnvme --hot-fracs 1.0,0.5,0.25,0.0 --controlled 0,2,4 --hot-rows 64 --capture-probe
fi

# ---- Level M (context): the served decode step through step_decomp, B=1 eager, profiler window 12 steps
EXP_GB=$(python -c "import json; i=json.load(open('$QA.index.json')); print(round(i['row_bytes']*i['n_layers']*i['n_experts_per_layer']/2**30, 3))")
SD="python -u $W/p66_step.py --model $QS --arena $QA --calib $W/calib.json --batch 1 --prompt-len 128 --gen-tokens 48 --amort off"
marm(){ local n=$1 need=$2 cap=$3; shift 3; can_run $need "$n" || return 0
  say "M $n"; perl -e "alarm $(arm_alarm $cap); exec @ARGV" "$@" --torch-profile-out $W/M/${n}_kernels.txt --sync-attr-out $W/M/${n}_sync.json --out $W/M/${n}.json > logs/$n.log 2>&1
  local rc=$?; echo "M $n rc=$rc" | tee -a summary.txt; grep -aE "TORCH_PROFILE_OUT|SYNC_ATTR_OUT|hybrid tier active|p66_step|REFUSE|Error" logs/$n.log | tail -5 | tee -a summary.txt; return 0; }
marm M_ref       480 1200 $SD --engine hybrid --placement-override all-vram
marm M_pipe-1.00 600 1500 $SD --engine pipelined --chunk 1
marm M_pipe-0.00 600 1500 env P66_HOT_FRAC=0.0 P66_HOT_SETS_OUT=$W/M/M_pipe-0.00_hot.json $SD --engine pipelined --chunk 1
# 256 tier rows, not 64: this arm PREFILLS 128 tokens in one chunk, and the tier refuses when one forward routes more
# distinct cold experts than it holds (enable_nvme_residency's hard floor); the solver, not a seed, picks VRAM here
marm M_hyb-0.50  720 1800 $SD --engine hybrid --vram-gb $(python -c "print(round($EXP_GB*0.5, 3))") --dram-gb 0 --hot-rows 256

# ---- reduce (the box's own read; the committed RESULTS are regenerated from the fetched receipts)
say "reduce"
python $W/p66_reduce.py $W/L --json $W/p66_read.json --md $W/RESULTS-p66-generated.md --level-m $W/M --step-budget $W/step_budget.py | tee RESULTS.txt
say "----- summary -----"; cat summary.txt; echo "----- versions -----"; cat versions.txt
[ -z "$L_FAILED" ] || { say "Level L arms incomplete:$L_FAILED -- receipts kept, success withheld"; echo "L_FAILED$L_FAILED" >> summary.txt; finish 32; }
finish 0
