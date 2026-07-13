#!/bin/bash
# BM4 box runner: gated ON-ladder rerun (prereg 2bbd2e2 bars) + H2D residue probe.
# Runs as ubuntu on Latitude g3.h100.small (ML-in-a-Box, raid-0). Detach with nohup.
# Writes /home/ubuntu/dq/BM4_SENT lines as the session-independent progress channel.
set -u
cd /home/ubuntu
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"  # pip --break-system-packages as non-root = user install
DQ=/home/ubuntu/dq
BM4=/home/ubuntu/bm4
mkdir -p $DQ $DQ/outputs $BM4/probe /home/ubuntu/fstore
SENT=$DQ/BM4_SENT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $SENT; }
say "BM4_START host=$(hostname) kernel=$(uname -r)"

# ---------- phase 0: system ----------
sudo DEBIAN_FRONTEND=noninteractive apt-get -qq update >/dev/null 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y cmake fio pciutils >/dev/null 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a $SENT
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT | grep -E "md|nvme" | tee -a $SENT

# ---------- phase 1: env build ----------
say "BUILD_START"
if [ ! -d /home/ubuntu/axolotl ]; then
  git clone --branch feature/expert-store --single-branch --depth 5 \
    https://github.com/pjordanandrsn/axolotl.git /home/ubuntu/axolotl >> $DQ/build.log 2>&1 || { say "BUILD_ABORT clone"; exit 1; }
fi
cd /home/ubuntu/axolotl
git config user.email bm4@local && git config user.name bm4
if ! git log --oneline -3 | grep -q "H2D stage-copy probe"; then
  git am $BM4/axolotl-probe.patch >> $DQ/build.log 2>&1 || { git am --abort; say "BUILD_ABORT patch"; exit 1; }
fi
say "BUILD axolotl HEAD=$(git log --oneline -1 | cut -c1-60)"
PIP="python3 -m pip install --break-system-packages"
$PIP --ignore-installed -e /home/ubuntu/axolotl >> $DQ/build.log 2>&1 || { say "BUILD_ABORT pip-axolotl"; exit 1; }
$PIP --ignore-installed "numpy>=2.4,<2.5" pytest >> $DQ/build.log 2>&1

# private code: RAM-only tmpfs
sudo mkdir -p /mnt/ramcode
mountpoint -q /mnt/ramcode || sudo mount -t tmpfs -o size=2g tmpfs /mnt/ramcode
sudo chown ubuntu:ubuntu /mnt/ramcode
mkdir -p /mnt/ramcode/e4b-ssdtier
tar -xzf $BM4/e4b-ssdtier-4073d01.tar.gz -C /mnt/ramcode/e4b-ssdtier
$PIP --no-deps -e /mnt/ramcode/e4b-ssdtier >> $DQ/build.log 2>&1 || { say "BUILD_ABORT pip-e4b"; exit 1; }

# bnb native lib needs the pip nvidia/*/lib dirs when torch is a cu13x wheel
NVLIBS=$(python3 - <<'PY'
import glob, os, sysconfig
sp = sysconfig.get_paths()["purelib"]
print(":".join(sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))))
PY
)
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$BM4/pyext:/home/ubuntu/axolotl/tests/integrations${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\"" > $BM4/env.sh
echo "export PYTHONPATH=\"$PYTHONPATH\"" >> $BM4/env.sh
echo "export PYTHONDONTWRITEBYTECODE=1" >> $BM4/env.sh
say "BUILD_DONE torch=$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null)"

# ---------- phase 2: gates ----------
say "GATES_START"
python3 - > $DQ/gate_quantize.txt 2>&1 <<'PY'
import torch
from bitsandbytes import functional as F
x = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
q, state = F.quantize_4bit(x, quant_type="nf4")
y = F.dequantize_4bit(q, state)
err = (x - y).abs().mean().item()
assert err < 0.1, err
print("quantize smoke OK, mean abs err", err)
PY
grep -q "quantize smoke OK" $DQ/gate_quantize.txt || { say "BUILD_ABORT quantize-smoke"; cat $DQ/gate_quantize.txt | tail -3 | tee -a $SENT; exit 1; }

cd /mnt/ramcode/e4b-ssdtier
python3 -m pytest tests/test_fused.py -q > $DQ/gate_pytest_private.txt 2>&1
tail -1 $DQ/gate_pytest_private.txt | tee -a $SENT
grep -q "44 passed" $DQ/gate_pytest_private.txt || { say "BUILD_ABORT pytest-private"; exit 1; }
cd /home/ubuntu/axolotl
python3 -m pytest tests/integrations/test_expert_offload.py::TestRecompute tests/integrations/test_expert_offload.py::TestCudaParametrized -q > $DQ/gate_pytest_public.txt 2>&1
tail -1 $DQ/gate_pytest_public.txt | tee -a $SENT
grep -qE "passed" $DQ/gate_pytest_public.txt || { say "BUILD_ABORT pytest-public"; exit 1; }

# fio QD1 on the stripe + pinned H2D link
cd /home/ubuntu
fio --name=seqread --filename=/home/ubuntu/fstore/fio.tmp --rw=read --bs=1M --size=8G \
    --direct=1 --numjobs=1 --iodepth=1 --runtime=25 --time_based --output-format=json \
    > $DQ/gate_fio.json 2>&1
S_GBPS=$(python3 -c "import json;d=json.load(open('$DQ/gate_fio.json'));print(round(d['jobs'][0]['read']['bw_bytes']/1e9,2))" 2>/dev/null || echo NA)
rm -f /home/ubuntu/fstore/fio.tmp
python3 - > $DQ/gate_link.txt 2>&1 <<'PY'
import time, torch
t = torch.empty(1 << 30, dtype=torch.uint8, pin_memory=True)
d = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
for _ in range(3): d.copy_(t, non_blocking=False)
torch.cuda.synchronize()
t0 = time.monotonic()
N = 10
for _ in range(N): d.copy_(t, non_blocking=False)
torch.cuda.synchronize()
dt = time.monotonic() - t0
print(f"pinned H2D {N} GiB in {dt:.3f}s = {N*(1<<30)/dt/1e9:.2f} GB/s")
PY
L_GBPS=$(grep -oE "= [0-9.]+ GB/s" $DQ/gate_link.txt | grep -oE "[0-9.]+" || echo NA)
say "GATES_DONE fio_qd1=${S_GBPS}GB/s link=${L_GBPS}GB/s"

# ---------- phase 3: preprocess (once per seqlen, outside timed arms) ----------
say "PREP_START"
command -v axolotl >/dev/null || { say "BUILD_ABORT axolotl-not-on-PATH"; exit 1; }
for cfg in res_v_s64 res_v_s512 res_v_s2048; do
  axolotl preprocess $BM4/configs/$cfg.yaml >> $DQ/prep.log 2>&1 || { say "BUILD_ABORT preprocess-$cfg"; exit 1; }
done
say "PREP_DONE"

# ---------- phase 4: arms ----------
drop_caches() { sudo sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'; }

run_arm() {  # run_arm <log-name> <config-name> [flash] [ENV=V ...]
  local name=$1 cfg=$2; shift 2
  local flash=0
  [ "${1:-}" = "flash" ] && { flash=1; shift; }
  local extra=("$@")
  local outdir
  outdir=$(grep "^output_dir:" $BM4/configs/$cfg.yaml | awk '{print $2}')
  rm -rf "$outdir"
  local sdir
  sdir=$(grep "^fused_store_dir:" $BM4/configs/$cfg.yaml | awk '{print $2}' || true)
  [ -n "$sdir" ] && rm -rf "$sdir"
  [ "$flash" = 1 ] && drop_caches
  say "ARM_START $name"
  env "${extra[@]}" axolotl train $BM4/configs/$cfg.yaml > $DQ/out_$name.log 2>&1
  local rc=$?
  local sps
  sps=$(grep -oE "'train_steps_per_second': [0-9.]+" $DQ/out_$name.log | tail -1 | grep -oE "[0-9.]+" || echo NA)
  local nloss
  nloss=$(grep -c "'loss':" $DQ/out_$name.log || true)
  local homed
  homed=$(grep -c "homed .* MoE blocks" $DQ/out_$name.log || true)
  say "ARM_DONE $name rc=$rc sps=$sps loss_lines=$nloss homed=$homed"
}

# warmup (discarded)
run_arm warmup warmup PYTHONUNBUFFERED=1

# residue + G2 reference first (seq64 pair shares prep with the ladder)
run_arm r64_noprobe res_r_s64 PYTHONUNBUFFERED=1
run_arm r64_probe   res_r_s64 PYTHONUNBUFFERED=1 E4B_H2D_PROBE=1 E4B_H2D_PROBE_OUT=$BM4/probe/r64.jsonl
run_arm v64  res_v_s64  PYTHONUNBUFFERED=1

# ladder OFF
run_arm off_f1p0_a off_f1p0_a PYTHONUNBUFFERED=1
run_arm off_f1p0_b off_f1p0_b PYTHONUNBUFFERED=1
run_arm off_f0p5   off_f0p5   flash PYTHONUNBUFFERED=1
run_arm off_f0p0   off_f0p0   flash PYTHONUNBUFFERED=1
run_arm cent_off   cent_off   flash PYTHONUNBUFFERED=1

# ladder ON (v3 gated) — stats on every arm
run_arm on_f1p0_a  on_f1p0_a  PYTHONUNBUFFERED=1 E4B_PREFETCH_STATS=1
run_arm on_f1p0_b  on_f1p0_b  PYTHONUNBUFFERED=1 E4B_PREFETCH_STATS=1
run_arm on_f0p9375 on_f0p9375 flash PYTHONUNBUFFERED=1 E4B_PREFETCH_STATS=1
run_arm on_f0p875  on_f0p875  flash PYTHONUNBUFFERED=1 E4B_PREFETCH_STATS=1
run_arm on_f0p75   on_f0p75   flash PYTHONUNBUFFERED=1 E4B_PREFETCH_STATS=1
run_arm on_f0p5    on_f0p5    flash PYTHONUNBUFFERED=1 E4B_PREFETCH_STATS=1
run_arm on_f0p0    on_f0p0    flash PYTHONUNBUFFERED=1 E4B_PREFETCH_STATS=1
run_arm cent_on    cent_on    flash PYTHONUNBUFFERED=1 E4B_PREFETCH_STATS=1

# residue long-seq arms
run_arm v512  res_v_s512  PYTHONUNBUFFERED=1
run_arm v2048 res_v_s2048 PYTHONUNBUFFERED=1
run_arm r512_probe    res_r_s512  PYTHONUNBUFFERED=1 E4B_H2D_PROBE=1 E4B_H2D_PROBE_OUT=$BM4/probe/r512.jsonl
run_arm r2048_probe_a res_r_s2048 PYTHONUNBUFFERED=1 E4B_H2D_PROBE=1 E4B_H2D_PROBE_OUT=$BM4/probe/r2048_a.jsonl
run_arm r2048_probe_b res_r_s2048 PYTHONUNBUFFERED=1 E4B_H2D_PROBE=1 E4B_H2D_PROBE_OUT=$BM4/probe/r2048_b.jsonl

# ---------- phase 5: reduce + pack ----------
say "REDUCE_START"
python3 $BM4/reduce_bm4.py $DQ $BM4/probe > $DQ/bm4_results.json 2> $DQ/reduce.err
tail -2 $DQ/reduce.err 2>/dev/null | tee -a $SENT
VERDICT=$(python3 -c "import json;d=json.load(open('$DQ/bm4_results.json'));print(d.get('verdict_line','(no verdict)'))" 2>/dev/null || echo "reduce-failed")
say "VERDICT $VERDICT"
cd /home/ubuntu
tar czf /home/ubuntu/bm4-final.tgz --exclude=dq/prep --exclude=dq/outputs \
  -C /home/ubuntu dq bm4/configs bm4/probe
sha256sum /home/ubuntu/bm4-final.tgz | tee -a $SENT
say "BM_FINAL_DONE"
