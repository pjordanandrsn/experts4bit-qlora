#!/bin/bash
# gen1_arms.sh — pod-side runner for the slow-link vram-tier validation (prereg_gen1_hunt.json).
# Runs as root on a QUALIFIED (L < 8 GB/s) RunPod community pod. Arm set mirrors prereg
# acabdfc / the 3090 session: warmup, V,R,V,R, fused fv in {0,0.25,0.5,0.75,1.0}, seq64, NO flash.
set -u
cd /root
DQ=/root/dq
mkdir -p $DQ $DQ/outputs
SENT=$DQ/GEN1_SENT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $SENT; }
say "GEN1_START $(hostname)"

# gates: PyPI reachable, driver >= 580
curl -s -m 15 -o /dev/null -w "pypi=%{http_code}\n" https://pypi.org/simple/ | tee -a $SENT | grep -q "pypi=200" || { say "ABORT pypi-unreachable"; exit 1; }
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
say "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
# cu128 wheels only need >=525 (we pre-pin torch 2.11.0+cu128 below; axolotl's
# torch>=2.9.1 range accepts it, so the cu130 wheel never gets pulled). The
# quantize smoke remains the functional gate.
[ "$DRV" -ge 525 ] || { say "ABORT driver-lt-525"; exit 1; }

PYBIN=$(for q in python3.12 python3.11 python3.10 python3; do command -v $q >/dev/null 2>&1 && $q -c "import torch" >/dev/null 2>&1 && { echo $q; break; }; done)
PYBIN=${PYBIN:-python3}
say "pybin=$PYBIN"

say "BUILD_START"
if [ ! -d /root/axolotl ]; then
  git clone --branch feature/expert-store --single-branch --depth 5 \
    https://github.com/pjordanandrsn/axolotl.git /root/axolotl >> $DQ/build.log 2>&1 || { say "ABORT clone"; exit 1; }
fi
$PYBIN -m pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 >> $DQ/build.log 2>&1 || { say "ABORT pip-torch-cu128"; exit 1; }
$PYBIN -m pip install -e /root/axolotl >> $DQ/build.log 2>&1 || { say "ABORT pip-axolotl"; exit 1; }
$PYBIN -m pip install "numpy>=2.4,<2.5" pytest >> $DQ/build.log 2>&1
mkdir -p /dev/shm/e4b
tar -xzf /root/gen1/e4b-ssdtier-4073d01.tar.gz -C /dev/shm/e4b
$PYBIN -m pip install --no-deps -e /dev/shm/e4b >> $DQ/build.log 2>&1 || { say "ABORT pip-e4b"; exit 1; }

NVLIBS=$($PYBIN - <<'PY'
import glob, os, sysconfig
sp = sysconfig.get_paths()["purelib"]
print(":".join(sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))))
PY
)
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/root/gen1/pyext:/root/axolotl/tests/integrations${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
say "BUILD_DONE torch=$($PYBIN -c 'import torch;print(torch.__version__)' 2>/dev/null)"

say "GATES_START"
$PYBIN - > $DQ/gate_quantize.txt 2>&1 <<'PY'
import torch
from bitsandbytes import functional as F
x = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
q, state = F.quantize_4bit(x, quant_type="nf4")
y = F.dequantize_4bit(q, state)
err = (x - y).abs().mean().item()
assert err < 0.1, err
print("quantize smoke OK", err)
PY
grep -q "quantize smoke OK" $DQ/gate_quantize.txt || { say "ABORT quantize-smoke"; tail -3 $DQ/gate_quantize.txt | tee -a $SENT; exit 1; }
cd /dev/shm/e4b
$PYBIN -m pytest tests/test_fused.py -q -k "ThreeTier or Placement" > $DQ/gate_pytest.txt 2>&1
tail -1 $DQ/gate_pytest.txt | tee -a $SENT
grep -qE "[0-9]+ passed" $DQ/gate_pytest.txt || { say "ABORT pytest"; exit 1; }
cd /root

# the qualification number, re-measured on the SAME pod the arms run on
$PYBIN - > $DQ/gate_link.txt 2>&1 <<'PY'
import time, torch
t = torch.empty(1 << 30, dtype=torch.uint8, pin_memory=True)
d = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
for _ in range(3): d.copy_(t, non_blocking=False)
torch.cuda.synchronize()
t0 = time.monotonic(); N = 8
for _ in range(N): d.copy_(t, non_blocking=False)
torch.cuda.synchronize()
dt = time.monotonic() - t0
print(f"pinned H2D = {N*(1<<30)/dt/1e9:.2f} GB/s")
PY
L=$(grep -oE "= [0-9.]+ GB/s" $DQ/gate_link.txt | grep -oE "[0-9.]+")
say "GATES_DONE link=${L}GB/s"

# configs
$PYBIN /root/gen1/gen_configs.py /root/gen1/configs
say "PREP_START"
axolotl preprocess /root/gen1/configs/v_a.yaml >> $DQ/prep.log 2>&1
say "PREP_DONE"

run_arm() {
  local name=$1 cfg=$2
  rm -rf $DQ/outputs/$name
  say "ARM_START $name"
  axolotl train /root/gen1/configs/$cfg.yaml > $DQ/out_$name.log 2>&1
  local rc=$?
  local sps
  sps=$(grep -oE "'train_steps_per_second': '?[0-9.]+'?" $DQ/out_$name.log | tail -1 | grep -oE "[0-9.]+" | tail -1 || echo NA)
  local nloss homed
  nloss=$(grep -c "'loss':" $DQ/out_$name.log || true)
  homed=$(grep -c "homed .* MoE blocks" $DQ/out_$name.log || true)
  say "ARM_DONE $name rc=$rc sps=$sps loss_lines=$nloss homed=$homed"
}

run_arm warmup warmup
run_arm v_a v_a
run_arm r_a r_a
run_arm v_b v_b
run_arm r_b r_b
run_arm fv000 fv000
run_arm fv025 fv025
run_arm fv050 fv050
run_arm fv075 fv075
run_arm fv100 fv100

say "REDUCE_START"
$PYBIN /root/gen1/reduce_gen1.py $DQ "$L" > $DQ/gen1_results.json 2> $DQ/reduce.err
tail -1 $DQ/reduce.err | tee -a $SENT
V=$($PYBIN -c "import json;print(json.load(open('$DQ/gen1_results.json')).get('verdict_line','?'))" 2>/dev/null || echo reduce-failed)
say "VERDICT $V"
tar czf /root/gen1-final.tgz --exclude=dq/prep --exclude=dq/outputs -C /root dq gen1/configs
sha256sum /root/gen1-final.tgz | tee -a $SENT
say "GEN1_DONE"
