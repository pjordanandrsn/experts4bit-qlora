#!/usr/bin/env bash
# bench/sc1/lmdeploy/install.sh -- LMDeploy 0.18.0 (TurboMind) venv for SC1, with the import tripwire. Exit codes: 10 venv,
# 11 torch, 12 lmdeploy, 13 tripwire, 142 alarm (perl alarm, the lane's convention).
#
# usage: install.sh VENV_DIR ALARM_S
#   env: SC1_PYTHON (default python3), SC1_LMD_TORCH (torch pin pre-installed from the cu128 index; default 2.8.0 = the
#        image's; set to "" to let pip resolve lmdeploy's own `torch<=2.12.1,>=2.0.0` from PyPI)
#
# Facts (InternLM/lmdeploy @ v0.18.0, 110965c7): README "Starting from v0.13.0, the default prebuilt wheels published on
# PyPI are built against CUDA 12.8, so `pip install lmdeploy` is sufficient for typical setups including GeForce RTX 50
# series" (README.md:225); the wheel builder defaults to CUDA 12.8 (builder/manywheel/build_all_wheel.sh:7) and CMake
# appends 120a-real for nvcc >= 12.8 (CMakeLists.txt:261-262). requirements/runtime_cuda.txt pins torch<=2.12.1,>=2.0.0,
# triton<=3.7.1,>=3.4.0, tilelang==0.1.11, apache-tvm-ffi==0.1.11, flash-linear-attention>=0.4.2, peft<=0.14.0.
# Python 3.10-3.13 per the install guide (docs/en/get_started/installation.md:15). TurboMind W4A16 on sm_120 runs the
# SM80 s16816 family ("SM12.x: no native kernels; falls back to the SM80 s16816 family", kernels/gemm/arch.h:38);
# the tripwire asks the engine itself which activation dtypes it can execute for the GPTQ/AWQ g128 weight format on
# device 0 (lmdeploy/turbomind/converter.py:49-70) -- an empty set is what `_resolve_dtype` turns into
# "no executable data type for this device" (converter.py:158-160) at load time.
set -uo pipefail
VENV=${1:?venv dir}; AL=${2:-2700}
PY=${SC1_PYTHON:-python3}
TORCH=${SC1_LMD_TORCH-2.8.0}
"$PY" -m venv "$VENV" || exit 10
V="$VENV/bin/python"
echo "lmdeploy install: lmdeploy==0.18.0 torch=${TORCH:-<pip-resolved>} python=$("$V" -c 'import sys;print(sys.version.split()[0])') alarm=$AL"
if [ -n "$TORCH" ]; then
  perl -e "alarm $AL; exec @ARGV" "$V" -m pip install -q --no-input --no-cache-dir "torch==$TORCH" --index-url https://download.pytorch.org/whl/cu128 || exit 11
fi
perl -e "alarm $AL; exec @ARGV" "$V" -m pip install -q --no-input --no-cache-dir "lmdeploy==0.18.0" "huggingface_hub>=0.23" || exit 12
# ---- tripwire ----
perl -e "alarm 900; exec @ARGV" "$V" - <<'PY' || exit 13
import json, sys, subprocess
import torch
import lmdeploy
from lmdeploy.turbomind import is_available, _import_error
cc = torch.cuda.get_device_capability(0)
out = {"lmdeploy": lmdeploy.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
       "python": sys.version.split()[0], "device": torch.cuda.get_device_name(0), "capability": list(cc),
       "torch_arch_list": torch.cuda.get_arch_list(), "turbomind_available": bool(is_available()),
       "turbomind_import_error": None if _import_error is None else repr(_import_error)[:300]}
try:
    out["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
except Exception as e:  # noqa: BLE001
    out["driver"] = f"n/a: {e!r}"
try:
    import triton; out["triton"] = triton.__version__
except Exception as e:  # noqa: BLE001
    out["triton"] = f"missing: {e!r}"
exe = {}
if is_available():
    from lmdeploy.turbomind.converter import _get_executable_dtypes
    from lmdeploy.turbomind.weight_format import AWQFormat, GPTQFormat
    for name, fmt in (("gptq_g128", GPTQFormat(block_in=128)), ("awq_g128", AWQFormat(block_in=128))):
        try:
            exe[name] = sorted(_get_executable_dtypes([fmt], 0))
        except Exception as e:  # noqa: BLE001
            exe[name] = f"error: {e!r}"[:200]
out["executable_activation_dtypes"] = exe
print("SC1_TRIPWIRE lmdeploy " + json.dumps(out), flush=True)
assert lmdeploy.__version__ == "0.18.0", lmdeploy.__version__
assert is_available(), f"TurboMind binding missing: {_import_error!r}"
assert cc[0] == 12, f"not consumer Blackwell: {cc}"
assert isinstance(exe.get("gptq_g128"), list) and exe["gptq_g128"], f"no executable dtype for W4A16 g128 on this device: {exe}"
PY
echo "lmdeploy install OK"
