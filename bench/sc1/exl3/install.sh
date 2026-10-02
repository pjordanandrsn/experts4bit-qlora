#!/usr/bin/env bash
# bench/sc1/exl3/install.sh -- ExLlamaV3 1.5.3 venv for SC1, with the import tripwire. Exit codes: 10 venv, 11 torch,
# 12 exllamav3 wheel, 13 tripwire, 142 alarm (perl alarm, the lane's convention).
#
# usage: install.sh VENV_DIR ALARM_S [FLAVOR]      FLAVOR = cu128 (default) | cu132
#   env: SC1_PYTHON (interpreter for the venv; default python3 = the image's), SC1_EXL3_TORCH (override the torch pin)
#
# Wheel matrix read from `gh api repos/turboderp-org/exllamav3/releases/tags/v1.5.3` on 2026-10-01 (62 assets):
#   exllamav3-1.5.3+cu128.torch{2.8.0,2.9.0,2.10.0}-cp3{10..13|14}-linux_x86_64.whl, +cu128.torch2.11.0 cp312-314 only
#   exllamav3-1.5.3+cu132.torch2.11.0 cp312-314 only; +cu132.torch{2.12.0,2.13.0}-cp3{10..14}-linux_x86_64.whl
#   (torch2.8.0 has cp310-313; torch2.9.0/2.10.0 have cp310-314)
# cu128 wheels need a torch+cu128 build (driver >= 570); cu132 wheels are "built with cu132 against torch==2.11.0+cu130"
# (release notes v0.0.34) and need a torch+cu130 build and a CUDA-13 driver (>= 580). The upstream README says the
# PyPI sdist JIT-compiles the extension at first import (minutes, needs nvcc) -- the release wheel is the recommended
# path (README "Prebuilt wheel - recommended"). requirements: torch>=2.6.0 only (pyproject.toml:19); NO flash-attn,
# NO xformers (both removed in v1.0.0; the attention kernels are built-in Triton, triton ships with torch on Linux).
set -uo pipefail
VENV=${1:?venv dir}; AL=${2:-2700}; FLAVOR=${3:-${SC1_EXL3_FLAVOR:-cu128}}
PY=${SC1_PYTHON:-python3}
case "$FLAVOR" in
  cu128) TORCH=${SC1_EXL3_TORCH:-2.8.0};  IDX=https://download.pytorch.org/whl/cu128 ;;
  cu132) TORCH=${SC1_EXL3_TORCH:-2.12.0}; IDX=https://download.pytorch.org/whl/cu130 ;;   # cu132.torch2.11.0 wheels are cp312-314 only; 2.12.0 covers cp310-314
  *) echo "unknown flavor $FLAVOR"; exit 2 ;;
esac
"$PY" -m venv "$VENV" || exit 10
V="$VENV/bin/python"
PYTAG=$("$V" -c 'import sys;print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
WHL="https://github.com/turboderp-org/exllamav3/releases/download/v1.5.3/exllamav3-1.5.3+${FLAVOR}.torch${TORCH}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
echo "exl3 install: flavor=$FLAVOR torch=$TORCH pytag=$PYTAG wheel=$WHL alarm=$AL"
perl -e "alarm $AL; exec @ARGV" "$V" -m pip install -q --no-input --no-cache-dir "torch==$TORCH" --index-url "$IDX" || exit 11
perl -e "alarm $AL; exec @ARGV" "$V" -m pip install -q --no-input --no-cache-dir "$WHL" "huggingface_hub>=0.23" || exit 12
# ---- tripwire: what proves the engine runs on THIS card without a model (the proving rental loads the model after) ----
perl -e "alarm 900; exec @ARGV" "$V" - <<'PY' || exit 13
import json, sys, subprocess
import torch
import exllamav3
from exllamav3 import ext as exl_ext
from exllamav3.ext import exllamav3_ext as E          # JIT-builds here if no prebuilt extension (minutes); the wheel has one
cc = torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
ver = getattr(exllamav3, "__version__", None) or exllamav3.version.__version__
caps = {k: hasattr(E, k) for k in ("exl3_gemm", "exl3_mgemm", "exl3_moe", "exl3_gemv", "exl3_gemv_int8",
                                   "BC_LinearEXL3", "BC_BlockSparseMLP", "BC_Attention")}
try:
    import triton; tv = triton.__version__
except Exception as e:  # noqa: BLE001
    tv = f"missing: {e!r}"
try:
    drv = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"], text=True).strip()
except Exception as e:  # noqa: BLE001
    drv = f"n/a: {e!r}"
live = "skipped"
try:  # best effort: one built-in kernel launch on the device (a 128-point Hadamard, the EXL3 input transform)
    x = torch.randn(1, 128, device="cuda", dtype=torch.half); y = torch.empty_like(x)
    E.had_r_128(x, y, None, None, 1.0); torch.cuda.synchronize(); live = "ok"
except Exception as e:  # noqa: BLE001
    live = f"error: {e!r}"[:200]
out = {"exllamav3": ver, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "triton": tv, "driver": drv,
       "device": torch.cuda.get_device_name(0), "capability": list(cc), "torch_arch_list": archs,
       "precompiled_extension": bool(exl_ext.is_precompiled_extension_available()), "ext_symbols": caps, "live_kernel": live,
       "python": sys.version.split()[0]}
print("SC1_TRIPWIRE exl3 " + json.dumps(out), flush=True)
sm = f"sm_{cc[0]}{cc[1]}"
assert ver == "1.5.3", ver
assert all(caps.values()), caps
assert any(a.startswith(sm) for a in archs) or any(a.startswith("compute_") for a in archs), f"torch build lacks {sm}: {archs}"
assert cc[0] == 12, f"not consumer Blackwell: {cc}"
PY
echo "exl3 install OK"
