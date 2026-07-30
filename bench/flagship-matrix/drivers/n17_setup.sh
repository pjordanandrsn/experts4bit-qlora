#!/bin/bash
# Build + GATE the pod before a single cell runs.
#
# The previous second-model attempt armed the driver with NO install step and
# all ten cells failed in seconds with ModuleNotFoundError -- a whole pod's
# provisioning paid for nothing. So: install, then PROVE the environment, then
# write a sentinel the matrix script refuses to start without.
set -u
cd /workspace/n17 || exit 1
LOG=/workspace/n17/setup.log
exec > >(tee -a "$LOG") 2>&1
echo "=== n17 setup $(date -u +%FT%TZ) ==="

# This image ships TWO pythons: python3 is 3.10 but pip installs into 3.12.
# Derive the interpreter from the one that actually has torch, never assume.
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
  command -v "$cand" >/dev/null 2>&1 || continue
  if "$cand" -c "import torch" >/dev/null 2>&1; then PY="$cand"; break; fi
done
[ -n "$PY" ] || { echo "FATAL: no interpreter has torch"; exit 1; }
echo "interpreter: $PY ($($PY -c 'import torch,sys;print(sys.version.split()[0], torch.__version__)'))"

E4B_REF="${E4B_REF:-e597be6}"
echo "--- installing (e4b @ ${E4B_REF}, gnf4 from PyPI) ---"
$PY -m pip install -q --upgrade pip
$PY -m pip install -q "transformers>=5.0" "datasets>=2.14" "accelerate>=0.30" "safetensors>=0.4" "huggingface_hub>=0.23" || exit 1
$PY -m pip install -q "grouped-nf4-gemm>=0.2.6" || exit 1
$PY -m pip install -q "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@${E4B_REF}" || exit 1

echo "--- GATES (every one must pass; a failure here is cheaper than ten failed cells) ---"
$PY - <<'PYEOF' || exit 1
import sys, torch, importlib.metadata as md

def ok(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}", flush=True)
    if not cond:
        sys.exit(1)

ok("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
cap = torch.cuda.get_device_capability()
ok("sm_89 (Ada, per prereg)", cap == (8, 9), f"got sm_{cap[0]}{cap[1]}")

import bitsandbytes  # noqa: F401
import transformers
ok("transformers", True, transformers.__version__)
import transformers.models.gemma4  # noqa: F401
ok("transformers has gemma4", True)

import experts4bit_qlora
ok("experts4bit-qlora", True, md.version("experts4bit-qlora"))
ok("grouped-nf4-gemm", True, md.version("grouped-nf4-gemm"))

# The fused arm silently degrades to the reference path without this import.
from nf4_qlora import fused_grouped_lora  # noqa: F401
ok("nf4_qlora.fused_grouped_lora importable", True)

# PR #43: the post-hook must not evict inside a backward, or every gemma4 cell
# dies in its first backward. Assert the FIX WORKS, not merely that it imports.
#
# `_in_backward()` is False outside a backward -- but it is ALSO False forever if
# torch._C._current_graph_task_id is missing, because offload.py deliberately
# degrades to the pre-fix behaviour rather than crashing. So checking only the
# False branch passes whether the detector works or not: a negative control with
# no positive control, which is the exact vacuity this campaign keeps finding.
# Drive a real backward and require it to flip.
from experts4bit_qlora.offload import _in_backward
seen = []


class _Probe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        seen.append(("forward", _in_backward()))
        return x * 2

    @staticmethod
    def backward(ctx, g):
        seen.append(("backward", _in_backward()))
        return g


_Probe.apply(torch.randn(4, requires_grad=True)).sum().backward()
ok("offload fix LIVE (_in_backward flips inside a real backward)",
   seen == [("forward", False), ("backward", True)] and _in_backward() is False,
   str(seen))

# A real quantize call -- imports succeeding is not the same as bnb working.
from bitsandbytes.functional import quantize_4bit, dequantize_4bit
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
q, st = quantize_4bit(x, quant_type="nf4")
y = dequantize_4bit(q, st)
ok("real nf4 quantize/dequantize on device", y.shape == x.shape and torch.isfinite(y).all())
print("ALL GATES PASS", flush=True)
PYEOF

echo "--- verifying the five datasets against the REGISTERED sha256s ---"
$PY - <<'PYEOF' || exit 1
import hashlib, json, sys
reg = json.load(open("/workspace/n17/data/ds_manifest.json"))
bad = []
for name in ("clinical", "code", "finance", "legal", "support"):
    h = hashlib.sha256(open(f"/workspace/n17/data/ds_{name}.json", "rb").read()).hexdigest()
    want = reg[name]["sha256"]
    print(f"  {name:9s} {h[:16]} vs {want[:16]} {'OK' if h == want else 'MISMATCH'}", flush=True)
    if h != want:
        bad.append(name)
if bad:
    print(f"FATAL: dataset(s) differ from the prereg: {bad}", flush=True)
    sys.exit(1)
print("DATASETS MATCH THE PREREG", flush=True)
PYEOF

echo "--- staging the model (the wall-clock cost is download, and we bill by the second) ---"
$PY - <<'PYEOF' || exit 1
import time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download("google/gemma-4-26B-A4B", allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"])
print(f"  staged in {(time.time()-t0)/60:.1f} min at {p}", flush=True)
PYEOF

echo "SETUP OK $(date -u +%FT%TZ)" | tee /workspace/n17/SETUP_OK
