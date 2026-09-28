"""Run a K3 driver UNMODIFIED under torch's deterministic-algorithms switch, with a
read-only trace of every MoE engine call. Built for experts4bit-qlora#761 section 4
(the forward drifts from run to run). Usage:

    python det_wrap.py /workspace/k3-rel-20260928/v3/k3_run_rel.py

K3_DET=1       torch.use_deterministic_algorithms(True). CUBLAS_WORKSPACE_CONFIG has to be
               in the environment before the first cuBLAS handle; the runner sets it.
K3_SHADOW=DIR  put DIR first on sys.path, so a candidate mxfp4_pipelined.py there is
               imported instead of the installed one.

torch.utils.deterministic.fill_uninitialized_memory is forced OFF in every arm. Under
deterministic mode torch.empty NaN-fills new tensors, and the driver's 1,251 dense
placeholders are host torch.empty tensors whose pages are otherwise never touched --
108.8 GB. Filling them would OOM this NAS.

The trace hashes each MoE call's input x, router ids, router weights (before the call)
and output (after it), in call order, into moe_trace.json beside the driver's result.
It only reads tensors (a device sync per call); nothing it computes reaches the forward.
"""
import hashlib
import json
import os
import runpy
import sys

import torch

driver = sys.argv[1]
torch.utils.deterministic.fill_uninitialized_memory = False
if os.environ.get("K3_SHADOW"):
    sys.path.insert(0, os.environ["K3_SHADOW"])
if os.environ.get("K3_DET", "0") == "1":
    torch.use_deterministic_algorithms(True)

import mxfp4_pipelined  # noqa: E402  -- after the shadow, before the driver imports it


def _sha(t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


def _file_sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


TRACE = []
_forward = mxfp4_pipelined.Mxfp4PipelinedGptOss.forward


def _traced(self, hidden_states, router_indices, router_scores):
    rec = {"layer": getattr(self, "layer", None),
           "T": int(router_indices.shape[0]) if router_indices.dim() > 1 else 1,
           "x": _sha(hidden_states), "ids": _sha(router_indices), "w": _sha(router_scores)}
    out = _forward(self, hidden_states, router_indices, router_scores)
    rec["out"] = _sha(out)
    TRACE.append(rec)
    return out


mxfp4_pipelined.Mxfp4PipelinedGptOss.forward = _traced
MOD = {"file": mxfp4_pipelined.__file__, "sha256": _file_sha(mxfp4_pipelined.__file__)}
print(f"[det_wrap] deterministic={torch.are_deterministic_algorithms_enabled()} "
      f"fill_uninitialized_memory={torch.utils.deterministic.fill_uninitialized_memory} "
      f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')} "
      f"mxfp4_pipelined={MOD['file']} sha256={MOD['sha256']}", flush=True)
sys.argv = [driver]
try:
    runpy.run_path(driver, run_name="__main__")
finally:
    out_dir = os.environ.get("K3_OUT", ".")
    with open(os.path.join(out_dir, "moe_trace.json"), "w") as f:
        json.dump({"deterministic": torch.are_deterministic_algorithms_enabled(),
                   "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "shadow": os.environ.get("K3_SHADOW"), "mxfp4_pipelined": MOD,
                   "calls": TRACE}, f, indent=1)
    print(f"[det_wrap] {len(TRACE)} MoE calls traced -> {out_dir}/moe_trace.json", flush=True)
