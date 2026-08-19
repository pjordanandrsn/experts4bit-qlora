# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""#171: what actually differs between the gate-1 cold arms, measured.

Four placements of the SAME experts through the SAME engine, one MoE layer at
OLMoE-1B-7B geometry (H=2048, I=1024), same inputs and the same routing:

  control_dram   the experts under test in the DRAM tier  -> CPU, fp32 kernels
  control_vram   the same experts in VRAM                 -> GPU, fused kernel
  cold_gpu       the same experts on NVMe, cold_dest="gpu"
  cold_cpu       the same experts on NVMe, cold_dest="cpu"

`placement.force_cold_mass` defaults to source="dram", so a gate arm that dials
cold mass is moving experts out of control_dram. The question this answers is
which control each cold arm has to be compared against, and the answer is the
one that keeps the EXECUTION DESTINATION fixed — otherwise the comparison reads
the CPU/GPU rounding path and reports it as a cold-path defect.

Every routed row lands on an expert under test, so the arms differ in nothing
but where those experts execute. `control_dram vs control_vram` is the control
for the whole probe: it contains no cold path at all.

Run (PYTHONPATH must carry gnf4's kernel/ and its repo root)::

    PROBE_TOKENS=8 python bench/hybrid-g9/issue171/destination_gap.py
"""
import json
import os
import struct
import tempfile

import torch
from nvme_arena import bake_expert_tensors

from experts4bit_qlora import Experts4bit
from experts4bit_qlora.engines import hybrid as hy
from experts4bit_qlora.engines.nvme_experts import NF4_SEGMENTS

E, INTER, H, K = 16, 1024, 2048, 4
UNDER_TEST = [4, 5, 6, 7, 8, 9, 10, 11]        # the experts that get moved
VRAM_REST, NVME_REST = [0, 1], [12, 13, 14, 15]
DRAM_REST = [2, 3]


class _Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k, self.num_experts, self.hidden_dim = K, E, H
        self.norm_topk_prob = False
        g = torch.Generator().manual_seed(9)
        self.weight = torch.nn.Parameter(torch.randn(E, H, generator=g) * 0.3)


class _Block(torch.nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.router, self.experts = _Router(), experts


def _st_bytes(tensors):
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data); off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def _manifest(spec):
    tiers = {"vram": [], "dram": [], "nvme": []}
    for tier, ids in spec.items():
        tiers[tier] += [[0, e] for e in ids]
    return {"schema": "e4b-placement/1", "tiers": tiers,
            "masses": {"vram_frac": 0, "dram_frac": 0, "nvme_frac": 0}}


g = torch.Generator().manual_seed(11)
gate_up = (torch.randn(E, 2 * INTER, H, generator=g) * (H ** -0.5))
down = (torch.randn(E, H, INTER, generator=g) * (INTER ** -0.5))
mod = Experts4bit.from_float(gate_up.to(torch.bfloat16), down.to(torch.bfloat16),
                             has_gate=True, activation=torch.nn.functional.silu,
                             quant_type="nf4", compute_dtype=torch.bfloat16)

dt = {torch.uint8: "U8", torch.float32: "F32"}
n1, k1 = mod._gate_up_shape
n2, k2 = mod._down_shape
payload = {"nf4.gate_up_blocks": mod.gate_up_proj.view(E, n1, k1 // 2),
           "nf4.gate_up_absmax": mod.gate_up_absmax.view(E, n1, k1 // 64).float(),
           "nf4.down_blocks": mod.down_proj.view(E, n2, k2 // 2),
           "nf4.down_absmax": mod.down_absmax.view(E, n2, k2 // 64).float()}
tmp = tempfile.mkdtemp(dir=os.environ.get("PROBE_TMP", "/tmp"))
snap = os.path.join(tmp, "snap"); os.makedirs(snap)
tensors = {}
for kind, stack in payload.items():
    for e in range(E):
        t = stack[e].contiguous().cpu()
        tensors[f"model.layers.0.mlp.experts.{e}.{kind}"] = (
            tuple(t.shape), dt[t.dtype], t.numpy().tobytes())
with open(os.path.join(snap, "model.safetensors"), "wb") as fh:
    fh.write(_st_bytes(tensors))
arena = os.path.join(tmp, "m.arena")
bake_expert_tensors(snap, arena,
                    name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
                    kinds=tuple(NF4_SEGMENTS.values()), align=4096, log=lambda *a: None)

model = torch.nn.ModuleList([_Block(mod.to("cuda"))])
T = int(os.environ.get("PROBE_TOKENS", "8"))
torch.manual_seed(3)
hidden = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.5
wts = torch.rand(T, K, device="cuda", dtype=torch.bfloat16)
# every token routes k slots at the experts under test, so the arms differ
# only in where THOSE experts execute
idx = torch.stack([torch.tensor([UNDER_TEST[(t * K + s) % len(UNDER_TEST)]
                                 for t in range(T)]) for s in range(K)],
                  dim=1).cuda()

ARMS = {
    "control_dram": (_manifest({"vram": VRAM_REST, "dram": DRAM_REST + UNDER_TEST,
                                "nvme": NVME_REST}), "gpu"),
    "control_vram": (_manifest({"vram": VRAM_REST + UNDER_TEST, "dram": DRAM_REST,
                                "nvme": NVME_REST}), "gpu"),
    "cold_gpu":     (_manifest({"vram": VRAM_REST, "dram": DRAM_REST,
                                "nvme": NVME_REST + UNDER_TEST}), "gpu"),
    "cold_cpu":     (_manifest({"vram": VRAM_REST, "dram": DRAM_REST,
                                "nvme": NVME_REST + UNDER_TEST}), "cpu"),
}
out = {}
for name, (man, dest) in ARMS.items():
    hy.enable_hybrid_tier(model, arena, man, hot_rows=E, cold_dest=dest)
    try:
        with torch.no_grad():
            out[name] = model[0].experts(hidden, idx, wts).float().clone()
        st = hy.cold_stats(model)
    finally:
        hy.disable_hybrid_tier(model)
    print(f"{name:<13} cold rows gpu/cpu = {st['cold_rows_gpu']}/{st['cold_rows_cpu']}")

ref = out["control_dram"]
print(f"\n|y| rms = {ref.pow(2).mean().sqrt():.5f}   tokens={T} k={K} "
      f"H={H} I={INTER} experts-under-test={len(UNDER_TEST)}")
print(f"{'pair':<34}{'rel RMS':>12}{'max abs':>12}  bitwise")
for a, b in (("control_dram", "cold_cpu"), ("control_vram", "cold_gpu"),
             ("control_dram", "cold_gpu"), ("control_dram", "control_vram"),
             ("cold_cpu", "cold_gpu")):
    d = out[a] - out[b]
    rel = (d.pow(2).mean().sqrt() / out[a].pow(2).mean().sqrt()).item()
    print(f"{a + ' vs ' + b:<34}{rel:12.3e}{d.abs().max().item():12.3e}"
          f"  {torch.equal(out[a], out[b])}")
