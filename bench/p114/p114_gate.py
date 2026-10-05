"""P114's correctness gate (bench/p114/PREREG-p114.md): before any energy is read, the arms must compute what they claim.

Imports the staged, unchanged `bench_energy.py` and `bench_energy_excluded.py` from the work dir and checks, on the card:
  * the dequant -> linear arm and the bnb.matmul_4bit arm compute the same projection over the same NF4 bytes, at
    decode M=1, train M=32 and prefill M=512: max |before - after| <= TOL * max |before| (TOL 5e-2: wrong weights or
    a wrong layout read O(1), a different summation order reads ~1e-2);
  * every arm runs forward, and the train workload's backward leaves a finite, non-zero input gradient;
  * the Part B layer (the fused 4-bit MoE forward) returns a finite output at batch 64.
Exit 0 and a `P114 gate: PASS` line, or exit 1 naming the first failure. The measured errors are printed either way.
"""
from __future__ import annotations

import sys

import torch

import bench_energy as be
import bench_energy_excluded as bx

TOL = 5e-2


def main() -> int:
    m, w_bf16 = be.build()

    def native(x):
        return torch.nn.functional.linear(x, w_bf16)

    def before(x):
        return torch.nn.functional.linear(
            x, m._dequantize_expert(m.gate_up_proj, m.gate_up_absmax, m._gate_up_shape, be.EXPERT, be.DTYPE))

    def after(x):
        return be._matmul4bit_proj(m, x)

    fails = []
    for M in (1, 32, 512):
        x = torch.randn(M, be.HIDDEN, dtype=be.DTYPE, device=be.DEV)
        with torch.no_grad():
            b, a, n = before(x).float(), after(x).float(), native(x).float()
        err = (b - a).abs().max().item() / max(b.abs().max().item(), 1e-30)
        ok = torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(n).all() and err <= TOL
        print(f"M={M}: max|dequant - matmul_4bit| / max|dequant| = {err:.3e} (bar {TOL}); finite={bool(ok)}")
        if not ok:
            fails.append(f"M={M} err {err:.3e}")
    for name, fn in (("native", native), ("dequant", before), ("matmul_4bit", after)):
        x = torch.randn(32, be.HIDDEN, dtype=be.DTYPE, device=be.DEV, requires_grad=True)
        fn(x).float().sum().backward()
        g = x.grad
        ok = g is not None and bool(torch.isfinite(g).all()) and g.abs().max().item() > 0
        print(f"train M=32 {name}: input grad finite and non-zero = {ok}")
        if not ok:
            fails.append(f"{name} backward")
    layer = bx.build_layer()
    idx = torch.stack([torch.randperm(bx.N_EXP, device=bx.DEV)[:8] for _ in range(64)])
    wts = torch.rand(64, 8, dtype=bx.DTYPE, device=bx.DEV)
    with torch.no_grad():
        y = layer(torch.randn(64, bx.HIDDEN, dtype=bx.DTYPE, device=bx.DEV), idx, wts)
    ok = bool(torch.isfinite(y).all()) and y.abs().max().item() > 0
    print(f"Part B layer, batch 64: output finite and non-zero = {ok}")
    if not ok:
        fails.append("Part B layer")
    torch.cuda.synchronize()
    if fails:
        print("P114 gate: FAIL -- " + "; ".join(fails))
        return 1
    print("P114 gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
