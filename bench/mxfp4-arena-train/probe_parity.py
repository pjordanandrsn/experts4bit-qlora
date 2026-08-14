"""Measured MXFP4 parity — the numbers behind the green tick.

Reuses the suite's own fixtures so this cannot drift from what the tests assert,
and grades the fused Triton kernel against the pure-torch oracle
(`dequantize_mxfp4` then matmul) — never against another accelerated lane, which
would measure whether two fast paths round alike rather than whether either is
right.

Writes probe.json. Exit status is NOT the verdict (pytest is); this exists so the
receipt carries magnitudes instead of a boolean.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mxfp4_arena_train as T  # noqa: E402

out = {"cuda": torch.cuda.is_available()}
if torch.cuda.is_available():
    out["gpu"] = torch.cuda.get_device_name(0)

if not torch.cuda.is_available():
    json.dump(out | {"skipped": "no cuda"}, open("probe.json", "w"), indent=1)
    print("NO CUDA — nothing measured")
    sys.exit(0)

from mxfp4_grouped import gemm_mxfp4_grouped  # noqa: E402

rows = []
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    gu, dn = (t.cuda() for t in T._dense_from_source())

    for tokens in (1, 6):
        d = tmp / f"t{tokens}"
        d.mkdir()
        x, idx, w = T._routing(tokens=tokens, device="cuda")
        mod = T._staged_v4(d, device="cuda")

        k = idx.shape[1]
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        token_rows = order // k
        counts = torch.bincount(flat, minlength=T.E)
        active = torch.nonzero(counts, as_tuple=False).view(-1)
        sizes = counts[active].tolist()
        eids = active.to(torch.int32).tolist()
        branch = "gemv (all groups size 1)" if max(sizes) == 1 else "grouped gemm"
        a_cat = x.index_select(0, token_rows).contiguous()

        # --- gate_up: kernel vs oracle -------------------------------------
        got = gemm_mxfp4_grouped(a_cat,
                                 mod.gate_up_proj.view(T.E, 2 * T.INTER, T.H // 2),
                                 mod.gate_up_absmax.view(T.E, 2 * T.INTER, T.H // 32),
                                 sizes, eids)
        want = torch.empty_like(got)
        at = 0
        for e, n in zip(eids, sizes):
            want[at:at + n] = a_cat[at:at + n] @ gu[e]
            at += n
        rows.append(("gate_up kernel vs oracle", tokens, branch, T._rel_err(got, want)))

        # --- down: different K and N through the same kernel ----------------
        h = (torch.randn(a_cat.shape[0], T.INTER, device="cuda") * 0.1).to(torch.bfloat16)
        got_d = gemm_mxfp4_grouped(h.contiguous(),
                                   mod.down_proj.view(T.E, T.H, T.INTER // 2),
                                   mod.down_absmax.view(T.E, T.H, T.INTER // 32),
                                   sizes, eids)
        want_d = torch.empty_like(got_d)
        at = 0
        for e, n in zip(eids, sizes):
            want_d[at:at + n] = h[at:at + n] @ dn[e]
            at += n
        rows.append(("down kernel vs oracle", tokens, branch, T._rel_err(got_d, want_d)))

        # --- whole forward: fused lane vs reference lane vs source oracle ---
        with torch.no_grad():
            fused = mod(x, idx, w)
            reference = mod._e4b_mxfp4_arena_ref(x, idx, w)
        oracle = T._oracle_forward(gu, dn, x, idx, w, limit=T.LIMIT)
        rows.append(("forward fused vs reference", tokens, branch, T._rel_err(fused, reference)))
        rows.append(("forward fused vs source oracle", tokens, branch, T._rel_err(fused, oracle)))
        rows.append(("forward reference vs source oracle", tokens, branch,
                     T._rel_err(reference, oracle)))
        # The control that keeps the tolerances meaningful: an unclamped SwiGLU
        # must be REJECTED, or the fixture cannot tell the epilogues apart and
        # every number above is compatible with the wrong activation.
        unclamped = T._oracle_forward(gu, dn, x, idx, w, limit=0.0)
        rows.append(("CONTROL fused vs UNCLAMPED oracle (must be large)", tokens, branch,
                     T._rel_err(fused, unclamped)))

print(f"{'quantity':52s} {'tokens':>6s}  {'branch':24s} {'rel max err':>12s}")
for name, tok, branch, err in rows:
    print(f"{name:52s} {tok:6d}  {branch:24s} {err:12.3e}")

out["measurements"] = [{"quantity": n, "tokens": t, "branch": b, "rel_max_err": e}
                       for n, t, b, e in rows]
json.dump(out, open("probe.json", "w"), indent=1)
print("\nwrote probe.json")
