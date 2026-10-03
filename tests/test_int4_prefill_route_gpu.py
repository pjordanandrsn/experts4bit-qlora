# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""E4B_INT4_PREFILL on a GPU, at Qwen3-30B-A3B's expert shapes (e4b#916): the real kernels, not stubs.

- ``batched`` is bit-identical to ``loop`` at prefill row counts (one 512-token and one 2048-token chunk's rows), with
  ``dequant_int4_ref`` never called. This is the matched-numerics claim the A/B rests on.
- ``k19`` takes K19 at prefill rows: the loop's operands in a different summation order. Measured against ``loop``:
  0.20 % (T = 512) and 0.43 % (T = 2048) relative Frobenius, about one bf16 ulp (A2000, 2026-10-03). The bound is
  0.8 %, well under the M-tile's ~1.3 %, so a ``k19`` that silently became the M-tile fails it.
- ``mtile`` takes the device-grouped int4 M-tile GEMM. Its output stays within 3 % of ``loop``'s: the int8
  activations move it, by about 1.3 % on these synthetic weights (A2000, 2026-10-03).

Neither route calls the reference decode. The bounds are tripwires for a broken route, not the quality gate (lane P102
gates quality on the model).

Synthetic int4 stores (random weights, ``pack_int4_b32``), skewed top-8 routing over 128 experts. Skips without CUDA or
the int4 kernels.
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
pytest.importorskip("int4_b32")
int4_pack_ref = pytest.importorskip("int4_pack_ref")

from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402

E, H, INTER, TOP = 128, 2048, 768, 8


@pytest.fixture(scope="module")
def stores():
    torch.manual_seed(0)

    def store(N, K):
        P, S = zip(*[int4_pack_ref.pack_int4_b32(torch.randn(N, K, device="cuda") * 0.02) for _ in range(E)])
        return {"packed": torch.stack(P).contiguous(), "scales": torch.stack(S).contiguous(), "N": N, "K": K}
    return {"gu": store(2 * INTER, H), "dn": store(H, INTER)}


def _rows(T, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    probs = 1.0 / (torch.arange(E, device="cuda", dtype=torch.float32) + 5.0) ** 0.8
    probs = probs[torch.randperm(E, device="cuda", generator=g)]
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, generator=g)
    ids = torch.multinomial((probs / probs.sum()).expand(T, E), TOP, generator=g).reshape(-1)
    return x.repeat_interleave(TOP, 0).contiguous(), ids


def _call(stores, xr, ids, mode, monkeypatch):
    monkeypatch.setenv("E4B_INT4_PREFILL", mode)
    singleton, device = hr._collapsed_grouping(xr.shape[0] // TOP, stores)
    sent = [torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(2)] + [torch.empty(0, device="cuda")] * 2
    return hr._fused_over_stack(xr, ids, sent[0], sent[2], sent[1], sent[3], (2 * INTER, H, H, INTER), True, F.silu,
                                singleton_groups=singleton, device_grouping=device, int4_stores=stores)


@pytest.mark.parametrize("T", [512, 2048])
def test_routes_at_prefill_rows(stores, monkeypatch, T):
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [False])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    xr, ids = _rows(T, 100 + T)
    n = [0]
    real = int4_pack_ref.dequant_int4_ref

    def counted(*a, **k):
        n[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(int4_pack_ref, "dequant_int4_ref", counted)
    y_loop = _call(stores, xr, ids, "loop", monkeypatch)
    torch.cuda.synchronize()
    assert n[0] == 2 * int(ids.unique().numel())
    n[0] = 0
    y_bat = _call(stores, xr, ids, "batched", monkeypatch)
    y_k19 = _call(stores, xr, ids, "k19", monkeypatch)
    y_mt = _call(stores, xr, ids, "mtile", monkeypatch)
    torch.cuda.synchronize()
    assert n[0] == 0
    assert torch.equal(y_loop, y_bat), "batched must be bit-identical to loop"

    def rel(y):
        return float((y.float() - y_loop.float()).norm() / y_loop.float().norm())
    assert rel(y_k19) < 0.008, rel(y_k19)
    assert rel(y_mt) < 0.03, rel(y_mt)
    print(f"T={T} rel k19={rel(y_k19):.5f} mtile={rel(y_mt):.5f}")
