# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_NF4_T1_DEVICE_GROUPING=1`` (lane P94's instrument) on a GPU, against the real served NF4 M-tile kernel.

With K25 off, T == 1 on the NF4 store takes the device tile table and ``gemm_4bit_grouped_captured`` (TF32 on the fp32
dequant), the arithmetic B=16 decode serves today. K8 is decode-shaped (one token per forward), so this is how it reads
that arithmetic. It stands for the B=16 rows only if a token's rows do not depend on the other rows sharing their
16-row tiles. The M-tile kernel has no split-K, so they should not; this file holds it to that on the card it runs on.

Pinned:
- the knob routes T == 1 to the served kernel, and the default does not;
- a token's rows are bit-equal alone (T = 1) and inside a B = 16 step;
- the T = 1 output agrees with an fp32 dequant oracle.

Random NF4 weights at Granite-3.1-3B-A800M's expert shapes (40 experts, top-8, H = 1536, I = 512).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the row invariance is a property of the compiled kernel")

DEV = "cuda"
E, H, INTER, TOP_K, B = 40, 1536, 512, 8, 16
_STACKS: list = []


def _stacks():
    if not _STACKS:
        from nf4_pack_ref import quantize_pack_nf4
        g = torch.Generator().manual_seed(0)

        def stack(N, K):
            ps, as_ = zip(*[quantize_pack_nf4(torch.randn(N, K, generator=g) * 0.02) for _ in range(E)])
            return torch.stack(ps).to(DEV), torch.stack(as_).to(DEV)
        _STACKS.append((*stack(2 * INTER, H), *stack(H, INTER)))
    return _STACKS[0]


def _step(seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(B, H, generator=g) * 0.5).to(DEV, torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(B)]).view(-1).to(DEV)
    return x.repeat_interleave(TOP_K, 0), ids


def _setup(monkeypatch):
    pytest.importorskip("nf4_grouped")
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "0")
    monkeypatch.setenv("E4B_NF4_T1_DEVICE_GROUPING", "1")
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", "0")
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    return hr


def _call(hr, st, T, xr, ids):
    s, gr = hr._collapsed_grouping(T, None)
    out = hr._fused_over_stack(xr, ids, *st, (2 * INTER, H, H, INTER), True, F.silu,
                               singleton_groups=s, device_grouping=gr, int4_stores=None)
    return out, (s, gr)


@needs_cuda
def test_t1_rows_take_the_served_m_tile_and_are_bit_equal_inside_b16(monkeypatch):
    hr = _setup(monkeypatch)
    import nf4_grouped
    seen = []
    real = nf4_grouped.gemm_4bit_grouped_captured
    monkeypatch.setattr(nf4_grouped, "gemm_4bit_grouped_captured", lambda *a, **kw: (seen.append(1), real(*a, **kw))[1])
    st = _stacks()
    xr, ids = _step(1)
    alone0, dec1 = _call(hr, st, 1, xr[:TOP_K], ids[:TOP_K])
    assert dec1 == (False, True) and len(seen) == 2, (dec1, len(seen))          # gate_up + down on the served kernel
    full, _ = _call(hr, st, B, xr, ids)
    torch.cuda.synchronize()
    for t in (0, 7, 15):
        sl = slice(t * TOP_K, (t + 1) * TOP_K)
        alone = alone0 if t == 0 else _call(hr, st, 1, xr[sl], ids[sl])[0]
        torch.cuda.synchronize()
        assert torch.equal(alone, full[sl]), f"token {t}: rows differ alone vs inside the B=16 tiles"


@needs_cuda
def test_t1_output_matches_the_oracle_and_the_default_stays_singleton(monkeypatch):
    pytest.importorskip("nf4_grouped")
    from nf4_grouped import dequant_ref
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.delenv("E4B_NF4_T1_DEVICE_GROUPING", raising=False)
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "0")
    assert hr._collapsed_grouping(1, None) == (True, False)
    hr = _setup(monkeypatch)
    st = _stacks()
    x, ids = _step(2)
    xr, ids = x[:TOP_K], ids[:TOP_K]
    out, _ = _call(hr, st, 1, xr, ids)
    gu_p, gu_a, dn_p, dn_a = (t_.cpu() for t_ in st)
    ref = torch.empty(TOP_K, H)
    for i in range(TOP_K):
        e = int(ids[i])
        gu = xr[i].float().cpu() @ dequant_ref(gu_p[e], gu_a[e], 2 * INTER, H).t()
        g_, u_ = gu.chunk(2)
        ref[i] = (F.silu(g_) * u_) @ dequant_ref(dn_p[e], dn_a[e], H, INTER).t()
    rel = float((out.float().cpu() - ref).abs().max() / ref.abs().max())
    assert rel < 0.02, rel
