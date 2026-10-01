# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K25's opt-in route (``E4B_NF4_GROUPED_SMALLM=1``) keeps a token's expert rows bit for bit the same whether the
token decodes alone (T = 1) or inside a B = 16 step, asserted with ``torch.equal`` on a GPU against the real kernel.

This is what lets a T = 1 instrument stand for the batched rows. K8 is decode-shaped, one token per forward, so under
the opt-in it reads K25 at T = 1. That reading covers the B = 16 rows only if each row's output depends on that row and
its expert alone, not on how many other rows share its 16-row tile. K25, like K19 and K21, has no split-K and a fixed
plan, so it should; this file holds it to that on the card it runs on.

Also pinned, on the real kernel:
- the route decision: T == 1 on the NF4 store takes the device tile table only under the opt-in;
- K25 really runs at T == 1, once for gate_up and once for down, at the route's plan;
- the output agrees with an fp32 dequant oracle (``nf4_grouped.dequant_ref``);
- under the lean glue (the default), a B = 16 step's token rows give the same bits as the expanded rows;
- the T == 1 route captures in a CUDA graph and replays bit-equal to eager on new inputs.

Random NF4 weights at Granite-3.1-3B-A800M's expert shapes: 40 experts, top-8, H = 1536, I = 512. Skipped without a
GPU or with a kernel package that lacks K25.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the row invariance is a property of the compiled kernel")

DEV = "cuda"
E, H, INTER, TOP_K, B = 40, 1536, 512, 8, 16                     # Granite-3.1-3B-A800M


def _k25():
    sm = pytest.importorskip("nf4_smallm")
    if not hasattr(sm, "gemm_nf4_grouped_smallm"):
        pytest.skip("the installed grouped-nf4-gemm lacks K25")
    return sm


_STACKS: list = []


def _stacks():
    """Quantised once per module: 40 experts x two projections is the slow part of this file."""
    if not _STACKS:
        _STACKS.append(_make_stacks())
    return _STACKS[0]


def _make_stacks():
    from nf4_pack_ref import quantize_pack_nf4
    g = torch.Generator().manual_seed(0)

    def stack(N, K):
        ps, as_ = zip(*[quantize_pack_nf4(torch.randn(N, K, generator=g) * 0.02) for _ in range(E)])
        return torch.stack(ps).to(DEV), torch.stack(as_).to(DEV)
    gu_p, gu_a = stack(2 * INTER, H)
    dn_p, dn_a = stack(H, INTER)
    return gu_p, gu_a, dn_p, dn_a


def _step(seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(B, H, generator=g) * 0.5).to(DEV, torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(B)]).view(-1).to(DEV)
    return x, ids


def _call(hr, st, T, xr, ids, x_tokens=None):
    s, gr = hr._collapsed_grouping(T, None)
    out = hr._fused_over_stack(xr, ids, *st, (2 * INTER, H, H, INTER), True, F.silu,
                               singleton_groups=s, device_grouping=gr, int4_stores=None, x_tokens=x_tokens)
    return out, (s, gr)


def _setup(monkeypatch, lean="0"):
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.setenv("E4B_NF4_GROUPED_SMALLM", "1")
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", lean)
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    return hr


@needs_cuda
def test_a_tokens_rows_are_bit_equal_alone_and_inside_a_b16_step(monkeypatch):
    sm = _k25()
    hr = _setup(monkeypatch)
    st = _stacks()
    x, ids = _step(1)
    xr = x.repeat_interleave(TOP_K, 0)
    seen = []
    k25 = sm.gemm_nf4_grouped_smallm
    monkeypatch.setattr(sm, "gemm_nf4_grouped_smallm", lambda *a, **kw: (seen.append(kw), k25(*a, **kw))[1])
    alone0, dec1 = _call(hr, st, 1, xr[:TOP_K], ids[:TOP_K])
    assert dec1 == (False, True) and len(seen) == 2, (dec1, len(seen))       # K25 ran at T == 1: gate_up + down
    assert all(kw == hr._K25_PLAN for kw in seen), seen
    full, dec16 = _call(hr, st, B, xr, ids)
    assert dec16 == (False, True)
    torch.cuda.synchronize()
    for t in (0, 7, 15):
        sl = slice(t * TOP_K, (t + 1) * TOP_K)
        alone = alone0 if t == 0 else _call(hr, st, 1, xr[sl], ids[sl])[0]
        torch.cuda.synchronize()
        assert torch.equal(alone, full[sl]), \
            f"token {t}: K25's rows differ alone vs inside the B=16 tiles (maxabs {(alone.float() - full[sl].float()).abs().max()})"


@needs_cuda
def test_t1_k25_matches_the_dequant_oracle_and_the_default_stays_singleton(monkeypatch):
    _k25()
    from nf4_grouped import dequant_ref
    from experts4bit_qlora.engines import hot_residency as hr
    st = _stacks()
    x, ids = _step(2)
    xr, ids = x[:1].repeat_interleave(TOP_K, 0), ids[:TOP_K]
    monkeypatch.delenv("E4B_NF4_GROUPED_SMALLM", raising=False)
    assert hr._collapsed_grouping(1, None) == (True, False)                   # the default route is untouched
    hr = _setup(monkeypatch)
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


@needs_cuda
def test_lean_token_rows_are_bit_equal_to_the_expanded_rows(monkeypatch):
    """The all-resident collapse hands K25 the step's token rows (x, row_token, top_k); under the lean glue gate_up
    reads them through gather_div and down scatters. Same bits as the expanded rows without the lean glue."""
    _k25()
    st = _stacks()
    x, ids = _step(4)
    rt = torch.arange(B * TOP_K, device=DEV) // TOP_K
    hr = _setup(monkeypatch, lean="0")
    want, _ = _call(hr, st, B, x.index_select(0, rt), ids)
    hr = _setup(monkeypatch, lean="1")
    got, _ = _call(hr, st, B, None, ids, x_tokens=(x, rt, TOP_K))
    torch.cuda.synchronize()
    assert torch.equal(got, want)


@needs_cuda
def test_the_t1_route_captures_in_a_cuda_graph_and_replays_bit_equal(monkeypatch):
    """The B=1 decode loop is a captured graph: T == 1 under the opt-in (the device tile table + K25) must capture
    with no host sync, and a replay on NEW inputs must equal the eager call on those inputs to the bit."""
    _k25()
    hr = _setup(monkeypatch)
    st = _stacks()
    x, ids = _step(3)
    xr = x.repeat_interleave(TOP_K, 0)
    x_s, id_s = xr[:TOP_K].clone(), ids[:TOP_K].clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):                                 # warm-up off the default stream (compiles K25)
        for _ in range(2):
            _call(hr, st, 1, x_s, id_s)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_s, _ = _call(hr, st, 1, x_s, id_s)
    for t in (5, 12):
        sl = slice(t * TOP_K, (t + 1) * TOP_K)
        x_s.copy_(xr[sl])
        id_s.copy_(ids[sl])
        g.replay()
        torch.cuda.synchronize()
        eager, _ = _call(hr, st, 1, xr[sl], ids[sl])
        torch.cuda.synchronize()
        assert torch.equal(out_s, eager), f"token {t}: the captured T == 1 route differs from eager"
