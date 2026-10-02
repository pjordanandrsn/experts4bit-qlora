# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K21's opt-in route (``E4B_MXFP4_GROUPED_SMALLM=1``) keeps a token's expert rows bit for bit the same whether the
token decodes alone (T = 1) or inside a B = 16 step, asserted with ``torch.equal`` on a GPU against the real kernel.

This is what lets a T = 1 instrument stand for the batched rows. The MXFP4 store's quality instrument (P44's
full-vocabulary KL against the dequant reference) is decode-shaped, one token per forward, so under the opt-in it reads
K21 at T = 1. That reading covers the B = 16 rows only if each row's output depends on that row and its expert alone,
not on how many other rows share its 16-row tile. K21, like K19, has no split-K and a fixed plan, so it should; this
file holds it to that on the card it runs on, through gpt-oss's own epilogue (per-expert biases, clamped GLU).

Also pinned, on the real kernel:
- the route decision: T == 1 on the MXFP4 store takes the device tile table only under the opt-in;
- K21 really runs at T == 1, once for gate_up and once for down, at the route's plan;
- the output agrees with an fp32 dequant oracle (``mxfp4_pack_ref.dequant_mxfp4``);
- the T == 1 route captures in a CUDA graph and replays bit-equal to eager on new inputs.

Random MXFP4 bytes at gpt-oss-20b's expert shapes: 32 experts, top-4, H = I = 2880 (K = 2880 is not a multiple of
KC 128, so the masked tail runs). Skipped without a GPU or with a kernel package that lacks K21's masked tail.
"""
from __future__ import annotations

import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the row invariance is a property of the compiled kernel")

DEV = "cuda"
E, H, INTER, TOP_K, B = 32, 2880, 2880, 4, 16                    # gpt-oss-20b
ALPHA, LIMIT = 1.702, 7.0


def _k21():
    mx = pytest.importorskip("mxfp4_grouped")
    from experts4bit_qlora.engines.hot_residency import _k21_has_masked_tail
    if not hasattr(mx, "gemm_mxfp4_grouped_smallm") or not _k21_has_masked_tail(mx):
        pytest.skip("the installed grouped-nf4-gemm lacks K21 with its masked K tail")
    return mx


def _store():
    g = torch.Generator().manual_seed(0)

    def stack(N, K):
        blocks = torch.randint(0, 256, (E, N, K // 2), generator=g, dtype=torch.uint8)
        scales = torch.randint(118, 122, (E, N, K // 32), generator=g, dtype=torch.uint8)
        return blocks.to(DEV), scales.to(DEV)
    gb, gs = stack(2 * INTER, H)
    db, ds = stack(H, INTER)
    st = {"kind": "mxfp4", "gu": {"blocks": gb, "scales": gs, "N": 2 * INTER, "K": H},
          "dn": {"blocks": db, "scales": ds, "N": H, "K": INTER}}
    bias = (torch.randn(E, 2 * INTER, generator=g) * 0.05).to(DEV, torch.bfloat16)
    dbias = (torch.randn(E, H, generator=g) * 0.05).to(DEV, torch.bfloat16)
    return st, (bias, dbias, ALPHA, LIMIT)


def _step(seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(B, H, generator=g) * 0.05).to(DEV, torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(B)]).view(-1).to(DEV)
    return x.repeat_interleave(TOP_K, 0), ids


def _call(hr, st, gptoss, T, xr, ids):
    s, gr = hr._collapsed_grouping(T, st)
    fg = torch.empty(0, 0, 0, dtype=torch.uint8, device=DEV)       # distinct objects: _mm names the slot by identity
    fd = torch.empty(0, 0, 0, dtype=torch.uint8, device=DEV)
    fa = torch.empty(0, 0, 0, device=DEV)
    out = hr._fused_over_stack(xr, ids, fg, fa, fd, fa, (2 * INTER, H, H, INTER), True, None, gptoss=gptoss,
                               singleton_groups=s, device_grouping=gr, int4_stores=st)
    return out, (s, gr)


def _setup(monkeypatch):
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.setenv("E4B_MXFP4_GROUPED_SMALLM", "1")
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    return hr


@needs_cuda
def test_a_tokens_rows_are_bit_equal_alone_and_inside_a_b16_step(monkeypatch):
    mx = _k21()
    hr = _setup(monkeypatch)
    st, gptoss = _store()
    xr, ids = _step(1)
    seen = []
    k21 = mx.gemm_mxfp4_grouped_smallm
    monkeypatch.setattr(mx, "gemm_mxfp4_grouped_smallm", lambda *a, **kw: (seen.append(kw), k21(*a, **kw))[1])
    alone0, dec1 = _call(hr, st, gptoss, 1, xr[:TOP_K], ids[:TOP_K])
    assert dec1 == (False, True) and len(seen) == 2, (dec1, len(seen))       # K21 ran at T == 1: gate_up + down
    assert all(kw == hr._K21_PLAN for kw in seen), seen
    full, dec16 = _call(hr, st, gptoss, B, xr, ids)
    assert dec16 == (False, True)
    torch.cuda.synchronize()
    for t in (0, 7, 15):
        alone = alone0 if t == 0 else _call(hr, st, gptoss, 1, xr[t * TOP_K:(t + 1) * TOP_K], ids[t * TOP_K:(t + 1) * TOP_K])[0]
        torch.cuda.synchronize()
        assert torch.equal(alone, full[t * TOP_K:(t + 1) * TOP_K]), \
            f"token {t}: K21's rows differ alone vs inside the B=16 tiles (maxabs {(alone.float() - full[t * TOP_K:(t + 1) * TOP_K].float()).abs().max()})"


@needs_cuda
def test_t1_k21_matches_the_dequant_oracle_and_the_default_stays_singleton(monkeypatch):
    _k21()
    from mxfp4_pack_ref import dequant_mxfp4
    from experts4bit_qlora.engines import hot_residency as hr
    st, gptoss = _store()
    xr, ids = _step(2)
    xr, ids = xr[:TOP_K], ids[:TOP_K]
    monkeypatch.delenv("E4B_MXFP4_GROUPED_SMALLM", raising=False)
    assert hr._collapsed_grouping(1, st) == (True, False)                     # the default route is untouched
    hr = _setup(monkeypatch)
    out, _ = _call(hr, st, gptoss, 1, xr, ids)
    gb, db, alpha, limit = gptoss
    ref = torch.empty(TOP_K, H)
    for i in range(TOP_K):
        e = int(ids[i])
        wg = dequant_mxfp4(st["gu"]["blocks"][e].cpu().view(2 * INTER, H // 32, 16), st["gu"]["scales"][e].cpu())
        wd = dequant_mxfp4(st["dn"]["blocks"][e].cpu().view(H, INTER // 32, 16), st["dn"]["scales"][e].cpu())
        gu = xr[i].float().cpu() @ wg.t() + gb[e].float().cpu()
        g_, u_ = gu.chunk(2)
        g_, u_ = g_.clamp(max=limit), u_.clamp(min=-limit, max=limit)
        ref[i] = ((u_ + 1) * (g_ * torch.sigmoid(g_ * alpha))) @ wd.t() + db[e].float().cpu()
    rel = float((out.float().cpu() - ref).abs().max() / ref.abs().max())
    assert rel < 0.02, rel


@needs_cuda
def test_the_t1_route_captures_in_a_cuda_graph_and_replays_bit_equal(monkeypatch):
    """The B=1 decode loop is a captured graph: T == 1 under the opt-in (the device tile table + K21) must capture
    with no host sync, and a replay on NEW inputs must equal the eager call on those inputs to the bit."""
    _k21()
    hr = _setup(monkeypatch)
    st, gptoss = _store()
    xr, ids = _step(3)
    x_s, id_s = xr[:TOP_K].clone(), ids[:TOP_K].clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):                                 # warm-up off the default stream (compiles K21)
        for _ in range(2):
            _call(hr, st, gptoss, 1, x_s, id_s)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_s, _ = _call(hr, st, gptoss, 1, x_s, id_s)
    for t in (5, 12):
        x_s.copy_(xr[t * TOP_K:(t + 1) * TOP_K])
        id_s.copy_(ids[t * TOP_K:(t + 1) * TOP_K])
        g.replay()
        torch.cuda.synchronize()
        eager, _ = _call(hr, st, gptoss, 1, xr[t * TOP_K:(t + 1) * TOP_K], ids[t * TOP_K:(t + 1) * TOP_K])
        torch.cuda.synchronize()
        assert torch.equal(out_s, eager), f"token {t}: the captured T == 1 route differs from eager"
