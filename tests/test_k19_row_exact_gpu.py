# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K19's opt-in route (``E4B_INT4_GROUPED_SMALLM=1``) keeps a token's expert rows bit for bit the same whether the
token decodes alone (T = 1) or inside a B = 16 step, asserted with ``torch.equal`` on a GPU.

This is what lets a T = 1 instrument stand for the batched rows. K8 scores through the T = 1 loop, so lane P87's
quality gate reads K19 at T = 1. That reading covers the B = 16 rows only if each row's output depends on that row
and its expert alone, not on how many other rows share its 16-row tile. K19 has no split-K and a fixed plan
(BLOCK_N, KC), so it should; this file holds it to that on the card it runs on.

Also pinned, on the real kernel:
- the route decision: T = 1 takes the device tile table only under the opt-in;
- K19 really runs at T = 1, once for gate_up and once for down;
- the output agrees with an fp32 dequant oracle.

Random weights at Qwen3-30B-A3B's expert shapes, 128 experts, top-8. A2000 (sm_86): bit-equal on tokens 0, 5 and 15
of a B = 16 step.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the row invariance is a property of the compiled kernel")

DEV = "cuda"
E, H, INTER, TOP_K, B = 128, 2048, 768, 8, 16          # Qwen3-30B-A3B


def _k19():
    return pytest.importorskip("int4_smallm").gemm_int4_b32_grouped_smallm


def _stores():
    from int4_pack_ref import pack_int4_b32
    g = torch.Generator().manual_seed(0)

    def stack(W):
        pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(W.shape[0])])
        return torch.stack(pk).to(DEV), torch.stack(sc).to(DEV)
    gp, gs = stack(torch.randn(E, 2 * INTER, H, generator=g) * 0.02)
    dp, ds = stack(torch.randn(E, H, INTER, generator=g) * 0.02)
    return {"gu": {"packed": gp, "scales": gs, "N": 2 * INTER, "K": H},
            "dn": {"packed": dp, "scales": ds, "N": H, "K": INTER}}


def _step():
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(B, H, generator=g) * 0.5).to(DEV, torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(B)]).view(-1).to(DEV)
    return x.repeat_interleave(TOP_K, 0), ids


def _run(hr, stores, T, xr, ids):
    s, gr = hr._collapsed_grouping(T, stores)
    fg = torch.empty(0, 0, 0, dtype=torch.uint8, device=DEV)      # distinct objects: _mm names the slot by identity
    fd = torch.empty(0, 0, 0, dtype=torch.uint8, device=DEV)
    fa = torch.empty(0, 0, 0, device=DEV)
    out = hr._fused_over_stack(xr, ids, fg, fa, fd, fa, (2 * INTER, H, H, INTER), True, F.silu,
                               singleton_groups=s, device_grouping=gr, int4_stores=stores)
    torch.cuda.synchronize()
    return out, (s, gr)


@needs_cuda
def test_a_tokens_rows_are_bit_equal_alone_and_inside_a_b16_step(monkeypatch):
    k19 = _k19()
    import int4_smallm
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "1")
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    stores = _stores()
    xr, ids = _step()
    seen = []
    monkeypatch.setattr(int4_smallm, "gemm_int4_b32_grouped_smallm",
                        lambda *a, **kw: (seen.append(1), k19(*a, **kw))[1])
    alone0, dec1 = _run(hr, stores, 1, xr[:TOP_K], ids[:TOP_K])
    assert dec1 == (False, True) and len(seen) == 2, (dec1, len(seen))     # K19 ran at T == 1, gate_up + down
    full, dec16 = _run(hr, stores, B, xr, ids)
    assert dec16 == (False, True)
    for t in (0, 5, 15):
        alone = alone0 if t == 0 else _run(hr, stores, 1, xr[t * TOP_K:(t + 1) * TOP_K], ids[t * TOP_K:(t + 1) * TOP_K])[0]
        assert torch.equal(alone, full[t * TOP_K:(t + 1) * TOP_K]), \
            f"token {t}: K19's rows differ alone vs inside the B=16 tiles (maxabs {(alone.float() - full[t * TOP_K:(t + 1) * TOP_K].float()).abs().max()})"


@needs_cuda
def test_t1_k19_matches_the_dequant_oracle_and_the_default_stays_singleton(monkeypatch):
    _k19()
    from int4_pack_ref import dequant_int4_ref
    from experts4bit_qlora.engines import hot_residency as hr
    stores = _stores()
    xr, ids = _step()
    xr, ids = xr[:TOP_K], ids[:TOP_K]
    monkeypatch.delenv("E4B_INT4_GROUPED_SMALLM", raising=False)
    assert hr._collapsed_grouping(1, stores) == (True, False)               # the default route is untouched
    monkeypatch.setenv("E4B_INT4_GROUPED_SMALLM", "1")
    out, _ = _run(hr, stores, 1, xr, ids)
    ref = torch.empty(TOP_K, H)
    gu, dn = stores["gu"], stores["dn"]
    for i in range(TOP_K):
        e = int(ids[i])
        wg = dequant_int4_ref(gu["packed"][e].cpu(), gu["scales"][e].cpu(), 2 * INTER, H).float()
        wd = dequant_int4_ref(dn["packed"][e].cpu(), dn["scales"][e].cpu(), H, INTER).float()
        g_, u_ = (xr[i].float().cpu() @ wg.t()).chunk(2)
        ref[i] = (F.silu(g_) * u_) @ wd.t()
    rel = float((out.float().cpu() - ref).abs().max() / ref.abs().max())
    assert rel < 0.02, rel
