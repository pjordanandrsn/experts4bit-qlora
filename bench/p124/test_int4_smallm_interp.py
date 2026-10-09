# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K16 small-M int4-b32 GEMM: the correctness contract, device-free (Triton interpreter mode, CPU).

Guards: (1) within one bf16 output ulp of ``x @ dequant_int4_ref(packed, scales).T`` on the registered
projection shapes (scaled down in K only where the interpreter would crawl) and on the checkout shapes
(K=64, 96; N not a multiple of BLOCK_N; M < 16); (2) deterministic for a fixed config; (3) within one
bf16 ulp across SK; (4) the plan legaliser refuses what the format cannot express and never changes
arithmetic silently; (5) the counter is left zeroed (the launch is re-armed); (6) up to 64 rows (a 32- or 64-row M
tile) the same contract, and at most 16 rows K16's launch exactly -- the same grid, tile and plan, the same bits as
the 0.43.0 wrapper's launch, not merely within an ulp. Set by conftest/CI:
TRITON_INTERPRET=1. Under the interpreter the dot runs with fp32 operands (numpy has no bf16 dot); the SAME
file run compiled with TRITON_INTERPRET=0 on a GPU exercises the bf16 tensor-core arithmetic -- that run owns
the numerics claim, this one owns the contract.
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

pytest.importorskip("triton", reason="interpreter mode needs triton (Linux-only dependency)")

import int4_smallm                                                 # noqa: E402
from _triton_shim import triton                                    # noqa: E402
from int4_pack_ref import dequant_int4_ref, pack_int4_b32          # noqa: E402
from int4_smallm import gemm_int4_b32_smallm, plan_smallm, smallm_block_m, smallm_workspace  # noqa: E402


INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
DEV = "cpu" if INTERP else ("cuda" if torch.cuda.is_available() else None)
if DEV is None:
    pytest.skip("compiled mode needs a CUDA device; set TRITON_INTERPRET=1 for the CPU contract run", allow_module_level=True)


def _case(N, K, M, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator=g) * 0.7
    packed, scales = pack_int4_b32(w)
    w_deq = dequant_int4_ref(packed, scales, N, K).float()
    x = (torch.randn(M, K, generator=g) * 0.5).to(torch.bfloat16)
    # the reference is the dequant-then-GEMM path's own arithmetic: bf16-rounded weights, fp32 accumulate
    ref = x.float() @ w_deq.to(torch.bfloat16).float().t()
    return x.to(DEV), packed.contiguous().to(DEV), scales.contiguous().to(DEV), ref


def _ulp_ok(y, ref):
    """K14's metric: the largest deviation from the dequant-then-GEMM reference, relative to the largest output --
    within one bf16 ulp at that magnitude (2^-7). Per-element relative bounds are meaningless where outputs cancel."""
    err = (y.float().cpu() - ref).abs().max()
    return bool(err <= 2 ** -7 * ref.abs().max()), float(err)


@pytest.mark.parametrize("N,K,M,bn,kc,sk", [
    (64, 256, 16, 64, 128, 2),       # a 2-split, two chunks each
    (100, 128, 16, 64, 128, 1),      # N not a multiple of BLOCK_N, single chunk
    (48, 64, 5, 32, 32, 2),          # checkout K=64 -> KC 32, M < 16 (padded rows)
    (40, 96, 3, 32, 32, 3),          # checkout K=96 -> KC 32, SK 3 (odd split)
    (256, 512, 16, 64, 256, 2),      # fat chunk
    (128, 1024, 16, 128, 128, 8),    # 8-way split
])
def test_matches_dequant_reference(N, K, M, bn, kc, sk):
    x, packed, scales, ref = _case(N, K, M)
    y = gemm_int4_b32_smallm(x, packed, scales, block_n=bn, kc=kc, sk=sk)
    assert y.shape == (M, N) and y.dtype == torch.bfloat16
    ok, err = _ulp_ok(y, ref)
    assert ok, f"max |y - ref| {err} exceeds one bf16 ulp on ({N},{K},{M}) bn{bn} kc{kc} sk{sk}"


def test_deterministic_and_within_ulp_across_sk():
    x, packed, scales, ref = _case(128, 1024, 16)
    ys = {}
    for sk in (1, 2, 4, 8):
        a = gemm_int4_b32_smallm(x, packed, scales, block_n=64, kc=128, sk=sk)
        b = gemm_int4_b32_smallm(x, packed, scales, block_n=64, kc=128, sk=sk)
        assert torch.equal(a, b), f"sk={sk}: two launches of one config differ"
        ys[sk] = a.float().cpu()
    base = ys[1]
    bound = 2 ** -7 * base.abs().max()
    for sk in (2, 4, 8):
        assert (ys[sk] - base).abs().max() <= bound, f"sk={sk} differs from sk=1 by more than one bf16 ulp at the output's magnitude"


def test_counter_rearmed_and_workspace_reuse():
    x, packed, scales, ref = _case(96, 512, 16)
    ws = smallm_workspace(96, block_n=32, sk=4, device=x.device)
    y1 = gemm_int4_b32_smallm(x, packed, scales, block_n=32, kc=128, sk=4, workspace=ws)
    assert int(ws[1].abs().sum()) == 0, "the counter must be left zeroed for the next launch"
    y2 = gemm_int4_b32_smallm(x, packed, scales, block_n=32, kc=128, sk=4, workspace=ws)
    assert torch.equal(y1, y2)
    ok, _ = _ulp_ok(y1, ref)
    assert ok


def test_plan_refuses_or_legalises_never_silently_rearithmetics():
    assert plan_smallm(4096, 2048) == (64, 128, 4)
    assert plan_smallm(512, 96, kc=128, sk=4) == (64, 32, 1)          # K=96: only KC=32 divides; 3 chunks -> sk falls to 1
    assert plan_smallm(512, 96, kc=32, sk=3) == (64, 32, 3)           # an odd split that divides is kept
    with pytest.raises(ValueError):
        plan_smallm(64, 48)                                           # not a multiple of the scale block
    with pytest.raises(ValueError):
        pk, sc = pack_int4_b32(torch.randn(8, 64))
        gemm_int4_b32_smallm(torch.zeros(65, 64, dtype=torch.bfloat16, device=DEV), pk.to(DEV), sc.to(DEV))


# -- (6) up to 64 rows ------------------------------------------------------------------------------------------------
class _Spy:
    """Stands in for the jitted kernel: records each launch's grid, tile and plan, then launches the real kernel."""

    KEYS = ("BLOCK_M", "BLOCK_N", "KC", "SK", "DOT_BF16", "num_warps", "num_stages")

    def __init__(self, fn):
        self.fn, self.launches = fn, []

    def __getitem__(self, grid):
        def launch(*args, **kw):
            self.launches.append((tuple(grid), {k: kw[k] for k in self.KEYS}))
            return self.fn[grid](*args, **kw)
        return launch


def _launch_as_0_43(x, packed, scales, block_n=64, kc=128, sk=4):
    """The 0.43.0 wrapper's launch, inlined: K16's plan, a 16-row M tile, 4 warps, 2 stages, a fresh workspace."""
    M, K = x.shape
    N = packed.shape[0]
    block_n, kc, sk = plan_smallm(N, K, block_n=block_n, kc=kc, sk=sk)
    part, cnt = smallm_workspace(N, block_n=block_n, sk=sk, device=x.device)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    int4_smallm._gemm_int4_b32_smallm[(triton.cdiv(N, block_n), sk)](
        x.contiguous(), packed, scales, part, cnt, out, M, N, K=K,
        BLOCK_M=16, BLOCK_N=block_n, KC=kc, SK=sk, DOT_BF16=not INTERP, num_warps=4, num_stages=2)
    return out


def test_block_m_rule():
    assert [smallm_block_m(m) for m in (0, 1, 2, 15, 16)] == [16] * 5
    assert [smallm_block_m(m) for m in (17, 24, 31, 32)] == [32] * 4
    assert [smallm_block_m(m) for m in (33, 48, 63, 64)] == [64] * 4
    with pytest.raises(ValueError, match="M <= 64"):
        smallm_block_m(65)


@pytest.mark.parametrize("N,K,M,bn,kc,sk", [
    (64, 256, 16, 64, 128, 2),
    (100, 128, 1, 64, 128, 1),
    (48, 64, 5, 32, 32, 2),
    (128, 1024, 16, 128, 128, 8),
    (128, 256, 9, 64, 128, 4),       # the shipped plan's defaults
])
def test_at_most_16_rows_is_k16s_launch_bit_for_bit(monkeypatch, N, K, M, bn, kc, sk):
    x, packed, scales, _ = _case(N, K, M, seed=M)
    ref = _launch_as_0_43(x, packed, scales, block_n=bn, kc=kc, sk=sk)
    spy = _Spy(int4_smallm._gemm_int4_b32_smallm)
    monkeypatch.setattr(int4_smallm, "_gemm_int4_b32_smallm", spy)
    y = gemm_int4_b32_smallm(x, packed, scales, block_n=bn, kc=kc, sk=sk)
    y16 = gemm_int4_b32_smallm(x, packed, scales, block_n=bn, kc=kc, sk=sk, block_m=16)
    pbn, pkc, psk = plan_smallm(N, K, block_n=bn, kc=kc, sk=sk)
    want = ((triton.cdiv(N, pbn), psk), {"BLOCK_M": 16, "BLOCK_N": pbn, "KC": pkc, "SK": psk, "DOT_BF16": not INTERP,
                                         "num_warps": 4, "num_stages": 2})
    assert spy.launches == [want, want], spy.launches
    assert torch.equal(y, ref) and torch.equal(y16, ref), "at most 16 rows must be the 0.43.0 launch's bits"


@pytest.mark.parametrize("N,K,M,bn,kc,sk", [
    (64, 256, 17, 64, 128, 2),       # one row past K16: a 32-row tile, 15 padded rows
    (100, 128, 32, 64, 128, 1),      # a full 32-row tile, N not a multiple of BLOCK_N
    (48, 64, 33, 32, 32, 2),         # checkout K=64, a 64-row tile, 31 padded rows
    (40, 96, 40, 32, 32, 3),         # odd split
    (256, 512, 64, 64, 256, 2),      # a full 64-row tile, fat chunk
    (128, 1024, 64, 128, 128, 8),    # 8-way split
    (80, 256, 24, 32, 128, 4),       # the shipped split, a 32-row tile
])
def test_up_to_64_rows_matches_dequant_reference(monkeypatch, N, K, M, bn, kc, sk):
    x, packed, scales, ref = _case(N, K, M, seed=M)
    spy = _Spy(int4_smallm._gemm_int4_b32_smallm)
    monkeypatch.setattr(int4_smallm, "_gemm_int4_b32_smallm", spy)
    y = gemm_int4_b32_smallm(x, packed, scales, block_n=bn, kc=kc, sk=sk)
    assert [launch[1]["BLOCK_M"] for launch in spy.launches] == [smallm_block_m(M)]
    assert y.shape == (M, N) and y.dtype == torch.bfloat16
    ok, err = _ulp_ok(y, ref)
    assert ok, f"max |y - ref| {err} exceeds one bf16 ulp on ({N},{K},{M}) bn{bn} kc{kc} sk{sk}"


def test_up_to_64_rows_deterministic_and_within_ulp_across_sk_and_tiles():
    x, packed, scales, ref = _case(128, 1024, 48)
    ys = {}
    for sk in (1, 2, 4, 8):
        a = gemm_int4_b32_smallm(x, packed, scales, block_n=64, kc=128, sk=sk)
        b = gemm_int4_b32_smallm(x, packed, scales, block_n=64, kc=128, sk=sk)
        assert torch.equal(a, b), f"sk={sk}: two launches of one config differ"
        ys[sk] = a.float().cpu()
    bound = 2 ** -7 * ys[1].abs().max()
    for sk in (2, 4, 8):
        assert (ys[sk] - ys[1]).abs().max() <= bound, f"sk={sk} differs from sk=1 by more than one bf16 ulp"
    x20 = x[:20].contiguous()
    t32 = gemm_int4_b32_smallm(x20, packed, scales, block_n=64, kc=128, sk=4).float().cpu()
    t64 = gemm_int4_b32_smallm(x20, packed, scales, block_n=64, kc=128, sk=4, block_m=64).float().cpu()
    assert (t64 - t32).abs().max() <= 2 ** -7 * t32.abs().max(), "the 64-row tile differs by more than one bf16 ulp"


def test_a_64_row_workspace_serves_every_tile():
    N, K = 96, 512
    ws = smallm_workspace(N, block_m=64, block_n=32, sk=4, device=DEV)
    for M in (64, 20, 8, 33):
        x, packed, scales, ref = _case(N, K, M, seed=M)
        shared = gemm_int4_b32_smallm(x, packed, scales, block_n=32, kc=128, sk=4, workspace=ws)
        assert int(ws[1].abs().sum()) == 0, "the counter must be left zeroed for the next launch"
        fresh = gemm_int4_b32_smallm(x, packed, scales, block_n=32, kc=128, sk=4)
        assert torch.equal(shared, fresh), f"M={M}: a shared 64-row workspace moved a bit"
        assert _ulp_ok(shared, ref)[0]


def test_wide_refusals():
    pk, sc = pack_int4_b32(torch.randn(8, 64))
    pk, sc = pk.to(DEV), sc.to(DEV)
    x17 = torch.zeros(17, 64, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError, match="cannot hold"):
        gemm_int4_b32_smallm(x17, pk, sc, block_m=16)           # an explicit tile must hold every row
    with pytest.raises(ValueError, match="cannot hold"):
        gemm_int4_b32_smallm(x17, pk, sc, block_m=48)           # not a supported tile
    pk, sc = pack_int4_b32(torch.randn(8, 512))                  # K=512 keeps the 4-way split (K=64 legalises to 1)
    x17 = torch.zeros(17, 512, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError, match="does not fit"):           # a 16-row workspace cannot hold 32-row partials
        gemm_int4_b32_smallm(x17, pk.to(DEV), sc.to(DEV), workspace=smallm_workspace(8, block_m=16, sk=4, device=DEV))
