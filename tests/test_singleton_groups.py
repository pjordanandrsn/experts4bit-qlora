# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""AMENDMENT-b1d-capture: the singleton-groups dispatch must be
value-identical to the grouped dispatch — the whole sort/group/unsort
algebra collapses away without changing any row's arithmetic. Tested
with DUPLICATE ids (harder than the exact T=1 case) through a mocked
GEMM so CI needs no CUDA.

The mock is installed per-test via monkeypatch, never module-level:
`sys.modules.setdefault` would silently NOT install under a full
.[test] install where an earlier test already imported the real
`nf4_grouped` (the mock loses), and a module-level insert would LEAK
the fake into later real-kernel tests in the other order (Bugbot,
e4b#229). monkeypatch covers both directions and auto-restores."""

import sys
import types

import pytest
import torch

from experts4bit_qlora.engines.hot_residency import _fused_over_stack

SHAPES = (8, 16, 8, 8)


def _fake_gemm(x, p, a, sizes, eids):
    ids = eids if torch.is_tensor(eids) else torch.as_tensor(eids)
    ids = ids.to(torch.float32)
    per_row = ids.repeat_interleave(torch.as_tensor(sizes))
    return x * (per_row.unsqueeze(1) + 2.0)


@pytest.fixture
def mock_gemm(monkeypatch):
    mod = sys.modules.get("nf4_grouped")
    if mod is None:
        mod = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "nf4_grouped", mod)
    monkeypatch.setattr(mod, "gemm_4bit_grouped", _fake_gemm,
                        raising=False)
    return mod


def test_singleton_matches_grouped_plain(mock_gemm):
    torch.manual_seed(3)
    x = torch.randn(8, 8)
    ids = torch.tensor([3, 1, 3, 0, 2, 1, 5, 5])
    g = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=False, act_fn=torch.nn.functional.silu)
    s = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=False, act_fn=torch.nn.functional.silu,
                          singleton_groups=True)
    assert torch.equal(g, s)


def test_singleton_matches_grouped_gated(mock_gemm):
    torch.manual_seed(4)
    x = torch.randn(6, 8)
    ids = torch.tensor([2, 2, 0, 4, 4, 4])
    g = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=True, act_fn=torch.nn.functional.silu)
    s = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=True, act_fn=torch.nn.functional.silu,
                          singleton_groups=True)
    assert torch.equal(g, s)


def test_singleton_matches_grouped_gptoss_epilogue(mock_gemm):
    torch.manual_seed(5)
    # widths matter here: the mock GEMM preserves width, and the gptoss
    # branch chunks gu in half before the down GEMM -- so gu_bias is 2x
    # the dn width or the bias add fails on shape, masking the equality
    x = torch.randn(5, 16)
    ids = torch.tensor([1, 0, 1, 3, 0])
    gu_b = torch.randn(6, 16)
    dn_b = torch.randn(6, 8)
    g = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=True, act_fn=torch.nn.functional.silu,
                          gptoss=(gu_b, dn_b, 1.702, 7.0))
    s = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=True, act_fn=torch.nn.functional.silu,
                          gptoss=(gu_b, dn_b, 1.702, 7.0),
                          singleton_groups=True)
    assert torch.equal(g, s)


def test_swiglu_and_combine_kernels_are_taken_when_present(monkeypatch):
    """CPU-side plumbing: with stand-in kernels installed the gated path
    calls swiglu_rows and the collapsed combine calls combine_rows; the
    A/B env arms and a missing kernel fall back to the torch chain."""
    from experts4bit_qlora.engines import hot_residency as hr
    calls = {"swiglu": 0, "combine": 0}

    def swiglu_rows(gu):
        calls["swiglu"] += 1
        g, u = gu.float().chunk(2, dim=-1)
        return (torch.nn.functional.silu(g) * u).to(torch.bfloat16)

    def combine_rows(dn, w, k):
        calls["combine"] += 1
        T = dn.shape[0] // k
        return (dn.float() * w[:, None]).view(T, k, -1).sum(1).to(torch.bfloat16)
    stub = types.SimpleNamespace(swiglu_rows=swiglu_rows, combine_rows=combine_rows)
    monkeypatch.setitem(sys.modules, "int4_b32", stub)
    hr._SWIGLU.clear()
    hr._COMBINE.clear()
    monkeypatch.delenv("E4B_FUSE_SWIGLU", raising=False)
    monkeypatch.delenv("E4B_FUSE_COMBINE", raising=False)
    assert hr._swiglu_kernel() is swiglu_rows and hr._combine_kernel() is combine_rows
    # CPU tensors never take the kernel (the guard is device + dtype), so
    # the routing predicate is exercised directly
    gu = torch.randn(4, 16, dtype=torch.bfloat16)
    gate, up = gu.chunk(2, dim=-1)
    h = hr._swiglu_or(torch.nn.functional.silu, gu, gate, up)
    assert calls["swiglu"] == 0 and h.shape == (4, 8)
    assert hr._is_silu(torch.nn.SiLU()) and not hr._is_silu(torch.nn.functional.gelu)
    hr._SWIGLU.clear()
    hr._COMBINE.clear()
    monkeypatch.setenv("E4B_FUSE_SWIGLU", "0")
    monkeypatch.setenv("E4B_FUSE_COMBINE", "0")
    assert hr._swiglu_kernel() is None and hr._combine_kernel() is None
    hr._SWIGLU.clear()
    hr._COMBINE.clear()
    monkeypatch.delenv("E4B_FUSE_SWIGLU")
    monkeypatch.delenv("E4B_FUSE_COMBINE")
    monkeypatch.setitem(sys.modules, "int4_b32", types.SimpleNamespace())   # an older cut
    hr._SWIGLU.clear()
    hr._COMBINE.clear()
    assert hr._swiglu_kernel() is None and hr._combine_kernel() is None
    hr._SWIGLU.clear()
    hr._COMBINE.clear()
