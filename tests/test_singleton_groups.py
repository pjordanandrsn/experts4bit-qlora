# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""AMENDMENT-b1d-capture: the singleton-groups dispatch must be
value-identical to the grouped dispatch — the whole sort/group/unsort
algebra collapses away without changing any row's arithmetic. Tested
with DUPLICATE ids (harder than the exact T=1 case) through a mocked
GEMM so CI needs no CUDA and no gnf4 kernels."""

import sys
import types

import torch


def _fake_gemm(x, p, a, sizes, eids):
    ids = eids if torch.is_tensor(eids) else torch.as_tensor(eids)
    ids = ids.to(torch.float32)
    reps = torch.as_tensor(sizes)
    per_row = ids.repeat_interleave(reps)
    return x * (per_row.unsqueeze(1) + 2.0)


sys.modules.setdefault(
    "nf4_grouped", types.SimpleNamespace(gemm_4bit_grouped=_fake_gemm))

from experts4bit_qlora.engines.hot_residency import _fused_over_stack  # noqa: E402

SHAPES = (8, 16, 8, 8)


def test_singleton_matches_grouped_plain():
    torch.manual_seed(3)
    x = torch.randn(8, 8)
    ids = torch.tensor([3, 1, 3, 0, 2, 1, 5, 5])
    g = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=False, act_fn=torch.nn.functional.silu)
    s = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=False, act_fn=torch.nn.functional.silu,
                          singleton_groups=True)
    assert torch.equal(g, s)


def test_singleton_matches_grouped_gated():
    torch.manual_seed(4)
    x = torch.randn(6, 8)
    ids = torch.tensor([2, 2, 0, 4, 4, 4])
    g = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=True, act_fn=torch.nn.functional.silu)
    s = _fused_over_stack(x, ids, None, None, None, None, SHAPES,
                          has_gate=True, act_fn=torch.nn.functional.silu,
                          singleton_groups=True)
    assert torch.equal(g, s)


def test_singleton_matches_grouped_gptoss_epilogue():
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
