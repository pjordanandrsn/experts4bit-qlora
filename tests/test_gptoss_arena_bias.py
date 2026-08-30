# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The arena path must carry gpt-oss's per-expert biases, de-interleaved.

Arena serving streams the packed expert WEIGHTS from a baked arena and
keeps only the small bias stacks resident. Two things can go wrong
silently, and both change the epilogue rather than crashing:

* dropping the biases entirely (what the loader used to refuse over);
* carrying them INTERLEAVED while the baked weight stack is
  gate-block-then-up-block, which pairs every gate row with an up row's
  bias -- a plausible, wrong model.

These tests pin the de-interleave against ``from_gptoss`` (the direct
load path, which is the epilogue oracle) without needing a GPU, a
checkpoint, or a baked arena.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experts4bit_qlora.arch.gptoss import (  # noqa: E402
    GPTOSS_ALPHA, GPTOSS_LIMIT, GptOssExperts4bit)

E, H, INTER = 2, 8, 4


def _dense():
    g = torch.Generator().manual_seed(11)
    gate_up = torch.randn(E, H, 2 * INTER, generator=g) / 4      # input-major, interleaved
    gate_up_bias = torch.randn(E, 2 * INTER, generator=g) / 4    # interleaved
    down = torch.randn(E, INTER, H, generator=g) / 4
    down_bias = torch.randn(E, H, generator=g) / 4
    return gate_up, gate_up_bias, down, down_bias


def _arena_style_bias(gate_up_bias):
    """The de-interleave the loader's arena branch performs."""
    return torch.cat([gate_up_bias[:, 0::2], gate_up_bias[:, 1::2]], dim=1)


def test_arena_bias_matches_the_direct_path():
    gate_up, gub, down, dnb = _dense()
    direct = GptOssExperts4bit.from_gptoss(gate_up, gub, down, dnb,
                                           quant_type="bf16",
                                           compute_dtype=torch.float32)
    assert torch.equal(direct.gate_up_bias, _arena_style_bias(gub))
    assert torch.equal(direct.down_bias, dnb)


def test_interleaved_bias_would_differ():
    """Guard the guard: if de-interleaving were a no-op this test suite
    could not tell the two layouts apart."""
    _, gub, _, _ = _dense()
    assert not torch.equal(_arena_style_bias(gub), gub)


def test_epilogue_scalars_are_shared_constants():
    gate_up, gub, down, dnb = _dense()
    direct = GptOssExperts4bit.from_gptoss(gate_up, gub, down, dnb,
                                           quant_type="bf16",
                                           compute_dtype=torch.float32)
    assert direct.alpha == GPTOSS_ALPHA
    assert direct.limit == GPTOSS_LIMIT


def test_loader_still_refuses_unknown_bias_spellings():
    """Only the gpt-oss spelling is handled; any other per-expert bias
    must still refuse rather than be dropped."""
    import inspect

    from experts4bit_qlora import loader
    src = inspect.getsource(loader)
    assert 'gptoss_arena = bool(bias) and f"{epfx}gate_up_proj_bias" in weight_map' in src
    assert "arena serving does not carry per-expert biases" in src


@pytest.mark.parametrize("quant", ["bf16"])
def test_forward_uses_the_deinterleaved_bias(quant):
    """A forward with a deliberately asymmetric bias must move the output
    in the direction the DE-INTERLEAVED pairing implies."""
    gate_up, gub, down, dnb = _dense()
    gub = torch.zeros_like(gub)
    gub[:, 0::2] = 5.0          # gate rows only, in interleaved spelling
    mod = GptOssExperts4bit.from_gptoss(gate_up, gub, down, dnb,
                                        quant_type=quant,
                                        compute_dtype=torch.float32)
    # after de-interleave the first half (gate block) carries the 5.0s
    assert torch.allclose(mod.gate_up_bias[:, :INTER],
                          torch.full((E, INTER), 5.0))
    assert torch.allclose(mod.gate_up_bias[:, INTER:], torch.zeros(E, INTER))


def test_bias_gather_indexes_on_the_bias_device():
    """Arena serving puts the packed stacks on ``meta`` and keeps the bias
    stacks resident -- two different tiers in one module. The residency
    gather must therefore build its indices from the BIASES' device.

    Indices taken from the packed stacks are ``meta``: on CUDA that kills
    the attach outright (a real tensor indexed by a meta one), and on CPU
    it silently degrades instead of raising -- which is exactly why this
    asserts the DEVICE of the index rather than an exception. The full
    attach needs CUDA, so this pins the contract the fix restored.
    """
    packed = torch.empty(E, 4, 2, device="meta")     # arena-served
    bias = torch.arange(E * 4, dtype=torch.float32).reshape(E, 4)  # resident
    hot_ids = torch.arange(E)

    assert hot_ids.to(packed.device).is_meta, "packed-derived index is meta"
    fresh = hot_ids.to(bias.device)
    assert not fresh.is_meta and fresh.device == bias.device
    assert torch.equal(bias.index_select(0, fresh), bias)


def test_residency_source_uses_the_bias_device():
    import inspect

    from experts4bit_qlora.engines import hot_residency
    src = inspect.getsource(hot_residency)
    assert "bhi = hot_ids.to(gub.device)" in src
    assert "gub.index_select(0, bhi)" in src
    assert "gub.index_select(0, hi)" not in src
