# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Attention 4-bit + LoRA: projections detected by STRUCTURE (#426 / #412).

A tiny synthetic decoder stack is enough: no checkpoint download. ``v_proj``
may be absent/None (Gemma-4 k_eq_v); a missing ``k_proj`` is refused; Qwen3 /
Mixtral / Granite / OLMoE-shaped stacks still count ``4 * n_layers``. The
bias refusal still fires so gpt-oss stays REFUSED. Quantize-and-store tests
need bitsandbytes Linear4bit and skip when that backend is absent.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from torch import nn  # noqa: E402

from experts4bit_qlora.lora import (  # noqa: E402
    LoRALinear,
    add_attention_lora,
    detect_attention_projections,
    quantize_attention_projections_4bit,
)

HID = 8
DTYPE = torch.float32


class _Attn(nn.Module):
    """Minimal q/k/v/o block. ``v_proj=None`` is Gemma-4 ``attention_k_eq_v``."""

    def __init__(self, hidden=HID, *, v_proj=True, k_proj=True, bias=False):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=bias)
        self.k_proj = nn.Linear(hidden, hidden, bias=bias) if k_proj else None
        self.v_proj = nn.Linear(hidden, hidden, bias=bias) if v_proj else None
        self.o_proj = nn.Linear(hidden, hidden, bias=bias)


class _KEqVAttn(nn.Module):
    """``value_states = key_states`` after the projection (transformers Gemma-4)."""

    def __init__(self, hidden=HID):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = None
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = k
        scale = k.shape[-1] ** 0.5
        ctx = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / scale, dim=-1), v)
        return self.o_proj(ctx)


class _Stack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


def _four_proj_stack(n_layers, **kw):
    return _Stack([_Attn(**kw) for _ in range(n_layers)])


def test_k_eq_v_layers_count_three_plus_four_rest():
    """Two k_eq_v layers (no v_proj) and two full layers: 3+3+4+4 = 14."""
    model = _Stack([_Attn(v_proj=False), _Attn(v_proj=False), _Attn(), _Attn()])
    census = detect_attention_projections(model)
    assert census.expected_count == 14
    names = [name for _, name in census.candidates]
    assert names.count("v_proj") == 2
    n = add_attention_lora(model, r=4, alpha=8, dtype=DTYPE)
    assert n == census.expected_count == 14
    wrapped = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(wrapped) == 14
    # Idempotent: LoRALinear is not nn.Linear, so the detector skips wrapped blocks.
    assert add_attention_lora(model, r=4, alpha=8, dtype=DTYPE) == 0


def test_missing_k_proj_is_refused():
    model = _Stack([_Attn(k_proj=False)])
    with pytest.raises(SystemExit, match="k_proj"):
        detect_attention_projections(model)
    with pytest.raises(SystemExit, match="k_proj"):
        add_attention_lora(model, r=4, alpha=8, dtype=DTYPE)
    with pytest.raises(SystemExit, match="k_proj"):
        quantize_attention_projections_4bit(model)


@pytest.mark.parametrize(
    "n_layers",
    [2, 3, 4, 5],
    ids=["qwen3", "mixtral", "granite", "olmoe"],
)
def test_four_family_stacks_still_count_four_per_layer(n_layers):
    """Qwen3 / Mixtral / Granite / OLMoE-shaped: every layer has q/k/v/o Linear."""
    model = _four_proj_stack(n_layers)
    expected = 4 * n_layers
    census = detect_attention_projections(model)
    assert census.expected_count == expected
    assert add_attention_lora(model, r=4, alpha=8, dtype=DTYPE) == expected
    names = [name for _, name in detect_attention_projections(model).candidates]
    assert names == []  # wrapped; LoRALinear is not nn.Linear, so the detector skips


def test_bias_refusal_fires_before_conversion():
    """gpt-oss: every attention projection carries a bias; the path must REFUSE.

    Bias is checked on the snapshot before bitsandbytes is imported, so this
    arm does not need a CUDA Linear4bit backend.
    """
    model = _four_proj_stack(2, bias=True)
    census = detect_attention_projections(model, exact_linear=True)
    assert census.expected_count == 8
    with pytest.raises(SystemExit, match="TRAIN_ATTN_4BIT: .* carries a bias"):
        quantize_attention_projections_4bit(model)
    # Nothing converted: still plain Linear.
    assert all(
        type(getattr(mod, name)) is nn.Linear
        for mod, name in census.candidates
    )


def test_shared_detector_wrap_count_matches_census():
    model = _Stack([_Attn(), _Attn(v_proj=False)])
    census = detect_attention_projections(model, exact_linear=False)
    assert add_attention_lora(model, r=2, alpha=4, dtype=DTYPE) == census.expected_count == 7


def test_keqv_lora_gradients_two_consumers_of_k_proj():
    """Warden #428: k_proj output is reused as V — one module, two consumers.

    Wrapping only ``k_proj`` (q/o stay plain Linear) still has to flow a
    gradient into the adapter when the same tensor is used as K and as V.
    """
    attn = _KEqVAttn()
    attn.k_proj = LoRALinear(attn.k_proj, r=4, alpha=8, dtype=DTYPE)
    x = torch.randn(2, 3, HID, dtype=DTYPE, requires_grad=True)
    attn(x).sum().backward()
    assert attn.k_proj.lora_B.grad is not None
    assert attn.k_proj.lora_B.grad.abs().sum() > 0
    # A is zero-delta at init (B=0) so dL/dA vanishes; B is the load-bearing adapter.
    assert attn.k_proj.base.weight.grad is None


def test_keqv_add_attention_lora_wraps_qko_and_trains():
    attn = _KEqVAttn()
    n = add_attention_lora(attn, r=4, alpha=8, dtype=DTYPE)
    assert n == 3
    assert isinstance(attn.q_proj, LoRALinear)
    assert isinstance(attn.k_proj, LoRALinear)
    assert attn.v_proj is None
    assert isinstance(attn.o_proj, LoRALinear)
    x = torch.randn(2, 3, HID, dtype=DTYPE)
    attn(x).sum().backward()
    assert attn.k_proj.lora_B.grad is not None
    assert attn.k_proj.lora_B.grad.abs().sum() > 0


def test_quantize_reports_structural_expected_count():
    """Conversion count equals the detector's expected_count, not 4 * n_layers."""
    model = _Stack([_Attn(v_proj=False), _Attn()])
    census = detect_attention_projections(model, exact_linear=True)
    assert census.expected_count == 7
    try:
        n = quantize_attention_projections_4bit(model)
    except (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError) as e:
        pytest.skip(f"bitsandbytes Linear4bit unavailable on this host: {e}")
    assert n == census.expected_count == 7
    # After conversion the exact-Linear detector sees nothing left to convert.
    assert detect_attention_projections(model, exact_linear=True).expected_count == 0
    # Linear4bit is still nn.Linear, so LoRA wraps the 4-bit bases.
    assert add_attention_lora(model, r=4, alpha=8, dtype=DTYPE) == 7
