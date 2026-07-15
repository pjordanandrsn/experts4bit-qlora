"""Tests for :func:`experts4bit_qlora.verify_moe_4bit`: does it correctly separate low-bit expert
stacks from the high-precision fused experts a stock ``from_pretrained(..., load_in_4bit=True)``
leaves behind (bitsandbytes#1849)?

CPU-only and cheap: modules are *constructed*, never quantized on a GPU (the helper classifies by
type/dtype, not by inspecting packed bytes), so this runs on any host with torch + bitsandbytes
importable — like ``tests/test_experts4bit_validation.py``.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import Experts4bit, ExpertsLoRA, verify_moe_4bit  # noqa: E402

E, HID, INTER = 2, 128, 192  # dims divisible by the 64 blocksize (mirrors the validation tests)


class Qwen3MoeExperts(torch.nn.Module):
    """Stand-in for a transformers v5 fused-expert module the stock 4-bit walker skipped: one fused
    3-D float ``nn.Parameter`` per projection, left in bf16. Class name matches the real one so the
    helper's name heuristic sees it."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.randn(E, 2 * INTER, HID, dtype=dtype))
        self.down_proj = torch.nn.Parameter(torch.randn(E, HID, INTER, dtype=dtype))


def _quantized_base():
    """A real Experts4bit stack, bare-constructed on CPU (no from_float quantization needed)."""
    return Experts4bit(E, HID, INTER, quant_type="nf4", device="cpu")


def _model(**modules):
    m = torch.nn.Module()
    for name, mod in modules.items():
        setattr(m, name, mod)  # registers as a submodule; named_modules() will walk it
    return m


def test_flags_unquantized_fused_experts():
    report = verify_moe_4bit(_model(block=Qwen3MoeExperts()))
    assert report["n_quantized"] == 0
    assert report["n_unquantized"] == 1
    entry = report["unquantized"][0]
    assert entry["module"].endswith("gate_up_proj")  # first 3-D float param on the stack
    assert entry["dtype"] == "bfloat16"
    assert entry["shape"] == (E, 2 * INTER, HID)


def test_recognizes_quantized_stack():
    report = verify_moe_4bit(_model(block=_quantized_base()))
    assert report["n_unquantized"] == 0
    assert report["n_quantized"] == 1
    assert report["quantized"][0]["quant_type"] == "nf4"


def test_expertslora_base_counted_once_not_as_unquantized():
    """The loader installs ``ExpertsLoRA(Experts4bit(...))``. ``named_modules()`` yields both the
    wrapper and its ``.base``: the base is the one quantized stack, and the wrapper — whose own LoRA
    adapters are 3-D fp32 Parameters — must not be miscounted as a high-precision fused stack."""
    report = verify_moe_4bit(_model(block=ExpertsLoRA(_quantized_base(), r=8, alpha=16)))
    assert report["n_quantized"] == 1  # the .base, counted exactly once
    assert report["n_unquantized"] == 0  # ExpertsLoRA excluded by name, so its 3-D LoRA params don't leak in


def test_mixed_model_and_strict_raises():
    model = _model(
        bf16=Qwen3MoeExperts(),
        q=_quantized_base(),
        lora=ExpertsLoRA(_quantized_base(), r=8, alpha=16),
    )
    report = verify_moe_4bit(model)
    assert report["n_quantized"] == 2  # q + lora.base
    assert report["n_unquantized"] == 1  # the bf16 stack
    with pytest.raises(RuntimeError, match="still high-precision"):
        verify_moe_4bit(model, strict=True)


def test_strict_passes_when_all_quantized():
    model = _model(q=_quantized_base(), lora=ExpertsLoRA(_quantized_base(), r=8, alpha=16))
    report = verify_moe_4bit(model, strict=True)  # must not raise
    assert report["n_unquantized"] == 0 and report["n_quantized"] == 2


def test_no_expert_modules_is_clean():
    """A model with no MoE experts at all: empty report, and strict does not raise."""
    report = verify_moe_4bit(_model(dense=torch.nn.Linear(HID, HID)), strict=True)
    assert report["n_quantized"] == 0 and report["n_unquantized"] == 0
