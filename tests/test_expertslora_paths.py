"""Patch coverage: ExpertsLoRA compute-path selection, mixed-dtype adapters, full save under offload.

Three behaviors introduced in v0.1.2:

1. **Path selection** (``ExpertsLoRA._use_matmul_4bit``): the frozen base projections route through
   ``bnb.matmul_4bit`` only when it helps (a backward will run) *and* is safe (never under offload —
   backward's re-dequant would read an evicted placeholder). The safety branches are CPU-testable and
   asserted here by spying on which ``Experts4bit`` method actually runs; the positive branch (matmul
   taken, and numerically identical to the dequantize path) is CUDA-only and marked accordingly.

2. **Mixed-dtype adapters** (``_lora`` / ``LoRALinear``): adapters in a different precision than the
   compute dtype (the conventional QLoRA setup keeps LoRA in fp32 over a bf16 base) used to crash with
   ``RuntimeError: expected m1 and m2 to have the same dtype``; the low-rank path now runs in the
   adapter dtype and casts the delta back. Forward *and* backward are asserted.

3. **Full save under offload**: covered by
   ``test_offload.py::test_offload_state_dict_full_save_correct_and_adapter_filter_unaffected``;
   this file adds the round-trip — a full state_dict taken from an offloaded model loads into a
   fresh, non-offloaded module and reproduces the same forward.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import Experts4bit, ExpertsLoRA, enable_expert_offload  # noqa: E402
from experts4bit_qlora import lora as lora_mod  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
_QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)
E, HID, INTER, TOP_K, N_TOK = 4, 128, 192, 2, 12


def _build(compute_dtype, adapter_dtype, seed=0):
    torch.manual_seed(seed)
    gate_up = (torch.randn(E, 2 * INTER, HID) * 0.1).to(DEVICE)
    down = (torch.randn(E, HID, INTER) * 0.1).to(DEVICE)
    try:
        base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=compute_dtype)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")
    return ExpertsLoRA(base, r=8, alpha=16, dtype=adapter_dtype).to(DEVICE)


def _inputs(dtype, requires_grad=False, seed=1):
    torch.manual_seed(seed)
    hs = torch.randn(N_TOK, HID, dtype=dtype, device=DEVICE, requires_grad=requires_grad)
    idx = torch.randint(0, E, (N_TOK, TOP_K), device=DEVICE)
    wts = torch.rand(N_TOK, TOP_K, dtype=dtype, device=DEVICE)
    return hs, idx, wts


class _Spy:
    """Wrap Experts4bit._expert_matmul / _dequantize_expert with call counters (class-level, so it
    covers every instance); restores on exit."""

    def __init__(self):
        self.matmul_calls = 0
        self.dequant_calls = 0

    def __enter__(self):
        self._orig_mm = Experts4bit._expert_matmul
        self._orig_dq = Experts4bit._dequantize_expert
        spy = self

        def mm(inner_self, *a, **k):
            spy.matmul_calls += 1
            return spy._orig_mm(inner_self, *a, **k)

        def dq(inner_self, *a, **k):
            spy.dequant_calls += 1
            return spy._orig_dq(inner_self, *a, **k)

        Experts4bit._expert_matmul = mm
        Experts4bit._dequantize_expert = dq
        return self

    def __exit__(self, *exc):
        Experts4bit._expert_matmul = self._orig_mm
        Experts4bit._dequantize_expert = self._orig_dq
        return False


# ---------------------------------------------------------------------------------------------------
# 1) Path selection — safety branches (CPU-runnable), positive branch (CUDA).
# ---------------------------------------------------------------------------------------------------
def test_gate_false_under_offload_even_if_probe_says_yes(monkeypatch):
    """The offload check must short-circuit *before* any capability probe: an offloaded layer never
    takes the matmul route, even on a bnb/device where matmul_4bit is supported."""
    lora = _build(torch.float32, torch.float32)
    monkeypatch.setattr(lora_mod, "_matmul_4bit_supported", lambda: True)
    hs, idx, wts = _inputs(torch.float32, requires_grad=True)

    enable_expert_offload(lora, DEVICE, pin=False)
    assert lora._use_matmul_4bit(hs) is False
    with _Spy() as spy:
        out = lora(hs, idx, wts)
        out.sum().backward()
    assert spy.matmul_calls == 0 and spy.dequant_calls > 0  # dequantize path only, fwd + recompute-free bwd


def test_gate_false_without_grad(monkeypatch):
    """Under no_grad nothing is saved for backward either way, so the simple dequantize path is used
    (this is also what keeps generation off bnb's single-token gemv path)."""
    lora = _build(torch.float32, torch.float32)
    monkeypatch.setattr(lora_mod, "_matmul_4bit_supported", lambda: True)
    hs, idx, wts = _inputs(torch.float32)
    with torch.no_grad():
        assert lora._use_matmul_4bit(hs) is False
        with _Spy() as spy:
            lora(hs, idx, wts)
    assert spy.matmul_calls == 0 and spy.dequant_calls > 0


@cuda
def test_gate_true_when_training_unoffloaded_and_paths_agree():
    """On CUDA with a supporting bnb: an un-offloaded training forward takes the matmul route, and its
    output matches the dequantize route within accumulation tolerance (§9a's 'numerically identical').
    If this bnb's probe fails (released <=0.49.x), the gate must stay False — asserted instead."""
    dtype = torch.bfloat16
    lora = _build(dtype, dtype)
    hs, idx, wts = _inputs(dtype, requires_grad=True)

    if not lora_mod._matmul_4bit_supported():
        assert lora._use_matmul_4bit(hs) is False
        pytest.skip("bnb matmul_4bit unsupported for [packed,1] here (<=0.49.x): gate correctly False")

    assert lora._use_matmul_4bit(hs) is True
    with _Spy() as spy:
        out_mm = lora(hs, idx, wts)
    assert spy.matmul_calls > 0 and spy.dequant_calls == 0

    with torch.no_grad():  # force the dequantize route for the comparison
        out_dq = lora(hs, idx, wts)
    torch.testing.assert_close(out_mm.detach(), out_dq, atol=3e-2, rtol=3e-2)

    out_mm.sum().backward()  # backward re-dequantizes the (resident) packed weight: must not raise
    assert lora.gate_up_lora_A.grad is not None and lora.base.gate_up_proj.grad is None


# ---------------------------------------------------------------------------------------------------
# 2) Mixed-dtype adapters (the fp32-adapters-over-bf16-base convention).
# ---------------------------------------------------------------------------------------------------
def test_fp32_adapters_over_bf16_base_forward_and_backward():
    lora = _build(torch.bfloat16, torch.float32)  # crashed pre-fix: F.linear(bf16, fp32)
    hs, idx, wts = _inputs(torch.bfloat16, requires_grad=True)

    out = lora(hs, idx, wts)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()

    # Zero-delta at init still holds exactly: B == 0 makes the delta identically zero in any dtype.
    torch.testing.assert_close(out, lora.base(hs, idx, wts))

    out.float().pow(2).sum().backward()
    for p in (lora.gate_up_lora_A, lora.gate_up_lora_B, lora.down_lora_A, lora.down_lora_B):
        assert p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()
    assert lora.base.gate_up_proj.grad is None  # frozen base still never differentiated


def test_loralinear_mixed_dtype():
    from experts4bit_qlora import LoRALinear

    torch.manual_seed(0)
    base = torch.nn.Linear(HID, INTER, bias=False, dtype=torch.bfloat16, device=DEVICE)
    ll = LoRALinear(base, r=4, alpha=8, dtype=torch.float32).to(DEVICE)  # crashed pre-fix
    x = torch.randn(3, HID, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
    y = ll(x)
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y, base(x))  # B=0 => exact base output at init
    y.float().sum().backward()
    assert ll.lora_A.grad is not None and ll.lora_A.grad.dtype == torch.float32


# ---------------------------------------------------------------------------------------------------
# 3) Full state_dict round-trip: offloaded save -> fresh (non-offloaded) module -> same forward.
# ---------------------------------------------------------------------------------------------------
def test_full_state_dict_roundtrip_from_offloaded_model():
    src = _build(torch.float32, torch.float32, seed=0)
    hs, idx, wts = _inputs(torch.float32, seed=2)
    with torch.no_grad():
        ref = src(hs, idx, wts)

    enable_expert_offload(src, DEVICE, pin=False)
    sd = src.state_dict()  # homes substituted for placeholders by the offload post-hook

    dst = _build(torch.float32, torch.float32, seed=7)  # different weights until load
    dst.load_state_dict({k: v for k, v in sd.items() if k != "base.code"}, strict=False)
    with torch.no_grad():
        got = dst(hs, idx, wts)
    torch.testing.assert_close(got, ref)  # the save carried the real experts, not placeholders
