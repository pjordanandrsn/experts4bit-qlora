"""ExpertsLoRA: base projection (recompute-in-backward), mixed-dtype adapters, output-dtype
contract, and full save under offload.

Behaviors covered:

1. **Base projection = recompute-in-backward** (``ExpertsNbit._project`` via
   :class:`_FrozenLinearRecomputeBackward`): the frozen base saves only its packed buffers and
   re-dequantizes in backward, so a grad-enabled forward+backward works, the frozen base never
   receives a gradient, and at LoRA init the adapted forward equals the base forward. (The old
   ``matmul_4bit`` train/no_grad *route selection* — ``_use_matmul_4bit`` / ``_expert_matmul`` —
   was removed when the package adopted the recompute-Function base; there is no route to select.)

2. **Mixed-dtype adapters** (``_lora`` / ``LoRALinear``): adapters in a different precision than the
   compute dtype (the conventional QLoRA setup keeps LoRA in fp32 over a bf16 base) run the low-rank
   path in the adapter dtype and cast the delta back. Forward *and* backward are asserted.

3. **Output-dtype contract**: the module returns the caller's dtype, not ``compute_dtype``.

4. **Full save under offload**: a full state_dict taken from an offloaded model loads into a fresh,
   non-offloaded module and reproduces the same forward.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import Experts4bit, ExpertsLoRA, enable_expert_offload  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


# ---------------------------------------------------------------------------------------------------
# 1) Base projection: recompute-in-backward, frozen base, init identity.
# ---------------------------------------------------------------------------------------------------
def test_base_recompute_backward_runs_and_leaves_base_frozen():
    """A grad-enabled forward+backward through the recompute-Function base works and never
    differentiates the frozen packed weights; at LoRA init (B=0) the adapted forward equals the
    base forward on the same route."""
    lora = _build(torch.float32, torch.float32)
    hs, idx, wts = _inputs(torch.float32, requires_grad=True)

    with torch.no_grad():
        assert torch.equal(lora(hs, idx, wts), lora.base(hs, idx, wts))  # zero-delta init identity

    out = lora(hs, idx, wts)
    out.sum().backward()  # recompute Function re-dequantizes the (resident) packed weight
    assert lora.gate_up_lora_A.grad is not None and lora.down_lora_A.grad is not None
    assert lora.base.gate_up_proj.grad is None and lora.base.down_proj.grad is None  # frozen


def test_output_dtype_follows_input_not_compute_dtype():
    """Drop-in contract: the module hands back the caller's dtype. A bf16 stream through a
    compute_dtype=fp32 module must come back bf16 (it used to come back fp32, silently upcasting
    the residual stream downstream)."""
    lora = _build(torch.float32, torch.float32)  # compute_dtype fp32
    hs, idx, wts = _inputs(torch.bfloat16)
    with torch.no_grad():
        out_lora = lora(hs, idx, wts)
        out_base = lora.base(hs, idx, wts)
    assert out_lora.dtype == torch.bfloat16 and out_base.dtype == torch.bfloat16
    assert torch.isfinite(out_lora.float()).all()


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
