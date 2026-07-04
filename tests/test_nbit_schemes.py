"""ExpertsNbit storage schemes beyond 4-bit: int8/fp8 (8-bit blockwise) and bf16/fp16 (passthrough).

The fold from ``Experts4bit`` to ``ExpertsNbit`` (bitsandbytes#1965) lets the package quantize the
fused expert stack at more than one precision. These tests exercise the non-4-bit schemes through
the same public surface the 4-bit path uses — construction, forward, per-expert LoRA training, the
state_dict round-trip, and the fidelity ordering (higher bits reconstruct the float weights more
faithfully) — so a regression in any scheme is caught on CPU.

``Experts4bit`` remains the 4-bit-only subclass and must reject the 8/16-bit types.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import Experts4bit, ExpertsNbit, ExpertsLoRA  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)
E, HID, INTER, TOP_K, N_TOK = 4, 128, 192, 2, 8
EIGHT_AND_SIXTEEN = ["int8", "fp8", "bf16", "fp16"]
ALL_SCHEMES = ["nf4", "fp4", *EIGHT_AND_SIXTEEN]


def _weights(seed=0):
    torch.manual_seed(seed)
    gate_up = (torch.randn(E, 2 * INTER, HID) * 0.1).to(DEVICE)
    down = (torch.randn(E, HID, INTER) * 0.1).to(DEVICE)
    return gate_up, down


def _build(quant_type, compute_dtype=torch.float32, seed=0):
    gate_up, down = _weights(seed)
    try:
        return ExpertsNbit.from_float(gate_up, down, quant_type=quant_type, compute_dtype=compute_dtype)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes {quant_type} quantize unavailable on {DEVICE}: {e}")


def _inputs(dtype=torch.float32, seed=1):
    torch.manual_seed(seed)
    hs = torch.randn(N_TOK, HID, dtype=dtype, device=DEVICE)
    idx = torch.randint(0, E, (N_TOK, TOP_K), device=DEVICE)
    wts = torch.rand(N_TOK, TOP_K, dtype=dtype, device=DEVICE)
    return hs, idx, wts


@pytest.mark.parametrize("quant_type", EIGHT_AND_SIXTEEN)
def test_build_and_forward(quant_type):
    base = _build(quant_type)
    hs, idx, wts = _inputs()
    with torch.no_grad():
        out = base(hs, idx, wts)
    assert out.shape == hs.shape and out.dtype == hs.dtype and torch.isfinite(out).all()
    # Passthrough schemes carry no absmax buffers; blockwise schemes do.
    if quant_type in ("bf16", "fp16"):
        assert base.gate_up_absmax is None and base.down_absmax is None
    else:
        assert base.gate_up_absmax is not None


@pytest.mark.parametrize("quant_type", EIGHT_AND_SIXTEEN)
def test_lora_trains_over_nbit_base(quant_type):
    """Per-expert LoRA trains over any storage scheme; the frozen base never gets a gradient."""
    base = _build(quant_type)
    lora = ExpertsLoRA(base, r=4, alpha=8, dtype=torch.float32).to(DEVICE)
    hs, idx, wts = _inputs()
    hs = hs.clone().requires_grad_(True)
    lora(hs, idx, wts).sum().backward()
    assert lora.gate_up_lora_A.grad is not None and lora.down_lora_A.grad is not None
    assert base.gate_up_proj.grad is None  # frozen storage, any scheme


def test_fidelity_ordering():
    """Reconstruction error should fall as bits rise: bf16 passthrough ~exact < int8 < nf4.
    (fp8's e4m3 codebook is coarser than int8's dynamic map, so it is not asserted in the chain.)"""
    gate_up, _ = _weights()
    err = {}
    for q in ("nf4", "int8", "bf16"):
        base = _build(q)
        deq = base._dequantize_expert(base.gate_up_proj, base.gate_up_absmax, base._gate_up_shape, 0, torch.float32)
        err[q] = (deq - gate_up[0]).abs().mean().item()
    assert err["bf16"] < err["int8"] < err["nf4"], err


@pytest.mark.parametrize("quant_type", ALL_SCHEMES)
def test_state_dict_roundtrip(quant_type):
    """A state_dict save/load reproduces the forward for every scheme (passthrough has no absmax
    keys; blockwise/4-bit carry absmax; the code buffer is non-persistent and rebuilt on load)."""
    base = _build(quant_type, seed=0)
    hs, idx, wts = _inputs(seed=2)
    with torch.no_grad():
        ref = base(hs, idx, wts)

    dst = _build(quant_type, seed=7)  # different weights until load
    sd = {k: v for k, v in base.state_dict().items() if not k.endswith("code")}
    dst.load_state_dict(sd, strict=False)
    with torch.no_grad():
        got = dst(hs, idx, wts)
    torch.testing.assert_close(got, ref)


def test_experts4bit_rejects_8_and_16_bit():
    gate_up, down = _weights()
    for q in EIGHT_AND_SIXTEEN:
        with pytest.raises(ValueError, match="quant_type must be one of"):
            Experts4bit.from_float(gate_up, down, quant_type=q)
