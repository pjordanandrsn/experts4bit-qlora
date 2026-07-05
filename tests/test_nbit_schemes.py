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
    """Reconstruction error falls as representable precision rises, across all six schemes:

        fp16 < bf16 < int8 < fp8 < nf4 < fp4

    fp16 < bf16: passthrough rounding, 11 vs 8 significand bits — by construction. int8 < fp8: both
    are 256-entry blockwise codebooks; int8's dynamic map is denser than fp8's e4m3 grid where
    ~Gaussian weights live — pinned by MEASUREMENT on bitsandbytes' current codebooks, not by
    construction. If a future bnb reshapes them and flips this one link, demote it to documentation
    rather than forcing the assert. fp8 < nf4 (256 vs 16 entries) and nf4 < fp4 (NF4's quantiles are
    optimal for normal weights; FP4's exponent grid is not) are robust multi-x gaps."""
    gate_up, _ = _weights()
    err = {}
    for q in ALL_SCHEMES:
        base = _build(q)
        deq = base._dequantize_expert(base.gate_up_proj, base.gate_up_absmax, base._gate_up_shape, 0, torch.float32)
        err[q] = (deq - gate_up[0]).abs().mean().item()
    order = ["fp16", "bf16", "int8", "fp8", "nf4", "fp4"]
    for finer, coarser in zip(order, order[1:]):
        assert err[finer] < err[coarser], (finer, coarser, err)


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


@pytest.mark.parametrize("quant_type", ["nf4", "bf16"])
def test_state_dict_roundtrip_strict(quant_type):
    """A same-config round-trip must load under ``strict=True`` — pinning the serialization contract
    across package versions (a checkpoint saved by this build loads strictly into this build; nf4
    covers the absmax-carrying family, bf16 the passthrough family with no absmax keys)."""
    base = _build(quant_type, seed=0)
    hs, idx, wts = _inputs(seed=2)
    with torch.no_grad():
        ref = base(hs, idx, wts)

    dst = _build(quant_type, seed=7)  # different weights until load
    dst.load_state_dict(base.state_dict(), strict=True)
    with torch.no_grad():
        got = dst(hs, idx, wts)
    torch.testing.assert_close(got, ref)


def test_experts4bit_rejects_8_and_16_bit():
    gate_up, down = _weights()
    for q in EIGHT_AND_SIXTEEN:
        with pytest.raises(ValueError, match="quant_type must be one of"):
            Experts4bit.from_float(gate_up, down, quant_type=q)


def test_extra_state_saved_and_scheme_mismatch_rejected():
    """state_dict carries construction metadata, and loading a checkpoint of one scheme into a
    module built for another raises — nf4 and fp4 packed bytes are shape-identical, so without the
    metadata this load would succeed and silently decode against the wrong codebook."""
    base = _build("nf4", seed=0)
    sd = base.state_dict()
    extra = sd.get("_extra_state")
    assert isinstance(extra, dict) and extra["quant_type"] == "nf4" and extra["blocksize"] == base.blocksize

    dst = _build("fp4", seed=7)  # identical dims/blocksize: every tensor shape matches
    with pytest.raises(ValueError, match="quant_type: checkpoint='nf4' vs module='fp4'"):
        dst.load_state_dict(sd, strict=True)


def test_extra_state_dim_or_blocksize_mismatch_rejected():
    """A config mismatch is named field-by-field (here blocksize), not surfaced as a bare tensor
    shape complaint."""
    base = _build("nf4", seed=0)  # blocksize 64
    gate_up, down = _weights(seed=7)
    try:
        dst = ExpertsNbit.from_float(gate_up, down, quant_type="nf4", blocksize=32)  # 32 divides HID and INTER
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes nf4/bs32 quantize unavailable on {DEVICE}: {e}")
    with pytest.raises(ValueError, match="blocksize: checkpoint=64 vs module=32"):
        dst.load_state_dict(base.state_dict(), strict=False)


def test_legacy_checkpoint_without_extra_state_loads():
    """A checkpoint from before the metadata existed (no ``_extra_state`` key) loads under BOTH
    strict modes, bit-identically to the old behavior — the metadata is additive, not gating."""
    base = _build("nf4", seed=0)
    hs, idx, wts = _inputs(seed=2)
    with torch.no_grad():
        ref = base(hs, idx, wts)

    legacy_sd = {k: v for k, v in base.state_dict().items() if k != "_extra_state"}
    for strict in (True, False):
        dst = _build("nf4", seed=7)
        dst.load_state_dict(dict(legacy_sd), strict=strict)
        with torch.no_grad():
            torch.testing.assert_close(dst(hs, idx, wts), ref)


def test_newer_extra_state_schema_rejected():
    """Metadata from a future package version fails loudly with an upgrade hint instead of being
    half-understood."""
    base = _build("nf4", seed=0)
    sd = base.state_dict()
    sd["_extra_state"] = dict(sd["_extra_state"], schema=99)
    with pytest.raises(ValueError, match="upgrade experts4bit-qlora"):
        _build("nf4", seed=7).load_state_dict(sd, strict=False)
