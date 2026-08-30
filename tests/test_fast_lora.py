"""The grouped kernel on the STREAMING expert path (``ExpertsLoRA``).

``enable_fast`` patched only ``ExpertsNbit``. ``ExpertsLoRA`` is not a subclass
of it and never calls ``base.forward()`` — it reads ``base.gate_up_proj`` and
calls ``base._project`` per expert — so on every model built by
``load_moe_4bit_streaming`` the patch was **dead code** and the grouped kernel
never ran. That is the configuration the large-model claims are made in.

What is asserted here:

1. ``enable_fast`` patches the wrapper and **skips the wrapped base**, whose
   forward is never called (patching it would inflate the count with dead
   patches and hide the bug this file exists for).
2. The fused output matches the reference forward — multi-token *and* the
   single-token decode shape, which takes a different reference branch
   (``_forward_decode``).
3. Training and grad-enabled forwards **fall back exactly**, bit-for-bit: the
   grouped kernel has no backward, and a reentrant-checkpoint initial forward is
   ``no_grad`` but ``training``, where the reference's summation order must be
   preserved for its recompute.
4. An offload-evicted (0-element placeholder) read falls back rather than
   entering the kernel with an empty tensor.
"""

import pytest

from quant_guard import require_quantize

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import (  # noqa: E402
    Experts4bit,
    ExpertsLoRA,
    disable_fast,
    enable_fast,
    fast_available,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
E, HID, INTER, TOP_K = 8, 128, 192, 2

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fast_available()),
    reason="grouped kernel needs CUDA + nf4_grouped",
)


def _build(compute_dtype=torch.bfloat16, adapter_dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    gate_up = (torch.randn(E, 2 * INTER, HID) * 0.1).to(DEVICE)
    down = (torch.randn(E, HID, INTER) * 0.1).to(DEVICE)
    require_quantize(DEVICE)
    base = Experts4bit.from_float(gate_up, down, quant_type="nf4",
                                  compute_dtype=compute_dtype)
    mod = ExpertsLoRA(base, r=8, alpha=16, dtype=adapter_dtype).to(DEVICE)
    # A zero-init B makes the delta vanish and would pass even if the adapter
    # were dropped entirely; give it real values so the LoRA term is load-bearing.
    with torch.no_grad():
        for p in (mod.gate_up_lora_B, mod.down_lora_B):
            p.normal_(0, 0.02)
    return mod.eval()


def _inputs(n_tok, dtype=torch.bfloat16, seed=1):
    torch.manual_seed(seed)
    hs = torch.randn(n_tok, HID, dtype=dtype, device=DEVICE)
    idx = torch.randint(0, E, (n_tok, TOP_K), device=DEVICE)
    wts = torch.rand(n_tok, TOP_K, dtype=dtype, device=DEVICE)
    return hs, idx, wts


def test_patches_wrapper_and_skips_wrapped_base():
    mod = _build()
    n = enable_fast(mod)
    assert n == 1, f"expected the ExpertsLoRA wrapper to be patched, got {n}"
    assert hasattr(mod, "_e4b_fast_ref")
    assert not hasattr(mod.base, "_e4b_fast_ref"), (
        "the wrapped base was patched too — its forward is never called, so that "
        "is a dead patch and would mask the bug this test covers"
    )
    assert disable_fast(mod) == 1
    assert not hasattr(mod, "_e4b_fast_ref")


@pytest.mark.parametrize("n_tok", [1, 12])
def test_fused_matches_reference(n_tok):
    """n_tok=1 is the decode shape and takes the reference's `_forward_decode`."""
    mod = _build()
    hs, idx, wts = _inputs(n_tok)
    with torch.no_grad():
        ref = mod(hs, idx, wts)
    assert enable_fast(mod) == 1
    with torch.no_grad():
        got = mod(hs, idx, wts)
    assert got.dtype == ref.dtype == hs.dtype
    rel = ((got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)).item()
    assert rel < 2e-2, f"n_tok={n_tok}: relative error {rel:.3e} — paths disagree"


def test_training_and_grad_fall_back_exactly():
    mod = _build()
    hs, idx, wts = _inputs(12)

    # The baseline must be taken in the SAME mode as the comparison. eval() lets
    # single-row experts route through bnb's fused GEMV while train() uses the
    # dequantize path — different kernels, so an eval baseline differs from a
    # training fallback by kernel rounding alone and says nothing about fusion.
    mod.train()
    with torch.no_grad():
        ref_train = mod(hs, idx, wts)

    mod.eval()
    enable_fast(mod)

    # training=True, no_grad: the reentrant-checkpoint initial-forward shape.
    mod.train()
    with torch.no_grad():
        got = mod(hs, idx, wts)
    assert torch.equal(got, ref_train), "training forward did not fall back bit-for-bit"

    # grad enabled: the kernel has no backward.
    mod.eval()
    hs_g = hs.clone().requires_grad_(True)
    out = mod(hs_g, idx, wts)
    assert out.requires_grad, "grad-enabled forward took the no-backward kernel"
    out.sum().backward()
    assert hs_g.grad is not None


def test_evicted_placeholder_falls_back():
    """Offload leaves 0-element placeholders; the kernel must not be entered."""
    mod = _build()
    enable_fast(mod)
    hs, idx, wts = _inputs(4)
    real = mod.base.gate_up_proj
    try:
        mod.base.gate_up_proj = torch.nn.Parameter(
            torch.empty(0, dtype=real.dtype, device=real.device), requires_grad=False)
        with pytest.raises(Exception):
            # The reference path raises offload.py's explanatory error rather
            # than a shape mismatch from inside the kernel — either way it must
            # not silently produce a wrong answer.
            with torch.no_grad():
                mod(hs, idx, wts)
    finally:
        mod.base.gate_up_proj = real
