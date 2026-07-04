"""Inference decode paths: the single-token fast-path and the no_grad 4-bit GEMV route.

Covers the two ``no_grad`` additions in :mod:`experts4bit_qlora.lora`:

1. **Decode fast-path** (``ExpertsLoRA._forward_decode``): a 1-token forward skips the expert-mask
   machinery and loops the token's ``top_k`` experts directly. Must be semantically identical to
   the mask path (same projections, fp32 accumulation; only summation order differs), must NOT
   fire in a grad-enabled forward, and must respect the ``E4B_DECODE_FASTPATH=0`` kill-switch.

2. **Inference GEMV gate** (``ExpertsLoRA._use_infer_gemv``): single-row base projections may
   route through ``bnb.matmul_4bit`` under ``no_grad`` — including under offload, where the
   training-time hazard (backward re-reading an evicted weight) cannot exist. CPU and kill-switch
   branches are asserted here; the positive branch (GEMV actually taken and numerically close to
   the dequantize route) is CUDA-only and gated on the decode-shape probe.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import Experts4bit, ExpertsLoRA, enable_expert_offload  # noqa: E402
from experts4bit_qlora import lora as lora_mod  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
_QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)
E, HID, INTER, TOP_K = 4, 128, 192, 2


def _build(compute_dtype=torch.float32, adapter_dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    gate_up = (torch.randn(E, 2 * INTER, HID) * 0.1).to(DEVICE)
    down = (torch.randn(E, HID, INTER) * 0.1).to(DEVICE)
    try:
        base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=compute_dtype)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")
    lora = ExpertsLoRA(base, r=8, alpha=16, dtype=adapter_dtype).to(DEVICE)
    with torch.no_grad():  # nonzero B so the adapter delta participates in every parity check
        lora.gate_up_lora_B.normal_(std=0.02)
        lora.down_lora_B.normal_(std=0.02)
    # Inference contract: the decode fast-path and the GEMV route require eval mode (a no_grad
    # forward of a train()-mode module is a reentrant-checkpoint shape, not inference).
    return lora.eval()


def _decode_inputs(dtype=torch.float32, seed=1, idx=None):
    torch.manual_seed(seed)
    hs = torch.randn(1, HID, dtype=dtype, device=DEVICE)
    if idx is None:
        idx = torch.randint(0, E, (1, TOP_K), device=DEVICE)
    wts = torch.rand(1, TOP_K, dtype=dtype, device=DEVICE)
    return hs, idx, wts


class _DecodeSpy:
    """Count ExpertsLoRA._forward_decode invocations (class-level); restores on exit."""

    def __init__(self):
        self.calls = 0

    def __enter__(self):
        self._orig = ExpertsLoRA._forward_decode
        spy = self

        def wrapped(inner_self, *a, **k):
            spy.calls += 1
            return spy._orig(inner_self, *a, **k)

        ExpertsLoRA._forward_decode = wrapped
        return self

    def __exit__(self, *exc):
        ExpertsLoRA._forward_decode = self._orig
        return False


# ---------------------------------------------------------------------------------------------------
# 1) Decode fast-path.
# ---------------------------------------------------------------------------------------------------
def test_decode_fastpath_taken_and_matches_mask_path(monkeypatch):
    monkeypatch.setenv("E4B_INFER_GEMV", "0")  # one compute route: compare routing machinery only
    lora = _build()
    hs, idx, wts = _decode_inputs()

    with torch.no_grad(), _DecodeSpy() as spy:
        out_fast = lora(hs, idx, wts)
        assert spy.calls == 1  # the fast-path actually ran
        monkeypatch.setenv("E4B_DECODE_FASTPATH", "0")
        out_ref = lora(hs, idx, wts)
        assert spy.calls == 1  # kill-switch respected: mask path ran instead

    # Same math, only the fp32 accumulation *order* differs (routing order vs expert-id order).
    torch.testing.assert_close(out_fast, out_ref)


def test_decode_fastpath_duplicate_expert_indices_parity(monkeypatch):
    """A duplicated expert index (not produced by topk, but legal input) must double-count the
    expert exactly like the mask path does — semantics preserved, not 'fixed'."""
    monkeypatch.setenv("E4B_INFER_GEMV", "0")  # one compute route: the mask path's duplicated
    # expert sees a 2-row input (dequant+GEMM) while the fast path stays 1-row (gemv) — pin both
    # to dequant so this compares summation order, not two CUDA kernels at default tolerances.
    lora = _build()
    hs, idx, wts = _decode_inputs(idx=torch.tensor([[1, 1]], device=DEVICE))

    with torch.no_grad():
        out_fast = lora(hs, idx, wts)
        monkeypatch.setenv("E4B_DECODE_FASTPATH", "0")
        out_ref = lora(hs, idx, wts)
    torch.testing.assert_close(out_fast, out_ref)


def test_decode_fastpath_not_taken_under_grad_or_multi_token():
    lora = _build()
    hs, idx, wts = _decode_inputs()

    with _DecodeSpy() as spy:
        # Grad-enabled single token: mask path (training semantics untouched), backward works.
        hs_g = hs.clone().requires_grad_(True)
        out = lora(hs_g, idx, wts)
        assert spy.calls == 0
        out.float().sum().backward()
        assert lora.gate_up_lora_A.grad is not None

        # Multi-token no_grad (prefill shape): mask path.
        torch.manual_seed(3)
        hs8 = torch.randn(8, HID, dtype=hs.dtype, device=DEVICE)
        idx8 = torch.randint(0, E, (8, TOP_K), device=DEVICE)
        wts8 = torch.rand(8, TOP_K, dtype=hs.dtype, device=DEVICE)
        with torch.no_grad():
            lora(hs8, idx8, wts8)
        assert spy.calls == 0


# ---------------------------------------------------------------------------------------------------
# 2) Inference GEMV gate.
# ---------------------------------------------------------------------------------------------------
def test_infer_gemv_gate_negative_branches(monkeypatch):
    lora = _build()
    hs, _, _ = _decode_inputs()

    # Grad-enabled: never (the route is an inference construct).
    monkeypatch.setattr(lora_mod, "_gemv_4bit_matches_dequant", lambda *a, **k: True)
    hs_g = hs.clone().requires_grad_(True)
    assert lora._use_infer_gemv(hs_g) is False

    with torch.no_grad():
        if DEVICE == "cpu":
            assert lora._use_infer_gemv(hs) is False  # CUDA-only kernel
        else:
            assert lora._use_infer_gemv(hs) is True
            monkeypatch.setenv("E4B_INFER_GEMV", "0")  # kill-switch
            assert lora._use_infer_gemv(hs) is False


def test_infer_gemv_gate_allowed_under_offload_no_grad(monkeypatch):
    """The offload hazard is a backward construct; under no_grad the gate must NOT veto on
    ``_offload`` (this is the deliberate refinement of the old always-False-under-offload rule —
    see the invariant note in :mod:`experts4bit_qlora.offload`)."""
    lora = _build()
    hs, _, _ = _decode_inputs()
    enable_expert_offload(lora, DEVICE, pin=False)

    monkeypatch.setattr(lora_mod, "_gemv_4bit_matches_dequant", lambda *a, **k: True)
    with torch.no_grad():
        expected = DEVICE == "cuda"  # still CUDA-gated
        assert lora._use_infer_gemv(hs) is expected
    # Grad-enabled: the GEMV route is inference-only, so it must veto regardless of offload — the
    # base then takes its recompute-in-backward projection (safe under checkpointed offload).
    hs_g = hs.clone().requires_grad_(True)
    assert lora._use_infer_gemv(hs_g) is False


@cuda
def test_decode_gemv_matches_dequant_route(monkeypatch):
    """Positive branch on real kernels: if the decode-shape probe passes, a 1-token decode via the
    GEMV route must match the dequantize route within kernel-rounding tolerance — offloaded and
    not. If the probe fails on this bitsandbytes, the gate must stay False."""
    dtype = torch.bfloat16
    lora = _build(dtype, dtype)
    hs, idx, wts = _decode_inputs(dtype)

    if not lora_mod._gemv_4bit_matches_dequant("nf4", 64, dtype):
        with torch.no_grad():
            assert lora._use_infer_gemv(hs) is False
        pytest.skip("bnb gemv_4bit incorrect for [packed, 1] decode shape here: gate correctly False")

    with torch.no_grad():
        out_gemv = lora(hs, idx, wts)
        monkeypatch.setenv("E4B_INFER_GEMV", "0")
        out_dq = lora(hs, idx, wts)
        monkeypatch.delenv("E4B_INFER_GEMV")
    torch.testing.assert_close(out_gemv, out_dq, atol=3e-2, rtol=3e-2)

    # Same parity with the experts offloaded (forward-only reads of the staged weight are safe).
    enable_expert_offload(lora, DEVICE, pin=False)
    with torch.no_grad():
        out_gemv_off = lora(hs, idx, wts)
        monkeypatch.setenv("E4B_INFER_GEMV", "0")
        out_dq_off = lora(hs, idx, wts)
    torch.testing.assert_close(out_gemv_off, out_dq_off, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(out_gemv_off, out_gemv, atol=3e-2, rtol=3e-2)


def test_decode_output_in_caller_dtype_not_compute_dtype(monkeypatch):
    """Dtype contract (parity with the base primitive): a bf16 stream through a compute_dtype=fp32
    module must come back bf16 — through the decode fast-path AND the mask path. The final cast
    used to be a no-op because ``hidden_states`` had been rebound to the compute-dtype cast."""
    lora = _build(compute_dtype=torch.float32, adapter_dtype=torch.float32)
    hs, idx, wts = _decode_inputs(dtype=torch.bfloat16)

    with torch.no_grad():
        assert lora(hs, idx, wts).dtype == torch.bfloat16  # fast-path
        monkeypatch.setenv("E4B_DECODE_FASTPATH", "0")
        assert lora(hs, idx, wts).dtype == torch.bfloat16  # mask path
