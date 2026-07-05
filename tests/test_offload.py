"""Unit tests for expert CPU-offloading (:mod:`experts4bit_qlora.offload`).

These run on CPU-only torch (no GPU required): ``Experts4bit.from_float`` and the ``ExpertsLoRA``
dequantize forward both work on CPU, and offload changes only tensor *location*, not the math — so
an offloaded forward must be bit-for-bit identical to a non-offloaded one. When CUDA is present the
same tests exercise the real host<->device streaming path. bitsandbytes must be able to 4-bit
quantize on the test device; if it can't, the affected test skips cleanly.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from torch.utils.checkpoint import checkpoint  # noqa: E402

from experts4bit_qlora import Experts4bit, ExpertsLoRA, ExpertsNbit  # noqa: E402
from experts4bit_qlora import enable_expert_offload, offload_model_experts  # noqa: E402
from experts4bit_qlora.offload import _ExpertOffload  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
N_EXP, HIDDEN, INTER, TOP_K, N_TOK = 4, 64, 64, 2, 8


@pytest.fixture(autouse=True)
def _reset_resident_slot():
    """The single-resident-slot is class-level global state; reset it around each test for isolation."""
    _ExpertOffload._resident = None
    yield
    _ExpertOffload._resident = None

# bnb signals a missing/broken 4-bit backend in several ways depending on the build; catch them all
# so a CPU-only host without a bnb 4-bit path SKIPS cleanly (as the module docstring promises)
# rather than erroring.
_QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)


def _build_experts_lora(seed=0, quant_type="nf4"):
    torch.manual_seed(seed)
    gate_up = torch.randn(N_EXP, 2 * INTER, HIDDEN, dtype=DTYPE, device=DEVICE)
    down = torch.randn(N_EXP, HIDDEN, INTER, dtype=DTYPE, device=DEVICE)
    cls = Experts4bit if quant_type in ("nf4", "fp4") else ExpertsNbit  # mirror the loader's dispatch
    try:
        base = cls.from_float(gate_up, down, quant_type=quant_type, compute_dtype=DTYPE)
    except _QUANTIZE_UNAVAILABLE as e:  # bnb can't quantize this scheme on this device
        pytest.skip(f"bitsandbytes {quant_type} quantize unavailable on {DEVICE}: {e}")
    return ExpertsLoRA(base, r=8, alpha=16, dtype=DTYPE).to(DEVICE)


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(N_TOK, HIDDEN, dtype=DTYPE, generator=g).to(DEVICE)
    top_k_index = torch.randint(0, N_EXP, (N_TOK, TOP_K), generator=g).to(DEVICE)
    top_k_weights = torch.rand(N_TOK, TOP_K, dtype=DTYPE, generator=g).to(DEVICE)
    return hidden_states, top_k_index, top_k_weights


@pytest.mark.parametrize("quant_type", ["nf4", "int8", "bf16"])
@pytest.mark.parametrize("pin", [True, False])
def test_offload_forward_is_math_identical(pin, quant_type):
    """Offload swaps where the experts live, not the computation: output must be unchanged, and
    the evict->reload cycle must be idempotent across repeated forwards. Covers both the pinned and
    the pageable (``OFFLOAD_PIN=0``) home paths, across the three storage families (4-bit packed,
    8-bit blockwise, 16-bit passthrough — the passthrough base carries NO absmax buffers, which the
    offload handle must tolerate rather than crash on)."""
    lora = _build_experts_lora(quant_type=quant_type)
    hs, idx, w = _inputs()
    with torch.no_grad():
        ref = lora(hs, idx, w)

    enable_expert_offload(lora, DEVICE, pin=pin)
    if quant_type == "bf16":
        # Passthrough: only the two projections have homes; the absmax buffers stay registered as
        # None — offload must preserve their None-ness (a placeholder here would break the
        # `absmax is None` passthrough test other code relies on).
        assert set(lora._offload.home) == {"gate_up_proj", "down_proj"}
        assert lora.base.gate_up_absmax is None and lora.base.down_absmax is None
    with torch.no_grad():
        out1 = lora(hs, idx, w)
        out2 = lora(hs, idx, w)  # a second pass proves stage->evict->stage round-trips cleanly

    assert torch.allclose(ref, out1, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ref, out2, atol=1e-6, rtol=1e-6)
    assert lora._offload.staged is False  # post-hook evicted after each forward
    if quant_type == "bf16":
        assert lora.base.gate_up_absmax is None  # still None after stage->evict cycles


# NB: the pre-ExpertsNbit test `test_offload_backward_uses_saved_dequant_not_evicted_base`
# (non-checkpointed offload backward) was removed with the fold to the recompute-in-backward base.
# ExpertsNbit._project re-dequantizes the packed weight in backward, so a *non-checkpointed*
# offload backward would read the post-hook-evicted placeholder — non-checkpointed offload training
# is now unsupported (see offload.py). Offload+backward correctness is covered by the checkpointed
# tests below, which is the only configuration the trainer ever runs.


def test_non_checkpointed_offload_backward_fails_loudly():
    """The unsupported configuration must fail with the pointed invariant error, not a bare shape/
    index error (and never silently wrong gradients): after the post-hook evicts the layer, a plain
    (non-checkpointed) backward's re-dequant reads the 0-element placeholder and the recompute
    Function translates that into the offload-invariant message."""
    lora = _build_experts_lora()
    enable_expert_offload(lora, DEVICE, pin=False)

    hs, idx, w = _inputs()
    hs = hs.clone().requires_grad_(True)
    out = lora(hs, idx, w)  # post-hook evicts the base after this forward
    with pytest.raises(RuntimeError, match="offload-evicted expert|gradient checkpointing"):
        out.sum().backward()


def test_offload_survives_gradient_checkpoint_recompute():
    """The load-bearing correctness invariant: under ``use_reentrant=False`` checkpointing the layer
    forward is recomputed in backward, so the pre-hook must RE-STAGE the evicted experts for the
    recompute. The pre-hook fires twice (initial forward + recompute); gradients match a non-offloaded,
    non-checkpointed reference — which is only possible if the experts were correctly re-staged for the
    recompute (an evicted 0-element base would crash or mis-grad).

    NB: the evict *post*-hook fires only ~once, not twice — PyTorch stops the recompute early once the
    saved tensors are regenerated, so it returns (and runs the post-hook) only on the initial pass.
    Eviction during backward is handled by the single-resident-slot policy, covered by the residency
    test below; this test asserts the correctness half."""
    lora = _build_experts_lora()
    torch.nn.init.normal_(lora.gate_up_lora_B, std=0.01)
    torch.nn.init.normal_(lora.down_lora_B, std=0.01)
    hs, idx, w = _inputs()

    # Reference: experts resident, no checkpointing.
    hs_ref = hs.clone().requires_grad_(True)
    lora.zero_grad(set_to_none=True)
    lora(hs_ref, idx, w).sum().backward()
    ref_grad_A = lora.gate_up_lora_A.grad.clone()
    ref_grad_hs = hs_ref.grad.clone()

    # Offloaded + gradient-checkpointed: count how many times the pre-hook (re-staging) fires.
    enable_expert_offload(lora, DEVICE, pin=True)
    pre_calls = {"n": 0}
    lora.register_forward_pre_hook(lambda m, a: pre_calls.__setitem__("n", pre_calls["n"] + 1))
    lora.zero_grad(set_to_none=True)
    hs_off = hs.clone().requires_grad_(True)
    out = checkpoint(lambda a, b, c: lora(a, b, c), hs_off, idx, w, use_reentrant=False)
    out.sum().backward()

    assert pre_calls["n"] == 2  # experts re-staged for the backward recompute, not just the forward
    assert torch.allclose(lora.gate_up_lora_A.grad, ref_grad_A, atol=1e-5, rtol=1e-5)
    assert torch.allclose(hs_off.grad, ref_grad_hs, atol=1e-5, rtol=1e-5)


def test_offload_single_layer_residency_through_checkpointed_backward():
    """The memory invariant the whole feature rests on: at most ONE layer's experts are GPU-resident
    at a time — through backward too. Because the evict post-hook does not fire on the early-stopped
    recompute, this is enforced by the single-resident-slot (staging a layer evicts the prior one).
    Without it, every layer recomputed in backward would stay staged and accumulate to the full
    footprint. Three checkpointed layers run fwd+bwd; after backward at most one base remains staged
    (the last recomputed), the rest evicted."""
    layers = [_build_experts_lora(seed=s) for s in range(3)]
    handles = []
    for lo in layers:
        torch.nn.init.normal_(lo.gate_up_lora_B, std=0.01)
        handles.append(enable_expert_offload(lo, DEVICE, pin=True))

    hs, idx, w = _inputs()
    cur = hs.clone().requires_grad_(True)
    for lo in layers:  # each layer is its own checkpointed region, like a transformers decoder stack
        cur = checkpoint(lambda a, b, c, m=lo: m(a, b, c), cur, idx, w, use_reentrant=False)
    cur.sum().backward()

    assert sum(h.staged for h in handles) <= 1  # single-slot held through backward (else 3 = leak)
    assert sum(getattr(lo.base, "gate_up_proj").numel() > 0 for lo in layers) <= 1  # ≤1 resident on GPU


def test_offload_placement_homes_on_cpu_code_and_lora_resident():
    """The four big tensors move to CPU (pinned under CUDA); the base holds 0-element placeholders
    while evicted; the NF4 code buffer and the LoRA adapters stay on the compute device."""
    lora = _build_experts_lora()
    h = enable_expert_offload(lora, DEVICE, pin=True)

    for n in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax"):
        assert h.home[n].device.type == "cpu"
        assert getattr(lora.base, n).numel() == 0  # placeholder while evicted
    if torch.cuda.is_available():
        assert h.pinned and all(h.home[n].is_pinned() for n in h.home)

    dev = torch.device(DEVICE).type
    assert lora.base.code.device.type == dev  # NF4 codebook never offloaded
    assert lora.gate_up_lora_A.device.type == dev and lora.down_lora_A.device.type == dev


def test_offload_state_dict_full_save_correct_and_adapter_filter_unaffected():
    """A full ``state_dict()`` of an *evicted* model must contain the real expert weights (the CPU
    home copies, substituted by the offload post-hook) — a placeholder state_dict would silently save
    a model with no experts. The adapter-save filter ('lora' in key) is key-based, so it never matches
    ``base.*`` and stays exactly as cheap as before."""
    lora = _build_experts_lora()
    handle = enable_expert_offload(lora, DEVICE, pin=True)
    sd = lora.state_dict()

    base_keys = [k for k in sd if k.startswith("base.")]
    assert base_keys and not any("lora" in k for k in base_keys)
    # Full save is CORRECT: the hook substituted the CPU homes for the 0-element placeholders,
    # as references (no copy) — bit-identical to the offload handle's home tensors.
    for name in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax"):
        t = sd[f"base.{name}"]
        assert t.numel() > 0 and t.device.type == "cpu"
        assert t.data_ptr() == handle.home[name].data_ptr()  # reference, not a copy
    # The module itself still holds placeholders (state_dict didn't re-stage anything).
    assert lora.base.gate_up_proj.numel() == 0 and handle.staged is False

    adapter = {k: v for k, v in sd.items() if "lora" in k}
    assert adapter and all(not k.startswith("base.") for k in adapter)


def test_enable_expert_offload_is_idempotent():
    """A second enable on the same module must return the existing handle, not build a new one:
    while evicted the base's registered tensors are 0-element placeholders, so a fresh handle would
    capture THOSE as its CPU homes (losing the weights) and stack a second pair of stage/evict
    hooks whose staging of empty homes breaks the forward. Both public entry points are covered —
    the direct enable and the model walker on an already-offloaded model."""
    import torch.nn as nn

    lora = _build_experts_lora()
    hs, idx, w = _inputs()
    with torch.no_grad():
        ref = lora(hs, idx, w)

    h1 = enable_expert_offload(lora, DEVICE, pin=False)
    h2 = enable_expert_offload(lora, DEVICE, pin=False)
    assert h2 is h1  # existing handle returned, no re-capture
    assert all(h1.home[n].numel() > 0 for n in h1.home)  # homes still hold the real weights

    with torch.no_grad():
        out = lora(hs, idx, w)  # forward still correct (would stage 0-element homes without the guard)
    assert torch.allclose(ref, out, atol=1e-6, rtol=1e-6)

    handles = offload_model_experts(nn.ModuleList([lora]), pin=False)
    assert handles == [h1]  # walking an already-offloaded model is a no-op, not a corruption


def test_offload_model_experts_raises_when_none_found():
    """Offload was requested and nothing would be offloaded: that must be a loud RuntimeError, not
    a silent no-op that leaves every expert GPU-resident (no ExpertsLoRA modules, no quantization —
    runs on any host)."""
    import torch.nn as nn

    with pytest.raises(RuntimeError, match="no ExpertsLoRA"):
        offload_model_experts(nn.Sequential(nn.Linear(4, 4)))


def test_offload_model_experts_walks_all_experts_lora():
    """offload_model_experts offloads every ExpertsLoRA in a container (the already-loaded-model
    entry point) and infers the compute device from the resident base."""
    import torch.nn as nn

    m1 = _build_experts_lora(seed=0)
    m2 = _build_experts_lora(seed=2)
    model = nn.ModuleList([m1, m2])

    handles = offload_model_experts(model, pin=True)

    assert len(handles) == 2
    for m in (m1, m2):
        assert m._offload.staged is False
        assert m.base.gate_up_proj.numel() == 0  # evicted to placeholder
        assert m._offload.home["gate_up_proj"].device.type == "cpu"

    hs, idx, w = _inputs()  # a forward still works: experts stream in per module
    with torch.no_grad():
        m1(hs, idx, w)
