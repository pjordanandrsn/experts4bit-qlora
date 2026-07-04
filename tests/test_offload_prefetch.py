"""Inference prefetch over offloaded experts (:func:`experts4bit_qlora.enable_inference_prefetch`).

Prefetch changes *when* a layer's experts cross the PCIe bus (on a side stream, one layer ahead),
never *what* is computed — so a prefetched no_grad pass must be bit-identical to a non-offloaded
one, residency must stay bounded at two layers, and a grad-enabled forward must sweep the machinery
back to the training single-slot invariant. The math-identity and policy tests are CUDA-only (the
whole feature is a CUDA-stream construct); the linking/validation surface is CPU-testable.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import Experts4bit, ExpertsLoRA  # noqa: E402
from experts4bit_qlora import enable_inference_prefetch, offload_model_experts  # noqa: E402
from experts4bit_qlora.offload import _ExpertOffload  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
_QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)
N_LAYERS, E, HID, INTER, TOP_K, N_TOK = 3, 4, 64, 64, 2, 1


@pytest.fixture(autouse=True)
def _reset_offload_class_state():
    """Both residency trackers are class-level global state; isolate every test."""
    _ExpertOffload._resident = None
    _ExpertOffload._staged_now = set()
    yield
    _ExpertOffload._resident = None
    _ExpertOffload._staged_now = set()


class _Chain(torch.nn.Module):
    """A minimal 'model': N sequential ExpertsLoRA layers sharing routing inputs, so the offload
    pre/post hooks fire in the same layer order a decoder would drive them."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        for _ in range(N_LAYERS):
            gate_up = (torch.randn(E, 2 * INTER, HID) * 0.1).to(DEVICE)
            down = (torch.randn(E, HID, INTER) * 0.1).to(DEVICE)
            try:
                base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=torch.float32)
            except _QUANTIZE_UNAVAILABLE as e:
                pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")
            layers.append(ExpertsLoRA(base, r=4, alpha=8, dtype=torch.float32).to(DEVICE))
        self.layers = torch.nn.ModuleList(layers)
        # Inference contract: prefetch (and the decode routes) require eval mode; grad-enabled
        # forwards below still take the training stage() path regardless of this flag.
        self.eval()

    def forward(self, hs, idx, wts):
        for layer in self.layers:
            hs = hs + layer(hs, idx, wts)  # residual, like a decoder block
        return hs


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(N_TOK, HID, dtype=torch.float32, generator=g).to(DEVICE)
    idx = torch.randint(0, E, (N_TOK, TOP_K), generator=g).to(DEVICE)
    wts = torch.rand(N_TOK, TOP_K, dtype=torch.float32, generator=g).to(DEVICE)
    return hs, idx, wts


# ---------------------------------------------------------------------------------------------------
# Linking / validation surface (CPU-testable).
# ---------------------------------------------------------------------------------------------------
def test_enable_prefetch_empty_and_single_are_noops():
    assert enable_inference_prefetch([]) == []
    if DEVICE == "cpu":
        return  # single-handle path needs a real handle; covered on CUDA below
    chain = _Chain()
    handles = offload_model_experts(torch.nn.ModuleList([chain.layers[0]]))
    out = enable_inference_prefetch(handles)
    assert out == handles and handles[0]._prefetch_next is None  # sync path preserved


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only negative branch")
def test_enable_prefetch_rejects_cpu_handles():
    chain = _Chain()
    handles = offload_model_experts(chain)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        enable_inference_prefetch(handles)


@cuda
def test_enable_prefetch_links_circularly():
    chain = _Chain()
    handles = offload_model_experts(chain)
    enable_inference_prefetch(handles)
    assert [h._prefetch_next for h in handles] == handles[1:] + handles[:1]


# ---------------------------------------------------------------------------------------------------
# Math identity + residency policy (CUDA).
# ---------------------------------------------------------------------------------------------------
@cuda
def test_prefetched_inference_is_math_identical(monkeypatch):
    monkeypatch.setenv("E4B_INFER_GEMV", "0")  # location-only comparison: keep one compute route
    chain = _Chain()
    hs, idx, wts = _inputs()
    with torch.no_grad():
        ref = chain(hs, idx, wts)

    handles = offload_model_experts(chain)
    enable_inference_prefetch(handles)
    with torch.no_grad():
        got1 = chain(hs, idx, wts)  # first pass: layer 0 cold-misses, rest arrive prefetched
        got2 = chain(hs, idx, wts)  # second pass: layer 0 arrives via the circular wrap-around
    torch.testing.assert_close(got1, ref)
    torch.testing.assert_close(got2, ref)


@cuda
def test_prefetch_residency_bounded_at_two_and_wraps():
    chain = _Chain()
    hs, idx, wts = _inputs()
    handles = offload_model_experts(chain)
    enable_inference_prefetch(handles)

    seen_max = 0
    orig = _ExpertOffload.stage_for_inference

    def counting(self):
        nonlocal seen_max
        orig(self)
        seen_max = max(seen_max, len(_ExpertOffload._staged_now))

    _ExpertOffload.stage_for_inference = counting
    try:
        with torch.no_grad():
            chain(hs, idx, wts)
            chain(hs, idx, wts)
    finally:
        _ExpertOffload.stage_for_inference = orig

    assert seen_max <= 2  # computing layer + in-flight next, never more
    # Between forwards: every computed layer was post-hook-evicted; only the wrap-around prefetch
    # of layer 0 (kicked by the last layer) remains, ready for the next token.
    assert _ExpertOffload._staged_now == {handles[0]}
    assert handles[0].staged and not handles[1].staged and not handles[2].staged


@cuda
def test_training_stage_sweeps_prefetch_leftovers():
    """A grad-enabled forward right after an (interrupted) generate() must restore the single-slot
    training invariant: the first training stage() sweeps prefetch leftovers, including ones staged
    on layers other than the one about to run (which cannot be cleaned by its early-return). The
    mid-forward residency observation below is what actually pins the sweep — end-state asserts
    alone pass even without it, drained by the ordinary post-hook evicts."""
    chain = _Chain()
    hs, idx, wts = _inputs()
    handles = offload_model_experts(chain)
    enable_inference_prefetch(handles)

    with torch.no_grad():
        chain(hs, idx, wts)
        # Interrupted generation: layer 1 staged mid-pass, post-hooks never fired. Leftovers land
        # on NON-first layers, so layer 0's training stage() must reach its sweep loop rather than
        # the staged early-return (which is what a full generate()'s wrap-around leftover hits).
        handles[1].stage_for_inference()
    assert _ExpertOffload._staged_now == {handles[1], handles[2]}

    seen_max = 0
    orig = _ExpertOffload.stage

    def counting(self):
        nonlocal seen_max
        orig(self)
        seen_max = max(seen_max, sum(h.staged for h in handles))

    _ExpertOffload.stage = counting
    try:
        hs_g = hs.clone().requires_grad_(True)
        out = chain(hs_g, idx, wts)  # grad-enabled: pre-hooks take the sync stage() path
        out.sum().backward()
    finally:
        _ExpertOffload.stage = orig

    assert seen_max == 1  # the FIRST training stage() swept the leftovers: single-slot throughout
    assert _ExpertOffload._staged_now == set()
    assert sum(h.staged for h in handles) <= 1
