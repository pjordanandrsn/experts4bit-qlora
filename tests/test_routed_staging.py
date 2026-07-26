"""Routed-only staging: copy the experts the router asked for, not all of them.

The offload pre-hook stages a layer's whole expert stack while a decode step
routes to `top_k`. At E=128 / top_k=8 that is 16x the bytes the routing needs,
and it measured as the dominant term in a 235B decode step (finding #21).

The load-bearing assertion is the first one: routed staging must produce **bit-
identical** output to bulk staging. It is the same weights through the same
kernels; only the rows that were copied differ, and the uncopied rows must never
be read. Anything less than exact equality means something read uninitialized
memory, which is precisely the failure this design has to rule out.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import (  # noqa: E402
    Experts4bit,
    ExpertsLoRA,
    enable_expert_offload,
    enable_inference_prefetch,
    enable_routed_staging,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
N_EXP, HIDDEN, INTER, TOP_K, N_TOK = 16, 128, 192, 2, 6
_QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="offload needs CUDA")


def _build(seed=0):
    torch.manual_seed(seed)
    gate_up = torch.randn(N_EXP, 2 * INTER, HIDDEN, dtype=DTYPE, device=DEVICE)
    down = torch.randn(N_EXP, HIDDEN, INTER, dtype=DTYPE, device=DEVICE)
    try:
        base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=DTYPE)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable: {e}")
    mod = ExpertsLoRA(base, r=8, alpha=16, dtype=DTYPE).to(DEVICE)
    with torch.no_grad():
        for p in (mod.gate_up_lora_B, mod.down_lora_B):
            p.normal_(0, 0.02)
    return mod.eval()


def _inputs(n_tok=N_TOK, n_experts_used=TOP_K, seed=1):
    torch.manual_seed(seed)
    hs = torch.randn(n_tok, HIDDEN, dtype=DTYPE, device=DEVICE)
    idx = torch.randint(0, n_experts_used, (n_tok, TOP_K), device=DEVICE)
    wts = torch.rand(n_tok, TOP_K, dtype=DTYPE, device=DEVICE)
    return hs, idx, wts


def test_routed_output_is_bit_identical_to_bulk():
    """The whole design rests on this: uncopied rows are never read."""
    hs, idx, wts = _inputs()

    bulk = _build()
    enable_expert_offload(bulk, DEVICE)
    with torch.no_grad():
        want = bulk(hs, idx, wts)

    routed = _build()
    h = enable_expert_offload(routed, DEVICE)
    enable_routed_staging([h])
    with torch.no_grad():
        got = routed(hs, idx, wts)

    assert torch.equal(got, want), (
        "routed staging changed the answer — a consumer read a row that was not copied"
    )


def test_routed_copies_less():
    """Only the routed rows should cross the link."""
    routed = _build()
    h = enable_expert_offload(routed, DEVICE)
    enable_routed_staging([h])
    hs, idx, wts = _inputs(n_experts_used=3)
    used = int(torch.unique(idx).numel())
    assert used <= 3
    with torch.no_grad():
        routed(hs, idx, wts)
    # Rows outside the routed set are allocated but never written, so the staged
    # tensor cannot equal the full home copy.
    full = h.home["gate_up_proj"].to(DEVICE)
    assert not torch.equal(routed.base.gate_up_proj, full), (
        "every row matched the home copy — the bulk path ran, not the routed one"
    )


def test_falls_back_to_bulk_above_crossover():
    """Prefill routes nearly everything; one bulk copy beats E strided ones."""
    routed = _build()
    h = enable_expert_offload(routed, DEVICE)
    enable_routed_staging([h], max_fraction=0.25)   # crossover at 4 of 16
    hs, idx, wts = _inputs(n_tok=32, n_experts_used=N_EXP)
    assert int(torch.unique(idx).numel()) > h._routed_max
    with torch.no_grad():
        out = routed(hs, idx, wts)
    full = h.home["gate_up_proj"].to(DEVICE)
    assert torch.equal(routed.base.gate_up_proj, full), "expected the bulk fallback"
    assert out.isfinite().all()


def test_refuses_prefetch_linked_handles():
    """Not a preference: next-layer routing is not knowable in time to prefetch."""
    mods = [_build(seed=i) for i in range(3)]
    handles = [enable_expert_offload(m, DEVICE) for m in mods]
    enable_inference_prefetch(handles)
    with pytest.raises(RuntimeError, match="incompatible with inference prefetch"):
        enable_routed_staging(handles)


def test_training_falls_back_to_bulk():
    """Grad and train() keep the conservative path so recompute re-stages identically."""
    routed = _build()
    h = enable_expert_offload(routed, DEVICE)
    enable_routed_staging([h])
    hs, idx, wts = _inputs(n_experts_used=2)

    routed.train()
    with torch.no_grad():
        routed(hs, idx, wts)
    full = h.home["gate_up_proj"].to(DEVICE)
    assert torch.equal(routed.base.gate_up_proj, full), (
        "train()-mode forward took the routed path; a checkpoint recompute could "
        "then read a row the initial forward never staged"
    )

    routed.eval()
    hs_g = hs.clone().requires_grad_(True)
    out = routed(hs_g, idx, wts)
    out.sum().backward()
    assert hs_g.grad is not None
