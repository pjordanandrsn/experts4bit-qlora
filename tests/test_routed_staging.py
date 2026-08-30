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

from quant_guard import require_quantize

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

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="offload needs CUDA")


def _build(seed=0):
    torch.manual_seed(seed)
    gate_up = torch.randn(N_EXP, 2 * INTER, HIDDEN, dtype=DTYPE, device=DEVICE)
    down = torch.randn(N_EXP, HIDDEN, INTER, dtype=DTYPE, device=DEVICE)
    require_quantize(DEVICE)
    base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=DTYPE)
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
    """Only the routed rows cross the link — asserted on bytes, not on a proxy.

    Note the module's own tensors are useless here: the post-hook evicts them to
    0-element placeholders the instant the forward returns, so a test that reads
    `routed.base.gate_up_proj` afterwards compares against a placeholder and
    passes no matter which path ran.
    """
    routed = _build()
    h = enable_expert_offload(routed, DEVICE)
    enable_routed_staging([h])
    hs, idx, wts = _inputs(n_experts_used=3)
    used = int(torch.unique(idx).numel())
    with torch.no_grad():
        routed(hs, idx, wts)
    assert h._last_stage_policy == "routed"
    expected = h._stage_nbytes * used / N_EXP
    assert h._last_stage_nbytes == pytest.approx(expected, rel=0.01), (
        f"moved {h._last_stage_nbytes} bytes for {used}/{N_EXP} experts; "
        f"routing implies ~{expected:.0f}"
    )
    assert h._last_stage_nbytes < h._stage_nbytes / 4


def test_falls_back_to_bulk_above_crossover():
    """Prefill routes nearly everything; one bulk copy beats E strided ones."""
    routed = _build()
    h = enable_expert_offload(routed, DEVICE)
    enable_routed_staging([h], max_fraction=0.25)   # crossover at 4 of 16
    hs, idx, wts = _inputs(n_tok=32, n_experts_used=N_EXP)
    assert int(torch.unique(idx).numel()) > h._routed_max
    with torch.no_grad():
        out = routed(hs, idx, wts)
    assert h._last_stage_policy != "routed", "expected the bulk fallback above the crossover"
    assert h._last_stage_nbytes == h._stage_nbytes
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
    assert h._last_stage_policy != "routed", (
        "train()-mode forward took the routed path; a checkpoint recompute could "
        "then read a row the initial forward never staged"
    )
    assert h._last_stage_nbytes == h._stage_nbytes

    # A grad-enabled forward must also take the bulk path. Deliberately NOT
    # completing a backward here: offload.py documents that non-checkpointed
    # offload training is unsupported (the re-dequant would read an evicted
    # placeholder), so calling .backward() would assert against a configuration
    # the package refuses by design rather than against routed staging.
    routed.eval()
    hs_g = hs.clone().requires_grad_(True)
    out = routed(hs_g, idx, wts)
    assert out.requires_grad
    assert h._last_stage_policy != "routed", "grad-enabled forward took the routed path"
    assert h._last_stage_nbytes == h._stage_nbytes


def test_composes_with_the_grouped_kernel():
    """Routed staging and the grouped kernel fix DIFFERENT halves of the step.

    Two gates, and the second is the strong one:

    1. vs bulk+reference, `rel < 2e-2` — the kernel boundary is documented as
       inexact (fp32 accumulation vs a bf16 materialization), so equality is the
       wrong ask here.
    2. **bulk+grouped == routed+grouped, BIT-EXACTLY.** Staging is held constant
       in the kernel and routed staging is bit-identical, so the kernel receives
       byte-identical inputs and must return byte-identical outputs. The original
       version of this test asserted only (1), which could not have caught a
       staging-dependent difference under the kernel — the exact gap that left
       finding #23's divergence unexplained.
    """
    from experts4bit_qlora import enable_fast, fast_available

    if not fast_available():
        pytest.skip("grouped kernel unavailable")

    hs, idx, wts = _inputs()

    ref_bulk = _build()
    enable_expert_offload(ref_bulk, DEVICE)
    with torch.no_grad():
        want = ref_bulk(hs, idx, wts)

    bulk_g = _build()
    enable_expert_offload(bulk_g, DEVICE)
    assert enable_fast(bulk_g) == 1
    with torch.no_grad():
        got_bulk = bulk_g(hs, idx, wts)

    routed_g = _build()
    h = enable_expert_offload(routed_g, DEVICE)
    enable_routed_staging([h])
    assert enable_fast(routed_g) == 1
    with torch.no_grad():
        got_routed = routed_g(hs, idx, wts)

    assert h._last_stage_policy == "routed", "routed staging stopped firing under [fast]"

    rel = ((got_routed.float() - want.float()).norm()
           / want.float().norm().clamp_min(1e-12)).item()
    assert rel < 2e-2, f"pair disagrees with bulk+reference (rel {rel:.3e})"

    assert torch.equal(got_bulk, got_routed), (
        "bulk+grouped != routed+grouped with the kernel held constant. Routed "
        "staging is bit-identical, so the kernel saw identical bytes and must "
        "return identical results — this means it consumed something staging "
        "changed."
    )
