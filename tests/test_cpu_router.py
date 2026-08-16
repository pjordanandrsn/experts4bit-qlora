# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""cpu_router (hybrid Phase 1): deterministic top-k rule, per-arch epilogue
math vs the reference formulas, patch/fallback/teardown behavior on CUDA,
and a positive control that the GPU cross-check actually raises."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.engines import cpu_router as cr  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)

E, H, K, ROWS = 8, 64, 3, 4


# --------------------------------------------------------------------------- #
# CPU-only: the routing math
# --------------------------------------------------------------------------- #

def test_deterministic_topk_ties_break_to_lower_index():
    logits = torch.tensor([[1.0, 3.0, 3.0, 0.5, 3.0, 2.0]])
    vals, idx = cr.deterministic_topk(logits, 3)
    assert idx.tolist() == [[1, 2, 4]]          # the three tied maxima, low→high
    assert vals.tolist() == [[3.0, 3.0, 3.0]]


def test_route_on_host_softmax_then_topk_matches_reference():
    torch.manual_seed(0)
    logits = torch.randn(ROWS, E)
    for norm in (False, True):
        w, idx = cr.route_on_host(logits, K, cr._SOFTMAX_THEN_TOPK, norm)
        # reference formula (olmoe/qwen3): softmax over ALL logits in fp32,
        # topk on probs, optional renorm
        probs = torch.softmax(logits.float(), dim=-1)
        ref_w, ref_idx = torch.topk(probs, K, dim=-1)
        if norm:
            ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)
        assert torch.equal(idx, ref_idx)         # no ties in randn draws
        assert torch.allclose(w, ref_w, atol=0, rtol=0)


def test_route_on_host_topk_then_softmax_matches_reference():
    torch.manual_seed(1)
    logits = torch.randn(ROWS, E)
    w, idx = cr.route_on_host(logits, K, cr._TOPK_THEN_SOFTMAX, False)
    ref_v, ref_idx = torch.topk(logits.float(), K, dim=-1)
    ref_w = torch.softmax(ref_v, dim=-1)
    assert torch.equal(idx, ref_idx)
    assert torch.allclose(w, ref_w, atol=0, rtol=0)


# --------------------------------------------------------------------------- #
# synthetic router modules (class NAMES drive the dispatch table)
# --------------------------------------------------------------------------- #

class OlmoeTopKRouter(torch.nn.Module):
    """Mirror of the transformers reference (softmax → topk → renorm)."""

    def __init__(self, norm=True, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.top_k, self.num_experts, self.hidden_dim = K, E, H
        self.norm_topk_prob = norm
        g = torch.Generator().manual_seed(7)
        self.weight = torch.nn.Parameter(
            torch.randn(E, H, generator=g).to(device=device, dtype=dtype)
        )

    def forward(self, hidden_states):
        h = hidden_states.reshape(-1, self.hidden_dim)
        logits = torch.nn.functional.linear(h, self.weight)
        probs = torch.softmax(logits, dtype=torch.float, dim=-1)
        val, idx = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            val = val / val.sum(dim=-1, keepdim=True)
        return logits, val.to(logits.dtype), idx


class WeirdTopKRouter(torch.nn.Module):
    """Unknown router class — enable() must skip it loudly, never guess."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(E, H))


def test_enable_skips_unknown_router_and_returns_zero():
    m = torch.nn.Sequential(WeirdTopKRouter())
    assert cr.enable_cpu_router(m) == 0


# --------------------------------------------------------------------------- #
# CUDA: served path vs reference, fallbacks, teardown, positive control
# --------------------------------------------------------------------------- #

def _cuda_router(dtype=torch.bfloat16):
    mod = OlmoeTopKRouter(dtype=dtype, device="cuda")
    mod.eval()          # a fresh Module is training=True and the served
    model = torch.nn.Sequential(mod)   # path (correctly) refuses train mode
    n = cr.enable_cpu_router(model, max_rows=ROWS, timing=True)
    assert n == 1
    return model, mod


@requires_cuda
def test_served_route_matches_reference_selection_and_weights():
    model, mod = _cuda_router()
    torch.manual_seed(3)
    hidden = torch.randn(1, ROWS, H, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        logits, w, idx = mod(hidden)
        ref = getattr(mod, cr._MARKER)
        r_logits, r_w, r_idx = ref(hidden)
    torch.cuda.synchronize()
    assert idx.is_cuda and w.is_cuda
    assert logits is None                        # served path pushes no logits
    assert w.dtype == torch.bfloat16
    # selections agree as sets per row (bf16-vs-fp32 boundary flips would
    # need near-ties; the seeded draw has none — checked by the weights too)
    for r in range(ROWS):
        assert set(idx[r].tolist()) == set(r_idx[r].tolist())
    # weights within bf16-linear vs fp32-linear tolerance, compared by pair
    ours = dict(zip(idx.reshape(-1).tolist(),
                    w.float().reshape(-1).tolist()))
    theirs = dict(zip(r_idx.reshape(-1).tolist(),
                      r_w.float().reshape(-1).tolist()))
    for e_id, val in ours.items():
        assert abs(val - theirs[e_id]) < 3e-2
    stats = cr.router_trip_stats(model)
    assert stats["served_calls"] == 1
    assert stats["trip_us_p50"] > 0


@requires_cuda
def test_grad_and_prefill_fall_back_to_reference():
    model, mod = _cuda_router()
    st = getattr(mod, cr._STATE)
    served0 = st.calls
    hidden = torch.randn(1, ROWS, H, device="cuda", dtype=torch.bfloat16,
                         requires_grad=True)
    _ = mod(hidden)                              # grad-mode → reference
    big = torch.randn(1, ROWS + 5, H, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        _ = mod(big)                             # prefill rows → reference
    assert st.calls == served0                   # nothing served


@requires_cuda
def test_disable_restores_and_double_patch_refuses():
    model, mod = _cuda_router()
    with pytest.raises(RuntimeError):
        cr.enable_cpu_router(model)
    assert cr.disable_cpu_router(model) == 1
    assert not hasattr(mod, cr._MARKER)
    assert not hasattr(mod, cr._STATE)
    torch.manual_seed(5)
    hidden = torch.randn(1, 2, H, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        logits, w, idx = mod(hidden)             # plain reference again
    assert logits.shape == (2, E)


@requires_cuda
def test_cross_check_positive_control_fires_on_corrupt_weights():
    """A checker that cannot fail is not a checker: corrupt the host router
    copy and require the GPU cross-check to raise."""
    model, mod = _cuda_router()
    st = getattr(mod, cr._STATE)
    st.assert_every = 1
    # corrupt IN PLACE: the served path reads the numpy VIEW of w_t, so a
    # rebinding corruption would silently miss it (this test caught exactly
    # that when the epilogue moved to numpy)
    st.np_w_t[:] = np.roll(st.np_w_t, 3, axis=1)    # wrong expert mapping
    hidden = torch.randn(1, 1, H, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), pytest.raises(RuntimeError, match="tie-break"):
        mod(hidden)


@requires_cuda
def test_route_inline_matches_route_on_host():
    """The dispatch-trimmed inline epilogue in route() must equal the
    route_on_host() spec exactly — this is the drift guard between them."""
    model, mod = _cuda_router()
    st = getattr(mod, cr._STATE)
    torch.manual_seed(9)
    hidden = torch.randn(1, ROWS, H, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        _, w, idx = mod(hidden)
    torch.cuda.synchronize()
    pin_h = st.pin_hidden[st.slot ^ 1][:ROWS]     # the slot route() used
    logits32 = pin_h.float() @ st.w_t
    ref_w, ref_idx = cr.route_on_host(logits32, st.k, st.kind,
                                      bool(mod.norm_topk_prob))
    assert torch.equal(idx.cpu(), ref_idx)
    assert torch.allclose(w.float().cpu(), ref_w.float(),
                          atol=1e-2, rtol=1e-2)   # bf16 staging cast only


@requires_cuda
def test_cross_check_passes_on_honest_weights():
    model, mod = _cuda_router()
    st = getattr(mod, cr._STATE)
    st.assert_every = 1
    torch.manual_seed(11)
    hidden = torch.randn(1, 2, H, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        mod(hidden)                              # must not raise
