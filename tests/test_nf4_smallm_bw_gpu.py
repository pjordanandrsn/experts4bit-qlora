# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_NF4_SMALLM_BW=1`` on the served NF4 collapse, on a GPU (serving roadmap lever B, step B0's kernel contract).

With the knob on, a step of 2..8 tokens takes singleton groups, and grouped-nf4-gemm serves every singleton call
through K33's bandwidth GEMV where it engages. On a tiny all-hot NF4 store with :data:`hot_residency.DEVICE_GROUPING`
on (the graph server's setting) and ``GNF4_GEMV_BW=1`` (these test shapes are not in K33's auto table), this pins:

* each of T = 2..8 dispatches exactly one ``bw_prmt32`` launch per projection and no dot-pad, scalar GEMV, K25 or
  M-tile call, and is counted in :data:`hot_residency.NF4_SMALLM_BW_SEEN`;
* EVERY token's output row equals that token's own T == 1 output BITWISE: K33 reads one row per program, so a row's
  result does not depend on what else is in the step;
* the T = 4 step captures in a CUDA graph and replays bitwise as eager;
* T = 9 and the knob off keep today's device-grouped route (no K33 launch);
* the knob's output reads the module's reference forward within the served tolerance;
* all of it with the lean glue off (expanded rows) and at its default (the token rows through ``gather_div``).

Skips without CUDA or without a grouped-nf4-gemm that carries ``GNF4_GEMV_BW`` (K33). Run on a rented box, never on
the A2000.
"""
import pytest
import torch

ng = pytest.importorskip("nf4_grouped", reason="needs grouped-nf4-gemm (the [fast] extra)")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from experts4bit_qlora import Experts4bit, enable_hot_residency  # noqa: E402
from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402

E, H, INTER, TOP_K = 16, 256, 128, 8
GEMV_KEYS = ("dotpad", "dotpad_splitk", "scalar", "scalar_splitk")


@pytest.fixture(autouse=True)
def _needs_bw():
    if not hasattr(ng, "_gemv_nf4_bw") or "bw_prmt32" not in ng.dispatch_counts():
        pytest.skip("this grouped-nf4-gemm has no GNF4_GEMV_BW (K33, grouped-nf4-gemm#500)")


@pytest.fixture(params=["0", "auto"], ids=["lean0", "lean-auto"])
def served(monkeypatch, request):
    """A tiny all-hot NF4 expert module on the served collapse, device grouping on, K33 forced on; with the lean glue
    off (the collapse hands over the expanded (token, slot) rows) and at its default (the token rows, gather_div)."""
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", request.param)
    monkeypatch.setattr(hr, "_NF4_SMALLM_BW_RESOLVED", [False])         # resolved below, never leaked to other tests
    for k in ("GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE", "E4B_NF4_GROUPED_SMALLM", "E4B_NF4_T1_DEVICE_GROUPING"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GNF4_GEMV_BW", "1")
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "1")
    monkeypatch.setattr(hr, "NF4_SMALLM_BW_SEEN", {})
    monkeypatch.setattr(hr, "ROUTE_SEEN", {})
    saved = (hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0])     # one-element lists: restored by hand below
    hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = True, False
    torch.manual_seed(0)
    mod = Experts4bit.from_float(gate_up_proj=torch.randn(E, 2 * INTER, H).cuda(),
                                 down_proj=torch.randn(E, H, INTER).cuda(),
                                 compute_dtype=torch.bfloat16, has_gate=True)

    def inputs(T, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", generator=g)
        p = torch.softmax(torch.randn(T, E, device="cuda", generator=g), -1)
        w, i = torch.topk(p, k=TOP_K, dim=-1)
        return x, i, w.to(torch.bfloat16)

    with torch.no_grad():
        ref = {T: mod(*inputs(T, 1 + T)) for T in (4,)}                # the module's own reference forward
    assert enable_hot_residency(mod, [torch.arange(E)], device="cuda") == 1
    mod._hot_residency.collapse_resident = True
    assert hr.resolve_nf4_smallm_bw([mod], "cuda")["on"]
    try:
        yield mod, inputs, ref
    finally:
        hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = saved


def _delta(before):
    after = ng.dispatch_counts()
    return {k: after[k] - before.get(k, 0) for k in after}


def _run(mod, x, i, w):
    c0 = ng.dispatch_counts()
    with torch.no_grad():
        out = mod(x, i, w)
    torch.cuda.synchronize()
    return out, _delta(c0)


@pytest.mark.parametrize("T", range(2, 9))
def test_each_row_is_bitwise_its_own_t1_row_through_k33_only(served, T):
    mod, inputs, _ref = served
    x, i, w = inputs(T, 10 + T)
    assert i.unique().numel() < T * TOP_K                               # rows share experts: no reuse is the point
    out, d = _run(mod, x, i, w)
    assert d["bw_prmt32"] == 2 and d["bw_tree"] == 0, d                 # one launch per projection, all T rows
    assert all(d[k] == 0 for k in GEMV_KEYS), d
    assert hr.NF4_SMALLM_BW_SEEN == {f"T{T}": 1}
    assert not any(k.startswith(("nf4_k25", "nf4_mtile")) for k in hr.ROUTE_SEEN), hr.ROUTE_SEEN
    for t in range(T):
        alone, _ = _run(mod, x[t:t + 1], i[t:t + 1], w[t:t + 1])
        torch.testing.assert_close(out[t:t + 1].float(), alone.float(), rtol=0, atol=0)


def test_the_t4_step_captures_and_replays_as_eager(served):
    mod, inputs, _ref = served
    x, i, w = inputs(4, 3)
    with torch.no_grad():
        eager = mod(x, i, w)
        for _ in range(2):                                                # warm: compile and bind before capture
            mod(x, i, w)
        torch.cuda.synchronize()
        c0 = ng.dispatch_counts()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static = mod(x, i, w)
        d = _delta(c0)
        g.replay()
        torch.cuda.synchronize()
    assert d["bw_prmt32"] == 2 and all(d[k] == 0 for k in GEMV_KEYS), d
    torch.testing.assert_close(static.float(), eager.float(), rtol=0, atol=0)


def test_nine_tokens_and_the_knob_off_keep_the_device_grouped_route(served, monkeypatch):
    mod, inputs, _ref = served
    _out, d = _run(mod, *inputs(9, 4))
    assert d["bw_prmt32"] == d["bw_tree"] == 0, d
    monkeypatch.setenv("E4B_NF4_SMALLM_BW", "0")
    _out, d = _run(mod, *inputs(4, 4))
    assert d["bw_prmt32"] == d["bw_tree"] == 0, d
    assert hr.NF4_SMALLM_BW_SEEN == {}


def test_the_knob_reads_the_reference_within_the_served_tolerance(served):
    mod, inputs, ref = served
    out, _ = _run(mod, *inputs(4, 5))
    rel = ((out.float() - ref[4].float()).abs().max() / ref[4].float().abs().max()).item()
    assert rel < 1.5e-2, rel
