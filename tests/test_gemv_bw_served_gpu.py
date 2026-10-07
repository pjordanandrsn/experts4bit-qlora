# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``GNF4_GEMV_BW=1`` on the served NF4 decode route (lane P116's premise; e4b#1313).

grouped-nf4-gemm's K33 (``_gemv_nf4_bw``, opt-in behind ``GNF4_GEMV_BW``) replaces dot-pad / the scalar GEMV in
``gemm_4bit_grouped``'s single-row decode branch. The served graph server reaches that branch only one way: the
all-resident collapse (``serve_paged.build_engine`` -> ``enable_hybrid_tier(..., collapse_resident=True)``) under
:data:`hot_residency.DEVICE_GROUPING`, where ``_collapsed_grouping`` sends ``T == 1`` to singleton groups and
``T > 1`` to device grouping (the captured M-tile GEMM). This file pins that on a tiny all-hot NF4 store:

* ``T == 1`` with the switch on dispatches every expert projection to ``bw_prmt32`` and nothing to dot-pad or the
  scalar GEMV; with it off, nothing to ``bw_*``;
* both routes read the dequantised reference within the served tolerance;
* the switch's step captures in a CUDA graph and replays bitwise as eager (PDL included, at its default);
* ``T > 1`` does not reach the decode GEMV at all, so P116's 16-request workload is a control, not the subject.

Skips without CUDA or without a grouped-nf4-gemm that carries ``GNF4_GEMV_BW``.
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


@pytest.fixture
def served(monkeypatch):
    """A tiny all-hot NF4 expert module on the served collapse, device grouping on (the graph server's setting)."""
    for k in ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE"):
        monkeypatch.delenv(k, raising=False)
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
        ref = {T: mod(*inputs(T, 1 + T)) for T in (1, 4)}           # the module's own reference forward
    assert enable_hot_residency(mod, [torch.arange(E)], device="cuda") == 1
    mod._hot_residency.collapse_resident = True
    try:
        yield mod, inputs, ref
    finally:
        hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = saved


def _delta(before):
    after = ng.dispatch_counts()
    return {k: after[k] - before.get(k, 0) for k in after}


def _b_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def _run(mod, inputs, T, seed):
    c0 = ng.dispatch_counts()
    with torch.no_grad():
        out = mod(*inputs(T, seed))
    torch.cuda.synchronize()
    return out, _delta(c0)


def test_t1_with_the_switch_on_reaches_bw_prmt32_only(served, monkeypatch):
    mod, inputs, ref = served
    monkeypatch.setenv("GNF4_GEMV_BW", "1")
    out, d = _run(mod, inputs, 1, 2)
    assert d["bw_prmt32"] == 2 and d["bw_tree"] == 0, d                 # gate_up and down, one launch each
    assert all(d[k] == 0 for k in GEMV_KEYS), d
    assert _b_rel(out, ref[1]) < 1.5e-2, _b_rel(out, ref[1])


def test_t1_with_the_switch_off_never_reaches_bw(served):
    mod, inputs, ref = served
    out, d = _run(mod, inputs, 1, 2)
    assert d["bw_prmt32"] == d["bw_tree"] == d["bw_splitk"] == 0, d
    assert sum(d[k] for k in GEMV_KEYS) == 2, d                          # dot-pad or the scalar GEMV, by shape
    assert _b_rel(out, ref[1]) < 1.5e-2, _b_rel(out, ref[1])


def test_t1_on_and_off_agree_within_the_served_tolerance(served, monkeypatch):
    mod, inputs, _ref = served
    off, _ = _run(mod, inputs, 1, 2)
    monkeypatch.setenv("GNF4_GEMV_BW", "1")
    on, _ = _run(mod, inputs, 1, 2)
    assert _b_rel(on, off) < 1.5e-2, _b_rel(on, off)


def test_t1_with_the_switch_on_captures_and_replays_as_eager(served, monkeypatch):
    mod, inputs, _ref = served
    monkeypatch.setenv("GNF4_GEMV_BW", "1")
    x, i, w = inputs(1, 3)
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


def test_t_above_1_takes_device_grouping_and_never_the_decode_gemv(served, monkeypatch):
    mod, inputs, ref = served
    monkeypatch.setenv("GNF4_GEMV_BW", "1")
    out, d = _run(mod, inputs, 4, 5)
    assert d["bw_prmt32"] == d["bw_tree"] == 0 and all(d[k] == 0 for k in GEMV_KEYS), d
    assert _b_rel(out, ref[4]) < 1.5e-2, _b_rel(out, ref[4])
