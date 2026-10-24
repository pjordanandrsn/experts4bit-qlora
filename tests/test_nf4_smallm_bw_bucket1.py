# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_NF4_SMALLM_BW`` is inert at bucket 1 (lane P234's control, on CPU): the served one-token decode call is the same
call for call with the knob unset and at ``1``.

The served path here is the real all-resident collapse (``_HotResidency._forward_collapsed``) of an all-hot NF4 expert
module at Mixtral's routing (8 experts, top-2), with :data:`hot_residency.DEVICE_GROUPING` on (the graph server's
setting), the lean glue off and at its default, and the switch resolved as the engine build resolves it
(:func:`hot_residency.resolve_nf4_smallm_bw`, K33's engagement stubbed). The kernels are the route tests' CPU oracles
(``test_nf4_grouped_smallm_route._install_stubs``), each wrapped by one recorder, so a step's log is, in order:
  - every decision of the switch (``_nf4_smallm_bw_takes``) and every grouping (``_collapsed_grouping``), with its result;
  - every kernel call, by name, with each argument (a tensor by dtype, shape and a hash of its bytes);
  - after the step, the route tally (``ROUTE_SEEN``), the switch's tally (``NF4_SMALLM_BW_SEEN``) and the output's
    bytes.
At T == 1 the two logs must be EQUAL over six routings. At T = 2 and 4 the same recorder must see them DIFFER (the
knob's singleton route against K25), so an equal bucket-1 log is not a recorder that sees nothing.
The kernel's own bits on a card (each row at T = 2..8 equal to its T == 1 row) are ``test_nf4_smallm_bw_gpu.py``'s.
"""
import functools
import hashlib
import os
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("nf4_grouped")

from test_nf4_grouped_smallm_route import _install_stubs  # noqa: E402

from experts4bit_qlora import Experts4bit, enable_hot_residency  # noqa: E402
from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402

E, H, INTER, TOP_K = 8, 64, 64, 2                  # Mixtral's routing: 8 experts, top-2
SEEDS = range(6)
KNOBS = ("E4B_NF4_SMALLM_BW", "E4B_NF4_GROUPED_SMALLM", "E4B_NF4_T1_DEVICE_GROUPING", "E4B_K25_LUT", "E4B_K25_PLAN",
         "GNF4_GEMV_BW", "E4B_INT4_LEAN_GLUE")


def _arg(a):
    if torch.is_tensor(a):
        t = a.detach().contiguous()
        return ("tensor", str(t.dtype), tuple(t.shape), hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(a, (list, tuple)):
        return tuple(_arg(x) for x in a)
    return repr(a)


def _recorded(log, name, fn):
    @functools.wraps(fn)                            # keeps the signature: hot_residency probes it for gather_div
    def rec(*args, **kw):
        out = fn(*args, **kw)
        log.append(("kernel", name, tuple(_arg(a) for a in args), tuple(sorted((k, _arg(v)) for k, v in kw.items()))))
        return out
    return rec


def _decision(log, name, fn):
    def rec(*args):
        out = fn(*args)
        log.append(("decision", name, args[0], out))
        return out
    return rec


def _step_log(monkeypatch, *, knob, lean, T):
    """One arm in a fresh process state: the module built, the switch resolved as build_engine resolves it, then one
    decode call of ``T`` tokens per seed through the collapse. Returns the ordered log."""
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("E4B_INT4_LEAN_GLUE", lean)
    if knob is not None:
        monkeypatch.setenv("E4B_NF4_SMALLM_BW", knob)
    monkeypatch.setattr(hr, "NF4_SMALLM_BW_SEEN", {})
    monkeypatch.setattr(hr, "ROUTE_SEEN", {})
    monkeypatch.setattr(hr, "_NF4_SMALLM_BW_RESOLVED", [False])
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    log = []
    _install_stubs(monkeypatch)
    import nf4_grouped
    import nf4_smallm

    def gemm_4bit_grouped(x, B, absmax, sizes, eids, gather_div=1):   # the singleton route, gather_div as gnf4#530's
        xr = x.repeat_interleave(gather_div, 0)
        e_l = eids.tolist() if torch.is_tensor(eids) else list(eids)
        out, r = torch.empty(xr.shape[0], B.shape[1], dtype=torch.bfloat16), 0
        for m, e in zip(sizes, e_l):
            w = nf4_grouped.dequant_ref(B[int(e)], absmax[int(e)], B.shape[1], B.shape[2] * 2)
            out[r:r + m] = (xr[r:r + m].float() @ w.t()).to(torch.bfloat16)
            r += m
        return out
    for name, fn in (("gemm_4bit_grouped", gemm_4bit_grouped),
                     ("gemm_4bit_grouped_captured", nf4_grouped.gemm_4bit_grouped_captured),
                     ("build_group_tiles_device", nf4_grouped.build_group_tiles_device)):
        monkeypatch.setattr(nf4_grouped, name, _recorded(log, name, fn))
    monkeypatch.setattr(nf4_smallm, "gemm_nf4_grouped_smallm",
                        _recorded(log, "k25", nf4_smallm.gemm_nf4_grouped_smallm))
    monkeypatch.setattr(nf4_grouped, "_bw_engages", lambda N, K, device: True, raising=False)
    for name in ("_nf4_smallm_bw_takes", "_collapsed_grouping"):
        monkeypatch.setattr(hr, name, _decision(log, name, getattr(hr, name)))

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mod = Experts4bit.from_float(gate_up_proj=torch.randn(E, 2 * INTER, H) * 0.1,
                                     down_proj=torch.randn(E, H, INTER) * 0.1,
                                     compute_dtype=torch.bfloat16, has_gate=True)
        assert enable_hot_residency(mod, [torch.arange(E)], device="cpu") == 1
    mod._hot_residency.collapse_resident = True
    res = hr.resolve_nf4_smallm_bw([mod], "cpu")
    assert res["on"] is (knob == "1"), res
    del log[:]                                      # the build's own traffic is not the step's
    for seed in SEEDS:
        g = torch.Generator().manual_seed(100 * T + seed)
        x = torch.randn(T, H, generator=g).to(torch.bfloat16)
        w, i = torch.topk(torch.softmax(torch.randn(T, E, generator=g), -1), k=TOP_K, dim=-1)
        with torch.no_grad():
            out = mod(x, i, w.to(torch.bfloat16))
        log.append(("step", seed, dict(hr.ROUTE_SEEN), dict(hr.NF4_SMALLM_BW_SEEN), _arg(out)))
    return log


def _arm(monkeypatch, **kw):
    with monkeypatch.context() as m:                # every patch undone before the other arm builds
        return _step_log(m, **kw)


def _kernels(log):
    return [e[1] for e in log if e[0] == "kernel"]


@pytest.mark.parametrize("knob", [None, "0"], ids=["unset", "zero"])
@pytest.mark.parametrize("lean", ["0", "auto"], ids=["lean0", "lean-auto"])
def test_bucket_1_is_the_same_call_for_call_with_the_knob_off_and_on(monkeypatch, lean, knob):
    off = _arm(monkeypatch, knob=knob, lean=lean, T=1)
    on = _arm(monkeypatch, knob="1", lean=lean, T=1)
    assert len(off) == len(on)
    for n, (a, b) in enumerate(zip(off, on)):
        assert a == b, f"entry {n} differs: knob {knob!r} {a} vs knob '1' {b}"
    # what both run: the singleton route, two gemm_4bit_grouped calls a step, no tile table and no K25
    assert _kernels(on) == ["gemm_4bit_grouped"] * (2 * len(SEEDS))
    assert [e[3] for e in on if e[0] == "decision" and e[1] == "_collapsed_grouping"] == [(True, False)] * len(SEEDS)
    assert all(e[3] is False for e in on if e[0] == "decision" and e[1] == "_nf4_smallm_bw_takes")
    last = on[-1]
    assert last[2] == {"nf4_singleton|le256": len(SEEDS)} and last[3] == {}, last


@pytest.mark.parametrize("T", [2, 4])
def test_the_recorder_sees_the_knob_act_at_buckets_2_and_4(monkeypatch, T):
    off = _arm(monkeypatch, knob=None, lean="auto", T=T)
    on = _arm(monkeypatch, knob="1", lean="auto", T=T)
    assert off != on
    assert "k25" in _kernels(off) and "k25" not in _kernels(on)
    assert set(_kernels(on)) == {"gemm_4bit_grouped"}
    assert on[-1][3] == {f"T{T}": len(SEEDS)} and off[-1][3] == {}
