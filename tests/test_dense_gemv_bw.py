# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_DENSE_GEMV_BW`` (engines/dense_gemv_bw.py; lane P151, grouped-nf4-gemm K35), on CPU with a stub kernel: e4b's CI
pins a grouped-nf4-gemm without ``dense_gemv``, and the routing, refusals and engagement record are this repo's.

The kernel's own contract (relative RMS at most 1.05 x cuBLAS's, one-hot exact, bitwise) is grouped-nf4-gemm's
``kernel/test_dense_gemv.py``, checked on a CUDA card; P151 reads it served."""
import inspect
import sys

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from experts4bit_qlora import serve_paged as sp  # noqa: E402
from experts4bit_qlora.engines import dense_gemv_bw as dg  # noqa: E402


class _Attn(nn.Module):
    def __init__(self, hidden=16, qkv=24, bias=False, dtype=torch.bfloat16):
        super().__init__()
        self.qkv_proj = nn.Linear(hidden, qkv, bias=bias, dtype=dtype)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, dtype=dtype)


class _Int4Like(nn.Linear):
    """A projection subclass (as int4 / LoRA projections are): never swapped."""


class _Toy(nn.Module):
    def __init__(self, layers=3, **kw):
        super().__init__()
        self.layers = nn.ModuleList([_Attn(**kw) for _ in range(layers)])
        self.mlp = nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)          # not an attention projection
        self.other = _Attn()
        self.other.o_proj = _Int4Like(16, 16, bias=False, dtype=torch.bfloat16)


class _Stub:
    """Records each call and computes the reference product, as gemv_bf16_bw's contract promises (within rounding)."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, w):
        assert x.shape == (1, w.shape[1]) and x.is_contiguous()
        self.calls.append(tuple(w.shape))
        return torch.nn.functional.linear(x, w)


def _apply(model, monkeypatch=None):
    stub = _Stub()
    rec = dg.apply_dense_gemv_bw(model, "1", gemv=stub)
    return stub, rec


def test_mode_0_changes_nothing():
    m = _Toy()
    before = {n: type(x) for n, x in m.named_modules()}
    assert dg.apply_dense_gemv_bw(m, "0") == {"mode": "0", "modules": 0}
    assert {n: type(x) for n, x in m.named_modules()} == before


@pytest.mark.parametrize("mode", ["", "2", "auto", "true"])
def test_other_modes_are_refused(mode):
    with pytest.raises(ValueError, match="0 or 1"):
        dg.apply_dense_gemv_bw(_Toy(), mode)


def test_it_swaps_exactly_the_attention_projections_and_shares_their_weights():
    m = _Toy(layers=3)
    w_before = {id(a.qkv_proj.weight): a for a in m.layers} | {id(a.o_proj.weight): a for a in m.layers}
    ptrs = [(a.qkv_proj.weight.data_ptr(), a.o_proj.weight.data_ptr()) for a in m.layers]
    _stub, rec = _apply(m)
    # 3 layers x (qkv, o) plus the `other` module's qkv_proj; its _Int4Like o_proj and the MLP are left alone
    assert rec["modules"] == 7 and rec["per_target"] == {"qkv_proj": 4, "o_proj": 3}
    assert rec["shapes"] == [[16, 16], [24, 16]]
    for a, (pq, po) in zip(m.layers, ptrs):
        assert isinstance(a.qkv_proj, dg.DenseGemvLinear) and isinstance(a.qkv_proj, nn.Linear)
        assert a.qkv_proj.weight.data_ptr() == pq and a.o_proj.weight.data_ptr() == po      # no copy
        assert id(a.qkv_proj.weight) in w_before and a.qkv_proj.bias is None
    assert type(m.mlp) is nn.Linear and type(m.other.o_proj) is _Int4Like
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in _Toy(layers=3).parameters())


def test_one_bf16_device_row_routes_and_everything_else_passes(monkeypatch):
    m = _Toy(layers=2)
    stub, _ = _apply(m)
    monkeypatch.setattr(dg, "_on_device", lambda x: True)
    lin = m.layers[0].qkv_proj
    x = torch.randn(1, 16).to(torch.bfloat16)
    for shape in ((1, 16), (1, 1, 16), (16,)):
        y = lin(x.reshape(shape))
        assert y.shape == (*shape[:-1], 24)
        assert torch.equal(y.reshape(-1), torch.nn.functional.linear(x, lin.weight).reshape(-1))
    assert len(stub.calls) == 3
    lin(torch.randn(2, 16).to(torch.bfloat16))                     # two rows: a bucket above 1, or a prefill
    lin(torch.randn(1, 3, 16).to(torch.bfloat16))                  # a 3-token prefill
    with pytest.raises(RuntimeError, match="dtype"):               # fp32: F.linear, as the plain nn.Linear (same error)
        lin(torch.randn(1, 16))
    assert len(stub.calls) == 3
    strided = torch.randn(32).to(torch.bfloat16)[::2]                 # one row, not contiguous: F.linear
    assert torch.equal(lin(strided), torch.nn.functional.linear(strided, lin.weight)) and len(stub.calls) == 3
    monkeypatch.setattr(dg, "_on_device", lambda x: False)          # CPU: never the kernel
    lin(x)
    assert len(stub.calls) == 3
    assert lin.routed == {1: 3} and lin.passed == {2: 1, 3: 1, 1: 3}


def test_the_engagement_record_counts_modules_per_row_count(monkeypatch):
    m = _Toy(layers=2)
    _apply(m)
    monkeypatch.setattr(dg, "_on_device", lambda x: True)
    for a in m.layers:                                              # a bucket-1 capture, then a bucket-4 capture
        a.qkv_proj(torch.randn(1, 16).to(torch.bfloat16))
        a.o_proj(torch.randn(1, 16).to(torch.bfloat16))
        a.qkv_proj(torch.randn(4, 16).to(torch.bfloat16))
        a.o_proj(torch.randn(4, 16).to(torch.bfloat16))
    e = dg.engagement(m)
    assert e == {"modules": 5, "routed_modules_by_rows": {"1": 4}, "passed_modules_by_rows": {"4": 4}}


def test_the_refusals(monkeypatch):
    with pytest.raises(RuntimeError, match="has a bias"):
        dg.apply_dense_gemv_bw(_Toy(bias=True), "1", gemv=_Stub())
    with pytest.raises(RuntimeError, match="not a bf16"):
        dg.apply_dense_gemv_bw(_Toy(dtype=torch.float32), "1", gemv=_Stub())
    with pytest.raises(RuntimeError, match="int4 attention makes them Int4Linear.*vacuous"):
        dg.apply_dense_gemv_bw(nn.Sequential(nn.Linear(4, 4)), "1", gemv=_Stub())
    monkeypatch.setitem(sys.modules, "dense_gemv", None)            # a grouped-nf4-gemm without K35
    with pytest.raises(RuntimeError, match="K35"):
        dg.apply_dense_gemv_bw(_Toy(), "1")


def test_a_second_apply_swaps_nothing_more():
    m = _Toy()
    _apply(m)
    with pytest.raises(RuntimeError, match="vacuous"):
        dg.apply_dense_gemv_bw(m, "1", gemv=_Stub())


def test_the_server_config_reads_the_knob(monkeypatch):
    monkeypatch.delenv("E4B_DENSE_GEMV_BW", raising=False)
    assert sp.PagedServeConfig.from_env().dense_gemv_bw == "0"
    monkeypatch.setenv("E4B_DENSE_GEMV_BW", "1")
    assert sp.PagedServeConfig.from_env().dense_gemv_bw == "1"
    assert sp.DENSE_GEMV_BW_ENV == dg.ENV == "E4B_DENSE_GEMV_BW"


def test_build_engine_applies_it_after_the_fusions_and_before_any_capture():
    src = inspect.getsource(sp.build_engine)
    at = [src.index(s) for s in ("_apply_fusions(model, cfg", "apply_dense_gemv_bw(model,",
                                 "runner.enable_decode_graphs(", 'info["dense_gemv_bw"] = {**dense_gemv, "engagement"')]
    assert at == sorted(at), at
