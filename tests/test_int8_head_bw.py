# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_INT8_HEAD_BW`` (engines/int8_head_bw.py; lane P153, grouped-nf4-gemm K36), on CPU with a stub kernel: e4b's CI
pins a grouped-nf4-gemm without ``gemv_i8_bw``, and the rounding, routing, refusals and engagement record are this
repo's.

The rounding is checked BITWISE against P150's I8HEAD round trip (``bench/i8dense/i8_fakequant.roundtrip_``), the grid
P150 ruled GO at both gates. The kernel's own contract (at most 1.05 x cuBLAS's error on the bf16 head, one-hot exact,
bitwise) is grouped-nf4-gemm's ``kernel/test_dense_gemv_i8.py``, checked on a CUDA card; P153 reads it served."""
import ast
import hashlib
import importlib.util
import inspect
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from experts4bit_qlora import serve_paged as sp  # noqa: E402
from experts4bit_qlora.engines import int8_head_bw as ih  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


def _fakequant():
    spec = importlib.util.spec_from_file_location("i8_fakequant", REPO / "bench" / "i8dense" / "i8_fakequant.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: sha256 of ``quantize_rows_int8``'s source (``def`` to its last line, LF): P148's prototype in bench/p148/p148_kernels.py,
#: which grouped-nf4-gemm K36 ships verbatim and pins by the same hash (kernel/test_dense_gemv_i8_interp.py, at 929a6906)
QUANTIZER_SHA256 = "8ea6dd9ee1b52594419f40339272059e413956aaed2a2f02d382450f854e8c6c"


def _source_segment(path, name):
    src = pathlib.Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, fn)


def _load_quantize():
    """P148's ``quantize_rows_int8`` EXECUTED from its own bytes (no restated copy), after its sha256 is checked against
    the hash K36 pins: if the shipped quantizer ever changes, this test stops reading what P150 ruled on, and says so."""
    seg = _source_segment(REPO / "bench" / "p148" / "p148_kernels.py", "quantize_rows_int8")
    assert hashlib.sha256(seg.encode("utf-8")).hexdigest() == QUANTIZER_SHA256
    ns = {"torch": torch}
    exec(compile(seg, "p148_kernels.quantize_rows_int8", "exec"), ns)
    return ns["quantize_rows_int8"]


_quantize = _load_quantize()


class _Stub:
    """Records each call and computes ``s * (q @ x)`` in fp32, as gemv_i8_bw's contract promises (within rounding)."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, q, s):
        assert x.shape == (1, q.shape[1]) and x.is_contiguous() and q.dtype == torch.int8 and s.dtype == torch.float32
        self.calls.append(tuple(q.shape))
        return (s * (q.float() @ x.float().reshape(-1))).to(torch.bfloat16).reshape(1, -1)


class _Int4Like(nn.Linear):
    """A head subclass (as an int4 head is an Int4Linear): never swapped."""


class _Toy(nn.Module):
    def __init__(self, vocab=40, hidden=16, tied=False, bias=False, dtype=torch.bfloat16, head_cls=nn.Linear, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab, hidden, dtype=dtype)
        self.lm_head = head_cls(hidden, vocab, bias=bias, dtype=dtype)
        with torch.no_grad():
            self.lm_head.weight.copy_((torch.randn(vocab, hidden, generator=g) / hidden ** 0.5).to(dtype))
            self.lm_head.weight[3] = 0                                   # an all-zero row: s = 1, q = 0
        if tied:
            self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, m):
        self.lm_head = m


def _apply(model):
    stub = _Stub()
    return stub, ih.apply_int8_head_bw(model, "1", gemv=stub, quantize=_quantize)


def test_mode_0_changes_nothing():
    m = _Toy()
    w = m.lm_head.weight.clone()
    assert ih.apply_int8_head_bw(m, "0") == {"mode": "0", "modules": 0}
    assert type(m.lm_head) is nn.Linear and torch.equal(m.lm_head.weight, w)


@pytest.mark.parametrize("mode", ["", "2", "auto", "true"])
def test_other_modes_are_refused(mode):
    with pytest.raises(ValueError, match="0 or 1"):
        ih.apply_int8_head_bw(_Toy(), mode)


def test_the_head_is_rounded_in_place_to_p150s_i8head_grid_bitwise():
    m, ref = _Toy(seed=1), _Toy(seed=1)
    _fakequant().roundtrip_(ref.lm_head.weight)                     # P150's I8HEAD, exactly as its box ran it
    ptr = m.lm_head.weight.data_ptr()
    _stub, rec = _apply(m)
    head = m.get_output_embeddings()
    assert isinstance(head, ih.Int8HeadLinear) and isinstance(head, nn.Linear)
    assert head.weight.data_ptr() == ptr and head.bias is None      # the same storage: no second bf16 copy
    assert torch.equal(head.weight, ref.lm_head.weight)             # bit for bit
    q, s = _quantize(_Toy(seed=1).lm_head.weight)                   # the codes and scales of the ORIGINAL weight
    assert torch.equal(head.q, q) and torch.equal(head.s, s) and float(head.s[3]) == 1.0 and not head.q[3].any()
    assert rec == {"mode": "1", "modules": 1, "shape": [40, 16], "int8_bytes": 640,
                   "grid": "quantize_rows_int8 (K36); the weight rounded in place to bf16(q * s), P150's I8HEAD"}


def test_the_chunked_grid_is_the_whole_call_bit_for_bit(monkeypatch):
    whole, chunked = _Toy(seed=2), _Toy(seed=2)
    _apply(whole)
    monkeypatch.setattr(ih, "CHUNK_ROWS", 7)                        # 40 rows in chunks of 7 (the last one ragged)
    ih.apply_int8_head_bw(chunked, "1", gemv=_Stub(), quantize=_quantize)
    a, b = whole.get_output_embeddings(), chunked.get_output_embeddings()
    assert torch.equal(a.weight, b.weight) and torch.equal(a.q, b.q) and torch.equal(a.s, b.s)


def test_knob_on_changes_multi_row_outputs_too_to_the_grid():
    m, orig = _Toy(seed=3), _Toy(seed=3)
    _apply(m)
    x = torch.randn(4, 16).to(torch.bfloat16)                       # a 4-row call: F.linear, on the rounded weight
    on = m.lm_head(x)
    assert torch.equal(on, torch.nn.functional.linear(x, m.lm_head.weight))
    assert not torch.equal(on, orig.lm_head(x)), "the grid moves a multi-row call (P150 ruled that grid at t1 and k16)"


def test_the_state_dict_and_parameters_are_unchanged():
    m, orig = _Toy(), _Toy()
    _apply(m)
    assert set(m.state_dict()) == set(orig.state_dict())             # q and s are not persistent
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in orig.parameters())


def test_one_bf16_device_row_routes_and_everything_else_passes(monkeypatch):
    m = _Toy(seed=4)
    stub, _ = _apply(m)
    monkeypatch.setattr(ih, "_on_device", lambda x: True)
    head = m.lm_head
    x = torch.randn(1, 16).to(torch.bfloat16)
    ref = (head.q.double() * head.s.double()[:, None]) @ x.double().reshape(-1)        # the grid, exactly, in fp64
    for shape in ((1, 16), (1, 1, 16), (16,)):
        y = head(x.reshape(shape))
        assert y.shape == (*shape[:-1], 40) and y.dtype == torch.bfloat16
        err = (y.reshape(-1).double() - ref).abs()                 # within one bf16 rounding of the output
        assert bool((err <= ref.abs() * 2.0 ** -8 + ref.abs().max() * 2.0 ** -16).all()), float(err.max())
    strided = torch.randn(32).to(torch.bfloat16)[::2]                # one row, not contiguous: COPIED and routed
    assert head(strided).shape == (40,)
    assert len(stub.calls) == 4
    head(torch.randn(2, 16).to(torch.bfloat16))                     # two rows: a bucket above 1, or a prefill
    head(torch.randn(1, 3, 16).to(torch.bfloat16))                  # a 3-token prefill
    with pytest.raises(RuntimeError, match="dtype"):                # fp32: F.linear, as the plain nn.Linear (same error)
        head(torch.randn(1, 16))
    monkeypatch.setattr(ih, "_on_device", lambda x: False)          # CPU: never the kernel
    head(x)
    assert len(stub.calls) == 4
    assert head.routed == {1: 4} and head.passed == {2: 1, 3: 1, 1: 2}


def test_the_engagement_record_counts_modules_per_row_count(monkeypatch):
    m = _Toy()
    _apply(m)
    monkeypatch.setattr(ih, "_on_device", lambda x: True)
    m.lm_head(torch.randn(1, 16).to(torch.bfloat16))                # a bucket-1 capture, then a bucket-4 capture
    m.lm_head(torch.randn(4, 16).to(torch.bfloat16))
    assert ih.engagement(m) == {"modules": 1, "routed_modules_by_rows": {"1": 1}, "passed_modules_by_rows": {"4": 1}}
    assert ih.engagement(_Toy()) == {"modules": 0, "routed_modules_by_rows": {}, "passed_modules_by_rows": {}}


def test_the_refusals(monkeypatch):
    with pytest.raises(RuntimeError, match="tied to the input embeddings"):
        ih.apply_int8_head_bw(_Toy(tied=True), "1", gemv=_Stub(), quantize=_quantize)
    with pytest.raises(RuntimeError, match="has a bias"):
        ih.apply_int8_head_bw(_Toy(bias=True), "1", gemv=_Stub(), quantize=_quantize)
    with pytest.raises(RuntimeError, match="not a bf16"):
        ih.apply_int8_head_bw(_Toy(dtype=torch.float32), "1", gemv=_Stub(), quantize=_quantize)
    with pytest.raises(RuntimeError, match="_Int4Like, not nn.Linear"):
        ih.apply_int8_head_bw(_Toy(head_cls=_Int4Like), "1", gemv=_Stub(), quantize=_quantize)
    with pytest.raises(RuntimeError, match="NoneType, not nn.Linear"):
        ih.apply_int8_head_bw(nn.Sequential(nn.Linear(4, 4)), "1", gemv=_Stub(), quantize=_quantize)
    monkeypatch.setitem(sys.modules, "dense_gemv", None)            # a grouped-nf4-gemm without K36
    with pytest.raises(RuntimeError, match="K36"):
        ih.apply_int8_head_bw(_Toy(), "1")


def test_a_second_apply_refuses_and_leaves_the_grid_alone():
    m = _Toy()
    _apply(m)
    w = m.lm_head.weight.clone()
    with pytest.raises(RuntimeError, match="Int8HeadLinear, not nn.Linear"):
        ih.apply_int8_head_bw(m, "1", gemv=_Stub(), quantize=_quantize)
    assert torch.equal(m.lm_head.weight, w)


def test_the_server_config_reads_the_knob(monkeypatch):
    monkeypatch.delenv("E4B_INT8_HEAD_BW", raising=False)
    assert sp.PagedServeConfig.from_env().int8_head_bw == "0"
    monkeypatch.setenv("E4B_INT8_HEAD_BW", "1")
    assert sp.PagedServeConfig.from_env().int8_head_bw == "1"
    assert sp.INT8_HEAD_BW_ENV == ih.ENV == "E4B_INT8_HEAD_BW"


def test_build_engine_applies_it_after_the_dense_gemv_and_before_any_capture():
    src = inspect.getsource(sp.build_engine)
    at = [src.index(s) for s in ("_apply_fusions(model, cfg", "apply_dense_gemv_bw(model,", "apply_int8_head_bw(model,",
                                 "runner.enable_decode_graphs(", 'info["int8_head_bw"] = {**int8_head, "engagement"')]
    assert at == sorted(at), at


def test_the_installed_quantizer_is_the_one_p150_ruled_on():
    """Where grouped-nf4-gemm's ``dense_gemv`` imports (K36 or later), its ``quantize_rows_int8`` is P148's bytes and
    reads bitwise as the reference above: the served grid cannot drift off P150's without this test saying so."""
    dg = pytest.importorskip("dense_gemv", reason="grouped-nf4-gemm without dense_gemv (e4b CI's pin predates K36)")
    if not hasattr(dg, "quantize_rows_int8"):
        pytest.skip("dense_gemv without K36's quantize_rows_int8")
    assert hashlib.sha256(_source_segment(dg.__file__, "quantize_rows_int8").encode("utf-8")).hexdigest() == QUANTIZER_SHA256
    w = (torch.randn(257, 96, generator=torch.Generator().manual_seed(9)) / 10).to(torch.bfloat16)
    w[5] = 0
    for got, want in zip(dg.quantize_rows_int8(w), _quantize(w)):
        assert torch.equal(got, want)
