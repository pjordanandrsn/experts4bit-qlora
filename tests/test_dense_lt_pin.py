# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_DENSE_LT_PIN`` (engines/dense_lt_pin.py; the serving roadmap's lever K, step K1), on CPU with a fake cuBLASLt
library, device properties and streams. Pins:

- the knob's parsing and refusals; the mapped-library lookup (one path taken, none or two refused);
- off changes nothing; on, o_proj (an exact nn.Linear, or K35's one-row module over one) and the router (only on the
  router epilogue's softmax-then-top-k forward, no bias) are routed;
- 2..8 rows with a pin route through the pinned call, 2..8 rows without one pass to what ran before (counted, by site,
  M and phase), every other row count passes uncounted;
- refusals by name: a captured bucket without a pin, another cuBLASLt version (no key), a configuration cuBLASLt
  rejects, a malformed or INPLACE pin, a biased or non-Linear o_proj, a router off the epilogue, a non-bf16 operand at a
  routed call, a call on a second stream;
- the router epilogue's GEMM is ``F.linear`` exactly when no pin is set.
The real library and its bits are ``test_dense_lt_pin_gpu.py``'s.
"""
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experts4bit_qlora.engines import dense_lt_pin as dl

N_O, K_O = 32, 64          # o_proj (N, K) of the toy model
N_R, K_R = 8, 32           # router (N, K)
VERSION, CC, SMS = 120805, (12, 0), 170
CFG = {"algo_id": 21, "tile": 0, "split_k": 9, "reduction_scheme": 4, "cta_swizzling": 0, "custom_option": 0,
       "stages": 0, "inner_shape": 0, "cluster_shape": 0}
MAPS = "7f00-7f10 r-xp 00000000 08:01 1 /usr/lib/python3/site-packages/nvidia/cublas/lib/libcublasLt.so.12\n"


class FakeLib:
    calls = []

    def __init__(self, path):
        self.path = path

    def version(self):
        return VERSION

    def problem(self, N, K, M):
        return {"N": N, "K": K, "M": M}

    def algo(self, cfg):
        return dict(cfg)

    def check(self, prob, a):
        if a["algo_id"] == 999:
            raise RuntimeError(f"{dl.ENV}: cuBLASLt rejects the pinned configuration {a}")
        return 4096

    def matmul(self, prob, a, x, w, out, ws, ws_bytes, stream):
        FakeLib.calls.append((prob["N"], prob["K"], prob["M"], a["algo_id"], ws_bytes, stream))
        out.copy_(F.linear(x, w))


class FakeAttention(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.o_proj = nn.Linear(K_O, N_O, bias=bias).to(torch.bfloat16)


class FakeRouter(nn.Module):
    def __init__(self, kind="softmax_topk", bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(N_R, K_R).to(torch.bfloat16))
        self.bias = bias
        if kind is not None:
            self._e4b_router_epi_kind = kind


class Model(nn.Module):
    def __init__(self, layers=2, **kw):
        super().__init__()
        self.layers = nn.ModuleList(nn.ModuleDict({"self_attn": FakeAttention(kw.get("bias", False)),
                                                   "gate": FakeRouter(kw.get("kind", "softmax_topk"),
                                                                      kw.get("rbias"))})
                                    for _ in range(layers))


class Stream:
    def __init__(self, n):
        self.n, self.cuda_stream = n, 1000 + n

    def __eq__(self, other):
        return isinstance(other, Stream) and other.n == self.n

    def __hash__(self):
        return self.n


@pytest.fixture(autouse=True)
def fake_cuda(monkeypatch):
    state = {"stream": Stream(0)}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda *a, **k: types.SimpleNamespace(major=CC[0], minor=CC[1], multi_processor_count=SMS))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: state["stream"])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(dl, "LT_PINS", {})
    monkeypatch.setattr(dl, "SEEN", {})
    monkeypatch.delenv(dl.ENV, raising=False)
    FakeLib.calls = []
    return state


def _pins(ms=(2, 4, 8), shapes=((N_O, K_O), (N_R, K_R)), cfg=CFG, version=VERSION):
    return {(version, CC, SMS, N, K, M): dict(cfg) for N, K in shapes for M in ms}


def _apply(model, value, buckets=(1, 2, 4, 8, 16), **kw):
    return dl.apply_dense_lt_pin(model, buckets, device="cpu", value=value, lib_factory=FakeLib, maps_text=MAPS, **kw)


@pytest.mark.parametrize("value, want", [("", ()), ("o_proj", ("o_proj",)), (" Router , o_proj ", ("o_proj", "router"))])
def test_the_knob_parses_to_registered_sites(value, want):
    assert dl.sites_env(value) == want


@pytest.mark.parametrize("value", ["qkv", "o_proj,o_proj", "lm_head", "1"])
def test_an_unknown_or_repeated_site_is_refused(value):
    with pytest.raises(ValueError, match=r"E4B_DENSE_LT_PIN=.*subset of \('o_proj', 'router'\)"):
        dl.sites_env(value)


def test_the_mapped_library_is_the_one_torch_loaded():
    assert dl.mapped_cublaslt(MAPS + MAPS).endswith("libcublasLt.so.12")              # repeated lines, one file
    with pytest.raises(RuntimeError, match=r"exactly one libcublasLt mapped .* found \[\]"):
        dl.mapped_cublaslt("7f00-7f10 r-xp 00000000 08:01 1 /usr/lib/libc.so.6\n")
    two = MAPS + MAPS.replace("site-packages/nvidia", "other/nvidia")
    with pytest.raises(RuntimeError, match=r"exactly one libcublasLt mapped"):
        dl.mapped_cublaslt(two)


def test_off_changes_nothing():
    m = Model()
    rec = _apply(m, "")
    assert rec == {"env": None, "on": False, "sites": []}
    assert type(m.layers[0]["self_attn"].o_proj) is nn.Linear and not hasattr(m.layers[0]["gate"], "_e4b_lt_pin")


def test_o_proj_routes_pinned_rows_passes_the_rest_and_counts_by_m(monkeypatch):
    monkeypatch.setattr(dl, "LT_PINS", _pins())
    m = Model()
    rec = _apply(m, "o_proj")
    assert rec["on"] and rec["sites"] == ["o_proj"] and rec["modules"] == {"o_proj": 2}
    assert rec["key"] == {"cublaslt": VERSION, "cc": list(CC), "sm_count": SMS} and rec["workspace_bytes"] == 4096
    assert rec["pinned_m"] == {f"o_proj|{N_O}x{K_O}": [2, 4, 8]} and rec["graphed_buckets"] == [2, 4, 8]
    lin = m.layers[0]["self_attn"].o_proj
    assert isinstance(lin, dl.PinnedLinear) and lin.weight is lin.inner.weight
    for M in (1, 2, 3, 8, 9):
        x = torch.randn(M, K_O).to(torch.bfloat16)
        torch.testing.assert_close(lin(x), F.linear(x, lin.weight), rtol=0, atol=0)
    assert [c[:3] for c in FakeLib.calls] == [(N_O, K_O, 2), (N_O, K_O, 8)]          # only the pinned M routed
    assert dl.engagement() == {"o_proj|M2|eager|pinned": 1, "o_proj|M3|eager|passed": 1, "o_proj|M8|eager|pinned": 1}


def test_the_phase_is_the_paged_contexts(monkeypatch):
    from experts4bit_qlora.engines import paged_attention as pa
    monkeypatch.setattr(dl, "LT_PINS", _pins())
    m = Model()
    _apply(m, "o_proj")
    monkeypatch.setattr(pa, "current_context", lambda: types.SimpleNamespace(mode="prefill"))
    m.layers[0]["self_attn"].o_proj(torch.randn(4, K_O).to(torch.bfloat16))
    monkeypatch.setattr(pa, "current_context", lambda: types.SimpleNamespace(mode="decode"))
    m.layers[0]["self_attn"].o_proj(torch.randn(5, K_O).to(torch.bfloat16))
    assert dl.engagement() == {"o_proj|M4|prefill|pinned": 1, "o_proj|M5|decode|passed": 1}


def test_a_captured_bucket_without_a_pin_is_refused(monkeypatch):
    monkeypatch.setattr(dl, "LT_PINS", _pins(ms=(2, 8)))
    with pytest.raises(RuntimeError, match=r"o_proj \(N 32, K 64\) has no pin for the captured decode bucket\(s\) \[4\]"):
        _apply(Model(), "o_proj")
    _apply(Model(), "o_proj", buckets=(1, 2, 8, 16))                                   # 4 not captured: no refusal


def test_another_cublaslt_version_finds_no_key(monkeypatch):
    monkeypatch.setattr(dl, "LT_PINS", _pins(version=VERSION + 1))
    with pytest.raises(RuntimeError, match=r"has no pin for the captured decode bucket\(s\) \[2, 4, 8\] under key "
                                           rf"\(cuBLASLt {VERSION}"):
        _apply(Model(), "o_proj")


def test_a_configuration_cublaslt_rejects_is_refused(monkeypatch):
    monkeypatch.setattr(dl, "LT_PINS", _pins(cfg=dict(CFG, algo_id=999)))
    with pytest.raises(RuntimeError, match=r"cuBLASLt rejects the pinned configuration"):
        _apply(Model(), "o_proj")


@pytest.mark.parametrize("bad", [dict(CFG, reduction_scheme=1), {k: v for k, v in CFG.items() if k != "stages"}],
                         ids=["inplace", "malformed"])
def test_an_inplace_or_malformed_pin_is_refused(monkeypatch, bad):
    monkeypatch.setattr(dl, "LT_PINS", _pins(cfg=bad))
    with pytest.raises(RuntimeError, match=r"malformed or uses a reduction scheme outside \(0, 2, 4\)"):
        _apply(Model(), "o_proj")


def test_a_biased_or_foreign_o_proj_is_refused(monkeypatch):
    monkeypatch.setattr(dl, "LT_PINS", _pins())
    with pytest.raises(RuntimeError, match=r"o_proj is Linear \(bias True\)"):
        _apply(Model(bias=True), "o_proj")
    m = Model()
    m.layers[0]["self_attn"].o_proj = type("Int8Linear", (nn.Module,), {})()
    with pytest.raises(RuntimeError, match=r"o_proj is Int8Linear"):
        _apply(m, "o_proj")


def test_k35s_one_row_module_composes(monkeypatch):
    """E4B_DENSE_GEMV_BW swaps o_proj for its one-row module; the pin wraps it, and one row still reaches it."""
    monkeypatch.setattr(dl, "LT_PINS", _pins())
    m = Model()
    hits = []

    class DenseGemvLinear(nn.Module):
        def __init__(self, src):
            super().__init__()
            self.in_features, self.out_features, self.weight = src.in_features, src.out_features, src.weight

        def forward(self, x):
            hits.append(x.shape[0])
            return F.linear(x, self.weight)
    for layer in m.layers:
        layer["self_attn"].o_proj = DenseGemvLinear(layer["self_attn"].o_proj)
    _apply(m, "o_proj")
    lin = m.layers[0]["self_attn"].o_proj
    lin(torch.randn(1, K_O).to(torch.bfloat16))
    lin(torch.randn(2, K_O).to(torch.bfloat16))
    assert hits == [1] and [c[2] for c in FakeLib.calls] == [2]


def test_the_router_routes_only_on_the_epilogues_forward(monkeypatch):
    monkeypatch.setattr(dl, "LT_PINS", _pins())
    m = Model()
    rec = _apply(m, "router")
    assert rec["modules"] == {"router": 2}
    r = m.layers[0]["gate"]
    for M in (1, 2, 3):
        x = torch.randn(M, K_R).to(torch.bfloat16)
        torch.testing.assert_close(dl.router_linear(r, x), F.linear(x, r.weight), rtol=0, atol=0)
    assert [c[:3] for c in FakeLib.calls] == [(N_R, K_R, 2)]
    for kw, why in (({"kind": None}, r"kind None"), ({"kind": "topk_softmax"}, r"kind 'topk_softmax'"),
                    ({"rbias": torch.zeros(N_R)}, r"bias True")):
        with pytest.raises(RuntimeError, match=rf"router named, but layers\.0\.gate is not on the router epilogue.*{why}"):
            _apply(Model(**kw), "router")


def test_without_a_pin_the_router_gemm_is_f_linear_exactly():
    r = FakeRouter()
    x = torch.randn(4, K_R).to(torch.bfloat16)
    torch.testing.assert_close(dl.router_linear(r, x), F.linear(x, r.weight), rtol=0, atol=0)


def test_a_non_bf16_operand_at_a_routed_call_is_refused(monkeypatch):
    monkeypatch.setattr(dl, "LT_PINS", _pins())
    m = Model()
    _apply(m, "router")
    r = m.layers[0]["gate"]
    with pytest.raises(RuntimeError, match=r"router got 2 rows as torch.float32 .* bf16 operands"):
        dl.router_linear(r, torch.randn(2, K_R))
    dl.router_linear(r, torch.randn(1, K_R).to(torch.bfloat16))                       # one row: not the pin's


def test_a_call_on_a_second_stream_is_refused(monkeypatch, fake_cuda):
    monkeypatch.setattr(dl, "LT_PINS", _pins())
    m = Model()
    _apply(m, "o_proj")
    fake_cuda["stream"] = Stream(1)
    with pytest.raises(RuntimeError, match=r"called on a stream other than the one recorded at build"):
        m.layers[0]["self_attn"].o_proj(torch.randn(2, K_O).to(torch.bfloat16))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)       # a capture records, never runs
    m.layers[0]["self_attn"].o_proj(torch.randn(2, K_O).to(torch.bfloat16))


def test_serve_paged_resolves_it_before_capture_and_reports_it():
    import pathlib
    src = (pathlib.Path(dl.__file__).parents[1] / "serve_paged.py").read_text(encoding="utf-8")
    at = [src.index(x) for x in ("dense_gemv = apply_dense_gemv_bw(", "spec_info = _enable_spec(",
                                 "dense_lt = apply_dense_lt_pin(", "graph_status = runner.enable_decode_graphs(")]
    assert at == sorted(at)
    assert "sorted({*cfg.buckets, *runner._verify_buckets}) if cfg.graphs else ()" in src
    assert 'info["dense_lt_pin"] = {' in src and '"engagement": dense_lt_engagement()}' in src


def test_the_router_epilogue_routes_its_gemm_and_marks_its_kind():
    import pathlib
    src = (pathlib.Path(dl.__file__).parent / "router_epilogue.py").read_text(encoding="utf-8")
    assert "from .dense_lt_pin import router_linear" in src
    assert "logits = router_linear(_m, rows)" in src and src.count("router_linear(") == 1   # the one GEMM call
    assert "mod._e4b_router_epi_kind = kind" in src
