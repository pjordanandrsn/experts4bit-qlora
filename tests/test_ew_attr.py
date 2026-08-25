# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""F1 Stage A attribution helpers (PREREG-f1-elementwise).

step_decomp.py is a bench script, not a package module, and importing
it drags in the serving stack -- so the pure helpers are lifted out by
AST and exercised directly. That keeps them under `pytest tests/`,
which CI actually runs (gnf4#245: a test file nothing invokes is not a
test)."""

import ast
import pathlib
import types

import pytest

SRC = (pathlib.Path(__file__).resolve().parents[1]
       / "bench" / "hybrid-g9" / "step_decomp.py")
WANT_FUNCS = ("_py_site", "_self_device_us", "_ew_attribute")
WANT_CONSTS = ("_EW_OPS", "_FRAME_SKIP", "_NON_EW_OPS")


@pytest.fixture(scope="module")
def helpers():
    tree = ast.parse(SRC.read_text())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in WANT_FUNCS)
            or (isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") in WANT_CONSTS)]
    got = {n.name for n in keep if isinstance(n, ast.FunctionDef)} | {
        n.targets[0].id for n in keep if isinstance(n, ast.Assign)}
    missing = set(WANT_FUNCS + WANT_CONSTS) - got
    assert not missing, (
        f"step_decomp.py no longer defines {sorted(missing)} at module "
        "level -- this test silently covers nothing if it cannot find "
        "them, so it fails loudly instead")
    mod = types.ModuleType("ew_attr_helpers")
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec"),
         mod.__dict__)
    return mod


def test_picks_our_frame_over_torch_internals(helpers):
    stack = [
        "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/"
        "module.py(15): _call_impl",
        "/root/e4b/experts4bit_qlora/engines/hot_residency.py(92): "
        "_forward_collapsed",
        "/root/e4b/bench/hybrid-g9/step_decomp.py(700): run",
    ]
    site = helpers._py_site(stack)
    assert "hot_residency.py(92)" in site, site


def test_harness_frame_is_never_the_answer(helpers):
    """step_decomp.py is the profiler's own caller: attributing the
    block to the harness would point every fix at the wrong file."""
    stack = ["/root/e4b/bench/hybrid-g9/step_decomp.py(700): run",
             "/root/e4b/experts4bit_qlora/engines/hybrid.py(41): forward"]
    assert "hybrid.py(41)" in helpers._py_site(stack)


def test_torch_only_stack_falls_back_visibly(helpers):
    stack = ["/site-packages/torch/nn/functional.py(2): silu"]
    assert helpers._py_site(stack) == stack[-1]


@pytest.mark.parametrize("empty", [[], None])
def test_empty_stack_is_labelled_not_blank(helpers, empty):
    assert helpers._py_site(empty) == "<no-python-frame>"


def test_device_time_reader_spans_torch_versions(helpers):
    new = type("E", (), {"self_device_time_total": 5.0})()
    old = type("E", (), {"self_cuda_time_total": 7.0})()
    neither = type("E", (), {})()
    assert helpers._self_device_us(new) == 5.0
    assert helpers._self_device_us(old) == 7.0
    assert helpers._self_device_us(neither) == 0.0


def test_op_set_is_elementwise_only(helpers):
    """A matmul leaking into the elementwise set would inflate the block
    F1 claims to remove."""
    for op in ("aten::copy_", "aten::mul", "aten::rsqrt", "aten::silu"):
        assert op in helpers._EW_OPS
    for op in ("aten::mm", "aten::matmul", "aten::linear", "aten::bmm",
               "aten::topk"):
        assert op not in helpers._EW_OPS


class _Evt:
    def __init__(self, key, us, count=1, stack=None):
        self.key = key
        self.self_device_time_total = us
        self.count = count
        self.stack = stack or []


def test_census_ops_are_all_covered(helpers):
    """Every aten op that carried >=50 us/step on the census receipt must
    land in the elementwise block. These five (447 us/step combined, the
    fp8 KV scale path plus scatter/fill helpers) were omitted from the
    first draft and would have shrunk the block Stage A attributes."""
    for op in ("aten::amax", "aten::abs", "aten::gt", "aten::fill_",
               "aten::index_add_"):
        assert op in helpers._EW_OPS, op


def test_unknown_op_is_reported_not_dropped(helpers):
    """The durable guard: a curated list drifts, so an op in NEITHER set
    must surface as `unclassified` instead of vanishing."""
    out = helpers._ew_attribute([
        _Evt("aten::mul", 100.0, 2, ["/e4b/engines/hot_residency.py(92): f"]),
        _Evt("aten::some_future_op", 250.0, 4),
        _Evt("aten::mm", 9999.0, 1),          # deliberately excluded
    ])
    assert out["attributed_us"] == 100.0
    assert "aten::some_future_op" in out["unclassified_ops"], out
    assert out["unclassified_ops"]["aten::some_future_op"]["us"] == 250.0
    assert "aten::mm" not in out["unclassified_ops"], "excluded != unknown"


def test_attribution_groups_by_call_site(helpers):
    site = "/e4b/experts4bit_qlora/engines/hot_residency.py(92): fwd"
    torch_frame = "/site-packages/torch/nn/modules/module.py(1): _call"
    out = helpers._ew_attribute([
        _Evt("aten::mul", 10.0, 1, [torch_frame, site]),
        _Evt("aten::add", 5.0, 2, [torch_frame, site]),
        _Evt("aten::silu", 7.0, 1, [torch_frame,
                                    "/e4b/engines/hybrid.py(4): g"]),
    ])
    assert out["attributed_us"] == 22.0
    assert out["by_site"][site]["us"] == 15.0
    assert out["by_site"][site]["calls"] == 3
    assert out["by_site"][site]["ops"] == {"aten::mul": 10.0,
                                           "aten::add": 5.0}
    # sorted by cost so the receipt leads with the site worth fixing
    assert list(out["by_site"])[0] == site


def test_zero_device_time_events_are_ignored(helpers):
    """CPU-only dispatch rows must not create phantom call sites."""
    out = helpers._ew_attribute([_Evt("aten::mul", 0.0, 5, ["/e4b/a.py(1): f"])])
    assert out["attributed_us"] == 0.0 and not out["by_site"]


def test_op_sets_are_disjoint(helpers):
    assert not (helpers._EW_OPS & helpers._NON_EW_OPS)
