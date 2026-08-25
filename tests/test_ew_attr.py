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
WANT_FUNCS = ("_py_site", "_self_device_us")
WANT_CONSTS = ("_EW_OPS", "_FRAME_SKIP")


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
