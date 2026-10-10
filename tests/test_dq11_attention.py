"""Actual CPU SDPA engagement plus refusal controls; no synthetic operator claims."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).parents[1] / "bench/dq11"))
import dq11_attention as attention


@pytest.fixture
def math_events():
    q, k, v = [torch.randn(1, 2, 8, 16, requires_grad=True) for _ in range(3)]
    with sdpa_kernel(SDPBackend.MATH), profile(activities=[ProfilerActivity.CPU]) as profiler:
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        output.square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    return [{"operator": event.name, "parent": event.cpu_parent.name if event.cpu_parent else None}
            for event in profiler.events()]


def test_actual_sdpa_math_forward_and_decomposed_backward(math_events):
    row = attention.dispatched(math_events, math_required=True)
    assert row["backend"] == "math"
    assert row["forward"] == ["aten::_scaled_dot_product_attention_math"]
    assert "aten::_softmax_backward_data" in row["backward"] and "aten::bmm" in row["backward"]


@pytest.mark.parametrize("missing", ["aten::_scaled_dot_product_attention_math", "SafeSoftmaxBackward0",
                                     "BmmBackward0", "aten::_softmax_backward_data"])
def test_forward_only_or_partial_backward_cannot_pass(math_events, missing):
    with pytest.raises(ValueError):
        attention.dispatched([row for row in math_events if row["operator"] != missing], math_required=True)


def test_unrelated_math_backward_cannot_supply_attention_witness(math_events):
    events = copy.deepcopy(math_events)
    for row in events:
        if row["operator"] == "aten::bmm":
            row["parent"] = "unrelated model matmul"
    with pytest.raises(ValueError, match="backward"):
        attention.dispatched(events, math_required=True)


def test_xformers_reentry_is_refused_despite_valid_sdpa_events(math_events):
    with pytest.raises(ValueError, match="optional attention"):
        attention.dispatched([*math_events, {"operator": "xformers::efficient_attention_backward", "parent": None}],
                             math_required=True)


@pytest.mark.parametrize("present", ["xformers", "flash_attn"])
def test_optional_package_presence_refuses_before_model_import(monkeypatch, present):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == present else None)
    with pytest.raises(ValueError, match="unimportable"):
        attention.require_install_set()
