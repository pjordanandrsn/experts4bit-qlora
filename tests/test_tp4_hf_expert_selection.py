# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""#542: the tp4 HF arm must select expert parameters BY STRUCTURE, and must REFUSE an empty selection.

The defect this file guards is not a crash -- it is a comparator that quietly becomes a different experiment.
``bench/tp4/tp4_arm.py::hf_targets`` picked PEFT's ``target_parameters`` with ``p.ndim == 3 and "experts" in name``:
a NAME substring, the shape #426/#435 already removed from the attention path. A family that calls the module
something else (GraniteMoe's own ``block_sparse_moe.input_linear`` / ``output_linear``) selects NOTHING, PEFT adapts
attention only, and the arm trains something other than what the head-to-head claims while every status stays "ok".

Each test below builds the storage layout by hand, so it runs against the pre-fix code too:

* ``test_granite_named_stacks_are_selected``   FAILS before the fix (0 selected, silently)
* ``test_no_expert_stack_refuses``             FAILS before the fix (empty list returned, no refusal)
* ``test_partial_selection_refuses``           FAILS before the fix (a half-adapted arm runs)
* ``test_fused_layout_selection_is_unchanged`` PASSES both ways -- it is the regression guard for the families the
  running lane measured (Qwen3 / Gemma-4 / Granite / OLMoE all present fused stacks under a module named ``experts``
  on transformers 5.17), NOT evidence for the fix.
"""

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
ARM_PATH = REPO / "bench" / "tp4" / "tp4_arm.py"


def _arm():
    spec = importlib.util.spec_from_file_location("tp4_arm_under_test", ARM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ARM = _arm()
_TAKES_LAYERS = len(inspect.signature(ARM.hf_targets).parameters) >= 3


def _select(model, n_layers=None, model_type=None):
    """The expert parameter names the arm hands PEFT, across the pre-#542 and post-#542 signatures."""
    out = ARM.hf_targets(model, n_layers, model_type) if _TAKES_LAYERS else ARM.hf_targets(model)
    return list(out[1])


def _model(layout, n_layers=3, n_experts=4, hidden=8, drop_last_stack=False):
    """A decoder whose MoE block stores its experts the way one of this lane's families does. Only the names and the
    storage shape differ between layouts; the attention projections are identical in all of them."""
    blocks = []
    for i in range(n_layers):
        b = nn.Module()
        b.self_attn = nn.Module()
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(b.self_attn, p, nn.Linear(hidden, hidden, bias=False))
        blk = nn.Module()
        blk.gate = nn.Linear(hidden, n_experts, bias=False)
        if layout == "fused_experts":          # transformers v5 fuses these under a module literally named `experts`
            blk.experts = nn.Module()
            blk.experts.gate_up_proj = nn.Parameter(torch.zeros(n_experts, hidden, 2 * hidden))
            if not (drop_last_stack and i == n_layers - 1):
                blk.experts.down_proj = nn.Parameter(torch.zeros(n_experts, 2 * hidden, hidden))
        elif layout == "granite_on_disk":      # GraniteMoe's OWN names: no `experts` component anywhere in the path
            blk.input_linear = nn.Module()
            blk.input_linear.weight = nn.Parameter(torch.zeros(n_experts, 2 * hidden, hidden))
            blk.output_linear = nn.Module()
            blk.output_linear.weight = nn.Parameter(torch.zeros(n_experts, hidden, hidden))
        elif layout == "per_expert_2d":        # OLMoE / Mixtral as stored: one 2-D Linear triple per expert
            blk.experts = nn.ModuleList()
            for _ in range(n_experts):
                e = nn.Module()
                e.gate_proj = nn.Linear(hidden, hidden, bias=False)
                e.up_proj = nn.Linear(hidden, hidden, bias=False)
                e.down_proj = nn.Linear(hidden, hidden, bias=False)
                blk.experts.append(e)
        elif layout == "dense":                # not an MoE at all
            blk.up_proj = nn.Linear(hidden, 2 * hidden, bias=False)
            blk.down_proj = nn.Linear(2 * hidden, hidden, bias=False)
        else:
            raise AssertionError(layout)
        b.mlp = blk
        blocks.append(b)
    inner = nn.Module()
    inner.layers = nn.ModuleList(blocks)
    m = nn.Module()
    m.model = inner
    m.config = type("Cfg", (), {"num_hidden_layers": n_layers, "num_experts": n_experts, "model_type": layout})()
    return m


def test_granite_named_stacks_are_selected():
    """The defect itself: 3-D expert stacks whose path carries no `experts` component are still expert stacks."""
    params = _select(_model("granite_on_disk"), 3, "granitemoe")
    assert len(params) == 6, f"expected 2 stacks x 3 layers, got {len(params)}: {params}"
    assert not any("experts" in n for n in params)          # nothing here would have matched the old substring
    assert not any(ARM.EXPERT_PARAM_RE.search(n) for n in params)   # nor the regex the issue proposed reusing


def test_no_expert_stack_refuses():
    """An empty selection must ABORT. Returning [] makes PEFT adapt attention only -- a different experiment that
    reports itself as the registered one."""
    for layout in ("per_expert_2d", "dense"):
        with pytest.raises(NotImplementedError) as ei:
            _select(_model(layout), 3, layout)
        msg = str(ei.value)
        assert "#542" in msg and "n_layers=3" in msg and layout in msg, msg   # names the family and what it looked for


def test_partial_selection_refuses():
    """5 stacks over 3 layers is not a whole number per layer: some layer went unadapted, so the arm refuses."""
    with pytest.raises(NotImplementedError) as ei:
        _select(_model("fused_experts", drop_last_stack=True), 3, "qwen3_moe")
    assert "whole number per layer" in str(ei.value), str(ei.value)


def test_fused_layout_selection_is_unchanged():
    """Regression guard, not evidence: every family the running lane measured presents fused stacks under a module
    named `experts`, and the structural rule must select exactly what the substring did -- same names, same order."""
    model = _model("fused_experts")
    legacy = [n for n, p in model.named_parameters() if p.ndim == 3 and "experts" in n]
    assert _select(model, 3, "qwen3_moe") == legacy and len(legacy) == 6


def test_a_refusal_is_recorded_as_a_refused_row():
    """The refusal rides the classifier run_arm already uses: a `refused` receipt with exit 3, never a silent arm."""
    assert ARM.classify_load_exception(NotImplementedError("x")) == ("refused", 3)
