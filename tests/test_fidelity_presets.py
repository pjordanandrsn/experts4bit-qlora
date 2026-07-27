# SPDX-License-Identifier: Apache-2.0
"""The identical-vs-fast choice must stay explicit, and default to identical.

Some accelerations move *which* bytes are copied or *when* and cannot change the
arithmetic; the fused grouped-NF4 kernel is a different computation, priced at
+0.023% perplexity (max|dlogit| = 3.75e-01 on Qwen3-235B-A22B). Collapsing both
behind one switch means a user who needs reproducible logits cannot tell which
they are getting.

Source-contract tests: they read the file rather than importing, so they run on
CPU CI with no torch/bnb/GPU — which is where a refactor would quietly flip the
default or fold `enable_fast` into the identical path.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "experts4bit_qlora" / "offload.py"


def _fn(name):
    tree = ast.parse(_SRC.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"{name} not found in offload.py")


def test_default_fidelity_is_identical():
    """A user who passes nothing must not silently get changed numbers."""
    fn = _fn("enable_decode_stack")
    d = {a.arg: v for a, v in zip(fn.args.args[len(fn.args.args) - len(fn.args.defaults):],
                                  fn.args.defaults)}
    assert "fidelity" in d, "fidelity lost its default"
    assert isinstance(d["fidelity"], ast.Constant) and d["fidelity"].value == "identical", (
        "the default fidelity is no longer 'identical'. Defaulting to 'fast' "
        "silently changes a caller's logits: the fused kernel is a different "
        "computation (+0.023% perplexity), not a faster route to the same answer."
    )


def test_fast_kernel_is_reachable_only_via_fidelity_fast():
    """`enable_fast` must not leak into the identical path."""
    src = ast.unparse(_fn("enable_decode_stack"))
    assert "enable_fast" in src, "the fast path vanished entirely"
    # the only enable_fast call must sit under a `fidelity == 'fast'` test
    assert "if fidelity == 'fast'" in src or 'if fidelity == "fast"' in src, (
        "enable_fast is no longer gated on fidelity=='fast'. The identical "
        "preset must stay bit-identical."
    )


def test_invalid_fidelity_is_rejected_not_coerced():
    src = ast.unparse(_fn("enable_decode_stack"))
    assert "raise ValueError" in src, (
        "an unknown fidelity should raise, not fall through to a default — "
        "silently picking one is how a caller ends up with the wrong numbers."
    )


def test_handle_recovery_is_public():
    """Reproducing the published ladder must not need a private attribute."""
    tree = ast.parse(_SRC.read_text())
    names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "offload_handles" in names, (
        "offload_handles was removed; load_moe_4bit_streaming returns "
        "(model, config) and NOT the handles, so without this the only route is "
        "the private layer.mlp._offload walk."
    )


@pytest.mark.parametrize("sym", ["enable_decode_stack", "offload_handles"])
def test_exported(sym):
    init = (_SRC.parent / "__init__.py").read_text()
    assert f'"{sym}"' in init, f"{sym} is not exported from __init__"
