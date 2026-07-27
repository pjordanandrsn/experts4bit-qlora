# SPDX-License-Identifier: Apache-2.0
"""The grouped kernel and routed/speculative staging must keep composing.

They are orthogonal by construction -- ``enable_fast`` swaps a module's
``forward`` (compute) while the staging policies drive the offload handle's
transfer state -- so neither guards against the other. That orthogonality is
held together only by a *positional* contract, and violating it fails
SILENTLY: routed staging drops back to bulk, moving E/top_k = 16x the bytes,
with no exception and no wrong answer. Nothing else in the suite catches it.

The measured cost of that silent fallback is the first rung of the ladder:
5.9041 -> 1.0917 s/token on Qwen3-235B-A22B (grouped-nf4-gemm finding #42).

This reads the SOURCE rather than importing, so it needs neither torch,
bitsandbytes, nor a GPU -- the contract is static, and a test that skips on
CPU CI would not defend it where it actually gets broken.
"""
from __future__ import annotations

import ast
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent / "experts4bit_qlora"


def _func(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path.name} -- renamed?")


def test_top_k_index_is_the_second_positional_arg():
    """``offload.py`` reads the pre-hook's ``args[1]`` as ``top_k_index``.

    A forward pre-hook sees the *positional* args of the call: ``args[0]`` is
    ``hidden_states`` and ``args[1]`` must be ``top_k_index``. Reordering the
    patched forward's parameters silently disables routed staging.
    """
    fn = _func(_PKG / "fast.py", "fused_experts_lora_forward")
    names = [a.arg for a in fn.args.args]
    assert names[:3] == ["mod", "hidden_states", "top_k_index"], (
        f"signature is {names[:3]}; routed staging reads args[1] of the forward "
        "pre-hook as top_k_index (offload.py, the `_routed_only` branch). Moving "
        "this parameter makes staging fall back to bulk SILENTLY -- 16x the "
        "bytes, no error, no wrong answer."
    )


def test_top_k_index_is_not_keyword_only():
    """A keyword-only ``top_k_index`` would never appear in ``args`` at all."""
    fn = _func(_PKG / "fast.py", "fused_experts_lora_forward")
    kwonly = [a.arg for a in fn.args.kwonlyargs]
    assert "top_k_index" not in kwonly, (
        "top_k_index is keyword-only; the pre-hook's `len(args) > 1` test would "
        "be False and routed staging would silently take the bulk path."
    )


def test_the_hook_still_reads_args_index_1():
    """The other half of the contract, asserted where it is consumed."""
    src = (_PKG / "offload.py").read_text()
    assert "torch.is_tensor(args[1])" in src, (
        "the routed-staging pre-hook no longer reads args[1]. If the contract "
        "moved, update test_top_k_index_is_the_second_positional_arg to match."
    )


def test_the_two_policies_do_not_guard_against_each_other():
    """Unlike hot_residency/pipelined, staging composes with the kernel.

    ``enable_fast`` deliberately skips ``_hot_residency`` and ``_e4b_pipe_ref``
    modules. It must NOT grow a similar skip for the staging attributes: the
    measured 7.04x -> 9.19x rung depends on composing them.
    """
    fn = _func(_PKG / "fast.py", "enable_fast")
    src = ast.unparse(fn)
    for attr in ("_spec_dev", "_spec_ids", "_routed_only"):
        assert attr not in src, (
            f"enable_fast now inspects {attr!r}. The grouped kernel and staging "
            "are orthogonal (compute vs transfer) and composing them is the "
            "third rung of the ladder -- do not add a guard without re-measuring."
        )
