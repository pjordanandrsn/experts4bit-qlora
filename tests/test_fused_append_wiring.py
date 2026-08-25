# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Wiring for AMENDMENT-f1-stageB-b2's fused append opt-in.

The KERNEL's bitwise gate lives in gnf4 (GPU-only by design). This
covers what must hold on any machine: the flag defaults OFF, reads the
env at construction, and the fused branch routes to the kernel with the
pool/side arguments in the right order — a swapped k/v row size would
corrupt the cache while still passing a smoke run that never reads it
back.
"""

import inspect

import pytest

torch = pytest.importorskip(
    "torch", reason="engine import needs torch (CI installs CPU torch; "
                    "the sibling suites use the same guard)")
mod = pytest.importorskip("experts4bit_qlora.engines.fp8_paged_kv")

_CLS = mod.Fp8PagedKV


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("E4B_FUSED_KV_APPEND", raising=False)
    src = inspect.getsource(mod)
    assert 'os.environ.get(\n            "E4B_FUSED_KV_APPEND", "0") == "1"' \
        in src, "the opt-in must default OFF until B2 RESULTS merge"


@pytest.mark.parametrize("env,expect", [("1", True), ("0", False),
                                        (None, False)])
def test_flag_reads_env_at_construction(monkeypatch, env, expect):
    if env is None:
        monkeypatch.delenv("E4B_FUSED_KV_APPEND", raising=False)
    else:
        monkeypatch.setenv("E4B_FUSED_KV_APPEND", env)
    kv = _CLS(n_layers=1, n_kv_heads=1, head_dim=64, batch=1,
              max_tokens_per_seq=8, device="cpu")
    assert kv._fused_append is expect


def test_fused_branch_argument_order():
    """The fused branch must pass V's row/payload with V's pool and K's
    with K's — read the source rather than executing CUDA. A transposed
    pair would write K bytes at V strides: silent cache corruption."""
    src = inspect.getsource(_CLS.append_graph_t1)
    v_call = src.split("fp8_kv_append_t1(")[1]
    k_call = src.split("fp8_kv_append_t1(")[2]
    assert "_g_vflat" in v_call and "self.v_row" in v_call \
        and "self._v_pay" in v_call and ", 1)" in v_call
    assert "_g_kflat" in k_call and "self.k_row" in k_call \
        and "self._k_pay" in k_call and "self.k_groups)" in k_call
    # the publish must remain OUTSIDE the kernel, after both sides
    after_k = src.split("fp8_kv_append_t1(")[2]
    assert ".add_(1)" in after_k, \
        "seq_lens publish must follow both fused writes"
    # and the fused branch must not fall through into the eager body
    fused_block = src.split("if self._fused_append:")[1].split(
        "vq, vs = self._quant_bytes")[0]
    assert "return" in fused_block, \
        "fused branch must return, not run the eager path too"
