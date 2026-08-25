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


def test_flag_defaults_on_with_env_rollback(monkeypatch):
    """B2 certified (RESULTS-f1-stageB-b2): the fused append is the
    default and the env is the rollback, not the opt-in."""
    monkeypatch.delenv("E4B_FUSED_KV_APPEND", raising=False)
    src = inspect.getsource(mod)
    assert 'os.environ.get(\n            "E4B_FUSED_KV_APPEND", "1") == "1"' \
        in src, "certified default must be ON with env rollback"


class _StubWithKernel:
    """A post-#253 fp8_kv surface: kernel symbol present."""
    E4M3_MAX = 448.0

    @staticmethod
    def fp8_kv_append_t1(*a, **k):
        raise AssertionError("flag test must not launch the kernel")


@pytest.mark.parametrize("env,expect", [("1", True), ("0", False),
                                        (None, True)])
def test_flag_reads_env_at_construction(monkeypatch, env, expect):
    """Flag semantics in isolation: a kernel-PRESENT stub, so the test
    holds on machines whose installed gnf4 predates #253 (CI installs
    from PyPI) as well as on the boxes."""
    import sys
    if env is None:
        monkeypatch.delenv("E4B_FUSED_KV_APPEND", raising=False)
    else:
        monkeypatch.setenv("E4B_FUSED_KV_APPEND", env)
    monkeypatch.setitem(sys.modules, "fp8_kv", _StubWithKernel())
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


class _StubFp8Kv:
    """A pre-#253 fp8_kv: has the torch surface, lacks the kernel."""
    E4M3_MAX = 448.0


def test_old_gnf4_degrades_to_eager_by_default(monkeypatch):
    """On an install whose gnf4 predates the kernel, the certified
    default must fall back to the intact eager path -- not crash graph
    decode with ImportError (Bugbot, e4b#238)."""
    import sys
    monkeypatch.delenv("E4B_FUSED_KV_APPEND", raising=False)
    monkeypatch.setitem(sys.modules, "fp8_kv", _StubFp8Kv())
    kv = _CLS(n_layers=1, n_kv_heads=1, head_dim=64, batch=1,
              max_tokens_per_seq=8, device="cpu")
    assert kv._fused_append is False


def test_old_gnf4_with_explicit_env_refuses_loudly(monkeypatch):
    """An EXPLICIT =1 on a kernel-less install must raise, never
    silently serve the eager path the caller opted out of."""
    import sys
    monkeypatch.setenv("E4B_FUSED_KV_APPEND", "1")
    monkeypatch.setitem(sys.modules, "fp8_kv", _StubFp8Kv())
    with pytest.raises(RuntimeError, match="gnf4#253"):
        _CLS(n_layers=1, n_kv_heads=1, head_dim=64, batch=1,
             max_tokens_per_seq=8, device="cpu")
