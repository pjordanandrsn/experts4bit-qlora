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


def test_cpu_construction_degrades_even_with_kernel(monkeypatch):
    """The regression that hit e4b#251 CI the hour gnf4 0.15.0 reached
    PyPI: the kernel IMPORTS fine on a CPU host, but launching it dies
    in triton's driver. A cpu-device KV must degrade to eager no
    matter what is installed."""
    import sys
    monkeypatch.delenv("E4B_FUSED_KV_APPEND", raising=False)
    monkeypatch.setitem(sys.modules, "fp8_kv", _stub(with_kernel=True))
    kv = _CLS(n_layers=1, n_kv_heads=1, head_dim=64, batch=1,
              max_tokens_per_seq=8, device="cpu")
    assert kv._fused_append is False


def test_resolver_cell_table():
    """B2 certified (RESULTS-f1-stageB-b2): fused is the default ON A
    CUDA DEVICE and the env is the rollback; every degrade has a
    loud-refuse twin under an explicit =1. Behavioral, on the factored
    resolver, so all cells run on CPU CI (the first version asserted
    the default via a cpu construction, which the e4b#251 fix
    correctly turned into a degrade)."""
    resolve = mod._resolve_fused_append
    present, absent = (lambda: True), (lambda: False)
    assert resolve(None, "cuda", present) is True      # certified default
    assert resolve(None, "cuda:1", present) is True
    assert resolve("0", "cuda", present) is False      # rollback
    assert resolve(None, "cpu", present) is False      # device degrade
    assert resolve(None, "cuda", absent) is False      # install degrade
    with pytest.raises(RuntimeError, match="device 'cpu'"):
        resolve("1", "cpu", present)
    with pytest.raises(RuntimeError, match="no fp8_kv_append_t1"):
        resolve("1", "cuda", absent)
    assert resolve("1", "cuda", present) is True


def _stub(with_kernel):
    """A stand-in fp8_kv module that DELEGATES to the real one and
    varies only the kernel symbol's presence. The first stubs replaced
    the module wholesale and omitted kv_block_bytes, so construction
    ImportError'd before ever reaching the logic under test -- the
    tests "covered" a path they never executed (Bugbot, e4b#238)."""
    import fp8_kv as real

    class _Stub:
        def __getattr__(self, name):
            if name == "fp8_kv_append_t1":
                if with_kernel:
                    def _kernel(*a, **k):
                        raise AssertionError(
                            "wiring test must not launch the kernel")
                    return _kernel
                raise AttributeError(name)
            return getattr(real, name)

    return _Stub()


@pytest.mark.parametrize("env,expect", [("0", False), (None, False)])
def test_flag_reads_env_at_construction(monkeypatch, env, expect):
    """Construction reads the env, on a cpu KV (what CI can build):
    "0" and unset both land eager here -- unset because the device
    gate degrades a non-cuda KV regardless of the certified ON
    default. The ON cells (cuda device) live in
    test_resolver_cell_table, which needs no cuda tensor to run."""
    import sys
    if env is None:
        monkeypatch.delenv("E4B_FUSED_KV_APPEND", raising=False)
    else:
        monkeypatch.setenv("E4B_FUSED_KV_APPEND", env)
    monkeypatch.setitem(sys.modules, "fp8_kv", _stub(with_kernel=True))
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


def test_old_gnf4_degrades_to_eager_by_default(monkeypatch):
    """On an install whose gnf4 predates the kernel, the certified
    default must fall back to the intact eager path -- not crash graph
    decode with ImportError (Bugbot, e4b#238)."""
    import sys
    monkeypatch.delenv("E4B_FUSED_KV_APPEND", raising=False)
    monkeypatch.setitem(sys.modules, "fp8_kv", _stub(with_kernel=False))
    kv = _CLS(n_layers=1, n_kv_heads=1, head_dim=64, batch=1,
              max_tokens_per_seq=8, device="cpu")
    assert kv._fused_append is False


def test_explicit_env_on_cpu_kv_refuses_loudly(monkeypatch):
    """An EXPLICIT =1 must raise when it cannot be honored, never
    silently serve the eager path the caller opted out of. On a cpu
    KV the DEVICE gate refuses first (correct precedence -- the
    kernel-missing refusal for a cuda KV is covered in
    test_resolver_cell_table)."""
    import sys
    monkeypatch.setenv("E4B_FUSED_KV_APPEND", "1")
    monkeypatch.setitem(sys.modules, "fp8_kv", _stub(with_kernel=False))
    with pytest.raises(RuntimeError, match="CUDA"):
        _CLS(n_layers=1, n_kv_heads=1, head_dim=64, batch=1,
             max_tokens_per_seq=8, device="cpu")
