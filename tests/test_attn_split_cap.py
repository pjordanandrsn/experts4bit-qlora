# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_ATTN_SPLIT_CAP`` (lane P143, Amendment 2; e4b#1313): the paged FP8 attention decode capped at ``min(cap, the
library's own plan)`` splits.

e4b computes no plan: it hands grouped-nf4-gemm's kernel ``split_cap`` (grouped-nf4-gemm#558), and the kernel's own
``decode_split_plan`` plans ``min(cap, plan)``. These tests call that real planner, so they need grouped-nf4-gemm >= 0.47.1.
In CI they never skip: there the import fails loudly. The cap the KV records is ``min(16, plan)`` across B 1..64 at 170 SMs
(RTX 5090) and 188 SMs (RTX PRO 6000), with serve-throughput's exact rows. The cap is handed over only when a cap is
set, the installed kernel takes it, and the compute is fp8; never over an explicit ``n_split``. The server knob's reading
and its per-family resolution: qwen3_moe takes 16 by default on P143 Amendment 3's binding read, every other family keeps
the library's plan."""
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora import serve_paged as sp  # noqa: E402
from experts4bit_qlora.engines import fp8_paged_kv as kvmod  # noqa: E402


def _require(module: str, reason: str):
    """Import ``module``; in CI a plain import (it must be there, never a skip), off CI a skip when it is absent."""
    if os.environ.get("CI"):
        import importlib
        return importlib.import_module(module)
    return pytest.importorskip(module, reason=reason)


def _planner():
    """grouped-nf4-gemm's real ``decode_split_plan``: required in CI, skipped only off it."""
    fp8_paged_attn = _require("fp8_paged_attn", "grouped-nf4-gemm is not installed (CI installs it)")
    fn = getattr(fp8_paged_attn, "decode_split_plan", None)
    if fn is None:
        if os.environ.get("CI"):
            raise AssertionError("CI's grouped-nf4-gemm has no decode_split_plan: the pin must be >= 0.47.1")
        pytest.skip("the installed grouped-nf4-gemm predates decode_split_plan (0.47.1)")
    return fn


def _plan(fn, B, sms, h=4, cap=None):
    return fn(B, h, 256, block_tokens=16, ktile=64, compute="fp8", pack_heads=False, sms=sms, split_cap=cap)


def test_serve_throughputs_rows_from_the_real_planner():
    """At Qwen3's 4 KV heads on a 5090 the plan is 32, 32, 31, 23, 18 at B 1..5 (all capped to 16) and 15 at B 6; on a
    PRO 6000 B 6 plans 17 (capped) and B 7 plans 14. Mixtral's 8 KV heads move at B 1..2 only."""
    fn = _planner()
    assert [_plan(fn, B, 170) for B in range(1, 7)] == [32, 32, 31, 23, 18, 15]
    assert [_plan(fn, B, 170) for B in (8, 16, 32, 64)] == [11, 5, 2, 1]
    assert (_plan(fn, 6, 188), _plan(fn, 7, 188)) == (17, 14)
    assert [_plan(fn, B, 170, h=8) for B in (1, 2, 3)] == [32, 23, 15]


def _recorded(monkeypatch, fn, *, B, sms, cap=16):
    """What the KV records for a call the cap met: Fp8PagedKV._record_split over a fake KV, the real planner behind it."""
    stub = types.ModuleType("fp8_paged_attn")
    stub.decode_split_plan = fn
    monkeypatch.setitem(sys.modules, "fp8_paged_attn", stub)
    monkeypatch.setattr(kvmod, "_sm_count", lambda d: sms)
    fake_kv = types.SimpleNamespace(Hs=[4], Ds=[128], kgs=[1], bt=16, split_cap=cap, split_plans={})
    kvmod.Fp8PagedKV._record_split(fake_kv, 0, torch.zeros(B, 32, 128, dtype=torch.bfloat16),
                                   torch.zeros(B, 256, dtype=torch.int32), {})
    return fake_kv.split_plans[B]


def test_the_cap_runs_min_16_and_the_library_plan_across_B_1_to_64(monkeypatch):
    fn = _planner()
    for sms in (170, 188):
        for B in range(1, 65):
            rec = _recorded(monkeypatch, fn, B=B, sms=sms)
            assert rec["plan"] == _plan(fn, B, sms) and rec["n_split"] == min(16, rec["plan"]), (sms, B, rec)
            assert rec["calls"] == 1


def _kv_with_stub_kernel(monkeypatch, *, takes_split_cap=True, compute="fp8"):
    _require("fp8_kv", "the KV class needs grouped-nf4-gemm (CI installs it)")          # never a skip in CI
    seen = []
    stub = types.ModuleType("fp8_paged_attn")
    if takes_split_cap:
        def fp8_paged_decode_attention(q, kf, vf, tbl, lens, *, n_kv_heads, head_dim, block_tokens, k_groups=1,
                                       k_row_bytes=None, v_row_bytes=None, window=0, sinks=None, split_cap=None, **kw):
            seen.append(dict(kw, split_cap=split_cap))
            return torch.zeros_like(q)
    else:
        def fp8_paged_decode_attention(q, kf, vf, tbl, lens, *, n_kv_heads, head_dim, block_tokens, k_groups=1,
                                       k_row_bytes=None, v_row_bytes=None, window=0, sinks=None, **kw):
            seen.append(dict(kw, split_cap="absent"))
            return torch.zeros_like(q)
    stub.fp8_paged_decode_attention = fp8_paged_decode_attention
    # the stub reads what the wrapper would: a ktile of 48 (not a fp8 tile) or v_groups != 1 makes fp8 unsupported
    stub._compute_default = lambda *a, **k: ("f32" if (k.get("ktile") == 48 or (len(a) > 3 and a[3] != 1)) else compute)
    stub.decode_split_plan = lambda B, h, cols, **k: 32 if k.get("split_cap") is None else min(k["split_cap"], 32)
    monkeypatch.setitem(sys.modules, "fp8_paged_attn", stub)
    monkeypatch.setattr(kvmod, "_sm_count", lambda d: 170)
    # _kernel_takes caches by id(fn): a collected stub's id can come back on the next test's stub with the old answer
    # (order-dependent: the ktile test read no cap when it ran after the no-split_cap stub), so each test starts empty
    monkeypatch.setattr(kvmod, "_KERNEL_KW_CACHE", {})
    kv = kvmod.Fp8PagedKV(1, 4, 128, batch=2, max_tokens_per_seq=4096, device="cpu")
    kv.append(0, 0, torch.randn(8, 4, 128, dtype=torch.bfloat16), torch.randn(8, 4, 128, dtype=torch.bfloat16))
    return kv, seen


def test_the_attention_call_hands_split_cap_only_when_set_and_never_over_an_explicit_n_split(monkeypatch):
    kv, seen = _kv_with_stub_kernel(monkeypatch)
    q = torch.randn(1, 32, 128, dtype=torch.bfloat16)
    kv.attention(0, q, slots=[0])
    kv.split_cap = 16
    kv.attention(0, q, slots=[0])
    kv.attention(0, q, slots=[0], n_split=4)
    assert [(k.get("split_cap"), k.get("n_split")) for k in seen] == [(None, None), (16, None), (None, 4)]
    assert kv.split_plans == {1: {"calls": 1, "plan": 32, "n_split": 16}}


def test_a_kernel_without_split_cap_changes_nothing(monkeypatch):
    kv, seen = _kv_with_stub_kernel(monkeypatch, takes_split_cap=False)
    kv.split_cap = 16
    kv.attention(0, torch.randn(1, 32, 128, dtype=torch.bfloat16), slots=[0])
    assert seen == [{"split_cap": "absent"}] and kv.split_plans == {}


def test_speculative_verify_never_takes_the_cap(monkeypatch):
    """The maintainer's final pass on #1768: lens_override marks the verify path (K + 1 rows over one slot), which
    P143 did not read, so it keeps the library's plan."""
    kv, seen = _kv_with_stub_kernel(monkeypatch)
    kv.split_cap = 16
    q = torch.randn(2, 32, 128, dtype=torch.bfloat16)
    kv.attention(0, q, slots=[0, 0], lens_override=torch.tensor([7, 8], dtype=torch.int32))
    assert seen[0]["split_cap"] is None and kv.split_plans == {}


def test_a_ktile_or_v_groups_that_rule_out_fp8_leaves_the_call_without_the_cap(monkeypatch):
    """The compute verdict is the kernel's own, from the same inputs its wrapper uses: a caller's ktile (or v_groups)
    that makes fp8 unsupported must not hand split_cap to what will run the f32 path."""
    kv, seen = _kv_with_stub_kernel(monkeypatch)
    kv.split_cap = 16
    q = torch.randn(1, 32, 128, dtype=torch.bfloat16)
    kv.attention(0, q, slots=[0], ktile=48)
    kv.attention(0, q, slots=[0], v_groups=2)
    kv.attention(0, q, slots=[0], ktile=64)
    assert [k.get("split_cap") for k in seen] == [None, None, 16]


def test_the_f32_path_is_left_to_the_library(monkeypatch):
    kv, seen = _kv_with_stub_kernel(monkeypatch, compute="f32")
    kv.split_cap = 16
    kv.attention(0, torch.randn(1, 32, 128, dtype=torch.bfloat16), slots=[0])
    assert seen[0]["split_cap"] is None and kv.split_plans == {}


@pytest.mark.parametrize("env,model_type,want_cap,want_source,handed", [
    (None, "qwen3_moe", 16, "default-allowlisted", 16),        # the flip: knob unset, the family's default
    (None, "mixtral", None, "default-off", None),              # unread family: the library's plan
    ("0", "qwen3_moe", None, "explicit", None),                # the way back
    ("16", "mixtral", 16, "explicit", 16),                     # explicit still works anywhere
])
def test_build_engines_cap_reaches_the_kernel_call_from_the_default(monkeypatch, env, model_type, want_cap, want_source,
                                                                   handed):
    """End to end on CPU through build_engine's own wiring (apply_attn_split_cap): the server config read from the
    environment -> the per-family resolution -> the KV -> the kwargs the KV hands fp8_paged_decode_attention. Only the
    kernel is a stub (A3 read the real one engaged on CUDA from kv.split_cap = 16)."""
    if env is None:
        monkeypatch.delenv("E4B_ATTN_SPLIT_CAP", raising=False)
    else:
        monkeypatch.setenv("E4B_ATTN_SPLIT_CAP", env)
    kv, seen = _kv_with_stub_kernel(monkeypatch)
    cfg = sp.PagedServeConfig.from_env()
    assert sp.apply_attn_split_cap(kv, cfg, model_type) == (want_cap, want_source)
    assert kv.split_cap == want_cap
    kv.attention(0, torch.randn(1, 32, 128, dtype=torch.bfloat16), slots=[0])
    assert seen[-1]["split_cap"] == handed
    assert kv.split_plans == ({1: {"calls": 1, "plan": 32, "n_split": 16}} if handed else {})


def test_build_engine_sets_the_cap_through_apply_attn_split_cap():
    import inspect
    src = inspect.getsource(sp.build_engine)
    assert "apply_attn_split_cap(kv, cfg," in src and "kv.split_cap =" not in src


def test_the_knob_is_read_and_refused():
    assert sp._attn_split_cap_env("") == sp.FUSION_UNSET and sp._attn_split_cap_env("  ") == sp.FUSION_UNSET
    assert sp._attn_split_cap_env("16") == "16" and sp._attn_split_cap_env("0") == "0" and sp._attn_split_cap_env("032") == "32"
    for bad in ("33", "-1", "auto", "16.0", "on"):
        with pytest.raises(ValueError, match="E4B_ATTN_SPLIT_CAP"):
            sp._attn_split_cap_env(bad)


def test_qwen3_moe_alone_takes_the_cap_by_default_on_p143_amendment_3():
    """P143 A3 (#1788, p143a3-5090-1) read FLIP for qwen3_moe at 16; nothing else is read. Mixtral's B 1-2 would move
    under the cap, so it stays on the library's plan until a read of its own. ``0`` restores the plan explicitly."""
    assert sp.ATTN_SPLIT_CAP_DEFAULT_FAMILIES == {"qwen3_moe": 16}
    assert sp.resolve_attn_split_cap(sp.FUSION_UNSET, "qwen3_moe") == (16, "default-allowlisted")
    for family in ("mixtral", "qwen3_5_moe_text", "gpt_oss", "granitemoe", "olmoe", None):
        assert sp.resolve_attn_split_cap(sp.FUSION_UNSET, family) == (None, "default-off"), family
    assert sp.resolve_attn_split_cap("0", "qwen3_moe") == (None, "explicit")           # the way back
    assert sp.resolve_attn_split_cap("16", "mixtral") == (16, "explicit")
    assert sp.resolve_attn_split_cap("8", "qwen3_moe") == (8, "explicit")


def test_the_server_config_reads_the_knob(monkeypatch):
    monkeypatch.setenv("E4B_ATTN_SPLIT_CAP", "16")
    assert sp.PagedServeConfig.from_env().attn_split_cap == "16"
    monkeypatch.delenv("E4B_ATTN_SPLIT_CAP")
    assert sp.PagedServeConfig.from_env().attn_split_cap == sp.FUSION_UNSET
