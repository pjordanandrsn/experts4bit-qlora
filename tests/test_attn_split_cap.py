# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_ATTN_SPLIT_CAP`` (lane P143, Amendment 2; e4b#1313): the paged FP8 attention decode capped at ``min(cap, the
library's own plan)`` splits.

The plan is grouped-nf4-gemm's inline block in ``fp8_paged_decode_attention``; e4b carries a copy
(``fp8_paged_kv.gnf4_split_plan``). These tests pin the INSTALLED block by sha256, then execute that block itself against
the copy over a grid, so a planner change in grouped-nf4-gemm fails here instead of diverging silently. Then: the cap
the KV hands the kernel is ``min(16, plan)`` across B 1..64 at 170 SMs (RTX 5090) and 188 SMs (RTX PRO 6000), with
serve-throughput's exact rows; it applies only when a cap is set, never over an explicit ``n_split``, and only on the fp8
split path; and the server knob's reading and per-family resolution (no family by default until P143's served read)."""
import hashlib
import importlib.util
import pathlib
import sys
import textwrap
import types

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora import serve_paged as sp  # noqa: E402
from experts4bit_qlora.engines import fp8_paged_kv as kvmod  # noqa: E402

START = "    if n_split is None:\n"
END = "        n_split = int(min(32, want, max_useful))\n"


def _installed_block():
    spec = importlib.util.find_spec("fp8_paged_attn")
    if spec is None or not spec.origin:
        pytest.skip("grouped-nf4-gemm is not installed (CI installs it)")
    src = pathlib.Path(spec.origin).read_text(encoding="utf-8").replace("\r\n", "\n")
    a = src.index(START)
    return src[a:src.index(END, a) + len(END)]


def test_the_installed_planner_block_is_the_one_e4b_copies():
    assert hashlib.sha256(_installed_block().encode()).hexdigest() == kvmod.GNF4_SPLIT_PLAN_SHA256


def _run_block(block, *, B, n_kv_heads, cols, block_tokens, sms, compute, pack_heads, ktile):
    class _Props:
        multi_processor_count = sms
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_properties=lambda d: _Props))
    ns = {"n_split": None, "torch": fake_torch, "q": types.SimpleNamespace(device="cuda"), "pack_heads": pack_heads,
          "B": B, "n_kv_heads": n_kv_heads, "block_tokens": block_tokens, "ktile": ktile, "compute": compute,
          "block_table": types.SimpleNamespace(shape=(B, cols))}
    exec(compile(textwrap.dedent(block), "gnf4_plan", "exec"), ns)
    return ns["n_split"]


def test_the_copy_is_the_installed_block_over_the_grid():
    block = _installed_block()
    for sms in (170, 188):
        for compute in ("fp8", "f32"):
            ktile = 64 if compute == "fp8" else 32               # the kernel's own defaults, resolved before the plan
            for h in (4, 8):
                for cols in (256, 64, 8):
                    for pack in (False, True):
                        for B in range(1, 65):
                            want = _run_block(block, B=B, n_kv_heads=h, cols=cols, block_tokens=16, sms=sms,
                                              compute=compute, pack_heads=pack, ktile=ktile)
                            got = kvmod.gnf4_split_plan(B, h, cols, 16, sms, compute, pack_heads=pack)
                            assert got == want, (sms, compute, h, cols, pack, B, got, want)


def _capped(monkeypatch, *, B, sms, h=4, cap=16, compute="fp8", kw=None):
    stub = types.ModuleType("fp8_paged_attn")
    stub._compute_default = lambda *a, **k: compute
    monkeypatch.setitem(sys.modules, "fp8_paged_attn", stub)
    monkeypatch.setattr(kvmod, "_sm_count", lambda d: sms)
    fake_kv = types.SimpleNamespace(Hs=[h], Ds=[128], kgs=[1], bt=16, split_cap=cap)
    q = torch.zeros(B, 32, 128, dtype=torch.bfloat16)
    tbl = torch.zeros(B, 256, dtype=torch.int32)                        # 16 slots x 4096 tokens: 256 blocks
    return kvmod.Fp8PagedKV._capped_n_split(fake_kv, 0, q, tbl, kw or {})


def test_the_cap_is_min_16_and_the_library_plan_across_B_1_to_64(monkeypatch):
    block = _installed_block()
    for sms in (170, 188):
        for B in range(1, 65):
            plan = _run_block(block, B=B, n_kv_heads=4, cols=256, block_tokens=16, sms=sms, compute="fp8",
                              pack_heads=False, ktile=64)
            assert _capped(monkeypatch, B=B, sms=sms) == min(16, plan), (sms, B)


def test_serve_throughputs_rows():
    """The plan at Qwen3's 4 KV heads on a 5090 is 32, 32, 31, 23, 18 at B 1..5 (all capped to 16) and 15 at B 6; on a
    PRO 6000 B 6 plans 17 (capped) and B 7 plans 14. Mixtral's 8 KV heads move at B 1..2 only."""
    plan = lambda B, sms, h=4: kvmod.gnf4_split_plan(B, h, 256, 16, sms, "fp8")  # noqa: E731
    assert [plan(B, 170) for B in range(1, 7)] == [32, 32, 31, 23, 18, 15]
    assert [plan(B, 170) for B in (8, 16, 32, 64)] == [11, 5, 2, 1]
    assert (plan(6, 188), plan(7, 188)) == (17, 14)
    assert [plan(B, 170, h=8) for B in (1, 2, 3)] == [32, 23, 15]


def test_the_cap_moves_only_where_the_plan_is_above_it(monkeypatch):
    for B in (1, 2, 3, 4, 5):
        assert _capped(monkeypatch, B=B, sms=170) == 16
    for B in (6, 8, 16, 32, 64):
        assert _capped(monkeypatch, B=B, sms=170) == kvmod.gnf4_split_plan(B, 4, 256, 16, 170, "fp8")


def test_the_f32_path_is_left_to_the_library(monkeypatch):
    assert _capped(monkeypatch, B=1, sms=170, compute="f32") is None


def test_an_explicit_compute_is_the_one_planned(monkeypatch):
    assert _capped(monkeypatch, B=1, sms=170, compute="f32", kw={"compute": "fp8"}) == 16


def test_the_attention_call_hands_the_cap_only_when_set_and_never_over_an_explicit_n_split(monkeypatch):
    pytest.importorskip("fp8_kv", reason="the KV class needs grouped-nf4-gemm (CI installs it)")
    seen = []
    stub = types.ModuleType("fp8_paged_attn")

    def fp8_paged_decode_attention(q, kf, vf, tbl, lens, *, n_kv_heads, head_dim, block_tokens, k_groups=1,
                                   k_row_bytes=None, v_row_bytes=None, window=0, sinks=None, **kw):
        seen.append(kw)
        return torch.zeros_like(q)
    stub.fp8_paged_decode_attention = fp8_paged_decode_attention
    stub._compute_default = lambda *a, **k: "fp8"
    monkeypatch.setitem(sys.modules, "fp8_paged_attn", stub)
    monkeypatch.setattr(kvmod, "_sm_count", lambda d: 170)
    kv = kvmod.Fp8PagedKV(1, 4, 128, batch=2, max_tokens_per_seq=4096, device="cpu")
    kv.append(0, 0, torch.randn(8, 4, 128, dtype=torch.bfloat16), torch.randn(8, 4, 128, dtype=torch.bfloat16))
    q = torch.randn(1, 32, 128, dtype=torch.bfloat16)
    kv.attention(0, q, slots=[0])
    kv.split_cap = 16
    kv.attention(0, q, slots=[0])
    kv.attention(0, q, slots=[0], n_split=4)
    assert [k.get("n_split") for k in seen] == [None, 16, 4]


def test_the_knob_is_read_and_refused():
    assert sp._attn_split_cap_env("") == sp.FUSION_UNSET and sp._attn_split_cap_env("  ") == sp.FUSION_UNSET
    assert sp._attn_split_cap_env("16") == "16" and sp._attn_split_cap_env("0") == "0" and sp._attn_split_cap_env("032") == "32"
    for bad in ("33", "-1", "auto", "16.0", "on"):
        with pytest.raises(ValueError, match="E4B_ATTN_SPLIT_CAP"):
            sp._attn_split_cap_env(bad)


def test_no_family_takes_the_cap_by_default_until_the_served_read():
    assert sp.ATTN_SPLIT_CAP_DEFAULT_FAMILIES == {}
    assert sp.resolve_attn_split_cap(sp.FUSION_UNSET, "qwen3_moe") == (None, "default-off")
    assert sp.resolve_attn_split_cap("16", "qwen3_moe") == (16, "explicit")
    assert sp.resolve_attn_split_cap("0", "qwen3_moe") == (None, "explicit")


def test_the_server_config_reads_the_knob(monkeypatch):
    monkeypatch.setenv("E4B_ATTN_SPLIT_CAP", "16")
    assert sp.PagedServeConfig.from_env().attn_split_cap == "16"
    monkeypatch.delenv("E4B_ATTN_SPLIT_CAP")
    assert sp.PagedServeConfig.from_env().attn_split_cap == sp.FUSION_UNSET
