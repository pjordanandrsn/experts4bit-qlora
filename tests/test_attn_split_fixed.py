# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_ATTN_SPLIT_FIXED`` (P223's follow-up; opt-in, e4b#1313): a fixed split count S for the paged FP8 attention decode
at every row count, with every other per-call choice of ``fp8_paged_decode_attention`` pinned, so the call differs
across buckets only in B. P223 read the decode schedule-invariant with one split at every bucket; S keeps that property
at a lone row's parallelism. These tests pin, on CPU with a stub kernel recording every keyword:
  - the knob's reading (unset, empty and 0 off; 1..32; anything else refused) and its refusal beside E4B_ATTN_SPLIT_CAP;
  - the wrapper's per-call choices, read from its REAL signature, each handed explicitly at every bucket 2/4/8/16, and
    the calls identical across those buckets but for B; a pin left on auto (fuse_combine, the review's mutant)
    fails that check;
  - the engagement record: n_split == S and the pins at EVERY bucket the call met (a fallback to the plan fails);
  - never on speculative verify, never over a caller's n_split, never on the f32 path; off, nothing changes;
  - build_engine's wiring: the fixed split replaces any cap (a per-family default included) and is reported."""
import inspect
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora import serve_paged as sp  # noqa: E402
from experts4bit_qlora.engines import fp8_paged_kv as kvmod  # noqa: E402

BUCKETS = (2, 4, 8, 16)
#: keyword arguments of the wrapper that default to None but carry data or a different knob, not a per-call choice:
#: the attention sinks, the pool row strides, and the cap (never handed beside n_split); sm_scale resolves from
#: head_dim alone, the same at every B
NOT_CHOICES = {"sinks", "k_row_bytes", "v_row_bytes", "split_cap", "sm_scale"}


def _require(module: str, reason: str):
    """Import ``module``; in CI a plain import (it must be there, never a skip), off CI a skip when it is absent."""
    if os.environ.get("CI"):
        import importlib
        return importlib.import_module(module)
    return pytest.importorskip(module, reason=reason)


try:      # the REAL wrapper's signature, read once at import, before any test swaps a stub into sys.modules
    import fp8_paged_attn as _real_fp8_paged_attn
    _REAL_SIG = inspect.signature(_real_fp8_paged_attn.fp8_paged_decode_attention)
except ImportError:
    _REAL_SIG = None


def _auto_choices() -> set:
    """Every keyword of the REAL wrapper that defaults to None and that the wrapper resolves per call: what a fixed
    split must hand over explicitly, read from the installed grouped-nf4-gemm, so a new one fails here. Never read
    through sys.modules inside a test: the stub kernel has no keyword-only parameters, which would make every check
    vacuous (an empty set), so an empty set is refused too."""
    if _REAL_SIG is None:
        _require("fp8_paged_attn", "grouped-nf4-gemm is not installed (CI installs it)")
        raise AssertionError("fp8_paged_attn imported now but not at module import")
    out = {n for n, p in _REAL_SIG.parameters.items() if p.kind is p.KEYWORD_ONLY and p.default is None} - NOT_CHOICES
    assert out, "no per-call choices read from the wrapper: the check would be vacuous"
    return out


def _kv(monkeypatch, *, compute="fp8"):
    _require("fp8_kv", "the KV class needs grouped-nf4-gemm (CI installs it)")
    seen = []
    stub = types.ModuleType("fp8_paged_attn")

    def fp8_paged_decode_attention(q, kf, vf, tbl, lens, **kw):
        seen.append({"B": int(q.shape[0]), **kw})
        return torch.zeros_like(q)
    stub.fp8_paged_decode_attention = fp8_paged_decode_attention
    stub._compute_default = lambda *a, **k: ("f32" if k.get("ktile") == 48 else compute)
    stub.decode_split_plan = lambda B, h, cols, **k: max(1, (11 * 170) // (5 * B * h))
    monkeypatch.setitem(sys.modules, "fp8_paged_attn", stub)
    monkeypatch.setattr(kvmod, "_sm_count", lambda d: 170)
    monkeypatch.setattr(kvmod, "_KERNEL_KW_CACHE", {})
    kv = kvmod.Fp8PagedKV(1, 8, 128, batch=16, max_tokens_per_seq=4096, device="cpu")
    for s in range(16):
        kv.append(0, s, torch.randn(8, 8, 128, dtype=torch.bfloat16), torch.randn(8, 8, 128, dtype=torch.bfloat16))
    return kv, seen


def _decode_at_buckets(kv, **kw):
    for B in BUCKETS:
        kv.attention(0, torch.randn(B, 32, 128, dtype=torch.bfloat16), slots=list(range(B)), **kw)


def fixed_call_faults(seen, S, choices) -> list:
    """The property: each call hands every per-call choice explicitly (never None), n_split is S, and the
    calls at the buckets are identical but for B."""
    faults = []
    for c in seen:
        missing = sorted(k for k in choices if c.get(k) is None)
        if missing:
            faults.append(f"B={c['B']}: left to the wrapper's per-call default: {missing}")
        if c.get("n_split") != S:
            faults.append(f"B={c['B']}: n_split {c.get('n_split')}, fixed {S}")
    rest = [{k: v for k, v in c.items() if k != "B"} for c in seen]
    if any(r != rest[0] for r in rest):
        faults.append(f"the calls differ beyond B: {rest}")
    return faults


def test_the_wrappers_per_call_choices_are_the_ones_pinned():
    """Read from the real signature: n_split plus every pinned key. A choice the library adds fails here until it is
    pinned (or named in NOT_CHOICES with a reason)."""
    choices = _auto_choices()
    assert choices == {"n_split"} | ({k for k in kvmod.SPLIT_FIXED_PINS} - {"num_stages", "pack_heads"}), choices
    assert kvmod.SPLIT_FIXED_PINS == {"compute": "fp8", "fuse_combine": True, "num_warps": 4, "ktile": 64,
                                      "num_stages": 3, "pack_heads": False}


@pytest.mark.parametrize("S", [1, 8, 12])
def test_a_fixed_split_hands_every_choice_explicitly_and_identically_at_every_bucket(monkeypatch, S):
    kv, seen = _kv(monkeypatch)
    kv.split_fixed = S
    _decode_at_buckets(kv)
    assert [c["B"] for c in seen] == list(BUCKETS)
    assert fixed_call_faults(seen, S, _auto_choices()) == []
    for c in seen:
        assert {k: c[k] for k in kvmod.SPLIT_FIXED_PINS} == kvmod.SPLIT_FIXED_PINS and "split_cap" not in c


def test_a_pin_left_on_auto_fails_the_check(monkeypatch):
    """The review's mutant: fuse_combine left to the wrapper (its per-call default) must fail the property."""
    kv, seen = _kv(monkeypatch)
    monkeypatch.setattr(kvmod, "SPLIT_FIXED_PINS", {k: v for k, v in kvmod.SPLIT_FIXED_PINS.items() if k != "fuse_combine"})
    kv.split_fixed = 8
    _decode_at_buckets(kv)
    faults = fixed_call_faults(seen, 8, _auto_choices())
    assert faults and all("fuse_combine" in f for f in faults), faults


def test_a_plan_fallback_at_one_bucket_fails_the_check(monkeypatch):
    kv, seen = _kv(monkeypatch)
    kv.split_fixed = 8
    _decode_at_buckets(kv)
    seen[2] = {**seen[2], "n_split": None}            # bucket 8 left to the plan
    faults = fixed_call_faults(seen, 8, _auto_choices())
    assert any("B=8" in f for f in faults), faults


def test_the_engagement_record_carries_s_and_the_pins_at_every_bucket(monkeypatch):
    kv, _ = _kv(monkeypatch)
    kv.split_fixed = 12
    _decode_at_buckets(kv)
    _decode_at_buckets(kv)
    assert sorted(kv.split_fixed_plans) == list(BUCKETS)
    for B in BUCKETS:
        rec = kv.split_fixed_plans[B]
        assert rec["n_split"] == 12 and rec["pins"] == kvmod.SPLIT_FIXED_PINS and rec["calls"] == 2, (B, rec)
        assert rec["plan"] == max(1, (11 * 170) // (5 * B * 8)), (B, rec)   # the plan it replaced, for reference
    assert kv.split_plans == {}, "the cap's record stays empty"


def test_a_callers_own_choice_stands_and_a_none_takes_the_pin(monkeypatch):
    kv, seen = _kv(monkeypatch)
    kv.split_fixed = 8
    q = torch.randn(2, 32, 128, dtype=torch.bfloat16)
    kv.attention(0, q, slots=[0, 1], num_warps=8)
    kv.attention(0, q, slots=[0, 1], num_warps=None)
    assert seen[0]["num_warps"] == 8 and seen[1]["num_warps"] == 4


def test_never_on_speculative_verify_over_a_callers_n_split_or_on_the_f32_path(monkeypatch):
    kv, seen = _kv(monkeypatch)
    kv.split_fixed = 8
    kv.attention(0, torch.randn(2, 32, 128, dtype=torch.bfloat16), slots=[0, 0],
                 lens_override=torch.tensor([7, 8], dtype=torch.int32))
    kv.attention(0, torch.randn(2, 32, 128, dtype=torch.bfloat16), slots=[0, 1], n_split=3)
    kv.attention(0, torch.randn(2, 32, 128, dtype=torch.bfloat16), slots=[0, 1], ktile=48)   # rules out fp8
    assert [c.get("n_split") for c in seen] == [None, 3, None]
    assert all("fuse_combine" not in c for c in seen) and kv.split_fixed_plans == {}
    kv32, seen32 = _kv(monkeypatch, compute="f32")
    kv32.split_fixed = 8
    _decode_at_buckets(kv32)
    assert all(c.get("n_split") is None for c in seen32) and kv32.split_fixed_plans == {}


def test_off_changes_nothing(monkeypatch):
    kv, seen = _kv(monkeypatch)
    _decode_at_buckets(kv)
    fixed_keys = {"n_split", *kvmod.SPLIT_FIXED_PINS}
    assert seen and all(not (set(c) & fixed_keys) for c in seen), seen
    assert kv.split_fixed is None and kv.split_fixed_plans == {}


def test_the_knob_is_read_and_refused():
    for off in ("", "  ", "0", "00"):
        assert sp._attn_split_fixed_env(off) is None, off
    assert sp._attn_split_fixed_env("8") == 8 and sp._attn_split_fixed_env("032") == 32 and sp._attn_split_fixed_env("1") == 1
    for bad in ("33", "-1", "auto", "8.0", "on"):
        with pytest.raises(ValueError, match="E4B_ATTN_SPLIT_FIXED"):
            sp._attn_split_fixed_env(bad)


@pytest.mark.parametrize("cap", ["16", "0"])
def test_the_config_refuses_it_beside_an_explicit_cap(monkeypatch, cap):
    monkeypatch.setenv("E4B_ATTN_SPLIT_FIXED", "8")
    monkeypatch.setenv("E4B_ATTN_SPLIT_CAP", cap)
    with pytest.raises(ValueError, match="E4B_ATTN_SPLIT_FIXED=8 and E4B_ATTN_SPLIT_CAP"):
        sp.PagedServeConfig.from_env()


@pytest.mark.parametrize("model_type", ["qwen3_moe", "mixtral"])
def test_build_engines_wiring_replaces_any_cap_with_the_fixed_split(monkeypatch, model_type):
    """End to end on CPU through apply_attn_split_cap: qwen3_moe's per-family default cap is superseded too."""
    monkeypatch.delenv("E4B_ATTN_SPLIT_CAP", raising=False)
    monkeypatch.setenv("E4B_ATTN_SPLIT_FIXED", "8")
    kv, seen = _kv(monkeypatch)
    cfg = sp.PagedServeConfig.from_env()
    assert cfg.attn_split_fixed == 8
    assert sp.apply_attn_split_cap(kv, cfg, model_type) == (None, "superseded: E4B_ATTN_SPLIT_FIXED")
    assert kv.split_fixed == 8 and kv.split_cap is None
    _decode_at_buckets(kv)
    assert fixed_call_faults(seen, 8, _auto_choices()) == []
    monkeypatch.delenv("E4B_ATTN_SPLIT_FIXED")
    cfg = sp.PagedServeConfig.from_env()
    kv2, _ = _kv(monkeypatch)
    assert sp.apply_attn_split_cap(kv2, cfg, model_type)[1] != "superseded: E4B_ATTN_SPLIT_FIXED"
    assert kv2.split_fixed is None


def test_build_engine_reports_the_fixed_split():
    src = inspect.getsource(sp.build_engine)
    assert "apply_attn_split_cap(kv, cfg," in src and 'info["attn_split_fixed"]' in src
    assert "kv.split_fixed_plans" in src


def test_one_rows_attention_is_bitwise_the_same_in_a_batch_of_2_and_of_16_at_a_fixed_split():
    """The kernel itself, on CPU (the review's $0 check, kept as a test, skipped without Triton):
    grouped-nf4-gemm's real fp8 split kernel under TRITON_INTERPRET=1 (tests/attn_split_rowcheck.py, a subprocess
    because Triton reads the variable when it defines the kernels). Row 0 decodes beside one other sequence and beside
    fifteen. The CONTROL hands each batch the plan P223 recorded on the card (12 splits at bucket 2, 2 at bucket 16),
    and row 0 must DIFFER, so the check can see a batch dependence. At a fixed S in 1, 8 and 12 it must be EQUAL."""
    import importlib.util
    import subprocess
    if importlib.util.find_spec("triton") is None or importlib.util.find_spec("fp8_paged_attn") is None:
        pytest.skip("needs Triton (its CPU interpreter) and grouped-nf4-gemm")
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "attn_split_rowcheck.py"), "fp8", "1", "8", "12"],
                       capture_output=True, text=True, timeout=900, cwd=here,
                       env={**os.environ, "TRITON_INTERPRET": "1",
                            "PYTHONPATH": os.pathsep.join([os.path.dirname(here), os.environ.get("PYTHONPATH", "")])})
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("CONTROL", "ROWCHECK"))]
    assert r.returncode == 0 and len(lines) == 4, r.stdout[-2000:] + r.stderr[-2000:]
    assert lines[0].startswith("CONTROL fp8 plan B=2 12 B=16 2: DIFFERS"), lines[0]
    for S, ln in zip((1, 8, 12), lines[1:]):
        assert ln.startswith(f"ROWCHECK fp8 S={S}: EQUAL max|d| 0 "), ln
