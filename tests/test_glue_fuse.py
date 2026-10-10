# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The RMSNorm fusion patches structurally and falls through safely."""
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experts4bit_qlora.engines.glue_fuse import fuse_t1_glue  # noqa: E402


class ToyRMSNorm(torch.nn.Module):
    def __init__(self, H=32, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.variance_epsilon = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True)
                                 + self.variance_epsilon)
                * self.weight.float()).to(x.dtype)


def _stub(monkeypatch, calls):
    stub = types.ModuleType("int4_b32")

    def rmsnorm_rows(x, w, eps):
        calls["fused"] += 1
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
                * w.float()).to(torch.bfloat16)
    stub.rmsnorm_rows = rmsnorm_rows
    monkeypatch.setitem(sys.modules, "int4_b32", stub)


def test_patches_and_falls_through(monkeypatch):
    monkeypatch.setenv("E4B_FUSE_T1_GLUE", "1")
    calls = {"fused": 0}
    _stub(monkeypatch, calls)
    m = torch.nn.Module()
    m.n1 = ToyRMSNorm()
    m.n2 = ToyRMSNorm()
    assert fuse_t1_glue(m) == 2
    x = torch.randn(1, 1, 32, dtype=torch.bfloat16)
    y = m.n1(x)
    assert calls["fused"] == 1 and y.dtype == torch.bfloat16
    ref = ToyRMSNorm()(x)
    assert torch.allclose(y.float(), ref.float(), rtol=2 ** -6, atol=2 ** -8)
    # prefill-size input falls through
    big = torch.randn(1, 4096, 32, dtype=torch.bfloat16)
    m.n1(big)
    assert calls["fused"] == 1
    # fp32 input falls through
    m.n2(torch.randn(1, 1, 32))
    assert calls["fused"] == 1


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("E4B_FUSE_T1_GLUE", raising=False)
    m = torch.nn.Module()
    m.n = ToyRMSNorm()
    assert fuse_t1_glue(m) == 0


def test_vacuous_refused(monkeypatch):
    monkeypatch.setenv("E4B_FUSE_T1_GLUE", "1")
    _stub(monkeypatch, {"fused": 0})
    with pytest.raises(RuntimeError, match="patched no RMSNorm"):
        fuse_t1_glue(torch.nn.Module())



class CenteredToyRMSNorm(ToyRMSNorm):
    """Gemma-style: x_norm * (1 + weight), weight near zero."""
    def __init__(self, H=32, eps=1e-6):
        super().__init__(H, eps)
        with torch.no_grad():
            self.weight.zero_()

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True)
                                 + self.variance_epsilon)
                * (1.0 + self.weight.float())).to(x.dtype)


def test_centered_variant_is_skipped(monkeypatch):
    """A name-matched module whose forward is the CENTERED formula must
    be refused by the semantic probe -- patching it would nearly zero
    the residual stream (review finding, High)."""
    monkeypatch.setenv("E4B_FUSE_T1_GLUE", "1")
    calls = {"fused": 0}
    _stub(monkeypatch, calls)
    m = torch.nn.Module()
    m.good = ToyRMSNorm()
    m.centered = CenteredToyRMSNorm()
    assert fuse_t1_glue(m) == 1          # only the matching norm
    x = torch.randn(1, 1, 32, dtype=torch.bfloat16)
    y = m.centered(x)                     # unpatched: its own formula
    assert calls["fused"] == 0
    ref = CenteredToyRMSNorm()(x)
    assert torch.allclose(y.float(), ref.float(), rtol=2 ** -6,
                          atol=2 ** -8)
    m.good(x)
    assert calls["fused"] == 1


def test_serve_assembly_point_invokes_glue(monkeypatch):
    """fuse_qkv is the advertised serve assembly point; the env flag
    must be live there (review finding: an env var read only by a
    function nothing calls is dead)."""
    from experts4bit_qlora.engines import qkv_fuse
    called = {}

    def rec(model):
        called["yes"] = True
        return 0
    monkeypatch.setattr("experts4bit_qlora.engines.glue_fuse.fuse_t1_glue",
                        rec)
    qkv_fuse.fuse_qkv(torch.nn.Module())
    assert called.get("yes")


def test_prefill_glue_lifts_the_norm_gate_and_counts(monkeypatch):
    """E4B_FUSE_PREFILL_GLUE (bench/prefill-glue/DESIGN.md, P1): off (the default), a prefill-size norm call keeps the
    eager chain, as test_patches_and_falls_through pins; at 1 it takes the fused kernel and is counted. A decode row is
    fused either way and never counted."""
    from experts4bit_qlora.engines import glue_fuse
    monkeypatch.setenv("E4B_FUSE_T1_GLUE", "1")
    for env, fused, seen in (("0", 0, {}), ("1", 1, {"norm": 1})):
        monkeypatch.setenv("E4B_FUSE_PREFILL_GLUE", env)
        monkeypatch.setattr(glue_fuse, "PREFILL_FOLD_SEEN", {})
        calls = {"fused": 0}
        _stub(monkeypatch, calls)
        m = torch.nn.Module()
        m.n = ToyRMSNorm()
        report = {}
        assert fuse_t1_glue(m, report=report) == 1 and report["prefill"] == ("on" if env == "1" else "off")
        big = torch.randn(1, 512, 32, dtype=torch.bfloat16)
        y = m.n(big)
        assert calls["fused"] == fused and glue_fuse.PREFILL_FOLD_SEEN == seen, env
        assert torch.allclose(y.float(), ToyRMSNorm()(big).float(), rtol=2 ** -6, atol=2 ** -8)
        m.n(torch.randn(1, 1, 32, dtype=torch.bfloat16))
        assert calls["fused"] == fused + 1 and glue_fuse.PREFILL_FOLD_SEEN == seen, env


def test_prefill_glue_knob_reads_zero_or_one_and_refuses_anything_else(monkeypatch):
    from experts4bit_qlora.engines.glue_fuse import DECODE_ROWS_MAX, prefill_glue_on, rows_cap
    monkeypatch.delenv("E4B_FUSE_PREFILL_GLUE", raising=False)
    assert prefill_glue_on() is False and rows_cap() == DECODE_ROWS_MAX == 64
    assert prefill_glue_on("") is False and prefill_glue_on("0") is False and prefill_glue_on("1") is True
    assert rows_cap("1") is None
    for bad in ("auto", "yes", "2", "on"):
        with pytest.raises(ValueError, match="E4B_FUSE_PREFILL_GLUE"):
            prefill_glue_on(bad)
    monkeypatch.setenv("E4B_FUSE_PREFILL_GLUE", "auto")
    m = torch.nn.Module()
    m.n = ToyRMSNorm()
    with pytest.raises(ValueError, match="E4B_FUSE_PREFILL_GLUE"):     # refused at patch time, not read as off
        fuse_t1_glue(m, mode="1")

