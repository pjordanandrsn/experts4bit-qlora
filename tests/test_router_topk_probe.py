# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k10 Stage A probe, checked WITHOUT a GPU.

The probe is an ATTRIBUTION instrument: it must count every torch.topk
call, record the shapes it saw, restore the original on exit (including
when the body raises), and pass `sorted` through faithfully -- an
override that silently did nothing would make Stage A's ablation
unfalsifiable, which is the exact failure K9 died of twice.
"""
import hashlib
import importlib.util
import pathlib

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "step_decomp_probe",
    pathlib.Path(__file__).resolve().parents[1]
    / "bench" / "hybrid-g9" / "step_decomp.py")


def _probe_cls(capturing=False):
    # step_decomp imports heavy deps at module scope; read the class out
    # by exec'ing only its definition block.
    src = _SPEC.origin
    text = pathlib.Path(src).read_text()
    start = text.index("class _TopkProbe:")
    end = text.index("\ndef _b1d_stage_a", start)
    # the exec namespace must carry every module the class body uses;
    # step_decomp imports these at module scope, but a text slice does
    # not bring them along
    ns = {"torch": torch, "hashlib": hashlib,
          "_capturing": lambda: capturing}
    exec(compile(text[start:end], src, "exec"), ns)
    return ns["_TopkProbe"]


def test_probe_counts_shapes_and_restores():
    P = _probe_cls()
    orig = torch.topk
    x = torch.randn(1, 128)
    with P() as p:
        p.begin_window(2)
        torch.topk(x, 8, dim=-1)
        torch.topk(x, 8, dim=-1)
        torch.topk(torch.randn(4, 64), 2, dim=-1)
    assert torch.topk is orig, "probe must restore torch.topk"
    r = p.report(layers=48)
    assert r["topk_calls"] == 3
    assert r["calls_per_step"] == 1.5
    assert r["shapes"]["(1, 128)|k=8"] == 2
    assert r["shapes"]["(4, 64)|k=2"] == 1
    assert r["sorted_override"] is None


def test_probe_restores_on_exception():
    P = _probe_cls()
    orig = torch.topk
    with pytest.raises(RuntimeError):
        with P():
            raise RuntimeError("boom")
    assert torch.topk is orig, "an exception must not leak the patch"


def test_sorted_override_actually_reaches_torch():
    """If the override silently did nothing, Stage A's ablation could
    never refute the attribution -- the whole point of the probe."""
    P = _probe_cls()
    seen = {}
    real = torch.topk

    def spy(input, k, dim=-1, largest=True, sorted=True, *a, **kw):
        seen["sorted"] = sorted
        return real(input, k, dim=dim, largest=largest, sorted=sorted,
                    *a, **kw)

    torch.topk = spy
    try:
        with P(sorted_override=False):
            torch.topk(torch.randn(1, 128), 8, dim=-1)
        assert seen["sorted"] is False
        seen.clear()
        with P(sorted_override=True):
            torch.topk(torch.randn(1, 128), 8, dim=-1)
        assert seen["sorted"] is True
        seen.clear()
        with P():                      # no override -> caller's value
            torch.topk(torch.randn(1, 128), 8, dim=-1, sorted=False)
        assert seen["sorted"] is False
    finally:
        torch.topk = real


def test_sorted_false_keeps_the_set_and_changes_only_order():
    """B1's premise, checked on CPU: sorted=False returns the same
    SELECTED SET. B1 refuses on a changed set regardless of perplexity."""
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        x = torch.randn(1, 128, generator=g)
        _, i_true = torch.topk(x, 8, dim=-1, sorted=True)
        _, i_false = torch.topk(x, 8, dim=-1, sorted=False)
        assert set(i_true[0].tolist()) == set(i_false[0].tolist())


def test_set_digest_is_order_invariant_and_set_sensitive():
    """B1-C is a REFUSE gate on the selected SET, so its digest must
    ignore ORDER (sorted=False permutes) while still catching a
    different set. A digest sensitive to order would refuse every
    sorted=False arm by construction; one insensitive to membership
    would pass a genuinely different model."""
    P = _probe_cls()
    x = torch.randn(1, 128, generator=torch.Generator().manual_seed(3))

    with P(sorted_override=True) as p_true:
        p_true.begin_window(1)
        torch.topk(x, 8, dim=-1)
    with P(sorted_override=False) as p_false:
        p_false.begin_window(1)
        torch.topk(x, 8, dim=-1)
    a = p_true.report(1)["selected_set_digest"]
    b = p_false.report(1)["selected_set_digest"]
    assert a == b, "same set, different order must digest identically"

    # a genuinely different set must NOT collide
    with P() as p_other:
        p_other.begin_window(1)
        torch.topk(x, 7, dim=-1)
    assert p_other.report(1)["selected_set_digest"] != a


def test_window_excludes_work_before_it():
    """The probe wraps the whole stage, so prefill and warm decode
    happen inside it. Counts divided by the WINDOW's step count would
    be inflated by that preamble -- which breaks the one-topk-per-layer
    attribution A1 depends on."""
    P = _probe_cls()
    x = torch.randn(1, 128)
    with P() as p:
        for _ in range(7):             # "prefill + warm"
            torch.topk(x, 8, dim=-1)
        p.begin_window(2)              # the measured window opens here
        torch.topk(x, 8, dim=-1)
        torch.topk(x, 8, dim=-1)
        r = p.report(layers=1)
    assert r["topk_calls"] == 2, "preamble leaked into the window"
    assert r["calls_per_step"] == 1.0


def test_report_without_a_window_refuses():
    """Reporting counts that were never scoped to a window would divide
    prefill+warm by a step count that never applied to them."""
    P = _probe_cls()
    with P() as p:
        torch.topk(torch.randn(1, 128), 8, dim=-1)
        with pytest.raises(RuntimeError, match="before begin_window"):
            p.report(layers=48)


def test_digest_is_skipped_while_capturing():
    """The digest does a D2H copy, which is ILLEGAL inside CUDA graph
    capture -- and the ablation arms capture. Under capture the probe
    must still COUNT (attribution) but must not sync."""
    P = _probe_cls(capturing=True)
    with P() as p:
        p.begin_window(1)
        torch.topk(torch.randn(1, 128), 8, dim=-1)
        r = p.report(layers=1)
    assert r["topk_calls"] == 1, "counting must survive capture"
    empty = __import__("hashlib").sha256().hexdigest()
    assert r["selected_set_digest"] == empty, "must not sync in capture"
