# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""Keep ``quant_guard`` honest: the refusal list complete, and the probe genuinely independent.

Both halves of that module are load-bearing and both can rot silently, so both are pinned here
rather than trusted. The refusal-list checks read ``loader.py``'s source and need neither torch
nor the package, so they run on any host; the probe checks import inside the function so they
skip cleanly instead of taking the module down with them.

``quant_guard`` exists so a loader REFUSAL cannot masquerade as an absent bnb backend and report as
a green skip. That only holds while the substring list covers every refusal the loader can raise with
a type ``QUANTIZE_UNAVAILABLE`` catches. A refusal added later and not listed here is invisible in
exactly the way the guard was built to prevent — the arm goes green, and nothing says otherwise.

So the list is checked against the loader's SOURCE rather than trusted. This module parses
``loader.py`` with ``ast``; it imports neither torch nor the package, so it runs on any host —
including the ones where every arm that would exercise the guard for real is skipped.
"""

import ast
import builtins
from pathlib import Path

import pytest

from quant_guard import LOADER_REFUSALS, is_loader_refusal

LOADER_SRC = Path(__file__).resolve().parents[1] / "experts4bit_qlora" / "loader.py"

#: Mirrors quant_guard.QUANTIZE_UNAVAILABLE. Spelled out rather than imported so that widening the
#: catch set there cannot quietly narrow what this test inspects.
_SWALLOWED = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)


def _swallowable_raises(tree):
    """(lineno, static message text) for every ``raise E(...)`` whose E is swallowed by the guard."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        if not isinstance(node.exc.func, ast.Name):
            continue
        exc_cls = getattr(builtins, node.exc.func.id, None)
        # Resolved against builtins on purpose: a non-builtin (MoEConventionError, a ValueError
        # subclass) is not caught by the guard, so it already fails loudly and needs no substring.
        if not (isinstance(exc_cls, type) and issubclass(exc_cls, _SWALLOWED)):
            continue
        # f-strings included: every refusal below carries its identifying phrase in a literal chunk,
        # which is what makes a substring match possible at runtime in the first place.
        text = "".join(
            n.value for n in ast.walk(node.exc) if isinstance(n, ast.Constant) and isinstance(n.value, str)
        )
        out.append((node.lineno, text))
    return out


def test_every_swallowable_loader_refusal_is_listed():
    """Each refusal the guard would otherwise swallow must match a ``LOADER_REFUSALS`` substring."""
    raises = _swallowable_raises(ast.parse(LOADER_SRC.read_text(encoding="utf-8")))
    assert raises, f"parsed no swallowable raises out of {LOADER_SRC} — this test stopped measuring"

    unlisted = [(ln, t) for ln, t in raises if not any(m in t for m in LOADER_REFUSALS)]
    assert not unlisted, (
        "loader.py raises these with a type quant_guard swallows, but no LOADER_REFUSALS substring "
        "matches — a test hitting one would report a green SKIP instead of failing:\n"
        + "\n".join(f"  loader.py:{ln}: {t[:120]!r}" for ln, t in unlisted)
        + "\nAdd a stable substring of each message to LOADER_REFUSALS."
    )


def test_no_dead_entries_in_the_refusal_list():
    """And the converse: a substring matching nothing is a rename that silently disarmed the guard."""
    raises = _swallowable_raises(ast.parse(LOADER_SRC.read_text(encoding="utf-8")))
    dead = [m for m in LOADER_REFUSALS if not any(m in t for _, t in raises)]
    assert not dead, (
        f"LOADER_REFUSALS entries match no raise in loader.py: {dead}. Either the message was "
        "reworded (update the substring) or the refusal is gone (drop it)."
    )


def test_is_loader_refusal_separates_the_two_causes():
    """The distinction itself: a loader refusal is one, a bnb backend complaint is not."""
    assert is_loader_refusal(RuntimeError("no fused expert stacks found in 'x' (model_type='y')"))
    assert is_loader_refusal(NotImplementedError("Unsupported model_type='z'."))
    # The real bnb-unavailable text this guard must keep skipping on.
    assert not is_loader_refusal(RuntimeError("CUDA Setup failed despite GPU being available."))
    assert not is_loader_refusal(ImportError("libbitsandbytes_cpu.so: cannot open shared object file"))


# ------------------------------------------------------------------------------------------------
# The quantize probe. These need torch/bnb, so they import inside the function: this module's
# top-level must stay import-light (see the note in test_every_swallowable_loader_refusal_is_listed)
# so the AST checks above still run on a host that has neither.
# ------------------------------------------------------------------------------------------------

def test_scheme_backend_table_covers_every_scheme_the_package_has():
    """A scheme added to the package but missed here would be probed against the wrong backend.

    `SCHEME_BACKEND` decides which bnb primitive answers "is this available", and a missing entry
    falls back to "4bit" — which for a new 8-bit or passthrough scheme is the wrong question, and
    would skip (or fail to skip) for a reason unrelated to the scheme actually under test.
    """
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    from experts4bit_qlora._vendor.experts import _SCHEME_BITS

    from quant_guard import SCHEME_BACKEND

    missing = sorted(set(_SCHEME_BITS) - set(SCHEME_BACKEND))
    assert not missing, f"schemes in the package but not in SCHEME_BACKEND: {missing}"
    stale = sorted(set(SCHEME_BACKEND) - set(_SCHEME_BITS))
    assert not stale, f"schemes in SCHEME_BACKEND the package no longer has: {stale}"

    expected = {4: "4bit", 8: "blockwise", 16: None}
    wrong = {q: (SCHEME_BACKEND[q], expected[b]) for q, b in _SCHEME_BITS.items()
             if SCHEME_BACKEND[q] != expected[b]}
    assert not wrong, f"scheme -> backend disagrees with the package's bit widths: {wrong}"


def test_probe_does_not_route_through_the_package_primitive():
    """The probe must survive a totally broken `from_float` — that is the whole point of it.

    A probe that called the code under test could be broken by the bug it exists to reveal: if
    `from_float` raised for every input, the probe would raise too, every arm would skip, and the
    guard would be back to reporting a dead primitive as green. Pinned by breaking `from_float`
    outright and asserting the probe still reports the backend as available.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    from experts4bit_qlora import ExpertsNbit

    import quant_guard

    if quant_guard.quantize_unavailable_reason("cpu", "nf4", 64) is not None:
        pytest.skip("no bnb 4-bit backend here, so there is nothing to decouple from")

    def _boom(*a, **kw):
        raise RuntimeError("simulated total break of from_float")

    original, quant_guard._PROBE_CACHE = ExpertsNbit.from_float, {}
    try:
        ExpertsNbit.from_float = classmethod(lambda cls, *a, **kw: _boom())
        assert quant_guard.quantize_unavailable_reason("cpu", "nf4", 64) is None
    finally:
        ExpertsNbit.from_float = original
        quant_guard._PROBE_CACHE = {}
    assert torch is not None  # keep the importorskip binding used


def test_probe_leaves_the_global_rng_untouched():
    """It runs lazily, usually after an arm's `manual_seed` — drawing here would shift the arm."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")

    import quant_guard

    torch.manual_seed(0)
    expected = torch.randn(4)

    quant_guard._PROBE_CACHE = {}
    torch.manual_seed(0)
    quant_guard.quantize_unavailable_reason("cpu", "nf4", 64)
    got = torch.randn(4)
    quant_guard._PROBE_CACHE = {}

    assert torch.equal(expected, got), "the probe consumed global RNG draws"
