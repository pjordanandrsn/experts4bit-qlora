# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""Keep ``loader_guard.LOADER_REFUSALS`` complete as ``loader.py`` grows new refusals.

``loader_guard`` exists so a loader REFUSAL cannot masquerade as an absent bnb backend and report as
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

from loader_guard import LOADER_REFUSALS, is_loader_refusal

LOADER_SRC = Path(__file__).resolve().parents[1] / "experts4bit_qlora" / "loader.py"

#: Mirrors loader_guard.QUANTIZE_UNAVAILABLE. Spelled out rather than imported so that widening the
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
        "loader.py raises these with a type loader_guard swallows, but no LOADER_REFUSALS substring "
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
