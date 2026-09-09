# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The int4 store's module docstring must not contradict its dispatch (#496).

A module docstring is a secondary description of a primary artefact, and this
one drifted into asserting the opposite of what the code does: it said flatly
that *"batched decode keeps the NF4 M-tile path"* while
``hot_residency._fused_over_stack`` has had a grouped captured int4-b32 GEMM
for ``T > 1`` since the b16 work. Worse, the register publishes
``e4b.serve.b16.qwen3-30b.int4.5090`` as **measured** -- 1,238 tok/s
aggregate, x2.50 over the NF4 batched baseline -- so the module implementing
the path said the path does not run while a public claim quantified it.

These are text checks over the two files, deliberately. The behaviour they
guard cannot be exercised on a CPU runner: the batched branch needs
``int4_b32`` and a GPU. What a text check CAN do is refuse the specific
contradiction that actually happened, and fail if the batched branch is
removed while the docstring still describes it -- drift in either direction.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "experts4bit_qlora" / "engines"
INT4_DOC = (ROOT / "int4_experts.py").read_text()
HOT_RES = (ROOT / "hot_residency.py").read_text()


def _docstring(src: str) -> str:
    m = re.search(r'^"""(.*?)"""', src, re.S | re.M)
    assert m, "module docstring not found"
    return m.group(1)


def test_a_batched_int4_decode_branch_exists_in_dispatch():
    """The premise of every other test here. If this fails, the docstring
    should go back to describing an NF4-only batched path and this module
    should be deleted with it."""
    assert "gemm_int4_b32_grouped_captured" in HOT_RES, (
        "the grouped captured int4-b32 GEMM is gone from dispatch — "
        "re-check int4_experts.py's Scope note (#496)"
    )
    assert re.search(r"elif\s+device_grouping\s*:", HOT_RES),         "the device_grouping branch is gone from dispatch"


def test_the_docstring_does_not_claim_batched_decode_is_nf4_only():
    """The exact regression: a flat assertion that batched decode keeps NF4."""
    doc = _docstring(INT4_DOC)
    offending = re.search(
        r"[Bb]atched decode keeps\s+the NF4", re.sub(r"\s+", " ", doc))
    assert offending is None, (
        "int4_experts.py's docstring again asserts batched decode keeps the "
        "NF4 path, which contradicts the device_grouping branch in "
        "hot_residency and the measured e4b.serve.b16.qwen3-30b.int4.5090 "
        "claim (#496)"
    )


def test_the_docstring_names_the_flag_that_decides_the_branch():
    """A reader must be able to tell WHICH configuration they are in.

    The fault in #496 was not only a wrong sentence: the file gave no way to
    discover that a flag selects between two kernels.
    """
    doc = _docstring(INT4_DOC)
    assert "DEVICE_GROUPING" in doc, (
        "the docstring no longer names hot_residency.DEVICE_GROUPING, the flag "
        "that decides whether batched decode reads the int4 store (#496)"
    )


def test_the_default_is_stated_and_is_still_the_default():
    """The docstring says the flag defaults off; dispatch must agree.

    This is the pair that made the original sentence half-true and therefore
    hard to spot: NF4 IS what batched decode uses by default, because nothing
    in the package ever turns the flag on.
    """
    assert re.search(r"DEVICE_GROUPING\s*=\s*\[\s*False\s*\]", HOT_RES),         "DEVICE_GROUPING's default changed — update int4_experts.py's Scope note"
    pkg = Path(__file__).resolve().parents[1] / "experts4bit_qlora"
    assigns = [
        f"{p.relative_to(pkg)}:{i}"
        for p in pkg.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(r"DEVICE_GROUPING\s*\[\s*0\s*\]\s*=", line)
    ]
    assert not assigns, (
        f"the package now assigns DEVICE_GROUPING[0] at {assigns} — the "
        f"docstring's claim that nothing in the package sets it is stale (#496)"
    )


@pytest.mark.parametrize("phrase", [
    "grouped captured",   # names the batched kernel
    "T == 1",             # names the singleton branch
])
def test_the_scope_note_describes_both_branches(phrase):
    doc = re.sub(r"\s+", " ", _docstring(INT4_DOC))
    assert phrase in doc, f"Scope note no longer describes {phrase!r} (#496)"
