"""Every per-chunk field the streamed calibration prints must be MERGED, not read once.

Streamed calibration (``enable_serve_experts_int4_calibrated``) enables layers in chunks and
prints its progress once per chunk. A reducer that reads the first line describes one chunk and
calls it the run. That has now bitten this lane twice: #536 fixed the gptq/rtn counts, and P39
box 4 then reported 140 overridden expert-roles where the five chunks summed to 714 -- the same
bug on the field next door, because the fix was applied per-field instead of per-shape.

So the shape gets a test: give each reader a log with SEVERAL chunks whose values differ, and
require the sum. A first-line read cannot pass these.
"""
import importlib.util
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "bench" / "p39" / "p39_reduce.py"
_spec = importlib.util.spec_from_file_location("p39_reduce", _SRC)
p39_reduce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p39_reduce)

SHA = "sha256:" + "60b3b951" * 8

# The five chunks P39 box 4 actually printed (48 layers as 10/10/10/10/8).
BOX4_LOG = "\n".join(
    f"INT4EXP calibrated experts: {g} gptq / {r} rtn (min_rows=32) over {n} layers\n"
    f"INT4EXP assignment honoured {SHA}: {d} expert-roles where local routing "
    "disagrees with the record (observed, not applied)"
    for g, r, n, d in [(2282, 278, 10, 140), (2254, 306, 10, 148), (2260, 300, 10, 154),
                       (2190, 370, 10, 150), (1834, 214, 8, 122)]
)


def test_counts_sum_across_chunks():
    assert p39_reduce._counts(BOX4_LOG) == (10820, 1468)


def test_disagreements_sum_across_chunks():
    """714, not the 140 of chunk one."""
    assert p39_reduce._honoured(BOX4_LOG) == (SHA, 714)


def test_no_calibration_lines_is_none_not_zero():
    """A run that never calibrated has no number; zero would read as 'agreed everywhere'."""
    assert p39_reduce._counts("nothing here") is None
    assert p39_reduce._honoured("nothing here") is None


def test_two_records_in_one_run_refuses():
    """Summing across chunks is only meaningful when every chunk honoured the SAME record."""
    other = "sha256:" + "ab" * 32
    mixed = BOX4_LOG + (
        f"\nINT4EXP assignment honoured {other}: 7 expert-roles where local routing disagrees")
    with pytest.raises(ValueError, match="2 different records"):
        p39_reduce._honoured(mixed)
