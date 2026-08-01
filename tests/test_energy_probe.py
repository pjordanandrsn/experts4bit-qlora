"""RAPL wrap arithmetic — the part that cannot be caught by running the probe.

The package energy counter is 32 bits. At this hardware's 61 uJ tick and ~60 W it wraps
every ~73 minutes, so a wrap bug does not show up in any normal test run: it silently
drops a full 262 kJ period, or reports a negative delta that quietly cancels real energy.
"""
import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "energy_probe", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "energy_probe.py")
energy_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(energy_probe)

WRAP = 1 << 32
wrap_delta = energy_probe.wrap_delta


def test_normal_increase():
    assert wrap_delta(1000, 1500) == 500


def test_no_change_is_zero():
    assert wrap_delta(42, 42) == 0


def test_wrap_is_not_negative():
    """The failure this exists to prevent: a naive cur-prev goes negative and cancels
    real energy already accumulated."""
    prev, cur = WRAP - 100, 50
    assert cur - prev < 0
    assert wrap_delta(prev, cur) == 150


def test_wrap_at_the_exact_boundary():
    assert wrap_delta(WRAP - 1, 0) == 1


def test_accumulation_across_a_wrap_matches_the_unwrapped_total():
    seq = [WRAP - 300, WRAP - 100, 100, 400]      # crosses once
    total = sum(wrap_delta(a, b) for a, b in zip(seq, seq[1:]))
    assert total == 700


@pytest.mark.parametrize("prev,cur,want", [(0, 1, 1), (WRAP - 1, WRAP - 1, 0), (5, 4, WRAP - 1)])
def test_edges(prev, cur, want):
    assert wrap_delta(prev, cur) == want


def test_a_tiny_decrease_is_read_as_a_wrap_not_a_negative():
    """Single-wrap is the documented limit; a 1-tick decrease is assumed to be a wrap
    rather than clamped to zero, which is what keeps accumulation monotonic."""
    assert wrap_delta(5, 4) > 0
