# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""`E4B_PIPELINED_TRAFFIC` parsing — CPU-testable on purpose.

Building an engine needs CUDA and pinned memory, so an `os.environ` read inlined
in `__init__` would ship unexercised on every machine without a GPU. The parse
lives in a module-level function so this can pin it anywhere.
"""
import pytest

from experts4bit_qlora.pipelined import traffic_counting_default


@pytest.mark.parametrize("value,expected", [
    (None, False),      # unset — counting is OFF by default, the whole point
    ("0", False),
    ("", False),
    ("no", False),
    ("2", False),       # only the accepted spellings count as on
    ("1", True),
    (" 1 ", True),      # tolerate stray whitespace from a shell export
    ("true", True),
    ("TRUE", True),
    ("yes", True),
    ("on", True),
])
def test_env_spellings(monkeypatch, value, expected):
    monkeypatch.delenv("E4B_PIPELINED_TRAFFIC", raising=False)
    if value is not None:
        monkeypatch.setenv("E4B_PIPELINED_TRAFFIC", value)
    assert traffic_counting_default() is expected


def test_default_is_off_when_unset(monkeypatch):
    """Stated separately because it is the behaviour change: the counters cost
    5.2% of the decode step and are now off unless asked for."""
    monkeypatch.delenv("E4B_PIPELINED_TRAFFIC", raising=False)
    assert traffic_counting_default() is False
