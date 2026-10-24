"""Assertions for a guard that must FAIL a test, not merely stop it.

``pytest.raises(pytest.fail.Exception)`` does not catch ``pytest.skip.Exception``: Skipped is its own outcome (both
derive from BaseException, neither from the other), so a guard whose CI branch regressed from ``pytest.fail`` to
``pytest.skip`` escapes the raises, the test reads SKIPPED, and CI stays green. These context managers catch both
outcomes and then require the right one, a subclass of it included."""
from __future__ import annotations

import contextlib
from typing import Iterator

import pytest

#: captured at import: a mutation check may replace pytest.fail, and with it pytest.fail.Exception
FAILED, SKIPPED = pytest.fail.Exception, pytest.skip.Exception


@contextlib.contextmanager
def must_fail(*also: type[BaseException], match: str | None = None) -> Iterator[pytest.ExceptionInfo]:
    """The body must FAIL (``pytest.fail``, or one of ``also``), its message matching ``match``. A skip of any Skipped
    subclass fails the test by name; a body that raises nothing fails as pytest.raises' DID NOT RAISE."""
    with pytest.raises((FAILED, SKIPPED, *also), match=match) as exc:
        yield exc
    if issubclass(exc.type, SKIPPED):
        pytest.fail(f"SKIPPED instead of failing: {exc.value}")


@contextlib.contextmanager
def expect_outcome(outcome: type[BaseException]) -> Iterator[pytest.ExceptionInfo]:
    """The body must end in ``outcome``, which is FAILED or SKIPPED (a subclass of it included); the other outcome,
    or no outcome at all, fails the test."""
    if outcome not in (FAILED, SKIPPED):
        raise TypeError(f"expect_outcome takes pytest.fail.Exception or pytest.skip.Exception, not {outcome!r}")
    with pytest.raises((FAILED, SKIPPED)) as exc:
        yield exc
    if not issubclass(exc.type, outcome):
        pytest.fail(f"expected {outcome.__name__}, got {exc.type.__name__}: {exc.value}")
