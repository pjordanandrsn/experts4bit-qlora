"""tests/guard_outcomes.py: must_fail and expect_outcome require the right outcome, and turn every wrong one (a skip, a
Skipped subclass, the other outcome, a quiet body) into a failure."""
import pytest

from guard_outcomes import FAILED, SKIPPED, expect_outcome, must_fail


class _SkippedSubclass(SKIPPED):
    """A Skipped subclass: an `is` check would let it through, issubclass does not."""


def test_must_fail_passes_on_a_failure_with_its_message():
    with must_fail(match="history"):
        pytest.fail("CI must hold the full history")


def test_must_fail_accepts_a_listed_extra_exception():
    with must_fail(ValueError):
        raise ValueError("refused")


@pytest.mark.parametrize("body", [lambda: pytest.skip("skipped"),
                                  lambda: (_ for _ in ()).throw(_SkippedSubclass("sub"))],
                         ids=["skip", "skipped-subclass"])
def test_must_fail_fails_on_a_skip_or_a_skipped_subclass(body):
    with pytest.raises(FAILED, match="SKIPPED instead of failing"):
        with must_fail():
            body()


def test_must_fail_fails_on_a_quiet_body_and_on_a_wrong_message():
    with pytest.raises(FAILED, match="DID NOT RAISE"):
        with must_fail():
            pass
    with pytest.raises(AssertionError, match="Regex pattern did not match"):
        with must_fail(match="history"):
            pytest.fail("something else")


def test_must_fail_lets_an_unlisted_exception_through():
    with pytest.raises(KeyError):
        with must_fail():
            raise KeyError("not a guard outcome")


@pytest.mark.parametrize("outcome,body", [(FAILED, lambda: pytest.fail("f")), (SKIPPED, lambda: pytest.skip("s")),
                                          (SKIPPED, lambda: (_ for _ in ()).throw(_SkippedSubclass("sub")))],
                         ids=["fail", "skip", "skipped-subclass"])
def test_expect_outcome_passes_on_its_outcome(outcome, body):
    with expect_outcome(outcome):
        body()


@pytest.mark.parametrize("outcome,body,got", [(FAILED, lambda: pytest.skip("s"), "Skipped"),
                                              (FAILED, lambda: (_ for _ in ()).throw(_SkippedSubclass("sub")),
                                               "_SkippedSubclass"),
                                              (SKIPPED, lambda: pytest.fail("f"), "Failed")],
                         ids=["fail-got-skip", "fail-got-skipped-subclass", "skip-got-fail"])
def test_expect_outcome_fails_on_the_other_outcome(outcome, body, got):
    with pytest.raises(FAILED, match=f"expected {outcome.__name__}, got {got}"):
        with expect_outcome(outcome):
            body()


def test_expect_outcome_fails_on_a_quiet_body_and_refuses_another_class():
    with pytest.raises(FAILED, match="DID NOT RAISE"):
        with expect_outcome(FAILED):
            pass
    with pytest.raises(TypeError):
        with expect_outcome(ValueError):
            pass
