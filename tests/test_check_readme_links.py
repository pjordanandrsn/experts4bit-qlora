"""The link checker's retry policy, which has exactly one way to be dangerous.

Retrying is added because GitHub throttles ~30 unauthenticated HEADs from a
shared Actions IP, and a gate that fires on load teaches people to re-run it —
after which the next REAL dead link gets re-run too. One nearly did: a genuine
404 sat inside a batch of rate-limit errors and was dismissed along with them.

So the property under test is not "retrying works". It is that retrying is
**scoped to answers that are not verdicts**. A 404 must fail on the first
response and must never be retried into a pass, because turning a dead link
green is the one outcome this script exists to prevent.

Network is never touched here: `_Session` is substituted -- the seam `_head`
actually calls -- which also lets each case assert the exact number of attempts
made.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_readme_links as clc  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Backoff is real seconds; the policy is what is under test, not the wait."""
    monkeypatch.setattr(clc.time, "sleep", lambda *_a: None)


class _FakeSession:
    """Stands in for `_Session`, which is the seam `_head` actually calls.

    An earlier cut of this file stubbed `urllib.request.urlopen` instead. When
    the transport moved to a pooled `http.client` connection those stubs went
    inert and the tests hit the real network — nine of them failed loudly, which
    is the good outcome, but the lesson is that a stub has to sit on the seam the
    code uses, not the one it used to use.
    """

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def head(self, url, headers, _depth=0):
        self.calls.append(url)
        o = self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]
        if isinstance(o, Exception):
            raise o
        return o


def _install(_monkeypatch, outcomes):
    """A session serving `outcomes` in order. `.calls` records one entry per
    attempt actually made, so each case can assert the retry count."""
    return _FakeSession(outcomes)


# ------------------------------------------------- the dangerous case, first --

def test_a_404_fails_immediately_and_is_never_retried(monkeypatch):
    """THE property. A dead link is an answer, not weather. If it were retried it
    would still fail here — but it would also burn the backoff budget, and the
    reported error would read like throttling, which is exactly how the last real
    404 got dismissed."""
    s = _install(monkeypatch, [404])
    status, err = clc._head("https://github.com/nope", {}, s)
    assert (status, err) == (404, None)
    assert len(s.calls) == 1, f"a 404 was retried {len(s.calls)} times"


def test_a_403_is_also_a_verdict(monkeypatch):
    s = _install(monkeypatch, [403])
    status, err = clc._head("https://github.com/private", {}, s)
    assert (status, err) == (403, None)
    assert len(s.calls) == 1


# ------------------------------------------------------ the transient cases --

@pytest.mark.parametrize("transient", [503, 429, 502, 500])
def test_a_throttled_answer_is_retried_and_can_succeed(monkeypatch, transient):
    """The failure that made this necessary: 503 and dropped connections on links
    whose paths demonstrably exist."""
    s = _install(monkeypatch, [transient, 200])
    assert clc._head("https://github.com/ok", {}, s) == (200, None)
    assert len(s.calls) == 2


def test_a_dropped_connection_is_retried(monkeypatch):
    s = _install(monkeypatch, [ConnectionResetError("Remote end closed"), 200])
    assert clc._head("https://github.com/ok", {}, s) == (200, None)
    assert len(s.calls) == 2


def test_persistent_throttling_is_reported_as_unanswered_not_dead(monkeypatch):
    """It still fails — but as "never answered", so nobody reads 28 rate-limit
    errors as 28 dead links."""
    s = _install(monkeypatch, [503])
    status, err = clc._head("https://github.com/slow", {}, s)
    assert status is None
    assert "503" in err and "attempts" in err
    assert len(s.calls) == clc.ATTEMPTS


def test_success_on_the_first_try_makes_exactly_one_request(monkeypatch):
    s = _install(monkeypatch, [200])
    assert clc._head("https://github.com/fine", {}, s) == (200, None)
    assert len(s.calls) == 1


# ------------------------------------------------------------------- auth --

def test_the_token_is_sent_when_present(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "t0ken")
    assert clc._auth_headers() == {"Authorization": "Bearer t0ken"}


def test_no_token_is_not_an_error(monkeypatch):
    """The retry loop is what carries this fix; auth is best-effort. Running
    without a token has to stay supported, or every local invocation breaks."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert clc._auth_headers() == {}


def test_retryable_set_excludes_every_4xx_verdict():
    """A drift guard: adding 404 or 403 to RETRYABLE_STATUS would silently make
    dead links survivable, and no other test in this file would notice if the
    set were edited rather than the code."""
    assert 404 not in clc.RETRYABLE_STATUS
    assert 403 not in clc.RETRYABLE_STATUS
    assert 401 not in clc.RETRYABLE_STATUS
    assert {429, 503}.issubset(clc.RETRYABLE_STATUS)
