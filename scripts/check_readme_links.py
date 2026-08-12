"""Release-time long_description link check: zero relative links, self-repo
links pin the current version tag, and every GitHub link answers 200.
Ref-consistency and --map-ref are scoped to THIS repo's URLs only — third-party
pins (e.g. upstream docs at their own tags) are checked for 200 but never
rewritten or version-matched (Bugbot, e4b PR #31 round 4).

**Throttling is not a dead link.** This checks ~40 GitHub URLs, and it used to
open a fresh TLS connection for every one of them — `urllib.request.urlopen`
gives you no choice. GitHub's edge drops some of that churn, which produced
three red runs in one day, once reporting 28 of 28 links "dead" (the CI badge,
LICENSE and third-party URLs included) on a tree where all 23 self-link paths
demonstrably existed. A gate that fires on load teaches people to re-run it, and
the next real dead link gets re-run too. One nearly did: a genuine 404 sat inside
a batch of rate-limit errors and was dismissed along with them.

**The fix is connection reuse**, not retrying. That was established by
measurement, not assumed: a URL that failed four `urlopen` attempts in a row
answered 200 three times in a row under `curl`, so it was never the URL, the
method, or the User-Agent. One pooled keep-alive connection per host took the
same README from 4 failures in 94 s to 40/40 in 33 s, unauthenticated, four runs
running.

Retry and auth are the belt to that suspenders. Retry covers the responses that
mean "ask again" and **only** those: 404 and 403 are answers, not weather, and
are never retried into a pass, because turning a dead link into a green check is
the one failure this script must not have.
"""
import http.client
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SKIP_SUBSTRINGS = ()

#: Statuses that mean "the server did not answer the question", not "no".
#: 429 = explicit rate limit; 5xx = server-side. Anything else, including 404,
#: is a verdict and is taken at face value.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

ATTEMPTS = 4
BACKOFF_BASE = 1.5      # seconds; grows 1.5 / 3 / 6 with jitter


def _auth_headers() -> dict:
    """Send the Actions-provided token when there is one.

    Authenticated traffic gets a higher ceiling, but this is best-effort and NOT
    what fixed the flakiness: the limit that bit here is on github.com HTML, not
    the REST API, and the check passes unauthenticated once connections are
    reused. Kept because it costs nothing and helps the API-shaped URLs (badges,
    workflow pages) — do not read a green run as evidence that the token worked.
    """
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _forward_headers(headers: dict, src: str, dst: str) -> dict:
    """Headers to carry across a redirect — **without** the credential when the
    destination is not the same origin.

    `urllib.request.urlopen` strips `Authorization` on a cross-host redirect;
    a hand-rolled `http.client` loop does not, and rebuilding the transport is
    exactly how that protection gets dropped by accident. GitHub 302s assets to
    `*.githubusercontent.com` and object storage, so forwarding blindly would
    hand the Actions token to hosts that have no business seeing it.

    Same origin means same scheme AND same host. A downgrade to http is treated
    as foreign even on the same host: the token would then cross the wire in
    clear text. (Cursor Bugbot, gnf4 #47: "Auth header follows cross-host
    redirects".)
    """
    s, d = urllib.parse.urlsplit(src), urllib.parse.urlsplit(dst)
    if (s.scheme, s.netloc) == (d.scheme, d.netloc) and d.scheme == "https":
        return headers
    return {k: v for k, v in headers.items() if k.lower() != "authorization"}


class _Session:
    """One keep-alive connection per host, reused across every link.

    THIS is the actual fix, not the retry loop. `urllib.request.urlopen` opens a
    fresh TLS connection per call, so checking ~30 links means ~30 handshakes to
    github.com in a tight loop, and GitHub's edge drops a few of them. Measured:
    the same URL that failed four urlopen attempts in a row answered 200 three
    times in a row under curl — it was never the URL, the method, or the
    User-Agent, only the connection churn. Reusing one connection collapses that
    to a single handshake.

    Redirects are followed by hand because `http.client` does not, and GitHub
    does redirect some paths (renames, `tree/` -> canonical). Missing that would
    turn a live link into a spurious 301.
    """

    def __init__(self):
        self._conns = {}

    def _conn(self, host: str):
        c = self._conns.get(host)
        if c is None:
            c = self._conns[host] = http.client.HTTPSConnection(host, timeout=20)
        return c

    def drop(self, host: str) -> None:
        """Discard a connection the server closed, so the next try re-handshakes
        instead of reusing a socket that is already gone."""
        c = self._conns.pop(host, None)
        if c is not None:
            try:
                c.close()
            except Exception:
                pass

    def head(self, url: str, headers: dict, _depth: int = 0) -> int:
        parts = urllib.parse.urlsplit(url)
        host = parts.netloc
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        conn = self._conn(host)
        try:
            conn.request("HEAD", path or "/",
                         headers={"User-Agent": "link-check", "Accept": "*/*",
                                  "Connection": "keep-alive", **headers})
            resp = conn.getresponse()
            status, loc = resp.status, resp.getheader("Location")
            resp.read()
        except Exception:
            self.drop(host)
            raise
        if status in (301, 302, 303, 307, 308) and loc and _depth < 5:
            nxt = urllib.parse.urljoin(url, loc)
            return self.head(nxt, _forward_headers(headers, url, nxt), _depth + 1)
        return status

    def close(self):
        for host in list(self._conns):
            self.drop(host)


def _head(url: str, headers: dict, session=None):
    """``(status, error)`` for one HEAD, retrying only transient answers.

    Returns ``(200, None)`` on success, ``(status, None)`` for a definitive
    non-200, or ``(None, str)`` when every attempt failed to get an answer.
    """
    session = session or _Session()
    last = None
    for attempt in range(ATTEMPTS):
        try:
            status = session.head(url, headers)
            if status == 200:
                return 200, None
            if status not in RETRYABLE_STATUS:
                return status, None                # a verdict: do not retry
            last = f"HTTP {status}"
        except urllib.error.HTTPError as e:         # kept: _Session may be stubbed
            if e.code not in RETRYABLE_STATUS:
                return e.code, None                # 404/403: answered, and it is a no
            last = f"HTTP {e.code}"
        except Exception as e:                      # connection reset, timeout, DNS
            last = str(e)
        if attempt < ATTEMPTS - 1:
            # Jittered exponential backoff: a fixed sleep would re-synchronise
            # every request onto the same window that just throttled them.
            time.sleep(BACKOFF_BASE * (2 ** attempt) * (0.5 + random.random()))
    return None, f"{last} (after {ATTEMPTS} attempts)"


def _self_slug() -> str:
    url = subprocess.run(["git", "remote", "get-url", "origin"],
                         capture_output=True, text=True, check=True).stdout.strip()
    m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", url)
    assert m, f"cannot derive owner/repo from origin: {url}"
    return m.group(1)


def main() -> int:
    args = sys.argv[1:]
    ref_map = None
    if args[:1] == ["--map-ref"]:
        ref_map = (args[1], args[2])
    slug = _self_slug()
    self_prefix = f"https://github.com/{slug}/"
    text = open("README.md").read()
    targets = re.findall(r"\]\(([^)]+)\)", text)
    rel = [t for t in targets if not t.startswith(("http://", "https://", "#"))]
    assert not rel, f"relative links present (dead on PyPI): {rel}"
    # ref-consistency (SELF-repo links only): every pinned blob/tree ref must be
    # the CURRENT version's tag, else a bump ships docs pointing at the old tag.
    try:
        import tomllib
        want = "v" + tomllib.load(open("pyproject.toml", "rb"))["project"]["version"]
        refs = set(re.findall(
            re.escape(self_prefix) + r"(?:blob|tree)/([^/]+)/", text))
        stale = refs - {want, "main"}
        assert not stale, f"README pins self-repo ref(s) {sorted(stale)} but project.version wants {want}"
    except FileNotFoundError:
        pass
    gh = [t for t in targets if t.startswith("https://github.com/")]
    if ref_map:
        gh = [t.replace(f"/{ref_map[0]}/", f"/{ref_map[1]}/")
              if t.startswith(self_prefix) else t
              for t in gh]
    headers = _auth_headers()
    session = _Session()
    bad, unanswered = [], []
    for t in gh:
        if any(s in t for s in SKIP_SUBSTRINGS):
            continue
        status, err = _head(t, headers, session)
        if err is not None:
            unanswered.append((t, err))
        elif status != 200:
            bad.append((t, status))
    # Separated on purpose, and the dead ones are reported FIRST. A real 404 was
    # once lost inside a batch of rate-limit errors and dismissed with them;
    # printing them apart means a genuine dead link cannot hide in the weather.
    session.close()
    assert not bad, f"dead GitHub links: {bad}"
    assert not unanswered, (
        f"{len(unanswered)} link(s) never answered after {ATTEMPTS} attempts — "
        f"GitHub throttling, not necessarily dead: {unanswered}")
    print(f"link check OK: 0 relative, self-repo refs consistent, "
          f"{len(gh)} GitHub links answer 200"
          f"{' (authenticated)' if headers else ' (unauthenticated)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
