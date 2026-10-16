"""Operational exact-wheel prefetch; the frozen science installer stays unchanged.

Only connection resets and timeouts are retried, at the identical locked URL.
Every completed attempt must match the locked SHA256 before cache promotion.
The existing box phase deadline and teardown reserve still bound this command.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import json
from pathlib import Path
import signal
import time
from urllib.error import HTTPError, URLError
import urllib.request
from urllib.parse import urlparse

import dq11_bootstrap as bootstrap
from dq11_common import file_sha

MAX_ATTEMPTS = 5
RETRY_SECONDS = 180
BACKOFF_SECONDS = (5, 10, 20, 40)
REQUEST_TIMEOUT_SECONDS = 20


class WheelDeadlineExceeded(TimeoutError):
    """A terminal wall-clock limit, including a peer that dribbles bytes."""


@contextmanager
def wheel_deadline():
    """The Linux box CLI runs on the main thread; interrupt blocked network IO."""
    def expired(signum, frame):
        raise WheelDeadlineExceeded("wheel retry budget exhausted")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, RETRY_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def transient(error):
    """Classify only reset/timeout transport failures, never HTTP/hash refusals."""
    if isinstance(error, (HTTPError, WheelDeadlineExceeded)):
        return False
    if isinstance(error, URLError):
        return transient(error.reason)
    return (isinstance(error, (ConnectionResetError, TimeoutError))
            or isinstance(error, OSError) and error.errno in (errno.ECONNRESET, errno.ETIMEDOUT))


def record(log, **fields):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as target:
        target.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **fields}) + "\n")


def download(row, cache, log):
    """Fetch one locked wheel with bounded retries and durable per-attempt evidence."""
    path = cache / row["filename"]
    if path.is_file():
        observed = file_sha(path)
        record(log, url=row["url"], attempt=0, outcome="cache", expected_sha256=row["sha256"],
               observed_sha256=observed)
        if observed != row["sha256"]:
            raise ValueError("resolved wheel hash mismatch")
        return
    temporary = path.with_suffix(".partial")
    started = time.monotonic()
    deadline = started + RETRY_SECONDS
    for attempt in range(1, MAX_ATTEMPTS + 1):
        digest = hashlib.sha256()
        size = 0
        record(log, url=row["url"], attempt=attempt, outcome="started",
               expected_sha256=row["sha256"], elapsed_seconds=time.monotonic() - started)
        try:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("wheel retry budget exhausted")
            request = urllib.request.Request(row["url"], headers={"User-Agent": bootstrap.FETCH_USER_AGENT})
            with urllib.request.urlopen(request, timeout=min(REQUEST_TIMEOUT_SECONDS, left)) as source:
                if not 200 <= source.status < 300:
                    raise bootstrap.WheelFetchError(row["url"], source.status, "non-2xx response")
                with temporary.open("wb") as target:
                    while True:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("wheel retry budget exhausted")
                        block = source.read(1 << 20)
                        if not block:
                            break
                        digest.update(block)
                        size += len(block)
                        target.write(block)
            observed = digest.hexdigest()
            if observed != row["sha256"]:
                raise ValueError("resolved wheel hash mismatch")
        except (OSError, URLError, bootstrap.WheelFetchError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            delay = BACKOFF_SECONDS[attempt - 1] if attempt < MAX_ATTEMPTS else 0
            retry = (transient(error) and attempt < MAX_ATTEMPTS
                     and time.monotonic() + delay < deadline)
            record(log, url=row["url"], attempt=attempt, outcome="retry" if retry else "refused",
                   expected_sha256=row["sha256"], observed_sha256=digest.hexdigest(), bytes=size,
                   elapsed_seconds=time.monotonic() - started, error=str(error),
                   error_type=type(error).__name__, backoff_seconds=delay if retry else 0)
            if not retry:
                raise
            time.sleep(delay)
            continue
        temporary.replace(path)
        record(log, url=row["url"], attempt=attempt, outcome="verified", bytes=size,
               expected_sha256=row["sha256"], observed_sha256=observed,
               elapsed_seconds=time.monotonic() - started)
        return


def install(directory, cache):
    """Prefetch exact bytes, then let the original installer recheck and install."""
    packages = json.loads((directory / "wheels.json").read_text())["packages"]
    bootstrap.validate_wheel_origins(packages)
    for row in packages:
        if Path(row["filename"]).name != row["filename"]:
            raise ValueError("unsafe wheel filename")
        if urlparse(row["url"]).scheme != "https":
            raise ValueError("locked wheel requires HTTPS")
    cache.mkdir(parents=True, exist_ok=True)
    log = directory / "receipts/wheel-fetch-attempts.jsonl"
    for row in packages:
        with wheel_deadline():
            download(row, cache, log)
    bootstrap.install(directory, cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path, default=Path("/root/.cache/dq11-wheels"))
    args = parser.parse_args()
    install(args.directory, args.cache)
