"""CPU-only transport failures; no provider calls or scientific measurements."""
import errno
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile
from urllib.error import HTTPError, URLError

import pytest

LANE = Path(__file__).parents[1] / "bench/dq11"
sys.path.insert(0, str(LANE))
import dq11_wheel_retry as retry  # noqa: E402

BASELINE_INSTALL = retry.bootstrap.install


class Response(io.BytesIO):
    status = 200


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    payload = b"exact pinned wheel bytes" * 128
    row = dict(name="fixture", filename="fixture.whl", version="1",
               url="https://files.pythonhosted.org/fixture.whl",
               sha256=hashlib.sha256(payload).hexdigest())
    (tmp_path / "wheels.json").write_text(json.dumps(dict(packages=[row])))
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(retry.time, "monotonic", lambda: clock[0])

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(retry.time, "sleep", sleep)
    installed = []

    def install(directory, cache):
        assert (cache / row["filename"]).read_bytes() == payload
        assert not list(cache.glob("*.partial"))
        installed.append((directory, cache))

    monkeypatch.setattr(retry.bootstrap, "install", install)
    return payload, row, clock, sleeps, installed


def outcomes(directory):
    return [json.loads(line) for line in (directory / "receipts/wheel-fetch-attempts.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("failures", [0, 1, 4])
@pytest.mark.parametrize("error", [ConnectionResetError(errno.ECONNRESET, "reset"),
                                  URLError(TimeoutError("timed out"))])
def test_transient_opener_failures_recover_exact_bytes(tmp_path, monkeypatch, fixture, failures, error):
    payload, row, clock, sleeps, installed = fixture
    calls = []

    def open_fixture(request, timeout):
        calls.append(request.full_url)
        assert request.get_header("User-agent") == retry.bootstrap.FETCH_USER_AGENT
        assert 0 < timeout <= 20
        if len(calls) <= failures:
            raise error
        return Response(payload)

    monkeypatch.setattr(retry.urllib.request, "urlopen", open_fixture)
    retry.install(tmp_path, tmp_path / "cache")
    assert len(installed) == 1
    assert calls == [row["url"]] * (failures + 1)
    assert sleeps == [5, 10, 20, 40][:failures]
    evidence = outcomes(tmp_path)
    assert len([r for r in evidence if r["outcome"] == "started"]) == failures + 1
    assert evidence[-1]["outcome"] == "verified"
    assert evidence[-1]["expected_sha256"] == evidence[-1]["observed_sha256"] == row["sha256"]


def test_reset_mid_transfer_discards_partial_before_full_retry(tmp_path, monkeypatch, fixture):
    payload, row, clock, sleeps, installed = fixture

    class BrokenResponse(Response):
        def read(self, size):
            if self.tell():
                raise URLError(ConnectionResetError(errno.ECONNRESET, "mid-read reset"))
            return super().read(12)

    responses = iter([BrokenResponse(payload), Response(payload)])
    monkeypatch.setattr(retry.urllib.request, "urlopen", lambda *a, **kw: next(responses))
    retry.install(tmp_path, tmp_path / "cache")
    assert len(installed) == 1 and sleeps == [5]
    assert outcomes(tmp_path)[1]["bytes"] == 12
    assert (tmp_path / "cache/fixture.whl").read_bytes() == payload


def test_wrong_bytes_after_transient_failure_refuse_without_install(tmp_path, monkeypatch, fixture):
    payload, row, clock, sleeps, installed = fixture
    calls = []

    def opener(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("transient")
        return Response(b"wrong wheel bytes")

    monkeypatch.setattr(retry.urllib.request, "urlopen", opener)
    with pytest.raises(ValueError, match="hash mismatch"):
        retry.install(tmp_path, tmp_path / "cache")
    assert len(calls) == 2 and not installed and sleeps == [5]
    assert not (tmp_path / "cache/fixture.whl").exists()
    assert not list((tmp_path / "cache").glob("*.partial"))
    last = outcomes(tmp_path)[-1]
    assert last["outcome"] == "refused"
    assert last["expected_sha256"] != last["observed_sha256"]


def test_five_failed_attempts_are_terminal(tmp_path, monkeypatch, fixture):
    payload, row, clock, sleeps, installed = fixture
    calls = []

    def opener(*a, **kw):
        calls.append(1)
        raise ConnectionResetError(errno.ECONNRESET, "persistent")

    monkeypatch.setattr(retry.urllib.request, "urlopen", opener)
    with pytest.raises(ConnectionResetError):
        retry.install(tmp_path, tmp_path / "cache")
    assert len(calls) == 5 and sleeps == [5, 10, 20, 40] and not installed
    assert outcomes(tmp_path)[-1]["outcome"] == "refused"


def test_deadline_prevents_more_network_attempts(tmp_path, monkeypatch, fixture):
    payload, row, clock, sleeps, installed = fixture
    calls = []

    def opener(*a, **kw):
        calls.append(1)
        clock[0] += 175
        raise TimeoutError("slow failure")

    monkeypatch.setattr(retry.urllib.request, "urlopen", opener)
    with pytest.raises(TimeoutError):
        retry.install(tmp_path, tmp_path / "cache")
    assert len(calls) == 1 and not sleeps and not installed
    assert outcomes(tmp_path)[-1]["elapsed_seconds"] == 175


def test_wall_deadline_interrupts_blocked_reader_and_restores_handler(tmp_path, monkeypatch):
    payload = b"wheel"
    row = dict(filename="fixture.whl", url="https://files.pythonhosted.org/fixture.whl",
               sha256=hashlib.sha256(payload).hexdigest())
    (tmp_path / "wheels.json").write_text(json.dumps(dict(packages=[row])))

    class BlockedResponse(Response):
        def read(self, size):
            time.sleep(5)
            pytest.fail("wall deadline must interrupt blocked read")

    monkeypatch.setattr(retry, "RETRY_SECONDS", 0.05)
    monkeypatch.setattr(retry.urllib.request, "urlopen", lambda *a, **kw: BlockedResponse(payload))
    monkeypatch.setattr(retry.bootstrap, "install", lambda *a: pytest.fail("expired fetch must not install"))
    previous = retry.signal.getsignal(retry.signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(retry.WheelDeadlineExceeded):
        retry.install(tmp_path, tmp_path / "cache")
    assert time.monotonic() - started < 2
    assert retry.signal.getsignal(retry.signal.SIGALRM) == previous
    assert retry.signal.getitimer(retry.signal.ITIMER_REAL)[0] == 0
    evidence = outcomes(tmp_path)
    assert [r["outcome"] for r in evidence] == ["started", "refused"]
    assert not list((tmp_path / "cache").glob("*.partial"))


@pytest.mark.parametrize("error", [HTTPError("https://files.pythonhosted.org/fixture.whl", 503, "unavailable", {}, None),
                                  URLError(OSError(errno.EACCES, "denied"))])
def test_nontransient_failure_does_not_retry(tmp_path, monkeypatch, fixture, error):
    payload, row, clock, sleeps, installed = fixture
    calls = []

    def opener(*a, **kw):
        calls.append(1)
        raise error

    monkeypatch.setattr(retry.urllib.request, "urlopen", opener)
    with pytest.raises(type(error)):
        retry.install(tmp_path, tmp_path / "cache")
    assert len(calls) == 1 and not sleeps and not installed


@pytest.mark.parametrize("valid", [False, True])
def test_cached_wheel_is_verified_without_fetch(tmp_path, monkeypatch, fixture, valid):
    payload, row, clock, sleeps, installed = fixture
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / row["filename"]).write_bytes(payload if valid else b"corrupt cache")
    monkeypatch.setattr(retry.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("cache must not fetch"))
    if valid:
        retry.install(tmp_path, cache)
        assert installed
    else:
        with pytest.raises(ValueError, match="hash mismatch"):
            retry.install(tmp_path, cache)
        assert not installed
    assert outcomes(tmp_path)[0]["outcome"] == "cache"


def test_entire_lock_checked_before_prefetch(tmp_path, monkeypatch, fixture):
    payload, row, clock, sleeps, installed = fixture
    bad = dict(row, url="https://unregistered.invalid/wheel.whl")
    (tmp_path / "wheels.json").write_text(json.dumps(dict(packages=[row, bad])))
    monkeypatch.setattr(retry.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("invalid lock must not fetch"))
    with pytest.raises(ValueError, match="unregistered wheel origin"):
        retry.install(tmp_path, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def test_frozen_installer_consumes_prefetched_cache_without_wheel_network(tmp_path, monkeypatch):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("fixture/__init__.py", "VALUE = 1\n")
    payload = stream.getvalue()
    row = dict(name="fixture", version="1", filename="fixture.whl",
               url="https://files.pythonhosted.org/fixture.whl",
               sha256=hashlib.sha256(payload).hexdigest())
    (tmp_path / "wheels.json").write_text(json.dumps(dict(packages=[row])))
    monkeypatch.setattr(retry.bootstrap, "install", BASELINE_INSTALL)
    monkeypatch.setattr(retry.bootstrap, "fetch_wheel", lambda *a, **kw: pytest.fail("frozen installer must use cache"))
    monkeypatch.setenv("E4B_SHA", "1" * 40)
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return Response(payload)

    commands = []
    monkeypatch.setattr(retry.urllib.request, "urlopen", opener)
    monkeypatch.setattr(retry.bootstrap.subprocess, "run", lambda command, **kw: commands.append(command))
    monkeypatch.setattr(retry.bootstrap.subprocess, "check_output", lambda *a, **kw: "")
    cache = tmp_path / "cache"
    retry.install(tmp_path, cache)
    assert calls == [row["url"]]
    assert "--no-index" in commands[0] and "--require-hashes" in commands[0]
    assert (cache / "requirements.local.txt").read_text() == f"fixture==1 --hash=sha256:{row['sha256']}\n"
    authority = json.loads((tmp_path / "source_authority.json").read_text())
    assert hashlib.sha256(b"VALUE = 1\n").hexdigest() in authority["sha256"]


def test_operational_helper_is_in_manifest_and_both_transport_hash_closures():
    import dq11_l_census_launch as launch

    assert "dq11_wheel_retry.py" in launch.OPERATIONAL_FILES
    drive = (LANE / "dq11_l_census_drive.sh").read_text()
    box = (LANE / "dq11_l_census_box.sh").read_text()
    assert "dq11_wheel_retry.py; do" in drive
    assert "dq11_wheel_retry.py ) >" in drive
    assert "dq11_wheel_retry.py > receipts/instrument-and-operational.sha256" in box
    assert "phase bootstrap 900 python dq11_wheel_retry.py" in box
    assert "sha256sum -c science.sha256" in box
