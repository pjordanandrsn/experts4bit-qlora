"""CPU route, staged-rate and real process deadline regressions."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

LANE = Path(__file__).parents[1] / "bench/dq11"
sys.path.insert(0, str(LANE))
import dq11_census_transport as transport  # noqa: E402


def test_offer_and_instance_direct_endpoint_refuse_missing_or_foreign_route():
    assert transport.direct_available({"public_ipaddr": "8.8.8.8", "direct_port_count": 1})
    for row in [
        {},
        {"public_ipaddr": "127.0.0.1", "direct_port_count": 1},
        {"public_ipaddr": "ssh2.vast.ai", "direct_port_count": 2},
        {"public_ipaddr": "8.8.8.8", "direct_port_count": 0},
        {"public_ipaddr": "8.8.8.8", "direct_port_count": True},
    ]:
        assert not transport.direct_available(row)
    row = {"id": 12, "public_ipaddr": "8.8.8.8", "ports": {"22/tcp": [{"HostPort": "1234"}]}}
    assert transport.direct_endpoint(row, "12") == ("8.8.8.8", 1234)
    for bad in [
        dict(row, id=13),
        dict(row, ports={}),
        dict(row, public_ipaddr="172.31.255.254"),
        dict(row, ports={"22/tcp": [{"HostPort": "0"}]}),
        dict(row, ports={"22/tcp": [{"HostPort": "1234"}, {"HostPort": "1235"}]}),
    ]:
        with pytest.raises(ValueError):
            transport.direct_endpoint(bad, "12")


def test_external_watchdog_cannot_be_cancelled_by_child_alarm(tmp_path):
    marker = tmp_path / "late"
    code = (
        "import signal,time,pathlib; signal.setitimer(signal.ITIMER_REAL,0); "
        f'time.sleep(1); pathlib.Path({str(marker)!r}).write_text("bad")'
    )
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        transport.supervised([sys.executable, "-c", code], 0.15)
    assert time.monotonic() - start < 2
    assert not marker.exists()


def test_outer_itimer_set_before_exec_survives_complete_wheel_fetch(tmp_path):
    # An actual inherited timer, not a mock of signal APIs. The old helper cancelled it.
    code = f"""
import sys,io,json,hashlib,time
from pathlib import Path
sys.path.insert(0,{str(LANE)!r})
import dq11_wheel_retry as w
root=Path({str(tmp_path)!r}); payload=b"exactwheel"
row=dict(name="x",version="1",filename="x.whl",url="https://files.pythonhosted.org/x.whl",sha256=hashlib.sha256(payload).hexdigest())
(root/"wheels.json").write_text(json.dumps(dict(packages=[row])))
class Response(io.BytesIO): status=200
w.urllib.request.urlopen=lambda *a,**k: Response(payload)
w.bootstrap.install=lambda *a: None
w.install(root,root/"cache")
(root/"fetch-complete").write_text("yes")
time.sleep(2)
"""
    wrapper = (
        "import os,signal,sys; signal.setitimer(signal.ITIMER_REAL,.8); "
        'os.execv(sys.executable,[sys.executable,"-c",sys.argv[1]])'
    )
    process = subprocess.Popen([sys.executable, "-c", wrapper, code])
    assert process.wait(timeout=4) == -signal.SIGALRM
    assert (tmp_path / "fetch-complete").exists()


def test_stage_refuses_a_throttled_process_before_remainder(tmp_path, monkeypatch):
    bulk = tmp_path / "bulk"
    bulk.write_bytes(b"x" * 100)
    small = tmp_path / "small"
    small.write_bytes(b"y")
    stub = tmp_path / "scp.py"
    stub.write_text("import time; time.sleep(2)\n")
    monkeypatch.setattr(transport, "STAGING_SECONDS", 0.15)
    monkeypatch.setenv("E4B_RENT_DEADLINE_EPOCH", str(int(time.time()) + 1000))
    log = tmp_path / "rate.jsonl"
    with pytest.raises(subprocess.TimeoutExpired):
        transport.stage([small, bulk], [sys.executable, str(stub)], "unused", log)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]["first_file"] == "bulk"
    assert rows[-1]["phase"] == "stage-refused"
    assert not any(r["phase"] == "stage-pass" for r in rows)


def test_stage_projects_first_bulk_and_shares_remaining_budget(tmp_path, monkeypatch):
    bulk = tmp_path / "bulk"
    bulk.write_bytes(b"x" * 100)
    small = tmp_path / "small"
    small.write_bytes(b"y" * 100)
    monkeypatch.setenv("E4B_RENT_DEADLINE_EPOCH", str(int(time.time()) + 1000))
    clock = [0.0]
    calls = []

    def run(cmd, timeout):
        calls.append((cmd, timeout))
        clock[0] += 10
        return 0

    log = tmp_path / "log"
    transport.stage([bulk, small], ["scp"], "remote", log, clock=lambda: clock[0], run=run)
    assert calls[0][1] == 90 and calls[1][1] == 170
    assert json.loads(log.read_text().splitlines()[-1])["elapsed_seconds"] == 20
    assert transport.projected_seconds(200, 100, 100) == 200


def test_drive_binds_instance_get_to_direct_environment_and_records_trust(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []
    record = {"id": 12, "public_ipaddr": "8.8.8.8", "ports": {"22/tcp": [{"HostPort": "1234"}]}}

    class Wire:
        def __init__(self, key):
            pass

        def request(self, method, path, **kwargs):
            calls.append((method, path))
            return 200, record

    class Provider:
        def __init__(self, wire, **kwargs):
            self.wire = wire

        def instance(self, iid):
            with pytest.raises(ValueError, match="instance-GET-only"):
                self.wire.request("POST", "/v0/asks/1/")
            return self.wire.request("GET", f"/v0/instances/{iid}/")[1]

    monkeypatch.setitem(
        sys.modules,
        "adertha.compute.vast_provider",
        SimpleNamespace(VastProvider=Provider, UrllibTransport=Wire, load_api_key=lambda: "fixture"),
    )
    for key, value in {
        "ADERTHA_REPO": str(tmp_path),
        "E4B_RENT_INSTANCE_ID": "12",
        "E4B_RENT_RUN_ID": "fixture",
        "E4B_RENT_RUN_DIR": str(tmp_path),
        "E4B_RENT_SSH_HOST": "ssh2.vast.ai",
        "E4B_RENT_SSH_PORT": "100",
    }.items():
        monkeypatch.setenv(key, value)
    captured = []
    monkeypatch.setattr(transport.os, "execvpe", lambda exe, cmd, env: captured.append(env))
    transport.drive(["bash", "fixture.sh"])
    assert calls == [("GET", "/v0/instances/12/")]
    assert captured[0]["E4B_RENT_SSH_HOST"] == "8.8.8.8"
    assert captured[0]["E4B_RENT_SSH_PORT"] == "1234"
    assert "direct-known_hosts" in captured[0]["E4B_RENT_SSH_OPTS"]
    evidence = json.loads((tmp_path / "direct-transport.jsonl").read_text())
    assert evidence["proxy_fallback"] is False
    assert "first-connect" in evidence["host_key_policy"]


def test_phase_cli_records_named_timing_and_refuses_exhausted_reserve(tmp_path):
    (tmp_path / "receipts").mkdir()
    helper = str(LANE / "dq11_census_transport.py")
    base = [sys.executable, helper, "phase", "--seconds", "2", "--name", "fixture"]
    command = [sys.executable, "-c", 'print("completed")']
    passed = subprocess.run(
        [*base, "--deadline", str(int(time.time()) + 1000), *command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert passed.returncode == 0, passed.stderr
    refused = subprocess.run(
        [*base, "--deadline", str(int(time.time()) + 299), *command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 11 and "completed" not in refused.stdout
    rows = [json.loads(line) for line in (tmp_path / "receipts/phase-timings.jsonl").read_text().splitlines()]
    assert [r["name"] for r in rows] == ["fixture"] * 4
    assert rows[1]["result"] == 0 and rows[-1]["result"] == "refused"


def test_watchdog_observes_exit_without_reaping_then_kills_group_then_reaps(monkeypatch):
    from types import SimpleNamespace

    events = []

    class Child:
        pid = 123
        returncode = None

        def wait(self):
            events.append("reap")
            self.returncode = 0

    monkeypatch.setattr(transport.subprocess, "Popen", lambda *a, **kw: Child())

    def waitid(kind, pid, flags):
        assert kind == os.P_PID and pid == 123
        assert flags & os.WNOWAIT and flags & os.WEXITED
        events.append("observe-unreaped")
        return SimpleNamespace(si_pid=123)

    monkeypatch.setattr(transport.os, "waitid", waitid)
    monkeypatch.setattr(transport.os, "killpg", lambda *a: events.append("signal-group"))
    assert transport.supervised(["fixture"], 1) == 0
    assert events == ["observe-unreaped", "signal-group", "reap"]


def test_slow_setup_not_multiplied_into_first_bulk_projection(tmp_path, monkeypatch):
    from contextlib import contextmanager

    a = tmp_path / "a"
    a.write_bytes(b"a" * 100)
    b = tmp_path / "b"
    b.write_bytes(b"b" * 100)
    clock = [0.0]
    monkeypatch.setenv("E4B_RENT_DEADLINE_EPOCH", str(int(time.time()) + 1000))

    @contextmanager
    def connect(*args):
        clock[0] = 99
        yield ["scp"], 99

    def run(*args):
        clock[0] += 1
        return 0

    monkeypatch.setattr(transport, "direct_connection", connect)
    log = tmp_path / "log"
    transport.stage([a, b], ["scp"], "remote", log, ssh=["ssh"], clock=lambda: clock[0], run=run)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[1]["connection_setup_seconds"] == 99
    assert rows[1]["transfer_seconds"] == 1
    assert rows[1]["projected_total_seconds"] == 101  # not (99 + 1) * 2
    assert rows[-1]["phase"] == "stage-pass"


@pytest.mark.parametrize("edge", ["deadline", "host-key", "empty-phase"])
def test_cli_edges_are_coded_refusals_without_tracebacks(tmp_path, edge):
    file = tmp_path / "asset"
    file.write_bytes(b"x")
    env = dict(
        os.environ,
        E4B_RENT_RUN_DIR=str(tmp_path),
        E4B_RENT_SSH_KNOWN_HOSTS=str(tmp_path / "missing"),
        E4B_RENT_DEADLINE_EPOCH=str(int(time.time()) + 1000),
    )
    base = [sys.executable, str(LANE / "dq11_census_transport.py")]
    if edge == "empty-phase":
        args = ["phase", "--seconds", "1", "--deadline", str(int(time.time()) + 1000), "--name", "empty"]
    elif edge == "deadline":
        env.pop("E4B_RENT_DEADLINE_EPOCH", None)
        args = ["stage", "--scp", "scp", "--ssh", "ssh root@8.8.8.8", "--remote", "unused", str(file)]
    else:
        args = ["stage", "--scp", "scp", "--ssh", "ssh root@8.8.8.8", "--remote", "unused", str(file)]
    result = subprocess.run([*base, *args], cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 11
    assert "CENSUS_REFUSAL:" in result.stderr and "Traceback" not in result.stderr
