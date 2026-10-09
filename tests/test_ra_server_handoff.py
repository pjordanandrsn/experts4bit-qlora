"""Synthetic server startup/lifecycle controls; never HTTP/GPU proof."""

import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def fixture(name):
    sp = importlib.util.spec_from_file_location("server_" + name, ROOT / "tests" / (name + ".py"))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


c = fixture("test_ra_capacity")
s = fixture("test_ra_sc2_handoff")
worker = s.f.worker


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def listening():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock


def test_socket_identity_does_not_close_inherited_descriptor(tmp_path):
    with listening() as sock:
        if sys.platform == "darwin":
            with pytest.raises(OSError):
                c.server_wrapper.listener_identity(sock.fileno())
            identity = fixture_identity(sock)
        else:
            identity = c.server_wrapper.listener_identity(sock.fileno())
        assert identity["address"] == list(sock.getsockname())
        assert identity["inode"] == os.fstat(sock.fileno()).st_ino
        if sys.platform != "darwin":
            assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
    p = tmp_path / "ordinary"
    p.write_text("file")
    with p.open("rb") as stream, pytest.raises(ValueError, match="not a socket"):
        c.server_wrapper.listener_identity(stream.fileno())
    with socket.socket() as sock, pytest.raises(ValueError, match="loopback"):
        c.server_wrapper.listener_identity(sock.fileno())


def fixture_identity(sock):
    # Darwin lacks SO_ACCEPTCONN getsockopt. Fixture claims identity only;
    # the production listener check is separately exercised on native Linux.
    st = os.fstat(sock.fileno())
    return {"fd": sock.fileno(), "device": st.st_dev, "inode": st.st_ino, "address": list(sock.getsockname())}


def setup(tmp_path, sock, effect=""):
    m, j, py, tool = s.setup(tmp_path)
    parent = Path(j.pop("sc2")["parent_out"])
    (parent / "server-child").mkdir()
    j.update(
        schema=4,
        phase="capacity_server",
        native_out=str(parent / "server-child/native"),
        server={"parent_pid": os.getpid(), "parent_out": str(parent), "listener": fixture_identity(sock)},
    )
    body = (ROOT / "bench/ra/ra_capacity_server.py").read_text().split('if __name__ == "__main__":')[0]
    if sys.platform == "darwin":
        body += "\ndef listener_identity(fd):\n    st=os.fstat(fd)\n    with socket.fromfd(fd,socket.AF_INET,socket.SOCK_STREAM) as probe: address=probe.getsockname()\n    return {'fd':fd,'device':st.st_dev,'inode':st.st_ino,'address':list(address)}\n"
    stub = """
if __name__ == '__main__':
    import json,hashlib
    assert 'ra_probe_e4b' in sys.modules and 'ra_probe_gnf4' in sys.modules
    a=dict(zip(sys.argv[1::2],sys.argv[2::2]))
    identity=listener_identity(int(a['--socket-fd']))
    out=Path(a['--receipt']);parent=out.parents[2]
    for n in ('request-trace.jsonl','step-trace.jsonl'):
        (parent/n).write_text('synthetic trace')
    record={'status':'SERVER_RETURNED_PENDING_GATES','pid':os.getpid(),'parent_pid':os.getppid(),
        'listener':identity,'closed':{'closed':True,'thread_alive':False},'engine_thread_alive':False,
        'trace_sha256':{n:hashlib.sha256((parent/n).read_bytes()).hexdigest()
        for n in ('request-trace.jsonl','step-trace.jsonl')}}
    out.write_text(json.dumps(record))
    out.parent.joinpath('target-pid').write_text(str(os.getpid()))
"""
    (tool / "ra_capacity_server.py").write_text(body + stub + effect + "\n")
    j["tools"] = {n: sha(tool / n) for n in worker.TOOLS}
    return m, j, py, tool


def run(tmp_path, m, j, py, tool, sock):
    mp, wp = tmp_path / "handoff-spec.json", tmp_path / "worker-spec.json"
    mp.write_text(json.dumps(m))
    wp.write_text(json.dumps(j))
    argv = [
        str(py),
        "-I",
        "-S",
        "-B",
        str(tool / "ra_handoff.py"),
        "--manifest",
        str(mp),
        "--out",
        str(tmp_path / "handoff.json"),
        "--worker-spec",
        str(wp),
        "--worker-sha256",
        sha(wp),
    ]
    return subprocess.run(
        argv,
        pass_fds=(sock.fileno(),),
        capture_output=True,
        text=True,
        timeout=45,
        env=dict(os.environ, E4B_PAGED_MAX_TOKENS_PER_SEQ="2048", E4B_PAGED_CHUNK_TOKENS="512"),
    )


@pytest.mark.parametrize(
    "mutation", [None, "parent", "inode", "device", "address", "fd", "output", "schema", "extra", "manifest"]
)
def test_fixed_server_route_passes_or_refuses_before_marker(tmp_path, mutation):
    with listening() as sock:
        m, j, py, tool = setup(tmp_path, sock)
        if mutation == "parent":
            j["server"]["parent_pid"] += 100000
        elif mutation == "inode":
            j["server"]["listener"]["inode"] += 1
        elif mutation == "device":
            j["server"]["listener"]["device"] += 1
        elif mutation == "address":
            j["server"]["listener"]["address"][1] += 1
        elif mutation == "fd":
            j["server"]["listener"]["fd"] = 0
        elif mutation == "output":
            j["native_out"] = str(tmp_path / "foreign")
        elif mutation == "schema":
            j["schema"] = 3
        elif mutation == "extra":
            j["server"]["command"] = "arbitrary"
        elif mutation == "manifest":
            j["handoff_manifest"]["sha256"] = "0" * 64
        result = run(tmp_path, m, j, py, tool, sock)
        marker = Path(j["native_out"]) / "target-pid"
        if mutation:
            assert result.returncode != 0 and not marker.exists(), result.stderr
        else:
            assert result.returncode == 0, result.stderr
            record = json.loads((tmp_path / "worker.json").read_text())
            assert int(marker.read_text()) == record["pid"]
            assert record["status"] == "WORKER_RETURNED_PENDING_GATES"
            assert record["nested_workers_verified"] is record["proves_gpu_engagement"] is False


@pytest.mark.parametrize(
    "effect",
    [
        "    sys.path.append('/foreign')",
        "    os.environ['E4B_FUSED_RMSNORM']='1'",
        "    record['listener']['inode'] += 1;out.write_text(json.dumps(record))",
        "    raise SystemExit(5)",
    ],
)
def test_server_post_return_mutants_retain_failed_worker(tmp_path, effect):
    with listening() as sock:
        m, j, py, tool = setup(tmp_path, sock, effect)
        result = run(tmp_path, m, j, py, tool, sock)
        assert result.returncode != 0
        assert json.loads((tmp_path / "handoff.json").read_text())["status"] == "PASSED"
        assert json.loads((tmp_path / "worker.json").read_text())["status"] == "FAILED"
        assert (Path(j["native_out"]) / "target-pid").exists()


@pytest.mark.parametrize("mutation", [None, "started", "exit", "force", "closed", "thread", "error", "queue"])
def test_lifecycle_requires_normal_drained_trace_return(mutation):
    server = types.SimpleNamespace(started=True, should_exit=True, force_exit=False)
    engine = types.SimpleNamespace(_thread=types.SimpleNamespace(is_alive=lambda: False), error=None, queue_depth=0)
    app = types.SimpleNamespace(
        state=types.SimpleNamespace(engine=engine, ra_trace_closed={"closed": True, "thread_alive": False})
    )
    if mutation == "started":
        server.started = False
    elif mutation == "exit":
        server.should_exit = False
    elif mutation == "force":
        server.force_exit = True
    elif mutation == "closed":
        app.state.ra_trace_closed = None
    elif mutation == "thread":
        engine._thread.is_alive = lambda: True
    elif mutation == "error":
        engine.error = "native failure"
    elif mutation == "queue":
        engine.queue_depth = 1
    if mutation:
        with pytest.raises(ValueError, match="return normally"):
            c.server_wrapper.lifecycle(server, app, {})
    else:
        result = c.server_wrapper.lifecycle(server, app, {})
        assert result["pid"] == os.getpid() and result["proves_native_context_absence"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "pid",
        "guard",
        "cleanup",
        "returncode",
        "status",
        "worker_phase",
        "handoff",
        "native",
        "trace",
        "listener",
        "parent",
        "closed",
        "engine",
        "spec",
        "missing",
    ],
)
def test_server_receipt_join_mutants(tmp_path, mutation):
    mp, wp, hp, rp, np = [tmp_path / n for n in ("manifest", "spec", "handoff", "worker", "lifecycle")]
    mp.write_text("{}")
    wp.write_text("{}")
    h = {
        "status": "PASSED",
        "phase": "COMPLETE",
        "manifest_sha256": sha(mp),
        "payload": {"proves_installed_payload": True},
        "proves_requested_release_imports": True,
    }
    hp.write_text(json.dumps(h))
    pin = {"parent_pid": 100, "listener": {"fd": 3, "device": 1, "inode": 2, "address": ["127.0.0.1", 12345]}}
    closed = {"closed": True, "thread_alive": False}
    traces = {"request-trace.jsonl": "a", "step-trace.jsonl": "b"}
    n = {
        "status": "SERVER_RETURNED_PENDING_GATES",
        "pid": 123,
        "parent_pid": 100,
        "listener": pin["listener"],
        "closed": closed,
        "engine_thread_alive": False,
        "trace_sha256": traces,
    }
    np.write_text(json.dumps(n))
    r = {
        "status": "WORKER_RETURNED_PENDING_GATES",
        "phase": "COMPLETE",
        "worker_phase": "capacity_server",
        "pid": 123,
        "spec_sha256": sha(wp),
        "handoff_sha256": sha(hp),
        "handoff_evidence": h,
        "native_receipt_sha256": sha(np),
    }
    process = {
        "status": "OK",
        "pid": 123,
        "returncode": 0,
        "cleanup_complete": True,
        "parent_death_guard": {"verified": True},
    }
    binding = {
        "manifest": {"path": str(mp), "sha256": sha(mp)},
        "worker_spec": {"path": str(wp), "sha256": sha(wp)},
        "handoff_receipt": str(hp),
        "worker_receipt": str(rp),
        "lifecycle": str(np),
        "server": pin,
    }
    if mutation == "pid":
        r["pid"] += 1
    elif mutation == "guard":
        process["parent_death_guard"]["verified"] = False
    elif mutation == "cleanup":
        process["cleanup_complete"] = False
    elif mutation == "returncode":
        process["returncode"] = -9
    elif mutation == "status":
        process["status"] = "TIMEOUT"
    elif mutation == "worker_phase":
        r["worker_phase"] = "capacity"
    elif mutation == "handoff":
        h["status"] = "FAILED"
        hp.write_text(json.dumps(h))
        r["handoff_sha256"] = sha(hp)
    elif mutation == "native":
        r["native_receipt_sha256"] = "0" * 64
    elif mutation == "trace":
        n["trace_sha256"] = {}
    elif mutation == "listener":
        n["listener"] = {}
    elif mutation == "parent":
        n["parent_pid"] = 101
    elif mutation == "closed":
        n["closed"] = {}
    elif mutation == "engine":
        n["engine_thread_alive"] = True
    elif mutation == "spec":
        wp.write_text("mutated")
    np.write_text(json.dumps(n))
    if mutation not in ("native",):
        r["native_receipt_sha256"] = sha(np)
    rp.write_text(json.dumps(r))
    if mutation == "missing":
        rp.unlink()
    if mutation:
        with pytest.raises((ValueError, OSError, KeyError)):
            worker.check_server_child(binding, process, closed, traces)
    else:
        result = worker.check_server_child(binding, process, closed, traces)
        assert result["proves_gpu_engagement"] is result["proves_native_context_absence"] is False


@pytest.mark.parametrize("mutation", [None, "foreign_pid", "returncode", "group", "timeout"])
def test_normal_shutdown_budget_and_owned_absence(monkeypatch, mutation):
    process = types.SimpleNamespace(pid=123, poll=lambda: None, wait=lambda timeout: 0)
    ack = {"accepted": True, "pid": 123, "parent_pid": os.getpid()}
    if mutation == "foreign_pid":
        ack["pid"] = 124
    elif mutation == "returncode":
        process.wait = lambda timeout: 1
    elif mutation == "timeout":
        process.wait = lambda timeout: (_ for _ in ()).throw(subprocess.TimeoutExpired("own", timeout))

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return json.dumps(ack).encode()

    monkeypatch.setattr(
        c.capacity.urllib.request, "build_opener", lambda *_: types.SimpleNamespace(open=lambda *_, **kw: Response())
    )

    def absence(pid, sig):
        assert pid == 123 and sig == 0
        if mutation != "group":
            raise ProcessLookupError

    monkeypatch.setattr(c.capacity.os, "killpg", absence)
    if mutation:
        with pytest.raises((ValueError, RuntimeError, subprocess.TimeoutExpired)):
            c.capacity.normal_shutdown("http://127.0.0.1:12345", process, 1)
    else:
        assert c.capacity.normal_shutdown("http://127.0.0.1:12345", process, 1) == ack


@pytest.mark.parametrize("mutant", [None, "shutdown", "join", "cleanup"])
def test_capacity_composition_joins_server_after_drivers(tmp_path, monkeypatch, mutant):
    spec, stage, _ = c.prepared(tmp_path)
    model, pf, _, _, driver = c.capacity.prepared(spec, stage)
    out = tmp_path / "out"
    calls = []
    guard_calls = []
    server_argv = ["/synthetic/python", "-I", "-S", "-B", "/synthetic/handoff"]
    monkeypatch.setattr(c.capacity.ra_stage, "verify", lambda _: {})

    class Process:
        pid = 12345
        returncode = None

        def __init__(self, argv, **kw):
            assert argv == server_argv and kw["start_new_session"] and len(kw["pass_fds"]) == 1
            for key in ("E4B_PAGED_TRACE", "E4B_PAGED_STEP_TRACE"):
                Path(kw["env"][key]).write_text("synthetic trace")

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0
            return 0

    monkeypatch.setattr(c.capacity.subprocess, "Popen", Process)

    def signal(pid, sig):
        assert pid == 12345
        if sig == 0:
            if mutant == "cleanup":
                raise PermissionError("synthetic inaccessible owned group")
            raise ProcessLookupError

    monkeypatch.setattr(c.capacity.os, "killpg", signal)

    def guarded(argv, receipt, record):
        record["parent_death_guard"] = {"required": True, "verified": False}
        return argv, None, None

    monkeypatch.setattr(c.capacity, "guarded_server", guarded)

    def guarded_evidence(record, guard, evidence, argv, pid):
        assert argv == server_argv and pid == 12345
        record["parent_death_guard"]["verified"] = True
        guard_calls.append(list(argv))

    monkeypatch.setattr(c.capacity.ra_process, "guard_evidence", guarded_evidence)
    admitted = [0]

    def http(base, endpoint, process, timeout):
        return {"ready": True, "pid": 12345} if endpoint == "/_ra/evidence" else c.health(admitted[0])

    monkeypatch.setattr(c.capacity, "get_json", http)
    closed = {"closed": True, "thread_alive": False}
    monkeypatch.setattr(c.capacity, "close_trace", lambda *_: closed)
    monkeypatch.setattr(c.capacity.ra_trace, "bind", lambda *_: {})

    def child(spec, stage, out, *, base, point):
        path = out / (point[0] + "-child.json")
        return (
            c.capacity.driver_command(
                Path(spec["venv"]) / "bin/python", stage, base, model, spec["prompts"]["path"], point, path
            ),
            path,
            {"label": point[0]},
        )

    def client(argv, **kw):
        options = dict(zip(argv[4::2], argv[5::2]))
        point = c.capacity.POINTS[len(calls)]
        Path(options["--out"]).write_text(json.dumps(c.native(driver, point, options["--base"], model, pf)))
        calls.append(point[0])
        admitted[0] += point[3]
        return {"status": "OK"}

    monkeypatch.setattr(c.capacity.ra_process, "run", client)

    def shutdown(base, process, timeout):
        assert calls == ["warm", "burst", "end"] and (out / "trace-binding.json").exists() is False
        if mutant == "shutdown":
            raise TimeoutError("synthetic owned shutdown timeout")
        process.returncode = 0

    monkeypatch.setattr(c.capacity, "normal_shutdown", shutdown)

    def joined(binding, process, closure, traces):
        assert process["status"] == "OK" and process["cleanup_complete"] and closure == closed
        assert calls == ["warm", "burst", "end"] and len(guard_calls) == 2
        assert set(traces) == {"request-trace.jsonl", "step-trace.jsonl"}
        if mutant == "join":
            raise ValueError("synthetic missing server receipt")
        return {"status": "synthetic joined server", "proves_gpu_engagement": False}

    monkeypatch.setitem(
        c.capacity.sys.modules,
        "ra_verified_worker",
        types.SimpleNamespace(
            CURRENT={},
            server_child=lambda *_: (server_argv, {"synthetic": True}),
            check_server_ready=lambda *_: "synthetic-hash",
            check_server_child=joined,
            sc2_child=child,
            check_sc2_child=lambda binding, process, point, native: {"native_receipt_sha256": sha(native)},
        ),
    )
    if mutant:
        with pytest.raises((TimeoutError, ValueError, RuntimeError)):
            c.capacity.execute(spec, stage, out)
    else:
        result = c.capacity.execute(spec, stage, out)
        assert result["server_startup_verified"] is True
        assert result["nested_workers_verified"] is result["proves_gpu_engagement"] is False
    record = json.loads((out / "server.json").read_text())
    assert record["status"] == ("FAILED" if mutant else "NATIVE_RECORDED_PENDING_ENGAGEMENT")
    assert len(guard_calls) >= 2 and (out / "server-startup-evidence.json").exists()
