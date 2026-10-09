"""CPU composition and ownership mutations; no serving capacity is measured."""
import importlib.util
import json
import signal
import socket
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench/ra" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for name in ("ra_env", "ra_stage", "ra_reduce", "ra_normalize", "ra_process", "ra_training", "ra_quality",
             "ra_fallback", "ra_routes", "ra_serving", "ra_trace"):
    load(name)
capacity = load("ra_capacity")
server_wrapper = load("ra_capacity_server")


def prepared(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    for name, source in (("source-pins.json", "bench/ra/source-pins.json"), ("sc2_driver.py", "bench/sc2/sc2_driver.py")):
        (stage / name).write_bytes((ROOT / source).read_bytes())
    pins = json.loads((stage / "source-pins.json").read_bytes())
    model = capacity.ra_training.MODEL["proof"][1]
    rows = [[i] + [0] * 511 for i in range(16)]
    pf = {"rows": rows, "prompts_sha256": capacity.ra_normalize.reducer.digest(rows)}
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(pf))
    arena, calib = tmp_path / "arena.bin", tmp_path / "calib.json"
    arena.write_bytes(b"synthetic arena")
    calib.write_text('{"synthetic_calibration":true}\n')
    spec = {"battery": "proof", "venv": str(tmp_path / "venv"), "cache": str(tmp_path / "cache"),
            "threads": 8, "allocator": "expandable_segments:True", "prompts": {"path": str(prompts),
            "sha256": capacity.ra_process.file_digest(prompts)}, "deadline_epoch_s": time.time() + 2000,
            "startup_s": 20, "driver_s": 30,
            "fixture": {"E4B_PAGED_MODEL": model, "E4B_PAGED_REVISION": pins["models"][model],
                        "E4B_PAGED_ARENA": str(arena), "E4B_PAGED_CALIB": str(calib),
                        "E4B_PAGED_MAX_TOKENS_PER_SEQ": "2048", "E4B_PAGED_CHUNK_TOKENS": "512"}}
    return spec, stage, pf


def native(driver, point, base, model, pf):
    _, mode, rate, n, seed = point
    plan = [list(p) for p in driver.plan(mode, rate, n, seed, 16, 64, 256)]
    return {"base": base, "model": model, "profile": "e4b", "mode": mode, "rate": rate, "n": n,
            "seed": seed, "max_tokens_range": [64, 256], "prompts_sha256": pf["prompts_sha256"], "plan": plan,
            "summary": {"valid": n, "invalid": 0},
            "requests": [{"i": i, "prompt_index": p, "planned_s": t, "max_tokens": m, "status": 200,
                          "valid": True, "prompt_tokens": 512, "prompt_len": 512, "completion_tokens": m,
                          "finish_reason": "length", "ttft_s": .1, "tpot_s": .01}
                         for i, (t, p, m) in enumerate(plan)]}


def health(admitted):
    return {"status": "ready", "error": None, "queue_depth": 0,
            "engine": {"buckets": [1, 2, 4, 8, 16], "graphs": True, "chunk_tokens": 512,
                       "max_tokens_per_seq": 2048, "max_tokens_limit": 2047,
                       "graph_status": {str(b): "graph" for b in (1, 2, 4, 8, 16)},
                       "graph_stats": {"16": {"replays": admitted, "eager_steps": 0}}},
            "prefill_graph": {"status": "on", "T": 512, "replays": admitted, "eager_chunks": 0},
            "kv_bookkeeping": {"requested": True, "bulk": True, "flush_layers": 0, "ready_layers": 0,
                               "ready_bulk": 0, "flush_bulk": admitted, "ready_at_flush": admitted}}


@pytest.mark.parametrize("key,value", [("seed", 0), ("profile", "vllm"), ("model", "wrong"),
                                      ("prompts_sha256", "changed"), ("rate", 11), ("n", 119)])
def test_native_plan_and_command_mutations(tmp_path, key, value):
    s, stage, _ = prepared(tmp_path)
    model, pf, _, _, driver = capacity.prepared(s, stage)
    row = native(driver, capacity.POINTS[-1], "http://127.0.0.1:1", model, pf)
    row[key] = value
    with pytest.raises(ValueError, match="mismatch"):
        capacity.check_native(row, capacity.POINTS[-1], driver, row["base"], model, pf)


def test_preflight_rejects_duplicate_prompts_and_unregistered_speed_env(tmp_path):
    s, stage, pf = prepared(tmp_path)
    s["fixture"]["E4B_PAGED_MAX_SEQS"] = "16"
    with pytest.raises(ValueError, match="unregistered"):
        capacity.prepared(s, stage)
    del s["fixture"]["E4B_PAGED_MAX_SEQS"]
    pf["rows"][1] = pf["rows"][0]
    p = Path(s["prompts"]["path"])
    p.write_text(json.dumps(pf))
    s["prompts"]["sha256"] = capacity.ra_process.file_digest(p)
    with pytest.raises(ValueError, match="distinct"):
        capacity.prepared(s, stage)


def test_trace_paths_cannot_target_foreign_resources(tmp_path):
    s, _, _ = prepared(tmp_path)
    out = tmp_path / "out"
    own = capacity.owned_traces(s, out)
    assert own["fixture"]["E4B_PAGED_TRACE"] == str(out / "request-trace.jsonl")
    assert "E4B_PAGED_TRACE" not in s["fixture"]
    s["fixture"]["E4B_PAGED_STEP_TRACE"] = str(tmp_path / "foreign.jsonl")
    with pytest.raises(ValueError, match="fresh component"):
        capacity.owned_traces(s, out)


def test_common_threads_and_prepared_arena_are_required(tmp_path):
    s, stage, _ = prepared(tmp_path)
    s["fixture"]["E4B_PAGED_TORCH_THREADS"] = "1"
    with pytest.raises(ValueError, match="common identity"):
        capacity.owned_traces(s, tmp_path / "out")
    del s["fixture"]["E4B_PAGED_TORCH_THREADS"]
    Path(s["fixture"]["E4B_PAGED_ARENA"]).unlink()
    with pytest.raises(ValueError, match="prepared arena"):
        capacity.prepared(s, stage)


def test_drained_rejects_dead_owned_process_before_http(monkeypatch):
    monkeypatch.setattr(capacity.urllib.request, "build_opener", lambda *_: pytest.fail("foreign HTTP must not run"))
    with pytest.raises(RuntimeError, match="exited"):
        capacity.drained("http://127.0.0.1:1", types.SimpleNamespace(poll=lambda: 1), .1)


def test_drained_distinguishes_loading_from_permanent_failure(monkeypatch):
    replies = iter([{"status": "loading"}, health(0)])
    monkeypatch.setattr(capacity, "get_json", lambda *_: next(replies))
    assert capacity.drained("http://127.0.0.1:1", None, 1)["status"] == "ready"
    monkeypatch.setattr(capacity, "get_json", lambda *_: {"status": "error", "error": "synthetic failure"})
    with pytest.raises(RuntimeError, match="failure"):
        capacity.drained("http://127.0.0.1:1", None, 1)


@pytest.mark.parametrize("mutant", [None, "request", "health", "inputs", "driver", "trace", "close", "binding", "wait", "driver_wait"])
def test_owned_socket_sequence_retention_and_failure_cleanup(tmp_path, monkeypatch, mutant):
    s, stage, _ = prepared(tmp_path)
    model, pf, _, _, driver = capacity.prepared(s, stage)
    observed, killed, calls = {}, [], []
    admitted = [0]
    monkeypatch.setattr(capacity.ra_stage, "verify", lambda _: {})
    monkeypatch.setenv("GNF4_GEMV_BW", "0")
    foreign = socket.socket()
    foreign.bind(("127.0.0.1", 0))
    foreign.listen(1)
    foreign_address = foreign.getsockname()

    class Process:
        pid = 12345  # Explicit synthetic ownership token; no OS signal is sent.
        returncode = None

        def __init__(self, argv, **kwargs):
            assert "GNF4_GEMV_BW" not in kwargs["env"]
            assert kwargs["start_new_session"] is True
            for key in ("E4B_PAGED_TRACE", "E4B_PAGED_STEP_TRACE"):
                p = Path(kwargs["env"][key])
                assert p.parent == kwargs["cwd"]
                if mutant != "trace":
                    p.write_text('{"synthetic_trace":true}\n')
            fd, = kwargs["pass_fds"]
            with socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) as duplicate:
                observed["address"] = duplicate.getsockname()
                assert observed["address"] != foreign_address
                assert duplicate.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_STREAM
                with socket.socket() as probe:
                    probe.settimeout(1)
                    assert probe.connect_ex(observed["address"]) == 0

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if mutant in ("wait", "driver_wait"):
                raise capacity.subprocess.TimeoutExpired("synthetic owned server", timeout)
            self.returncode = -signal.SIGKILL

    monkeypatch.setattr(capacity.subprocess, "Popen", Process)
    def own_signal(pid, sig):
        if sig == 0:
            raise ProcessLookupError
        killed.append((pid, sig))
    monkeypatch.setattr(capacity.os, "killpg", own_signal)

    def client(argv, **kwargs):
        options = dict(zip(argv[4::2], argv[5::2]))
        point = capacity.POINTS[len(calls)]
        assert options["--mode"] == point[1] and options["--seed"] == str(point[4])
        assert options["--base"] == "http://127.0.0.1:" + str(observed["address"][1])
        row = native(driver, point, options["--base"], model, pf)
        if mutant == "request":
            row["requests"][0]["status"] = 503
        Path(options["--out"]).write_text(json.dumps(row))
        calls.append(point[0])
        admitted[0] += point[3]
        if mutant in ("driver", "driver_wait"):
            raise RuntimeError("synthetic driver failure")
        if mutant == "inputs":
            Path(s["prompts"]["path"]).write_text("mutated input")
        return {"status": "OK"}

    def get(base, endpoint, process, timeout):
        assert base.endswith(":" + str(observed["address"][1]))
        if endpoint == "/_ra/evidence":
            return {"ready": True, "synthetic": True}
        h = health(admitted[0])
        if mutant == "health" and admitted[0] == 188:
            h["prefill_graph"]["eager_chunks"] = 1
        return h

    monkeypatch.setattr(capacity.ra_process, "run", client)
    monkeypatch.setattr(capacity, "get_json", get)
    def close(*_):
        assert calls == ["warm", "burst", "end"] and admitted[0] == 188
        if mutant == "close":
            raise TimeoutError("synthetic trace close timeout")
        return {"closed": True, "synthetic": True}

    monkeypatch.setattr(capacity, "close_trace", close)
    # This suite tests process composition; native joins have their own mutation
    # and production-engine tests in test_ra_trace.py.
    def bind(*_):
        if mutant == "binding":
            raise ValueError("synthetic trace join failure")
        return {"synthetic": True}

    monkeypatch.setattr(capacity.ra_trace, "bind", bind)
    out = tmp_path / "out"
    try:
        if mutant and mutant != "wait":
            with pytest.raises((ValueError, RuntimeError, TimeoutError, capacity.ra_normalize.reducer.Invalid)) as error:
                capacity.execute(s, stage, out)
            if mutant == "driver_wait":
                assert type(error.value) is RuntimeError
                assert str(error.value) == "synthetic driver failure"
        else:
            result = capacity.execute(s, stage, out)
            assert result["status"] == "NATIVE_RECORDED_PENDING_ENGAGEMENT"
            assert result["proves_gpu_engagement"] is False
            assert calls == ["warm", "burst", "end"]
        rec = json.loads((out / "server.json").read_bytes())
        assert rec["status"] == ("FAILED" if mutant and mutant != "wait" else "NATIVE_RECORDED_PENDING_ENGAGEMENT")
        assert rec["kill_wait_timeout"] is (mutant in ("wait", "driver_wait"))
        assert rec["returncode"] == (None if mutant in ("wait", "driver_wait") else -signal.SIGKILL)
        assert rec["attempts"] == 1 and killed == [(Process.pid, signal.SIGKILL)]
        assert (out / "capacity_warm.json").is_file() and (out / "server.log").is_file()
        assert foreign.getsockname() == foreign_address
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(foreign_address) == 0
    finally:
        foreign.close()


def test_server_hooks_before_capture_builds_once_and_redacts_token(monkeypatch):
    cfg = types.SimpleNamespace(token="synthetic-secret", graphs=True)
    app = types.SimpleNamespace(routes={})
    app.get = lambda path: lambda fn: app.routes.setdefault(path, fn)
    app.post = app.get
    parts = types.SimpleNamespace(info={"fusion_modes": {"graph": "auto"}, "census": 1})
    def original(cfg):
        return parts
    server = types.SimpleNamespace(build_engine=original)
    server.create_app = lambda cfg: app
    seen = []

    def instrument(server, instrument, cfg, *, builder, listener=None):
        assert builder is original
        seen.append("before-capture")
        return builder(cfg), types.SimpleNamespace(snapshot=lambda: {"calls": 1}), types.SimpleNamespace(
            snapshot=lambda: {"qkv_calls": 1}, ra_defaults={"synthetic": True})

    monkeypatch.setattr(server_wrapper.ra_serving, "build_instrumented", instrument)
    instrumented = server_wrapper.instrumented_app(server, types.SimpleNamespace(CENSUS_KEYS=("census",)), cfg)
    assert instrumented.routes["/_ra/evidence"]() == {"ready": False}
    assert server.build_engine(cfg) is parts
    evidence = instrumented.routes["/_ra/evidence"]()
    assert "token" not in evidence["config"] and evidence["kernels"] == {"calls": 1}
    assert evidence["resolved_defaults"] == {"synthetic": True}
    assert seen == ["before-capture"]
    with pytest.raises(RuntimeError, match="only once"):
        server.build_engine(cfg)


def test_parent_death_guard_refuses_missing_guard_or_changed_parent(monkeypatch):
    seen = []
    monkeypatch.setattr(server_wrapper.sys, "platform", "linux")
    monkeypatch.setattr(server_wrapper.ctypes, "CDLL", lambda *a, **k: types.SimpleNamespace(
        prctl=lambda *args: seen.append(args) or 0))
    monkeypatch.setattr(server_wrapper.os, "getppid", lambda: 123)
    server_wrapper.parent_death_guard(123)
    assert seen == [(1, signal.SIGKILL, 0, 0, 0)]
    with pytest.raises(RuntimeError, match="already exited"):
        server_wrapper.parent_death_guard(124)
    monkeypatch.setattr(server_wrapper.ctypes, "CDLL", lambda *a, **k: types.SimpleNamespace(prctl=lambda *args: -1))
    with pytest.raises(RuntimeError, match="unavailable"):
        server_wrapper.parent_death_guard(123)
