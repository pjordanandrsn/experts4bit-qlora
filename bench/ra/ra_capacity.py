#!/usr/bin/env python3
"""One owned capacity server and the frozen SC2 warm/burst/point processes.

Native records and counters are retained; independent provenance and complete
default/fallback coverage remain external gates. This is no rental controller.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import ra_env
import ra_normalize
import ra_process
import ra_stage
import ra_training
import ra_trace

POINTS = (("warm", "serial", 0, 4, 999), ("burst", "poisson", 1000, 64, 998),
          ("end", "poisson", 12, 120, 112))
SPEC = {"battery", "venv", "cache", "threads", "allocator", "fixture", "prompts",
        "deadline_epoch_s", "startup_s", "driver_s"}


def owned_traces(spec, out):
    fixture = dict(spec["fixture"])
    fixture.setdefault("E4B_PAGED_DEVICE", "cuda")
    fixture.setdefault("E4B_PAGED_PLACEMENT", "all-vram")
    threads = str(spec["threads"])
    if fixture.get("E4B_PAGED_TORCH_THREADS", threads) != threads:
        raise ValueError("capacity torch threads differ from common identity")
    fixture["E4B_PAGED_TORCH_THREADS"] = threads
    for key, filename in (("E4B_PAGED_TRACE", "request-trace.jsonl"),
                          ("E4B_PAGED_STEP_TRACE", "step-trace.jsonl")):
        path = str(out / filename)
        if key in fixture and fixture[key] != path:
            raise ValueError("trace destination must belong to this fresh component")
        fixture[key] = path
    return {**spec, "fixture": fixture}


def prepared(spec, stage):
    if set(spec) != SPEC or spec["battery"] not in ra_training.MODEL:
        raise ValueError("capacity spec fields/battery")
    ra_process.window(spec["deadline_epoch_s"], spec["startup_s"] + 3 * spec["driver_s"])
    for value in (spec["startup_s"], spec["driver_s"]):
        if type(value) not in (int, float) or value <= 0:
            raise ValueError("positive phase timeouts required")
    prompt_path = ra_process.check_input(spec["prompts"])
    pf = json.loads(prompt_path.read_bytes())
    rows = pf["rows"]
    if len(rows) != 16 or len({tuple(r) for r in rows}) != 16 or any(len(r) != 512 or
            any(type(t) is not int or t < 0 for t in r) for r in rows):
        raise ValueError("capacity requires sixteen distinct 512-token prompts")
    if ra_normalize.reducer.digest(rows) != pf["prompts_sha256"]:
        raise ValueError("capacity prompt payload digest")
    _, model = ra_training.MODEL[spec["battery"]]
    pins = json.loads((stage / "source-pins.json").read_bytes())
    if spec["fixture"].get("E4B_PAGED_MODEL") != model or \
            spec["fixture"].get("E4B_PAGED_REVISION") != pins["models"][model]:
        raise ValueError("capacity model/revision fixture")
    for key in ("E4B_PAGED_ARENA", "E4B_PAGED_CALIB"):
        path = Path(spec["fixture"].get(key, ""))
        if not path.is_absolute() or not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("capacity requires prepared arena/calibration files")
    env, removed = ra_env.clean(os.environ, component="capacity", fixture=spec["fixture"],
                               venv=Path(spec["venv"]), cache=Path(spec["cache"]),
                               threads=spec["threads"], allocator=spec["allocator"])
    driver = types.ModuleType("ra_capacity_frozen_driver")
    path = stage / "sc2_driver.py"
    exec(compile(path.read_bytes(), str(path), "exec"), driver.__dict__)
    return model, pf, env, removed, driver


def get_json(base, endpoint, process, timeout):
    if process.poll() is not None:
        raise RuntimeError("owned capacity server exited")
    # A fixed numeric loopback URL and a held listener exclude proxy/foreign services.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + endpoint, timeout=timeout) as response:
        return json.load(response)


def drained(base, process, timeout):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        try:
            health = get_json(base, "/health", process, min(2, until - time.monotonic()))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(min(.05, max(0, until - time.monotonic())))
            continue
        if health.get("status") == "error" or health.get("error"):
            raise RuntimeError("capacity server health reports failure")
        if health.get("status") == "ready" and health.get("queue_depth") == 0:
            return health
        time.sleep(min(.05, max(0, until - time.monotonic())))
    raise TimeoutError("owned capacity server did not become drained/ready")


def close_trace(base, process, timeout):
    if process.poll() is not None:
        raise RuntimeError("owned capacity server exited")
    if not ra_trace.number(timeout) or timeout <= 0:
        raise TimeoutError("no phase budget for native trace closure")
    request = urllib.request.Request(base + "/_ra/close-trace?timeout=" + str(min(10, timeout / 2)), method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def driver_command(python, stage, base, model, prompt_path, point, output):
    _, mode, rate, n, seed = point
    return [str(python), "-B", str(stage / "sc2_driver.py"), "run", "--base", base, "--model", model,
            "--prompts", str(prompt_path), "--mode", mode, "--rate", str(rate), "--n", str(n),
            "--seed", str(seed), "--max-tokens-lo", "64", "--max-tokens-hi", "256", "--profile", "e4b",
            "--out", str(output)]


def check_native(native, point, driver, base, model, pf):
    _, mode, rate, n, seed = point
    expected = (base, model, "e4b", mode, rate, n, seed, [64, 256], pf["prompts_sha256"])
    actual = tuple(native[k] for k in ("base", "model", "profile", "mode", "rate", "n", "seed",
                                      "max_tokens_range", "prompts_sha256"))
    if actual != expected or native["plan"] != [list(r) for r in driver.plan(mode, rate, n, seed, 16, 64, 256)]:
        raise ValueError("native SC2 command/plan mismatch")
    ra_normalize.requests(native)


def execute(spec, stage, out):
    if not stage.is_absolute() or not out.is_absolute() or any(
            p.is_symlink() for path in (stage, out) for p in (path, *path.parents)):
        raise ValueError("absolute non-symlink component paths required")
    ra_stage.verify(stage)
    spec = owned_traces(spec, out)
    model, pf, env, removed, driver = prepared(spec, stage)
    python = Path(spec["venv"]) / "bin/python"
    out.mkdir(parents=True, exist_ok=False)
    record = {"status": "STARTING", "started_at": ra_process.clock(), "attempts": 1,
              "removed_environment_keys": removed, "proves_gpu_engagement": False}
    (out / "server.json").write_text(json.dumps(record, indent=2) + "\n")
    process = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener, (out / "server.log").open("xb") as log:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            base = "http://127.0.0.1:" + str(listener.getsockname()[1])
            argv = [str(python), "-B", str(Path(__file__).with_name("ra_capacity_server.py")),
                    "--socket-fd", str(listener.fileno()), "--parent-pid", str(os.getpid()),
                    "--instruments", str(stage)]
            record.update(argv=argv, base=base)
            ra_process.window(spec["deadline_epoch_s"], spec["startup_s"] + 3 * spec["driver_s"])
            process = subprocess.Popen(argv, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=(listener.fileno(),))
            record["pid"] = process.pid
            (out / "health_start.json").write_text(json.dumps(drained(base, process, spec["startup_s"]), indent=2) + "\n")
            raw, workloads = {}, []
            for point in POINTS:
                label = point[0]
                name = "capacity" if label == "end" else "capacity_" + label
                # Driver completion and health retrieval together fit this phase timeout.
                started = time.monotonic()
                ra_process.window(spec["deadline_epoch_s"], spec["driver_s"])
                argv = driver_command(python, stage, base, model, spec["prompts"]["path"], point, out / f"{name}.json")
                ra_process.run(argv, env=env, cwd=out, log=out / f"{name}.log", receipt=out / f"{name}_process.json",
                               deadline=spec["deadline_epoch_s"], timeout=spec["driver_s"])
                native = json.loads((out / f"{name}.json").read_bytes())
                check_native(native, point, driver, base, model, pf)
                workloads.append(native)
                remaining = spec["driver_s"] - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("no phase budget for capacity health retrieval")
                health = drained(base, process, remaining)
                raw[f"capacity_health_{label}"] = health
                (out / f"capacity_health_{label}.json").write_text(json.dumps(health, indent=2) + "\n")
                remaining = spec["driver_s"] - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("no phase budget for capacity counter retrieval")
                evidence = get_json(base, "/_ra/evidence", process, min(2, remaining))
                if evidence.get("ready") is not True:
                    raise ValueError("capacity counter instrumentation not ready")
                (out / f"capacity_evidence_{label}.json").write_text(json.dumps(evidence, indent=2) + "\n")
            ra_normalize.capacity_health(raw)
            remaining = spec["driver_s"] - (time.monotonic() - started)
            closed = close_trace(base, process, remaining)
            (out / "trace-close.json").write_text(json.dumps(closed, indent=2, allow_nan=False) + "\n")
            ra_trace.bind(out, workloads, closed, raw["capacity_health_end"])
            ra_process.check_input(spec["prompts"])
            ra_stage.verify(stage)
            for filename in ("request-trace.jsonl", "step-trace.jsonl"):
                path = out / filename
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError("missing native capacity trace")
                record[filename + "_sha256"] = ra_process.file_digest(path)
            ra_process.window(spec["deadline_epoch_s"], .001)
            record["status"] = "NATIVE_RECORDED_PENDING_ENGAGEMENT"
    except Exception as exc:
        record.update(status="FAILED", error_type=type(exc).__name__)
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Retain the failure receipt and any original exception even if
                # the owned child is stuck during GPU-driver teardown.
                record["returncode"] = None
                record["kill_wait_timeout"] = True
            else:
                record["returncode"] = process.returncode
                record["kill_wait_timeout"] = False
        record["finished_at"] = ra_process.clock()
        if (out / "server.log").is_file():
            record["log_sha256"] = ra_process.file_digest(out / "server.log")
        (out / "server.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--instruments", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if not args.spec.is_absolute():
        ap.error("absolute spec required")
    execute(json.loads(args.spec.read_bytes()), args.instruments, args.out)


if __name__ == "__main__":
    main()
