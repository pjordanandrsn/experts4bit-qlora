# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235 Amendment 16 (bench/q235/PREREG-q235.md): the box's cap sized on the box, the load's per-layer record, the VOID_BY_HOST
carve-out and the host CPU record. No network and no GPU: the runner's functions run from its own text, as Amendment 15's tests do."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "bench" / "q235"
RUN = (Q / "q235_run.sh").read_text()


def _fn(start, end):
    return RUN[RUN.index(start):RUN.index(end)]


def _workspace(tmp_path):
    w = tmp_path / "w"
    (w / "receipts" / "box").mkdir(parents=True)
    (w / "logs").mkdir()
    pybin = tmp_path / "pybin"
    pybin.mkdir()
    (pybin / "python").symlink_to(sys.executable)
    return w, {**os.environ, "PATH": f"{pybin}:{os.environ['PATH']}"}


def _box_cap(tmp_path, left_s, pinned=""):
    """The runner's own box_cap against a deadline ``left_s`` seconds away."""
    w, env = _workspace(tmp_path)
    share = re.search(r"^BOX_SHARE=(\d+) CAP_BOX=\"\"$", RUN, re.M).group(1)
    fn = _fn("# --- box_cap (Amendment 16) ---", "# --- end box_cap ---")
    deadline = int(time.time()) + left_s
    r = subprocess.run(["bash", "-c", f'{fn}\nTC1_DEADLINE_EPOCH={deadline}; BOX_SHARE={share}; CAP_BOX="{pinned}"; box_cap'], cwd=w,
                       capture_output=True, text=True, env=env, timeout=60)
    rec = json.loads((w / "receipts" / "box-cap.json").read_text()) if r.returncode == 0 else None
    return r, rec


@pytest.mark.parametrize("run,left,cap,used", [("q235-pro6000-7", 17678, 8689, 3031), ("q235-pro6000-6", 17829, 8764, 2425),
                                               ("q235-pro6000-3", 20462, 10081, 2349), ("q235-pro6000-2", 20710, 10205, 1185)])
def test_the_box_cap_is_half_the_time_left_less_the_reserve(tmp_path, run, left, cap, used):
    """The worked examples, from each science box's time left after its tokens phase (receipts/phase-timings.jsonl). -7 was VOIDed by
    the sealed 2,997 s with 4.9 h left; its half-cap holds every box on record. One second may pass between the deadline's arithmetic
    here and the runner's own clock read."""
    r, rec = _box_cap(tmp_path, left)
    assert r.returncode == 0, r.stderr
    assert int(r.stdout) in (cap, cap - 1) and rec["derived_cap_s"] == int(r.stdout) and rec["share"] == 2 and rec["reserve_s"] == 300
    assert rec["amendment"] == 16 and "rehearsal_cap_s" not in rec and int(r.stdout) > used


def test_the_box_cap_refuses_when_nothing_is_left_and_records_a_rehearsal_pin(tmp_path):
    r, rec = _box_cap(tmp_path / "a", 250)
    assert r.returncode != 0 and r.stdout.strip() == "" and rec is None
    r, rec = _box_cap(tmp_path / "b", 4000, pinned="900")
    assert r.stdout.strip() == "900" and rec["rehearsal_cap_s"] == 900 and rec["derived_cap_s"] in (1850, 1849)


@pytest.mark.parametrize("lines,unfinished", [
    (None, False),                                                                 # no record: the box died before its load began
    ([{"schema": "q235-load-layers/1"}], True),                                     # the load began, no layer finished
    ([{"schema": "q235-load-layers/1"}, {"layer": 1, "t_s": 22.1}, {"layer": 2, "t_s": 44.0}], True),
    ([{"schema": "q235-load-layers/1"}, {"layer": 1, "t_s": 3.8}, {"done": True, "layers": 1, "load_s": 4.0}], False)])
def test_load_unfinished_reads_the_per_layer_record(tmp_path, lines, unfinished):
    w, env = _workspace(tmp_path)
    if lines is not None:
        (w / "receipts" / "box" / "load_layers.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    fn = _fn("load_unfinished(){", "# --- end box_cap ---")
    r = subprocess.run(["bash", "-c", f"{fn}\nload_unfinished"], cwd=w, capture_output=True, text=True, env=env, timeout=60)
    assert (r.returncode == 0) is unfinished


def test_the_runner_sizes_the_box_on_the_box_and_only_a_load_past_its_cap_is_void_by_host():
    """Amendment 16's wiring: no sealed box cap; the box's cap is box_cap's, taken right before the box phase; a box record that cannot
    be sized is a VOID (11); a box killed at its cap (rc 142) with its load unfinished exits 18 (a host floor, which names the machine)
    as VOID_BY_HOST; every other box failure, and every repeat failure, stays a VOID (11). Only a rehearsal pins the box cap."""
    assert not re.search(r"\bCAP_BOX=\d", RUN) and "2997" not in RUN.split("# --- box_cap")[1]
    loop = _fn("for run in box repeat; do", "if [ \"$MODE\" = pro6000 ]; then   # Amendment 11: the second")
    assert 'cap=$(box_cap) && [ -n "$cap" ] || {' in loop and "Amendment 16)'; finish 11; }" in loop
    assert loop.index("cap=$(box_cap)") < loop.index("cap=$(repeat_cap)") < loop.index('phase "$run" "$cap" python q235_box.py')
    carve = loop[loop.index('if [ "$run" = box ] && [ "$PHASE_RC" = 142 ] && load_unfinished; then'):]
    assert carve.index("finish 18") < carve.index('say "VOID: $run (rc $PHASE_RC)"; finish 11')
    assert "VOID_BY_HOST" in carve[:carve.index("finish 18")]
    rehearsal = _fn('if [ "$MODE" != science ]; then', "must require-git")
    assert "CAP_BOX=${Q235_CAP_BOX:-}" in rehearsal and RUN.count("Q235_CAP_BOX") == 1


def test_the_cpu_record_is_written_after_the_install_and_never_gates():
    block = _fn("# Amendment 16: what the host-side load runs on", "DIRECT=()")
    assert RUN.index('must install "$CAP_INSTALL"') < RUN.index("# Amendment 16: what the host-side load runs on")
    assert "receipts/cpu.json" in block and "finish" not in block and "recorded only, never gated" in block
    for key in ("cpu_model", "logical_cpus", "physical_cores", "threads_per_core", "affinity_cpus", "cpu_quota", "torch_default_pool"):
        assert f'"{key}"' in block, key


def _box_module():
    spec = importlib.util.spec_from_file_location("q235_box_a16", Q / "instrument" / "q235_box.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_load_profile_writes_one_flushed_line_per_layer_and_an_end_line_only_on_return(tmp_path):
    """On the CPU rehearsal (no offload) a layer is finished at its quantize; the end line is the load returning, so a box killed (or
    raising) mid-load leaves a record with no end line: what load_unfinished reads."""
    box = _box_module()
    path = tmp_path / "box" / "load_layers.jsonl"
    prof = box.LoadProfile(False, layers_path=str(path))
    with prof:
        quant = prof._timed("quantize", lambda: None)
        for _ in range(3):
            quant()
        mid = [json.loads(x) for x in path.read_text().splitlines()]      # flushed as written, before the load returns
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert mid[0]["schema"] == "q235-load-layers/1" and mid[0]["event"] == "quantize" and [x["layer"] for x in mid[1:]] == [1, 2, 3]
    assert lines[-1]["done"] is True and lines[-1]["layers"] == 3 and len(lines) == 5
    path2 = tmp_path / "repeat" / "load_layers.jsonl"
    prof = box.LoadProfile(False, layers_path=str(path2))
    with pytest.raises(RuntimeError):
        with prof:
            prof._timed("quantize", lambda: None)()
            raise RuntimeError("killed mid-load")
    assert not any('"done"' in x for x in path2.read_text().splitlines())


def test_the_load_profile_without_a_path_writes_nothing(tmp_path):
    box = _box_module()
    prof = box.LoadProfile(False)
    with prof:
        prof._timed("quantize", lambda: None)()
    assert prof._layers is None and not list(tmp_path.iterdir())


def test_the_fit_checks_the_box_budget_against_its_half_cap_on_the_expected_timeline():
    spec = importlib.util.spec_from_file_location("q235_fit_a16", Q / "q235_fit.py")
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    caps = f.sealed_caps(Q / "q235_run.sh")
    assert "box" not in caps and "repeat" not in caps and f.ON_BOX_CAPS == ("box", "repeat") and f.BOX_SHARE == 2
    timings = {"require-git": 0, "create-venv": 3, "tokens": 6, "hostrate": 22, "box": 85, "repeat": 20}
    fetch = {"bytes": 4e9, "hash_seconds": 4.0}
    rep = f.fit(timings, 155, fetch, 3e9, 474.252e9, 806, 1.85, caps)
    box = rep["phases"]["box"]
    expected = {k: v for k, v in timings.items() if k not in ("box", "repeat")}
    bw = 806e6 / 8
    left = 6 * 3600 - 1800 - (sum(expected.values()) + 3e9 / bw + 155 + 474.252e9 / bw + 474.252e9 / 1e9)
    assert box["half_cap_s"] == int((left - 300) // 2) and box["budget_s"] == 2997 and box["fits"] and box["sealed_cap_s"] is None
    slow = f.fit({**timings, "box": 400}, 155, fetch, 3e9, 474.252e9, 806, 1.85, caps)
    assert not slow["phases"]["box"]["fits"] and not slow["fits"]          # 400 x 94/4 x 1.5 = 14,100 s > the half-cap
