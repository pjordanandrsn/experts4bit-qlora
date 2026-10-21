"""bench/common/phases.sh and bench/common/phases.py (bench/common/README.md, "Phase stamps"): the real P149 record first,
then the writer's refusals and the reader's faults."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "bench" / "common" / "phases.sh"
REAL = ROOT / "bench" / "p149" / "receipts" / "p149-5090-2" / "phases.jsonl"
REAL_SHA256 = "5b281475af7730eadf328dd2ad3371925492ee7508268b529d5633dfde9ffa7d"


def _mod():
    spec = importlib.util.spec_from_file_location("phases_common", ROOT / "bench" / "common" / "phases.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sh(tmp_path, script):
    return subprocess.run(["bash", "-c", f'set -u; . "{HELPER}"; {script}'], cwd=tmp_path, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "PHASES_FILE": str(tmp_path / "phases.jsonl")})


def test_the_real_p149_record_reads_clean_and_its_kinds_are_unknown_not_guessed():
    assert hashlib.sha256(REAL.read_bytes()).hexdigest() == REAL_SHA256
    m = _mod()
    phases, faults = m.read(REAL.read_text().splitlines())
    s = m.summary(phases)
    assert faults == [] and s["open"] == [] and s["phases"] == 18
    assert [p["name"] for p in phases][:3] == ["install", "fetch", "bake"] and phases[-1]["name"] == "reduce"
    assert s["seconds_by_kind"] == {"setup": 0, "measure": 0, "reduce": 0, "other": 0, "unknown": 5422}
    assert s["first_measure_start"] is None              # no kind recorded: nothing is read as measurement
    assert (s["first_stamp"], s["last_stamp"]) == (1792523069, 1792528649)


def test_the_helper_writes_lines_the_reader_classifies(tmp_path):
    r = _sh(tmp_path, "phase_start install setup; phase_end install; phase_start armA measure; phase_end armA; "
                      "phase_start reduce reduce; phase_end reduce")
    assert r.returncode == 0, r.stderr
    lines = (tmp_path / "phases.jsonl").read_text().splitlines()
    assert [json.loads(x)["event"] for x in lines] == ["start", "end"] * 3
    m = _mod()
    phases, faults = m.read(lines)
    s = m.summary(phases)
    assert faults == [] and [p["kind"] for p in phases] == ["setup", "measure", "reduce"]
    assert s["first_measure_start"] == phases[1]["start"]


@pytest.mark.parametrize("script", ['phase_start "bad name" setup', 'phase_start ok badkind', 'phase_end "x\\"y"',
                                    'phase_start "" setup', 'phase_start onlyname', 'phase_start', 'phase_end'])
def test_the_helper_refuses_a_bad_name_or_kind_and_writes_nothing(tmp_path, script):
    r = _sh(tmp_path, script)
    assert r.returncode == 2 and not (tmp_path / "phases.jsonl").exists(), (r.returncode, r.stderr)


def _lines(*recs):
    return [json.dumps(r) for r in recs]


@pytest.mark.parametrize("lines,fault", [
    (_lines({"phase": "a", "event": "start", "t": 1}, {"phase": "b", "event": "start", "t": 2}), "starts while a is open"),
    (_lines({"phase": "a", "event": "start", "t": 5}, {"phase": "a", "event": "end", "t": 4}), "goes backwards"),
    (_lines({"phase": "a", "event": "start", "kind": "warmup", "t": 1}), "bad kind"),
    (_lines({"phase": "a", "event": "end", "t": 1}), "ends but nothing is open"),
    (_lines({"phase": "a b", "event": "start", "t": 1}), "bad phase name"),
    (_lines({"phase": "a", "event": "start", "t": "1"}), "not an integer"),
    (["{not json"], "not JSON"),
])
def test_the_reader_names_each_fault(lines, fault):
    _, faults = _mod().read(lines)
    assert any(fault in f for f in faults), faults


def test_a_backwards_end_credits_no_seconds():
    m = _mod()
    phases, faults = m.read(_lines({"phase": "a", "event": "start", "kind": "setup", "t": 9},
                                   {"phase": "a", "event": "end", "t": 4}))
    assert any("goes backwards" in f for f in faults)
    assert phases[0]["seconds"] is None and m.summary(phases)["seconds_by_kind"]["setup"] == 0


def test_an_open_phase_is_reported_not_dropped():
    m = _mod()
    phases, faults = m.read(_lines({"phase": "fetch", "event": "start", "kind": "setup", "t": 10}))
    assert faults == [] and m.summary(phases)["open"] == ["fetch"]
    assert m.summary(phases)["seconds_by_kind"]["setup"] == 0     # an open phase adds no seconds


def test_check_exits_nonzero_on_a_fault(tmp_path):
    bad = tmp_path / "p.jsonl"
    bad.write_text("\n".join(_lines({"phase": "a", "event": "end", "t": 1})) + "\n")
    r = subprocess.run([sys.executable, str(ROOT / "bench" / "common" / "phases.py"), "--check", str(bad)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "ends but nothing is open" in r.stdout
    ok = subprocess.run([sys.executable, str(ROOT / "bench" / "common" / "phases.py"), "--check", str(REAL)],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout == ""
