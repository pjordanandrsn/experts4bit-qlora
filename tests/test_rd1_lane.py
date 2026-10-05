"""Lane RD1 (bench/moegen/rd1/RD1-PREREG.md): the controller can carry the lane, and the bar the reducer applies is the one
registered.

* Every staged piece exists, the runner parses, and the rehearsal knob never reaches a rented box.
* The probe's family shapes are the registration's table, and the reducer's bar (0.85, 3 of the 7 many-group families,
  both seqs, the skewed draw) is the registration's.
* The committed A2000 receipt is correctness only: no timing field, read by the reducer's --gate-only mode.
* The correctness gate keeps a fast, wrong arm out of the bar, and missing error fields count for nothing.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "moegen" / "rd1"
RUN = LANE / "rd1_run.sh"
PREREG = LANE / "RD1-PREREG.md"
DRIVE = REPO / "bench" / "tc1" / "tc1_drive.sh"


def _forwarded():
    drive = DRIVE.read_text()
    a = drive.index("for v in TC1_FAMILIES")
    return set(drive[a:drive.index("; do", a)].split())


def test_the_staged_pieces_exist_and_the_runner_parses():
    stage = re.search(r"`TC1_EXTRA_STAGE` = `([^`]+)`", PREREG.read_text(), re.S)
    assert stage, "the registration names the staged pieces"
    pieces = stage.group(1).split()
    assert "bench/moegen/rd1/rd1_run.sh" in pieces
    for p in pieces:
        assert (REPO / p).is_file() and (REPO / p).stat().st_size > 0, p
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0


def test_the_rehearsal_knob_never_reaches_a_rented_box():
    assert "${RD1_REHEARSAL" in RUN.read_text()
    assert not any(v.startswith("RD1_") for v in _forwarded()), "tc1_drive.sh must not forward any RD1_ knob"


def _assign(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not in {path}")


def test_the_probe_shapes_are_the_registered_table():
    fams = _assign(LANE / "rd_probe.py", "FAMS")
    text = PREREG.read_text()
    for fam, (E, k, H, inter, gated) in fams.items():
        row = re.search(rf"^\| `{fam}` \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (gated|non-gated)", text, re.M)
        assert row, f"{fam} has no row in the registration's shape table"
        assert (int(row.group(1)), int(row.group(2)), int(row.group(3)), int(row.group(4))) == (E, k, H, inter), fam
        assert (row.group(5) == "gated") == gated, fam


def test_the_reducer_bar_is_the_registered_one():
    many = _assign(LANE / "rd_table.py", "MANY")
    text = PREREG.read_text()
    assert len(many) == 7 and "at least 3 of the 7" in text and "0.85" in text
    for fam in many:
        assert f"`{fam}`" in text, fam
    src = (LANE / "rd_table.py").read_text()
    assert src.count("<= 0.85") >= 2 and ">= 3" in src and '"skew"' in src
    assert _assign(LANE / "rd_table.py", "GATE_X") == 2.0 and "at most 2 times dense's" in text


def _synthetic(tmp_path, decoded_err_x, with_err=True, gpu="NVIDIA GeForce RTX 5090"):
    """A registered-shape receipt where decoded_cap is 0.5x every other arm on qwen3, qwen36 and graniteh (both seqs, skewed)
    and level elsewhere; qwen3's 512 gate_up forward has decoded_cap's fp32 error at ``decoded_err_x`` times dense's."""
    import json
    many = _assign(LANE / "rd_table.py", "MANY")
    cells = []
    for fam in many:
        for seq in (512, 2048):
            proj = {}
            for call in ("gate_up N=1 K=1", "down N=1 K=1"):
                res = {}
                for mode in ("fwd", "dgrad"):
                    fast = fam in ("qwen3", "qwen36", "graniteh")
                    arm = lambda ev, err=0.002: {"device_ms": ev, "event_ms": ev, "peak_mib": 1.0, "rel_err": 0.0,   # noqa: E731
                                                 **({"rel_err32": err} if with_err else {})}
                    bad = fam == "qwen3" and seq == 512 and call.startswith("gate_up") and mode == "fwd"
                    res[mode] = {"v1": arm(1.0), "v3": arm(1.1), "dense": arm(1.0), "decoded": arm(0.5),
                                 "decoded_cap": {**arm(0.5 if fast else 1.0, 0.002 * (decoded_err_x if bad else 1.0)),
                                                 "chunks": 1}}
                proj[call] = res
            cells.append({"fam": fam, "seq": seq, "routing": "skew", "groups": 64, "groups_under_16_rows": 0, "proj": proj})
    path = tmp_path / "rd.json"
    path.write_text(json.dumps({"gpu": gpu, "torch": "t", "triton": "t", "cap_mib": 256, "reps": 20, "cells": cells,
                                "host_load1_probe": {"median": 2.0, "max": 3.0, "samples": 40, "gate": 5.0}}))
    p = subprocess.run([sys.executable, str(LANE / "rd_table.py"), str(path)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout


def test_the_gate_keeps_a_fast_wrong_arm_out_of_the_bar(tmp_path):
    ok = _synthetic(tmp_path, 1.0)
    assert "BAR DECODED (gated): 3/7" in ok and "-> HELD" in ok and "every arm passed" in ok
    bad = _synthetic(tmp_path, 3.0)                       # 3x dense's error on ONE call: qwen3 no longer counts
    assert "BAR DECODED (gated): 2/7" in bad and "NOT HELD" in bad
    assert "FAIL decoded_cap skew/qwen3/512 gate_up N=1 K=1/fwd" in bad


def test_any_other_card_gets_the_gate_and_never_a_timing_bar(tmp_path):
    """A receipt from a card other than the RTX 5090 (the A2000 rehearsal) prints the correctness gate only: no timing
    column, no BAR line, no decision -- even though its cells carry event times."""
    out = _synthetic(tmp_path, 1.0, gpu="NVIDIA RTX A2000 12GB")
    assert "NOT THE REGISTERED CARD" in out and "CORRECTNESS ONLY" in out and "GATE:" in out
    assert "BAR" not in out and "DECISION:" not in out and "v1 ev ms" not in out


def test_amendment_1_constants_are_the_registered_ones():
    run, prereg = RUN.read_text(), PREREG.read_text()
    assert "LOAD_MAX=5.0; LOAD_WAIT_S=600; ANCHOR_TRIES=3" in run
    assert "## Amendment 1 (2026-10-05" in prereg and "load1 at or under 5.0" in prereg and "at most 3 anchor attempts" in prereg


def test_a_loaded_probe_decides_nothing(tmp_path):
    import json
    _synthetic(tmp_path, 1.0)
    path = tmp_path / "rd.json"
    rec = json.loads(path.read_text())
    for load, expect in ((3.2, "DECISION:"), (7.5, "NOT A DECISION"), (None, "NOT A DECISION")):
        rec["host_load1_probe"] = {"median": load, "max": load, "samples": 40, "gate": 5.0} if load else {"samples": 0, "gate": 5.0}
        path.write_text(json.dumps(rec))
        out = subprocess.run([sys.executable, str(LANE / "rd_table.py"), str(path)], capture_output=True, text=True).stdout
        assert expect in out, (load, out[-400:])
        if expect == "NOT A DECISION":
            assert "\nDECISION:" not in out


def test_missing_error_fields_count_for_nothing(tmp_path):
    out = _synthetic(tmp_path, 1.0, with_err=False)
    assert "BAR DECODED (gated): 0/7" in out and "unread" in out


def test_the_a2000_receipt_is_correctness_only():
    """The QNAP A2000 is a correctness testbed only (standing policy, 2026-07-27): its committed receipt carries no timing
    field, and the reducer reads it in --gate-only mode, which prints no bar and no decision."""
    import json
    path = LANE / "a2000" / "rd_a2000_correctness.json"
    timing = {"device_ms", "event_ms", "wall_s", "device_ms_vs_fused", "event_ms_vs_fused"}

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    assert not timing & set(keys(json.load(open(path)))), "a timing field in the A2000 receipt"
    assert not (LANE / "a2000" / "rd_a2000.json").exists() and not (LANE / "a2000" / "RESULTS-rd-a2000.txt").exists()
    p = subprocess.run([sys.executable, str(LANE / "rd_table.py"), "--gate-only", str(path)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "CORRECTNESS ONLY" in p.stdout and "GATE:" in p.stdout and "No timing is read" in p.stdout
    assert "BAR" not in p.stdout and "DECISION" not in p.stdout