"""Lane RD1 (bench/moegen/rd1/RD1-PREREG.md): the controller can carry the lane, and the bar the reducer applies is the one
registered.

* Every staged piece exists, the runner parses, and the rehearsal knob never reaches a rented box.
* The probe's family shapes are the registration's table, and the reducer's bar (0.85, 3 of the 7 many-group families,
  both seqs, the skewed draw) is the registration's.
* The committed A2000 receipt is correctness only: no timing field, read by the reducer's --gate-only mode.
* The correctness gate keeps a fast, wrong arm out of the bar, and missing error fields count for nothing.
* The committed reading (rd1-rp-5090-2) re-derives from its receipt: the box's own table, the bar, the decision and every
  registered prediction, each scored on its own text. An anchor that crashes is a harness error, never a refusal.
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
    assert "${RD1_REHEARSAL" in RUN.read_text() and "${RD1_PROVE" in RUN.read_text()
    rd1 = {v for v in _forwarded() if v.startswith("RD1_")}
    assert rd1 == {"RD1_PROVE"}, f"tc1_drive.sh forwards exactly RD1_PROVE of RD1's knobs, never RD1_REHEARSAL: {rd1}"


def test_the_runner_installs_rsync_before_anything_is_fetched():
    """Amendment 2: tc1_drive fetches with rsync; on RunPod the image has none (tc1c-h100-19 fetched zero files)."""
    run = RUN.read_text()
    assert run.index("apt-get install -y -qq rsync") < run.index("train anchor (attempt")
    assert "RSYNC INSTALL FAIL" in run and "finish 9" in run


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


def _synthetic(tmp_path, decoded_err_x, with_err=True, gpu="NVIDIA GeForce RTX 5090", fast=("qwen3", "qwen36", "graniteh")):
    """A registered-shape receipt where decoded_cap is 0.5x every other arm on the ``fast`` families (both seqs, skewed)
    and level elsewhere; qwen3's 512 gate_up forward has decoded_cap's fp32 error at ``decoded_err_x`` times dense's."""
    fast_fams = fast
    import json
    many = _assign(LANE / "rd_table.py", "MANY")
    cells = []
    for fam in many:
        for seq in (512, 2048):
            proj = {}
            for call in ("gate_up N=1 K=1", "down N=1 K=1"):
                res = {}
                for mode in ("fwd", "dgrad"):
                    fast = fam in fast_fams
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
                                "host_load1_probe": {"median": 2.0, "max": 3.0, "samples": 40},
                                "anchor_post": {"rc": 0, "class": "pcie-full/launch-fast"}}))
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


def test_amendment_3_constants_are_the_registered_ones():
    run, prereg = RUN.read_text(), PREREG.read_text()
    assert "ANCHOR_TRIES=3; POST_ANCHOR=1" in run and "wait_load" not in run and "LOAD_MAX" not in run
    assert run.index("python rd_probe.py") < run.index('ANCHOR_OUT=$W/anchor_post.json') < run.index("python rd_table.py")
    assert "## Amendment 3 (2026-10-05" in prereg and "post-probe anchor" in prereg


def test_only_a_passing_post_probe_anchor_decides(tmp_path):
    import json
    _synthetic(tmp_path, 1.0)
    path = tmp_path / "rd.json"
    rec = json.loads(path.read_text())
    # host load is informational: a passing post-probe anchor decides even at a median load1 of 30
    for post, load, expect in (({"rc": 0, "class": "c"}, 30.0, "DECISION:"), ({"rc": 3, "class": "c"}, 2.0, "NOT A DECISION"),
                               (None, 2.0, "NOT A DECISION")):
        rec["host_load1_probe"] = {"median": load, "max": load, "samples": 40}
        rec.pop("anchor_post", None)
        if post:
            rec["anchor_post"] = post
        path.write_text(json.dumps(rec))
        out = subprocess.run([sys.executable, str(LANE / "rd_table.py"), str(path)], capture_output=True, text=True).stdout
        assert expect in out, (post, load, out[-400:])
        if expect == "NOT A DECISION":
            assert "\nDECISION:" not in out


def test_missing_error_fields_count_for_nothing(tmp_path):
    out = _synthetic(tmp_path, 1.0, with_err=False)
    assert "BAR DECODED (gated): 0/7" in out and "unread" in out


def test_p1_is_scored_on_the_families_it_names(tmp_path):
    """P1 names graniteh, qwen3 and qwen36: the bar holding on other families does not make it hold."""
    named = _synthetic(tmp_path, 1.0)
    assert "BAR DECODED (gated): 3/7" in named and "P1 DECODED holds, at least on graniteh, qwen3, qwen36 -> HELD" in named
    other = _synthetic(tmp_path, 1.0, fast=("olmoe", "lfm2", "ernie", "graniteh"))
    assert "BAR DECODED (gated): 4/7" in other and "-> HELD" in other
    assert "P1 DECODED holds, at least on graniteh, qwen3, qwen36 -> REFUTED: DECODED held (4/7)" in other
    assert "qwen3 fails" in other and "qwen36 fails" in other and "graniteh passes" in other
    # P5 names mixtral, which this receipt lacks: unread, never held
    assert "P5 on mixtral, dense <= decoded_cap on event time at both seqs -> UNREAD" in other


READ = LANE / "rd1-rp-5090-2"


def test_the_committed_reading_rederives_from_its_receipt():
    """rd1-rp-5090-2 (RunPod Secure RTX 5090): the reducer on the committed receipt reproduces RESULTS-rd1.txt exactly, and
    every line the box's own reducer wrote is in it, in order -- the predictions and the per-length line only add lines."""
    p = subprocess.run([sys.executable, str(LANE / "rd_table.py"), str(READ / "receipts" / "rd1.json")],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert p.stdout == (LANE / "RESULTS-rd1.txt").read_text()
    lines = iter(p.stdout.splitlines())
    for box_line in (READ / "RESULTS-rd1.txt").read_text().splitlines():
        assert any(line == box_line for line in lines), f"the box's line is not re-derived: {box_line[:100]}"
    for want in ("post-probe anchor: {'rc': 0, 'class': 'pcie-full/launch-fast'}",
                 "BAR DECODED (gated): 4/7", "seq 512 4/7", "seq 2048 7/7", "BAR V3 (gated): 0/7",
                 "DECISION: an opt-in decoded route in grouped-nf4-gemm, then a TC1 full-step A/B",
                 "P0 every arm passes the correctness gate on every call -> HELD (160 arm-cells: 0 failed, 0 unread)",
                 "graniteh, qwen3, qwen36 -> REFUTED", "V3 no prediction was registered",
                 "P3 decoded_cap event / device <= 1.5 on every many-group skewed cell -> HELD",
                 "seq 512, skewed, on graniteh, qwen3, qwen36 -> HELD",
                 "P5 on mixtral, dense <= decoded_cap on event time at both seqs -> REFUTED (skew holds"):
        assert want in p.stdout, want


def test_an_anchor_crash_is_a_harness_error_not_a_refusal():
    """rd1-rp-5090-1: train_anchor.py crashed (CUDA "invalid argument" on a pinned allocation) and the lane exited 12, the
    strict-anchor refusal. The gate exits 0 or 3; any other rc now finishes 9, before the refusal branch can see it."""
    run = RUN.read_text()
    crash = run.index('if [ "$arc" -ne 0 ] && [ "$arc" -ne 3 ]; then')
    assert crash < run.index("BOX REFUSED by train anchor") and run.index("finish 9", crash) < run.index("finish 12", crash)
    assert "ANCHOR HARNESS ERROR" in run[crash:run.index("BOX REFUSED by train anchor")]
    assert "rc=1 class=" in (LANE / "attempts" / "rd1-rp-5090-1" / "summary.txt").read_text()
    assert "pin_memory=True" in (LANE / "attempts" / "rd1-rp-5090-1" / "anchor.log").read_text()


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