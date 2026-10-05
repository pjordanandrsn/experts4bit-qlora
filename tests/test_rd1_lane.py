"""Lane RD1 (bench/moegen/rd1/RD1-PREREG.md): the controller can carry the lane, and the bar the reducer applies is the one
registered.

* Every staged piece exists, the runner parses, and the rehearsal knob never reaches a rented box.
* The probe's family shapes are the registration's table, and the reducer's bar (0.85, 3 of the 7 many-group families,
  both seqs, the skewed draw) is the registration's.
* The reducer runs on the committed A2000 filter receipt and names the registered grid as complete.
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


def test_the_reducer_reads_the_a2000_filter_receipt():
    p = subprocess.run([sys.executable, str(LANE / "rd_table.py"), str(LANE / "a2000" / "rd_a2000.json")],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "BAR DECODED:" in p.stdout and "BAR V3:" in p.stdout
    assert "NOT THE REGISTERED CARD (NVIDIA RTX A2000" in p.stdout and "DECISION:" not in p.stdout, "an A2000 filter decides nothing"
