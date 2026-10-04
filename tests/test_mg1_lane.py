"""Lane MG1 (bench/moegen/MG1-PREREG.md): the controller can carry the lane, and the reading is tp1's instrument.

* Every ``MG1_*`` knob the box runner reads reaches the box through ``tc1_drive.sh``'s forwarded list, except
  ``MG1_REHEARSAL``: the A2000 rehearsal's anchor bypass, which a rented box must never be able to set.
* Every piece the preregistration stages exists, and the runner is the staged one.
* ``mg1_reduce.py``'s verdict is tp1_reduce v2.1's, as the preregistration says it is copied: the same AST, not a lookalike.
"""
import ast
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "bench" / "moegen" / "mg1_run.sh"
DRIVE = REPO / "bench" / "tc1" / "tc1_drive.sh"
PREREG = REPO / "bench" / "moegen" / "MG1-PREREG.md"


def _forwarded():
    drive = DRIVE.read_text()
    a = drive.index("for v in TC1_FAMILIES")
    return set(drive[a:drive.index("; do", a)].split())


def test_every_mg1_knob_the_runner_reads_is_forwarded_and_the_rehearsal_bypass_is_not():
    read = set(re.findall(r"\$\{(MG1_[A-Z0-9_]+)", RUN.read_text()))
    assert {"MG1_FAMILIES", "MG1_STEPS", "MG1_PROVE", "MG1_REHEARSAL"} <= read, read
    forwarded = _forwarded()
    missing = sorted(k for k in read - {"MG1_REHEARSAL"} if k not in forwarded)
    assert not missing, f"read by mg1_run.sh but never forwarded by tc1_drive.sh: {missing}"
    assert "MG1_REHEARSAL" not in forwarded, "the rehearsal's anchor bypass must never reach a rented box"


def test_the_staged_pieces_exist_and_the_runner_parses():
    text = PREREG.read_text()
    stage = re.search(r"`TC1_EXTRA_STAGE` = `([^`]+)`", text, re.S)
    assert stage, "the preregistration names the staged pieces"
    pieces = stage.group(1).split()
    assert "bench/moegen/mg1_run.sh" in pieces
    for p in pieces:
        assert (REPO / p).is_file() and (REPO / p).stat().st_size > 0, p
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0


def _fn(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.dump(node)
    raise AssertionError(f"{name} not in {path}")


def test_the_verdict_is_tp1s():
    tp1 = REPO / "bench" / "train-parity-20260905" / "tp1" / "tp1_reduce.py"
    mg1 = REPO / "bench" / "moegen" / "mg1_reduce.py"
    for name in ("verdict", "c1_ok"):
        assert _fn(mg1, name) == _fn(tp1, name), name
    band = [re.search(r"^BAND = (.+)$", p.read_text(), re.M).group(1) for p in (tp1, mg1)]
    assert band[0] == band[1], band


def test_amendment_2_shape_is_the_registered_one():
    """The runner's registered amendment-2 args are exactly the ones MG1-PREREG.md registers, and the knobs reach the box."""
    run, prereg = RUN.read_text(), PREREG.read_text()
    a2 = re.search(r'^A2_ARGS="([^"]+)"$', run, re.M).group(1)
    assert f"MG1_LADDER_ARGS='{a2}'" in prereg
    assert {"MG1_LADDER_ONLY", "MG1_LADDER_ARGS"} <= _forwarded()
