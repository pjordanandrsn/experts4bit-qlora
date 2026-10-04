"""Lane MG1 (bench/moegen/MG1-PREREG.md): the controller can carry the lane, and the reading is tp1's instrument.

* Every ``MG1_*`` knob the box runner reads reaches the box through ``tc1_drive.sh``'s forwarded list, except
  ``MG1_REHEARSAL``: the A2000 rehearsal's anchor bypass, which a rented box must never be able to set.
* Every piece the preregistration stages exists, and the runner is the staged one.
* ``mg1_reduce.py``'s verdict is tp1_reduce v2.1's, as the preregistration says it is copied: the same AST, not a lookalike.
"""
import ast
import json
import os
import re
import subprocess
import sys
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


def test_amendment_3_runs_tp1s_driver_file_under_the_hook():
    """Amendment 3 reads P2 on tp1's arm driver itself: the runner starts the staged driver file through p2_hook.py (never a
    copy or an edit of it), the knob reaches the box, and the hook is a staged piece."""
    run, prereg = RUN.read_text(), PREREG.read_text()
    assert 'py="python p2_hook.py"' in run and "$py tp1_train_smoke.py" in run
    assert "MG1_P2_ARM" in _forwarded()
    assert "MG1_P2_ARM=1" in prereg and "bench/moegen/p2_hook.py" in prereg
    assert "AMENDMENT 3 SHAPE (registered)" in run


def _hook(tmp_path, body, code):
    """Run p2_hook.py on a stand-in driver that loads a stand-in nf4_qlora, bumps its counters and exits with ``code``."""
    (tmp_path / "nf4_qlora.py").write_text(
        'DGRAD_STATS = {"kernel": 0, "grouped_mm": 0, "dense": 0, "loop": 0, "loop_reasons": {}}\n')
    drv = tmp_path / "driver.py"
    drv.write_text("import sys\nimport nf4_qlora\n" + body +
                   f"\nif __name__ == '__main__':\n    assert sys.argv[1:] == ['--arm', 'fused'], sys.argv\n    sys.exit({code})\n")
    out = tmp_path / "census.json"
    env = {**os.environ, "MG1_P2_OUT": str(out), "PYTHONPATH": str(tmp_path)}
    p = subprocess.run([sys.executable, str(REPO / "bench" / "moegen" / "p2_hook.py"), str(drv), "--arm", "fused"],
                       env=env, cwd=tmp_path, capture_output=True, text=True)
    return p, out, drv


def test_the_hook_records_the_counters_and_keeps_the_drivers_exit_code(tmp_path):
    import hashlib
    p, out, drv = _hook(tmp_path, 'nf4_qlora.DGRAD_STATS["kernel"] += 160', 0)
    assert p.returncode == 0, p.stderr
    c = json.loads(out.read_text())
    assert c["dgrad"]["kernel"] == 160 and c["dgrad"]["loop"] == 0 and c["reason"] is None
    assert c["driver_sha256"] == hashlib.sha256(drv.read_bytes()).hexdigest()
    p, out, _ = _hook(tmp_path, 'nf4_qlora.DGRAD_STATS["loop"] += 2', 5)       # a driver stub's exit (OOM = 5) still records
    assert p.returncode == 5 and json.loads(out.read_text())["dgrad"]["loop"] == 2
