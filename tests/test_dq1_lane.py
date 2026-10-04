"""Lane DQ1 (bench/dq1/DQ1-PREREG.md): the reducer's self-test passes, and the box, the census, the reducer and the
pre-registration agree on the pins, the arms and the subject. CPU-only: nothing here needs a GPU."""
import ast
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "dq1"
RUN = (LANE / "dq1_run.sh").read_text()
PREREG = (LANE / "DQ1-PREREG.md").read_text()
GNF4_SHA = "a5edec8789735bff1c0da4708ae5fc93260a1410"   # grouped-nf4-gemm v0.39.0


def _const(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path.name}")


def test_the_reducer_self_test_passes():
    out = subprocess.run([sys.executable, str(LANE / "dq1_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (30 cases)" in out.stdout, out.stdout + out.stderr


def test_census_and_reducer_agree_on_arms_and_shapes():
    assert _const(LANE / "dq1_census.py", "ARMS") == _const(LANE / "dq1_reduce.py", "ARMS")
    assert _const(LANE / "dq1_census.py", "SHAPES") == _const(LANE / "dq1_reduce.py", "SHAPES_T")


def test_the_registered_rule_constants():
    r = LANE / "dq1_reduce.py"
    assert _const(r, "SELF_PAIR_BAND") == (0.95, 1.05)
    assert _const(r, "NOISY_FRACTION") == 0.10
    assert _const(r, "PARITY_MAX") == 1e-2
    src = r.read_text()
    for clause in ("h2 >= 0.15", "h2 < 0.10 and h4 < 0.07", "g2 < 0.95", "g4 < 0.95", "g2 >= 1.05 and g4 >= 1.05",
                   "l2 >= 0.15", "r2 >= 1.5", '"gemm_slowdown") <= 1.05', '"copy_ratio") >= 0.8', "r4 < 1.0"):
        assert clause in src, clause


def test_the_box_pins_what_the_prereg_registers():
    assert f"DQ1_GNF4_SHA={GNF4_SHA}" in RUN and GNF4_SHA in PREREG
    assert "DQ1_BNB_VER=0.50.2" in RUN and "bitsandbytes 0.50.2" in PREREG
    assert 'train_gemm_route(torch.device("cuda", 0), 1) == "dense"' in RUN      # the premise, on the card
    assert "dq1_reduce.py --self-test" in RUN


def test_the_box_speaks_tc1_drives_contract():
    for marker in ('> TC1_RUN_NONCE', 'TC1_EXIT_CODE.$NONCE', 'TC1_SUCCESS.$NONCE', 'TP_DONE.$NONCE', "summary.txt"):
        assert marker in RUN, marker
    assert RUN.startswith("#!/bin/bash") and "W=/root/tc1" in RUN
    out = subprocess.run(["bash", "-n", str(LANE / "dq1_run.sh")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
