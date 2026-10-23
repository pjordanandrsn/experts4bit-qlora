# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P228, K1's selection and microbench (bench/p228/PREREG-p228.md). The staged pins; the registered constants as the
PREREG states them; the runner's order, premise and guard fit; the drive's refusals; the runner's READ line on a mixed
record; the self-tests and the reducer mutants they must kill."""
import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p228"
SOURCES = {"p228_bench.py": LANE, "p228_reduce.py": LANE, "p228_run.sh": LANE,
           "test_dense_lt_pin_gpu.py": REPO / "tests", "test_dense_lt_pin.py": REPO / "tests"}
RUN = (LANE / "p228_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p228_drive.sh").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p228.md").read_text(encoding="utf-8")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


R = _load("p228_reduce_under_test", LANE / "p228_reduce.py")


def test_every_staged_file_is_pinned_and_matches():
    pins = dict(reversed(line.split()) for line in (LANE / "staged-p228.sha256").read_text(encoding="ascii").splitlines())
    assert set(pins) == set(SOURCES)
    for name, where in SOURCES.items():
        assert hashlib.sha256((where / name).read_bytes()).hexdigest() == pins[name], name


def test_the_self_tests_pass():
    R.self_test()
    r = subprocess.run([sys.executable, str(LANE / "p228_bench.py"), "--self-test"], capture_output=True, text=True)
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


def test_the_registered_constants_and_the_prereg_agree():
    assert (R.BAR, R.TOL, R.TIE, R.RULED_MS, R.MS, R.SELF_BAND, R.MIN_POWER_W) == \
        (0.85, 1.5e-2, 1.02, (2, 4), (2, 4, 8), (0.99, 1.01), 575)
    assert R.SITES == {"o_proj": {"N": 2048, "K": 4096, "copies": 48}, "router": {"N": 128, "K": 2048, "copies": 384}}
    assert R.ARMS == ["D", "D2", "P", "T1"] and R.ITERS == 10 and R.READ_STEPS == tuple(range(60))
    assert (R.SELECT_STEPS, R.SELECT_ITERS) == (8, 5) and R.REDUCTION_OK == (0, 2, 4)
    for s in ("**0.85**", "**within 1.5e-2**", "**1.02x**", "| `o_proj` | (2048, 4096) | 48 |", "| router gate | (128, 2048) | 384 |",
              "exactly **37 passed**", "under 575 W", "seed 2280 + M", "seed 22800 + M", "**$1.45**"):
        assert s in PREREG, s


def test_the_runner_installs_e4b_alone_checks_k1_and_runs_the_premise_on_the_card():
    assert "grouped-nf4-gemm.git" not in RUN and "GNF4_SHA" not in RUN
    assert 'assert dl.LT_PINS == {}, "the commit ships no pins: this lane selects them"' in RUN
    assert "assert re_.router_linear is dl.router_linear" in RUN
    assert "test_dense_lt_pin_gpu.py test_dense_lt_pin.py -q -rs" in RUN and "grep -qE '^37 passed" in RUN
    order = ["sha256sum -c staged-p228.sha256", "PL=$(nvidia-smi --query-gpu=power.limit", '[ "$LEFT" -ge "$FIT_NEED" ]',
             "pipx logs/pip_e4b.log", "TRIPWIRE FAIL", "grep -qE '^37 passed", "p228_bench.py --out", "p228_reduce.py --bench"]
    at = [RUN.index(x) for x in order]
    assert at == sorted(at), dict(zip(order, at))
    assert "MIN_POWER_W=575 " in RUN and "finish 17; }" in RUN
    assert "p228-5090-1|p228-5090-2|p228-5090-3)" in RUN and "p228-shake-1|p228-shake-2|p228-shake-3) SHAKE=1;;" in RUN
    assert "--shakedown --reps 4 --copies-o 8 --copies-r 32" in RUN


def test_the_guard_fit():
    get = lambda v: int(re.search(rf"^{v}=(\d+)", RUN, re.M).group(1))  # noqa: E731
    install, need, reserve = get("INSTALL_TO_GATE_S"), get("NEED_BENCH"), get("RESERVE")
    assert (install, need, reserve) == (600, 900, 600)
    fit = install + need + reserve
    assert fit == 2100 and int(fit * 1.25) == 2625 and (300 + fit) * 1.25 == 3000 <= 3600
    for s in ("`NEED_BENCH` = **900 s**", "`FIT` = 600 + 900 + 600 = 2100 s", "2100 x 1.25 = 2625 s",
              "(boot 300 s + 2100 s) x 1.25 = 3000 s <= 3600 s (1.0 h)"):
        assert s in PREREG, s
    assert abs(1.0 * 1.05 + 10 * 0.04 - 1.45) < 1e-9


def test_the_drive_never_creates_compute_and_refuses_what_is_not_registered():
    assert "p228-5090-[1-3]|p228-shake-[1-3]) :;;" in DRIVE and "the checkout is dirty" in DRIVE
    assert "tests/test_dense_lt_pin_gpu.py tests/test_dense_lt_pin.py; do" in DRIVE
    for word in ("vast.py", "destroy", "/api/", "rent-watch"):
        assert word not in DRIVE.split("set -uo pipefail", 1)[1], word


def test_a_record_with_no_site_read_is_not_a_reading(tmp_path):
    code = re.search(r'^READ=\$\(python -c "(.*)"\)$', RUN, re.M).group(1)

    def read_count(v):
        p = tmp_path / "verdict_p228.json"
        p.write_text(json.dumps(v), encoding="utf-8")
        r = subprocess.run([sys.executable, "-c", code.replace("$W/verdict_p228.json", p.as_posix())],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return int(r.stdout.strip())
    q = R._fake()
    q["sites"]["o_proj"]["cells"]["2"]["reps"][4]["p_rel"] = 0.05                    # o_proj: TOLERANCE_FAIL (read)
    q["sites"]["router"]["cells"]["8"]["reps"][1]["d2_identical"] = False             # router: VOID
    v = R.verdict(q)
    assert v["verdict"] == "o_proj=TOLERANCE_FAIL router=VOID" and read_count(v) == 1
    q["sites"]["o_proj"]["cells"]["2"]["reps"][4]["d2_identical"] = False
    v = R.verdict(q)
    assert v["verdict"] == "o_proj=VOID router=VOID" and read_count(v) == 0
    assert '"1/o_proj=SHAKEDOWN router=SHAKEDOWN") finish 0;;' in RUN


REDUCER_MUTANTS = [
    ("BAR = 0.85 ", "BAR = 0.95 "),
    ("RULED_MS = (2, 4) ", "RULED_MS = (2,) "),
    ('ok = all(cells[M]["p_over_d"]["ub"] <= BAR and cells[M]["p_over_d"]["ub_clock_adj"] <= BAR for M in RULED_MS)',
     'ok = all(cells[M]["p_over_d"]["median"] <= BAR for M in RULED_MS)'),
    ('p["ub_clock_adj"] = p["ub"] * max(1.0, ratio)', 'p["ub_clock_adj"] = p["ub"] * min(1.0, ratio)'),
    ('two = [c for c in ok if c["config"]["reduction_scheme"] == 2 and select_ms(c["step_ms"]) <= TIE * best]', 'two = []'),
    ('return (c.get("check_ok") is True and c.get("bitwise") is True', 'return (c.get("check_ok") is True'),
    ('if rep.get("d2_identical") is not True:', 'if False:'),
    ('bad = [M for M, c in cells.items() if c["p_rel_max"] > TOL]', 'bad = []'),
    ('if sel.get("chosen") != cfg:', 'if False:'),
]


@pytest.mark.parametrize("old, new", REDUCER_MUTANTS, ids=[m[0][:28] for m in REDUCER_MUTANTS])
def test_the_self_test_kills_each_reducer_mutant(tmp_path, old, new):
    src = (LANE / "p228_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p228_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    r = subprocess.run([sys.executable, str(tmp_path / "p228_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert r.returncode != 0 and "FAILED" in r.stdout, (old, r.stdout)
