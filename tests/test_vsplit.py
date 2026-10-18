# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P138, the verify split (#1313; bench/vsplit/PREREG-vsplit.md): the staged pins, the runner's shape, the
reducer's rule on real records first, the anti-circularity condition, the registered predictions, and the driver's dry
run."""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "vsplit"
SOURCES = {"vsplit_run.sh": LANE, "p132_box.py": B / "p132", "vsplit_reduce.py": LANE,
           "kernel_classes_nf4_v1_2.json": B / "p132", "sc1b_census.py": B / "sc1b", "sd2_box.py": B / "sd2",
           "sd2_reduce.py": B / "sd2", "sd2_audit.py": B / "sd2", "audit_read.tsv": B / "sd2", "sd1_box.py": B / "sd1",
           "sd1_eagle3.py": B / "sd1", "chat_prompts.json": B / "sd1", "expect_w1.json": B / "sd1",
           "p109_box.py": B / "p109", "p115_quality.py": B / "p115", "p110_box.py": B / "p110", "p108_box.py": B / "p108",
           "p97_box.py": B / "p97", "k8_bake.py": B / "p39", "calib.json": B / "p39"}
# sd2-5090-3's stage V1, committed with its receipt (bench/sd2/receipts/sd2-5090-3/SHA256SUMS lists v.json)
SD2_V = B / "sd2" / "receipts" / "sd2-5090-3" / "v.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _red():
    sys.path.insert(0, str(B / "sd2"))
    return _load("vsplit_reduce_t", LANE / "vsplit_reduce.py")


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-vsplit.sha256")
    assert set(pins) == set(SOURCES), sorted(set(pins) ^ set(SOURCES))
    for name, want in pins.items():
        assert hashlib.sha256((SOURCES[name] / name).read_bytes()).hexdigest() == want, name


def test_p132s_box_and_class_map_are_reused_byte_identical():
    mine, p132 = _pins(LANE / "staged-vsplit.sha256"), _pins(B / "p132" / "staged-p132.sha256")
    for name in ("p132_box.py", "kernel_classes_nf4_v1_2.json", "sc1b_census.py", "sd2_box.py", "sd2_reduce.py"):
        assert mine[name] == p132[name], name


def test_the_driver_and_the_runner_stage_exactly_the_pinned_pieces():
    drive = (LANE / "vsplit_drive.sh").read_text(encoding="utf-8")
    stage = re.search(r'^STAGE="([^"]+)"', drive, re.M).group(1).split()
    assert {pathlib.PurePosixPath(p).name for p in stage} - {"staged-vsplit.sha256"} == set(SOURCES)
    run = (LANE / "vsplit_run.sh").read_text(encoding="utf-8")
    listed = re.search(r"^for f in (.*?); do$", run, re.M | re.S).group(1).replace("\\\n", " ").split()
    assert set(listed) - {"staged-vsplit.sha256"} == set(SOURCES) - {"vsplit_run.sh"}


def test_the_runner_pins_the_subject_the_reducer_reads():
    run = (LANE / "vsplit_run.sh").read_text(encoding="utf-8")
    red = _red()
    assert re.search(rf"^E4B_T={red.E4B_T}\b", run, re.M) and re.search(rf"^GNF4_T={red.GNF4_T}\b", run, re.M)
    assert f"REV={red.REV}" in run
    assert re.search(rf"^SKIP=32; STEPS={red.STEPS}; CANARY_STEPS=12; MOE_LAYERS=48$", run, re.M)
    assert "--min-steps $((CANARY_STEPS - 4))" in run                 # P132 Amendment 1's canary floor, kept


def test_any_5090_runs_and_both_timing_passes_bracket_the_census():
    """No card-uuid refusal (every number is the card's own); pass A before the census, pass B (the self-pair) after."""
    run = (LANE / "vsplit_run.sh").read_text(encoding="utf-8")
    assert "REF_UUID" not in run and "finish 18" not in run and "finish 15" in run     # the class check stays
    i_canary = run.index("--mode canary")
    i_a = run.index("--out $W/timing.json")
    i_census = run.index("for n in 1 2 3 4; do")
    i_b = run.index("--out $W/timing_b.json")
    i_reduce = run.index("python $W/vsplit_reduce.py --dir $W")
    assert i_canary < i_a < i_census < i_b < i_reduce


@pytest.mark.parametrize("script", ["vsplit_reduce.py", "../p132/p132_box.py"])
def test_the_self_tests_pass(script):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(B / d) for d in ("sd2", "sd1", "p109", "p115", "p110", "p108",
                                                                           "p97")))
    out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True, timeout=300,
                         env=env)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_rho_is_the_registered_desk_ratio():
    desk = json.loads((B / "p132" / "desk_out.json").read_text())
    assert _red().RHO == {int(n): v for n, v in desk["rho"].items()}


def test_the_predictions_are_desk_v2_as_fractions():
    fr = _load("desk_fractions_t", LANE / "desk_fractions.py")
    desk = json.loads((B / "p132" / "desk_out.json").read_text())
    assert fr.fractions(desk) == json.loads((LANE / "desk_fractions.json").read_text())


def test_need_on_the_reference_record_is_p132s_registered_threshold():
    """Real record first: the in-box need(n), fed sd2-5090-3's own V1 terms (both passes), reproduces the threshold P132
    registered from those terms. The formula is unchanged; only where its inputs come from moves into the box."""
    v1 = json.loads(SD2_V.read_text())["v1"]
    sums = (B / "sd2" / "receipts" / "sd2-5090-3" / "SHA256SUMS").read_text()
    assert hashlib.sha256(SD2_V.read_bytes()).hexdigest() in sums
    red = _red()
    anchor, verify, full, per = red.timing_terms([v1, v1])
    need = red.need_ms(anchor, full, red._sd2())
    p132 = _load("p132_reduce_t", B / "p132" / "p132_reduce.py")
    assert abs(anchor - p132.ANCHOR_REG) < 1e-12
    assert all(abs(verify[n] - p132.VERIFY_REG[n]) < 1e-12 for n in red.NS)
    assert all(abs(need[n] - p132.NEED[n]) < 1e-12 for n in red.NS), (need, p132.NEED)


def test_a_noisy_verdict_carries_no_quotable_label():
    """Planner's R3: H, BUILD and build_at sit at the top level only on READ; a NOISY run keeps them under "outcome"."""
    red = _red()
    with tempfile.TemporaryDirectory() as d:
        red._fake(d, anchor=(4.10, 4.27), terms=red._world(0.9))
        r = red.judge(d)
    assert r["verdict"] == "NOISY" and not {"H", "BUILD", "build_at"} & set(r)
    assert r["outcome"]["H"] == "SUPPORT" and r["outcome"]["BUILD"] == "BUILD"


def test_the_guard_and_the_launch_terms_are_the_registered_ones():
    """R4 and R5: the runner's guard is 2.0 h, and the PREREG's disk order and worst follow from the runner's floor and
    the stated caps."""
    run = (LANE / "vsplit_run.sh").read_text(encoding="utf-8")
    assert re.search(r"^GUARD_S=7200;.*# --wallclock-h 2\.0$", run, re.M)
    floor = int(re.search(r"MIN_DISK_GB=\$\{VSPLIT_MIN_DISK_GB:-(\d+)\}", run).group(1))
    pre = (LANE / "PREREG-vsplit.md").read_text(encoding="utf-8")
    assert f"≥ {floor + 20} GB" in pre and "inet_down ≥ 1000" in pre
    assert f"**${7200 / 3600 * 0.85 + 100 * 0.011:.2f}**" in pre


def test_the_outcome_alone_flips_the_verdict():
    """The maintainer's condition 1: with every input held fixed (the timing passes, the n = 1 census, tau, G_BAR, rho),
    the n >= 2 shares alone take the reading from SUPPORT/BUILD to REFUTE/NO_BUILD."""
    red = _red()
    with tempfile.TemporaryDirectory() as d:
        red._fake(d, terms=red._world(0.9))
        hi = red.judge(d)
    with tempfile.TemporaryDirectory() as d:
        red._fake(d, terms={1: red._world(0.9)[1], **{n: red._world(0.2)[n] for n in red.NS}})
        lo = red.judge(d)
    assert hi["inputs"]["need"] == lo["inputs"]["need"] and hi["inputs"]["bytes"] == lo["inputs"]["bytes"]
    assert (hi["verdict"], hi["H"], hi["BUILD"]) == ("READ", "SUPPORT", "BUILD")
    assert (lo["verdict"], lo["H"], lo["BUILD"]) == ("READ", "REFUTE", "NO_BUILD")


def test_ms_come_only_from_the_unprofiled_passes():
    """The maintainer's condition 2: inflating every census term and its period by the same factor (a profiler's
    uniform inflation) changes no millisecond the rule uses."""
    red = _red()
    outs = []
    for k in (1.0, 1.7):
        with tempfile.TemporaryDirectory() as d:
            red._fake(d, terms=red._world(0.8))
            for n in (1,) + red.NS:
                p = pathlib.Path(d) / f"census_n{n}.json"
                c = json.loads(p.read_text())
                c["terms"] = {kk: v * k for kk, v in c["terms"].items()}
                c["P_ms"] *= k
                p.write_text(json.dumps(c))
            outs.append(red.judge(d))
    a, b = outs
    assert a["outcome"]["S"] == pytest.approx(b["outcome"]["S"], abs=1e-12)
    assert a["outcome"]["share"] == pytest.approx(b["outcome"]["share"], abs=1e-12)
    assert a["reporting"]["census_over_unprofiled"] != b["reporting"]["census_over_unprofiled"]   # reported, not used


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "VSPLIT_DRIVE_DRYRUN": "1",
            "E4B_SHA": "1" * 40, "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q",
            "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800",
            "E4B_RENT_INSTANCE_ID": "0"}


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    out = subprocess.run(["bash", str(LANE / "vsplit_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "vsplit-5090-dry"), timeout=120)
    if "is not in" in out.stdout + out.stderr:
        pytest.skip("SD2's proven or read commit is not in this checkout")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/vsplit" in out.stdout and "bash vsplit_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    out = subprocess.run(["bash", str(LANE / "vsplit_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p132-5090-9"), timeout=120)
    assert out.returncode == 78 and "is not vsplit-5090-N" in out.stdout + out.stderr


def test_the_reference_records_carry_an_excess_beyond_the_bytes():
    """Real records first, for the NO_EXCESS rung: sd2-5090-3's V1 increment against the bytes term built from P123's
    real B = 1 census share (p123-5090-1, the same model and route) has an excess above the registered floor at every n,
    so the reference case reads H over all three n rather than NO_EXCESS. One pass stands for both, so the spread term
    is zero and the floor is its 5 % branch."""
    red = _red()
    v1 = json.loads(SD2_V.read_text())["v1"]
    anchor, verify, full, per = red.timing_terms([v1, v1])
    p123 = B / "p123" / "receipts" / "p123-5090-1"
    raw = (p123 / "census_b1.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() in (p123 / "SHA256SUMS").read_text()
    sh1 = red.shares(json.loads(raw))
    by = {n: (red.RHO[n] - 1.0) * sh1["moe_expert"] * anchor for n in red.NS}
    inc = {n: verify[n] - anchor for n in red.NS}
    floor = red.excess_floor(inc, inc, inc)
    assert floor == {n: red.EXCESS_FRAC * inc[n] for n in red.NS}
    assert all(inc[n] - by[n] > floor[n] for n in red.NS), (inc, by, floor)
