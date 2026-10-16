# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P132 (#1313; bench/p132/PREREG-p132.md): the staged pins, the self-tests, the registered terms against the desk
prediction's own output, the class map's one change, and the driver's dry run."""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p132"
B = REPO / "bench"
SOURCES = {"p132_run.sh": LANE, "p132_box.py": LANE, "p132_reduce.py": LANE, "kernel_classes_nf4_v1_2.json": LANE,
           "sc1b_census.py": B / "sc1b", "sd2_box.py": B / "sd2", "sd2_reduce.py": B / "sd2", "sd2_audit.py": B / "sd2",
           "audit_read.tsv": B / "sd2", "sd1_box.py": B / "sd1", "sd1_eagle3.py": B / "sd1", "chat_prompts.json": B / "sd1",
           "expect_w1.json": B / "sd1", "p109_box.py": B / "p109", "p115_quality.py": B / "p115", "p110_box.py": B / "p110",
           "p108_box.py": B / "p108", "p97_box.py": B / "p97", "k8_bake.py": B / "p39", "calib.json": B / "p39"}
# the desk prediction v2, accepted as final on the bus (2026-10-11T00:25Z): its script's bytes are the registration
DESK_SHA256 = "71229f08df388d12"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pins():
    out = {}
    for line in (LANE / "staged-p132.sha256").read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def test_every_staged_piece_matches_its_source():
    pins = _pins()
    assert set(pins) == set(SOURCES), sorted(set(pins) ^ set(SOURCES))
    for name, want in pins.items():
        got = hashlib.sha256((SOURCES[name] / name).read_bytes()).hexdigest()
        assert got == want, f"{name}: {got} != staged {want}"


def test_the_driver_and_the_runner_stage_exactly_the_pinned_pieces():
    drive = (LANE / "p132_drive.sh").read_text(encoding="utf-8")
    stage = re.search(r'^STAGE="([^"]+)"', drive, re.M).group(1).split()
    assert {pathlib.PurePosixPath(p).name for p in stage} - {"staged-p132.sha256"} == set(SOURCES)
    run = (LANE / "p132_run.sh").read_text(encoding="utf-8")
    listed = re.search(r"^for f in (.*?); do$", run, re.M | re.S).group(1).replace("\\\n", " ").split()
    assert set(listed) - {"staged-p132.sha256"} == set(SOURCES) - {"p132_run.sh"}


def test_the_runner_and_the_reducer_pin_the_same_subject():
    run = (LANE / "p132_run.sh").read_text(encoding="utf-8")
    red = _load("p132_reduce_t", LANE / "p132_reduce.py")
    assert re.search(rf"^E4B_T={red.E4B_T}\b", run, re.M)
    assert re.search(rf"^GNF4_T={red.GNF4_T}\b", run, re.M)
    assert re.search(r"^E4B_T_PROVE=539a2d2694096a81c6b272b335e26d818647cd95\b", run, re.M)   # SD2's proven build
    assert f"REV={red.REV}" in run
    assert re.search(rf"^SKIP=32; STEPS={red.STEPS}; CANARY_STEPS=8; MOE_LAYERS=48$", run, re.M)
    # SD2's read server, as stage V built it
    assert ("E4B_PAGED_MAX_SEQS=16 E4B_INT4_TILE_PROGRAMS=1 E4B_PAGED_SPEC=eagle3 E4B_PAGED_SPEC_K=3 "
            "E4B_PAGED_SPEC_HEAD=$HEAD_DIR") in run
    sd2 = (B / "sd2" / "sd2_run.sh").read_text(encoding="utf-8")
    unset_p132 = re.search(r"^unset (.*?)\n: > summary", run, re.M | re.S).group(1).split()
    unset_sd2 = re.search(r"^unset (.*?)\n: > summary", sd2, re.M | re.S).group(1).split()
    assert set(unset_p132) - {"\\"} == set(unset_sd2) - {"\\"}, "the unset list must be SD2's"


def test_the_canary_runs_before_timing_and_the_census():
    run = (LANE / "p132_run.sh").read_text(encoding="utf-8")
    i_canary, i_timing = run.index("--mode canary"), run.index("--mode timing")
    i_census = run.index("for n in 1 2 3 4; do")
    assert i_canary < i_timing < i_census
    assert "finish 26" in run[i_canary:i_timing]                  # a canary failure stops the lane (rc 26)


@pytest.mark.parametrize("script", ["p132_reduce.py", "p132_box.py"])
def test_the_self_tests_pass(script):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(B / "sd2"), str(B / "sd1"), str(B / "p109"), str(B / "p115"),
                                                        str(B / "p110"), str(B / "p108"), str(B / "p97")]))
    out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True, timeout=300,
                         env=env)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_the_desk_prediction_is_the_accepted_script_and_the_reducer_carries_its_numbers():
    assert hashlib.sha256((LANE / "desk_prediction.py").read_bytes()).hexdigest().startswith(DESK_SHA256)
    d = json.loads((LANE / "desk_out.json").read_text())
    red = _load("p132_reduce_d", LANE / "p132_reduce.py")
    for n in red.NS:
        p = d["per_n"][str(n)]
        assert abs(p["I"] - red.I_REG[n]) < 1e-9
        assert p["bytes"] == red.BYTES[n] and p["known"][1] == red.KNOWN_C[n] and p["route"][1] == red.ROUTE_C
        assert d["need"][str(n)]["threshold"] == red.NEED[n]
    assert d["anchor_ms"] == red.ANCHOR_REG
    v1 = json.loads((B / "sd2" / "receipts" / "sd2-5090-3" / "v.json").read_text())["v1"]
    assert {n: v1[f"k{n - 1}"]["verify_ms"] for n in red.NS} == red.VERIFY_REG


def test_the_class_map_changes_one_name_and_leaves_p123s_census_unmoved():
    """The maintainer's check: v1.2 adds _rope_norm_qk (gnf4 0.44, absent from P123's exports) to norm_elem, and every
    kernel name in P123's own census classifies exactly as under v1.1, so the desk's P123 terms cannot move."""
    sc = _load("sc1b_census_t", B / "sc1b" / "sc1b_census.py")
    v11 = sc.load_classes(str(B / "p123" / "kernel_classes_nf4.json"), "e4b")
    v12 = sc.load_classes(str(LANE / "kernel_classes_nf4_v1_2.json"), "e4b")

    def base(spec, name):
        it = {"name": name, "demangled": name, "kind": "kernel", "grid": (1, 1, 1)}
        for r in spec["rules"]:
            if sc._hit(it, r["match"]) and sc._where_ok(it, r.get("where"), True):
                return r["class"]
        return None
    names = set()
    for b in (1, 16):
        names |= set(json.loads((B / "p123" / "receipts" / "p123-5090-1" / f"census_b{b}.json").read_text())["node"]["name_map"])
    assert len(names) >= 20
    assert [n for n in sorted(names) if base(v11, n) != base(v12, n)] == []
    assert not any("_rope_norm_qk" in n for n in names)
    assert (base(v11, "_rope_norm_qk"), base(v12, "_rope_norm_qk")) == (None, "norm_elem")
    strip = lambda s: {k: v for k, v in s.items() if k != "rules"}  # noqa: E731
    assert strip(v11) == strip(v12)                                # segment, expert and matmul names unchanged


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P132_DRIVE_DRYRUN": "1",
           "E4B_SHA": "1" * 40, "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p132-5090-dry", "E4B_RENT_DEADLINE_EPOCH": "4102444800",
           "E4B_RENT_INSTANCE_ID": "0"}
    out = subprocess.run(["bash", str(LANE / "p132_drive.sh")], capture_output=True, text=True, env=env, timeout=120)
    if "is not in" in out.stdout + out.stderr:
        pytest.skip("SD2's proven or read commit is not in this checkout")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p132" in out.stdout and "bash p132_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P132_DRIVE_DRYRUN": "1",
           "E4B_SHA": "1" * 40, "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "sd2-5090-9", "E4B_RENT_DEADLINE_EPOCH": "4102444800",
           "E4B_RENT_INSTANCE_ID": "0"}
    out = subprocess.run(["bash", str(LANE / "p132_drive.sh")], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 78 and "is not p132-5090-N" in out.stdout + out.stderr


# ---- strike 1 (p132-5090-1): the canary's step floor -------------------------------------------------------------------
FIX = REPO / "tests" / "fixtures" / "p132-5090-1"


def _run_vars():
    run = (LANE / "p132_run.sh").read_text(encoding="utf-8")
    steps = int(re.search(r"\bCANARY_STEPS=(\d+)", run).group(1))
    trim = int(re.search(r"^CANARY_MIN_STEPS=\$\(\(CANARY_STEPS - (\d+)\)\)$", run, re.M).group(1))
    full = int(re.search(r"\bSTEPS=(\d+)", run).group(1))
    return steps, steps - trim, full, run


def _short_census(tmp, n_replays, long_at=None):
    """Exports of the canary's shape, built with SC1b's own ``_mkdb``. Graph mode gets ``n_replays`` bracketed replays of
    one decode graph; node mode gets ``n_replays`` graph launches with their in-graph kernels. ``long_at`` makes one
    period 2.5x the others."""
    sc1b = _load("sc1b_census", B / "sc1b" / "sc1b_census.py")
    us, t = 1000, 0
    graphs, kernels, launches = [], [], []
    for i in range(n_replays):
        t += (2500 if i == long_at else 1000) * us
        graphs.append((t, t + 600 * us, 100 + i, 5))
        c = 1000 + i
        launches.append((t - 20, t - 10, c))
        kernels += [(t, t + 100 * us, c, 1, "rmsnorm", (1, 1, 1)), (t + 150 * us, t + 350 * us, c, 2, "gemv", (8, 1, 1))]
    g, n = tmp / "g.sqlite", tmp / "n.sqlite"
    sc1b._mkdb(str(g), graphs=graphs)
    sc1b._mkdb(str(n), kernels=kernels, launches=launches)
    return sc1b, sc1b.load(str(g)), sc1b.load(str(n))


def test_strike_1_the_canary_floor_was_unreachable_and_the_classifier_read_it_right(tmp_path):
    """p132-5090-1 voided at the canary, though everything before the classifier passed. Every bucket was captured, every
    replay returned (canary.json), and both short captures bracketed 8 replays (the driver records).

    SC1b then kept 5 steady steps of 8 against a floor of 6. That floor could never be met: 8 replays delimit 7 periods,
    and SC1b trims the first and the last.

    Exports of that shape reproduce both of the record's verdicts byte for byte at the old floor, and read ok at the
    runner's floor."""
    rec = json.loads((FIX / "canary.json").read_text(encoding="utf-8"))
    census = json.loads((FIX / "canary_census.json").read_text(encoding="utf-8"))
    assert rec["missing_buckets"] == [] and set(rec["replays"].values()) == {"ok"}
    assert rec["census"] == {"status": "void", "labels": [], "blocking": []}
    bracketed = {json.loads((FIX / f"canary_drv_n2_{m}.json").read_text(encoding="utf-8"))["steps_bracketed"]
                 for m in ("graph", "node")}
    steps, floor, _full, _run = _run_vars()
    assert bracketed == {steps} == {8}
    sc1b, g, n = _short_census(tmp_path, steps)
    spec = sc1b.load_classes(str(LANE / "kernel_classes_nf4_v1_2.json"), "e4b")
    old = steps - 2                                                    # the floor p132-5090-1 ran with
    assert sc1b.graph_mode(g, min_steps=old)["why"] == census["graph"]["why"] == "5 steady steps kept < 6 (dropped {})"
    assert sc1b.node_mode(n, spec, min_steps=old, moe_layers=48)["why"] == census["node"]["why"]
    ok_g, ok_n = sc1b.graph_mode(g, min_steps=floor), sc1b.node_mode(n, spec, min_steps=floor, moe_layers=48)
    assert ok_g["status"] == ok_n["status"] == "ok" and ok_g["steps_kept"] == ok_n["steps_kept"] == steps - 3


def test_the_canary_floor_is_every_step_the_classifier_can_keep_and_no_fewer(tmp_path):
    """The fix does not loosen the canary. Its floor is SC1b's structural maximum, so one more lost step (a long period,
    which SC1b drops) still voids it. The full census keeps its margin: 64 replays leave 61 steps against SC1b's 56."""
    steps, floor, full, run = _run_vars()
    assert floor == steps - 3 and "--min-steps $CANARY_MIN_STEPS" in run and "CANARY_STEPS - 2" not in run
    sc1b, g, _n = _short_census(tmp_path, steps, long_at=4)
    r = sc1b.graph_mode(g, min_steps=floor)
    assert r["status"] == "void" and r["dropped"] == {"period_over_1.5x_median": 1}, r
    assert full - 3 >= sc1b.MIN_STEPS
