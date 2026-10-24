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


def _history_missing(why: str):
    """A history-dependent assert whose commits this checkout lacks. CI fetches full history, so there it is a broken
    checkout and fails; a local shallow clone skips (the e4b#1898 rule)."""
    if os.environ.get("CI"):
        pytest.fail(why + " (CI must fetch full history)")
    pytest.skip(why)


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
    assert re.search(rf"^SKIP=32; STEPS={red.STEPS}; CANARY_STEPS=12; MOE_LAYERS=48$", run, re.M)   # Amendment 1: 8 -> 12
    assert "--min-steps $((CANARY_STEPS - 4))" in run                                               # Amendment 1: - 2 -> - 4
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
        _history_missing("SD2's proven or read commit is not in this checkout")
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


# ---- Amendment 1 (2026-10-16): the canary's floor. p132-5090-1 stopped at rc 26 because SC1b keeps at most N - 3 of N
# bracketed replays (one row per period between consecutive replays, then both edges trimmed) and the registration's
# floor was N - 2. Real records first: the run's own canary records, byte-identical to its receipt, digests checked.
FIX = REPO / "tests" / "fixtures" / "p132_canary"
FIX_SHA256 = {"canary.json": "a5f5068ff57c6f5da5889031e46ba5e305384e1523326f52683a2f5a2f05e316",
              "canary_census.json": "1467f9d2a388e1d96d1caea8446a0644e89d6f372fccc3ec2c5f2bc42a128ee8",
              "canary_drv_n2_graph.json": "ba4db33afbd61ba8608c521e3ff0f0fdc6583a40980792a2760a2007b4293c37",
              "canary_drv_n2_node.json": "862123a5d959a70b62e43f2aec4da256904f7eecddfff7529beeeb0e487ebaf5"}
REGISTERED_OFFSET = 2          # the registration's `--min-steps $((CANARY_STEPS - 2))` (#1632, 172d8802): the old rule


def _record(name):
    raw = (FIX / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIX_SHA256[name], f"{name} is not p132-5090-1's record"
    return json.loads(raw)


def _canary_rule():
    """(CANARY_STEPS, the floor's offset, STEPS, MOE_LAYERS) as p132_run.sh runs them; the floor is set once."""
    run = (LANE / "p132_run.sh").read_text(encoding="utf-8")
    m = re.search(r"^SKIP=\d+; STEPS=(\d+); CANARY_STEPS=(\d+); MOE_LAYERS=(\d+)$", run, re.M)
    floors = re.findall(r"--min-steps \$\(\(CANARY_STEPS - (\d+)\)\)", run)
    assert len(floors) == 1 and run.count("--min-steps") == 1, "only the canary's census sets a floor"
    return int(m.group(2)), int(floors[0]), int(m.group(1)), int(m.group(3))


def _bracket(n, long_periods=()):
    """An export holding n replays of one decode graph (as a census capture brackets them), each a launch with in-graph
    kernels; a period whose index is in long_periods runs 2x (SC1b drops periods over 1.5x the median)."""
    graphs, launches, kernels, t = [], [], [], 1_000_000_000
    for i in range(n):
        graphs.append({"start": t, "end": t + 9_000_000, "corr": 100 + i, "graph": 7, "exec": 70})
        launches.append({"start": t - 5_000, "end": t - 1_000, "corr": 100 + i})
        for j in range(4):
            a = t + j * 2_250_000
            kernels.append({"start": a, "end": a + 2_000_000, "stream": 7, "corr": 100 + i, "node": 1 + j,
                            "name": "_gemv_nf4_bw", "demangled": "_gemv_nf4_bw", "grid": (1, 1, 1), "kind": "kernel"})
        t += 20_000_000 if i in long_periods else 10_000_000
    return {"kernels": kernels, "copies": [], "launches": launches, "graphs": graphs, "diagnostics": []}


def _read(mode, ex, floor, moe_layers=48):
    sc = _load("sc1b_census_a1", B / "sc1b" / "sc1b_census.py")       # the staged piece (pinned above)
    if mode == "graph":
        return sc.graph_mode(ex, min_steps=floor)
    return sc.node_mode(ex, sc.load_classes(str(LANE / "kernel_classes_nf4_v1_2.json"), "e4b"), min_steps=floor,
                        moe_layers=moe_layers)


@pytest.mark.parametrize("mode", ["graph", "node"])
def test_a1_i_the_registered_floor_reproduces_the_runs_void(mode):
    """The old-rule mutant, on the real record: the record's bracket count read at the registration's floor gives back
    the run's own `why` string byte for byte, and the census was the canary's only failed check."""
    canary, census, drv = _record("canary.json"), _record("canary_census.json"), _record(f"canary_drv_n2_{mode}.json")
    assert canary["status"] == "failed" and canary["missing_buckets"] == [] and set(canary["replays"].values()) == {"ok"}
    assert canary["census"] == {"status": "void", "labels": [], "blocking": []}
    assert (drv["status"], drv["n"], drv["capture"], drv["steps_bracketed"]) == ("ok", 2, mode, 8)
    n = drv["steps_bracketed"]
    got = _read(mode, _bracket(n), n - REGISTERED_OFFSET)
    assert got["status"] == census[mode]["status"] == "void"
    assert got["why"] == census[mode]["why"] and got["dropped"] == census[mode]["dropped"] == {}


@pytest.mark.parametrize("mode", ["graph", "node"])
def test_a1_ii_the_amended_floor_fits_what_the_reader_keeps(mode):
    canary_steps, offset, _, moe = _canary_rule()
    for n in (_record(f"canary_drv_n2_{mode}.json")["steps_bracketed"], canary_steps):
        ceiling = _read(mode, _bracket(n), 1, moe)["steps_kept"]
        assert ceiling == n - 3                                  # the reader's own rule, read, not assumed
        assert n - offset <= ceiling and _read(mode, _bracket(n), n - offset, moe)["status"] == "ok"
    mid = canary_steps // 2
    one = _read(mode, _bracket(canary_steps, {mid}), canary_steps - offset, moe)
    assert one["status"] == "ok" and one["dropped"] == {"period_over_1.5x_median": 1}
    two = _read(mode, _bracket(canary_steps, {mid - 2, mid}), canary_steps - offset, moe)
    assert two["status"] == "void"                               # a second long period still stops the lane


@pytest.mark.parametrize("mode", ["graph", "node"])
def test_a1_iii_the_main_census_keeps_at_least_sc1bs_floor(mode):
    sc = _load("sc1b_census_a1m", B / "sc1b" / "sc1b_census.py")
    _, _, steps, moe = _canary_rule()
    got = _read(mode, _bracket(steps), sc.MIN_STEPS, moe)
    assert got["status"] == "ok" and got["steps_kept"] == steps - 3 >= sc.MIN_STEPS


# ---- Amendment 1 through SC1b's export route: the same assertions, read from an nsys-shaped sqlite written by SC1b's
# own _mkdb and parsed by its own load(), so the schema, the column mapping, the sort and the graph-launch rows are the
# reader's, not this file's. Each case is also read through _bracket and must agree with it exactly.
def _export(tmp, n, long_periods=()):
    """_bracket's replays, written as an export with SC1b's _mkdb and read back with SC1b's load()."""
    sc = _load("sc1b_census_a1x", B / "sc1b" / "sc1b_census.py")
    ex = _bracket(n, long_periods)
    path = tmp / f"n{n}_long{'-'.join(map(str, sorted(long_periods))) or 'none'}.sqlite"
    sc._mkdb(str(path),
             kernels=[(k["start"], k["end"], k["corr"], k["node"], k["name"], k["grid"]) for k in ex["kernels"]],
             launches=[(x["start"], x["end"], x["corr"]) for x in ex["launches"]],
             graphs=[(g["start"], g["end"], g["corr"], g["exec"]) for g in ex["graphs"]])
    return sc.load(str(path))


@pytest.mark.parametrize("mode", ["graph", "node"])
def test_a1_export_route_reproduces_the_runs_void_byte_for_byte(mode, tmp_path):
    census, drv = _record("canary_census.json"), _record(f"canary_drv_n2_{mode}.json")
    n = drv["steps_bracketed"]
    got = _read(mode, _export(tmp_path, n), n - REGISTERED_OFFSET)
    assert got["why"] == census[mode]["why"] and got["dropped"] == census[mode]["dropped"] == {}
    assert got == _read(mode, _bracket(n), n - REGISTERED_OFFSET)


@pytest.mark.parametrize("mode", ["graph", "node"])
def test_a1_export_route_old_floor_voids_and_new_floor_reads_at_8_and_12(mode, tmp_path):
    canary_steps, offset, _, moe = _canary_rule()
    n_record = _record(f"canary_drv_n2_{mode}.json")["steps_bracketed"]
    assert (n_record, canary_steps) == (8, 12)
    for n in (n_record, canary_steps):
        ex = _export(tmp_path, n)
        assert _read(mode, ex, 1, moe)["steps_kept"] == n - 3                  # the reader's ceiling, read from the export
        old, new = _read(mode, ex, n - REGISTERED_OFFSET, moe), _read(mode, ex, n - offset, moe)
        assert old["status"] == "void" and new["status"] == "ok" and new["steps_kept"] == n - 3
        assert old == _read(mode, _bracket(n), n - REGISTERED_OFFSET, moe)
        assert new == _read(mode, _bracket(n), n - offset, moe)


@pytest.mark.parametrize("mode", ["graph", "node"])
def test_a1_export_route_one_long_period_reads_and_two_void(mode, tmp_path):
    canary_steps, offset, _, moe = _canary_rule()
    mid = canary_steps // 2
    one = _read(mode, _export(tmp_path, canary_steps, {mid}), canary_steps - offset, moe)
    assert one["status"] == "ok" and one["dropped"] == {"period_over_1.5x_median": 1}
    two = _read(mode, _export(tmp_path, canary_steps, {mid - 2, mid}), canary_steps - offset, moe)
    assert two["status"] == "void" and two["dropped"] == {"period_over_1.5x_median": 2}
    assert one == _read(mode, _bracket(canary_steps, {mid}), canary_steps - offset, moe)
    assert two == _read(mode, _bracket(canary_steps, {mid - 2, mid}), canary_steps - offset, moe)


def _expect_outcome(call, outcome):
    """``call()`` must end in exactly ``outcome`` (pytest.fail.Exception or pytest.skip.Exception). pytest.raises(outcome)
    lets the other outcome escape, so a CI branch that regressed to a skip would be reported SKIPPED and pass CI."""
    try:
        call()
    except (pytest.fail.Exception, pytest.skip.Exception) as e:
        if not isinstance(e, outcome):
            pytest.fail(f"expected {outcome.__name__}, got {type(e).__name__}: {e}")
        return
    pytest.fail(f"expected {outcome.__name__}, but the call returned")


@pytest.mark.parametrize("ci,outcome", [("1", pytest.fail.Exception), ("", pytest.skip.Exception)],
                         ids=["ci-fails", "local-skips"])
def test_missing_history_fails_under_ci_and_skips_locally(monkeypatch, ci, outcome):
    if ci:
        monkeypatch.setenv("CI", ci)
    else:
        monkeypatch.delenv("CI", raising=False)
    _expect_outcome(lambda: _history_missing("the commits are not in this checkout"), outcome)
