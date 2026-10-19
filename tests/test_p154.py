# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P154, the W16 NF4 expert byte floor on the replay's route (bench/p154/PREREG-p154.md): the staged pins, with P152's
shared pieces at P152's bytes and P152's verdict at its committed receipt's; the runner cut to the routing pass, with the
power floor before the install and K25's GPU tests as the premise; the routing pass on the replay's route (DEVICE_GROUPING
set by the pass, K25 counted, the NF4 store read where K25 reads it, the arena the box's own); the library premise; every
time-left check fitting the registered guard with the measured setup; the self-tests and the mutants they must kill; the
reducer on the REAL records first (P152's verdict, and P152's off-route routing record, which must VOID); the run ids."""
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
B = REPO / "bench"
LANE = B / "p154"
P152R = B / "p152" / "receipts" / "p152-5090-2"
SOURCES = {"p154_run.sh": LANE, "p154_reduce.py": LANE, "p154_route.py": LANE, "p136_box.py": B / "p136",
           "sc1b_e4b_census.py": B / "sc1b", "p109_box.py": B / "p109", "k8_bake.py": B / "p39", "calib.json": B / "p39",
           "verdict_p152.json": P152R, "test_k25_row_exact_gpu.py": REPO / "tests"}
PYPATH = os.pathsep.join(str(B / d) for d in ("p136", "sc1b", "p109"))
RUN = (LANE / "p154_run.sh").read_text(encoding="utf-8")
ROUTE = (LANE / "p154_route.py").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p154.md").read_text(encoding="utf-8")
GUARD_H = 1.0
HEADROOM = 1.25   # the guard over the largest check: every term is one host's walls (P152's)
# P152's read box (p152-5090-2's drive log, in the private receipt store): create to the drive's start 68 s, the power
# floor 36 s and the install with its self-tests about 70 s more; the fetch about 610 s; the bake and prompts about 60 s;
# the routing pass 83 s. The allowances below are those, rounded UP: boot 300, install 450, fetch 900, bake + prompts 180.
ALLOW = {"boot": 300, "install": 450, "fetch": 900, "bake_prompts": 180}
RESERVE = 600   # can_run's own margin after each step


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _run(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=300, cwd=cwd,
                          env=dict(os.environ, PYTHONPATH=PYPATH))


def _reducer():
    spec = importlib.util.spec_from_file_location("p154_reduce_under_test", LANE / "p154_reduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p154.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_shared_pieces_are_at_p152s_pins():
    mine, p152 = _pins(LANE / "staged-p154.sha256"), _pins(B / "p152" / "staged-p152.sha256")
    for name in ("p136_box.py", "sc1b_e4b_census.py", "p109_box.py", "k8_bake.py", "calib.json"):
        assert mine[name] == p152[name], name


def test_p152s_verdict_is_staged_and_pinned_at_its_committed_bytes():
    red = _reducer()
    sha = hashlib.sha256((P152R / "verdict_p152.json").read_bytes()).hexdigest()
    sums = (P152R / "SHA256SUMS").read_text(encoding="ascii")
    assert red.P152_VERDICT_SHA256 == sha == _pins(LANE / "staged-p154.sha256")["verdict_p152.json"]
    assert f"{sha}  ./verdict_p152.json" in sums                # the record's own checksum list


def test_the_runner_is_p152s_cut_to_the_routing_pass():
    assert ("p154-5090-1|p154-5090-2|p154-5090-3) FAMILY=qwen3; MODEL=Qwen/Qwen3-30B-A3B; "
            "REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39; MOE_LAYERS=48; MIN_DISK_GB=150;;") in RUN
    gnf4 = re.search(r"^GNF4_SHA=([0-9a-f]{40}) ", RUN, re.M)
    p152_run = (B / "p152" / "p152_run.sh").read_text(encoding="utf-8")
    assert gnf4 and f"GNF4_SHA={gnf4.group(1)} " in p152_run and gnf4.group(1) in PREREG, "P152's grouped-nf4-gemm pin"
    for gone in ("nsys", "p136_box.py --prompts", "p136_census.py", "pername_table.py", "sc1b_census.py arm"):
        assert gone not in RUN, gone                            # no speed arm, no Nsight, no capture: compared as recorded
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    for pin in ('hr._k25_mode_env() == "auto"', "hr._collapsed_grouping(16, None) == (False, False)",
                "hr._collapsed_grouping(16, None) == (False, True)", "nf4_smallm.gemm_nf4_grouped_smallm",
                'md.version("grouped-nf4-gemm") == "0.47.1"'):
        assert pin in trip, pin
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_NF4_GROUPED_SMALLM", "E4B_PAGED_GRAPHS", "E4B_NF4_T1_DEVICE_GROUPING", "E4B_INT4_PREFILL", "E4B_INT8_HEAD_BW"):
        assert re.search(rf"\b{knob}\b", unset), knob
    order = [RUN.index(x) for x in ('say "install e4b @', "python -m pytest test_k25_row_exact_gpu.py",
                                    "can_run $NEED_FETCH fetch", "can_run $NEED_BAKE bake",
                                    "python $W/p154_route.py --prompts", "python $W/p154_reduce.py --route")]
    assert order == sorted(order)
    assert "grep -qE '^4 passed(, [0-9]+ warnings?)? in" in RUN


def test_the_power_floor_refuses_before_the_install():
    floor = RUN.index('[ "$PL" -ge "$MIN_POWER_W" ] || {')
    assert "MIN_POWER_W=575" in RUN and "finish 17; }" in RUN[floor:floor + 260]
    assert "power.limit unreadable" in RUN and floor < RUN.index('say "install e4b @')
    assert "${P154_MIN_POWER_W:-}" in RUN                       # a frozen reading knob


def test_the_routing_pass_runs_on_the_replays_route():
    route = RUN[RUN.index("# ---- the ROUTING pass"):RUN.index("python $W/p154_route.py --prompts") + 200]
    for x in ("E4B_PAGED_GRAPHS=0", "P136_BATCH=16", "P154_STARTED_AT=$STARTED_AT", "--skip $SKIP --steps $STEPS",
              "$ENGINE_ENV", 'gpu_snap "route_b16 start"'):
        assert x in route, x
    assert "E4B_PAGED_MAX_SEQS=16 E4B_INT8_HEAD_BW=0\"" in RUN     # P152's bf16 head (e4b#1838 made int8 the default)
    assert 'out["head"], head_faults = head_record(model, os.environ.get("E4B_INT8_HEAD_BW"))' in ROUTE
    assert RUN.index("STARTED_AT=$(date +%s)") < RUN.index('say "bake the NF4 arena"')
    build = ROUTE.index("parts = build_engine(cfg)")
    dg = ROUTE.index("hot_residency.DEVICE_GROUPING[0] = True")
    window = ROUTE.index("census_window(sched, batch, a.skip, a.steps, _start, _stop")
    assert "        rec.start()\n" in ROUTE and "        rec.stop()\n" in ROUTE   # the hooks inside the snapshots
    assert build < dg < window, "DEVICE_GROUPING is set after the build (which would reset nothing) and before the window"
    assert ROUTE.index("rec.count_k25(nf4_smallm)") < window
    # the library's own route counter, read across the same window, is the second witness of the route
    lib = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text(encoding="utf-8")
    assert 'ROUTE_SEEN[key] = ROUTE_SEEN.get(key, 0) + 1' in lib and '_route = "nf4_k25"' in lib
    assert 'seen["before"] = dict(hot_residency.ROUTE_SEEN)' in ROUTE and 'seen["after"] = dict(hot_residency.ROUTE_SEEN)' in ROUTE
    assert 'os.environ.get("E4B_PAGED_GRAPHS") != "0"' in ROUTE and "default_env_ok(env)" in ROUTE


def test_the_library_device_groups_16_nf4_rows_only_under_device_grouping(monkeypatch):
    """The premise the route record checks, on the library itself: with no int4 store, a 16-row call is device-grouped
    (K25's route) exactly when DEVICE_GROUPING is on, as the graph server sets it; eager with it off host-groups (P152)."""
    pytest.importorskip("torch")
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.delenv("E4B_NF4_GROUPED_SMALLM", raising=False)
    monkeypatch.delenv("E4B_NF4_T1_DEVICE_GROUPING", raising=False)
    monkeypatch.setattr(hr, "FORCE_SINGLETON_GROUPS", [False])
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [False])
    assert hr._collapsed_grouping(16, None) == (False, False)
    monkeypatch.setattr(hr, "DEVICE_GROUPING", [True])
    assert hr._collapsed_grouping(16, None) == (False, True)
    assert hr._k25_mode_env() == "auto"


def test_the_pass_counts_and_reads_k25_where_the_library_calls_it():
    lib = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text(encoding="utf-8")
    # K25 is looked up on its module at each MoE call, so the pass's wrapper on the module is what runs
    assert "import nf4_smallm as _k25_mod" in lib and "_k25 = _k25_mod.gemm_nf4_grouped_smallm" in lib
    # the stacks K25 reads are the residency state's, the ones the pass weighs
    assert "_fused_over_stack(xr, flat, self.h_gu_p, self.h_gu_a,\n                               self.h_dn_p, self.h_dn_a" in lib
    assert "gp, ga, dp, da = hr.h_gu_p, hr.h_gu_a, hr.h_dn_p, hr.h_dn_a" in ROUTE


def _checks(need):
    """Each time-left check's total: the measured setup's elapsed time before it, plus its need, plus the reserve."""
    at = {"FETCH": ALLOW["boot"] + ALLOW["install"]}
    at["BAKE"] = at["FETCH"] + ALLOW["fetch"]
    at["ROUTE"] = at["BAKE"] + ALLOW["bake_prompts"]
    return {k: at[k] + need[k] + RESERVE for k in at}


def test_the_time_left_check_prints_what_it_compares():
    """A skip line must print the quantity the decision compares, computed once (box K -6's lesson: it printed the need
    while the test compared the need plus the reserve)."""
    line = next(x for x in RUN.splitlines() if x.startswith("can_run(){"))
    assert 'fit=$((need + 600))' in line and '[ "$fit" -le "$left" ]' in line
    assert "needs ${fit}s" in line and "only ${left}s left" in line and line.count("600") == 2   # once computed, once named


def test_every_time_left_check_fits_the_guard():
    assert f"guard {GUARD_H} h" in PREREG
    need = {k: int(v) for k, v in re.findall(r"NEED_(FETCH|BAKE|ROUTE)=(\d+)", RUN)}
    assert set(need) == {"FETCH", "BAKE", "ROUTE"}
    checks = _checks(need)
    assert checks == {"FETCH": 2550, "BAKE": 2550, "ROUTE": 2730}, checks            # the PREREG's arithmetic
    assert max(checks.values()) * HEADROOM <= GUARD_H * 3600, "the guard keeps 25 % headroom over the largest check"
    assert "2730 x 1.25 = 3413 s <= 3600 s" in PREREG
    assert "**A strike:** a time-left skip (STOP-2, rc 40)" in PREREG


@pytest.mark.parametrize("script", ["p154_reduce.py", "p154_route.py"])
def test_the_self_test_passes(script):
    out = _run(LANE / script, "--self-test")
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_the_reducer_on_the_real_records_first():
    red = _reducer()
    raw = (P152R / "verdict_p152.json").read_bytes()
    p152 = json.loads(raw)
    assert red.recorded_gemm(p152) == {"gate_up": 8.395073, "down": 3.35296}
    off_route = json.loads((P152R / "route_b16.json").read_text(encoding="utf-8"))
    v = red.reduce(off_route, p152, hashlib.sha256(raw).hexdigest())
    assert v["verdict"] == "VOID" and any("routing-pass faults" in w for w in v["why"]), v["why"]   # P152's own pass


REDUCE_MUTANTS = [
    # the routing pass's faults ignored (P152's off-route record would read a floor)
    ('        if route.get("faults"):\n', "        if False:\n"),
    # P152's verdict accepted at any bytes
    ('    if p152 is None or p152_sha != P152_VERDICT_SHA256:\n', "    if p152 is None:\n"),
    # the floor over the box's own rate instead of the registered P141 rate
    ("        f = floor_ms(u, b, COPY_GBPS)\n", "        f = floor_ms(u, b, box or COPY_GBPS)\n"),
    # the fraction over the floor's own time instead of the RECORDED GEMM time
    ('"fraction_of_copy": round(f / gemm[p], 4)', '"fraction_of_copy": round(f / f, 4)'),
]
ROUTE_MUTANTS = [
    # device grouping not checked (P152's eager host-grouped route would pass)
    ('    if rec["collapsed_grouping"] != [False, True]:\n', "    if False:\n"),
    # the K25 count not checked
    ("    return [] if count == want else", "    return [] if True else"),
    # an arena older than the run accepted (a shipped file)
    ("        if st.st_mtime < started_at:\n", "        if False:\n"),
    # the library's own route record not checked
    ("    return [] if delta == want else", "    return [] if True else"),
    # an int8 head accepted (today's default, not P152's server)
    ('    faults = [] if (n == 0 and env_value == "0") else', '    faults = [] if True else'),
    # an int4 store accepted as NF4
    ('    bad = [i for i, r in enumerate(layers) if r.get("error") or r["int4_stores"] or not r.get("experts")',
     '    bad = [i for i, r in enumerate(layers) if r.get("error") or not r.get("experts")'),
]


@pytest.mark.parametrize("script,old,new", [("p154_reduce.py", o, n) for o, n in REDUCE_MUTANTS]
                         + [("p154_route.py", o, n) for o, n in ROUTE_MUTANTS])
def test_each_mutant_fails_the_self_test(tmp_path, script, old, new):
    src = (LANE / script).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / script).write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / script, "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P154_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p154_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p154-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p154" in out.stdout and "bash p154_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p154_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p152-5090-1"), timeout=120)
    assert out.returncode == 78 and "is not p154-5090-N" in out.stdout + out.stderr
