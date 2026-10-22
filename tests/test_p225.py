# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P225, the Train->Serve successor to P224 (bench/p225/PREREG-p225.md). The six self-tests; every runner call through
its script's own argument check; the needs and alarms in the runner against the derivation that registers
them; the guard inside the launcher's wallclock cap; the host screen before any install, refusing only through the
floors; the held-out set extending P224's byte for byte; the staged pins; the run ids across runner, driver and
PREREG. (Private names are the repository guards' job, not this file's.)"""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p225"
P224 = REPO / "bench" / "p224"
PREREG = (LANE / "PREREG-p225.md").read_text(encoding="utf-8")
RUN = (LANE / "p225_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p225_drive.sh").read_text(encoding="utf-8")
LAUNCHER_CAP_H = 6.0          # the launcher's teardown max_wallclock_h; the guard keeps a 0.1 h margin under it
#: the per-adapter SDs of PA - REF and the largest per-row kurtosis, from the five named receipts (PREREG "The held-out set")
SD_INPUTS = [0.0481, 0.0663, 0.0470, 0.0566, 0.0621]
KURTOSIS_MAX = 4.6
SOURCES = {"p225_run.sh": LANE, "p225_probe.py": LANE, "p225_gate.py": LANE, "p225_canary.py": LANE,
           "p225_reduce.py": LANE, "p224_box.py": P224, "p224_canary.py": P224, "p224_reduce.py": P224,
           "phases.sh": REPO / "bench" / "common", "calib.json": REPO / "bench" / "p39",
           "test_adapter_schema.py": REPO / "tests"}
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([str(LANE), str(P224)])}


def _load(name):
    d = LANE if (LANE / f"{name}.py").exists() else P224
    spec = importlib.util.spec_from_file_location(name, d / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path[:0] = [str(LANE), str(P224)]
    try:
        spec.loader.exec_module(mod)
    finally:
        del sys.path[:2]
    return mod


@pytest.mark.parametrize("name", ["p225_probe", "p225_gate", "p225_reduce", "p225_canary", "p225_size", "p225_derive"])
def test_self_tests_pass(name):
    r = subprocess.run([sys.executable, str(LANE / f"{name}.py"), "--self-test"], capture_output=True, text=True,
                       timeout=300, cwd=str(LANE), env=ENV)
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


def _joined_lines(text):
    out, cur = [], ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            cur += line.rstrip()[:-1] + " "
            continue
        out.append(cur + line)
        cur = ""
    return out


def _calls(script):
    """Every ``python $W/<script> ...`` call in the runner as an argv: $-variables become placeholders, the optional
    ``$([ ... ] && echo --x)`` becomes ``--x``, redirections and pipes are cut."""
    calls = []
    for line in _joined_lines(RUN):
        m = re.search(r"python \$W/" + re.escape(script) + r"(?P<rest>.*)", line)
        if not m:
            continue
        rest = re.sub(r"\$\(\[[^)]*&& echo (--[a-z-]+)\)", r"\1", m["rest"])
        rest = re.split(r"\s(?:\d?>|>>|\||;|&&|\|\|)", " " + rest)[0]
        for var, val in (("$ARM", "PA"), ("$arm", "REF"), ("$TAG", "b0B"), ("$TIER", "a"), ('"$1"', "install")):
            rest = rest.replace(var, val)
        calls.append([re.sub(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", "X", tok) for tok in shlex.split(rest)])
    return calls


@pytest.mark.parametrize("script", ["p225_probe.py", "p225_gate.py", "p225_reduce.py", "p225_canary.py",
                                    "p224_box.py"])
def test_every_runner_call_passes_the_scripts_own_argument_check(script):
    calls = _calls(script)
    assert calls, script
    src = SOURCES[script] / script
    for argv in calls:
        r = subprocess.run([sys.executable, str(src), *argv, "--check-args"], capture_output=True, text=True,
                           timeout=60, cwd=str(src.parent), env=ENV)
        assert r.returncode == 0 and "ARGS OK" in r.stdout, (argv, r.stdout + r.stderr)


def test_a_reduce_call_without_gates_fails_the_argument_check():
    r = subprocess.run([sys.executable, str(LANE / "p225_reduce.py"), "--dir", "X", "--pins", "X", "--out", "X",
                        "--check-args"], capture_output=True, text=True, timeout=60, cwd=str(LANE), env=ENV)
    assert r.returncode == 2 and "--gates" in r.stderr


def _needs():
    """The runner's NEED_* and *_ALARM assignments, parsed from its own text (never restated here)."""
    vals = {}
    for line in RUN.splitlines():
        if re.match(r"(NEED_(FETCH|BAKE|TRAIN|REF|SLOT|SCORE)|(FETCH|BAKE|TRAIN|REF|SLOT)_ALARM)=", line):
            vals.update(re.findall(r"([A-Z_]+)=(\S+?)(?:;|$)", line))
    assert len(vals) == 11 and all(v.isdigit() for v in vals.values()), f"unfilled or missing: {vals}"
    return {k: int(v) for k, v in vals.items()}


def test_runner_needs_and_alarms_are_the_derivation():
    D, C = _load("p225_derive"), _load("p225_canary")
    plan = D.plan("a", C.N_HELDOUT)
    v = _needs()
    for k in ("fetch", "bake", "train", "ref", "slot", "score"):
        assert v[f"NEED_{k.upper()}"] == plan["need"][k], k
    for k in ("fetch", "bake", "train", "ref", "slot"):
        assert v[f"{k.upper()}_ALARM"] == plan["alarm"][k], k
    for k in ("fetch", "bake", "train", "ref"):
        assert v[f"{k.upper()}_ALARM"] >= D.ALARM_X * v[f"NEED_{k.upper()}"]
    assert v["SLOT_ALARM"] >= D.ALARM_X * (v["NEED_SLOT"] + v["NEED_SCORE"])


def test_every_phase_alarm_is_a_named_constant():
    for lit in ("alarm 2700", "alarm 3600", "alarm 1800", "alarm 1600", "alarm 1500"):
        assert f"$(alarm {lit.split()[1]})" not in RUN, lit
    for name in ("FETCH_ALARM", "BAKE_ALARM", "TRAIN_ALARM", "REF_ALARM", "SLOT_ALARM"):
        assert f"$(alarm ${name})" in RUN, name


def test_the_guard_fits_under_the_launchers_wallclock_cap():
    D, C = _load("p225_derive"), _load("p225_canary")
    plan = D.plan("a", C.N_HELDOUT)
    assert plan["guard_h"] <= LAUNCHER_CAP_H - 0.1, plan["guard_h"]
    assert plan["worst_usd"] <= 15.0
    assert set(D.FLOOR_MOPS) == {"a"} and set(_load("p225_probe").FLOORS) == {"a"}     # no fallback tier at this N


def test_the_held_out_size_is_the_ruled_rule_on_the_five_sd_receipts():
    S, C = _load("p225_size"), _load("p225_canary")
    r = S.size(SD_INPUTS, KURTOSIS_MAX)
    assert (r["topics"], r["rows"]) == (C.K, C.N_HELDOUT) == (162, 1296)
    assert r["p_inconclusive"] <= 0.10 < r["p_inconclusive_one_topic_less"]
    for text in ("K = 162", "1,296", "0.0991", "4.6"):
        assert text in PREREG, text


def test_the_strike_call_is_the_reducers_map_of_every_runner_exit():
    """Every ``finish N`` the runner can reach before the marker is classified, and the PREREG table names it."""
    R = _load("p225_reduce")
    pre = RUN[:RUN.index("> $W/run/STRIKE_ELIGIBLE")]
    codes = {int(c) for c in re.findall(r"finish (\d+)", pre)}
    assert codes and codes <= set(R.HOST_CODES) | set(R.KIT_CODES), codes - set(R.HOST_CODES) - set(R.KIT_CODES)
    for rc in R.HOST_CODES:
        c = R.classify(rc, False)
        assert c["event"] == "HOST_REFUSED" and c["host_refused_cap"] and not c["strike"], rc
    for rc in R.KIT_CODES:
        c = R.classify(rc, False)
        assert c["event"] == "PRE_INSTALL_HALT" and c["halt"] and not (c["strike"] or c["host_refused_cap"]), rc
    assert R.classify(0, True)["event"] == "ANSWER" and R.classify(27, True)["strike"]
    for text in ("**HOST_REFUSED**: no strike; counts toward the cap of 4", "**PRE_INSTALL_HALT**", "--classify --work W"):
        assert text in PREREG, text
    assert 10 in R.KIT_CODES and 10 not in R.HOST_CODES and "noimport" in RUN     # a failed torch import is the image's
    assert 28 in R.KIT_CODES and "finish 28" in pre                             # the image pin is checked before the marker
    assert pre.index("finish 28") < RUN.index("phase_start install")


def _box(root, rc, marker, screen=None, name="box"):
    w = root / name
    (w / "run").mkdir(parents=True)
    (w / "RUN_ID").write_text("p225-5090-1\n", encoding="utf-8")
    (w / "P225_EXIT_CODE.ab12").write_text(f"{rc}\n", encoding="utf-8")
    if marker:
        (w / "run" / "STRIKE_ELIGIBLE").write_text("2026-10-23T06:00:00Z\n", encoding="utf-8")
    if screen is not None:
        (w / "summary.txt").write_text("P225_SCREEN " + json.dumps(screen) + "\n", encoding="utf-8")
    return w


def _classify(work):
    return subprocess.run([sys.executable, str(LANE / "p225_reduce.py"), "--classify", "--work", str(work)],
                          capture_output=True, text=True, timeout=60, cwd=str(LANE), env=ENV)


def test_the_classify_command_line_fails_closed_and_maps_each_case(tmp_path):
    struck = _box(tmp_path, 27, True, name="struck")
    r = _classify(struck)
    assert r.returncode == 0 and json.loads(r.stdout.split("P225_CLASS ", 1)[1])["event"] == "STRIKE", r.stdout
    assert json.loads(r.stdout.split("P225_CLASS ", 1)[1])["screen_read"] is False
    refused = _box(tmp_path, 16, False, name="refused")
    assert json.loads(_classify(refused).stdout.split("P225_CLASS ", 1)[1])["event"] == "HOST_REFUSED"
    halted = _box(tmp_path, 10, False, name="halted")
    assert json.loads(_classify(halted).stdout.split("P225_CLASS ", 1)[1])["event"] == "PRE_INSTALL_HALT"
    for wrong in (struck / "run", tmp_path, tmp_path / "missing"):          # one level too deep, too shallow, absent
        r = _classify(wrong)
        assert r.returncode == 2 and "P225_CLASS {" not in r.stdout, (wrong, r.stdout, r.stderr)
    (struck / "P225_EXIT_CODE.cd34").write_text("0\n", encoding="utf-8")   # two exit files: ambiguous, refused
    assert _classify(struck).returncode == 2


def test_a_fast_family_host_refused_on_speed_emits_the_retirement_line(tmp_path):
    screen = {"floors": {"mops": 30.0}, "mops": {"median": 24.0},
              "recorded": {"cpu_model": "AMD EPYC 9115 16-Core Processor", "cpu_family": "fast"}}
    r = _classify(_box(tmp_path, 19, False, screen, name="fast"))
    assert "P225_EVENT FAMILY_RULE_RETIRED" in r.stdout and '"cpu_model": "AMD EPYC 9115' in r.stdout, r.stdout
    slow = dict(screen, recorded={"cpu_model": "AMD EPYC 7763 64-Core Processor", "cpu_family": "slow"})
    assert "FAMILY_RULE_RETIRED" not in _classify(_box(tmp_path, 19, False, slow, name="slow")).stdout
    P = _load("p225_probe")
    assert P.cpu_family(screen["recorded"]["cpu_model"]) == "fast" and P.cpu_family("AMD EPYC 7763") == "slow"


def test_the_fetch_carries_the_unnamed_margin():
    D = _load("p225_derive")
    p = D.plan("a", 688)
    assert p["need"]["fetch"] == D._ceil10(D.FETCH_S * D.MARGIN_UNNAMED * D.NEED_X)
    assert "| fetch | 806 (slowest measured) | 2,120 (x 1.75 x 1.5) | 2,650 |" in PREREG


def test_the_registered_lane_events_have_their_rows():
    for row in ("| **UNRUNNABLE** |", "| FAMILY_RULE_RETIRED (an event, not an outcome) |", "| **HOST_REFUSED** |"):
        assert row in PREREG, row
    assert "The slow-family exclusion stays" in PREREG and "option (b)" not in PREREG


def test_the_derivation_never_extrapolates_below_the_measured_speeds():
    D = _load("p225_derive")
    assert (D.MOPS_SLOW, D.MOPS_FAST) == (18.6, 36.4)
    with pytest.raises(ValueError):
        D.interp(D.TRAIN_S, 18.0)
    P = _load("p225_probe")
    assert all(D.MOPS_SLOW <= D.FLOOR_MOPS[t] == P.FLOORS[t]["mops"] for t in P.FLOORS)


def test_the_host_screen_runs_before_any_install_and_refuses_only_as_host_refused():
    screen = RUN.index("python $W/p225_probe.py --floors $TIER")
    eligible = RUN.index("STRIKE_ELIGIBLE")
    install = RUN.index("phase_start install")
    fit = RUN.index("FIT_HEADROOM_PCT=125")
    assert screen < fit < eligible < install
    after = RUN[screen:RUN.index("\n", RUN.index("SCR\" = 0", screen))]
    assert "finish 19" in after and "HOST_REFUSED" in after
    assert _load("p225_probe").HOST_REFUSED == 19


def test_inherited_checks_are_never_stricter_than_the_screen():
    P = _load("p225_probe")
    ram, disk = (int(re.search(rf"^(?:[A-Z_]+=\d+;\s*)*{n}=(\d+)", RUN, re.M)[1]) for n in ("MIN_RAM_GB", "MIN_DISK_GB"))
    for f in P.FLOORS.values():
        assert ram <= f["ram_gib"]
        assert disk * 2**30 < f["disk_gb"] * 10**9           # df -BG counts GiB; the screen counts GB


def test_the_screen_judges_only_through_the_floors():
    P = _load("p225_probe")
    ok = {"mops": {"median": 40.0}, "download": {"mb_s": 300.0, "status": 200}, "ram_gib": 120, "free_gb": 400}
    assert P.judge("a", ok) == []
    for k, bad in (("mops", {"median": 29.9}), ("download", {"mb_s": 59.0, "status": 200}),
                   ("download", {"mb_s": 300.0, "status": 403}), ("ram_gib", 59), ("free_gb", 99)):
        assert len(P.judge("a", {**ok, k: bad})) == 1, (k, bad)
    assert P.judge("a", {**ok, "unrelated": "anything"}) == []           # only the four floor measurements count


def test_the_probe_loop_is_the_launchers():
    src = (LANE / "p225_probe.py").read_text(encoding="utf-8")
    assert "x = (x + i * 7) & 0xFFFF" in src and "LOOP_N = 1_000_000" in src


def test_the_held_out_set_extends_p224s_byte_exact():
    C, C224 = _load("p225_canary"), _load("p224_canary")
    d, d224 = C.build(), C224.build()
    assert d["train"] == d224["train"] and d["http"] == d224["http"]
    assert d["heldout"][:len(d224["heldout"])] == d224["heldout"]
    assert len(d["heldout"]) == C.N_HELDOUT == C.K * 8
    rows = [json.dumps(r, sort_keys=True) for r in d["heldout"]]
    assert len(set(rows)) == len(rows)
    train = {hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in d["train"]}
    assert not train & {hashlib.sha256(r.encode()).hexdigest() for r in rows}


def test_the_runner_pins_the_canary_it_stages():
    m = re.search(r"^CANARY_SHA256=([0-9a-f]{64})\b", RUN, re.M)
    assert m, "CANARY_SHA256 is unfilled"
    assert m[1] == _load("p225_canary").canary_sha256()


def test_the_reducer_judges_at_the_canarys_n():
    R = _load("p225_reduce")
    assert R.R.N_HELDOUT == _load("p225_canary").N_HELDOUT


def test_staged_pins_match_their_sources():
    lines = [ln.split() for ln in (LANE / "staged-p225.sha256").read_text(encoding="ascii").splitlines() if ln.strip()]
    assert sorted(n for _, n in lines) == sorted(SOURCES)
    for want, name in lines:
        got = hashlib.sha256((SOURCES[name] / name).read_bytes()).hexdigest()
        assert got == want, f"{name}: staged {want}, source {got}"


def test_run_ids_agree_across_runner_driver_and_prereg():
    for rid in (f"p225-5090-{i}" for i in (1, 2, 3)):
        assert rid in RUN and rid in DRIVE
    assert "p225b" not in RUN and "p225b" not in DRIVE                    # the fallback tier is withdrawn
    assert "p224-5090" not in DRIVE and "p224-shake" not in DRIVE
    assert "`p225-5090-1`" in PREREG


def test_no_shakedown_and_no_fixture_generation():
    assert "SHAKE=0" in RUN and "fixture_base" not in RUN and "FIXTURE" not in RUN
    assert "fixture_base.py" not in DRIVE


def test_the_runner_finishes_with_an_answer_only_on_pass_fail_or_inconclusive():
    assert 'case "$V" in PASS|FAIL|INCONCLUSIVE) finish 0;; *) say "P225 $V -- no reading"; finish 27;; esac' in RUN
    R = _load("p225_reduce")
    assert set(R.ANSWERS) == {"PASS", "FAIL", "INCONCLUSIVE"}
