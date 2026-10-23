# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""HO3, the planner's second-box and host-offload held-out read (bench/ho3/PREREG-ho3.md): the reducer's self-test and the
mutants it must kill; the staged pins; the stack under test pinned to HO2's exactly (never the launch commit or CI's current
pins); the registered plans agreeing across the reducer and the registration; the run ids across runner, driver and
registration; each card's fit inside its guard with the boot slack bench/common requires; HO2's machines refused; a real
Loggetta receipt read through the reducer; Amendment 2's driver-peak probe (Loggetta 0.6.2's own sampler, before the
install, rc 19 a host refusal counting toward nothing, three in a row stop the card)."""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "ho3"
HO2 = REPO / "bench" / "ho2"
PREREG = (LANE / "PREREG-ho3.md").read_text(encoding="utf-8")
RUN = (LANE / "ho3_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "ho3_drive.sh").read_text(encoding="utf-8")
REDUCE = (LANE / "ho3_reduce.py").read_text(encoding="utf-8")
HO2_RUN = (HO2 / "ho2_run.sh").read_text(encoding="utf-8")
SOURCES = {"ho3_run.sh": LANE, "ho3_reduce.py": LANE, "ho3_drvprobe.py": LANE, "loggetta_measure.py": LANE,
           "phases.sh": REPO / "bench" / "common"}
#: Loggetta 0.6.2's loggetta/measure.py (release commit 9348f8b8, the runner's LOGGETTA_SHA), staged byte for byte
LOGGETTA_062_MEASURE_SHA256 = "360363898e265a83aeefa252e96baab8082ce4431a30da9936ada6d1004b4596"
#: each card's guard (PREREG "Boxes") and bench/common's boot slack on Vast (ssh_ready_s 420 + a 120 s staging margin)
GUARD_S = {"5090": 7200, "4090": 7200, "a4000": 11400}
BOOT_SLACK_S, HEADROOM = 540, 1.25
HO2_STACK = "f6e66fd65b28fe493a89062b02aa91d3970ff995"     # HO2's registration merge, the e4b its boxes installed
FIXTURE = REPO / "tests" / "fixtures" / "ho2" / "loggetta-train-receipt.json"
FIXTURE_SHA256 = "22ca086ef4461090cebe5b0d45d039858a0e391701541f2f7abf59eeaa0c3903"
FIXTURE_RUN = {"families": {"olmoe": ("allenai/OLMoE-1B-7B-0924", None)}, "data": ("tatsu-lab/alpaca", None), "format": "alpaca",
               "steps": 12, "micro_batch": 2, "cards": {"a2000": "NVIDIA RTX A2000 12GB"}}


def _var(text, name):
    m = re.search(rf"(?:^|[;\s]){name}=([^\s;]+)", text, re.M)
    assert m, name
    return m.group(1)


def _self_test(path):
    return subprocess.run([sys.executable, str(path), "--self-test"], capture_output=True, text=True, timeout=120, cwd=str(path.parent))


def _reducer():
    spec = importlib.util.spec_from_file_location("ho3_reduce", LANE / "ho3_reduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reducer_self_test_passes():
    r = _self_test(LANE / "ho3_reduce.py")
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


MUTANTS = [
    ('d.get("tokens") == steps * mb * seq', "True"),                                           # an early stop reads as a row
    ('gpu.get("name") == reg["cards"][card]', "True"),                                         # another card's receipt
    ('all(ln.get("basis") == "inferred" for ln in over)', "True"),                             # receipts on file
    ("under, short = driver_peak > plan_total, allocated_peak > estimate", "under, short = False, False"),
    ('if len(found) != 1:', "if not found:"),                                                  # two receipts for one row
    ("host_under = h_meas > h_est", "host_under = False"),                                     # the host side unchecked
    ('if residency == "host":', 'if False:'),                                                  # a host row without host numbers
    ('box = ("VOID" if "VOID" in verdicts else', 'box = ("COVERED" if True else'),             # a VOID cohort hidden
    ("cause = cause or why", "cause = None"),                                                  # a bad row read as covered
    ("plan_diff = \"; \".join(diffs) or None", "plan_diff = None"),                            # a plan difference unrecorded
    ('if residency not in ("device", "host"):', "if False:"),                                  # an unknown residency skips the host check
    ("if not found and tag in skipped:", "if not found:"),                                     # a lost row read as a deadline skip
    ('else "UNREAD" if "UNREAD" in verdicts', 'else "COVERED" if "UNREAD" in verdicts'),        # an UNREAD cohort read as covered
]


@pytest.mark.parametrize("i", range(len(MUTANTS)))
def test_self_test_kills_the_mutants(i, tmp_path):
    old, new = MUTANTS[i]
    assert REDUCE.count(old) == 1, old
    (tmp_path / "ho3_reduce.py").write_text(REDUCE.replace(old, new), encoding="utf-8")
    assert _self_test(tmp_path / "ho3_reduce.py").returncode != 0, f"mutant survived: {old!r}"


def test_staged_pins_match_their_sources():
    lines = [ln.split() for ln in (LANE / "staged-ho3.sha256").read_text(encoding="ascii").splitlines() if ln.strip()]
    assert sorted(n for _, n in lines) == sorted(SOURCES)
    for want, name in lines:
        assert hashlib.sha256((SOURCES[name] / name).read_bytes()).hexdigest() == want, name


def test_the_stack_under_test_is_ho2s_exactly():
    # rule 1 of bench/common: a replication pins the replicated lane's stack, so a cohort's two boxes differ only in the machine
    assert _var(RUN, "E4B_STACK_SHA") == HO2_STACK and HO2_STACK in PREREG
    for name in ("GNF4_SHA", "LOGGETTA_SHA", "ALPACA_REV", "LFM2_REV", "ERNIE_REV"):
        assert _var(RUN, name) == _var(HO2_RUN, name), name
    pip, ho2_pip = (t[t.index("pipx logs/pip_e4b.log"):t.index("PIP FAIL (e4b)")] for t in (RUN, HO2_RUN))
    # the same packages and versions, plus only the fetch-side brotli floor an unpinned huggingface_hub needs
    assert pip.replace("$E4B_STACK_SHA", "$E4B_SHA").replace(' "brotli>=1.2.0"', "") == ho2_pip
    assert '"brotli>=1.2.0"' in pip
    assert 'git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_STACK_SHA' in RUN
    assert "experts4bit-qlora.git@$E4B_SHA" not in RUN                                          # the launch commit is never installed
    assert 'WANT_E4B=$E4B_STACK_SHA' in RUN and "2.8.0+cu128" in RUN


AMEND2_TRIPWIRE = (
    "# the probe ran the rows' sampler: the installed loggetta.measure is the staged copy, byte for byte (Amendment 2)\n"
    "import hashlib, loggetta.measure  # noqa: E401\n"
    '_h = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()\n'
    'assert _h(loggetta.measure.__file__) == _h("/root/ho3/loggetta_measure.py"), "installed loggetta.measure is not the probed copy"\n'
)


def test_the_tripwire_is_the_one_ho2_ran_on_this_stack():
    def trip(text):
        i = text.index('say "TRIPWIRE FAIL"')
        return text[i:text.index("\nPYT\n", i)]
    # HO2's tripwire plus one counted edit (Amendment 2): the installed loggetta.measure is the probed copy
    amend = AMEND2_TRIPWIRE
    assert trip(RUN).count(amend) == 1
    assert trip(RUN).replace(amend, "").replace("/root/ho3/", "/root/ho2/") == trip(HO2_RUN)


def test_a_failed_fetch_is_retried_once_before_the_box_refuses():
    fetch = RUN[RUN.index("phase_start fetch"):RUN.index("phase_end fetch")]
    assert "for attempt in 1 2; do" in fetch and fetch.index("for attempt in 1 2") < fetch.index('say "DL FAIL $1"')


def test_the_rows_are_ho2s_and_the_planners_own():
    assert "--format text --text-field text" in RUN and "--micro-batch 2 --steps 20" in RUN
    cmd = RUN[RUN.index("python -m loggetta train"):RUN.index("--adapter-out")]
    assert "--fix" not in cmd and "--observations" not in cmd
    assert RUN.index("python -m loggetta plan") < RUN.index("python -m loggetta train")       # the box plans before any row trains


def test_registered_plans_agree_with_the_registration():
    red = _reducer()
    for card, rows in red.REGISTERED_PLANS.items():
        for tag, (residency, device_bytes, host_bytes) in rows.items():
            assert f"{device_bytes:,}" in PREREG, (card, tag, device_bytes)
            if host_bytes is not None:
                assert f"{host_bytes:,}" in PREREG, (card, tag, host_bytes)
    assert red.REGISTERED_PLANS["a4000"]["ernie_1024"][0] == "host" and red.REGISTERED_PLANS["a4000"]["ernie_2048"][0] == "host"
    assert red.REGISTERED_PLANS["4090"] == red.REGISTERED_PLANS["5090"]                         # HO2's plan, byte-identical across cards


def test_run_ids_agree_across_runner_driver_and_prereg():
    readings = ["ho3-5090-1", "ho3-4090-1", "ho3-a4000-1", "ho3-a4000-2"]
    reruns = ["ho3-5090-2", "ho3-4090-2", "ho3-a4000-3"]                     # the cap's re-runs need ids of their own
    for rid in readings + reruns:
        assert rid in RUN and rid in DRIVE and f"`{rid}`" in PREREG
    for card in ("5090", "4090", "a4000"):
        assert f"ho3-{card}-r[1-8]" in RUN and f"ho3-{card}-r[1-8]" in DRIVE
    assert "`ho3-<card>-r1` to `-r8`" in PREREG


@pytest.mark.parametrize("card", sorted(GUARD_S))
def test_each_cards_fit_is_inside_its_guard_with_the_boot_slack(card):
    n = {k: int(v) for k, v in re.findall(r"\b(INSTALL_TO_GATE_S|NEED_FETCH|NEED_PLAN|NEED_REDUCE|RESERVE)=(\d+)", RUN)}
    m = re.search(rf"ho3-{card}-[^)]*\).*?NEED_TRAIN=(\d+)", RUN)
    assert m, card
    fit = n["INSTALL_TO_GATE_S"] + n["NEED_FETCH"] + n["NEED_PLAN"] + 4 * int(m.group(1)) + n["NEED_REDUCE"] + n["RESERVE"]
    need = int(fit * HEADROOM)
    assert GUARD_S[card] - need >= BOOT_SLACK_S, (card, fit, need)
    assert f"{fit:,} s" in PREREG, (card, fit)


def test_the_driver_refuses_ho2s_machines_and_an_unrecorded_one():
    machines = re.search(r'^HO2_MACHINES="([^"]*)"', DRIVE, re.M).group(1).split()
    assert machines == ["124996", "154091"] and all(f"`{m}`" in PREREG for m in machines)
    assert 'instance.json' in DRIVE and "recorded no machine id" in DRIVE
    assert DRIVE.index("HO2_MACHINES=") < DRIVE.index('$SCP $STAGE')                             # refused before anything is staged


def test_a_deadline_overrun_is_recorded_as_a_skip_the_reducer_reads():
    assert 'echo "SKIPPED $2" >> summary.txt' in RUN                                               # can_run's skip before a row
    assert '[ "$rc" = 142 ] && {' in RUN and 'echo "SKIPPED train $tag" >> summary.txt' in RUN     # a row that hit its alarm
    assert "skipped=load_skipped(a.dir)" in REDUCE


def _dry_run(tmp_path, run_id, machine, earlier=()):
    """The driver's $0 dry run in a fake runs root: ``earlier`` lists (run id, machine or None, ran[, the runner's exit])
    for boxes before it, in launch order (the launcher's instance.json "at")."""
    root = tmp_path / "runs" / "2026-10-24"
    for i, (rid, m, ran, *rc) in enumerate(earlier):
        d = root / rid
        d.mkdir(parents=True)
        if m is not None:
            (d / "instance.json").write_text(json.dumps({"machine_id": m, "at": f"2026-10-24T10:{i:02d}:00Z"}))
        if ran:
            (d / "ho3").mkdir()
            if rc:
                (d / "ho3" / f"HO3_EXIT_CODE.{'0' * 64}").write_text(f"{rc[0]}\n")
    me = root / run_id
    me.mkdir(parents=True)
    (me / "instance.json").write_text(json.dumps({"machine_id": machine}))
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "HO3_DRIVE_DRYRUN": "1",
           "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(me),
           "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "9999999999", "E4B_RENT_INSTANCE_ID": "1"}
    return subprocess.run(["bash", str(LANE / "ho3_drive.sh")], capture_output=True, text=True, env=env, timeout=60)


def _clean_checkout():
    r = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


@pytest.mark.skipif(not _clean_checkout(), reason="the driver refuses a dirty checkout before its machine checks")
def test_part_b_needs_two_machines_and_fails_closed(tmp_path):
    ok = _dry_run(tmp_path / "a", "ho3-a4000-2", "777", earlier=[("ho3-a4000-1", "778", True)])
    assert ok.returncode == 0 and "DRYRUN" in ok.stdout, ok.stdout + ok.stderr
    same = _dry_run(tmp_path / "b", "ho3-a4000-2", "777", earlier=[("ho3-a4000-1", "777", True)])
    assert same.returncode == 78 and "Part B needs a second machine" in same.stdout, same.stdout
    unread = _dry_run(tmp_path / "c", "ho3-a4000-3", "777", earlier=[("ho3-a4000-1", None, True)])
    assert unread.returncode == 78 and "machine is unreadable" in unread.stdout, unread.stdout
    refused = _dry_run(tmp_path / "d", "ho3-a4000-2", "777", earlier=[("ho3-a4000-r1", "777", False)])
    assert refused.returncode == 0, refused.stdout                       # a box that never ran the runner holds no machine
    ho2 = _dry_run(tmp_path / "e", "ho3-5090-1", "124996")
    assert ho2.returncode == 78 and "ran HO2's rows" in ho2.stdout, ho2.stdout


def test_the_driver_never_fetches_tensors():
    assert "--exclude 'run/adapter_*'" in DRIVE and "--exclude '*.safetensors'" in DRIVE


def test_a_real_loggetta_receipt_reads_through_the_reducer():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    red, r = _reducer(), json.loads(raw)
    row, why = red.row_of("olmoe_512", r, "a2000", FIXTURE_RUN)
    assert row is not None, why
    assert row["plan_total_bytes"] == r["plan"]["selected"]["device_bytes"] and row["outcome"] == "UNDER+SHORT"
    assert row["plan_matches_registration"] and row["plan_diff"] is None                         # no plan registered for it
    row, why = red.row_of("lfm2_1024", r, "a4000")
    assert row is None and "card 'NVIDIA RTX A2000 12GB'" in why


# ---- Amendment 2: the driver-peak probe

def _drvprobe(path=LANE / "ho3_drvprobe.py"):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("ho3_drvprobe", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(path.parent))


def test_the_staged_sampler_is_loggetta_062s_own():
    raw = (LANE / "loggetta_measure.py").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == LOGGETTA_062_MEASURE_SHA256
    assert _var(RUN, "LOGGETTA_SHA") == "9348f8b865f9e429f0628ae6e5be412787447331"
    assert "if p.strip() == pid and mib.strip().isdigit():" in raw.decode()        # the rows' own match rule


def test_drvprobe_self_test_passes():
    r = _self_test(LANE / "ho3_drvprobe.py")
    assert r.returncode == 0 and "self-test OK (7 fixtures)" in r.stdout, r.stdout + r.stderr


def test_probe_and_sampler_agree_on_the_same_fixtures():
    """The probe's decision IS the sampler's result: a row's sampler takes no sample exactly where the probe refuses."""
    mod = _drvprobe()
    pid = os.getpid()
    for name, out, want in mod.FIXTURES:
        text = out.format(pid=pid, other=pid + 7919)
        samples, peak = mod._with_smi(text)
        assert mod.decide(samples, peak)[0] is want, name
        assert (samples > 0 and peak >= mod.ALLOC_BYTES) is want, name    # what the rows' sampler itself reads
    samples, _ = mod._with_smi("")                                        # ho3-4090-r1's host: no per-process rows
    assert samples == 0 and not mod.decide(samples, 0)[0]


PROBE_MUTANTS = [
    ("ho3_drvprobe.py", "if peak < ALLOC_BYTES:", "if False:"),                                   # a peak below the allocation passes
    ("loggetta_measure.py", "if p.strip() == pid and mib.strip().isdigit():", "if mib.strip().isdigit():"),  # any process's row
]


@pytest.mark.parametrize("i", range(len(PROBE_MUTANTS)))
def test_drvprobe_self_test_kills_the_mutants(i, tmp_path):
    target, old, new = PROBE_MUTANTS[i]
    for f in ("ho3_drvprobe.py", "loggetta_measure.py"):
        text = (LANE / f).read_text(encoding="utf-8")
        if f == target:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        (tmp_path / f).write_text(text, encoding="utf-8")
    assert _self_test(tmp_path / "ho3_drvprobe.py").returncode != 0, f"mutant survived: {old!r}"


def test_the_probe_runs_before_the_install_and_refuses_with_rc_19():
    probe = RUN.index("python $W/ho3_drvprobe.py | tee -a summary.txt")
    assert RUN.index("python $W/ho3_drvprobe.py --self-test") < probe
    assert RUN.index('-ge "$MIN_RAM_GB"') < probe < RUN.index("phase_start install") < RUN.index("pip install")
    assert RUN.index('finish 41; }') < probe       # after the guard fit: it runs another file (tests/test_lane_guard_fit.py)
    assert 'echo "refused: driver peak not sampleable" > REFUSAL; finish 19;;' in RUN
    # only the probe's rc 19 is a host refusal; any other nonzero is a kit halt (21), never a refusal
    case = RUN[RUN.index('case "$DRV_RC" in'):RUN.index("esac", RUN.index('case "$DRV_RC" in'))]
    assert re.findall(r"^  (\S+)\)", case, re.M) == ["0", "19", "*"] and "finish 21;;" in case.split("*)")[1]
    assert "merge commit (Amendment 3)." in PREREG                                    # every unstarted box launches from it


def test_the_tripwire_ties_the_installed_sampler_to_the_probed_copy():
    trip = RUN[RUN.index("PYT' || {"):RUN.index("\nPYT\n")]
    assert '_h(loggetta.measure.__file__) == _h("/root/ho3/loggetta_measure.py")' in trip


def test_every_pre_install_exit_is_classified():
    red = _reducer()
    before = RUN[:RUN.index("phase_start install")]
    exits = {int(n) for n in re.findall(r"finish (\d+)", before)}
    kit_faults = {9, 10, 21, 78, 130}    # staging, a dud box, a self-test, configuration, a signal: bench/common decides
    assert set(red.PRE_INSTALL) <= exits and exits - set(red.PRE_INSTALL) == kit_faults, exits


def test_rc_19_is_a_host_refusal_that_counts_toward_nothing():
    r = subprocess.run([sys.executable, str(LANE / "ho3_reduce.py"), "--classify", "19"], capture_output=True, text=True, timeout=60)
    c = json.loads(r.stdout)
    assert c == {"rc": 19, "class": "HOST_REFUSED", "started": False, "strike": False, "counts_toward": "nothing"}, c
    assert "rc 19 is a host refusal before the install" in PREREG and "counts toward nothing" in PREREG


@pytest.mark.skipif(not _clean_checkout(), reason="the driver refuses a dirty checkout before its machine checks")
def test_three_consecutive_driver_peak_refusals_stop_the_card(tmp_path):
    three = [("ho3-4090-r2", "1", True, 19), ("ho3-4090-r3", "2", True, 19), ("ho3-4090-r4", "3", True, 19)]
    stop = _dry_run(tmp_path / "a", "ho3-4090-r5", "777", earlier=three)
    assert stop.returncode == 78 and "3 consecutive driver-peak refusals" in stop.stdout, stop.stdout
    broken = [three[0], ("ho3-4090-2", "4", True, 0), *three[1:]]
    ok = _dry_run(tmp_path / "b", "ho3-4090-r5", "777", earlier=broken)
    assert ok.returncode == 0 and "DRYRUN" in ok.stdout, ok.stdout + ok.stderr
    other = _dry_run(tmp_path / "c", "ho3-5090-r5", "777", earlier=three)    # another card's refusals never stop this one
    assert other.returncode == 0 and "DRYRUN" in other.stdout, other.stdout + other.stderr
    unread = _dry_run(tmp_path / "d", "ho3-4090-r5", "777", earlier=[("ho3-4090-r2", None, True, 19)])
    assert unread.returncode == 78 and "unreadable" in unread.stdout, unread.stdout


def test_the_fetch_never_imports_links():
    """Amendment 3: every rsync in the driver skips links, and a fetched link refuses before anything reads the tree."""
    rsyncs = re.findall(r"^\s*rsync\b[^\n]*", DRIVE, re.M)
    assert rsyncs and all("--no-links" in r for r in rsyncs), rsyncs
    check = DRIVE.index('find "$RUN_DIR/ho3" -type l -print -quit')
    assert DRIVE.index("rsync -az --no-links") < check < DRIVE.index("HO3_RUN_NONCE\" 2>/dev/null")


def test_a_fetched_link_is_recorded_removed_and_refused(tmp_path):
    """The refusal block itself, run on a tree with links: each is recorded (never followed) and removed, then exit 22."""
    start = DRIVE.index('if [ -n "$(find "$RUN_DIR/ho3" -type l -print -quit)" ]; then')
    block = DRIVE[start:DRIVE.index("\nfi\n", start) + 4]
    run = tmp_path / "run"
    (run / "ho3" / "logs").mkdir(parents=True)
    (run / "ho3" / "summary.txt").write_text("ok\n")
    secret = tmp_path / "secret.env"
    secret.write_text("TOKEN=x\n")
    (run / "ho3" / "logs" / "a b.log").symlink_to(secret)
    (run / "ho3" / "up").symlink_to("../../../etc")
    script = 'say(){ echo "$*"; }\nRUN_DIR="$1"\n' + block + "echo passed\n"
    r = subprocess.run(["bash", "-c", script, "x", str(run)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 22 and "2 symlink(s) arrived" in r.stdout, r.stdout + r.stderr
    assert not [p for p in (run / "ho3").rglob("*") if p.is_symlink()]
    rec = (run / "ho3" / "refused_links.txt").read_text()
    assert "TOKEN" not in rec and str(secret) in rec and "../../../etc" in rec and rec.count("\n") == 2, rec
    assert (run / "ho3" / "summary.txt").read_text() == "ok\n" and secret.read_text() == "TOKEN=x\n"


def test_a_box_supplied_refused_links_file_is_replaced_never_followed(tmp_path):
    """The box controls the fetched tree: its own refused_links.txt (a file, or a link to a controller file) is replaced."""
    start = DRIVE.index('if [ -n "$(find "$RUN_DIR/ho3" -type l -print -quit)" ]; then')
    block = DRIVE[start:DRIVE.index("\nfi\n", start) + 4]
    script = 'say(){ echo "$*"; }\nRUN_DIR="$1"\n' + block
    for kind in ("file", "link"):
        run = tmp_path / kind / "run"
        (run / "ho3").mkdir(parents=True)
        target = tmp_path / kind / "controller.txt"
        target.write_text("keep\n")
        rl = run / "ho3" / "refused_links.txt"
        if kind == "file":
            rl.write_text("forged -> nothing\n")
            (run / "ho3" / "x").symlink_to(target)
        else:
            rl.symlink_to(target)
        r = subprocess.run(["bash", "-c", script, "x", str(run)], capture_output=True, text=True, timeout=30)
        assert r.returncode == 22, r.stdout + r.stderr
        assert target.read_text() == "keep\n" and not rl.is_symlink() and not [p for p in run.rglob("*") if p.is_symlink()]
        rec = rl.read_text()
        assert "forged" not in rec and rec.startswith("# the box shipped"), rec

