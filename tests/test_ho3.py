# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""HO3, the planner's second-box and host-offload held-out read (bench/ho3/PREREG-ho3.md): the reducer's self-test and the
mutants it must kill; the staged pins; the stack under test pinned to HO2's exactly (never the launch commit or CI's current
pins); the registered plans agreeing across the reducer and the registration; the run ids across runner, driver and
registration; each card's fit inside its guard with the boot slack bench/common requires; HO2's machines refused; a real
Loggetta receipt read through the reducer."""
import hashlib
import importlib.util
import json
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
SOURCES = {"ho3_run.sh": LANE, "ho3_reduce.py": LANE, "phases.sh": REPO / "bench" / "common"}
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
    ('box = "VOID" if "VOID" in verdicts else', 'box = "COVERED" if True else'),               # a VOID cohort hidden
    ("cause = cause or why", "cause = None"),                                                  # a bad row read as covered
    ("plan_diff = \"; \".join(diffs) or None", "plan_diff = None"),                            # a plan difference unrecorded
    ('if residency not in ("device", "host"):', "if False:"),                                  # an unknown residency skips the host check
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
    assert pip.replace("$E4B_STACK_SHA", "$E4B_SHA") == ho2_pip                                 # the same packages, same versions
    assert 'git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_STACK_SHA' in RUN
    assert "experts4bit-qlora.git@$E4B_SHA" not in RUN                                          # the launch commit is never installed
    assert 'WANT_E4B=$E4B_STACK_SHA' in RUN and "2.8.0+cu128" in RUN


def test_the_tripwire_is_the_one_ho2_ran_on_this_stack():
    def trip(text):
        i = text.index('say "TRIPWIRE FAIL"')
        return text[i:text.index("\nPYT\n", i)]
    assert trip(RUN).replace("/root/ho3/", "/root/ho2/") == trip(HO2_RUN)


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
    for rid in readings:
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
