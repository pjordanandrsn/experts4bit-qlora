# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""HO2, the planner's first held-out training read (bench/ho2/PREREG-ho2.md): the reducer's self-test and the mutants it must
kill; the staged pins; the runner's pins, data, install and tripwire against the registration; the run IDs across runner,
driver and registration; the fit inside the guard."""
import hashlib
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "ho2"
PREREG = (LANE / "PREREG-ho2.md").read_text(encoding="utf-8")
RUN = (LANE / "ho2_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "ho2_drive.sh").read_text(encoding="utf-8")
REDUCE = (LANE / "ho2_reduce.py").read_text(encoding="utf-8")
SOURCES = {"ho2_run.sh": LANE, "ho2_reduce.py": LANE, "phases.sh": REPO / "bench" / "common"}
GUARD_S, BOOT_S, HEADROOM = 7200, 300, 1.25


def _var(text, name):
    m = re.search(rf"(?:^|[;\s]){name}=([^\s;]+)", text, re.M)
    assert m, name
    return m.group(1)


def _self_test(path):
    return subprocess.run([sys.executable, str(path), "--self-test"], capture_output=True, text=True, timeout=120, cwd=str(path.parent))


def test_reducer_self_test_passes():
    r = _self_test(LANE / "ho2_reduce.py")
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


MUTANTS = [
    ('d.get("tokens") == STEPS * MICRO_BATCH * seq', "True"),                                  # an early stop reads as a row
    ('gpu.get("name") == CARDS[card]', "True"),                                                # another card's receipt
    ('all(ln.get("basis") == "inferred" for ln in over)', "True"),                             # receipts on file
    ("under, short = driver_peak > plan_total, allocated_peak > estimate", "under, short = False, False"),
    ('if len(found) != 1:', "if not found:"),                                                  # two receipts for one row
]


@pytest.mark.parametrize("i", range(len(MUTANTS)))
def test_self_test_kills_the_mutants(i, tmp_path):
    old, new = MUTANTS[i]
    assert REDUCE.count(old) == 1, old
    (tmp_path / "ho2_reduce.py").write_text(REDUCE.replace(old, new), encoding="utf-8")
    assert _self_test(tmp_path / "ho2_reduce.py").returncode != 0, f"mutant survived: {old!r}"


def test_staged_pins_match_their_sources():
    lines = [ln.split() for ln in (LANE / "staged-ho2.sha256").read_text(encoding="ascii").splitlines() if ln.strip()]
    assert sorted(n for _, n in lines) == sorted(SOURCES)
    for want, name in lines:
        assert hashlib.sha256((SOURCES[name] / name).read_bytes()).hexdigest() == want, name


def test_runner_pins_are_the_registered_ones():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert f"grouped-nf4-gemm.git@{_var(RUN, 'GNF4_SHA')}" in ci                       # e4b CI's pin
    assert _var(RUN, "ALPACA_REV") == "dce01c9b08f87459cf36a430d809084718273017" and _var(RUN, "ALPACA_REV") in PREREG
    for fam, rev in (("LFM2", "c1c44ff9fc00db3ebf4516970563f5f383d23670"), ("ERNIE", "87db95487941cb39592ee0abca3b9155a6d19c5c")):
        assert _var(RUN, f"{fam}_REV") == rev and rev in PREREG and rev in REDUCE
    assert "--format text --text-field text" in RUN and "--micro-batch 2 --steps 20" in RUN
    cmd = RUN[RUN.index("python -m loggetta train"):RUN.index("--adapter-out")]
    assert "--fix" not in cmd and "--observations" not in cmd                          # the planner's own plan, nothing on file


def test_loggetta_is_a_pinned_release_commit():
    sha = _var(RUN, "LOGGETTA_SHA")
    assert re.fullmatch(r"[0-9a-f]{40}", sha), f"LOGGETTA_SHA is {sha!r}: fill in loggetta 0.6.2's release commit before registering"


def test_install_carries_datasets_and_the_tripwire_imports_the_train_path_before_the_fetch():
    pip_line = RUN[RUN.index("pipx logs/pip_e4b.log"):RUN.index("PIP FAIL (e4b)")]
    assert '"datasets>=2.14"' in pip_line and '"transformers==5.17.0"' in pip_line
    i_trip = RUN.index("import torch, datasets")
    assert "import loggetta.data, loggetta.backends.experts4bit_train" in RUN[i_trip:i_trip + 300]
    assert i_trip < RUN.index("phase_start fetch")


def test_run_ids_agree_across_runner_driver_and_prereg():
    for rid in [f"ho2-{card}-{i}" for card in ("5090", "4090") for i in (1, 2, 3)]:
        assert rid in RUN and rid in DRIVE
    assert "`ho2-5090-1`" in PREREG and "`ho2-4090-1`" in PREREG


def test_fit_inside_the_guard():
    n = {k: int(v) for k, v in re.findall(r"\b(INSTALL_TO_GATE_S|NEED_FETCH|NEED_TRAIN|NEED_REDUCE|RESERVE)=(\d+)", RUN)}
    fit = n["INSTALL_TO_GATE_S"] + n["NEED_FETCH"] + 4 * n["NEED_TRAIN"] + n["NEED_REDUCE"] + n["RESERVE"]
    assert fit == 3930 and "3,930 s" in PREREG
    assert BOOT_S + fit * HEADROOM <= GUARD_S


def test_the_driver_never_fetches_tensors():
    assert "--exclude 'run/adapter_*'" in DRIVE and "--exclude '*.safetensors'" in DRIVE
