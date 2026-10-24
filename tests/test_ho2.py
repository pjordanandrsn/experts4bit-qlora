# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""HO2, the planner's first held-out training read (bench/ho2/PREREG-ho2.md): the reducer's self-test and the mutants it must
kill; the staged pins; the runner's pins, data, install and tripwire against the registration; the run IDs across runner,
driver and registration; the fit inside the guard; a real Loggetta receipt read through the reducer."""
import hashlib
import importlib
import importlib.util
import json
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
#: a real Loggetta training receipt, paths and GPU uuids redacted (tests/fixtures/ho2/README.md names its source and sha256)
FIXTURE = REPO / "tests" / "fixtures" / "ho2" / "loggetta-train-receipt.json"
FIXTURE_SHA256 = "22ca086ef4461090cebe5b0d45d039858a0e391701541f2f7abf59eeaa0c3903"
#: the fixture's own run, as a registration the reducer reads it against
FIXTURE_RUN = {"families": {"olmoe": ("allenai/OLMoE-1B-7B-0924", None)}, "data": ("tatsu-lab/alpaca", None), "format": "alpaca",
               "steps": 12, "micro_batch": 2, "cards": {"a2000": "NVIDIA RTX A2000 12GB"}}


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
    ('d.get("tokens") == steps * mb * seq', "True"),                                           # an early stop reads as a row
    ('gpu.get("name") == reg["cards"][card]', "True"),                                         # another card's receipt
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


#: HO2 is CLOSED: its read merged at 512614e6 (e4b#1987, RESULTS-ho2.md), where CI's pin and the runner's GNF4_SHA were
#: last asserted equal. A closed lane asserts its own literal pins against that commit, never the live tree, so a later
#: move of CI's pin (e4b#2030) does not re-pin or fail it.
HO2_FROZEN = "512614e6c698fdc0d5ac724ebdfc2f725553113b"


def at_frozen(rel):
    import os

    r = subprocess.run(["git", "-C", str(REPO), "show", f"{HO2_FROZEN}:{rel}"], capture_output=True)
    if r.returncode != 0:
        # CI checks out full history (fetch-depth: 0), so a miss there is a failure, never a skip
        if os.environ.get("CI") == "true":
            pytest.fail(f"HO2's frozen commit {HO2_FROZEN[:12]} is not readable: {r.stderr.decode()[:200]}")
        pytest.skip("HO2's frozen commit is not in this checkout's history (a shallow clone)")
    return r.stdout


def test_runner_pins_are_the_registered_ones():
    ci = at_frozen(".github/workflows/ci.yml").decode("utf-8")
    assert f"grouped-nf4-gemm.git@{_var(RUN, 'GNF4_SHA')}" in ci                       # e4b CI's pin at HO2's close
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
    trip = _tripwire()
    # Loggetta imports these lazily, so importing loggetta's modules alone proves none of them installed
    for name in ("import torch, numpy, bitsandbytes, safetensors.torch", "from datasets import Dataset, load_dataset",
                 "from transformers import AutoTokenizer", "from experts4bit_qlora import ExpertsNbit, offload_handles",
                 "prepare_qlora_training", "estimate_qlora_footprint", "from experts4bit_qlora.arch.topology import describe_moe",
                 "import experts4bit_qlora.lora, nf4_qlora", "from nf4_route import MIN_CAPABILITY, route_for",
                 "loggetta.backends.experts4bit_train"):
        assert name in trip, name
    assert RUN.index(trip) < RUN.index("phase_start fetch")


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


def _tripwire():
    i = RUN.index('say "TRIPWIRE FAIL"')
    return RUN[i:RUN.index("\nPYT\n", i)]


def test_the_tripwires_experts4bit_and_gnf4_names_exist_at_this_checkout():
    pytest.importorskip("torch")
    trip = _tripwire()
    names = re.findall(r"^from ((?:experts4bit_qlora|nf4_route)[\w.]*) import ([\w, ]+)", trip, re.M)
    assert {m for m, _ in names} == {"experts4bit_qlora", "experts4bit_qlora.recipe", "experts4bit_qlora.arch.topology", "nf4_route"}
    for mod, listed in names:
        if mod == "nf4_route" and importlib.util.find_spec("nf4_route") is None:
            continue                                                                    # gnf4 not installed here; CI installs it
        m = importlib.import_module(mod)
        for n in listed.split(","):
            assert hasattr(m, n.strip()), f"{mod}.{n.strip()} is gone: the box would refuse at its tripwire"
    importlib.import_module("experts4bit_qlora.lora")


def _reducer():
    spec = importlib.util.spec_from_file_location("ho2_reduce", LANE / "ho2_reduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _real_receipt():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    return json.loads(raw)


def test_a_real_loggetta_receipt_reads_through_the_reducer():
    red, r = _reducer(), _real_receipt()
    row, why = red.row_of("olmoe_512", r, "a2000", FIXTURE_RUN)
    assert row is not None, why
    sel, cmp = r["plan"]["selected"], r["comparison"]
    # execution.compare: the driver estimate is the plan's device total; the allocator estimate drops the context and reserve lines
    ctx = sum(ln["bytes"] for ln in sel["lines"]
              if ln["where"] == "device" and ln["name"].startswith(("CUDA context", "allocator reserve")))
    assert row["plan_total_bytes"] == sel["device_bytes"] == 7_312_146_022
    assert row["estimate_bytes"] == sel["device_bytes"] - ctx == 5_646_062_592
    assert row["driver_peak_bytes"] == cmp["device_driver"]["measured"] and row["allocated_peak_bytes"] == cmp["device_allocator"]["measured"]
    assert row["outcome"] == "UNDER+SHORT"                         # that run's plan was under on both: by 7 MB and by 209 MB
    assert red._no_receipts_on_file(r)                              # its plan used no receipts on file: every overhead line inferred


def test_a_real_receipt_voids_against_the_registration_and_when_it_learned_from_receipts():
    red, r = _reducer(), _real_receipt()
    row, why = red.row_of("lfm2_1024", r, "5090")
    assert row is None and "model allenai/OLMoE-1B-7B-0924" in why and "card 'NVIDIA RTX A2000 12GB'" in why
    assert red.reduce({t: [r] for t in red.TAGS}, "ho2-5090-1", {"e4b": "a" * 40, "gnf4": "b" * 40, "loggetta": "c" * 40})["verdict"] == "VOID"
    learned = json.loads(json.dumps(r))
    next(ln for ln in learned["plan"]["selected"]["lines"] if ln["name"].startswith("CUDA context"))["basis"] = "measured"
    row, why = red.row_of("olmoe_512", learned, "a2000", FIXTURE_RUN)
    assert row is None and "receipts on file" in why
