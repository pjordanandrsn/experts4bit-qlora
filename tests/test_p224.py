# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P224, one trained attention adapter on serve_paged priced against its reference path (bench/p224/PREREG-p224.md).
The canary data's pin and its splits; the three self-tests, and the mutants of the rule they must kill; the fit table
against the guard with 25 % headroom; the registration's constants as the reducer carries them. (Private names are
the repository guards' job, not this file's.)"""
import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p224"
PREREG = (LANE / "PREREG-p224.md").read_text(encoding="utf-8")
CANARY_SHA256 = "ed67a18fde1fea23c70ca00c9769ce23cf2399633fa24a89c0bffd431e8b37b0"
GUARD_S = 9000        # 2.5 h
BOOT_S = 300
HEADROOM = 1.25
FIT = {"install": 450, "fetch": 1200, "bake": 600, "train": 2 * 360, "ref": 2 * 240, "servers": 7 * 360,
       "offload_test": 60, "reduce": 60, "reserve": 600}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path.insert(0, str(LANE))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(LANE))
    return mod


def _self_test(path):
    return subprocess.run([sys.executable, str(path), "--self-test"], capture_output=True, text=True, timeout=120,
                          cwd=str(path.parent))


@pytest.mark.parametrize("name", ["p224_canary", "p224_reduce", "p224_box"])
def test_self_tests_pass(name):
    r = _self_test(LANE / f"{name}.py")
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


def test_canary_pin_is_the_registered_one():
    c = _load("p224_canary")
    assert c.canary_sha256() == CANARY_SHA256
    d = c.build()
    assert (len(d["train"]), len(d["heldout"]), len(d["http"]["triggers"])) == (512, 64, 32)


def test_reducer_constants_are_the_registration():
    R = _load("p224_reduce")
    assert (R.N_HELDOUT, R.RATIO_BAR, R.FLOOR, R.CAP, R.FAITHFUL, R.SE_FRACTION, R.MIN_REF_GAIN, R.VRAM_TOL) == \
        (64, 0.95, 0.01, 0.05, 0.025, 0.2, 0.05, 0.01)
    assert R.EXPECTED_ATTN_PROJECTIONS == {"Qwen/Qwen3-30B-A3B": 192} and R.ABBA_BLOCKS == 2 and R.BAND_FLOOR == 0.01
    assert R.N_HELDOUT == _load("p224_canary").N_HELDOUT
    for text in ("0.95 of REF's", "min(0.05, max(0.01, 2 × |loss(P0) − loss(HF0)|))", "> 0.025 nats", "0.2 × 0.01",
                 "less than 0.05 nats", "/ P0 > 1 %", "192 for Qwen3-30B-A3B",
                 "1 % of the mean of the four A slots", "G3 read at the 1% floor", "(B = 1)", "unreleased, the 0.6.2 candidate"):
        assert text in PREREG, text


def test_box_slots_are_two_abba_blocks_then_pa_dq():
    b = _load("p224_box")
    assert [(t, a) for t, a, _ in b.SLOTS] == [("b0A1", "P0"), ("b0B", "PA"), ("b0A2", "P0"), ("b1A1", "P0"),
                                              ("b1B", "PA"), ("b1A2", "P0"), ("dq", "PA-DQ")]
    assert {a for _, a, s in b.SLOTS if s} == {"P0", "PA", "PA-DQ"}
    assert b.DQ_VALUE == {"REF": "0", "REF-DQ": None} and b.ACCEPT_VALUE == "experts"


def test_fit_inside_the_guard():
    total = BOOT_S + sum(FIT.values())
    assert total == 6990 and "**6,990**" in PREREG
    assert total * HEADROOM <= GUARD_S, (total, total * HEADROOM, GUARD_S)
    assert (total - BOOT_S) * HEADROOM == pytest.approx(8362.5) and "8,363 s" in PREREG
    assert "8,738 s" in PREREG and "(9,000 s)" in PREREG


SHAKE_GUARD_S = 6120   # 1.7 h (Amendment 1)
SHAKE_FIT = {"install": 450, "fetch": 300, "bake": 200, "train": 2 * 200, "ref": 2 * 120, "servers": 7 * 150,
             "offload_test": 60, "reduce": 60, "fixtures": 1000, "reserve": 600}
AMENDMENT_1 = PREREG.split("## Amendment 1", 1)[1] if "## Amendment 1" in PREREG else ""


def test_shakedown_fit_with_the_fixtures_inside_its_guard():
    """Amendment 1's table: the fixtures term (two trains on the pinned data, fixture-base, the fetch) against 1.7 h."""
    total = BOOT_S + sum(SHAKE_FIT.values())
    assert total == 4660 and "**4,660**" in AMENDMENT_1 and "5,450 s" in AMENDMENT_1 and "5,825 s" in AMENDMENT_1
    assert total * HEADROOM <= SHAKE_GUARD_S and "(6,120 s)" in AMENDMENT_1 and "**1.7 h guard**" in AMENDMENT_1
    assert (total - BOOT_S) * HEADROOM == 5450
    assert round(1.7 * 0.18 + (14 + 5 + 3.4) * 0.011, 2) == 0.55 and "**$0.55**" in AMENDMENT_1
    assert 2 * 410 + 180 == SHAKE_FIT["fixtures"]       # the runner's per-fixture can_run (410 s) and fixture-base's


def test_amendment_1_pins_the_fixture_data_and_declares_fixture_base():
    for text in ("--dataset tatsu-lab/alpaca --dataset-revision dce01c9b08f87459cf36a430d809084718273017 --format text "
                 "--text-field text", "958 rows, 102,400 tokens", "row 284", "fixture-base/stacks.safetensors",
                 "fixture-base/absmax_dq.safetensors", "`correctness.frozen_expert_digest`", "refuses fixture-base only"):
        assert text in AMENDMENT_1, text
    for marker in ("**The trains** (the data and the invocation superseded by Amendment 1)",
                   "**The fit** (superseded by Amendment 1)", "**Worst case** (superseded by Amendment 1)"):
        assert marker in PREREG, marker


def test_fixture_generation_is_declared_and_never_graded():
    sec = PREREG.split("## Fixture generation", 1)[1].split("\n## ", 1)[0]
    for text in ("NEVER affect its verdict", "--steps 200 --seed 42", "`fixture-dq`", "`fixture-nodq` at `E4B_ABSMAX_DQ=0`",
                 "`resolved_model_revision` equals", "`double-quantized`", "--fix expert_residency=device",
                 "`grouped_nf4`", "never a P224 record",
                 "7f1c97f440f06ce36705e4f2b843edb5925f4498"):
        assert text in sec, text


MUTANTS = [
    ("p224_reduce", "    if ratio is None or ratio < RATIO_BAR:", "    if ratio is None:"),
    ("p224_reduce", "    if gap > FAITHFUL:", "    if gap > 1.0:"),
    ("p224_reduce", "    if None in shas.values() or len(set(shas.values())) != 1:", "    if False:"),
    ("p224_reduce", "    if thr and not shake:", "    if thr:"),
    ("p224_reduce", "                if dev > band[m][w]:", "                if dev > 2 * band[m][w]:"),
    ("p224_reduce", "    if ident.get(\"base_name_status\") != 404:", "    if False:"),
    ("p224_reduce", "    band = {m: {w: max(raw[m][w], floor[m][w]) for w in COST_WS} for m in COST_METRICS}",
     "    band = raw"),
    ("p224_canary", "SPLIT = {\"train\": TOPICS[:32], \"heldout\": TOPICS[32:40], \"http\": TOPICS[40:48]}",
     "SPLIT = {\"train\": TOPICS[:33], \"heldout\": TOPICS[32:40], \"http\": TOPICS[40:48]}"),
    ("p224_box", "    test = (\"passed\" if line.startswith(\"1 passed\") and \"skipped\" not in line",
     "    test = (\"passed\" if line"),
    ("p224_box", "    same = all(len({by_mode[m][i] for m in modes}) == 1 for i in range(n))", "    same = True"),
    ("p224_box", "        if env.get(DQ_ENV) is not None:", "        if False:"),
]


@pytest.mark.parametrize("i", range(len(MUTANTS)))
def test_self_tests_kill_the_mutants(i, tmp_path):
    name, old, new = MUTANTS[i]
    for f in LANE.glob("p224_*.py"):
        (tmp_path / f.name).write_bytes(f.read_bytes())
    src = (tmp_path / f"{name}.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, f"mutant {i} does not apply: {old!r}"
    (tmp_path / f"{name}.py").write_text(src.replace(old, new), encoding="utf-8", newline="\n")
    r = _self_test(tmp_path / f"{name}.py")
    assert r.returncode != 0, f"mutant {i} survived {name}'s self-test: {r.stdout}"

