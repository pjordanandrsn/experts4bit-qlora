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



# ------------------------------------------------------------------------------------------------------------- the kit

RUN = (LANE / "p224_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p224_drive.sh").read_text(encoding="utf-8")
SOURCES = {"p224_run.sh": LANE, "p224_box.py": LANE, "p224_canary.py": LANE, "p224_reduce.py": LANE,
           "fixture_base.py": LANE, "phases.sh": REPO / "bench" / "common", "k8_bake.py": REPO / "bench" / "p39",
           "calib.json": REPO / "bench" / "p39", "test_adapter_schema.py": REPO / "tests"}


def _var(text, name):
    import re
    m = re.search(rf"^\s*(?:[A-Z_]+=\S+;\s*)*{name}=(\S+?)[;\s]", text, re.M)
    assert m, name
    return m.group(1)


def test_staged_pins_match_their_sources():
    import hashlib
    lines = [ln.split() for ln in (LANE / "staged-p224.sha256").read_text(encoding="ascii").splitlines() if ln.strip()]
    assert sorted(n for _, n in lines) == sorted(SOURCES)
    for want, name in lines:
        got = hashlib.sha256((SOURCES[name] / name).read_bytes()).hexdigest()
        assert got == want, f"{name}: staged {want}, source {got}"


def test_runner_pins_are_the_registered_ones():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert f"grouped-nf4-gemm.git@{_var(RUN, 'GNF4_SHA')}" in ci            # e4b CI's pin
    assert _var(RUN, "LOGGETTA_SHA").startswith("90a7697d") and len(_var(RUN, "LOGGETTA_SHA")) == 40
    assert _var(RUN, "CANARY_SHA256") == CANARY_SHA256
    assert _var(RUN, "FIXTURE_REV") == "7f1c97f440f06ce36705e4f2b843edb5925f4498" and _var(RUN, "FIXTURE_REV") in PREREG
    assert _var(RUN, "ALPACA_REV") == "dce01c9b08f87459cf36a430d809084718273017" and _var(RUN, "ALPACA_REV") in AMENDMENT_1
    assert "REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in RUN and "`ad44e777`" in PREREG
    assert "--format text --text-field text" in RUN and "--fix expert_residency=device" in RUN


def test_runner_terms_are_the_fit_tables():
    def need(block):
        env = {}
        for part in block.replace("\n", ";").split(";"):
            part = part.strip()
            if part.startswith("NEED_") and "=" in part and "$((" not in part:
                k, v = part.split("=", 1)
                env[k] = int(v.split()[0])
        return env
    shake_block = RUN.split('if [ "$SHAKE" = 1 ]; then', 1)[1].split("else", 1)[0]
    read_block = RUN.split('if [ "$SHAKE" = 1 ]; then', 1)[1].split("else", 1)[1].split("fi", 1)[0]
    common = {"NEED_GATE": 60, "NEED_REDUCE": 60}
    assert "NEED_GATE=60; NEED_REDUCE=60; RESERVE=600" in RUN and "INSTALL_TO_GATE_S=450" in RUN
    for block, table in ((read_block, FIT), (shake_block, SHAKE_FIT)):
        n = {**need(block), **common}
        work = (n["NEED_FETCH"] + n["NEED_BAKE"] + 2 * n["NEED_TRAIN"] + 2 * n["NEED_REF"] + 7 * n["NEED_SLOT"]
                + n["NEED_GATE"] + n["NEED_REDUCE"] + n["NEED_FIXTURES"])
        assert BOOT_S + 450 + work + 600 == BOOT_S + sum(table.values()), (n, table)


def test_run_ids_agree_across_runner_driver_and_prereg():
    ids = [f"p224-5090-{i}" for i in (1, 2, 3)] + [f"p224-shake-{i}" for i in (1, 2, 3)]
    for rid in ids:
        assert rid in RUN and rid in DRIVE
    assert "`p224-shake-1..3`" in PREREG and "`p224-5090-1`" in PREREG


def test_the_fixture_base_step_is_after_the_trains_and_never_changes_the_verdict():
    i_reduce = RUN.index("p224_reduce.py --dir")
    i_trains = RUN.index('for FX in dq nodq; do')
    i_base = RUN.index("fixture_base.py --plan")
    i_finish = RUN.rindex('case "$SHAKE/$V" in')
    assert i_reduce < i_trains < i_base < i_finish
    assert "$W/fx/receipts_fixture-nodq" in RUN and "['plan']" in RUN


AMENDMENT_2 = PREREG.split("## Amendment 2", 1)[1] if "## Amendment 2" in PREREG else ""


def test_amendment_2_the_shakedown_card_and_its_guard():
    """The L4 at <= $0.40/h; the fit table unchanged (4,660 s); a 1.9 h guard for the slower card; worst $1.0064."""
    total = BOOT_S + sum(SHAKE_FIT.values())
    assert total == 4660 and total * HEADROOM <= 1.9 * 3600 == 6840
    for text in ("one L4 (sm_89, 24 GB", "most $0.40/h", "**1.9 h** (6,840 s)", "5,825 s, inside the 6,840 s guard",
                 "(4,660 − 300) × 1.25 = 5,450 s", "**$1.0064**", "`phases.jsonl`"):
        assert text in AMENDMENT_2, text
    assert round(1.9 * 0.40 + (14 + 5 + 3.4) * 0.011, 4) == 1.0064
    for marker in ("(the card and the cap superseded by Amendment 2)", "**The guard** (superseded by Amendment 2)",
                   "**Worst case** (superseded by Amendment 2)"):
        assert marker in PREREG, marker


def test_loggetta_requirements_are_installed_and_imported_before_the_fetch():
    """Loggetta goes in --no-deps (its requirements would re-resolve the pinned e4b and grouped-nf4-gemm from PyPI), so
    the runner names datasets itself (the experts4bit path's one other requirement) and the tripwire imports it and the
    train path before any fetch."""
    import re
    i_pip = RUN.index('"datasets>=2.14" ||')
    i_logg = RUN.index("pipx logs/pip_logg.log 900 --no-deps")
    i_trip = RUN.index("import loggetta.data, loggetta.backends.experts4bit_train")
    i_fetch = RUN.index("phase_start fetch setup")
    assert i_pip < i_logg < i_trip < i_fetch
    assert re.search(r"^import datasets\b", RUN, re.M)
    assert '"peft' not in RUN and "import peft" not in RUN     # dense-backend only: P224 never needs it


AMENDMENT_3 = PREREG.split("## Amendment 3", 1)[1] if "## Amendment 3" in PREREG else ""
FIT_A3 = {"install": 450, "fetch": 1200, "bake": 600, "train": 360, "ref": 240, "servers": 6 * 360,
          "offload_test": 60, "reduce": 60, "reserve": 600}
SHAKE_FIT_A3 = {"install": 450, "fetch": 300, "bake": 200, "train": 200, "ref": 120, "servers": 6 * 150,
                "offload_test": 60, "reduce": 60, "fixtures": 1000, "reserve": 600}


def test_amendment_3_fits_inside_the_unchanged_guards():
    reading, shake = BOOT_S + sum(FIT_A3.values()), BOOT_S + sum(SHAKE_FIT_A3.values())
    assert reading == 6030 and "**6,030**" in AMENDMENT_3 and reading * HEADROOM <= GUARD_S
    assert "7,537 s" in AMENDMENT_3 and "7,162 s" in AMENDMENT_3
    assert shake == 4190 and "**4,190**" in AMENDMENT_3 and shake * HEADROOM <= 1.9 * 3600
    assert "5,237 s" in AMENDMENT_3 and "4,862 s" in AMENDMENT_3
    assert "**$3.35**" in AMENDMENT_3 and "**$1.0064**" in AMENDMENT_3


def test_amendment_3_registers_the_bake_the_canary_check_and_the_dropped_arms():
    for text in ("`bake_nf4(SNAP, work/nf4.arena)`", "`arena_source(index)` equals the registered `(MODEL, REV)`",
                 "HARNESS_ERROR before any slot runs", "never on a\n     hand-built dict",
                 "`expert_absmax.form` `fp32` AND the setup's `expert_kernel` `reference`", "A mismatch VOIDs",
                 "REF-DQ and PA-DQ are dropped", "Six fresh server processes", "`pins.json`",
                 "`--fix expert_kernel=grouped_nf4`", "WHATEVER the reduce outcome"):
        assert text in AMENDMENT_3, text
    for marker in ("canary B: trained at the default (dropped by Amendment 3)", "(superseded by Amendment 3: six processes)",
                   "- **The fit** (superseded by Amendment 3)", "superseded by Amendment 3: `bake_nf4`"):
        assert marker in PREREG, marker
