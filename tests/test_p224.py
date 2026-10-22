# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P224, one trained attention adapter on serve_paged priced against its reference path (bench/p224/PREREG-p224.md).
The canary data's pin and its splits; the three self-tests, and the mutants of the rule they must kill; the fit table
against the guard with 25 % headroom; the registration's constants as the reducer carries them. (Private names are
the repository guards' job, not this file's.)"""
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p224"
PREREG = (LANE / "PREREG-p224.md").read_text(encoding="utf-8")
CANARY_SHA256 = "287706ba066909f83c63a77bd552bc64f7fbf080c0b90fc07ea6cd4ae456efe7"   # Amendment 5
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
    assert (len(d["train"]), len(d["heldout"]), len(d["http"]["triggers"])) == (512, 256, 32)   # Amendment 5


def test_reducer_constants_are_the_registration():
    R = _load("p224_reduce")
    assert (R.N_HELDOUT, R.RATIO_BAR, R.FLOOR, R.CAP, R.FAITHFUL, R.SE_FRACTION, R.MIN_REF_GAIN, R.VRAM_TOL) == \
        (256, 0.95, 0.01, 0.05, 0.025, 0.2, 0.05, 0.01)                                     # Amendment 5: N 256
    assert R.EXPECTED_ATTN_PROJECTIONS == {"Qwen/Qwen3-30B-A3B": 192} and R.ABBA_BLOCKS == 2 and R.BAND_FLOOR == 0.01
    assert R.N_HELDOUT == _load("p224_canary").N_HELDOUT
    for text in ("0.95 of REF's", "min(0.05, max(0.01, 2 × |loss(P0) − loss(HF0)|))", "> 0.025 nats", "0.2 × 0.01",
                 "less than 0.05 nats", "/ P0 > 1 %", "192 for Qwen3-30B-A3B",
                 "1 % of the mean of the four A slots", "G3 read at the 1% floor", "(B = 1)", "unreleased, the 0.6.2 candidate"):
        assert text in PREREG, text


def test_box_slots_are_two_abba_blocks():
    b = _load("p224_box")
    assert [(t, a) for t, a, _ in b.SLOTS] == [("b0A1", "P0"), ("b0B", "PA"), ("b0A2", "P0"), ("b1A1", "P0"),
                                              ("b1B", "PA"), ("b1A2", "P0")]                   # Amendment 3: six
    assert {a for _, a, s in b.SLOTS if s} == {"P0", "PA"}
    assert b.DQ_VALUE == {"REF": "0"} and b.ACCEPT_VALUE == "experts"


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
    ("p224_reduce", "    if a[\"ratio\"] is None or a[\"ratio\"] < RATIO_BAR:", "    if a[\"ratio\"] is None:"),
    ("p224_reduce", "    if gap > FAITHFUL:", "    if gap > 1.0:"),
    ("p224_reduce", "    if None in shas.values() or len(set(shas.values())) != 1:", "    if False:"),
    ("p224_reduce", "    if thr and not shake:", "    if thr:"),
    ("p224_reduce", "                if dev > band[m][w]:", "                if dev > 2 * band[m][w]:"),
    ("p224_reduce", "    if ident.get(\"base_name_status\") != 404:", "    if False:"),
    ("p224_reduce", "            band.setdefault(m, {})[w] = max(r, f)", "            band.setdefault(m, {})[w] = r"),
    ("p224_canary", "SPLIT = {\"train\": TOPICS[:32], \"heldout\": TOPICS[32:40] + TOPICS[48:72], \"http\": TOPICS[40:48]}",
     "SPLIT = {\"train\": TOPICS[:33], \"heldout\": TOPICS[32:40] + TOPICS[48:72], \"http\": TOPICS[40:48]}"),
    # Amendment 5
    ("p224_reduce", "    elif keys and len(set(keys)) != len(keys):", "    elif False:"),
    ("p224_reduce", "    if meta.get(\"heldout_train_overlap\") != 0:", "    if False:"),
    ("p224_reduce", "    a = st[\"assistant\"]", "    a = st[\"all\"]"),
    ("p224_reduce", "GRADED_COST = {(\"ttft_any_ms\", \"W1\"), (\"itl_wall_ms\", \"W1\"), (\"tok_s\", \"W1\")}",
     "GRADED_COST = {(m, w) for m in COST_METRICS for w in COST_WS}"),
    ("p224_reduce", "        if k not in seen:", "        if True:"),
    ("p224_reduce", "            \"flag\": abs(x[\"did\"]) > DID_SE_FLAG * x[\"se_did\"] and abs(x[\"did\"]) > margin}",
     "            \"flag\": False}"),
    ("p224_box", "(r[4] - r[3]) / (r[2] - 1)", "(r[4] - r[3]) / r[2]"),
    ("p224_box", "                    heldout_train_overlap=sum(h in train for h in held))",
     "                    heldout_train_overlap=0)"),
    ("p224_canary", "for t in SPLIT[\"heldout\"] for j in range(4) for trig in (True, False)]",
     "for t in SPLIT[\"heldout\"] for j in range(3) for trig in (True, False)]"),
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
           "fixture_base.py": LANE, "phases.sh": REPO / "bench" / "common",
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


def test_run_ids_agree_across_runner_driver_and_prereg():
    ids = [f"p224-5090-{i}" for i in (1, 2, 3)] + [f"p224-shake-{i}" for i in (1, 2, 3, 4, 5, 6)]   # -4..-6: Amendment 4
    for rid in ids:
        assert rid in RUN and rid in DRIVE
    assert "`p224-shake-1..3`" in PREREG and "`p224-5090-1`" in PREREG


def test_the_fixture_base_step_is_after_the_trains_and_never_changes_the_verdict():
    i_reduce = RUN.index("p224_reduce.py --dir")
    i_trains = RUN.index('for FX in dq nodq; do')
    i_base = RUN.index("fixture_base.py --plan")
    i_fail = RUN.index('[ "$V" != NONE ] || { say "REDUCER FAILED"; finish 22; }')
    i_finish = RUN.rindex('case "$SHAKE/$V" in')
    assert i_reduce < i_trains < i_base < i_fail < i_finish        # Amendment 3: the fixtures run whatever the reduce did
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
                 "`--fix expert_kernel=grouped_nf4`", "WHATEVER the reduce outcome",
                 "--seq 512 --micro-batch 1 --steps 200 --seed 42 --fix expert_residency=device --fix expert_kernel=grouped_nf4"):
        assert text in AMENDMENT_3, text
    for marker in ("canary B: trained at the default (dropped by Amendment 3)", "(superseded by Amendment 3: six processes)",
                   "- **The fit** (superseded by Amendment 3)", "superseded by Amendment 3: `bake_nf4`"):
        assert marker in PREREG, marker


# ---------------------------------------------------------------------------------------------- Amendment 3's kit

def _joined_lines(text):
    """The runner's lines with backslash continuations joined."""
    out, cur = [], ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            cur += line.rstrip()[:-1] + " "
            continue
        out.append(cur + line)
        cur = ""
    return out


def _calls(script):
    """Every ``python $W/<script> ...`` call in the runner, as an argv: $-variables become placeholders, the optional
    ``$([ ... ] && echo --x)`` becomes ``--x``, redirections and pipes are cut."""
    import re
    import shlex
    calls = []
    for line in _joined_lines(RUN):
        m = re.search(r"python \$W/" + re.escape(script) + r"(?P<rest>.*)", line)
        if not m:
            continue
        rest = re.sub(r"\$\(\[[^)]*&& echo (--[a-z-]+)\)", r"\1", m["rest"])
        rest = re.split(r"\s(?:\d?>|>>|\||;|&&|\|\|)", " " + rest)[0]
        for var, val in (("$ARM", "PA"), ("$arm", "REF"), ("$TAG", "b0B")):    # the runner's arm and slot variables
            rest = rest.replace(var, val)
        argv = [re.sub(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", "X", tok) for tok in shlex.split(rest)]
        calls.append(argv)
    return calls


@pytest.mark.parametrize("script", ["p224_box.py", "p224_canary.py", "p224_reduce.py"])
def test_every_runner_call_passes_the_scripts_own_argument_check(script):
    """Every call the runner makes is run through the script's own --check-args, which applies the same checks as a
    real run (argparse plus the after-parse rules, such as the reducer's --pins) and does no work. p224-shake-1's
    reduce call without --pins fails here."""
    calls = _calls(script)
    assert calls, script
    for argv in calls:
        if "--self-test" in argv:
            continue
        r = subprocess.run([sys.executable, str(LANE / script), *argv, "--check-args"], capture_output=True, text=True,
                           timeout=60, cwd=str(LANE))
        assert r.returncode == 0 and "ARGS OK" in r.stdout, (argv, r.stdout + r.stderr)


def test_a_reduce_call_without_pins_fails_the_argument_check():
    r = subprocess.run([sys.executable, str(LANE / "p224_reduce.py"), "--dir", "X", "--out", "X", "--check-args"],
                       capture_output=True, text=True, timeout=60, cwd=str(LANE))
    assert r.returncode == 2 and "--pins" in r.stderr


def test_the_fixture_base_call_carries_every_flag_its_parser_requires():
    """fixture_base.py is staged at the fixture owner's bytes (no --check-args); its four flags are required=True."""
    import re
    req = re.findall(r'"(--[a-z-]+)"', (LANE / "fixture_base.py").read_text(encoding="utf-8").split("for k in (", 1)[1]
                     .split(")", 1)[0])
    assert req == ["--plan", "--receipt-nodq", "--receipt-dq", "--out"]
    (argv,) = _calls("fixture_base.py")
    assert all(f in argv for f in req), argv


def test_only_the_fixture_trains_take_the_fused_kernel():
    trains = [ln for ln in _joined_lines(RUN) if "python -m loggetta train" in ln]
    assert len(trains) == 2, trains
    canary = next(t for t in trains if "$W/run/data/train.jsonl" in t)
    fixture = next(t for t in trains if "tatsu-lab/alpaca" in t)
    assert "expert_kernel" not in canary and "--fix train_experts=false" in canary
    assert "--fix expert_kernel=grouped_nf4" in fixture


def test_the_fixture_train_is_the_registered_command():
    """Amendment 3's authoritative fixture command, token for token, with the runner's variables resolved."""
    import re
    import shlex
    block = AMENDMENT_3.split("The authoritative fixture command", 1)[1].split("```", 2)[1]
    want = shlex.split(" ".join(ln.strip() for ln in block.strip().splitlines()))
    line = next(ln for ln in _joined_lines(RUN) if "python -m loggetta train" in ln and "tatsu-lab/alpaca" in ln)
    line = line.split("python -m loggetta train", 1)[1].split(">", 1)[0]
    subs = {"$FIXTURE_MODEL": "allenai/OLMoE-1B-7B-0924-Instruct", "$FIXTURE_REV": _var(RUN, "FIXTURE_REV"),
            "$ALPACA_REV": _var(RUN, "ALPACA_REV"), "$W/fx/": "", "$FX": "<form>"}
    for k, v in subs.items():
        line = line.replace('"' + k + '"', v).replace(k, v)
    got = ["python", "-m", "loggetta", "train", *shlex.split(line)]
    assert got == want, (got, want)
    assert re.search(r"E4B_ABSMAX_DQ=0", RUN.split("for FX in dq nodq; do", 1)[1].split("python -m loggetta", 1)[0])


def test_the_arena_is_baked_by_bake_nf4_and_checked_before_any_server():
    i_bake = RUN.index("from nvme_bake_nf4 import bake_nf4; bake_nf4('$SNAP', '$W/work/nf4.arena')")
    i_check = RUN.index("from experts4bit_qlora.adapter_dir import arena_source")
    i_halt = RUN.index('[ "$ARC" = 0 ] || {')
    i_slot = RUN.index("for spec in b0A1:P0:1")
    assert i_bake < i_check < i_halt < i_slot and "k8_bake" not in RUN
    assert '("nf4", 64)' in RUN and "import nvme_bake_nf4" in RUN


def test_the_paged_refusals_pass_on_an_index_a_real_bake_nf4_writes(tmp_path):
    """Refusals 3 and 4 on the index grouped-nf4-gemm's bake_nf4 ACTUALLY writes (a tiny MoE in an HF-cache-shaped
    snapshot, quantized on the CPU through the bake's own quantizer), never a hand-built dict; and a relocation bake's
    index (a local snapshot path, no quantizer) is refused on both, as p224-shake-1's was."""
    import json
    import types
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    st = pytest.importorskip("safetensors.torch")
    nb = pytest.importorskip("nvme_bake_nf4")
    from experts4bit_qlora import adapter_dir as AD
    rev = "c" * 40
    snap = tmp_path / "models--org--tiny-moe" / "snapshots" / rev
    snap.mkdir(parents=True)
    g, tensors, wm = torch.Generator().manual_seed(0), {}, {}
    for li in range(2):
        for e in range(2):
            for proj, shape in (("gate_proj", (64, 128)), ("up_proj", (64, 128)), ("down_proj", (128, 64))):
                name = f"model.layers.{li}.mlp.experts.{e}.{proj}.weight"
                tensors[name] = torch.randn(*shape, generator=g).to(torch.bfloat16)
                wm[name] = "model-00001-of-00001.safetensors"
    st.save_file(tensors, str(snap / "model-00001-of-00001.safetensors"))
    (snap / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}), encoding="utf-8")
    out = str(tmp_path / "nf4.arena")
    nb.bake_nf4(str(snap), out, quantize_fn=nb.default_quantize_expert(dev="cpu"), log=lambda *a, **k: None)
    index = json.loads(open(out + ".index.json", encoding="utf-8").read())
    assert AD.arena_source(index) == ("org/tiny-moe", rev)
    ad = types.SimpleNamespace(revision=rev, setup={"quant_type": "nf4", "blocksize": 64, "attn_4bit": False,
                                                    "train_experts": False}, attention={}, expert_tensors=(),
                               other_tensors=(), tensors={}, manifest={})
    kw = dict(served_revision=rev, requested_revision=None, levers={}, served_model="org/tiny-moe",
              accept_unverified_base=frozenset({"experts"}))
    assert AD.servable_refusals(ad, arena_index=index, **kw) == []
    reloc = {k: v for k, v in index.items() if k != "quantizer"}
    reloc["snapshot"] = str(tmp_path / "nf4snap")
    assert sorted({c for c, _ in AD.servable_refusals(ad, arena_index=reloc, **kw)}) == [3, 4]


def test_amendment_4_registers_three_spare_shake_ids_and_nothing_else():
    a4 = PREREG.split("## Amendment 4", 1)[1]
    for text in ("`p224-shake-4`, `-5` and `-6`", "never a reading and never a strike", "the L4 first",
                 "at least half", "the id is kept", "$1.0064"):
        assert text in a4, text
    assert "spares `-4` to `-6` added by Amendment 4" in PREREG
    for rid in ("p224-shake-7", "p224-5090-4"):          # still unregistered: both scripts refuse them
        assert rid not in RUN and rid not in DRIVE, rid


# ------------------------------------------------------------------------------------------------------------- Amendment 5

AMENDMENT_5 = PREREG.split("## Amendment 5", 1)[1] if "## Amendment 5" in PREREG else ""
FIT_A5 = {"install": 450, "fetch": 1200, "bake": 600, "train": 360, "ref": 720, "servers": 6 * 360, "score": 2 * 240,
          "offload_test": 60, "reduce": 60, "reserve": 600}
SHAKE_FIT_A5 = {"install": 450, "fetch": 300, "bake": 200, "train": 360, "ref": 600, "servers": 6 * 150,
                "score": 2 * 260, "offload_test": 60, "reduce": 60, "fixtures": 1000, "reserve": 600}
SHAKE3_CELLS_SHA256 = "77e4b12049c4f0e55b47aae75f522e67724dd30ff40b9db8152bfbd764112caf"


def _need(block):
    env = {}
    for part in block.replace("\n", ";").split(";"):
        part = part.strip()
        if part.startswith("NEED_") and "=" in part and "$((" not in part:
            k, v = part.split("=", 1)
            env[k] = int(v.split()[0])
    return env


def test_amendment_5_the_runner_fit_is_the_registered_table_inside_both_guards():
    shake_block = RUN.split('if [ "$SHAKE" = 1 ]; then', 1)[1].split("else", 1)[0]
    read_block = RUN.split('if [ "$SHAKE" = 1 ]; then', 1)[1].split("else", 1)[1].split("fi", 1)[0]
    assert "6 * NEED_SLOT + 2 * NEED_SCORE" in RUN and 'can_run $((NEED_SLOT + SCORE * NEED_SCORE)) "slot $TAG"' in RUN
    for block, table, guard in ((read_block, FIT_A5, GUARD_S), (shake_block, SHAKE_FIT_A5, 1.9 * 3600)):
        n = {**_need(block), "NEED_GATE": 60, "NEED_REDUCE": 60}
        work = (n["NEED_FETCH"] + n["NEED_BAKE"] + n["NEED_TRAIN"] + n["NEED_REF"] + 6 * n["NEED_SLOT"]
                + 2 * n["NEED_SCORE"] + n["NEED_GATE"] + n["NEED_REDUCE"] + n["NEED_FIXTURES"])
        assert 450 + work + 600 == sum(table.values()), (n, table)
        assert BOOT_S + sum(table.values()) * HEADROOM <= guard, (table, guard)
    assert sum(FIT_A5.values()) == 6690 and sum(SHAKE_FIT_A5.values()) == 5050
    for text in ("**6,690**", "8,363 s", "8,663 s", "**5,050**", "6,313 s", "6,613 s", "alarm (1,200 s)"):
        assert text in AMENDMENT_5, text


def test_amendment_5_the_held_out_set_is_distinct_disjoint_and_the_training_unchanged():
    import hashlib
    c = _load("p224_canary")
    d = c.build()
    held = [c.row_sha256(r) for r in d["heldout"]]
    assert len(held) == len(set(held)) == 256 == c.N_HELDOUT
    assert not set(held) & {c.row_sha256(r) for r in d["train"]}
    assert set(c.TOPICS[32:40]) <= set(c.SPLIT["heldout"]) and len(c.SPLIT["heldout"]) == 32
    assert not set(c.SPLIT["heldout"]) & (set(c.SPLIT["train"]) | set(c.SPLIT["http"]))
    fs = c.files(d)
    assert hashlib.sha256(fs["train.jsonl"]).hexdigest() == c.A3_TRAIN_SHA256       # the canary trains on the same rows
    assert hashlib.sha256(fs["http.json"]).hexdigest() == c.A3_HTTP_SHA256
    assert sum(r["trigger"] for r in d["heldout"]) == 128


def _shake3():
    import hashlib
    raw = (LANE / "shake3_cells.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHAKE3_CELLS_SHA256          # the real records, byte-exact
    return json.loads(raw)


def test_amendment_5_regrades_p224_shake_3_real_cells():
    """p224-shake-3's own per-sequence cells under the amended rule: 46 distinct of 64 rows; the graded (assistant) G1
    passes; the SE is too large on the distinct rows (so a reading at this size would VOID); the all-token G1 passes
    over distinct rows; the assistant DiD is flagged (the served adapter applies about 0.5 % more than the reference)."""
    R = _load("p224_reduce")
    fx = _shake3()
    ls = R.loss_stats(fx["per_seq_loss"], fx["per_seq_loss_assistant"], fx["row_keys"])
    al, a = ls["sets"]["all"], ls["sets"]["assistant"]
    assert (ls["rows"], ls["distinct"]) == (64, 46)
    assert round(al["base_gap"], 4) == 0.0107 and round(al["pa_minus_ref"], 4) == 0.0161 and round(ls["margin"], 4) == 0.0214
    assert round(al["did"], 4) == 0.0054 and round(al["se_did"], 4) == 0.0080
    assert round(a["pa_minus_ref"], 4) == -0.0062 and round(a["se_pa_ref"], 5) == 0.00255 and round(a["ratio"], 4) == 1.0060
    assert round(a["se_pa_ref_every_row"], 5) == 0.00219            # what the registered SE read (duplicates counted)
    assert round(a["did"], 4) == -0.0316 and round(a["se_did"], 4) == 0.0114
    assert ls["fails"] == []                                         # graded G1 passes on the assistant positions
    assert ls["reported"]["all_token_g1"]["would_fail"] == []        # ... and the all-token G1 passes over distinct rows
    assert ls["reported"]["did_assistant"]["flag"] is True and ls["reported"]["did_all"]["flag"] is False
    assert ls["reported"]["over_application_assistant"]["flag"] is False      # 1.006 < 1 / 0.95
    assert a["se_pa_ref"] >= R.SE_FRACTION * R.FLOOR                  # too small to judge at 64 rows


def test_amendment_5_regrades_p224_shake_3_cost_on_the_per_token_wall():
    """W1's per-token wall time from the records ((tokens / tok_s / requests - TTFT) / (tokens per request - 1)): P0 and
    PA equal, inside the band; the text-cadence ITL (the old graded cell) is outside it, and at W16 tok/s PA is faster:
    both REPORTED."""
    R = _load("p224_reduce")
    fx = _shake3()
    def per_token(c):
        req = c["tokens"] / c["requests"]
        return (c["tokens"] / c["tok_s"] / c["requests"] * 1000 - c["ttft_ms"]) / (req - 1)
    cells = {t: dict(fx["slots"][t]) for t in fx["slots"]}
    for t, c in cells.items():
        c["W1"] = dict(c["W1"], itl_wall_ms=per_token(c["W1"]))
    p0 = [cells[t]["W1"]["itl_wall_ms"] for t in ("b0A1", "b0A2", "b1A1", "b1A2")]
    pa = [cells[t]["W1"]["itl_wall_ms"] for t in ("b0B", "b1B")]
    assert round(sum(p0) / 4, 3) == 10.883 and round(sum(pa) / 2, 3) == 10.858
    blocks = [{s: {m: {w: cells[t][w].get(m) for w in R.COST_WS} for m in ("itl_wall_ms", "tok_s", "itl_ms")}
               for s, t in (("A1", f"b{k}A1"), ("B", f"b{k}B"), ("A2", f"b{k}A2"))} for k in (0, 1)]
    for b in blocks:                                              # per-token wall is a W1 cell here
        for s in b.values():
            s["itl_wall_ms"]["W16"] = s["itl_wall_ms"]["W1"]
    cg = R.cost_grade(blocks, metrics=("itl_wall_ms", "tok_s", "itl_ms"))
    assert not [f for f in cg["fails"] if "itl_wall_ms W1" in f or "tok_s W1" in f], cg["fails"]
    assert any("itl_ms W1" in w for w in cg["reported_outside_band"])          # the text cadence, reported
    assert any("tok_s W16" in w for w in cg["reported_outside_band"])          # PA faster at W16, reported


def test_amendment_5_rule_text():
    for text in ("ASSISTANT positions", "not established at n=64", "46 distinct", "0.00255", "0.00219",
                 "about 1 %", "first chunk of any kind", "(wall − first chunk) / (completion tokens − 1)",
                 "287706ba066909f83c63a77bd552bc64f7fbf080c0b90fc07ea6cd4ae456efe7", "a held-out row equal to a training row",
                 "|DiD| > 2 SE", "1 / 0.95"):
        assert text in AMENDMENT_5, text
    assert "falsif" not in AMENDMENT_5.lower()
