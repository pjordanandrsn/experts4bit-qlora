# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P149 (bench/p149/PREREG-p149.md), box Q+S and Amendments 4 and 5: the staged pins, the runner's order (the host
floors and the guard fit before any install, the compiled premises before the fetch, the early gates before any scored
arm, Q in S0a and S1a and its reduce before any other process, T1 and T2 in the rotation), the box's and the dry run's
guard tables against the registration's, the dry mode, the self-tests run as the box runs them (one directory), the
mutants the self-tests must kill, the windows against P147's record, and the driver's run ids."""
import hashlib
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
LANE = B / "p149"
SOURCES = {"p149_run.sh": LANE, "p149_box.py": LANE, "p149_reduce.py": LANE, "p149_slice.py": LANE, "test_dense_i8.py": LANE,
           "test_int8_attn_interp.py": REPO / "tests", "p109_box.py": B / "p109", "p125_box.py": B / "p125",
           "p97_box.py": B / "p97", "p115_quality.py": B / "p115", "p108_box.py": B / "p108", "p110_box.py": B / "p110",
           "k8_bake.py": B / "p39", "calib.json": B / "p39"}
RUN = (LANE / "p149_run.sh").read_text(encoding="utf-8")


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _run(script, *args, cwd=None, env=None):
    """A staged script in its own directory, as the box runs it; the package resolves from this tree when it is not
    installed (the box installs it)."""
    env = dict(os.environ if env is None else env)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=300, cwd=cwd,
                          env=env)


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p149.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_the_runner_and_the_reducer_pin_the_same_grouped_nf4_gemm():
    red = (LANE / "p149_reduce.py").read_text(encoding="utf-8")
    sha = re.search(r"^GNF4_SHA=([0-9a-f]{40})", RUN, re.M).group(1)
    assert f'GNF4_SHA = "{sha}"' in red


def test_the_runner_refuses_before_any_install_and_reads_q_before_any_other_process():
    assert "p149-5090-1|p149-5090-2|p149-5090-3) DRY=0;;" in RUN and "p149-dry-2|p149-dry-3|p149-dry-4) DRY=1;;" in RUN
    assert "MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1; REV=eba92302a2861cdc0098cc54bc9f17cb2c47eb61" in RUN
    assert "MIN_POWER_W=575" in RUN and 'assert triton.__version__ == "3.4.0"' in RUN and "TORCH_FULL\" = 2.8.0+cu128" in RUN
    order = ['[ "${P149_DRY:-0}" = "$DRY" ]', '[ "$PL" -ge "$MIN_POWER_W" ]', "finish 17; }", '[ "$LEFT" -ge "$FIT_NEED" ]',
             "finish 19; }", "pipx(){",
             "pipx logs/pip_e4b.log",
             "python -m pytest test_dense_i8.py", "python -m pytest test_int8_attn_interp.py", "grep -qE '^28 passed",
             "grep -qE '^4 passed", "can_run $NEED_FETCH fetch", "p149_slice.py --snapshot", "can_run $NEED_BAKE bake",
             "--prompts-only", "--windows-only", "can_run $NEED_GATE", "--mode gate",
             "gate_check $W/gate_S1w.json build S1 43", "gate_check $W/gate_S1w.json instrument S1 44", "run_arm S0a --q",
             "gate_check $W/arm_S0a.json build S0 45", "run_arm S1a --q", "--q-only", "QUALITY_FAIL) say", "run_tf T1",
             "run_arm S1b", "run_arm S0b", "run_arm S0c", "run_arm S1c", "run_tf T2", "run_arm S1d", "run_arm S0d",
             "for PH in S0 S1", "--mode sane",
             "--out $W/verdict_p149.json --e4b-sha \"$E4B_SHA\" $RD 2>&1 | tee -a summary.txt\n[ \"${PIPESTATUS[0]}\"",
             'if [ "$DRY" = 1 ]; then   # never a reading', "finish 29", 'case "$V" in FLIP']
    at = []
    for x in order:
        assert x in RUN, x
        at.append(RUN.index(x, at[-1] if at else 0))
    assert at == sorted(at), dict(zip(order, at))
    assert 'case "$V" in FLIP|NO_FLIP|FUNCTION_FAIL|QUALITY_FAIL) ;;' in RUN, "NOISY and VOID are not readings"
    assert 'LV="E4B_SERVE_ATTN_INT8=1"' in RUN and "E4B_SERVE_ATTN_INT8 E4B_DENSE_GEMV_BW E4B_INT8_HEAD_BW" in RUN
    assert "E4B_PAGED_MAX_TOKENS_PER_SEQ=768 E4B_PAGED_BUCKETS=1,2,4,8,16" in RUN


TERMS = r"\b(INSTALL|NEED_FETCH|NEED_BAKE|NEED_ARM|NEED_Q|NEED_REDUCE|NEED_TF|NEED_SANE|NEED_GATE|RESERVE|MIN_DISK_GB)=(\d+)"


def _fit(v):
    return (v["INSTALL"] + v["NEED_FETCH"] + v["NEED_BAKE"] + v["NEED_GATE"] + 8 * v["NEED_ARM"] + 2 * v["NEED_Q"]
            + v["NEED_REDUCE"] + 2 * v["NEED_TF"] + 2 * v["NEED_SANE"] + v["RESERVE"])


def _tables():
    """The box's table (every assignment before the dry block) and the dry run's (the box's, overridden in the block)."""
    head, dry = RUN.split('if [ "$DRY" = 1 ]; then   # the dry table (Amendment 5)', 1)
    box = {k: int(x) for k, x in re.findall(TERMS, head)}
    return box, {**box, **{k: int(x) for k, x in re.findall(TERMS, dry.split("\nfi\n", 1)[0])}}


BOOT_TOLERANCE_S = 1200      # Amendment 5: the guard leaves boot at least 20 min before the runner's own check refuses


def test_the_guard_table_is_the_registrations_and_fits_six_hours_with_a_quarter_to_spare():
    """Amendments 4 and 5: every allowance the runner checks time left against is the table's, the pre-install fit sums
    them all, boot + the fit fits the 6.0 h guard (the rental cap) at >= 25 % headroom, and boot keeps at least 20 min
    of tolerance."""
    pre = (LANE / "PREREG-p149.md").read_text(encoding="utf-8")
    assert "**The guard is 6.0 h** (21,600 s)" in pre and "**total** | **16,290 (4.53 h)**" in pre
    v, _dry = _tables()
    assert v == {"INSTALL": 450, "NEED_FETCH": 3000, "NEED_BAKE": 1500, "NEED_ARM": 480, "NEED_Q": 1200, "NEED_REDUCE": 120,
                 "NEED_TF": 840, "NEED_SANE": 900, "NEED_GATE": 600, "RESERVE": 600, "MIN_DISK_GB": 200}, v
    for term, s in (("fetch (93.4 GB)", "3,000"), ("bake", "1,500"), ("the reserve", "600"), ("the Q reduce", "120"),
                    ("the early gates' process S1w (Amendment 5), a load and one Q pass per gate", "600")):
        assert f"| {term} | {s} |" in pre, term
    fit = _fit(v)
    assert ("FIT=$((INSTALL + NEED_FETCH + NEED_BAKE + NEED_GATE + 8 * NEED_ARM + 2 * NEED_Q + NEED_REDUCE + 2 * NEED_TF"
            " + 2 * NEED_SANE + RESERVE))") in RUN
    boot, guard = 300, 6.0 * 3600
    assert "**Box Q+S keeps its 6.0 h guard** (21,600 s)" in pre and "$7.38 + $3.31 = **$10.69**" in pre
    assert boot + fit == 16290 and (boot + fit) * 1.25 <= guard
    assert "FIT_HEADROOM_PCT=125" in RUN and "FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100))" in RUN
    assert guard - fit * 1.25 >= BOOT_TOLERANCE_S     # the runtime check (fit x 125 %) leaves boot its tolerance


def test_the_dry_table_is_amendment_5s_and_fits_its_guard_with_a_quarter_to_spare():
    pre = (LANE / "PREREG-p149.md").read_text(encoding="utf-8")
    _box, v = _tables()
    assert v == {"INSTALL": 450, "NEED_FETCH": 900, "NEED_BAKE": 600, "NEED_ARM": 240, "NEED_Q": 480, "NEED_REDUCE": 120,
                 "NEED_TF": 240, "NEED_SANE": 240, "NEED_GATE": 240, "RESERVE": 600, "MIN_DISK_GB": 60}, v
    boot, guard = 300, 3.0 * 3600
    assert boot + _fit(v) == 7050 and (boot + _fit(v)) * 1.25 <= guard and guard - _fit(v) * 1.25 >= BOOT_TOLERANCE_S
    assert "**The dry guard is 3.0 h** (10,800 s)" in pre and "**dry total** | **7,050 (1.96 h)**" in pre
    assert "**$3.31**" in pre


def test_the_dry_mode_is_the_run_ids_and_never_a_reading():
    """Amendment 5: P149_DRY must match the run id before anything runs; the dry reads the slice and the --dry reducer;
    its gates report and go on; its rc is 0 or 29, never a reading's."""
    fit_at = RUN.index('[ "$LEFT" -ge "$FIT_NEED" ]')
    assert RUN.index('[ "${P149_DRY:-0}" = "$DRY" ] || {') < fit_at and RUN.index('RD=""; [ "$DRY" = 1 ] && RD=--dry') < fit_at
    assert "'model-00001-of-00019.safetensors','model-00019-of-00019.safetensors','*.json','tokenizer*'" in RUN
    assert '[ "$DRY" = 1 ] && ENGINE_MODEL="E4B_PAGED_MODEL=$W/slice"' in RUN
    assert RUN.count("$RD") == 3                         # the gate check, the Q reduce and the final reduce run --dry
    gate = RUN[RUN.index("gate_check(){"):RUN.index("can_run $NEED_GATE")]
    assert 'if [ "$DRY" = 1 ]; then echo "DRY_GATE_FAULT rc=$4"' in gate and 'finish "$4"' in gate
    assert 'echo "DRY: the Q reduce read ${QV:-nothing} on the slice; a dry run goes on"' in RUN

@pytest.fixture(scope="module")
def box_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("p149_box")
    for name, src in SOURCES.items():
        shutil.copyfile(src / name, d / name)
    return d


def _engine():
    try:
        import experts4bit_qlora.engines.paged_runner  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("script", ["p149_box.py", "p149_reduce.py"])
def test_the_self_tests_pass(box_dir, script):
    out = _run(box_dir / script, "--self-test", cwd=box_dir)
    assert out.returncode == 0 and re.search(r"self-test: (\d+)/\1 OK", out.stdout), out.stdout + out.stderr
    if script == "p149_box.py" and _engine():                # the Forcer and Q cases need the package: none skipped
        assert "self-test: 23/23 OK" in out.stdout, out.stdout


BOX_MUTANTS = [
    # Q forces the wrong first token: the window-sourced check must see it in the graph's buffer
    ("        runner.tokens[rid][-1] = w[PROMPT - 1]                 # forced: position 511's true token, not the argmax",
     "        runner.tokens[rid][-1] = w[PROMPT - 2]"),
    # the early gate's Q runs one window at W16 instead of a full pass of sixteen
    ("    return ws[:Q_SETS[name][1]]", "    return ws[:1]"),
]


@pytest.mark.parametrize("old,new", BOX_MUTANTS)
def test_each_box_mutant_fails_the_self_test(tmp_path, old, new):
    if not _engine():
        pytest.skip("the Q forcing case needs torch and the package")
    for name, d in SOURCES.items():
        shutil.copyfile(d / name, tmp_path / name)
    src = (LANE / "p149_box.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p149_box.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p149_box.py", "--self-test", cwd=tmp_path)
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def test_the_preread_self_test_passes(tmp_path):
    """Amendment 4's A2000 pre-read (lab only, never staged on the box) reduces with the box's own route_stats."""
    for name in ("p149_preread.py", "p149_box.py"):
        shutil.copyfile(LANE / name, tmp_path / name)
    out = _run(tmp_path / "p149_preread.py", "--self-test", cwd=tmp_path)
    assert out.returncode == 0 and "self-test: 3/3 OK" in out.stdout, out.stdout + out.stderr


def test_the_windows_are_p147s_by_its_own_record():
    run = json.loads((B / "p147" / "receipts" / "p147-a2000-1" / "run.json").read_text(encoding="utf-8"))
    box = (LANE / "p149_box.py").read_text(encoding="utf-8")
    red = (LANE / "p149_reduce.py").read_text(encoding="utf-8")
    for g, n in (("W1", 196), ("W16", 123)):
        w = run["windows"][g]
        assert w["windows"] == n and f'"{w["sha256"]}"' in box and f'"{w["sha256"]}"' in red, g


MUTANTS = [
    ("NOISY_BAND = (0.985, 1.015)", "NOISY_BAND = (0.98, 1.02)"),
    ("FLIP_G1, NOREG_G16 = 1.03, 0.99", "FLIP_G1, NOREG_G16 = 1.019, 0.99"),
    ("FLIP_G1, NOREG_G16 = 1.03, 0.99", "FLIP_G1, NOREG_G16 = 1.03, 0.97"),
    ("ARGMAX_MIN, UPPER_Z, PPL_BUDGET = 0.95, 1.645, 0.05", "ARGMAX_MIN, UPPER_Z, PPL_BUDGET = 0.0, 1.645, 0.05"),
    ("ARGMAX_MIN, UPPER_Z, PPL_BUDGET = 0.95, 1.645, 0.05", "ARGMAX_MIN, UPPER_Z, PPL_BUDGET = 0.95, 1.645, 0.5"),
    ("        elif set(routed) != {str(b)} or passed:", "        elif False:"),
    ('        if r.get("replays") != steps or r.get("eager_steps"):', '        if r.get("replays") != steps:'),
    ('        if f.get("buf_mismatch") != 0 or f.get("ptr_mismatch") != 0 or f.get("steps") != steps:', "        if False:"),
    ('        if got != want or r.get("int8_modules") != PROJ:', "        if False:"),
    ('        if c.get("buf_mismatch") != 0 or not b.get("ptr_ok")', '        if not b.get("ptr_ok")'),
    ('        if b.get("stream") == "B" and not same and not c.get("argmax_mismatch"):', "        if False:"),
    ('    elif rec["streams_identical"] or rho + e <= m / 2:', '    elif rec["streams_identical"] or rho <= m / 2:'),
    ('    if qv["verdict"] == "QUALITY_FAIL" and not any(t in arms for t in TAGS[2:]):', "    if False:"),
    ('    out["default_flip_pr"] = (("proceeds" if lab == "ROUTING_CLEAR" else f"held ({lab})")',
     '    out["default_flip_pr"] = (("proceeds" if True else f"held ({lab})")'),
    ("    g1_min, g16_min = min(pairs[\"W1\"].values()), min(pairs[\"W16\"].values())",
     "    g1_min, g16_min = max(pairs[\"W1\"].values()), min(pairs[\"W16\"].values())"),
    # Amendment 5: the closed forms, the gates, the dry run
    ("        elif routed[str(b)] != CAPTURE_CALLS * PROJ:", "        elif False:"),
    ("CAPTURE_CALLS = 3 ", "CAPTURE_CALLS = 2 "),
    ('"fuse_t1_glue_n": 2 * layers + 1,', '"fuse_t1_glue_n": 2 * layers,'),
    ('        if [g.get(str(b)) for b in BUCKETS] != ["graph"] * len(BUCKETS):\n            out.append(f"gate: graph_status {g}")',
     "        if False:\n            pass"),
    ('gates={g: (group, group, sha) for g, (_n, group, sha) in Q_GATES.items()}', "gates=None"),
    ("    PROJ, CENSUS = 4 * LAYERS, census(LAYERS)", "    PROJ, CENSUS = 128, census(LAYERS)"),
    ("    for t in T_TAGS:\n        faults += tf_faults(t, tf.get(t))", "    for t in T_TAGS:\n        pass"),
    ('    faults += [f"sane {ph}: missing or not ok"', '    faults += [] and [f"sane {ph}: missing or not ok"'),
    ('    faults = gate_faults(gate_rec, "build", "S1", e4b_sha) + gate_faults(gate_rec, "instrument", "S1", e4b_sha)',
     "    faults = []"),
]


@pytest.mark.parametrize("old,new", MUTANTS)
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p149_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p149_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p149_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P149_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


@pytest.mark.parametrize("run_id,ok", [("p149-5090-1", True), ("p149-5090-3", True), ("p149-5090-4", False),
                                       ("p149k-5090-1", False), ("p149-dry-2", True), ("p149-dry-4", True),
                                       ("p149-dry-1", False), ("p149-dry-5", False)])
def test_the_driver_takes_the_registered_ids_only(tmp_path, run_id, ok):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p149_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, run_id), timeout=120)
    if ok:
        assert out.returncode == 0 and "DRYRUN stage -> root@h:/root/p149 " in out.stdout + " ", out.stdout + out.stderr
        assert "bash p149_run.sh" in out.stdout
        assert ("P149_DRY=1" in out.stdout) == run_id.startswith("p149-dry-"), out.stdout
    else:
        assert out.returncode == 78 and "is not p149-5090-N" in out.stdout + out.stderr
