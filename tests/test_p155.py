# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P155, K25's plan at Qwen3-30B-A3B's W16 shapes on S0's real router picks (bench/p155/PREREG-p155.md): the staged pins;
the routing on the REAL record first (P153's committed router replay, sha-checked, and the checks stated before data);
the call form against the served route; today's plan against the library's; the runner's refusals, guard fit with 25 %
headroom and premise; no instance id or host name on the box; the self-tests and the mutants they must kill; the run
ids."""
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
LANE = B / "p155"
RR = B / "p153" / "router-replay"
SOURCES = {"p155_run.sh": LANE, "p155_bench.py": LANE, "p155_reduce.py": LANE, "rr.json.pt": RR,
           "test_k25_row_exact_gpu.py": REPO / "tests"}
RUN = (LANE / "p155_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p155_drive.sh").read_text(encoding="utf-8")
BENCH = (LANE / "p155_bench.py").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p155.md").read_text(encoding="utf-8")
LIB = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text(encoding="utf-8")
GUARD_H = 1.5
BOOT_S = 300          # the launcher's create-to-runner allowance (P152's and P154's, kept)
HEADROOM = 1.25       # e4b#1845's cross-lane rule


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


def _reducer():
    return _load("p155_reduce_under_test", LANE / "p155_reduce.py")


def _run(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=300)


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p155.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


SHAKE = LANE / "shakedown" / "shake_p155.json"
SHAKE_SHA256 = "7be425530f27165b993740204985489bd1f1c7b92b959f1cb66d225f2fbc5471"


def test_the_reducer_on_the_real_shakedown_record():
    """The real record first: the $0 A2000 shakedown's own bench record (sha-pinned) through the registered reducer. It
    must read SHAKEDOWN with every structural check passed, never graded; every plan that ran on sm_86 kept today's bits;
    the real-tile prefix held on every table; and the per-plan compile metadata was read."""
    raw = SHAKE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHAKE_SHA256
    rec = json.loads(raw)
    red = _reducer()
    v = red.verdict(rec)
    assert v["verdict"] == "SHAKEDOWN" and v["why"] == ["a shakedown record is never a reading"], v["why"]
    assert set(v["predictions"].values()) == {"UNREAD (a shakedown)"}
    assert rec["cc"] == [8, 6] and rec["layers"] == 4 and rec["tile_faults"] == []
    for p in ("gu", "dn"):
        rows = rec["select"][p]
        assert sum(r["identical"] is True for r in rows) == 126 and sum(r["identical"] is False for r in rows) == 0
        assert {r["error"].split(":")[0] for r in rows if r["error"]} == {"OutOfResources"}       # 18 on sm_86's 99 KB
        assert rec["today_compiled"][p] == [{"n_regs": 71, "n_spills": 0, "shared": 14592, "num_warps": 4,
                                             "num_stages": 3}]                                   # b1-decode's sm_86 numbers
    assert all(rep["identical"][a] for rep in rec["confirm"]["reps"] for a in rep["identical"])
    bad = json.loads(raw)                                    # the old rule's mutant, on the real record: a hand-carried
    bad["carried"] = list(reversed(bad["carried"]))          # selection is VOID
    assert red.verdict(bad)["verdict"] == "VOID"


def test_need_bench_is_the_shakedowns_walls_times_one_and_a_half():
    """The maintainer's rule: NEED_BENCH is the measured walls x1.5, not an estimate (PREREG, "The shakedown")."""
    rec = json.loads(SHAKE.read_bytes())
    compile_s = sum(r["first_call_s"] for p in ("gu", "dn") for r in rec["select"][p] if r.get("first_call_s") is not None)
    ran = sum(r["identical"] is True for p in ("gu", "dn") for r in rec["select"][p])
    host_s = (rec["walls"]["select_s"] - compile_s) / (ran * 2 + 2 * 2 * 2)        # per selection graph, host-bound
    conf_s = rec["walls"]["confirm_s"] / (8 * 6)
    gpu = 12 * 11.748e-3                                                             # 12 replays of P152's two GEMM lines
    box = (compile_s * 144 / 126 + (2 * 144 + 4) * 8 * (host_s + gpu) + 64 * 6 * (conf_s + gpu) + 20)
    assert round(compile_s, 1) == 145.3 and round(host_s, 3) == 0.415 and round(conf_s, 3) == 0.215
    assert 1550 <= box <= 1700 and box * 1.5 <= 2450, box


def test_the_router_replay_is_the_committed_record():
    red = _reducer()
    raw = (RR / "rr.json.pt").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == red.RR_SHA256 == _pins(LANE / "staged-p155.sha256")["rr.json.pt"]
    assert f"{sha}  ./rr.json.pt" in (RR / "SHA256SUMS").read_text(encoding="ascii")  # P153's own checksum list
    assert "3c59e72f947daa425e61f727401204099a1e782b902fb39687e04c46e338b14f" in PREREG


def test_s0s_picks_pass_every_check_stated_before_data():
    """The real record first: the box's own loader on the committed file, and P153's numbers out of it."""
    torch = pytest.importorskip("torch")
    bench = _load("p155_bench_under_test", LANE / "p155_bench.py")
    red = bench.R
    s0, sha = bench.s0_picks(str(RR / "rr.json.pt"))
    assert sha == red.S0_SHA256 and tuple(s0.shape) == (48, 16, 128, 8) and s0.dtype == torch.int16
    nreal = bench.n_real_table(s0, 48)
    flat = [v for row in nreal for v in row]
    assert sum(flat) == red.S0_DISTINCT_TOTAL == 350868                     # P153's record
    assert round(sum(flat) / len(flat), 3) == 57.107 and min(flat) == 40 and max(flat) == 82
    rows_per_expert = max(int(torch.bincount(s0[li, :, s, :].reshape(-1).long(), minlength=128).max())
                          for li in range(48) for s in range(128))
    assert rows_per_expert <= 16, "an expert past 16 rows would take two tiles"
    for figure in ("350,868", "57.107", "40 to 82", "3020dbce"):
        assert figure in PREREG, figure
    p153 = (RR / "README.md").read_text(encoding="utf-8")
    assert "350,868" in p153 and "57.107" in p153


def test_the_routing_checks_refuse_what_they_name(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    bench = _load("p155_bench_under_test", LANE / "p155_bench.py")
    d = torch.load(str(RR / "rr.json.pt"), weights_only=True)
    with pytest.raises(SystemExit, match="sha256 is not the pinned"):              # any other bytes
        other = tmp_path / "other.pt"
        torch.save({"picks": d["picks"]}, str(other))
        bench.s0_picks(str(other))
    picks = d["picks"].clone()
    picks[3, 2, 7, 1] = picks[3, 2, 7, 0]                                           # one row picks an expert twice
    dup = tmp_path / "dup.pt"
    torch.save({"picks": picks}, str(dup))
    monkeypatch.setattr(bench.R, "RR_SHA256", hashlib.sha256(dup.read_bytes()).hexdigest())
    monkeypatch.setattr(bench.R, "S0_SHA256", hashlib.sha256(
        picks[:, 0:16].contiguous().numpy().astype("<i2").tobytes()).hexdigest())
    with pytest.raises(SystemExit, match="picks one expert twice"):
        bench.s0_picks(str(dup))


def test_todays_plan_is_the_served_plan():
    pytest.importorskip("torch")
    from experts4bit_qlora.engines import hot_residency as hr
    red = _reducer()
    assert hr._K25_PLAN == dict(red.TODAY, lut="tree", dot_bf16=False)
    assert red.TODAY in red.GRID and len(red.GRID) == 144
    assert '"block_n": 32, "kc": 64, "warps": 4, "stages": 3, "lut": "tree", "dot_bf16": False}' in RUN   # the tripwire


def test_the_bench_calls_k25_as_the_served_route():
    # the served route at 16 NF4 rows under K23's lean glue: gate_up reads the token rows (gather_div), down stores into
    # the caller's order (scatter), over the one-launch lean table
    assert "return _k25(_tok[0].to(torch.bfloat16), pk, am, t_row0, t_rows, t_grp, order,\n" \
           "                            gather_div=_tok[1], **_k25_kw)" in LIB
    assert 'kw = {"scatter": order} if (_lean and pk is dn_p) else {}' in LIB
    assert "_fused_tiles(\n                local_ids, _n_exp, 16, lean=True, sorted_ids=True)" in LIB
    assert "k25(X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK, lut=\"tree\", dot_bf16=False, **plan)" in BENCH
    assert "k25(H[li], p, am, row0, rows, grp, None, scatter=order, lut=\"tree\", dot_bf16=False, **plan)" in BENCH
    assert "build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)" in BENCH
    assert "hr._lean_glue_supported(int4_b32.build_group_tiles_fused, nf4_smallm.gemm_nf4_grouped_smallm)" in RUN


def test_the_compaction_arm_is_the_real_tile_prefix():
    # b1-decode's invariant, checked on the box for every table and a VOID if it fails: the real tiles are the prefix
    assert "compact[s].append((row0[:n], rows[:n], grp[:n], order))" in BENCH
    assert "any(v <= 0 for v in rv[:n]) or any(v != 0 for v in rv[n:])" in BENCH


def test_the_runner_refuses_before_the_install():
    first_install = RUN.index('say "install e4b @')
    for refusal in ('[ "$PL" -ge "$MIN_POWER_W" ] || {', '[ "$CC" = 12.0 ] || {', '[ "$LEFT" -ge "$FIT_NEED" ] || {',
                    '[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || {'):
        assert RUN.index(refusal) < first_install, refusal
    assert RUN.index('[ "$LEFT" -ge "$FIT_NEED" ] || {') < RUN.index("pipx(){")
    floor = RUN.index('[ "$PL" -ge "$MIN_POWER_W" ] || {')
    assert "MIN_POWER_W=575" in RUN and "finish 17; }" in RUN[floor:floor + 260]
    assert "${P155_MIN_POWER_W:-}" in RUN and "${P155_NEED_BENCH:-}" in RUN            # frozen reading knobs
    assert 'md.version("grouped-nf4-gemm") == "0.48.0"' in RUN
    assert "GNF4_SHA=ef2faf3816b30ecfd0e2a69cb27fb049a7d56a59 " in RUN
    for sha in ("96176de875af5273d3f13edd9c3dfd1371069dea98a7bf6d618f1dac05e2d412",
                "773eff5db7be77dea3d1f9048a6e708a94e79634baf17f452bf2848e24a2072f"):
        assert sha in RUN and sha[:8] in PREREG
    assert "grep -qE '^4 passed(, [0-9]+ warnings?)? in" in RUN
    order = [RUN.index(x) for x in ('say "install e4b @', "python -m pytest test_k25_row_exact_gpu.py",
                                    "can_run $NEED_BENCH bench", "python $W/p155_bench.py --rr",
                                    "python $W/p155_reduce.py --bench")]
    assert order == sorted(order)
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_NF4_GROUPED_SMALLM", "E4B_INT4_LEAN_GLUE", "TRITON_INTERPRET", "E4B_INT4_WIDE_TILES"):
        assert re.search(rf"\b{knob}\b", unset), knob


def test_the_guard_fits_with_25_percent_headroom():
    """e4b#1845's rule on both sides: the runner refuses before the install below FIT x 125 %, and the registered guard
    holds the boot plus that fit with 25 % headroom."""
    k = {n: int(v) for n, v in re.findall(r"^(INSTALL_TO_GATE_S|NEED_BENCH|NEED_NCU|RESERVE)=(\d+)", RUN, re.M)}
    assert k == {"INSTALL_TO_GATE_S": 600, "NEED_BENCH": 2450, "NEED_NCU": 300, "RESERVE": 600}, k
    assert ("FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + NEED_NCU + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100))"
            in RUN)
    assert re.search(r"^FIT_HEADROOM_PCT=125 ", RUN, re.M)
    assert "two consecutive headroom refusals count as one strike" in RUN
    fit = sum(k.values())
    assert fit * 125 // 100 == 4937 and "`FIT_NEED` = 3950 x 1.25 =\n    4937 s" in PREREG
    assert (BOOT_S + fit) * HEADROOM <= GUARD_H * 3600
    assert "4250 x 1.25 = 5313 s <= 5400 s (1.5 h)" in PREREG and "**guard 1.5 h**" in PREREG
    line = next(x for x in RUN.splitlines() if x.startswith("can_run(){"))
    assert "fit=$((need + RESERVE))" in line and "needs ${fit}s" in line and "only ${left}s left" in line


def test_the_nsight_step_is_reported_only():
    """It runs after the verdict is read, probes the counters first, and every path out of it returns 0: the exit code is
    the verdict's alone (the maintainer's ruling, and loggetta's ERR_NVGPUCTRPERM note)."""
    body = RUN[RUN.index("ncu_step(){"):RUN.index("\n}\n", RUN.index("ncu_step(){"))]
    assert "finish" not in body and "exit" not in body
    assert body.index("--metrics sm__cycles_elapsed.avg") < body.index("--launch-skip 6 --launch-count 2")
    assert "ERR_NVGPUCTRPERM" in body and "counters unavailable on this host" in body
    tail = RUN[RUN.index('V=$(python -c "import json'):]
    assert tail.index('if can_run $NEED_NCU "ncu (reported only)"; then') < tail.index('case "$V" in GO|NO_GO) finish 0;;')
    assert "range(4)" in BENCH[BENCH.index("def ncu_target"):BENCH.index("def n_real_table")]   # 3 warm rounds + 1


def test_no_instance_id_or_host_name_reaches_the_box():
    for name, text in (("runner", RUN), ("bench", BENCH), ("reducer", (LANE / "p155_reduce.py").read_text("utf-8"))):
        for marker in ("gethostname", "socket.", "INSTANCE_ID", "hostname"):
            assert marker not in text, (name, marker)
    pass_line = next(x for x in DRIVE.splitlines() if x.startswith("PASS="))
    assert "INSTANCE" not in pass_line and "P155_DEADLINE_EPOCH=$DEADLINE" in pass_line
    assert "HF_TOKEN" not in DRIVE                                                       # nothing is fetched


def test_the_constants_match_their_records():
    red = _reducer()
    p152 = json.loads((B / "p152" / "receipts" / "p152-5090-2" / "verdict_p152.json").read_text(encoding="utf-8"))
    lines = {c["key"]: c["removable_ms"] for c in p152["W16"]["candidates"]}
    assert red.P152_BOTH_MS == lines["_gemm_nf4_grouped_smallm|136x48x1"] + lines["_gemm_nf4_grouped_smallm|136x64x1"]
    p154 = json.loads((B / "p154" / "receipts" / "p154-5090-1" / "verdict_p154.json").read_text(encoding="utf-8"))
    assert red.P154_HEADROOM_MS == p154["both"]["headroom_ms"]
    assert red.SHAPES == {"gu": (1536, 2048), "dn": (2048, 768)}                     # P154's store: 1536x1024, 2048x384 B


@pytest.mark.parametrize("script", ["p155_reduce.py", "p155_bench.py"])
def test_the_self_test_passes(script):
    out = _run(LANE / script, "--self-test")
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


REDUCE_MUTANTS = [
    # GO on the median alone, without the 2-SE guard
    ('    elif winner["lb"] >= GO_BAR:\n', '    elif winner["median"] >= GO_BAR:\n'),
    # the median's lower bound taken with the MEAN's standard error (loggetta's review)
    ("    se = se_median(q) if n > 1 else math.inf\n", "    se = statistics.stdev(q) / math.sqrt(n) if n > 1 else math.inf\n"),
    # the carried plans taken on trust instead of re-derived from the recorded times
    ('    if rec.get("carried") != carried:\n', "    if False:\n"),
    # the self-pair band not enforced
    ("    if not SELF_BAND[0] <= self_pair[\"median\"] <= SELF_BAND[1]:\n", "    if False:\n"),
    # a carried plan that moved bits kept as a candidate
    ("        if moved:\n", "        if False:\n"),
    # today's plan allowed to replay different bits
    ('    if not all(rep["identical"]["T2"] is True for rep in reps):\n', "    if False:\n"),
    # the rotation not checked
    ('        if rep.get("step") != steps[r] or rep.get("rotation") != rotation(arms, r):\n',
     '        if rep.get("step") != steps[r]:\n'),
    # any card read as the 5090
    ('        if "5090" not in str(rec.get("device", "")) or rec.get("cc") != [12, 0]:\n', "        if False:\n"),
    # S0's picks not checked
    ('    if rec.get("s0_sha256") != S0_SHA256:\n', "    if False:\n"),
]
BENCH_MUTANTS = [
    # gate_up without the served gather (it would read 128 expanded rows the served route never builds)
    ("k25(X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK,", "k25(X[li], p, am, row0, rows, grp, order,"),
    # down without the served scatter (it would store in sorted order, a different kernel specialisation)
    ("k25(H[li], p, am, row0, rows, grp, None, scatter=order,", "k25(H[li], p, am, row0, rows, grp, None,"),
]


@pytest.mark.parametrize("script,old,new", [("p155_reduce.py", o, n) for o, n in REDUCE_MUTANTS]
                         + [("p155_bench.py", o, n) for o, n in BENCH_MUTANTS])
def test_each_mutant_fails_the_self_test(tmp_path, script, old, new):
    src = (LANE / script).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / script).write_bytes(src.replace(old, new).encode("utf-8"))
    if script == "p155_bench.py":                                    # the bench imports its sibling reducer
        shutil.copy(LANE / "p155_reduce.py", tmp_path / "p155_reduce.py")
    out = _run(tmp_path / script, "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P155_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p155_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p155-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p155" in out.stdout and "bash p155_run.sh" in out.stdout
    assert "INSTANCE" not in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p155_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p154-5090-1"), timeout=120)
    assert out.returncode == 78 and "is not p155-5090-N" in out.stdout + out.stderr


def test_the_run_ids_are_registered_in_both_scripts():
    for text in (RUN, DRIVE):
        assert "p155-5090-1|p155-5090-2|p155-5090-3)" in text
    assert "`p155-5090-1` to `-3`" in PREREG
