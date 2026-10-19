# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P156, K39's decode against today's select tree at Qwen3-30B-A3B's W16 shapes on S0's real router picks
(bench/p156/PREREG-p156.md): the staged pins; the shakedown on the REAL record first; the routing (P155's checks, on the
committed router replay); the call form against the served route, with the decode the only moving argument; the
runner's pins, refusals, guard fit with 25 % headroom (from P155's measured box walls) and premise; no instance id or
host name on the box; the self-tests and the mutants they must kill; the run ids."""
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p156"
RR = B / "p153" / "router-replay"
SOURCES = {"p156_run.sh": LANE, "p156_bench.py": LANE, "p156_reduce.py": LANE, "rr.json.pt": RR,
           "test_k25_row_exact_gpu.py": REPO / "tests"}
RUN = (LANE / "p156_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p156_drive.sh").read_text(encoding="utf-8")
BENCH = (LANE / "p156_bench.py").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p156.md").read_text(encoding="utf-8")
LIB = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text(encoding="utf-8")
GUARD_H = 1.25
BOOT_S = 300          # the launcher's create-to-runner allowance (P155's, kept)
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
    return _load("p156_reduce_under_test", LANE / "p156_reduce.py")


def _run(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=300)


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p156.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


SHAKE = LANE / "shakedown" / "shake_p156.json"
SHAKE_SHA256 = "b7fd7320f3f055f538da3e7428bc491df024fb017931597135a9e8e3dacc8a40"


def test_the_reducer_on_the_real_shakedown_record():
    """The real record first: the $0 A2000 shakedown's own bench record (sha-pinned) through the registered reducer. It
    must read SHAKEDOWN with every structural check passed, never graded; every prmt plan ran on sm_86 and kept T's
    bits; P and every carried plan kept them on every held-out rep; and both decodes' compile metadata was read."""
    raw = SHAKE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHAKE_SHA256
    rec = json.loads(raw)
    red = _reducer()
    v = red.verdict(rec)
    assert v["verdict"] == "SHAKEDOWN" and v["why"] == ["a shakedown record is never a reading"], v["why"]
    assert set(v["predictions"].values()) == {"UNREAD (a shakedown)"}
    assert rec["cc"] == [8, 6] and rec["layers"] == 4 and rec["tile_faults"] == [] and rec["triton"] == "3.4.0"
    assert rec["luts"] == {"T": "tree", "P": "prmt", "sweep": "prmt"}
    for p in ("gu", "dn"):
        rows = rec["select"][p]
        assert sum(r["identical"] is True for r in rows) == 144 and not any(r["error"] for r in rows)
        assert rec["compiled"]["T"][p] == [{"n_regs": 71, "n_spills": 0, "shared": 14592, "num_warps": 4,
                                            "num_stages": 3}]                                    # P155's sm_86 numbers
        assert rec["compiled"]["P"][p] == [{"n_regs": 64, "n_spills": 0, "shared": 6400, "num_warps": 4,
                                            "num_stages": 3}]
    assert all(rep["identical"][a] for rep in rec["confirm"]["reps"] for a in rep["identical"])
    assert v["contract_findings"] == [] and v["P"]["moved_bits_on_steps"] == []
    bad = json.loads(raw)                                    # the old rule's mutant, on the real record: a hand-carried
    bad["carried"] = list(reversed(bad["carried"]))          # sweep is VOID
    assert red.verdict(bad)["verdict"] == "VOID"


#: P155's box bench (p155-5090-1; its record is e4b#1856): the walls of the same shape, 252 of the 288 plans running
P155_BOX_WALLS_S, P155_BOX_RAN = 0.4 + 0.4 + 2.6 + 782.0 + 110.2, 252


def test_need_bench_is_the_larger_measured_wall_times_one_and_a_half():
    """The maintainer's rule: NEED_BENCH is measured walls x1.5, from two measurements, the larger taken: P155's box
    bench scaled to all 288 plans running (the shakedown: no OutOfResources at K39's shared memory), and this lane's own
    shakedown walls through P155's formula (tests/test_p155.py)."""
    rec = json.loads(SHAKE.read_bytes())
    compile_s = sum(r["first_call_s"] for p in ("gu", "dn") for r in rec["select"][p] if r.get("first_call_s") is not None)
    ran = sum(r["identical"] is True for p in ("gu", "dn") for r in rec["select"][p])
    host_s = (rec["walls"]["select_s"] - compile_s) / (ran * 2 + 2 * 2 * 2)        # per selection graph, host-bound
    conf_s = rec["walls"]["confirm_s"] / (8 * 6)
    gpu = 12 * 11.748e-3                                                             # 12 replays of P152's two GEMM lines
    shake_box = compile_s + (2 * 144 + 4) * 8 * (host_s + gpu) + 64 * 6 * (conf_s + gpu) + 20
    assert ran == 288 and round(compile_s, 1) == 133.3 and round(host_s, 3) == 0.131 and round(conf_s, 3) == 0.165
    p155_box = P155_BOX_WALLS_S * 288 / P155_BOX_RAN
    assert round(shake_box) == 907 and round(p155_box) == 1024
    need = int(re.search(r"^NEED_BENCH=(\d+)", RUN, re.M).group(1))
    assert need == math.ceil(max(shake_box, p155_box) * 1.5 / 100) * 100 == 1600
    for figure in ("**896 s**", "about 1,024 s", "about 907 s", "x1.5 = 1,536 s"):
        assert figure in PREREG, figure


def test_the_router_replay_is_the_committed_record():
    red = _reducer()
    raw = (RR / "rr.json.pt").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == red.RR_SHA256 == _pins(LANE / "staged-p156.sha256")["rr.json.pt"]
    assert f"{sha}  ./rr.json.pt" in (RR / "SHA256SUMS").read_text(encoding="ascii")  # P153's own checksum list
    assert red.RR_SHA256[:8] in PREREG and red.S0_SHA256[:8] in PREREG and "350,868" in PREREG


def test_s0s_picks_pass_every_check_stated_before_data():
    """The real record first: the box's own loader on the committed file, and P153's total out of it."""
    torch = pytest.importorskip("torch")
    bench = _load("p156_bench_under_test", LANE / "p156_bench.py")
    red = bench.R
    s0, sha = bench.s0_picks(str(RR / "rr.json.pt"))
    assert sha == red.S0_SHA256 and tuple(s0.shape) == (48, 16, 128, 8) and s0.dtype == torch.int16
    assert sum(v for row in bench.n_real_table(s0, 48) for v in row) == red.S0_DISTINCT_TOTAL == 350868


def test_the_routing_checks_refuse_what_they_name(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    bench = _load("p156_bench_under_test", LANE / "p156_bench.py")
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


def test_the_registration_shares_p155s_routing_grid_and_constants():
    """P155's instrument, unchanged except the decode: the same router replay, S0 slice, grid, today's plan, steps,
    shapes and floors (its reducer is merged; this lane's must agree with it)."""
    red, p155 = _reducer(), _load("p155_reduce_for_p156", B / "p155" / "p155_reduce.py")
    for name in ("RR_SHA256", "S0_SHA256", "S0_DISTINCT_TOTAL", "GRID", "TODAY", "SHAPES", "SELECT_STEPS",
                 "CONFIRM_STEPS", "IDENTITY_STEP", "CARRY", "ITERS", "SELF_BAND", "P152_BOTH_MS", "P154_HEADROOM_MS",
                 "BOOT_B"):
        assert getattr(red, name) == getattr(p155, name), name
    assert red.ORDER_SEED == red.BOOT_SEED == 156 and red.plan_order() != p155.plan_order()   # this lane's own shuffle
    assert (red.T_LUT, red.P_LUT, red.GO_BAR) == ("tree", "prmt", 1.05)


def test_todays_plan_is_the_served_plan():
    pytest.importorskip("torch")
    from experts4bit_qlora.engines import hot_residency as hr
    red = _reducer()
    assert hr._K25_PLAN == dict(red.TODAY, lut=red.T_LUT, dot_bf16=False)
    assert red.TODAY in red.GRID and len(red.GRID) == 144
    assert '"block_n": 32, "kc": 64, "warps": 4, "stages": 3, "lut": "tree", "dot_bf16": False}' in RUN   # the tripwire


def test_the_bench_calls_k25_as_the_served_route_with_only_the_decode_moving():
    # the served route at 16 NF4 rows under K23's lean glue: gate_up reads the token rows (gather_div), down stores into
    # the caller's order (scatter), over the one-launch lean table
    assert "return _k25(_tok[0].to(torch.bfloat16), pk, am, t_row0, t_rows, t_grp, order,\n" \
           "                            gather_div=_tok[1], **_k25_kw)" in LIB
    assert 'kw = {"scatter": order} if (_lean and pk is dn_p) else {}' in LIB
    assert "_fused_tiles(\n                local_ids, _n_exp, 16, lean=True, sorted_ids=True)" in LIB
    assert "k25(X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK, lut=lut, dot_bf16=False, **plan)" in BENCH
    assert "k25(H[li], p, am, row0, rows, grp, None, scatter=order, lut=lut, dot_bf16=False, **plan)" in BENCH
    assert BENCH.count("lut=lut, dot_bf16=False, **plan)") == 2 and not re.search(r'k25\([^\n]*lut="', BENCH)
    assert '"T": (R.TODAY, R.TODAY, R.T_LUT), "T2": (R.TODAY, R.TODAY, R.T_LUT), "P": (R.TODAY, R.TODAY, R.P_LUT)' in BENCH
    assert "build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)" in BENCH
    assert "hr._lean_glue_supported(int4_b32.build_group_tiles_fused, nf4_smallm.gemm_nf4_grouped_smallm)" in RUN
    assert "any(v <= 0 for v in rv[:n]) or any(v != 0 for v in rv[n:])" in BENCH        # the real-tile prefix, checked


def test_the_runner_pins_k25_and_k39_and_refuses_before_the_install():
    first_install = RUN.index('say "install e4b @')
    for refusal in ('[ "$PL" -ge "$MIN_POWER_W" ] || {', '[ "$CC" = 12.0 ] || {', '[ "$LEFT" -ge "$FIT_NEED" ] || {',
                    '[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || {'):
        assert RUN.index(refusal) < first_install, refusal
    assert RUN.index('[ "$LEFT" -ge "$FIT_NEED" ] || {') < RUN.index("pipx(){")
    floor = RUN.index('[ "$PL" -ge "$MIN_POWER_W" ] || {')
    assert "MIN_POWER_W=575" in RUN and "finish 17; }" in RUN[floor:floor + 260]
    assert "${P156_MIN_POWER_W:-}" in RUN and "${P156_NEED_BENCH:-}" in RUN            # frozen reading knobs
    gnf4 = re.search(r"^GNF4_SHA=([0-9a-f]{40}) ", RUN, re.M)
    assert gnf4 and gnf4.group(1)[:8] in PREREG, "grouped-nf4-gemm is pinned at #569's merge, by its full sha"
    assert re.search(r'md\.version\("grouped-nf4-gemm"\) == "\d+\.\d+\.\d+"', RUN)
    for sha in ("ee46969a4f786c36abe26bfaf0a52e426db4465406a5e31c6085fd3270b26568",    # nf4_smallm.py: K25 + K39
                "773eff5db7be77dea3d1f9048a6e708a94e79634baf17f452bf2848e24a2072f",    # int4_b32.py
                "a7090915485c6e9943f568e79315b3f533c754f24edbdb7160b05dcc18619a23"):   # nf4_grouped.py: K39's tables
        assert sha in RUN and sha[:8] in PREREG
    assert 'nf4_smallm._LUT_MODES.get("prmt") == 4' in RUN
    assert "grep -qE '^4 passed(, [0-9]+ warnings?)? in" in RUN
    order = [RUN.index(x) for x in ('say "install e4b @', "python -m pytest test_k25_row_exact_gpu.py",
                                    "can_run $NEED_BENCH bench", "python $W/p156_bench.py --rr",
                                    "python $W/p156_reduce.py --bench")]
    assert order == sorted(order)
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_NF4_GROUPED_SMALLM", "E4B_INT4_LEAN_GLUE", "TRITON_INTERPRET", "E4B_INT4_WIDE_TILES"):
        assert re.search(rf"\b{knob}\b", unset), knob


def test_the_guard_fits_with_25_percent_headroom():
    """e4b#1845's rule on both sides: the runner refuses before the install below FIT x 125 %, and the registered guard
    holds the boot plus that fit with 25 % headroom."""
    k = {n: int(v) for n, v in re.findall(r"^(INSTALL_TO_GATE_S|NEED_BENCH|NEED_NCU|RESERVE)=(\d+)", RUN, re.M)}
    assert k == {"INSTALL_TO_GATE_S": 600, "NEED_BENCH": 1600, "NEED_NCU": 300, "RESERVE": 600}, k
    assert ("FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + NEED_NCU + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100))"
            in RUN)
    assert re.search(r"^FIT_HEADROOM_PCT=125 ", RUN, re.M)
    assert "two consecutive headroom refusals count as one strike" in RUN
    fit = sum(k.values())
    assert fit * 125 // 100 == 3875 and "`FIT_NEED` = 3100 x 1.25 =\n    3875 s" in PREREG
    assert (BOOT_S + fit) * HEADROOM <= GUARD_H * 3600
    assert "3400 x 1.25 = 4250 s <= 4500 s (1.25 h)" in PREREG and "**guard 1.25 h**" in PREREG
    assert "1.25 x $1.05 + 10 x $0.011 =\n    **$1.42**" in PREREG
    line = next(x for x in RUN.splitlines() if x.startswith("can_run(){"))
    assert "fit=$((need + RESERVE))" in line and "needs ${fit}s" in line and "only ${left}s left" in line


def test_the_nsight_step_is_reported_only():
    """It runs after the verdict is read, probes the counters first, and every path out of it returns 0: the exit code is
    the verdict's alone. It captures T's and P's two calls: 3 warm rounds of 4, then the measured round."""
    body = RUN[RUN.index("ncu_step(){"):RUN.index("\n}\n", RUN.index("ncu_step(){"))]
    assert "finish" not in body and "exit" not in body
    assert body.index("--metrics sm__cycles_elapsed.avg") < body.index("--launch-skip 12 --launch-count 4")
    assert "ERR_NVGPUCTRPERM" in body and "counters unavailable on this host" in body
    tail = RUN[RUN.index('V=$(python -c "import json'):]
    assert tail.index('if can_run $NEED_NCU "ncu (reported only)"; then') < tail.index('case "$V" in GO|NO_GO) finish 0;;')
    target = BENCH[BENCH.index("def ncu_target"):BENCH.index("def n_real_table")]
    assert "range(4)" in target and "for lut in (R.T_LUT, R.P_LUT):" in target


def test_no_instance_id_or_host_name_reaches_the_box():
    for name, text in (("runner", RUN), ("bench", BENCH), ("reducer", (LANE / "p156_reduce.py").read_text("utf-8"))):
        for marker in ("gethostname", "socket.", "INSTANCE_ID", "hostname"):
            assert marker not in text, (name, marker)
    pass_line = next(x for x in DRIVE.splitlines() if x.startswith("PASS="))
    assert "INSTANCE" not in pass_line and "P156_DEADLINE_EPOCH=$DEADLINE" in pass_line
    assert "HF_TOKEN" not in DRIVE                                                       # nothing is fetched


def test_the_constants_match_their_records():
    red = _reducer()
    p152 = json.loads((B / "p152" / "receipts" / "p152-5090-2" / "verdict_p152.json").read_text(encoding="utf-8"))
    lines = {c["key"]: c["removable_ms"] for c in p152["W16"]["candidates"]}
    assert red.P152_BOTH_MS == lines["_gemm_nf4_grouped_smallm|136x48x1"] + lines["_gemm_nf4_grouped_smallm|136x64x1"]
    p154 = json.loads((B / "p154" / "receipts" / "p154-5090-1" / "verdict_p154.json").read_text(encoding="utf-8"))
    assert red.P154_HEADROOM_MS == p154["both"]["headroom_ms"]
    assert round(red.BYTE_FLOOR_MS, 2) == 4.94 and "**4.94 ms**" in PREREG


def test_the_predictions_are_the_registered_models():
    red = _reducer()
    assert red.MODELS == {"M1 overlap": (1.02, 1.11), "M3 serial": (1.15, 1.18), "M2 overlap + occupancy": (1.19, 1.28)}
    assert red.PRED["headline"] == (1.05, 1.18) and red.PRED["regs"] == {"T": 80, "P": 64}
    for figure in ("**1.02-1.11**", "**1.15-1.18**", "1.19-1.28", "**the least likely**", "**1.05-1.18**",
                   "**8.404 ms**", "**8.317 ms**", "12.075 ms", "0.027", "**17.00**", "**0.70**", "0.51",
                   "grouped-nf4-gemm#570"):
        assert figure in PREREG, figure


@pytest.mark.parametrize("script", ["p156_reduce.py", "p156_bench.py"])
def test_the_self_test_passes(script):
    out = _run(LANE / script, "--self-test")
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


REDUCE_MUTANTS = [
    # GO on the median alone, without the 2-SE guard
    ('    elif p["lb"] >= GO_BAR:\n', '    elif p["median"] >= GO_BAR:\n'),
    # GO ruled by the sweep's best instead of the pre-declared P
    ('    elif p["lb"] >= GO_BAR:\n', '    elif (best or p)["lb"] >= GO_BAR:\n'),
    # the median's lower bound taken with the MEAN's standard error (loggetta's review of P155)
    ("    se = se_median(q) if n > 1 else math.inf\n", "    se = statistics.stdev(q) / math.sqrt(n) if n > 1 else math.inf\n"),
    # the carried plans taken on trust instead of re-derived from the recorded times
    ('    if rec.get("carried") != carried:\n', "    if False:\n"),
    # today's plan carried by the sweep (it is P, the registered question)
    ('_eligible(r, n_steps) and r["plan"] != TODAY]', "_eligible(r, n_steps)]"),
    # the self-pair band not enforced
    ("    if not SELF_BAND[0] <= self_pair[\"median\"] <= SELF_BAND[1]:\n", "    if False:\n"),
    # a P that moved bits ruled on as a timing
    ("    elif p_moved:\n", "    elif False:\n"),
    # a carried plan that moved bits kept as a candidate
    ("        if moved:\n", "        if False:\n"),
    # today's tree allowed to replay different bits
    ('    if not all(rep["identical"]["T2"] is True for rep in reps):\n', "    if False:\n"),
    # the rotation not checked
    ('        if rep.get("step") != steps[r] or rep.get("rotation") != rotation(arms, r):\n',
     '        if rep.get("step") != steps[r]:\n'),
    # any card read as the 5090
    ('        if "5090" not in str(rec.get("device", "")) or rec.get("cc") != [12, 0]:\n', "        if False:\n"),
    # S0's picks not checked
    ('    if rec.get("s0_sha256") != S0_SHA256:\n', "    if False:\n"),
    # the decodes not checked (P could be the tree)
    ('    if rec.get("luts") != {"T": T_LUT, "P": P_LUT, "sweep": P_LUT}:\n', "    if False:\n"),
]
BENCH_MUTANTS = [
    # gate_up without the served gather (it would read 128 expanded rows the served route never builds)
    ("k25(X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK,", "k25(X[li], p, am, row0, rows, grp, order,"),
    # down without the served scatter (it would store in sorted order, a different kernel specialisation)
    ("k25(H[li], p, am, row0, rows, grp, None, scatter=order,", "k25(H[li], p, am, row0, rows, grp, None,"),
    # P timed with the tree (the question unasked)
    ('"P": (R.TODAY, R.TODAY, R.P_LUT)}', '"P": (R.TODAY, R.TODAY, R.T_LUT)}'),
    # the sweep run with the tree
    ("plan, R.P_LUT) for li in range(L)]\n                    torch.cuda.synchronize()",
     "plan, R.T_LUT) for li in range(L)]\n                    torch.cuda.synchronize()"),
]


@pytest.mark.parametrize("script,old,new", [("p156_reduce.py", o, n) for o, n in REDUCE_MUTANTS]
                         + [("p156_bench.py", o, n) for o, n in BENCH_MUTANTS])
def test_each_mutant_fails_the_self_test(tmp_path, script, old, new):
    src = (LANE / script).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / script).write_bytes(src.replace(old, new).encode("utf-8"))
    if script == "p156_bench.py":                                    # the bench imports its sibling reducer
        shutil.copy(LANE / "p156_reduce.py", tmp_path / "p156_reduce.py")
    out = _run(tmp_path / script, "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P156_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p156_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p156-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p156" in out.stdout and "bash p156_run.sh" in out.stdout
    assert "INSTANCE" not in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p156_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p155-5090-1"), timeout=120)
    assert out.returncode == 78 and "is not p156-5090-N" in out.stdout + out.stderr


def test_the_run_ids_are_registered_in_both_scripts():
    for text in (RUN, DRIVE):
        assert "p156-5090-1|p156-5090-2|p156-5090-3)" in text
    assert "`p156-5090-1` to `-3`" in PREREG
