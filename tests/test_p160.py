# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P160, decode x plan (bench/p160/PREREG-p160.md): K39's decode at P156's carried tile plans against prmt at today's
plan, the served default. The staged pins; the shakedown on the REAL record first; the carried plans are P156's record's
and never re-chosen; the routing (both streams, on the committed router replay); the call form against the served route,
with only the plan (C1-C3) or only the decode (T) moving against P; the runner's pins, refusals, tripwire, premise and
guard fit with 25 % headroom (from P156's measured box walls); no instance id or host name on the box; the self-tests and
the mutants they must kill; the run ids."""
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
LANE = B / "p160"
RR = B / "p153" / "router-replay"
SOURCES = {"p160_run.sh": LANE, "p160_bench.py": LANE, "p160_reduce.py": LANE, "rr.json.pt": RR,
           "test_k25_row_exact_gpu.py": REPO / "tests", "test_nf4_smallm_prmt_gpu.py": B / "p158" / "premise"}
RUN = (LANE / "p160_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p160_drive.sh").read_text(encoding="utf-8")
BENCH = (LANE / "p160_bench.py").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p160.md").read_text(encoding="utf-8")
LIB = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text(encoding="utf-8")
GUARD_H = 1.0
BOOT_S = 300          # the launcher's create-to-runner allowance (P155's and P156's, kept)
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
    return _load("p160_reduce_under_test", LANE / "p160_reduce.py")


def _run(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=300)


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p160.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name
    # the premise pieces are the ones P156 and P158 staged, byte for byte
    assert pins["test_k25_row_exact_gpu.py"] == _pins(B / "p156" / "staged-p156.sha256")["test_k25_row_exact_gpu.py"]
    assert pins["test_nf4_smallm_prmt_gpu.py"] == _pins(B / "p158" / "staged-p158.sha256")["test_nf4_smallm_prmt_gpu.py"]


SHAKE = LANE / "shakedown" / "shake_p160.json"
SHAKE_SHA256 = "3c6bf3df9e3a67b08635f040e1f1415399ebc9b59b5d754a21cde0307173ed2a"


def test_the_reducer_on_the_real_shakedown_record():
    """The real record first: the $0 A2000 shakedown's own bench record (sha-pinned) through the registered reducer. It
    must read SHAKEDOWN with every structural check passed, never graded; every arm kept T's bits and P2 kept P's on
    every rep of both blocks; the spacer ran before every timed replay; the clocks were read for every arm."""
    raw = SHAKE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHAKE_SHA256
    rec = json.loads(raw)
    red = _reducer()
    v = red.verdict(rec)
    assert v["verdict"] == "SHAKEDOWN" and v["why"] == ["a shakedown record is never a reading"], v["why"]
    assert set(v["predictions"].values()) == {"UNREAD (a shakedown)"}
    assert rec["cc"] == [8, 6] and rec["layers"] == 4 and rec["triton"] == "3.4.0" and rec["torch"] == "2.8.0+cu128"
    assert v["contract_findings"] == []
    for b in red.BLOCKS:
        reps = rec["blocks"][b]["reps"]
        assert len(reps) == 6 and rec["blocks"][b]["tile_faults"] == []
        assert all(rep["p2_is_p"] is True and all(rep["identical"].values()) and rep["spacer_replays"] == 60
                   for rep in reps)
        assert all(rec["clocks"][b][a]["n"] > 100 for a in red.ARMS)
    assert rec["clocks"]["error"] is None
    # sm_86's compile of the carried plans (the box reads sm_120's; P156 recorded C1 at 194 and 127 there)
    assert rec["compiled"]["P"]["gu"] == [{"n_regs": 64, "n_spills": 0, "shared": 6400, "num_warps": 4, "num_stages": 3}]
    assert rec["compiled"]["C1"]["gu"][0]["shared"] == 17408 and rec["compiled"]["C1"]["dn"][0]["shared"] == 26624
    bad = json.loads(raw)                                        # a mutant on the real record: a rep out of its row
    bad["blocks"]["S1"]["reps"][2]["order"] = bad["blocks"]["S1"]["reps"][3]["order"]
    assert red.verdict(bad)["verdict"] == "VOID"


#: P156's box confirmation (p156-5090-1): 64 reps x 6 arms of the same graphs (both GEMMs, 48 layers, 10 replays each)
P156_CONFIRM_S, P156_GRAPHS = 60.8, 64 * 6


def test_need_bench_is_p156s_box_walls_through_this_shape_times_one_and_a_half():
    """NEED_BENCH from a measured box wall, x1.5: P156's confirmation graphs on an RTX 5090 (the same call form, stores
    and arms' shapes), each graph's host cost being its wall less its 10 replays at the arms' recorded medians; this lane
    captures 7 graphs a rep (the spacer and six arms) and replays 60 timed and 60 spacer (P's time) a rep."""
    p156 = json.loads((B / "p156" / "receipts" / "p156-5090-1" / "bench_p156.json").read_text(encoding="utf-8"))
    assert p156["walls"]["confirm_s"] == P156_CONFIRM_S and len(p156["confirm"]["reps"]) * 6 == P156_GRAPHS
    import statistics
    med = {a: statistics.median(statistics.median(r["replays_ms"][a]) for r in p156["confirm"]["reps"])
           for a in p156["confirm"]["arms"]}
    mean_ms = statistics.fmean(med.values())
    host_s = P156_CONFIRM_S / P156_GRAPHS - 10 * mean_ms / 1e3
    red = _reducer()
    reps = sum(len(s["steps"]) for s in red.BLOCKS.values())
    per_rep = 7 * host_s + 60 * mean_ms / 1e3 + 60 * med["P"] / 1e3
    # the compiles, sourced: P156's slowest first call of a prmt variant on the same card class and toolchain, for each
    # of this lane's 12 fresh variants; P156's routing, stores and two tile passes
    first = [r["first_call_s"] for p in ("gu", "dn") for r in p156["select"][p] if r.get("first_call_s") is not None]
    w = p156["walls"]
    setup = w["routing_s"] + w["stores_s"] + 2 * w["tiles_s"]
    box = reps * per_rep + setup + 12 * max(first)
    assert reps == 186 and round(host_s, 3) == 0.063 and round(per_rep, 2) == 1.67 and round(reps * per_rep, 1) == 310.7
    assert len(first) == 288 and max(first) == 2.65 and round(statistics.median(first), 2) == 0.25 and round(setup, 1) == 3.6
    assert round(box) == 346 and round(box * 1.5) == 519
    need = int(re.search(r"^NEED_BENCH=(\d+)", RUN, re.M).group(1))
    assert need == math.ceil(box * 1.5 / 100) * 100 == 600
    for figure in ("**60.8 s**", "slowest **2.65 s**", "12 x 2.65 = 31.8 s", "About 346 s; x1.5 = 519 s"):
        assert figure in PREREG, figure


def test_the_router_replay_is_the_committed_record():
    red = _reducer()
    raw = (RR / "rr.json.pt").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == red.RR_SHA256 == _pins(LANE / "staged-p160.sha256")["rr.json.pt"]
    assert f"{sha}  ./rr.json.pt" in (RR / "SHA256SUMS").read_text(encoding="ascii")  # P153's own checksum list
    for name in ("S0", "S1"):
        assert red.ROWS_SHA256[name][:8] in PREREG
    assert "350,529" in PREREG and "350,868" in PREREG


def test_both_streams_pass_every_check_stated_before_data():
    """The real record first: the box's own loader on the committed file, and P153's totals out of it."""
    torch = pytest.importorskip("torch")
    bench = _load("p160_bench_under_test", LANE / "p160_bench.py")
    red = bench.R
    picks, shas = bench.stream_picks(str(RR / "rr.json.pt"))
    assert shas == red.ROWS_SHA256
    for name in ("S0", "S1"):
        assert tuple(picks[name].shape) == (48, 16, 128, 8) and picks[name].dtype == torch.int16
        assert sum(v for row in bench.n_real_table(picks[name], 48) for v in row) == red.DISTINCT_TOTAL[name]
    # S1 is partly S0's: the rows whose picks equal S0's at every layer, stated in the registration
    same = (picks["S0"].sort(-1).values == picks["S1"].sort(-1).values).all(-1).all(0)   # [16 rows, 128 steps]
    assert round(float(same.float().mean()), 3) == 0.433 and int(same.all(0).sum()) == 0
    assert "43.3 %" in PREREG and "no step has all 16 rows identical" in PREREG
    band = lambda steps: round(100 * float(same[:, steps].float().mean()), 1)          # noqa: E731
    assert [band(list(range(lo, lo + 16))) for lo in range(0, 128, 16)] == [68.8, 59.8, 50.0, 44.5, 38.3, 32.0, 28.1, 25.0]
    sel = list(range(0, 64, 8))
    assert band(sel) == 57.0 and [int(same[:, s].sum()) for s in sel] == [11, 11, 11, 9, 8, 8, 8, 7]
    assert band([s for s in range(126) if s not in sel]) == 42.7 and band(list(range(64, 126))) == 31.0
    for figure in ("| 68.8 % | 59.8 % | 50.0 % | 44.5 % | 38.3 % | 32.0 % | 28.1 % | 25.0 % |", "**57.0 %**", "42.7 %",
                   "**31.0 %**"):
        assert figure in PREREG, figure
    red = bench.R
    assert red.P156_SELECT_STEPS == tuple(range(0, 64, 8))


def test_the_routing_checks_refuse_what_they_name(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    bench = _load("p160_bench_under_test", LANE / "p160_bench.py")
    d = torch.load(str(RR / "rr.json.pt"), weights_only=True)
    with pytest.raises(SystemExit, match="sha256 is not the pinned"):
        other = tmp_path / "other.pt"
        torch.save({"picks": d["picks"]}, str(other))
        bench.stream_picks(str(other))
    picks = d["picks"].clone()
    picks[3, 20, 7, 1] = picks[3, 20, 7, 0]                                          # an S1 row picks an expert twice
    dup = tmp_path / "dup.pt"
    torch.save({"picks": picks}, str(dup))
    monkeypatch.setattr(bench.R, "RR_SHA256", hashlib.sha256(dup.read_bytes()).hexdigest())
    monkeypatch.setattr(bench.R, "ROWS_SHA256", {n: hashlib.sha256(picks[:, lo:lo + 16].contiguous().numpy().astype(
        "<i2").tobytes()).hexdigest() for n, lo in (("S0", 0), ("S1", 16))})
    with pytest.raises(SystemExit, match="S1: a row picks one expert twice"):
        bench.stream_picks(str(dup))


def test_the_carried_plans_are_p156s_record_never_rechosen():
    red = _reducer()
    p156 = json.loads((B / "p156" / "receipts" / "p156-5090-1" / "bench_p156.json").read_text(encoding="utf-8"))
    assert red.CARRIED == p156["carried"]
    p156r = _load("p156_reduce_for_p160", B / "p156" / "p156_reduce.py")
    for name in ("RR_SHA256", "SHAPES", "TODAY", "T_LUT", "P_LUT", "ITERS", "SELF_BAND", "BOOT_B", "E", "TOPK", "LAYERS"):
        assert getattr(red, name) == getattr(p156r, name), name
    assert red.ROWS_SHA256["S0"] == p156r.S0_SHA256 and red.DISTINCT_TOTAL["S0"] == p156r.S0_DISTINCT_TOTAL
    assert set(red.BLOCKS["S0R"]["steps"]) <= set(p156r.CONFIRM_STEPS)          # P156's held-out steps only
    assert not set(red.BLOCKS["S0R"]["steps"]) & set(p156r.SELECT_STEPS)
    assert red.BOOT_SEED == 160 and red.GO_BAR == 1.05
    # the figure re-asked: P156's reported C1 over P on its held-out steps, recomputed from its record
    import statistics
    q = [statistics.median(r["replays_ms"]["P"]) / statistics.median(r["replays_ms"]["C1"]) for r in p156["confirm"]["reps"]]
    assert round(statistics.median(q), 4) == red.P156_C1_OVER_P == 1.6167


def test_only_the_plan_or_only_the_decode_moves_against_p():
    red = _reducer()
    P = red.PLANS["P"]
    assert P == (red.TODAY, red.TODAY, "prmt") and red.PLANS["P2"] == P
    for i in range(3):
        c = red.PLANS[f"C{i + 1}"]
        assert c[2] == "prmt" and c[:2] == (red.CARRIED[i]["gu"], red.CARRIED[i]["dn"]) and c[:2] != P[:2]
    assert red.PLANS["T"] == (red.TODAY, red.TODAY, "tree")
    assert red.ARMS == ["P", "P2", "C1", "C2", "C3", "T"] and red.READING == "S1"


def test_todays_plan_is_the_served_plan_and_p_the_served_default():
    pytest.importorskip("torch")
    from experts4bit_qlora.engines import hot_residency as hr
    red = _reducer()
    assert hr._K25_PLAN == dict(red.TODAY, lut=red.T_LUT, dot_bf16=False)
    assert hr.K25_LUT_DEFAULT_FAMILIES == {"qwen3_moe": "prmt"}                  # P158's FLIP (e4b#1868)
    assert '_r = hr.resolve_k25_lut("qwen3_moe")' in RUN
    assert '(_r["env"], _r["lut"], _r["source"]) == (None, "prmt", "default-allowlisted")' in RUN
    assert '"block_n": 32, "kc": 64, "warps": 4, "stages": 3, "lut": "prmt", "dot_bf16": False}' in RUN


def test_the_bench_calls_k25_as_the_served_route():
    assert "return _k25(_tok[0].to(torch.bfloat16), pk, am, t_row0, t_rows, t_grp, order,\n" \
           "                            gather_div=_tok[1], **_k25_kw)" in LIB
    assert 'kw = {"scatter": order} if (_lean and pk is dn_p) else {}' in LIB
    assert "_fused_tiles(\n                local_ids, _n_exp, 16, lean=True, sorted_ids=True)" in LIB
    assert "k25(X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK, lut=lut, dot_bf16=False, **plan)" in BENCH
    assert "k25(H[li], p, am, row0, rows, grp, None, scatter=order, lut=lut, dot_bf16=False, **plan)" in BENCH
    assert BENCH.count("lut=lut, dot_bf16=False, **plan)") == 2 and not re.search(r'k25\([^\n]*lut="', BENCH)
    assert "build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)" in BENCH
    assert "any(v <= 0 for v in rv[:n]) or any(v != 0 for v in rv[n:])" in BENCH


def test_the_runner_pins_k25_and_k39_and_refuses_before_the_install():
    first_install = RUN.index('say "install e4b @')
    for refusal in ('[ "$PL" -ge "$MIN_POWER_W" ] || {', '[ "$CC" = 12.0 ] || {', '[ "$LEFT" -ge "$FIT_NEED" ] || {',
                    '[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || {'):
        assert RUN.index(refusal) < first_install, refusal
    floor = RUN.index('[ "$PL" -ge "$MIN_POWER_W" ] || {')
    assert "MIN_POWER_W=575" in RUN and "finish 17; }" in RUN[floor:floor + 260]
    assert "${P160_MIN_POWER_W:-}" in RUN and "${P160_NEED_BENCH:-}" in RUN            # frozen reading knobs
    gnf4 = re.search(r"^GNF4_SHA=([0-9a-f]{40}) ", RUN, re.M)
    assert gnf4 and gnf4.group(1) == "41ef086e6a3e2be5b158ff1bb38e0de67862a949" and gnf4.group(1)[:8] in PREREG
    assert gnf4.group(1) in (B / "p158" / "p158_run.sh").read_text(encoding="utf-8")   # P158's release pin
    assert 'md.version("grouped-nf4-gemm") == "0.50.0"' in RUN
    for sha in ("ee46969a4f786c36abe26bfaf0a52e426db4465406a5e31c6085fd3270b26568",    # nf4_smallm.py: K25 + K39
                "773eff5db7be77dea3d1f9048a6e708a94e79634baf17f452bf2848e24a2072f",    # int4_b32.py
                "a7090915485c6e9943f568e79315b3f533c754f24edbdb7160b05dcc18619a23"):   # nf4_grouped.py: K39's tables
        assert sha in RUN and sha[:8] in PREREG
    assert 'nf4_smallm._LUT_MODES.get("prmt") == 4' in RUN
    assert "python -m pytest test_k25_row_exact_gpu.py test_nf4_smallm_prmt_gpu.py" in RUN
    assert "grep -qE '^27 passed(, [0-9]+ warnings?)? in" in RUN
    order = [RUN.index(x) for x in ('say "install e4b @', "python -m pytest test_k25_row_exact_gpu.py",
                                    "can_run $NEED_BENCH bench", "python $W/p160_bench.py --rr",
                                    "python $W/p160_reduce.py --bench")]
    assert order == sorted(order)
    assert "ncu" not in RUN.lower()                                                    # no Nsight step in this lane
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_NF4_GROUPED_SMALLM", "E4B_INT4_LEAN_GLUE", "TRITON_INTERPRET", "E4B_INT4_WIDE_TILES"):
        assert re.search(rf"\b{knob}\b", unset), knob


def test_the_guard_fits_with_25_percent_headroom():
    k = {n: int(v) for n, v in re.findall(r"^(INSTALL_TO_GATE_S|NEED_BENCH|NEED_NCU|RESERVE)=(\d+)", RUN, re.M)}
    assert k == {"INSTALL_TO_GATE_S": 600, "NEED_BENCH": 600, "RESERVE": 600}, k
    assert "FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100))" in RUN
    assert re.search(r"^FIT_HEADROOM_PCT=125 ", RUN, re.M)
    fit = sum(k.values())
    assert fit * 125 // 100 == 2250 and "`FIT_NEED` = 1800 x 1.25 = 2250 s" in PREREG
    assert (BOOT_S + fit) * HEADROOM <= GUARD_H * 3600
    assert "2100 x 1.25 = 2625 s <= 3600 s (1.0 h)" in PREREG and "**guard 1.0 h**" in PREREG
    assert "1.0 x $1.05 + 10 x $0.011 = **$1.16**" in PREREG
    assert GUARD_H * 3600 - fit * 125 // 100 == 1350 and "3600 - 2250 = 1350 s to reach it" in PREREG
    line = next(x for x in RUN.splitlines() if x.startswith("can_run(){"))
    assert "fit=$((need + RESERVE))" in line and "needs ${fit}s" in line and "only ${left}s left" in line


def test_no_instance_id_or_host_name_reaches_the_box():
    for name, text in (("runner", RUN), ("bench", BENCH), ("reducer", (LANE / "p160_reduce.py").read_text("utf-8"))):
        for marker in ("gethostname", "socket.", "INSTANCE_ID", "hostname", "nvmlDeviceGetUUID", "nvmlDeviceGetSerial"):
            assert marker not in text, (name, marker)
    pass_line = next(x for x in DRIVE.splitlines() if x.startswith("PASS="))
    assert "INSTANCE" not in pass_line and "P160_DEADLINE_EPOCH=$DEADLINE" in pass_line
    assert "HF_TOKEN" not in DRIVE                                                       # nothing is fetched


@pytest.mark.parametrize("script", ["p160_reduce.py", "p160_bench.py"])
def test_the_self_test_passes(script):
    out = _run(LANE / script, "--self-test")
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


REDUCE_MUTANTS = [
    # GO on the median alone, without the 2-SE guard or the clock bound
    ('    elif c1["lb"] >= GO_BAR and c1["lb_clock_adj"] >= GO_BAR:\n', '    elif c1["median"] >= GO_BAR:\n'),
    # the clock bound dropped
    ('    elif c1["lb"] >= GO_BAR and c1["lb_clock_adj"] >= GO_BAR:\n', '    elif c1["lb"] >= GO_BAR:\n'),
    # a SLOWER-clocked C1 credited with the clock difference (the bound is one-sided)
    ('        s["lb_clock_adj"] = s["lb"] * min(1.0, base / clk)\n', '        s["lb_clock_adj"] = s["lb"] * (base / clk)\n'),
    # the median's lower bound taken with the MEAN's standard error
    ("    se = se_median(q) if n > 1 else math.inf\n", "    se = statistics.stdev(q) / math.sqrt(n) if n > 1 else math.inf\n"),
    # GO read on a reported arm
    ('    c1 = reads[READING]["C1"]\n', '    c1 = reads[READING]["C2"]\n'),
    # GO read on the reproduction block
    ('READING = "S1"\n', 'READING = "S0R"\n'),
    # the self-pair band not enforced
    ('    if not SELF_BAND[0] <= sp["median"] <= SELF_BAND[1]:\n', "    if False:\n"),
    # P allowed to replay different bits in a second capture
    ('            if rep["p2_is_p"] is not True:\n', "            if False:\n"),
    # a C1 or P that moved bits ruled on as a timing
    ("    elif c1_moved:\n", "    elif False:\n"),
    # the Williams row not checked
    ('            if rep.get("step") != steps[r] or rep.get("order") != arm_order(r):\n',
     '            if rep.get("step") != steps[r]:\n'),
    # the spacer not checked
    ('            if rep.get("spacer_replays") != ITERS * len(ARMS):\n', "            if False:\n"),
    # a timed replay under a reduced enforced limit accepted
    ("        if any(not _num(w) or w < MIN_POWER_W for w in lim):\n", "        if False:\n"),
    # unread clocks accepted
    ('    if clk.get("error") or not all(', '    if False and not all('),
    # any card read as the 5090
    ('        if "5090" not in str(rec.get("device", "")) or rec.get("cc") != [12, 0]:\n', "        if False:\n"),
    # the streams' picks not checked
    ('    if rec.get("rows_sha256") != ROWS_SHA256:\n', "    if False:\n"),
    # the memory-clock label dropped
    ("            if abs(mclk / mbase - 1) > MEM_CLOCK_TOL:\n", "            if False:\n"),
    # the late split read off the whole block (it would not move when the early steps carry the headline)
    ('S1_SPLITS = {"steps_64_125": lambda s: s >= 64,', 'S1_SPLITS = {"steps_64_125": lambda s: True,'),
]
BENCH_MUTANTS = [
    # gate_up without the served gather
    ("k25(X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK,", "k25(X[li], p, am, row0, rows, grp, order,"),
    # down without the served scatter
    ("k25(H[li], p, am, row0, rows, grp, None, scatter=order,", "k25(H[li], p, am, row0, rows, grp, None,"),
    # the spacer is not P's work (a light predecessor)
    ('pg, pd, lut = R.PLANS["P" if arm == R.SPACER else arm]', 'pg, pd, lut = R.PLANS["C1" if arm == R.SPACER else arm]'),
    # no spacer before the timed replay
    ("                        graphs[R.SPACER][0].replay()      # untimed: every timed replay follows the same work\n",
     ""),
    # the sampler tags after the replay
    ("                        sampler.tag = (b, arm)\n                        rp[arm].append(replay_ms(graphs[arm][0]))\n",
     "                        rp[arm].append(replay_ms(graphs[arm][0]))\n                        sampler.tag = (b, arm)\n"),
]


@pytest.mark.parametrize("script,old,new", [("p160_reduce.py", o, n) for o, n in REDUCE_MUTANTS]
                         + [("p160_bench.py", o, n) for o, n in BENCH_MUTANTS])
def test_each_mutant_fails_the_self_test(tmp_path, script, old, new):
    src = (LANE / script).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / script).write_bytes(src.replace(old, new).encode("utf-8"))
    if script == "p160_bench.py":                                    # the bench imports its sibling reducer
        shutil.copy(LANE / "p160_reduce.py", tmp_path / "p160_reduce.py")
    out = _run(tmp_path / script, "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P160_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p160_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p160-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p160" in out.stdout and "bash p160_run.sh" in out.stdout
    assert "INSTANCE" not in out.stdout


def test_unregistered_run_ids_are_refused(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree")
    out = subprocess.run(["bash", str(LANE / "p160_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p160-5090-4"), timeout=120)
    assert out.returncode == 78 and "unregistered run ID" in out.stdout
    assert "p160-5090-1|p160-5090-2|p160-5090-3) :;;" in RUN
