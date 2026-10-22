# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P222, K25's decode x plan at Mixtral-8x7B's W16 shapes (bench/p222/PREREG-p222.md). The staged pins; the seeded
routing (both blocks' draws, pinned, with the statistics the registration states); the arms, with only the plan or only
the decode moving against T, Mixtral's served default; c1 is e4b's registered pair; the call form against the served
route; the runner's pins, refusals (and the shakedown's narrower ones), tripwire, premise and guard fit with 25 %
headroom, from P156's and P145's records; no instance id or host name on the box; the self-tests and the mutants they
must kill; the run ids."""
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p222"
SOURCES = {"p222_run.sh": LANE, "p222_bench.py": LANE, "p222_reduce.py": LANE,
           "test_k25_row_exact_gpu.py": REPO / "tests", "test_nf4_smallm_prmt_gpu.py": B / "p158" / "premise"}
RUN = (LANE / "p222_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p222_drive.sh").read_text(encoding="utf-8")
BENCH = (LANE / "p222_bench.py").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p222.md").read_text(encoding="utf-8")
LIB = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text(encoding="utf-8")
GUARD_H = 1.5
BOOT_S = 300          # the launcher's create-to-runner allowance (P155's, P156's and P160's, kept)
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
    return _load("p222_reduce_under_test", LANE / "p222_reduce.py")


def _run(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=300)


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p222.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name
    # the premise pieces are the ones P160 staged, byte for byte
    p160 = _pins(B / "p160" / "staged-p160.sha256")
    for name in ("test_k25_row_exact_gpu.py", "test_nf4_smallm_prmt_gpu.py"):
        assert pins[name] == p160[name], name


def test_the_seeded_routing_is_pinned_and_as_stated():
    red = _reducer()
    for b, seed in red.SEEDS.items():
        pk = red.picks(seed)
        assert red.route_sha256(pk) == red.ROUTE_SHA256[b] and red.ROUTE_SHA256[b][:8] in PREREG
        assert len(pk) == 8 and all(len(layer) == 128 and all(len(step) == 16 for step in layer) for layer in pk)
        assert all(len(set(row)) == 2 and all(0 <= e < 8 for e in row) for layer in pk for step in layer for row in step)
    assert red.SEEDS == {"A": 2220, "B": 2221} and "seed 2220" in PREREG and "seed 2221" in PREREG
    # the reading block's distinct experts per layer and step, stated in the registration
    flat = [v for step in red.n_real(red.picks(red.SEEDS["B"])) for v in step]
    assert Counter(flat) == {8: 944, 7: 76, 6: 4} and round(statistics.fmean(flat), 3) == 7.918
    assert "944 of B's 1,024 (layer, step) cells touch all 8 experts" in PREREG and "**7.918**" in PREREG
    # the expected count under uniform top-2 of 8 at 16 rows: 8 x (1 - (21/28)^16)
    assert round(8 * (1 - (21 / 28) ** 16), 3) == 7.920 and "7.920" in PREREG


def test_the_sweep_and_the_arms():
    red = _reducer()
    p156 = _load("p156_reduce_for_p222", B / "p156" / "p156_reduce.py")
    assert red.GRID == p156.GRID and len(red.GRID) == 144                        # P156's grid, unchanged
    assert red.TODAY == p156.TODAY and red.SELECT_ITERS == 5 and red.ITERS == 10
    assert red.SELECT_STEPS == tuple(range(0, 128, 16)) and red.READ_STEPS == tuple(range(120))
    assert sorted(red.sweep_order()) == list(range(144)) and red.ORDER_SEED == 222
    m = {"gu": red.GRID[100], "dn": red.GRID[7]}
    P = red.plans_for(m)
    assert P["T"] == P["T2"] == (red.TODAY, red.TODAY, "tree")
    assert P["P"] == (red.TODAY, red.TODAY, "prmt")                                  # only the decode moves
    assert P["C1"] == (red.C1["gu"], red.C1["dn"], "prmt") and P["M"] == (m["gu"], m["dn"], "prmt")
    assert "TM" not in P                                                            # Amendment 1 dropped it
    assert red.ARMS == ["T", "T2", "P", "C1", "M"] and red.GO_BAR == 1.10 and red.TIE_BAR == 1.02
    w = red.williams(len(red.ARMS))
    assert len(w) == 10 and len(red.READ_STEPS) % len(w) == 0                      # 12 whole squares of 10 orders


def test_c1_is_e4bs_registered_pair_and_mixtral_serves_the_tree_at_todays_plan():
    red = _reducer()
    c1 = re.search(r'"c1": \((\{[^}]*\}), (\{[^}]*\})\)', LIB)
    assert c1 and (json.loads(c1.group(1)), json.loads(c1.group(2))) == (red.C1["gu"], red.C1["dn"])
    assert re.search(r'^_K25_PLAN = \{"block_n": 32, "kc": 64, "warps": 4, "stages": 3, "lut": "tree", "dot_bf16": False\}',
                     LIB, re.M)
    lut = re.search(r"^K25_LUT_DEFAULT_FAMILIES: dict = \{(.*?)\n\}", LIB, re.M | re.S)
    plan = re.search(r"^K25_PLAN_DEFAULT_FAMILIES: dict = \{(.*?)\n\}", LIB, re.M | re.S)
    assert lut and plan and "mixtral" not in lut.group(1) and "mixtral" not in plan.group(1)   # T is the served default
    assert '_r = hr.resolve_k25_lut("mixtral")' in RUN
    assert '(_r["env"], _r["lut"]) == (None, "tree") and _r["source"].startswith("default-off")' in RUN
    assert '_p = hr.resolve_k25_plan(_r, model_type="mixtral")' in RUN and '(_p["env"], _p["plan"]) == (None, "today")' in RUN
    assert 'hr._K25_PLANS["c1"] == (R.C1["gu"], R.C1["dn"])' in RUN


def test_an_arm_that_cannot_compile_or_capture_is_recorded_never_a_crash():
    """Amendment 1 (p222-shake-1 crashed on an arm whose compile ran out of shared memory): the compiled-metadata pass
    and block B's captures record the fault in the bench record and end the bench; the reducer VOIDs with the reason."""
    assert BENCH.count('rec["arm_fault"] = ') == 2
    assert "except (Poisoned, ArmFault) as e:" in BENCH and "return 32 if isinstance(e, Poisoned) else 34" in BENCH
    assert 'if rec.get("arm_fault"):' in (LANE / "p222_reduce.py").read_text(encoding="utf-8")


def test_the_bench_calls_k25_as_the_served_route():
    assert "return _k25(_tok[0].to(torch.bfloat16), pk, am, t_row0, t_rows, t_grp, order,\n" \
           "                            gather_div=_tok[1], **_k25_gu_kw)" in LIB      # main at this registration
    assert 'kw = {"scatter": order} if (_lean and pk is dn_p) else {}' in LIB
    assert "_fused_tiles(\n                local_ids, _n_exp, 16, lean=True, sorted_ids=True)" in LIB
    assert "k25(X[li], p, am, row0, rows, grp, order_, gather_div=R.TOPK, lut=lut, dot_bf16=False, **plan)" in BENCH
    assert "k25(H[li], p, am, row0, rows, grp, None, scatter=order_, lut=lut, dot_bf16=False, **plan)" in BENCH
    assert BENCH.count("lut=lut, dot_bf16=False, **plan)") == 2 and not re.search(r'k25\([^\n]*lut="', BENCH)
    assert "build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)" in BENCH
    red = _reducer()
    assert red.SHAPES == {"gu": (28672, 4096), "dn": (4096, 14336)} and (red.E, red.TOPK, red.TOKENS) == (8, 2, 16)


def test_the_runner_pins_k25_and_k39_and_refuses_before_the_install():
    first_install = RUN.index('say "install e4b @')
    for refusal in ('[ "$SHAKE" = 1 ] || [ "$PL" -ge "$MIN_POWER_W" ] || {', '[ "$CC" = 12.0 ] || {',
                    '[ "$LEFT" -ge "$FIT_NEED" ] || {', '[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || {',
                    '[ "${VRAM_MIB:-0}" -ge "$VRAM_FLOOR" ] || {'):
        assert RUN.index(refusal) < first_install, refusal
    assert "MIN_POWER_W=575" in RUN and "${P222_MIN_POWER_W:-}" in RUN and "${P222_NEED_BENCH:-}" in RUN
    gnf4 = re.search(r"^GNF4_SHA=([0-9a-f]{40}) ", RUN, re.M)
    assert gnf4 and gnf4.group(1) == "41ef086e6a3e2be5b158ff1bb38e0de67862a949" and gnf4.group(1)[:8] in PREREG
    p160 = (B / "p160" / "p160_run.sh").read_text(encoding="utf-8")
    for sha in ("ee46969a4f786c36abe26bfaf0a52e426db4465406a5e31c6085fd3270b26568",
                "773eff5db7be77dea3d1f9048a6e708a94e79634baf17f452bf2848e24a2072f",
                "a7090915485c6e9943f568e79315b3f533c754f24edbdb7160b05dcc18619a23"):
        assert sha in RUN and sha in p160 and sha[:8] in PREREG                    # P160's kernel files, kept
    assert 'nf4_smallm._LUT_MODES.get("prmt") == 4' in RUN
    assert "grep -qE '^27 passed(, [0-9]+ warnings?)? in" in RUN
    order = [RUN.index(x) for x in ('say "install e4b @', "python -m pytest test_k25_row_exact_gpu.py",
                                    "can_run $NEED_BENCH bench", "python $W/p222_bench.py --out",
                                    "python $W/p222_reduce.py --bench")]
    assert order == sorted(order)
    assert "ncu" not in RUN.lower() and "rr.json" not in RUN + DRIVE
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_NF4_GROUPED_SMALLM", "E4B_INT4_LEAN_GLUE", "TRITON_INTERPRET", "E4B_INT4_WIDE_TILES"):
        assert re.search(rf"\b{knob}\b", unset), knob


def test_the_shakedown_is_never_a_reading():
    """p222-shake-N runs on any sm_86+ card with 15 GB, with no power floor, at the shakedown's knobs, and it finishes 0
    only on the reducer's SHAKEDOWN; a 5090 run id keeps the class, cc and power refusals and finishes 0 only on GO or
    NO_GO."""
    assert "p222-shake-1|p222-shake-2|p222-shake-3) SHAKE=1;;" in RUN          # shake-3: Amendment 2
    assert "8.6|8.9|9.0|10.0|12.0) ;;" in RUN and "VRAM_FLOOR=30000; [ \"$SHAKE\" = 1 ] && VRAM_FLOOR=15000" in RUN
    assert '--shakedown --layers 2 --reps 10 --grid 6 --select-steps 2' in RUN
    assert 'case "$SHAKE/$V" in 0/GO|0/NO_GO|1/SHAKEDOWN) finish 0;;' in RUN
    assert ("p222-5090-1|p222-5090-2|p222-5090-3|p222-5090-4|p222-5090-5|p222-5090-6|p222-shake-1|p222-shake-2|"
            "p222-shake-3) :;;") in DRIVE                                     # -4..-6: Amendment 3


#: P156's box sweep (p156-5090-1): 288 plan x projection rows, each a first call; P145's per-name record: K25 per call
#: at Mixtral's W16 shapes under the tree at today's plan (p145-5090-1, pername_b16.json)
def test_need_bench_is_derived_from_p156s_and_p145s_records_times_one_and_a_half():
    p156 = json.loads((B / "p156" / "receipts" / "p156-5090-1" / "bench_p156.json").read_text(encoding="utf-8"))
    p145 = json.loads((B / "p145" / "receipts" / "p145-5090-1" / "pername_b16.json").read_text(encoding="utf-8"))
    red = _reducer()
    first = [r["first_call_s"] for p in ("gu", "dn") for r in p156["select"][p] if r.get("first_call_s") is not None]
    assert len(first) == 288 and max(first) == 2.65
    rows = {r["grid"]: r for r in p145["rows"] if r["name"] == "_gemm_nf4_grouped_smallm"}
    call = {"gu": rows["10x896x1"]["dur_ms"] / rows["10x896x1"]["n_per_step"],
            "dn": rows["10x128x1"]["dur_ms"] / rows["10x128x1"]["n_per_step"]}
    assert round(call["gu"], 3) == 0.924 and round(call["dn"], 3) == 0.592

    def sel_ms(r):
        return statistics.fmean(statistics.median(x) for x in r["step_replays_ms"])
    slow = {}
    for p in ("gu", "dn"):
        rr = [r for r in p156["select"][p] if r.get("step_replays_ms")]
        today = next(r for r in rr if r["plan"] == red.TODAY)
        slow[p] = statistics.fmean(sel_ms(r) / sel_ms(today) for r in rr)
    assert round(slow["gu"], 3) == 1.155 and round(slow["dn"], 3) == 1.739
    host = 0.063                                     # P160's per-graph host cost from P156's confirmation (test_p160)
    L, n_sel = red.LAYERS, len(red.SELECT_STEPS)
    compiles = len(first) * max(first)               # every compile at P156's slowest first call
    sweep_gpu = sum(144 * n_sel * (2 + red.SELECT_ITERS) * L * call[p] * slow[p] / 1e3 for p in ("gu", "dn"))
    sweep_host = len(first) * n_sel * host
    t_ms = L * (call["gu"] + call["dn"])             # T's graph: every arm taken at the tree's time (conservative)
    graphs = len(red.ARMS) + 1                       # the spacer and every arm, captured each rep
    per_rep = graphs * host + (graphs * 2 + 2 * len(red.ARMS) * red.ITERS) * t_ms / 1e3
    read = len(red.READ_STEPS) * per_rep
    meta = 2 * len(red.ARMS) * max(first)
    box = compiles + sweep_gpu + sweep_host + read + meta
    assert round(compiles, 1) == 763.2 and round(sweep_gpu, 1) == 135.2 and round(sweep_host, 1) == 145.2
    assert round(t_ms, 2) == 12.13 and round(per_rep, 3) == 1.736 and round(read, 1) == 208.3 and round(meta, 1) == 26.5
    assert round(box) == 1278 and round(box * 1.5) == 1918
    need = int(re.search(r"^NEED_BENCH=(\d+)", RUN, re.M).group(1))
    assert need == 2100 >= math.ceil(box * 1.5 / 100) * 100 == 2000     # Amendment 1 keeps the registered, larger bound
    for figure in ("288 x 2.65 = 763.2 s", "135.2 s", "145.2 s", "120 x 1.736 = 208.3 s", "10 x 2.65 = 26.5 s",
                   "about 1,278 s; x1.5 = 1,918 s"):
        assert figure in PREREG, figure


def test_the_guard_fits_with_25_percent_headroom():
    k = {n: int(v) for n, v in re.findall(r"^(INSTALL_TO_GATE_S|NEED_BENCH|NEED_NCU|RESERVE)=(\d+)", RUN, re.M)}
    assert k == {"INSTALL_TO_GATE_S": 600, "NEED_BENCH": 2100, "RESERVE": 600}, k
    assert "FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100))" in RUN
    fit = sum(k.values())
    assert fit * 125 // 100 == 4125 and "`FIT_NEED` = 3300 x 1.25 = 4125 s" in PREREG
    assert (BOOT_S + fit) * HEADROOM <= GUARD_H * 3600
    assert "3600 x 1.25 = 4500 s <= 5400 s (1.5 h)" in PREREG and "**guard 1.5 h**" in PREREG
    assert "1.5 x $1.05 + 10 x $0.011 = **$1.69**" in PREREG
    assert GUARD_H * 3600 - fit * 125 // 100 == 1275 and "5400 - 4125 = 1275 s to reach it" in PREREG
    assert "AL=$(step_alarm 2700)" in RUN and 2700 >= k["NEED_BENCH"]


def test_the_shakedown_fits_its_own_guard_at_its_stated_price():
    """A shakedown id takes its own NEED_BENCH, so its 0.75 h guard at <= $0.40/h passes the runner's fit check."""
    assert '[ "$SHAKE" = 1 ] && NEED_BENCH=600' in RUN
    order = [RUN.index(x) for x in ('p222-shake-1|p222-shake-2|p222-shake-3) SHAKE=1;;', '[ "$SHAKE" = 1 ] && NEED_BENCH=600',
                                    'FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + RESERVE))')]
    assert order == sorted(order)
    fit = 600 + 600 + 600
    assert fit * 125 // 100 == 2250 and (BOOT_S + fit) * HEADROOM == 2625 <= 0.75 * 3600
    for figure in ("**guard 0.75 h**", "0.75 x $0.40 + 10 x $0.011 = **$0.41**", "(300 + 1800) x 1.25 = 2625 s <= 2700 s"):
        assert figure in PREREG, figure


def test_no_instance_id_or_host_name_reaches_the_box():
    for name, text in (("runner", RUN), ("bench", BENCH), ("reducer", (LANE / "p222_reduce.py").read_text("utf-8"))):
        for marker in ("gethostname", "socket.", "INSTANCE_ID", "hostname", "nvmlDeviceGetUUID", "nvmlDeviceGetSerial"):
            assert marker not in text, (name, marker)
    pass_line = next(x for x in DRIVE.splitlines() if x.startswith("PASS="))
    assert "INSTANCE" not in pass_line and "P222_DEADLINE_EPOCH=$DEADLINE" in pass_line
    assert "HF_TOKEN" not in DRIVE


@pytest.mark.parametrize("script", ["p222_reduce.py", "p222_bench.py"])
def test_the_self_test_passes(script):
    out = _run(LANE / script, "--self-test")
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


REDUCE_MUTANTS = [
    # GO on the median alone, without the 2-SE guard or the clock bound
    ('    elif ma["lb"] >= GO_BAR and ma["lb_clock_adj"] >= GO_BAR:\n', '    elif ma["median"] >= GO_BAR:\n'),
    # the clock bound dropped
    ('    elif ma["lb"] >= GO_BAR and ma["lb_clock_adj"] >= GO_BAR:\n', '    elif ma["lb"] >= GO_BAR:\n'),
    # a SLOWER-clocked M credited with the clock difference (the bound is one-sided)
    ('        s["lb_clock_adj"] = s["lb"] * min(1.0, base / clk)', '        s["lb_clock_adj"] = s["lb"] * (base / clk)'),
    # the median's lower bound taken with the MEAN's standard error
    ("    se = se_median(q) if len(q) > 1 else math.inf\n",
     "    se = statistics.stdev(q) / math.sqrt(len(q)) if len(q) > 1 else math.inf\n"),
    # GO read on C1, the knob arm, not on the selected M
    ('    ma = rd["M"]\n', '    ma = rd["C1"]\n'),
    # the selection's winner not checked against the plan block B timed
    ('    elif rec.get("m") != m:\n', "    elif False:\n"),
    # the reading ignores the self-pair band (Amendment 2 keeps it ruling a reading)
    ("    if not inside and not shake:                 # Amendment 2: the band rules a reading; a shakedown reports it\n",
     "    if False:\n"),
    # the shakedown VOIDs on the self-pair band (Amendment 2 reports it there)
    ("    if not inside and not shake:                 # Amendment 2: the band rules a reading; a shakedown reports it\n",
     "    if not inside:\n"),
    # T allowed to replay different bits in a second capture
    ('        if ident["T2"] is not True:\n', "        if False:\n"),
    # a P, C1 or M that moved bits ruled on as a timing
    ("    elif moved:\n", "    elif False:\n"),
    # the Williams row not checked
    ('        if rep.get("step") != steps[r] or rep.get("order") != arm_order(r):\n',
     '        if rep.get("step") != steps[r]:\n'),
    # the spacer not checked
    ('        if rep.get("spacer_replays") != ITERS * len(ARMS):\n', "        if False:\n"),
    # a timed replay under a reduced enforced limit accepted
    ("        if any(not _num(w) or w < MIN_POWER_W for w in lim):\n", "        if False:\n"),
    # any card read as the 5090
    ('        if "5090" not in str(rec.get("device", "")) or rec.get("cc") != [12, 0]:\n', "        if False:\n"),
    # the routing draw not checked
    ('    if rec.get("route_sha256") != ROUTE_SHA256 and not shake:\n', "    if False:\n"),
    # the tie-break dropped: M served even within TIE_BAR of c1
    ("    if m == C1 or m_over_c1 < TIE_BAR:\n", "    if m == C1:\n"),
    # a recorded arm fault not read as a VOID (Amendment 1)
    ('    if rec.get("arm_fault"):\n', "    if False:\n"),
    # the sweep's order not checked
    ('    if not shake and (rec.get("select_steps") != list(SELECT_STEPS) or order != sweep_order()):\n',
     "    if False:\n"),
    # a plan that moved a bit eligible for selection
    ('    return (row.get("identical") is True and not row.get("error") and isinstance(sr, list)',
     '    return (not row.get("error") and isinstance(sr, list)'),
    # a plan that errored eligible for selection
    ('    return (row.get("identical") is True and not row.get("error") and isinstance(sr, list)',
     '    return (row.get("identical") is True and isinstance(sr, list)'),
    # a tie goes to the higher grid index
    ('select_ms(r["step_replays_ms"]), _GRID_INDEX[plan_key(r["plan"])]))',
     'select_ms(r["step_replays_ms"]), -_GRID_INDEX[plan_key(r["plan"])]))'),
    # the margin read against the slowest plan, not the runner-up
    ("        out[proj] = ts[1] / ts[0] if len(ts) > 1 else None\n", "        out[proj] = ts[-1] / ts[0] if len(ts) > 1 else None\n"),
]
BENCH_MUTANTS = [
    # Amendment 2: an arm whose variant was compiled earlier gets no metadata (no fallback)
    ("    if earlier and earlier[0]:\n", "    if False:\n"),
    # Amendment 2: block A's reference compile (the tree at today's plan, T and T2's cubin) not recorded
    ('        first[(p, R.plan_key(R.TODAY), R.T_LUT)] = (new_variants(kfn, before), "block A reference")\n', ""),
    # gate_up without the served gather
    ("k25(X[li], p, am, row0, rows, grp, order_, gather_div=R.TOPK,", "k25(X[li], p, am, row0, rows, grp, order_,"),
    # down without the served scatter
    ("k25(H[li], p, am, row0, rows, grp, None, scatter=order_,", "k25(H[li], p, am, row0, rows, grp, None,"),
    # the spacer is not T's work (a light predecessor)
    ('pg, pd, lut = plans["T" if arm == R.SPACER else arm]', 'pg, pd, lut = plans["M" if arm == R.SPACER else arm]'),
    # no spacer before the timed replay
    ("                    graphs[R.SPACER][0].replay()      # untimed: every timed replay follows the same work\n", ""),
    # the sweep under the tree
    ("                    outs = [call[proj](li, sel_tabs[sel_steps[0]][li], plan, R.P_LUT) for li in range(L)]",
     "                    outs = [call[proj](li, sel_tabs[sel_steps[0]][li], plan, R.T_LUT) for li in range(L)]"),
]


@pytest.mark.parametrize("script,old,new", [("p222_reduce.py", o, n) for o, n in REDUCE_MUTANTS]
                         + [("p222_bench.py", o, n) for o, n in BENCH_MUTANTS])
def test_each_mutant_fails_the_self_test(tmp_path, script, old, new):
    src = (LANE / script).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / script).write_bytes(src.replace(old, new).encode("utf-8"))
    if script == "p222_bench.py":                                    # the bench imports its sibling reducer
        shutil.copy(LANE / "p222_reduce.py", tmp_path / "p222_reduce.py")
    out = _run(tmp_path / script, "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P222_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


@pytest.mark.parametrize("run_id", ["p222-5090-1", "p222-shake-1"])
def test_the_drivers_dry_run_names_the_launch(tmp_path, run_id):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p222_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, run_id), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p222" in out.stdout and "bash p222_run.sh" in out.stdout
    assert "INSTANCE" not in out.stdout


def test_unregistered_run_ids_are_refused(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree")
    out = subprocess.run(["bash", str(LANE / "p222_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p222-5090-7"), timeout=120)
    assert out.returncode == 78 and "unregistered run ID" in out.stdout
    assert "p222-5090-1|p222-5090-2|p222-5090-3|p222-5090-4|p222-5090-5|p222-5090-6) :;;" in RUN


# ------------------------------------------------------------------ Amendment 2: the real p222-shake-2 record first
SHAKE2 = REPO / "tests" / "fixtures" / "p222_shake2"
SHAKE2_PINS = {"bench_p222.json": "434cc9753c9a0b5c7daa1918c7df068f08c8681686d0ee7290008af3c391251d",
               "verdict_p222.json": "f93575790e8dc89f047817a02f96219145bde5dc4dd82704fc2a7b8a9b5961ca"}


def _shake2(name):
    raw = (SHAKE2 / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHAKE2_PINS[name], name           # the receipt's bytes, verbatim
    return json.loads(raw)


def test_the_real_shake2_record_reads_shakedown_with_the_band_reported():
    """p222-shake-2's own record: the 035cedc4 reducer read it VOID on the self-pair band alone; the amended reducer
    reads the same record SHAKEDOWN, reports the band, and computes the same read."""
    rec, old = _shake2("bench_p222.json"), _shake2("verdict_p222.json")
    assert old["verdict"] == "VOID" and old["why"] == ["self-pair T/T2 median 1.0217 outside (0.99, 1.01)"]
    o = _reducer().verdict(rec)
    assert o["verdict"] == "SHAKEDOWN", o["why"]
    assert o["self_pair_band"] == {"band": [0.99, 1.01], "median": old["read"]["self_pair"]["median"],
                                   "inside": False, "rules": False}
    assert o["why"][1].startswith("self-pair T/T2 median 1.0217 outside (0.99, 1.01): reported, never ruling")
    assert o["read"] == old["read"] and o["m"] == old["m"]                        # only the ruling moved
    assert set(o["predictions"].values()) == {"UNREAD (a shakedown)"}
    reading = dict(rec, shakedown=False)
    assert _reducer().verdict(reading)["verdict"] == "VOID"                     # never a reading: structure still rules


def test_every_arm_carries_its_compiled_metadata_on_a_shakedown_shaped_record():
    """Amendment 2 (2), on shake-2's real record: the metadata pass's own rows (C1) and the sweep's first compiles (P,
    M) give those arms their registers, spills and shared memory; T and T2 take block A's reference compile, stated
    here because the 035cedc4 bench never recorded it, which is the gap."""
    bench = _load("p222_bench_under_test", LANE / "p222_bench.py")
    R = bench.R
    rec = _shake2("bench_p222.json")
    assert [a for a in R.ARMS if any(rec["compiled"][a][p] for p in ("gu", "dn"))] == ["C1"]   # the gap, as recorded
    tree = ([{"n_regs": 64, "n_spills": 0, "shared": 6400, "num_warps": 4, "num_stages": 3}], "block A reference")
    first = {(proj, R.plan_key(R.TODAY), R.T_LUT): tree for proj in ("gu", "dn")}
    for proj in ("gu", "dn"):
        for row in rec["select"][proj]:
            if row["compiled"]:
                first.setdefault((proj, R.plan_key(row["plan"]), R.P_LUT), (row["compiled"], "block A sweep"))
    got = {}
    for arm in R.ARMS:
        pg, pd, lut = rec["plans"][arm]
        for proj, plan in (("gu", pg), ("dn", pd)):
            got[(arm, proj)] = bench.arm_compiled(rec["compiled"][arm][proj], first.get((proj, R.plan_key(plan), lut)))
    for (arm, proj), rows in got.items():
        assert rows and all(r.get(f) is not None for r in rows for f in ("n_regs", "n_spills", "shared")), (arm, proj)
    src = {arm: {rows[0]["source"] for (a, _), rows in got.items() if a == arm} for arm in R.ARMS}
    assert src == {"T": {"block A reference"}, "T2": {"block A reference"}, "P": {"block A sweep"},
                   "C1": {"block B metadata pass"}, "M": {"block A sweep"}}
    assert got[("M", "gu")][0]["n_spills"] == 2                                  # M's gate_up plan, as the sweep compiled it


def test_amendment_2_is_additive():
    """The registered body stands: each superseded line keeps its text and gains the marker, and the new section sits
    after Amendment 1 and before the receipts."""
    assert sum(x.endswith(" *(superseded by Amendment 2)*") for x in PREREG.splitlines()) == 3
    for line in ("  - **T2 differs from T by a bit on any rep**, or T/T2's median is outside 0.99-1.01;",
                 "- **Run ids** `p222-5090-1` to `-3`, and `p222-shake-1` / `-2` for the shakedown."):
        assert line in PREREG, line
    assert PREREG.index("## Amendment 1") < PREREG.index("## Amendment 2") < PREREG.index("## Receipts")
    assert "**P(VOID | no real difference)\n   is under 1 in 20,000**" in PREREG


def test_amendment_3_registers_three_more_reading_ids_and_excludes_the_ghost_host():
    a3 = PREREG.split("## Amendment 3", 1)[1]
    for text in ("`p222-5090-4`, `-5` and `-6`", "excludes machine 45379", "byte-identical", "none a strike"):
        assert text in a3, text
    assert "(the reading's ids extended by Amendment 3)" in PREREG


# ---------------------------------------------------------------------------------------------------- the reading

READ = LANE / "receipts" / "p222-5090-5"


def test_the_reading_record_is_byte_exact_and_carries_no_identifier():
    """p222-5090-5's public files hash to SHA256SUMS; the store-only files are named by sha256 only; no file names an
    instance, an offer, an endpoint or the GPU's UUID."""
    lines = [ln.split() for ln in (READ / "SHA256SUMS").read_text(encoding="ascii").splitlines() if ln.strip()]
    assert len(lines) == 9
    for want, name in lines:
        assert hashlib.sha256((READ / name).read_bytes()).hexdigest() == want, name
    store = [ln for ln in (READ / "STORE-ONLY.sha256").read_text(encoding="ascii").splitlines() if ln and ln[0] != "#"]
    assert sorted(ln.split()[1] for ln in store) == ["instance.json", "p222/forensics.txt", "p222/outer.log",
                                                     "teardown-proof.json"]
    for p in READ.rglob("*"):
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            assert not re.search(r"GPU-[0-9a-f]{8}-|vast\.ai|ssh\d*\.|instance[_ ]id", text), p.name


def test_the_reading_rederives_byte_for_byte_and_reads_go_with_served_arm_m(tmp_path):
    """The registered reducer on the public bench record writes the box's verdict, byte for byte (Python 3.11 is CI's;
    another Python can move the last digit of an SE, so the check is exact only there)."""
    o = _reducer().verdict(json.loads((READ / "bench_p222.json").read_text(encoding="utf-8")))
    box = json.loads((READ / "verdict_p222.json").read_text(encoding="utf-8"))
    assert o["verdict"] == box["verdict"] == "GO" and o["served_arm"] == "M" and o["why"] == box["why"]
    assert round(o["M"]["median"], 4) == 2.2678 and round(o["M"]["lb_clock_adj"], 4) == 2.2624
    assert o["contract_findings"] == [] and o["self_pair_band"]["inside"] is True
    if sys.version_info[:2] == (3, 11):
        out = subprocess.run([sys.executable, str(LANE / "p222_reduce.py"), "--bench", str(READ / "bench_p222.json"),
                              "--out", str(tmp_path / "v.json")], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert (tmp_path / "v.json").read_bytes() == (READ / "verdict_p222.json").read_bytes()
    for text in ("2.2678", "2.2624", "**GO.**", "served arm is M", "power probe"):
        assert text in (LANE / "RESULTS-p222.md").read_text(encoding="utf-8"), text
