# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P157, what bounds K25: its stages knocked out one at a time at today's plan on Qwen3-30B-A3B's W16 shapes
(bench/p157/PREREG-p157.md): the staged pins; the shakedown on the REAL record first; the routing (P155's checks, on the
committed router replay); the served call form; the arms' kernel against K25's own source; the runner's pins, refusals,
guard fit with 25 % headroom and premise; the Nsight bytes pass; no instance id or host name on the box; the
self-tests and the mutants they must kill; the run ids."""
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

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p157"
RR = B / "p153" / "router-replay"
SOURCES = {"p157_run.sh": LANE, "p157_bench.py": LANE, "p157_reduce.py": LANE, "p157_arms.py": LANE,
           "p157_cuda.py": LANE, "sass_count.py": LANE, "rr.json.pt": RR, "test_k25_row_exact_gpu.py": REPO / "tests"}
RUN = (LANE / "p157_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p157_drive.sh").read_text(encoding="utf-8")
BENCH = (LANE / "p157_bench.py").read_text(encoding="utf-8")
ARMS = (LANE / "p157_arms.py").read_text(encoding="utf-8")
CUDA = (LANE / "p157_cuda.py").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p157.md").read_text(encoding="utf-8")
LIB = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text(encoding="utf-8")
GUARD_H = 1.5
BOOT_S = 300          # the launcher's create-to-runner allowance (P155's, kept)
HEADROOM = 1.25       # e4b#1845's cross-lane rule
SASS_COUNT_SHA256 = "d479fad5a865d76f6162b904caf08a9214957c96da2935cbf9c5844cfb37e748"   # gnf4#571's file


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
    return _load("p157_reduce_under_test", LANE / "p157_reduce.py")


def _run(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=300)


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p157.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name
    assert pins["sass_count.py"] == SASS_COUNT_SHA256 and SASS_COUNT_SHA256[:8] in PREREG


SHAKE = LANE / "shakedown" / "shake_p157.json"
SHAKE_SHA256 = "977a1b18612c937c3b547424068fdf4deb20448cad5caea134f1f8fe9995924e"


def test_the_reducer_on_the_real_shakedown_record():
    """The real record first: the $0 A2000 shakedown's own bench record (sha-pinned) through the registered reducer. It
    must read SHAKEDOWN with every structural check passed, never graded; every arm's static gate passes on sm_86; the
    pins hold A0's CTAs per SM wherever a pad can; the graph nodes read back the pinned shared bytes; and T2, A0p and
    ACX0 kept K25's bits on every rep."""
    raw = SHAKE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHAKE_SHA256
    rec = json.loads(raw)
    red = _reducer()
    v = red.verdict(rec)
    assert v["state"] == "SHAKEDOWN" and v["predictions"] == "UNREAD (a shakedown)"
    assert rec["cc"] == [8, 6] and rec["layers"] == 4 and rec["triton"] == "3.4.0" and len(rec["reps"]) == 14
    assert rec["spacer"] == "S" and rec["arms"] == list(red.ARMS)
    assert all(rp["identical"] == {"T2": True, "A0p": True, "ACX0": True} for rp in rec["reps"])
    assert v["gates"] == {arm: [] for arm in red.ARMS[2:]}
    for arm, by in rec["occupancy"].items():
        for proj, o in by.items():
            assert o["node_shared"] == [o["launch_shared"]], (arm, proj)
            if arm != "A2b":
                assert o["ctas_per_sm"] == rec["occupancy"]["A0"][proj]["ctas_per_sm"] == 6, (arm, proj)
    assert v["labels"]["A2b"] == ["gu: 5 CTAs/SM against A0's 6"]                  # 89 registers on sm_86
    assert rec["clocks"]["error"] is None and rec["clocks"]["A0"]["enforced_limit_w"] == [70.0]
    bad = json.loads(raw)                                    # the old rule's mutant, on the real record: an order that
    bad["reps"][3]["order"] = list(red.ARMS)                 # is not the registered design is VOID
    assert red.verdict(bad)["state"] == "VOID"


#: P155's box bench (p155-5090-1; its record is e4b#1856): 6 arms x 64 reps of 48-layer graphs in 110.2 s
P155_CONFIRM_S, P155_GRAPHS = 110.2, 6 * 64


def test_need_bench_is_the_larger_measured_wall_times_one_and_a_half():
    """NEED_BENCH is measured walls x1.5, from two measurements, the larger taken: this lane's shakedown (its host time
    per launch, scaled to 48 layers, plus every graph at P155's 12.075 ms) and P155's box (0.287 s a graph)."""
    rec = json.loads(SHAKE.read_bytes())
    arms, reps, layers = rec["arms"], rec["reps"], rec["layers"]

    def med(rp, a):
        return statistics.median(rp["replays_ms"][a])
    gpu = statistics.fmean((10 * sum(med(rp, a) for a in arms) + 10 * len(arms) * med(rp, "A0")
                            + 3 * (len(arms) + 1) * med(rp, "A0")) / 1000 for rp in reps)
    host = (rec["walls"]["reps_s"] / len(reps) - gpu) / (3 * (len(arms) + 1) * 2 * layers)
    a0 = 12.075e-3
    per_rep = (10 * len(arms) + 10 * len(arms) + 3 * (len(arms) + 1)) * a0 + host * 3 * (len(arms) + 1) * 2 * 48
    shake_box = 70 * per_rep + rec["walls"]["setup_s"] * 48 / layers + rec["walls"]["pin_static_s"]
    p155_box = 70 * ((len(arms) + 1) * P155_CONFIRM_S / P155_GRAPHS + 10 * len(arms) * a0) + 3.4 + \
        rec["walls"]["pin_static_s"]
    need = int(re.search(r"^NEED_BENCH=(\d+)", RUN, re.M).group(1))
    assert need == math.ceil(max(shake_box, p155_box) * 1.5 / 100) * 100 == 1100, (shake_box, p155_box)
    for figure in (f"about {round(shake_box):,} s", f"about {round(p155_box):,} s"):
        assert figure in PREREG, figure


def test_the_router_replay_is_the_committed_record():
    red = _reducer()
    raw = (RR / "rr.json.pt").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == red.RR_SHA256 == _pins(LANE / "staged-p157.sha256")["rr.json.pt"]
    assert f"{sha}  ./rr.json.pt" in (RR / "SHA256SUMS").read_text(encoding="ascii")
    assert red.RR_SHA256[:8] in PREREG and red.S0_SHA256[:8] in PREREG


def test_s0s_picks_pass_every_check_and_the_dose_counts_distinct_masked_experts():
    torch = pytest.importorskip("torch")
    bench = _load("p157_bench_under_test", LANE / "p157_bench.py")
    s0, sha = bench.s0_picks(str(RR / "rr.json.pt"))
    assert sha == bench.R.S0_SHA256 and tuple(s0.shape) == (48, 16, 128, 8) and s0.dtype == torch.int16
    for step in (58, 90, 127):
        sl = bench.slabs(s0, step, 48)
        ids = [set(s0[li, :, step, :].reshape(-1).tolist()) for li in range(48)]
        assert sl["127"]["gu"] == [len(x) for x in ids] == sl["127"]["dn"]
        assert sl["0"]["gu"] == [1] * 48 and all(a <= 4 for a in sl["3"]["gu"]) and all(a <= 16 for a in sl["15"]["gu"])
    f = [bench.R.dose(bench.slabs(s0, s, 48), m) / bench.R.dose(bench.slabs(s0, s, 48), 127)
         for s in bench.R.CONFIRM_STEPS for m in (0,)]
    assert 0.01 < statistics.median(f) < 0.03                                         # "about 2 %"


def test_the_registration_shares_p155s_routing_shapes_and_rules():
    red, p155 = _reducer(), _load("p155_reduce_for_p157", B / "p155" / "p155_reduce.py")
    for name in ("RR_SHA256", "S0_SHA256", "SHAPES", "TODAY", "ITERS", "SELF_BAND", "BOOT_B", "E", "TOPK", "TOKENS",
                 "LAYERS", "STEPS"):
        assert getattr(red, name) == getattr(p155, name), name
    assert red.BOOT_SEED == 157 and red.CONFIRM_STEPS == tuple(range(58, 128))
    q = [math.cos(i) for i in range(70)]
    assert red.se_median(q, seed=155) == p155.se_median(q)                              # the same rule, seeds apart


def test_todays_plan_is_the_served_plan():
    pytest.importorskip("torch")
    from experts4bit_qlora.engines import hot_residency as hr
    red = _reducer()
    assert hr._K25_PLAN == dict(red.TODAY, lut="tree", dot_bf16=False)
    assert '"block_n": 32, "kc": 64, "warps": 4, "stages": 3, "lut": "tree", "dot_bf16": False}' in RUN


def test_a0_is_k25_as_the_served_route_calls_it_and_the_arms_take_the_same_form():
    assert "return _k25(_tok[0].to(torch.bfloat16), pk, am, t_row0, t_rows, t_grp, order,\n" \
           "                            gather_div=_tok[1], **_k25_kw)" in LIB
    assert 'kw = {"scatter": order} if (_lean and pk is dn_p) else {}' in LIB
    assert "self.k25(self.X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK, lut=\"tree\",\n" in BENCH
    assert "self.k25(self.H[li], p, am, row0, rows, grp, None, scatter=order, lut=\"tree\", dot_bf16=False," in BENCH
    assert "self.A.launch(arm, self.X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK, **plan)" in BENCH
    assert "self.A.launch(arm, self.H[li], p, am, row0, rows, grp, None, scatter=order, **plan)" in BENCH
    assert "K=K, BLOCK_M=16, BLOCK_N=block_n, KC=kc, LUT=3, GATHER=bool(gather), DOT_BF16=False," in ARMS
    assert "build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)" in BENCH


#: K25's lines that _k25_ablate changes, and how (the PREREG's "copied verbatim except")
K25_EDITS = {
    "for c in range(0, NKC):": "for c in tl.range(0, NKC, num_stages=3, loop_unroll_factor=1):",
    "wbase = w_ptr + eid * stride_we + offs_n.to(tl.int64)[:, None] * stride_wn":
        "wbase = w_ptr + eid * stride_we + offs_w.to(tl.int64)[:, None] * stride_wn",
    "abase = am_ptr + eid * stride_ae + offs_n.to(tl.int64)[:, None] * stride_an":
        "abase = am_ptr + eid * stride_ae + offs_w.to(tl.int64)[:, None] * stride_an",
}


def _body(src, name):
    """A function's statements after its docstring, comments and blank lines dropped, stripped of indentation."""
    at = src.index(f"def {name}(")
    indent = at - src.rindex("\n", 0, at) - 1 + 4                                  # the body's indentation
    start = src.index('"""', src.index('"""', at) + 3) + 3                         # after the docstring
    lines = []
    for ln in src[start:].splitlines()[1:]:
        if ln.strip() and not ln.startswith(" " * indent):
            break
        s = ln.split("#")[0].strip()
        if s:
            lines.append(s)
    return lines


def test_the_arms_kernel_is_k25_verbatim_except_the_registered_edits():
    """Every statement of K25's kernel body (at the pin) appears in _k25_ablate, in order, after the registered edits
    (the loop header, and the weight rows read through ``offs_w``; under ``n_mask`` the loads read ``w_mask``, equal
    to it when RESID is off). Skipped unless the installed grouped-nf4-gemm carries the pinned nf4_smallm.py."""
    try:
        import nf4_smallm
    except ImportError:
        pytest.skip("grouped-nf4-gemm is not installed")
    want = re.search(r"^NF4_SMALLM_SHA256=([0-9a-f]{64})", RUN, re.M).group(1)
    path = pathlib.Path(nf4_smallm.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != want:
        pytest.skip("the installed nf4_smallm.py is not the pinned one")
    k25 = _body(path.read_text(encoding="utf-8"), "_gemm_nf4_grouped_smallm")
    arms = [x.replace("w_mask", "n_mask") for x in _body(ARMS, "_k25_ablate")]
    i = 0
    for line in k25:
        line = K25_EDITS.get(line, line)
        while i < len(arms) and arms[i] != line:
            i += 1
        assert i < len(arms), f"K25's line is not in _k25_ablate in order: {line}"
        i += 1


def test_the_arm_table_and_the_one_cubin_sets():
    A = _load("p157_arms_under_test", LANE / "p157_arms.py")
    red = _reducer()
    assert set(A.ARMS) == (set(red.ARMS) - {"A0", "T2"}) and A.LADDER == red.LADDER
    for arm in red.CUBINS["ladder"]:
        assert A.ARMS[arm][:2] == (0, True) and A.SHARES.get(arm, "AC") == "AC"
    assert {A.ARMS[a][2] for a in red.LADDER} == {0, 3, 15, 127} and A.ARMS["ACX0"][3] == 0
    assert all(A.ARMS[a][3] == 1 for a in red.LADDER + ("A3", "AL"))
    assert A.ARMS["A3"][:3] == (4, True, 127) and A.ARMS["AL"][:3] == (4, True, 0) and A.SHARES["AL"] == "A3"
    assert all(A.ARMS[a][1] is False for a in ("A0p", "A1", "A2b", "A12", "AE"))
    assert '@triton.jit(do_not_specialize=["emask", "nrot"])' in ARMS
    with pytest.raises(RuntimeError, match="P157_ABLATION=1"):
        os.environ.pop("P157_ABLATION", None)
        A.launch("A0p", None, None, None, None, None, None)


def test_the_a2a_arithmetic_is_stated():
    for figure in ("48 distinct operand registers", "ceil(48 / 2) = **24 three-input `lop3`**", "against 8 HMMA",
                   "trades\ntensor-pipe time for extra ALU issues"):
        assert figure in PREREG, figure
    assert "A2a" not in _reducer().ARMS


def test_the_runner_pins_k25_and_refuses_before_the_install():
    first_install = RUN.index('say "install e4b @')
    for refusal in ('[ "$PL" -ge "$MIN_POWER_W" ] || {', '[ "$CC" = 12.0 ] || {', '[ "$LEFT" -ge "$FIT_NEED" ] || {',
                    '[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || {'):
        assert RUN.index(refusal) < first_install, refusal
    floor = RUN.index('[ "$PL" -ge "$MIN_POWER_W" ] || {')
    assert "MIN_POWER_W=575" in RUN and "finish 17; }" in RUN[floor:floor + 260]
    assert "${P157_MIN_POWER_W:-}" in RUN and "${P157_NEED_BENCH:-}" in RUN
    gnf4 = re.search(r"^GNF4_SHA=([0-9a-f]{40}) ", RUN, re.M)
    assert gnf4 and gnf4.group(1) == "41ef086e6a3e2be5b158ff1bb38e0de67862a949" and "41ef086e" in PREREG
    assert 'md.version("grouped-nf4-gemm") == "0.50.0"' in RUN
    for sha in ("ee46969a4f786c36abe26bfaf0a52e426db4465406a5e31c6085fd3270b26568",
                "773eff5db7be77dea3d1f9048a6e708a94e79634baf17f452bf2848e24a2072f",
                "a7090915485c6e9943f568e79315b3f533c754f24edbdb7160b05dcc18619a23"):
        assert sha in RUN
    assert "grep -qE '^4 passed(, [0-9]+ warnings?)? in" in RUN
    order = [RUN.index(x) for x in ('say "install e4b @', "python -m pytest test_k25_row_exact_gpu.py",
                                    "can_run $NEED_BENCH bench", "python $W/p157_bench.py --rr",
                                    "ncu_step(){", "python $W/p157_reduce.py --bench")]
    assert order == sorted(order)
    for f in ("p157_arms.py", "p157_cuda.py", "sass_count.py"):
        assert f in RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]


def test_the_guard_fits_with_25_percent_headroom():
    k = {n: int(v) for n, v in re.findall(r"^(INSTALL_TO_GATE_S|NEED_BENCH|NEED_NCU|RESERVE)=(\d+)", RUN, re.M)}
    assert k == {"INSTALL_TO_GATE_S": 600, "NEED_BENCH": 1100, "NEED_NCU": 900, "RESERVE": 600}, k
    assert ("FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + NEED_NCU + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100))"
            in RUN)
    fit = sum(k.values())
    assert fit * 125 // 100 == 4000 and "`FIT_NEED` = 3200 x 1.25 =\n    4000 s" in PREREG
    assert (BOOT_S + fit) * HEADROOM <= GUARD_H * 3600
    assert "3500 x 1.25 = 4375 s <= 5400 s\n    (1.5 h)" in PREREG and "**guard 1.5 h**" in PREREG
    assert "1.5 x $1.05 + 10 x $0.011 =\n    **$1.69**" in PREREG


def test_the_nsight_step_feeds_the_reduce_and_never_ends_the_run():
    """It runs before the reduce (the bytes axis), probes the counters first, and every path out of it returns 0. Its
    launch window is the bench's own constants: the pins' launches and three warm rounds skipped, one round measured."""
    body = RUN[RUN.index("ncu_step(){"):RUN.index("\n}\n", RUN.index("ncu_step(){"))]
    assert "finish" not in body and "exit" not in body
    assert body.index("ERR_NVGPUCTRPERM") < body.index("for arm in $NCU_ARMS")
    assert 'python -c "import p157_bench as B; print(B.NCU_SKIP)"' in body and "--launch-skip \"$skip\"" in body
    assert 'NCU_ARMS="A0p A3 AL AC AC4 AC16 ACX"' in RUN and "--ncu-collect" in body
    assert RUN.index('if can_run $NEED_NCU "ncu (the bytes axis)"') < RUN.index("python $W/p157_reduce.py --bench")
    bench = _load("p157_bench_for_ncu", LANE / "p157_bench.py")
    assert bench.NCU_SKIP == 2 * (len(bench.R.ARMS) - 1) + 3 * 2 * bench.NCU_LAYERS and bench.NCU_COUNT == 8
    pin = BENCH[BENCH.index("    def pin(self, step):"):BENCH.index("        for proj in (\"gu\", \"dn\"):\n            a0")]
    assert 'if arm == "T2":\n                continue' in pin and "self.call(arm, proj, 0, tab)" in pin
    target = BENCH[BENCH.index("def ncu_target"):BENCH.index("def ncu_collect")]
    assert "range(4)" in target and "box.pin(s)" in target


def test_no_instance_id_or_host_name_reaches_the_box():
    for name, text in (("runner", RUN), ("bench", BENCH), ("arms", ARMS), ("cuda", CUDA),
                       ("reducer", (LANE / "p157_reduce.py").read_text("utf-8"))):
        for marker in ("gethostname", "socket.", "INSTANCE_ID", "hostname"):
            assert marker not in text, (name, marker)
    pass_line = next(x for x in DRIVE.splitlines() if x.startswith("PASS="))
    assert "INSTANCE" not in pass_line and "P157_DEADLINE_EPOCH=$DEADLINE" in pass_line
    assert "HF_TOKEN" not in DRIVE


def test_the_predictions_are_registered():
    red = _reducer()
    assert red.PRED == {"a0_ms": (11.5, 13.5), "copy": (0.995, 1.005), "r_bands": ("LATENCY_BOUND", "MIXED"),
                        "decode_ms": (1.9, 6.0), "mma_ms": (0.0, 1.5), "mem_ms": (2.0, 5.5), "onchip_ms": (5.0, 10.0)}
    for figure in ("**11.5-13.5 ms**", "**0.995-1.005**", "**LATENCY_BOUND or MIXED**", "**1.9-6.0 ms**",
                   "**0-1.5 ms**", "**2.0-5.5 ms**", "**5.0-10.0 ms**", "AC <= AC4 <= AC16 <= ACX", "**4.94 ms**",
                   "**12.075 ms**", "**24.1 lane instructions per weight on sm_120**"):
        assert figure in PREREG, figure
    assert (red.COPY_TOL, red.CLOCK_TOL, red.WIDE) == (0.01, 0.02, 0.5) and red.TEST_RUNGS == ("AC4", "AC16")


@pytest.mark.parametrize("script", ["p157_reduce.py", "p157_bench.py"])
def test_the_self_test_passes(script):
    out = _run(LANE / script, "--self-test")
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


REDUCE_MUTANTS = [
    # full taken from A0p instead of inside the ladder's cubin
    ('    full = [m["ACX"] - m["AE"] for m in ms]', '    full = [m["A0p"] - m["AE"] for m in ms]'),
    # the ladder's doses from zero, not from AC's (a serial kernel would read AC4 as MIXED)
    ('            g = (f[k] - f["AC"]) / (1 - f["AC"])', '            g = f[k]'),
    # NOT_ADDITIVE not split by side in the ladder check (a serial rung just above 1 would disagree)
    ('    ok = COMPATIBLE[regime(call, call_lo)] | {"UNRESOLVED"}', '    ok = {call, "UNRESOLVED"}'),
    # the bands without NOT_ADDITIVE first
    ('    if hi < 0 or lo > 1:\n        return "NOT_ADDITIVE"\n', ''),
    # no small-denominator rule
    ('    small = min(statistics.median(mem), statistics.median(onchip)) <= 2 * self_se', '    small = False'),
    # the mean's SE for the median's
    ('    se = se_median(q)\n    return {"n"', '    se = statistics.stdev(q) / math.sqrt(len(q))\n    return {"n"'),
    # an uninformative rung allowed to rule
    ('            row["informative"] = row["role"] == "test" and row["width"] <= WIDE',
     '            row["informative"] = row["role"] == "test"'),
    # ACX's labels not blocking the call
    ('    used = ("A3", "AL", "AC", "ACX")', '    used = ("A3", "AL", "AC")'),
    # no spacer check
    ('    if rec.get("spacer") != SPACER:\n', '    if False:\n'),
    # the order not checked
    ('        if rp.get("order") != order(ARMS, i):\n', '        if False:\n'),
    # the one-cubin sets not checked
    ('            if len(hs) != 1 or None in hs:\n', '            if False:\n'),
    # ACX0 allowed to move bits
    ('        for arm in ("T2", "A0p", "ACX0"):\n', '        for arm in ("T2", "A0p"):\n'),
    # the self-pair band not enforced
    ('    if not SELF_BAND[0] <= self_pair["median"] <= SELF_BAND[1]:\n', '    if False:\n'),
    # load parity not gated
    ('                elif r.get("loads_per_weight") != a0.get("loads_per_weight"):\n', '                elif False:\n'),
    # the graph nodes' shared bytes not checked
    ('            if not nodes or any(n != o.get("launch_shared") for n in nodes):\n', '            if False:\n'),
    # the clock rule not applied
    ('        if _num(base) and _num(c) and abs(c / base - 1) > CLOCK_TOL:\n', '        if False:\n'),
    # Nsight's dose check not applied
    ('        passed = ok and all(abs(checks[k] - 1) <= NCU_DOSE_TOL for k in LADDER + ("AL",)) and abs(',
     '        passed = ok and all(abs(checks[k] - 1) <= 9 for k in LADDER + ("AL",)) and abs('),
    # a broken Williams square
    ('            first.append(hi)\n            hi -= 1\n', '            first.append(lo)\n            lo += 1\n'),
    # the copy gap passing silently
    ('    if not copy_["gap_within_tol"]:\n', '    if False:\n'),
    # an MMA arm that compiled without one read as if it had (b1-decode's (2))
    ('            if r.get("has_mma") is not (arm in MMA_ARMS):\n', '            if False:\n'),
    # a refused static count hidden behind "no static record" (b1-decode's (1))
    ('            if r and r.get("error"):\n', '            if False:\n'),
    # a drifted enforced power limit not labelled
    ('        if lim0 and lim and lim != lim0:\n', '        if False:\n'),
]
BENCH_MUTANTS = [
    # no spacer before the timed replay
    ("                    graphs[R.SPACER][0].replay()      # untimed: every timed replay follows the same work\n", ""),
    # gate_up without the served gather
    ("self.A.launch(arm, self.X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK, **plan)",
     "self.A.launch(arm, self.X[li], p, am, row0, rows, grp, order, **plan)"),
    # down without the served scatter
    ("self.A.launch(arm, self.H[li], p, am, row0, rows, grp, None, scatter=order, **plan)",
     "self.A.launch(arm, self.H[li], p, am, row0, rows, grp, None, **plan)"),
    # the old cyclic rotation
    ("            rot = R.order(R.ARMS, r)", "            rot = list(R.ARMS[r % 14:]) + list(R.ARMS[:r % 14])"),
    # no pad applied
    ("                        self.A.set_pad(k, pad)", "                        pass"),
    # the graph nodes not read back
    ("            nodes = capture_kernel_nodes(torch.cuda.current_stream().cuda_stream)", "            nodes = []"),
    # ACX0's bits not checked
    ('                     for arm in ("T2", "A0p", "ACX0")}', '                     for arm in ("T2", "A0p")}'),
    # has_mma assumed instead of read from the SASS
    ('    has_mma = any(sc.classify(op) == "mma" for _, op, _ in rows)', '    has_mma = True'),
]
CUDA_MUTANTS = [                                          # run through the bench's self-test, which drives a fake NVML
    # a sample straddling a tag change kept (b1-decode's (5))
    ("        if self.tag != tag:\n            self.dropped += 1\n            return\n", ""),
    # a failed power read recorded as 0 W (b1-decode's (3))
    ("        mw_v = mw.value / 1000.0 if nv.nvmlDeviceGetPowerUsage(h, ctypes.byref(mw)) == 0 else None",
     "        nv.nvmlDeviceGetPowerUsage(h, ctypes.byref(mw))\n        mw_v = mw.value / 1000.0"),
    # the field-value union read without its type (b1-decode's (4))
    ('        return struct.unpack("<d", raw.to_bytes(8, "little"))[0]', "        return raw"),
]


@pytest.mark.parametrize("script,old,new", [("p157_reduce.py", o, n) for o, n in REDUCE_MUTANTS]
                         + [("p157_bench.py", o, n) for o, n in BENCH_MUTANTS]
                         + [("p157_cuda.py", o, n) for o, n in CUDA_MUTANTS])
def test_each_mutant_fails_the_self_test(tmp_path, script, old, new):
    src = (LANE / script).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / script).write_bytes(src.replace(old, new).encode("utf-8"))
    if script != "p157_reduce.py":                    # the bench's self-test imports its siblings
        for sib in ("p157_reduce.py", "p157_bench.py", "p157_cuda.py", "sass_count.py"):
            if sib != script:
                shutil.copy(LANE / sib, tmp_path / sib)
    out = _run(tmp_path / ("p157_reduce.py" if script == "p157_reduce.py" else "p157_bench.py"), "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P157_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p157_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p157-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p157" in out.stdout and "bash p157_run.sh" in out.stdout
    assert "INSTANCE" not in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p157_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p156-5090-1"), timeout=120)
    assert out.returncode == 78 and "is not p157-5090-N" in out.stdout + out.stderr


def test_the_run_ids_are_registered_in_both_scripts():
    for text in (RUN, DRIVE):
        assert "p157-5090-1|p157-5090-2|p157-5090-3)" in text
    assert "`p157-5090-1` to `-3`" in PREREG
