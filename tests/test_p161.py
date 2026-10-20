# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P161 (bench/p161/PREREG-p161.md; e4b#1313), the served read of K25 at plan c1 under K39's decode, through
E4B_K25_PLAN: the staged pins (every piece shared with P158's kit at P158's pins, e4b's served-route contract frozen with
c1's cases), the runner's order (the power floor and the guard fit before the install, the premise at 62, nine processes
at five reps, K39's kernel bytes and the plan table in the tripwire, no SANE phase), the guard fit with 25 % headroom on
both sides, the self-tests run as the box runs them (one directory), P153's pieces carried verbatim, the plan's two
witnesses (the record and the shape spy) read from the code the box installs, the mutants the self-tests must kill, and
the driver's run ids (-1..-3)."""
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p161"
P153 = B / "p153"
P158 = B / "p158"
PREREG = (LANE / "PREREG-p161.md").read_text(encoding="utf-8")
SOURCES = {"p161_run.sh": LANE, "p161_box.py": LANE, "p161_reduce.py": LANE, "p109_box.py": B / "p109",
           "k8_bake.py": B / "p39", "calib.json": B / "p39",
           "test_decode_graph_buckets.py": REPO / "tests", "test_kv_step_select.py": REPO / "tests",
           "test_fused_glue_decode_graphs_gpu.py": B / "p136" / "premise", "test_gemv_bw_served_gpu.py": REPO / "tests",
           "test_nf4_smallm_prmt_gpu.py": LANE / "premise", "test_k25_lut_gpu.py": LANE / "premise"}
#: premise/test_k25_lut_gpu.py is tests/test_k25_lut_gpu.py frozen at this registration's base (dd535cd6, the first main
#: with c1's cases, e4b#1877): the lane stages its own bytes, never the live test, which later knobs extend.
FROZEN_K25_ROUTE_TEST_SHA256 = "9ee0028ee57c4bc33e5290f73fba5a68eda10fa83d61d6d455bdf0ca2f7f2338"
#: grouped-nf4-gemm kernel/test_nf4_smallm_prmt_gpu.py at v0.50.0 (41ef086e6a3e2be5b158ff1bb38e0de67862a949), its sha256
K39_COMPILED_TEST_SHA256 = "9f6366287957f75bdfe53ff8165b401c30fb9e97d4da582254ad7e98bf702868"
GUARD_H = 1.5
BOOT_S = 300          # the launcher's create-to-runner allowance (P152's, P154's and P155's, kept)
HEADROOM = 1.25


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _run(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=180, cwd=cwd)


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p161.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_every_piece_shared_with_p158_is_at_p158s_pins_and_the_route_contract_carries_c1():
    mine, p158 = _pins(LANE / "staged-p161.sha256"), _pins(P158 / "staged-p158.sha256")
    shared = set(mine) & set(p158)
    assert shared == {"p109_box.py", "k8_bake.py", "calib.json", "test_decode_graph_buckets.py", "test_kv_step_select.py",
                      "test_fused_glue_decode_graphs_gpu.py", "test_gemv_bw_served_gpu.py", "test_nf4_smallm_prmt_gpu.py",
                      "test_k25_lut_gpu.py"}
    assert all(mine[n] == p158[n] for n in shared - {"test_k25_lut_gpu.py"})
    assert mine["test_nf4_smallm_prmt_gpu.py"] == K39_COMPILED_TEST_SHA256
    assert mine["test_k25_lut_gpu.py"] == FROZEN_K25_ROUTE_TEST_SHA256 != p158["test_k25_lut_gpu.py"]
    route = (LANE / "premise" / "test_k25_lut_gpu.py").read_text(encoding="utf-8")
    # six functions, 20 tests once parametrized (the runner's premise counts 20 of its 62); three are c1's
    assert route.count("\ndef test_") == 6 and route.count("_under_c1_") == 3 and "E4B_K25_PLAN" in route
    assert not {"p115_quality.py", "p110_box.py", "p108_box.py", "p97_box.py"} & set(mine), "no SANE phase"


def test_the_runner_refuses_below_575_w_and_without_the_fit_before_the_install_and_reads_only_a_rule_verdict():
    run = (LANE / "p161_run.sh").read_text(encoding="utf-8")
    assert "GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949" in run and 'md.version("grouped-nf4-gemm") == "0.50.0"' in run
    assert "p161-5090-1|p161-5090-2|p161-5090-3) MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39;;" in run
    assert "SHORT=32; LONG=160; REPS=5\n" in run and "MIN_POWER_W=575 " in run
    assert 'finish 17; }' in run and 'echo "refused: power.limit unreadable" > REFUSAL; finish 15; }' in run
    assert "E4B_ATTN_SPLIT_CAP E4B_DENSE_GEMV_BW E4B_INT8_HEAD_BW E4B_K25_LUT E4B_K25_PLAN\n" in run, "every knob starts unset"
    assert "ee46969a4f786c36abe26bfaf0a52e426db4465406a5e31c6085fd3270b26568" in run, "K25/K39's kernel bytes in the tripwire"
    assert 'assert sp.INT8_HEAD_BW_DEFAULT_FAMILIES == {"qwen3_moe": ((151936, 2048),)}' in run, "the int8 head's default"
    assert 'assert sp.ATTN_SPLIT_CAP_DEFAULT_FAMILIES == {"qwen3_moe": 16}' in run
    assert '== (None, "prmt", None, "default-allowlisted"), _r' in run, "the decode at qwen3_moe's default in the tripwire"
    assert 'assert hr._K25_PLANS == {"today": ({}, {}), "c1": ({"block_n": 64, "kc": 256, "warps": 4, "stages": 2},' in run
    assert 'assert "E4B_K25_PLAN=c1" in str(_e), _e' in run, "c1 under the tree refused in the tripwire"
    order = ["for f in p161_box.py", "PL=$(nvidia-smi --query-gpu=power.limit", '[ "$LEFT" -ge "$FIT_NEED" ]',
             "pipx logs/pip_e4b.log", "grep -qE '^62 passed", "snapshot_download", "python $W/k8_bake.py", "--prompts-only",
             "for TAG in S0w S0a S1a S1b S0b S0c S1c S1d S0d; do", "python $W/p161_reduce.py --dir"]
    at = [run.index(x) for x in order]
    assert at == sorted(at), dict(zip(order, at))
    assert "test_gemv_bw_served_gpu.py test_nf4_smallm_prmt_gpu.py test_k25_lut_gpu.py -q -rs" in run
    assert run.count("E4B_K25_PLAN=$PLAN P161_ARM=$ARM") == 1 and 'PLAN=today; [ "$ARM" = S1 ] && PLAN=c1' in run
    for knob in ("E4B_ATTN_SPLIT_CAP=", "E4B_DENSE_GEMV_BW=", "E4B_INT8_HEAD_BW=", "E4B_NF4_GROUPED_SMALLM=", "E4B_K25_LUT="):
        assert knob not in run, f"{knob} stays unset (its default) in both arms"
    assert "--mode sane" not in run and "p115_quality" not in run, "no SANE phase"
    assert 'case "$V" in FLIP|NO_FLIP|FUNCTION_FAIL) ;;' in run, "NOISY and VOID are not readings"
    # the early gates: after S0w (rc 43) and S1a (rc 44), each before the arms after it
    assert "    S0w) GATE_RC=43;;\n    S1a) GATE_RC=44;;\n" in run
    assert "python $W/p161_reduce.py --gate $W/arm_$TAG.json --gate-arm $ARM" in run and "finish $GATE_RC" in run
    assert "not a strike; the same gate firing on two consecutive runs counts as one strike" in run.replace("\n  # ", " ")


def test_the_layercut_record_is_the_licence_half_and_names_its_limit():
    lc = LANE / "layercut"
    import json
    spy = {"today": {"32/64/4/3/prmt": 32}, "c1": {"64/256/4/2/prmt": 16, "128/128/8/3/prmt": 16}}
    for plan in ("today", "c1"):
        r = json.loads((lc / f"layercut_{plan}.json").read_text(encoding="utf-8"))
        assert r["layers"] == 2 and r["k25_seen_build"] == {"prmt|le256": 16} and r["k25_lut"]["lut"] == "prmt", r
        assert r["k25_lut"]["source"] == "default-allowlisted" and r["k25_lut"]["env"] is None, r
        assert r["k25_plan"]["plan"] == plan and r["k25_plan_seen_build"] == {f"{plan}|le256": 16}, r
        assert r["k25_shapes_build"] == spy[plan] and not r["k25_plan_seen_before"] and not r["k25_shapes_before"], r
        assert sum(r["dispatch_build"].values()) == 8 and r["graph_status"] == {}, r
    readme = (lc / "README.md").read_text(encoding="utf-8")
    assert "needs sm_89+" in readme and "960 = 48 x (2 + 3) x 4 in all nine processes" in readme
    assert "That is a carry from another lane's measurement, labelled as such" in readme


def test_the_guard_fits_with_25_percent_headroom():
    """e4b#1845's rule on both sides: the runner refuses before the install below FIT x 125 %, and the registered guard
    holds the boot plus that fit with 25 % headroom."""
    run = (LANE / "p161_run.sh").read_text(encoding="utf-8")
    val = {k: int(re.search(rf"^{k}=(\d+)", run, re.M).group(1)) for k in ("INSTALL_TO_GATE_S", "NEED_BENCH", "RESERVE")}
    fit = val["INSTALL_TO_GATE_S"] + val["NEED_BENCH"] + val["RESERVE"]
    assert (val["INSTALL_TO_GATE_S"], val["NEED_BENCH"], val["RESERVE"]) == (600, 2100, 600) and fit == 3300
    assert "FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + RESERVE))" in run and "FIT_HEADROOM_PCT=125" in run
    assert (BOOT_S + fit) * HEADROOM <= GUARD_H * 3600
    assert "(300 + 3300) x 1.25 = 4500 s <= 5400 s (1.5 h)" in PREREG and "**guard 1.5 h**" in PREREG
    # NEED_BENCH: the larger of the two measured served reads' fetch-to-reduce walls, at nine processes, x1.5
    assert math_ceil_to(1.5 * max(1347, 1039), 100) == val["NEED_BENCH"]


def math_ceil_to(x, step):
    return -(-int(round(x)) // step) * step


@pytest.fixture(scope="module")
def box_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("p161_box")
    for name, src in SOURCES.items():
        shutil.copyfile(src / name, d / name)
    return d


@pytest.mark.parametrize("script", ["p161_box.py", "p161_reduce.py"])
def test_the_self_tests_pass(box_dir, script):
    out = _run(box_dir / script, "--self-test", cwd=box_dir)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_p153s_checks_are_carried_verbatim():
    """P153's helpers and record checks verbatim; its W1 check's statements carried inside this lane's, which adds the
    GEMV's closed-form build tally."""
    p153 = (P153 / "p153_reduce.py").read_text(encoding="utf-8")
    mine = (LANE / "p161_reduce.py").read_text(encoding="utf-8")
    helpers = p153[p153.index("def gemv_faults(name, d) -> list:"):p153.index("def w1_faults(t, r) -> list:")]
    records = p153[p153.index("def record_faults("):p153.index("def cap_faults(")]
    assert helpers in mine and records in mine
    w1 = p153[p153.index("def w1_faults(t, r) -> list:"):p153.index("def record_faults(")]
    for stmt in ('    gs1 = (r.get("graph_stats") or {}).get("1") or {}\n',
                 '    out = gemv_faults(f"{t} W1 capture", r.get("dispatch_build"))\n',
                 '    if not gs1.get("replays"):\n', '    if gs1.get("eager_steps") or any((r["workloads"]["W1"].get("dispatch") or {}).values()):\n'):
        assert stmt in w1 and stmt in mine, stmt


def test_the_build_counts_are_derived_from_the_code_the_box_installs():
    """960 = 48 MoE layers x the four multi-row buckets x five forwards (the residual licence's two probes, then
    enable_decode_graphs' two warm-ups and one capture); 480 = 48 x the same five forwards at T == 1 x two GEMV calls
    (P153's build read 480). Each fact is read from the code the box installs: the licence before the capture, its
    two forwards per layer and row count, the warm-up default, the bucket set, T == 1's route, K25's row bound."""
    serve = (REPO / "experts4bit_qlora" / "serve_paged.py").read_text(encoding="utf-8")
    lic = serve.index("    license_moe_residual(model, sorted({1, *cfg.buckets, *runner._verify_buckets}),")
    cap = serve.index("    graph_status = runner.enable_decode_graphs(cfg.buckets) if cfg.graphs else None")
    assert lic < cap, "the licence probes run before the capture, inside the build's record"
    glue = (REPO / "experts4bit_qlora" / "engines" / "glue_r2.py").read_text(encoding="utf-8")
    body = glue[glue.index("def license_moe_residual("):glue.index("def _patch_attention(")]
    assert "            for t in rows:" in body and "                    want = mod.mlp(h)\n" in body
    assert "                    got = _moe_with_residual(mod.mlp, h, r, width)\n" in body
    assert "    return mlp.experts(h2, idx, w, residual=residual.reshape(-1, width)).reshape(residual.shape)" in glue
    runner = (REPO / "experts4bit_qlora" / "engines" / "paged_runner.py").read_text(encoding="utf-8")
    assert re.search(r"def enable_decode_graphs\(self, buckets=DEFAULT_BUCKETS, \*, capture: bool = True,\s+"
                     r"warmup: int = 2,", runner)
    assert "for _ in range(max(1, warmup)):" in runner and "with torch.cuda.graph(g):\n                self._padded_step(b)" in runner
    hr = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text(encoding="utf-8")
    assert "    singleton = T == 1 or" in hr and "R_rows <= 256 and _k25_mode != \"0\"" in hr
    sys.path.insert(0, str(LANE))
    try:
        import p161_reduce as r
    finally:
        sys.path.remove(str(LANE))
    assert (r.LICENCE_FORWARDS, r.WARMUPS, r.CAPTURES) == (2, 2, 1) and r.K25_BUCKETS == (2, 4, 8, 16)
    assert r.K25_BUILD_CALLS == 48 * 4 * 5 == 960 and r.BW_BUILD_CALLS == 48 * 5 * 2 == 480
    assert "= 960" in PREREG and "= 480" in PREREG
    # the plan's witnesses: one kernel call per projection (gate_up, down) per K25 call, each at its plan's block shape
    assert hr.count("**(_k25_gu_kw if pk is gu_p else _k25_dn_kw))") == 1 and hr.count("gather_div=_tok[1], **_k25_gu_kw)") == 1
    assert '_K25_PLAN = {"block_n": 32, "kc": 64, "warps": 4, "stages": 3,' in hr and r.TODAY_SHAPE == {
        "block_n": 32, "kc": 64, "warps": 4, "stages": 3}
    c1 = hr[hr.index("_K25_PLANS: dict = {"):hr.index("}", hr.index('"c1": (', hr.index("_K25_PLANS: dict = {")) + 1)]
    assert '"block_n": 64, "kc": 256, "warps": 4, "stages": 2' in c1 and r.PLAN_PAIR["c1"][0] == {
        "block_n": 64, "kc": 256, "warps": 4, "stages": 2}
    assert r.KERNEL_CALLS_PER_K25 == 2 and r.shapes_build("S0") == {"32/64/4/3/prmt": 1920}
    assert r.shapes_build("S1") == {"64/256/4/2/prmt": 960, "128/128/8/3/prmt": 960}


REDUCER_MUTANTS = [
    # g16 from the best pair, not the worst of four
    ('    g16_min, g1_min = min(pairs["W16"].values()), min(pairs["W1"].values())',
     '    g16_min, g1_min = max(pairs["W16"].values()), min(pairs["W1"].values())'),
    # only the first ABBA block's pairs
    ('PAIRS = (("S1a", "S0a"), ("S1b", "S0b"), ("S1c", "S0c"), ("S1d", "S0d"))', 'PAIRS = (("S1a", "S0a"), ("S1b", "S0b"))'),
    # the NOISY band widened
    ("NOISY_BAND = (0.985, 1.015)", "NOISY_BAND = (0.98, 1.02)"),
    # NOISY read at W1 only (P153's form): the bar is on W16
    ("             for w in WORKLOADS for k, v in selfp[w].items() if not NOISY_BAND[0]",
     "             for w in (\"W1\",) for k, v in selfp[w].items() if not NOISY_BAND[0]"),
    # NOISY never read
    ("    if noisy:\n        out.update(verdict=\"NOISY\"", "    if False:\n        out.update(verdict=\"NOISY\""),
    # the warm-up process's absence accepted
    ('    missing = [t for t in TAGS + (WARMUP,) if t not in arms', '    missing = [t for t in TAGS if t not in arms'),
    # the five reps never checked
    ('        if r.get("reps") != REPS:', "        if False:"),
    # the W1 guard dropped from the flip
    ("    elif g16_min >= FLIP_G16 and g1_min >= NOREG_G1:", "    elif g16_min >= FLIP_G16:"),
    # the bars read the ROUNDED mins
    ("    elif g16_min >= FLIP_G16 and g1_min >= NOREG_G1:",
     "    elif round(g16_min, 4) >= FLIP_G16 and round(g1_min, 4) >= NOREG_G1:"),
    # THE NULL never read: a non-deterministic today reads as a contract break against c1
    ('    if null:\n        out.update(verdict="VOID"', '    if False:\n        out.update(verdict="VOID"'),
    # the null without the warm-up
    ('    null = _token_diffs(arms, ("S0b", "S0c", "S0d", WARMUP), "S0a", lens)',
     '    null = _token_diffs(arms, ("S0b", "S0c", "S0d"), "S0a", lens)'),
    # the identity gate never read
    ('    if fn:\n        out["contract"]', '    if False:\n        out["contract"]'),
    # S1 held to S1a's tokens only (the contract is S1 == S0)
    ('    fn = _token_diffs(arms, ("S1a", "S1b", "S1c", "S1d"), "S0a", lens)',
     '    fn = _token_diffs(arms, ("S1b", "S1c", "S1d"), "S1a", lens)'),
    # S1's own timed reps unchecked
    ('lens) + _rep_diffs(arms, ("S1a", "S1b", "S1c", "S1d"))', 'lens)'),
    # K25's engagement never checked
    ("        void += k25_faults(t, r, t[:2])\n", ""),
    # the build count not exact
    ('    if r.get("k25_seen_build") != {key: K25_BUILD_CALLS}:',
     '    if not (r.get("k25_seen_build") or {}).get(key):'),
    # the licence's probes dropped from the closed form (the reviewed 576)
    ("LICENCE_FORWARDS, WARMUPS, CAPTURES = 2, 2, 1", "LICENCE_FORWARDS, WARMUPS, CAPTURES = 0, 2, 1"),
    # the early gate blind to the GEMV tally
    ("    if bw != BW_BUILD_CALLS:\n        out.append(f\"{r.get('tag') or arm}: bw_prmt32", "    if False:\n        out.append(f\"{r.get('tag') or arm}: bw_prmt32"),
    # the S1a gate reading S1a as today
    ('    out = [x for x in k25_faults(r.get("tag") or arm, r, arm)', '    out = [x for x in k25_faults(r.get("tag") or arm, r, "S0")'),
    # the GEMV's build tally not exact
    ('    if (r.get("dispatch_build") or {}).get("bw_prmt32") != BW_BUILD_CALLS:', '    if False:'),
    # the decode's resolution unchecked: an env-named or refused decode accepted
    ('    if {f: k.get(f) for f in want_lut} != want_lut:', '    if k.get("lut") != lut:'),
    # the decode's source unchecked (an explicit prmt is not qwen3_moe's default)
    ('want_lut = {"env": None, "lut": lut, "why": None, "source": "default-allowlisted"}',
     'want_lut = {"env": None, "lut": lut, "why": None}'),
    # the plan's resolution unchecked
    ('    if (kp.get("env"), kp.get("plan"), kp.get("why")) != (plan, plan, None):', '    if False:'),
    # the plan record's build count not exact
    ('    if r.get("k25_plan_seen_build") != {pkey: K25_BUILD_CALLS}:', '    if not (r.get("k25_plan_seen_build") or {}).get(pkey):'),
    # another plan accepted after the runs
    ('    if set(pafter) != {pkey} or pafter.get(pkey, 0) < K25_BUILD_CALLS:', '    if pafter.get(pkey, 0) < K25_BUILD_CALLS:'),
    # the spy never read at the build: the record's word taken for the kernel's shapes
    ('    if r.get("k25_shapes_build") != want_sh:', '    if False:'),
    # another shape accepted after the runs
    ('    if set(sh_after) != set(want_sh) or any(', '    if any('),
    # the spy's count not tied to the plan's
    ('    if sh_after and n_spy != KERNEL_CALLS_PER_K25 * n_plan:', '    if False:'),
    # the plan and decode records not tied
    ('    if n_plan != n_dec:', '    if False:'),
    # c1's pair swapped (down's shape for gate_up's)
    ('             "c1": ({"block_n": 64, "kc": 256, "warps": 4, "stages": 2}, {"block_n": 128, "kc": 128, "warps": 8, "stages": 3})}',
     '             "c1": ({"block_n": 128, "kc": 128, "warps": 8, "stages": 3}, {"block_n": 128, "kc": 128, "warps": 8, "stages": 3})}'),
    # the early gate blind to the plan's resolution
    (' or "k25_plan" in x or "before the build" in x]', ' or "before the build" in x]'),
    # another decode accepted after the runs
    ('    if set(after) != {key} or after.get(key, 0) < K25_BUILD_CALLS:', '    if after.get(key, 0) < K25_BUILD_CALLS:'),
    # the run counts not compared across processes
    ('        if len(set(runs.values())) != 1:', '        if False:'),
    # the int8 head's default unchecked
    ("        void += default_faults(t, r)\n", ""),
    # an unreadable enforced limit accepted
    ("                    if pl is None or pl < MIN_POWER_W:", "                    if pl is not None and pl < MIN_POWER_W:"),
    # only power.limit read, not the enforced limit
    ('                for f in ("power.limit", "enforced.power.limit"):', '                for f in ("power.limit",):'),
    # the floor lowered to a capped board's
    ("MIN_POWER_W = 575 ", "MIN_POWER_W = 450 "),
    # an explicit cap of 16 taken for the default
    ('    if sc.get("cap") != CAP[arm] or sc.get("source") != "default-allowlisted":', '    if sc.get("cap") != CAP[arm]:'),
    # the clock label one-sided
    ("        elif 100 * abs(ca - cb) > CLOCK_TOL_PCT * cb:", "        elif 100 * (ca - cb) > CLOCK_TOL_PCT * cb:"),
    # an unread clock read as a value
    ("    return statistics.median(xs) if xs else None", "    return statistics.median(xs) if xs else 0.0"),
]


@pytest.mark.parametrize("old,new", REDUCER_MUTANTS)
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p161_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p161_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p161_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


BOX_MUTANTS = [
    # K25's mode may be set in an arm
    ("    bad.update({k: env.get(k) for k in (K25_KNOB, LUT_KNOB, HEAD_KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS",
     "    bad.update({k: env.get(k) for k in (LUT_KNOB, HEAD_KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS"),
    # K25's decode may be set in an arm (the subject is qwen3_moe's default, prmt)
    ("    bad.update({k: env.get(k) for k in (K25_KNOB, LUT_KNOB, HEAD_KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS",
     "    bad.update({k: env.get(k) for k in (K25_KNOB, HEAD_KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS"),
    # the int8 head may be set in an arm
    ("    bad.update({k: env.get(k) for k in (K25_KNOB, LUT_KNOB, HEAD_KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS",
     "    bad.update({k: env.get(k) for k in (K25_KNOB, LUT_KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS"),
    # the arms' values swapped
    ('ARM_VALUE = {"S0": "today", "S1": "c1"}', 'ARM_VALUE = {"S0": "c1", "S1": "today"}'),
    # the spy without K25's signature: the route's lean-glue check would read the spy, not the kernel
    ("    @functools.wraps(k25)\n    def spy(", "    def spy("),
    # the spy's key without the decode
    ('    return "/".join(str(kw.get(k)) for k in ("block_n", "kc", "warps", "stages", "lut"))',
     '    return "/".join(str(kw.get(k)) for k in ("block_n", "kc", "warps", "stages"))'),
    # the spy counting nothing
    ("        SHAPES[key] = SHAPES.get(key, 0) + 1\n", ""),
    # an unreadable field read as zero
    ("        return float(v)\n    except (TypeError, ValueError):\n        return None",
     "        return float(v)\n    except (TypeError, ValueError):\n        return 0.0"),
    # a malformed sample line kept
    ("        if len(vals) != len(SAMPLE):\n            continue", "        if not vals:\n            continue"),
]


@pytest.mark.parametrize("old,new", BOX_MUTANTS)
def test_each_box_mutant_fails_the_self_test(box_dir, tmp_path, old, new):
    src = (LANE / "p161_box.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p161_box.py").write_bytes(src.replace(old, new).encode("utf-8"))
    shutil.copyfile(box_dir / "p109_box.py", tmp_path / "p109_box.py")
    out = _run(tmp_path / "p161_box.py", "--self-test", cwd=tmp_path)
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P161_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


@pytest.mark.parametrize("run_id,ok", [("p161-5090-1", True), ("p161-5090-3", True), ("p161-5090-4", False),
                                       ("p153-5090-1", False)])
def test_the_driver_takes_three_run_ids_and_no_other(tmp_path, run_id, ok):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p161_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, run_id), timeout=120)
    if ok:
        assert out.returncode == 0 and "DRYRUN stage -> root@h:/root/p161" in out.stdout and "bash p161_run.sh" in out.stdout, \
            out.stdout + out.stderr
    else:
        assert out.returncode == 78 and "is not p161-5090-N" in out.stdout + out.stderr
