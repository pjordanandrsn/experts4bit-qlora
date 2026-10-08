"""P123's staged-file pin must match the repo (the e4b#642 check, mirrored for lane P123; e4b#1313).

`bench/p123/p123_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p123/staged-p123.sha256`;
this test runs the same comparison in CI. It also runs the lane's self-tests and pins its shape:
- P109's box, P39's bake and calibration and the premise tests at the bytes P115 Phase D pinned; SC1b's census reducer
  and window at SC1b's bytes;
- the v1 class map: SC1b's v0 e4b map with only the NF4 expert kernels added;
- the reducer reads a census record that SC1b's own `arm()` writes (synthetic exports), and its fakes carry that
  record's keys (P115 Phase D's Amendment 4 lesson);
- the order: refusals, install and tripwire, self-tests, the premise on the card (19), nsys, fetch, bake, prompts,
  two speed arms, (the proof only, Amendment 1) the router build, four captures, the census arms, the reducer;
- Amendment 1: the router build sets E4B_FUSE_ROUTER_EPI=auto and no other lever, on the proof only, and its record
  carries the keys the reducer's fake carries;
- the subject: the shipped default (every lever unset, 16 slots named); every time-left check inside its guard; the
  exit codes.
"""
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p123"
SC1B = REPO / "bench" / "sc1b"
PIN = LANE / "staged-p123.sha256"
SOURCES = {
    "p123_run.sh": LANE / "p123_run.sh",
    "p123_box.py": LANE / "p123_box.py",
    "p123_census.py": LANE / "p123_census.py",
    "p123_reduce.py": LANE / "p123_reduce.py",
    "kernel_classes_nf4.json": LANE / "kernel_classes_nf4.json",
    "sc1b_census.py": SC1B / "sc1b_census.py",
    "sc1b_e4b_census.py": SC1B / "sc1b_e4b_census.py",
    "p109_box.py": REPO / "bench" / "p109" / "p109_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
    "test_kv_step_select.py": REPO / "tests" / "test_kv_step_select.py",
    "test_fused_glue_decode_graphs_gpu.py": REPO / "tests" / "test_fused_glue_decode_graphs_gpu.py",
    "test_gemv_bw_served_gpu.py": REPO / "tests" / "test_gemv_bw_served_gpu.py",
}
PREMISE = ("test_decode_graph_buckets.py", "test_kv_step_select.py", "test_fused_glue_decode_graphs_gpu.py",
           "test_gemv_bw_served_gpu.py")
RUN = (LANE / "p123_run.sh").read_text()
DRIVE = (LANE / "p123_drive.sh").read_text()
PREREG = (LANE / "PREREG-p123.md").read_text(encoding="utf-8")
NF4_EXPERTS = ("_gemv_nf4_bw", "_gemv_nf4_dotpad", "_gemv_nf4_dotpad_splitk", "_gemv_nf4_grouped",
               "_gemv_nf4_grouped_splitk", "_gemm_nf4_grouped", "_gemm_nf4_grouped_smallm")


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _env():
    base = {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ}}
    sep = ";" if sys.platform.startswith("win") else ":"
    return {"PYTHONPATH": sep.join(str(p) for p in (REPO / "bench" / "p109", SC1B)), **base}


def _import(name, *dirs):
    sys.path[:0] = [str(d) for d in dirs]
    try:
        return __import__(name)
    finally:
        del sys.path[:len(dirs)]


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next P123 launch refuses ON A RENTED BOX"


def test_borrowed_files_run_at_their_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    phase_d = dict((n, w) for w, n in _entries(REPO / "bench" / "p115" / "staged-d.sha256"))
    for name in ("p109_box.py", "k8_bake.py", "calib.json") + PREMISE:
        assert mine[name] == phase_d[name], f"{name} is not at P115 Phase D's bytes"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged-p123.sha256"]:
        if name != "p123_run.sh":
            assert name in staged, name
    assert 'done < "$HERE/staged-p123.sha256"' in DRIVE
    assert "--exclude 'census/*.nsys-rep' --exclude 'census/*.sqlite'" in DRIVE


def test_the_self_tests_pass():
    for script, arg, want in (("p123_reduce.py", "--self-test", "p123_reduce self-test OK (23 cases)"),
                              ("p123_box.py", "--self-test", "p123_box self-test OK (9 cases)")):
        out = subprocess.run([sys.executable, str(LANE / script), arg], capture_output=True, text=True, env=_env())
        assert out.returncode == 0 and want in out.stdout, out.stdout + out.stderr
    out = subprocess.run([sys.executable, str(SC1B / "sc1b_census.py"), "--self-test"], capture_output=True, text=True,
                         env=_env())
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_the_class_map_is_sc1b_v0_with_only_the_nf4_experts_named():
    v0 = json.loads((SC1B / "kernel_classes.json").read_text())["e4b"]
    v1 = json.loads((LANE / "kernel_classes_nf4.json").read_text())["e4b"]
    assert v1["rules"] == v0["rules"] and v1["segment"] == v0["segment"]
    assert v1["inherit_next"] == v0["inherit_next"] and v1["inherit_prev"] == v0["inherit_prev"]
    assert set(v1["expert_names"]) == set(v0["expert_names"]) | set(NF4_EXPERTS)
    assert set(v1["matmul_names"]) == set(v0["matmul_names"]) | set(NF4_EXPERTS)


def _census_record(tmp_path, batch, moe_layers=1):
    """A census arm record written by SC1b's own arm() from synthetic exports: per replay one layer of the default NF4
    step (rmsnorm, the fused q/k/v GEMV, attention, the router epilogue, two bandwidth GEMVs and swiglu, the combine)."""
    cen = _import("sc1b_census", SC1B)
    spec = cen.load_classes(str(LANE / "kernel_classes_nf4.json"), "e4b")
    us = 1000
    layer = [("_rmsnorm_rows", 30), ("gemv2T_kernel_val", 200), ("_fp8_paged_decode_split", 60), ("_router_epilogue", 20),
             ("_gemv_nf4_bw", 120), ("_swiglu_rows", 10), ("_gemv_nf4_bw", 60), ("_combine_rows", 10)]
    nk, nl, gg, gk = [], [], [], []
    for i in range(10):
        t, c = 1_000_000 * (i + 1), 1000 + i
        nl.append((t - 20, t - 10, c))
        x = t
        for j, (name, dur) in enumerate(layer):
            nk.append((x, x + dur * us, c, j + 1, name, (8, 1, 1)))
            x += (dur + 5) * us                      # a 5 us launch gap between kernels: in-graph idle
        gg.append((t, x, 4000 + i, 5))
        gk.append((x + 20 * us, x + 40 * us, 5000 + i, None, "ArgMaxOps", (1, 1, 1)))
    pn, pg = tmp_path / f"n{batch}.sqlite", tmp_path / f"g{batch}.sqlite"
    cen._mkdb(str(pn), kernels=nk, launches=nl)
    cen._mkdb(str(pg), kernels=gk, graphs=gg)
    return cen.arm(cen.load(str(pg)), cen.load(str(pn)), "e4b", batch, spec, unprofiled_ms=1.0, min_steps=3,
                   moe_layers=moe_layers)


def test_the_reducer_reads_the_census_record_sc1b_writes(tmp_path):
    r = _import("p123_reduce", LANE)
    c1, c16 = _census_record(tmp_path, 1), _census_record(tmp_path, 16)
    assert c1["status"] in ("ok", "labelled") and "CLASS_MAP_INCOMPLETE" not in c1["labels"], c1["labels"]
    assert c1["terms"]["moe_expert"] > 0 and c1["terms"]["dense_gemm"] > 0 and c1["terms"]["attn"] > 0
    v = r.reduce({"S1a": r.fake_arm("S1a", w1=1.0), "S1b": r.fake_arm("S1b", w1=1.0)},
                 {b: r.fake_drv(b) for b in r.BATCHES}, {1: c1, 16: c16}, r.E)
    assert v["verdict"] == "READ", v["reasons"]
    b1 = v["batches"]["1"]
    assert abs(b1["RESIDUAL"]["ms"] - (b1["P_ms"] - sum(b1["classes_ms"].values()))) < 1e-3
    fake = r.fake_census(1)
    assert set(fake) <= set(c1), set(fake) - set(c1)
    assert set(fake["terms"]) == set(c1["terms"])
    assert set(fake["gates"]) <= set(c1["gates"]) and "kernels_per_graph_modal" in c1["node"]


def test_amendment_1_the_proof_licenses_every_router_under_the_router_knob_alone():
    """The router build runs on the proof only, after the speed arms, with E4B_FUSE_ROUTER_EPI=auto and nothing else; the
    reducer VOIDs the proof unless all of Granite's 32 routers license (fam-prove-1 read 27; #1398's regression check),
    and its fake router record carries the keys ``routers_main`` writes and the fold's own report keys."""
    import ast
    r = _import("p123_reduce", LANE)
    box = _import("p123_box", LANE, REPO / "bench" / "p109")
    assert box.routers_env_ok({"E4B_FUSE_ROUTER_EPI": "auto"})[0]
    assert not box.routers_env_ok({"E4B_FUSE_ROUTER_EPI": "auto", "E4B_PAGED_FUSE_QKV": "auto"})[0]
    assert not box.routers_env_ok({"E4B_FUSE_ROUTER_EPI": "auto", "GNF4_GEMV_BW": "1"})[0]
    step = RUN[RUN.index('if [ "$PROVE" = 1 ] && [ "$ARMS_OK" = 1 ]; then'):RUN.index("# ---- the census")]
    assert "$ENGINE_ENV E4B_FUSE_ROUTER_EPI=auto E4B_SHA" in step and "--out $W/router_census.json" in step
    fn = next(n for n in ast.walk(ast.parse((LANE / "p123_box.py").read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "routers_main")
    rec = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "rec")
    assert {k.value for k in rec.keys} == set(r.fake_routers()), "the fake router record drifted from routers_main's"
    fold = (REPO / "experts4bit_qlora" / "engines" / "router_epilogue.py").read_text()
    for key in r.fake_routers()["router_report"]:
        assert f"{key}=" in fold, f"the fold's report has no {key!r}"
    gran = {"S1a": r.fake_arm("S1a", r.GRAN), "S1b": r.fake_arm("S1b", r.GRAN)}
    drvs = {b: r.fake_drv(b, r.GRAN) for b in r.BATCHES}
    cen = {1: r.fake_census(1), 16: r.fake_census(16, p=17.0)}
    assert r.reduce(gran, drvs, cen, r.E, proof=True, routers=r.fake_routers())["routers"]["licensed"] == 32
    assert r.reduce(gran, drvs, cen, r.E, proof=True, routers=r.fake_routers(n=27))["verdict"] == "VOID"
    assert r.reduce(gran, drvs, cen, r.E, proof=True)["verdict"] == "VOID"
    assert "Amendment 1" in PREREG and "router_census.json" in PREREG


def test_the_rule_is_the_registered_rule():
    r = _import("p123_reduce", LANE)
    assert r.GNF4_SHA == "6ee2e10408161a9d3c874975c9191a7f2957e6f4" and f"GNF4_SHA={r.GNF4_SHA}" in RUN
    for model, rev in r.REVS.items():
        assert rev in RUN, model
    assert r.DEFAULT[r.QWEN]["census"] == {"fuse_qkv_n": 48, "fuse_t1_glue_n": 193, "fuse_t1_glue_r2_n": [48, 48],
                                           "fuse_router_epilogue_n": 48}, "the default's Qwen3 census (P115)"
    assert r.DEFAULT[r.QWEN]["source"] == "default-allowlisted" and r.DEFAULT[r.GRAN]["source"] == "default-off"
    assert r.STEPS == 64 and "SKIP=32; STEPS=64" in RUN and r.SELF_PAIR == (0.97, 1.03)
    assert set(r.BLOCKING) == {"CLASS_MAP_INCOMPLETE", "CLASS_MAP_SEGMENT_BROKEN", "NSYS_DIAGNOSTIC_ERRORS", "CLOCK_MISMATCH"}
    cen = _import("sc1b_census", SC1B)
    assert r.CLASSES == cen.CLASSES, "the census's classes, unchanged"
    assert "MOE_LAYERS=48" in RUN and "MOE_LAYERS=32" in RUN and "--moe-layers $MOE_LAYERS" in RUN
    for b in r.BATCHES:
        assert set(r.PREDICTED[b]) >= {"dense_gemm", "moe_expert", "RESIDUAL", "busy_fraction"}
    assert "READ" in PREREG and "NOISY" in PREREG and "RESIDUAL" in PREREG and "busy fraction" in PREREG


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["CUDA_PROBE=$(python -", "REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @',
             "python - <<'PYT'", "p123_reduce.py --self-test", "p123_box.py --self-test", "sc1b_census.py --self-test",
             "python -m pytest " + " ".join(PREMISE), 'echo "premise ok"', 'say "fetch $MODEL @ $REV"',
             "python $W/k8_bake.py", "p109_box.py --prompts-only", "for TAG in S1a S1b; do",
             'if [ "$PROVE" = 1 ] && [ "$ARMS_OK" = 1 ]; then', "p123_box.py --mode routers", "  install_nsys\n",
             "for B in 1 16; do", "for MODE in graph node; do", "sc1b_census.py arm --graph",
             "python $W/p123_reduce.py --dir $W --out $W/verdict_p123.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'echo "$LASTL" | grep -q "19 passed"' in RUN and "19 passed" in PREREG
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert 'sp.resolve_fusion_modes(dict(unset), "qwen3_moe")' in trip and '{"default-allowlisted"}' in trip
    assert 'sp.resolve_fusion_modes(dict(unset), "granitemoe")' in trip and '{"default-off"}' in trip
    assert 'ng._bw() == "auto" and {(1536, 2048), (2048, 768)} <= set(ng._BW_SHAPES)' in trip
    assert 'md.version("grouped-nf4-gemm") == "0.43.0"' in trip and 'transformers.__version__ == "5.17.0"' in trip


def test_the_subject_is_the_shipped_default():
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in ("E4B_PAGED_GRAPHS", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2",
                 "E4B_FUSE_ROUTER_EPI", "E4B_PAGED_DECODE_LOOKAHEAD", "GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_PDL",
                 "TRITON_INTERPRET", "E4B_PAGED_STEP_TRACE"):
        assert re.search(rf"\b{knob}\b", unset), knob
    assert ('ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena '
            'E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"') in RUN
    after = RUN.split(": > summary.txt")[1]
    assert not re.search(r"E4B_PAGED_FUSE_QKV=|E4B_FUSE_T1_GLUE=|GNF4_GEMV_BW=", after)
    code = "\n".join(line for line in after.splitlines() if not line.lstrip().startswith("#"))
    assert re.findall(r"E4B_FUSE_ROUTER_EPI=\S*", code) == ["E4B_FUSE_ROUTER_EPI=auto"], "only Amendment 1's build"
    box = (LANE / "p123_box.py").read_text()
    census = (LANE / "p123_census.py").read_text()
    assert "default_env_ok(os.environ)" in box and "default_env_ok(os.environ)" in census
    assert "(cfg.max_seqs, cfg.placement) != (16, \"all-vram\")" in box and "from sc1b_e4b_census import census_window" in census


def test_every_time_left_check_fits_its_own_guard():
    assert "guard 0.75 h" in PREREG and "guard 1.5 h" in PREREG
    keys = {"FETCH", "BAKE", "ARM", "CAPTURE"}
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|ARM|CAPTURE)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|ARM|CAPTURE)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == keys
    for need in prove.values():
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 1.5 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_ARM", "$NEED_ARM", "$NEED_CAPTURE"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen
    assert {9, 10, 15, 16, 21, 22, 25, 26, 27, 40} <= codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p123-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P123_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p123_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p123"), out.stdout + out.stderr
