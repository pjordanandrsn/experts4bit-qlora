"""P125's staged-file pin must match the repo (the e4b#642 check, mirrored for lane P125; e4b#1313).

`bench/p125/p125_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p125/staged-p125.sha256`;
this test runs the same comparison in CI. It also runs the lane's self-tests and pins its shape:
- P115 Phase D's quality instrument and the boxes it imports, P109's box, P39's bake and calibration and Phase D's
  premise tests at the bytes Phase D pinned;
- the arms: one lever each (A none, B calibrated int4 attention, C + the calibrated int4 head, M RTN attention, K B with
  its scales rolled), the runner's levers equal to the box's, the speed arms palindromic, the quality arm A first;
- the gates as the maintainer ruled them: K8's budget in nats from A's own windows, argmax 0.95, the window counts sized
  on Phase D's per-window spread (SE <= bound / 3), the UNDERPOWERED rung, the sure-fail mutant as the VOID rung;
- the reducer reads records shaped as the box writes them (P115 Phase D's Amendment 4 lesson);
- the order, the subject (every lever unset but the arm's), every time-left check inside its guard, the exit codes.
"""
import ast
import hashlib
import math
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p125"
PIN = LANE / "staged-p125.sha256"
SOURCES = {
    "p125_run.sh": LANE / "p125_run.sh",
    "p125_box.py": LANE / "p125_box.py",
    "p125_reduce.py": LANE / "p125_reduce.py",
    "p115_quality.py": REPO / "bench" / "p115" / "p115_quality.py",
    "p110_box.py": REPO / "bench" / "p110" / "p110_box.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "p109_box.py": REPO / "bench" / "p109" / "p109_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
    "test_kv_step_select.py": REPO / "tests" / "test_kv_step_select.py",
    "test_fused_glue_decode_graphs_gpu.py": REPO / "tests" / "test_fused_glue_decode_graphs_gpu.py",
    "test_gemv_bw_served_gpu.py": REPO / "tests" / "test_gemv_bw_served_gpu.py",
    "test_int4_attn.py": REPO / "tests" / "test_int4_attn.py",
    "test_int4_attn_calib.py": REPO / "tests" / "test_int4_attn_calib.py",
}
PHASE_D = ("p115_quality.py", "p110_box.py", "p108_box.py", "p97_box.py", "p109_box.py", "k8_bake.py", "calib.json",
           "test_decode_graph_buckets.py", "test_kv_step_select.py", "test_fused_glue_decode_graphs_gpu.py",
           "test_gemv_bw_served_gpu.py")
PREMISE = PHASE_D[7:] + ("test_int4_attn.py", "test_int4_attn_calib.py")
RUN = (LANE / "p125_run.sh").read_text()
DRIVE = (LANE / "p125_drive.sh").read_text()
PREREG = (LANE / "PREREG-p125.md").read_text(encoding="utf-8")


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _env():
    base = {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ}}
    return {"PYTHONPATH": str(REPO / "bench" / "p109"), **base}


def _import(name, *dirs):
    sys.path[:0] = [str(d) for d in dirs]
    try:
        return __import__(name)
    finally:
        del sys.path[:len(dirs)]


def _box():
    return _import("p125_box", LANE, REPO / "bench" / "p109")


def _reduce():
    return _import("p125_reduce", LANE)


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next P125 launch refuses ON A RENTED BOX"


def test_borrowed_files_run_at_phase_d_s_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    phase_d = dict((n, w) for w, n in _entries(REPO / "bench" / "p115" / "staged-d.sha256"))
    for name in PHASE_D:
        assert mine[name] == phase_d[name], f"{name} is not at P115 Phase D's bytes"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged-p125.sha256"]:
        if name != "p125_run.sh":
            assert name in staged, name
    assert 'done < "$HERE/staged-p125.sha256"' in DRIVE
    assert "--include 'work/' --include 'work/bake.json' --exclude 'work/*'" in DRIVE, "the reference log-probs stay on the box"


def test_the_self_tests_pass():
    for script, want in (("p125_reduce.py", "p125_reduce self-test OK (25 cases)"),
                         ("p125_box.py", "p125_box self-test OK (16 cases)")):
        out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True, env=_env())
        assert out.returncode == 0 and want in out.stdout, out.stdout + out.stderr


def test_the_gates_are_the_maintainer_s_ruling():
    box, red = _box(), _reduce()
    assert {g[0]: g[1:] for g in box.GATES} == red.GATES, "the box and the reducer read the same gates"
    assert red.RULED == ("t1", "k16") and red.K8_BUDGET == 0.05 and red.ARGMAX_MIN == 0.95 and red.POWER == 3.0
    nll = [2.0, 2.2, 2.4]
    assert abs(red.bound_of(nll) - math.log(1 + 0.05 / math.exp(2.2))) < 1e-12, "K8's budget in nats, from A's own ppl"
    p = red.PRIOR
    assert (p["sd_d"], p["ppl_R"]) == (0.01957, 8.762), "Phase D's p115d-5090-1 wikitext windows"
    assert abs(math.log(1 + 0.05 / p["ppl_R"]) - p["bound"]) < 5e-6
    need = math.ceil((p["sd_d"] / (p["bound"] / 3)) ** 2)
    assert need == p["windows_needed"] == 107
    t1, k16 = red.GATES["t1"], red.GATES["k16"]
    assert t1[1] == 1 and t1[2] >= need and k16[1] == 16 and k16[2] >= need and k16[2] % 16 == 0
    assert t1[2] == 108 and k16[2] == 112, "the registered counts: 107 rounded up (to a multiple of 16 for K16)"
    assert red.CONT == 128 and "--cont" not in RUN, "128 continuation positions, P115's default, never overridden"
    g = red.gate([2.0] * 4, [2.0 + 0.01 * (i % 2) for i in range(4)], [0.99] * 4)
    assert g["outcome"] == "UNDERPOWERED", g
    assert red.gate([2.0] * 12, [3.0] * 12, [0.3] * 12)["outcome"] == "FAIL", "a gross fail is decisive at 12 windows"
    for word in ("UNDERPOWERED", "ln(1 + 0.05 / ppl_A)", "107", "0.01957", "rolled", "RTN", "auto slot", "first prefill"):
        assert word in PREREG, word


def test_the_arms_are_one_lever_each_and_the_runner_sets_the_box_s():
    box, red = _box(), _reduce()
    assert box.ARMS == red.QUALITY_ARMS == ("A", "B", "C", "M", "K") and box.SPEED_ARMS == ("A", "B", "C")
    lever = RUN[RUN.index("lever(){"):RUN.index("esac; }") + 7]
    for arm, env in box.ARM_ENV.items():
        set_ = " ".join(f"{k}={v}" for k, v in env.items())
        assert re.search(rf"(?:^|[\s(;])(?:[A-Z]\|)*{arm}(?:\|[A-Z])*\) echo \"{re.escape(set_)}\"", lever), (arm, set_, lever)
        assert box.arm_env_ok(arm, env)[0]
        for other in box.LEVERS:
            if other not in env:
                assert not box.arm_env_ok(arm, {**env, other: "1"})[0], (arm, other)
    assert red.SPEED_TAGS == ("A1", "B1", "C1", "C2", "B2", "A2") and "for TAG in A1 B1 C1 C2 B2 A2; do" in RUN
    assert "for ARM in A B C M K; do" in RUN, "A first: it writes the references"
    quality = RUN[RUN.index("for ARM in A B C M K; do"):RUN.index('PF=""; [ "$PROVE" = 1 ]')]
    speed = RUN[RUN.index("for TAG in A1 B1 C1 C2 B2 A2; do"):RUN.index("for ARM in A B C M K; do")]
    assert "E4B_PAGED_GRAPHS=0" in quality and "E4B_PAGED_GRAPHS" not in speed, "quality eager, speed on the graph default"


def test_the_subject_is_the_shipped_default_with_the_arm_s_lever_alone():
    box = _box()
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    for knob in box.LEVERS + box.KNOBS + box.GEMV_KNOBS + ("E4B_PAGED_GRAPHS", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_BUCKETS",
                                                           "E4B_INT4_WIDE_TILES", "TRITON_INTERPRET"):
        assert re.search(rf"\b{knob}\b", unset), knob
    assert ('ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena '
            'E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"') in RUN
    after = "\n".join(line for line in RUN.split(": > summary.txt")[1].splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"E4B_PAGED_FUSE_QKV=|E4B_FUSE_T1_GLUE=|E4B_FUSE_ROUTER_EPI=|GNF4_GEMV_BW=|E4B_CALIB_\w+=", after)
    src = (LANE / "p125_box.py").read_text()
    assert "arm_env_ok(arm, os.environ)" in src and '(16, "all-vram", (1, 2, 4, 8, 16))' in src


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["CUDA_PROBE=$(python -", "REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @',
             "python - <<'PYT'", "p125_reduce.py --self-test", "p125_box.py --self-test",
             "python -m pytest " + " ".join(PREMISE), 'echo "premise ok"', 'say "fetch $MODEL @ $REV"',
             "C4_FETCH shard", "python $W/k8_bake.py", "p109_box.py --prompts-only", "for TAG in A1 B1 C1 C2 B2 A2; do",
             "for ARM in A B C M K; do", "python $W/p125_reduce.py --dir $W --out $W/verdict_p125.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'echo "$LASTL" | grep -q "39 passed"' in RUN and "39 passed" in PREREG
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    for s in ('sp.resolve_fusion_modes(dict(unset), "qwen3_moe")', '(Int4Linear.GEMV_ROWS_MAX, Int4Linear.SMALLM_ROWS_MAX) == (1, 16)',
              "calibrate_attention_hessians", "gemv_int4_b32", 'md.version("grouped-nf4-gemm") == "0.43.0"',
              'transformers.__version__ == "5.17.0"', 'E4B_CALIB_SOURCE") is None'):
        assert s in trip, s


def test_every_time_left_check_fits_its_own_guard():
    assert "guard 0.75 h" in PREREG and "guard 2.5 h" in PREREG
    keys = {"FETCH", "BAKE", "ARM", "QUALITY"}
    prove = dict(re.findall(r"NEED_(FETCH|BAKE|ARM|QUALITY)=(\d+)", RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("else\n")]))
    reading = dict(re.findall(r"NEED_(FETCH|BAKE|ARM|QUALITY)=(\d+)", RUN[RUN.index("else\n"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(prove) == set(reading) == keys
    for need in prove.values():
        assert int(need) + 600 <= 0.75 * 3600 - 900, prove
    for need in reading.values():
        assert int(need) + 600 <= 2.5 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_BAKE", "$NEED_ARM", "$NEED_QUALITY"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen
    assert {9, 10, 11, 12, 15, 16, 19, 21, 22, 25, 27, 40} <= codes


def _rec_keys(fn_names):
    """Every string key the named functions write: dict literals and subscript stores (``rec["memory"]["x"] = ...``)."""
    tree = ast.parse((LANE / "p125_box.py").read_text())
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in fn_names:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    keys |= {k.value for k in sub.keys if isinstance(k, ast.Constant)}
                if isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store) and isinstance(sub.slice, ast.Constant):
                    keys.add(sub.slice.value)
    return keys


def test_the_reducer_reads_records_shaped_as_the_box_writes_them():
    red = _reduce()
    speed_keys = _rec_keys({"speed_main", "_common"})
    fake = red.fake_speed("B1")
    assert set(fake) <= speed_keys, set(fake) - speed_keys
    assert set(fake["memory"]) <= speed_keys, set(fake["memory"]) - speed_keys
    quality_keys = _rec_keys({"quality_main", "_common"})
    fq = red.fake_quality("K")
    assert set(fq) <= quality_keys | {"workloads"}, set(fq) - quality_keys
    gate = next(iter(fq["gates"].values()))
    q = (REPO / "bench" / "p115" / "p115_quality.py").read_text()
    assert set(gate) - {"per_window"} <= quality_keys and '"per_window"' in q, "per_window comes from measure_phase"
    for k in ("nll", "argmax_agree", "window"):
        assert f'"{k}"' in q, k
    box = _box()
    assert set(red.fake_speed("C1")["slots"]) == {str(t) for t in box.SLOT_TOKENS}
    src = (LANE / "p125_box.py").read_text()
    assert '"corrected_slots"' in src and '"free_before_load"' in src and '"peak_first_prefill"' in src


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p125-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P125_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p125_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p125"), out.stdout + out.stderr
