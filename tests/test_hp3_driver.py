"""The qwen3hp3 / qwen3hp3shake tokens' control flow (bench/tc1/tc1_run.sh, tc1_hp3_family) with every harness call stubbed, on the
same harness as test_hp1_driver: which arms it starts, in which venv, when it ends the draw and when it signals the (R) fallback.
Bash only; no model, no GPU."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_hp1_driver as D  # noqa: E402

HP3_PY = r"""  *"hp3_reduce.py g1"*) out=$(echo "$*" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
    case "${STUB_H3G1:-pass}" in
      close) echo '{"verdict": "CLOSE", "frameworks": {}}' > "$out"; echo "HP3_G1=CLOSE"; exit 4;;
      partial) echo '{"verdict": "PARTIAL", "roundtrip_zero_centred_tol": 0.01, "frameworks": {"e4b": {"g1": "PASS"}, "unsloth": {"g1": "PASS"}, "axolotl": {"g1": "CLOSE"}}}' > "$out"; echo "HP3_G1=PARTIAL";;
      *) echo '{"verdict": "PASS", "roundtrip_zero_centred_tol": 0.01, "frameworks": {"e4b": {"g1": "PASS"}, "unsloth": {"g1": "PASS"}, "axolotl": {"g1": "PASS"}}}' > "$out"; echo "HP3_G1=PASS";;
    esac;;
  *"hp3_reduce.py verdict"*) out=$(echo "$*" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
    case "${STUB_H3VOID:-}" in
      ax) echo '{"void": {"axolotl seed 3410": "engagement: kernels on", "e4b seed 3411": "absent"}}' > "$out";;
      *) echo '{"void": {"e4b seed 3411": "absent", "e4b seed 3412": "absent"}}' > "$out";;
    esac; echo "HP3_VERDICT three_way=NOT_READ";;
  *seed_plan.json*"import json,sys,time"*|*"import json,sys,time"*seed_plan.json*) [ -n "${STUB_PLAN_FAIL:-}" ] && exit 1; python3 -c "${@:2}";;
  *"hp3_build.py classify"*) echo "${STUB_CLASS:-R | RuntimeError: an unsupported op}";;
"""


def drive(tmp_path, **env):
    """test_hp1_driver's harness with HP3's interpreters, the build flag, a per-arm HP3_VENV log and the HP3 reducer stubs. The
    shakedown token runs under a registered shakedown id unless a test names another (the runner refuses any other id)."""
    if env.get("FAMILY_FAM") == "qwen3hp3shake":
        env.setdefault("TC1_RUN_ID", "hp3-shake-1")
    if env.get("FAMILY_FAM") == "qwen3hp3rshake":
        env.setdefault("TC1_RUN_ID", "hp3r-shake-1")
    fakepy, harness = D.FAKEPY, D.HARNESS
    try:
        D.FAKEPY = fakepy.replace('  *"hp2_reduce.py verdict"*)', HP3_PY + '  *"hp2_reduce.py verdict"*)')
        D.HARNESS = (harness.replace("PY_E4B=$W/fakepy;", "PY_E4B=$W/fakepy; PY_H3E=$W/fakepy; PY_H3U=$W/fakepy; PY_H3A=$W/fakepy; H3_OK=${STUB_H3OK:-1}; "
                                     "H3_REASON=${STUB_H3REASON:-}; H3R_OK=${STUB_H3ROK:-1}; H3R_REASON=${STUB_H3RREASON:-}; DS_ALPACA_SHA=d;")
                     .replace('echo "$1 $2 $3 UNS_VENV=${UNS_VENV:-}" >> "$W/venv.log";',
                              'echo "$1 $2 $3 UNS_VENV=${UNS_VENV:-}" >> "$W/venv.log"; echo "$1 $2 $3 HP3_VENV=${HP3_VENV:-}" >> "$W/h3venv.log";')
                     .replace("5400 2400 3600 3600", "5400 2400 3600 5400 ${HP3_SHAKE:-0} ${HP3_MODE:-p}")
                     .replace("say(){", "left(){ echo ${STUB_LEFT:-100000}; }\nsay(){"))
        return D.drive(tmp_path, FAMILY_FN="tc1_hp3_family", **env)
    finally:
        D.FAKEPY, D.HARNESS = fakepy, harness


def test_the_read_runs_calibration_then_three_seeds_with_e4b_first_every_arm_in_its_hp3_venv(tmp_path):
    arms, summary = drive(tmp_path, FAMILY_FAM="qwen3hp3")
    t = [a[2] for a in arms if a[0] == "qwen3hp3"]
    assert t[:3] == ["hp3cal_e4b_k00_s3407", "hp3cal_unsloth_s3407", "hp3cal_axolotl_s3407"]
    assert t[3:] == [f"hp3_{r}_s{s}" for s in (3410, 3411, 3412, 3413, 3414) for r in ("k00", "unsloth", "axolotl")]
    assert all(ln.endswith("HP3_VENV=1") for ln in (tmp_path / "h3venv.log").read_text().splitlines())
    exporters = [a[2] for a in arms if a[0] == "qwen3hp3" and "--hp1-export-base" in a and not a[2].startswith("hp3cal")]
    assert exporters == ["hp3_k00_s3410", "hp3_unsloth_s3410", "hp3_axolotl_s3410"]
    assert (tmp_path / "fetch_allow.log").read_text().strip() == ""            # the read fetches the whole checkpoint
    assert "HP3 verdict:" in summary and "SHAKEDOWN" not in summary


def test_axolotl_trains_kernels_off_and_its_receipts_reach_the_reducer(tmp_path):
    arms, _ = drive(tmp_path, FAMILY_FAM="qwen3hp3")
    for a in arms:
        if a[1] == "axolotl":
            i = a.index("--axolotl-kernels")
            assert a[i + 1] == "off" and a[a.index("--axolotl-best") + 1] == "0", a
    py = (tmp_path / "py.log").read_text()
    ver = next(ln for ln in py.splitlines() if "hp3_reduce.py verdict" in ln)
    assert all(f"--arm-receipt {tmp_path}/qwen3hp3_axolotl_hp3_axolotl_s{s}.json" in ver for s in (3410, 3411, 3412, 3413, 3414))
    g1_eval = next(ln for ln in py.splitlines() if "hp1_eval.py" in ln and "eval_cal" in ln)
    assert all(f"base_{fw}_qwen3hp3" in g1_eval for fw in ("e4b", "unsloth", "axolotl"))


def test_a_failed_build_runs_nothing_and_the_shakedown_names_the_fallback(tmp_path):
    arms, summary = drive(tmp_path, FAMILY_FAM="qwen3hp3shake", HP3_SHAKE="1", STUB_H3OK="0", STUB_H3REASON="the three venvs are not one build")
    assert arms == [] and "HP3 BUILD_FAILED: the three venvs are not one build -> the (R) fallback signal" in summary


def test_the_shakedown_runs_the_2_layer_cut_with_one_seed_and_passes_on_non_void_runs(tmp_path):
    arms, summary = drive(tmp_path, FAMILY_FAM="qwen3hp3shake", HP3_SHAKE="1")
    t = [a[2] for a in arms if a[0] == "qwen3hp3shakeL2"]
    assert t == ["hp3cal_e4b_k00_s3407", "hp3cal_unsloth_s3407", "hp3cal_axolotl_s3407", "hp3_k00_s3410", "hp3_unsloth_s3410", "hp3_axolotl_s3410"]
    assert "model-00001-of-00016.safetensors model-00016-of-00016.safetensors" in (tmp_path / "fetch_allow.log").read_text()
    assert "HP3 SHAKEDOWN_OK" in summary


def test_a_shakedown_void_other_than_absent_fails_it(tmp_path):
    _, summary = drive(tmp_path, FAMILY_FAM="qwen3hp3shake", HP3_SHAKE="1", STUB_H3VOID="ax")
    assert "HP3 SHAKEDOWN_FAILED" in summary and "engagement" in summary


def test_a_whole_box_close_ends_the_draw_before_any_confirmation_arm(tmp_path):
    for fam, shake in (("qwen3hp3", "0"), ("qwen3hp3shake", "1")):
        w = tmp_path / fam
        w.mkdir()
        arms, summary = drive(w, FAMILY_FAM=fam, HP3_SHAKE=shake, STUB_H3G1="close")
        assert not [a for a in arms if not a[2].startswith("hp3cal")] and "HP3 G1_CLOSED: bases differ" in summary
        assert ("the (R) fallback signal" in summary) == (shake == "1")


def test_a_partial_g1_reads_the_surviving_pairs_on_the_read_and_signals_the_fallback_in_the_shakedown(tmp_path):
    (tmp_path / "r").mkdir()
    arms, summary = drive(tmp_path / "r", FAMILY_FAM="qwen3hp3", STUB_H3G1="partial")
    conf = [a for a in arms if not a[2].startswith("hp3cal")]
    assert len(conf) == 10 and not [a for a in conf if a[1] == "axolotl"]
    (tmp_path / "s").mkdir()
    arms, summary = drive(tmp_path / "s", FAMILY_FAM="qwen3hp3shake", HP3_SHAKE="1", STUB_H3G1="partial")
    assert not [a for a in arms if not a[2].startswith("hp3cal")]
    assert "is not PASS for every framework (PASS: e4b unsloth) -> the (R) fallback signal" in summary


def test_the_drive_stages_hp3s_pieces_for_both_tokens():
    s = (D.REPO / "bench" / "tc1" / "tc1_drive.sh").read_text()
    line = next(ln for ln in s.splitlines() if '" qwen3hp3 "' in ln and "STAGE=" in ln)
    for f in ("hp3_reduce.py", "hp3_build.py", "axo1767_reduce.py", "hp1_eval.py", "hp1_reduce.py", "hp1_trunc.py", "hp1_caltokens.py"):
        assert f in line, f
    assert '" qwen3hp3shake "' in line


def test_the_build_probe_finds_one_bnb_library_and_compare_names_each_mismatch(tmp_path):
    sys.path.insert(0, str(D.REPO / "bench" / "hp3"))
    import hp3_build as B
    lib = tmp_path / "libbitsandbytes_cuda130.so"
    lib.write_bytes(b"x")
    maps = "\n".join(f"7f0{i}000-7f0{i}fff r-xp 00000000 08:01 42 {lib}" for i in range(3)) + "\n7f09000-7f09fff r--p 0 0:0 0 /usr/lib/libc.so.6"
    assert B.bnb_lib_path(maps) == str(lib) and B.bnb_lib_path("") is None
    p = {"torch": "2.14.0+cu130", "cuda": "13.0", "bitsandbytes": "0.50.2", "bnb_lib_sha256": "a" * 64, "triton": "3.6.0"}
    probes = [dict(p, framework=f, transformers=t) for f, t in (("e4b", "5.18.0"), ("unsloth", "5.17.0"), ("axolotl", "5.17.0"))]
    assert B.compare(probes, "2.14.0", "0.50.2")["verdict"] == "MATCHED"         # transformers differ: reported, never compared
    probes[2] = dict(probes[2], torch="2.14.1+cu130")
    rec = B.compare(probes, "2.14.0", "0.50.2")
    assert rec["verdict"] == "MISMATCH" and any("axolotl: torch 2.14.1+cu130 != the pin 2.14.0" in x for x in rec["problems"])
    assert any(x.startswith("torch differs across venvs") for x in rec["problems"])
    json.dumps(rec)


def _probes(**over):
    p = {"torch": "2.14.0+cu130", "cuda": "13.0", "bitsandbytes": "0.50.2", "bnb_lib_sha256": "a" * 64, "triton": "3.6.0"}
    return [dict(p, framework=f, **over) for f in ("e4b", "unsloth", "axolotl")]


def test_compare_refuses_a_missing_bnb_library_hash_even_when_all_three_agree():
    sys.path.insert(0, str(D.REPO / "bench" / "hp3"))
    import hp3_build as B
    rec = B.compare(_probes(bnb_lib_sha256=None), "2.14.0", "0.50.2")
    assert rec["verdict"] == "MISMATCH" and any(x.startswith("bnb_lib_sha256 differs across venvs") for x in rec["problems"])


def test_compare_refuses_three_matching_venvs_that_are_not_the_pinned_cu130_build():
    sys.path.insert(0, str(D.REPO / "bench" / "hp3"))
    import hp3_build as B
    rec = B.compare(_probes(torch="2.14.0+cu128", cuda="12.8"), "2.14.0", "0.50.2")
    assert rec["verdict"] == "MISMATCH" and sum("is not a cu130 build" in x for x in rec["problems"]) == 3
    rec = B.compare(_probes(bitsandbytes="0.50.1"), "2.14.0", "0.50.2")
    assert rec["verdict"] == "MISMATCH" and sum("!= the pin 0.50.2" in x for x in rec["problems"]) == 3
    rec = B.compare(_probes(torch="2.13.1+cu130"), "2.14.0", "0.50.2")
    assert rec["verdict"] == "MISMATCH" and sum("!= the pin 2.14.0" in x for x in rec["problems"]) == 3


def test_classify_names_only_the_pre_named_host_classes_and_quotes_the_line():
    sys.path.insert(0, str(D.REPO / "bench" / "hp3"))
    import hp3_build as B
    oom = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
    assert B.classify(oom, 12288, 16384) == ("HOST_OOM_UNDER_FLOOR", oom)
    assert B.classify(oom, 24576, 16384)[0] == "R"                       # OOM on a card at or over the floor is the build's
    assert B.classify(oom, None, 16384)[0] == "R"
    assert B.classify("RuntimeError: CUDA error: GPU has fallen off the bus (NVRM: Xid 79)", 32768, 16384)[0] == "HOST_DRIVER"
    assert B.classify("RuntimeError: CUDA error: uncorrectable ECC error encountered", 32768, 16384)[0] == "HOST_DRIVER"
    assert B.classify("", 32768, 16384, reason="the instance was preempted by the provider")[0] == "HOST_PREEMPT"
    for text in ("Linux 6.8.0 #1 SMP PREEMPT_DYNAMIC\nTraceback (most recent call last):\nNotImplementedError: aten::_grouped_mm",
                 "triton.compiler.errors.CompilationError: at 12:4", "ImportError: undefined symbol: _ZN3c10"):
        cls, line = B.classify(text, 32768, 16384)
        assert cls == "R" and line == text.splitlines()[-1], (text, cls, line)
    assert B.classify("", 32768, 16384) == ("R", "(no error text captured)")


def test_classify_reads_only_the_terminal_error_so_an_incidental_mention_is_r():
    """The review's three cases: an earlier mention of NVML, pynvml or ECC never makes a later real failure host-attributable."""
    sys.path.insert(0, str(D.REPO / "bench" / "hp3"))
    import hp3_build as B
    fail = "RuntimeError: Triton compilation failed: ptxas error"
    for pre in ("UserWarning: Can't initialize NVML", '  File "x.py", line 3, in <module>\n    import pynvml',
                "gpu: name=RTX 4000 Ada ECC errors=0 Xid=none-seen"):
        assert B.classify(pre + "\nTraceback (most recent call last):\n" + fail, 32768, 16384) == ("R", fail), pre
    oom = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
    assert B.classify(oom + "\nRuntimeError: unsupported op", 12288, 16384) == ("R", "RuntimeError: unsupported op")
    assert B.classify("noise\n" + oom, 12288, 16384) == ("HOST_OOM_UNDER_FLOOR", oom)
    assert B.classify("x", 32768, 16384, reason="the instance was preempted by the provider")[0] == "HOST_PREEMPT"
    assert B.classify("kernel: Linux 6.8 #1 SMP PREEMPT_DYNAMIC\nValueError: bad shape", 32768, 16384)[0] == "R"


def test_a_shakedown_arm_that_does_not_run_signals_r_unless_its_class_is_a_host_class(tmp_path):
    for cls, want in (("R | NotImplementedError: aten::_grouped_mm", "R -> the (R) fallback signal | NotImplementedError: aten::_grouped_mm"),
                      ("HOST_DRIVER | NVRM: Xid 79", "host-attributable (HOST_DRIVER): the spare | NVRM: Xid 79")):
        w = tmp_path / cls.split()[0]
        w.mkdir()
        arms, summary = drive(w, FAMILY_FAM="qwen3hp3shake", HP3_SHAKE="1", FAIL_FAM="qwen3hp3shakeL2", FAIL_STATUS="error", STUB_CLASS=cls)
        assert f"HP3 ARM_FAILED e4b/hp3cal_e4b_k00_s3407 error: {want}" in summary, summary
        assert not [a for a in arms if not a[2].startswith("hp3cal")]


def test_the_read_box_reports_a_calibration_failure_without_any_fallback_signal(tmp_path):
    _, summary = drive(tmp_path, FAMILY_FAM="qwen3hp3", FAIL_FAM="qwen3hp3", FAIL_STATUS="error")
    assert "HP3 CALIB_ARM_FAILED e4b error" in summary and "fallback" not in summary and "ARM_FAILED e4b/" not in summary


def test_the_read_writes_its_seed_plan_before_seed_3411_and_hands_it_to_the_reducer(tmp_path):
    """Amendment 1 part 7: after seed 3410, the plan (timing only) is written before any later seed's arm, and the verdict reads it."""
    arms, summary = drive(tmp_path, FAMILY_FAM="qwen3hp3")
    plan = json.loads((tmp_path / "hp3" / "seed_plan.json").read_text())
    assert plan["n"] == 5 and plan["basis"] == "timing" and plan["seeds"] == [3410, 3411, 3412, 3413, 3414] and "HP3 SEED_PLAN n=5" in summary
    first_3411 = tmp_path / "qwen3hp3_e4b_hp3_k00_s3411.json"
    assert (tmp_path / "hp3" / "seed_plan.json").stat().st_mtime_ns <= first_3411.stat().st_mtime_ns
    ver = next(ln for ln in (tmp_path / "py.log").read_text().splitlines() if "hp3_reduce.py verdict" in ln)
    assert f"--seed-plan {tmp_path}/hp3/seed_plan.json" in ver
    ev = next(ln for ln in (tmp_path / "py.log").read_text().splitlines() if "hp1_eval.py" in ln and "eval_P" in ln)
    assert f"--seed-plan {tmp_path}/hp3/seed_plan.json" in ev                     # the evaluator hashes the plan it is handed
    assert "written_at_utc" in plan and "written_at" not in {k for k in plan if k != "written_at_utc"}


def test_a_short_deadline_reads_four_seeds_and_a_shorter_one_stops_after_seed_3410(tmp_path):
    (tmp_path / "four").mkdir()
    arms, summary = drive(tmp_path / "four", FAMILY_FAM="qwen3hp3", STUB_LEFT="6400")     # need(5) 6300 > 5500 >= need(4) 5220
    conf = [a[2] for a in arms if a[0] == "qwen3hp3" and not a[2].startswith("hp3cal")]
    assert conf == [f"hp3_{r}_s{s}" for s in (3410, 3411, 3412, 3413) for r in ("k00", "unsloth", "axolotl")] and "HP3 SEED_PLAN n=4" in summary
    (tmp_path / "none").mkdir()
    arms, summary = drive(tmp_path / "none", FAMILY_FAM="qwen3hp3", STUB_LEFT="3000")
    conf = [a[2] for a in arms if a[0] == "qwen3hp3" and not a[2].startswith("hp3cal")]
    assert conf == ["hp3_k00_s3410", "hp3_unsloth_s3410", "hp3_axolotl_s3410"] and "HP3 HOST_LIMITED" in summary
    assert "eval_P" not in (tmp_path / "none" / "py.log").read_text()


def test_the_shakedown_writes_no_seed_plan(tmp_path):
    _, summary = drive(tmp_path, FAMILY_FAM="qwen3hp3shake", HP3_SHAKE="1")
    assert not (tmp_path / "hp3" / "seed_plan.json").exists() and "SEED_PLAN" not in summary
    assert "--seed-plan" not in (tmp_path / "py.log").read_text()                  # the shakedown never hands a plan to the evaluator


def _membox(tmp_path, cls, mib, fams="qwen3hp3shake"):
    """Run tc1_run.sh's part-8 block alone, with a fake nvidia-smi reporting `mib`; returns (rc, summary)."""
    import subprocess
    src = (D.REPO / "bench" / "tc1" / "tc1_run.sh").read_text()
    block = src[src.index("# HP3 Amendment 1 part 8 begin"):src.index("# HP3 Amendment 1 part 8 end")]
    (tmp_path / "bin").mkdir(exist_ok=True)
    (tmp_path / "bin" / "nvidia-smi").write_text(f"#!/bin/sh\necho {mib}\n")
    (tmp_path / "bin" / "nvidia-smi").chmod(0o755)
    (tmp_path / "t.sh").write_text("say(){ :; }\nfinish(){ exit $1; }\n: > summary.txt\n" + block + "\nexit 0\n")
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "FAMILIES": fams, "GPU_CLASS": cls}
    p = subprocess.run(["bash", "t.sh"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    return p.returncode, (tmp_path / "summary.txt").read_text()


def test_the_box_refuses_a_card_whose_memory_is_outside_its_class_band(tmp_path):
    """Amendment 1 part 8: TC1_GPU_CLASS from the pin covers the name; memory.total must fall in the class band too (rc 18, the
    launcher's other host floor; a host refusal before any staging, so the spare)."""
    for cls, mib in (("A4000", 16376), ("L4", 23034), ("4090", 24564), ("L40S", 46068), ("5090", 32607)):
        assert _membox(tmp_path, cls, mib)[0] == 0, cls
    rc, summary = _membox(tmp_path, "4090", 49140)                             # a modded 48 GB 4090
    assert rc == 18 and "gpu_mem=49140MiB class=4090" in summary
    assert _membox(tmp_path, "A4000", 8192)[0] == 18 and _membox(tmp_path, "3090", 24576)[0] == 18   # no band registered: refuse
    assert _membox(tmp_path, "4090", 49140, fams="qwen3")[0] == 0              # other tokens are untouched


def _seed_plan(tmp_path, wall, nf, left):
    """hp3_seed_plan sourced alone from tc1_run.sh with left() stubbed: (rc, stdout, plan)."""
    import subprocess
    src = (D.REPO / "bench" / "tc1" / "tc1_run.sh").read_text()
    fn = src[src.index("hp3_seed_plan(){"):src.index("  echo $n; }") + len("  echo $n; }")]
    (tmp_path / "hp3").mkdir(exist_ok=True)
    (tmp_path / "t.sh").write_text(f"W={tmp_path}; PY_E4B=python3\nleft(){{ echo {left}; }}\n" + fn + f"\nhp3_seed_plan {wall} {nf}\n")
    p = subprocess.run(["bash", "t.sh"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    f = tmp_path / "hp3" / "seed_plan.json"
    return p.returncode, p.stdout.strip(), (json.loads(f.read_text()) if f.exists() else None)


def test_the_seed_plan_counts_seed_3410s_wall_time():
    """The evaluator's required test (2026-10-23 20:57Z): at W = 1200 s, NF = 3 the needs are 11100 / 8820 / 6540, so left 9800 reads
    4, 9000 reads 3 and 7000 reads 0. Without the W term 9000 would read 5, so this kills that mutant (stub seeds take ~0 s)."""
    import tempfile
    for left, want in ((20000, 5), (9800, 4), (9000, 3), (7000, 0)):
        with tempfile.TemporaryDirectory() as d:
            rc, out, plan = _seed_plan(Path(d), 1200, 3, left)
            assert rc == 0 and out == str(want), (left, out)
            assert plan["need_s"] == {"5": 11100, "4": 8820, "3": 6540} and plan["seed1_wall_s"] == 1200 and plan["n"] == want
            assert plan["seeds"] == [3410, 3411, 3412, 3413, 3414][:want] and "written_at_utc" in plan


def test_an_unwritable_seed_plan_is_a_harness_fault_never_host_limited(tmp_path):
    """The maintainer 20:58Z: a plan-write failure prints SEED_PLAN_UNWRITABLE (a NOT_READ naming the harness), never HOST_LIMITED."""
    arms, summary = drive(tmp_path, FAMILY_FAM="qwen3hp3", STUB_PLAN_FAIL="1")
    assert "HP3 SEED_PLAN_UNWRITABLE" in summary and "HOST_LIMITED" not in summary
    conf = [a[2] for a in arms if a[0] == "qwen3hp3" and not a[2].startswith("hp3cal")]
    assert conf == ["hp3_k00_s3410", "hp3_unsloth_s3410", "hp3_axolotl_s3410"] and "eval_P" not in (tmp_path / "py.log").read_text()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "hp3").mkdir()
        (Path(d) / "hp3").chmod(0o500)                                          # the plan file cannot be created
        rc, out, _ = _seed_plan(Path(d), 600, 3, 100000)
        (Path(d) / "hp3").chmod(0o700)
        assert rc == 1 and out == "UNWRITABLE"




def test_the_shakedown_evaluator_skips_the_registered_trainable_count_and_a_reading_id_never_does(tmp_path):
    """hp3-shake-1 (2026-10-23): the 2-layer cut trains 26,771,456 parameters, not the registered 642,514,944, so the evaluator VOIDed
    every calibration arm. --expect-trainable 0 turns the check OFF, so it is reachable only from a registered shakedown id: the
    shakedown passes it to both evaluator calls; a READING id (hp3-5090-1) never carries it, so the registered count applies."""
    (tmp_path / "s").mkdir()
    drive(tmp_path / "s", FAMILY_FAM="qwen3hp3shake", HP3_SHAKE="1", TC1_RUN_ID="hp3-shake-2")
    evs = [ln for ln in (tmp_path / "s" / "py.log").read_text().splitlines() if "hp1_eval.py" in ln]
    assert len(evs) == 2 and all("--expect-trainable 0" in ln for ln in evs), evs
    for rid in ("hp3-5090-1", "hp3-5090-2"):
        (tmp_path / rid).mkdir()
        drive(tmp_path / rid, FAMILY_FAM="qwen3hp3", TC1_RUN_ID=rid)
        evs = [ln for ln in (tmp_path / rid / "py.log").read_text().splitlines() if "hp1_eval.py" in ln]
        assert len(evs) == 2 and not any("--expect-trainable" in ln for ln in evs), evs     # the registered count, by default


def test_the_shakedown_token_under_a_reading_id_refuses_before_any_arm(tmp_path):
    arms, summary = drive(tmp_path, FAMILY_FAM="qwen3hp3shake", HP3_SHAKE="1", TC1_RUN_ID="hp3-5090-1")
    assert "HP3 REFUSED: the shakedown token under run id 'hp3-5090-1'" in summary
    log = tmp_path / "py.log"                                                  # it refuses before any interpreter call at all
    assert not arms and (not log.exists() or "hp1_eval.py" not in log.read_text())


def _cal_argv(tmp_path, fam, shake, rid, mode="p"):
    """The calibration evaluator's argv exactly as tc1_hp3_family builds it (the driver records every interpreter call)."""
    import re
    import shlex
    drive(tmp_path, FAMILY_FAM=fam, HP3_SHAKE=shake, TC1_RUN_ID=rid, HP3_MODE=mode)
    line = next(ln for ln in (tmp_path / "py.log").read_text().splitlines() if "hp1_eval.py" in ln and "eval_cal" in ln)
    argv = shlex.split(re.sub(r"^PY(\[[^]]*\])?: ", "", line))                 # PY: or PY[<interpreter>]: (the (R) t28 venv)
    return argv[next(i for i, x in enumerate(argv) if x.endswith("hp1_eval.py")) + 1:]


_REAL = {}


def _run_hp1_eval(tmp_path, argv, monkeypatch, trainable):
    """hp1_eval.main() on the runner's own argv (its real parse, tokens and evaluate()); only the model and the receipts are fixtures:
    each receipt is a cut's arm with `trainable` parameters, and each base a store whose digest matches its training run."""
    import importlib
    import test_hp1_eval as TE
    hp1_eval = importlib.import_module("hp1_eval")
    tok = tmp_path / "tokens.json"
    tok.write_text(json.dumps({"sha256": TE.TOK, "eval": [[1, 2]] * 300}))
    out = tmp_path / "eval_cal.json"
    a = list(argv)
    a[a.index("--tokens") + 1], a[a.index("--out") + 1] = str(tok), str(out)
    recs = a[a.index("--receipts") + 1:a.index("--bases")]
    bases = a[a.index("--bases") + 1:a.index("--tokens")]
    for r in recs:
        Path(r).parent.mkdir(parents=True, exist_ok=True)
        Path(r).write_text("{}")
    fw_of = lambda p: next(f for f in ("e4b", "unsloth", "axolotl") if f"_{f}_" in Path(p).name)                     # noqa: E731
    monkeypatch.setattr(hp1_eval, "load_tc1_arm", lambda: type("T", (), {})())
    monkeypatch.setattr(hp1_eval, "receipt_arm", lambda p: TE.arm(Path(p).stem.split("_", 2)[2], framework=fw_of(p), trainable_params=trainable))
    h = TE.Harness(stores={b: TE.Store() for b in bases})
    real = _REAL.setdefault("evaluate", hp1_eval.evaluate)                     # the true evaluate(), cached before any patch

    def evaluate(arms, base_dirs, rows, open_base, make_scorer, load_adapter, file_sha, tokens_sha, expect, write=None, release=None, **kw):
        return real(arms, base_dirs, rows, h.open_base, h.make_scorer, h.load_adapter, lambda p: "sha-" + p.split(".")[0], tokens_sha,
                    expect, write, None, **kw)
    monkeypatch.setattr(hp1_eval, "evaluate", evaluate)
    monkeypatch.setattr(sys, "argv", ["hp1_eval.py"] + a)
    hp1_eval.main()
    return json.loads(out.read_text())


def test_the_shakedowns_evaluator_call_runs_on_a_cut_with_the_runners_real_arguments(tmp_path, monkeypatch):
    """The maintainer (2026-10-24): a $0 CPU run of the shakedown's EVALUATOR path, with the arguments the runner builds, against a cut.
    On 6602e763's runner (no --expect-trainable) every cut arm is VOID as "trainable 26771456 != the registered 642514944"; on the fix
    no arm is VOID on the trainable count. The read's argv, on the same fixture, still applies the registered count."""
    sys.path.insert(0, str(D.REPO / "bench" / "hp1"))
    sys.path.insert(0, str(D.REPO / "tests"))
    (tmp_path / "s").mkdir()
    ev = _run_hp1_eval(tmp_path / "s", _cal_argv(tmp_path / "s", "qwen3hp3shake", "1", "hp3-shake-2"), monkeypatch, trainable=26771456)
    bad = [x for x in ev["arms"] if "trainable" in str(x.get("reason") or "")]
    assert len(ev["arms"]) == 3 and not bad, bad
    (tmp_path / "r").mkdir()
    ev = _run_hp1_eval(tmp_path / "r", _cal_argv(tmp_path / "r", "qwen3hp3", "0", "hp3-5090-1"), monkeypatch, trainable=26771456)
    assert ev["expect_trainable"] == 642514944
    assert all("trainable 26771456 != the registered 642514944" in str(x.get("reason")) for x in ev["arms"])


def test_the_r_fallback_runs_e4b_and_unsloth_on_the_t28_build_with_the_r_reducer(tmp_path):
    """HP3 Amendment 2 (R): e4b in venv-unsloth-t28 (t28u) and Unsloth 2026.9.14 (t28), no axolotl arm, the evaluator in that venv,
    and the reducer in its --fallback-r mode, for calibration and five confirmation seeds."""
    arms, summary = drive(tmp_path, FAMILY_FAM="qwen3hp3r", HP3_MODE="r", TC1_RUN_ID="hp3r-5090-1")
    t = [a[2] for a in arms if a[0] == "qwen3hp3r"]
    assert t[:2] == ["hp3cal_e4b_k00_s3407", "hp3cal_unsloth_s3407"] and not [x for x in t if "axolotl" in x]
    assert t[2:] == [f"hp3_{r}_s{s}" for s in (3410, 3411, 3412, 3413, 3414) for r in ("k00", "unsloth")]
    ev = (tmp_path / "e4bvenv.log").read_text().splitlines()
    assert all(ln.endswith("E4B_VENV=t28u") for ln in ev if " e4b " in ln), ev
    uv = (tmp_path / "venv.log").read_text().splitlines()
    assert all(ln.endswith("UNS_VENV=t28") for ln in uv if " unsloth " in ln), uv
    assert all(ln.endswith("HP3_VENV=") for ln in (tmp_path / "h3venv.log").read_text().splitlines())
    py = (tmp_path / "py.log").read_text().splitlines()
    assert [ln for ln in py if "hp1_eval.py" in ln] and all(ln.startswith("PY[fakepy-t28]") for ln in py if "hp1_eval.py" in ln)
    red = [ln for ln in py if "hp3_reduce.py" in ln]
    assert len(red) == 2 and all("--fallback-r" in ln for ln in red) and all(ln.startswith("PY[fakepy-t28]") for ln in red)


def test_the_r_fallback_refuses_without_its_build(tmp_path):
    arms, summary = drive(tmp_path, FAMILY_FAM="qwen3hp3r", HP3_MODE="r", STUB_H3ROK="0", STUB_H3RREASON="venv-unsloth-t28 did not install/import")
    assert arms == [] and "HP3 BUILD_FAILED: venv-unsloth-t28 did not install/import" in summary and "fallback signal" not in summary


def test_the_r_shakedown_runs_the_cut_end_to_end_on_the_t28_build(tmp_path):
    """The (R) shakedown (the maintainer, 2026-10-24 04:36Z): the 2-layer cut, e4b (t28u) and Unsloth (t28) calibration then one
    seed each, no axolotl; both evaluator calls skip the trainable count; both reducer calls in --fallback-r mode; SHAKEDOWN_OK."""
    arms, summary = drive(tmp_path, FAMILY_FAM="qwen3hp3rshake", HP3_MODE="r", HP3_SHAKE="1")
    t = [a[2] for a in arms if a[0] == "qwen3hp3rshakeL2"]
    assert t == ["hp3cal_e4b_k00_s3407", "hp3cal_unsloth_s3407", "hp3_k00_s3410", "hp3_unsloth_s3410"]
    assert "model-00001-of-00016.safetensors model-00016-of-00016.safetensors" in (tmp_path / "fetch_allow.log").read_text()
    assert all(ln.endswith("E4B_VENV=t28u") for ln in (tmp_path / "e4bvenv.log").read_text().splitlines() if " e4b " in ln)
    py = (tmp_path / "py.log").read_text().splitlines()
    evs = [ln for ln in py if "hp1_eval.py" in ln]
    assert len(evs) == 2 and all("--expect-trainable 0" in ln and ln.startswith("PY[fakepy-t28]") for ln in evs), evs
    red = [ln for ln in py if "hp3_reduce.py" in ln]
    assert len(red) == 2 and all("--fallback-r" in ln for ln in red)
    assert "HP3 SHAKEDOWN_OK" in summary and "seed_plan" not in "\n".join(py)


def test_an_r_shakedown_failure_is_a_kit_fault_never_a_fallback_signal(tmp_path):
    for k, kw, want in (("b", dict(STUB_H3ROK="0", STUB_H3RREASON="t28u did not import"), "HP3 BUILD_FAILED: t28u did not import -> an (R) kit fault"),
                        ("c", dict(STUB_H3G1="close"), "HP3 G1_CLOSED: bases differ: outside this registration -> an (R) kit fault"),
                        ("a", dict(FAIL_FAM="qwen3hp3rshakeL2", FAIL_STATUS="error"), "HP3 ARM_FAILED e4b/hp3cal_e4b_k00_s3407 error: R -> an (R) kit fault: fix it, then the spare")):
        (tmp_path / k).mkdir()
        _, summary = drive(tmp_path / k, FAMILY_FAM="qwen3hp3rshake", HP3_MODE="r", HP3_SHAKE="1", **kw)
        assert want in summary and "the (R) fallback signal" not in summary, (k, summary)


def test_each_modes_shakedown_refuses_the_other_modes_ids(tmp_path):
    for k, fam, mode, rid, ids in (("p", "qwen3hp3shake", "p", "hp3r-shake-1", "hp3-shake-1/hp3-shake-2"),
                                   ("r", "qwen3hp3rshake", "r", "hp3-shake-2", "hp3r-shake-1/hp3r-shake-2"),
                                   ("x", "qwen3hp3rshake", "r", "hp3r-5090-1", "hp3r-shake-1/hp3r-shake-2")):
        (tmp_path / k).mkdir()
        arms, summary = drive(tmp_path / k, FAMILY_FAM=fam, HP3_MODE=mode, HP3_SHAKE="1", TC1_RUN_ID=rid)
        assert arms == [] and f"HP3 REFUSED: the shakedown token under run id '{rid}', not a registered shakedown id ({ids})" in summary
    for rid in ("hp3r-shake-1", "hp3r-shake-2"):                          # the read token never carries the check-off flag
        (tmp_path / rid).mkdir()
        drive(tmp_path / rid, FAMILY_FAM="qwen3hp3r", HP3_MODE="r", TC1_RUN_ID=rid)
        assert "--expect-trainable" not in (tmp_path / rid / "py.log").read_text()


def test_the_r_shakedowns_evaluator_call_runs_on_a_cut_with_the_runners_real_arguments(tmp_path, monkeypatch):
    sys.path.insert(0, str(D.REPO / "bench" / "hp1"))
    sys.path.insert(0, str(D.REPO / "tests"))
    ev = _run_hp1_eval(tmp_path, _cal_argv(tmp_path, "qwen3hp3rshake", "1", "hp3r-shake-1", mode="r"), monkeypatch, trainable=26771456)
    assert len(ev["arms"]) == 2 and not [x for x in ev["arms"] if "trainable" in str(x.get("reason") or "")], ev["arms"]


def test_the_r_build_record_compares_against_torch_280_cu128_with_bnb_unpinned():
    sys.path.insert(0, str(D.REPO / "bench" / "hp3"))
    import hp3_build as B
    p = {"torch": "2.8.0+cu128", "cuda": "12.8", "bitsandbytes": "0.49.1", "bnb_lib_sha256": "b" * 64, "triton": "3.4.0"}
    probes = [dict(p, framework=f) for f in ("e4b", "unsloth")]
    assert B.compare(probes, "2.8.0", "any", "cu128")["verdict"] == "MATCHED"
    rec = B.compare(probes, "2.8.0", "any", "cu130")
    assert rec["verdict"] == "MISMATCH" and sum("is not a cu130 build" in x for x in rec["problems"]) == 2
    assert B.compare([dict(p, framework="e4b"), dict(p, framework="unsloth", bnb_lib_sha256="c" * 64)], "2.8.0", "any", "cu128")["verdict"] == "MISMATCH"


def test_the_part_8_memory_band_covers_the_r_tokens(tmp_path):
    assert _membox(tmp_path, "5090", 32607, fams="qwen3hp3r")[0] == 0
    assert _membox(tmp_path, "5090", 49140, fams="qwen3hp3r")[0] == 18
    assert _membox(tmp_path, "L4", 23034, fams="qwen3hp3rshake")[0] == 0
    assert _membox(tmp_path, "4090", 49140, fams="qwen3hp3rshake")[0] == 18


def test_the_r_shakedown_token_is_wired_like_the_r_token():
    """Every gate that names qwen3hp3r names qwen3hp3rshake too (no axolotl, no cu130 Unsloth, t28u, the build record, staging)."""
    run = (D.REPO / "bench" / "tc1" / "tc1_run.sh").read_text()
    for ln in [ln for ln in run.splitlines() if '" qwen3hp3r "' in ln]:
        assert '" qwen3hp3rshake "' in ln, ln
    assert "qwen3hp3rshake) tc1_hp3_family qwen3hp3rshake Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 1800 1800 1800 1 r;;" in run
    drv = (D.REPO / "bench" / "tc1" / "tc1_drive.sh").read_text()
    assert '*" qwen3hp3rshake "*) STAGE="$STAGE' in drv
