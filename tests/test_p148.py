# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P148 (bench/p148/PREREG-p148.md; e4b#1313), the dense bf16 decode GEMVs microbench: the staged pins (P136's
premise tests at P136's bytes), the runner's order (the kernels' own suite runs compiled after the premise and before the
bench), the self-tests run as the box runs them (one directory), the mutants the reducer's self-test must kill, and the
driver's run ids."""
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p148"
SOURCES = {"p148_run.sh": LANE, "p148_bench.py": LANE, "p148_kernels.py": LANE, "p148_reduce.py": LANE,
           "test_p148_kernels_interp.py": REPO / "tests", "test_decode_graph_buckets.py": REPO / "tests",
           "test_kv_step_select.py": REPO / "tests", "test_fused_glue_decode_graphs_gpu.py": B / "p136" / "premise",
           "test_gemv_bw_served_gpu.py": REPO / "tests"}
PREMISE = {"test_decode_graph_buckets.py", "test_kv_step_select.py", "test_fused_glue_decode_graphs_gpu.py",
           "test_gemv_bw_served_gpu.py"}


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _run(script, *args, cwd=None, env=None):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=180, cwd=cwd,
                          env=env)


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p148.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_the_premise_tests_are_p136s():
    mine, p136 = _pins(LANE / "staged-p148.sha256"), _pins(B / "p136" / "staged-p136.sha256")
    assert PREMISE <= set(p136) and all(mine[n] == p136[n] for n in PREMISE)


def test_the_runner_runs_the_kernels_suite_compiled_after_the_premise_and_reads_only_a_rule_verdict():
    run = (LANE / "p148_run.sh").read_text(encoding="utf-8")
    assert "GNF4_SHA=38b308887688ea5bda7dd449bd3cc3dc1426b995" in run and 'md.version("grouped-nf4-gemm") == "0.47.1"' in run
    assert "p148-5090-1|p148-5090-2|p148-5090-3) FAMILY=qwen3-dense-gemv;" in run
    order = ["grep -qE '^19 passed", "TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_p148_kernels_interp.py",
             "grep -qE '^61 passed", "python $W/p148_bench.py --out $W/p148.json", "python $W/p148_reduce.py --bench $W/p148.json"]
    at = [run.index(x) for x in order]
    assert at == sorted(at), dict(zip(order, at))
    assert 'case "$V" in D1_*+D2_*) ;;' in run, "NOISY and VOID are not readings"
    for gone in ("snapshot_download", "k8_bake.py", "n_split", "Fp8PagedKV", "p143_bench"):
        assert gone not in run, gone


def test_the_kernels_suite_counts_61_and_sets_the_interpreter_only_by_default():
    src = (REPO / "tests" / "test_p148_kernels_interp.py").read_text(encoding="utf-8")
    assert 'os.environ.setdefault("TRITON_INTERPRET", "1")' in src           # the runner sets 0 explicitly
    import ast
    tree = ast.parse(src)
    n = 0
    for f in (x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name.startswith("test_")):
        par = [d for d in f.decorator_list if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "parametrize"]
        if not par:
            n += 1
            continue
        arg = par[0].args[1]
        n += {"K.CONFIGS_BF16": 36, "K.CONFIGS_I8": 18}.get(ast.unparse(arg)) or len(arg.elts)
    assert n == 61


@pytest.fixture(scope="module")
def box_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("p148_box")
    for name, src in SOURCES.items():
        shutil.copyfile(src / name, d / name)
    return d


@pytest.mark.parametrize("script", ["p148_bench.py", "p148_reduce.py"])
def test_the_self_tests_pass(box_dir, script):
    if script == "p148_bench.py":
        pytest.importorskip("triton")
        pytest.importorskip("torch")
    out = _run(box_dir / script, "--self-test", cwd=box_dir)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


MUTANTS = [
    # the bar at 1 % of P
    ("BAR_SHARE = 0.02", "BAR_SHARE = 0.01"),
    # a shape's saving from the LARGER pair's projection
    ('        proj = min(ANCHOR_MS[s] * (1 - r["B1"] / r["A1"]), ANCHOR_MS[s] * (1 - r["B2"] / r["A2"]))',
     '        proj = max(ANCHOR_MS[s] * (1 - r["B1"] / r["A1"]), ANCHOR_MS[s] * (1 - r["B2"] / r["A2"]))'),
    # the NOISY band widened
    ("NOISY_BAND = (0.985, 1.015)", "NOISY_BAND = (0.98, 1.02)"),
    # the transfer anchor never checked
    ("            if not ANCHOR[0] <= r[t] / ANCHOR_MS[s] <= ANCHOR[1]:", "            if False:"),
    # the contract never binds
    ('        passing = {c for c, e in con[s]["configs"].items() if e <= CONTRACT_RATIO * con[s]["cublas"]}',
     '        passing = {c for c, e in con[s]["configs"].items() if e <= 10 * con[s]["cublas"]}'),
    # D1 from its better shape instead of the sum of both
    ('        d1 = sum(shapes[s]["projected_save_ms"] for s in D1_SHAPES)',
     '        d1 = max(shapes[s]["projected_save_ms"] for s in D1_SHAPES)'),
    # the bars on the raw harness ms instead of the projected in-situ saving (the maintainer's ruling on #1796)
    ('        d1 = sum(shapes[s]["projected_save_ms"] for s in D1_SHAPES)',
     '        d1 = sum(shapes[s]["harness_save_ms"] for s in D1_SHAPES)'),
    ('        d2 = shapes["head"]["projected_save_ms"]', '        d2 = shapes["head"]["harness_save_ms"]'),
    # the choice never checked against the selection
    ("        elif cho.get(s) != min(sel[s], key=sel[s].get):", "        elif False:"),
    # the harness's cuBLAS kernel never held to P142's in-situ grid (the review of #1796)
    ("        if GRID_P142[s] not in grids:", "        if False:"),
    # the bar read on a rounded saving (A3's final-pass lesson)
    ('"verdict": "GO" if d1 >= BAR else "PARK"', '"verdict": "GO" if round(d1, 6) >= BAR else "PARK"'),
]


@pytest.mark.parametrize("old,new", MUTANTS)
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p148_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p148_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p148_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P148_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p148_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p148-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p148" in out.stdout and "bash p148_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p148_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p143a3-5090-1"), timeout=120)
    assert out.returncode == 78 and "is not p148-5090-N" in out.stdout + out.stderr
