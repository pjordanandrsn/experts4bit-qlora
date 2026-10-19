# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P149 (bench/p149/PREREG-p149.md), box K: the staged pins, the runner's order (the power floor before any install;
the kernels' own suite compiled before the bench), the self-tests run as the box runs them (one directory), the mutants the
reducer's self-test must kill, the interpreter suite in its own process, and the driver's run ids."""
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
LANE = B / "p149"
SOURCES = {"p149k_run.sh": LANE, "p149k_bench.py": LANE, "p149_kernels.py": LANE, "p149k_reduce.py": LANE,
           "test_p149_kernels_interp.py": REPO / "tests", "i8_fakequant.py": B / "i8dense"}


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
    pins = _pins(LANE / "staged-p149k.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_the_runner_refuses_a_capped_board_before_any_install_and_runs_the_suite_compiled_before_the_bench():
    run = (LANE / "p149k_run.sh").read_text(encoding="utf-8")
    assert "GNF4_SHA=38b308887688ea5bda7dd449bd3cc3dc1426b995" in run and 'md.version("grouped-nf4-gemm") == "0.47.1"' in run
    assert 'assert triton.__version__ == "3.4.0"' in run and "TORCH_FULL\" = 2.8.0+cu128" in run
    assert "p149k-5090-1|p149k-5090-2|p149k-5090-3) FAMILY=mixtral-attn-int8;" in run
    assert "MIN_POWER_W=575" in run
    order = ["PL=$(nvidia-smi --query-gpu=power.limit", 'finish 15; }', '[ "$PL" -ge "$MIN_POWER_W" ] ||', "finish 17; }",
             "pipx logs/pip_e4b.log", "TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_p149_kernels_interp.py",
             "grep -qE '^135 passed", "python $W/p149k_bench.py --out $W/p149k.json", "python $W/p149k_reduce.py --bench $W/p149k.json"]
    at = []
    for x in order:
        assert x in run, x
        at.append(run.index(x, at[-1] if at else 0))
    assert at == sorted(at), dict(zip(order, at))
    assert 'case "$V" in K_GO|KERNEL_NO_GAIN) ;;' in run, "NOISY and VOID are not readings"
    for gone in ("test_decode_graph_buckets.py", "premise.log", "resolve_fusion_modes", "p148_bench", "snapshot_download"):
        assert gone not in run, gone


def test_the_kernels_suite_counts_135_and_sets_the_interpreter_only_by_default():
    src = (REPO / "tests" / "test_p149_kernels_interp.py").read_text(encoding="utf-8")
    assert 'os.environ.setdefault("TRITON_INTERPRET", "1")' in src           # the runner sets 0 explicitly
    assert 'type(triton.language.zeros).__name__ != "InterpretedFunction"' in src   # skips after a compiled import
    import ast
    tree = ast.parse(src)
    n = 0
    for f in (x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name.startswith("test_")):
        par = [d for d in f.decorator_list if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "parametrize"]
        if not par:
            n += 1
            continue
        arg = par[0].args[1]
        n += {"K.CONFIGS_GEMV": 54, "K.CONFIGS_SMALL": 72}.get(ast.unparse(arg)) or len(arg.elts)
    assert n == 135


def test_the_kernels_suite_passes_interpreted_in_its_own_process():
    """CI's full suite imports triton compiled before the kernels' suite can set TRITON_INTERPRET, so the file skips itself
    in-process; here it runs in a fresh process, as the boxes run it (compiled there)."""
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                          str(REPO / "tests" / "test_p149_kernels_interp.py")], capture_output=True, text=True,
                         cwd=REPO, env={**os.environ, "TRITON_INTERPRET": "1"}, timeout=1200)
    last = (out.stdout.strip().splitlines() or [""])[-1]
    ok = out.returncode == 0 and re.match(r"135 passed(, \d+ warnings?)? in ", last)
    assert ok, out.stdout[-3000:] + out.stderr[-2000:]


@pytest.fixture(scope="module")
def box_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("p149k_box")
    for name, src in SOURCES.items():
        shutil.copyfile(src / name, d / name)
    return d


@pytest.mark.parametrize("script", ["p149k_bench.py", "p149k_reduce.py"])
def test_the_self_tests_pass(box_dir, script):
    if script == "p149k_bench.py":
        pytest.importorskip("triton")
        pytest.importorskip("torch")
    out = _run(box_dir / script, "--self-test", cwd=box_dir)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


MUTANTS = [
    # the bar at 1 % of P
    ("BAR_SHARE = 0.02", "BAR_SHARE = 0.01"),
    # a shape's saving from the LARGER pair's projection
    ('                row.update(projected_save_ms=min(ANCHOR_MS[s] * (1 - r["B1"] / r["A1"]), ANCHOR_MS[s] * (1 - r["B2"] / r["A2"])),',
     '                row.update(projected_save_ms=max(ANCHOR_MS[s] * (1 - r["B1"] / r["A1"]), ANCHOR_MS[s] * (1 - r["B2"] / r["A2"])),'),
    # the NOISY band widened
    ("NOISY_BAND = (0.985, 1.015)", "NOISY_BAND = (0.98, 1.02)"),
    # NOISY read on M == 16 too (it is reported, never ruled)
    ('            if m == "1":\n                noisy +=', '            if True:\n                noisy +='),
    # the transfer anchor never checked
    ("                    if not ANCHOR[0] <= r[t] / ANCHOR_MS[s] <= ANCHOR[1]:", "                    if False:"),
    # the contract never binds
    ('    return {c for c, e in con["configs"].items() if e <= CONTRACT_RATIO * con["cublas"]}',
     '    return {c for c, e in con["configs"].items() if e <= 10 * con["cublas"]}'),
    # a contract failure at M == 16 ignored
    ('                fails.append(f"{s} M={m}: no config passes the contract', '                pass  # fails.append(f"{s} M={m}: no config passes the contract'),
    # W1 from the better shape instead of the sum of both
    ('    w1 = sum(shapes[s]["1"]["projected_save_ms"] for s in SHAPES if "1" in shapes[s])',
     '    w1 = max(shapes[s]["1"]["projected_save_ms"] for s in SHAPES if "1" in shapes[s])'),
    # the bar on the raw harness ms instead of the projected in-situ saving
    ('    w1 = sum(shapes[s]["1"]["projected_save_ms"] for s in SHAPES if "1" in shapes[s])',
     '    w1 = sum(shapes[s]["1"]["harness_save_ms"] for s in SHAPES if "1" in shapes[s])'),
    # the choice never checked against the selection
    ("            elif (cho.get(s) or {}).get(m) != min(sm, key=sm.get):", "            elif False:"),
    # the harness's cuBLAS kernel never held to P145's in-situ grid
    ("        if GRID_P145[s] not in grids:", "        if False:"),
    # the bar read on a rounded saving
    ("    if w1 >= BAR:", "    if round(w1, 6) >= BAR:"),
]


@pytest.mark.parametrize("old,new", MUTANTS)
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p149k_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p149k_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p149k_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P149_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p149k_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p149k-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p149k" in out.stdout and "bash p149k_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p149k_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p149-5090-1"), timeout=120)
    assert out.returncode == 78 and "is not p149k-5090-N" in out.stdout + out.stderr
