# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P143 step 1 (bench/p143/PREREG-p143.md; e4b#1313): the staged pins and P136's premise tests byte-identical, the
runner's grid and steps, the bench reproducing the served call's arguments, the self-tests, the mutants the reducer's
self-test must kill, and the driver's run ids."""
import ast
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p143"
SOURCES = {"p143_run.sh": LANE, "p143_bench.py": LANE, "p143_reduce.py": LANE, "test_decode_graph_buckets.py": REPO / "tests",
           "test_kv_step_select.py": REPO / "tests", "test_fused_glue_decode_graphs_gpu.py": B / "p136" / "premise",
           "test_gemv_bw_served_gpu.py": REPO / "tests"}


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _run(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=120)


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p143.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_the_premise_tests_are_staged_at_p136s_pins():
    mine, p136 = _pins(LANE / "staged-p143.sha256"), _pins(B / "p136" / "staged-p136.sha256")
    shared = set(mine) & set(p136)
    assert shared == {"test_decode_graph_buckets.py", "test_kv_step_select.py", "test_fused_glue_decode_graphs_gpu.py",
                      "test_gemv_bw_served_gpu.py"}
    assert all(mine[n] == p136[n] for n in shared)


def test_the_runner_runs_the_bench_after_the_premise_and_reads_only_a_rule_verdict():
    run = (LANE / "p143_run.sh").read_text(encoding="utf-8")
    assert "GNF4_SHA=70bc03cb7a61b03c4e801e7423c58c103a1b6d7d" in run
    assert "p143-5090-1|p143-5090-2|p143-5090-3) FAMILY=qwen3-attn;" in run
    assert run.index("grep -qE '^19 passed") < run.index("python $W/p143_bench.py --out") < run.index("python $W/p143_reduce.py --bench")
    assert 'case "$V" in GO_FIXED|GO_ADAPTIVE|PARK) ;;' in run
    for gone in ("snapshot_download", "k8_bake.py", "nsys", "p136_box.py"):
        assert gone not in run, gone


def test_the_bench_hands_the_kernel_the_served_calls_arguments():
    """``Fp8PagedKV.attention`` passes n_kv_heads, head_dim, block_tokens, k_groups, k_row_bytes and v_row_bytes from the
    KV object; ``paged_attention`` adds sm_scale, window and sinks. The bench's call must name exactly those, plus n_split."""
    tree = ast.parse((LANE / "p143_bench.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "fp8_paged_decode_attention"]
    assert len(calls) == 1
    kws = {k.arg for k in calls[0].keywords}
    assert kws == {"n_kv_heads", "head_dim", "block_tokens", "k_groups", "k_row_bytes", "v_row_bytes", "sm_scale", "window",
                   "sinks", None}, kws          # None: the **kw that carries n_split
    kv_src = (REPO / "experts4bit_qlora" / "engines" / "fp8_paged_kv.py").read_text(encoding="utf-8")
    for arg in ("n_kv_heads=self.Hs[layer]", "head_dim=self.Ds[layer], block_tokens=self.bt", "k_groups=self.kgs[layer], k_row_bytes=self.k_row",
                "v_row_bytes=self.v_row, **kw"):
        assert arg in kv_src, arg


def test_every_contexts_kernel_args_outlive_its_graphs():
    """The review of #1759: ``kernel_args`` returns fresh copies of the table and lengths, and a CUDA graph keeps only
    their pointers. Each context's copies are held for the run and bound into the captured step as defaults (a late
    lookup of a rebound name would read the next context's), and the held lengths are read back before any timing."""
    src = (LANE / "p143_bench.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    step = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "step"]
    assert len(step) == 1
    defaults = [ast.unparse(d) for d in step[0].args.defaults]
    assert step[0].args.args[0].arg == "args" and defaults[0] == "held[ctx]", defaults
    assert "held = hold_args(kv, LAYERS)" in src
    assert src.index("    check_held(held)") < src.index("order = arms()"), "checked before timing"


def _bench():
    import importlib.util
    spec = importlib.util.spec_from_file_location("p143_bench_t", LANE / "p143_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_bench_setup_runs_on_the_served_kv_class():
    """Amendment 1 (p143-5090-1 failed here on the box): the bench's own fill / hold_args / check_held, on e4b's
    Fp8PagedKV on CPU, at the registered contexts and capacity (two layers). Every context's held lengths read it."""
    pytest.importorskip("fp8_kv", reason="the KV class needs grouped-nf4-gemm (CI installs it)")
    torch = pytest.importorskip("torch")
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    b = _bench()
    kv = Fp8PagedKV(2, b.HKV, b.D, batch=2, max_tokens_per_seq=b.CAP, device="cpu")
    b.fill(kv, torch, 2, b.CAP, "cpu")
    held = b.hold_args(kv, 2)
    assert set(held) == set(b.CTX)
    b.check_held(held)
    assert all(int(a[3][0]) == ctx for ctx, args in held.items() for a in args)
    with pytest.raises(ValueError, match="rewind forward"):          # the order p143-5090-1 used
        kv.rewind(0, b.CTX[0])
        kv.rewind(0, b.CTX[1])


@pytest.mark.parametrize("script", ["p143_bench.py", "p143_reduce.py"])
def test_the_self_tests_pass(script):
    out = _run(LANE / script, "--self-test")
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


MUTANTS = [
    # the 4096 loss ignored: a split that only wins at short context reads GO_FIXED
    ('        fixed.append({"n_split": s, "save_640_ms": round(save, 6), "loss_4096_ms": round(loss, 6), "noise_4096_ms": round(tol, 6),\n'
     '                      "clears": save >= BAR and loss <= tol})',
     '        fixed.append({"n_split": s, "save_640_ms": round(save, 6), "loss_4096_ms": round(loss, 6), "noise_4096_ms": round(tol, 6),\n'
     '                      "clears": save >= BAR})'),
    # the bar set against the wrong period (1 % instead of 3 %)
    ("BAR = round(0.03 * P142_P_MS, 6)", "BAR = round(0.01 * P142_P_MS, 6)"),
    # the self-pair never checked
    ("            if s is None and not SELF_PAIR[0] <= b / a <= SELF_PAIR[1]:", "            if False:"),
    # the transfer anchor never checked
    ("        if not ANCHOR[0] <= anchor <= ANCHOR[1]:", "        if False:"),
]


@pytest.mark.parametrize("old,new", MUTANTS)
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p143_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p143_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p143_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P143_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p143_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p143-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p143" in out.stdout and "bash p143_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p143_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "pername-5090-1"), timeout=120)
    assert out.returncode == 78 and "is not p143-5090-N" in out.stdout + out.stderr
