# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P143 Amendment 2 (bench/p143/PREREG-p143.md; e4b#1313), the served read of E4B_ATTN_SPLIT_CAP: the staged pins
(P136's premise tests and P115 Phase D's served pieces byte-identical to their own registrations), the runner's order,
the kernel script reproducing the served call and holding its graphs' arguments, its setup on e4b's KV class, the SANE
pass's KV served at the server's capacity (P115's pass builds 656 tokens, where the cap cannot bind), the self-tests run
as the box runs them (one directory), the mutants the reducer's self-test must kill, and the driver's run ids."""
import ast
import hashlib
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p143"
SOURCES = {"p143a2_run.sh": LANE, "p143a2_kernel.py": LANE, "p143a2_box.py": LANE, "p143a2_reduce.py": LANE,
           "p115_quality.py": B / "p115", "p109_box.py": B / "p109", "p110_box.py": B / "p110", "p108_box.py": B / "p108",
           "p97_box.py": B / "p97", "k8_bake.py": B / "p39", "calib.json": B / "p39",
           "test_decode_graph_buckets.py": REPO / "tests", "test_kv_step_select.py": REPO / "tests",
           "test_fused_glue_decode_graphs_gpu.py": B / "p136" / "premise", "test_gemv_bw_served_gpu.py": REPO / "tests"}
PREMISE = {"test_decode_graph_buckets.py", "test_kv_step_select.py", "test_fused_glue_decode_graphs_gpu.py",
           "test_gemv_bw_served_gpu.py"}
SERVED = {"p115_quality.py", "p109_box.py", "p110_box.py", "p108_box.py", "p97_box.py", "k8_bake.py", "calib.json"}


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _run(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=180, cwd=cwd)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p143a2.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_the_premise_is_p136s_and_the_served_pieces_are_p115_phase_ds():
    mine = _pins(LANE / "staged-p143a2.sha256")
    p136, p115d = _pins(B / "p136" / "staged-p136.sha256"), _pins(B / "p115" / "staged-d.sha256")
    assert PREMISE <= set(p136) and all(mine[n] == p136[n] for n in PREMISE)
    assert SERVED <= set(p115d) and all(mine[n] == p115d[n] for n in SERVED)


def test_the_runner_reads_the_kernel_before_the_fetch_and_only_a_rule_verdict():
    run = (LANE / "p143a2_run.sh").read_text(encoding="utf-8")
    assert "GNF4_SHA=38b308887688ea5bda7dd449bd3cc3dc1426b995" in run and 'md.version("grouped-nf4-gemm") == "0.47.1"' in run
    assert "p143a2-5090-1|p143a2-5090-2|p143a2-5090-3) MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39;;" in run
    order = ["grep -qE '^19 passed", "python $W/p143a2_kernel.py --out", "snapshot_download", "python $W/k8_bake.py",
             "--prompts-only", "for TAG in S0a S1a S1b S0b; do", "for PH in off on; do", "python $W/p143a2_reduce.py --dir"]
    at = [run.index(x) for x in order]
    assert at == sorted(at), dict(zip(order, at))
    assert 'ARM=${TAG:0:2}; CAPV=0; [ "$ARM" = S1 ] && CAPV=16' in run and 'ARM=S0; CAPV=0; [ "$PH" = on ] && { ARM=S1; CAPV=16; }' in run
    assert run.count("E4B_ATTN_SPLIT_CAP=$CAPV P143A2_ARM=$ARM") == 2
    assert "E4B_ATTN_SPLIT_CAP" in run.split("unset ", 1)[1].split(": > summary.txt")[0], "the cap starts unset"
    assert 'case "$V" in FLIP|NO_FLIP|KERNEL_FAIL|FUNCTION_FAIL|QUALITY_FAIL) ;;' in run


def test_the_kernel_timing_hands_the_kernel_the_served_calls_arguments():
    """As step 1's bench: the arguments ``Fp8PagedKV.attention`` and ``paged_attention`` pass, plus the **kw that carries
    ``split_cap`` (the cap arm) or nothing (the plan arm)."""
    tree = ast.parse((LANE / "p143a2_kernel.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "fp8_paged_decode_attention"]
    assert len(calls) == 1
    assert {k.arg for k in calls[0].keywords} == {"n_kv_heads", "head_dim", "block_tokens", "k_groups", "k_row_bytes",
                                                  "v_row_bytes", "sm_scale", "window", "sinks", None}
    src = (LANE / "p143a2_kernel.py").read_text(encoding="utf-8")
    assert 'kw = {"split_cap": SPLIT_CAP} if arm == "cap" else {}' in src


def test_every_timing_graphs_arguments_outlive_it():
    src = (LANE / "p143a2_kernel.py").read_text(encoding="utf-8")
    step = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "step"]
    assert len(step) == 1
    assert [a.arg for a in step[0].args.args] == ["args", "q", "kw"]
    assert [ast.unparse(d) for d in step[0].args.defaults] == ["held[B, ctx]", "q", "kw"]
    assert "held = hold_args(kv)" in src and src.index("    check_held(held)") < src.index("order = timing_arms()")


def test_the_contract_runs_the_served_call_both_ways():
    """The contract goes through ``Fp8PagedKV.attention`` (the cap handed over by the KV, as served), unset then 16."""
    src = (LANE / "p143a2_kernel.py").read_text(encoding="utf-8")
    body = src[src.index("def contract("):src.index("def hold_args(")]
    assert body.index("kv.split_cap = None") < body.index("o_plan = kv.attention(layer, q, slots=slots, sm_scale=scale)") \
        < body.index("kv.split_cap = SPLIT_CAP") < body.index("o_cap = kv.attention(layer, q, slots=slots, sm_scale=scale)")
    assert "kv.reference_kv(layer, s, dtype=torch.float64)" in body


def test_the_kernel_setup_runs_on_the_served_kv_class():
    """fill / hold_args / check_held on e4b's Fp8PagedKV on CPU (two layers), and the fp64 oracle the contract reads."""
    pytest.importorskip("fp8_kv", reason="the KV class needs grouped-nf4-gemm (CI installs it)")
    torch = pytest.importorskip("torch")
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    k = _load("p143a2_kernel_t", LANE / "p143a2_kernel.py")
    kv = Fp8PagedKV(2, k.HKV, k.D, batch=k.SLOTS, max_tokens_per_seq=k.CAP, device="cpu")
    k.fill(kv, torch, k.CONTRACT_SLOTS + k.TIMING_SLOTS, "cpu", layers=2)
    held = k.hold_args(kv, layers=2)
    assert set(held) == {(b, c) for b in k.TIMING_B for c in k.TIMING_CTX}
    k.check_held(held)
    assert int(kv.block_table[0].shape[1]) == 256
    for s in k.CONTRACT_SLOTS:                                          # the contract's own slots, untouched by timing
        kv.rewind(s, 640)
    kk, vv = kv.reference_kv(1, 2, dtype=torch.float64)
    assert kk.shape == (640, k.HKV, k.D) and kk.dtype == torch.float64
    out = k.ref_attention(torch, torch.randn(k.HQ, k.D, dtype=torch.float64), kk, vv, k.D ** -0.5)
    assert out.shape == (k.HQ, k.D) and torch.isfinite(out).all()


def test_the_sane_pass_kv_is_served_at_the_servers_capacity_with_its_cap():
    """P110's pass builds its KV by keyword at ``P + C + 16`` tokens; inside ServedKV that KV is the server's 4096 (256
    blocks: the library plans 32 at B = 1, where 656 tokens plan 11 and a cap of 16 cannot bind), carries the cap, and
    its split_plans reach the record."""
    pytest.importorskip("fp8_kv", reason="the KV class needs grouped-nf4-gemm (CI installs it)")
    pytest.importorskip("torch")
    p110 = (B / "p110" / "p110_box.py").read_text(encoding="utf-8")
    assert "batch=n, max_tokens_per_seq=P + C + 16," in p110
    sys.path.insert(0, str(B / "p109"))
    try:
        box = _load("p143a2_box_t", LANE / "p143a2_box.py")
    finally:
        sys.path.remove(str(B / "p109"))
    from experts4bit_qlora.engines import fp8_paged_kv as m
    orig = m.Fp8PagedKV.__init__
    with box.ServedKV(4096, 16) as s:
        kv = m.Fp8PagedKV(2, 4, 128, batch=1, max_tokens_per_seq=512 + 128 + 16, device="cpu", scratch_slots=16)
        kv.split_plans.setdefault(1, {"calls": 0, "plan": 32, "n_split": 16})["calls"] += 2
    assert m.Fp8PagedKV.__init__ is orig
    assert (kv.blocks_per_seq, kv.split_cap) == (256, 16)
    r = s.report()
    assert r["asked"] == [656] and r["max_tokens_per_seq"] == [4096] and r["blocks_per_seq"] == [256]
    assert r["plans"] == {"1": {"calls": 2, "plan": 32, "n_split": 16}}


@pytest.fixture(scope="module")
def box_dir(tmp_path_factory):
    """The box's directory: every staged piece side by side, as the runner sees them."""
    d = tmp_path_factory.mktemp("p143a2_box")
    for name, src in SOURCES.items():
        shutil.copyfile(src / name, d / name)
    return d


@pytest.mark.parametrize("script", ["p143a2_kernel.py", "p143a2_box.py", "p143a2_reduce.py"])
def test_the_self_tests_pass(box_dir, script):
    out = _run(box_dir / script, "--self-test", cwd=box_dir)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


MUTANTS = [
    # the contract's bar loosened to 1.5x
    ("CONTRACT_B, CONTRACT_CTX, CONTRACT_RATIO = (1, 2, 4), (256, 640, 1024, 2048, 4096), 1.05",
     "CONTRACT_B, CONTRACT_CTX, CONTRACT_RATIO = (1, 2, 4), (256, 640, 1024, 2048, 4096), 1.5"),
    # the timing loss never compared with the pass-to-pass noise
    ("            if loss > noise:", "            if False:"),
    # g1 from the better pair, not the worse
    ('out["speed"] = {"decode_tok_s": rates, "g1": round(min(pairs["W1"]), 4)',
     'out["speed"] = {"decode_tok_s": rates, "g1": round(max(pairs["W1"]), 4)'),
    # W16 (the no-change check) never compared across arms
    ('        if _tok(S1a, "W16", n) != _tok(S0a, "W16", n):', '        if False:'),
    # the SANE pass's KV capacity never checked (P115's 656 tokens, where the cap cannot bind, would read)
    ('        if not served or (sk.get("passes") or 0) < SANE_WINDOWS',
     '        if (sk.get("passes") or 0) < SANE_WINDOWS'),
    # S1's engagement never checked
    ('    if arm == "S0":\n        if sc.get("cap") is not None or plans:', '    if True:\n        if False:'),
    # the SANE gate's argmax bar dropped
    ('    elif abs(sane["bias"]) > SANE_BIAS or sane["argmax_agree"] < SANE_ARGMAX:', '    elif abs(sane["bias"]) > SANE_BIAS:'),
]


@pytest.mark.parametrize("old,new", MUTANTS)
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p143a2_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p143a2_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p143a2_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P143A2_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p143a2_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p143a2-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p143a2" in out.stdout and "bash p143a2_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p143a2_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p143-5090-1"), timeout=120)
    assert out.returncode == 78 and "is not p143a2-5090-N" in out.stdout + out.stderr
