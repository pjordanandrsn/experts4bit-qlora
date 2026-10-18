# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The W8A16 quality premise (bench/i8dense/PREMISE-w8a16.md; e4b#1313): the int8 round trip on CPU (its error bound,
only the named weights touched, shapes unchanged), the staged pins and P125's instrument byte-identical, the runner on
P125's subject, the self-tests, and the driver's run ids."""
import hashlib
import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "i8dense"
SOURCES = {"i8_run.sh": LANE, "i8_box.py": LANE, "i8_fakequant.py": LANE, "i8_reduce.py": LANE, "p125_box.py": B / "p125",
           "p125_reduce.py": B / "p125", "p115_quality.py": B / "p115", "p110_box.py": B / "p110", "p108_box.py": B / "p108",
           "p97_box.py": B / "p97", "p109_box.py": B / "p109", "k8_bake.py": B / "p39", "calib.json": B / "p39",
           "test_decode_graph_buckets.py": REPO / "tests", "test_kv_step_select.py": REPO / "tests",
           "test_fused_glue_decode_graphs_gpu.py": REPO / "tests", "test_gemv_bw_served_gpu.py": REPO / "tests",
           "test_int4_attn.py": REPO / "tests", "test_int4_attn_calib.py": REPO / "tests"}
PYPATH = os.pathsep.join(str(B / d) for d in ("p125", "p109", "p115", "p110", "p108", "p97"))


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _fq():
    spec = importlib.util.spec_from_file_location("i8_fakequant_t", LANE / "i8_fakequant.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-i8dense.sha256")
    assert set(pins) == set(SOURCES), sorted(set(pins) ^ set(SOURCES))
    for name, want in pins.items():
        assert hashlib.sha256((SOURCES[name] / name).read_bytes()).hexdigest() == want, name


def test_p125s_instrument_is_staged_byte_identical():
    mine, p125 = _pins(LANE / "staged-i8dense.sha256"), _pins(B / "p125" / "staged-p125.sha256")
    shared = set(mine) & set(p125)
    assert shared == set(SOURCES) - {"i8_run.sh", "i8_box.py", "i8_fakequant.py", "i8_reduce.py"}
    for name in shared:
        assert mine[name] == p125[name], name


def test_the_driver_and_the_runner_stage_exactly_the_pinned_pieces():
    drive = (LANE / "i8_drive.sh").read_text(encoding="utf-8")
    stage = re.search(r'^STAGE="([^"]+)"', drive, re.M).group(1).split()
    assert {pathlib.PurePosixPath(p).name for p in stage} - {"staged-i8dense.sha256"} == set(SOURCES)
    run = (LANE / "i8_run.sh").read_text(encoding="utf-8")
    listed = re.search(r"^for f in (.*?); do$", run, re.M | re.S).group(1).replace("\\\n", " ").split()
    assert set(listed) - {"staged-i8dense.sha256"} == set(SOURCES) - {"i8_run.sh"}


def test_the_runner_installs_p125s_subject_and_runs_a_then_i8():
    run = (LANE / "i8_run.sh").read_text(encoding="utf-8")
    assert re.search(r"^E4B_T=4934aba9c88da7ff49e12e4911354538ae0fdf48\b", run, re.M)
    assert re.search(r"^GNF4_SHA=6ee2e10408161a9d3c874975c9191a7f2957e6f4\b", run, re.M)
    assert "MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in run
    assert "experts4bit-qlora.git@$E4B_T" in run and "WANT_E4B=$E4B_T" in run
    assert "for ARM in A I8; do" in run and "I8_ARM=$ARM" in run and "--mode speed" not in run
    assert run.index("python $W/i8_reduce.py --dir $W") > run.index("for ARM in A I8; do")
    assert "E4B_SERVE_ATTN_INT4" not in run.split("for ARM in A I8; do")[1]      # no int4 lever reaches an arm


@pytest.mark.parametrize("script", ["i8_fakequant.py", "i8_box.py", "i8_reduce.py"])
def test_the_self_tests_pass(script):
    out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True, timeout=120,
                         env=dict(os.environ, PYTHONPATH=PYPATH))
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_the_round_trip_touches_only_the_named_weights_and_keeps_shapes():
    """The maintainer's unit tests: the error bound per row, only attention projections and the head rounded, every
    other parameter bitwise unchanged, and every shape and dtype unchanged."""
    torch = pytest.importorskip("torch")
    fq = _fq()

    class Qwen3MoeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = torch.nn.Linear(32, 48, bias=False, dtype=torch.bfloat16)
            self.o_proj = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
            self.q_norm = torch.nn.LayerNorm(8, dtype=torch.bfloat16)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.ModuleList([Qwen3MoeAttention() for _ in range(3)])
            self.gate = torch.nn.Linear(32, 8, bias=False, dtype=torch.bfloat16)
            self.experts = torch.nn.Parameter(torch.randn(8, 16, 32, dtype=torch.bfloat16))
            self.embed = torch.nn.Embedding(50, 32, dtype=torch.bfloat16)
            self.lm_head = torch.nn.Linear(32, 50, bias=False, dtype=torch.bfloat16)

        def get_input_embeddings(self):
            return self.embed

    torch.manual_seed(3)
    m = Toy()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    rec = fq.apply(m)
    named = {f"attn.{i}.{p}.weight" for i in range(3) for p in ("qkv_proj", "o_proj")} | {"lm_head.weight"}
    assert rec["modules"] == 7 and rec["attention_modules"] == 6
    for n, p in m.named_parameters():
        assert p.shape == before[n].shape and p.dtype == before[n].dtype, n
        if n in named:
            w0 = before[n].float()
            half = w0.abs().amax(dim=1, keepdim=True) / fq.QMAX / 2
            err = (p.float() - w0).abs()
            assert bool((err <= half + w0.abs().amax(dim=1, keepdim=True) * 2.0 ** -8 + 1e-12).all()), n
            assert not torch.equal(p, before[n]), f"{n} was not rounded"
        else:
            assert torch.equal(p, before[n]), f"{n} changed: only the named weights may"


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "I8_DRIVE_DRYRUN": "1",
            "E4B_SHA": "1" * 40, "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q",
            "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800",
            "E4B_RENT_INSTANCE_ID": "0"}


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    out = subprocess.run(["bash", str(LANE / "i8_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "w8a16-premise-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/i8dense" in out.stdout and "bash i8_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    out = subprocess.run(["bash", str(LANE / "i8_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p125-5090-9"), timeout=120)
    assert out.returncode == 78 and "is not w8a16-premise-5090-N" in out.stdout + out.stderr
