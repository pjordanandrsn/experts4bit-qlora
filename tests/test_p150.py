# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P150 (bench/p150/PREREG-p150.md; e4b#1313), D2's quality ruling on HELD-OUT windows: the staged pins (the split
premise's box and round trip, P125's instrument and premise tests, all at the bytes bench/i8dense pins), the held-out
windows (P125's loader with only the split changed; their registered sha256s equal in the box, the reducer and the
PREREG), the int8 grid equal to P148's lever, the runner on P125's subject with arms A then I8HEAD, the self-tests and
the reducer's mutants, and the driver's run ids."""
import ast
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
LANE, I8 = B / "p150", B / "i8dense"
SOURCES = {"p150_run.sh": LANE, "p150_box.py": LANE, "p150_reduce.py": LANE, "i8_box.py": I8, "i8_fakequant.py": I8, "p125_box.py": B / "p125",
           "p125_reduce.py": B / "p125", "p115_quality.py": B / "p115", "p110_box.py": B / "p110", "p108_box.py": B / "p108",
           "p97_box.py": B / "p97", "p109_box.py": B / "p109", "k8_bake.py": B / "p39", "calib.json": B / "p39",
           "test_decode_graph_buckets.py": REPO / "tests", "test_kv_step_select.py": REPO / "tests",
           "test_fused_glue_decode_graphs_gpu.py": REPO / "tests", "test_gemv_bw_served_gpu.py": REPO / "tests",
           "test_int4_attn.py": REPO / "tests", "test_int4_attn_calib.py": REPO / "tests"}
PYPATH = os.pathsep.join(str(B / d) for d in ("i8dense", "p125", "p109", "p115", "p110", "p108", "p97"))


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p150.sha256")
    assert set(pins) == set(SOURCES), sorted(set(pins) ^ set(SOURCES))
    for name, want in pins.items():
        assert hashlib.sha256((SOURCES[name] / name).read_bytes()).hexdigest() == want, name


def test_the_instrument_is_the_split_premises_at_its_pinned_bytes():
    """Everything but this lane's runner and reducer is staged at the bytes bench/i8dense/staged-i8dense.sha256 pins: the
    split premise's box and round trip, P125's instrument and P125's premise tests."""
    mine, i8 = _pins(LANE / "staged-p150.sha256"), _pins(I8 / "staged-i8dense.sha256")
    shared = set(SOURCES) - {"p150_run.sh", "p150_box.py", "p150_reduce.py"}
    assert shared <= set(i8)
    for name in shared:
        assert mine[name] == i8[name], name


def test_the_driver_and_the_runner_stage_exactly_the_pinned_pieces():
    drive = (LANE / "p150_drive.sh").read_text(encoding="utf-8")
    stage = re.search(r'^STAGE="([^"]+)"', drive, re.M).group(1).split()
    assert {pathlib.PurePosixPath(p).name for p in stage} - {"staged-p150.sha256"} == set(SOURCES)
    run = (LANE / "p150_run.sh").read_text(encoding="utf-8")
    listed = re.search(r"^for f in (.*?); do$", run, re.M | re.S).group(1).replace("\\\n", " ").split()
    assert set(listed) - {"staged-p150.sha256"} == set(SOURCES) - {"p150_run.sh"}
    assert "source \"$LANE_HELPER\"" in drive and "lane_two_missing" in drive        # liveness by launch identity


def test_the_runner_installs_p125s_subject_and_rules_only_on_i8head():
    run = (LANE / "p150_run.sh").read_text(encoding="utf-8")
    assert re.search(r"^E4B_T=4934aba9c88da7ff49e12e4911354538ae0fdf48\b", run, re.M)
    assert re.search(r"^GNF4_SHA=6ee2e10408161a9d3c874975c9191a7f2957e6f4\b", run, re.M)
    assert "MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in run
    loop = "for ARM in A I8HEAD; do"
    assert loop in run and "I8_ARM=$ARM" in run and "I8ATTN" not in run
    assert "skipped on the deadline" not in run                                 # the candidate never skips: a deadline is 40
    assert run.index("python $W/p150_reduce.py --dir $W --out $W/verdict_p150.json") > run.index(loop)
    assert 'case "$V" in GO|PARK) ;; *) say "P150 $V -- no reading (a VOID is not a ruling)"; finish 27;; esac' in run
    assert "E4B_SERVE_ATTN_INT4" not in run.split(loop)[1]                       # no int4 lever reaches an arm
    assert "python $W/p150_box.py --windows-check" in run and "python $W/p150_box.py --mode quality" in run
    assert "i8_box.py --" not in run                                           # every box call goes through the held-out wrapper


def test_the_rounding_is_p148s_lever_bitwise():
    """I8HEAD stores the same values P148's int8 head computes with: i8_fakequant's per-row grid equals
    p148_kernels.quantize_rows_int8 dequantized, bit for bit, including a zero row and a row with one large entry."""
    torch = pytest.importorskip("torch")
    fq = _load(I8 / "i8_fakequant.py", "i8_fakequant_p150")
    sys.path.insert(0, str(B / "p148"))
    try:
        k = _load(B / "p148" / "p148_kernels.py", "p148_kernels_p150")
    except ImportError as e:                      # p148_kernels imports triton at module scope
        pytest.skip(f"p148_kernels needs triton: {e}")
    finally:
        sys.path.pop(0)
    g = torch.Generator().manual_seed(149)
    w = (torch.randn(64, 256, generator=g) * 0.02).to(torch.bfloat16)
    w[3] = 0
    w[5, 7] = 3.0
    rt = w.clone()
    fq.roundtrip_(rt)
    q, s = k.quantize_rows_int8(w)
    assert torch.equal(rt, (q.float() * s[:, None]).to(torch.bfloat16))
    assert torch.equal(rt[3], torch.zeros_like(rt[3]))


def _heldout(path):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "HELDOUT":
            return ast.literal_eval(node.value)
    raise AssertionError(f"no HELDOUT in {path}")


def test_the_heldout_hashes_are_one_registration():
    """The box checks and records the held-out windows, the reducer VOIDs on any other: one registration, three copies
    (the box, the reducer, the PREREG), held equal here."""
    box, red = _heldout(LANE / "p150_box.py"), _heldout(LANE / "p150_reduce.py")
    assert box == red and box["split"] == "validation" and box["stride"] == 1024 and set(box["sha256"]) == {"t1", "k16"}
    prereg = (LANE / "PREREG-p150.md").read_text(encoding="utf-8")
    assert all(h in prereg for h in box["sha256"].values())


def test_the_heldout_loader_is_p125s_with_only_the_split_changed():
    """The same corpus, join, tokenisation and window slicing as P125's wikitext loader; only the split differs."""
    def body(path, name):
        src = path.read_text(encoding="utf-8")
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.get_source_segment(src, fn)
    mine, p125 = body(LANE / "p150_box.py", "heldout_wikitext_windows"), body(B / "p125" / "p125_box.py", "wikitext_windows")
    assert 'load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")' in p125
    assert 'load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=HELDOUT["split"])' in mine
    for line in ('text = "\\n\\n".join(t for t in ds["text"] if t.strip())', 'ids = tok(text, return_tensors="pt").input_ids[0]',
                 "for k in range(n):", "w = ids[k * stride:k * stride + prompt + cont]", "assert w.numel() == prompt + cont",
                 "out.append(w.tolist())"):
        assert line in mine and line in p125, line


@pytest.mark.parametrize("script", ["p150_reduce.py", "p150_box.py"])
def test_the_self_tests_pass(script):
    out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True,
                         timeout=600, env=dict(os.environ, PYTHONPATH=PYPATH))
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


MUTANTS = [
    # the point d instead of the one-sided upper bound (the split premise's own mutant)
    ("UPPER_Z = 1.645 ", "UPPER_Z = 0.0 "),
    # either gate instead of both
    ('out["verdict"] = "GO" if all(out["gates"][g]["passes"] for g in RULED) else "PARK"',
     'out["verdict"] = "GO" if any(out["gates"][g]["passes"] for g in RULED) else "PARK"'),
    # t1 alone
    ('RULED, CANDIDATE = ("t1", "k16"), "I8HEAD"', 'RULED, CANDIDATE = ("t1",), "I8HEAD"'),
    # the argmax gate never read
    ('    if s["argmax"] < pr.ARGMAX_MIN:', "    if False:"),
    # the scope never checked (an arm that also rounds attention, or not the head, would rule)
    ("        if {k: fq.get(k) for k in SCOPE[CANDIDATE]} != SCOPE[CANDIDATE]:", "        if False:"),
    # the ruled window counts never checked
    ("    if n_want is not None and (len(ra) != n_want or len(rx) != n_want):", "    if False:"),
    # the recorded windows (counts and stride) never checked
    ('        elif rec.get("windows") != WINDOWS:', "        elif False:"),
    # the held-out windows never checked: a read on P125's seen windows (or any other) would rule
    ('        elif rec.get("heldout") != HELDOUT:', "        elif False:"),
]


@pytest.mark.parametrize("old,new", MUTANTS)
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p150_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p150_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = subprocess.run([sys.executable, str(tmp_path / "p150_reduce.py"), "--self-test"], capture_output=True,
                         text=True, timeout=600, env=dict(os.environ, PYTHONPATH=PYPATH))
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P150_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0"}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_the_drivers_dry_run_names_the_launch(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p150_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "p150-5090-1"), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN stage -> root@h:/root/p150" in out.stdout and "bash p150_run.sh" in out.stdout


def test_the_driver_refuses_another_run_id(tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree before it reads the run id")
    out = subprocess.run(["bash", str(LANE / "p150_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, "w8a16-split-5090-1"), timeout=120)
    assert out.returncode == 78 and "is not p150-5090-N" in out.stdout + out.stderr
