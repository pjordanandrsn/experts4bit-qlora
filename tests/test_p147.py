# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P147, the Mixtral attention-format quality premise (bench/p147/PREREG-p147.md): the staged closure at its pinned
bytes, with the reused pieces at their own lanes' pins; the round trips', the box's and the reducer's self-tests; the
mutants the reducer must kill; the model lock; the driver's dry run and refusals; the runner's start floor and
closure tripwire."""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p147"
SOURCES = {"p147_box.py": LANE, "p147_fakequant.py": LANE, "p147_reduce.py": LANE, "p147_run.sh": LANE,
           "model_files.json": LANE, "i8_fakequant.py": B / "i8dense", "p97_box.py": B / "p97", "p108_box.py": B / "p108",
           "p109_box.py": B / "p109", "p110_box.py": B / "p110", "p115_quality.py": B / "p115", "p125_box.py": B / "p125",
           "quality_off_mixtral.json": B / "p141" / "receipts" / "p141-5090-2"}
PREREG = (LANE / "PREREG-p147.md").read_text(encoding="utf-8")


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _run(script, *args, env=None):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=600,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", **(env or {})))


def test_every_staged_piece_matches_its_source():
    pins = _pins(LANE / "staged-p147.sha256")
    assert set(pins) == set(SOURCES)
    for name, d in SOURCES.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == pins[name], name


def test_the_reused_pieces_are_at_their_own_lanes_pins():
    mine = _pins(LANE / "staged-p147.sha256")
    elsewhere = {}
    for f in B.glob("*/staged*.sha256"):
        if f.parent.name != "p147":
            for name, sha in _pins(f).items():
                elsewhere.setdefault(name, set()).add(sha)
    for name in ("i8_fakequant.py", "p97_box.py", "p108_box.py", "p109_box.py", "p110_box.py", "p115_quality.py",
                 "p125_box.py"):
        assert mine[name] in elsewhere.get(name, set()), name


@pytest.mark.parametrize("script", ["p147_fakequant.py", "p147_reduce.py", "p147_box.py"])
def test_the_self_tests_pass(script):
    out = _run(LANE / script, "--self-test")
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


MUTANTS = [
    ("UPPER_Z = 1.645", "UPPER_Z = 0.0"),                       # the point d instead of the one-sided upper bound
    ("ARGMAX_MIN = 0.95", "ARGMAX_MIN = 0.0"),                  # no argmax gate
    ("CHECK_B_TOL = 0.001 ", "CHECK_B_TOL = 1.0 "),             # no repeat check
    ('if (c.get("format")', 'if False and (c.get("format")'),   # no round-trip census
]


@pytest.mark.parametrize("old,new", MUTANTS)
def test_each_mutant_fails_the_reducer_self_test(tmp_path, old, new):
    src = (LANE / "p147_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p147_reduce.py").write_text(src.replace(old, new), encoding="utf-8")
    out = _run(tmp_path / "p147_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


def test_the_model_lock_is_the_registered_revision():
    lock = json.loads((LANE / "model_files.json").read_text())
    assert lock["revision"] == "eba92302a2861cdc0098cc54bc9f17cb2c47eb61"
    assert len(lock["shards"]) == 19 and sum(s["size"] for s in lock["shards"]) == lock["safetensors_bytes"]
    assert {s["file"] for s in lock["small"]} == {"config.json", "model.safetensors.index.json"}
    small = {s["file"]: s["blob_id"] for s in lock["small"]}
    assert small["config.json"].startswith("de132a80") and small["model.safetensors.index.json"].startswith("fc6de58e")
    assert "de132a80" in PREREG and "fc6de58e" in PREREG and "eba92302" in PREREG


def test_the_prereg_registers_the_rule_and_the_checks():
    for s in ("argmax agreement", "d + 1.645 SE ≤ bound", "ln(1 + 0.05 / ppl_A)", "**196**", "**123**",
              "|ΔNLL| ≤ 0.02 nats per text", "served FP8-KV paged reference", "|Δppl| ≤ 0.001 per set", "10,000 MiB", "38b308887688ea5bda7dd449bd3cc3dc1426b995",
              "No timing from this box is reported or used"):
        assert s in PREREG, s
    run = (LANE / "p147_run.sh").read_text()
    assert "FREE_MIB_MIN=10000" in run and "GNF4_SHA=38b308887688ea5bda7dd449bd3cc3dc1426b995" in run
    red = (LANE / "p147_reduce.py").read_text()
    for s in ("ARGMAX_MIN = 0.95", "UPPER_Z = 1.645", "PPL_BUDGET = 0.05", "CHECK_A_TOL = 0.02", "CHECK_B_TOL = 0.001 "):
        assert s in red, s


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def _bash_or_skip():
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")


def test_the_driver_refuses_another_run_id(tmp_path):
    _bash_or_skip()
    out = subprocess.run(["bash", str(LANE / "p147_drive.sh"), "p147-5090-1"], capture_output=True, text=True, timeout=60,
                         env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P147_DRIVE_DRYRUN": "1",
           "P147_CARD_HOST": "card@host", "P147_BUS_CMD": "cat", "P147_BUS_ME": "lane/label"})
    assert out.returncode == 78 and "is not p147-a2000-N" in out.stdout


def test_the_drivers_dry_run_stages_the_closure_under_the_claim(tmp_path):
    _bash_or_skip()
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = subprocess.run(["bash", str(LANE / "p147_drive.sh"), "p147-a2000-1"], capture_output=True, text=True, timeout=120,
                         env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P147_DRIVE_DRYRUN": "1",
           "P147_CARD_HOST": "card@host", "P147_BUS_CMD": "cat", "P147_BUS_ME": "lane/label"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DRYRUN bus claim" in out.stdout and "gpu:a2000" in out.stdout
    assert "DRYRUN stage -> card@host:/share/gpu-dev/p147/p147-a2000-1 (13 files" in out.stdout
    assert "p147_run.sh" in out.stdout and "DRYRUN bus release" in out.stdout


@pytest.mark.parametrize("var", ["P147_CARD_HOST", "P147_BUS_CMD", "P147_BUS_ME"])
def test_the_driver_has_no_lab_defaults(tmp_path, var):
    _bash_or_skip()
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P147_DRIVE_DRYRUN": "1",
           "P147_CARD_HOST": "card@host", "P147_BUS_CMD": "cat", "P147_BUS_ME": "lane/label"}
    del env[var]
    out = subprocess.run(["bash", str(LANE / "p147_drive.sh"), "p147-dry-1"], capture_output=True, text=True, timeout=60, env=env)
    assert out.returncode != 0 and f"set {var}" in out.stderr and "DRYRUN bus" not in out.stdout


def _runner_dir(tmp_path, free_mib):
    d = tmp_path / "run"
    d.mkdir()
    for name, src in SOURCES.items():
        shutil.copy(src / name, d / name)
    shutil.copy(LANE / "staged-p147.sha256", d / "staged-p147.sha256")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text(
        "#!/bin/sh\ncase \"$*\" in *memory.free*) echo %d;; *compute-apps*) ;; esac\n" % free_mib)
    (bin_ / "nvidia-smi").chmod(0o755)
    env = {"PATH": f"{bin_}:{os.environ.get('PATH', '/usr/bin:/bin')}", "HOME": str(tmp_path), "E4B_SHA": "a" * 40,
           "P147_VENV": str(tmp_path / "no-venv"), "P147_HF_HOME": str(tmp_path / "hf")}
    return d, env


def test_the_runner_exits_not_run_on_a_busy_card(tmp_path):
    _bash_or_skip()
    d, env = _runner_dir(tmp_path, 500)
    out = subprocess.run(["bash", str(d / "p147_run.sh")], capture_output=True, text=True, timeout=60, env=env)
    assert out.returncode == 75 and (d / "P147_DONE").read_text().strip() == "75" and (d / "NOT_RUN").exists()
    assert "NOT_RUN: the card is busy" in (d / "summary.txt").read_text()


def test_the_runner_voids_a_closure_off_its_pins(tmp_path):
    _bash_or_skip()
    d, env = _runner_dir(tmp_path, 11000)
    (d / "p147_reduce.py").write_text("# changed\n")
    out = subprocess.run(["bash", str(d / "p147_run.sh")], capture_output=True, text=True, timeout=60, env=env)
    assert out.returncode == 9 and "VOID: the staged closure does not match" in (d / "summary.txt").read_text()


def test_a_dry_id_runs_the_shakedown_and_a_registered_id_never_does(tmp_path):
    _bash_or_skip()
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P147_DRIVE_DRYRUN": "1",
           "P147_CARD_HOST": "card@host", "P147_BUS_CMD": "cat", "P147_BUS_ME": "lane/label"}
    dry = subprocess.run(["bash", str(LANE / "p147_drive.sh"), "p147-dry-1"], capture_output=True, text=True, timeout=120, env=env)
    reg = subprocess.run(["bash", str(LANE / "p147_drive.sh"), "p147-a2000-2"], capture_output=True, text=True, timeout=120, env=env)
    assert dry.returncode == 0 and "P147_SHAKEDOWN=1" in dry.stdout, dry.stdout + dry.stderr
    assert reg.returncode == 0 and "P147_SHAKEDOWN=0" in reg.stdout, reg.stdout + reg.stderr


def test_the_shakedown_scores_no_real_window():
    box = (LANE / "p147_box.py").read_text()
    shake = box[box.index("    if a.shakedown:"):box.index("    snap = fq.snapshot(model)")]
    assert "torch.randint" in shake and "wins[" in shake and "score(model, synth" in shake
    assert "score_arm" not in shake and 'wins[s]["windows"]' not in shake    # only the windows' sha256 is read
    run = (LANE / "p147_run.sh").read_text()
    assert '[ -n "$SK" ] && done_ "$rc"' in run   # the shakedown ends before the reduce
