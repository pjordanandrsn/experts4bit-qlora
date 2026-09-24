"""P67's draw machinery, EXECUTED (bench/p67/P67-PREREG.md).

P56's draw 1 lost every arm to a line of `tp4_run.sh` that three dry runs never executed (e4b#672): `bash -n`
parses without running, `--selftest` never goes through the run script, and the driver's dry run stops before the
box does anything. So each path this lane adds is driven here for real, with only the GPU work stubbed:

* the P67 block of `tp4_run.sh` -- extracted from the real file and run, so the per-arm order provably reaches the
  arm's CHILD process, only on the perm arms, and does not leak into the shell afterwards;
* `tp4_arm.py`'s refusal of the switch on a non-reference arm (a real process, exit 19, a real stub);
* `p67_run.sh`'s guard on a fake box directory: pins, registered knobs, the fixture lock, host RAM -- every
  refusal writes the markers the controller waits for;
* `p67_drive.sh` -> `tp4_drive.sh` in dry-run: the box is told to start the guard, with the registered knobs.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p67"
TP4 = REPO / "bench" / "tp4"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="no bash")


def _knobs():
    out = {}
    for ln in (LANE / "registered.knobs").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            k, v = ln.split("|", 1)
            out[k] = v
    return out


# ----------------------------------------------------------------------------- tp4_run.sh's P67 block
def _p67_block():
    sh = (TP4 / "tp4_run.sh").read_text()
    m = re.search(r'^(\s*)if \[ "\$\{TP4_P67:-0\}" = 1 \]; then\n.*?^\1fi\n', sh, re.M | re.S)
    assert m, "the P67 block is not in the shape this test drives"
    return m.group(0)


def _run_block(tmp_path, env_extra):
    script = f"""
set -uo pipefail
cd {tmp_path}
can_run(){{ return 0; }}
say(){{ echo "SAY $*"; }}
arm(){{ echo "ARM tag=$3 arm=$4 child_order=$(sh -c 'echo "${{E4B_REFERENCE_EXPERT_ORDER:-none}}"')"; }}
FAM=gemma4 RAL=5400 MID=m REV=r OFF=0 TOK=t TS=s
{_p67_block()}
echo "AFTER order=${{E4B_REFERENCE_EXPERT_ORDER:-none}}"
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TP4_", "E4B_REFERENCE"))}
    env.update(env_extra)
    return subprocess.run([BASH, "-c", script], capture_output=True, text=True, env=env)


def test_the_p67_block_sets_the_order_on_the_perm_arms_only(tmp_path):
    r = _run_block(tmp_path, {"TP4_P67": "1", "TP4_P67_PERMS": "1 2 x 4"})
    assert r.returncode == 0, r.stderr
    arms = re.findall(r"ARM tag=(\S+) arm=(\S+) child_order=(\S+)", r.stdout)
    assert arms == [("reference_attn4_repeat", "reference", "none"), ("reference_attn4_perm1", "reference", "perm:1"),
                    ("reference_attn4_perm2", "reference", "perm:2"), ("reference_attn4_perm4", "reference", "perm:4")], r.stdout
    assert "AFTER order=none" in r.stdout                      # the prefix is scoped to the call; nothing leaks
    assert "P67 BAD_SEED x" in (tmp_path / "summary.txt").read_text()


def test_the_p67_block_is_opt_in(tmp_path):
    r = _run_block(tmp_path, {})
    assert r.returncode == 0 and "ARM " not in r.stdout


def test_the_registered_perms_are_the_default(tmp_path):
    r = _run_block(tmp_path, {"TP4_P67": "1"})
    assert re.findall(r"tag=reference_attn4_perm(\d)", r.stdout) == ["1", "2", "3", "4"]
    assert _knobs()["TP4_P67_PERMS"] == "1 2 3 4"


# ----------------------------------------------------------------------------- tp4_arm.py's refusal
def test_tp4_arm_refuses_the_switch_on_a_non_reference_arm(tmp_path):
    pytest.importorskip("torch")
    env = dict(os.environ, E4B_REFERENCE_EXPERT_ORDER="perm:1")
    r = subprocess.run([sys.executable, str(TP4 / "tp4_arm.py"), "--framework", "e4b", "--arm", "fused", "--tag", "fused_attn4",
                        "--fam", "gemma4", "--model", "none", "--revision", "none", "--tokens", str(tmp_path / "t.json"),
                        "--prereg", "bench/p67/P67-PREREG.md", "--out", str(tmp_path), "--adapter-dir", str(tmp_path / "a")],
                       capture_output=True, text=True, env=env, cwd=REPO, timeout=300)
    assert r.returncode == 19, r.stdout[-2000:] + r.stderr[-2000:]
    import json
    rec = json.loads((tmp_path / "gemma4_e4b_fused_attn4.json").read_text())
    assert rec["status"] == "harness_error" and "reference-arm switch only" in rec["reason"]


# ----------------------------------------------------------------------------- p67_run.sh's guard
def _box(tmp_path, ram_gib=120, cgroup=None, tamper=None):
    """A fake /root/tp4: the staged files as the driver would stage them, and a fake meminfo / cgroup."""
    box = tmp_path / "box"
    box.mkdir()
    for line in (LANE / "staged.sha256").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name = line.split(None, 1)[1].strip()
        src = {"p67_run.sh": LANE / name, "registered.knobs": LANE / name, "n9_datasets.py": REPO / "bench/flagship-matrix/drivers" / name,
               "ds_manifest.json": REPO / "bench/flagship-matrix" / name}.get(name, TP4 / name)
        shutil.copy(src, box / name)
    shutil.copy(LANE / "staged.sha256", box / "staged.sha256")
    if tamper:
        with open(box / tamper, "a") as f:
            f.write("\n# tampered\n")
    (tmp_path / "meminfo").write_text(f"MemTotal:       {ram_gib * 1048576} kB\nMemFree: 1 kB\n")
    (tmp_path / "cgmax").write_text("max\n" if cgroup is None else f"{cgroup * 1024 ** 3}\n")
    return box


def _guard(tmp_path, box, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TP4_", "E4B_REFERENCE", "GNF4_"))}
    env.update({"P67_BOX_DIR": str(box), "P67_GUARD_ONLY": "1", "TP4_RUN_NONCE": "n0nce",
                "P67_MEMINFO": str(tmp_path / "meminfo"), "P67_CGROUP_MEMMAX": str(tmp_path / "cgmax")})
    env.update(extra or {})
    return subprocess.run([BASH, str(box / "p67_run.sh")], capture_output=True, text=True, env=env)


needs_sha256sum = pytest.mark.skipif(shutil.which("sha256sum") is None, reason="the box-side guard uses sha256sum")


@needs_sha256sum
def test_the_guard_passes_a_good_box_and_sets_every_registered_knob(tmp_path):
    box = _box(tmp_path)
    r = _guard(tmp_path, box)
    assert r.returncode == 0, r.stdout + r.stderr
    g = (box / "p67_guard.txt").read_text()
    assert "PINS OK" in g and "GUARD OK" in g
    for k, v in _knobs().items():
        assert f"KNOB {k}={v}" in g


@needs_sha256sum
@pytest.mark.parametrize("ram,cg", [(64, None), (120, 32), (95, None)])
def test_the_guard_refuses_a_host_below_the_shard_class(tmp_path, ram, cg):
    box = _box(tmp_path, ram_gib=ram, cgroup=cg)
    r = _guard(tmp_path, box)
    assert r.returncode == 18, r.stdout
    assert (box / "TP4_EXIT_CODE.n0nce").read_text().strip() == "18" and (box / "TP_DONE.n0nce").exists()
    assert (box / "TP4_RUN_NONCE").read_text().strip() == "n0nce"


@needs_sha256sum
@pytest.mark.parametrize("extra", [{"TP4_STEPS": "60"}, {"TP4_FAMILIES": "gemma4 mixtral"}, {"GNF4_SHA": "d9fd170d83305fbe3f9b5ae2e661d45307c0a2c5"},
                                   {"TP4_SEQ": "512"}, {"TP4_EVAL_N": "8"}, {"E4B_REFERENCE_EXPERT_ORDER": "perm:1"}])
def test_the_guard_refuses_a_knob_that_is_not_the_registration(tmp_path, extra):
    box = _box(tmp_path)
    r = _guard(tmp_path, box, extra)
    assert r.returncode == 78, r.stdout
    assert (box / "TP_DONE.n0nce").exists()


@needs_sha256sum
def test_the_guard_accepts_a_knob_passed_with_its_registered_value(tmp_path):
    box = _box(tmp_path)
    r = _guard(tmp_path, box, {"TP4_STEPS": "20", "TP4_P67_PERMS": "1 2 3 4"})
    assert r.returncode == 0, r.stdout


@needs_sha256sum
@pytest.mark.parametrize("victim", ["tp4_arm.py", "registered.knobs", "tp4_run.sh"])
def test_the_guard_refuses_a_staged_file_that_is_not_the_pinned_one(tmp_path, victim):
    box = _box(tmp_path, tamper=victim)
    r = _guard(tmp_path, box)
    assert r.returncode == 9 and "differs from bench/p67/staged.sha256" in r.stdout


# ----------------------------------------------------------------------------- p67_drive.sh -> tp4_drive.sh
def _drive(extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TP4_", "E4B_", "GNF4_"))}
    env.update({"E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "22", "E4B_RENT_RUN_DIR": "/tmp/x", "E4B_RENT_RUN_ID": "p67-test-1",
                "E4B_RENT_DEADLINE_EPOCH": "9999999999", "E4B_RENT_INSTANCE_ID": "1", "E4B_SHA": "a" * 40, "TP4_DRIVE_DRYRUN": "1"})
    env.update(extra or {})
    return subprocess.run([BASH, str(LANE / "p67_drive.sh")], capture_output=True, text=True, env=env, timeout=60)


def test_the_driver_tells_the_box_to_start_the_guard_with_the_registered_knobs():
    r = _drive()
    assert r.returncode == 0, r.stdout + r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("DRYRUN"))
    assert "bash p67_run.sh" in line
    for f in ("p67_run.sh", "registered.knobs", "staged.sha256", "tp4_run.sh", "tp4_arm.py"):
        assert f in line
    k = _knobs()
    for key in ("TP4_BOX", "TP4_FAMILIES", "TP4_STEPS", "TP4_P56", "TP4_P67", "TP4_PREREG", "GNF4_SHA"):
        assert f"{key}={k[key]}" in line, key


def test_the_driver_refuses_an_override_before_staging():
    r = _drive({"TP4_STEPS": "60"})
    assert r.returncode == 78 and "differs from the registered" in r.stdout


def test_tp4_drive_refuses_a_runner_it_did_not_stage():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TP4_", "E4B_", "GNF4_"))}
    env.update({"E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "22", "E4B_RENT_RUN_DIR": "/tmp/x", "E4B_RENT_RUN_ID": "r",
                "E4B_RENT_DEADLINE_EPOCH": "9999999999", "E4B_RENT_INSTANCE_ID": "1", "E4B_SHA": "a" * 40, "TP4_BOX": "C",
                "TP4_DRIVE_DRYRUN": "1", "TP4_RUNNER": "nowhere.sh"})
    r = subprocess.run([BASH, str(TP4 / "tp4_drive.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 78 and "not a staged file" in r.stdout


def test_tp4_drive_default_runner_is_unchanged():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TP4_", "E4B_", "GNF4_"))}
    env.update({"E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "22", "E4B_RENT_RUN_DIR": "/tmp/x", "E4B_RENT_RUN_ID": "r",
                "E4B_RENT_DEADLINE_EPOCH": "9999999999", "E4B_RENT_INSTANCE_ID": "1", "E4B_SHA": "a" * 40, "TP4_BOX": "C",
                "TP4_DRIVE_DRYRUN": "1"})
    r = subprocess.run([BASH, str(TP4 / "tp4_drive.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and "bash tp4_run.sh" in r.stdout and "p67_run.sh" not in r.stdout


def test_every_registered_knob_is_written_in_the_prereg():
    """registered.knobs is what the box enforces; the prereg is what was registered. They must be one text."""
    pre = (LANE / "P67-PREREG.md").read_text()
    for k, v in _knobs().items():
        assert f"{k}|{v}" in pre, f"{k}|{v} is enforced on the box but not written in P67-PREREG.md"


def test_the_staged_list_hashes_are_sha256():
    for line in (LANE / "staged.sha256").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            want = line.split()[0]
            assert re.fullmatch(r"[0-9a-f]{64}", want)
