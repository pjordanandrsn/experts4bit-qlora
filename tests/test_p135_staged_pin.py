# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P135's staged-file pin must match the repo, and the runner, driver, box, reducer and registration must agree
(bench/p135/PREREG-p135.md).

``bench/p135/p135_drive.sh`` refuses to run when a staged file's sha256 differs from ``bench/p135/staged.sha256``; this test
runs the same comparison in CI, where it costs nothing. It also pins:
- the driver's stage list, the runner's stage check and the pin naming the same files;
- the files P135 shares with FAM at FAM's pinned bytes (``bench/fam/staged.sha256``), so the prompt, the bake and the
  premise are the ones FAM's Mixtral reads ran;
- the model, the knobs and the server's size as the box, the runner and the registration state them;
- the rule's constants as the registration states them;
- every time-left check, plus 600 s of fetch-back, inside the guard.
"""
import hashlib
import importlib.util
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p135"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p135_run.sh": LANE / "p135_run.sh",
    "p135_speed.py": LANE / "p135_speed.py",
    "p135_reduce.py": LANE / "p135_reduce.py",
    "p115_quality.py": REPO / "bench" / "p115" / "p115_quality.py",
    "p110_box.py": REPO / "bench" / "p110" / "p110_box.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_fusion_modes.py": REPO / "tests" / "test_fusion_modes.py",
    "test_fused_glue_decode_graphs_gpu.py": REPO / "bench" / "p136" / "premise" / "test_fused_glue_decode_graphs_gpu.py",
}
SHARED_WITH_FAM = ("p115_quality.py", "p110_box.py", "p108_box.py", "p97_box.py", "k8_bake.py", "calib.json",
                   "test_fusion_modes.py")
RUN = (LANE / "p135_run.sh").read_text()
DRIVE = (LANE / "p135_drive.sh").read_text()
PREREG = (LANE / "PREREG-p135.md").read_text(encoding="utf-8")
GUARD_H = 2.0


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _load(name):
    if str(LANE) not in sys.path:
        sys.path.insert(0, str(LANE))
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_staged_file_matches_its_pin():
    names = [n for _, n in _entries()]
    assert sorted(names) == sorted(SOURCES) and len(names) == len(set(names))
    for want, name in _entries():
        assert hashlib.sha256(SOURCES[name].read_bytes()).hexdigest() == want, name


def test_the_files_shared_with_fam_are_at_fam_s_pinned_bytes():
    fam = dict((n, w) for w, n in _entries(REPO / "bench" / "fam" / "staged.sha256"))
    mine = dict((n, w) for w, n in _entries())
    for name in SHARED_WITH_FAM:
        assert mine[name] == fam[name], name


def test_the_driver_and_the_runner_stage_exactly_the_pinned_files():
    stage = re.search(r'^STAGE="([^"]+)"', DRIVE, re.M).group(1).split()
    assert sorted(p.rsplit("/", 1)[-1] for p in stage) == sorted(list(SOURCES) + ["staged.sha256"])
    listed = re.search(r"for f in (.+?); do\n", RUN, re.S).group(1).replace("\\\n", " ").split()
    assert sorted(listed) == sorted(list(SOURCES) + ["staged.sha256"]) or \
        sorted(listed) == sorted([n for n in SOURCES if n != "p135_run.sh"] + ["staged.sha256"])


def test_the_driver_keeps_its_literal_launcher_env_loop_and_dry_run():
    assert re.search(r"^for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID "
                     r"E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do$", DRIVE, re.M)
    assert "fam" not in DRIVE.replace("bench/fam/fam_drive.sh", "")        # every FAM name substituted


def test_the_box_runner_and_registration_agree_on_the_model_knobs_and_server():
    box = _load("p135_speed")
    assert f"MODEL={box.MODEL}; REV={box.REVISION}" in RUN
    assert box.MODEL in PREREG and box.REVISION in PREREG
    knobs = " ".join(f"{k}={v}" for k, v in box.KNOBS.items())
    assert knobs in RUN.replace(" \\\n  ", " ")
    assert "E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=2 E4B_PAGED_MAX_TOKENS_PER_SEQ=768" in RUN
    gnf4 = re.search(r"^GNF4_SHA=([0-9a-f]{40})$", RUN, re.M).group(1)
    # the lane is registered at 724ccc45 (v0.45.0, e4b CI's pin at registration); CI has since moved on (e4b#1768: v0.47.1)
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert gnf4 in PREREG
    assert gnf4 in ci or gnf4 == "724ccc454f006c1a46836e434e997f31f293747f", "e4b CI's grouped-nf4-gemm at registration"
    for k in box.OFF_KNOBS:
        assert k in RUN                                                    # unset before anything runs


def test_the_rule_s_constants_are_the_registration_s():
    red = _load("p135_reduce")
    assert (red.FAST, red.SLOW, red.NOISE) == (0.98, 1.02, 0.015)
    assert red.CENSUS == [0, 65, [32, 32], 32] and red.FOLDS == {"glue_fuse": 65, "glue_r2": 64, "router_epilogue": 32}
    for s in ("0.98", "1.02", "1.5 %", "0 / 65 / [32, 32] / 32", "161"):
        assert s in PREREG, s


def test_every_time_left_check_fits_the_guard():
    need = {k: int(v) for k, v in re.findall(r"\bNEED_(\w+)=(\d+)", RUN)}
    assert set(need) == {"FETCH", "BAKE", "SPEED"}
    for k, v in need.items():
        assert v + 600 <= GUARD_H * 3600, k
    assert f"guard {GUARD_H:g} h" in PREREG.lower()


def test_the_runner_and_driver_parse():
    for f in ("p135_run.sh", "p135_drive.sh"):
        subprocess.run(["bash", "-n", str(LANE / f)], check=True)


def test_the_driver_refuses_an_unregistered_run_id(tmp_path):
    case = re.search(r'case "\$RUN_ID" in\n\s*([^)]+)\) :;;', DRIVE).group(1)
    assert case.split("|") == ["p135-5090-1", "p135-5090-2", "p135-5090-3"]
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "22",
           "E4B_RENT_SSH_OPTS": "-o x=y", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_DEADLINE_EPOCH": "9999999999",
           "E4B_RENT_INSTANCE_ID": "1", "E4B_SHA": "a" * 40, "P135_DRIVE_DRYRUN": "1"}
    bad = subprocess.run(["bash", str(LANE / "p135_drive.sh")], env={**env, "E4B_RENT_RUN_ID": "p135-5090-4"},
                         capture_output=True, text=True)
    assert bad.returncode == 78 and "not a registered P135 id" in bad.stdout, (bad.returncode, bad.stdout, bad.stderr)
    ok = subprocess.run(["bash", str(LANE / "p135_drive.sh")], env={**env, "E4B_RENT_RUN_ID": "p135-5090-1"},
                        capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout.startswith("DRYRUN stage"), (ok.returncode, ok.stdout, ok.stderr)

