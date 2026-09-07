"""bench/p41 drivers (P41 R1, e4b#433): the lane script's plan is the pre-registration's grid, in its order, with its numbers;
the controller driver refuses to run blind and stages exactly the harness + lane script + admission rules; the admission rules
turn a receipt that fails a validity rule into a VOID row and the p41c probe into a footprint row. No box, no network, no compute."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "bench" / "p41" / "p41_run.sh"
DRIVE = REPO / "bench" / "p41" / "p41_drive.sh"
ADMIT = REPO / "bench" / "p41" / "p41_admit.py"
sys.path.insert(0, str(ADMIT.parent))
import p41_admit  # noqa: E402


def _bash(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", *args], capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=120
    )


def test_both_scripts_parse_and_are_executable():
    for s in (RUN, DRIVE):
        assert _bash("-n", str(s)).returncode == 0, s
        assert os.access(s, os.X_OK), s


def _plan(env: dict | None = None) -> list[dict]:
    out = _bash(str(RUN), "--plan", env=env)
    assert out.returncode == 0, out.stderr
    rows = []
    for line in out.stdout.splitlines():
        if line.startswith("PLAN TOTAL"):
            rows.append({"total": dict(kv.split("=", 1) for kv in line.split()[2:])})
        elif line.startswith("PLAN "):
            parts = line.split()
            rows.append(
                {"n": int(parts[1]), "fam": parts[2], "tag": parts[3], **dict(kv.split("=", 1) for kv in parts[4:])}
            )
    return rows


def test_r1_plan_is_the_granite_grid_in_the_preregistered_order():
    rows = _plan()
    total = rows[-1]["total"]
    arms = rows[:-1]
    assert total["arms"] == "19" and len(arms) == 19, "9 cells x (fused + reference) + the p41c probe"
    assert total["prereg"] == "p41/P41-PREREG.md" and total["steps"] == "60"
    # order: seq ascending, rank ascending, the (512, 8) anchor first, fused (PRIMARY) then reference per cell, the probe last
    cells = [(int(a["seq"]), int(a["r"]), a["arm"]) for a in arms[:18]]
    expected = [(s, r, arm) for s in (512, 1024, 2048) for r in (8, 16, 32) for arm in ("fused", "reference")]
    assert cells == expected
    assert (
        arms[0]["tag"] == "fused_attn4_s512_r8"
        and arms[0]["anchor"] == "yes"
        and arms[1]["tag"] == "reference_attn4_s512_r8"
    )
    assert (
        arms[18]["tag"] == "fused_attn4_s4096_r8_probe" and arms[18]["probe"] == "p41c" and arms[18]["seq"] == "4096"
    )
    # alpha = 2r; expected trainable = (r/8) x the tp2 anchor count 49,807,360 (PREREG "Expected trainable counts")
    for a in arms:
        assert int(a["alpha"]) == 2 * int(a["r"])
        assert int(a["expect_trainable"]) == 49_807_360 * int(a["r"]) // 8
        assert a["attn4_census"] == "128" and a["offload"] == "0"
        assert int(a["alarm"]) > 0
    # the planning curve: reference arms get longer alarms than fused; longer sequences get longer alarms
    by = {a["tag"]: int(a["alarm"]) for a in arms}
    assert by["reference_attn4_s512_r8"] > by["fused_attn4_s512_r8"]
    assert by["fused_attn4_s2048_r8"] > by["fused_attn4_s1024_r8"] > by["fused_attn4_s512_r8"]
    assert by["fused_attn4_s512_r8"] == by["fused_attn4_s512_r32"], (
        "rank does not enter the planning curve (P40's rule: s512 anchor x seq exponent)"
    )


def test_plan_follows_the_family_table_for_a_later_run():
    rows = _plan({"P41_FAMILIES": "olmoe", "P41_RUN_ID": "p41-r2-olmoe"})
    arms = rows[:-1]
    assert len(arms) == 19 and all(a["fam"] == "olmoe" for a in arms)
    assert int(arms[0]["expect_trainable"]) == 60_817_408 and arms[0]["attn4_census"] == "64"
    bad = _bash(str(RUN), "--plan", env={"P41_FAMILIES": "gptoss"})
    assert bad.returncode != 0 and "unknown family gptoss" in bad.stdout, (
        "gpt-oss is REFUSED by the pre-registration: no row, no arms"
    )


def test_driver_refuses_without_the_launcher_environment_and_stages_only_the_harness():
    env = {
        k: ""
        for k in (
            "E4B_RENT_SSH_HOST",
            "E4B_RENT_SSH_PORT",
            "E4B_RENT_RUN_DIR",
            "E4B_RENT_RUN_ID",
            "E4B_RENT_DEADLINE_EPOCH",
            "E4B_RENT_USD_PER_HOUR",
            "E4B_RENT_EST_USD",
            "E4B_RENT_INSTANCE_ID",
        )
    }
    out = _bash(str(DRIVE), env=env)
    assert out.returncode == 78 and "E4B_RENT_SSH_HOST is not set" in out.stdout
    env = {
        "E4B_RENT_SSH_HOST": "ssh5.vast.ai",
        "E4B_RENT_SSH_PORT": "12345",
        "E4B_RENT_RUN_DIR": "/tmp/run",
        "E4B_RENT_RUN_ID": "p41-r1-granite",
        "E4B_RENT_DEADLINE_EPOCH": "1788800000",
        "E4B_RENT_WALLCLOCK_S": "14400.0",
        "E4B_RENT_INSTANCE_ID": "50059999",
        "E4B_RENT_PROVIDER": "vast:verified-secure",
        "P41_DRIVE_DRYRUN": "1",
    }
    out = _bash(str(DRIVE), env=env)
    assert out.returncode == 78 and "E4B_RENT_USD_PER_HOUR is not set" in out.stdout, (
        "no rate/estimate from the launcher → STOP-4 would be blind → refuse"
    )
    env.update({"E4B_RENT_USD_PER_HOUR": "0.66", "E4B_RENT_EST_USD": "2.64"})
    out = _bash(str(DRIVE), env=env)
    assert out.returncode == 78 and "P41_PLAN_EST_USD is not set" in out.stdout, (
        "the approval line is the guard, not STOP-4's estimate (PREREG amendment, #467) → refuse without the planning estimate"
    )
    env.update({"P41_PLAN_EST_USD": "1.87"})
    out = _bash(str(DRIVE), env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = out.stdout.splitlines()
    assert any(
        ln.startswith("DRYRUN stage:")
        and "bench/tp3/tp3_arm.py" in ln
        and "bench/p41/p41_run.sh" in ln
        and "bench/p41/p41_admit.py" in ln
        and "root@ssh5.vast.ai:/root/p41/" in ln
        for ln in lines
    )
    assert any(
        ln.startswith("DRYRUN start:")
        and "P41_RUN_ID=p41-r1-granite" in ln
        and "P41_DEADLINE_EPOCH=1788800000" in ln
        and "P41_USD_PER_HOUR=0.66" in ln
        and "P41_PLAN_EST_USD=1.87" in ln
        and "P41_APPROVAL_EST_USD=2.64" in ln
        and "P41_WALLCLOCK_S=14400.0" in ln
        and "P41_INSTANCE_ID=50059999" in ln
        and "P41_PROVIDER=vast:verified-secure" in ln
        for ln in lines
    )
    assert any(ln.startswith("DRYRUN fetch:") and "/tmp/run/p41/" in ln and "excluding adapters" in ln for ln in lines)
    assert not any("ssh-add" in ln or "PRIVATE" in ln for ln in lines)


def test_lane_script_never_calls_the_provider_and_names_no_credential():
    text = RUN.read_text() + DRIVE.read_text() + ADMIT.read_text()
    # the drivers drive a box the launcher already rented; they never touch the provider API, a key file or a token
    for forbidden in (
        "vast_provider",
        "VAST_API_KEY",
        "secrets.env",
        "rent.py --provider",
        "/instances/",
        "vast-destroy",
        "api/v0",
        "api/v1",
        "xoxb-",
        "sk-ant-",
    ):
        assert forbidden not in text, forbidden
    run = RUN.read_text()
    assert "TP_DONE" in run and all(f"STOP-{n}" in run for n in (1, 2, 3, 4, 5))
    # the helpers come from a COMMIT, never a tag, each file pinned by sha256 and verified before anything runs (Warden MEDIUM on #466)
    assert "refs/tags" not in run and 'bash -c "curl' not in run
    import re

    default_ref = re.search(r"E4B_SRC_REF=\$\{P41_E4B_SRC_REF:-([0-9a-f]+)\}", run).group(1)
    assert len(default_ref) == 40
    pins = dict(kv.split("=") for kv in re.search(r'HELPER_SHAS="([^"]+)"', run).group(1).split())
    assert set(pins) == {
        "bench/flagship-matrix/drivers/n9_datasets.py",
        "bench/flagship-matrix/ds_manifest.json",
        "bench/train-anchor/train_anchor.py",
        "bench/train-anchor/train_anchor_gate.py",
    }
    assert all(len(v) == 64 for v in pins.values())
    # the registered fixture sha is asserted, not narrated (PREREG "Fixture")
    assert "76fb9036de80f3bb495fe4c8894159fcb1d399d2437293e012e264d81949f791" in run
    # the registered box class is refused if absent; the exact machine is recorded
    assert 'GPU_CLASS=${P41_GPU_CLASS:-"RTX 5090"}' in run and "BOX_REFUSED" in run and "box.json" in run


def test_helper_archive_fetch_survives_the_registered_images_without_curl():
    """R1 attempt 3 proved the registered PyTorch image lacks curl; the lane must not pass preflight then die here."""
    run = RUN.read_text()
    start = run.index('for tool in curl wget python3; do')
    end = run.index('GOT_ARCHIVE_SHA=', start)
    fetch = run[start:end]
    assert fetch.index('command -v curl') < fetch.index('command -v wget') < fetch.index('command -v python3')
    assert "urllib.request.urlretrieve" in fetch
    assert 'SRC FETCH TOOL $FETCH_TOOL rc=$rc' in fetch
    assert '[ "$rc" -eq 0 ] || { echo "SRC FETCH FAIL rc=$rc' in fetch, (
        "a failed downloader's rc is retained for the receipt and refusal rather than overwritten by the tar gate"
    )
    assert 'GOT_ARCHIVE_SHA=$(sha256sum "$W/e4b-src.tar.gz"' in run
    assert "SOURCE ARCHIVE MISMATCH" in run
    assert run.index('tar xzf "$W/e4b-src.tar.gz"') > run.index("pip(e4b-source)"), (
        "the exact archive is verified and installed before its byte-identical helpers are extracted"
    )


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _fake_driver(tmp_path: Path, remote: Path) -> subprocess.CompletedProcess:
    fake_bin = tmp_path / "driver-bin"
    fake_bin.mkdir(exist_ok=True)
    _write_executable(
        fake_bin / "ssh",
        """#!/bin/bash
cmd=""
for arg in "$@"; do cmd=$arg; done
case "$cmd" in
  *"test -f /root/p41/TP_DONE"*) test -f "$FAKE_REMOTE/TP_DONE" ;;
  *"tail -n 1 /root/p41/summary.txt"*) tail -n 1 "$FAKE_REMOTE/summary.txt" 2>/dev/null || true ;;
  *) exit 0 ;;
esac
""",
    )
    _write_executable(fake_bin / "scp", "#!/bin/bash\nexit 0\n")
    _write_executable(
        fake_bin / "rsync",
        """#!/bin/bash
dest=""
for arg in "$@"; do dest=$arg; done
mkdir -p "$dest"
cp -R "$FAKE_REMOTE"/. "$dest"/
""",
    )
    run_dir = tmp_path / "driver-run"
    env = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_REMOTE": str(remote),
        "E4B_RENT_SSH_HOST": "fake.vast.invalid",
        "E4B_RENT_SSH_PORT": "12345",
        "E4B_RENT_RUN_DIR": str(run_dir),
        "E4B_RENT_RUN_ID": "p41-no-gpu-regression",
        "E4B_RENT_DEADLINE_EPOCH": str(int(time.time()) + 60),
        "E4B_RENT_WALLCLOCK_S": "60",
        "E4B_RENT_USD_PER_HOUR": "0.66",
        "E4B_RENT_EST_USD": "2.64",
        "E4B_RENT_INSTANCE_ID": "50000000",
        "E4B_RENT_PROVIDER": "vast:verified-secure",
        "P41_PLAN_EST_USD": "1.87",
        "P41_POLL_S": "1",
    }
    return _bash(str(DRIVE), env=env)


def test_exact_tripwire_failure_is_terminal_nonzero_and_propagates_through_driver(tmp_path: Path):
    work = tmp_path / "remote-p41"
    work.mkdir()
    (work / "tp3_arm.py").write_text("# staged\n")
    (work / "p41_admit.py").write_text("# staged\n")
    archive = tmp_path / "detector-source.tar.gz"
    archive.write_bytes(b"no-gpu source archive fixture")
    fake_bin = tmp_path / "lane-bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "curl",
        """#!/bin/bash
out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then shift; out=$1; fi
  shift
done
cp "$P41_TEST_ARCHIVE" "$out"
""",
    )
    _write_executable(
        fake_bin / "python",
        """#!/bin/bash
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pip" ]; then exit 0; fi
if [ "${1:-}" = "-" ]; then
  cat >/dev/null
  echo "ImportError: cannot import name 'detect_attention_projections' from 'experts4bit_qlora.lora'" >&2
  exit 1
fi
exec python3 "$@"
""",
    )
    env = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "P41_TEST_ARCHIVE": str(archive),
        "P41_WORKDIR": str(work),
        "P41_USD_PER_HOUR": "0.66",
        "P41_PLAN_EST_USD": "1.87",
        "P41_APPROVAL_EST_USD": "2.64",
        "P41_DEADLINE_EPOCH": str(int(time.time()) + 600),
        "P41_E4B_ARCHIVE_SHA256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    lane = _bash(str(RUN), env=env)
    assert lane.returncode == 9
    assert "TRIPWIRE FAIL (e4b)" in lane.stdout
    assert "ImportError: cannot import name 'detect_attention_projections'" in lane.stderr
    assert (work / "P41_EXIT_CODE").read_text() == "9\n"
    assert (work / "P41_EXIT_CODE").stat().st_mtime_ns <= (work / "TP_DONE").stat().st_mtime_ns
    assert not (work / "P41_SUCCESS").exists()

    driven = _fake_driver(tmp_path, work)
    assert driven.returncode == 9, driven.stdout + driven.stderr
    assert "lane failed with remote rc=9" in driven.stdout
    assert "admission:" not in driven.stdout and "[p41_drive] done" not in driven.stdout


@pytest.mark.parametrize(
    ("rc_text", "success", "expected_rc", "message"),
    [
        (None, False, 24, "no P41_EXIT_CODE"),
        ("not-a-number\n", False, 24, "malformed P41_EXIT_CODE"),
        ("0\n1\n", False, 24, "expected one line"),
        ("999\n", False, 24, "malformed P41_EXIT_CODE"),
        ("7\n", False, 7, "remote rc=7"),
        ("0\n", False, 24, "without P41_SUCCESS"),
        ("0\n", True, 0, "done"),
    ],
)
def test_driver_requires_exact_remote_exit_and_success_marker(
    tmp_path: Path, rc_text: str | None, success: bool, expected_rc: int, message: str
):
    remote = tmp_path / "remote-result"
    remote.mkdir()
    (remote / "TP_DONE").touch()
    (remote / "summary.txt").write_text("")
    if rc_text is not None:
        (remote / "P41_EXIT_CODE").write_text(rc_text)
    if success:
        (remote / "P41_SUCCESS").touch()
    out = _fake_driver(tmp_path, remote)
    assert out.returncode == expected_rc, out.stdout + out.stderr
    assert message in out.stdout
    if expected_rc:
        assert "admission:" not in out.stdout and "[p41_drive] done" not in out.stdout


def test_source_install_and_prereg_amendment_bind_the_detector_commit():
    run = RUN.read_text()
    prereg = (REPO / "bench" / "p41" / "P41-PREREG.md").read_text()
    commit = "5dad2a7fe7020ced76df0ddd0cfc548627223496"
    archive_sha = "8c2562d23a213ae5773ed8f01170b7807e1cf6551f1f54fa5f694d7ae573e479"
    assert f"E4B_SRC_REF=${{P41_E4B_SRC_REF:-{commit}}}" in run
    assert f"E4B_ARCHIVE_SHA256=${{P41_E4B_ARCHIVE_SHA256:-{archive_sha}}}" in run
    assert '--force-reinstall "$W/e4b-src.tar.gz"' in run
    assert "e4b {e.__version__} (source {source})" in run and "e4b {e.__version__} (PyPI)" not in run
    assert "P41_EXIT_CODE" in run and "P41_SUCCESS" in run
    assert commit in prereg and archive_sha in prereg and "terminal-status amendment" in prereg


def test_stop4_projects_the_remaining_cells_at_the_planning_curve_not_the_alarm_sum(capsys):
    """CEO HIGH-1 on #466: at arm 1 of R1 (≈ 900 s of setup elapsed, all 19 arms remaining) the projection at the registered
    planning curve must NOT trip STOP-4 for the approval line (rate $0.66/h, estimate $1.98); the alarm sum would have."""
    rows = _plan()
    total = rows[-1]["total"]
    plan_sum, alarm_sum = int(total["plan_sum_s"]), int(total["alarm_sum_s"])
    assert sum(int(r["plan"]) for r in rows[:-1]) == plan_sum and plan_sum < alarm_sum
    assert all(int(r["plan"]) < int(r["alarm"]) for r in rows[:-1])
    assert 8000 < plan_sum < 11000, (
        plan_sum
    )  # ≈ 6,500 s of steps + 19 × 120 s (the PR body's arithmetic), not 27,595 s of alarms
    rc = p41_admit.main(
        ["budget", "--elapsed", "900", "--remaining", str(plan_sum), "--rate", "0.66", "--est", "1.87"]
    )
    assert rc == 1 and "STOP-4 ok" in capsys.readouterr().out
    rc = p41_admit.main(
        ["budget", "--elapsed", "900", "--remaining", str(alarm_sum), "--rate", "0.66", "--est", "1.87"]
    )
    assert rc == 0, "the alarm sum would have fired STOP-4 before the first arm -- that is why the lane never uses it"
    rc = p41_admit.main(
        ["budget", "--elapsed", "900", "--remaining", str(plan_sum), "--rate", str(3 * 0.66), "--est", "1.87"]
    )
    assert rc == 0 and "STOP-4 DUE" in capsys.readouterr().out
    run = RUN.read_text()
    assert "remaining_plan_sum" in run and "remaining_alarm_sum" not in run
    # the amended R1 planning estimate ($1.87 at $0.66/h; PREREG amendment, #467) is STOP-4's estimate: it trips above 1.5 × 1.87 = $2.81,
    # not above the approval line's 1.5 × 2.64 = $3.96 — a projection between the two is DUE on the estimate and would not have been on the line
    rem = str(int(3.00 / 0.66 * 3600))
    rc_plan = p41_admit.main(["budget", "--elapsed", "0", "--remaining", rem, "--rate", "0.66", "--est", "1.87"])
    rc_line = p41_admit.main(["budget", "--elapsed", "0", "--remaining", rem, "--rate", "0.66", "--est", "2.64"])
    assert (rc_plan, rc_line) == (0, 1) and "$2.81" in capsys.readouterr().out
    assert "EST=${P41_PLAN_EST_USD:-0}" in run and "APPROVAL_EST=${P41_APPROVAL_EST_USD:-0}" in run
    assert "plan_est_usd" in run and "approval_est_usd" in run, "both numbers in the RUN line and stop_state.json"
    assert "GUARD_T0" in run and "now - GUARD_T0" in run, "STOP-5 counts from the guard's start (CEO LOW-1)"


# ---- the admission rules (bench/p41/p41_admit.py): the harness's receipt vs the pre-registration's validity rules
GRANITE = dict(steps=60, tokens_sha="a" * 64, expect_trainable=49_807_360, n_layers=32, attn4_census=128)


def _ok_receipt(**over) -> dict:
    rec = {
        "framework": "e4b",
        "fam": "granite",
        "arm": "fused",
        "tag": "fused_attn4_s512_r8",
        "status": "ok",
        "steps": 60,
        "seq": 512,
        "r": 8,
        "alpha": 16,
        "step_ms": [641.0] * 60,
        "tokens": {"sha256": "a" * 64, "n_train": 1200},
        "trainable_params": 49_807_360,
        "trainable_mismatch": None,
        "n_attn4": 128,
        "structural_expected_n_attn4": 128,
        "n_patched": 32,
        "kernel_calls_per_step_min": 64,
        "engagement_banners": ["[e4b] fast train: fused_grouped_lora on 32 layers"],
        "C1_bit_exact": True,
        "C1_experts_changed": 0,
        "s_per_step_median_11plus": 0.641,
        "peak_vram_gb": 2.511,
        "eval_loss_final": 0.31,
        "offload": False,
        "grad_ckpt": "hf-nonreentrant",
        "tokens_per_step": [512] * 60,
        "prereg": "p41/P41-PREREG.md",
    }
    rec.update(over)
    return rec


def _admit(tmp_path: Path, rec: dict, arm: str = "fused", **over) -> tuple[int, dict]:
    p = tmp_path / f"granite_e4b_{rec['tag']}.json"
    p.write_text(json.dumps(rec))
    kw = {**GRANITE, **over}
    rc = p41_admit.main(
        [
            "admit",
            str(p),
            "--steps",
            str(kw["steps"]),
            "--tokens-sha",
            kw["tokens_sha"],
            "--expect-trainable",
            str(kw["expect_trainable"]),
            "--n-layers",
            str(kw["n_layers"]),
            "--attn4-census",
            str(kw["attn4_census"]),
            "--arm",
            arm,
        ]
    )
    return rc, json.loads(p.read_text())


def test_admission_admits_a_receipt_that_meets_every_registered_rule(tmp_path: Path):
    rc, rec = _admit(tmp_path, _ok_receipt())
    assert rc == 0 and rec["admitted"] is True and rec["status"] == "ok"
    rc, rec = _admit(
        tmp_path,
        _ok_receipt(
            arm="reference",
            tag="reference_attn4_s512_r8",
            n_patched=0,
            kernel_calls_per_step_min=0,
            engagement_banners=[],
        ),
        arm="reference",
    )
    assert rc == 0 and rec["admitted"] is True, (
        "the fused-only rules (n_patched, kernel calls, banner) do not apply to the reference arm"
    )


@pytest.mark.parametrize(
    "over,cls",
    [
        ({"step_ms": [641.0] * 59}, "steps"),
        ({"tokens": {"sha256": "b" * 64}}, "tokens"),
        (
            {"trainable_params": 49_807_361, "trainable_mismatch": {"expected": 49_807_360, "got": 49_807_361}},
            "trainable",
        ),
        ({"n_attn4": 127}, "attn4"),
        ({"n_patched": 31}, "engagement"),
        ({"kernel_calls_per_step_min": 63}, "engagement"),
        ({"engagement_banners": ["NO '[e4b] fast train' banner on stdout (the census below decides)"]}, "engagement"),
        ({"C1_bit_exact": False, "C1_experts_changed": 3}, "c1"),
    ],
)
def test_admission_rewrites_a_rule_failure_as_a_void_row(tmp_path: Path, over: dict, cls: str):
    rc, rec = _admit(tmp_path, _ok_receipt(**over))
    assert rc == 1 and rec["status"] == "void" and rec["status_harness"] == "ok" and rec["admitted"] is False
    assert rec["void_class"] == cls and cls in rec["void_reason"]
    assert "s_per_step_median_11plus" in rec, "the measurement stays in the row; it just never enters a reading"


def test_admission_leaves_the_harness_own_rows_and_names_a_harness_void_class(tmp_path: Path):
    rc, rec = _admit(tmp_path, _ok_receipt(status="oom", reason="CUDA out of memory"))
    assert rc == 2 and rec["status"] == "oom" and rec["admitted"] is False
    rc, rec = _admit(tmp_path, _ok_receipt(status="void_trainable", reason="non-adapter trainables"))
    assert rc == 1 and rec["status"] == "void_trainable" and rec["void_class"] == "trainable"
    assert (
        p41_admit.main(
            [
                "admit",
                str(tmp_path / "absent.json"),
                "--steps",
                "60",
                "--tokens-sha",
                "x",
                "--expect-trainable",
                "1",
                "--n-layers",
                "1",
                "--attn4-census",
                "1",
                "--arm",
                "fused",
            ]
        )
        == 3
    )


def test_footprint_row_states_fit_never_speed(tmp_path: Path):
    probe = tmp_path / "granite_e4b_fused_attn4_s4096_r8_probe.json"
    probe.write_text(
        json.dumps(
            _ok_receipt(tag="fused_attn4_s4096_r8_probe", seq=4096, peak_vram_gb=19.42, tokens_per_step=[4096] * 60)
        )
    )
    vram = tmp_path / "vram.txt"
    vram.write_text(
        "1788700000 21000, 99, 410.2\n1788700001 25123, 100, 420.0\n1788700002 24000, 98, 400.0\nbad line\n"
    )
    out = tmp_path / "granite_p41c_footprint.json"
    assert (
        p41_admit.main(
            [
                "footprint",
                str(probe),
                "--fam",
                "granite",
                "--seq",
                "4096",
                "--r",
                "8",
                "--vram",
                str(vram),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    row = json.loads(out.read_text())
    assert (
        row["fit_status"] == "OK"
        and row["peak_vram_gb_allocator"] == 19.42
        and row["peak_vram_gb_nvsmi"] == round(25123 / 1024, 3)
    )
    assert (
        row["seq_probe"] == 4096
        and row["batch"] == 1
        and row["offload_design"] == "none"
        and row["tokens_per_step"] == 4096
    )
    assert (
        "never speed" in row["feeds"]
        and "s_per_step_median" not in json.dumps(row)
        and "joules" not in json.dumps(row)
    )
    probe.write_text(
        json.dumps(_ok_receipt(tag="fused_attn4_s4096_r8_probe", seq=4096, status="oom", reason="CUDA out of memory"))
    )
    assert p41_admit.main(["footprint", str(probe), "--fam", "granite", "--seq", "4096", "--out", str(out)]) == 0
    row = json.loads(out.read_text())
    assert row["fit_status"] == "OOM" and row["peak_vram_gb_nvsmi"] is None and "out of memory" in row["reason"]
