"""bench/p41 drivers (P41 R1, e4b#433): the lane script's plan is the pre-registration's grid, in its order, with its numbers;
the controller driver refuses to run blind and stages exactly the harness + lane script. No box, no network, no compute."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "bench" / "p41" / "p41_run.sh"
DRIVE = REPO / "bench" / "p41" / "p41_drive.sh"


def _bash(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", *args], capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=120)


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
            rows.append({"n": int(parts[1]), "fam": parts[2], "tag": parts[3], **dict(kv.split("=", 1) for kv in parts[4:])})
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
    assert arms[0]["tag"] == "fused_attn4_s512_r8" and arms[0]["anchor"] == "yes" and arms[1]["tag"] == "reference_attn4_s512_r8"
    assert arms[18]["tag"] == "fused_attn4_s4096_r8_probe" and arms[18]["probe"] == "p41c" and arms[18]["seq"] == "4096"
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
    assert by["fused_attn4_s512_r8"] == by["fused_attn4_s512_r32"], "rank does not enter the planning curve (P40's rule: s512 anchor x seq exponent)"


def test_plan_follows_the_family_table_for_a_later_run():
    rows = _plan({"P41_FAMILIES": "olmoe", "P41_RUN_ID": "p41-r2-olmoe"})
    arms = rows[:-1]
    assert len(arms) == 19 and all(a["fam"] == "olmoe" for a in arms)
    assert int(arms[0]["expect_trainable"]) == 60_817_408 and arms[0]["attn4_census"] == "64"
    bad = _bash(str(RUN), "--plan", env={"P41_FAMILIES": "gptoss"})
    assert bad.returncode != 0 and "unknown family gptoss" in bad.stdout, "gpt-oss is REFUSED by the pre-registration: no row, no arms"


def test_driver_refuses_without_the_launcher_environment_and_stages_only_the_harness():
    env = {k: "" for k in ("E4B_RENT_SSH_HOST", "E4B_RENT_SSH_PORT", "E4B_RENT_RUN_DIR", "E4B_RENT_RUN_ID", "E4B_RENT_DEADLINE_EPOCH")}
    out = _bash(str(DRIVE), env=env)
    assert out.returncode == 78 and "E4B_RENT_SSH_HOST is not set" in out.stdout
    env = {"E4B_RENT_SSH_HOST": "ssh5.vast.ai", "E4B_RENT_SSH_PORT": "12345", "E4B_RENT_RUN_DIR": "/tmp/run", "E4B_RENT_RUN_ID": "p41-r1-granite",
           "E4B_RENT_DEADLINE_EPOCH": "1788800000", "E4B_RENT_INSTANCE_ID": "1", "P41_DRIVE_DRYRUN": "1", "P41_RATE_USD_H": "0.60", "P41_EST_USD": "1.98"}
    out = _bash(str(DRIVE), env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = out.stdout.splitlines()
    assert any(l.startswith("DRYRUN stage:") and "bench/tp3/tp3_arm.py" in l and "bench/p41/p41_run.sh" in l and "root@ssh5.vast.ai:/root/p41/" in l for l in lines)
    assert any(l.startswith("DRYRUN start:") and "P41_RUN_ID=p41-r1-granite" in l and "P41_DEADLINE_EPOCH=1788800000" in l and "P41_USD_PER_HOUR=0.60" in l and "P41_EST_USD=1.98" in l for l in lines)
    assert any(l.startswith("DRYRUN fetch:") and "/tmp/run/p41/" in l and "excluding adapters" in l for l in lines)
    assert not any("ssh-add" in l or "PRIVATE" in l for l in lines)


def test_lane_script_never_calls_the_provider_and_names_no_credential():
    text = RUN.read_text() + DRIVE.read_text()
    # the drivers drive a box the launcher already rented; they never touch the provider API, a key file or a token
    for forbidden in ("vast_provider", "VAST_API_KEY", "secrets.env", "rent.py --provider", "/instances/", "vast-destroy", "api/v0", "api/v1", "xoxb-", "sk-ant-"):
        assert forbidden not in text, forbidden
    assert "TP_DONE" in RUN.read_text() and "STOP-5" in RUN.read_text() and "STOP-1" in RUN.read_text() and "STOP-4" in RUN.read_text()
