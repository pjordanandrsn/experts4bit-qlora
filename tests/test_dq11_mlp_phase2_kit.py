"""DQ11 MLP-adapter BUILD Phase 2's paid kit (Amendment 5 section 8): the box and drive scripts, the launch manifest
template and its binding, and the static fit. Text and arithmetic only; nothing here rents, installs or loads."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
D = REPO / "bench" / "dq11"
sys.path.insert(0, str(D))
import dq11_mlp_phase2_run as r2  # noqa: E402

BOX = (D / "dq11_mlp_phase2_box.sh").read_text()
DRIVE = (D / "dq11_mlp_phase2_drive.sh").read_text()
TEMPLATE = json.loads((D / "dq11_mlp_phase2_manifest_template.json").read_text())
CENSUS = json.loads((D / "dq11_l_census_manifest_template.json").read_text())


def test_both_scripts_parse():
    for f in ("dq11_mlp_phase2_box.sh", "dq11_mlp_phase2_drive.sh"):
        subprocess.run(["bash", "-n", str(D / f)], check=True)


def test_the_box_checks_the_closure_and_fits_the_guard_before_it_installs():
    order = ["sha256sum -c science.sha256", "sha256sum -c dq11-mlp-phase2-stage.sha256", "sha256sum -c inputs.sha256",
             "--part fit", "create-venv", "dq11_wheel_retry.py", "dq11_prepare.py"]
    at = [BOX.index(s) for s in order]
    assert at == sorted(at), dict(zip(order, at))
    assert "finish 41" in BOX[BOX.index("--part fit"):BOX.index("create-venv")]
    assert "13b55990" not in BOX                                     # the census's pinned science head never gates Phase 2
    assert "DQ11_GPU_CLAIM_CONFIRMED=1" in BOX and "HF_HUB_OFFLINE=1" in BOX[BOX.index("dq11_prepare.py"):]
    assert "--target 5090" in BOX and "--target a2000" not in BOX


def test_the_box_runs_every_part_in_order_and_drift_never_fails_it():
    stage, b = BOX.index("--part stage-check"), BOX.index("--part b-read")
    timed = BOX.index("for arm in " + " ".join(r2.TIMED) + ";")
    done, drift, reduce = BOX.index("TIMED_DONE"), BOX.index("for arm in D0 D1;"), BOX.index("phase reduce")
    assert stage < b < timed < done < drift < reduce
    block = BOX[drift:reduce]
    assert "run_phase" in block and re.search(r"\bphase \"\$arm\"", block) is None      # never the fatal wrapper
    assert "drift: not read" in block and "sleep 150" in BOX[done:drift]


def test_the_drive_stages_exactly_the_instrument_the_box_checks():
    inst = re.search(r'^INSTRUMENT="([^"]+)"', DRIVE, re.M).group(1).split()
    for f in inst:
        assert (D / f).is_file(), f
    for f in ("dq11_mlp_phase2_run.py", "dq11_mlp_phase2.py", "dq11_mlp_phase1.py", "dq11_mlp_phase1_run.py",
              "dq11_mlp_phase2_box.sh", "dq11_census_transport.py", "dq11_wheel_retry.py"):
        assert f in inst, f
    assert "TC1_RUNNER=dq11_mlp_phase2_box.sh" in DRIVE and "dq11-mlp-phase2-stage.sha256" in DRIVE
    assert "PHASE2_REFUSAL: dirty launch source" in DRIVE and "PHASE2_REFUSAL: unmerged launch source" in DRIVE
    assert "TC1_L_CENSUS" not in DRIVE                               # the census's own staging and size gates stay off


def test_the_template_is_the_census_shape_with_phase_2s_caps():
    assert set(TEMPLATE) == set(CENSUS)                              # the launcher refuses unknown top-level keys
    assert TEMPLATE["usd_per_hour"] == 0.85 and TEMPLATE["wallclock_h"] * 3600 == r2.GUARD_S == 7200
    assert TEMPLATE["est_usd"] == pytest.approx(TEMPLATE["usd_per_hour"] * TEMPLATE["wallclock_h"])
    assert TEMPLATE["est_usd"] + TEMPLATE["download_gb"] * TEMPLATE["max_download_usd_per_gb"] <= \
        TEMPLATE["lane_record"]["phase2"]["all_in_cap_usd"] <= 15
    assert TEMPLATE["preregistration"] == "bench/dq11/DQ11-MLP-BUILD-AMENDMENT-5.md"
    assert TEMPLATE["gpu"] == "RTX 5090" and TEMPLATE["run_id"] == "__DQ11_PHASE2_RUN_ID__"
    for g in TEMPLATE["guards"]:
        if g["repo"] == "e4b":
            assert g["pattern"] in (REPO / g["path"]).read_text(), g


def test_binding_pins_by_inclusion_and_is_write_once_outside_the_tree(tmp_path):
    out = tmp_path / "m.json"
    kw = dict(e4b_head="a" * 40, adertha_head="b" * 40, machine=1234, offers=[11, 12], driver="590.44.01",
              assets=tmp_path, output=out)
    m = r2.bind_manifest("dq11-mlp-phase2-1", **kw)
    assert json.loads(out.read_text()) == m
    assert m["vast_pin_machine"] == "1234" and m["vast_pin_offers"] == ["11", "12"] and m["vast_required_driver"] == "590.44.01"
    assert m["command"].endswith("bash bench/dq11/dq11_mlp_phase2_drive.sh") and "DQ11_REQUIRED_DRIVER=590.44.01" in m["command"]
    assert m["lane_record"]["min_disk_gb"] == r2.BOX_MIN_DISK_GB == 123                    # the launcher's preflight disk check
    assert m["heads"] == {"e4b": "a" * 40, "adertha": "b" * 40} and set(m) == set(CENSUS) | {"heads", "command",
                                                                                              "vast_pin_machine", "vast_pin_offers",
                                                                                              "vast_required_driver"}
    with pytest.raises(ValueError, match="write-once"):
        r2.bind_manifest("dq11-mlp-phase2-1", **kw)
    bad = [dict(kw, output=tmp_path / "1.json", driver="575.1.2"), dict(kw, output=tmp_path / "2.json", offers=[]),
           dict(kw, output=tmp_path / "3.json", offers=[11, 11]), dict(kw, output=tmp_path / "4.json", machine="x1"),
           dict(kw, output=tmp_path / "5.json", e4b_head="main"), dict(kw, output=REPO / "bench" / "m.json")]
    for b in bad:
        with pytest.raises(ValueError, match="PHASE2_REFUSAL"):
            r2.bind_manifest("dq11-mlp-phase2-1", **b)
    with pytest.raises(ValueError, match="dq11-mlp-phase2-N"):
        r2.bind_manifest("dq11-l-census-9", **dict(kw, output=tmp_path / "6.json"))


def test_the_fit_refuses_while_the_b_read_seconds_are_unknown(monkeypatch):
    monkeypatch.setattr(r2, "B_READ_S", None)
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        r2.box_need()


def test_the_static_fit_on_the_registered_parts(monkeypatch):
    monkeypatch.setattr(r2, "B_READ_S", 1800)
    parts = r2.registered_parts()
    assert len(parts) == 4 + 1 + len(r2.TIMED) + len(r2.QUALITY_ARMS) + len(r2.DRIFT) + 1
    fit = r2.static_fit(parts)
    assert fit["fits"] and fit["need_s"] == (r2.BOOT_S + sum(parts)) * r2.FIT_FACTOR
    assert r2.box_need() == int(sum(r2.registered_parts(drift=False)) * r2.FIT_FACTOR + 0.999999)
    big = 7200 / r2.FIT_FACTOR - r2.BOOT_S - sum(r2.registered_parts(b_read_s=0)) + 1
    assert not r2.static_fit(r2.registered_parts(b_read_s=big))["fits"]


def test_the_fit_part_prints_the_box_need_without_torch(monkeypatch, capsys):
    monkeypatch.setattr(r2, "B_READ_S", 1800)
    assert r2.main(["--target", "5090", "--part", "fit", "--directory", ".", "--cache", ".", "--out", "unused"]) == 0
    assert int(capsys.readouterr().out.strip()) == r2.box_need()


def test_the_declared_disk_floor_is_the_box_scripts_own():
    """bind_manifest declares lane_record.min_disk_gb: the box script's own free-disk floor in decimal GB, rounded
    up. The box reads `df -Pk`, whose blocks are 1024 bytes."""
    import math

    assert "df -Pk" in BOX
    floor_kb = int(re.search(r'\[ "\$free_kb" -ge ([0-9]+) \]', BOX).group(1))
    assert math.ceil(floor_kb * 1024 / 1e9) == r2.BOX_MIN_DISK_GB
