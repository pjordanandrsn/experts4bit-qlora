"""TC1 amendment 78's kit caps: the offer screen and the launch template refuse the same offers, and the lane fits the 15-dollar rail.

The kit's market watch admits an offer only through ``bench/tc1/pos78_kit.py``'s ``offer_why``. The launcher drops any offer that bills
download above the template's ``max_download_usd_per_gb``. Both read one ceiling, so an offer at 0.021 USD/GB is refused in both places.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TC1 = REPO / "bench" / "tc1"
TEMPLATE = json.loads((TC1 / "pos78_manifest_template.json").read_text())
PREREG = (TC1 / "TC1-PREREG.md").read_text()
A78 = PREREG[PREREG.index("### Amendment 78 "):]
_spec = importlib.util.spec_from_file_location("pos78_kit", TC1 / "pos78_kit.py")
kit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kit)


def test_the_screen_refuses_an_offer_above_the_download_ceiling():
    assert kit.DL_USD_GB == 0.02
    assert "download 0.021 USD/GB is over 0.02" in kit.offer_why(24564, 0.67, 0.021)
    assert kit.offer_why(24564, 0.67, 0.0201) != ""
    assert kit.offer_why(24564, 0.67, None) != ""                      # an unread price refuses
    assert kit.offer_why(24564, 0.67, 0.02) == "" and kit.offer_why(24564, 0.67, 0.003) == ""


def test_the_screen_refuses_a_48_gb_card_a_dear_hour_and_two_gpus():
    assert "24 GiB" in kit.offer_why(49140, 0.67, 0.01)                   # a 48 GB "RTX 4090" by its listing
    assert kit.offer_why(23999, 0.67, 0.01) != "" and kit.offer_why(None, 0.67, 0.01) != ""
    assert kit.offer_why(24564, 1.01, 0.01) != "" and kit.offer_why(24564, None, 0.01) != ""
    assert kit.offer_why(24564, 0.67, 0.01, num_gpus=2) != ""
    assert kit.offer_why(24000, 1.00, 0.02) == "" and kit.offer_why(24600, 1.00, 0.02) == ""


def test_the_template_carries_the_screens_caps():
    """The launcher refuses an offer above the template's download ceiling, so it must be the screen's, and the other caps with it."""
    assert TEMPLATE["max_download_usd_per_gb"] == kit.DL_USD_GB
    assert TEMPLATE["usd_per_hour"] == kit.RATE_USD_H and TEMPLATE["wallclock_h"] == kit.GUARD_H
    assert TEMPLATE["download_gb"] == kit.DOWNLOAD_GB and TEMPLATE["storage"] == TEMPLATE["vast_disk_gb"] == kit.DISK_GB
    assert TEMPLATE["ssh_ready_s"] == kit.SSH_READY_S == 420
    assert TEMPLATE["gpu"] == "RTX 4090" and TEMPLATE["est_usd"] <= TEMPLATE["usd_per_hour"] * TEMPLATE["wallclock_h"]
    assert TEMPLATE["avoid_vast_machine_receipts"] == []                  # filled at launch with every earlier read draw's receipt
    cmd = TEMPLATE["command"]
    for want in ("GNF4_SHA=efdda943d5a310343ed7fef2e23d1cbf2df27e51", "TC1_GPU_CLASS=4090", "TC1_FAMILIES=qwen3pos78fit",
                 "TC1_STEPS=60", "bash bench/tc1/tc1_drive.sh"):
        assert want in cmd, want
    rec = TEMPLATE["lane_record"]
    assert rec["worst_usd"] == kit.box_worst() and rec["lane_worst_usd"] == kit.lane_worst()


def test_every_template_guard_names_text_that_exists():
    for g in TEMPLATE["guards"]:
        if g["repo"] == "e4b":
            assert g["pattern"] in (REPO / g["path"]).read_text(), g


def test_the_worst_cases_fit_the_rail():
    assert kit.NONSTART_S == 1440 and kit.box_worst() == 3.50 and kit.nonstart_worst() == 0.43
    assert kit.lane_worst() == 13.94 <= 15 and kit.box_worst() <= 15
    assert kit.CAP_STARTED == 3 and kit.RETRY_IDS == 8


def test_the_screens_card_band_sits_inside_the_boxs():
    run = (TC1 / "tc1_run.sh").read_text()
    lo, hi = (int(re.search(rf"(?:^|; )POS78_MIB_{k}=(\d+)", run, re.MULTILINE).group(1)) for k in ("MIN", "MAX"))
    assert lo <= kit.GPU_RAM_MIB[0] and kit.GPU_RAM_MIB[1] <= hi, (lo, hi, kit.GPU_RAM_MIB)


def test_the_amendment_states_the_same_terms():
    for want in ("$0.02/GB", "**$3.50**", "**$0.43**", "**$13.94**", "bench/tc1/pos78_kit.py", "bench/tc1/pos78_manifest_template.json", "$0.021/GB"):
        assert want in A78, want
    assert "$17.02" not in A78 and "$14.78" not in A78
