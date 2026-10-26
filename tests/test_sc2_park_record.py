# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The SC2f/SC2h/SC2i park record of 2026-10-26: the three PREREGs carry it, the lane spend adds up, and the committed
market reads after SC2f Amendment 10's file are the 39 the record names."""
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
SC2 = REPO / "bench" / "sc2"
FLAT = {k: re.sub(r"\s+", " ", (SC2 / f"{k}-PREREG.md").read_text()) for k in ("SC2f", "SC2h", "SC2i")}
ATTEMPTS = {"sc2f-prove-11": ("rc 25, VOID", 0.319), "sc2f-prove-12": ("rc 13", 0.121), "sc2f-prove-13": ("rc 18, a host fault", 0.305)}


def test_each_lane_records_the_park():
    rec = FLAT["SC2f"][FLAT["SC2f"].index("### Record (2026-10-26): SC2f PARKED"):]
    for run, (rc, cost) in ATTEMPTS.items():
        assert re.search(rf"`{run}` \| \d+ \| [\d /W]+ \| \*\*{re.escape(rc)}\*\* \|.*?\| \${cost:.3f} \|", rec), run
    assert round(sum(c for _, c in ATTEMPTS.values()), 3) == 0.745 and "Lane spend: **$0.745** against the registered $14.42 worst" in rec
    assert "**before a lane's first create, run its full judge chain once at $0 on a real committed record.**" in rec
    for lane in ("SC2h", "SC2i"):
        assert f"### Record (2026-10-26): {lane} PARKED with SC2f" in FLAT[lane] and f"{lane}'s spend: $0." in FLAT[lane]


def test_each_prereg_opens_with_the_parked_status_above_its_registered_sentence():
    for lane in ("SC2f", "SC2h", "SC2i"):
        text = (SC2 / f"{lane}-PREREG.md").read_text()
        status = "**Status: PARKED on 2026-10-26.** See the record at the end of this file.\n\n**Registered**"
        assert status in text and text.index(status) < 1000, lane      # its own paragraph, at the head, before the registered sentence


def test_the_market_record_closes_the_screen():
    reads = json.loads((SC2 / "sc2f-park" / "screen_reads_after_am10.json").read_text())
    assert len(reads) == 39 and "the 39 screen reads after Amendment 10's committed file" in FLAT["SC2f"]
    assert all({"machine_id", "offer_id", "usd_h", "disk_gb", "listed_power_w"} <= set(o) for rd in reads for o in rd)
    assert any("storage_usd_gb_month" in o for rd in reads for o in rd)       # screen7's rows carry the storage rate
