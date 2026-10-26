# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""SC2f Amendment 11 (SC2h Amendment 6, SC2i Amendment 5), on the CPU: one $1.25/h ceiling for the three lanes, the
replay of which reads it admits (from the committed reads and census snapshot), and each lane's restated worst cases
recomputed from its own file's formula."""
import collections
import importlib.util
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
SC2 = REPO / "bench" / "sc2"
_spec = importlib.util.spec_from_file_location("sc2_qualify", SC2 / "sc2_qualify.py")
Q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Q)
FLAT = {k: re.sub(r"\s+", " ", (SC2 / f"{k}-PREREG.md").read_text()) for k in ("SC2f", "SC2h", "SC2i")}
AM11 = FLAT["SC2f"][FLAT["SC2f"].index("### Amendment 11"):]
REPLAY = json.loads((SC2 / "sc2f-am11" / "ceiling_replay.json").read_text())


def _reads(usd):
    return [[{"machine_id": "152938", "offer_id": "x", "usd_h": usd, "disk_gb": 400, "listed_power_w": 600.0}]] * 3


def test_the_self_test_passes():
    assert Q.self_test() == 0


def test_one_ceiling_of_1_25():
    assert Q.CEILING_USD_H == 1.25 and "moves from $1.05 to **$1.25**" in AM11
    assert Q.admit([], _reads(1.25), False, clocks={}).get("machine") == "152938"
    assert Q.admit([], _reads(1.26), False, clocks={})["verdict"] == "NONE"


def test_the_replay_is_re_derived_from_the_committed_reads():
    reads = json.loads((REPO / REPLAY["reads_file"]).read_text())
    assert len(reads) == REPLAY["reads"] == 228
    clocks = {m: {"ok": ok, "run": "", "why": ""} for m, ok in REPLAY["clock_ok"].items()}
    assert (len(clocks), sum(REPLAY["clock_ok"].values())) == (49, 25) and "(49 machines, 25 pass)" in AM11
    saved = Q.CEILING_USD_H
    try:
        for ceiling, want in REPLAY["by_ceiling"].items():
            Q.CEILING_USD_H = float(ceiling)
            pts = collections.Counter()
            for i in range(Q.STABLE_READS, len(reads) + 1):
                r = Q.admit(REPLAY["attempts"], reads[:i], False, clocks=clocks)
                if r["verdict"] == "ADMIT":
                    pts[r["machine"]] += 1
            assert {"admit_reads": sum(pts.values()), "machines": dict(sorted(pts.items()))} == want
    finally:
        Q.CEILING_USD_H = saved
    lo, hi = REPLAY["by_ceiling"]["1.05"], REPLAY["by_ceiling"]["1.25"]
    assert f"at $1.05, **{lo['admit_reads']}** reads ADMIT ({len(lo['machines'])} machines)" in AM11
    assert f"at $1.25, **{hi['admit_reads']}** reads ADMIT ({len(hi['machines'])} machines)" in AM11
    assert "55337" not in hi["machines"] and hi["machines"]["152938"] == 24 and "152938 (24 reads)" in AM11


def _c(x):
    return round(x + 1e-9, 2)                        # cents, half up


def test_each_lanes_worst_by_its_own_formula():
    c = Q.CEILING_USD_H
    # SC2f (Amendment 7): guard hours x ceiling, rounded, plus Granite ~$0.10 and the burn's torch ~$0.03; reading +$0.73 + $0.03
    f_proof, f_read = _c(c + 0.10 + 0.03), _c(_c(3.5 * c) + 0.73 + 0.03)
    assert (f_proof, f_read, _c(3 * f_proof + 2 * f_read)) == (1.38, 5.14, 14.42)
    # SC2h: two proofs, one reading (4.0 h), one re-run counted as a reading
    h_proof, h_read = _c(c + 0.10), _c(4.0 * c + 0.73)
    assert (h_proof, h_read, _c(2 * h_proof + 2 * h_read)) == (1.35, 5.73, 14.16)
    # SC2i: as SC2h's with a 4.5 h reading, summed unrounded (its $13.21 at $1.05 is 2 x 1.15 + 2 x 5.455)
    assert _c(2 * 1.15 + 2 * (4.5 * 1.05 + 0.73)) == 13.21
    i_proof, i_read_raw = _c(c + 0.10), 4.5 * c + 0.73
    assert (i_proof, _c(i_read_raw), _c(2 * i_proof + 2 * i_read_raw)) == (1.35, 6.36, 15.41)
    assert max(f_proof, f_read, h_proof, h_read, i_proof, _c(i_read_raw)) <= 15          # the per-run bar
    # with SC2h Am. 5 / SC2i Am. 4's 3 proof attempts and the burn (~$0.03 on each run), each in its own rounding convention
    burn_proof = _c(c + 0.10 + 0.03)
    assert _c(3 * burn_proof + 2 * _c(4.0 * c + 0.73 + 0.03)) == 15.66
    assert _c(3 * burn_proof + 2 * (4.5 * c + 0.73 + 0.03)) == 16.91
    assert "the two lane totals would be $15.66 and $16.91" in AM11
    for txt in ("a proof $1.18 -> **$1.38**", "the reading $4.44 -> **$5.14**", "$12.42 -> **$14.42**",
                "a proof $1.15 -> **$1.35**", "$4.93 -> **$5.73**", "$12.16 -> **$14.16**", "$5.46 -> **$6.36**",
                "$13.21 -> **$15.41**", "SC2i's lane total is above $15, by $0.41", "the largest is SC2i's reading, at $6.36"):
        assert txt in AM11, txt


def test_sc2h_and_sc2i_record_their_own_restatement():
    assert "### Amendment 6 (registered before any SC2h run): the price ceiling is $1.25" in FLAT["SC2h"]
    assert "2 x $1.35 + 2 x $5.73 = **$14.16** (was $12.16)" in FLAT["SC2h"]
    assert "### Amendment 5 (registered before any SC2i run): the price ceiling is $1.25" in FLAT["SC2i"]
    assert "2 x $6.355 = **$15.41** (was $13.21" in FLAT["SC2i"] and "above $15 by $0.41" in FLAT["SC2i"]
