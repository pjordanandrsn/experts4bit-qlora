# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""SC2f Amendment 10, on the CPU: the {575, 600} W class, the recorded-clock rule (``recorded_clocks`` on a planted
store, and through ``admit``'s CLI), the qualify read and the post-proof records against the qualify reading, the
wrapper on a stubbed 600 W box, and the amendment's numbers against ``bench/sc2/sc2f-am10/census.json``."""
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SC2 = REPO / "bench" / "sc2"
AM10_DIR = SC2 / "sc2f-am10"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Q = _load("sc2_qualify", SC2 / "sc2_qualify.py")
CENSUS = _load("power_census", AM10_DIR / "power_census.py")
CC = _load("card_clock", REPO / "bench" / "common" / "card_clock.py")
PREREG = (SC2 / "SC2f-PREREG.md").read_text()
AM10 = PREREG[PREREG.index("### Amendment 10"):]
AM10_FLAT = re.sub(r"\s+", " ", AM10)
C = json.loads((AM10_DIR / "census.json").read_text())


def test_the_self_test_passes():
    assert Q.self_test() == 0


def test_the_constants_are_the_texts():
    assert Q.POWER_W == (575.0, 600.0) and "listed-power class is {575, 600} W (`POWER_W`)" in AM10_FLAT
    assert Q.CLOCK_FLOOR_MHZ == 2700 and "`card_clock.passes(summary, 2700, mode=\"clock\")`" in AM10_FLAT
    assert Q.RECORD_MIN_BUSY == 10 and ">= 10 busy samples" in AM10_FLAT
    assert Q.CLOCK_RECORDS == ("card_clock_burn.json", "card_clock.json")
    assert (Q.MAX_ATTEMPTS, Q.CEILING_USD_H, Q.MIN_DISK_GB, Q.STABLE_READS) == (3, 1.05, 320, 3)   # unchanged
    assert "the deadline 2026-10-28T14:12:34Z" in AM10_FLAT and "proof worst $1.18 per attempt, lane worst $12.42" in AM10_FLAT


# ---- the recorded-clock census on a planted store
def _run(store, day, run, machine, started, gpu="NVIDIA GeForce RTX 5090", logs=(), burns=()):
    d = store / day / run
    d.mkdir(parents=True)
    (d / "receipt.json").write_text(json.dumps({"instance_id": "1", "started_at": started,
                                                "environment": {"vast_machine_id": machine, "vast_gpu_name": gpu}}))
    for i, (mhz, n) in enumerate(logs):
        (d / f"gpuclk_arm{i}.txt").write_text("".join(f"{1792000000 + k} {mhz}, 13801, 60, 0x0000000000000000 | 1 1 1 | cpu 0\n"
                                                       for k in range(n)))
    for i, summ in enumerate(burns):
        sub = d / "qualify" / "box" if i else d
        sub.mkdir(parents=True, exist_ok=True)
        (sub / ("card_clock_burn.json" if i == 0 else "card_clock.json")).write_text(json.dumps(summ))
    return d


def _burn(mhz, n=74, legacy=False):
    s = CC.summarize([(mhz, 13801, 60, 0x4)] * n)
    if legacy:                                    # a summary written before TC1 amendment 83: no per-sample Reliability count
        s.pop("busy_reliability_samples")
    return s


def test_the_latest_receipt_decides(tmp_path):
    s = tmp_path / "store"
    _run(s, "2026-10-01", "a-1", "100", "2026-10-01T00:00:00Z", logs=[(2900, 30)])
    _run(s, "2026-10-05", "a-2", "100", "2026-10-05T00:00:00Z", logs=[(2600, 30)])
    _run(s, "2026-10-01", "b-1", "200", "2026-10-01T00:00:00Z", logs=[(2600, 30)])
    _run(s, "2026-10-05", "b-2", "200", "2026-10-05T00:00:00Z", logs=[(2900, 30)])
    c = Q.recorded_clocks(s)
    assert not c["100"]["ok"] and c["100"]["run"] == "2026-10-05/a-2"
    assert c["200"]["ok"] and c["200"]["run"] == "2026-10-05/b-2"


def test_every_record_in_the_latest_receipt_must_pass(tmp_path):
    s = tmp_path / "store"
    _run(s, "2026-10-05", "a-1", "100", "2026-10-05T00:00:00Z", logs=[(2900, 30), (2650, 30)])
    _run(s, "2026-10-05", "b-1", "200", "2026-10-05T00:00:00Z", logs=[(2900, 30)], burns=[_burn(2827), _burn(2647)])
    c = Q.recorded_clocks(s)
    assert not c["100"]["ok"] and "2650" in c["100"]["why"]
    assert not c["200"]["ok"] and "2647" in c["200"]["why"]       # the qualify burn (qualify/box/card_clock.json) counts


def test_a_legacy_burn_fails_and_a_thin_record_is_no_record(tmp_path):
    s = tmp_path / "store"
    _run(s, "2026-10-05", "a-1", "100", "2026-10-05T00:00:00Z", burns=[_burn(2900, legacy=True)])
    _run(s, "2026-10-05", "b-1", "200", "2026-10-05T00:00:00Z", logs=[(2500, 9)])
    _run(s, "2026-10-05", "c-1", "300", "2026-10-05T00:00:00Z", burns=[{"error": "RuntimeError: no CUDA", "refusal": "probe_error"}])
    c = Q.recorded_clocks(s)
    assert not c["100"]["ok"] and "legacy" in c["100"]["why"]
    assert "200" not in c and "300" not in c                      # 9 busy samples; a harness error summary


def test_an_unparseable_record_fails_and_a_non_5090_is_ignored(tmp_path):
    s = tmp_path / "store"
    d = _run(s, "2026-10-05", "a-1", "100", "2026-10-05T00:00:00Z", logs=[(2900, 30)])
    (d / "card_clock_burn.json").write_text("{not json")
    _run(s, "2026-10-05", "b-1", "200", "2026-10-05T00:00:00Z", gpu="RTX 4090", logs=[(2400, 30)])
    c = Q.recorded_clocks(s)
    assert not c["100"]["ok"] and "unparseable" in c["100"]["why"] and "200" not in c


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through a mode-000 directory")
def test_a_store_error_makes_the_census_unread(tmp_path):
    s = tmp_path / "store"
    d = _run(s, "2026-10-05", "a-1", "100", "2026-10-05T00:00:00Z", logs=[(2900, 30)])
    os.chmod(d.parent, 0)
    try:
        assert Q.recorded_clocks(s) is None
    finally:
        os.chmod(d.parent, 0o755)
    assert Q.recorded_clocks(tmp_path / "absent") == {}           # a missing store is an empty census, not an error


def _mutant(tmp_path, old, new):
    """sc2_qualify.py with one census line restored to a skip, to show the test below is not vacuous."""
    src = (SC2 / "sc2_qualify.py").read_text()
    assert src.count(old) == 1
    path = tmp_path / "sc2_qualify_mutant.py"
    path.write_text(src.replace(old, new))
    (tmp_path / "common").mkdir(exist_ok=True)
    m = _load("sc2_qualify_mutant", path)
    m._card_clock = Q._card_clock
    return m


def _hidden_fail(tmp_path, receipt_text=None, started="2026-10-05T00:00:00Z"):
    s = tmp_path / "store"
    _run(s, "2026-10-01", "a-1", "100", "2026-10-01T00:00:00Z", logs=[(2900, 30)])
    d = _run(s, "2026-10-05", "a-2", "100", started, logs=[(2600, 30)])
    if receipt_text is not None:
        (d / "receipt.json").write_text(receipt_text)
    return s, d


def test_an_unparseable_receipt_with_a_clock_record_makes_the_census_unread(tmp_path):
    s, d = _hidden_fail(tmp_path, receipt_text="{truncated")
    assert Q.recorded_clocks(s) is None                          # never the older passing record
    m = _mutant(tmp_path, "            except ValueError:\n                return None\n            if not isinstance(rj, dict):",
                "            except ValueError:\n                continue\n            if not isinstance(rj, dict):")
    assert m.recorded_clocks(s)["100"] == {"run": "2026-10-01/a-1", "ok": True, "why": ""}   # the skip would admit it
    (d / "receipt.json").write_text("[1, 2]")
    assert Q.recorded_clocks(s) is None                          # parses, but not an object
    (d / "receipt.json").unlink()
    assert Q.recorded_clocks(s)["100"]["run"] == "2026-10-01/a-1"  # no receipt: the run names no machine


@pytest.mark.parametrize("started", [None, "", "yesterday", "2026-10-05 00:00:00", "2026-13-05T00:00:00Z"])
def test_a_missing_or_unparseable_started_at_makes_the_census_unread(tmp_path, started):
    s, _ = _hidden_fail(tmp_path, started=started)
    assert Q.recorded_clocks(s) is None
    m = _mutant(tmp_path, "            if started is None:\n                return None\n",
                "            if started is None:\n                started = _utc('1970-01-01T00:00:00Z')\n")
    assert m.recorded_clocks(s)["100"]["ok"]                      # sorting it last would hide the fail


def test_started_at_parses_whole_and_fractional_seconds():
    assert Q._utc("2026-10-05T01:02:03Z") is not None and Q._utc("2026-10-05T01:02:03.456Z") is not None
    assert Q._utc("2026-10-05T01:02:03") is None and Q._utc("2026-10-05T25:02:03Z") is None


def _reads(*offers, n=3):
    return [[{"machine_id": m, "offer_id": f"o{m}{w}", "usd_h": 0.9, "disk_gb": 400, "listed_power_w": w} for m, w in offers]] * n


def test_admit_on_the_cli_excludes_a_recorded_fail(tmp_path):
    s = tmp_path / "store"
    _run(s, "2026-10-05", "a-1", "100", "2026-10-05T00:00:00Z", logs=[(2647, 30)])
    _run(s, "2026-10-05", "b-1", "200", "2026-10-05T00:00:00Z", logs=[(2940, 30)])
    reads = tmp_path / "reads.json"
    reads.write_text(json.dumps(_reads(("100", 600.0), ("200", 600.0), ("300", 575.0))))
    cr = tmp_path / "cr.json"
    cr.write_text(json.dumps({"100": 9, "300": 5}))
    out = subprocess.run([sys.executable, str(SC2 / "sc2_qualify.py"), "admit", "--store", str(s), "--lane", "sc2f",
                          "--reads", str(reads), "--clean-runs", str(cr)], capture_output=True, text=True)
    rec = json.loads(out.stdout.split(" ", 1)[1])
    assert out.returncode == 0 and rec["machine"] == "200", out.stdout + out.stderr
    assert rec["walk"] == ["52780", "145701", "200", "300"] and "100" in rec["clock_refused"]


def test_the_walk_without_the_rule_would_admit_the_recorded_fail():
    """The rule is not vacuous: an empty census (nothing recorded) admits the machine whose record fails."""
    reads = _reads(("100", 600.0))
    assert Q.admit([], reads, False, clean_runs={"100": 9}, clocks={})["machine"] == "100"
    assert Q.admit([], reads, False, clean_runs={"100": 9}, clocks={"100": {"ok": False, "run": "d/r", "why": "x"}})["verdict"] == "NONE"


def test_clocks_has_no_default():
    with pytest.raises(TypeError):
        Q.admit([], _reads(("100", 600.0)), False)


# ---- the card checks
@pytest.mark.parametrize("w,want", [("575.00", "PASS"), ("600.00", "PASS"), ("600", "PASS"), ("500.00", "REFUSE"),
                                    ("599.9", "REFUSE"), ("600.1", "REFUSE"), ("574.99", "REFUSE")])
def test_the_enforced_read(w, want):
    assert Q.power(f"{w} W, 600.00 W, 600.00 W")["verdict"] == want


FX = "GPU 0: NVIDIA GeForce RTX 5090\npower.limit,clocks.max.sm {} W, 3105 MHz\n"


@pytest.mark.parametrize("fw,hf,qw,want", [(600.0, 600.0, 600.0, "PASS"), (575.0, 575.0, 575.0, "PASS"),
                                           (600.0, 600.0, 575.0, "REFUSE"), (575.0, 600.0, 600.0, "REFUSE"),
                                           (500.0, 500.0, 500.0, "REFUSE"), (600.0, 600.0, None, "UNREAD")])
def test_the_post_proof_records_against_the_qualify_reading(fw, hf, qw, want):
    q = None if qw is None else {"enforced_w": qw}
    assert Q.judge_proof(FX.format(f"{fw:.2f}"), {"gpu_power_limit_w": hf}, q)["verdict"] == want


# ---- the wrapper on a stubbed box (test_sc2f_am7's harness)
AM7T = _load("test_sc2f_am7", REPO / "tests" / "test_sc2f_am7.py")


def test_a_600_w_card_qualifies_and_its_proof_passes(tmp_path):
    out, drove, run = AM7T._wrap(tmp_path, power="600.00 W, 600.00 W, 600.00 W", proof_w="600.00", host_facts=600.0)
    assert out.returncode == 0 and drove, out.stdout + out.stderr
    pj = json.loads((run / "qualify" / "proof_judge.json").read_text())
    assert pj["verdict"] == "PASS" and pj["qualify_enforced_w"] == 600.0


def test_a_proof_whose_records_differ_from_the_qualify_reading_is_a_host_fault(tmp_path):
    out, drove, run = AM7T._wrap(tmp_path, power="600.00 W, 600.00 W, 600.00 W", proof_w="575.00", host_facts=575.0)
    assert out.returncode == 18 and drove, out.stdout + out.stderr
    assert "differs from the qualify reading" in json.dumps(json.loads((run / "qualify" / "proof_judge.json").read_text()))


def test_a_500_w_card_is_still_refused_before_anything(tmp_path):
    out, drove, _ = AM7T._wrap(tmp_path, power="500.00 W, 600.00 W, 600.00 W")
    assert out.returncode == 17 and not drove


def test_the_wrapper_always_passes_the_qualify_reading():
    drive = (SC2 / "sc2_qualify_drive.sh").read_text()
    assert '--qualify-power "$QD/power.json"' in drive and drive.count("judge-proof") >= 1


# ---- the amendment's numbers are the census
def test_the_market_half_is_re_derived_from_the_committed_reads():
    reads = json.loads((AM10_DIR / "screen_reads.json").read_text())
    assert CENSUS.market(reads) == C["market"] and C["market"]["reads"] == 228 == len(reads)


def test_the_prereg_numbers_are_the_census():
    by = C["market"]["by_listed_w"]
    for w, txt in (("575", "575 W 14 / 10 / 3, 7 points"), ("600", "600 W 17 / 13 / 7, 61 points"),
                   ("500", "500 W 13 / 10 / 6, 49 points"), ("450", "450 W 7 / 4 / 3, 85 points")):
        b = by[w]
        assert f"{w} W {b['machines']} / {b['qualified']} / {b['with_admit_point']}, {b['admit_points']} points" == txt
        assert txt in AM10_FLAT
    assert "575 W listings reached an admit point 7 times (3 machines)" in AM10_FLAT
    assert "600 W listings reached one 61 times (7 machines)" in AM10_FLAT
    t = C["training_by_enforced_w"]
    for w, txt in (("600", "at 600 W, 13 of 19 runs, and 4 of 8 machines on every run, held >= 2,700 MHz (median 2,940)"),
                   ("575", "at 575 W, 33 of 41 runs and 9 of 14 machines (median 2,865)"),
                   ("500", "at 500 W, 5 of 7 runs and 3 of 5 machines (median 2,857)")):
        r = t[w]
        nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", txt)[1:]]
        assert [r["runs_at_floor"], r["runs"], r["machines_every_run_at_floor"], r["machines"]] == [n for n in nums if n != 2700][:4]
        assert round(r["run_median_median"]) == nums[-1] and txt in AM10_FLAT
    burns = {b["power_w"]: b["sm_median"] for b in C["burns"]}
    assert burns == {"575": 2827.0, "500": 2647.0, "470": 2482.0, "430": 2493.5, "400": 1972.0} and "600" not in burns
    assert "575 W 2,827 MHz (pass); 500 W 2,647; 470 W 2,482; 430 W 2,493.5; 400 W 1,972" in AM10_FLAT
    lve = C["listed_vs_enforced"]
    assert (lve["machines"], lve["agree"], len(lve["differ"])) == (30, 22, 8) and "of 30 machines in both the reads and the store, 22 agree" in AM10_FLAT
    assert sorted(v["enforced"][0] for v in lve["differ"].values() if v["listed"] == [600.0]) == [500.0, 520.0]
    assert C["recorded_clocks"] == {"machines": 48, "pass": 24} and "48 RTX 5090 machines carry a loaded clock record, and 24 pass" in AM10_FLAT
