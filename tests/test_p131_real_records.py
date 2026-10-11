# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P131's decode check (``bench/p131/p131_reduce.py``) on REAL records first, synthetic departures second.

The records are K19's dispatch counters (``K19_DISPATCH_SEEN``) as #1625's A2000 serve smoke wrote them
(``tests/fixtures/smoke1625/``, byte-identical to the receipt, digest checked). They are the counter the box's decode
check records per arm, written by the route itself: at knob 0 the 1,024-row calls take the WIDE one-launch table, which
is what the lane's first decode check (``chained|lean|gt256`` x layers) would have called touched.
- In each cell's own process, the knob-0 forward is the m0 record and the cell's forward the arm's: untouched at 0 and
  0r, touched at 32 and 64;
- across the serve path (``serve_k19_seen``, two requests x two layers = 4 calls above 256 rows), the same;
- the reducer, end to end, on records whose decode check is the real one.
Then departures from the real m0 record, each a way the knob could reach decode or m0 could be off.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "smoke1625" / "pass2_summary.json"
SHA256 = "1f4960b1d6b05f774c7fb8c0c922f1b85aab84b333263b3e113c1404b32ab9bb"
FAMILIES = ("qwen3_moe", "granitemoe", "mixtral")
KNOBS = ("0", "0r", "32", "64")
TOUCHED = {"0": False, "0r": False, "32": True, "64": True}
SERVE_CALLS = 4                  # two requests' prefill, one K19 call a layer over two layers, above 256 rows
FORWARD_CALLS = 2                # one forced forward over two layers


def _reducer():
    spec = importlib.util.spec_from_file_location("p131_reduce", ROOT / "bench" / "p131" / "p131_reduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _reducer()


def _cells() -> dict:
    data = json.loads(FIX.read_bytes())
    return {(c["family"], str(c["knob"])): c for c in data["cells"]}


def test_the_fixture_is_the_smoke_receipt():
    raw = FIX.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256
    assert b"\r" not in raw
    assert set(_cells()) == {(f, k) for f in FAMILIES for k in KNOBS}


@pytest.mark.parametrize("family", FAMILIES)
def test_real_m0_takes_the_wide_table_not_the_chained_one(family):
    """The fact the first decode check had wrong: at this size knob 0 is one_launch, so 'chained x layers' would read
    every arm DECODE_TOUCHED. The new check takes m0's table as found."""
    c = _cells()
    for k in KNOBS:
        k0 = R.gt256(c[(family, k)]["seen_knob0"])
        assert k0 == {"one_launch|lean|gt256": FORWARD_CALLS}
        assert not any(key.startswith("chained|") for key in k0)
        assert not R.decode_m0_bad(c[(family, k)]["seen_knob0"], FORWARD_CALLS)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("knob", KNOBS)
def test_real_in_process_records_read_touched_iff_the_knob_is_set(family, knob):
    cell = _cells()[(family, knob)]
    assert R.decode_touched(cell["seen_own"], cell["seen_knob0"]) is TOUCHED[knob]


@pytest.mark.parametrize("family", FAMILIES)
def test_real_serve_records_read_touched_iff_the_knob_is_set(family):
    c = _cells()
    m0 = c[(family, "0")]["serve_k19_seen"]
    assert not R.decode_m0_bad(m0, SERVE_CALLS)
    for k in KNOBS:
        assert R.decode_touched(c[(family, k)]["serve_k19_seen"], m0) is TOUCHED[k], k
    # qwen3_moe's serve record also holds its decode steps' small calls (one_launch|lean|le256 x24): not read
    if family == "qwen3_moe":
        assert c[(family, "32")]["serve_k19_seen"].get("one_launch|lean|le256")


def _reduce(tmp_path, decode, L=FORWARD_CALLS):
    for n, r in R._run_recs(L=L, decode=decode).items():
        (tmp_path / f"proc_{n}.json").write_text(json.dumps(r), encoding="utf-8")
    (tmp_path / "summary.txt").write_text("premise ok\n", encoding="utf-8")
    return R.reduce(tmp_path, "a" * 40)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_reducer_licenses_on_real_untouched_records(family, tmp_path):
    c = _cells()
    real = {"m0": c[(family, "0")]["seen_own"], "m32": c[(family, "0r")]["seen_own"],
            "m64": c[(family, "0r")]["seen_own"]}
    o = _reduce(tmp_path, real)
    assert o["verdict"] == "M32_LICENSED M64_LICENSED DEFAULT_m64", o.get("reasons")


@pytest.mark.parametrize("family", FAMILIES)
def test_the_reducer_reads_a_real_split_record_as_decode_touched(family, tmp_path):
    c = _cells()
    real = {"m0": c[(family, "32")]["seen_knob0"], "m32": c[(family, "32")]["seen_own"],
            "m64": c[(family, "0r")]["seen_own"]}
    o = _reduce(tmp_path, real)
    assert o["arms"]["m32"]["verdict"] == "DECODE_TOUCHED"
    assert o["arms"]["m64"]["verdict"] == "LICENSED"


# synthetic departures from the real m0 record ----------------------------------------------------------------------

def _m0():
    return dict(_cells()[("qwen3_moe", "0")]["serve_k19_seen"])


def test_departure_a_split_at_m0_is_bad():
    k = _m0()
    k["split|lean|gt256"] = 1
    assert R.decode_m0_bad(k, SERVE_CALLS + 1)


def test_departure_a_gather_dispatch_at_m0_is_bad():
    k = {key.replace("|lean|", "|gather|"): n for key, n in _m0().items()}
    assert R.decode_m0_bad(k, SERVE_CALLS)


def test_departure_a_missing_call_at_m0_is_bad():
    k = _m0()
    k["one_launch|lean|gt256"] -= 1
    assert R.decode_m0_bad(k, SERVE_CALLS)


def test_departure_another_table_than_m0s_is_touched():
    m0 = _m0()
    arm = {key.replace("one_launch|", "chained|") if key.endswith("|gt256") else key: n for key, n in m0.items()}
    assert R.decode_touched(arm, m0)


def test_departure_an_extra_call_above_256_rows_is_touched():
    m0, arm = _m0(), _m0()
    arm["one_launch|lean|gt256"] += 1
    assert R.decode_touched(arm, m0)


def test_departure_a_split_beside_m0s_table_is_touched():
    m0, arm = _m0(), _m0()
    arm["split|lean|gt256"] = 1
    assert R.decode_touched(arm, m0)


def test_departure_small_calls_alone_are_not_read():
    m0, arm = _m0(), _m0()
    arm["one_launch|lean|le256"] += 8
    assert not R.decode_touched(arm, m0)
