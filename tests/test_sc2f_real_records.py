# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC2f's FLIPS gate (``bench/sc2/sc2f_reduce.py``, Amendment 1) on REAL records first, synthetic departures second.

Real records:
- ``tests/fixtures/smoke1625/pass2_summary.json``: #1625's A2000 serve smoke, byte-identical to its receipt (digest
  checked). Each cell's ``routes`` is ``serve_paged.prefill_routes()`` as the served process reported it, the block FLIPS
  reads under ``prefill_routes``, at the release's prefill defaults. The smoke served no HTTP, so ``levers.prefill_glue``
  is what ``/health`` reports it from: ``serve_paged.resolve_prefill_glue`` on the family. Its 1,024-row calls on 4
  experts took the WIDE one-launch table on the lean dispatch: the pre-amendment set ``{"chained|lean|gt256"}`` refuses
  them for the table alone.
- SC2e's committed start ``/health`` on the RTX 5090 (``bench/h2h-2026-10-02/sc2e/receipts/sc2e-5090-1/``, digests
  checked), Qwen3-30B-A3B at 64 slots. With buckets up to 64 (s64a, SC2f's server shape) the startup captures make 384 K19
  calls above 256 rows; with buckets up to 16 (s16, s64c) 240. Bucket 64's decode capture adds 144 = 48 layers x 3
  forwards above 256 rows beside the prefill graph's: calls the pre-amendment set had no place for. Those records predate
  the dispatch counter, so they fix the count, not the table.
- ``tests/fixtures/smokeflips/summary.json``: both of SC2f's arms on the A2000 (record-only smoke, digest checked), one
  fresh process each. Each cell's ``health`` is the two blocks FLIPS reads, built by the /health handler's own
  expressions, after the startup-shaped calls an SC2f server makes. The model is the smoke's qwen3_moe at top-4: the
  serve path's 2,048-row chunks take the chained table, and a 512-row forward (bucket 64's count) takes the one-launch
  table. ``on`` and ``off`` each pass their own arm with nothing added.
The departures start from those real records.
"""
import copy
import hashlib
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "smoke1625" / "pass2_summary.json"
FIX_SHA256 = "1f4960b1d6b05f774c7fb8c0c922f1b85aab84b333263b3e113c1404b32ab9bb"
SC2E = REPO / "bench" / "h2h-2026-10-02" / "sc2e" / "receipts" / "sc2e-5090-1" / "sc2"
SC2E_SHA256 = {
    "health_e4b_s16_d1_start.json": "03e93f1a77adc11f7ad191114ed1cf2e947f28f31a7705fc0a39ee8ba90f568a",
    "health_e4b_s16_d2_start.json": "f3b8544a5d659aa4eb1d21388a6712cda61801d767bdd8079ef2abfb8f3f1fef",
    "health_e4b_s64a_d1_start.json": "bde1659ae84868773bf137f7b5bb8f70ed6cd7394e3646169b9dbaab0fab2c70",
    "health_e4b_s64a_d2_start.json": "c5e91edf3f966c8407546f1d440478a594de27d5652637c7f70c6f6ac5a72b62",
    "health_e4b_s64c_d1_start.json": "1c2d6f7f7ad68d758132d1742f2b759dc861850d925860f693f5ec63fd4d9e35",
    "health_e4b_s64c_d2_start.json": "c70ae99af72e3df4aca67b69b2cb0544ef3010c7bd22fab67dc8c05267e4f35c",
}
FLIPS_FIX = REPO / "tests" / "fixtures" / "smokeflips" / "summary.json"
FLIPS_SHA256 = "1eeb6e02670cb63e1d7c28c3b0e8c2f34116575e09d62a7a41435f267166b0de"
FAMILIES = ("qwen3_moe", "granitemoe", "mixtral")
LAYERS_30B = 48


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "bench" / "sc2" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RED = _mod("sc2f_reduce")


def _cells() -> dict:
    return {(c["family"], str(c["knob"])): c for c in json.loads(FIX.read_bytes())["cells"]}


def _arms() -> dict:
    return {c["arm"]: c for c in json.loads(FLIPS_FIX.read_bytes())["cells"]}


def _health(family, knob, glue_value=None):
    """A start /health as FLIPS reads it: the cell's real ``prefill_routes``, and ``levers.prefill_glue`` as /health
    resolves it for the family."""
    from experts4bit_qlora import serve_paged as s
    mode, source = s.resolve_prefill_glue(glue_value or s.FUSION_UNSET, family)
    return {"prefill_routes": copy.deepcopy(_cells()[(family, knob)]["routes"]),
            "levers": {"prefill_glue": {"mode": mode, "source": source}}}


def _pre_amendment(seen, arm) -> bool:
    """SC2f's FLIPS rule above 256 rows as merged (#1619): exactly the one chained key, so True is a refusal."""
    want = {"on": {"chained|lean|gt256"}, "off": {"chained|gather|gt256"}}[arm]
    return {k for k, n in (seen.get("moe_k19_dispatch") or {}).items() if k.endswith("|gt256") and n} != want


def test_the_records_are_the_receipts():
    raw = FIX.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIX_SHA256 and b"\r" not in raw
    raw = FLIPS_FIX.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FLIPS_SHA256 and b"\r" not in raw
    for name, sha in SC2E_SHA256.items():
        assert hashlib.sha256((SC2E / name).read_bytes()).hexdigest() == sha, name


# real records ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["on", "off"])
def test_real_start_health_reads_as_its_own_arm_and_not_the_other(arm):
    cell = _arms()[arm]
    assert cell["status"] == "RECORDED" and cell["config_top_k"] == 4
    assert RED.flips_bad(cell["health"], arm) == {}
    other = RED.flips_bad(cell["health"], {"on": "off", "off": "on"}[arm])
    assert {"levers.prefill_glue", "prefill_routes.prefill_glue_env", "prefill_routes.prefill_lean_dispatch_env",
            "prefill_routes.prefill_lean_dispatch", "seen.moe_k19_dispatch.gt256", "seen.moe_k19_dispatch.chained",
            "seen.prefill_folds"} == set(other)


@pytest.mark.parametrize("arm", ["on", "off"])
def test_real_records_take_both_tables_on_the_arms_dispatch(arm):
    """The serve path's 2,048-row chunks on the chained table, the 512-row forward on the one-launch table, both on the
    arm's dispatch: the merged rule refuses either arm for the one-launch call."""
    cell, d = _arms()[arm], RED.WANT_FLIPS[arm]["dispatch"]
    assert cell["seen_serve"] == {f"chained|{d}|gt256": 4}
    assert cell["seen_eager_512_rows"] == {f"one_launch|{d}|gt256": 2}
    assert _pre_amendment(cell["health"]["prefill_routes"]["seen"], arm)


def test_real_decode_calls_stay_lean_on_both_arms_and_are_not_read():
    """The prefill dispatch knob governs rows above 256 only: decode's small calls are lean under both arms."""
    for arm, cell in _arms().items():
        le = {k: n for k, n in cell["health"]["prefill_routes"]["seen"]["moe_k19_dispatch"].items()
              if k.endswith("|le256")}
        assert le == {"one_launch|lean|le256": 24}, arm

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("knob", ["0", "0r"])
def test_real_calls_above_256_rows_read_the_lean_dispatch(family, knob):
    """Every real call reads as on's dispatch and not as off's; the one departure is the prefill capture's chained call,
    which a 1,024-row forward on 4 experts never routes (an SC2f server's 4,096-row capture does)."""
    k19 = _cells()[(family, knob)]["routes"]["seen"]["moe_k19_dispatch"]
    assert RED.k19_gt256_bad(k19, "lean") == {"seen.moe_k19_dispatch.chained": ["one_launch|lean|gt256"]}
    assert RED.k19_gt256_bad(k19, "gather")["seen.moe_k19_dispatch.gt256"] == ["one_launch|lean|gt256"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("knob", ["32", "64"])
def test_real_split_records_are_refused_on_either_arm(family, knob):
    k19 = _cells()[(family, knob)]["routes"]["seen"]["moe_k19_dispatch"]
    assert k19 == {"split|lean|gt256": 2}
    for d in ("lean", "gather"):
        assert RED.k19_gt256_bad(k19, d)["seen.moe_k19_dispatch.gt256"] == ["split|lean|gt256"]


def test_the_real_qwen3_record_reads_as_on_but_for_the_prefill_capture():
    """The glue lever, both raw env values, the lean setting and the three prefill folds all read as on from the real
    record; off's reading departs on each knob."""
    h = _health("qwen3_moe", "0")
    assert RED.flips_bad(h, "on") == {"seen.moe_k19_dispatch.chained": ["one_launch|lean|gt256"]}
    off = RED.flips_bad(h, "off")
    assert {"levers.prefill_glue", "prefill_routes.prefill_glue_env", "prefill_routes.prefill_lean_dispatch_env",
            "prefill_routes.prefill_lean_dispatch", "seen.moe_k19_dispatch.gt256", "seen.prefill_folds"} <= set(off)


@pytest.mark.parametrize("family", ["granitemoe", "mixtral"])
def test_real_records_off_the_allowlist_read_as_the_proofs_form(family):
    """Outside the glue allowlist the default resolves off and no fold runs: the proof's on form, whatever the K19 table."""
    h = _health(family, "0")
    assert RED.flips_bad(h, "on", proof=True) == {}
    assert RED.flips_bad(h, "on") != {}


def test_the_real_record_with_the_prefill_capture_passes_and_the_merged_rule_refused_it():
    """The real record plus the one call it could not route: on passes. The merged rule refuses it for the one-launch
    call, which is the table bucket 64's decode capture takes on Qwen3-30B (P122's bound: 128 experts x 512 rows)."""
    h = _health("qwen3_moe", "0")
    h["prefill_routes"]["seen"]["moe_k19_dispatch"]["chained|lean|gt256"] = 5 * LAYERS_30B
    assert RED.flips_bad(h, "on") == {}
    assert _pre_amendment(h["prefill_routes"]["seen"], "on")


@pytest.mark.parametrize("draw", ["d1", "d2"])
def test_sc2e_s64a_startup_makes_bucket_64_calls_above_256_rows(draw):
    def rec(arm):
        return json.loads((SC2E / f"health_e4b_{arm}_{draw}_start.json").read_bytes())
    gt = {arm: rec(arm)["prefill_routes"]["seen"]["moe"]["int4_k19|gt256"] for arm in ("s16", "s64c", "s64a")}
    assert gt == {"s16": 5 * LAYERS_30B, "s64c": 5 * LAYERS_30B, "s64a": 5 * LAYERS_30B + 3 * LAYERS_30B}
    assert 64 in rec("s64a")["engine"]["buckets"] and max(rec("s64c")["engine"]["buckets"]) == 16


# synthetic departures from the real record -------------------------------------------------------------------------

def _on():
    return copy.deepcopy(_arms()["on"]["health"])


def _off():
    return copy.deepcopy(_arms()["off"]["health"])


def test_departure_on_with_a_gather_call_is_refused():
    h = _on()
    k = h["prefill_routes"]["seen"]["moe_k19_dispatch"]
    k["one_launch|gather|gt256"] = k.pop("one_launch|lean|gt256")
    assert set(RED.flips_bad(h, "on")) == {"seen.moe_k19_dispatch.gt256"}


def test_departure_off_with_a_lean_call_is_refused():
    h = _off()
    k = h["prefill_routes"]["seen"]["moe_k19_dispatch"]
    k["one_launch|lean|gt256"] = k.pop("one_launch|gather|gt256")
    assert set(RED.flips_bad(h, "off")) == {"seen.moe_k19_dispatch.gt256"}


def test_departure_a_split_beside_the_prefill_capture_is_refused():
    h = _on()
    h["prefill_routes"]["seen"]["moe_k19_dispatch"]["split|lean|gt256"] = 2
    assert RED.flips_bad(h, "on") == {"seen.moe_k19_dispatch.gt256": ["split|lean|gt256"]}


def test_departure_small_calls_are_not_read():
    h = _on()
    h["prefill_routes"]["seen"]["moe_k19_dispatch"]["one_launch|gather|le256"] = 24
    assert RED.flips_bad(h, "on") == {}
