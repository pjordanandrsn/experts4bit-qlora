# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Stage-3 gate 1: cold mass must be DIALLED, not waited for.

`force_cold_mass` is the instrument the first experimental gate runs on, so
its failure modes are gate failures: a manifest that changed something other
than the tier of the moved experts, a reported fraction that is not the
measured one, or a non-deterministic choice of which experts go cold. Each
would make the arms incomparable while still producing numbers.
"""
import pytest

from experts4bit_qlora.engines.placement import SCHEMA, force_cold_mass


def _manifest(vram, dram, nvme=()):
    return {"schema": SCHEMA,
            "tiers": {"vram": [list(p) for p in vram],
                      "dram": [list(p) for p in dram],
                      "nvme": [list(p) for p in nvme]},
            "masses": {"vram_frac": 0.0, "dram_frac": 0.0, "nvme_frac": 0.0}}


@pytest.fixture()
def setup():
    # one layer, 10 experts, mass 0.19 .. 0.01 descending (sums to 1.0)
    mass = {(0, e): (10 - e) / 55.0 for e in range(10)}
    man = _manifest(vram=[(0, 0), (0, 1)],
                    dram=[(0, e) for e in range(2, 10)])
    return man, mass


def test_target_is_reached_and_the_achieved_fraction_is_reported(setup):
    man, mass = setup
    out = force_cold_mass(man, mass, 0.10)
    fc = out["forced_cold"]
    assert fc["achieved_frac"] >= 0.10
    # discreteness: overshoot is bounded by one expert's mass
    assert fc["achieved_frac"] - 0.10 <= max(mass.values())
    # the REPORTED fraction is the MEASURED one, not the requested one
    got = sum(mass[(lay, e)] for lay, e in out["tiers"]["nvme"])
    assert fc["achieved_frac"] == pytest.approx(got)
    assert out["masses"]["nvme_frac"] == pytest.approx(got)


def test_tail_order_moves_what_a_smaller_dram_would_have_dropped(setup):
    man, mass = setup
    out = force_cold_mass(man, mass, 0.10, order="tail")
    moved = {(m["layer"], m["expert"]) for m in out["forced_cold"]["moved"]}
    coldest = sorted(range(2, 10), key=lambda e: mass[(0, e)])
    assert moved <= {(0, e) for e in coldest[:len(moved)]}
    assert (0, 9) in moved, "the coldest DRAM expert must go first"


def test_head_order_moves_the_hottest_few(setup):
    man, mass = setup
    out = force_cold_mass(man, mass, 0.10, order="head")
    moved = out["forced_cold"]["moved"]
    assert (0, 2) in {(m["layer"], m["expert"]) for m in moved}
    tail = force_cold_mass(man, mass, 0.10, order="tail")
    assert len(moved) < tail["forced_cold"]["experts_moved"], (
        "head order should reach the same mass with fewer, hotter experts — "
        "that difference IS the burst-locality axis")


def test_nothing_but_the_tier_of_moved_experts_changes(setup):
    """An arm is only comparable to its control if one thing differs."""
    man, mass = setup
    out = force_cold_mass(man, mass, 0.10)
    before = {tuple(p) for t in ("vram", "dram", "nvme") for p in man["tiers"][t]}
    after = {tuple(p) for t in ("vram", "dram", "nvme") for p in out["tiers"][t]}
    assert before == after, "an expert was invented or lost"
    for t in ("vram", "dram", "nvme"):
        assert len({tuple(p) for p in out["tiers"][t]}) == len(out["tiers"][t])
    assert out["schema"] == man["schema"]


def test_the_input_manifest_is_not_mutated(setup):
    man, mass = setup
    dram_before = [list(p) for p in man["tiers"]["dram"]]
    force_cold_mass(man, mass, 0.20)
    assert man["tiers"]["dram"] == dram_before
    assert "forced_cold" not in man


def test_the_choice_is_deterministic(setup):
    man, mass = setup
    a = force_cold_mass(man, mass, 0.15)
    b = force_cold_mass(man, mass, 0.15)
    assert a["tiers"] == b["tiers"] and a["forced_cold"] == b["forced_cold"]


def test_vram_is_untouched_by_default(setup):
    man, mass = setup
    out = force_cold_mass(man, mass, 0.30)
    assert out["tiers"]["vram"] == man["tiers"]["vram"]


def test_already_cold_mass_counts_toward_the_target(setup):
    """A sweep point is a TOTAL cold fraction. Ignoring pre-existing NVMe
    placement would silently overshoot every arm."""
    man, mass = setup
    man["tiers"]["dram"].remove([0, 9])
    man["tiers"]["nvme"].append([0, 9])
    out = force_cold_mass(man, mass, 0.05)
    assert out["forced_cold"]["already_cold_frac"] == pytest.approx(mass[(0, 9)])
    assert out["forced_cold"]["achieved_frac"] >= 0.05


def test_zero_target_moves_nothing(setup):
    man, mass = setup
    out = force_cold_mass(man, mass, 0.0)
    assert out["forced_cold"]["experts_moved"] == 0
    assert out["tiers"]["dram"] == man["tiers"]["dram"]


def test_exhausting_the_source_tier_says_so(setup):
    """A gate that quietly delivered less cold mass than the sweep point asked
    for would report a knee at the wrong x."""
    man, mass = setup
    out = force_cold_mass(man, mass, 0.99, source="dram")
    assert "short" in out["forced_cold"]
    assert out["tiers"]["dram"] == []
    assert out["tiers"]["vram"] == man["tiers"]["vram"]


def test_both_source_can_use_vram_too(setup):
    man, mass = setup
    out = force_cold_mass(man, mass, 0.99, source="both")
    assert "short" not in out["forced_cold"]
    assert out["tiers"]["dram"] == [] and out["tiers"]["vram"] == []


@pytest.mark.parametrize("kwargs,match", [
    ({"order": "middle"}, "order"),
    ({"source": "disk"}, "source"),
])
def test_bad_arguments_are_named_errors(setup, kwargs, match):
    man, mass = setup
    with pytest.raises(ValueError, match=match):
        force_cold_mass(man, mass, 0.1, **kwargs)


@pytest.mark.parametrize("frac", [-0.1, 1.5])
def test_target_outside_the_unit_interval_is_an_error(setup, frac):
    man, mass = setup
    with pytest.raises(ValueError, match="target_frac"):
        force_cold_mass(man, mass, frac)


def test_empty_routing_mass_is_an_error(setup):
    man, _ = setup
    with pytest.raises(ValueError, match="routing mass"):
        force_cold_mass(man, {}, 0.1)
