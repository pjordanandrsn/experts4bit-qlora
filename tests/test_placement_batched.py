# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Batched placement law (Phase 8): the solver's currency changes from
routed activations to UNIQUE expert weight reads.

The properties worth pinning are the ones a wrong implementation would
break silently: batch=1 must reproduce Phase 3 EXACTLY (a "generalization"
that shifts the batch-1 answer is a regression wearing a new name), the
law must match the gate's closed form on uniform routing (that is the
falsifiable claim), and skewed routing must amortize BETTER than uniform
(otherwise the per-expert law is buying nothing over the closed form).
"""

import json

import pytest

from experts4bit_qlora.engines.placement import (  # noqa: E402
    amortization_factor,
    expected_weight_reads,
    routing_probabilities,
    solve_placement,
)

CALIB = {"cpu_bench": {"scatter_best": {"gbs": 264.3}},
         "gpu_bench": {"devices": [{"b_vram_triad_gbs": 1573.4}]}}
GEO = dict(n_layers=4, n_experts=32, bytes_per_expert=1 << 20)


def _profile(tmp_path, skew=1.0):
    """JSONL profile: expert e in each layer gets weight e**-skew."""
    path = tmp_path / "profile.jsonl"
    with open(path, "w") as f:
        for layer in range(GEO["n_layers"]):
            for e in range(GEO["n_experts"]):
                f.write(json.dumps({
                    "row": "expert", "layer_id": layer, "expert_id": e,
                    "tokens_routed": 1000.0 / ((e + 1) ** skew)}) + "\n")
    return path


def test_batch_one_is_the_phase3_law_exactly(tmp_path):
    prof = _profile(tmp_path)
    common = dict(**GEO, vram_budget_bytes=16 << 20,
                  dram_budget_bytes=48 << 20, calibration=CALIB,
                  profile_path=str(prof))
    a = solve_placement(**common)
    b = solve_placement(**common, batch=1, top_k=4)
    assert a["tiers"] == b["tiers"], "batch=1 must not move any expert"


def test_uniform_law_matches_the_gate_closed_form():
    """expected_weight_reads summed over E uniform experts, normalized by
    B*k activations, IS the gate's factor(B) — the two must not drift."""
    E, k = 128, 8
    for B in (1, 2, 4, 8, 16, 32, 64, 128):
        summed = E * expected_weight_reads(k / E, B) / (B * k)
        assert summed == pytest.approx(amortization_factor(E, k, B), rel=1e-12)


def test_factor_is_one_at_batch_one_and_decreases():
    E, k = 128, 8
    assert amortization_factor(E, k, 1) == pytest.approx(1.0)
    prev = 1.0
    for B in (2, 4, 8, 16, 32, 64, 128, 256):
        f = amortization_factor(E, k, B)
        assert 0.0 < f < prev, f"factor must fall with batch (B={B})"
        prev = f
    # deep-batch limit: every expert touched, cost E reads per B*k acts
    assert amortization_factor(E, k, 4096) == pytest.approx(
        E / (4096 * k), rel=1e-6)


def test_skewed_routing_amortizes_better_than_uniform(tmp_path):
    """A hot expert saturates (p -> 1-(1-p)^B -> 1) sooner than a uniform
    one, so measured-profile placements should book LESS total weight-read
    cost than the uniform closed form predicts. If this ever inverts, the
    per-expert law is worse than the constant it replaced."""
    mass = {}
    for layer in range(2):
        for e in range(32):
            mass[(layer, e)] = 1000.0 / ((e + 1) ** 1.5)
    probs = routing_probabilities(mass, 2, 32, top_k=4)
    for B in (4, 16, 64):
        skewed = sum(expected_weight_reads(probs[(0, e)], B)
                     for e in range(32))
        uniform = 32 * expected_weight_reads(4 / 32, B)
        assert skewed < uniform, f"skew should amortize better (B={B})"


def test_batched_solve_rebalances_and_records_its_law(tmp_path):
    prof = _profile(tmp_path, skew=1.2)
    common = dict(**GEO, vram_budget_bytes=8 << 20,
                  dram_budget_bytes=40 << 20, calibration=CALIB,
                  profile_path=str(prof))
    m1 = solve_placement(**common)
    m16 = solve_placement(**common, batch=16, top_k=4)
    assert m16["batch"]["solved_for"] == 16
    assert m16["batch"]["cost_law"].startswith("unique-expert-reads")
    assert m16["batch"]["uniform_factor"] == pytest.approx(
        amortization_factor(32, 4, 16))
    # the placement is a real solve, not a copy: capacity is unchanged, so
    # the SAME number of experts land on each tier, but the balance proxy
    # is computed under the batched law
    assert m16["batch"]["balance_ratio"] is not None
    assert {len(v) for v in m1["tiers"].values()} == \
        {len(v) for v in m16["tiers"].values()}


def test_batched_solve_refuses_without_top_k():
    with pytest.raises(ValueError, match="needs top_k"):
        solve_placement(**GEO, vram_budget_bytes=1 << 20,
                        dram_budget_bytes=1 << 20, calibration=CALIB,
                        batch=8)


def test_probability_edges():
    assert expected_weight_reads(0.0, 64) == 0.0
    assert expected_weight_reads(1.0, 1) == 1.0
    with pytest.raises(ValueError):
        expected_weight_reads(1.5, 4)
    with pytest.raises(ValueError):
        expected_weight_reads(0.5, 0)


def test_uniform_probabilities_when_layer_absent_from_profile():
    """A layer the profile never saw must not read as zero-probability —
    that would place its experts as if they were never routed."""
    probs = routing_probabilities({(0, e): 10.0 for e in range(8)},
                                  n_layers=2, n_experts=8, top_k=2)
    assert probs[(1, 0)] == pytest.approx(2 / 8)
    assert probs[(0, 0)] == pytest.approx(2 / 8)   # flat profile


def test_compute_term_moves_concentrated_experts_to_the_gpu(tmp_path):
    """The measured finding the term encodes: hot experts serve many rows
    per step, and the CPU tier pays per row while the GPU does not. With
    the term on, the solver must put MORE routed mass on VRAM than the
    bandwidth-only solve chose — at identical budgets."""
    prof = _profile(tmp_path, skew=1.5)
    common = dict(**GEO, vram_budget_bytes=8 << 20,
                  dram_budget_bytes=40 << 20, calibration=CALIB,
                  profile_path=str(prof), batch=16, top_k=4)
    bw_only = solve_placement(**common)
    with_term = solve_placement(**common, cpu_us_fixed=700.0,
                                cpu_us_per_row=600.0)
    assert with_term["batch"]["cpu_cost_model"]["us_per_row"] == 600.0
    assert bw_only["batch"]["cpu_cost_model"] == "bandwidth-only"
    assert with_term["masses"]["vram_frac"] > bw_only["masses"]["vram_frac"], \
        "compute term did not shift routed mass toward the GPU"


def test_fixed_term_alone_is_honored(tmp_path):
    """cpu_us_fixed WITHOUT cpu_us_per_row must still open the compute
    term — on AVX-512 hosts the fixed call floor is the operative half
    (rows are free post-hoist), so fixed-only is the natural call there.
    A gate keyed on per_row alone silently degraded it to bandwidth-only
    (bugbot, PR #155)."""
    prof = _profile(tmp_path, skew=1.5)
    common = dict(**GEO, vram_budget_bytes=8 << 20,
                  dram_budget_bytes=40 << 20, calibration=CALIB,
                  profile_path=str(prof), batch=16, top_k=4)
    bw_only = solve_placement(**common)
    fixed_only = solve_placement(**common, cpu_us_fixed=700.0)
    assert fixed_only["batch"]["cpu_cost_model"] == {
        "us_fixed": 700.0, "us_per_row": None}
    assert fixed_only["masses"]["vram_frac"] > bw_only["masses"]["vram_frac"], \
        "fixed-only term did not shift routed mass toward the GPU"


def test_bandwidth_only_costing_is_placement_identical_to_phase3_units():
    """The cost refactor rescaled both buses by bytes_per_expert — a
    monotone transform that must not move a single expert."""
    common = dict(**GEO, vram_budget_bytes=16 << 20,
                  dram_budget_bytes=48 << 20, calibration=CALIB)
    a = solve_placement(**common)
    b = solve_placement(**common, batch=1, top_k=4)
    assert a["tiers"] == b["tiers"]
