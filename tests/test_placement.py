# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Placement solver (hybrid Phase 3): objective properties, determinism,
manifest round-trip, and the verifier's refusal behaviors."""

import json

import pytest

from experts4bit_qlora.engines import placement as pl

CALIB = {
    "cpu_bench": {"scatter_best": {"gbs": 200.0}},
    "gpu_bench": {"devices": [{"b_vram_triad_gbs": 1600.0}]},
}
L, E, BPE = 4, 16, 10 * 1024 * 1024


def _profile(tmp_path, hot=(0, 1, 2)):
    """Zipf-ish: a few hot experts per layer, everything else cold."""
    p = tmp_path / "prof.jsonl"
    rows = []
    for layer in range(L):
        for e in range(E):
            tok = 1000 // (1 + e) if e in hot else (5 if e < 12 else 0)
            rows.append({"row": "expert", "layer_id": layer, "expert_id": e,
                         "tokens_routed": tok})
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _solve(tmp_path, vram_slots=8, dram_slots=24, profile=True):
    return pl.solve_placement(
        n_layers=L, n_experts=E, bytes_per_expert=BPE,
        vram_budget_bytes=vram_slots * BPE, dram_budget_bytes=dram_slots * BPE,
        calibration=CALIB,
        profile_path=_profile(tmp_path) if profile else None)


def test_nvme_gets_only_overflow_and_coldest(tmp_path):
    # 12 nonzero-mass experts per layer x 4 layers = 48; give the compute
    # tiers exactly enough combined capacity for all of them
    m = _solve(tmp_path, vram_slots=16, dram_slots=32)
    assert len(m["tiers"]["vram"]) == 16
    assert len(m["tiers"]["dram"]) == 32
    assert len(m["tiers"]["nvme"]) == L * E - 48
    # objective (1): NVMe routing mass is minimized — with capacity for every
    # nonzero-mass expert elsewhere, nvme mass must be exactly zero
    assert m["masses"]["nvme_frac"] == 0.0


def test_hottest_experts_land_on_the_faster_tier(tmp_path):
    m = _solve(tmp_path)
    vram = {tuple(p) for p in m["tiers"]["vram"]}
    # expert 0 is the hottest everywhere; with B_vram >> B_dram the greedy
    # sends the very hottest to VRAM first
    assert any((layer, 0) in vram for layer in range(L))


def test_ratio_tracks_bandwidths(tmp_path):
    m = _solve(tmp_path, vram_slots=32, dram_slots=32)
    r = m["masses"]["achieved_gpu_cpu_ratio"]
    target = m["masses"]["target_gpu_cpu_ratio"]
    assert target == pytest.approx(8.0)
    # capacity is ample, so the completion-time greedy should land within a
    # small factor of the target ratio
    assert r == pytest.approx(target, rel=0.5)


def test_deterministic(tmp_path):
    a = _solve(tmp_path)
    b = _solve(tmp_path)
    assert a["tiers"] == b["tiers"]


def test_uniform_fallback_is_marked(tmp_path):
    m = _solve(tmp_path, profile=False)
    assert m["profile"]["kind"] == "uniform-assumed"
    assert m["profile"]["sha256"] is None


def test_refuses_blob_without_bandwidths(tmp_path):
    with pytest.raises(ValueError, match="refuses to guess"):
        pl.solve_placement(n_layers=L, n_experts=E, bytes_per_expert=BPE,
                           vram_budget_bytes=BPE, dram_budget_bytes=BPE,
                           calibration={"cpu_bench": {}, "gpu_bench": {}})


def test_manifest_roundtrip_and_verify(tmp_path):
    m = _solve(tmp_path)
    path = tmp_path / "placement.json"
    sha = pl.save_manifest(m, path)
    assert len(sha) == 64
    loaded = pl.load_manifest(path)
    assert loaded["tiers"] == m["tiers"]
    rep = pl.verify_manifest(path)
    assert rep["manifest_sha256"] == sha
    assert rep["reduction_order_current"] is True
    look = pl.tier_lookup(loaded)
    assert len(look) == L * E


def test_verify_catches_duplicates_and_holes(tmp_path):
    m = _solve(tmp_path)
    m["tiers"]["dram"].append(m["tiers"]["vram"][0])       # duplicate
    p = tmp_path / "dup.json"
    pl.save_manifest(m, p)
    with pytest.raises(ValueError, match="placed twice"):
        pl.verify_manifest(p)
    m2 = _solve(tmp_path)
    m2["tiers"]["nvme"] = m2["tiers"]["nvme"][:-1]         # hole
    p2 = tmp_path / "hole.json"
    pl.save_manifest(m2, p2)
    with pytest.raises(ValueError, match="covers"):
        pl.verify_manifest(p2)
