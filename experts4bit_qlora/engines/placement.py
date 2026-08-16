# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Static placement solver — hybrid tier Phase 3.

Assigns every (layer, expert) to a tier — ``vram`` (hot, GPU-resident),
``dram`` (warm, computed in place by the CPU kernels), ``nvme`` (cold,
streamed NVMe→GPU) — from two measured inputs and nothing else:

  * the calibration blob (``gnf4-hybrid-calib/1``, bench/calibrate.py):
    achieved ``B_vram``, ``B_dram`` (grouped-scatter), ``B_nvme`` on the box
    that will run — never spec sheets;
  * a routing-frequency profile (``expert_profile`` JSONL from a profiling
    pass over a representative corpus). With no profile, routing is assumed
    uniform and the solver SAYS so in the manifest — an index-ordered hot
    set is a uniform random draw (measured: statistically identical to
    streaming), so an unprofiled placement is honest but weak.

Objective, in the directive's priority order: (1) minimize NVMe-resident
routing mass; (2) approach a GPU:CPU routing-mass ratio of
``B_vram_effective : B_dram_grouped`` subject to VRAM capacity after the
attention/KV reservation. The greedy below encodes both directly: walk
experts hottest-first, assign each to whichever compute tier currently has
the smaller completion time (mass/bandwidth) and free capacity; overflow —
and only overflow — goes to NVMe.

The result is a *placement manifest*: placement map + the input hashes +
per-backend reduction-order ids. Same artifact + same manifest ⇒ the same
run, which is what ``verify_manifest`` checks offline.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

SCHEMA = "e4b-placement/1"

# reduction-order identifiers: bumped only when an accumulation order
# changes, which is a determinism-contract change
REDUCTION_ORDER_IDS = {
    "gpu": "gnf4-fused-grouped/1",
    "cpu": "gnf4-native-locked-tree/1",
}


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_routing_mass(profile_path, n_layers: int, n_experts: int):
    """Per-(layer, expert) routed-token counts from an expert_profile JSONL.
    Returns a dict {(layer, expert): count} plus the profile sha256; cold
    experts are absent from the profile by design and count 0."""
    mass = defaultdict(float)
    with open(profile_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("row") == "expert":
                mass[(int(row["layer_id"]), int(row["expert_id"]))] += float(
                    row.get("tokens_routed", 0.0))
    return dict(mass), _sha256_file(profile_path)


def solve_placement(
    *,
    n_layers: int,
    n_experts: int,
    bytes_per_expert: int,
    vram_budget_bytes: int,
    dram_budget_bytes: int,
    calibration: dict | str | Path,
    profile_path=None,
    b_vram_override: float | None = None,
    b_dram_override: float | None = None,
) -> dict:
    """Returns the placement manifest (a plain dict; write with
    ``save_manifest``). ``calibration`` is the blob dict or its path."""
    calib_hash = None
    if not isinstance(calibration, dict):
        calib_hash = _sha256_file(calibration)
        calibration = json.loads(Path(calibration).read_text())
    cb = calibration.get("cpu_bench") or {}
    gb = calibration.get("gpu_bench") or {}
    b_dram = b_dram_override or (cb.get("scatter_best") or {}).get("gbs")
    devs = gb.get("devices") or []
    b_vram = b_vram_override or (
        devs[0].get("b_vram_triad_gbs") if devs else None)
    if not b_dram or not b_vram:
        raise ValueError(
            "calibration blob lacks scatter/vram numbers and no overrides "
            "given — the solver refuses to guess bandwidths")

    if profile_path is not None:
        mass, profile_hash = load_routing_mass(profile_path, n_layers,
                                               n_experts)
        profile_kind = "measured"
    else:
        mass, profile_hash = {}, None
        profile_kind = "uniform-assumed"

    items = [
        (mass.get((layer, e), 0.0), layer, e)
        for layer in range(n_layers)
        for e in range(n_experts)
    ]
    # hottest first; ties by (layer, expert) for determinism
    items.sort(key=lambda t: (-t[0], t[1], t[2]))

    vram_slots = int(vram_budget_bytes // bytes_per_expert)
    dram_slots = int(dram_budget_bytes // bytes_per_expert)

    tiers = {"vram": [], "dram": [], "nvme": []}
    t_gpu = 0.0   # completion-time proxies: mass / bandwidth
    t_cpu = 0.0
    for m, layer, e in items:
        gpu_ok = len(tiers["vram"]) < vram_slots
        cpu_ok = len(tiers["dram"]) < dram_slots
        if gpu_ok and (not cpu_ok or (t_gpu + m / b_vram) <=
                       (t_cpu + m / b_dram)):
            tiers["vram"].append([layer, e])
            t_gpu += m / b_vram
        elif cpu_ok:
            tiers["dram"].append([layer, e])
            t_cpu += m / b_dram
        else:
            tiers["nvme"].append([layer, e])

    total_mass = sum(m for m, _, _ in items) or 1.0
    nvme_mass = sum(mass.get((layer, e), 0.0) for layer, e in tiers["nvme"])
    vram_mass = sum(mass.get((layer, e), 0.0) for layer, e in tiers["vram"])
    dram_mass = sum(mass.get((layer, e), 0.0) for layer, e in tiers["dram"])

    return {
        "schema": SCHEMA,
        "geometry": {"n_layers": n_layers, "n_experts": n_experts,
                     "bytes_per_expert": bytes_per_expert},
        "budgets": {"vram_bytes": int(vram_budget_bytes),
                    "dram_bytes": int(dram_budget_bytes)},
        "bandwidths_gbs": {"vram": b_vram, "dram_grouped": b_dram},
        "profile": {"kind": profile_kind, "sha256": profile_hash},
        "calibration_sha256": calib_hash,
        "reduction_order_ids": dict(REDUCTION_ORDER_IDS),
        "tiers": tiers,
        "masses": {
            "vram_frac": vram_mass / total_mass,
            "dram_frac": dram_mass / total_mass,
            "nvme_frac": nvme_mass / total_mass,
            "target_gpu_cpu_ratio": b_vram / b_dram,
            "achieved_gpu_cpu_ratio":
                (vram_mass / dram_mass) if dram_mass else None,
        },
    }


def save_manifest(manifest: dict, path) -> str:
    """Write the manifest; returns its sha256 (the run-identity handle)."""
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return _sha256_file(path)


def load_manifest(path) -> dict:
    m = json.loads(Path(path).read_text())
    if m.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} manifest: {m.get('schema')!r}")
    return m


def tier_lookup(manifest: dict):
    """{(layer, expert) -> 'vram'|'dram'|'nvme'} with completeness check."""
    geo = manifest["geometry"]
    out = {}
    for tier, pairs in manifest["tiers"].items():
        for layer, e in pairs:
            key = (int(layer), int(e))
            if key in out:
                raise ValueError(f"{key} placed twice")
            out[key] = tier
    want = geo["n_layers"] * geo["n_experts"]
    if len(out) != want:
        raise ValueError(f"placement covers {len(out)} of {want} experts")
    return out


def verify_manifest(manifest_path, *, arena_manifest_path=None) -> dict:
    """Offline reproducibility check: the placement manifest is internally
    complete, its reduction-order ids are the current ones, and (when the
    gnf4 arena manifest is given) the artifact hashes it references exist.
    Returns a report dict; raises on structural violations."""
    m = load_manifest(manifest_path)
    tier_lookup(m)                       # completeness / duplicates
    report = {"manifest_sha256": _sha256_file(manifest_path),
              "schema": m["schema"],
              "reduction_order_current":
                  m["reduction_order_ids"] == REDUCTION_ORDER_IDS,
              "profile": m["profile"], "masses": m["masses"]}
    if arena_manifest_path is not None:
        am = json.loads(Path(arena_manifest_path).read_text())
        n_hashes = sum(1 for v in (am.get("entries") or am.get("rows") or [])
                       if isinstance(v, dict) and v.get("sha256"))
        report["arena_manifest_sha256"] = _sha256_file(arena_manifest_path)
        report["arena_hashed_entries"] = n_hashes
    return report
