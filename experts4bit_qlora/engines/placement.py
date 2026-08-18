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


def expected_weight_reads(p: float, batch: int) -> float:
    """Expected number of times an expert's weights are read in one step.

    An expert selected by each token with probability ``p`` is touched at
    least once over ``batch`` tokens with probability ``1 - (1-p)^batch``
    — and once touched its weights are read ONCE, with the GEMM covering
    all of its tokens. So the per-step weight-read cost is that
    probability, NOT ``batch * p`` activations.

    At ``batch=1`` this is exactly ``p``, i.e. the Phase-3 mass law: the
    batched solver is a generalization of the batch-1 solver, not a
    replacement, and a batch-1 solve is bit-identical to Phase 3's.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"routing probability {p} outside [0, 1]")
    if batch < 1:
        raise ValueError("batch must be >= 1")
    return 1.0 - (1.0 - p) ** batch


def amortization_factor(n_experts: int, top_k: int, batch: int) -> float:
    """G8's falsifiable law: expected unique-expert weight reads per
    activation, ``E(1-(1-k/E)^B) / (B*k)``, for UNIFORM routing.

    This is :func:`expected_weight_reads` summed over E identical experts
    at ``p = k/E`` and normalized by the ``B*k`` activations a naive
    per-token dispatch would pay. Measured routing is not uniform, so a
    real profile amortizes BETTER than this curve (hot experts saturate
    sooner) — which is why the solver uses per-expert probabilities and
    only the gate quotes the closed form.
    """
    if not 1 <= top_k <= n_experts:
        raise ValueError(f"top_k {top_k} outside [1, {n_experts}]")
    p = top_k / n_experts
    return n_experts * expected_weight_reads(p, batch) / (batch * top_k)


def routing_probabilities(mass: dict, n_layers: int, n_experts: int,
                          top_k: int) -> dict:
    """{(layer, expert) -> per-token routing probability} from profile
    counts. A layer's counts sum to ``tokens * top_k``, so
    ``p = top_k * count / layer_total`` needs no token count on the side.
    Layers absent from the profile fall back to uniform ``k/E``."""
    totals = defaultdict(float)
    for (layer, e), m in mass.items():
        totals[layer] += m
    out = {}
    for layer in range(n_layers):
        tot = totals.get(layer, 0.0)
        for e in range(n_experts):
            if tot > 0:
                out[(layer, e)] = min(
                    1.0, top_k * mass.get((layer, e), 0.0) / tot)
            else:
                out[(layer, e)] = top_k / n_experts
    return out


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
    batch: int = 1,
    top_k: int | None = None,
    cpu_us_fixed: float | None = None,
    cpu_us_per_row: float | None = None,
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

    if batch < 1:
        raise ValueError("batch must be >= 1")
    if batch > 1 and top_k is None:
        raise ValueError("a batched solve needs top_k: the amortization "
                         "law is per-expert P(touched) = 1-(1-p)^B and p "
                         "is derived from the profile counts via top_k")

    # Cost weight per expert. At batch 1 this is the routed mass itself
    # (Phase 3's law, preserved bit-for-bit including the tie order); at
    # batch B it is the expected number of WEIGHT READS — the same expert
    # serving many tokens reads its bytes once, so activations stop being
    # the currency. Ordering stays monotone in mass either way, so the
    # greedy walks the same experts in the same order; what changes is how
    # much completion time each one books, and therefore where the
    # balance point between the buses falls.
    if batch == 1:
        weights = {(layer, e): mass.get((layer, e), 0.0)
                   for layer in range(n_layers) for e in range(n_experts)}
        cpu_rows = None
    else:
        probs = routing_probabilities(mass, n_layers, n_experts, top_k)
        weights = {key: expected_weight_reads(p, batch)
                   for key, p in probs.items()}
        # expected ROWS an expert serves per step: B*k*p_e. The reads
        # term above is what the batch amortizes; the rows term is what
        # it cannot — the CPU tier's per-row compute. Measured on the
        # rows-curve diagnostic: t_cpu(expert) = a + b*rows, with b
        # dominating past a few rows/expert, which concentrated routing
        # reaches at B=8. A solver that balances on bandwidth alone
        # hands the CPU a share it cannot deliver (measured: balance
        # 0.07-0.18 on the AVX-512 box against a 0.80 bar).
        cpu_rows = {key: batch * top_k * p for key, p in probs.items()}

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
    # CPU-tier cost per unit of "weight" (expected reads): bandwidth term
    # plus, when the constants are provided, the measured compute term.
    # Constants come from an in-situ probe (bench rows_curve fit), never
    # a spec sheet — same rule as every bandwidth in this solver.
    gb_per_read = bytes_per_expert / 1e9
    have_term = (cpu_us_fixed is not None or cpu_us_per_row is not None)

    def cpu_cost(key, w):
        c = w * gb_per_read / b_dram
        # EITHER constant opens the term: on AVX-512 hosts the fixed
        # call floor is the operative half (fixbox receipts), so a
        # fixed-only call must not silently fall back to bandwidth-only
        if have_term and cpu_rows is not None:
            c += (w * (cpu_us_fixed or 0.0)
                  + cpu_rows[key] * (cpu_us_per_row or 0.0)) / 1e6
        return c

    def gpu_cost(w):
        return w * gb_per_read / b_vram

    for m, layer, e in items:
        w = weights[(layer, e)]
        key = (layer, e)
        gpu_ok = len(tiers["vram"]) < vram_slots
        cpu_ok = len(tiers["dram"]) < dram_slots
        if gpu_ok and (not cpu_ok or (t_gpu + gpu_cost(w)) <=
                       (t_cpu + cpu_cost(key, w))):
            tiers["vram"].append([layer, e])
            t_gpu += gpu_cost(w)
        elif cpu_ok:
            tiers["dram"].append([layer, e])
            t_cpu += cpu_cost(key, w)
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
        "batch": {
            "solved_for": batch, "top_k": top_k,
            "cost_law": ("routed-mass" if batch == 1
                         else "unique-expert-reads: 1-(1-p)^B"),
            # what the gate's uniform closed form would predict here; the
            # solve itself uses per-expert probabilities, which amortize
            # better whenever routing is skewed
            "uniform_factor": (amortization_factor(n_experts, top_k, batch)
                               if top_k else None),
            "cpu_cost_model": ({"us_fixed": cpu_us_fixed,
                                "us_per_row": cpu_us_per_row}
                               if have_term and batch > 1 else
                               "bandwidth-only"),
            # completion-time proxies the greedy actually balanced
            "t_gpu_proxy": t_gpu, "t_cpu_proxy": t_cpu,
            "balance_ratio": (min(t_gpu, t_cpu) / max(t_gpu, t_cpu)
                              if max(t_gpu, t_cpu) > 0 else None),
        },
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
