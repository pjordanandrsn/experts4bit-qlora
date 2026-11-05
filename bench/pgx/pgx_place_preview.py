"""PGX placement preview: E-hyb's placement at each registered budget, from the solver itself, before any timed cell.

    python bench/pgx/pgx_place_preview.py --calib calib.json --profile bench/pgx/fit_profile.jsonl --out preview.json

Runs ``engines/placement.solve_placement`` exactly as ``serve_paged.build_engine`` calls it at the solver placement
(batch 1, top_k 8, NF4 bytes per expert, ``E4B_PAGED_VRAM_GB`` = the decimal budget in GiB rounded to 3 places,
``E4B_PAGED_DRAM_GB=16``), with the measured profile (E-hyb) and without it (E-hyb-u). Per budget it reports the VRAM
tier's experts and bytes against the decimal budget, the DRAM and NVMe tier counts, and the share of the profile's
routed mass the VRAM tier holds. That share is E-hyb's predicted VRAM hit rate on the profile's own prompts, and the
registration's E-hyb expectation is computed this way from the BOX'S calib.json (the shakedown's calibrate.py) before
the reading. It depends on the host: the solver balances GPU against CPU completion time, so a host with a fast CPU
scatter keeps more of the routed mass on its DRAM tier (measured on the sc1 5090 host's blob at $0: 0.709 at 8 GB
against a top-by-mass ceiling of 0.843).

Stdlib only (placement.py is stdlib); runs on any CPU.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLACEMENT = os.path.join(_HERE, "..", "..", "experts4bit_qlora", "engines", "placement.py")
NF4_BYTES_PER_EXPERT = 2_654_208          # P233's placement record for this model's NF4 arena
BUDGETS_GB = (4, 8, 12)
DRAM_GIB = 16.0


def placement_module():
    """The solver module: the repo's own file beside this kit, or, staged flat on the box (no repo layout), the
    installed package's, which the tripwire has checked is the launch commit's."""
    if not os.path.exists(_PLACEMENT):
        return importlib.import_module("experts4bit_qlora.engines.placement")
    spec = importlib.util.spec_from_file_location("pgx_placement", _PLACEMENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vram_gib(budget_gb: float) -> float:
    """``E4B_PAGED_VRAM_GB`` for a decimal budget: serve_paged reads GiB (vram_gb * 2**30)."""
    return round(budget_gb * 1e9 / 2**30, 3)


def preview(calib: dict, profile: str, layers=48, experts=128, bpe=NF4_BYTES_PER_EXPERT, top_k=8) -> dict:
    pl = placement_module()
    mass, _ = pl.load_routing_mass(profile, layers, experts)
    total = sum(mass.values())
    out = {}
    for b in BUDGETS_GB:
        for arm, prof in (("E-hyb", profile), ("E-hyb-u", None)):
            m = pl.solve_placement(n_layers=layers, n_experts=experts, bytes_per_expert=bpe,
                                   vram_budget_bytes=int(vram_gib(b) * 2**30),
                                   dram_budget_bytes=int(DRAM_GIB * 2**30), calibration=calib, profile_path=prof,
                                   batch=1, top_k=top_k)
            vram = [tuple(x) for x in m["tiers"]["vram"]]
            out[f"{arm}|{b}"] = {
                "vram_experts": len(vram), "vram_bytes": len(vram) * bpe, "budget_bytes": int(b * 1e9),
                "dram_experts": len(m["tiers"]["dram"]), "nvme_experts": len(m["tiers"]["nvme"]),
                "vram_mass_share": sum(mass.get(k, 0.0) for k in vram) / total,
                "profile_kind": m["profile"]["kind"]}
        n = int(vram_gib(b) * 2**30) // bpe
        out[f"ceiling|{b}"] = {"vram_experts": n,
                               "vram_mass_share": sum(sorted(mass.values(), reverse=True)[:n]) / total}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--profile", default=os.path.join(_HERE, "fit_profile.jsonl"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.calib, encoding="utf-8") as f:
        calib = json.load(f)
    p = preview(calib, a.profile)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=1, sort_keys=True)
    for k, v in sorted(p.items()):
        print(f"{k:12s} vram {v['vram_experts']:5d} share {v['vram_mass_share']:.3f}")
    bad = [k for k, v in p.items() if not k.startswith("ceiling")
           and (v["vram_bytes"] > v["budget_bytes"] or v["nvme_experts"] != 0)]
    if bad:
        print(f"REFUSED: over budget or NVMe tier used at {bad}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
