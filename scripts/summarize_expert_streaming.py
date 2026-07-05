"""Summarize expert-streaming profile JSONL: is the offload transfer wall hot-expert concentrated?

Reads one or more profiles written by ``experts4bit_qlora.expert_profile`` and reports routing
concentration, projected pinning budgets, and a build/don't-build decision against the
pre-registered criterion. Prints observed-in-this-run; no universal claims.

Load-bearing honesty (stated in every relevant table): staging is LAYER-granular, so measured
H2D stall is per layer, not per (layer, expert). Per-expert stall is a PROJECTION under one
explicit attribution rule — a layer's staging stall is shared among the experts that were routed
in that layer, weighted by tokens_routed. That rule is what a per-expert pinning policy would
have to beat; it is not a measurement of isolated per-expert transfer (which does not exist
today). Both interpretations are printed so the reader can pick.

Decision criterion (pre-registered): build hot-static only if the top 10% of (layer, expert)
pairs account for >=40% of projected stall, or the top 20% account for >=60%.

Usage:
    python scripts/summarize_expert_streaming.py \\
      --input runs/expert_streaming/<job>/profile.jsonl \\
      --out-md runs/expert_streaming/<job>/profile.md
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mode_matrix_common import read_jsonl  # noqa: E402

BUDGETS_GB = (0.25, 0.5, 1.0, 2.0)


def load(path):
    rows = read_jsonl(path)
    meta = next((r for r in rows if r.get("row") == "meta"), {})
    layers = {r["layer_id"]: r for r in rows if r.get("row") == "layer"}
    experts = [r for r in rows if r.get("row") == "expert"]
    return meta, layers, experts


def build_pairs(layers, experts):
    """One record per routed (layer, expert): hits, tokens, per-expert bytes, and the PROJECTED
    stall share (layer stall * this expert's token fraction within the layer)."""
    tokens_in_layer = {}
    for e in experts:
        tokens_in_layer[e["layer_id"]] = tokens_in_layer.get(e["layer_id"], 0) + e["tokens_routed"]
    pairs = []
    for e in experts:
        lid = e["layer_id"]
        layer = layers.get(lid, {})
        layer_stall = layer.get("h2d_ms_total", 0.0)
        frac = e["tokens_routed"] / tokens_in_layer[lid] if tokens_in_layer.get(lid) else 0.0
        # per_expert_bytes recorded by the profiler reads 0 when the base was offload-evicted at
        # attach time (0-element placeholders). Derive it from the layer's staged bytes instead:
        # h2d_bytes = stage_nbytes x stage_copies, so one stage / num_experts = per-expert bytes.
        per_expert = layer.get("per_expert_bytes", 0)
        copies, nexp = layer.get("stage_copies", 0), layer.get("num_experts", 0)
        if not per_expert and copies and nexp:
            per_expert = layer.get("h2d_bytes", 0) / copies / nexp
        pairs.append({
            "layer_id": lid,
            "expert_id": e["expert_id"],
            "hits": e["hits"],
            "tokens_routed": e["tokens_routed"],
            "per_expert_bytes": per_expert,
            "projected_stall_ms": layer_stall * frac,
            "h2d_bytes_share": per_expert * copies,
        })
    return pairs


def _table(pairs, key, top=12):
    ordered = sorted(pairs, key=lambda p: p[key], reverse=True)[:top]
    lines = [f"| layer | expert | {key} | hits | tokens |", "|---|---|---|---|---|"]
    for p in ordered:
        v = p[key]
        vs = f"{v:.2f}" if isinstance(v, float) else str(v)
        lines.append(f"| {p['layer_id']} | {p['expert_id']} | {vs} | {p['hits']} | {p['tokens_routed']} |")
    return "\n".join(lines)


def concentration(pairs, key):
    total = sum(p[key] for p in pairs)
    n = len(pairs)
    out = []
    if total <= 0 or n == 0:
        return out, total
    ordered = sorted(pairs, key=lambda p: p[key], reverse=True)
    for pct in (1, 5, 10, 20):
        k = max(1, round(n * pct / 100))
        share = sum(p[key] for p in ordered[:k]) / total
        out.append((pct, k, share))
    return out, total


def _pin_score(p, score):
    b = p["per_expert_bytes"]
    if not b:
        return 0.0
    if score == "stall-per-byte":
        return p["projected_stall_ms"] / b
    if score == "bytes-per-byte":  # fallback when stall timing is noisy
        return p["h2d_bytes_share"] / b
    if score == "hits-bytes-per-byte":
        return p["hits"] * p["h2d_bytes_share"] / b
    raise ValueError(f"unknown score {score!r}")


def budget_projection(pairs, total_stall, score="stall-per-byte"):
    """Greedy by ``score`` (default: projected stall per resident byte): fill each budget with
    the highest-value pairs whose cumulative per-expert bytes fit. Returns one row per budget:
    (budget_gb, selected_pairs, used_gb, stall_covered_frac, h2d_gb_avoided, hit_covered_frac).
    Greedy, not a knapsack — v1 by design."""
    ordered = sorted(pairs, key=lambda p: _pin_score(p, score), reverse=True)
    total_hits = sum(p["hits"] for p in pairs)
    rows = []
    for gb in BUDGETS_GB:
        cap = gb * 1e9
        used = 0
        covered = 0.0
        avoided = 0
        hits_cov = 0
        selected = []
        for p in ordered:
            b = p["per_expert_bytes"]
            if b and used + b <= cap:
                used += b
                covered += p["projected_stall_ms"]
                avoided += p["h2d_bytes_share"]
                hits_cov += p["hits"]
                selected.append(p)
        rows.append((gb, selected, used / 1e9,
                     covered / total_stall if total_stall else 0.0,
                     avoided / 1e9,
                     hits_cov / total_hits if total_hits else 0.0))
    return rows


def decide(pairs):
    conc, total = concentration(pairs, "projected_stall_ms")
    by_pct = {pct: share for pct, _, share in conc}
    hit_10 = by_pct.get(10, 0) >= 0.40
    hit_20 = by_pct.get(20, 0) >= 0.60
    if total <= 0:
        return "INSUFFICIENT DATA (no staging stall recorded — is this an offload profile on CUDA?)"
    if hit_10 or hit_20:
        why = []
        if hit_10:
            why.append(f"top 10% = {by_pct[10]:.0%} of projected stall (>=40%)")
        if hit_20:
            why.append(f"top 20% = {by_pct[20]:.0%} (>=60%)")
        return "BUILD hot-static: concentration meets the criterion — " + "; ".join(why)
    return (f"DO NOT build hot-static: stall is diffuse (top 10% = {by_pct.get(10, 0):.0%}, "
            f"top 20% = {by_pct.get(20, 0):.0%}); the offload wall is not hot-expert concentrated "
            "for this model/path.")


def render(meta, layers, experts):
    pairs = build_pairs(layers, experts)
    md = ["# Expert-streaming profile", "",
          f"- model: {meta.get('model')} | phase: {meta.get('phase')} | storage: "
          f"{next(iter({lr['storage_mode'] for lr in layers.values()}), '?')} | "
          f"offload: {meta.get('offload')} | seed: {meta.get('seed')}",
          f"- host: {meta.get('gpu')} | torch {meta.get('torch_version')} | "
          f"bnb {meta.get('bitsandbytes_version')}",
          f"- methodology: {meta.get('methodology')}",
          "", "**Staging is layer-granular.** Measured H2D stall is per layer; per-(layer,expert) "
          "stall below is a PROJECTION (layer stall shared by token fraction) — the number a "
          "per-expert pinning policy would have to beat, not a measurement of isolated transfer.",
          ""]

    total_stall = sum(layer.get("h2d_ms_total", 0.0) for layer in layers.values())
    total_bytes = sum(layer.get("h2d_bytes", 0) for layer in layers.values())
    md += [f"- layers profiled: {len(layers)} | routed (layer,expert) pairs: {len(pairs)}",
           f"- total measured H2D stall: {total_stall:.1f} ms across {total_bytes / 1e9:.2f} GB staged",
           ""]

    md += ["## Top (layer, expert) by projected stall ms", "", _table(pairs, "projected_stall_ms"), ""]
    md += ["## Top (layer, expert) by hits", "", _table(pairs, "hits"), ""]
    md += ["## Top (layer, expert) by tokens routed", "", _table(pairs, "tokens_routed"), ""]

    md += ["## Concentration (share of total held by the hottest pairs)", "",
           "| metric | top 1% | top 5% | top 10% | top 20% |", "|---|---|---|---|---|"]
    for key in ("projected_stall_ms", "hits", "tokens_routed"):
        conc, _ = concentration(pairs, key)
        shares = {pct: share for pct, _, share in conc}
        md.append(f"| {key} | " + " | ".join(f"{shares.get(p, 0):.0%}" for p in (1, 5, 10, 20)) + " |")
    md.append("")

    md += ["## Projected pinning budgets (greedy by stall-per-byte)", "",
           "Estimated coverage if the hottest experts were held resident within a GPU cache "
           "budget (projections under the attribution rule, not measured speedups). Offload is "
           "not binary: this table is the dial a user would spend spare VRAM on.", "",
           "| budget GB | pinned experts | added VRAM GB | projected stall covered | "
           "H2D GB avoided | hit % covered |", "|---|---|---|---|---|---|"]
    for gb, selected, used, frac, avoided, hitfrac in budget_projection(pairs, total_stall):
        md.append(f"| {gb:.2f} | {len(selected)} | {used:.2f} | {frac:.0%} | "
                  f"{avoided:.2f} | {hitfrac:.0%} |")
    md.append("")

    md += ["## Decision", "", decide(pairs), "",
           "## What is not claimed", "",
           "- No measurement of isolated per-expert transfer (staging is layer-granular).",
           "- No speedup is claimed here — pinning coverage is a projection; a hot-static "
           "validation run is required to measure real s/step / stall change.",
           "- Concentration is observed for this model/path/host; it is not asserted to generalize "
           "(Qwen3 gets a sentinel profile only if OLMoE shows concentration worth scaling).", ""]
    return "\n".join(md)


def write_policies(meta, layers, experts, policies_dir, score="stall-per-byte"):
    """Controller-generated machine-readable hot-static policies, one per budget. Field names are
    honest: per-expert stall is stall_ms_projected (layer-granular staging; see module docstring),
    not an observation of isolated per-expert transfer."""
    pairs = build_pairs(layers, experts)
    total_stall = sum(lr.get("h2d_ms_total", 0.0) for lr in layers.values())
    storage = next(iter({lr["storage_mode"] for lr in layers.values()}), "unknown")
    os.makedirs(policies_dir, exist_ok=True)
    written = []
    for gb, selected, used, frac, avoided, hitfrac in budget_projection(pairs, total_stall, score):
        policy = {
            "policy": "hot-static",
            "model": meta.get("model"),
            "storage_mode": storage,
            "offload": meta.get("offload"),
            "phase_profiled": meta.get("phase"),
            "seed": meta.get("seed"),
            "budget_gb": gb,
            "score": score,
            "attribution": "per-expert stall is a PROJECTION: layer staging stall shared by "
                           "token fraction (staging is layer-granular)",
            "added_vram_gb": round(used, 4),
            "projected_stall_covered": round(frac, 4),
            "h2d_gb_avoided": round(avoided, 4),
            "hit_fraction_covered": round(hitfrac, 4),
            "selected_experts": [
                {"layer": p["layer_id"], "expert": p["expert_id"],
                 "resident_bytes": p["per_expert_bytes"],
                 "score": round(_pin_score(p, score), 6),
                 "stall_ms_projected": round(p["projected_stall_ms"], 3),
                 "hits": p["hits"], "tokens_routed": p["tokens_routed"]}
                for p in selected
            ],
        }
        name = (f"{'olmoe' if 'OLMoE' in str(meta.get('model')) else 'model'}_{storage}_"
                f"{'offload' if meta.get('offload') else 'resident'}_hotstatic_budget{gb}gb.json")
        path = os.path.join(policies_dir, name)
        with open(path, "w") as f:
            json.dump(policy, f, indent=2, sort_keys=True)
            f.write("\n")
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--policies-out", default=None,
                    help="controller-only: also write hot-static policy JSONs per budget here")
    ap.add_argument("--score", default="stall-per-byte",
                    choices=("stall-per-byte", "bytes-per-byte", "hits-bytes-per-byte"),
                    help="greedy pin score; the byte-based fallbacks are for noisy stall timing")
    args = ap.parse_args()
    meta, layers, experts = load(args.input)
    md = render(meta, layers, experts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(md)
    print(md)
    if args.policies_out:
        for path in write_policies(meta, layers, experts, args.policies_out, args.score):
            print(f"wrote policy {path}")
    print(f"\nwrote {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
