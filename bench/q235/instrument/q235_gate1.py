# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235 gate 1's verdict, applied mechanically from the registered rules (bench/q235/PREREG-q235.md), never read by eye.

From a rehearsal directory (``rehearse.sh``'s layout: ``ref/``, ``box/``, ``repeat/``, ``plant_routing/``, ``plant_epilogue/``):

- **box**: Amendment 2's STRICT rules and the frozen base, re-derived from ``box/box.json`` (fail-closed), and Amendment 1's bands
  against the layer-streamed bf16 reference: median per-layer relative error <= 0.25, maximum <= 0.50, minimum router overlap >= 0.80.
- **repeat** (rule 6): the fresh-process repeat of step 0's loss within the larger of the box's in-process spread and 1e-4 relative.
- **plants**: each must FAIL the bands as registered -- the routing error by a maximum relative error above 0.50 or a minimum
  overlap below 0.80, the epilogue error by a maximum relative error above 0.50.

Verdict and exit code: PASS (0) when the box passes, the repeat passes and both plants fail; FAIL (1) when the box or the repeat
does not; BANDS_VOID (2) when a plant passes, which voids the bands (Amendment 1: a new amendment sets them before any box);
HARNESS_ERROR (3) when a record is missing or unreadable. ``gate1.json`` holds every number the verdict used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import torch

MEDIAN_MAX, MAX_MAX, OVERLAP_MIN = 0.25, 0.50, 0.80      # Amendment 1, rule 7
LOSS_FLOOR_REL = 1e-4                                   # Amendment 1, rule 6
EXIT = {"PASS": 0, "FAIL": 1, "BANDS_VOID": 2, "HARNESS_ERROR": 3}


def _sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def band_failures(cmp):
    """The bands a comparison (q235_compare.compare against the reference) violates, by name. Empty = within the bands."""
    bad = []
    if not cmp["rel_err_median"] <= MEDIAN_MAX:
        bad.append(f"rel_err_median {cmp['rel_err_median']:.4g} > {MEDIAN_MAX}")
    if not cmp["rel_err_max"] <= MAX_MAX:
        bad.append(f"rel_err_max {cmp['rel_err_max']:.4g} > {MAX_MAX}")
    if not cmp["router_overlap_min"] >= OVERLAP_MIN:
        bad.append(f"router_overlap_min {cmp['router_overlap_min']:.4g} < {OVERLAP_MIN}")
    return bad


def plant_fails_as_registered(kind, cmp):
    """Amendment 1's 'must fail' line for each planted error."""
    if kind == "routing":
        return not cmp["rel_err_max"] <= MAX_MAX or not cmp["router_overlap_min"] >= OVERLAP_MIN
    if kind == "epilogue":
        return not cmp["rel_err_max"] <= MAX_MAX
    raise ValueError(kind)


def repeat_within_tolerance(box, repeat_loss):
    """Rule 6: |repeat - box| on step 0's loss within max(the box's in-process spread, 1e-4 relative)."""
    ip = box["in_process"]
    l0 = float(ip["loss_0"])
    tol = max(abs(float(ip["loss_1"]) - l0), LOSS_FLOOR_REL * abs(l0))
    gap = abs(float(repeat_loss) - l0)
    return gap <= tol, {"loss_box": l0, "loss_repeat": float(repeat_loss), "gap": gap, "tolerance": tol,
                        "in_process_spread": abs(float(ip["loss_1"]) - l0)}


def judge(run_dir, tokens):
    compare = _sibling("q235_compare").compare
    strict_failures = _sibling("q235_box").strict_failures
    mask = torch.load(tokens, weights_only=True)["attention_mask"]
    d = {k: os.path.join(run_dir, k) for k in ("ref", "box", "repeat", "plant_routing", "plant_epilogue")}
    out = {"run_dir": os.path.abspath(run_dir), "bands": {"rel_err_median_max": MEDIAN_MAX, "rel_err_max_max": MAX_MAX,
                                                            "router_overlap_min": OVERLAP_MIN, "loss_floor_rel": LOSS_FLOOR_REL}}
    try:
        box = json.load(open(os.path.join(d["box"], "box.json")))
        rep = json.load(open(os.path.join(d["repeat"], "box.json")))
        cmp_box = compare(d["box"], d["ref"], mask)
        cmp_plants = {k: compare(d[f"plant_{k}"], d["ref"], mask) for k in ("routing", "epilogue")}
        plants_strict = {k: strict_failures(json.load(open(os.path.join(d[f"plant_{k}"], "box.json")))) for k in ("routing", "epilogue")}
    except (OSError, KeyError, ValueError, AssertionError, RuntimeError) as e:
        out.update(verdict="HARNESS_ERROR", why=f"{type(e).__name__}: {e}")
        return out
    box_strict = strict_failures(box)
    box_bands = band_failures(cmp_box)
    repeat_ok, repeat_rec = repeat_within_tolerance(box, rep["repeat"]["loss"])
    plants = {k: {"fails_as_registered": plant_fails_as_registered(k, c), "band_failures": band_failures(c),
                  "rel_err_max": c["rel_err_max"], "router_overlap_min": c["router_overlap_min"], "rel_err": c["rel_err"],
                  "strict_failures": plants_strict[k]} for k, c in cmp_plants.items()}
    out.update(box={"strict_failures": box_strict, "band_failures": box_bands, "compare": cmp_box},
               repeat={"within_tolerance": repeat_ok, **repeat_rec}, plants=plants)
    if not all(p["fails_as_registered"] for p in plants.values()):
        out.update(verdict="BANDS_VOID", why="a planted error passed the bands: " +
                   ", ".join(k for k, p in plants.items() if not p["fails_as_registered"]))
    elif box_strict or box_bands or not repeat_ok:
        out.update(verdict="FAIL", why="; ".join(box_strict + box_bands + ([] if repeat_ok else
                                                                          [f"repeat gap {repeat_rec['gap']:.3g} > {repeat_rec['tolerance']:.3g}"])))
    else:
        out.update(verdict="PASS", why="the box within the bands and its STRICT rules, the repeat within tolerance, both plants failing")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = judge(a.run_dir, a.tokens)
    json.dump(r, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({"verdict": r["verdict"], "why": r["why"]}))
    sys.exit(EXIT[r["verdict"]])


if __name__ == "__main__":
    main()
