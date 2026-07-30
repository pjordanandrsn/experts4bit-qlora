#!/usr/bin/env python3
"""Adjudicate the second model's cells against PREREG-flagship-matrix-model2.md.

Written BEFORE the cells finish, on purpose. C4 in particular registers a rule
that is easy to bend once you can see which way it falls:

    "A margin smaller than 10x the measured zero-adapter floor (0.099 %) is
     reported as NOT SEPARABLE, not as a win."

Encoding it now makes the verdict mechanical. The first ten cells produced
exactly the situation the rule anticipates (reference 3, fused 2, every margin
near the floor), and the temptation to call a winner is easiest to resist before
seeing the numbers -- which is why the prereg says so in those words.

Usage: n17_summarize.py <cells-dir>
"""
import json
import os
import sys

ZERO_ADAPTER_FLOOR = 0.099          # %, measured separately (LoRA B still zero)
SEPARABLE_AT = 10 * ZERO_ADAPTER_FLOOR   # 0.99 %
C2_BAND = 0.05                      # |delta final eval| registered band
DATASETS = ("clinical", "code", "finance", "legal", "support")


def load(cells_dir):
    out = {}
    for fn in sorted(os.listdir(cells_dir)):
        if not fn.endswith(".json"):
            continue
        c = json.load(open(os.path.join(cells_dir, fn)))
        out[(c["dataset"], c["arm"])] = c
    return out


def main():
    cells_dir = sys.argv[1] if len(sys.argv) > 1 else "cells"
    C = load(cells_dir)
    have = sorted({d for d, _ in C})
    print(f"cells present: {len(C)}/10  datasets: {have}\n")

    # ---- C1: bit-exactness, HARD GATE -------------------------------------
    print("C1 — bit-exactness (HARD GATE)")
    c1_fail = []
    for k, c in sorted(C.items()):
        ok = (c.get("C1_bit_exact") and c.get("C1_bytes_hashed", 0) > 0
              and c.get("C1_empties_skipped", 1) == 0
              and c.get("C1_control_detects_flipped_byte"))
        gb = c.get("C1_bytes_hashed", 0) / 1e9
        print(f"  {k[0]:9s} {k[1]:9s} {'PASS' if ok else 'FAIL'}  "
              f"{gb:7.2f} GB hashed, {c.get('C1_experts_changed')} changed, "
              f"empties={c.get('C1_empties_skipped')}, control={c.get('C1_control_detects_flipped_byte')}")
        if not ok:
            c1_fail.append(k)
    print(f"  => {'PASSES' if not c1_fail else 'FAILS: ' + str(c1_fail)} "
          f"({len(C)}/10 cells)\n")
    if c1_fail:
        print("  C1 FAILED: per the prereg, publish the failure and make NO speed "
              "or energy claim.\n")

    # ---- C2: loss parity ---------------------------------------------------
    print(f"C2 — loss parity, registered band |delta final eval| <= {C2_BAND}")
    for ds in DATASETS:
        r, f = C.get((ds, "reference")), C.get((ds, "fused"))
        if not (r and f):
            print(f"  {ds:9s} (missing)")
            continue
        d = abs(f["eval_loss_final"] - r["eval_loss_final"])
        print(f"  {ds:9s} delta={d:.5f}  {'PASS' if d <= C2_BAND else 'FAIL'}"
              f"  ({C2_BAND/d:.0f}x inside)" if d else f"  {ds:9s} delta=0")

    # ---- C3: cost, reported not gated -------------------------------------
    print("\nC3 — cost (reported, not gated)")
    ratios = {"s_per_step": [], "peak_vram_gb": [], "joules_per_step": []}
    for ds in DATASETS:
        r, f = C.get((ds, "reference")), C.get((ds, "fused"))
        if not (r and f):
            continue
        sp = r["s_per_step"] / f["s_per_step"]
        vm = f["peak_vram_gb"] / r["peak_vram_gb"]
        en = (f["joules_per_step"] / r["joules_per_step"]) if r.get("joules_per_step") else None
        ratios["s_per_step"].append(sp)
        ratios["peak_vram_gb"].append(vm)
        if en:
            ratios["joules_per_step"].append(en)
        print(f"  {ds:9s} speed {sp:.3f}x   vram {vm:.3f}x   energy "
              + (f"{en:.3f}x" if en else "n/a"))
    if ratios["s_per_step"]:
        print(f"  => speed {min(ratios['s_per_step']):.2f}-{max(ratios['s_per_step']):.2f}x faster, "
              f"vram {min(ratios['peak_vram_gb']):.3f}-{max(ratios['peak_vram_gb']):.3f}x, "
              + (f"energy {min(ratios['joules_per_step']):.3f}-{max(ratios['joules_per_step']):.3f}x"
                 if ratios["joules_per_step"] else "energy n/a"))

    # ---- C4: "best result", with the registered separability rule ----------
    print(f"\nC4 — best result (NOT SEPARABLE below {SEPARABLE_AT:.3f} % margin, "
          f"= 10x the {ZERO_ADAPTER_FLOOR} % zero-adapter floor)")
    tally = {"reference": 0, "fused": 0, "not separable": 0}
    for ds in DATASETS:
        r, f = C.get((ds, "reference")), C.get((ds, "fused"))
        if not (r and f):
            continue
        rv, fv = r["eval_loss_final"], f["eval_loss_final"]
        # Relative to the REFERENCE, matching the convention the first ten cells
        # published (finance: 0.00723/0.44563 = 1.62 %). Dividing by max() gives
        # 1.60 % instead -- same verdict there, but the two definitions can fall
        # either side of the 0.99 % threshold, so pin the published one.
        margin = abs(fv - rv) / rv * 100
        if margin < SEPARABLE_AT:
            verdict = "not separable"
        else:
            verdict = "fused" if fv < rv else "reference"
        tally[verdict] += 1
        print(f"  {ds:9s} ref {rv:.5f}  fused {fv:.5f}  margin {margin:.3f} %  -> {verdict}")
    print(f"  => tally {tally}")
    if tally["not separable"] >= max(tally["reference"], tally["fused"]):
        print("  => The two paths are NOT SEPARABLE at this sample size. The correct "
              "reading is that the fused path REPRODUCES the reference, which is what "
              "C2 tests -- not that either 'won'.")

    # ---- C5: cross-model transfer -----------------------------------------
    print("\nC5 — cross-model transfer (qualitative reproduction, NOT numeric match)")
    if ratios["s_per_step"]:
        checks = [
            ("fused faster per step", all(x > 1.0 for x in ratios["s_per_step"])),
            ("peak VRAM ratio below 1.0", all(x < 1.0 for x in ratios["peak_vram_gb"])),
            ("VRAM ratio flat across datasets",
             (max(ratios["peak_vram_gb"]) - min(ratios["peak_vram_gb"])) < 0.02),
            ("energy ratio below 1.0",
             bool(ratios["joules_per_step"]) and all(x < 1.0 for x in ratios["joules_per_step"])),
        ]
        for label, ok in checks:
            print(f"  [{'YES' if ok else 'NO '}] {label}")
        print(f"  => topology {'REPRODUCES' if all(o for _, o in checks) else 'PARTIALLY reproduces'} "
              "on the second model")
        print("  NB: the first ten cells measured 1.75-1.81x speed / 0.754-0.755x vram / "
              "0.797-0.846x energy. A different ratio here is NOT a failure -- C5 registers a "
              "direction check, and a different architecture has no obligation to match.")


if __name__ == "__main__":
    main()
