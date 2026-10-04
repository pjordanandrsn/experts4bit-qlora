#!/usr/bin/env python
"""Lane MG1 (and its A2000 rehearsal) reducer: tp1's verdict in tp1's units, per family, plus the ladder summary.

`c1_ok` and `verdict` are tp1_reduce.py v2.1's (bench/train-parity-20260905/tp1/tp1_reduce.py:185-213) copied VERBATIM, so
a row here means exactly what a tp1 row means: the fused arm PASSES when both its final-step loss and its median per-step
loss sit within BAND = 0.05 of the reference arm's, after the identity checks (same init, C1 bit-exact frozen experts on
both arms, the kernel engaged on every patched layer every step). Never touches eval loss.

    python mg1_reduce.py <receipts_dir>
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

BAND = 0.05


def c1_ok(r):
    return bool(r.get("C1_bit_exact")) and r.get("C1_bytes_hashed", 0) > 0 \
        and r.get("C1_empties_skipped", 1) == 0 and bool(r.get("C1_control_detects_flipped_byte"))


def verdict(ref, arm):
    """(verdict, d_final, med, why) in the registered units for an OK accelerated row. Never touches eval loss."""
    if ref is None or ref.get("status") != "ok":
        return "VOID", None, None, "no OK reference arm to read against"
    why = []
    if arm.get("init_sha") != ref.get("init_sha"):
        why.append("init_sha differs from reference (arms did not start identical)")
    for tag, r in (("ref", ref), ("arm", arm)):
        if not c1_ok(r):
            why.append(f"C1 not clean on {tag}")
    if arm.get("n_patched", 0) == 0:
        why.append("n_patched == 0")
    need = 2 * arm.get("n_patched", 0)
    if arm.get("kernel_calls_per_step_min", 0) < need:
        why.append(f"kernel calls/step min {arm.get('kernel_calls_per_step_min')} < 2*n_patched={need} (not engaged on every layer)")
    if len(arm.get("losses", [])) != len(ref.get("losses", [])) or not arm.get("losses"):
        why.append("step counts differ")
    if why:
        return "VOID", None, None, "; ".join(why)
    d_final = abs(arm["loss_last"] - ref["loss_last"])
    med = statistics.median(abs(x - y) for x, y in zip(arm["losses"], ref["losses"]))
    return ("PASS" if (d_final <= BAND and med <= BAND) else "FAIL"), d_final, med, ""


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    d = sys.argv[1]
    fams = sorted({os.path.basename(p).split("_train_")[0] for p in glob.glob(os.path.join(d, "*_train_*.json"))})
    print(f"MG1 reduce: {d}  (BAND {BAND}, tp1 v2.1 verdict)")
    print(f"{'family':10} {'ref':10} {'fused':10} {'verdict':8} {'d_final':>8} {'med':>8} {'ref s/st':>9} {'fus s/st':>9} "
          f"{'ref GB':>7} {'fus GB':>7} {'n_patched':>9}  notes")
    for f in fams:
        ref, fus = load(os.path.join(d, f"{f}_train_reference.json")), load(os.path.join(d, f"{f}_train_fused.json"))
        rs, fs = (ref or {}).get("status", "missing"), (fus or {}).get("status", "missing")
        v, dfin, med, why = ("VOID", None, None, "fused arm not OK") if fs != "ok" else verdict(ref, fus)
        fmt = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"  # noqa: E731
        print(f"{f:10} {rs:10} {fs:10} {v:8} {fmt(dfin):>8} {fmt(med):>8} {fmt((ref or {}).get('s_per_step'), 3):>9} "
              f"{fmt((fus or {}).get('s_per_step'), 3):>9} {fmt((ref or {}).get('peak_vram_gb'), 2):>7} "
              f"{fmt((fus or {}).get('peak_vram_gb'), 2):>7} {str((fus or {}).get('n_patched', '-')):>9}  "
              f"{why or ((fus or {}).get('reason') or (ref or {}).get('reason') or '')}")
    for p in sorted(glob.glob(os.path.join(d, "*_ladder.json"))):
        lad = load(p) or {}
        s = lad.get("summary")
        if not s:
            print(f"{os.path.basename(p)}: no summary (rungs read: {[r.get('rung') for r in lad.get('rungs', [])]})")
            continue
        print(f"{os.path.basename(p)}: " + "; ".join(
            f"{k} {v['s_per_step_mean']} s/step (x{v.get('ratio_to_fused')}, spread {v.get('spread')}, {v['peak_alloc_gb']} GB)"
            for k, v in s.items()))


if __name__ == "__main__":
    main()
