"""Box-acceptance gate for TRAINING receipts.

Serving has kernel/decode_anchor.py; training had nothing, so a slow BOX and a
slow BUILD were indistinguishable (3.7 vs 6.1 s/step, same config, 2026-08-26/27).
This is the committed reference and the REFUSAL rule.

    python train_anchor_gate.py anchor.json                  # class one box
    python train_anchor_gate.py --calibrate a.json b.json c.json

Exit 0 = accepted. Exit 3 = REFUSED; do not run measured arms on it.
"""
import json, statistics, sys

# ---------------------------------------------------------------------------
# CALIBRATED 2026-08-27 across 3 distinct RTX 5090 machines (9105, 137831,
# 137833). The population is BIMODAL, not a single dispersed class:
#
#   box       power   launches/s   h2d GB/s
#   137831    525 W     128,453      25.95
#   137833    525 W     129,370      25.97      <- agree to 0.7% / 0.1%
#   9105      600 W     100,226      13.35      <- 0.78x / 0.51x
#
# So a tolerance BAND is the wrong instrument here: dispersion/2 on h2d gives
# +/-49%, which accepts everything. FLOPs *are* tight (2.5% across all three)
# and do get a band -- but they are also the probe that does NOT discriminate,
# which is exactly why a FLOPs-only gate would have passed both boxes that
# later differed 1.65x in training step time.
#
# Instead: launch rate and PCIe bandwidth are CLASS LABELS. A receipt records
# the class it was measured on, and comparing across classes is invalid.
# ---------------------------------------------------------------------------
FLOPS_REF = 185.19          # median of 3 boxes, dispersion 1.025x
FLOPS_BAND = 0.05           # generous vs the 2.5% observed

# Class boundaries sit in the empty space between the two observed modes.
PCIE_FULL_MIN = 20.0        # observed modes: 13.35 and ~25.96
LAUNCH_FAST_MIN = 115000    # observed modes: 100,226 and ~128,900

SELF_PAIR_MAX = 1.03        # a box that disagrees with ITSELF is unusable


def box_class(a):
    """(pcie, launch) class labels. Cross-class comparison is not valid."""
    h = a["h2d"]["h2d_gb_s"]["median"]
    l = a["launch"]["launches_per_s"]["median"]
    return ("pcie-full" if h >= PCIE_FULL_MIN else "pcie-half",
            "launch-fast" if l >= LAUNCH_FAST_MIN else "launch-slow")


def _med(d, probe, key):
    return d[probe][key]["median"]


def classify(a, expect_class=None):
    """expect_class: ("pcie-full","launch-fast") the receipt was measured on.
    None = just report the class, refuse only on self-consistency/FLOPs."""
    out, fatal = [], False
    if a.get("status") != "OK":
        return [("status", a.get("status"), "FATAL")], True
    if a.get("gpu") != "NVIDIA GeForce RTX 5090":
        out.append(("gpu", a.get("gpu"), "FATAL: wrong GPU")); fatal = True

    # 1. Self-consistency. Needs no reference; invalidates a box on its own
    #    terms. The 600 W outlier failed here (launch spread 1.116) as well as
    #    on class -- unstable and slow travel together.
    for probe, key in (("flops", "tflops"), ("launch", "launches_per_s"), ("h2d", "h2d_gb_s")):
        sp = a[probe][key]["spread"]
        ok = sp is not None and sp <= SELF_PAIR_MAX
        out.append((probe + ".self_pair", sp, "ok" if ok else f"FATAL: >{SELF_PAIR_MAX}"))
        if not ok:
            fatal = True

    # 2. FLOPs band -- the one probe tight enough to band.
    tf = a["flops"]["tflops"]["median"]
    rel = tf / FLOPS_REF
    ok = (1 - FLOPS_BAND) <= rel <= (1 + FLOPS_BAND)
    out.append(("flops", f"{tf} ({rel:.3f}x ref)", "ok" if ok else "REFUSE: out of band"))
    if not ok:
        fatal = True

    # 3. Class labels -- reported always, enforced only against a declared class.
    cls = box_class(a)
    out.append(("class", "/".join(cls), "reported"))
    if expect_class is not None:
        match = tuple(expect_class) == cls
        out.append(("class_match", f"expected {'/'.join(expect_class)}",
                    "ok" if match else "REFUSE: cross-class comparison is invalid"))
        if not match:
            fatal = True
    return out, fatal


def calibrate(paths):
    ds = [json.load(open(p)) for p in paths]
    ds = [d for d in ds if d.get("status") == "OK"]
    if len(ds) < 3:
        print(f"WARNING: {len(ds)} box(es); >=3 needed to separate box dispersion "
              f"from run noise. Band below is NOT trustworthy yet.\n")
    res = {}
    for probe, key in (("flops", "tflops"), ("launch", "launches_per_s"), ("h2d", "h2d_gb_s")):
        vals = [_med(d, probe, key) for d in ds]
        disp = max(vals) / min(vals)
        res[key] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals),
                    "dispersion": round(disp, 4),
                    "suggested_band": round((disp - 1) / 2 + 0.02, 4)}
        print(f"{key:>16}: median={res[key]['median']} min={min(vals)} max={max(vals)} "
              f"dispersion={disp:.3f}x  band>={res[key]['suggested_band']}")
    print("\nBand must be >= observed dispersion/2, else it refuses in-class boxes.")
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--calibrate":
        calibrate(args[1:]); sys.exit(0)
    if not args:
        sys.exit("usage: train_anchor_gate.py anchor.json [pcie-full/launch-fast]\n"
                 "       train_anchor_gate.py --calibrate a.json b.json c.json")
    expect = tuple(args[1].split("/")) if len(args) > 1 else None
    rows, fatal = classify(json.load(open(args[0])), expect)
    for k, v, verdict in rows:
        print(f"  {k:<20} {str(v):<30} {verdict}")
    print("\nBOX REFUSED" if fatal else "\nBOX ACCEPTED")
    sys.exit(3 if fatal else 0)
