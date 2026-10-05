"""bench/dq3/dq3_rehearsal_check.py -- lane DQ3's A2000 rehearsal gate (Amendment 3): correctness and the lane's
QUANTITY, never timing. ``python dq3_rehearsal_check.py <dir>`` reads the rehearsal's arm receipts (A<k>_<arm>.json:
6 layers, hidden 1024, R S S0 S0 S R; B<k>_<arm>.json: 4 layers at Qwen3-32B width, R S S0) and exits 1 unless every
arm finished, parity is bitwise across arms, and each streamed arm's allocator peak sits (L-2) layers' streamed bytes
below the resident arm's (half a layer of slack). The last check is why this exists: the first rehearsal printed
S 6.47 > R 5.82 GiB and passed, and a rented 5090 found it (dq3-5090-3).
"""
from __future__ import annotations

import glob
import json
import sys


def check(arms_by_shape: dict) -> tuple[list, list]:
    """``{shape: [(name, receipt), ...]}`` -> (report lines, failures)."""
    lines, bad = [], []
    for shape, arms in arms_by_shape.items():
        if not arms:
            bad.append(f"{shape}: no receipts")
            continue
        unfinished = [n for n, a in arms if "timing" not in a]
        if unfinished:
            bad.append(f"{shape}: arm(s) did not finish {unfinished}")
        arms = [(n, a) for n, a in arms if "timing" in a]
        r_peaks = [a["timing"]["peak_alloc"] for _n, a in arms if a["arm"] == "R"]
        if not r_peaks:
            bad.append(f"{shape}: no finished R arm to compare against")
            continue
        ref = next(a for _n, a in arms if a["arm"] == "R")["parity"]
        n_layers = arms[0][1]["config_overrides"]["num_hidden_layers"]
        for name, a in arms:
            t = a["timing"]
            same = all(p["loss_bits"] == r["loss_bits"] and p["grads"] == r["grads"] for p, r in zip(a["parity"], ref))
            line = f"{shape} {name:10s} parity_eq={same} peak={t['peak_alloc'] / 2**30:.3f}GiB"
            if not same:
                bad.append(f"{name}: parity")
            if a["arm"] in ("S", "S0"):
                per = a["offload"]["per_layer_bytes"]
                bound = min(r_peaks) - (n_layers - 2) * per + per // 2
                ok = t["peak_alloc"] <= bound
                line += f" | memory: <= {bound / 2**30:.3f} GiB: {ok}"
                if not ok:
                    bad.append(f"{name}: MEMORY DIRECTION (peak not (L-2) layers below resident)")
            lines.append(line)
    return lines, bad


def main(d: str) -> int:
    by_shape = {s: [(f.rsplit("/", 1)[1], json.load(open(f))) for f in sorted(glob.glob(f"{d}/{s}[0-9]*.json"))]
                for s in ("A", "B")}
    lines, bad = check(by_shape)
    print("\n".join(lines))
    print("REHEARSAL CHECK", "OK" if not bad else f"FAIL {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
