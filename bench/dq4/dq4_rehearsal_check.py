"""bench/dq4/dq4_rehearsal_check.py -- lane DQ4's A2000 rehearsal gate: correctness and the lane's QUANTITY, never timing.
``python dq4_rehearsal_check.py <dir>`` reads the rehearsal's ladder receipts (``r_R.json``, ``r_S.json``: 8 layers at
Qwen3-32B width, chunked loss) and exits 1 unless both ladders finished with finite losses, the chunked loss engaged in
both, S streamed every layer through the late-bound backward, and S's boundary sits ABOVE R's -- the direction the lane
exists to measure (DQ3's lesson: a rehearsal that only checks that things run passes while that number goes the wrong way).
"""
from __future__ import annotations

import json
import os
import sys


def check(r: dict, s: dict, layers: int = 8) -> tuple[list, list]:
    lines, bad = [], []
    for tag, a in (("R", r), ("S", s)):
        if not a.get("finished_at"):
            bad.append(f"{tag}: did not finish")
        if not (a.get("chunked_loss") or {}).get("patched"):
            bad.append(f"{tag}: the chunked loss did not engage")
        if any(r_["ok"] and not all(st["finite"] for st in r_["steps"]) for r_ in a.get("rungs", [])):
            bad.append(f"{tag}: non-finite loss")
        lines.append(f"{tag}: L* {a.get('max_ok')} first OOM {a.get('first_oom')}")
    off = s.get("offload") or {}
    if off.get("layers") != layers or off.get("late_bound_4bit") != 7 * layers:
        bad.append(f"S: offload {off} (want {layers} layers, late_bound_4bit {7 * layers})")
    lr, ls = r.get("max_ok"), s.get("max_ok")
    if lr is None or ls is None or ls <= lr:
        bad.append(f"CAPACITY DIRECTION: L*_S {ls} is not above L*_R {lr}")
    if r.get("first_oom") is None:
        bad.append("R reached the ladder's top: raise --top for the rehearsal")
    return lines, bad


def main(d: str) -> int:
    r = json.load(open(os.path.join(d, "r_R.json")))
    s = json.load(open(os.path.join(d, "r_S.json")))
    lines, bad = check(r, s)
    print("\n".join(lines))
    print("REHEARSAL CHECK", "OK" if not bad else f"FAIL {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
