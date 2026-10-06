"""bench/dq6/dq6_reduce.py -- lane DQ6's registered rule (bench/dq6/DQ6-PREREG.md): DQ4's rule (``bench/dq4/dq4_reduce.py``,
staged beside this file), unchanged, with two registered differences:

* the registered device is the RTX 4090 (a 5090 receipt is VOID here, as a 4090 receipt is VOID under DQ4);
* every receipt must come from a **24 GB** card: ``total_bytes`` in [23, 24.5] GiB, else VOID. A 48 GB-modded 4090 reports
  the same name, and the runner's gate is the first line of defence; this is the second.

``python dq6_reduce.py <dir>`` or ``python dq6_reduce.py --self-test`` (DQ4's 25 cases on the 4090, plus DQ6's own).
"""
from __future__ import annotations

import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "dq4")]     # staged flat on the box; bench/dq4/ in the repo
import dq4_reduce as R  # noqa: E402

REGISTERED_DEVICE = "NVIDIA GeForce RTX 4090"
MIN_TOTAL, MAX_TOTAL = 23.0 * 2**30, 24.5 * 2**30
SYNTH_TOTAL = 23_75 * 2**30 // 100     # self-test receipts only: a synthetic total inside the band, not a measured 4090

R.REGISTERED_DEVICE = REGISTERED_DEVICE
_dq4_void, _dq4_synth = R._void, R.synth_receipt


def _void(tag, a, loss, alloc):
    out = _dq4_void(tag, a, loss, alloc)
    tb = a.get("total_bytes")
    if not (isinstance(tb, int) and MIN_TOTAL <= tb <= MAX_TOTAL):
        out.append(f"{tag}: total memory {tb!r} is not a 24 GB card")
    return out


def _synth(*a, total=SYNTH_TOTAL, **kw):
    r = _dq4_synth(*a, **kw)
    r["total_bytes"] = total
    return r


R._void, R.synth_receipt = _void, _synth


def reduce_dir(d: str) -> dict:
    out = R.reduce_dir(d)
    out["schema"] = "dq6-read/1"
    out["registered_device"] = REGISTERED_DEVICE
    return out


def self_test() -> int:
    if R.self_test() != 0:
        return 1
    fails = []
    V = lambda o: o["verdict"]  # noqa: E731

    def case(name, rec, check):
        try:
            o = R.reduce_config("c_def", rec)
        except Exception as exc:  # a crash is a failure of the self-test
            fails.append(f"{name}: crashed {exc!r}")
            return
        if not check(o):
            fails.append(f"{name}: {o['verdict']} G={o['G']} void={o['void'][:2]}")

    case("a 5090 receipt is VOID under DQ6", R.synth_config(device="NVIDIA GeForce RTX 5090"), lambda o: V(o) == "VOID")
    case("a 48 GB card is VOID", R.synth_config(total=48 * 2**30), lambda o: V(o) == "VOID")
    case("a missing total is VOID", R.synth_config(total=None), lambda o: V(o) == "VOID")
    rec = R.synth_config()
    rec["S_ok"]["total_bytes"] = 32 * 2**30
    case("one confirmation from another card is VOID", rec, lambda o: V(o) == "VOID")

    def ladder(r_oom, s_oom, start=512, step=512, top=32768):
        seqs = range(start, top + 1, step)
        rec = {"R_ladder": R.synth_receipt("R", "ladder", seqs=seqs, oom_at=r_oom),
               "S_ladder": R.synth_receipt("S", "ladder", seqs=seqs, oom_at=s_oom)}
        for arm, oom in (("R", r_oom), ("S", s_oom)):
            if rec[f"{arm}_ladder"]["max_ok"] is not None:
                rec[f"{arm}_ok"] = R.synth_receipt(arm, "confirm", seq=rec[f"{arm}_ladder"]["max_ok"])
            rec[f"{arm}_oom"] = R.synth_receipt(arm, "confirm", seq=oom, oom_at=oom)
        return rec

    case("the predicted read (R 1536, S 9728 on a 512 ladder) is CAP_REAL", ladder(2048, 10240),
         lambda o: V(o) == "CAP_REAL" and o["L_R"] == 1536 and o["L_S"] == 9728 and abs(o["G"] - 9728 / 1536) < 1e-9)
    case("R passing only the first rung still grades", ladder(1024, 4096),
         lambda o: V(o) == "CAP_REAL" and o["L_R"] == 512 and abs(o["G"] - 7.0) < 1e-9)
    case("R at the first rung of a 512 ladder is VOID", ladder(512, 4096), lambda o: V(o) == "VOID")
    if fails:
        print("dq6 self-test FAILED:")
        print("\n".join(fails))
        return 1
    print("dq6 self-test OK (DQ4's 25 cases on the RTX 4090 + 7)")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        raise SystemExit(self_test())
    json.dump(reduce_dir(sys.argv[1]), sys.stdout, indent=1,
              default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
    print()
