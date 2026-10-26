"""TC1 amendment 83's $0 evidence scan: every per-second clock log in the private receipt store, read by card_clock.py's own parser
(amendment 80's bytes, the rule being amended), split by load, GPU class, busy median against the floor, and the recorded power limit.
It prints counts and quantiles only: no run id, machine, instance or GPU id leaves it (``--detail`` writes a local per-log file that does
name runs; it is never committed).

    python3 am83_scan.py STORE_RECEIPTS_DIR CARD_CLOCK_PY [--detail OUT.json] > am83_scan.json

- TRAINING: every ``gpuclk_*.txt`` (TC1 amendment 7's per-second sampler: ``EPOCH sm, mem, temp, reasons | ...``), at least 10 busy
  samples (SM above card_clock's BUSY_MHZ). The GPU class and power limit come from the run's forensics.txt (or box.json).
- SERVING: no per-second log with reason bits exists; the ``gpu_phases.csv`` phase start/end SNAPSHOTS are counted as rows.
"""
import csv
import glob
import importlib.util
import json
import os
import re
import statistics
import sys
from functools import reduce

ROOT, CC = sys.argv[1], sys.argv[2]
DETAIL = sys.argv[sys.argv.index("--detail") + 1] if "--detail" in sys.argv else None
spec = importlib.util.spec_from_file_location("card_clock", CC)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
FLOOR = {"RTX 5090": 2700, "RTX 4090": 2500}       # the floors the lanes register (TC1 amendment 80's 5090; P237's 4090)
OTHER_HARD = cc.HARD_BITS & ~0x400                 # every amendment 80 hard bit but Reliability: the slowdowns and BoardLimit
REL, SWPC = 0x400, 0x4


def run_dir(path):
    parts = path.split(os.sep)
    i = parts.index("receipts")
    return os.sep.join(parts[: i + 4])             # .../receipts/<project>/<date>/<run>


def card(rd):
    """(GPU class, power limit W) from the run's forensics.txt, else its box.json; ("unknown", None) when neither names one."""
    for f in glob.glob(os.path.join(rd, "*", "forensics.txt")) + glob.glob(os.path.join(rd, "forensics.txt")):
        try:
            first = open(f, encoding="utf-8", errors="replace").readline()
        except OSError:
            continue
        m = re.search(r"RTX (\d{4})", first)
        if m:
            w = re.search(r",\s*([\d.]+) W", first)
            return f"RTX {m.group(1)}", float(w.group(1)) if w else None
    for f in glob.glob(os.path.join(rd, "*", "box.json")):
        try:
            m = re.search(r"RTX (\d{4})", json.load(open(f)).get("name", ""))
        except (OSError, ValueError, AttributeError):
            continue
        if m:
            return f"RTX {m.group(1)}", None
    return "unknown", None


rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "**", "gpuclk_*.txt"), recursive=True)):
    samples = [r[1:] for r in (cc.parse_gpuclk_line(ln) for ln in open(path, encoding="utf-8", errors="replace")) if r is not None]
    busy = [s for s in samples if s[0] > cc.BUSY_MHZ]
    if len(busy) < 10:
        continue
    rd = run_dir(path)
    gpu, plim = card(rd)
    med = statistics.median(s[0] for s in busy)
    other = reduce(lambda a, s: a | (s[3] & OTHER_HARD), samples, 0)
    rows.append({"run": os.path.basename(rd), "file": os.path.basename(path), "gpu": gpu, "power_limit_w": plim, "busy": len(busy),
                 "sm_median": med, "above_floor": None if gpu not in FLOOR else med >= FLOOR[gpu],
                 "swpc_share": sum(1 for s in busy if s[3] & SWPC) / len(busy),
                 "rel_share": sum(1 for s in busy if s[3] & REL) / len(busy),
                 "other_hard": cc.decode(other)})


def q(xs):
    xs = sorted(xs)
    if not xs:
        return None
    pick = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]  # noqa: E731
    return {"n": len(xs), "min": round(xs[0], 4), "p10": round(pick(0.1), 4), "median": round(pick(0.5), 4),
            "p90": round(pick(0.9), 4), "max": round(xs[-1], 4)}


def group(rs):
    return {"logs": len(rs), "runs": len({r["run"] for r in rs}), "busy_samples": sum(r["busy"] for r in rs),
            "sm_median": q([r["sm_median"] for r in rs]),
            "swpowercap_share": q([r["swpc_share"] for r in rs]),
            "logs_swpowercap_over_5pct": sum(r["swpc_share"] > 0.05 for r in rs),
            "logs_with_any_reliability": sum(r["rel_share"] > 0 for r in rs),
            "reliability_share_where_set": q([r["rel_share"] for r in rs if r["rel_share"] > 0]),
            "logs_reliability_over_half": sum(r["rel_share"] > 0.5 for r in rs),
            "logs_other_hard_any": sum(bool(r["other_hard"]) for r in rs),
            "other_hard_bits_seen": sorted({b for r in rs for b in r["other_hard"]})}


table = {"source": "the private receipt store's per-second clock logs, card_clock.py at the bytes given",
         "card_clock_sha256": __import__("hashlib").sha256(open(CC, "rb").read()).hexdigest(),
         "training": {"logs_read": len(rows), "groups": {}}, "serving": {}}
for gpu in sorted({r["gpu"] for r in rows}):
    for above in (True, False, None):
        rs = [r for r in rows if r["gpu"] == gpu and r["above_floor"] is above]
        if not rs:
            continue
        key = f"{gpu} | busy median {'>=' if above else '<'} {FLOOR[gpu]}" if above is not None else f"{gpu} | no registered floor"
        table["training"]["groups"][key] = group(rs)
        if above and gpu == "RTX 5090":
            for name, keep in ((">= 575 W", lambda w: w is not None and w >= 575), ("< 575 W", lambda w: w is not None and w < 575),
                               ("power limit unread", lambda w: w is None)):
                sub = [r for r in rs if keep(r["power_limit_w"])]
                if sub:
                    table["training"]["groups"][f"{key} | {name}"] = group(sub)

snap = []
for f in glob.glob(os.path.join(ROOT, "**", "gpu_phases.csv"), recursive=True):
    for r in csv.DictReader(open(f, encoding="utf-8", errors="replace")):
        try:
            snap.append((f, int(float(r["clocks.sm"])), int(r["clocks_event_reasons.active"].strip(), 16), float(r["power.draw"])))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
busy = [s for s in snap if s[1] > cc.BUSY_MHZ]
hi = [s for s in busy if s[1] >= 2700]
table["serving"] = {"per_second_logs_with_reasons": 0, "phase_snapshot_files": len({s[0] for s in snap}), "snapshot_rows": len(snap),
                    "busy_rows": len(busy), "busy_rows_ge_2700": len(hi),
                    "ge_2700_with_swpowercap": sum(1 for s in hi if s[2] & SWPC), "ge_2700_with_reliability": sum(1 for s in hi if s[2] & REL),
                    "ge_2700_with_other_hard": sum(1 for s in hi if s[2] & OTHER_HARD),
                    "busy_power_draw_median_w": statistics.median(s[3] for s in busy) if busy else None}
print(json.dumps(table, indent=1, sort_keys=True))
if DETAIL:
    json.dump(rows, open(DETAIL, "w"), indent=1)
