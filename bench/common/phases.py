#!/usr/bin/env python3
"""phases.py -- read and check a rental box's ``phases.jsonl`` (bench/common/README.md, "Phase stamps"). Stdlib only.

    python bench/common/phases.py PATH            # the summary as JSON
    python bench/common/phases.py --check PATH    # exit 1, with one line per fault, on any format fault

A line is ``{"phase": NAME, "event": "start"|"end", "t": EPOCH_SECONDS}``; a start line may carry ``"kind"``: setup,
measure, reduce or other. Lines written before the kind existed (P149's) carry none: their kind reads ``unknown``, never
a guess from the name. Phases run one at a time: each start is closed by its own end before the next starts, and ``t``
never goes backwards. A phase still open at the end of the file (the box died, or the run is live) is reported, not
dropped.
"""
import json
import re
import sys

KINDS = ("setup", "measure", "reduce", "other")
NAME = re.compile(r"^[A-Za-z0-9_.:-]+$")


def read(lines):
    """(phases, faults). Each phase: {name, kind, start, end, seconds}; ``end``/``seconds`` are None while open."""
    phases, faults, open_ = [], [], None
    last_t = None
    for n, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            faults.append(f"line {n}: not JSON")
            continue
        if not isinstance(rec, dict):
            faults.append(f"line {n}: not an object")
            continue
        name, event, t = rec.get("phase"), rec.get("event"), rec.get("t")
        if not isinstance(name, str) or not NAME.match(name):
            faults.append(f"line {n}: bad phase name {name!r}")
            continue
        if not isinstance(t, int) or isinstance(t, bool):
            faults.append(f"line {n}: t is not an integer epoch")
            continue
        if last_t is not None and t < last_t:
            faults.append(f"line {n}: t goes backwards ({t} < {last_t})")
        last_t = t if last_t is None else max(last_t, t)
        if event == "start":
            kind = rec.get("kind", "unknown")
            if kind != "unknown" and kind not in KINDS:
                faults.append(f"line {n}: bad kind {kind!r}")
                kind = "unknown"
            if open_ is not None:
                faults.append(f"line {n}: {name} starts while {open_['name']} is open")
            open_ = {"name": name, "kind": kind, "start": t, "end": None, "seconds": None}
            phases.append(open_)
        elif event == "end":
            if open_ is None or open_["name"] != name:
                faults.append(f"line {n}: {name} ends but {open_['name'] if open_ else 'nothing'} is open")
                continue
            open_["end"] = t
            open_["seconds"] = t - open_["start"] if t >= open_["start"] else None   # backwards: a fault, no seconds
            open_ = None
        else:
            faults.append(f"line {n}: bad event {event!r}")
    return phases, faults


def summary(phases):
    """Seconds per kind (closed phases only), the first measurement's start, and the phases left open."""
    out = {k: 0 for k in KINDS + ("unknown",)}
    for p in phases:
        if p["seconds"] is not None:
            out[p["kind"]] += p["seconds"]
    first = [p for p in phases if p["kind"] == "measure"]
    return {"seconds_by_kind": out, "phases": len(phases),
            "first_measure_start": first[0]["start"] if first else None,
            "first_stamp": phases[0]["start"] if phases else None,
            "last_stamp": max((p["end"] or p["start"]) for p in phases) if phases else None,
            "open": [p["name"] for p in phases if p["end"] is None]}


def main(argv):
    check = "--check" in argv
    paths = [a for a in argv if a != "--check"]
    if len(paths) != 1:
        print(__doc__.strip().splitlines()[2])
        return 2
    with open(paths[0]) as f:
        phases, faults = read(f)
    if check:
        for line in faults:
            print(line)
        return 1 if faults else 0
    print(json.dumps({**summary(phases), "faults": faults}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
