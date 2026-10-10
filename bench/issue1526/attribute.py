"""Live allocations at the allocated peak of the recorded step, grouped by the first frame in e4b/gnf4/transformers/torch."""
import json
import pickle
import sys

MiB = 1 << 20
KEEP = ("experts4bit_qlora/", "nf4_", "transformers/", "bitsandbytes/", "torch/nn/", "torch/utils/checkpoint")


def site(frames):
    for fr in frames or ():
        fn = fr.get("filename", "")
        if any(k in fn for k in KEEP) and "torch/cuda/memory" not in fn:
            short = fn.split("site-packages/")[-1]
            return f"{short}:{fr.get('line')} {fr.get('name')}"
    return "(no frame)"


for path in sys.argv[1:]:
    snap = pickle.load(open(path, "rb"))
    trace = snap["device_traces"][0]
    end = {}
    for seg in snap["segments"]:
        a = seg["address"]
        for b in seg["blocks"]:
            if b["state"] == "active_allocated":
                end[a] = (b["size"], b.get("frames") or [])
            a += b["size"]
    allocated_in_trace = {e["addr"] for e in trace if e["action"] == "alloc"}
    freed_before_alloc = set()
    seen = set()
    for e in trace:
        if e["action"] == "alloc":
            seen.add(e["addr"])
        elif e["action"] in ("free_requested", "free_completed") and e["addr"] not in seen:
            freed_before_alloc.add((e["addr"], e["size"]))
    live = {a: v for a, v in end.items() if a not in allocated_in_trace}       # predates the recording, still live
    for a, s in freed_before_alloc:
        live.setdefault(a, (s, []))
    total = sum(v[0] for v in live.values())
    peak, at = total, dict(live)
    for e in trace:
        if e["action"] == "alloc":
            live[e["addr"]] = (e["size"], e.get("frames") or [])
            total += e["size"]
            if total > peak:
                peak, at = total, dict(live)
        elif e["action"] == "free_completed" and e["addr"] in live:
            total -= live.pop(e["addr"])[0]
    groups = {}
    for s, fr in at.values():
        k = site(fr) if fr else "(allocated before the recorded step)"
        groups[k] = groups.get(k, 0) + s
    print(f"== {path}: replayed peak {peak / MiB:.1f} MiB")
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1])[:18]:
        print(f"  {v / MiB:9.1f} MiB  {k}")
