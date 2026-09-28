"""Compare the v5 runs: p(' Paris') to all printed digits, prefill expert rows, and the MoE trace.

For every pair of runs it finds the FIRST MoE call (layer order) whose hashes differ, and says
which of input x / router ids / router weights / output differs there. If x, ids and w agree
and out does not, the engine itself returned different bits for identical inputs. If x differs
while every earlier output agreed, the divergence started outside the MoE engines.
Usage: python compare_v5.py RUN_DIR ...   (writes nothing; prints)
"""
import itertools
import json
import os
import sys

runs = {}
for d in sys.argv[1:]:
    name = os.path.basename(d.rstrip("/"))
    try:
        res = json.load(open(os.path.join(d, "k3_gen_n1_pin0.json")))
        tr = json.load(open(os.path.join(d, "moe_trace.json")))
    except FileNotFoundError as e:
        print(f"{name}: missing {e.filename}")
        continue
    runs[name] = (res, tr)

print(f"{'run':8s} {'p(Paris) repr':22s} {'rows':>5s} {'det':>5s} {'shadow':>6s} calls  mxfp4_pipelined sha256[:12]  prefill_s")
for n, (res, tr) in runs.items():
    print(f"{n:8s} {res['probs'][0]!r:22s} {res['prefill_counters']['disk_reads']:5d} "
          f"{str(tr['deterministic']):>5s} {('yes' if tr['shadow'] else 'no'):>6s} {len(tr['calls']):5d}  "
          f"{tr['mxfp4_pipelined']['sha256'][:12]}                 {res['prefill_s']:.1f}")


def first_diff(a, b):
    ca, cb = a["calls"], b["calls"]
    if len(ca) != len(cb):
        return f"call counts differ {len(ca)} vs {len(cb)}"
    outs_differ = sum(x["out"] != y["out"] for x, y in zip(ca, cb))
    for i, (x, y) in enumerate(zip(ca, cb)):
        fields = [f for f in ("x", "ids", "w", "out") if x[f] != y[f]]
        if fields:
            where = ("ENGINE: identical x/ids/w, different out" if fields == ["out"] else
                     "UPSTREAM of the engine: x differs" if "x" in fields else
                     "router differs with identical x")
            return (f"first diff at call {i} (layer {x['layer']}): {','.join(fields)} -> {where}; "
                    f"outputs differ at {outs_differ}/{len(ca)} calls")
    return "IDENTICAL: every call's x, ids, w and out"


print()
for (na, (ra, ta)), (nb, (rb, tb)) in itertools.combinations(runs.items(), 2):
    same_p = ra["probs"][0] == rb["probs"][0]
    same_r = ra["prefill_counters"]["disk_reads"] == rb["prefill_counters"]["disk_reads"]
    print(f"{na:>7s} vs {nb:7s} p {'==' if same_p else '!='}  rows {'==' if same_r else '!='}  "
          f"{first_diff(ta, tb)}")
