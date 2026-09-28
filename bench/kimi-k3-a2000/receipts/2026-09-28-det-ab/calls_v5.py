"""Per-call detail for two v5 runs: every MoE call where any of x / ids / w / out differs, and
for each, whether the engine's INPUTS agreed (a fresh engine-level difference) or not
(inherited from upstream). Usage: python calls_v5.py RUN_A RUN_B"""
import json
import os
import sys

a, b = (json.load(open(os.path.join(d, "moe_trace.json")))["calls"] for d in sys.argv[1:3])
fresh = inherited = 0
for i, (x, y) in enumerate(zip(a, b)):
    f = [k for k in ("x", "ids", "w", "out") if x[k] != y[k]]
    if f:
        kind = "fresh (inputs equal)" if f == ["out"] else "inherited"
        fresh += f == ["out"]
        inherited += f != ["out"]
        print(f"  call {i:2d} layer {x['layer']:2d}: {','.join(f):14s} {kind}")
print(f"{os.path.basename(sys.argv[1])} vs {os.path.basename(sys.argv[2])}: "
      f"{fresh} fresh engine-level differences, {inherited} calls with differing inputs")
