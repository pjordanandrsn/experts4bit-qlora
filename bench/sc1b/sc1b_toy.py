#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc1b_toy.py -- lane SC1b (#846) proof item 2: does Nsight Systems record what the census reads, the way the census reads it?

`run`: a SPAWNED child (as vLLM's EngineCore and SGLang's scheduler are) fills >= 90 % of free GPU memory, captures a
3-kernel CUDA graph (x*2, +1, relu: three elementwise kernels) BEFORE calling cudaProfilerStart (every engine captures at
warm-up), then replays it REPLAYS times inside the cudaProfilerApi range -- each replay followed by one EAGER kernel and one
host-to-device copy, as every engine's out-of-graph step work is -- and calls cudaProfilerStop.
`check G.sqlite N.sqlite`: through the census reducer's own loader --
  graph mode: REPLAYS GRAPH_TRACE rows, one graphExecId, the column present;
  node mode: every cudaGraphLaunch correlation groups exactly 3 kernels, each with a graphNodeId and a non-zero grid;
  both modes: the eager kernels and the H2D copies land in the reducer's NON-graph set (graphNodeId NULL or 0; round 2 M7).

  nsys profile ... --cuda-graph-trace=graph|node --capture-range=cudaProfilerApi -o T python sc1b_toy.py run
  sc1b_toy.py check G.sqlite N.sqlite
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

REPLAYS = 20


def child(replays: int = REPLAYS):
    import torch
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    hog = torch.empty(max(0, int(free * 0.92) - (512 << 20)), dtype=torch.uint8, device="cuda")     # >= 90 % in use
    x = torch.randn(1 << 20, device="cuda")

    def f(t):
        return torch.relu(t * 2.0 + 1.0)
    host = torch.randn(1 << 16).pin_memory()
    dev = torch.empty(1 << 16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            f(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = f(x)
    torch.cuda.synchronize()
    used = 1 - torch.cuda.mem_get_info()[0] / total
    torch.cuda.profiler.start()
    for _ in range(replays):
        g.replay()
        x.mul_(1.0)                                     # one eager kernel
        dev.copy_(host, non_blocking=True)              # one H2D copy
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("SC1B_TOY " + json.dumps({"replays": replays, "mem_used_fraction": round(used, 3), "y0": float(y[0]), "hog_gb": round(hog.numel() / 2**30, 1)}), flush=True)


def check(graph_db, node_db, replays: int = REPLAYS):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sc1b_census as C
    con = sqlite3.connect(f"file:{graph_db}?mode=ro", uri=True)
    cols = [r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_GRAPH_TRACE)")]
    con.close()
    out = {"graph_trace_columns_ok": "graphExecId" in cols}
    g = C.load(graph_db)
    out["graph_rows"] = len(g["graphs"])
    out["graph_execs"] = len({r["exec"] for r in g["graphs"]})
    n = C.load(node_db)
    by = {}
    for k in n["kernels"]:
        if not C._nongraph(k):
            by.setdefault(k["corr"], []).append(k)
    launches = [x for x in n["launches"] if x["corr"] in by]
    out["node_launches"] = len(launches)
    out["kernels_per_launch"] = sorted({len(by[x["corr"]]) for x in launches})
    out["all_have_node_id_and_grid"] = all(k["node"] is not None and all(v > 0 for v in k["grid"]) for ks in by.values() for k in ks)
    for tag, ex in (("graph", g), ("node", n)):
        out[f"{tag}_eager_kernels"] = sum(1 for k in ex["kernels"] if C._nongraph(k))
        out[f"{tag}_h2d_copies"] = sum(1 for c in ex["copies"] if C._nongraph(c) and c["kind"] == "memcpy" and not c["d2h"])
    eager_ok = all(out[f"{t}_eager_kernels"] >= replays and out[f"{t}_h2d_copies"] >= replays for t in ("graph", "node"))
    out["eager_work_is_non_graph"] = eager_ok
    ok = (out["graph_trace_columns_ok"] and out["graph_rows"] == replays and out["graph_execs"] == 1
          and out["node_launches"] == replays and out["kernels_per_launch"] == [3] and out["all_have_node_id_and_grid"] and eager_ok)
    out["ok"] = ok
    print("SC1B_TOY_CHECK " + json.dumps(out), flush=True)
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        import multiprocessing as mp
        p = mp.get_context("spawn").Process(target=child)
        p.start()
        p.join()
        return p.exitcode or 0
    if argv[:1] == ["check"] and len(argv) == 3:
        return check(argv[1], argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
