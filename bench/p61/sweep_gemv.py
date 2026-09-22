#!/usr/bin/env python3
"""sweep_gemv.py -- P61: what bounds the served int4-b32 expert GEMV at B=16 -- per-row work or per-expert bytes.

grouped-nf4-gemm's shipped `gemv_int4_b32` only (no model, no new kernel), at Qwen3-30B-A3B's expert shapes
(gate_up N=1536 K=2048, down N=2048 K=768), 128 experts per layer, synthetic int4-b32 weights. Timing does not depend
on the weight values -- only on which experts a call touches and how many rows it serves.

  grid       for R rows in {16, 32, 64, 128} and D distinct experts in {1, 2, ..., 128} (D | R, rows per expert R/D,
             rows interleaved expert-by-expert -- order is worth nothing, P60 P3): one CUDA graph of L calls per cell
             and projection, call l on layer l's OWN weight store, so no call can be served from the previous layer's
             L2 lines (P60's replay shared one store across layers). Graph time and profiler kernel time per call.
  recorded   P60's recorded B=16 routing (bench/p60/receipts/eids_b16.int16.bin, 128 steps x 48 layers): served
             (R=128) and dedup (one row per distinct expert), each step's 96 calls in one graph, on per-layer stores
             AND on one shared store (P60's replay, the instrument check).
  probes     device-to-device copy bandwidth of a 2 GiB buffer (DRAM) and of two buffers sized to a quarter of the L2
             each (L2-resident), read+write bytes per second.

    python sweep_gemv.py --eids eids_b16.int16.bin --meta eids_b16.int16.json --out p61_rows.json     # the lane
    python sweep_gemv.py --self-test                                                                    # tiny shapes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys

import numpy as np
import torch

SHAPES = {"gate_up": (1536, 2048), "down": (2048, 768)}
GRID_R = (16, 32, 64, 128)
GRID_D = (1, 2, 4, 8, 16, 32, 64, 128)
E_DEFAULT = 128


def bytes_per_expert(N: int, K: int) -> int:
    return N * (K // 2) + N * (K // 32) * 2                      # packed int4 + fp16 scales


def grid_cells(rs=GRID_R, ds=GRID_D):
    return [(r, d) for r in rs for d in ds if d <= r and r % d == 0]


def build(L: int, E: int, dev, shapes=SHAPES, R_max: int = 128, seed: int = 0):
    """Per-layer weight stores [L, E, N, K//2] (+ fp16 scales), one activation block per projection."""
    from int4_b32 import quant_x_rows
    g = torch.Generator(device=dev).manual_seed(seed)
    W = {}
    for name, (N, K) in shapes.items():
        packed = torch.empty(L, E, N, K // 2, dtype=torch.uint8, device=dev)
        for li in range(L):                                       # one layer at a time: no int64 temporary of 10 GB
            packed[li] = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev, generator=g)
        scales = torch.full((L, E, N, K // 32), 1e-3, dtype=torch.float16, device=dev)
        x = (torch.randn(R_max, K, device=dev, generator=g) / 4).to(torch.bfloat16)
        xq, xs = quant_x_rows(x)
        W[name] = {"N": N, "K": K, "packed": packed, "scales": scales, "xq": xq, "xs": xs}
    return W


def make_bufs(W, dev, R_max: int = 128):
    from int4_b32 import _plan, _sm_count
    bufs = {}
    for name, w in W.items():
        sks = {r: _plan(w["N"], w["K"], r, _sm_count(dev))[2] for r in (1, 2, 4, 8, 16, 32, 64, R_max)}
        sk = max(sks.values())
        bufs[name] = {"sk": sk, "plan_sk_by_R": sks,
                      "part": torch.empty(sk * R_max, w["N"], dtype=torch.float32, device=dev),
                      "out": torch.empty(R_max, w["N"], dtype=torch.bfloat16, device=dev)}
    return bufs


def call(W, bufs, name, layer, eids):
    from int4_b32 import gemv_int4_b32
    w, b = W[name], bufs[name]
    R = eids.numel()
    return gemv_int4_b32(w["xq"][:R], w["xs"][:R], w["packed"][layer], w["scales"][layer], eids, w["N"], w["K"],
                         part=b["part"][: b["sk"] * R], out=b["out"][:R], fused_reduce=False)


def calls(W, bufs, names, el, layer_of):
    """A capturable closure: for every layer li, each projection in ``names`` on ids el[li], store layer_of(li)."""
    def fn():
        for li in range(len(el)):
            for n in names:
                call(W, bufs, n, layer_of(li), el[li])
    return fn


def cell_eids(R: int, D: int, L: int, dev):
    """Rows interleaved over D experts; the same ids on every layer (each layer has its own store)."""
    e = (torch.arange(R) % D).to(torch.int32).to(dev)
    return [e] * L


def time_graph(fn, iters: int):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return g, statistics.median(ts)


def profile_graph(g, replays: int = 4):
    """Kernel self-time per replay by kernel name (torch.profiler over graph replays, P42's census instrument)."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            g.replay()
        torch.cuda.synchronize()
    gemv = red = 0.0
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None) or getattr(ev, "self_cuda_time_total", 0)
        if not t:
            continue
        if "_gemv_int4_b32" in ev.key and "grouped" not in ev.key:
            gemv += t / 1000.0 / replays
        elif "_reduce_partials" in ev.key:
            red += t / 1000.0 / replays
    return gemv, red                                              # ms per replay


def copy_gbs(dev, nbytes: int, reps: int) -> float:
    a = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    b = torch.empty_like(a)
    def fn():
        for _ in range(reps):
            b.copy_(a)
    _, ms = time_graph(fn, iters=10)
    return 2 * nbytes * reps / (ms / 1000.0) / 1e9                # read + write bytes per second


def _layer0(li: int) -> int:
    return 0                                                      # the shared store: P60's replay


def main_box(a) -> int:
    dev = torch.device("cuda")
    raw = open(a.eids, "rb").read()
    meta = json.load(open(a.meta))
    assert hashlib.sha256(raw).hexdigest() == meta["sha256"], "eids digest mismatch"
    eids_all = np.frombuffer(raw, dtype="<i2").reshape(meta["shape"])                   # [S, L, 16, k]
    S, L = eids_all.shape[:2]
    S, L = min(S, a.steps), min(L, a.layers)                        # caps for a rehearsal on a smaller card only
    eids_all = eids_all[:S, :L]
    from int4_b32 import _sm_count
    props = torch.cuda.get_device_properties(dev)
    l2 = int(getattr(props, "L2_cache_size", 0) or 0)
    rows = {"device": props.name, "sm_count": _sm_count(dev), "torch": torch.__version__, "l2_bytes": l2,
            "layers": L, "steps": S, "eids_sha256": meta["sha256"],
            "bytes_per_expert": {n: bytes_per_expert(*s) for n, s in SHAPES.items()}}
    print(f"[p61] {props.name} ({rows['sm_count']} SMs), L2 {l2 >> 20} MiB, {L} layers, {S} steps", flush=True)

    dram = copy_gbs(dev, 2 * 2**30, reps=4)
    l2buf = max(l2 // 4, 1 << 20)
    l2bw = copy_gbs(dev, l2buf, reps=200)
    rows["bandwidth"] = {"dram_copy_gbs": dram, "l2_copy_gbs": l2bw, "l2_buf_bytes": l2buf}
    torch.cuda.empty_cache()                                      # the probes' 4 GiB before the 16 GB of stores
    print(f"[p61] copy bandwidth: DRAM {dram:.0f} GB/s, L2-resident ({l2buf >> 20} MiB x2) {l2bw:.0f} GB/s", flush=True)

    W = build(L, E_DEFAULT, dev)
    bufs = make_bufs(W, dev)
    rows["plans"] = {n: bufs[n]["plan_sk_by_R"] for n in bufs}

    grid = []
    for name in SHAPES:
        for R, D in grid_cells():
            el = cell_eids(R, D, L, dev)
            g, ms = time_graph(calls(W, bufs, (name,), el, int), iters=a.iters)
            gemv_ms, red_ms = profile_graph(g)
            del g
            cell = {"proj": name, "R": R, "D": D, "graph_us_per_call": ms * 1000 / L,
                    "gemv_us_per_call": gemv_ms * 1000 / L, "reduce_us_per_call": red_ms * 1000 / L}
            grid.append(cell)
            print(f"   grid {name:8s} R={R:3d} D={D:3d}  graph {cell['graph_us_per_call']:7.2f}  "
                  f"gemv {cell['gemv_us_per_call']:7.2f}  reduce {cell['reduce_us_per_call']:5.2f} us/call", flush=True)
    rows["grid"] = grid

    rec = {}
    distinct = []
    for s in range(S):
        distinct.append([int(np.unique(eids_all[s, li]).size) for li in range(L)])
    rows["distinct_per_call"] = {"mean": float(np.mean(distinct)), "min": int(np.min(distinct)),
                                 "max": int(np.max(distinct)),
                                 "rows_removed_per_step_mean": float(np.mean([sum(128 - d for d in st) for st in distinct]))}
    for store in ("per_layer", "shared"):
        per = {"served": [], "dedup": []}
        prof = {"served": [], "dedup": []}
        for s in range(S):
            for arm in ("served", "dedup"):
                el = []
                for li in range(L):
                    e = torch.from_numpy(eids_all[s, li].reshape(-1).astype(np.int32))
                    if arm == "dedup":
                        e = torch.unique(e)
                    el.append(e.to(dev))
                g, ms = time_graph(calls(W, bufs, tuple(SHAPES), el, int if store == "per_layer" else _layer0),
                                   iters=a.iters)
                per[arm].append(ms)
                if s < a.profile_steps:
                    prof[arm].append(profile_graph(g)[0])
                del g
        rec[store] = {arm: {"step_ms_median": statistics.median(v), "step_ms_mean": statistics.mean(v),
                            "min": min(v), "max": max(v),
                            "gemv_kernel_ms_median": statistics.median(prof[arm]) if prof[arm] else None}
                      for arm, v in per.items()}
        print(f"[p61] recorded ({store:9s}) served {rec[store]['served']['step_ms_median']:.3f}  "
              f"dedup {rec[store]['dedup']['step_ms_median']:.3f} ms/step", flush=True)
    rows["recorded"] = rec
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"[p61] rows -> {a.out}", flush=True)
    return 0


def self_test() -> int:
    """Tiny shapes: each call reads its own layer's store (bitwise the served GEMV on that layer's tensors, and two
    layers differ), the grid cells have the R and D they claim, and (on CUDA) a captured cell replays into the eager
    bits."""
    import os
    dev = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1" else "cpu")
    shapes = {"gate_up": (64, 64), "down": (64, 32)}
    L, E, R = 3, 8, 16
    W = build(L, E, dev, shapes=shapes, R_max=R, seed=1)
    bufs = make_bufs(W, dev, R_max=R)
    from int4_b32 import gemv_int4_b32
    fails = 0
    for name in shapes:
        w = W[name]
        e = (torch.arange(R) % 5).to(torch.int32).to(dev)
        for li in range(L):
            got = call(W, bufs, name, li, e).clone()
            want = gemv_int4_b32(w["xq"][:R], w["xs"][:R], w["packed"][li].contiguous(), w["scales"][li].contiguous(),
                                 e, w["N"], w["K"], fused_reduce=False)
            fails += not torch.equal(got, want)
        other = call(W, bufs, name, 1, e).clone()
        fails += torch.equal(call(W, bufs, name, 0, e), other)                         # layers must differ
    for R_, D_ in grid_cells((4, 8, 16), (1, 2, 4, 8)):
        ids = cell_eids(R_, D_, L, dev)[0]
        fails += int(torch.unique(ids).numel() != D_) + int(ids.numel() != R_)
    if dev.type == "cuda":
        e = (torch.arange(R) % 4).to(torch.int32).to(dev)
        eager = [call(W, bufs, "gate_up", li, e).clone() for li in range(L)]
        g, _ = time_graph(calls(W, bufs, ("gate_up",), [e] * L, int), iters=2)
        g.replay()
        torch.cuda.synchronize()
        fails += not torch.equal(bufs["gate_up"]["out"][:R], eager[-1])
    print(f"self-test on {dev.type}: {'OK' if fails == 0 else f'{fails} FAILURES'}")
    return 0 if fails == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eids"), ap.add_argument("--meta"), ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--profile-steps", type=int, default=8)
    ap.add_argument("--layers", type=int, default=48, help="cap the layers (rehearsal on a smaller card)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.eids and a.meta and a.out):
        ap.error("--eids, --meta and --out are required (or --self-test)")
    return main_box(a)


if __name__ == "__main__":
    sys.exit(main())
