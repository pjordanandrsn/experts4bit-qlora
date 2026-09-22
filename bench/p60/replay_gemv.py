#!/usr/bin/env python3
"""replay_gemv.py -- P60: replay RECORDED B=16 routing through the shipped int4-b32 expert GEMV and three variants.

grouped-nf4-gemm only (no model): synthetic int4-b32 packed weights for 128 experts per projection at Qwen3-30B-A3B's
expert shapes (gate_up N=1536 K=2048; down N=2048 K=768) -- the bytes and the L2 behaviour depend on WHICH experts a
call touches and how often, not on the weight values -- and for every recorded decode step, the step's 96 expert-GEMV
calls (48 layers x gate_up + down, R = 16 x 8 = 128 rows each, the step's real ids) captured into one CUDA graph and
replayed. Arms, each its own graph per step:

  served   the shipped call: R=128, rows in routing order (token-major), gnf4's own plan         -- the instrument check
  dedup    one row per DISTINCT expert of that call (R = distinct count): the cost if every expert's bytes were read
           once -- the lower bound any grouped kernel (one weight read per expert for all its rows) could reach on loads
  sorted   R=128 with rows ordered by expert id: the cost of the served work with repeats made adjacent (L2 locality)
  floor    no kernel: distinct-expert bytes per step / this box's measured device-to-device bandwidth

The instrument check (registered): `served`'s `_gemv_int4_b32` kernel time per step, read with torch.profiler over
graph replays exactly as P42's census reads it, must reproduce P57's census row (6.340 ms/step) within +-15 %, or
nothing else here is read.

    python replay_gemv.py --eids eids_b16.pt --out replay.json      # on a CUDA box
    python replay_gemv.py --self-test                               # tiny shapes: arms run, dedup/sorted exact vs served rows
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

SHAPES = {"gate_up": (1536, 2048), "down": (2048, 768)}
E_DEFAULT = 128


def _bytes_per_expert(N: int, K: int) -> int:
    return N * (K // 2) + N * (K // 32) * 2                      # packed int4 + fp16 scales


def build(E: int, dev, seed: int = 0, shapes=SHAPES, R: int = 128):
    from int4_b32 import quant_x_rows
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = {}
    for name, (N, K) in shapes.items():
        packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g).to(dev)
        scales = (torch.rand(E, N, K // 32, generator=g) * 0.01 + 1e-3).to(torch.float16).to(dev)
        x = (torch.randn(R, K, generator=g) / 4).to(torch.bfloat16).to(dev)
        xq, xs = quant_x_rows(x)
        W[name] = {"N": N, "K": K, "packed": packed, "scales": scales, "xq": xq, "xs": xs}
    return W


def step_fn(W, eids_layers, arm: str, bufs):
    """One decode step's expert GEMVs for this arm. ``eids_layers``: list over layers of int32 [R] device tensors
    (already deduplicated / sorted for those arms). Buffers are preallocated (capture forbids allocation)."""
    from int4_b32 import gemv_int4_b32
    def fn():
        for eids in eids_layers:
            R = eids.numel()
            for name in ("gate_up", "down"):
                w = W[name]
                b = bufs[name]
                gemv_int4_b32(w["xq"][:R], w["xs"][:R], w["packed"], w["scales"], eids, w["N"], w["K"],
                              part=b["part"][: b["sk"] * R], out=b["out"][:R], fused_reduce=False)
    return fn


def make_bufs(W, dev, R_max: int = 128):
    from int4_b32 import _plan, _sm_count
    bufs = {}
    for name, w in W.items():
        sk = _plan(w["N"], w["K"], R_max, _sm_count(dev))[2]
        # the plan's sk depends on R on small-SM parts; take the max over R <= R_max so every slice fits
        sk = max(_plan(w["N"], w["K"], r, _sm_count(dev))[2] for r in (1, 2, 4, 8, 16, 32, 64, R_max))
        bufs[name] = {"sk": sk, "part": torch.empty(sk * R_max, w["N"], dtype=torch.float32, device=dev),
                      "out": torch.empty(R_max, w["N"], dtype=torch.bfloat16, device=dev)}
    return bufs


def arm_eids(eids_step: torch.Tensor, arm: str, dev):
    """eids_step [layers, 16, k] int16 -> list over layers of int32 [R] device tensors for the arm."""
    out = []
    for li in range(eids_step.shape[0]):
        e = eids_step[li].reshape(-1).to(torch.int32)
        if arm == "dedup":
            e = torch.unique(e)
        elif arm == "sorted":
            e = torch.sort(e, stable=True).values
        out.append(e.to(dev))
    return out


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
    return g, statistics.median(ts), ts


def profile_graph(g, replays: int = 4):
    """Kernel self-time per replay by kernel name (CUPTI via torch.profiler), the census's instrument."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            g.replay()
        torch.cuda.synchronize()
    rows = {}
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None) or getattr(ev, "self_cuda_time_total", 0)
        if t:
            rows[ev.key] = rows.get(ev.key, 0.0) + t / 1000.0 / replays          # ms per replay
    return rows


def bandwidth_gbs(dev, gib: float = 2.0) -> float:
    n = int(gib * 2**30)
    a = torch.empty(n, dtype=torch.uint8, device=dev)
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        b.copy_(a)
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) / 1000.0)
    return 2 * n / statistics.median(ts) / 1e9                               # read + write bytes per second


def main_box(a) -> int:
    dev = torch.device("cuda")
    rec = torch.load(a.eids, map_location="cpu")
    eids = rec["eids"]                                                        # [steps, layers, 16, k]
    S = min(eids.shape[0], a.steps)
    W = build(E_DEFAULT, dev)
    bufs = make_bufs(W, dev)
    bw = bandwidth_gbs(dev)
    per_expert = sum(_bytes_per_expert(*sh) for sh in SHAPES.values())
    out = {"device": torch.cuda.get_device_name(), "sm_count": torch.cuda.get_device_properties(dev).multi_processor_count,
           "torch": torch.__version__, "steps_replayed": S, "eids_shape": list(eids.shape), "prompts_sha256": rec.get("prompts_sha256"),
           "copy_bandwidth_gbs": bw, "bytes_per_expert_layer": per_expert, "plans": {n: bufs[n]["sk"] for n in bufs}, "arms": {}}
    t0 = time.time()
    for arm in ("served", "sorted", "dedup"):
        per_step, kernel_rows = [], []
        for s in range(S):
            el = arm_eids(eids[s], arm, dev)
            g, med, _ = time_graph(step_fn(W, el, arm, bufs), a.iters)
            per_step.append(med)
            if s < a.profile_steps:
                kernel_rows.append(profile_graph(g))
            del g
        gemv = [r.get("_gemv_int4_b32", 0.0) for r in kernel_rows]
        red = [sum(v for k, v in r.items() if "reduce_partials" in k) for r in kernel_rows]
        out["arms"][arm] = {"step_ms_median": statistics.median(per_step), "step_ms_mean": statistics.mean(per_step),
                            "step_ms_min": min(per_step), "step_ms_max": max(per_step),
                            "gemv_kernel_ms_median": statistics.median(gemv) if gemv else None,
                            "reduce_kernel_ms_median": statistics.median(red) if red else None,
                            "profiled_steps": len(kernel_rows)}
        print(f"P60ARM {arm} " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out["arms"][arm].items()}), flush=True)
    distinct = [[len(set(eids[s, li].flatten().tolist())) for li in range(eids.shape[1])] for s in range(S)]
    floor = [sum(d) * per_expert / (bw * 1e9) * 1e3 for d in distinct]
    out["arms"]["floor"] = {"step_ms_median": statistics.median(floor), "step_ms_mean": statistics.mean(floor),
                            "mean_distinct": statistics.mean(x for row in distinct for x in row),
                            "at_1528_gbs_ms": statistics.median(sum(d) * per_expert / 1.528e12 * 1e3 for d in distinct)}
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"), indent=1)
    print("P60REPLAY " + json.dumps({"bw_gbs": round(bw, 1), "floor_ms": round(out["arms"]["floor"]["step_ms_median"], 3),
                                     "mean_distinct": round(out["arms"]["floor"]["mean_distinct"], 2)}), flush=True)
    return 0


def self_test() -> int:
    """CUDA if present (tiny shapes): the three arms build and run; the dedup and sorted arms compute, for every row they
    keep, exactly what the served arm computes for a row with the same expert -- the arms differ in WORK, never in math."""
    if not torch.cuda.is_available():
        print("self-test needs CUDA (the kernel is Triton): skipped")
        return 0
    from int4_b32 import gemv_int4_b32
    dev = torch.device("cuda")
    shapes = {"gate_up": (256, 512), "down": (512, 256)}
    W = build(16, dev, shapes=shapes, R=32)
    bufs = make_bufs(W, dev, R_max=32)
    eids = torch.randint(0, 16, (3, 2, 4, 8)).to(torch.int16)                 # 3 steps, 2 layers, B=4, k=8 -> R=32
    for arm in ("served", "sorted", "dedup"):
        el = arm_eids(eids[0], arm, dev)
        _, med, _ = time_graph(step_fn(W, el, arm, bufs), 5)
        assert med > 0
    w = W["gate_up"]
    e_served = eids[0, 0].reshape(-1).to(torch.int32).to(dev)
    # identical input rows so a row's output depends only on its expert
    xq = w["xq"][:1].expand(32, -1).contiguous()
    xs = w["xs"][:1].expand(32, -1).contiguous()
    y = gemv_int4_b32(xq, xs, w["packed"], w["scales"], e_served, w["N"], w["K"], fused_reduce=False)
    u = torch.unique(e_served)
    yu = gemv_int4_b32(xq[: u.numel()], xs[: u.numel()], w["packed"], w["scales"], u, w["N"], w["K"], fused_reduce=False)
    for i, ex in enumerate(u.tolist()):
        rows = (e_served == ex).nonzero().flatten()
        assert all(torch.equal(y[r], yu[i]) for r in rows.tolist()), f"expert {ex}: dedup row differs from a served row"
    print("self-test OK: served/sorted/dedup graphs capture and replay; dedup rows bit-equal served rows of the same expert")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--eids")
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--profile-steps", type=int, default=8)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.eids and a.out):
        ap.error("--eids and --out are required")
    return main_box(a)


if __name__ == "__main__":
    sys.exit(main())
