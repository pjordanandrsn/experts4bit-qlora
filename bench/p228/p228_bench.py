# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P228's microbench (bench/p228/PREREG-p228.md): K1's selection and reading on one RTX 5090.

For each site (Qwen3-30B-A3B's ``o_proj`` [2048, 4096] and router gate [128, 2048]) and M in 2, 4, 8, over ``copies``
distinct random bf16 weights (enough to stream past L2 per replay):

- **Block A, the selection:** every configuration on cuBLASLt's own heuristic list (the K1 binding's
  ``LtLib.heuristics``: 64 requested, INPLACE excluded) is checked (``AlgoCheck``), run twice (bitwise), compared with
  ``F.linear`` (the tolerance) and timed: one CUDA graph of the ``copies`` calls, 5 replays on each of 8 input draws.
  ``p228_reduce.select`` names the pin.
- **Block B, the reading:** the pins go into ``dense_lt_pin.LT_PINS`` under the loaded key, and e4b's own
  ``apply_dense_lt_pin`` installs them on toy modules holding the same weights. Four arms per cell, one graph each over a
  static input buffer: D (``F.linear``, the served default), D2 (D captured again), P (the pinned call through
  ``PinnedLinear`` / ``router_linear``) and T1 (``F.linear`` at one row). 60 reps: each copies its step's input in,
  replays in its Williams order, 10 timed replays per arm, each right after an untimed replay of the spacer S (D's
  work), with NVML clocks tagged ``site|M|arm``; then D2 against D bitwise and P against D within the tolerance.

    python p228_bench.py --out bench_p228.json
    python p228_bench.py --out shake.json --shakedown --reps 4 --copies-o 8 --copies-r 32
    python p228_bench.py --self-test
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p228_reduce as R  # noqa: E402


class ClockSampler(threading.Thread):
    """P222's NVML sampler, tagged per (site, M, arm): SM and memory clocks, power and the enforced power limit."""

    def __init__(self, index=0, period=0.005):
        super().__init__(daemon=True)
        self.tag, self.period, self.samples, self.error = None, period, [], None
        self._stop_evt = threading.Event()
        try:
            self.nv = ctypes.CDLL("libnvidia-ml.so.1")
            if self.nv.nvmlInit_v2() != 0:
                raise OSError("nvmlInit_v2")
            self.h = ctypes.c_void_p(0)
            if self.nv.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(self.h)) != 0:
                raise OSError("nvmlDeviceGetHandleByIndex_v2")
        except Exception as e:                    # noqa: BLE001  (recorded: the reducer VOIDs on unread clocks)
            self.error = f"{type(e).__name__}: {e}"

    def _read(self):
        sm, mem, mw, lim = ctypes.c_uint(0), ctypes.c_uint(0), ctypes.c_uint(0), ctypes.c_uint(0)
        self.nv.nvmlDeviceGetClockInfo(self.h, 1, ctypes.byref(sm))
        self.nv.nvmlDeviceGetClockInfo(self.h, 2, ctypes.byref(mem))
        self.nv.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(mw))
        self.nv.nvmlDeviceGetEnforcedPowerLimit(self.h, ctypes.byref(lim))
        return sm.value, mem.value, mw.value / 1000.0, lim.value / 1000.0

    def run(self):
        if self.error:
            return
        while not self._stop_evt.is_set():
            tag = self.tag
            if tag is not None:
                try:
                    self.samples.append((tag, *self._read()))
                except Exception as e:            # noqa: BLE001
                    self.error = f"{type(e).__name__}: {e}"
                    return
            time.sleep(self.period)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)

    def summary(self):
        out = {"error": self.error, "n": len(self.samples), "period_s": self.period}
        for tag in sorted({s[0] for s in self.samples}):
            xs = [s for s in self.samples if s[0] == tag]
            out[tag] = {"n": len(xs), "sm_mhz_median": statistics.median(s[1] for s in xs),
                        "mem_mhz_median": statistics.median(s[2] for s in xs),
                        "power_w_median": statistics.median(s[3] for s in xs),
                        "enforced_limit_w": sorted({s[4] for s in xs})}
        return out


def main(a):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from experts4bit_qlora.engines import dense_lt_pin as dl
    dev = torch.device("cuda")
    t0 = time.time()
    shake = bool(a.shakedown)
    copies = {"o_proj": a.copies_o, "router": a.copies_r}
    steps = list(R.READ_STEPS[:a.reps])
    if not shake and (copies != {s: R.SITES[s]["copies"] for s in R.SITE_ORDER} or steps != list(R.READ_STEPS)):
        raise SystemExit("reading knobs are frozen: --copies-* and --reps are shakedown-only")
    os.environ.pop(dl.ENV, None)
    F.linear(torch.ones(2, 8, dtype=torch.bfloat16, device=dev), torch.ones(8, 8, dtype=torch.bfloat16, device=dev))
    torch.cuda.synchronize()
    lib = dl.LtLib(dl.mapped_cublaslt())
    props = torch.cuda.get_device_properties(dev)
    key = (lib.version(), (props.major, props.minor), props.multi_processor_count)
    rec = {"lane": R.LANE, "shakedown": shake, "device": torch.cuda.get_device_name(), "cc": list(key[1]),
           "sm_count": key[2], "torch": torch.__version__, "arms": list(R.ARMS), "spacer": R.SPACER, "iters": R.ITERS,
           "key": {"cublaslt": key[0], "cc": list(key[1]), "sm_count": key[2]}, "library": lib.path, "sites": {}}
    # the copy rate: the floor's divisor (reported)
    x_, y_ = torch.empty(2 * 1024 ** 3, dtype=torch.uint8, device=dev), torch.empty(2 * 1024 ** 3, dtype=torch.uint8,
                                                                                      device=dev)
    for _ in range(3):
        y_.copy_(x_)
    cs = []
    for _ in range(20):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        y_.copy_(x_)
        e1.record()
        e1.synchronize()
        cs.append(e0.elapsed_time(e1) / 1e3)
    del x_, y_
    torch.cuda.empty_cache()
    rec["copy_gbs"] = 2 * 2 * 1024 ** 3 / statistics.median(cs) / 1e9

    def replay_ms(gr):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        gr.replay()
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1)

    def capture(fn, warm=2):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            outs = fn()
        torch.cuda.synchronize()
        return gr, outs

    def rel(outs, refs):
        return max(((o.float() - r.float()).abs().max() / r.float().abs().max()).item() for o, r in zip(outs, refs))

    g = torch.Generator(device=dev).manual_seed(228)
    W = {s: [(torch.randn(R.SITES[s]["N"], R.SITES[s]["K"], generator=g, device=dev) * 0.02).to(torch.bfloat16)
             for _ in range(copies[s])] for s in R.SITE_ORDER}
    # ---- block A: the selection
    t = time.time()
    for site in R.SITE_ORDER:
        N, K, L = R.SITES[site]["N"], R.SITES[site]["K"], copies[site]
        srec = {"N": N, "K": K, "copies": L, "select": {}, "cells": {}}
        rec["sites"][site] = srec
        for M in R.MS:
            ga = torch.Generator(device=dev).manual_seed(2280 + M)
            xa = [(torch.randn(M, K, generator=ga, device=dev) * 0.5).to(torch.bfloat16) for _ in range(R.SELECT_STEPS)]
            prob = lib.problem(N, K, M)
            heur = lib.heuristics(prob)
            cands = []
            for i, (cfg, ws_hint) in enumerate(heur):
                c = {"index": i, "config": cfg, "workspace": ws_hint, "check_ok": False, "bitwise": None, "rel": None,
                     "step_ms": None}
                try:
                    algo = lib.algo(cfg)
                    ws = lib.check(prob, algo)
                    c["check_ok"] = True
                    wsb = torch.empty(max(ws, 1), dtype=torch.uint8, device=dev)
                    xs = xa[0].clone()
                    outs = [torch.empty(M, N, dtype=torch.bfloat16, device=dev) for _ in range(L)]

                    def run(_xs=xs, _outs=outs, _algo=algo, _ws=ws, _wsb=wsb):
                        st = torch.cuda.current_stream().cuda_stream
                        for li in range(L):
                            lib.matmul(prob, _algo, _xs, W[site][li], _outs[li], _wsb.data_ptr(), _ws, st)
                        return _outs
                    run()
                    first = [o.clone() for o in outs]
                    run()
                    c["bitwise"] = all(torch.equal(a_, b_) for a_, b_ in zip(first, outs))
                    c["rel"] = rel(outs, [F.linear(xs, w) for w in W[site]])
                    gr, _ = capture(run)
                    c["step_ms"] = []
                    for s_ in range(R.SELECT_STEPS):
                        xs.copy_(xa[s_])
                        gr.replay()
                        c["step_ms"].append([replay_ms(gr) for _ in range(R.SELECT_ITERS)])
                    del gr, wsb, outs
                except Exception as e:           # noqa: BLE001  (recorded; an ineligible candidate is never chosen)
                    c["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    torch.cuda.synchronize()
                cands.append(c)
                torch.cuda.empty_cache()
            chosen, idx = R.select(cands)
            srec["select"][str(M)] = {"candidates": cands, "chosen": chosen, "chosen_index": idx}
            print(f"P228_SELECT {site} M={M} candidates={len(cands)} chosen={idx} {json.dumps(chosen)} "
                  f"margin={R.selection_margin(cands)}", flush=True)
            if chosen is None:
                rec["arm_fault"] = f"{site} M={M}: no eligible candidate"
                rec["wall_s"] = round(time.time() - t0, 1)
                with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(rec, f, indent=1)
                return 33
            dl.LT_PINS[(*key, N, K, M)] = chosen
    rec["select_s"] = round(time.time() - t, 1)

    # ---- the pinned path, installed by e4b's own resolver on modules holding the same weights
    class ToyAttention(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.o_proj = nn.Linear(w.shape[1], w.shape[0], bias=False, device=dev, dtype=torch.bfloat16)
            self.o_proj.weight = nn.Parameter(w, requires_grad=False)

    class ToyRouter(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.weight, self.bias = nn.Parameter(w, requires_grad=False), None
            self._e4b_router_epi_kind = "softmax_topk"

    model = nn.Module()
    model.attn = nn.ModuleList(ToyAttention(w) for w in W["o_proj"])
    model.routers = nn.ModuleList(ToyRouter(w) for w in W["router"])
    rec["pin_record"] = {k: v for k, v in dl.apply_dense_lt_pin(model, R.MS, device=dev, value="o_proj,router").items()
                         if not k.startswith("_")}

    def arm_fn(site, arm, x):
        if arm == "T1":
            x1 = x[:1]
            return lambda: [F.linear(x1, w) for w in W[site]]
        if arm in ("D", "D2", R.SPACER):
            return lambda: [F.linear(x, w) for w in W[site]]
        if site == "o_proj":
            return lambda: [m.o_proj(x) for m in model.attn]
        return lambda: [dl.router_linear(r, x) for r in model.routers]

    # ---- block B: the reading
    t = time.time()
    sampler = ClockSampler()
    sampler.start()
    try:
        for site in R.SITE_ORDER:
            N, K, L = R.SITES[site]["N"], R.SITES[site]["K"], copies[site]
            for M in R.MS:
                gb = torch.Generator(device=dev).manual_seed(22800 + M)
                xb = [(torch.randn(M, K, generator=gb, device=dev) * 0.5).to(torch.bfloat16) for _ in steps]
                x = xb[0].clone()
                cell = {"steps": steps, "reps": [], "engaged": {}}
                rec["sites"][site]["cells"][str(M)] = cell
                for arm in ("D", "P", "T1"):          # the engagement: one eager pass per arm before any rep
                    s0 = dict(dl.SEEN)
                    arm_fn(site, arm, x)()
                    torch.cuda.synchronize()
                    cell["engaged"][arm] = {k: v - s0.get(k, 0) for k, v in dl.SEEN.items() if v - s0.get(k, 0)}
                graphs = {R.SPACER: capture(arm_fn(site, R.SPACER, x))}
                for arm in R.ARMS:
                    graphs[arm] = capture(arm_fn(site, arm, x))
                for r, s_ in enumerate(steps):
                    x.copy_(xb[r])
                    rot = R.arm_order(r)
                    rp, n_spacer = {arm: [] for arm in rot}, 0
                    for _ in range(R.ITERS):
                        for arm in rot:
                            graphs[R.SPACER][0].replay()      # untimed: every timed replay follows the same work
                            n_spacer += 1
                            sampler.tag = R.clock_tag(site, M, arm)
                            rp[arm].append(replay_ms(graphs[arm][0]))
                            sampler.tag = None
                    d, d2, p = graphs["D"][1], graphs["D2"][1], graphs["P"][1]
                    cell["reps"].append({"step": s_, "order": rot, "replays_ms": rp, "spacer_replays": n_spacer,
                                         "d2_identical": all(torch.equal(u, v) for u, v in zip(d, d2)),
                                         "p_rel": rel(p, d)})
                    print(f"P228_REP {site} M={M} {r} " + " ".join(f"{k}={statistics.median(v) / L * 1e3:.2f}us"
                                                                    for k, v in rp.items()), flush=True)
                del graphs
                torch.cuda.empty_cache()
    except Exception as e:                       # noqa: BLE001
        rec["arm_fault"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        sampler.stop()
    rec["read_s"] = round(time.time() - t, 1)
    rec["clocks"] = sampler.summary()
    rec["wall_s"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    rc = 34 if rec.get("arm_fault") else 0
    print(f"P228_BENCH rc={rc} in {rec['wall_s']} s; clocks n={rec['clocks']['n']} error={rec['clocks']['error']}",
          flush=True)
    return rc


def self_test():
    """The CPU-checkable wiring: the rules come from the reducer; the selection's candidates are cuBLASLt's own list
    through the K1 binding; P runs e4b's installed path; the spacer precedes every timed replay; the order is the
    reducer's."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    src = src[:src.index("def " + "self_test():")]
    n = 0

    def check(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1
    check("heur = lib.heuristics(prob)" in src and "ws = lib.check(prob, algo)" in src, "the binding's own list")
    check('c["bitwise"] = all(torch.equal(a_, b_) for a_, b_ in zip(first, outs))' in src, "bitwise run to run")
    check("chosen, idx = R.select(cands)" in src and "dl.LT_PINS[(*key, N, K, M)] = chosen" in src, "the rule's pin")
    check('dl.apply_dense_lt_pin(model, R.MS, device=dev, value="o_proj,router")' in src, "e4b's resolver installs")
    check("return lambda: [m.o_proj(x) for m in model.attn]" in src, "P through PinnedLinear")
    check("return lambda: [dl.router_linear(r, x) for r in model.routers]" in src, "P through router_linear")
    check('self._e4b_router_epi_kind = "softmax_topk"' in src, "the router's epilogue mark")
    check("graphs[R.SPACER][0].replay()      # untimed" in src, "spacer")
    seq = [src.index(x) for x in ("graphs[R.SPACER][0].replay()", "sampler.tag = R.clock_tag(site, M, arm)",
                                  "rp[arm].append(replay_ms(graphs[arm][0]))", "sampler.tag = None")]
    check(seq == sorted(seq), "S, tag, timed replay, untag")
    seq = [src.index(x) for x in ("chosen, idx = R.select(cands)", "dl.apply_dense_lt_pin(", "sampler = ClockSampler()")]
    check(seq == sorted(seq), "selected and installed before block B")
    check(R.ARMS == ["D", "D2", "P", "T1"] and R.ITERS == 10 and len(R.READ_STEPS) == 60, "the registered shape")
    print(f"p228_bench self-test OK ({n} checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--reps", type=int, default=len(R.READ_STEPS))
    ap.add_argument("--copies-o", type=int, default=R.SITES["o_proj"]["copies"])
    ap.add_argument("--copies-r", type=int, default=R.SITES["router"]["copies"])
    ap.add_argument("--shakedown", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        try:
            self_test()
        except AssertionError as e:
            print(f"p228_bench self-test FAILED: {e}")
            sys.exit(1)
        sys.exit(0)
    if not args.out:
        ap.error("--out is required")
    sys.exit(main(args))
