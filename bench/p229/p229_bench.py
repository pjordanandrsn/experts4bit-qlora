# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P229's microbench (bench/p229/PREREG-p229.md): K2b's int8 lm_head route at 2, 4 and 8 rows on one RTX 5090.

One random bf16 head at Qwen3-30B-A3B's shape (151936 x 2048, seed 229) is swapped by e4b's own resolvers:
``apply_int8_head_bw(model, "1")`` (rounded in place to its int8 grid, K36 at one row) and
``apply_int8_head_rows(model, 8)`` (K37's ``linear_i8`` at 2..8 rows). Four arms per M, one CUDA graph each over a
static input buffer: D (``F.linear`` on the grid's bf16 rounding: the served path at those rows), D2 (D captured again),
I (the head's installed route at M rows: K37) and K (the head at one row: K36). 60 reps: each copies its step's input
in, replays in its Williams order, 10 timed replays per arm, each right after an untimed replay of the spacer S (D's
work), with NVML clocks tagged ``M|arm``; then D2 against D bitwise and I against D within the tolerance.

    python p229_bench.py --out bench_p229.json
    python p229_bench.py --out shake.json --shakedown --reps 4
    python p229_bench.py --self-test
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
import p229_reduce as R  # noqa: E402


class ClockSampler(threading.Thread):
    """P222's NVML sampler, tagged per (M, arm): SM and memory clocks, power and the enforced power limit."""

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


def gnf4_commit():
    """The installed grouped-nf4-gemm's commit, from pip's direct_url.json (None when unread)."""
    import importlib.metadata as md
    try:
        d = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
        return (d.get("vcs_info") or {}).get("commit_id")
    except Exception:                             # noqa: BLE001  (unread: the reducer VOIDs)
        return None


def main(a):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from experts4bit_qlora.engines import int8_head_bw as ih
    dev = torch.device("cuda")
    t0 = time.time()
    shake = bool(a.shakedown)
    steps = list(R.READ_STEPS[:a.reps])
    if not shake and steps != list(R.READ_STEPS):
        raise SystemExit("reading knobs are frozen: --reps is shakedown-only")
    for k in (ih.ENV, ih.ROWS_ENV):
        os.environ.pop(k, None)
    N, K = R.HEAD["N"], R.HEAD["K"]
    rec = {"lane": R.LANE, "shakedown": shake, "device": torch.cuda.get_device_name(),
           "cc": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "gnf4_commit": gnf4_commit(),
           "arms": list(R.ARMS), "spacer": R.SPACER, "iters": R.ITERS, "head": {"N": N, "K": K}, "cells": {}}
    # the copy rate: the floors' divisor (reported)
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

    # ---- the head, swapped by e4b's own resolvers: the grid (K36 at one row), then the rows route (K37 at 2..8)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator(device=dev).manual_seed(229)
            self.embed = nn.Embedding(8, K, device=dev, dtype=torch.bfloat16)
            self.lm_head = nn.Linear(K, N, bias=False, device=dev, dtype=torch.bfloat16)
            with torch.no_grad():
                self.lm_head.weight.copy_((torch.randn(N, K, generator=g, device=dev) * 0.02).to(torch.bfloat16))

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, m):
            self.lm_head = m

    model = Toy()
    rec["grid_record"] = ih.apply_int8_head_bw(model, "1")
    rec["rows_record"] = ih.apply_int8_head_rows(model, 8)
    head = model.lm_head
    torch.cuda.synchronize()

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
            out = fn()
        torch.cuda.synchronize()
        return gr, out

    def rel(o, r):
        return ((o.float() - r.float()).abs().max() / r.float().abs().max()).item()

    def arm_fn(arm, x):
        if arm == "K":
            x1 = x[:1]
            return lambda: head(x1)
        if arm == "I":
            return lambda: head(x)
        return lambda: F.linear(x, head.weight)          # D, D2 and the spacer: the served path on the grid

    # ---- the reading
    t = time.time()
    sampler = ClockSampler()
    sampler.start()
    try:
        for M in R.MS:
            gb = torch.Generator(device=dev).manual_seed(22900 + M)
            xb = [torch.randn(M, K, generator=gb, device=dev).to(torch.bfloat16) for _ in steps]
            x = xb[0].clone()
            cell = {"steps": steps, "reps": [], "engaged": {}}
            rec["cells"][str(M)] = cell
            for arm in R.ARMS:                            # the engagement: one eager pass per arm before any rep
                s0 = dict(head.routed)
                arm_fn(arm, x)()
                torch.cuda.synchronize()
                cell["engaged"][arm] = {str(k): v - s0.get(k, 0) for k, v in head.routed.items() if v - s0.get(k, 0)}
            graphs = {R.SPACER: capture(arm_fn(R.SPACER, x))}
            for arm in R.ARMS:
                graphs[arm] = capture(arm_fn(arm, x))
            for r, s_ in enumerate(steps):
                x.copy_(xb[r])
                rot = R.arm_order(r)
                rp, n_spacer = {arm: [] for arm in rot}, 0
                for _ in range(R.ITERS):
                    for arm in rot:
                        graphs[R.SPACER][0].replay()      # untimed: every timed replay follows the same work
                        n_spacer += 1
                        sampler.tag = R.clock_tag(M, arm)
                        rp[arm].append(replay_ms(graphs[arm][0]))
                        sampler.tag = None
                d, d2, i_ = graphs["D"][1], graphs["D2"][1], graphs["I"][1]
                cell["reps"].append({"step": s_, "order": rot, "replays_ms": rp, "spacer_replays": n_spacer,
                                     "d2_identical": bool(torch.equal(d, d2)), "i_rel": rel(i_, d)})
                print(f"P229_REP M={M} {r} " + " ".join(f"{k}={statistics.median(v) * 1e3:.1f}us" for k, v in rp.items()),
                      flush=True)
            del graphs
            torch.cuda.empty_cache()
    except Exception as e:                           # noqa: BLE001
        rec["arm_fault"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        sampler.stop()
    rec["read_s"] = round(time.time() - t, 1)
    rec["engagement"] = ih.engagement(model)
    rec["clocks"] = sampler.summary()
    rec["wall_s"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    rc = 34 if rec.get("arm_fault") else 0
    print(f"P229_BENCH rc={rc} in {rec['wall_s']} s; clocks n={rec['clocks']['n']} error={rec['clocks']['error']}",
          flush=True)
    return rc


def self_test():
    """The CPU-checkable wiring: e4b's resolvers install the route; I and K call the swapped head; D, D2 and S call
    F.linear on its grid; the spacer precedes every timed replay; the order and the rules are the reducer's."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    src = src[:src.index("def " + "self_test():")]
    n = 0

    def check(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1
    check('rec["grid_record"] = ih.apply_int8_head_bw(model, "1")' in src, "e4b swaps the head onto its grid")
    check('rec["rows_record"] = ih.apply_int8_head_rows(model, 8)' in src, "e4b installs the rows route at 8")
    check("return lambda: head(x)" in src and "return lambda: head(x1)" in src, "I and K through the swapped head")
    check("return lambda: F.linear(x, head.weight)" in src, "D on the grid's bf16 rounding")
    check("graphs[R.SPACER][0].replay()      # untimed" in src, "spacer")
    seq = [src.index(x) for x in ("graphs[R.SPACER][0].replay()", "sampler.tag = R.clock_tag(M, arm)",
                                  "rp[arm].append(replay_ms(graphs[arm][0]))", "sampler.tag = None")]
    check(seq == sorted(seq), "S, tag, timed replay, untag")
    seq = [src.index(x) for x in ("ih.apply_int8_head_rows(model, 8)", "sampler = ClockSampler()")]
    check(seq == sorted(seq), "the route is installed before the reading")
    check(R.ARMS == ["D", "D2", "I", "K"] and R.ITERS == 10 and len(R.READ_STEPS) == 60 and R.MS == (2, 4, 8),
          "the registered shape")
    print(f"p229_bench self-test OK ({n} checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--reps", type=int, default=len(R.READ_STEPS))
    ap.add_argument("--shakedown", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        try:
            self_test()
        except AssertionError as e:
            print(f"p229_bench self-test FAILED: {e}")
            sys.exit(1)
        sys.exit(0)
    if not args.out:
        ap.error("--out is required")
    sys.exit(main(args))
