# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P227's microbench (bench/p227/PREREG-p227.md): the serving roadmap's lever B. At 2, 4 and 8 decode tokens, e4b's
MoE expert call on the NF4 store, made exactly as the all-resident collapse makes it
(``hot_residency._fused_over_stack`` with the token rows, ``x_tokens``), through two routes:

- **K**, the CONTROL: device grouping, so the tile table and K25 at the family's served decode and plan (Qwen3: prmt and
  c1; Mixtral: the tree and today's plan). K2 is K captured again (the self-pair).
- **B**: ``E4B_NF4_SMALLM_BW=1``, resolved by e4b's own ``resolve_nf4_smallm_bw``. One group per (token, slot) row, so
  grouped-nf4-gemm's K33 bandwidth GEMV, one row per program.

Per family and token count, 60 reps. Each rep captures the spacer S (K's work, its own outputs, never timed) and the
three arms in the rep's Williams order, one graph per arm holding every layer's call, then replays the arms interleaved:
10 timed replays each, every one right after an untimed replay of S. An NVML thread samples clocks, power and the
enforced limit, tagged ``family|T|arm``. Each rep compares K2 with K bit for bit and B with K within the tolerance.
Before the reps, one eager pass per arm records the engagement: gnf4's dispatch counts and e4b's route tally.

    python p227_bench.py --replay rr.json.pt --out bench_p227.json
    python p227_bench.py --replay rr.json.pt --out shake.json --shakedown --layers-qwen3 4 --layers-mixtral 2 --reps 6
    python p227_bench.py --self-test
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p227_reduce as R  # noqa: E402  (the registered rules: the staged sibling)

KNOB = "E4B_NF4_SMALLM_BW"
GEMV_KEYS = ("bw_prmt32", "bw_tree", "bw_splitk", "dotpad", "dotpad_splitk", "scalar", "scalar_splitk")


class Poisoned(RuntimeError):
    """A failure left the CUDA context unusable; nothing after it can be timed."""


class _FieldValue(ctypes.Structure):          # nvmlFieldValue_t
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_uint), ("nvmlReturn", ctypes.c_uint),
                ("value", ctypes.c_ulonglong)]


class ClockSampler(threading.Thread):
    """P222's NVML sampler: every ``period`` seconds while :attr:`tag` names the arm replaying, SM and memory clocks,
    power, the clock-event reasons and the enforced power limit. :meth:`summary` gives each tag its medians."""

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
            self.reasons = (getattr(self.nv, "nvmlDeviceGetCurrentClocksEventReasons", None)
                            or self.nv.nvmlDeviceGetCurrentClocksThrottleReasons)
        except Exception as e:                    # noqa: BLE001  (recorded: the reducer VOIDs on unread clocks)
            self.error = f"{type(e).__name__}: {e}"

    def _read(self):
        sm, mem, mw, rs, lim = (ctypes.c_uint(0), ctypes.c_uint(0), ctypes.c_uint(0), ctypes.c_ulonglong(0),
                                ctypes.c_uint(0))
        self.nv.nvmlDeviceGetClockInfo(self.h, 1, ctypes.byref(sm))
        self.nv.nvmlDeviceGetClockInfo(self.h, 2, ctypes.byref(mem))
        self.nv.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(mw))
        self.reasons(self.h, ctypes.byref(rs))
        self.nv.nvmlDeviceGetEnforcedPowerLimit(self.h, ctypes.byref(lim))
        return sm.value, mem.value, mw.value / 1000.0, rs.value, lim.value / 1000.0

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
            reasons = {}
            for s in xs:
                reasons[hex(s[4])] = reasons.get(hex(s[4]), 0) + 1
            out[tag] = {"n": len(xs), "sm_mhz_median": statistics.median(s[1] for s in xs),
                        "sm_mhz_min": min(s[1] for s in xs), "sm_mhz_max": max(s[1] for s in xs),
                        "mem_mhz_median": statistics.median(s[2] for s in xs),
                        "power_w_median": statistics.median(s[3] for s in xs),
                        "enforced_limit_w": sorted({s[5] for s in xs}), "reasons": reasons}
        return out


class _State:
    """The face :func:`hot_residency.resolve_nf4_smallm_bw` reads of one MoE layer's engine: its shapes, the collapse,
    every expert resident, no int4 store. The bench calls the collapse's ``_fused_over_stack`` itself, so the engine is
    this record of it, and the resolver's own checks (K33 engages at the shapes, on this device) run unchanged."""

    def __init__(self, shapes):
        self.shapes, self.collapse_resident, self._int4_stores = tuple(shapes), True, None

    def _all_hot(self):
        return True


def main(a):
    import torch
    import torch.nn.functional as F
    import nf4_grouped as ng
    from experts4bit_qlora.engines import hot_residency as hr
    dev = torch.device("cuda")
    t0 = time.time()
    shake = bool(a.shakedown)
    layers = {"qwen3": a.layers_qwen3, "mixtral": a.layers_mixtral}
    steps = list(R.STEPS[:a.reps])
    if not shake and (layers != {f: R.FAMILIES[f]["layers"] for f in R.FAMILY_ORDER} or steps != list(R.STEPS)):
        raise SystemExit("reading knobs are frozen: --layers-* and --reps are shakedown-only")
    with open(a.replay, "rb") as fh:
        replay_sha = hashlib.sha256(fh.read()).hexdigest()
    if replay_sha != R.REPLAY_SHA256:
        raise SystemExit(f"the router replay is {replay_sha}, registered {R.REPLAY_SHA256}")
    replay = torch.load(a.replay, weights_only=False)["picks"]          # [48 layers, 32 rows, 128 steps, top-8]
    mpicks = R.mixtral_picks()
    cap = torch.cuda.get_device_capability()
    rec = {"lane": R.LANE, "shakedown": shake, "device": torch.cuda.get_device_name(), "cc": list(cap),
           "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "torch": torch.__version__,
           "arms": list(R.ARMS), "spacer": R.SPACER, "iters": R.ITERS, "replay_sha256": replay_sha, "families": {}}
    try:
        import triton
        import importlib.metadata as md
        rec["triton"], rec["gnf4"] = triton.__version__, md.version("grouped-nf4-gemm")
        rec["e4b"] = md.version("experts4bit-qlora")
    except Exception as e:                       # noqa: BLE001  (recorded, never fatal)
        rec["versions_error"] = f"{type(e).__name__}: {e}"
    for k in (KNOB, "E4B_NF4_GROUPED_SMALLM", "E4B_NF4_T1_DEVICE_GROUPING", "E4B_INT4_LEAN_GLUE", "GNF4_GEMV_BW",
              "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE", "GNF4_PDL", "E4B_K25_LUT", "E4B_K25_PLAN"):
        os.environ.pop(k, None)                  # every knob at its default except the two this lane moves

    def write(rc):
        rec["wall_s"] = round(time.time() - t0, 1)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        return rc

    def check_context():
        try:
            torch.cuda.synchronize()
        except Exception as e:
            raise Poisoned(f"{type(e).__name__}: {str(e)[:200]}") from e

    # ---- the copy rate: the floor's divisor (K33's floor: bytes / copy), median of 20 copies of 2 GiB
    a_ = torch.empty(2 * 1024 ** 3, dtype=torch.uint8, device=dev)
    b_ = torch.empty_like(a_)
    for _ in range(3):
        b_.copy_(a_)
    torch.cuda.synchronize()
    cs = []
    for _ in range(20):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        b_.copy_(a_)
        e1.record()
        e1.synchronize()
        cs.append(e0.elapsed_time(e1) / 1e3)
    del a_, b_
    torch.cuda.empty_cache()
    rec["copy"] = {"gbs": 2 * 2 * 1024 ** 3 / statistics.median(cs) / 1e9, "samples_s": cs}

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

    sampler = ClockSampler()
    sampler.start()
    try:
        for fam in R.FAMILY_ORDER:
            spec = R.FAMILIES[fam]
            L, E, k = layers[fam], spec["E"], spec["topk"]
            n1, k1, n2, k2 = spec["shapes"]
            g = torch.Generator(device=dev).manual_seed(227)
            W = [(torch.randint(0, 256, (E, n1, k1 // 2), dtype=torch.uint8, generator=g, device=dev),
                  torch.rand(E, n1, k1 // 64, generator=g, device=dev) * 0.05 + 0.01,
                  torch.randint(0, 256, (E, n2, k2 // 2), dtype=torch.uint8, generator=g, device=dev),
                  torch.rand(E, n2, k2 // 64, generator=g, device=dev) * 0.05 + 0.01) for _ in range(L)]
            os.environ[KNOB] = "1"                               # resolved by e4b's own resolver, at these shapes
            resolved = hr.resolve_nf4_smallm_bw([types.SimpleNamespace(_hot_residency=_State(spec["shapes"]))
                                                 for _ in range(L)], dev)
            frec = {"layers": L, "shapes": list(spec["shapes"]), "k25": dict(spec["k25"]), "resolved": resolved,
                    "store_bytes": sum(sum(t.numel() * t.element_size() for t in w) for w in W), "cells": {}}
            rec["families"][fam] = frec
            for T in R.TS:
                if fam == "qwen3":
                    tab = [[[[int(e) for e in replay[li, R.QWEN3_ROWS0 + r, s].tolist()] for r in range(T)]
                            for s in R.STEPS] for li in range(L)]
                else:
                    tab = [[[list(row) for row in mpicks[li][s][:T]] for s in R.STEPS] for li in range(L)]
                full = (L == spec["layers"])
                cell = {"steps": steps, "route_sha256": R.route_sha256(tab) if full else None,
                        "distinct": [[len({e for row in tab[li][s] for e in row}) for li in range(L)] for s in steps],
                        "reps": []}
                frec["cells"][str(T)] = cell
                X = [(torch.randn(T, k1, generator=g, device=dev) * 0.3).to(torch.bfloat16) for _ in range(L)]
                rt = torch.arange(T * k, device=dev) // k

                def ids_for(s):
                    return [torch.tensor([e for row in tab[li][s] for e in row], dtype=torch.long, device=dev)
                            for li in range(L)]

                def arm_fn(arm, ids):
                    flags = (True, False) if arm == "B" else (False, True)

                    def fn():
                        return [hr._fused_over_stack(None, ids[li], *W[li], spec["shapes"], True, F.silu,
                                                     singleton_groups=flags[0], device_grouping=flags[1],
                                                     int4_stores=None, x_tokens=(X[li], rt, k)) for li in range(L)]
                    return fn

                def set_arm(arm):
                    os.environ.update(spec["k25"])
                    os.environ[KNOB] = "1" if arm == "B" else "0"

                # the engagement: one eager pass per arm on the first step, before any rep
                ids0 = ids_for(steps[0])
                cell["engaged"] = {}
                for arm in ("K", "B"):
                    set_arm(arm)
                    if arm == "B":
                        assert hr._collapsed_grouping(T, None) == (True, False), "the knob does not route this T"
                    c0, r0 = dict(ng.dispatch_counts()), dict(hr.ROUTE_SEEN)
                    arm_fn(arm, ids0)()
                    torch.cuda.synchronize()
                    c1 = ng.dispatch_counts()
                    eng = {key: c1.get(key, 0) - c0.get(key, 0) for key in GEMV_KEYS}
                    eng["route"] = {key: v - r0.get(key, 0) for key, v in hr.ROUTE_SEEN.items() if v - r0.get(key, 0)}
                    cell["engaged"][arm] = eng
                for r, s in enumerate(steps):
                    ids = ids_for(s)
                    rot = R.arm_order(r)
                    set_arm("K")
                    graphs = {R.SPACER: capture(arm_fn("K", ids))}          # its own buffers: never timed
                    for arm in rot:
                        set_arm(arm)
                        graphs[arm] = capture(arm_fn(arm, ids))
                    rp, n_spacer = {arm: [] for arm in rot}, 0
                    for _ in range(R.ITERS):
                        for arm in rot:
                            graphs[R.SPACER][0].replay()      # untimed: every timed replay follows the same work
                            n_spacer += 1
                            sampler.tag = R.clock_tag(fam, T, arm)
                            rp[arm].append(replay_ms(graphs[arm][0]))
                            sampler.tag = None
                    ko, k2o, bo = graphs["K"][1], graphs["K2"][1], graphs["B"][1]
                    same = all(torch.equal(x, y) for x, y in zip(ko, k2o))
                    rel = max(((b.float() - x.float()).abs().max() / x.float().abs().max()).item()
                              for b, x in zip(bo, ko))
                    cell["reps"].append({"step": s, "order": rot, "replays_ms": rp, "spacer_replays": n_spacer,
                                         "k2_identical": same, "b_rel_diff": rel})
                    del graphs, ko, k2o, bo
                    torch.cuda.empty_cache()
                    print(f"P227_REP {fam} T={T} {r} step {s} " + " ".join(
                        f"{x}={statistics.median(rp[x]):.3f}" for x in R.ARMS) + f" k2_same={same} b_rel={rel:.4f}",
                        flush=True)
                X = None                                  # released before the next cell's stores
            W = None
            torch.cuda.empty_cache()
    except Poisoned as e:
        rec["poisoned"] = str(e)
    except Exception as e:                       # noqa: BLE001  (an arm that cannot resolve, launch or capture)
        try:
            check_context()
            rec["arm_fault"] = f"{type(e).__name__}: {str(e)[:300]}"
        except Poisoned as p:
            rec["poisoned"] = str(p)
    finally:
        sampler.stop()
        for k_ in (KNOB, "E4B_K25_LUT", "E4B_K25_PLAN"):
            os.environ.pop(k_, None)
    rec["clocks"] = sampler.summary()
    rc = 32 if rec.get("poisoned") else (34 if rec.get("arm_fault") else 0)
    print(f"P227_BENCH rc={rc} in {round(time.time() - t0, 1)} s; clocks n={rec['clocks']['n']} "
          f"error={rec['clocks']['error']}", flush=True)
    return write(rc)


def self_test():
    """The CPU-checkable wiring: the rules come from the reducer, both arms make the collapse's own call with the token
    rows, B is the knob resolved by e4b's resolver, every timed replay follows the spacer, and the order is the
    reducer's."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    src = src[:src.index("def " + "self_test():")]          # the code under test, never these needles
    n = 0

    def check(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1
    check("hr._fused_over_stack(None, ids[li], *W[li], spec[\"shapes\"], True, F.silu," in src, "the collapse's call")
    check("int4_stores=None, x_tokens=(X[li], rt, k)) for li in range(L)]" in src, "the token rows")
    check('flags = (True, False) if arm == "B" else (False, True)' in src, "B singleton, K device-grouped")
    check("hr.resolve_nf4_smallm_bw([types.SimpleNamespace(_hot_residency=_State(spec[\"shapes\"]))" in src, "resolved")
    check('os.environ[KNOB] = "1" if arm == "B" else "0"' in src and "os.environ.update(spec[\"k25\"])" in src, "env")
    check("graphs[R.SPACER][0].replay()      # untimed" in src, "spacer")
    seq = [src.index(x) for x in ("graphs[R.SPACER][0].replay()", "sampler.tag = R.clock_tag(fam, T, arm)",
                                  "rp[arm].append(replay_ms(graphs[arm][0]))", "sampler.tag = None")]
    check(seq == sorted(seq), "S, then the tag, the timed replay, the untag")
    check("rot = R.arm_order(r)" in src and "graphs = {R.SPACER: capture(arm_fn(\"K\", ids))}" in src, "order, spacer K")
    check("same = all(torch.equal(x, y) for x, y in zip(ko, k2o))" in src, "K2 bitwise")
    check("rt = torch.arange(T * k, device=dev) // k" in src, "token-major rows")
    check("R.QWEN3_ROWS0 + r, s]" in src and "mpicks[li][s][:T]" in src, "the registered rows")
    check(R.ARMS == ["K", "K2", "B"] and R.ITERS == 10 and len(R.STEPS) == 60, "the registered shape")
    st = _State([1536, 2048, 2048, 768])
    check(st._all_hot() and st.collapse_resident and st._int4_stores is None and st.shapes == (1536, 2048, 2048, 768),
          "the resolver's face")
    print(f"p227_bench self-test OK ({n} checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--replay")
    ap.add_argument("--layers-qwen3", type=int, default=R.FAMILIES["qwen3"]["layers"])
    ap.add_argument("--layers-mixtral", type=int, default=R.FAMILIES["mixtral"]["layers"])
    ap.add_argument("--reps", type=int, default=len(R.STEPS))
    ap.add_argument("--shakedown", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        try:
            self_test()
        except AssertionError as e:
            print(f"p227_bench self-test FAILED: {e}")
            sys.exit(1)
        sys.exit(0)
    if not args.out or not args.replay:
        ap.error("--out and --replay are required")
    sys.exit(main(args))
