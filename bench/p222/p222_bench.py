# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P222's microbench (bench/p222/PREREG-p222.md): K25's decode x plan at Mixtral-8x7B's W16 shapes, where the served
default is still the select tree at today's plan.

K25 (grouped-nf4-gemm ``nf4_smallm.gemm_nf4_grouped_smallm``) is called exactly as e4b's served route calls it on the
NF4 store at 16 rows (P156's call form): gate_up reads the 16 token rows through the lean tile table's ``order``
(``gather_div=2``, Mixtral's top-2), and down reads the sorted rows and stores into the caller's order
(``scatter=order``), TF32 on fp32 weights. Random NF4 stores for 8 MoE layers (one layer is ~0.8 GB, far beyond L2).

The routing is seeded synthetic top-2 (``p222_reduce.picks``): no Mixtral router replay exists.
- **Block A, the SELECTION (reported):** each plan of P156's 144-plan grid under ``lut="prmt"``, per projection, in the
  registered shuffled order: its outputs against the tree at today's plan (bit for bit, every layer), then 5 timed
  replays on each of 8 steps of A. ``p222_reduce.select`` names M.
- **Block B, the READING:** five arms (``p222_reduce.ARMS``, Amendment 1) on 120 steps of B. Each rep captures the spacer S (T's calls
  with their own outputs) and every arm's graph (both GEMMs of every layer) in the rep's Williams row, then replays the
  arms interleaved, 10 timed replays each, every one right after an untimed replay of S. An NVML thread samples clocks,
  power and the enforced limit, tagged with the arm replaying. After the replays, every arm's outputs are compared with
  T's, bit for bit.

    python p222_bench.py --out bench_p222.json
    python p222_bench.py --out shake.json --shakedown --layers 2 --reps 10 --grid 6 --select-steps 2
    python p222_bench.py --self-test
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
import p222_reduce as R  # noqa: E402  (the registered rules: the staged sibling)


class Poisoned(RuntimeError):
    """A failure left the CUDA context unusable; nothing after it can be timed."""


class ArmFault(RuntimeError):
    """An arm's graph could not be captured (Amendment 1): recorded in the bench record, and the reducer VOIDs."""


class _FieldValue(ctypes.Structure):          # nvmlFieldValue_t
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_uint), ("nvmlReturn", ctypes.c_uint),
                ("value", ctypes.c_ulonglong)]


_FI_POWER_INSTANT = 186                       # NVML_FI_DEV_POWER_INSTANT (mW)


class ClockSampler(threading.Thread):
    """NVML samples every ``period`` seconds while :attr:`tag` names the arm replaying, None between replays (P160's
    sampler, one block): SM and memory clocks, the averaged and instantaneous power, the clock-event reasons and the
    enforced power limit. :meth:`summary` gives each arm its medians."""

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
        fv = _FieldValue(fieldId=_FI_POWER_INSTANT)
        inst = None
        if self.nv.nvmlDeviceGetFieldValues(self.h, 1, ctypes.byref(fv)) == 0 and fv.nvmlReturn == 0:
            inst = (fv.value & 0xFFFFFFFF) / 1000.0
        return sm.value, mem.value, mw.value / 1000.0, rs.value, lim.value / 1000.0, inst

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
        for arm in R.ARMS:
            xs = [s for s in self.samples if s[0] == arm]
            if not xs:
                continue
            reasons = {}
            for s in xs:
                reasons[hex(s[4])] = reasons.get(hex(s[4]), 0) + 1
            inst = [s[6] for s in xs if s[6] is not None]
            out[arm] = {"n": len(xs), "sm_mhz_median": statistics.median(s[1] for s in xs),
                        "sm_mhz_min": min(s[1] for s in xs), "sm_mhz_max": max(s[1] for s in xs),
                        "mem_mhz_median": statistics.median(s[2] for s in xs),
                        "power_w_median": statistics.median(s[3] for s in xs),
                        "power_instant_w_median": statistics.median(inst) if inst else None,
                        "power_instant_w_max": max(inst) if inst else None,
                        "enforced_limit_w": sorted({s[5] for s in xs}), "reasons": reasons}
        return out


def compiled_variants(fn):
    """Every compiled variant in a Triton JIT function's per-device caches (P156's reader; reported, never a stop)."""
    out = {}
    try:
        for dev in getattr(fn, "device_caches", {}).values():
            cache = dev[0] if isinstance(dev, tuple) else dev
            if isinstance(cache, dict):
                out.update(cache)
    except Exception:                                # noqa: BLE001
        return {}
    return out


def new_variants(fn, before):
    rows = []
    for key, k in compiled_variants(fn).items():
        if key in before:
            continue
        try:
            m = getattr(k, "metadata", None)
            rows.append({"n_regs": getattr(k, "n_regs", None), "n_spills": getattr(k, "n_spills", None),
                         "shared": getattr(m, "shared", None), "num_warps": getattr(m, "num_warps", None),
                         "num_stages": getattr(m, "num_stages", None)})
        except Exception as e:                       # noqa: BLE001
            rows.append({"unread": type(e).__name__})
    return rows


def arm_compiled(rows, earlier):
    """Amendment 2: one arm's compiled metadata for one projection, each row naming the pass it came from. A variant
    the metadata pass did not compile was compiled earlier in the run (block A's reference or its sweep), and its first
    compile stands in. ``earlier`` is ``(rows, pass)`` for the same (projection, plan, decode), or None."""
    if rows:
        return [dict(r, source="block B metadata pass") for r in rows]
    if earlier and earlier[0]:
        return [dict(r, source=earlier[1]) for r in earlier[0]]
    return [{"unread": "no recording pass saw this variant compile", "source": None}]


def main(a):
    import torch
    import nf4_smallm
    from int4_b32 import build_group_tiles_fused
    k25 = nf4_smallm.gemm_nf4_grouped_smallm
    dev = "cuda"
    t0 = time.time()
    walls = {}
    L = a.layers
    sel_steps = list(R.SELECT_STEPS[:a.select_steps])
    read_steps = list(R.READ_STEPS[:a.reps])
    order = R.sweep_order()[:a.grid]
    if a.shakedown:                               # a shakedown sweeps today's plan first, so the tree's own plan is timed
        today = R.GRID.index(R.TODAY)
        order = [today] + [i for i in R.sweep_order() if i != today][:a.grid - 1]
    if not a.shakedown and (L != R.LAYERS or sel_steps != list(R.SELECT_STEPS) or read_steps != list(R.READ_STEPS)
                            or len(order) != len(R.GRID)):
        raise SystemExit("reading knobs are frozen: --layers, --reps, --grid and --select-steps are shakedown-only")
    route = {b: R.picks(seed, layers=L) for b, seed in R.SEEDS.items()}
    cap = torch.cuda.get_device_capability()
    rec = {"lane": "P222", "shakedown": bool(a.shakedown), "device": torch.cuda.get_device_name(), "cc": list(cap),
           "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "torch": torch.__version__,
           "layers": L, "route_sha256": ({b: R.route_sha256(pk) for b, pk in route.items()} if L == R.LAYERS
                                         else {b: R.route_sha256(R.picks(s)) for b, s in R.SEEDS.items()}),
           "arms": list(R.ARMS), "spacer": R.SPACER, "iters": R.ITERS, "select_steps": sel_steps, "order": order}
    try:
        import triton
        rec["triton"] = triton.__version__
    except Exception as e:                       # recorded, never fatal: the kernel module imported it already
        rec["triton"] = f"unread: {type(e).__name__}"

    # ---- the stores: random NF4 bytes and fp32 absmax, the served layout, every layer distinct
    t = time.time()
    g = torch.Generator(device=dev).manual_seed(222)
    W = []
    for _ in range(L):
        st = {}
        for proj, (N, K) in R.SHAPES.items():
            st[proj] = (torch.randint(0, 256, (R.E, N, K // 2), dtype=torch.uint8, generator=g, device=dev),
                        torch.rand(R.E, N, K // 64, generator=g, device=dev) * 0.05 + 0.01)
        W.append(st)
    X = [(torch.randn(R.TOKENS, R.SHAPES["gu"][1], generator=g, device=dev) / 4).to(torch.bfloat16) for _ in range(L)]
    H = [(torch.randn(R.TOKENS * R.TOPK, R.SHAPES["dn"][1], generator=g, device=dev) / 4).to(torch.bfloat16)
         for _ in range(L)]
    torch.cuda.synchronize()
    walls["stores_s"] = round(time.time() - t, 1)
    rec["store_bytes"] = sum(p.numel() + a_.numel() * 4 for st in W for p, a_ in st.values())

    def gu(li, tab, plan, lut):                   # the served gate_up call: token rows, gathered through order
        row0, rows, grp, order_ = tab
        p, am = W[li]["gu"]
        return k25(X[li], p, am, row0, rows, grp, order_, gather_div=R.TOPK, lut=lut, dot_bf16=False, **plan)

    def dn(li, tab, plan, lut):                   # the served down call: sorted rows in, stored to the caller's order
        row0, rows, grp, order_ = tab
        p, am = W[li]["dn"]
        return k25(H[li], p, am, row0, rows, grp, None, scatter=order_, lut=lut, dot_bf16=False, **plan)

    call = {"gu": gu, "dn": dn}

    def check_context():
        try:
            torch.cuda.synchronize()
        except Exception as e:
            raise Poisoned(f"{type(e).__name__}: {str(e)[:200]}") from e

    def capture(fn, warm=2):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            outs = fn()
        torch.cuda.synchronize()
        return gr, outs

    def replay_ms(gr):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        gr.replay()
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1)

    def tables(b, s):
        """One step's tile tables, as the served route builds them; faults if the real tiles are not a prefix."""
        tabs, faults = [], []
        for li in range(L):
            flat = [e for row in route[b][li][s] for e in row]
            ids = torch.tensor(flat, dtype=torch.long, device=dev)
            row0, rows, grp, order_, _counts, _sids = build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)
            tabs.append((row0, rows, grp, order_))
            rv = rows.tolist()
            n = sum(1 for v in rv if v > 0)
            want = len(set(flat))
            if n != want or any(v <= 0 for v in rv[:n]) or any(v != 0 for v in rv[n:]):
                faults.append({"block": b, "step": s, "layer": li, "real": n, "distinct": want,
                               "fault": "the real tiles are not a prefix of one tile per expert"})
        return tabs, faults

    def proj_replays(proj, plan, lut, tabs):
        gr, outs = capture(lambda: [call[proj](li, tabs[li], plan, lut) for li in range(L)])
        xs = [replay_ms(gr) for _ in range(R.SELECT_ITERS)]
        del gr, outs
        return xs

    # ---- block A: the reference (the tree at today's plan, the first selection step) and the SWEEP under prmt
    t = time.time()
    kfn = nf4_smallm._gemm_nf4_grouped_smallm
    sel_tabs, faults = {}, []
    for s in sel_steps:
        sel_tabs[s], f = tables("A", s)
        faults += f
    rec["select_tile_faults"] = faults
    ref, first = {}, {}                  # first[(proj, plan, lut)]: the variants its first compile made, and the pass
    for p in ("gu", "dn"):
        before = set(compiled_variants(kfn))
        ref[p] = [call[p](li, sel_tabs[sel_steps[0]][li], R.TODAY, R.T_LUT) for li in range(L)]
        torch.cuda.synchronize()
        first[(p, R.plan_key(R.TODAY), R.T_LUT)] = (new_variants(kfn, before), "block A reference")
    sel = {"gu": [], "dn": []}
    try:
        for proj in ("gu", "dn"):
            for gi in order:
                plan = R.GRID[gi]
                row = {"plan": dict(plan), "identical": None, "error": None, "step_replays_ms": None}
                try:
                    tc = time.time()
                    before = set(compiled_variants(kfn))
                    outs = [call[proj](li, sel_tabs[sel_steps[0]][li], plan, R.P_LUT) for li in range(L)]
                    torch.cuda.synchronize()
                    row["first_call_s"] = round(time.time() - tc, 2)
                    row["compiled"] = new_variants(kfn, before)
                    if row["compiled"]:
                        first.setdefault((proj, R.plan_key(plan), R.P_LUT), (row["compiled"], "block A sweep"))
                    diff = [li for li in range(L) if not torch.equal(outs[li], ref[proj][li])]
                    row["identical"] = not diff
                    if diff:
                        d = (outs[diff[0]].float() - ref[proj][diff[0]].float()).abs()
                        row["mismatch"] = {"layers": diff[:8], "n_layers": len(diff), "max_abs": float(d.max()),
                                           "n_elems": int((d > 0).sum())}
                    del outs
                    if row["identical"]:
                        row["step_replays_ms"] = [proj_replays(proj, plan, R.P_LUT, sel_tabs[s]) for s in sel_steps]
                except Exception as e:                   # a plan that cannot compile or launch: recorded, excluded
                    row["identical"], row["step_replays_ms"] = None, None
                    row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    check_context()
                sel[proj].append(row)
                torch.cuda.empty_cache()
                print(f"P222_PLAN {proj} {R.plan_key(plan)} identical={row['identical']} err={bool(row['error'])} "
                      f"first={row.get('first_call_s')}", flush=True)
    except Poisoned as e:
        rec.update(select=sel, poisoned=str(e), walls=walls, wall_s=round(time.time() - t0, 1))
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        print(f"P222_POISONED {e}", flush=True)
        return 32
    del ref
    rec["select"] = sel
    walls["select_s"] = round(time.time() - t, 1)
    m = R.select(sel, len(sel_steps))
    if m is None:
        rec.update(m=None, walls=walls, wall_s=round(time.time() - t0, 1))
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        print("P222_SELECT_NONE no eligible plan for a projection", flush=True)
        return 33
    plans = R.plans_for(m)
    rec["m"], rec["plans"] = m, {arm: list(p) for arm, p in plans.items()}
    print("P222_M " + json.dumps(m) + " margin " + json.dumps(R.selection_margin(sel, len(sel_steps))), flush=True)

    # ---- each arm's compiled metadata (registers, spills, shared memory), per projection, on B's first step
    t = time.time()
    tabs0, _ = tables("B", read_steps[0])
    rec["compiled"], seen = {}, {}
    for arm in R.ARMS:
        pg, pd, lut = plans[arm]
        rec["compiled"][arm] = {}
        for proj, plan in (("gu", pg), ("dn", pd)):
            key = (proj, R.plan_key(plan), lut)
            if key in seen:                       # T2 is T's cubin, and M may equal C1 or today: read once
                rec["compiled"][arm][proj] = seen[key]
                continue
            before = set(compiled_variants(kfn))
            try:
                for li in range(L):
                    call[proj](li, tabs0[li], plan, lut)
                torch.cuda.synchronize()
            except Exception as e:                       # Amendment 1: recorded; the reducer VOIDs with the reason
                check_context()
                rec["arm_fault"] = f"{arm} {proj} {R.plan_key(plan)} {lut}: {type(e).__name__}: {str(e)[:200]}"
                rec.update(walls=walls, wall_s=round(time.time() - t0, 1))
                with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(rec, f, indent=1)
                print(f"P222_ARM_FAULT {rec['arm_fault']}", flush=True)
                return 34
            seen[key] = rec["compiled"][arm][proj] = arm_compiled(new_variants(kfn, before), first.get(key))
    walls["compile_s"] = round(time.time() - t, 1)

    def graph_fn(arm, tabs):
        pg, pd, lut = plans["T" if arm == R.SPACER else arm]
        return lambda: [(gu(li, tabs[li], pg, lut), dn(li, tabs[li], pd, lut)) for li in range(L)]

    # ---- block B: the reading; one paired rep per step in its Williams row
    t = time.time()
    sampler = ClockSampler()
    sampler.start()
    blk = {"steps": read_steps, "n_real": R.n_real(route["B"]), "tile_faults": [], "reps": []}
    rec["read"] = blk
    try:
        for r, s in enumerate(read_steps):
            tabs, f = tables("B", s)
            blk["tile_faults"] += f
            rot = R.arm_order(r)
            try:
                graphs = {R.SPACER: capture(graph_fn(R.SPACER, tabs))}   # its own buffers: never timed
                for arm in rot:
                    graphs[arm] = capture(graph_fn(arm, tabs))
            except Poisoned:
                raise
            except Exception as e:                       # Amendment 1: a capture that fails is recorded, never a crash
                check_context()
                rec["arm_fault"] = f"rep {r} step {s} capture: {type(e).__name__}: {str(e)[:200]}"
                raise ArmFault(rec["arm_fault"]) from e
            rp, n_spacer = {arm: [] for arm in rot}, 0
            for _ in range(R.ITERS):
                for arm in rot:
                    graphs[R.SPACER][0].replay()      # untimed: every timed replay follows the same work
                    n_spacer += 1
                    sampler.tag = arm
                    rp[arm].append(replay_ms(graphs[arm][0]))
                    sampler.tag = None
            ref_outs = graphs["T"][1]
            same = lambda x, y: all(torch.equal(o[0], q[0]) and torch.equal(o[1], q[1])  # noqa: E731
                                    for o, q in zip(x, y))
            ident = {arm: same(graphs[arm][1], ref_outs) for arm in rot if arm != "T"}
            blk["reps"].append({"step": s, "order": rot, "replays_ms": rp, "spacer_replays": n_spacer,
                                "identical": ident})
            del graphs, ref_outs, tabs
            torch.cuda.empty_cache()
            print(f"P222_REP {r} step {s} " + " ".join(f"{a_}={statistics.median(rp[a_]):.3f}" for a_ in R.ARMS)
                  + f" identical={all(ident.values())}", flush=True)
    except (Poisoned, ArmFault) as e:
        sampler.stop()
        rec.update(clocks=sampler.summary(), walls=walls, wall_s=round(time.time() - t0, 1))
        if isinstance(e, Poisoned):
            rec["poisoned"] = str(e)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        print(f"P222_{'POISONED' if isinstance(e, Poisoned) else 'ARM_FAULT'} {e}", flush=True)
        return 32 if isinstance(e, Poisoned) else 34
    sampler.stop()
    walls["read_s"] = round(time.time() - t, 1)
    rec["clocks"] = sampler.summary()
    rec["walls"], rec["wall_s"] = walls, round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print(f"P222_BENCH done in {rec['wall_s']} s; walls {json.dumps(walls)}; clocks n={rec['clocks']['n']} "
          f"error={rec['clocks']['error']}", flush=True)
    return 0


def self_test():
    """The CPU-checkable wiring: the rules come from the reducer, the call form is the served route's at Mixtral's top-2
    with only the plan or the decode moving, every timed replay follows the spacer, and the order is the reducer's."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    src = src[:src.index("def " + "self_test():")]          # the code under test, never these needles
    assert "gather_div=R.TOPK, lut=lut, dot_bf16=False, **plan" in src and R.TOPK == 2
    assert "None, scatter=order_, lut=lut, dot_bf16=False, **plan" in src
    assert "build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)" in src and R.E == 8
    assert 'pg, pd, lut = plans["T" if arm == R.SPACER ' 'else arm]' in src             # the spacer is T's work
    assert "graphs[R.SPACER][0].replay()      # untimed" in src
    seq = [src.index(x) for x in ("graphs[R.SPACER][0].replay()", "sampler.tag = arm",
                                  "rp[arm].append(replay_ms(graphs[arm][0]))", "sampler.tag = None")]
    assert seq == sorted(seq)                                   # S, then the tag, the timed replay, the untag
    assert "rot = R.arm_order(r)" in src and "for arm in rot:" in src
    assert "m = R.select(sel, len(sel_steps))" in src and "plans = R.plans_for(m)" in src
    seq = [src.index(x) for x in ('rec["select"] = sel', "m = R.select(sel", "sampler = ClockSampler()")]
    assert seq == sorted(seq)                                   # A is swept and M named before B is timed
    assert "outs = [call[proj](li, sel_tabs[sel_steps[0]][li], plan, R.P_LUT)" in src   # the sweep is under prmt
    assert "ref[p] = [call[p](li, sel_tabs[sel_steps[0]][li], R.TODAY, R.T_LUT) for li in range(L)]" in src  # the tree
    # Amendment 2: every arm's compiled metadata, from the pass that compiled it
    assert 'first[(p, R.plan_key(R.TODAY), R.T_LUT)] = (new_variants(kfn, before), "block A reference")' in src
    assert 'first.setdefault((proj, R.plan_key(plan), R.P_LUT), (row["compiled"], "block A sweep"))' in src
    assert "arm_compiled(new_variants(kfn, before), first.get(key))" in src
    row = {"n_regs": 64, "n_spills": 0, "shared": 6400, "num_warps": 4, "num_stages": 3}
    assert arm_compiled([row], None) == [dict(row, source="block B metadata pass")]
    assert arm_compiled([], ([row], "block A sweep")) == [dict(row, source="block A sweep")]
    assert arm_compiled([], None)[0]["source"] is None and "unread" in arm_compiled([], ([], "block A sweep"))[0]
    print("p222_bench self-test OK (17 checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--layers", type=int, default=R.LAYERS)
    ap.add_argument("--reps", type=int, default=len(R.READ_STEPS))
    ap.add_argument("--grid", type=int, default=len(R.GRID))
    ap.add_argument("--select-steps", type=int, default=len(R.SELECT_STEPS))
    ap.add_argument("--shakedown", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        try:
            self_test()
        except AssertionError as e:
            print(f"p222_bench self-test FAILED: {e}")
            sys.exit(1)
        sys.exit(0)
    if not args.out:
        ap.error("--out is required")
    sys.exit(main(args))
