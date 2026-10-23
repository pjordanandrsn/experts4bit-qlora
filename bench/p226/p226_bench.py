# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P226's microbench (bench/p226/PREREG-p226.md): the K25 plan at bucket 32. P156's sweep and P160's instrument at
Qwen3-30B-A3B's W32 shapes (32 token rows, 256 routed rows), on the box's own routing capture (``p226_capture.py``).

K25 (grouped-nf4-gemm ``nf4_smallm.gemm_nf4_grouped_smallm``) is called exactly as e4b's served route calls it on the
NF4 store (P156's call form): gate_up reads the 32 token rows through the lean tile table's ``order`` (``gather_div=8``),
and down reads the sorted rows and stores into the caller's order (``scatter=order``), TF32 on fp32 weights, decode
``prmt``. The tables are the served builder's (``build_group_tiles_fused(ids, 128, 16, lean=True, sorted_ids=True)``);
at 32 rows an expert can take more than 16 routed rows, so the check is one 16-row tile per 16 rows of each expert.
Full-size random NF4 stores for all 48 MoE layers, so no replay is served from L2.

1. SELECTION (reported only): every plan of P156's 144-plan grid, per projection, in the registered shuffled order, on
   the 8 selection steps; each plan's outputs are compared with c1's at the first selection step, bit for bit (a plan
   changes tiling and staging, never the values), and only identical plans are timed. c1 opens and closes each
   projection's sweep (the drift probe, reported). ``p226_reduce.select`` carries three pairs, never c1's own.
2. CONFIRMATION (the ruling): one paired rep per held-out step (64-127) in its Williams row over C, C2, K1-K3 and D. Each
   rep captures the spacer S (C's calls with their own outputs) and every arm's graph (both GEMMs of all 48 layers), then
   replays the arms interleaved, 10 timed replays each, every one right after an untimed replay of S. An NVML thread
   samples clocks, power and the enforced limit, tagged with the arm replaying. After the replays every arm's outputs
   are compared with C's, bit for bit.

    python p226_bench.py --picks capture.pt --out bench_p226.json
    python p226_bench.py --picks capture.pt --out shake.json --shakedown --layers 4 --select-steps 2 --confirm-reps 6
    python p226_bench.py --self-test
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p226_reduce as R  # noqa: E402  (the registered rules: the staged sibling)


class Poisoned(RuntimeError):
    """A failure left the CUDA context unusable; nothing after it can be timed."""


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


def load_picks(path):
    """The capture's picks, checked before any data: shape and dtype, ids in [0, 128), each row's 8 picks distinct.
    Returns the [48, 32, 128, 8] int16 tensor and the sha256 of its <i2 bytes (the reducer matches it to the capture)."""
    import torch
    picks = torch.load(path, weights_only=True)["picks"]
    if tuple(picks.shape) != (R.LAYERS, R.TOKENS, R.STEPS, R.TOPK) or picks.dtype != torch.int16:
        raise SystemExit(f"P226_ROUTING_FAULT picks {tuple(picks.shape)} {picks.dtype}")
    if int(picks.min()) < 0 or int(picks.max()) >= R.E:
        raise SystemExit("P226_ROUTING_FAULT an expert id outside [0, 128)")
    srt = picks.sort(dim=-1).values
    if bool((srt[..., 1:] == srt[..., :-1]).any()):
        raise SystemExit("P226_ROUTING_FAULT a row picks one expert twice")
    return picks, hashlib.sha256(picks.contiguous().numpy().astype("<i2").tobytes()).hexdigest()


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


def main(a):
    import torch
    import nf4_smallm
    from int4_b32 import build_group_tiles_fused
    k25 = nf4_smallm.gemm_nf4_grouped_smallm
    dev = "cuda"
    t0 = time.time()
    walls = {}
    picks, sha = load_picks(a.picks)
    L = a.layers
    sel_steps = list(R.SELECT_STEPS[:a.select_steps])
    conf_steps = list(R.CONFIRM_STEPS[:a.confirm_reps])
    if not a.shakedown and (L != R.LAYERS or sel_steps != list(R.SELECT_STEPS) or conf_steps != list(R.CONFIRM_STEPS)):
        raise SystemExit("reading knobs are frozen: --layers, --select-steps and --confirm-reps are shakedown-only")
    cap = torch.cuda.get_device_capability()
    rec = {"lane": "P226", "shakedown": bool(a.shakedown), "device": torch.cuda.get_device_name(), "cc": list(cap),
           "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "torch": torch.__version__,
           "layers": L, "picks_sha256": sha, "arms": list(R.ARMS), "spacer": R.SPACER, "lut": R.LUT,
           "c1": R.C1, "today": R.TODAY, "grid": R.GRID, "order": R.plan_order(), "iters": R.ITERS}
    try:
        import triton
        rec["triton"] = triton.__version__
    except Exception as e:                       # recorded, never fatal: the kernel module imported it already
        rec["triton"] = f"unread: {type(e).__name__}"
    walls["routing_s"] = round(time.time() - t0, 1)

    # ---- the stores: random NF4 bytes and fp32 absmax, the served layout, every layer distinct (P156's)
    t = time.time()
    g = torch.Generator(device=dev).manual_seed(226)
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

    # ---- the tile tables, as the served route builds them; the real tiles must be one per 16 rows of each expert
    t = time.time()
    full, faults = {}, []
    for s in sorted({*sel_steps, *conf_steps}):
        full[s] = []
        for li in range(L):
            ids = picks[li, :, s, :].reshape(-1).to(torch.long).to(dev)
            row0, rows, grp, order, _counts, _sids = build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)
            full[s].append((row0, rows, grp, order))
            rv = rows.tolist()
            counts = torch.bincount(ids, minlength=R.E).tolist()
            want = R.tiles_for(counts)
            n = sum(1 for v in rv if v > 0)
            if n != want or any(v <= 0 for v in rv[:n]) or any(v != 0 for v in rv[n:]) or sum(rv) != R.TOKENS * R.TOPK:
                faults.append({"step": s, "layer": li, "real": n, "want": want,
                               "fault": "the real tiles are not a prefix of one tile per 16 rows of each expert"})
    torch.cuda.synchronize()
    walls["tiles_s"] = round(time.time() - t, 1)

    def gu(li, tab, plan):                        # the served gate_up call: token rows, gathered through order
        row0, rows, grp, order = tab
        p, am = W[li]["gu"]
        return k25(X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK, lut=R.LUT, dot_bf16=False, **plan)

    def dn(li, tab, plan):                        # the served down call: sorted rows in, stored to the caller's order
        row0, rows, grp, order = tab
        p, am = W[li]["dn"]
        return k25(H[li], p, am, row0, rows, grp, None, scatter=order, lut=R.LUT, dot_bf16=False, **plan)

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

    def proj_replays(proj, plan, s):
        gr, outs = capture(lambda: [call[proj](li, full[s][li], plan) for li in range(L)])
        xs = [replay_ms(gr) for _ in range(R.ITERS)]
        del gr, outs
        return xs

    # ---- 1. SELECTION (reported): every grid plan, identity against c1 at the first selection step, then timed
    t = time.time()
    kfn = nf4_smallm._gemm_nf4_grouped_smallm
    s_id = sel_steps[0]
    ref = {p: [call[p](li, full[s_id][li], R.C1[p]) for li in range(L)] for p in ("gu", "dn")}
    torch.cuda.synchronize()
    sel, drift = {"gu": [], "dn": []}, {}
    try:
        for proj in ("gu", "dn"):
            probe0 = statistics.fmean(statistics.median(proj_replays(proj, R.C1[proj], s)) for s in sel_steps)
            for gi in rec["order"]:
                plan = R.GRID[gi]
                row = {"plan": dict(plan), "identical": None, "error": None, "step_replays_ms": None}
                try:
                    tc = time.time()
                    before = set(compiled_variants(kfn))
                    outs = [call[proj](li, full[s_id][li], plan) for li in range(L)]
                    torch.cuda.synchronize()
                    row["first_call_s"] = round(time.time() - tc, 2)
                    row["compiled"] = new_variants(kfn, before)
                    diff = [li for li in range(L) if not torch.equal(outs[li], ref[proj][li])]
                    row["identical"] = not diff
                    if diff:
                        d = (outs[diff[0]].float() - ref[proj][diff[0]].float()).abs()
                        row["mismatch"] = {"layers": diff[:8], "n_layers": len(diff), "max_abs": float(d.max()),
                                           "n_elems": int((d > 0).sum())}
                    del outs
                    if row["identical"]:
                        row["step_replays_ms"] = [proj_replays(proj, plan, s) for s in sel_steps]
                except Exception as e:                   # a plan that cannot compile or launch: recorded, excluded
                    row["identical"], row["step_replays_ms"] = None, None
                    row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    check_context()
                sel[proj].append(row)
                torch.cuda.empty_cache()
                print(f"P226_PLAN {proj} {R.plan_key(plan)} identical={row['identical']} err={bool(row['error'])} "
                      f"first={row.get('first_call_s')}", flush=True)
            probe1 = statistics.fmean(statistics.median(proj_replays(proj, R.C1[proj], s)) for s in sel_steps)
            drift[proj] = probe1 / probe0
    except Poisoned as e:
        rec.update(selection={"steps": sel_steps, "rows": sel}, poisoned=str(e), walls=walls,
                   wall_s=round(time.time() - t0, 1))
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        print(f"P226_POISONED {e}", flush=True)
        return 32
    del ref
    rec["selection"], rec["drift"] = {"steps": sel_steps, "rows": sel}, drift
    walls["select_s"] = round(time.time() - t, 1)
    carried = R.select(sel, len(sel_steps))
    rec["carried"] = carried
    print("P226_CARRIED " + json.dumps(carried), flush=True)
    if len(carried) != R.CARRY:
        rec.update(walls=walls, wall_s=round(time.time() - t0, 1))
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        print(f"P226_CARRY_SHORT {len(carried)} pairs", flush=True)
        return 33
    plans = R.plans_for(carried)
    rec["plans"] = {arm: [p for p in v] for arm, v in plans.items()}

    def graph_fn(arm, tabs):
        pg, pd = plans["C" if arm == R.SPACER else arm]
        return lambda: [(gu(li, tabs[li], pg), dn(li, tabs[li], pd)) for li in range(L)]

    # ---- each arm's compiled metadata (registers, spills, shared memory), per projection, on the first held-out step
    rec["compiled"] = {}
    seen = {}
    for arm in R.ARMS:
        pg, pd = plans[arm]
        rec["compiled"][arm] = {}
        for proj, plan in (("gu", pg), ("dn", pd)):
            key = (proj, R.plan_key(plan))
            if key in seen:
                rec["compiled"][arm][proj] = seen[key]
                continue
            before = set(compiled_variants(kfn))
            for li in range(L):
                call[proj](li, full[conf_steps[0]][li], plan)
            torch.cuda.synchronize()
            seen[key] = rec["compiled"][arm][proj] = new_variants(kfn, before)

    # ---- 2. CONFIRMATION (the ruling): one paired rep per held-out step in its Williams row
    t = time.time()
    sampler = ClockSampler()
    sampler.start()
    conf = {"steps": conf_steps, "tile_faults": faults, "reps": []}
    rec["confirmation"] = conf
    try:
        for r, s in enumerate(conf_steps):
            tabs = full[s]
            rot = R.arm_order(r)
            try:
                graphs = {R.SPACER: capture(graph_fn(R.SPACER, tabs))}   # its own buffers: never timed
                for arm in rot:
                    graphs[arm] = capture(graph_fn(arm, tabs))
            except Exception:
                check_context()
                raise
            rp, n_spacer = {arm: [] for arm in rot}, 0
            for _ in range(R.ITERS):
                for arm in rot:
                    graphs[R.SPACER][0].replay()      # untimed: every timed replay follows the same work
                    n_spacer += 1
                    sampler.tag = arm
                    rp[arm].append(replay_ms(graphs[arm][0]))
                    sampler.tag = None
            ref_c = graphs["C"][1]
            same = lambda x, y: all(torch.equal(o[0], q[0]) and torch.equal(o[1], q[1])  # noqa: E731
                                    for o, q in zip(x, y))
            ident = {arm: same(graphs[arm][1], ref_c) for arm in rot if arm != "C"}
            conf["reps"].append({"step": s, "order": rot, "replays_ms": rp, "spacer_replays": n_spacer, "identical": ident})
            del graphs, ref_c
            torch.cuda.empty_cache()
            print(f"P226_REP {r} step {s} " + " ".join(f"{a_}={statistics.median(rp[a_]):.3f}" for a_ in R.ARMS)
                  + f" identical={all(ident.values())}", flush=True)
    except Poisoned as e:
        sampler.stop()
        rec.update(clocks=sampler.summary(), poisoned=str(e), walls=walls, wall_s=round(time.time() - t0, 1))
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        print(f"P226_POISONED {e}", flush=True)
        return 32
    sampler.stop()
    walls["confirm_s"] = round(time.time() - t, 1)
    rec["clocks"] = sampler.summary()
    rec["walls"], rec["wall_s"] = walls, round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print(f"P226_BENCH done in {rec['wall_s']} s; walls {json.dumps(walls)}; clocks n={rec['clocks']['n']} "
          f"error={rec['clocks']['error']}", flush=True)
    return 0


def self_test():
    """The CPU-checkable wiring: the rules come from the reducer, the call form is the served route's with only the plan
    moving, identity is against c1, every timed replay follows the spacer, and the order is the reducer's."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    src = src[:src.index("def " + "self_test():")]          # the code under test, never these needles
    assert "gather_div=R.TOPK, lut=R.LUT, dot_bf16=False, **plan" in src
    assert "None, scatter=order, lut=R.LUT, dot_bf16=False, **plan" in src
    assert "build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)" in src
    assert "want = R.tiles_for(counts)" in src                 # one tile per 16 rows of each expert, not one an expert
    assert 'pg, pd = plans["C" if arm == R.SPACER ' 'else arm]' in src       # the spacer is C's (c1's) work
    assert "graphs[R.SPACER][0].replay()      # untimed" in src
    seq = [src.index(x) for x in ("graphs[R.SPACER][0].replay()", "sampler.tag = arm",
                                  "rp[arm].append(replay_ms(graphs[arm][0]))", "sampler.tag = None")]
    assert seq == sorted(seq)                                   # S, then the tag, the timed replay, the untag
    assert "rot = R.arm_order(r)" in src and "for arm in rot:" in src
    assert "ref = {p: [call[p](li, full[s_id][li], R.C1[p])" in src   # sweep identity is against c1
    assert "carried = R.select(sel, len(sel_steps))" in src and "plans = R.plans_for(carried)" in src
    assert R.LUT == "prmt" and R.TOKENS == 32 and R.TOKENS * R.TOPK == 256
    assert not set(R.SELECT_STEPS) & set(R.CONFIRM_STEPS)
    assert R.ARMS[0] == "C" and R.plans_for([{"gu": R.TODAY, "dn": R.TODAY}] * 3)["C"] == (R.C1["gu"], R.C1["dn"])
    print("p226_bench self-test OK (13 checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks")
    ap.add_argument("--out")
    ap.add_argument("--layers", type=int, default=R.LAYERS)
    ap.add_argument("--select-steps", type=int, default=len(R.SELECT_STEPS))
    ap.add_argument("--confirm-reps", type=int, default=len(R.CONFIRM_STEPS))
    ap.add_argument("--shakedown", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-args", action="store_true", help="apply the argument checks a real run applies, then stop")
    args = ap.parse_args()
    if args.self_test:
        try:
            self_test()
        except AssertionError as e:
            print(f"p226_bench self-test FAILED: {e}")
            sys.exit(1)
        sys.exit(0)
    if not (args.picks and args.out):
        ap.error("--picks and --out are required")
    if args.check_args:
        print("ARGS OK")
        sys.exit(0)
    sys.exit(main(args))
