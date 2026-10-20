# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P160's microbench (bench/p160/PREREG-p160.md): decode x plan. K39's decode (K25's ``lut="prmt"``) at P156's carried
tile plans against prmt at today's plan (qwen3_moe's served default since P158), at Qwen3-30B-A3B's W16 shapes, on a
router-replay stream P156 never timed.

K25 (grouped-nf4-gemm ``nf4_smallm.gemm_nf4_grouped_smallm``) is called exactly as e4b's served route calls it on the
NF4 store at 16 rows (P156's call form): gate_up reads the 16 token rows through the lean tile table's ``order``
(``gather_div=8``), and down reads the sorted rows and stores into the caller's order (``scatter=order``), TF32 on fp32
weights. Full-size random NF4 stores for all 48 MoE layers (16.3 GB), so no replay is served from L2.

Six arms, every plan named before data (``p160_reduce.PLANS``): P and P2 (prmt at today's plan, twice: the self-pair),
C1-C3 (prmt at P156's carried plan pairs), T (the tree at today's plan). Two blocks, in order: S1, P153's second stream
(rows 16-31), steps 0-125, the READING; S0R, S0 (rows 0-15) steps 64-123, P156's own held-out steps, reported.

Each rep (one router-replay step) captures the spacer S (P's calls with their own outputs) and every arm's graph (both
GEMMs of all 48 layers) in the rep's Williams row, then replays the arms interleaved, 10 timed replays each, every one
right after an untimed replay of S. An NVML thread samples clocks, power and the enforced limit, tagged with the block
and the arm replaying. After the replays, every arm's outputs are compared with T's, and P2's with P's, bit for bit.

    python p160_bench.py --rr rr.json.pt --out bench_p160.json
    python p160_bench.py --rr rr.json.pt --out shake.json --shakedown --layers 4 --reps 6
    python p160_bench.py --self-test
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
import p160_reduce as R  # noqa: E402  (the registered rules: the staged sibling)


class Poisoned(RuntimeError):
    """A failure left the CUDA context unusable; nothing after it can be timed."""


class _FieldValue(ctypes.Structure):          # nvmlFieldValue_t
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_uint), ("nvmlReturn", ctypes.c_uint),
                ("value", ctypes.c_ulonglong)]


_FI_POWER_INSTANT = 186                       # NVML_FI_DEV_POWER_INSTANT (mW)


class ClockSampler(threading.Thread):
    """NVML samples every ``period`` seconds while :attr:`tag` names (block, arm) replaying, None between replays
    (P157's sampler): SM and memory clocks, the averaged and instantaneous power, the clock-event reasons and the
    enforced power limit. :meth:`summary` gives each block's arms their medians."""

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
        for b in R.BLOCKS:
            out[b] = {}
            for arm in R.ARMS:
                xs = [s for s in self.samples if s[0] == (b, arm)]
                if not xs:
                    continue
                reasons = {}
                for s in xs:
                    reasons[hex(s[4])] = reasons.get(hex(s[4]), 0) + 1
                inst = [s[6] for s in xs if s[6] is not None]
                out[b][arm] = {"n": len(xs), "sm_mhz_median": statistics.median(s[1] for s in xs),
                               "sm_mhz_min": min(s[1] for s in xs), "sm_mhz_max": max(s[1] for s in xs),
                               "mem_mhz_median": statistics.median(s[2] for s in xs),
                               "power_w_median": statistics.median(s[3] for s in xs),
                               "power_instant_w_median": statistics.median(inst) if inst else None,
                               "power_instant_w_max": max(inst) if inst else None,
                               "enforced_limit_w": sorted({s[5] for s in xs}), "reasons": reasons}
        return out


def stream_picks(rr_path):
    """Both streams' picks from P153's router replay, checked before any data: the pinned file, each slice's hash, the
    shape and dtype, ids in [0, 128), each row's 8 picks distinct. Returns ({"S0": ..., "S1": ...} [48, 16, 128, 8]
    int16, their sha256)."""
    import torch
    raw = open(rr_path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != R.RR_SHA256:
        raise SystemExit(f"P160_ROUTING_FAULT rr.json.pt sha256 is not the pinned {R.RR_SHA256[:8]}")
    picks = torch.load(rr_path, weights_only=True)["picks"]
    if tuple(picks.shape) != (R.LAYERS, 2 * R.TOKENS, R.STEPS, R.TOPK) or picks.dtype != torch.int16:
        raise SystemExit(f"P160_ROUTING_FAULT picks {tuple(picks.shape)} {picks.dtype}")
    out, shas = {}, {}
    for name, lo in (("S0", 0), ("S1", R.TOKENS)):
        s = picks[:, lo:lo + R.TOKENS].contiguous()
        shas[name] = hashlib.sha256(s.numpy().astype("<i2").tobytes()).hexdigest()
        if shas[name] != R.ROWS_SHA256[name]:
            raise SystemExit(f"P160_ROUTING_FAULT {name}'s picks hash {shas[name][:8]}, pinned {R.ROWS_SHA256[name][:8]}")
        if int(s.min()) < 0 or int(s.max()) >= R.E:
            raise SystemExit(f"P160_ROUTING_FAULT {name}: an expert id outside [0, 128)")
        srt = s.sort(dim=-1).values
        if bool((srt[..., 1:] == srt[..., :-1]).any()):
            raise SystemExit(f"P160_ROUTING_FAULT {name}: a row picks one expert twice")
        out[name] = s
    return out, shas


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


def n_real_table(s, layers):
    """[step][layer] distinct experts across the 16 rows (one tile each: no expert gets more than 16 rows)."""
    return [[len(set(s[li, :, t, :].reshape(-1).tolist())) for li in range(layers)] for t in range(R.STEPS)]


def main(a):
    import torch
    import nf4_smallm
    from int4_b32 import build_group_tiles_fused
    k25 = nf4_smallm.gemm_nf4_grouped_smallm
    dev = "cuda"
    t0 = time.time()
    walls = {}
    picks, shas = stream_picks(a.rr)
    L = a.layers
    steps = {b: list(spec["steps"][:a.reps]) for b, spec in R.BLOCKS.items()}
    if not a.shakedown and (L != R.LAYERS or any(steps[b] != R.BLOCKS[b]["steps"] for b in R.BLOCKS)):
        raise SystemExit("reading knobs are frozen: --layers and --reps are shakedown-only")
    cap = torch.cuda.get_device_capability()
    rec = {"lane": "P160", "shakedown": bool(a.shakedown), "device": torch.cuda.get_device_name(), "cc": list(cap),
           "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "torch": torch.__version__,
           "layers": L, "rr_sha256": R.RR_SHA256, "rows_sha256": shas, "arms": list(R.ARMS), "spacer": R.SPACER,
           "plans": {arm: list(R.PLANS[arm]) for arm in R.ARMS}, "iters": R.ITERS}
    try:
        import triton
        rec["triton"] = triton.__version__
    except Exception as e:                       # recorded, never fatal: the kernel module imported it already
        rec["triton"] = f"unread: {type(e).__name__}"
    walls["routing_s"] = round(time.time() - t0, 1)

    # ---- the stores: random NF4 bytes and fp32 absmax, the served layout, every layer distinct (P156's)
    t = time.time()
    g = torch.Generator(device=dev).manual_seed(156)
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
        row0, rows, grp, order = tab
        p, am = W[li]["gu"]
        return k25(X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK, lut=lut, dot_bf16=False, **plan)

    def dn(li, tab, plan, lut):                   # the served down call: sorted rows in, stored to the caller's order
        row0, rows, grp, order = tab
        p, am = W[li]["dn"]
        return k25(H[li], p, am, row0, rows, grp, None, scatter=order, lut=lut, dot_bf16=False, **plan)

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
        src = picks[R.STREAM[b]]
        for li in range(L):
            ids = src[li, :, s, :].reshape(-1).to(torch.long).to(dev)
            row0, rows, grp, order, _counts, _sids = build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)
            tabs.append((row0, rows, grp, order))
            rv = rows.tolist()
            n = sum(1 for v in rv if v > 0)
            want = len(set(src[li, :, s, :].reshape(-1).tolist()))
            if n != want or any(v <= 0 for v in rv[:n]) or any(v != 0 for v in rv[n:]):
                faults.append({"step": s, "layer": li, "real": n, "distinct": want,
                               "fault": "the real tiles are not a prefix of one tile per expert"})
        return tabs, faults

    def graph_fn(arm, tabs):
        pg, pd, lut = R.PLANS["P" if arm == R.SPACER else arm]
        return lambda: [(gu(li, tabs[li], pg, lut), dn(li, tabs[li], pd, lut)) for li in range(L)]

    # ---- each arm's compiled metadata (registers, spills, shared memory), per projection, on S1's first step
    t = time.time()
    kfn = nf4_smallm._gemm_nf4_grouped_smallm
    tabs0, _ = tables("S1", steps["S1"][0])
    rec["compiled"] = {}
    seen = {}
    for arm in R.ARMS:
        pg, pd, lut = R.PLANS[arm]
        rec["compiled"][arm] = {}
        for proj, plan, fn in (("gu", pg, gu), ("dn", pd, dn)):
            key = (proj, R.plan_key(plan), lut)
            if key in seen:                       # P2 is P's cubin: the same variants, read once
                rec["compiled"][arm][proj] = seen[key]
                continue
            before = set(compiled_variants(kfn))
            for li in range(L):
                fn(li, tabs0[li], plan, lut)
            torch.cuda.synchronize()
            seen[key] = rec["compiled"][arm][proj] = new_variants(kfn, before)
    walls["compile_s"] = round(time.time() - t, 1)

    # ---- the blocks: S1 (the reading), then S0R (reported); one paired rep per step in its Williams row
    sampler = ClockSampler()
    sampler.start()
    rec["blocks"] = {}
    try:
        for b, spec in R.BLOCKS.items():
            t = time.time()
            blk = {"rows": list(spec["rows"]), "steps": steps[b],
                   "n_real": n_real_table(picks[R.STREAM[b]], L), "tile_faults": [], "reps": []}
            rec["blocks"][b] = blk
            for r, s in enumerate(steps[b]):
                tabs, faults = tables(b, s)
                blk["tile_faults"] += faults
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
                        sampler.tag = (b, arm)
                        rp[arm].append(replay_ms(graphs[arm][0]))
                        sampler.tag = None
                ref = graphs["T"][1]
                same = lambda x, y: all(torch.equal(o[0], q[0]) and torch.equal(o[1], q[1])  # noqa: E731
                                        for o, q in zip(x, y))
                ident = {arm: same(graphs[arm][1], ref) for arm in rot if arm != "T"}
                blk["reps"].append({"step": s, "order": rot, "replays_ms": rp, "spacer_replays": n_spacer,
                                    "identical": ident, "p2_is_p": same(graphs["P2"][1], graphs["P"][1])})
                del graphs, ref, tabs
                torch.cuda.empty_cache()
                print(f"P160_REP {b} {r} step {s} " + " ".join(f"{a_}={statistics.median(rp[a_]):.3f}" for a_ in R.ARMS)
                      + f" identical={all(ident.values())}", flush=True)
            walls[f"{b}_s"] = round(time.time() - t, 1)
    except Poisoned as e:
        sampler.stop()
        rec.update(clocks=sampler.summary(), poisoned=str(e), walls=walls, wall_s=round(time.time() - t0, 1))
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        print(f"P160_POISONED {e}", flush=True)
        return 32
    sampler.stop()
    rec["clocks"] = sampler.summary()
    rec["walls"], rec["wall_s"] = walls, round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print(f"P160_BENCH done in {rec['wall_s']} s; walls {json.dumps(walls)}; clocks n={rec['clocks']['n']} "
          f"error={rec['clocks']['error']}", flush=True)
    return 0


def self_test():
    """The CPU-checkable wiring: the rules come from the reducer, the call form is the served route's with only the plan
    (C1-C3) or the decode (T) moving against P, every timed replay follows the spacer, and the order is the reducer's."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    src = src[:src.index("def " + "self_test():")]          # the code under test, never these needles
    assert "gather_div=R.TOPK, lut=lut, dot_bf16=False, **plan" in src
    assert "None, scatter=order, lut=lut, dot_bf16=False, **plan" in src
    assert "build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)" in src
    assert 'pg, pd, lut = R.PLANS["P" if arm == R.SPACER ' 'else arm]' in src       # the spacer is P's work
    assert "graphs[R.SPACER][0].replay()      # untimed" in src
    seq = [src.index(x) for x in ("graphs[R.SPACER][0].replay()", "sampler.tag = (b, arm)",
                                  "rp[arm].append(replay_ms(graphs[arm][0]))", "sampler.tag = None")]
    assert seq == sorted(seq)                                   # S, then the tag, the timed replay, the untag
    assert "rot = R.arm_order(r)" in src and "for arm in rot:" in src
    assert '"p2_is_p": same(graphs["P2"][1], graphs["P"][1])' in src
    assert (R.T_LUT, R.P_LUT) == ("tree", "prmt")
    P, C1, T = R.PLANS["P"], R.PLANS["C1"], R.PLANS["T"]
    assert P[2] == C1[2] == R.P_LUT and P[:2] == (R.TODAY, R.TODAY) and C1[:2] != P[:2]   # C1: only the plan moves
    assert T[:2] == P[:2] and T[2] == R.T_LUT                                             # T: only the decode moves
    assert R.PLANS["P2"] == P and [R.PLANS[f"C{i + 1}"][:2] for i in range(3)] == [(c["gu"], c["dn"]) for c in R.CARRIED]
    assert R.READING == "S1" and list(R.BLOCKS) == ["S1", "S0R"] and R.STREAM == {"S1": "S1", "S0R": "S0"}
    assert R.BLOCKS["S1"]["rows"] == [16, 32] and R.BLOCKS["S0R"]["rows"] == [0, 16]
    assert set(R.BLOCKS["S0R"]["steps"]) <= set(range(64, 128))          # P156's held-out steps, never its selection
    print("p160_bench self-test OK (15 checks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr")
    ap.add_argument("--out")
    ap.add_argument("--layers", type=int, default=R.LAYERS)
    ap.add_argument("--reps", type=int, default=max(len(s["steps"]) for s in R.BLOCKS.values()))
    ap.add_argument("--shakedown", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        try:
            self_test()
        except AssertionError as e:
            print(f"p160_bench self-test FAILED: {e}")
            sys.exit(1)
        sys.exit(0)
    if not (args.rr and args.out):
        ap.error("--rr and --out are required")
    sys.exit(main(args))
