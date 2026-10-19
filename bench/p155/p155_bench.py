# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P155's microbench (bench/p155/PREREG-p155.md): K25's plan at Qwen3-30B-A3B's W16 shapes, on S0's real router picks.

K25 (grouped-nf4-gemm ``nf4_smallm.gemm_nf4_grouped_smallm``) is called exactly as e4b's served route calls it on the
NF4 store at 16 rows (``hot_residency._fused_over_stack``, K23's lean glue): gate_up reads the 16 token rows through the
tile table's ``order`` (``gather_div=8``), and down reads the sorted rows and stores into the caller's order
(``scatter=order``), over ``int4_b32.build_group_tiles_fused(ids, 128, 16, lean=True, sorted_ids=True)``, at the served
precision (select-tree decode, TF32 on fp32 weights). Full-size random NF4 stores for all 48 MoE layers (16.3 GB, the
arena's size), so no replay is served from L2. The routing is P153's router replay, rows 0-15 (S0), 128 steps.

- IDENTITY: every grid plan's outputs (144 plans) on step 0, all layers, both projections, against today's plan (``torch.equal``).
  A plan that fails to compile or launch is recorded with its error and excluded; one that moves a bit is excluded.
- SELECTION: per projection, each plan's graph (48 calls) on the 8 selection steps; ``p155_reduce.select`` carries three.
- CONFIRMATION: on the 64 held-out steps, one paired rep each: today (T), today again (T2, the self-pair), the carried
  C1..C3 and today over the compacted table (K, reported). Each arm's graph holds both GEMMs of all 48 layers. Capture
  and replay orders rotate per rep (``p155_reduce.rotation``); replays interleave arm by arm. After the replays, every
  arm's outputs are compared with T's, bit for bit.

The record holds the raw replay times; ``p155_reduce.py`` derives the selection and the verdict from them.

    python p155_bench.py --rr rr.json.pt --out bench_p155.json
    python p155_bench.py --rr rr.json.pt --out shake.json --shakedown --layers 4 --select-steps 2 --confirm-reps 8
    python p155_bench.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p155_reduce as R  # noqa: E402  (the registered rules: the staged sibling)


class Poisoned(RuntimeError):
    """A failure left the CUDA context unusable; nothing after it can be timed."""


def s0_picks(rr_path):
    """S0's picks from P153's router replay, checked before any data: the pinned file, the slice's hash, the shape and
    dtype, ids in [0, 128), each row's 8 picks distinct, and P153's distinct total. Returns (picks [48, 16, 128, 8]
    int16, the slice's sha256)."""
    import torch
    raw = open(rr_path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != R.RR_SHA256:
        raise SystemExit(f"P155_ROUTING_FAULT rr.json.pt sha256 is not the pinned {R.RR_SHA256[:8]}")
    d = torch.load(rr_path, weights_only=True)
    picks = d["picks"]
    if tuple(picks.shape) != (R.LAYERS, 2 * R.TOKENS, R.STEPS, R.TOPK) or picks.dtype != torch.int16:
        raise SystemExit(f"P155_ROUTING_FAULT picks {tuple(picks.shape)} {picks.dtype}")
    s0 = picks[:, 0:R.TOKENS].contiguous()
    sha = hashlib.sha256(s0.numpy().astype("<i2").tobytes()).hexdigest()
    if sha != R.S0_SHA256:
        raise SystemExit(f"P155_ROUTING_FAULT S0's picks hash {sha[:8]}, pinned {R.S0_SHA256[:8]}")
    if int(s0.min()) < 0 or int(s0.max()) >= R.E:
        raise SystemExit("P155_ROUTING_FAULT an expert id outside [0, 128)")
    srt = s0.sort(dim=-1).values
    if bool((srt[..., 1:] == srt[..., :-1]).any()):
        raise SystemExit("P155_ROUTING_FAULT a row picks one expert twice")
    return s0, sha


def compiled_variants(fn):
    """Every compiled variant in a Triton JIT function's per-device caches, {key: CompiledKernel}; empty if this Triton
    lays them out differently (the per-plan metadata is REPORTED, never a reason to stop)."""
    out = {}
    try:
        for dev in getattr(fn, "device_caches", {}).values():
            cache = dev[0] if isinstance(dev, tuple) else dev
            if isinstance(cache, dict):
                out.update(cache)
    except Exception:                                # noqa: BLE001  (reported only)
        return {}
    return out


def new_variants(fn, before):
    """The variants compiled since ``before`` (a key set): registers, spills and shared memory per variant, as Triton
    compiled them for this launch, the inputs of theoretical occupancy."""
    rows = []
    for key, k in compiled_variants(fn).items():
        if key in before:
            continue
        try:
            m = getattr(k, "metadata", None)
            rows.append({"n_regs": getattr(k, "n_regs", None), "n_spills": getattr(k, "n_spills", None),
                         "shared": getattr(m, "shared", None), "num_warps": getattr(m, "num_warps", None),
                         "num_stages": getattr(m, "num_stages", None)})
        except Exception as e:                       # noqa: BLE001  (reported only)
            rows.append({"unread": type(e).__name__})
    return rows


def ncu_target(a):
    """What Nsight Compute captures (REPORTED only): layer 0 at the first confirmation step's real routing, at today's
    plan. Three warm rounds of (gate_up, down), then the round the runner's ``ncu --launch-skip 6 --launch-count 2``
    measures. Prints nothing, so ncu's CSV stays clean."""
    import torch
    import nf4_smallm
    from int4_b32 import build_group_tiles_fused
    k25 = nf4_smallm.gemm_nf4_grouped_smallm
    s0, _ = s0_picks(a.rr)
    g = torch.Generator(device="cuda").manual_seed(155)
    st = {proj: (torch.randint(0, 256, (R.E, N, K // 2), dtype=torch.uint8, generator=g, device="cuda"),
                 torch.rand(R.E, N, K // 64, generator=g, device="cuda") * 0.05 + 0.01)
          for proj, (N, K) in R.SHAPES.items()}
    x = (torch.randn(R.TOKENS, R.SHAPES["gu"][1], generator=g, device="cuda") / 4).to(torch.bfloat16)
    h = (torch.randn(R.TOKENS * R.TOPK, R.SHAPES["dn"][1], generator=g, device="cuda") / 4).to(torch.bfloat16)
    ids = s0[0, :, R.CONFIRM_STEPS[0], :].reshape(-1).to(torch.long).cuda()
    row0, rows, grp, order, _counts, _sids = build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)
    for _ in range(4):
        k25(x, *st["gu"], row0, rows, grp, order, gather_div=R.TOPK, lut="tree", dot_bf16=False, **R.TODAY)
        k25(h, *st["dn"], row0, rows, grp, None, scatter=order, lut="tree", dot_bf16=False, **R.TODAY)
    torch.cuda.synchronize()
    return 0


def n_real_table(s0, layers):
    """[step][layer] distinct experts across the 16 rows (one tile each: no expert gets more than 16 rows)."""
    return [[len(set(s0[li, :, s, :].reshape(-1).tolist())) for li in range(layers)] for s in range(R.STEPS)]


def main(a):
    import torch
    import nf4_smallm
    from int4_b32 import build_group_tiles_fused
    k25 = nf4_smallm.gemm_nf4_grouped_smallm
    dev = "cuda"
    t0 = time.time()
    walls = {}
    s0, s0_sha = s0_picks(a.rr)
    L = a.layers
    sel_steps = list(R.SELECT_STEPS[:a.select_steps])
    conf_steps = list(R.CONFIRM_STEPS[:a.confirm_reps])
    if not a.shakedown and (L != R.LAYERS or sel_steps != list(R.SELECT_STEPS) or conf_steps != list(R.CONFIRM_STEPS)):
        raise SystemExit("reading knobs are frozen: --layers, --select-steps and --confirm-reps are shakedown-only")
    cap = torch.cuda.get_device_capability()
    rec = {"lane": "P155", "shakedown": bool(a.shakedown), "device": torch.cuda.get_device_name(), "cc": list(cap),
           "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "torch": torch.__version__,
           "layers": L, "rr_sha256": R.RR_SHA256, "s0_sha256": s0_sha, "grid": R.GRID, "today": R.TODAY,
           "order": R.plan_order(), "select_steps": sel_steps, "confirm_steps": conf_steps, "iters": R.ITERS}
    try:
        import triton
        rec["triton"] = triton.__version__
    except Exception as e:                       # recorded, never fatal: the kernel module imported it already
        rec["triton"] = f"unread: {type(e).__name__}"
    rec["n_real"] = n_real_table(s0, L)
    walls["routing_s"] = round(time.time() - t0, 1)

    # ---- the stores: random NF4 bytes and fp32 absmax, the served layout, every layer distinct
    t = time.time()
    g = torch.Generator(device=dev).manual_seed(155)
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

    # ---- the tile tables, as the served route builds them; the compacted table is the real-tile prefix
    t = time.time()
    need = sorted({R.IDENTITY_STEP, *sel_steps, *conf_steps})
    full, compact, faults = {}, {}, []
    for s in need:
        full[s], compact[s] = [], []
        for li in range(L):
            ids = s0[li, :, s, :].reshape(-1).to(torch.long).to(dev)
            row0, rows, grp, order, _counts, _sids = build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)
            full[s].append((row0, rows, grp, order))
            rv = rows.tolist()
            n = sum(1 for v in rv if v > 0)
            if n != rec["n_real"][s][li] or any(v <= 0 for v in rv[:n]) or any(v != 0 for v in rv[n:]):
                faults.append({"step": s, "layer": li, "real": n, "distinct": rec["n_real"][s][li],
                               "fault": "the real tiles are not a prefix of one tile per expert"})
            compact[s].append((row0[:n], rows[:n], grp[:n], order))
    torch.cuda.synchronize()
    rec["tile_faults"] = faults
    rec["tiles_budget"] = int(full[need[0]][0][0].numel())
    walls["tiles_s"] = round(time.time() - t, 1)

    def gu(li, tab, plan):                        # the served gate_up call: token rows, gathered through order
        row0, rows, grp, order = tab
        p, am = W[li]["gu"]
        return k25(X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK, lut="tree", dot_bf16=False, **plan)

    def dn(li, tab, plan):                        # the served down call: sorted rows in, stored to the caller's order
        row0, rows, grp, order = tab
        p, am = W[li]["dn"]
        return k25(H[li], p, am, row0, rows, grp, None, scatter=order, lut="tree", dot_bf16=False, **plan)

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

    # ---- IDENTITY and SELECTION, plan by plan in the registered shuffled order; today's plan also opens and closes
    # each projection's sweep (the drift probe, reported)
    t = time.time()
    kfn = nf4_smallm._gemm_nf4_grouped_smallm
    ref, rec["today_compiled"] = {}, {}
    for p in ("gu", "dn"):
        before = set(compiled_variants(kfn))
        ref[p] = [call[p](li, full[R.IDENTITY_STEP][li], R.TODAY) for li in range(L)]
        rec["today_compiled"][p] = new_variants(kfn, before)
    torch.cuda.synchronize()
    sel, drift = {"gu": [], "dn": []}, {}
    try:
        for proj in ("gu", "dn"):
            probe0 = statistics.fmean(statistics.median(proj_replays(proj, R.TODAY, s)) for s in sel_steps)
            for gi in rec["order"]:
                plan = R.GRID[gi]
                row = {"plan": dict(plan), "identical": None, "error": None, "step_replays_ms": None}
                try:
                    tc = time.time()
                    before = set(compiled_variants(kfn))
                    outs = [call[proj](li, full[R.IDENTITY_STEP][li], plan) for li in range(L)]
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
                print(f"P155_PLAN {proj} {R.plan_key(plan)} identical={row['identical']} "
                      f"err={bool(row['error'])} first={row.get('first_call_s')}", flush=True)
            probe1 = statistics.fmean(statistics.median(proj_replays(proj, R.TODAY, s)) for s in sel_steps)
            drift[proj] = probe1 / probe0
    except Poisoned as e:
        rec.update(select=sel, poisoned=str(e), walls=walls, wall_s=round(time.time() - t0, 1))
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        print(f"P155_POISONED {e}", flush=True)
        return 32
    rec["select"], rec["drift"] = sel, drift
    walls["select_s"] = round(time.time() - t, 1)
    carried = R.select(sel, len(sel_steps))
    rec["carried"] = carried
    print("P155_CARRIED " + json.dumps(carried), flush=True)

    # ---- CONFIRMATION: one paired, rotated rep per held-out step; GO is read here only
    t = time.time()
    arms = R.confirm_arms(len(carried))
    plans = {"T": (R.TODAY, R.TODAY, False), "T2": (R.TODAY, R.TODAY, False), "K": (R.TODAY, R.TODAY, True)}
    for i, c in enumerate(carried):
        plans[f"C{i + 1}"] = (c["gu"], c["dn"], False)
    reps = []
    for r, s in enumerate(conf_steps):
        rot = R.rotation(arms, r)
        graphs = {}
        for arm in rot:
            pg, pd, cmp_ = plans[arm]
            tabs = compact[s] if cmp_ else full[s]
            graphs[arm] = capture(lambda tabs=tabs, pg=pg, pd=pd: [(gu(li, tabs[li], pg), dn(li, tabs[li], pd))
                                                                   for li in range(L)])
        rp = {arm: [] for arm in rot}
        for _ in range(R.ITERS):
            for arm in rot:
                rp[arm].append(replay_ms(graphs[arm][0]))
        ref_outs = graphs["T"][1]
        ident = {arm: all(torch.equal(o[0], q[0]) and torch.equal(o[1], q[1]) for o, q in zip(graphs[arm][1], ref_outs))
                 for arm in rot if arm != "T"}
        reps.append({"step": s, "rotation": rot, "replays_ms": rp, "identical": ident})
        del graphs, ref_outs
        torch.cuda.empty_cache()
        print(f"P155_REP {r} step {s} " + " ".join(f"{a_}={statistics.median(rp[a_]):.3f}" for a_ in arms)
              + f" identical={all(ident.values())}", flush=True)
    rec["confirm"] = {"arms": arms, "reps": reps}
    walls["confirm_s"] = round(time.time() - t, 1)
    rec["walls"], rec["wall_s"] = walls, round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print(f"P155_BENCH done in {rec['wall_s']} s; walls {json.dumps(walls)}", flush=True)
    return 0


def self_test():
    """The CPU-checkable wiring: the rules come from the reducer, the call form is the served route's, the grid holds
    today's plan, the steps split cleanly, and every arm takes every rotation position equally often."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    assert "gather_div=R.TOPK, lut=\"tree\", dot_bf16=False, **plan" in src
    assert "None, scatter=order, lut=\"tree\", dot_bf16=False, **plan" in src
    assert "build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)" in src
    assert R.TODAY in R.GRID and len(R.GRID) == 144 and len({R.plan_key(p) for p in R.GRID}) == 144
    assert not set(R.SELECT_STEPS) & set(R.CONFIRM_STEPS) and R.IDENTITY_STEP in R.SELECT_STEPS
    assert max(R.SELECT_STEPS) < min(R.CONFIRM_STEPS) and len(R.CONFIRM_STEPS) == 64
    arms = R.confirm_arms(3)
    pos = {a_: [0] * len(arms) for a_ in arms}
    for r in range(len(R.CONFIRM_STEPS) - len(R.CONFIRM_STEPS) % len(arms)):
        for i, a_ in enumerate(R.rotation(arms, r)):
            pos[a_][i] += 1
    assert all(len(set(v)) == 1 for v in pos.values()), pos
    assert sorted(R.plan_order()) == list(range(len(R.GRID))) and R.plan_order() == R.plan_order()
    print("p155_bench self-test OK (5 cases)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr")
    ap.add_argument("--out")
    ap.add_argument("--layers", type=int, default=R.LAYERS)
    ap.add_argument("--select-steps", type=int, default=len(R.SELECT_STEPS))
    ap.add_argument("--confirm-reps", type=int, default=len(R.CONFIRM_STEPS))
    ap.add_argument("--shakedown", action="store_true")
    ap.add_argument("--ncu-target", action="store_true", help="the calls Nsight Compute captures (reported only)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        try:
            self_test()
        except AssertionError as e:
            print(f"p155_bench self-test FAILED: {e}")
            sys.exit(1)
        sys.exit(0)
    if args.ncu_target:
        if not args.rr:
            ap.error("--rr is required")
        sys.exit(ncu_target(args))
    if not (args.rr and args.out):
        ap.error("--rr and --out are required")
    sys.exit(main(args))
