#!/usr/bin/env python3
"""p141_select.py -- lane P141's phase 1 (bench/p141/PREREG-p141.md): choose K33's plan for each of Mixtral-8x7B's two
expert shapes on THIS card, by a rule registered before any data, then hand the served read (phase 2) its
``GNF4_GEMV_BW_PLAN``. No model is loaded: per layer, a seeded synthetic NF4 store at the served shape.

The calls are the served one-request ones, as P141's call-site trace recorded them on e4b's decode:
  * gate_up (28672, 4096): ONE token row read for both routed slots (``gather_div=2``), two experts;
  * down (4096, 14336): two rows, one per routed slot, two experts.
Each projection runs over all 32 layers (each its own bytes, so nothing stays in L2: K24's lesson), under B1's own
environment (``GNF4_GEMV_BW=1``; decode and PDL at their defaults).

**The rule** (K33's discipline, ``kernel/k33_bench.py``), per projection:
  1. every plan in :data:`PLANS` (K33's twelve) is checked on layer 0 BEFORE any timing: ``bw_prmt32`` bitwise
     ``bw_tree``, and both within the fp32 reference's tolerance (error <= 1.05x the scalar route's, or <= 2^-9 of the
     output scale). A plan that fails to compile or misses the contract is not selectable;
  2. each selectable plan's 32-layer chain is captured in one CUDA graph; after :data:`WARM` untimed rounds, the graphs
     are replayed for :data:`SELECT_ROUNDS` rounds, order reversed every round (CUDA events);
  3. the plan with the lowest median wins; a tie goes to the earlier plan in :data:`PLANS`.
  4. no selectable plan on a projection: ``status`` names it, and the box stops before the fetch (FUNCTION_FAIL).
Reported, never ruled and never quoted as the gain (picking the best of twelve on the same samples biases it upward):
each plan's median, the incumbent's median (``GNF4_GEMV_BW`` unset: the scalar GEMV at these shapes), and the copy floor.

    python p141_select.py --family mixtral --out select.json
    python p141_select.py --family tiny --out select.json        (the dry run's fixture shapes)
    python p141_select.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

#: per family: layers, the served (N, K) of each projection, and the decode call's form (gather_div: token rows read
#: once per routed slot, e4b's lean singleton dispatch for gate_up)
FAMILIES = {"mixtral": {"layers": 32, "top_k": 2, "gate_up": (28672, 4096), "down": (4096, 14336)},
            "tiny": {"layers": 2, "top_k": 2, "gate_up": (512, 512), "down": (512, 256)}}   # p133_tiny.py's fixture
PROJ = ("gate_up", "down")
GATHER = {"gate_up": True, "down": False}
PLANS = [(16, 256, 4, 1), (16, 512, 4, 1), (16, 1024, 4, 1), (16, 256, 8, 1), (16, 512, 8, 1),
         (32, 256, 4, 1), (32, 512, 4, 1), (32, 512, 8, 1), (32, 1024, 8, 1), (16, 256, 4, 2), (16, 512, 4, 2),
         (32, 256, 4, 2)]                                                 # K33's list (kernel/k33_bench.py PLANS)
WARM, SELECT_ROUNDS = 20, 40
TOL_X, TOL_ABS = 1.05, 2.0 ** -9
B1_ENV = {"GNF4_GEMV_BW": "1", "GNF4_GEMV_BW_DECODE": None, "GNF4_PDL": None, "GNF4_GEMV_BW_PLAN": None,
          "GNF4_GEMV_DOTPAD": None}
INCUMBENT_ENV = {"GNF4_GEMV_BW": None, "GNF4_GEMV_BW_DECODE": None, "GNF4_PDL": None, "GNF4_GEMV_BW_PLAN": None,
                 "GNF4_GEMV_DOTPAD": None}


def plan_env(selected: dict, family: str) -> str:
    """``GNF4_GEMV_BW_PLAN`` for the selected plans: ``"N,K=bn,kc,w,sk;N,K=..."``, gate_up first."""
    spec = FAMILIES[family]
    return ";".join(f"{spec[p][0]},{spec[p][1]}=" + ",".join(str(int(x)) for x in selected[p]) for p in PROJ)


def parse_plan_env(s: str) -> dict:
    """The inverse: ``{(N, K): (bn, kc, w, sk)}``, refusing anything malformed (the kernel's own parser would raise)."""
    out = {}
    for entry in (s or "").split(";"):
        shape, sep, cfg = entry.partition("=")
        n_s, sep2, k_s = shape.partition(",")
        vals = cfg.split(",")
        if not (sep and sep2 and len(vals) == 4):
            raise ValueError(f"malformed GNF4_GEMV_BW_PLAN entry {entry!r}")
        out[(int(n_s), int(k_s))] = tuple(int(v) for v in vals)
    return out


def contract_ok(err: dict) -> bool:
    """K33's tolerance contract on one plan's errors: both decodes within max(1.05 x scalar, 2^-9)."""
    return all(err[d] <= max(TOL_X * err["scalar"], TOL_ABS) for d in ("bw_tree", "bw_prmt32"))


def select(medians: dict, selectable: list) -> tuple | None:
    """The registered choice: the lowest median among the selectable plans, ties to the earlier plan in PLANS."""
    cand = [p for p in PLANS if p in selectable and p in medians]
    if not cand:
        return None
    return min(cand, key=lambda p: (medians[p], PLANS.index(p)))


def main(family: str, out_path: str) -> int:
    import torch

    import int4_b32
    import nf4_grouped as ng
    spec = FAMILIES[family]
    dev = "cuda"
    t0 = time.time()
    rec = {"family": family, "spec": {k: list(v) if isinstance(v, tuple) else v for k, v in spec.items()},
           "device": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability()),
           "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "torch": torch.__version__,
           "plans": [list(p) for p in PLANS], "warm": WARM, "select_rounds": SELECT_ROUNDS,
           "numerics": {}, "plan_refused": {}, "plan_ms": {}, "incumbent_ms": {}, "floor_ms": {}, "selected": {},
           "tally": {}, "reported_never_ruled": ["plan_ms", "incumbent_ms", "floor_ms"]}
    g = torch.Generator(device=dev).manual_seed(139)

    def env(kv):
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        int4_b32.pdl_refresh()

    # the copy floor: this card's streaming bandwidth (read + write of a 512 MiB copy), reported
    src = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        dst.copy_(src)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    rec["copy_gbps"] = 2 * src.numel() / (statistics.median(ts) * 1e6)
    del src, dst

    k = spec["top_k"]
    for proj in PROJ:
        N, K = spec[proj]
        L = spec["layers"]
        stores = [(torch.randint(0, 256, (k, N, K // 2), dtype=torch.uint8, generator=g, device=dev),
                   torch.rand(k, N, K // 64, generator=g, device=dev) + 0.5) for _ in range(L)]
        ids = [torch.randperm(k, generator=g, device=dev).to(torch.int32) for _ in range(L)]
        rows = 1 if GATHER[proj] else k
        x = torch.randn(rows, K, generator=g, device=dev).to(torch.bfloat16)
        kw = {"gather_div": k} if GATHER[proj] else {}
        rec["floor_ms"][proj] = L * k * N * (K // 2 + (K // 64) * 4) / (rec["copy_gbps"] * 1e6)

        def call(B, A, e, plan=None):
            return ng.gemm_4bit_grouped(x, B, A, [1] * k, e, bw_config=plan, **kw)

        # (1) the contract at every plan, on layer 0, before any timing
        B0, A0 = stores[0]
        e0 = ids[0]
        xr = x.repeat_interleave(k, 0) if GATHER[proj] else x
        ref = torch.stack([ng.dequant_ref(B0[e], A0[e], N, K).float() @ xr[r].float()
                           for r, e in enumerate(e0.tolist())])
        scale = ref.abs().max().clamp_min(1e-6)
        env({**INCUMBENT_ENV, "GNF4_GEMV_BW": "0", "GNF4_GEMV_DOTPAD": "0"})
        err_scalar = ((call(B0, A0, e0).float() - ref).abs() / scale).max().item()
        selectable = []
        for plan in PLANS:
            key = ",".join(map(str, plan))
            try:
                outs = {}
                for d in ("tree", "prmt32"):
                    env({**B1_ENV, "GNF4_GEMV_BW_DECODE": d})
                    outs[d] = call(B0, A0, e0, plan)
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001 -- a plan that cannot run is not selectable
                rec["plan_refused"].setdefault(proj, []).append([list(plan), f"{type(exc).__name__}: {str(exc)[:160]}"])
                continue
            err = {"scalar": err_scalar, **{f"bw_{d}": ((o.float() - ref).abs() / scale).max().item()
                                            for d, o in outs.items()}}
            ok = bool(torch.equal(outs["tree"], outs["prmt32"])) and contract_ok(err)
            rec["numerics"].setdefault(proj, {})[key] = {"bitwise_tree_prmt32": bool(torch.equal(outs["tree"],
                                                                                                outs["prmt32"])),
                                                         "err": err, "contract": ok}
            if ok:
                selectable.append(plan)
            else:
                rec["plan_refused"].setdefault(proj, []).append([list(plan), "missed the numerics contract"])

        # (2) capture each selectable plan's 32-layer chain, and the incumbent's, under their own environments
        def capture(fn, envd):
            env(envd)
            fn()
            torch.cuda.synchronize()
            ng.reset_dispatch_counts()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn()
            torch.cuda.synchronize()
            return gr, dict(ng.dispatch_counts())

        graphs = {}
        for plan in selectable:
            gr, tally = capture(lambda p=plan: [call(B, A, e, p) for (B, A), e in zip(stores, ids)], B1_ENV)
            graphs[plan] = gr
            rec["tally"][f"{proj}/{','.join(map(str, plan))}"] = tally
        inc, tally = capture(lambda: [call(B, A, e) for (B, A), e in zip(stores, ids)], INCUMBENT_ENV)
        rec["tally"][f"{proj}/incumbent"] = tally
        order = list(graphs) + ["incumbent"]
        allg = {**graphs, "incumbent": inc}
        times = {p: [] for p in order}
        for rnd in range(WARM + SELECT_ROUNDS):
            for p in (order if rnd % 2 == 0 else order[::-1]):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                allg[p].replay()
                e.record()
                e.synchronize()
                if rnd >= WARM:
                    times[p].append(s.elapsed_time(e))
        med = {p: statistics.median(v) for p, v in times.items() if p != "incumbent"}
        rec["plan_ms"][proj] = {",".join(map(str, p)): v for p, v in med.items()}
        rec["incumbent_ms"][proj] = statistics.median(times["incumbent"])
        choice = select(med, selectable)
        if choice is not None:
            rec["selected"][proj] = list(choice)
        print(f"P141_SELECT {proj} {N}x{K} selectable {len(selectable)}/{len(PLANS)} selected {choice} "
              f"(reported: {med.get(choice, float('nan')):.4f} ms vs incumbent {rec['incumbent_ms'][proj]:.4f} ms, "
              f"floor {rec['floor_ms'][proj]:.4f} ms)", flush=True)
        graphs = allg = inc = stores = None                              # release this projection's stacks
        torch.cuda.empty_cache()

    env(INCUMBENT_ENV)
    missing = [p for p in PROJ if p not in rec["selected"]]
    rec["status"] = "ok" if not missing else "no_plan:" + ",".join(missing)
    if not missing:
        rec["env"] = plan_env({p: tuple(rec["selected"][p]) for p in PROJ}, family)
    rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(out_path, "w"), indent=1)
    print(f"P141_SELECTED status={rec['status']} env={rec.get('env')}", flush=True)
    return 0


def self_test() -> int:
    sel = {"gate_up": (16, 1024, 4, 1), "down": (16, 256, 4, 2)}
    s = plan_env(sel, "mixtral")
    med = {PLANS[0]: 1.0, PLANS[1]: 0.9, PLANS[2]: 0.9, PLANS[3]: 0.8}
    ok = [s == "28672,4096=16,1024,4,1;4096,14336=16,256,4,2",
          parse_plan_env(s) == {(28672, 4096): (16, 1024, 4, 1), (4096, 14336): (16, 256, 4, 2)},
          plan_env(sel, "tiny") == "512,512=16,1024,4,1;512,256=16,256,4,2",
          select(med, list(med)) == PLANS[3],                                   # the lowest median
          select(med, [PLANS[0], PLANS[1], PLANS[2]]) == PLANS[1],              # a refused plan is never chosen; tie -> earlier
          select(med, []) is None, select({}, PLANS) is None,
          contract_ok({"scalar": 0.0028, "bw_tree": 0.0028, "bw_prmt32": 0.0029}),
          contract_ok({"scalar": 0.0, "bw_tree": 2.0 ** -10, "bw_prmt32": 2.0 ** -10}),       # the absolute floor
          not contract_ok({"scalar": 0.0028, "bw_tree": 0.0028, "bw_prmt32": 0.0031}),
          len(PLANS) == 12 and len(set(PLANS)) == 12, FAMILIES["mixtral"]["layers"] == 32,
          GATHER == {"gate_up": True, "down": False}]
    for bad in ("28672,4096=16,1024,4", "28672=16,1024,4,1", "28672,4096:16,1024,4,1", ""):
        try:
            parse_plan_env(bad)
            ok.append(False)
        except ValueError:
            ok.append(True)
    print(f"p141_select self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES))
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if not (a.family and a.out):
        sys.exit("REFUSED: --family and --out, or --self-test")
    sys.exit(main(a.family, a.out))
