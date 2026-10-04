#!/usr/bin/env python3
"""p112_box.py -- lane P112 (bench/p112/PREREG-p112.md), ONE arm in its own process: does ``GNF4_PDL=1``
(grouped-nf4-gemm's programmatic dependent launch) decode SC1's int4 serving configuration's tokens exactly, and faster?

The engine is the default server -- ``PagedServeConfig.from_env()`` + ``build_engine(cfg)``, graphs on -- with SC1's
int4_sched levers in the environment (the runner's INT4_ENV). The arms differ ONLY in ``GNF4_PDL``, which gnf4 reads at
its first launch:
  P0  off (the shipped default);
  P1  on: gnf4's decode-row kernels launch as programmatic dependents of the kernel before them.

A Triton launch hook is armed for ``build_engine`` only (the warm passes and the decode-graph capture) and disarmed
before anything is timed. It records how many launches of each switched gnf4 kernel the build made and how many of them
carried ``launch_pdl``.

The workloads, timing and token records are P109's, imported from ``p109_box.py`` at its registered bytes:
- W16: 16 distinct 512-token prompts added at once; W1: row 0 alone.
- SHORT and LONG new tokens; one warm pass, then REPS timed passes; p37's slope.
- Every timed pass's token digest; the last pass's tokens per row.

    python p112_box.py --prompts prompts.json --out arm_P0a.json --tag P0a       (arm from P112_ARM; engine from env)
    python p112_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p112_box.py --census --out census_default.json     (the default server, no levers: its build's switched launches)
    python p112_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)

ARMS = ("P0", "P1")
# grouped-nf4-gemm's switched decode-row kernels (#448)
SWITCHED = ("_quant_x_rows", "_quant_x_rows_gathered", "_gemv_int4_b32", "_reduce_partials", "_rmsnorm_rows",
            "_rmsnorm_resid_rows", "_scaled_resid_add_rows", "_rope_norm_heads", "_rope_heads", "_router_epilogue",
            "_swiglu_rows", "_combine_rows")
INT4_KEYS = ("int4_expert_layers", "int4_attn_projections", "fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n",
             "fuse_router_epilogue_n", "moe_layers")


def pdl_accounting(launches, int4_b32) -> dict:
    """``{kernel: [launches, launches with launch_pdl, compile launches, compiled variants, variants with launch_pdl]}``
    over the switched kernels, from the hook's (name, function) records and the kernels' compiled variants.

    Amendment 1: the launch that COMPILES a variant reaches Triton 3.4's launch hook before the variant's handle exists
    (``launch_metadata`` reads ``kernel.function`` before ``kernel.run`` initialises it), so the hook sees
    ``function=None``. p112-5090-1 counted those as launches without PDL. They are now counted as compile launches and
    checked against the compiled variants instead, each of which records whether it was built with ``launch_pdl``."""
    fn_pdl, variants = {}, {}
    for name in SWITCHED:
        k = getattr(int4_b32, name, None)
        for _dev, (cache, *_rest) in (getattr(k, "device_caches", None) or {}).items():
            for ck in cache.values():
                pdl = bool(getattr(ck.metadata, "launch_pdl", False))
                fn_pdl[ck.function] = pdl
                v = variants.setdefault(name, [0, 0])
                v[0] += 1
                v[1] += int(pdl)
    out = {}
    for name, fn in launches:
        if name in SWITCHED:
            row = out.setdefault(name, [0, 0, 0] + variants.get(name, [0, 0]))
            row[0] += 1
            if fn is None:
                row[2] += 1
            else:
                row[1] += int(fn_pdl.get(fn, False))
    return out


def census_main(a) -> int:
    """Build the server the environment describes (the runner passes the default server, no levers) with the launch hook
    armed, and record which switched kernels its build launched. Reported, not ruled on."""
    import triton
    import int4_b32
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    seen = []
    triton.knobs.runtime.launch_enter_hook = lambda md: seen.append((md.data["name"], md.data["function"]))
    try:
        parts = build_engine(cfg)
    finally:
        triton.knobs.runtime.launch_enter_hook = None
    acct = pdl_accounting(seen, int4_b32)
    rec = {"e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"), "pdl_active": bool(int4_b32.pdl_active("cuda")),
           "switched": acct, "switched_kernels": sorted(acct), "triton_launches": len(seen),
           "triton_kernels": sorted({n for n, _ in seen}), "int4": {k_: parts.info.get(k_) for k_ in INT4_KEYS},
           "graph_status": ({str(k): v for k, v in parts.info["graph_status"].items()} if parts.info.get("graph_status") else None),
           "levers_env": parts.info.get("levers_env")}
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P112_CENSUS " + json.dumps({"switched_kernels": rec["switched_kernels"], "switched_launches": sum(v[0] for v in acct.values()),
                                       "triton_launches": len(seen), "int4": rec["int4"]}), flush=True)
    return 0


def arm_main(a) -> int:
    arm = os.environ.get("P112_ARM", "")
    if arm not in ARMS:
        raise SystemExit(f"REFUSED: P112_ARM={arm!r}, expected one of {ARMS}")
    want = arm == "P1"
    if (os.environ.get("GNF4_PDL", "0").strip() == "1") != want:
        raise SystemExit(f"REFUSED: arm {arm} with GNF4_PDL={os.environ.get('GNF4_PDL')!r}")
    for k in ("E4B_SERVE_EXP_INT4", "E4B_SERVE_ATTN_INT4", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI",
              "E4B_PAGED_FUSE_QKV"):
        if os.environ.get(k) != "1":
            raise SystemExit(f"REFUSED: {k}={os.environ.get(k)!r}; the subject is SC1's int4 configuration")
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"] or len(rows) != p109_box.ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    import torch
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if not cfg.graphs or (cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, "all-vram", (1, 2, 4, 8, 16)):
        raise SystemExit(f"REFUSED: not the default graph server: graphs={cfg.graphs} max_seqs={cfg.max_seqs} "
                         f"placement={cfg.placement} buckets={cfg.buckets}")
    import triton
    import int4_b32
    seen = []
    triton.knobs.runtime.launch_enter_hook = lambda md: seen.append((md.data["name"], md.data["function"]))
    t0 = time.perf_counter()
    try:
        parts = build_engine(cfg)
    finally:
        triton.knobs.runtime.launch_enter_hook = None              # disarmed before anything is timed
    load_s = time.perf_counter() - t0
    pdl = pdl_accounting(seen, int4_b32)
    info = parts.info
    rec = {"arm": arm, "tag": a.tag, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "model": cfg.model, "revision": cfg.revision, "torch": torch.__version__, "load_s": round(load_s, 2),
           "pdl_active": bool(int4_b32.pdl_active("cuda")), "pdl_launches_build": pdl,
           "int4": {k_: info.get(k_) for k_ in INT4_KEYS},
           "kv_step_select": bool(getattr(parts.runner.kv, "_step_select", False)),
           "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
           "grouping": info.get("grouping"),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "mem_after_load": p109_box._mem(torch), "workloads": {}, "status": "ok"}
    for wname, b in p109_box.WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, last = {}, {}, {}
        for n in (a.short, a.long):
            p109_box.run_pass(parts, torch, wrows, n)                      # warm, untimed
            ws, ds = [], []
            for _ in range(a.reps):
                wall, _steps, toks = p109_box.run_pass(parts, torch, wrows, n)
                ws.append(round(wall, 5))
                ds.append(p109_box.digest(toks))
                last[str(n)] = toks
            walls[str(n)], digests[str(n)] = ws, ds
        s = p109_box.slope(walls[str(a.short)], walls[str(a.long)], b, a.short, a.long)
        rec["workloads"][wname] = {"batch": b, "walls": walls, "rep_digests": digests, "tokens": last, **s}
        print(f"P112_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} median={s['decode_tok_s_median']} "
              f"walls={walls}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P112_ARM " + json.dumps({"arm": arm, "tag": a.tag, "pdl_active": rec["pdl_active"], "load_s": rec["load_s"],
                                    "pdl_launches": sum(v[0] for v in pdl.values()), "pdl_with": sum(v[1] for v in pdl.values()),
                                    "int4": rec["int4"],
                                    "graph_status": rec["graph_status"], "W16": rec["workloads"]["W16"]["decode_tok_s"],
                                    "W1": rec["workloads"]["W1"]["decode_tok_s"]}), flush=True)
    return 0


def self_test() -> int:
    class _CK:
        def __init__(self, fn, pdl):
            self.function, self.metadata = fn, type("M", (), {"launch_pdl": pdl})()

    class _K:
        def __init__(self, cks):
            self.device_caches = {0: ({i: ck for i, ck in enumerate(cks)}, None, None, None)}

    class _Mod:
        _gemv_int4_b32 = _K([_CK(11, False), _CK(12, True)])
        _rmsnorm_rows = _K([_CK(21, True)])
    # a compiling launch reaches the hook with function=None (Amendment 1)
    acct = pdl_accounting([("_gemv_int4_b32", None), ("_gemv_int4_b32", 11), ("_gemv_int4_b32", 12),
                           ("_rmsnorm_rows", None), ("_rmsnorm_rows", 21), ("other", 99)], _Mod)
    ok = [ARMS == ("P0", "P1"), p109_box.WORKLOADS == {"W16": 16, "W1": 1}, p109_box.self_test() == 0,
          acct == {"_gemv_int4_b32": [3, 1, 1, 2, 1], "_rmsnorm_rows": [2, 1, 1, 1, 1]}, len(SWITCHED) == 12]
    print(f"p112_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--census", action="store_true")
    p.add_argument("--model")
    p.add_argument("--revision", default="")
    p.add_argument("--prompts")
    p.add_argument("--out")
    p.add_argument("--tag", default="")
    p.add_argument("--short", type=int, default=32)
    p.add_argument("--long", type=int, default=160)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return p109_box.prompts_main(a)
    if a.census:
        return census_main(a)
    return arm_main(a)


if __name__ == "__main__":
    sys.exit(main())
