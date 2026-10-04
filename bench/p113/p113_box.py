#!/usr/bin/env python3
"""p113_box.py -- lane P113 (bench/p113/PREREG-p113.md), ONE arm in its own process: does grouped-nf4-gemm's
programmatic dependent launch (``GNF4_PDL``), on every switched launch or capped to small launches
(``GNF4_PDL_MAX_ROWS=8``), decode SC1's int4 serving configuration's tokens exactly, and faster, under decode-only timing?

The engine is P112's subject: the default server -- ``PagedServeConfig.from_env()`` + ``build_engine(cfg)``, graphs on --
with SC1's int4_sched levers in the environment (the runner's INT4_ENV). The arms differ ONLY in the switch, which
gnf4 reads at its first launch:
  OFF  ``GNF4_PDL=0`` (the shipped default);
  ALL  ``GNF4_PDL=1``: every switched decode-row launch is a programmatic dependent of the kernel before it (P112's P1);
  CAP  ``GNF4_PDL=1 GNF4_PDL_MAX_ROWS=8``: only launches of at most 8 activation rows are (B=1 decode carries 1 or 8,
       B=16 decode 16 or more).

**Decode-only timing** (P112 Amendment 2; SC1's A10). P109's slope differences whole-pass walls, which carry every
row's prefill; on a shared host that jitter voided P112's second reading. Here each pass records every row's
``Request.ttft`` (from arrival; the rows are added together), and the decode-only time is the wall minus the pass's
largest ttft. The slope (p37's, P109's ``slope``) is taken over those times; the whole-pass slope is reported beside it.

The workloads, prompts and token records are P109's, imported from ``p109_box.py`` at its registered bytes: W16 (16
distinct 512-token prompts at once) and W1 (row 0 alone); SHORT and LONG new tokens; one warm pass, then REPS timed.

A Triton launch hook is armed for ``build_engine`` only and disarmed before anything is timed: P112 Amendment 1's
accounting of each switched kernel's launches, those with PDL, compile launches (no handle) and compiled variants.

    python p113_box.py --prompts prompts.json --out arm_OFF1.json --tag OFF1      (arm from P113_ARM; engine from env)
    python p113_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p113_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, slope, digest, _mem, _smi)

ARMS = {"OFF": {"GNF4_PDL": "0", "GNF4_PDL_MAX_ROWS": None},
        "ALL": {"GNF4_PDL": "1", "GNF4_PDL_MAX_ROWS": None},
        "CAP": {"GNF4_PDL": "1", "GNF4_PDL_MAX_ROWS": "8"}}
CAP_ROWS = 8
# grouped-nf4-gemm's switched decode-row kernels (#448)
SWITCHED = ("_quant_x_rows", "_quant_x_rows_gathered", "_gemv_int4_b32", "_reduce_partials", "_rmsnorm_rows",
            "_rmsnorm_resid_rows", "_scaled_resid_add_rows", "_rope_norm_heads", "_rope_heads", "_router_epilogue",
            "_swiglu_rows", "_combine_rows")
INT4_KEYS = ("int4_expert_layers", "int4_attn_projections", "fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n",
             "fuse_router_epilogue_n", "moe_layers")


def pdl_accounting(launches, int4_b32) -> dict:
    """``{kernel: [launches, launches with launch_pdl, compile launches, compiled variants, variants with launch_pdl]}``
    over the switched kernels (P112 Amendment 1). A compiling launch reaches Triton 3.4's launch hook with no handle
    (``launch_metadata`` reads ``kernel.function`` before ``kernel.run`` initialises it), so it is counted as a compile
    launch and checked against the compiled variants, each of which records whether it was built with ``launch_pdl``."""
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


def run_pass_ttft(parts, torch, rows, n_tokens):
    """P109's ``run_pass`` (every row added at once, stepped to idle, exactly n_tokens per row and the prompt length
    asserted) that also returns every row's ``Request.ttft``: ``(wall, steps, tokens, ttfts)``."""
    sched = parts.scheduler
    before = len(sched.done)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rids = [sched.add_request(list(r), max_new_tokens=n_tokens) for r in rows]
    steps = sched.run_until_idle()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    new = {r.rid: r for r in sched.done[before:]}
    if sorted(new) != sorted(rids) or sched.active or sched.queue:
        raise AssertionError(f"scheduler did not drain: done {sorted(new)} vs added {sorted(rids)}")
    for rid, row in zip(rids, rows):
        q = new[rid]
        if len(q.out) != n_tokens or q.prompt_len != len(row):
            raise AssertionError(f"request {rid}: {len(q.out)} tokens from {q.prompt_len} prompt tokens, "
                                 f"expected {n_tokens} from {len(row)}")
    return wall, steps, [[int(t) for t in new[r].out] for r in rids], [new[r].ttft for r in rids]


def decode_only(walls, ttfts):
    """Each pass's wall minus its largest ttft (SC1's A10), or None when a pass lacks a row's ttft."""
    out = []
    for w, ts in zip(walls, ttfts):
        if not ts or any(t is None or t < 0 for t in ts) or max(ts) >= w:
            return None
        out.append(round(w - max(ts), 5))
    return out


def arm_main(a) -> int:
    arm = os.environ.get("P113_ARM", "")
    if arm not in ARMS:
        raise SystemExit(f"REFUSED: P113_ARM={arm!r}, expected one of {sorted(ARMS)}")
    for k, want in ARMS[arm].items():
        if os.environ.get(k) != want:
            raise SystemExit(f"REFUSED: arm {arm} with {k}={os.environ.get(k)!r} (registered {want!r})")
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
           "pdl_active": bool(int4_b32.pdl_active("cuda")), "pdl_cap": int(int4_b32._PDL_CAP[0]),
           "pdl_launches_build": pdl, "int4": {k_: info.get(k_) for k_ in INT4_KEYS},
           "kv_step_select": bool(getattr(parts.runner.kv, "_step_select", False)),
           "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
           "grouping": info.get("grouping"),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "mem_after_load": p109_box._mem(torch), "workloads": {}, "status": "ok"}
    for wname, b in p109_box.WORKLOADS.items():
        wrows = rows[:b]
        walls, ttfts, digests, last = {}, {}, {}, {}
        for n in (a.short, a.long):
            run_pass_ttft(parts, torch, wrows, n)                          # warm, untimed
            ws, ts, ds = [], [], []
            for _ in range(a.reps):
                wall, _steps, toks, tt = run_pass_ttft(parts, torch, wrows, n)
                ws.append(round(wall, 5))
                ts.append([round(x, 5) if x is not None else None for x in tt])
                ds.append(p109_box.digest(toks))
                last[str(n)] = toks
            walls[str(n)], ttfts[str(n)], digests[str(n)] = ws, ts, ds
        dec = {n: decode_only(walls[n], ttfts[n]) for n in walls}
        whole = p109_box.slope(walls[str(a.short)], walls[str(a.long)], b, a.short, a.long)
        if dec[str(a.short)] is None or dec[str(a.long)] is None:
            s = {"decode_tok_s": None, "decode_tok_s_median": None, "status": "void: a pass lacks a row's ttft"}
        else:
            s = p109_box.slope(dec[str(a.short)], dec[str(a.long)], b, a.short, a.long)
        rec["workloads"][wname] = {"batch": b, "walls": walls, "ttfts": ttfts, "decode_only_walls": dec,
                                   "rep_digests": digests, "tokens": last, **s,
                                   "whole_pass": {k: v for k, v in whole.items()}}
        print(f"P113_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} median={s['decode_tok_s_median']} "
              f"whole_pass={whole['decode_tok_s']} decode_only_walls={dec}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P113_ARM " + json.dumps({"arm": arm, "tag": a.tag, "pdl_active": rec["pdl_active"], "pdl_cap": rec["pdl_cap"],
                                    "load_s": rec["load_s"], "pdl_launches": sum(v[0] for v in pdl.values()),
                                    "pdl_with": sum(v[1] for v in pdl.values()), "int4": rec["int4"],
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
    acct = pdl_accounting([("_gemv_int4_b32", None), ("_gemv_int4_b32", 11), ("_gemv_int4_b32", 12),
                           ("_rmsnorm_rows", None), ("_rmsnorm_rows", 21), ("other", 99)], _Mod)
    ok = [sorted(ARMS) == ["ALL", "CAP", "OFF"], ARMS["CAP"]["GNF4_PDL_MAX_ROWS"] == str(CAP_ROWS),
          p109_box.WORKLOADS == {"W16": 16, "W1": 1}, p109_box.self_test() == 0,
          acct == {"_gemv_int4_b32": [3, 1, 1, 2, 1], "_rmsnorm_rows": [2, 1, 1, 1, 1]}, len(SWITCHED) == 12,
          decode_only([6.0, 7.4], [[0.5, 5.0], [0.4, 6.1]]) == [1.0, 1.3],
          decode_only([6.0], [[0.5, None]]) is None and decode_only([1.0], [[1.2]]) is None]
    print(f"p113_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
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
    return arm_main(a)


if __name__ == "__main__":
    sys.exit(main())
