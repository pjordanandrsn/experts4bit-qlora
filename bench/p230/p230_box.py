#!/usr/bin/env python3
"""p230_box.py -- lane P230 (bench/p230/PREREG-p230.md), ONE arm in its own process: does grouped-nf4-gemm's programmatic
dependent launch, as it ships (``GNF4_PDL`` on, capped to launches of at most 8 rows), decode the DEFAULT NF4 server's
tokens exactly, and faster, at one request, under decode-only timing? P113's box (bench/p113/p113_box.py) by named
edits.

The engine is the default server -- ``PagedServeConfig.from_env()`` + ``build_engine(cfg)``, graphs on, every serving
lever unset -- on Qwen3-30B-A3B's NF4 store. P113 read SC1's int4 configuration; the NF4 default then reached only two
switched kernels. Since K33 (grouped-nf4-gemm's single-row NF4 decode GEMV) carries the switch, the default reaches four
a layer at one request: K33's gate_up and down launches, ``swiglu_rows`` and ``combine_rows``. The arms differ ONLY in
the switch, which gnf4 reads at its first launch:
  OFF  ``GNF4_PDL=0``;
  DEF  ``GNF4_PDL`` and ``GNF4_PDL_MAX_ROWS`` unset: the shipped default (on, capped at 8 rows).

**Decode-only timing** (P112 Amendment 2, P113): each pass records every row's ``Request.ttft``, and the decode-only time
is the wall minus the pass's largest ttft; p37's slope (P109's ``slope``) is taken over those times. The whole-pass
slope is reported beside it.

The workloads, prompts and token records are P109's, imported from ``p109_box.py`` at its registered bytes: W1 (row 0
alone: where the cap lets PDL engage) and W16 (16 distinct 512-token prompts at once: every switched launch is above
the cap there, so W16 is the control); SHORT and LONG new tokens; one warm pass, then REPS timed.

A Triton launch hook is armed for ``build_engine`` only (it captures the decode graphs) and disarmed before anything is
timed: P112 Amendment 1's accounting of each switched kernel's launches, those with PDL, compile launches (no handle)
and compiled variants, over int4_b32's switched kernels AND K33 (``nf4_grouped._gemv_nf4_bw``). Each launch is also
tagged with the decode bucket being captured (``PagedModelRunner._capture_bucket`` is wrapped for the build only), so
the receipt READS that bucket 1 carries PDL launches and bucket 16 none, rather than arguing it from the code.

    python p230_box.py --prompts prompts.json --out arm_OFF1.json --tag OFF1      (arm from P230_ARM; engine from env)
    python p230_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p230_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, slope, digest, _mem, _smi)

ARMS = {"OFF": {"GNF4_PDL": "0", "GNF4_PDL_MAX_ROWS": None},
        "DEF": {"GNF4_PDL": None, "GNF4_PDL_MAX_ROWS": None}}
CAP_ROWS = 8
# grouped-nf4-gemm's switched decode-row kernels: int4_b32's (test_pdl.py's KERNELS, plus the gathered quant) and K33
SWITCHED_INT4 = ("_quant_x_rows", "_quant_x_rows_gathered", "_gemv_int4_b32", "_reduce_partials", "_rmsnorm_rows",
                 "_rmsnorm_resid_rows", "_scaled_resid_add_rows", "_rope_norm_heads", "_rope_heads", "_rope_norm_qk",
                 "_router_epilogue", "_swiglu_rows", "_combine_rows")
SWITCHED_NF4 = ("_gemv_nf4_bw",)
SWITCHED = SWITCHED_INT4 + SWITCHED_NF4
# every serving lever the default server must NOT carry (the subject is the default)
LEVERS_UNSET = ("E4B_SERVE_EXP_INT4", "E4B_SERVE_ATTN_INT4", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI",
                "E4B_PAGED_FUSE_QKV", "E4B_PAGED_GRAPHS", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_BUCKETS", "GNF4_GEMV_BW",
                "E4B_FUSE_SWIGLU", "E4B_FUSE_COMBINE", "E4B_K25_PLAN", "E4B_K25_LUT")


def pdl_accounting(launches, modules) -> dict:
    """``{kernel: [launches, launches with launch_pdl, compile launches, compiled variants, variants with launch_pdl]}``
    over the switched kernels (P112 Amendment 1). ``modules`` maps each kernel name to the module that defines it. A
    compiling launch reaches Triton's launch hook with no handle, so it is counted as a compile launch and checked
    against the compiled variants, each of which records whether it was built with ``launch_pdl``."""
    fn_pdl, variants = {}, {}
    for name in SWITCHED:
        k = getattr(modules.get(name), name, None)
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


def pdl_by_bucket(launches, modules) -> dict:
    """``{bucket: [switched launches, handled launches with launch_pdl]}`` over launches tagged ``(name, fn, bucket)``;
    launches outside a bucket's capture are filed under ``"none"``. A compile launch (no handle) never counts as PDL."""
    fn_pdl = {}
    for name in SWITCHED:
        k = getattr(modules.get(name), name, None)
        for _dev, (cache, *_rest) in (getattr(k, "device_caches", None) or {}).items():
            for ck in cache.values():
                fn_pdl[ck.function] = bool(getattr(ck.metadata, "launch_pdl", False))
    out = {}
    for name, fn, bucket in launches:
        if name in SWITCHED:
            row = out.setdefault(str(bucket) if bucket is not None else "none", [0, 0])
            row[0] += 1
            row[1] += int(fn is not None and fn_pdl.get(fn, False))
    return out


def run_pass_ttft(parts, torch, rows, n_tokens):
    """P109's ``run_pass`` that also returns every row's ``Request.ttft``: ``(wall, steps, tokens, ttfts)``."""
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
    """Each pass's wall minus its largest ttft, or None when a pass lacks a row's ttft."""
    out = []
    for w, ts in zip(walls, ttfts):
        if not ts or any(t is None or t < 0 for t in ts) or max(ts) >= w:
            return None
        out.append(round(w - max(ts), 5))
    return out


def arm_main(a) -> int:
    arm = os.environ.get("P230_ARM", "")
    if arm not in ARMS:
        raise SystemExit(f"REFUSED: P230_ARM={arm!r}, expected one of {sorted(ARMS)}")
    for k, want in ARMS[arm].items():
        if os.environ.get(k) != want:
            raise SystemExit(f"REFUSED: arm {arm} with {k}={os.environ.get(k)!r} (registered {want!r})")
    for k in LEVERS_UNSET:
        if os.environ.get(k) is not None:
            raise SystemExit(f"REFUSED: {k}={os.environ.get(k)!r}; the subject is the default server, every lever unset")
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"] or len(rows) != p109_box.ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    import torch
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if not cfg.graphs or cfg.placement != "all-vram" or not ({1, 16} <= set(cfg.buckets)):
        raise SystemExit(f"REFUSED: not the default graph server with buckets 1 and 16: graphs={cfg.graphs} "
                         f"placement={cfg.placement} buckets={cfg.buckets}")
    import triton
    import int4_b32
    import nf4_grouped
    from experts4bit_qlora.engines import paged_runner
    modules = {**{n: int4_b32 for n in SWITCHED_INT4}, **{n: nf4_grouped for n in SWITCHED_NF4}}
    seen, cur = [], [None]
    orig_capture = paged_runner.PagedModelRunner._capture_bucket

    def _tagged_capture(self, b, warmup):                         # the bucket under capture, for the hook's tag
        cur[0] = int(b)
        try:
            return orig_capture(self, b, warmup)
        finally:
            cur[0] = None
    paged_runner.PagedModelRunner._capture_bucket = _tagged_capture
    triton.knobs.runtime.launch_enter_hook = lambda md: seen.append((md.data["name"], md.data["function"], cur[0]))
    t0 = time.perf_counter()
    try:
        parts = build_engine(cfg)
    finally:
        triton.knobs.runtime.launch_enter_hook = None              # disarmed before anything is timed
        paged_runner.PagedModelRunner._capture_bucket = orig_capture
    load_s = time.perf_counter() - t0
    pdl = pdl_accounting([(n, f) for n, f, _b in seen], modules)
    by_bucket = pdl_by_bucket(seen, modules)
    info = parts.info
    rec = {"arm": arm, "tag": a.tag, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "model": cfg.model, "revision": cfg.revision, "torch": torch.__version__, "load_s": round(load_s, 2),
           "pdl_active": bool(int4_b32.pdl_active("cuda")), "pdl_cap": int(int4_b32._PDL_CAP[0]),
           "pdl_launches_build": pdl, "pdl_by_bucket": by_bucket, "config": {"max_seqs": cfg.max_seqs, "buckets": list(cfg.buckets),
                                                 "placement": cfg.placement, "graphs": bool(cfg.graphs)},
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
        print(f"P230_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} median={s['decode_tok_s_median']} "
              f"whole_pass={whole['decode_tok_s']} decode_only_walls={dec}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P230_ARM " + json.dumps({"arm": arm, "tag": a.tag, "pdl_active": rec["pdl_active"], "pdl_cap": rec["pdl_cap"],
                                    "load_s": rec["load_s"], "pdl_launches": sum(v[0] for v in pdl.values()),
                                    "pdl_with": sum(v[1] for v in pdl.values()),
                                    "k33_launches": (pdl.get("_gemv_nf4_bw") or [0])[0], "by_bucket": by_bucket,
                                    "config": rec["config"],
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

    class _I4:
        _swiglu_rows = _K([_CK(11, False), _CK(12, True)])

    class _NF4:
        _gemv_nf4_bw = _K([_CK(21, True)])
    mods = {**{n: _I4 for n in SWITCHED_INT4}, **{n: _NF4 for n in SWITCHED_NF4}}
    acct = pdl_accounting([("_swiglu_rows", None), ("_swiglu_rows", 11), ("_swiglu_rows", 12),
                           ("_gemv_nf4_bw", None), ("_gemv_nf4_bw", 21), ("other", 99)], mods)
    ok = [sorted(ARMS) == ["DEF", "OFF"], ARMS["DEF"] == {"GNF4_PDL": None, "GNF4_PDL_MAX_ROWS": None},
          ARMS["OFF"]["GNF4_PDL"] == "0", p109_box.WORKLOADS == {"W16": 16, "W1": 1}, p109_box.self_test() == 0,
          acct == {"_swiglu_rows": [3, 1, 1, 2, 1], "_gemv_nf4_bw": [2, 1, 1, 1, 1]},
          len(SWITCHED) == 14 and "_gemv_nf4_bw" in SWITCHED and "GNF4_PDL" not in LEVERS_UNSET,
          pdl_by_bucket([("_swiglu_rows", None, 1), ("_swiglu_rows", 12, 1), ("_swiglu_rows", 11, 16),
                         ("_gemv_nf4_bw", 21, 1), ("_gemv_nf4_bw", 21, None), ("other", 99, 1)], mods)
          == {"1": [3, 2], "16": [1, 0], "none": [1, 1]},
          decode_only([6.0, 7.4], [[0.5, 5.0], [0.4, 6.1]]) == [1.0, 1.3],
          decode_only([6.0], [[0.5, None]]) is None and decode_only([1.0], [[1.2]]) is None]
    print(f"p230_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
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
    p.add_argument("--check-args", action="store_true", help="parse and validate the arguments, then exit (tests)")
    a = p.parse_args(argv)
    if a.check_args:
        if a.prompts_only:
            ok = bool(a.model and a.revision and a.out)
        else:
            ok = bool(a.prompts and a.out and a.tag) and a.short > 0 and a.long > a.short and a.reps > 0
        print("ARGS OK" if ok else "ARGS BAD")
        return 0 if ok else 2
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return p109_box.prompts_main(a)
    return arm_main(a)


if __name__ == "__main__":
    sys.exit(main())
