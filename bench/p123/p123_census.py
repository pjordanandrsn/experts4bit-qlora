#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""p123_census.py -- lane P123 (#1313): bracket exactly N steady decode steps of the shipped default ``serve_paged`` for
an Nsight Systems capture (``nsys profile --capture-range=cudaProfilerApi ...``; bench/p123/PREREG-p123.md).

The engine is the speed arm's: every lever unset but ``E4B_PAGED_MAX_SEQS=16``, graphs on, built by ``build_engine``.
B rows of P109's prompts are warmed through ``p109_box.run_pass`` (every bucket the window uses replayed once), then
SC1b's ``census_window`` (``sc1b_e4b_census.py``, staged at SC1b's bytes and imported, never edited) ramps to full
decode, skips K steps and brackets exactly N decode-only full-batch steps between cudaProfilerStart and Stop. The
profiled wall is recorded and is NEVER a speed reading: the speed arms (p123_box.py) are the unprofiled reference.

    env: P123_BATCH (1 or 16), P123_OUT, and the engine env (E4B_PAGED_*)
    p123_census.py --prompts prompts.json [--skip 32 --steps 64]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOKENS = 160          # each row's budget: the window (skip + steps after the ramp) must not outlive the first row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--skip", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    a = ap.parse_args(argv)
    import p109_box
    from p123_box import default_env_ok
    from sc1b_e4b_census import census_window

    ok, why = default_env_ok(os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: not the shipped default ({why})")
    batch = int(os.environ["P123_BATCH"])
    pf = json.load(open(a.prompts))
    rows = pf["rows"][:batch]
    if p109_box.digest(pf["rows"]) != pf["prompts_sha256"] or len(rows) != batch:
        raise SystemExit("REFUSED: prompts.json does not match its own digest, or has too few rows")
    import torch

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if not cfg.graphs or (cfg.max_seqs, cfg.placement) != (16, "all-vram"):
        raise SystemExit(f"REFUSED: graphs={cfg.graphs} max_seqs={cfg.max_seqs} placement={cfg.placement}")
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    info = parts.info
    rec = {"engine": "e4b", "batch": batch, "mode": "census", "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "model": cfg.model, "revision": cfg.revision,
           "load_s": round(time.perf_counter() - t0, 1), "torch": torch.__version__,
           "fusions": {k: info.get(k) for k in ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n",
                                                "fuse_router_epilogue_n")},
           "fusion_sources": info.get("fusion_sources"), "prompts_sha256": pf["prompts_sha256"]}
    p109_box.run_pass(parts, torch, rows, 32)                      # warm (untimed): the window's bucket replayed
    sched = parts.scheduler
    for r in rows:
        sched.add_request(list(r), max_new_tokens=TOKENS)
    rec.update(census_window(sched, batch, a.skip, a.steps, torch.cuda.profiler.start, torch.cuda.profiler.stop,
                             torch.cuda.synchronize, tokens=TOKENS))
    sched.run_until_idle()
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
    rec["max_memory_reserved"] = int(torch.cuda.max_memory_reserved())
    with open(os.environ["P123_OUT"], "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("P123_CENSUS " + json.dumps({k: rec[k] for k in ("batch", "steps_bracketed", "profiled_ms_per_step")}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
