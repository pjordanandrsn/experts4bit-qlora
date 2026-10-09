#!/usr/bin/env python3
"""Lane P126's measurement (bench/p126/PREREG-p126.md; e4b#846): does splitting the one-launch cumsum tile table above
256 routed rows over P programs (``E4B_INT4_TILE_PROGRAMS=P``: grouped-nf4-gemm #524, e4b #1433) make SC2e's 64-row
decode step faster, with identical tokens?

Lane P122 (#1393) read the one-program chunked table at 2.51 ms of Qwen3-30B-A3B's 64-row eager step (52 us a launch over
48 layers), still 1.75x the chained builder's named kernels. #524 gives each of P programs a slice of the experts; every
program takes all the counts from one histogram of the ids. The tables are the same integers at every P, so outputs are
bit-identical and the read gates token identity, as P120 and P122 did.

On ONE RTX 5090, SC2e's int4 stack built eager with one slot (P117's to P124's build), with ``E4B_INT4_WIDE_TILES`` at
its default (``auto``: the one-launch cumsum table at 64 rows), this box runs:
- **profile** at P = 1, 4 and 8: P119's ``decode_bracket`` at its registered bytes (64 rows, buckets up to 64, 3 warm +
  8 profiled padded eager steps under ``torch.profiler``): the premise (``_tile_table_r1``'s share of the step at P = 1)
  and the mechanism (``_tile_table_cumsum_mp``'s time at P = 4 and 8);
- **served**, interleaved as P124's Amendment 1 does it: for each candidate P in (4, 8), blocks ``a`` (P = 1 first in
  every pair) and ``b`` (the candidate first), each with ONE runner per setting, both alive, its bucket graphs CAPTURED
  under its setting, decoding in strict alternation step by step: 5 warm, ``--steps`` (256) timed and ``BUSY`` (32)
  traced steps of each. Every decode step's tokens, each block's peak memory and its GPU log;
- **mutant**: P = 8 with P120's ``_Mutant`` (every live tile's expert id shifted by one, the builder's signature kept):
  its tokens must differ from P = 1's, so the token gate can fail.

``p126_reduce.py`` applies the registered rule. ``measure()`` takes the model and the windows, so
``tests/test_p126_box.py`` runs it on CPU with the runner, the pool, the profiler and the builder stood in.

    python p126_box.py --out OUT.json [--rows 64 --prompt 512 --steps 256]   (engine from E4B_PAGED_* env)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p119_box  # noqa: E402  (staged at P119's registered bytes: decode_bracket, _pool, _Grouped, _sync)
import p117_box  # noqa: E402  (staged at P117's registered bytes: windows(), the bucket lists)
import p120_box  # noqa: E402  (staged at P120's registered bytes: _Mutant, served_arm)
import p124_box  # noqa: E402  (staged at P124's registered bytes as amended: _Clock, _busy, _mem, _reset_peak)

B64 = p117_box.B64
SETTINGS = ("1", "4", "8")       # E4B_INT4_TILE_PROGRAMS, fixed before any data
CANDIDATES = ("4", "8")
BLOCKS = ("a", "b")              # a: P = 1 first in every pair; b: the candidate first
WARM, BUSY = 5, 32
MUTANT_STEPS = 16
TABLE_ONE, TABLE_MP = "_tile_table_r1", "_tile_table_cumsum_mp"


def _set_programs(p):
    if p is None:
        os.environ.pop("E4B_INT4_TILE_PROGRAMS", None)
    else:
        os.environ["E4B_INT4_TILE_PROGRAMS"] = str(p)


def _calls(table: dict, name: str) -> float:
    return round(sum(c for k, c, _ms in table["kernels"] if name in k), 3)


def _ms(table: dict, name: str) -> float:
    return round(sum(ms for k, _c, ms in table["kernels"] if name in k), 4)


def profile_arm(model, ws, P, device, *, programs: str, rows: int, bulk_kv: bool):
    """P119's decode bracket at 64 rows with ``E4B_INT4_TILE_PROGRAMS`` set (it is read on every call)."""
    _set_programs(programs)
    try:
        t = p119_box.decode_bracket(model, ws, P, device, buckets=B64, rows=rows, warm=3, steps=8, bulk_kv=bulk_kv)
    finally:
        _set_programs(None)
    return {"programs": programs, "device_ms": t["device_ms"], "profiled": t["profiled"],
            "graph_status": t["graph_status"], "layers": t["layers"], "kernels": t["kernels"], "classes": t["classes"],
            "table_one_calls": _calls(t, TABLE_ONE), "table_one_ms": _ms(t, TABLE_ONE),
            "table_mp_calls": _calls(t, TABLE_MP), "table_mp_ms": _ms(t, TABLE_MP),
            "radix_sort_calls": _calls(t, "radixSortKVInPlace")}


def interleaved_pair(model, ws, P, device, *, candidate: str, block: str, rows: int, steps: int, bulk_kv: bool,
                     trace_dir: str):
    """One block: a runner at P = 1 and one at ``candidate``, both alive, each captured with its setting, then decoded
    in strict alternation (block ``a``: P = 1 first in every pair; ``b``: the candidate first). P124 Amendment 1's
    method; each timed step is ``run_decode``'s synchronised wall."""
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    from experts4bit_qlora.engines.step_trace import StepTrace
    p124_box._reset_peak(device)
    first = ("1", candidate) if block == "a" else (candidate, "1")
    rec = {"candidate": candidate, "block": block, "order": list(first), "rows": rows, "steps": steps, "warm": WARM,
           "busy_steps": BUSY, "bulk_kv": bool(bulk_kv), "arms": {}}
    run, kvs, traces = {}, [], {}
    try:
        with p119_box._Grouped(), torch.no_grad():
            for st in first:                                     # each runner captured under its own setting
                kv, _layers = p119_box._pool(model, rows, P + WARM + steps + BUSY + 16, max(B64), device)
                kvs.append(kv)
                runner = PagedModelRunner(model, kv, device=device, bulk_kv=bulk_kv)
                run[st] = runner
                _set_programs(st)
                try:
                    gs = {str(k): v for k, v in runner.enable_decode_graphs(B64, capture=True, verbose=False).items()}
                finally:
                    _set_programs(None)
                rec["arms"][st] = {"programs": st, "graph_status": gs, "step_ms": [], "tokens": []}
            for st in first:
                for rid in range(rows):
                    run[st].bind(rid, rid, ws[rid][:P])
                    run[st].run_prefill([(rid, 0, P)])
            order = list(range(rows))
            with p124_box._Clock() as clock:
                for i in range(WARM + steps):
                    for st in first:                             # strict alternation, one step of each per pair
                        p119_box._sync(device)
                        t0 = time.perf_counter()
                        got = run[st].run_decode(order)
                        p119_box._sync(device)
                        arm = rec["arms"][st]
                        if i >= WARM:
                            arm["step_ms"].append(round((time.perf_counter() - t0) * 1e3, 4))
                        arm["tokens"].append([int(got[r]) for r in order])
                for st in first:
                    path = os.path.join(trace_dir, f"trace_p{candidate}_{block}_P{st}.jsonl")
                    if os.path.exists(path):
                        os.remove(path)
                    traces[st] = (path, StepTrace(path, cuda=str(device).startswith("cuda"), flush_every=10**6))
                    run[st].tracer = traces[st][1]
                try:
                    for _ in range(BUSY):
                        for st in first:
                            p119_box._sync(device)
                            tr = traces[st][1]
                            tr.begin()
                            got = run[st].run_decode(order)
                            tr.end()
                            rec["arms"][st]["tokens"].append([int(got[r]) for r in order])
                finally:
                    for st in first:
                        run[st].tracer = None
                        traces[st][1].close()
            rec["clock"] = clock.rows
        for st in first:
            gs = getattr(run[st], "graph_stats", None) or {}
            rec["arms"][st]["graph_stats"] = {str(k): dict(v) for k, v in gs.items()}
        rec["memory"] = p124_box._mem(device)
    finally:
        for runner in run.values():
            dis = getattr(runner, "disable_decode_graphs", None)
            if dis is not None:
                dis()
        del run, kvs
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    for st in first:
        arm = rec["arms"][st]
        arm["median_ms"] = round(statistics.median(arm["step_ms"]), 4)
        arm["busy"] = p124_box._busy(traces[st][0])
        arm["tokens_sha256"] = hashlib.sha256(json.dumps(arm["tokens"]).encode()).hexdigest()
    return rec


def mutant_arm(model, ws, P, device, *, rows: int, bulk_kv: bool):
    """P = 8 with P120's ``_Mutant`` (live tiles' expert ids shifted by one; the builder's signature kept, so the
    ``programs=`` capability still reads true): P120's served arm, its tokens only."""
    _set_programs("8")
    try:
        rec = p120_box.served_arm(model, ws, P, device, on=True, rows=rows, steps=MUTANT_STEPS, bulk_kv=bulk_kv,
                                  mutant=True)
    finally:
        _set_programs(None)
        os.environ.pop("E4B_INT4_WIDE_TILES", None)            # P120's arm sets it; the lane runs it at its default
    return {"programs": "8", "mutant": True, "rows": rows, "steps": MUTANT_STEPS, "warm": rec["warm"],
            "graph_status": rec["graph_status"], "graph_stats": rec["graph_stats"], "tokens": rec["tokens"]}


def measure(model, ws, *, prompt, device, rows=64, steps=256, bulk_kv=True, trace_dir="."):
    if not 32 < rows <= 64:
        raise SystemExit(f"REFUSED: rows {rows}: the step is one bucket-64 piece (33 to 64 rows)")
    t0 = time.time()
    rec = {"prompt": prompt, "rows": rows, "steps": steps, "profile": {}, "served": {}}
    for p in SETTINGS:
        rec["profile"][p] = profile_arm(model, ws, prompt, device, programs=p, rows=rows, bulk_kv=bulk_kv)
        print(f"P126_PROFILE P={p} table one {rec['profile'][p]['table_one_ms']} ms mp "
              f"{rec['profile'][p]['table_mp_ms']} ms at {time.time() - t0:.0f} s", flush=True)
    for c in CANDIDATES:
        rec["served"][c] = {}
        for b in BLOCKS:
            blk = interleaved_pair(model, ws, prompt, device, candidate=c, block=b, rows=rows, steps=steps,
                                   bulk_kv=bulk_kv, trace_dir=trace_dir)
            rec["served"][c][b] = blk
            meds = " ".join(f"P={st} {blk['arms'][st]['median_ms']} ms" for st in blk["order"])
            print(f"P126_BLOCK P={c} {b} {meds} at {time.time() - t0:.0f} s", flush=True)
    rec["mutant"] = mutant_arm(model, ws, prompt, device, rows=rows, bulk_kv=bulk_kv)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--stride", type=int, default=3072)
    a = ap.parse_args()
    import transformers

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram" or cfg.max_seqs != 1:
        raise SystemExit(f"REFUSED: the engine is built eager at all-vram with one slot (graphs={cfg.graphs}, "
                         f"placement={cfg.placement}, max_seqs={cfg.max_seqs}); the arms build their own pools")
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    ws = p117_box.windows(parts.tokenizer, a.rows, a.prompt, 0, a.stride)
    rec = measure(model, ws, prompt=a.prompt, device=cfg.device, rows=a.rows, steps=a.steps, bulk_kv=cfg.bulk_kv,
                  trace_dir=os.path.dirname(os.path.abspath(a.out)))
    rec = {"model": cfg.model, "revision": cfg.revision, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "transformers": transformers.__version__, "torch": torch.__version__,
           "load_s": round(load_s, 1), "stride": a.stride,
           "tile_programs_env": os.environ.get("E4B_INT4_TILE_PROGRAMS"),
           "wide_tiles_env": os.environ.get("E4B_INT4_WIDE_TILES"),
           "census_build": {k: parts.info.get(k) for k in ("moe_layers", "experts", "top_k", "model_type",
                                                           "int4_expert_layers", "int4_attn_projections",
                                                           "fuse_qkv_n", "fuse_t1_glue_n", "fuse_router_epilogue_n")},
           **rec, "gpu": torch.cuda.get_device_name(0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    open(a.out, "w").write(json.dumps(rec, indent=1))
    s = rec["served"]
    line = " | ".join(f"P={c} {b} " + " ".join(f"{st}:{blk['arms'][st]['median_ms']}" for st in blk["order"])
                      for c, blocks in s.items() for b, blk in blocks.items())
    print(f"P126_BOX: {line} | load {load_s:.0f}s | {rec['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
