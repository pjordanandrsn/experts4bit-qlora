#!/usr/bin/env python3
"""Lane P120's measurement (bench/p120/PREREG-p120.md; e4b#846): does building the tile table above 256 routed rows in
ONE launch (``E4B_INT4_WIDE_TILES=1``: grouped-nf4-gemm's ``build_group_tiles_fused(..., rank="cumsum")``, #515/#1357)
make SC2e's 64-row decode step faster, with identical tokens?

Lane P119 (#1353) read the 64-row step at 15.64 ms of device time, 2.87 ms of it the chained tile-table builder that
calls above 256 routed rows take (argsort, scatter, cumsum, searchsorted, index_select), against 0.93 ms for the
one-launch table at 256 rows. The cumsum rank gives the chained builder's integers, so outputs are bit-identical.

On ONE RTX 5090, SC2e's int4 stack built eager with one slot (P117's and P119's build), this box runs:
- **profile** ``off`` and ``on``: P119's ``decode_bracket`` (64 windows, buckets up to 64, 3 warm + 8 profiled padded
  eager steps under ``torch.profiler``) with ``E4B_INT4_WIDE_TILES`` 0 and 1: the premise (the chained builder's
  kernels are there with 0) and the engagement (one ``_tile_table_r1`` launch per layer and no radix sort with 1);
- **served** ``OFF_a``, ``ON_a``, ``ON_b``, ``OFF_b`` (ABBA): each a fresh pool and runner whose bucket-64 graph is
  CAPTURED under its setting, 64 windows prefilled, 5 warm steps, then ``--steps`` (256) timed decode steps (each
  ``run_decode``'s wall: the graph replay and the step's host work, synchronised); every decode step's tokens recorded;
- **mutant**: ``ON`` with the table's expert ids shifted by one on every live tile (``(grp + 1) % E``): its tokens must
  differ from ``OFF``'s, so the token gate can fail.

``p120_reduce.py`` applies the registered rule. ``measure()`` takes the model and the windows, so
``tests/test_p120_box.py`` runs it on CPU with the runner, the pool and the profiler stood in.

    python p120_box.py --out OUT.json [--rows 64 --prompt 512 --steps 256]   (engine from E4B_PAGED_* env)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p119_box  # noqa: E402  (staged at P119's registered bytes: decode_bracket, _pool, _Grouped, _sync, kernel_table)
import p117_box  # noqa: E402  (staged at P117's registered bytes: windows())

B64 = p117_box.B64
SERVED = ("OFF_a", "ON_a", "ON_b", "OFF_b")
#: kernels only the chained builder launches at 64 rows (frozen before the box; P119's d64 table against d32's): its
#: argsort, its scatter_add, its two cumsums, its searchsorted and its aranges. Its generic fills, gathers and
#: elementwise ops share names with the route's own and are left out, so this is a lower bound on its device time.
BUILDER = ("radixSortKVInPlace", "_cuda_scatter_gather_internal_kernel<true", "DeviceScan", "searchsorted",
           "arange_cuda_out")
TABLE = "_tile_table_r1"     # the one-launch table (pairwise at <= 256 rows, cumsum above with E4B_INT4_WIDE_TILES=1)
WARM = 5


def _set_wide(on: bool):
    os.environ["E4B_INT4_WIDE_TILES"] = "1" if on else "0"


def _builder_ms(table: dict) -> float:
    return round(sum(ms for k, _c, ms in table["kernels"] if any(b in k for b in BUILDER)), 4)


def _calls(table: dict, name: str) -> float:
    return round(sum(c for k, c, _ms in table["kernels"] if name in k), 3)


def _ms(table: dict, name: str) -> float:
    return round(sum(ms for k, _c, ms in table["kernels"] if name in k), 4)


def profile_arm(model, ws, P, device, *, on: bool, rows: int, bulk_kv: bool):
    """P119's decode bracket at 64 rows with the setting given: kernel table per step, builder time, the counts the
    rule reads."""
    _set_wide(on)
    t = p119_box.decode_bracket(model, ws, P, device, buckets=B64, rows=rows, warm=3, steps=8, bulk_kv=bulk_kv)
    return {"wide": on, "device_ms": t["device_ms"], "builder_ms": _builder_ms(t),
            "table_calls": _calls(t, TABLE), "table_ms": _ms(t, TABLE), "radix_sort_calls": _calls(t, "radixSortKVInPlace"),
            "profiled": t["profiled"], "graph_status": t["graph_status"], "layers": t["layers"],
            "kernels": t["kernels"], "classes": t["classes"]}


class _Mutant:
    """``int4_b32.build_group_tiles_fused`` with every live tile's expert id shifted by one: rows land on the wrong
    expert's weights, so the outputs (and tokens) must move. Keeps the builder's signature so e4b's ``rank=`` check
    still passes; restored on exit."""

    def __enter__(self):
        import int4_b32
        self.mod, self.real = int4_b32, int4_b32.build_group_tiles_fused
        real = self.real

        def shifted(ids, n_exp, block_m, *a, **kw):
            out = real(ids, n_exp, block_m, *a, **kw)
            row0, rows, grp = out[0], out[1], out[2]
            grp = torch.where(rows > 0, (grp + 1) % n_exp, grp)
            return (row0, rows, grp) + tuple(out[3:])
        shifted.__signature__ = inspect.signature(real)
        int4_b32.build_group_tiles_fused = shifted
        return self

    def __exit__(self, *exc):
        self.mod.build_group_tiles_fused = self.real
        return False


def served_arm(model, ws, P, device, *, on: bool, rows: int, steps: int, bulk_kv: bool, mutant: bool = False):
    """A fresh pool and runner whose bucket graphs are captured under the setting given; ``rows`` windows prefilled;
    WARM untimed decode steps, then ``steps`` timed ones (each ``run_decode`` synchronised: its wall is the replay and
    the step's host work). Every decode step's tokens are recorded, warm ones first."""
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    _set_wide(on)
    kv, _layers = p119_box._pool(model, rows, P + WARM + steps + 16, max(B64), device)
    runner = PagedModelRunner(model, kv, device=device, bulk_kv=bulk_kv)
    rec = {"wide": on, "mutant": mutant, "rows": rows, "steps": steps, "warm": WARM, "bulk_kv": bool(bulk_kv)}
    ctx = _Mutant() if mutant else None
    ms, toks = [], []
    try:
        if ctx is not None:
            ctx.__enter__()
        with p119_box._Grouped(), torch.no_grad():
            rec["graph_status"] = {str(k): v for k, v in
                                   runner.enable_decode_graphs(B64, capture=True, verbose=False).items()}
            for rid in range(rows):
                runner.bind(rid, rid, ws[rid][:P])
                runner.run_prefill([(rid, 0, P)])
            order = list(range(rows))
            for i in range(WARM + steps):
                p119_box._sync(device)
                t0 = time.perf_counter()
                got = runner.run_decode(order)
                p119_box._sync(device)
                if i >= WARM:
                    ms.append((time.perf_counter() - t0) * 1e3)
                toks.append([int(got[r]) for r in order])
        gs = getattr(runner, "graph_stats", None) or {}
        rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()}
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
        dis = getattr(runner, "disable_decode_graphs", None)
        if dis is not None:
            dis()
        del runner, kv
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec["step_ms"] = [round(x, 4) for x in ms]
    rec["median_ms"] = round(statistics.median(ms), 4)
    rec["tokens"] = toks
    rec["tokens_sha256"] = hashlib.sha256(json.dumps(toks).encode()).hexdigest()
    return rec


def measure(model, ws, *, prompt, device, rows=64, steps=256, bulk_kv=True):
    t0 = time.time()
    rec = {"prompt": prompt, "rows": rows, "steps": steps, "profile": {}, "served": {}}
    rec["profile"]["off"] = profile_arm(model, ws, prompt, device, on=False, rows=rows, bulk_kv=bulk_kv)
    rec["profile"]["on"] = profile_arm(model, ws, prompt, device, on=True, rows=rows, bulk_kv=bulk_kv)
    for label in SERVED:
        rec["served"][label] = served_arm(model, ws, prompt, device, on=label.startswith("ON"), rows=rows, steps=steps,
                                          bulk_kv=bulk_kv)
        print(f"P120_ARM {label} median {rec['served'][label]['median_ms']} ms at {time.time() - t0:.0f} s", flush=True)
    rec["served"]["MUTANT"] = served_arm(model, ws, prompt, device, on=True, rows=rows, steps=steps, bulk_kv=bulk_kv,
                                         mutant=True)
    os.environ.pop("E4B_INT4_WIDE_TILES", None)
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
    rec = measure(model, ws, prompt=a.prompt, device=cfg.device, rows=a.rows, steps=a.steps, bulk_kv=cfg.bulk_kv)
    rec = {"model": cfg.model, "revision": cfg.revision, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "transformers": transformers.__version__, "torch": torch.__version__,
           "load_s": round(load_s, 1), "stride": a.stride,
           "census_build": {k: parts.info.get(k) for k in ("moe_layers", "experts", "top_k", "model_type",
                                                           "int4_expert_layers", "int4_attn_projections",
                                                           "fuse_qkv_n", "fuse_t1_glue_n", "fuse_router_epilogue_n")},
           **rec, "gpu": torch.cuda.get_device_name(0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    open(a.out, "w").write(json.dumps(rec, indent=1))
    s = rec["served"]
    line = " | ".join(f"{k} {v['median_ms']} ms" for k, v in s.items())
    p = rec["profile"]
    print(f"P120_BOX: {line} | builder off {p['off']['builder_ms']} on {p['on']['builder_ms']} ms | "
          f"load {load_s:.0f}s | {rec['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
