#!/usr/bin/env python3
"""Lane P119's census (bench/p119/PREREG-p119.md; e4b#846): where the 64-slot server's steps spend their device time.
Descriptive only: no gate, licence, default or claim.

SC2e (``bench/h2h-2026-10-02/sc2e``) read 64 slots with buckets up to 64 at 12 req/s on Qwen3-30B-A3B int4, one RTX
5090: a 64-row decode step runs in 18.4 ms (16.0 on the device), and a 512-token prefill forward takes ~40 ms. The
capacity model on SC2e's costs says prefill and the decode step together set the next ceiling. Nobody has attributed
either by kernel on this stack. This box does, on the eager twins of the served graphs: each decode bucket's graph is
captured from its padded eager step and the first-chunk prefill graph from the eager forward -- the same code path at
the same shapes (P109's bit-identical outputs establish numerical equivalence, not identical kernels). ``torch.profiler``
(CUDA activities) attributes the eager steps' device time per kernel, as P102 and P107 did for prefill; the captured
kernels stay unprofiled.

Brackets, each on a fresh ``Fp8PagedKV`` and ``PagedModelRunner`` over SC2e's int4 model (device grouping on, as the
graph server runs it):
- **decode** ``d16``, ``d32``, ``d64``: ``rows`` windows prefilled with ``--prompt`` tokens, ``--warm`` decode steps,
  then ``--steps`` profiled decode steps, each one padded eager step of its bucket (``capture=False``);
- **decode** ``d64x4``: the same 64 rows on buckets 1-16, so every step runs as four 16-row pieces (SC2e's s64c);
- **prefill** ``p512_off`` / ``p512_on``: ``--reps`` profiled 512-token first-chunk prefills with
  ``E4B_PAGED_LAST_LOGITS`` off and on (#1337; ON is refused, and recorded as refused, if the model's forward has no
  explicit keyword);
- **head**: the model's LM head alone on bf16 rows of ``HEAD_ROWS`` sizes, so its share is read without guessing which
  GEMM in a forward it is.

Per bracket: every device kernel's calls and device ms per step (or per prefill), the classes (``CLASSES``, frozen
before the box), device-to-device copies per step and per layer, the runner's bucket statistics and graph status, and
the profiled wall per step (reported, never a speed reading: speeds are SC2e's).

``measure()`` takes the model and the windows, so ``tests/test_p119_box.py`` runs it on CPU with the runner stood in.

    python p119_box.py --out OUT.json [--rows 64 --prompt 512 --warm 3 --steps 8 --reps 3]   (engine from E4B_PAGED_* env)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p117_box  # noqa: E402  (staged at P117's registered bytes: windows(), the bucket lists)

B16, B32, B64 = p117_box.B16, p117_box.B32, p117_box.B64
DECODE = (("d16", B16, 16), ("d32", B32, 32), ("d64", B64, 64), ("d64x4", B16, 64))
PREFILL = (("p512_off", False), ("p512_on", True))
HEAD_ROWS = (1, 16, 64, 512)
# kernel name (lower case) -> class: the first rule whose any substring matches wins. Frozen before the box.
CLASSES = (
    ("memcpy_dtod", ("memcpy dtod", "memcpy device -> device", "memcpy d2d")),
    ("memcpy_other", ("memcpy", "memset")),
    ("attn", ("_fp8_paged_decode", "_fp8_combine", "_fp8_append", "flash", "fmha", "sdpa", "attention")),
    ("expert_int4", ("_gemm_int4_b32_grouped", "grouped_smallm", "gemm_4bit_grouped", "_mtile")),
    ("moe_route", ("_tile_table", "build_group_tiles", "_gather_rows", "_combine_rows", "_router_epilogue", "topk",
                   "sort", "scan", "cumsum", "histogram", "bincount")),
    ("dense_int4", ("_gemv_int4_b32", "_gemm_int4_b32_smallm", "int4_b32")),
    ("dense_gemm", ("nvjet", "gemm", "gemv", "cutlass", "cublas", "xmma", "sm90_", "sm100_", "sm120_")),
    ("norm_glue", ("rmsnorm", "rms_norm", "_rope", "_scaled_resid", "glue", "_fused_")),
    ("sample", ("argmax", "max_kernel")),
    ("elementwise", ("elementwise", "vectorized", "index", "copy", "fill", "cat", "reduce", "softmax", "where")),
)


def kclass(name: str) -> str:
    n = str(name).lower()
    for cls, keys in CLASSES:
        if any(k in n for k in keys):
            return cls
    return "other"


def kernel_table(avgs, per: int) -> dict:
    """``prof.key_averages()`` -> device kernels (self device time > 0), per step: ``{"kernels": [[name, calls,
    ms], ...] sorted by ms, "classes": {class: [calls, ms]}, "device_ms": total, "dtod": calls}``."""
    rows = []
    for a in avgs:
        t = getattr(a, "self_device_time_total", None)
        if t is None:
            t = getattr(a, "self_cuda_time_total", 0)
        if t and t > 0:
            rows.append([str(a.key)[:160], int(a.count), float(t) / 1e3])
    per = max(int(per), 1)
    rows = [[k, round(c / per, 3), round(ms / per, 4)] for k, c, ms in rows]
    rows.sort(key=lambda r: -r[2])
    classes: dict = {}
    for k, c, ms in rows:
        cl = classes.setdefault(kclass(k), [0.0, 0.0])
        cl[0] = round(cl[0] + c, 3)
        cl[1] = round(cl[1] + ms, 4)
    return {"kernels": rows, "classes": dict(sorted(classes.items(), key=lambda kv: -kv[1][1])),
            "device_ms": round(sum(r[2] for r in rows), 4), "dtod": classes.get("memcpy_dtod", [0.0, 0.0])[0]}


def _profile(fn, n, device):
    """Run ``fn`` ``n`` times under ``torch.profiler`` (CUDA activities on a CUDA device); returns (key averages,
    wall ms per call). The wall is reported, never read as a speed."""
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CUDA] if str(device).startswith("cuda") else [ProfilerActivity.CPU]
    _sync(device)
    with profile(activities=acts) as prof:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        _sync(device)
        wall = (time.perf_counter() - t0) * 1e3 / max(n, 1)
    return prof.key_averages(), round(wall, 3)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _pool(model, rows, tokens, scratch, device):
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import kv_layers
    from experts4bit_qlora.serve_paged import _kv_geometry
    cfg = getattr(model.config, "text_config", None) or model.config
    hkv, hd = _kv_geometry(model.config)
    layers = kv_layers(model, int(cfg.num_hidden_layers))
    return Fp8PagedKV(layers, hkv, hd, batch=rows, max_tokens_per_seq=tokens, device=device, scratch_slots=scratch), layers


class _Grouped:
    """The graph server's expert grouping for the bracket (serve_paged sets it above one sequence), restored after."""

    def __enter__(self):
        from experts4bit_qlora.engines import hot_residency as hr
        self.hr, self.saved = hr, (hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0])
        hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = True, False
        return self

    def __exit__(self, *exc):
        self.hr.DEVICE_GROUPING[0], self.hr.FORCE_SINGLETON_GROUPS[0] = self.saved
        return False


def decode_bracket(model, ws, P, device, *, buckets, rows, warm, steps, bulk_kv=True):
    """``rows`` windows prefilled, ``warm`` decode steps, then ``steps`` profiled decode steps through the bucketed
    path with ``capture=False`` (each piece a padded eager step of its bucket)."""
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    kv, layers = _pool(model, rows, P + warm + steps + 16, max(buckets), device)
    runner = PagedModelRunner(model, kv, device=device, bulk_kv=bulk_kv)
    rec = {"buckets": list(buckets), "rows": rows, "warm": warm, "steps": steps, "layers": len(layers),
           "bulk_kv": bool(bulk_kv)}
    with _Grouped(), torch.no_grad():
        rec["graph_status"] = {str(k): v for k, v in
                               runner.enable_decode_graphs(buckets, capture=False, verbose=False).items()}
        for rid in range(rows):
            runner.bind(rid, rid, ws[rid][:P])
            runner.run_prefill([(rid, 0, P)])
        order = list(range(rows))
        for _ in range(warm):
            runner.run_decode(order)
        before = {str(k): dict(v) for k, v in (getattr(runner, "graph_stats", None) or {}).items()}
        avgs, wall = _profile(lambda: runner.run_decode(order), steps, device)
        after = {str(k): dict(v) for k, v in (getattr(runner, "graph_stats", None) or {}).items()}
    rec["profiled"] = {b: {k: after[b][k] - before.get(b, {}).get(k, 0) for k in after[b]} for b in after}
    rec["wall_ms_per_step"] = wall
    rec.update(kernel_table(avgs, steps))
    rec["dtod_per_layer"] = round(rec["dtod"] / max(len(layers), 1), 3)
    return rec


def prefill_bracket(model, ws, P, device, *, last_logits, reps, bulk_kv=True):
    """One warm 512-token first-chunk prefill, then ``reps`` profiled ones, each its own window, with
    ``last_logits`` as given. ON is refused by the runner if the forward has no explicit keyword: recorded."""
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    kv, layers = _pool(model, reps + 1, P + 16, 1, device)
    try:
        runner = PagedModelRunner(model, kv, device=device, bulk_kv=bulk_kv, last_logits=last_logits)
    except ValueError as e:
        return {"last_logits": last_logits, "refused": str(e)[:300]}
    rec = {"last_logits": last_logits, "reps": reps, "tokens": P, "layers": len(layers), "bulk_kv": bool(bulk_kv)}
    with _Grouped(), torch.no_grad():
        runner.bind(0, 0, ws[0][:P])
        runner.run_prefill([(0, 0, P)])
        for r in range(1, reps + 1):
            runner.bind(r, r, ws[r][:P])
        nxt = iter(range(1, reps + 1))
        avgs, wall = _profile(lambda: runner.run_prefill([(next(nxt), 0, P)]), reps, device)
    stats = getattr(runner, "last_logits_stats", None)
    rec["last_logits_stats"] = stats() if callable(stats) else None
    rec["wall_ms_per_prefill"] = wall
    rec.update(kernel_table(avgs, reps))
    rec["dtod_per_layer"] = round(rec["dtod"] / max(len(layers), 1), 3)
    return rec


def head_bracket(model, device, *, rows=HEAD_ROWS, reps=5):
    """The LM head alone on bf16 rows: its shapes, dtype and module, and device ms per call at each row count."""
    head = model.get_output_embeddings()
    w = getattr(head, "weight", None)
    cfg = getattr(model.config, "text_config", None) or model.config
    rec = {"module": type(head).__name__, "weight_shape": list(w.shape) if w is not None else None,
           "weight_dtype": str(w.dtype) if w is not None else None, "rows": {}}
    dt = w.dtype if w is not None and w.is_floating_point() else torch.bfloat16
    with torch.no_grad():
        for n in rows:
            h = torch.randn(1, n, int(cfg.hidden_size), dtype=dt, device=device)
            head(h)
            avgs, wall = _profile(lambda: head(h), reps, device)
            t = kernel_table(avgs, reps)
            rec["rows"][str(n)] = {"device_ms": t["device_ms"], "kernels": t["kernels"][:4], "wall_ms": wall}
    return rec


def measure(model, ws, *, prompt, device, warm=3, steps=8, reps=3, decode=DECODE, prefill=PREFILL, head=True,
            bulk_kv=True):
    t0 = time.time()
    rec = {"prompt": prompt, "decode": {}, "prefill": {}}
    for label, buckets, rows in decode:
        rec["decode"][label] = decode_bracket(model, ws, prompt, device, buckets=buckets, rows=rows, warm=warm,
                                              steps=steps, bulk_kv=bulk_kv)
        print(f"P119_BRACKET {label} device {rec['decode'][label]['device_ms']} ms/step at {time.time() - t0:.0f} s",
              flush=True)
    for label, ll in prefill:
        rec["prefill"][label] = prefill_bracket(model, ws, prompt, device, last_logits=ll, reps=reps, bulk_kv=bulk_kv)
        print(f"P119_BRACKET {label} {rec['prefill'][label].get('device_ms', 'refused')} at {time.time() - t0:.0f} s",
              flush=True)
    if head:
        rec["head"] = head_bracket(model, device)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--stride", type=int, default=3072)
    a = ap.parse_args()
    import transformers

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram" or cfg.max_seqs != 1:
        raise SystemExit(f"REFUSED: the engine is built eager at all-vram with one slot (graphs={cfg.graphs}, "
                         f"placement={cfg.placement}, max_seqs={cfg.max_seqs}); the brackets build their own pools")
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    ws = p117_box.windows(parts.tokenizer, max(a.rows, a.reps + 1), a.prompt, 0, a.stride)
    decode = tuple((lb, b, min(r, a.rows)) for lb, b, r in DECODE)
    rec = measure(model, ws, prompt=a.prompt, device=cfg.device, warm=a.warm, steps=a.steps, reps=a.reps, decode=decode,
                  bulk_kv=cfg.bulk_kv)                       # the server's KV bookkeeping (bulk by default since #1200)
    rec = {"model": cfg.model, "revision": cfg.revision, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "transformers": transformers.__version__, "torch": torch.__version__,
           "load_s": round(load_s, 1), "stride": a.stride,
           "census_build": {k: parts.info.get(k) for k in ("moe_layers", "experts", "top_k", "model_type",
                                                           "int4_expert_layers", "int4_attn_projections",
                                                           "int4_store_kinds", "fuse_qkv_n", "fuse_t1_glue_n",
                                                           "fuse_router_epilogue_n", "levers_env")},
           **rec, "gpu": torch.cuda.get_device_name(0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    open(a.out, "w").write(json.dumps(rec, indent=1))
    d = rec["decode"]
    line = " | ".join(f"{k} {v['device_ms']} ms" for k, v in d.items())
    pf = rec["prefill"]
    pl = " | ".join(f"{k} {v.get('device_ms', 'refused')} ms" for k, v in pf.items())
    print(f"P119_BOX: decode {line} | prefill {pl} | load {load_s:.0f}s | mem {rec['max_mem_gb']} GB | {rec['seconds']} s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
