#!/usr/bin/env python3
"""Lane P124's measurement (bench/p124/PREREG-p124.md; e4b#846): does serving the attention projections of a 32- or
64-row decode step on the int4 small-M GEMM (``E4B_ATTN_INT4_WIDE=1``: grouped-nf4-gemm's ``block_m=`` #522, e4b
#1410) make SC2e's decode step faster, without costing teacher-forced quality beyond P110's floor?

Lane P119 (#1353) read SC2e's 64-row step at 15.64 ms of device time, 2.05 ms of it ``Int4Linear``'s cached bf16 copy
on cuBLAS above 16 rows: 1.81 GB a step where the int4 grid it is dequantised from is 510 MB. With the route on, the
same int4 bytes the 2..16-row K16 route reads serve 17..64 rows with a 32- or 64-row tile. The arithmetic changes
(in-register dequant, bf16 MMA over 128-wide K chunks, a 4-way split-K, against cuBLAS on the dequantised bf16 copy),
so the read gates quality, not token identity.

On ONE RTX 5090, SC2e's int4 stack built eager with one slot (P117's, P119's and P120's build), this box runs:
- **profile** ``off`` / ``on`` at 64 and at 32 rows: P119's ``decode_bracket`` (buckets up to 64, 3 warm + 8 profiled
  padded eager steps under ``torch.profiler``), at its registered bytes: the premise (with ``0``, the cuBLAS class
  carries the projections) and the mechanism (with ``1``, one ``_gemm_int4_b32_smallm`` launch per projection);
- **served**, interleaved (Amendment 1), at 64 rows, then at 32: blocks ``a`` and ``b``, each with ONE runner per
  setting, both alive, each with its own pool and its bucket graphs CAPTURED under its setting. Both are prefilled,
  then they decode in strict alternation, step by step in lockstep: 5 warm pairs, ``--steps`` (256) timed pairs (each
  step ``run_decode``'s synchronised wall, P120's method), then ``BUSY`` (32) traced pairs with each runner's
  ``StepTrace`` attached (the replay's device time over the step's wall is the GPU busy fraction). Block ``a`` runs
  OFF first in every pair, block ``b`` ON first. Every decode step's tokens are recorded, the block's peak memory, and
  the GPU's SM and memory clocks, power, temperature and performance state every 5 s (reported, never gated);
- **quality**: teacher-forced passes of 64 windows through P117's ``paged_pass`` (at its registered bytes): ``R``
  (route off, one 64-row piece a step: today's served arithmetic), ``rep``, the floor ``half`` (route off, two 32-row
  pieces) and ``chunk`` (route off, prefill in 256-token pieces), the subjects ``ON64`` (route on, one 64-row piece)
  and ``ON32`` (route on, two 32-row pieces), ``mutant_scale`` (``ON64`` with P108's halved decode scale),
  ``mutant_wide`` (``ON64`` with the route reading each 32-block's scale from the next block: the route is the one
  scored) and ``G64on`` (``ON64`` with the graphs CAPTURED: its emitted tokens must equal ``ON64``'s, FUNCTION).

The setting is every ``Int4Linear``'s ``_wide`` flag, which is what ``E4B_ATTN_INT4_WIDE`` sets at enable
(``engines.int4_attn.resolve_wide``); the engine is built with ``1`` so the enable path's capability check runs on the
card, then each arm sets the flag before its captures or passes. :class:`Route` also counts, per call that reaches
Python (eager steps, prefills and graph captures; replays never do), which route each projection took at which row
count: ``gemv``, ``k16`` (2..16 rows), ``wide`` (17..64) or ``bf16`` (the cached copy).

``p124_reduce.py`` applies the registered rule. ``measure()`` takes the model and the windows, so
``tests/test_p124_box.py`` runs it on CPU with the runner, the pool, the profiler and the passes stood in.

    python p124_box.py --out OUT.json [--rows 64 --prompt 512 --cont 128 --steps 256]   (engine from E4B_PAGED_* env)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p119_box  # noqa: E402  (staged at P119's registered bytes: decode_bracket, _pool, _Grouped, _sync)
import p117_box  # noqa: E402  (staged at P117's registered bytes: windows(), paged_pass(), the bucket lists)
import p108_box  # noqa: E402  (staged at P108's registered bytes: _score, _kl, _release; imported by P117's box)

B32, B64 = p117_box.B32, p117_box.B64
BLOCKS = ("a", "b")        # Amendment 1: block a runs OFF first in every pair, block b ON first
SETTINGS = ("OFF", "ON")
DEPTHS = ("64", "32")      # the step's bucket: ``rows`` rows (33-64, one bucket-64 piece), then 32 (one bucket-32 piece)
WARM = 5
BUSY = 32                  # steps after the timed ones, with the runner's StepTrace attached
FLOORS = ("half", "chunk")
SUBJECTS = ("ON64", "ON32")
MUTANTS = ("mutant_scale", "mutant_wide")
QUALITY = ("R", "rep") + FLOORS + SUBJECTS + MUTANTS + ("G64on",)
#: per quality arm: the bucket list (the runner splits a step into pieces of its largest bucket), the route, P108's
#: attention mutant, the wide-route mutant, and whether the graphs are captured
Q_KW = {"R": {"buckets": B64, "wide": False}, "rep": {"buckets": B64, "wide": False},
        "half": {"buckets": B32, "wide": False}, "chunk": {"buckets": B64, "wide": False},
        "ON64": {"buckets": B64, "wide": True}, "ON32": {"buckets": B32, "wide": True},
        "mutant_scale": {"buckets": B64, "wide": True, "mutant": "scale"},
        "mutant_wide": {"buckets": B64, "wide": True, "mutant_wide": True},
        "G64on": {"buckets": B64, "wide": True, "capture": True}}


def int4_linears(model) -> list:
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    return [m for m in model.modules() if isinstance(m, Int4Linear)]


class Route:
    """Sets every ``Int4Linear``'s ``_wide`` flag for an arm and counts each projection call by route and row count,
    ``"<route>:<rows>" -> calls``. ``mutant_wide``: the wide route reads its scales rolled by one 32-block along K (every
    weight on the wrong scale), so a quality pass that scores the route must fail the bar. Restored on exit."""

    def __init__(self, model, wide: bool, mutant_wide: bool = False):
        self.mods, self.wide, self.mutant = int4_linears(model), bool(wide), bool(mutant_wide)
        self.counts: dict = {}

    def _add(self, route: str, rows: int) -> None:
        k = f"{route}:{int(rows)}"
        self.counts[k] = self.counts.get(k, 0) + 1

    def __enter__(self):
        self.saved = []
        for m in self.mods:
            self.saved.append((m, m._wide, m._smallm, m._gemv))
            m._wide = self.wide
            rows_now = [0]
            real_fwd, real_bf16, real_sm, real_gemv = m.forward, m._bf16_weight, m._smallm, m._gemv

            def forward(x, _real=real_fwd, _rows=rows_now):
                _rows[0] = x.numel() // x.shape[-1]
                return _real(x)

            def bf16_weight(_real=real_bf16, _rows=rows_now):
                self._add("bf16", _rows[0])
                return _real()

            def gemv(*a, _real=real_gemv, _rows=rows_now, **kw):
                self._add("gemv", _rows[0])
                return _real(*a, **kw)

            m.forward, m._bf16_weight, m._gemv = forward, bf16_weight, gemv
            if real_sm is not None:
                def smallm(x, packed, scales, *a, _real=real_sm, **kw):
                    r = int(x.shape[0])
                    self._add("k16" if r <= 16 else "wide", r)
                    if self.mutant and r > 16:
                        scales = scales.roll(-1, dims=1)
                    return _real(x, packed, scales, *a, **kw)
                m._smallm = smallm
        return self

    def __exit__(self, *exc):
        for m, wide, sm, gemv in self.saved:
            m._wide, m._smallm, m._gemv = wide, sm, gemv
            for name in ("forward", "_bf16_weight"):
                m.__dict__.pop(name, None)
        return False


def _mem(device) -> dict:
    from experts4bit_qlora.engines.int4_attn import wide_workspace_bytes
    cuda = str(device).startswith("cuda")
    return {"max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
            "max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1) if cuda else None,
            "wide_workspace_mib": round(wide_workspace_bytes() / 2**20, 3)}


def _reset_peak(device) -> None:
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def profile_arm(model, ws, P, device, *, on: bool, rows: int, bulk_kv: bool):
    """P119's decode bracket at ``rows`` rows under the setting: the kernel table per step, its classes, the route
    counts of the profiled steps' Python calls."""
    with Route(model, on) as route:
        t = p119_box.decode_bracket(model, ws, P, device, buckets=B64, rows=rows, warm=3, steps=8, bulk_kv=bulk_kv)
    return {"wide": on, "rows": rows, "device_ms": t["device_ms"], "profiled": t["profiled"],
            "graph_status": t["graph_status"], "layers": t["layers"], "kernels": t["kernels"], "classes": t["classes"],
            "route": dict(sorted(route.counts.items()))}


def _busy(trace_path: str) -> list:
    """Per traced step: [replay device ms (``dec_issue - dec_prep``), the step's host wall ms]."""
    out = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            g = r.get("gpu") or {}
            if "dec_issue" in g and "dec_prep" in g:
                out.append([round(g["dec_issue"] - g["dec_prep"], 4), r.get("step_ms")])
    return out


class _Clock:
    """Samples the GPU's SM and memory clocks, power, temperature and performance state every PERIOD seconds while a
    block runs (Amendment 1; reported, never gated). Without ``nvidia-smi`` (the CPU tests) it records nothing."""

    PERIOD = 5.0
    QUERY = "clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate"

    def __init__(self):
        self.rows, self._stop = [], threading.Event()

    def _run(self):
        t0 = time.time()
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.SubprocessError):
                return
            line = out.stdout.strip().splitlines()[:1]
            if out.returncode == 0 and line:
                self.rows.append([round(time.time() - t0, 1)] + [x.strip() for x in line[0].split(",")])
            self._stop.wait(self.PERIOD)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=15)
        return False


def interleaved_block(model, ws, P, device, *, block: str, rows: int, steps: int, bulk_kv: bool, trace_dir: str,
                      depth: str):
    """Amendment 1: ONE runner per setting, both alive, each with its own pool and its bucket graphs captured under its
    setting; both prefilled; then WARM, ``steps`` timed and BUSY traced decode steps of each in strict alternation, step
    by step in lockstep (block ``a``: OFF then ON in every pair; ``b``: ON then OFF), so the two steps of a pair share
    the context length and the GPU's state. Each timed step is ``run_decode``'s synchronised wall (P120's method).
    Every decode step's tokens are recorded, warm ones first."""
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    from experts4bit_qlora.engines.step_trace import StepTrace
    _reset_peak(device)
    first = ("OFF", "ON") if block == "a" else ("ON", "OFF")
    rec = {"block": block, "order": list(first), "rows": rows, "steps": steps, "warm": WARM, "busy_steps": BUSY,
           "bulk_kv": bool(bulk_kv), "arms": {}}
    run, kvs, traces = {}, [], {}
    try:
        with p119_box._Grouped(), torch.no_grad():
            for st in first:                                     # each runner captured under its own setting
                kv, _layers = p119_box._pool(model, rows, P + WARM + steps + BUSY + 16, max(B64), device)
                kvs.append(kv)
                runner = PagedModelRunner(model, kv, device=device, bulk_kv=bulk_kv)
                run[st] = runner
                with Route(model, st == "ON") as route:
                    gs = {str(k): v for k, v in runner.enable_decode_graphs(B64, capture=True, verbose=False).items()}
                rec["arms"][st] = {"wide": st == "ON", "graph_status": gs,
                                   "capture_route": dict(sorted(route.counts.items())), "step_ms": [], "tokens": []}
            for st in first:
                for rid in range(rows):
                    run[st].bind(rid, rid, ws[rid][:P])
                    run[st].run_prefill([(rid, 0, P)])
            order = list(range(rows))
            with _Clock() as clock:
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
                    path = os.path.join(trace_dir, f"trace_d{depth}_{block}_{st}.jsonl")
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
        rec["memory"] = _mem(device)
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
        arm["busy"] = _busy(traces[st][0])
        arm["tokens_sha256"] = hashlib.sha256(json.dumps(arm["tokens"]).encode()).hexdigest()
    return rec


def quality(model, ws, *, prompt, cont, chunk, floor_chunk, device, stand_in=False, arms=QUALITY):
    """The teacher-forced passes: per window, each arm's mean continuation NLL, argmax agreement and KL against R;
    per pass, P117's engagement record plus the route counts; R's repeatability; G64on's function gate."""
    from experts4bit_qlora.engines import paged_attention
    P, C = prompt, cont
    paged_attention.register(model)
    per = {a: [] for a in arms if a != "G64on"}
    eng = {}
    start = time.time()
    refs = ref_am = None
    rep_identical = function = None
    on64_tk = None

    def add(arm, wi, lp, ref):
        nll, a = p108_box._score(lp, ws[wi][P:P + C])
        rec = {"window": wi, "nll": sum(nll) / len(nll),
               "argmax_agree": sum(int(x == y) for x, y in zip(a, ref_am[wi])) / len(a)}
        if ref is not None:
            kl = p108_box._kl(ref, lp)
            rec["kl"] = sum(kl) / len(kl)
        per[arm].append(rec)

    for arm in arms:
        kw = dict(Q_KW[arm])
        wide, mw = kw.pop("wide"), kw.pop("mutant_wide", False)
        _reset_peak(device)
        with Route(model, wide, mutant_wide=mw) as route:
            lps, tk, e = p117_box.paged_pass(model, ws, P, C, floor_chunk if arm == "chunk" else chunk, device,
                                             stand_in=stand_in, **kw)
        e["route"] = dict(sorted(route.counts.items()))
        e["wide"], e["mutant_wide"] = wide, mw
        e["memory"] = _mem(device)
        eng[arm] = e
        if arm == "R":
            refs = lps
            ref_am = [r.argmax(-1).tolist() for r in refs]
            for i, r in enumerate(refs):
                add("R", i, r, None)
        elif arm == "G64on":
            if on64_tk is not None:        # the replay's emitted tokens against the padded eager step's, every step
                diff = sum(1 for a, b in zip(tk, on64_tk) for x, y in zip(a, b) if x != y)
                function = {"positions": sum(len(a) for a in tk), "differ": diff}
        else:
            if arm == "rep":
                rep_identical = all(torch.equal(a, b) for a, b in zip(lps, refs))
            if arm == "ON64":
                on64_tk = tk
            for i, x in enumerate(lps):
                add(arm, i, x, refs[i])
        del lps
        p108_box._release(device)
        print(f"P124_QUALITY {arm} done at {time.time() - start:.0f} s", flush=True)
    del refs
    p108_box._release(device)
    cfg = getattr(model.config, "text_config", None) or model.config
    return {"windows": len(ws), "prompt": P, "cont": C, "chunk": chunk, "floor_chunk": floor_chunk,
            "layers": int(cfg.num_hidden_layers), "rep_identical": rep_identical, "function": function, "per_window": per, "engagement": eng,
            "seconds": round(time.time() - start, 1)}


def measure(model, ws, *, prompt, cont, device, rows=64, steps=256, chunk=512, floor_chunk=256, bulk_kv=True,
            trace_dir=".", stand_in=False):
    if not 32 < rows <= 64:
        raise SystemExit(f"REFUSED: rows {rows}: the 64-bucket step is one bucket-64 piece (33 to 64 rows)")
    t0 = time.time()
    rec = {"prompt": prompt, "rows": rows, "steps": steps, "int4_linears": len(int4_linears(model)),
           "profile": {}, "served": {}}
    for d in DEPTHS:
        n = rows if d == "64" else 32
        for label, on in (("off", False), ("on", True)):
            rec["profile"][f"{label}{d}"] = profile_arm(model, ws, prompt, device, on=on, rows=n, bulk_kv=bulk_kv)
        rec["served"][d] = {}
        for b in BLOCKS:
            blk = interleaved_block(model, ws, prompt, device, block=b, rows=n, steps=steps, bulk_kv=bulk_kv,
                                    trace_dir=trace_dir, depth=d)
            rec["served"][d][b] = blk
            meds = " ".join(f"{st} {blk['arms'][st]['median_ms']} ms" for st in SETTINGS)
            print(f"P124_BLOCK d{d} {b} ({'-'.join(blk['order'])}) {meds} at {time.time() - t0:.0f} s", flush=True)
    rec["quality"] = quality(model, ws[:rows], prompt=prompt, cont=cont, chunk=chunk, floor_chunk=floor_chunk,
                             device=device, stand_in=stand_in)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--cont", type=int, default=128)
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
    ws = p117_box.windows(parts.tokenizer, a.rows, a.prompt, a.cont, a.stride)
    rec = measure(model, ws, prompt=a.prompt, cont=a.cont, device=cfg.device, rows=a.rows, steps=a.steps,
                  bulk_kv=cfg.bulk_kv, trace_dir=os.path.dirname(os.path.abspath(a.out)))
    rec = {"model": cfg.model, "revision": cfg.revision, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "transformers": transformers.__version__, "torch": torch.__version__,
           "load_s": round(load_s, 1), "stride": a.stride, "attn_int4_wide_env": os.environ.get("E4B_ATTN_INT4_WIDE"),
           "census_build": {k: parts.info.get(k) for k in ("moe_layers", "experts", "top_k", "model_type",
                                                           "int4_expert_layers", "int4_attn_projections",
                                                           "fuse_qkv_n", "fuse_t1_glue_n", "fuse_router_epilogue_n")},
           **rec, "gpu": torch.cuda.get_device_name(0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    open(a.out, "w").write(json.dumps(rec, indent=1))
    s = rec["served"]
    line = " | ".join(f"d{d} {b} {st} {blk['arms'][st]['median_ms']} ms" for d, blocks in s.items()
                      for b, blk in blocks.items() for st in SETTINGS)
    print(f"P124_BOX: {line} | int4 linears {rec['int4_linears']} | rep identical {rec['quality']['rep_identical']} | "
          f"function {rec['quality']['function']} | load {load_s:.0f}s | {rec['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
