#!/usr/bin/env python3
"""p148_bench.py -- lane P148 (bench/p148/PREREG-p148.md; e4b#1313): the dense bf16 decode GEMVs of Qwen3-30B-A3B's
B=1 step on one card, cuBLAS against lane P148's prototype kernels (p148_kernels.py), in one CUDA-graph harness. No model.

Shapes (P142's per-name table): ``qkv`` 48 x [5120, 2048] and ``o`` 48 x [2048, 4096] (D1, bf16), ``head`` 1 x
[151936, 2048] (D2: cuBLAS on the bf16 head against the int8 kernel on its quantized copy). Every one of the 48 launches
reads its own weight tensor, so a replay streams the step's bytes from DRAM as the served step does (1.0 GB / 0.8 GB /
0.62 GB, all well above the L2). cuBLAS is ``torch.nn.functional.linear`` on a ``[1, K]`` row, as ``nn.Linear`` serves it.

Phases, in order:
  contract  per shape, layer 0's weight and 4 random rows: the relative RMS error against an fp64 reference, for cuBLAS
            and every registered config (D2: the int8 kernel against its dequantized weights).
  select    per shape, every config whose error is <= 1.05 x cuBLAS's: one graph of the shape's launches, 10 warm + 50
            timed replays (median); the fastest is the shape's choice.
  read      per shape, fresh graphs A (cuBLAS) and B (the choice), timed A, B, B, A: 20 warm + 200 timed replays each,
            each replay by its own event pair, medians. Pairs (A1, B1) and (B2, A2) run in opposite orders.
  cublas    per shape, one eager cuBLAS call profiled (torch.profiler, CUDA activity): every kernel's name and grid, so the
            reducer can hold the harness's cuBLAS kernel to P142's in-situ grid (the review of #1796).

    python p148_bench.py --out p148.json
    python p148_bench.py --contract-only --out contract.json      (any CUDA card: correctness, no timing)
    python p148_bench.py --self-test
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SHAPES = {"qkv": (5120, 2048, 48), "o": (2048, 4096, 48), "head": (151936, 2048, 1)}
D1_SHAPES, D2_SHAPES = ("qkv", "o"), ("head",)
CONTRACT_X, CONTRACT_RATIO = 4, 1.05
SEL_WARM, SEL_REPS = 10, 50
READ_WARM, READ_REPS = 20, 200
ORDER = ("A1", "B1", "B2", "A2")


def _median_ms(torch, g, warm, reps):
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in ev:
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    return round(statistics.median(a.elapsed_time(b) for a, b in ev), 6)


def _capture(torch, step):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    return g


def cublas_kernels(torch, w, x, out_dir):
    """Per shape, the CUDA kernels one eager ``F.linear`` launches: ``[{name, grid}]``, from the profiler's trace."""
    import tempfile
    from torch.profiler import ProfilerActivity, profile
    out = {}
    for s in SHAPES:
        torch.nn.functional.linear(x[s][None, :], w[s][0])          # warm: the heuristic and the handle exist
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            torch.nn.functional.linear(x[s][None, :], w[s][0])
            torch.cuda.synchronize()
        path = os.path.join(out_dir or tempfile.gettempdir(), f"p148_cublas_{s}.trace.json")
        prof.export_chrome_trace(path)
        ev = json.load(open(path)).get("traceEvents", [])
        out[s] = [{"name": e.get("name"), "grid": (e.get("args") or {}).get("grid")}
                  for e in ev if e.get("cat") == "kernel"]
        os.remove(path)
        print(f"P148_CUBLAS {s}: {[(k['name'][:40], k['grid']) for k in out[s]]}", flush=True)
    return out


def rel_rms(got, ref):
    return float(((got.double() - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())


def build(torch, dev, one_layer=False):
    """The weights (one tensor per launch; layer 0 only when ``one_layer``), the int8 head and the rows."""
    import p148_kernels as K
    gen = torch.Generator(device=dev).manual_seed(148)
    w, x = {}, {}
    for s, (n, k, layers) in SHAPES.items():
        w[s] = [(torch.randn(n, k, device=dev, generator=gen) / k ** 0.5).to(torch.bfloat16)
                for _ in range(1 if one_layer else layers)]
        x[s] = torch.randn(k, device=dev, generator=gen).to(torch.bfloat16)
    q, sc = K.quantize_rows_int8(w["head"][0])
    return w, x, (q, sc)


def contract(torch, w, x, head_q, dev):
    import p148_kernels as K
    out = {}
    gen = torch.Generator(device=dev).manual_seed(1480)
    for s, (n, k, _layers) in SHAPES.items():
        rows = [torch.randn(k, device=dev, generator=gen).to(torch.bfloat16) for _ in range(CONTRACT_X)]
        W = w[s][0]
        Wd = W.double()                                  # once per shape: the head's fp64 copy is 2.5 GB
        refs = [Wd @ r.double() for r in rows]
        del Wd
        cub = [rel_rms(torch.nn.functional.linear(r[None, :], W)[0], ref) for r, ref in zip(rows, refs)]
        rec = {"cublas": max(cub), "configs": {}}
        if s in D1_SHAPES:
            for cfg in K.CONFIGS_BF16:
                o = torch.empty(n, dtype=torch.bfloat16, device=dev)
                errs = [rel_rms(K.gemv_bf16(r, W, o, cfg), ref) for r, ref in zip(rows, refs)]
                rec["configs"][",".join(map(str, cfg))] = max(errs)
        else:
            q, sc = head_q
            Qd = q.double() * sc.double()[:, None]
            refs_q = [Qd @ r.double() for r in rows]
            del Qd
            for cfg in K.CONFIGS_I8:
                o = torch.empty(n, dtype=torch.bfloat16, device=dev)
                errs = [rel_rms(K.gemv_i8(r, q, sc, o, cfg), ref) for r, ref in zip(rows, refs_q)]
                rec["configs"][",".join(map(str, cfg))] = max(errs)
        out[s] = rec
        print(f"P148_CONTRACT {s}: cublas {rec['cublas']:.3e}; configs {min(rec['configs'].values()):.3e}"
              f"..{max(rec['configs'].values()):.3e}", flush=True)
    return out


def passing(rec):
    """The configs whose error is <= CONTRACT_RATIO x cuBLAS's, as tuples."""
    return [tuple(int(v) for v in c.split(",")) for c, e in rec["configs"].items() if e <= CONTRACT_RATIO * rec["cublas"]]


def steps(torch, s, w, x, head_q, dev):
    """``(cublas_step, kernel_step_for(cfg))`` for shape ``s``: each runs the shape's launches over its own weights."""
    import p148_kernels as K
    n, k, layers = SHAPES[s]
    out = torch.empty(n, dtype=torch.bfloat16, device=dev)
    x2 = x[s][None, :]

    def cublas():
        for W in w[s]:
            torch.nn.functional.linear(x2, W)

    def kernel_for(cfg):
        ws = torch.empty(max(cfg[3], 1), n, dtype=torch.float32, device=dev)
        if s in D1_SHAPES:
            def step():
                for W in w[s]:
                    K.gemv_bf16(x[s], W, out, cfg, ws=ws)
        else:
            q, sc = head_q

            def step():
                K.gemv_i8(x[s], q, sc, out, cfg)
        return step
    return cublas, kernel_for


def run(out_path, contract_only=False) -> int:
    import torch
    dev = "cuda"
    w, x, head_q = build(torch, dev, one_layer=contract_only)
    rec = {"status": "ok", "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "device": {"name": torch.cuda.get_device_name(0), "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
                      "cc": list(torch.cuda.get_device_capability(0))},
           "torch": torch.__version__, "shapes": {s: list(v) for s, v in SHAPES.items()},
           "contract_x": CONTRACT_X, "contract_ratio": CONTRACT_RATIO,
           "sel": {"warm": SEL_WARM, "reps": SEL_REPS}, "read": {"warm": READ_WARM, "reps": READ_REPS, "order": list(ORDER)}}
    import triton
    rec["triton"] = triton.__version__
    rec["contract"] = contract(torch, w, x, head_q, dev)
    rec["cublas_kernels"] = cublas_kernels(torch, w, x, os.path.dirname(os.path.abspath(out_path)))
    if contract_only:
        rec["mode"] = "contract-only"
        json.dump(rec, open(out_path, "w"), indent=1)
        return 0
    rec["select"], rec["choice"], rec["reads"] = {}, {}, {}
    for s in SHAPES:
        cublas, kernel_for = steps(torch, s, w, x, head_q, dev)
        ok = passing(rec["contract"][s])
        sel = {}
        for cfg in ok:
            g = _capture(torch, kernel_for(cfg))
            sel[",".join(map(str, cfg))] = _median_ms(torch, g, SEL_WARM, SEL_REPS)
            del g
        rec["select"][s] = sel
        if not sel:
            print(f"P148_SELECT {s}: no config passes the contract", flush=True)
            continue
        choice = min(sel, key=sel.get)
        rec["choice"][s] = choice
        cfg = tuple(int(v) for v in choice.split(","))
        ga, gb = _capture(torch, cublas), _capture(torch, kernel_for(cfg))
        r = {}
        for tag in ORDER:
            r[tag] = _median_ms(torch, ga if tag[0] == "A" else gb, READ_WARM, READ_REPS)
        rec["reads"][s] = r
        print(f"P148_READ {s}: choice {choice}; A1 {r['A1']:.4f} B1 {r['B1']:.4f} B2 {r['B2']:.4f} A2 {r['A2']:.4f} ms",
              flush=True)
        del ga, gb
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(rec, open(out_path, "w"), indent=1)
    return 0


def self_test() -> int:
    import p148_kernels as K
    cases = [("the registered shapes (P142's served GEMVs)", SHAPES == {"qkv": (5120, 2048, 48), "o": (2048, 4096, 48),
                                                                        "head": (151936, 2048, 1)}),
             ("D1 is qkv + o, D2 the head", D1_SHAPES == ("qkv", "o") and D2_SHAPES == ("head",)),
             ("the read order is ABBA", ORDER == ("A1", "B1", "B2", "A2")),
             ("the contract bar", (CONTRACT_X, CONTRACT_RATIO) == (4, 1.05)),
             ("the timing counts", (SEL_WARM, SEL_REPS, READ_WARM, READ_REPS) == (10, 50, 20, 200)),
             ("the grids", len(K.CONFIGS_BF16) == 36 and len(K.CONFIGS_I8) == 18),
             ("passing keeps <= 1.05x", passing({"cublas": 1.0, "configs": {"4,512,4,1": 1.05, "8,512,4,1": 1.0501,
                                                                           "2,1024,8,2": 0.9}})
              == [(4, 512, 4, 1), (2, 1024, 8, 2)])]
    bad = [n for n, ok in cases if not ok]
    print(f"p148_bench self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--contract-only", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    return run(a.out, contract_only=a.contract_only)


if __name__ == "__main__":
    sys.exit(main())
