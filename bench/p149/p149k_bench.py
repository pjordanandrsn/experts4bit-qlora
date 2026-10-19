#!/usr/bin/env python3
"""p149k_bench.py -- lane P149, box K (bench/p149/PREREG-p149.md): Mixtral-8x7B's attention projections at decode on one
card, cuBLAS bf16 against lane P149's prototype W8A16 kernels (p149_kernels.py), in one CUDA-graph harness. No model.

Shapes (P145's per-name table): ``qo`` 64 x [4096, 4096] (q and o, 32 layers each) and ``kv`` 64 x [1024, 4096] (k and v).
Every launch reads its own weight tensor, so a replay streams the step's bytes from DRAM as the served step does (2.1 GB
and 0.54 GB of bf16, far above the L2). cuBLAS is ``torch.nn.functional.linear`` on ``[M, K]`` rows, as ``nn.Linear``
serves it, against the int8 kernel on the same weights' quantized copy: ``gemv_i8`` at M == 1, ``gemm_i8_small`` at M == 16.

Phases, in order, per shape and per M (1, then 16):
  contract  layer 0's weight and 4 random inputs: the relative RMS error against an fp64 reference, cuBLAS against the
            bf16 weights and every registered config against its dequantized weights.
  select    every config whose error is <= 1.05 x cuBLAS's: one graph of the shape's 64 launches, 10 warm + 50 timed
            replays (median); the fastest is the choice.
  read      fresh graphs A (cuBLAS) and B (the choice), timed A1, B1, B2, A2: 20 warm + 200 timed replays each, each
            replay by its own event pair, medians. The GPU's temperature and SM clock are read before A1 and after A2.
  cublas    one eager cuBLAS call per shape and M, profiled: every kernel's name and grid (the harness must run P145's
            in-situ grid at M == 1).
The GPU state (power.limit, enforced.power.limit, clocks.max.sm, temperature, clocks.sm) is recorded at the start and end.

    python p149k_bench.py --out p149k.json
    python p149k_bench.py --contract-only --out contract.json      (any CUDA card: correctness, no timing)
    python p149k_bench.py --self-test
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SHAPES = {"qo": (4096, 4096, 64), "kv": (1024, 4096, 64)}
MS = (1, 16)
CONTRACT_X, CONTRACT_RATIO = 4, 1.05
SEL_WARM, SEL_REPS = 10, 50
READ_WARM, READ_REPS = 20, 200
ORDER = ("A1", "B1", "B2", "A2")
GPU_FIELDS = "power.limit,enforced.power.limit,clocks.max.sm,temperature.gpu,clocks.sm"


def gpu_state():
    """The card's power limits, max SM clock, temperature and current SM clock (reported, never ruled)."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        return dict(zip(GPU_FIELDS.split(","), (v.strip() for v in out.split(","))))
    except Exception as e:  # noqa: BLE001 - reported only
        return {"error": f"{type(e).__name__}: {e}"[:200]}


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


def rel_rms(got, ref):
    return float(((got.double() - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())


def key(cfg):
    return ",".join(map(str, cfg))


def build(torch, dev, one_layer=False):
    """Per shape: the bf16 weights (one per launch; layer 0 only when ``one_layer``), their int8 copies, and the rows."""
    import p149_kernels as K
    gen = torch.Generator(device=dev).manual_seed(149)
    w, qs, x = {}, {}, {}
    for s, (n, k, layers) in SHAPES.items():
        w[s] = [(torch.randn(n, k, device=dev, generator=gen) / k ** 0.5).to(torch.bfloat16)
                for _ in range(1 if one_layer else layers)]
        qs[s] = [K.quantize_rows_int8(W) for W in w[s]]
        x[s] = {m: torch.randn(m, k, device=dev, generator=gen).to(torch.bfloat16) for m in MS}
    return w, qs, x


def configs(m):
    import p149_kernels as K
    return K.CONFIGS_GEMV if m == 1 else K.CONFIGS_SMALL


def kernel_call(m):
    """``(x, q, s, out, cfg, ws)`` -> the int8 kernel for ``m`` rows."""
    import p149_kernels as K
    if m == 1:
        return lambda x, q, s, out, cfg, ws=None: K.gemv_i8(x[0], q, s, out[0], cfg, ws=ws)
    return lambda x, q, s, out, cfg, ws=None: K.gemm_i8_small(x, q, s, out, cfg, ws=ws)


def contract(torch, w, qs, dev):
    out = {}
    gen = torch.Generator(device=dev).manual_seed(1490)
    for s, (n, k, _layers) in SHAPES.items():
        W = w[s][0]
        q, sc = qs[s][0]
        Wd = W.double()
        Qd = q.double() * sc.double()[:, None]
        out[s] = {}
        for m in MS:
            xs = [torch.randn(m, k, device=dev, generator=gen).to(torch.bfloat16) for _ in range(CONTRACT_X)]
            refs = [xx.double() @ Wd.T for xx in xs]
            refs_q = [xx.double() @ Qd.T for xx in xs]
            cub = [rel_rms(torch.nn.functional.linear(xx, W), r) for xx, r in zip(xs, refs)]
            rec = {"cublas": max(cub), "configs": {}}
            call = kernel_call(m)
            for cfg in configs(m):
                o = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
                errs = [rel_rms(call(xx, q, sc, o, cfg), r) for xx, r in zip(xs, refs_q)]
                rec["configs"][key(cfg)] = max(errs)
            out[s][str(m)] = rec
            print(f"P149K_CONTRACT {s} M={m}: cublas {rec['cublas']:.3e}; configs {min(rec['configs'].values()):.3e}"
                  f"..{max(rec['configs'].values()):.3e}", flush=True)
        del Wd, Qd
    return out


def passing(rec):
    """The configs whose error is <= CONTRACT_RATIO x cuBLAS's, as tuples."""
    return [tuple(int(v) for v in c.split(",")) for c, e in rec["configs"].items() if e <= CONTRACT_RATIO * rec["cublas"]]


def steps(torch, s, m, w, qs, x, dev):
    """``(cublas_step, kernel_step_for(cfg))`` for shape ``s`` at ``m`` rows: each runs the shape's 64 launches."""
    n, k, layers = SHAPES[s]
    out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    xm = x[s][m]
    call = kernel_call(m)

    def cublas():
        for W in w[s]:
            torch.nn.functional.linear(xm, W)

    def kernel_for(cfg):
        ws = torch.empty(max(cfg[3], 1), m * n, dtype=torch.float32, device=dev)
        wsv = ws.view(cfg[3], n) if m == 1 else ws.view(cfg[3], m, n)

        def step():
            for q, sc in qs[s]:
                call(xm, q, sc, out, cfg, ws=wsv)
        return step
    return cublas, kernel_for


def cublas_kernels(torch, w, x, out_dir):
    """Per shape and M, the CUDA kernels one eager ``F.linear`` launches: ``[{name, grid}]``, from the profiler's trace."""
    import tempfile
    from torch.profiler import ProfilerActivity, profile
    out = {}
    for s in SHAPES:
        out[s] = {}
        for m in MS:
            torch.nn.functional.linear(x[s][m], w[s][0])
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                torch.nn.functional.linear(x[s][m], w[s][0])
                torch.cuda.synchronize()
            path = os.path.join(out_dir or tempfile.gettempdir(), f"p149k_cublas_{s}_{m}.trace.json")
            prof.export_chrome_trace(path)
            ev = json.load(open(path)).get("traceEvents", [])
            out[s][str(m)] = [{"name": e.get("name"), "grid": (e.get("args") or {}).get("grid")}
                              for e in ev if e.get("cat") == "kernel"]
            os.remove(path)
            print(f"P149K_CUBLAS {s} M={m}: {[(kk['name'][:40], kk['grid']) for kk in out[s][str(m)]]}", flush=True)
    return out


def run(out_path, contract_only=False) -> int:
    import torch
    import triton
    dev = "cuda"
    rec = {"status": "ok", "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "device": {"name": torch.cuda.get_device_name(0), "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
                      "cc": list(torch.cuda.get_device_capability(0))},
           "torch": torch.__version__, "triton": triton.__version__, "shapes": {s: list(v) for s, v in SHAPES.items()},
           "ms": list(MS), "contract_x": CONTRACT_X, "contract_ratio": CONTRACT_RATIO,
           "sel": {"warm": SEL_WARM, "reps": SEL_REPS}, "read": {"warm": READ_WARM, "reps": READ_REPS, "order": list(ORDER)},
           "gpu_state": {"start": gpu_state()}}
    w, qs, x = build(torch, dev, one_layer=contract_only)
    rec["contract"] = contract(torch, w, qs, dev)
    rec["cublas_kernels"] = cublas_kernels(torch, w, x, os.path.dirname(os.path.abspath(out_path)))
    if contract_only:
        rec["mode"] = "contract-only"
        rec["gpu_state"]["end"] = gpu_state()
        json.dump(rec, open(out_path, "w"), indent=1)
        return 0
    rec["select"], rec["choice"], rec["reads"] = {}, {}, {}
    for s in SHAPES:
        for sel_d in (rec["select"], rec["choice"], rec["reads"]):
            sel_d[s] = {}
        for m in MS:
            cublas, kernel_for = steps(torch, s, m, w, qs, x, dev)
            ok = passing(rec["contract"][s][str(m)])
            sel = {}
            for cfg in ok:
                g = _capture(torch, kernel_for(cfg))
                sel[key(cfg)] = _median_ms(torch, g, SEL_WARM, SEL_REPS)
                del g
            rec["select"][s][str(m)] = sel
            if not sel:
                print(f"P149K_SELECT {s} M={m}: no config passes the contract", flush=True)
                continue
            choice = min(sel, key=sel.get)
            rec["choice"][s][str(m)] = choice
            cfg = tuple(int(v) for v in choice.split(","))
            ga, gb = _capture(torch, cublas), _capture(torch, kernel_for(cfg))
            r = {"gpu_before": gpu_state()}
            for tag in ORDER:
                r[tag] = _median_ms(torch, ga if tag[0] == "A" else gb, READ_WARM, READ_REPS)
            r["gpu_after"] = gpu_state()
            rec["reads"][s][str(m)] = r
            print(f"P149K_READ {s} M={m}: choice {choice}; A1 {r['A1']:.4f} B1 {r['B1']:.4f} B2 {r['B2']:.4f} "
                  f"A2 {r['A2']:.4f} ms", flush=True)
            del ga, gb
    rec["gpu_state"]["end"] = gpu_state()
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(rec, open(out_path, "w"), indent=1)
    return 0


def self_test() -> int:
    import p149_kernels as K
    cases = [("the registered shapes (Mixtral's attention projections, P145's calls)",
              SHAPES == {"qo": (4096, 4096, 64), "kv": (1024, 4096, 64)}),
             ("M == 1 rules, M == 16 is read", MS == (1, 16)),
             ("the read order is ABBA", ORDER == ("A1", "B1", "B2", "A2")),
             ("the contract bar", (CONTRACT_X, CONTRACT_RATIO) == (4, 1.05)),
             ("the timing counts", (SEL_WARM, SEL_REPS, READ_WARM, READ_REPS) == (10, 50, 20, 200)),
             ("the grids", len(K.CONFIGS_GEMV) == 54 and len(K.CONFIGS_SMALL) == 72),
             ("passing keeps <= 1.05x", passing({"cublas": 1.0, "configs": {"4,512,4,1": 1.05, "8,512,4,1": 1.0501,
                                                                           "16,1024,8,2": 0.9}})
              == [(4, 512, 4, 1), (16, 1024, 8, 2)]),
             ("the GPU state names the power limits, clocks and temperature",
              GPU_FIELDS == "power.limit,enforced.power.limit,clocks.max.sm,temperature.gpu,clocks.sm")]
    bad = [n for n, ok in cases if not ok]
    print(f"p149k_bench self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
