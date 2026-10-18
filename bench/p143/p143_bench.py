#!/usr/bin/env python3
"""p143_bench.py -- lane P143 step 1: the paged FP8 attention decode at B=1, by context and split count, in CUDA graphs
(bench/p143/PREREG-p143.md; e4b#1313).

The served call, reproduced: e4b's own ``Fp8PagedKV`` at Qwen3-30B-A3B's geometry (48 layers, 4 KV heads, head_dim 128,
16 slots, 4096 tokens per slot, ``k_groups`` auto), slot 0 filled to 4096 tokens per layer by one bulk ``append`` and cut
back to each context by ``rewind``. Each layer's kernel arguments are ``kernel_args(layer, [0])`` -- exactly what
``Fp8PagedKV.attention`` hands ``fp8_paged_decode_attention`` -- precomputed outside the graph, with Qwen3's
``sm_scale`` (head_dim ** -0.5), no window and no sinks, as ``paged_attention`` passes them. The only change per arm is
``n_split`` (``None`` = the library's plan, which sizes splits from the block table's capacity: 32 at B=1).

Per (context, n_split): one CUDA graph of the 48 layers' calls, warmed, then ``REPS`` replays each timed by its own
event pair; two passes, the second in reverse arm order (the self-pair). Writes the per-replay medians in ms, per pass.

    python p143_bench.py --out p143.json
    python p143_bench.py --self-test
"""
import argparse
import json
import statistics
import sys
import time

CTX = (256, 640, 1024, 2048, 4096)
SPLITS = (None, 4, 8, 16, 32)          # None: the library's plan
LAYERS, HKV, HQ, D, SLOTS, CAP = 48, 4, 32, 128, 16, 4096
REPS, WARM = 200, 20


def arms():
    return [(c, s) for c in CTX for s in SPLITS]


def run(out_path) -> int:
    import torch
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from fp8_paged_attn import fp8_paged_decode_attention

    dev = "cuda"
    torch.manual_seed(143)
    kv = Fp8PagedKV(LAYERS, HKV, D, batch=SLOTS, max_tokens_per_seq=CAP, device=dev)
    for layer in range(LAYERS):
        k = torch.randn(CAP, HKV, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(CAP, HKV, D, device=dev, dtype=torch.bfloat16)
        kv.append(layer, 0, k, v)
    torch.cuda.synchronize()
    q = torch.randn(1, HQ, D, device=dev, dtype=torch.bfloat16)
    scale = D ** -0.5
    rec = {"geometry": {"layers": LAYERS, "kv_heads": HKV, "q_heads": HQ, "head_dim": D, "slots": SLOTS, "cap_tokens": CAP,
                        "block_tokens": kv.bt, "k_groups": kv.kgs[0], "block_table_cols": None},
           "reps": REPS, "passes": {}, "status": "ok"}
    graphs = {}
    # kernel_args returns FRESH index_select copies of the block table and lengths. A CUDA graph keeps only their
    # pointers and does not keep them alive, so every context's copies are held for the whole run (review of #1759):
    # otherwise the allocator hands a freed context's blocks to the next context's copies and its graphs read those.
    held = {}
    for ctx in CTX:
        kv.rewind(0, ctx)
        held[ctx] = [kv.kernel_args(layer, [0]) for layer in range(LAYERS)]
        rec["geometry"]["block_table_cols"] = int(held[ctx][0][2].shape[1])
        for s in SPLITS:
            kw = {} if s is None else {"n_split": s}

            def step(args=held[ctx], kw=kw):
                for layer, (kf, vf, tbl, lens) in enumerate(args):
                    fp8_paged_decode_attention(q, kf, vf, tbl, lens, n_kv_heads=kv.Hs[layer], head_dim=kv.Ds[layer],
                                               block_tokens=kv.bt, k_groups=kv.kgs[layer], k_row_bytes=kv.k_row,
                                               v_row_bytes=kv.v_row, sm_scale=scale, window=0, sinks=None, **kw)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step()
            graphs[(ctx, s)] = g
    torch.cuda.synchronize()
    for ctx, args in held.items():                       # the held lengths still read their context, every layer
        got = sorted({int(lens[0].item()) for _kf, _vf, _tbl, lens in args})
        if got != [ctx]:
            raise SystemExit(f"REFUSED: context {ctx}'s held lengths read {got} before timing")
    rec["held_lengths_checked"] = True
    order = arms()
    for p, seq in (("1", order), ("2", list(reversed(order)))):
        res = {}
        for ctx, s in seq:
            g = graphs[(ctx, s)]
            for _ in range(WARM):
                g.replay()
            torch.cuda.synchronize()
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(REPS)]
            for a, b in ev:
                a.record()
                g.replay()
                b.record()
            torch.cuda.synchronize()
            ms = sorted(a.elapsed_time(b) for a, b in ev)
            res[f"{ctx}/{s}"] = {"median_ms": round(statistics.median(ms), 6),
                                 "p10_ms": round(ms[len(ms) // 10], 6), "p90_ms": round(ms[(9 * len(ms)) // 10], 6)}
            print(f"P143 pass {p} ctx {ctx} n_split {s}: {res[f'{ctx}/{s}']['median_ms']:.4f} ms per 48 launches", flush=True)
        rec["passes"][p] = res
    rec["device"] = {"name": torch.cuda.get_device_name(0), "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=1)
    return 0


def self_test() -> int:
    cases = [("the registered grid", CTX == (256, 640, 1024, 2048, 4096) and SPLITS == (None, 4, 8, 16, 32)),
             ("25 arms, each once per pass", len(arms()) == 25 and len(set(arms())) == 25),
             ("Qwen3-30B-A3B's attention geometry", (LAYERS, HKV, HQ, D) == (48, 4, 32, 128)),
             ("the served pool: 16 slots of 4096 tokens", (SLOTS, CAP) == (16, 4096))]
    bad = [n for n, ok in cases if not ok]
    print(f"p143_bench self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    return run(a.out)


if __name__ == "__main__":
    sys.exit(main())
