#!/usr/bin/env python3
"""p143a2_kernel.py -- lane P143 Amendment 2, the kernel half (bench/p143/PREREG-p143.md, "Amendment 2"; e4b#1313).
No model: e4b's own ``Fp8PagedKV`` at Qwen3-30B-A3B's attention geometry (48 layers, 4 KV heads, 32 query heads,
head_dim 128, 16 slots of 4096 tokens, ``k_groups`` auto), as step 1 (``p143_bench.py``) built it.

(a) The tolerance contract (the cap is not bitwise: 16 splits regroup the fp32 log-sum-exp combine). Slots 0..3 filled
    to 4096 tokens per layer and cut back to each context, longest first. At each B in {1, 2, 4} and context in {256,
    640, 1024, 2048, 4096}, the SERVED call (``Fp8PagedKV.attention``, Qwen3's ``sm_scale``) runs every layer twice: once
    with the KV's cap unset (the library's plan) and once at ``split_cap = 16`` (the KV hands the kernel the cap, as the
    server does). Each output is held against an fp64 reference from the KV's own oracle (``reference_kv`` at float64:
    the stored e4m3 bytes and scales, dequantized), softmax(q k^T * scale) v in fp64 with grouped-query heads (query
    head h reads KV head h // 8). Recorded per cell: the RMS error of each arm over 48 layers x B rows x 32 heads x 128,
    its max, the reference's RMS, and whether the two arms agree bitwise.
(b) The timing add-on. Slots 4..7, cut back to 2048, 1024 and 640 tokens (longest first); at each B in {1, 2, 4} (B = 1
    repeats step 1's seen data; the rule reads B 2 and 4) each context's ``kernel_args`` are copied and HELD for the run
    (a CUDA graph keeps only pointers: #1759's review). One graph of the 48 layers' kernel calls per (B, context, arm),
    arm = plan (no ``split_cap``) or cap (``split_cap = 16``); 20 warm replays, then 200 each timed by its own event pair;
    two passes, the second in reverse order. Per-replay medians in ms.

    python p143a2_kernel.py --out kernel.json
    python p143a2_kernel.py --self-test
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

LAYERS, HKV, HQ, D, SLOTS, CAP = 48, 4, 32, 128, 16, 4096
SPLIT_CAP = 16
CONTRACT_B, CONTRACT_CTX = (1, 2, 4), (256, 640, 1024, 2048, 4096)
TIMING_B, TIMING_CTX = (1, 2, 4), (640, 1024, 2048)     # the rule reads B 2 and 4; B = 1 is step 1's, reported
CONTRACT_SLOTS, TIMING_SLOTS = (0, 1, 2, 3), (4, 5, 6, 7)
REPS, WARM = 200, 20


def timing_arms():
    return [(b, c, arm) for b in TIMING_B for c in TIMING_CTX for arm in ("plan", "cap")]


def fill(kv, torch, slots, dev, layers=LAYERS):
    """Each slot of every layer filled to ``CAP`` tokens by one bulk ``append`` (random bf16 K and V)."""
    for layer in range(layers):
        for s in slots:
            kv.append(layer, s, torch.randn(CAP, HKV, D, device=dev, dtype=torch.bfloat16),
                      torch.randn(CAP, HKV, D, device=dev, dtype=torch.bfloat16))


def ref_attention(torch, q, k, v, scale):
    """fp64: q [HQ, D]; k, v [t, HKV, D]; query head h reads KV head h // (HQ // HKV)."""
    g = q.shape[0] // k.shape[1]
    kk = k.permute(1, 0, 2).repeat_interleave(g, 0)
    vv = v.permute(1, 0, 2).repeat_interleave(g, 0)
    p = torch.softmax(torch.einsum("hd,htd->ht", q, kk) * scale, dim=-1)
    return torch.einsum("ht,htd->hd", p, vv)


def contract(kv, torch, q_all, scale) -> dict:
    out = {}
    for ctx in sorted(CONTRACT_CTX, reverse=True):          # rewind refuses a forward move (Amendment 1)
        for s in CONTRACT_SLOTS:
            kv.rewind(s, ctx)
        refs = {}
        for layer in range(LAYERS):
            for s in CONTRACT_SLOTS:
                k, v = kv.reference_kv(layer, s, dtype=torch.float64)
                if k.shape[0] != ctx:
                    raise SystemExit(f"REFUSED: slot {s} layer {layer} reads {k.shape[0]} tokens, not {ctx}")
                refs[(layer, s)] = ref_attention(torch, q_all[s].double(), k, v, scale)
        for B in CONTRACT_B:
            slots = list(CONTRACT_SLOTS[:B])
            q = q_all[:B].contiguous()
            sq = {"plan": 0.0, "cap": 0.0}
            mx = {"plan": 0.0, "cap": 0.0}
            ref_sq, n, same = 0.0, 0, True
            for layer in range(LAYERS):
                ref = torch.stack([refs[(layer, s)] for s in slots])
                kv.split_cap = None
                o_plan = kv.attention(layer, q, slots=slots, sm_scale=scale)
                kv.split_cap = SPLIT_CAP
                o_cap = kv.attention(layer, q, slots=slots, sm_scale=scale)
                kv.split_cap = None
                same = same and bool(torch.equal(o_plan, o_cap))
                for arm, o in (("plan", o_plan), ("cap", o_cap)):
                    e = o.double() - ref
                    sq[arm] += float((e * e).sum())
                    mx[arm] = max(mx[arm], float(e.abs().max()))
                ref_sq += float((ref * ref).sum())
                n += ref.numel()
            cell = {"err_plan": math.sqrt(sq["plan"] / n), "err_cap": math.sqrt(sq["cap"] / n),
                    "max_plan": mx["plan"], "max_cap": mx["cap"], "ref_rms": math.sqrt(ref_sq / n),
                    "bitwise_equal": same, "out_dtype": str(o_plan.dtype)}
            out[f"{B}/{ctx}"] = cell
            print(f"P143A2_CONTRACT B={B} ctx={ctx} err_plan={cell['err_plan']:.4e} err_cap={cell['err_cap']:.4e} "
                  f"ratio={cell['err_cap'] / cell['err_plan']:.4f} bitwise={same}", flush=True)
    return out


def hold_args(kv, layers=LAYERS):
    """``(B, ctx) -> per-layer kernel_args``, longest context first (``rewind`` refuses a forward move); the copies are
    held for the run (a CUDA graph keeps only their pointers)."""
    held = {}
    for ctx in sorted(TIMING_CTX, reverse=True):
        for s in TIMING_SLOTS:
            kv.rewind(s, ctx)
        for B in TIMING_B:
            held[(B, ctx)] = [kv.kernel_args(layer, list(TIMING_SLOTS[:B])) for layer in range(layers)]
    return held


def check_held(held):
    for (B, ctx), args in held.items():
        got = sorted({int(x) for _kf, _vf, _tbl, lens in args for x in lens.tolist()})
        if got != [ctx] or any(int(lens.shape[0]) != B for _kf, _vf, _tbl, lens in args):
            raise SystemExit(f"REFUSED: (B={B}, ctx={ctx})'s held lengths read {got} before timing")


def timing(kv, torch, q_all, scale, fp8_paged_decode_attention) -> dict:
    held = hold_args(kv)
    graphs, keep = {}, []
    for B, ctx, arm in timing_arms():
        q = q_all[:B].contiguous()
        keep.append(q)
        kw = {"split_cap": SPLIT_CAP} if arm == "cap" else {}

        def step(args=held[(B, ctx)], q=q, kw=kw):
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
        graphs[(B, ctx, arm)] = g
    torch.cuda.synchronize()
    check_held(held)                                         # before any timing
    res = {}
    order = timing_arms()
    for p, seq in (("p1", order), ("p2", list(reversed(order)))):
        for B, ctx, arm in seq:
            g = graphs[(B, ctx, arm)]
            for _ in range(WARM):
                g.replay()
            torch.cuda.synchronize()
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(REPS)]
            for a, b in ev:
                a.record()
                g.replay()
                b.record()
            torch.cuda.synchronize()
            ms = statistics.median(a.elapsed_time(b) for a, b in ev)
            res.setdefault(f"{B}/{ctx}/{arm}", {})[p] = round(ms, 6)
            print(f"P143A2_TIMING {p} B={B} ctx={ctx} {arm}: {ms:.4f} ms per 48 launches", flush=True)
    return res


def run(out_path) -> int:
    import torch
    import fp8_paged_attn
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from fp8_paged_attn import fp8_paged_decode_attention

    dev = "cuda"
    torch.manual_seed(1432)
    kv = Fp8PagedKV(LAYERS, HKV, D, batch=SLOTS, max_tokens_per_seq=CAP, device=dev)
    fill(kv, torch, CONTRACT_SLOTS + TIMING_SLOTS, dev)
    torch.cuda.synchronize()
    q_all = torch.randn(len(CONTRACT_SLOTS), HQ, D, device=dev, dtype=torch.bfloat16)
    scale = D ** -0.5
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    cols = int(kv.block_table[0].shape[1])
    rec = {"status": "ok", "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "geometry": {"layers": LAYERS, "kv_heads": HKV, "q_heads": HQ, "head_dim": D, "slots": SLOTS, "cap_tokens": CAP,
                        "block_tokens": kv.bt, "k_groups": kv.kgs[0], "block_table_cols": cols},
           "device": {"name": torch.cuda.get_device_name(0), "sm_count": sms},
           "split_cap": SPLIT_CAP, "reps": REPS, "warm": WARM}
    rec["contract"] = contract(kv, torch, q_all, scale)
    rec["contract_plans"] = {str(b): dict(r) for b, r in sorted(kv.split_plans.items())}
    plan = fp8_paged_attn.decode_split_plan
    kw = dict(block_tokens=kv.bt, ktile=64, compute="fp8", pack_heads=False, sms=sms)
    rec["timing_plans"] = {str(b): {"plan": plan(b, HKV, cols, **kw), "n_split": plan(b, HKV, cols, split_cap=SPLIT_CAP, **kw)}
                           for b in TIMING_B}
    rec["timing"] = timing(kv, torch, q_all, scale, fp8_paged_decode_attention)
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=1)
    return 0


def self_test() -> int:
    cases = [("the registered contract grid", (CONTRACT_B, CONTRACT_CTX) == ((1, 2, 4), (256, 640, 1024, 2048, 4096))),
             ("the registered timing grid", (TIMING_B, TIMING_CTX) == ((1, 2, 4), (640, 1024, 2048))),
             ("18 timing arms, each once per pass", len(timing_arms()) == 18 and len(set(timing_arms())) == 18),
             ("the contract and the timing own different slots", not set(CONTRACT_SLOTS) & set(TIMING_SLOTS)
              and max(CONTRACT_SLOTS + TIMING_SLOTS) < SLOTS and len(CONTRACT_SLOTS) >= max(CONTRACT_B)
              and len(TIMING_SLOTS) >= max(TIMING_B)),
             ("Qwen3-30B-A3B's attention geometry", (LAYERS, HKV, HQ, D) == (48, 4, 32, 128)),
             ("the served pool: 16 slots of 4096 tokens", (SLOTS, CAP) == (16, 4096)),
             ("the cap", SPLIT_CAP == 16)]
    try:
        import torch
        g = torch.Generator().manual_seed(0)
        q = torch.randn(HQ, D, dtype=torch.float64, generator=g)
        k = torch.randn(9, HKV, D, dtype=torch.float64, generator=g)
        v = torch.randn(9, HKV, D, dtype=torch.float64, generator=g)
        got = ref_attention(torch, q, k, v, D ** -0.5)
        h = 13                                              # query head 13 reads KV head 1
        p = torch.softmax(k[:, 1] @ q[h] * D ** -0.5, 0)
        cases.append(("the fp64 reference: query head 13 reads KV head 1", torch.allclose(got[h], p @ v[:, 1])))
        one = torch.zeros(1, HKV, D, dtype=torch.float64)
        one[0, 3] = 1.0
        cases.append(("one token: the output is its V", torch.allclose(ref_attention(torch, q, one, one, 1.0)[31], one[0, 3])))
    except ImportError:
        pass
    bad = [n for n, ok in cases if not ok]
    print(f"p143a2_kernel self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
