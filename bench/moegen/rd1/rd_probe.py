#!/usr/bin/env python3
"""Lane RD1's probe (bench/moegen/rd1/RD1-PREREG.md): grouped-nf4-gemm's frozen-expert GEMM routes, per call, at each
family's registered expert shapes, under a uniform and a skewed router draw.

Arms (forward and dgrad of each projection):
  v1       the fused NF4 kernels as shipped: ``gemm_4bit_grouped`` (default variant: TF32 tensor cores, decode in the loop)
           and ``dgrad_4bit_grouped`` (fp32 operands into tl.dot).
  v3       the fused kernels' bf16 MMA: ``gemm_4bit_grouped(..., prefill_variant=3)`` (opt-in in grouped-nf4-gemm) and, for
           dgrad, which has no bf16 variant there, a probe-local copy of ``_dgrad_nf4_grouped`` that casts ``g`` and the
           decoded ``w`` to bf16 before tl.dot (best of three tile configs).
  dense    ``nf4_route.dense_forward`` / ``dense_dgrad``: per-expert dequant + torch.mm (``auto``'s route for <= 16 groups).
  decoded  ``nf4_route.dequant_groups`` (one launch, every present expert) + ONE Triton grouped bf16 GEMM launch (best of three
           tile configs), uncapped.
  decoded_cap  the same with the decode transient capped at --cap-mib: groups in chunks whose decoded bytes fit the cap, one
           dequant + one GEMM launch per chunk (the chunk plan is built once per call shape, outside the timed region, as a
           route would reuse its plan).

Per arm: device ms per call (profiler kernel time), event ms per call (CUDA events over --reps back-to-back calls: includes
launch and host gaps), peak MiB above the inputs (max_memory_allocated), relative error against v1, and ``rel_err32``: the
relative error against an fp32 reference (``dequant_ref`` in fp32 times the fp32 activations), which rd_table.py's
correctness gate reads. v1 is measured against that reference too (it runs TF32), so every arm sits in one frame.

usage: rd_probe.py --out rd.json [--seqs 512,2048] [--fams ...] [--routings uniform,skew] [--cap-mib 256] [--reps 20]
"""
import argparse
import json
import time

import torch
import triton
import triton.language as tl

import nf4_grouped as ng
import nf4_route as nr

# family -> (experts E, top-k, hidden H, expert intermediate I, gated): each checkpoint's config.json
FAMS = {
    "olmoe": (64, 8, 2048, 1024, True), "lfm2": (32, 4, 2048, 1792, True), "ernie": (64, 6, 2560, 1536, True),
    "graniteh": (64, 6, 1536, 512, True), "qwen3": (128, 8, 2048, 768, True), "nemotron": (128, 6, 2688, 1856, False),
    "qwen36": (256, 8, 2048, 512, True), "mixtral": (8, 2, 4096, 14336, True),
}
GEMM_CFGS = [(64, 128, 64, 4, 3), (32, 128, 64, 4, 3), (128, 128, 32, 8, 3)]
DGRAD16_CFGS = [(32, 64, 64, 2), (64, 64, 64, 4), (32, 128, 64, 4)]


@triton.jit
def _grouped_bf16_gemm(a_ptr, w_ptr, out_ptr, t_row0_ptr, t_rows_ptr, t_grp_ptr,
                       N_OUT, R, s_am, s_wg, s_wr, s_wj, s_om,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out[row0+i, j] = sum_r a[row0+i, r] * W[g, r, j] over one (M-tile, N-tile); W by strides, so one kernel serves the
    forward (a @ W_g^T) and the dgrad (grad_out @ W_g). One launch covers every group of the plan."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row0 = tl.load(t_row0_ptr + pid_m)
    rows = tl.load(t_rows_ptr + pid_m)
    g = tl.load(t_grp_ptr + pid_m).to(tl.int64)
    rm = tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mmask = rm < rows
    nmask = rn < N_OUT
    a_ptrs = a_ptr + (row0 + rm).to(tl.int64)[:, None] * s_am + rk[None, :]
    w_ptrs = w_ptr + g * s_wg + rk[:, None].to(tl.int64) * s_wr + rn[None, :].to(tl.int64) * s_wj
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, R, BK):
        kmask = (k0 + rk) < R
        a = tl.load(a_ptrs, mask=mmask[:, None] & kmask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BK
        w_ptrs += BK * s_wr
    o_ptrs = out_ptr + (row0 + rm).to(tl.int64)[:, None] * s_om + rn[None, :]
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _dgrad_nf4_grouped_bf16(g_ptr, b_ptr, amax_ptr, out_ptr, lut_ptr, t_row0_ptr, t_rows_ptr, t_group_ptr, expert_ids_ptr,
                            K, N, stride_be, stride_bn, stride_ae, stride_an,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """grouped-nf4-gemm's ``_dgrad_nf4_grouped`` (main @9622144) with ONE change: ``g`` and the decoded ``w`` enter tl.dot as
    bf16 (bf16 MMA, fp32 accumulate) instead of fp32. Decode, tiling and accumulation order are otherwise the same."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    row0 = tl.load(t_row0_ptr + pid_m).to(tl.int64)
    rows = tl.load(t_rows_ptr + pid_m)
    grp = tl.load(t_group_ptr + pid_m)
    eid = tl.load(expert_ids_ptr + grp).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = offs_m < rows
    k_mask = offs_k < K
    lut_reg = tl.load(lut_ptr + tl.arange(0, 16))
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    g_base = g_ptr + (row0 + offs_m)[:, None] * N
    g0 = (pid_k * BLOCK_K) // 64
    for n0 in range(0, N, BLOCK_N):
        nn = n0 + offs_n
        n_mask = nn < N
        bytes_ = tl.load(b_ptr + eid * stride_be + nn[:, None] * stride_bn + (offs_k[None, :] // 2),
                         mask=n_mask[:, None] & k_mask[None, :], other=0).to(tl.int32)
        nib = tl.where((offs_k[None, :] % 2) == 0, (bytes_ >> 4) & 0xF, bytes_ & 0xF)
        w = tl.reshape(tl.gather(lut_reg, tl.reshape(nib, [BLOCK_N * BLOCK_K]), 0), [BLOCK_N, BLOCK_K])
        am = tl.load(amax_ptr + eid * stride_ae + nn * stride_an + g0, mask=n_mask, other=0.0)
        w = (w * am[:, None]).to(tl.bfloat16)
        g = tl.load(g_base + nn[None, :], mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.bfloat16)
        acc += tl.dot(g, w)
    out_ptrs = out_ptr + (row0 + offs_m)[:, None] * K + offs_k[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & k_mask[None, :])


def dgrad_bf16(grad_out, B, absmax, sizes, eids_dev, cfg):
    E, N, half = B.shape
    K = half * 2
    bm, bn, bk, warps = cfg
    t_row0, t_rows, t_group = ng.build_group_tiles(sizes, bm, grad_out.device)
    out = torch.empty(grad_out.shape[0], K, dtype=torch.bfloat16, device=grad_out.device)
    _dgrad_nf4_grouped_bf16[(t_row0.numel(), triton.cdiv(K, bk))](
        grad_out, B, absmax, out, ng._lut(grad_out.device), t_row0, t_rows, t_group, eids_dev, K, N,
        B.stride(0), B.stride(1), absmax.stride(0), absmax.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps)
    return out


def decoded_plan(sizes, present, N, K, bm, cap_bytes, dev):
    """Chunks of consecutive groups whose decoded bf16 bytes (n_groups * N * K * 2) fit cap_bytes (None: one chunk). Per
    chunk: (row0, row1, device expert ids, device M-tiles relative to the chunk). Built once per call shape."""
    per = N * K * 2
    step = len(present) if cap_bytes is None else max(1, cap_bytes // per)
    plan, r0 = [], 0
    for i in range(0, len(present), step):
        sz = sizes[i:i + step]
        r1 = r0 + sum(sz)
        tiles = ng.to_device_i32(_tiles(sz, bm), dev)
        plan.append((r0, r1, torch.tensor(present[i:i + step], dtype=torch.int32, device=dev), tiles))
        r0 = r1
    return plan


def _tiles(sizes, bm):
    t_row0, t_rows, t_grp, row = [], [], [], 0
    for g, m in enumerate(sizes):
        left = m
        while left > 0:
            take = min(bm, left)
            t_row0.append(row + (m - left))
            t_rows.append(take)
            t_grp.append(g)
            left -= take
        row += m
    return t_row0, t_rows, t_grp


def decoded_call(x, B, absmax, plan, mode, cfg):
    E, N, half = B.shape
    K = half * 2
    BM, BN, BK, warps, stages = cfg
    n_out = N if mode == "fwd" else K
    out = torch.empty(x.shape[0], n_out, device=x.device, dtype=torch.bfloat16)
    for r0, r1, eids, (row0, rows, grp) in plan:
        W = nr.dequant_groups(B, absmax, eids, N, K)                                   # [g, N, K] bf16, one launch
        if mode == "fwd":
            R, s_wr, s_wj = K, W.stride(2), W.stride(1)
        else:
            R, s_wr, s_wj = N, W.stride(1), W.stride(2)
        xc, oc = x[r0:r1], out[r0:r1]
        _grouped_bf16_gemm[(row0.numel(), triton.cdiv(n_out, BN))](
            xc, W, oc, row0, rows, grp, n_out, R, xc.stride(0), W.stride(0), s_wr, s_wj, oc.stride(0),
            BM=BM, BN=BN, BK=BK, num_warps=warps, num_stages=stages)
        del W
    return out


def ref32_call(x, B, absmax, sizes, present, mode, row_chunk=4096):
    """The fp32 reference every arm is gated against: per group, ``dequant_ref`` (grouped-nf4-gemm's pure-torch decode, bit-equal
    to bitsandbytes' ``dequantize_4bit``) in fp32, times the activations in fp32. The weight is decoded in row chunks to bound
    its memory. Forward: ``x_g @ W_g^T``; dgrad: ``g_g @ W_g``."""
    E, N, half = B.shape
    K = half * 2
    out = torch.zeros(x.shape[0], N if mode == "fwd" else K, device=x.device, dtype=torch.float32)
    r0 = 0
    for e, n in zip(present, sizes):
        xg = x[r0:r0 + n].float()
        for c0 in range(0, N, row_chunk):
            c1 = min(N, c0 + row_chunk)
            w = ng.dequant_ref(B[e, c0:c1].contiguous(), absmax[e, c0:c1].contiguous(), c1 - c0, K)    # [c, K] fp32
            if mode == "fwd":
                out[r0:r0 + n, c0:c1] = xg @ w.t()
            else:
                out[r0:r0 + n] += xg[:, c0:c1] @ w
        r0 += n
    return out


def measure(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    event_ms = e0.elapsed_time(e1) / reps
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    dev_us = sum(getattr(e, "self_device_time_total", 0) or 0 for e in prof.key_averages())
    return {"device_ms": round(dev_us / 1000 / reps, 4), "event_ms": round(event_ms, 4), "peak_mib": round(peak / 2**20, 1)}


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def route_draw(E, k, seq, routing, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if routing == "uniform":
        choice = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(seq)])
    else:                                    # grouped-nf4-gemm bench/host-reuse/moe_host.py's router: hot and cold experts
        logits = torch.randn(seq, E, generator=g) + torch.linspace(1.5, -1.5, E)
        choice = logits.topk(k, dim=1).indices
    counts = torch.bincount(choice.flatten(), minlength=E)
    present = [int(e) for e in torch.nonzero(counts).flatten()]
    return present, [int(counts[e]) for e in present]


def best_of(cfgs, make, ref, ref32, reps, label):
    best, skipped = None, []
    for cfg in cfgs:
        try:
            y = make(cfg)()
            err, err32 = rel(y, ref), rel(y, ref32)
            del y
            m = measure(make(cfg), reps)
        except Exception as e:                        # a tile config that does not fit this card is skipped and named
            skipped.append(f"{label} {cfg}: {type(e).__name__}")
            continue
        m["rel_err"], m["rel_err32"], m["cfg"] = round(err, 5), round(err32, 6), list(cfg)
        if best is None or m["event_ms"] < best["event_ms"]:
            best = m
    return best, skipped


def cell(fam, seq, routing, cap_bytes, reps, dev):
    E, k, H, inter, gated = FAMS[fam]
    present, sizes = route_draw(E, k, seq, routing)
    A = sum(sizes)
    eids_dev = torch.tensor(present, dtype=torch.int32, device=dev)
    out = {"fam": fam, "seq": seq, "routing": routing, "E": E, "k": k, "groups": len(present),
           "rows_mean": round(A / len(present), 1), "rows_max": max(sizes), "rows_min": min(sizes),
           "groups_under_16_rows": sum(1 for s in sizes if s < 16), "proj": {}}
    for pname, N, K in [("gate_up" if gated else "up", (2 * inter if gated else inter), H), ("down", H, inter)]:
        B = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
        absmax = (torch.rand(E, N, K // 64, device=dev) * 0.05 + 0.01).float()
        res = {}
        for mode in ("fwd", "dgrad"):
            x = (torch.randn(A, K if mode == "fwd" else N, device=dev) * 0.5).to(torch.bfloat16)
            r, skipped = {}, []
            ref32 = ref32_call(x, B, absmax, sizes, present, mode)
            if mode == "fwd":
                v1 = lambda: ng.gemm_4bit_grouped(x, B, absmax, sizes, present)                          # noqa: E731
                ref = v1()
                r["v1"] = measure(v1, reps)
                r["v1"]["rel_err32"] = round(rel(ref, ref32), 6)
                v3 = lambda: ng.gemm_4bit_grouped(x, B, absmax, sizes, present, prefill_variant=3)        # noqa: E731
                try:
                    r["v3"] = measure(v3, reps)
                    y3 = v3()
                    r["v3"]["rel_err"], r["v3"]["rel_err32"] = round(rel(y3, ref), 5), round(rel(y3, ref32), 6)
                    del y3
                except Exception as e:                    # recorded as a missing arm, never silently dropped
                    skipped.append(f"v3-fwd: {type(e).__name__}: {str(e)[:160]}")
                dense = lambda: nr.dense_forward(x, B, absmax, sizes, present)                           # noqa: E731
            else:
                v1 = lambda: ng.dgrad_4bit_grouped(x, B, absmax, sizes, present)                         # noqa: E731
                ref = v1()
                r["v1"] = measure(v1, reps)
                r["v1"]["rel_err32"] = round(rel(ref, ref32), 6)
                r["v3"], sk = best_of(DGRAD16_CFGS, lambda c: (lambda: dgrad_bf16(x, B, absmax, sizes, eids_dev, c)),
                                      ref, ref32, reps, "v3-dgrad")
                skipped += sk
                dense = lambda: nr.dense_dgrad(x, B, absmax, sizes, present)                             # noqa: E731
            r["dense"] = measure(dense, reps)
            yd = dense()
            r["dense"]["rel_err"], r["dense"]["rel_err32"] = round(rel(yd, ref), 5), round(rel(yd, ref32), 6)
            del yd
            plans = {c[0]: decoded_plan(sizes, present, N, K, c[0], None, dev) for c in GEMM_CFGS}
            r["decoded"], sk = best_of(GEMM_CFGS, lambda c: (lambda: decoded_call(x, B, absmax, plans[c[0]], mode, c)),
                                       ref, ref32, reps, "decoded")
            skipped += sk
            if r["decoded"]:
                c = tuple(r["decoded"]["cfg"])
                cplan = decoded_plan(sizes, present, N, K, c[0], cap_bytes, dev)
                m = measure(lambda: decoded_call(x, B, absmax, cplan, mode, c), reps)
                yc = decoded_call(x, B, absmax, cplan, mode, c)
                m["rel_err"], m["rel_err32"] = round(rel(yc, ref), 5), round(rel(yc, ref32), 6)
                del yc
                m["cfg"], m["chunks"] = list(c), len(cplan)
                r["decoded_cap"] = m
            if skipped:
                r["skipped"] = skipped
            res[mode] = r
            del ref32
        out["proj"][f"{pname} N={N} K={K}"] = res
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs", default="512,2048")
    ap.add_argument("--fams", default=",".join(FAMS))
    ap.add_argument("--routings", default="uniform,skew")
    ap.add_argument("--cap-mib", type=int, default=256)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    dev = torch.device("cuda")
    rec = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
           "gnf4_file": ng.__file__, "cap_mib": a.cap_mib, "reps": a.reps,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cells": []}
    for routing in a.routings.split(","):
        for seq in [int(s) for s in a.seqs.split(",")]:
            for fam in a.fams.split(","):
                t0 = time.time()
                c = cell(fam, seq, routing, a.cap_mib * 2**20, a.reps, dev)
                c["wall_s"] = round(time.time() - t0, 1)
                rec["cells"].append(c)
                json.dump(rec, open(a.out, "w"), indent=1)
                print(f"{routing:7s} {fam:9s} seq={seq:5d} G={c['groups']:3d} rows~{c['rows_mean']:6.1f} "
                      f"(<16: {c['groups_under_16_rows']}) {c['wall_s']} s", flush=True)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(rec, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
