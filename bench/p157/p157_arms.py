# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P157's ablation kernels (bench/p157/PREREG-p157.md): K25 with one stage knocked out per arm. BENCH-ONLY.

``_k25_ablate`` is grouped-nf4-gemm's ``nf4_smallm._gemm_nf4_grouped_smallm`` (K25, at the pin the PREREG names),
copied verbatim except:
- the K loop is ``tl.range(0, NKC, num_stages=3, loop_unroll_factor=1)``: Triton 3.4.0 multi-buffers loads only when
  they feed a ``tl.dot``, so every arm names its pipelining explicitly, and an arm without the dot keeps the loads the
  served kernel pipelines; and no arm's loop is unrolled away (a light body over 12 iterations would be), so every arm
  has one counted loop;
- after K25's own parameters, two constexprs, ``ARM`` and ``RESID``, and two runtime arguments, ``emask`` and ``nrot``
  (used only under ``RESID``). The constexprs' branches are pruned at compile time, so each (ARM, RESID) compiles as a
  kernel of its own, and ``ARM == 0`` without ``RESID`` is K25 with only the explicit ``tl.range`` (the placebo, A0p);
- arms without the select tree sum its 16 codebook values before the K loop and add the sum to the accumulator after
  it: they load the codebook as K25 does (every arm's loads outside the K loop are K25's), without holding sixteen
  registers through the loop that only the tree needs.

| arm | ARM | RESID: emask, nrot | what is removed | what stays |
|---|---|---|---|---|
| A0p | 0 | no | nothing (the placebo) | everything |
| A1 | 1 | no | the select tree: the nibble is converted to fp32 | the loads, nibble extraction, absmax, staging, MMA |
| A2b | 2 | no | the MMA and its operand staging: per-thread fp32 folds | the loads, decode, absmax |
| A12 | 3 | no | both A1's and A2b's | the loads, nibble extraction, absmax |
| A3 | 4 | 127, 1 | everything but the loads: per-thread XOR/add folds | the loads, A0's bytes |
| AL | 4 | 0, 1 | A3's work on expert 0's slab only (~2 % of A0's DRAM bytes) | A3's loads, from L2 after the first touch |
| AC | 0 | 0, 1 | DRAM bytes: every program reads expert ``eid & 0`` (one slab a call) | the whole K loop |
| AC4 | 0 | 3, 1 | DRAM bytes: at most 4 slabs a call | the whole K loop |
| AC16 | 0 | 15, 1 | DRAM bytes: at most 16 slabs a call | the whole K loop |
| ACX | 0 | 127, 1 | nothing: ``eid & 127`` is the identity on [0, 128), so A0's own bytes | the whole K loop |
| AE | 9 | no | everything: each program returns at once | the grid and its launches |

``RESID`` is a constexpr; ``emask`` and ``nrot`` are runtime arguments compiled with ``do_not_specialize``. So the
ladder (AC, AC4, AC16, ACX) is ONE cubin and A3 with AL is another: within each, arms differ by data only, and every
difference r is built from is taken inside one cubin (full = ACX - AE, onchip = AC - AE, mem_dram = A3 - AL). With
``nrot`` = 1 a program reads its weights from n-tile (pid_n + g) mod NT, so programs dispatched together (consecutive
g, one pid_n) read different lines even when the mask gives them one expert: without it, ~57 programs read the same
32 rows at once, and the lab A2000 timed one slab a call 7 % SLOWER than all of them. Every rung pays a compulsory
first touch per call (each layer's slabs are evicted by the other 47 layers' traffic before it comes round again), so
a rung moves (distinct masked slabs) x (slab bytes) of DRAM a call: a dose series, from ~2 % of A0's bytes (emask 0)
to all of them (emask 127). ``ACX0`` (``nrot`` = 0) is the ladder's cubin unrotated: never timed, its outputs must be
K25's bit for bit, which checks the cubin's math.

A2a, the PTX-level MMA knock-out, is NOT BUILT (pre-stated best effort). A faithful replacement keeps every MMA operand
alive: 48 distinct operand registers an iteration (32 A from scalar ``cvt.f32.bf16``, 16 B from ``ld.shared.v4``) fold
into one accumulator in at least ceil(48 / 2) = 24 three-input ``lop3`` against 8 HMMA. It trades tensor-pipe time for
extra ALU issues, so its delta mixes MMA removal with added issue pressure and cannot be read as either.

Every arm stores its accumulator through K25's own masked store, and every accumulator depends on all of its arm's
operands, so no work can be eliminated as dead. The outputs are WRONG by design except A0p's (bit-identical to K25's,
checked on every rep) and ACX0's. None is ever compared to a reference or reachable from the library. ``launch``
refuses unless ``P157_ABLATION=1`` (set by ``p157_bench.py``).
"""
from __future__ import annotations

import os

ARMS = {                               # arm: (ARM, RESID, emask, nrot)
    "A0p": (0, False, 127, 0), "A1": (1, False, 127, 0), "A2b": (2, False, 127, 0), "A12": (3, False, 127, 0),
    "A3": (4, True, 127, 1), "AL": (4, True, 0, 1),
    "AC": (0, True, 0, 1), "AC4": (0, True, 3, 1), "AC16": (0, True, 15, 1), "ACX": (0, True, 127, 1),
    "ACX0": (0, True, 127, 0),         # the ladder's cubin unrotated: never timed, its outputs must be K25's
    "AE": (9, False, 127, 0)}
LADDER = ("AC", "AC4", "AC16", "ACX")
SHARES = {"AC4": "AC", "AC16": "AC", "ACX": "AC", "ACX0": "AC", "AL": "A3"}   # one cubin each, by construction

try:                                   # grouped-nf4-gemm at the pin: the box and the shakedown have it, CI need not
    import torch
    from _triton_shim import triton, tl
    from nf4_grouped import _TL_INTERLEAVE, _lut
    from nf4_smallm import _prmt_decode
except ImportError:                    # pragma: no cover (CPU CI reads this file as text)
    triton = None


if triton is not None:
    @triton.jit(do_not_specialize=["emask", "nrot"])
    def _k25_ablate(x_ptr, ord_ptr, w_ptr, am_ptr, lut_ptr, row0_ptr, rows_ptr, grp_ptr, out_ptr,
                    N, stride_we, stride_wn, stride_ae, stride_an,
                    K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KC: tl.constexpr,
                    LUT: tl.constexpr, GATHER: tl.constexpr, DOT_BF16: tl.constexpr, EVEN_K: tl.constexpr,
                    sct_ptr=None, SCATTER: tl.constexpr = False, GDIV: tl.constexpr = 1, ARM: tl.constexpr = 0,
                    RESID: tl.constexpr = False, emask=127, nrot=0):
        """K25's grid, tile table and call form (see the module docstring for what each ARM removes)."""
        g = tl.program_id(0)
        pid_n = tl.program_id(1)
        if ARM == 9:                                       # AE: the grid and its launches, no work
            return
        rows = tl.load(rows_ptr + g)
        if rows == 0:
            return
        row0 = tl.load(row0_ptr + g).to(tl.int64)
        eid = tl.load(grp_ptr + g).to(tl.int64)
        if RESID:                                          # the ladder and AL: a runtime mask on the expert id, so
            eid = eid & emask                              # the rungs share one cubin and differ by data only
        offs_m = tl.arange(0, BLOCK_M)
        m_mask = offs_m < rows
        if GATHER:
            src = tl.load(ord_ptr + row0 + offs_m, mask=m_mask, other=0).to(tl.int64)
            if GDIV != 1:
                src = src // GDIV
        else:
            src = row0 + offs_m
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        if RESID:                                          # weight rows from n-tile (pid_n + g * nrot) mod NT, so the
            nt = tl.cdiv(N, BLOCK_N)                       # programs dispatched together read different lines even
            offs_w = ((pid_n + g * nrot) % nt) * BLOCK_N + tl.arange(0, BLOCK_N)   # when they share an expert
        else:
            offs_w = offs_n
        w_mask = offs_w < N
        KB: tl.constexpr = K // 64
        NKC: tl.constexpr = (K + KC - 1) // KC
        NSC: tl.constexpr = KC // 64
        offs_kc = tl.arange(0, KC)
        offs_kh = tl.arange(0, KC // 2)
        offs_sc = tl.arange(0, NSC)
        wbase = w_ptr + eid * stride_we + offs_w.to(tl.int64)[:, None] * stride_wn
        abase = am_ptr + eid * stride_ae + offs_w.to(tl.int64)[:, None] * stride_an
        if LUT == 3:                                       # the 16 codebook values, once per program (the select tree)
            c0 = tl.load(lut_ptr + 0)
            c1 = tl.load(lut_ptr + 1)
            c2 = tl.load(lut_ptr + 2)
            c3 = tl.load(lut_ptr + 3)
            c4 = tl.load(lut_ptr + 4)
            c5 = tl.load(lut_ptr + 5)
            c6 = tl.load(lut_ptr + 6)
            c7 = tl.load(lut_ptr + 7)
            c8 = tl.load(lut_ptr + 8)
            c9 = tl.load(lut_ptr + 9)
            c10 = tl.load(lut_ptr + 10)
            c11 = tl.load(lut_ptr + 11)
            c12 = tl.load(lut_ptr + 12)
            c13 = tl.load(lut_ptr + 13)
            c14 = tl.load(lut_ptr + 14)
            c15 = tl.load(lut_ptr + 15)
            if ARM == 1 or ARM == 3 or ARM == 4:           # no select tree, but its codebook is loaded as K25 loads it,
                csum = (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15)  # summed once
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if ARM == 2 or ARM == 3:                           # A2b / A12: per-thread folds stand in for the MMA
            fw = tl.zeros((BLOCK_N, KC), dtype=tl.float32)
            fa = tl.zeros((BLOCK_M, KC), dtype=tl.float32)
        if ARM == 4:                                       # A3: per-thread folds of every loaded word
            xw = tl.zeros((BLOCK_N, KC // 2), dtype=tl.int32)
            xm = tl.zeros((BLOCK_N, NSC), dtype=tl.float32)
            xa = tl.zeros((BLOCK_M, KC), dtype=tl.float32)
        for c in tl.range(0, NKC, num_stages=3, loop_unroll_factor=1):
            k0 = c * KC
            if EVEN_K:
                a = tl.load(x_ptr + src[:, None] * K + k0 + offs_kc[None, :], mask=m_mask[:, None], other=0.0)
                wb = tl.load(wbase + (k0 // 2) + offs_kh[None, :], mask=w_mask[:, None], other=0).to(tl.int32)
                am = tl.load(abase + (k0 // 64) + offs_sc[None, :], mask=w_mask[:, None], other=0.0).to(tl.float32)
            else:
                a = tl.load(x_ptr + src[:, None] * K + k0 + offs_kc[None, :],
                            mask=m_mask[:, None] & ((k0 + offs_kc) < K)[None, :], other=0.0)
                wb = tl.load(wbase + (k0 // 2) + offs_kh[None, :],
                             mask=w_mask[:, None] & (((k0 // 2) + offs_kh) < (K // 2))[None, :], other=0).to(tl.int32)
                am = tl.load(abase + (k0 // 64) + offs_sc[None, :],
                             mask=w_mask[:, None] & (((k0 // 64) + offs_sc) < KB)[None, :], other=0.0).to(tl.float32)
            if DOT_BF16:
                a = a.to(tl.bfloat16)
            else:
                a = a.to(tl.float32)
            if ARM == 4:                                   # A3: every loaded word folded, nothing else
                xw = xw ^ wb
                xm = xm + am
                xa = xa + a
            else:
                if LUT == 2:
                    pv = tl.load(lut_ptr + wb)                                       # [BN, KC/2] int64: both values
                    ev = pv.to(tl.int32).to(tl.float32, bitcast=True)                # low word = element 2j
                    od = (pv >> 32).to(tl.int32).to(tl.float32, bitcast=True)        # high word = element 2j + 1
                    w = _TL_INTERLEAVE(ev, od)                                       # [BN, KC] in k order
                elif LUT == 3:
                    if ARM == 1 or ARM == 3:               # A1 / A12: the select tree removed
                        nib = _TL_INTERLEAVE((wb >> 4) & 0xF, wb & 0xF)              # element 2j = HIGH nibble
                        w = nib.to(tl.float32)
                    else:
                        # a 4-level select tree on the nibble's bits over the 16 fp32 codebook values: the same
                        # weights as a lookup, with no load per element (lane K26 found the lookup holds K25 back)
                        nib = _TL_INTERLEAVE((wb >> 4) & 0xF, wb & 0xF)              # element 2j = HIGH nibble
                        b0 = (nib & 1) != 0
                        b1 = (nib & 2) != 0
                        b2 = (nib & 4) != 0
                        b3 = (nib & 8) != 0
                        s0 = tl.where(b0, c1, c0)
                        s1 = tl.where(b0, c3, c2)
                        s2 = tl.where(b0, c5, c4)
                        s3 = tl.where(b0, c7, c6)
                        s4 = tl.where(b0, c9, c8)
                        s5 = tl.where(b0, c11, c10)
                        s6 = tl.where(b0, c13, c12)
                        s7 = tl.where(b0, c15, c14)
                        u0 = tl.where(b1, s1, s0)
                        u1 = tl.where(b1, s3, s2)
                        u2 = tl.where(b1, s5, s4)
                        u3 = tl.where(b1, s7, s6)
                        v0 = tl.where(b2, u1, u0)
                        v1 = tl.where(b2, u3, u2)
                        w = tl.where(b3, v1, v0)
                elif LUT == 4:
                    w = _prmt_decode(wb.to(tl.uint8))
                else:
                    nib = _TL_INTERLEAVE((wb >> 4) & 0xF, wb & 0xF)                  # element 2j = HIGH nibble
                    w = tl.load(lut_ptr + nib)
                w3 = tl.reshape(w, (BLOCK_N, NSC, 64)) * am[:, :, None]                # per-64 absmax, fp32
                wsc = tl.reshape(w3, (BLOCK_N, KC))
                if DOT_BF16:
                    wsc = wsc.to(tl.bfloat16)
                if ARM == 2 or ARM == 3:                   # A2b / A12: no MMA, no operand staging
                    fw = fw + wsc
                    fa = fa + a
                else:
                    acc += tl.dot(a, tl.trans(wsc), out_dtype=tl.float32)
        if LUT == 3:
            if ARM == 1 or ARM == 3 or ARM == 4:           # here, so one register, not sixteen, lives through the loop
                acc += csum
        if ARM == 2 or ARM == 3:
            acc += tl.sum(fw, axis=1)[None, :] + tl.sum(fa, axis=1)[:, None]
        if ARM == 4:
            acc += (tl.sum(xw.to(tl.float32), axis=1)[None, :] + tl.sum(xm, axis=1)[None, :]
                    + tl.sum(xa, axis=1)[:, None])
        if SCATTER:
            dst = tl.load(sct_ptr + row0 + offs_m, mask=m_mask, other=0).to(tl.int64)
        else:
            dst = row0 + offs_m
        ooff = dst[:, None] * N + offs_n[None, :]
        tl.store(out_ptr + ooff, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def launch(arm: str, x, packed, absmax, t_row0, t_rows, t_group, order=None, *, block_n=32, kc=64, warps=4,
           stages=3, scatter=None, gather_div=1, pad=0):
    """One ablation arm, K25's served call form at today's plan (``lut="tree"``, TF32: ``dot_bf16=False``). Refused
    outside the bench.

    ``pad`` > 0 pins occupancy: the arm's compiled kernel launches with ``pad`` bytes of dynamic shared memory over its
    own, so a lighter arm holds A0's CTAs per SM (the bench sizes ``pad`` with the driver's occupancy query). It is set
    on the compiled kernel's launch metadata, which is what a CUDA graph captures as the kernel node's
    ``sharedMemBytes``; one cubin takes one pad, and a second, different pad is refused."""
    if os.environ.get("P157_ABLATION") != "1":
        raise RuntimeError("P157's ablation kernels compute wrong outputs by design; they run only under p157_bench.py "
                           "(P157_ABLATION=1)")
    if triton is None:
        raise RuntimeError("grouped-nf4-gemm is not importable")
    code, resid, emask, nrot = ARMS[arm]
    R, K = x.shape
    if gather_div != 1:
        R = order.numel()
    E, N, _kh = packed.shape
    out = torch.empty(R, N, dtype=torch.bfloat16, device=x.device)
    extra = {} if scatter is None else {"sct_ptr": scatter, "SCATTER": True}
    if gather_div != 1:
        extra["GDIV"] = int(gather_div)
    gather = order is not None
    grid = (t_row0.numel(), triton.cdiv(N, block_n))
    args = (x.contiguous(), order if gather else t_row0, packed, absmax, _lut(x.device), t_row0, t_rows, t_group, out,
            N, packed.stride(0), packed.stride(1), absmax.stride(0), absmax.stride(1))
    kw = dict(K=K, BLOCK_M=16, BLOCK_N=block_n, KC=kc, LUT=3, GATHER=bool(gather), DOT_BF16=False,
              EVEN_K=K % kc == 0, ARM=code, RESID=resid, emask=emask, nrot=nrot, num_warps=warps, num_stages=stages,
              **extra)
    if pad:
        set_pad(_k25_ablate.warmup(*args, grid=grid, **kw), pad)
    _k25_ablate[grid](*args, **kw)
    return out


def set_pad(kernel, pad: int):
    """Launch ``kernel`` (a Triton ``CompiledKernel``) with ``pad`` extra bytes of dynamic shared memory: Triton 3.4
    launches with ``packed_metadata = (num_warps, num_ctas, shared, *cluster_dims)``, built once from the compiled
    metadata, and every later launch of the cached kernel reads it. Returns the launch's shared bytes."""
    want = int(kernel.metadata.shared) + int(pad)
    pm = tuple(kernel.packed_metadata)
    if pm[2] == want:
        return want
    if pm[2] != int(kernel.metadata.shared):
        raise RuntimeError(f"{kernel.name}: already padded to {pm[2]} B, not {want} B (one cubin takes one pad)")
    if want > 48 * 1024:
        raise RuntimeError(f"{kernel.name}: a pad past 48 KiB would need the opt-in attribute; refused")
    kernel.packed_metadata = (pm[0], pm[1], want, *pm[3:])
    return want
