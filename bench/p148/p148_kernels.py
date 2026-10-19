"""p148_kernels.py -- lane P148's prototype decode GEMVs (bench/p148/PREREG-p148.md; e4b#1313). Not a library kernel:
a falsify-first prototype, timed against cuBLAS on one card before any kernel work in grouped-nf4-gemm.

D1, ``gemv_bf16``: ``out[n] = sum_k W[n, k] * x[k]`` for one bf16 row ``x`` and bf16 ``W [N, K]`` (the served
attention projections at T == 1: fused q/k/v 5120 x 2048, o 2048 x 4096). Program ``(pid_n, pid_k)`` owns ``BLOCK_N``
rows over a K span, loads ``[BLOCK_N, KC]`` tiles (16-byte vectors) and accumulates in fp32. K33's shape
(grouped-nf4-gemm ``_gemv_nf4_bw``) without the NF4 decode: ``SPLIT_K`` > 1 stores fp32 partials
``ws[pid_k, n]`` that the host sums in split order, with one bf16 downcast.

D2, ``gemv_i8``: ``out[n] = s[n] * sum_k q[n, k] * x[k]`` for int8 ``q [N, K]`` and an fp32 per-row scale ``s`` (an
int8 lm_head, 151936 x 2048): half the bf16 bytes. ``quantize_rows_int8`` is the symmetric per-row absmax
quantizer (``s = absmax / 127``, round to nearest, clamp to [-127, 127]), as the w8a16 premise's fake-quant used.

Every config is ``(BLOCK_N, KC, num_warps, split_k)`` (``split_k`` is 1 for D2). The bench sweeps the registered grids
below; nothing here chooses.
"""
import torch
import triton
import triton.language as tl

CONFIGS_BF16 = [(bn, kc, w, sk) for bn in (2, 4, 8) for kc in (512, 1024) for w in (4, 8) for sk in (1, 2, 4)]
CONFIGS_I8 = [(bn, kc, w, 1) for bn in (4, 8, 16) for kc in (512, 1024, 2048) for w in (4, 8)]


@triton.jit
def _gemv_bf16_kernel(w_ptr, x_ptr, out_ptr, N, K, KSPAN, stride_wn,
                      BLOCK_N: tl.constexpr, KC: tl.constexpr, SPLITK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_c = tl.arange(0, KC)
    k_lo = pid_k * KSPAN
    k_hi = tl.minimum(k_lo + KSPAN, K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_rows = w_ptr + offs_n.to(tl.int64)[:, None] * stride_wn
    for k0 in range(k_lo, k_hi, KC):
        offs_k = k0 + offs_c
        k_mask = offs_k < k_hi
        x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(w_rows + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    if SPLITK:
        tl.store(out_ptr + pid_k * N + offs_n, acc, mask=n_mask)
    else:
        tl.store(out_ptr + offs_n, acc.to(tl.bfloat16), mask=n_mask)


@triton.jit
def _gemv_i8_kernel(q_ptr, s_ptr, x_ptr, out_ptr, N, K, stride_qn,
                    BLOCK_N: tl.constexpr, KC: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_c = tl.arange(0, KC)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    q_rows = q_ptr + offs_n.to(tl.int64)[:, None] * stride_qn
    for k0 in range(0, K, KC):
        offs_k = k0 + offs_c
        k_mask = offs_k < K
        x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        q = tl.load(q_rows + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0).to(tl.float32)
        acc += tl.sum(q * x[None, :], axis=1)
    s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
    tl.store(out_ptr + offs_n, (acc * s).to(tl.bfloat16), mask=n_mask)


def check_config(cfg, *, int8=False):
    bn, kc, warps, sk = (int(v) for v in cfg)
    ok = (bn > 0 and bn & (bn - 1) == 0 and kc >= 16 and kc & (kc - 1) == 0 and warps in (1, 2, 4, 8, 16) and sk >= 1
          and (sk == 1 or not int8))
    if not ok:
        raise ValueError(f"not a registered config shape: {cfg}")
    return bn, kc, warps, sk


def gemv_bf16(x, w, out, cfg, ws=None):
    """``out [N] (bf16) = w [N, K] (bf16) @ x [K] (bf16)``; ``ws`` (fp32 ``[split_k, N]``) is preallocated by graph
    callers, since a capture must not allocate. Returns ``out``."""
    bn, kc, warps, sk = check_config(cfg)
    n, k = w.shape
    assert x.shape == (k,) and out.shape == (n,) and w.stride(1) == 1
    span = -(-k // sk)
    span = -(-span // kc) * kc                       # each split covers whole KC chunks (only the last split is ragged)
    grid = (triton.cdiv(n, bn), sk)
    if sk == 1:
        _gemv_bf16_kernel[grid](w, x, out, n, k, span, w.stride(0), BLOCK_N=bn, KC=kc, SPLITK=False, num_warps=warps)
        return out
    if ws is None:
        ws = torch.empty(sk, n, dtype=torch.float32, device=w.device)
    _gemv_bf16_kernel[grid](w, x, ws, n, k, span, w.stride(0), BLOCK_N=bn, KC=kc, SPLITK=True, num_warps=warps)
    out.copy_(ws.sum(dim=0))                           # split order, one bf16 downcast (K33's split-K contract)
    return out


def gemv_i8(x, q, s, out, cfg):
    """``out [N] (bf16) = s [N] * (q [N, K] (int8) @ x [K] (bf16))``, fp32 accumulation. Returns ``out``."""
    bn, kc, warps, _sk = check_config(cfg, int8=True)
    n, k = q.shape
    assert x.shape == (k,) and s.shape == (n,) and out.shape == (n,) and q.stride(1) == 1
    _gemv_i8_kernel[(triton.cdiv(n, bn),)](q, s, x, out, n, k, q.stride(0), BLOCK_N=bn, KC=kc, num_warps=warps)
    return out


def quantize_rows_int8(w):
    """``(q int8 [N, K], s fp32 [N])``: symmetric per-row absmax, ``s = absmax / 127`` (1.0 for an all-zero row)."""
    wf = w.float()
    amax = wf.abs().amax(dim=1)
    s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.clamp(torch.round(wf / s[:, None]), -127, 127).to(torch.int8)
    return q, s.contiguous()


def ref_bf16(x, w):
    """fp64 reference of ``w @ x``."""
    return w.double() @ x.double()


def ref_i8(x, q, s):
    """fp64 reference of the int8 GEMV: the dequantized weights ``q * s``, times ``x``."""
    return (q.double() * s.double()[:, None]) @ x.double()
