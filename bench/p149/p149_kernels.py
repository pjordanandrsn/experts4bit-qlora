"""p149_kernels.py -- lane P149's prototype W8A16 kernels for Mixtral-8x7B's attention projections (bench/p149/PREREG-p149.md).
Not a library kernel: a falsify-first prototype, timed against cuBLAS on one card (box K) before any kernel work in
grouped-nf4-gemm. After K_GO the kernels move into grouped-nf4-gemm verbatim, with K's chosen configs.

Both compute ``out[m, n] = s[n] * sum_k q[n, k] * x[m, k]`` for int8 ``q [N, K]`` (row-contiguous), an fp32 per-row scale
``s [N]`` and bf16 activations, with fp32 accumulation and one bf16 downcast:

- ``gemv_i8`` (M == 1): P148's ``gemv_i8`` with K33's split-K (``SPLIT_K`` > 1 stores fp32 partials ``ws[pid_k, n]``
  that the host sums in split order, then scales and downcasts once). Program ``(pid_n, pid_k)`` owns ``BLOCK_N`` rows
  over a K span and loads ``[BLOCK_N, KC]`` int8 tiles.
- ``gemm_i8_small`` (2 <= M <= 16): one ``BLOCK_M = 16`` row tile; each program dequantizes ``[BLOCK_N, BK]`` int8 tiles
  to bf16 in registers (exact: |q| <= 127) and runs ``tl.dot`` with fp32 accumulation, applying ``s`` once at the end.
  Rows past M are masked. ``SPLIT_K`` as above, over ``[split_k, M, N]`` partials.

``quantize_rows_int8`` is the symmetric per-output-row absmax quantizer of ``bench/i8dense/i8_fakequant.py``
(``s = absmax / 127``, round to nearest, clamp to [-127, 127], a zero row keeps scale 1): its codes and scales are the
fake-quant's, so ``(q * s).to(bf16)`` equals the fake-quant's round trip bit for bit (P147's I8ATTN weights). The kernels
dequantize in fp32, not through bf16, which is the less lossy direction.

Every config is ``(BLOCK_N, KC_or_BK, num_warps, split_k)``. The bench sweeps the registered grids below; nothing here
chooses.
"""
import os

import torch
import triton
import triton.language as tl

# Triton's interpreter runs tl.dot on bf16 operands' raw bits, so under TRITON_INTERPRET=1 the small-M kernel's dot runs in
# fp32 (IEEE) instead; compiled code always runs the bf16 dot. CI checks the logic (masks, split-K, the scale); the
# compiled bf16 dot is checked against fp64 on the card (the A2000 shakedown and box K run the suite compiled).
DOT_F32 = os.environ.get("TRITON_INTERPRET") == "1"

CONFIGS_GEMV = [(bn, kc, w, sk) for bn in (4, 8, 16) for kc in (512, 1024, 2048) for w in (4, 8) for sk in (1, 2, 4)]
CONFIGS_SMALL = [(bn, bk, w, sk) for bn in (32, 64, 128) for bk in (64, 128, 256) for w in (4, 8) for sk in (1, 2, 4, 8)]
BLOCK_M = 16
QMAX = 127


@triton.jit
def _gemv_i8_kernel(q_ptr, s_ptr, x_ptr, out_ptr, N, K, KSPAN, stride_qn,
                    BLOCK_N: tl.constexpr, KC: tl.constexpr, SPLITK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_c = tl.arange(0, KC)
    k_lo = pid_k * KSPAN
    k_hi = tl.minimum(k_lo + KSPAN, K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    q_rows = q_ptr + offs_n.to(tl.int64)[:, None] * stride_qn
    for k0 in range(k_lo, k_hi, KC):
        offs_k = k0 + offs_c
        k_mask = offs_k < k_hi
        x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        q = tl.load(q_rows + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0).to(tl.float32)
        acc += tl.sum(q * x[None, :], axis=1)
    if SPLITK:
        tl.store(out_ptr + pid_k * N + offs_n, acc, mask=n_mask)
    else:
        s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
        tl.store(out_ptr + offs_n, (acc * s).to(tl.bfloat16), mask=n_mask)


@triton.jit
def _gemm_i8_small_kernel(q_ptr, s_ptr, x_ptr, out_ptr, M, N, K, KSPAN, stride_qn, stride_xm, stride_om,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BK: tl.constexpr, SPLITK: tl.constexpr,
                          DOT_F32: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    offs_c = tl.arange(0, BK)
    k_lo = pid_k * KSPAN
    k_hi = tl.minimum(k_lo + KSPAN, K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    q_rows = q_ptr + offs_n.to(tl.int64)[:, None] * stride_qn
    x_rows = x_ptr + offs_m.to(tl.int64)[:, None] * stride_xm
    for k0 in range(k_lo, k_hi, BK):
        offs_k = k0 + offs_c
        k_mask = offs_k < k_hi
        x = tl.load(x_rows + offs_k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        q = tl.load(q_rows + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0).to(tl.float32)
        if DOT_F32:
            acc = tl.dot(x.to(tl.float32), tl.trans(q), acc, input_precision="ieee")
        else:
            acc = tl.dot(x, tl.trans(q.to(tl.bfloat16)), acc)   # the codes are exact in bf16 (|q| <= 127)
    if SPLITK:
        ws = out_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
        tl.store(ws, acc, mask=m_mask[:, None] & n_mask[None, :])
    else:
        s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
        o = out_ptr + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :]
        tl.store(o, (acc * s[None, :]).to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def check_config(cfg, *, small=False):
    bn, kc, warps, sk = (int(v) for v in cfg)
    ok = (bn > 0 and bn & (bn - 1) == 0 and kc >= 16 and kc & (kc - 1) == 0 and warps in (1, 2, 4, 8, 16) and sk >= 1
          and (not small or bn >= 16))
    if not ok:
        raise ValueError(f"not a registered config shape: {cfg}")
    return bn, kc, warps, sk


def _span(k, sk, chunk):
    span = -(-k // sk)
    return -(-span // chunk) * chunk                 # each split covers whole chunks (only the last split is ragged)


def gemv_i8(x, q, s, out, cfg, ws=None):
    """``out [N] (bf16) = s [N] * (q [N, K] (int8) @ x [K] (bf16))``, fp32 accumulation. ``ws`` (fp32
    ``[split_k, N]``) is preallocated by graph callers, since a capture must not allocate. Returns ``out``."""
    bn, kc, warps, sk = check_config(cfg)
    n, k = q.shape
    assert x.shape == (k,) and s.shape == (n,) and out.shape == (n,) and q.stride(1) == 1 and q.dtype == torch.int8
    span = _span(k, sk, kc)
    grid = (triton.cdiv(n, bn), sk)
    if sk == 1:
        _gemv_i8_kernel[grid](q, s, x, out, n, k, span, q.stride(0), BLOCK_N=bn, KC=kc, SPLITK=False, num_warps=warps)
        return out
    if ws is None:
        ws = torch.empty(sk, n, dtype=torch.float32, device=q.device)
    _gemv_i8_kernel[grid](q, s, x, ws, n, k, span, q.stride(0), BLOCK_N=bn, KC=kc, SPLITK=True, num_warps=warps)
    out.copy_(ws.sum(dim=0) * s)                       # split order, then the scale, one bf16 downcast
    return out


def gemm_i8_small(x, q, s, out, cfg, ws=None):
    """``out [M, N] (bf16) = (x [M, K] (bf16) @ q [N, K]^T (int8)) * s [N]`` for 1 <= M <= 16, fp32 accumulation.
    ``ws`` (fp32 ``[split_k, M, N]``) is preallocated by graph callers. Returns ``out``."""
    bn, bk, warps, sk = check_config(cfg, small=True)
    n, k = q.shape
    m = x.shape[0]
    assert 1 <= m <= BLOCK_M and x.shape == (m, k) and x.stride(1) == 1 and out.shape == (m, n) and out.stride(1) == 1
    assert s.shape == (n,) and q.stride(1) == 1 and q.dtype == torch.int8
    span = _span(k, sk, bk)
    grid = (triton.cdiv(n, bn), sk)
    if sk == 1:
        _gemm_i8_small_kernel[grid](q, s, x, out, m, n, k, span, q.stride(0), x.stride(0), out.stride(0),
                                    BLOCK_M=BLOCK_M, BLOCK_N=bn, BK=bk, SPLITK=False, DOT_F32=DOT_F32, num_warps=warps)
        return out
    if ws is None:
        ws = torch.empty(sk, m, n, dtype=torch.float32, device=q.device)
    _gemm_i8_small_kernel[grid](q, s, x, ws, m, n, k, span, q.stride(0), x.stride(0), out.stride(0),
                                BLOCK_M=BLOCK_M, BLOCK_N=bn, BK=bk, SPLITK=True, DOT_F32=DOT_F32, num_warps=warps)
    out.copy_(ws.sum(dim=0) * s[None, :])
    return out


def quantize_rows_int8(w):
    """``(q int8 [N, K], s fp32 [N])``: the fake-quant's symmetric per-row absmax (``s = absmax / 127``, round to nearest,
    clamp to [-127, 127], 1.0 for an all-zero row)."""
    wf = w.detach().float()
    amax = wf.abs().amax(dim=1)
    s = torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))
    q = torch.clamp(torch.round(wf / s[:, None]), -QMAX, QMAX).to(torch.int8)
    return q.contiguous(), s.contiguous()


def ref_i8(x, q, s):
    """fp64 reference of the int8 product: the dequantized weights ``q * s`` times ``x`` (``x`` is ``[K]`` or ``[M, K]``)."""
    deq = q.double() * s.double()[:, None]
    return deq @ x.double() if x.dim() == 1 else x.double() @ deq.T


def ref_bf16(x, w):
    """fp64 reference of cuBLAS's bf16 product on the original weights (the contract's other side)."""
    return w.double() @ x.double() if x.dim() == 1 else x.double() @ w.double().T
