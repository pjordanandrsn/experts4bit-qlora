# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K36 (kernel/PREREG-k36-int8-head-gemv.md): ``dense_gemv.gemv_i8_bw`` COMPILED, on a CUDA card.

The registered tolerance contract, fixed before any timing (P148's D2 contract, unchanged):
1. **Against cuBLAS, at the served head:** one 151936 x 2048 bf16 head ``W`` and its int8 copy
   ``(q, s) = quantize_rows_int8(W)``, read by 16 rows. For each row, the kernel's relative RMS against the fp64
   reference of its own dequantized weights (``q * s``) is at most 1.05 x cuBLAS's against fp64 on the bf16 head
   (``torch.nn.functional.linear(x, W)`` against ``W @ x``).
2. **One-hot readback, exact:** a one-hot row reads ``bf16(s * q[:, j])`` bit for bit.
3. **Bitwise run to run:** the same call repeats bit for bit (no atomics).
4. **Every output within bf16's rounding** of the fp64 reference of the dequantized weights, at the served head and at
   ragged shapes (the compiled store rounds to nearest).

The int8 grid's effect on a model's quality is not checked here: experts4bit-qlora P150 ruled on it (GO). The interpreter
contract is kernel/test_dense_gemv_i8_interp.py's. Timing is not checked here: P148 timed this kernel body
(experts4bit-qlora ``bench/p148``), and a consumer's served read is the timing of record.
"""
import os

import pytest
import torch

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("the compiled checks; the interpreter contract is test_dense_gemv_i8_interp.py's", allow_module_level=True)

pytest.importorskip("triton", reason="the compiled kernel needs triton")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled kernel needs a CUDA device")

import dense_gemv as D  # noqa: E402

HEAD = (151936, 2048)
RAGGED = [(1031, 2050), (37, 2560), (5, 1000)]
ROWS = 16
CHUNK = 16384          # rows per temporary: the head's fp32 or fp64 copy whole would be 1.2 or 2.5 GB on the card


def _w(n, k, seed):
    """A bf16 ``[n, k]`` weight, ``randn / sqrt(k)``, drawn on the card in row chunks."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.empty(n, k, dtype=torch.bfloat16, device="cuda")
    for i in range(0, n, CHUNK):
        w[i:i + CHUNK] = (torch.randn(min(CHUNK, n - i), k, device="cuda", generator=g) / k ** 0.5).to(torch.bfloat16)
    return w


def _quantize(w):
    """``quantize_rows_int8`` in row chunks: it is row-wise, so only its temporaries' size changes."""
    parts = [D.quantize_rows_int8(w[i:i + CHUNK]) for i in range(0, w.shape[0], CHUNK)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def _fp64(rows_of, n, xs):
    """fp64 ``W @ xs.T`` for rows ``xs [R, K]``, one row chunk of ``W`` (``rows_of(a, b)``, fp64) at a time."""
    xd = xs.double().T
    return torch.cat([rows_of(i, min(i + CHUNK, n)) @ xd for i in range(0, n, CHUNK)])


def _dequant(q, s):
    return lambda a, b: q[a:b].double() * s[a:b].double()[:, None]


def _rel_rms(got, ref):
    got = got.reshape(-1).double()
    return float(((got - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())


@needs_cuda
def test_the_contract_against_cublas_at_the_served_head():
    n, k = HEAD
    w = _w(n, k, 1360)
    q, s = _quantize(w)
    xs = torch.randn(ROWS, k, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1361))
    xs = xs.to(torch.bfloat16)
    ref_w = _fp64(lambda a, b: w[a:b].double(), n, xs)
    ref_q = _fp64(_dequant(q, s), n, xs)
    worst = 0.0
    for r in range(ROWS):
        mine = _rel_rms(D.gemv_i8_bw(xs[r], q, s), ref_q[:, r])
        cublas = _rel_rms(torch.nn.functional.linear(xs[r:r + 1], w)[0], ref_w[:, r])
        worst = max(worst, mine / cublas)
        assert mine <= 1.05 * cublas, (r, mine, cublas)
    print(f"K36_CONTRACT {n}x{k}: worst mine/cuBLAS {worst:.4f} over {ROWS} rows")


@needs_cuda
@pytest.mark.parametrize("n,k,j", [(19, 2048, 1531), (151936, 2048, 2047), (5, 1000, 0)])
def test_a_one_hot_row_reads_one_scaled_column_exactly(n, k, j):
    q, s = _quantize(_w(n, k, 5))
    x = torch.zeros(k, dtype=torch.bfloat16, device="cuda")
    x[j] = 1.0
    assert torch.equal(D.gemv_i8_bw(x, q, s), (q[:, j].float() * s).to(torch.bfloat16))


@needs_cuda
def test_the_same_call_repeats_bit_for_bit():
    q, s = _quantize(_w(*HEAD, 11))
    x = torch.randn(HEAD[1], device="cuda", generator=torch.Generator(device="cuda").manual_seed(18)).to(torch.bfloat16)
    assert torch.equal(D.gemv_i8_bw(x, q, s), D.gemv_i8_bw(x, q, s))


@needs_cuda
@pytest.mark.parametrize("n,k", [HEAD] + RAGGED, ids=lambda v: str(v))
def test_every_output_is_within_bf16_rounding(n, k):
    q, s = _quantize(_w(n, k, n + k))
    x = torch.randn(1, k, device="cuda", generator=torch.Generator(device="cuda").manual_seed(n)).to(torch.bfloat16)
    ref = _fp64(_dequant(q, s), n, x)[:, 0]
    out = D.gemv_i8_bw(x[0], q, s).double()
    assert bool(((out - ref).abs() <= ref.abs() * 2.0 ** -8 + ref.abs().max() * 2.0 ** -20).all())
