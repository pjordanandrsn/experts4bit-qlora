# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K37 (kernel/PREREG-k37-int8-attention-projections.md): ``dense_i8.linear_i8`` COMPILED, on a CUDA card.

The registered tolerance contract, fixed before any timing (P149's box K contract, unchanged):
1. **Against cuBLAS, at Mixtral's two projection shapes** (q and o: 4096 x 4096; k and v: 1024 x 4096), at M = 1 and
   M = 16: one bf16 weight ``W`` (``randn / sqrt(K)``) and its int8 copy ``(q, s) = quantize_rows_int8(W)``, read by four
   activation draws. The kernel's worst relative RMS against the fp64 reference of its own dequantized weights
   (``q * s``) is at most 1.05 x cuBLAS's worst against fp64 on the bf16 weights (``torch.nn.functional.linear(x, W)``).
2. **One-hot readback, exact:** a one-hot row reads ``bf16(s * q[:, j])`` bit for bit, on both paths.
3. **Bitwise run to run:** the same call repeats bit for bit (no atomics).
4. **Every output within bf16's rounding** of the fp64 reference of the dequantized weights, at Mixtral's shapes and at
   ragged ones, for M in 1, 2, 7 and 16 (the compiled store rounds to nearest).

The int8 grid's effect on a model's quality is not checked here: experts4bit-qlora P147 ruled on it (I8ATTN GO). The
interpreter contract is kernel/test_dense_i8_interp.py's. Timing is not checked here: P149's box K timed these kernel
bodies (experts4bit-qlora ``bench/p149``), and a consumer's served read is the timing of record.
"""
import os

import pytest
import torch

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("the compiled checks; the interpreter contract is test_dense_i8_interp.py's", allow_module_level=True)

pytest.importorskip("triton", reason="the compiled kernel needs triton")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled kernel needs a CUDA device")

import dense_i8 as D  # noqa: E402

SHAPES = {"qo": (4096, 4096), "kv": (1024, 4096)}
RAGGED = [(1031, 2050), (37, 2560), (5, 1000)]
CONTRACT_X, CONTRACT_RATIO = 4, 1.05


def _w(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, device="cuda", generator=g) / k ** 0.5).to(torch.bfloat16)


def _rel_rms(got, ref):
    return float(((got.double() - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())


@needs_cuda
@pytest.mark.parametrize("m", [1, 16])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_contract_against_cublas_at_mixtrals_projections(shape, m):
    n, k = SHAPES[shape]
    w = _w(n, k, 149)
    q, s = D.quantize_rows_int8(w)
    wd, qd = w.double(), q.double() * s.double()[:, None]
    g = torch.Generator(device="cuda").manual_seed(1490)
    xs = [torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16) for _ in range(CONTRACT_X)]
    cublas = max(_rel_rms(torch.nn.functional.linear(x, w), x.double() @ wd.T) for x in xs)
    mine = max(_rel_rms(D.linear_i8(x, q, s), x.double() @ qd.T) for x in xs)
    assert mine <= CONTRACT_RATIO * cublas, (shape, m, mine, cublas)


@needs_cuda
@pytest.mark.parametrize("m", [1, 16])
def test_a_one_hot_row_reads_bf16_of_s_times_q_exactly(m):
    n, k, j = 1024, 4096, 2049
    q, s = D.quantize_rows_int8(_w(n, k, 6))
    x = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    x[:, j] = 1.0
    got = D.linear_i8(x, q, s)
    want = (s * q[:, j].float()).to(torch.bfloat16)
    assert all(torch.equal(got[r], want) for r in range(m))


@needs_cuda
@pytest.mark.parametrize("m", [1, 16])
def test_the_same_call_repeats_bit_for_bit(m):
    q, s = D.quantize_rows_int8(_w(4096, 4096, 8))
    x = torch.randn(m, 4096, device="cuda").to(torch.bfloat16)
    assert torch.equal(D.linear_i8(x, q, s), D.linear_i8(x, q, s))


@needs_cuda
@pytest.mark.parametrize("n,k", list(SHAPES.values()) + RAGGED, ids=lambda v: str(v))
@pytest.mark.parametrize("m", [1, 2, 7, 16])
def test_every_output_within_bf16_rounding_of_the_dequantized_reference(n, k, m):
    q, s = D.quantize_rows_int8(_w(n, k, n + k))
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    got = D.linear_i8(x, q, s).double()
    ref = x.double() @ (q.double() * s.double()[:, None]).T
    # one bf16 rounding of the output (2^-9 relative) plus fp32 accumulation over K terms
    assert bool(((got - ref).abs() <= 2.0 ** -8 * ref.abs() + 1e-4).all()), (n, k, m)
