# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""K35 (kernel/PREREG-k35-dense-bf16-decode-gemv.md): ``dense_gemv.gemv_bf16_bw`` COMPILED, on a CUDA card.

The registered tolerance contract, fixed before any timing:
1. **Against cuBLAS:** relative RMS against an fp64 reference at most 1.05 x cuBLAS's own
   (``torch.nn.functional.linear`` on the same bf16 row and weight). The served shapes are checked over 48 independent
   weights each (one model's layers' worth), a ragged shape over 8.
2. **One-hot readback, exact:** a one-hot row reads one weight column bit for bit.
3. **Bitwise run to run:** the same call repeats bit for bit (no atomics).
4. **Every output within bf16's rounding** of the fp64 reference (the compiled store rounds to nearest).

The interpreter contract is kernel/test_dense_gemv_interp.py's. Timing is not checked here: P148 timed this kernel body
(experts4bit-qlora ``bench/p148``), and a consumer's served read is the timing of record.
"""
import os

import pytest
import torch

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("the compiled checks; the interpreter contract is test_dense_gemv_interp.py's", allow_module_level=True)

pytest.importorskip("triton", reason="the compiled kernel needs triton")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled kernel needs a CUDA device")

import dense_gemv as D  # noqa: E402

SERVED = [(5120, 2048), (2048, 4096)]
RAGGED = (1031, 2050)


def _w(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, k, generator=g) / k ** 0.5).to(torch.bfloat16).cuda()


def _x(k, seed):
    g = torch.Generator().manual_seed(seed + 7)
    return torch.randn(k, generator=g).to(torch.bfloat16).cuda()


def _ref(x, w):
    return w.double().cpu() @ x.reshape(-1).double().cpu()


def _rel_rms(got, ref):
    got = got.reshape(-1).double().cpu()
    return float(((got - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())


@needs_cuda
@pytest.mark.parametrize("n,k", SERVED + [RAGGED], ids=lambda v: str(v))
def test_the_contract_against_cublas(n, k):
    seeds = range(48 if (n, k) in SERVED else 8)
    worst = 0.0
    for s in seeds:
        w, x = _w(n, k, 1000 + s), _x(k, 1000 + s)
        ref = _ref(x, w)
        mine = _rel_rms(D.gemv_bf16_bw(x, w), ref)
        cublas = _rel_rms(torch.nn.functional.linear(x.reshape(1, k), w), ref)
        worst = max(worst, mine / cublas)
        assert mine <= 1.05 * cublas, (s, mine, cublas)
    print(f"K35_CONTRACT {n}x{k}: worst mine/cuBLAS {worst:.4f} over {len(seeds)} weights")


@needs_cuda
@pytest.mark.parametrize("n,k,j", [(19, 2048, 1531), (2048, 4096, 4095), (5120, 2048, 0)])
def test_a_one_hot_row_reads_one_weight_column_exactly(n, k, j):
    w = _w(n, k, 5)
    x = torch.zeros(k, dtype=torch.bfloat16, device="cuda")
    x[j] = 1.0
    assert torch.equal(D.gemv_bf16_bw(x, w), w[:, j])


@needs_cuda
@pytest.mark.parametrize("n,k", SERVED, ids=lambda v: str(v))
def test_the_same_call_repeats_bit_for_bit(n, k):
    w, x = _w(n, k, 11), _x(k, 11)
    assert torch.equal(D.gemv_bf16_bw(x, w), D.gemv_bf16_bw(x, w))


@needs_cuda
@pytest.mark.parametrize("n,k", SERVED + [RAGGED, (5, 1000)], ids=lambda v: str(v))
def test_every_output_is_within_bf16_rounding(n, k):
    w, x = _w(n, k, n + k), _x(k, n + k)
    ref = _ref(x, w)
    out = D.gemv_bf16_bw(x, w).double().cpu()
    assert bool(((out - ref).abs() <= ref.abs() * 2.0 ** -8 + ref.abs().max() * 2.0 ** -20).all())
