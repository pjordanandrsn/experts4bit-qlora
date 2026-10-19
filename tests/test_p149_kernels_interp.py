# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P149's prototype W8A16 kernels (bench/p149/p149_kernels.py), correctness only: under Triton's interpreter on CPU in
CI, and compiled on the card as the boxes' premise (the runner sets TRITON_INTERPRET=0 explicitly). Every registered config
at ragged shapes against an fp64 reference of the dequantized weights; one config at each of Mixtral's two shapes; a
one-hot activation reads q * s exactly; split-K matches the unsplit kernel; the quantizer's codes and scales are
bench/i8dense/i8_fakequant.py's, so their bf16 product is P147's I8ATTN round trip bit for bit."""
import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
# triton decorates its own @jit library at its first import, so the interpreter env must precede that import. In CI's
# full suite another module imports triton first; this file then skips itself here, and tests/test_p149.py runs it in a
# fresh interpreted process (as the boxes run it, compiled, in their own process).
if os.environ["TRITON_INTERPRET"] == "1" and type(triton.language.zeros).__name__ != "InterpretedFunction":
    pytest.skip("triton was imported compiled earlier in this process; tests/test_p149.py runs this file in a fresh "
                "interpreted process", allow_module_level=True)

HERE = pathlib.Path(__file__).resolve().parent
for cand in (HERE, HERE.parent / "bench" / "p149"):
    if (cand / "p149_kernels.py").exists():
        sys.path.insert(0, str(cand))
        break
for cand in (HERE, HERE.parent / "bench" / "i8dense"):
    if (cand / "i8_fakequant.py").exists():
        sys.path.insert(0, str(cand))
        break
import i8_fakequant  # noqa: E402
import p149_kernels as K  # noqa: E402

DEV = "cuda" if os.environ.get("TRITON_INTERPRET") == "0" and torch.cuda.is_available() else "cpu"
TOL = 4e-3                                             # bf16 output rounding (2^-9) with fp32 accumulation


def _w(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, k, generator=g) / k ** 0.5).to(torch.bfloat16).to(DEV)


def _x(shape, seed):
    g = torch.Generator().manual_seed(seed + 1)
    return torch.randn(*shape, generator=g).to(torch.bfloat16).to(DEV)


def _rel_rms(got, ref):
    return float(((got.double().cpu() - ref.cpu()) ** 2).mean().sqrt() / (ref.cpu() ** 2).mean().sqrt())


def _bf16_conv(f32):
    """The fp32 -> bf16 conversion each mode uses: the interpreter truncates, compiled code (cvt.rn) rounds to even."""
    f32 = f32.cpu()
    if os.environ.get("TRITON_INTERPRET") == "1":
        return (f32.view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)
    return f32.to(torch.bfloat16)


@pytest.mark.parametrize("cfg", K.CONFIGS_GEMV, ids=lambda c: "gemv-%d-%d-%d-%d" % c)
def test_every_gemv_config_reads_its_dequantized_weights_at_a_ragged_shape(cfg):
    n, k = 45, 2048 + 512 + 64                         # n not a BLOCK_N multiple; k not a KC multiple
    q, s = K.quantize_rows_int8(_w(n, k, 1))
    x = _x((k,), 1)
    out = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    K.gemv_i8(x, q, s, out, cfg)
    assert _rel_rms(out, K.ref_i8(x, q, s)) < TOL


@pytest.mark.parametrize("cfg", K.CONFIGS_SMALL, ids=lambda c: "small-%d-%d-%d-%d" % c)
def test_every_small_m_config_reads_its_dequantized_weights_at_a_ragged_shape(cfg):
    m, n, k = 5, 70, 1024 + 64                         # m < 16 (masked rows); n, k ragged
    q, s = K.quantize_rows_int8(_w(n, k, 2))
    x = _x((m, k), 2)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=DEV)
    K.gemm_i8_small(x, q, s, out, cfg)
    assert _rel_rms(out, K.ref_i8(x, q, s)) < TOL


@pytest.mark.parametrize("m", [1, 2, 16])
def test_the_small_m_kernel_covers_its_row_range(m):
    n, k = 96, 2048
    q, s = K.quantize_rows_int8(_w(n, k, 3))
    x = _x((m, k), 3)
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=DEV)
    K.gemm_i8_small(x, q, s, out, (32, 128, 4, 2))
    assert torch.isfinite(out.float()).all() and _rel_rms(out, K.ref_i8(x, q, s)) < TOL


@pytest.mark.parametrize("n,k", [(4096, 4096), (1024, 4096)], ids=["q_o", "k_v"])
def test_one_config_each_at_mixtrals_shapes(n, k):
    q, s = K.quantize_rows_int8(_w(n, k, 4))
    x1, x16 = _x((k,), 4), _x((16, k), 4)
    o1 = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    o16 = torch.empty(16, n, dtype=torch.bfloat16, device=DEV)
    K.gemv_i8(x1, q, s, o1, (8, 1024, 4, 2))
    K.gemm_i8_small(x16, q, s, o16, (64, 128, 4, 4))
    assert _rel_rms(o1, K.ref_i8(x1, q, s)) < TOL and _rel_rms(o16, K.ref_i8(x16, q, s)) < TOL


def test_a_one_hot_activation_reads_q_times_s_exactly():
    n, k, j = 23, 2048, 1531
    q, s = K.quantize_rows_int8(_w(n, k, 5))
    x = torch.zeros(k, dtype=torch.bfloat16, device=DEV)
    x[j] = 1.0
    out = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    K.gemv_i8(x, q, s, out, (8, 1024, 4, 1))
    want = _bf16_conv(q[:, j].float() * s)
    assert torch.equal(out.cpu(), want)
    xm = torch.zeros(3, k, dtype=torch.bfloat16, device=DEV)
    xm[1, j] = 1.0
    om = torch.empty(3, n, dtype=torch.bfloat16, device=DEV)
    K.gemm_i8_small(xm, q, s, om, (32, 128, 4, 1))
    assert torch.equal(om[1].cpu(), want) and float(om[0].float().abs().max()) == 0.0


def test_split_k_matches_the_unsplit_kernels_within_rounding():
    n, k = 64, 4096
    q, s = K.quantize_rows_int8(_w(n, k, 6))
    x1, xm = _x((k,), 6), _x((7, k), 6)
    ref1, refm = K.ref_i8(x1, q, s), K.ref_i8(xm, q, s)
    o1 = {sk: K.gemv_i8(x1, q, s, torch.empty(n, dtype=torch.bfloat16, device=DEV), (4, 512, 4, sk)) for sk in (1, 2, 4)}
    om = {sk: K.gemm_i8_small(xm, q, s, torch.empty(7, n, dtype=torch.bfloat16, device=DEV), (32, 128, 4, sk))
          for sk in (1, 2, 4, 8)}
    for outs, ref in ((o1, ref1), (om, refm)):
        for o in outs.values():
            assert _rel_rms(o, ref) < TOL
        base = outs[1].double().cpu()
        for sk, o in outs.items():
            assert float((o.double().cpu() - base).abs().max()) <= float(ref.abs().max()) * 2 ** -7, sk


def test_the_quantizer_is_the_fake_quants_so_the_served_weights_are_p147s():
    w = _w(9, 384, 7)
    w[4] = 0.0
    q, s = K.quantize_rows_int8(w)
    assert q.dtype == torch.int8 and int(q.abs().max()) == 127 and float(s[4]) == 1.0 and int(q[4].abs().max()) == 0
    fq = w.clone()
    i8_fakequant.roundtrip_(fq)
    assert torch.equal((q.float() * s[:, None]).to(w.dtype).cpu(), fq.cpu())


def test_unregistered_configs_refuse():
    for bad, small in (((3, 512, 4, 1), False), ((8, 500, 4, 1), False), ((8, 512, 3, 1), False), ((8, 64, 4, 1), True)):
        with pytest.raises(ValueError):
            K.check_config(bad, small=small)
