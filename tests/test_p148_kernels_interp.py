# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P148's prototype GEMVs (bench/p148/p148_kernels.py), correctness only: under Triton's interpreter on CPU in CI,
and compiled on the card as the box's premise (the runner sets TRITON_INTERPRET=0 explicitly). Every registered config
at ragged shapes against an fp64 reference; a one-hot activation reads one weight column exactly; split-K matches
the unsplit kernel; the int8 kernel reads its dequantized weights; the quantizer's range and zero-row scale."""
import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
# triton decorates its own @jit library (tl.zeros, tl.sum) at its first import, so the interpreter env must precede
# that import. In CI's full suite another module imports triton first; this file then skips itself here, and
# tests/test_p148.py runs it in a fresh interpreted process (as the box runs it, compiled, in its own process).
if os.environ["TRITON_INTERPRET"] == "1" and type(triton.language.zeros).__name__ != "InterpretedFunction":
    pytest.skip("triton was imported compiled earlier in this process; tests/test_p148.py runs this file in a fresh "
                "interpreted process", allow_module_level=True)

HERE = pathlib.Path(__file__).resolve().parent
for cand in (HERE, HERE.parent / "bench" / "p148"):
    if (cand / "p148_kernels.py").exists():
        sys.path.insert(0, str(cand))
        break
import p148_kernels as K  # noqa: E402

DEV = "cuda" if os.environ.get("TRITON_INTERPRET") == "0" and torch.cuda.is_available() else "cpu"


def _w(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, k, generator=g) / k ** 0.5).to(torch.bfloat16).to(DEV)


def _x(k, seed):
    g = torch.Generator().manual_seed(seed + 1)
    return torch.randn(k, generator=g).to(torch.bfloat16).to(DEV)


def _rel_rms(got, ref):
    return float(((got.double() - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())


@pytest.mark.parametrize("cfg", K.CONFIGS_BF16, ids=lambda c: "bf16-%d-%d-%d-%d" % c)
def test_every_bf16_config_reads_the_fp64_reference_at_a_ragged_shape(cfg):
    n, k = 37, 2560                                    # n not a BLOCK_N multiple; k not a KC multiple at KC 1024
    w, x = _w(n, k, 1), _x(k, 1)
    out = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    K.gemv_bf16(x, w, out, cfg)
    assert _rel_rms(out, K.ref_bf16(x, w)) < 4e-3          # bf16 output rounding (2^-9) with fp32 accumulation


@pytest.mark.parametrize("cfg", [(2, 512, 4, 1), (8, 1024, 8, 4), (4, 512, 4, 2)])
def test_a_one_hot_activation_reads_one_weight_column_exactly(cfg):
    n, k, j = 19, 2048, 1531
    w = _w(n, k, 2)
    x = torch.zeros(k, dtype=torch.bfloat16, device=DEV)
    x[j] = 1.0
    out = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    K.gemv_bf16(x, w, out, cfg)
    assert torch.equal(out.cpu(), w[:, j].cpu())


def test_split_k_matches_the_unsplit_kernel_within_rounding():
    n, k = 64, 4096
    w, x = _w(n, k, 3), _x(k, 3)
    ref = K.ref_bf16(x, w)
    outs = {}
    for sk in (1, 2, 4):
        o = torch.empty(n, dtype=torch.bfloat16, device=DEV)
        K.gemv_bf16(x, w, o, (4, 512, 4, sk))
        outs[sk] = o
        assert _rel_rms(o, ref) < 4e-3
    for sk in (2, 4):
        assert float((outs[sk].double() - outs[1].double()).abs().max()) <= float(ref.abs().max()) * 2 ** -7


@pytest.mark.parametrize("cfg", K.CONFIGS_I8, ids=lambda c: "i8-%d-%d-%d-%d" % c)
def test_every_int8_config_reads_its_dequantized_weights(cfg):
    n, k = 45, 2048 + 512
    q, s = K.quantize_rows_int8(_w(n, k, 4))
    x = _x(k, 4)
    out = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    K.gemv_i8(x, q, s, out, cfg)
    assert _rel_rms(out, K.ref_i8(x, q, s)) < 4e-3


def test_the_int8_one_hot_reads_q_times_s_exactly():
    n, k, j = 23, 2048, 77
    q, s = K.quantize_rows_int8(_w(n, k, 5))
    x = torch.zeros(k, dtype=torch.bfloat16, device=DEV)
    x[j] = 1.0
    out = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    K.gemv_i8(x, q, s, out, (8, 1024, 4, 1))
    f32 = (q[:, j].float() * s).cpu()
    if os.environ.get("TRITON_INTERPRET") == "1":
        # Triton's interpreter converts fp32 -> bf16 by truncation; compiled code (cvt.rn) rounds to nearest even.
        # Either way the fp32 product is exact; the check is exact against the conversion each mode uses.
        want = (f32.view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)
    else:
        want = f32.to(torch.bfloat16)
    assert torch.equal(out.cpu(), want)


def test_the_quantizer_spans_minus_127_to_127_and_keeps_a_zero_row():
    w = _w(6, 256, 6).float()
    w[2] = 0.0
    q, s = K.quantize_rows_int8(w.to(torch.bfloat16))
    assert int(q.abs().max()) == 127 and q.dtype == torch.int8 and float(s[2]) == 1.0 and int(q[2].abs().max()) == 0
    assert torch.all(q.abs().amax(dim=1)[[0, 1, 3, 4, 5]] == 127)


def test_the_registered_grids_and_refusals():
    assert len(K.CONFIGS_BF16) == 36 and len(K.CONFIGS_I8) == 18
    for bad in ((3, 512, 4, 1), (4, 500, 4, 1), (4, 512, 3, 1), (4, 512, 4, 0)):
        with pytest.raises(ValueError):
            K.check_config(bad)
    with pytest.raises(ValueError):
        K.check_config((4, 512, 4, 2), int8=True)
