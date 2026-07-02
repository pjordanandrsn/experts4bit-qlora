"""Correctness + torch.compile tests for the fused Triton NF4 double-dequant kernel.

Requires CUDA + bitsandbytes (skips cleanly otherwise). Validates the single-kernel double-dequant
against ``bitsandbytes.functional.dequantize_4bit`` in both fp16 and bf16, and that the
``triton_op`` path compiles with ``fullgraph=True`` (which raises on any graph break).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

if not torch.cuda.is_available():
    pytest.skip("Triton NF4 kernel is CUDA-only", allow_module_level=True)

import bitsandbytes.functional as F  # noqa: E402
from bitsandbytes.nn import Linear4bit  # noqa: E402

from experts4bit_qlora.triton_nf4 import your_dequantize_nf4, dequantize_nf4_compiled  # noqa: E402

_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)


def _make(in_features, out_features, dtype):
    try:
        lin = Linear4bit(
            in_features, out_features, bias=None, compute_dtype=dtype,
            compress_statistics=True, quant_type="nf4",
        ).to("cuda")
    except _UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit unavailable: {e}")
    lin.weight.quant_state.dtype = dtype
    return lin


def _bnb(lin):
    return F.dequantize_4bit(lin.weight.data, lin.weight.quant_state)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1024, 4096), (2048, 8192), (4096, 1024)])
def test_matches_bitsandbytes(dtype, shape):
    """One fused kernel reproduces bnb's two-step double-dequant, in f16 and bf16."""
    lin = _make(shape[0], shape[1], dtype)
    ref = _bnb(lin)
    got = your_dequantize_nf4(lin)
    assert got.dtype == dtype and got.shape == ref.shape
    torch.testing.assert_close(got, ref)  # bit-exact for bf16; within fp16 ULP for fp16


@pytest.mark.skipif(dequantize_nf4_compiled is None, reason="torch without triton_op")
def test_torch_compile_no_graph_break():
    """The triton_op path compiles under fullgraph=True (raises on any graph break) and matches eager."""
    from experts4bit_qlora.triton_nf4 import _dequantize_nf4_op

    lin = _make(1024, 4096, torch.bfloat16)
    qs = lin.weight.quant_state
    args = (
        lin.weight.data.reshape(-1), qs.absmax, qs.state2.code, qs.state2.absmax, qs.code,
        qs.offset, qs.shape[0], qs.shape[1], qs.blocksize * qs.state2.blocksize, qs.dtype,
    )
    eager = _dequantize_nf4_op(*args)

    torch._dynamo.reset()
    compiled = torch.compile(_dequantize_nf4_op, fullgraph=True)  # fullgraph -> errors on graph break
    out = compiled(*args)
    torch.testing.assert_close(out, eager)
