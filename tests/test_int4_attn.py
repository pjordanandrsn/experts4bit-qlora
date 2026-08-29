# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Serving attention on the int4-b32 grid: swap structure, refusal
paths, and (CUDA) forward parity of both branches against the dequant
reference. Skips wholesale until the grouped-nf4-gemm cut carrying
``int4_b32`` is installed."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("int4_pack_ref",
                    reason="needs grouped-nf4-gemm with int4_b32")

from torch import nn  # noqa: E402

from experts4bit_qlora.engines.int4_attn import (  # noqa: E402
    Int4Linear,
    enable_serve_attn_int4,
)


class _Attention(nn.Module):          # structural: name ends in Attention
    def __init__(self, bias=False):
        super().__init__()
        self.qkv_proj = nn.Linear(64, 160, bias=bias)
        self.o_proj = nn.Linear(128, 64, bias=bias)
        self.q_norm = nn.LayerNorm(64)   # non-Linear child: untouched


class _Model(nn.Module):
    def __init__(self, n=2, bias=False):
        super().__init__()
        self.layers = nn.ModuleList(_Attention(bias) for _ in range(n))
        self.lm_head = nn.Linear(64, 512, bias=False)


def test_swap_counts_and_lm_head_untouched():
    m = _Model(n=3)
    n = enable_serve_attn_int4(m)
    assert n == 6
    for lyr in m.layers:
        assert isinstance(lyr.qkv_proj, Int4Linear)
        assert isinstance(lyr.o_proj, Int4Linear)
        assert isinstance(lyr.q_norm, nn.LayerNorm)
    assert type(m.lm_head) is nn.Linear      # NEVER on this grid (+0.18 ppl)


def test_bias_refused_loudly():
    with pytest.raises(RuntimeError, match="bias"):
        enable_serve_attn_int4(_Model(bias=True))


def test_vacuous_enable_refused():
    class Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
    with pytest.raises(RuntimeError, match="vacuous|matched no"):
        enable_serve_attn_int4(Bare())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_decode_and_prefill_parity():
    pytest.importorskip("triton")
    torch.manual_seed(0)
    lin = nn.Linear(64, 160, bias=False).cuda().to(torch.bfloat16)
    q = Int4Linear(lin)
    wref = q._deq()                        # the module's own int4 truth
    x1 = torch.randn(1, 1, 64, device="cuda", dtype=torch.bfloat16) * 0.2
    got = q(x1)
    ref = (x1.reshape(-1, 64).to(torch.bfloat16) @ wref.t()).reshape_as(got)
    assert (got.float() - ref.float()).abs().max() \
        <= ref.float().abs().max() * 2 ** -6   # two bf16 roundings + int8 act
    xm = torch.randn(1, 7, 64, device="cuda", dtype=torch.bfloat16) * 0.2
    gm = q(xm)
    rm = (xm.reshape(-1, 64) @ wref.t()).reshape_as(gm)
    assert torch.allclose(gm.float(), rm.float(), rtol=1e-2, atol=1e-2)
    assert gm.dtype == xm.dtype
