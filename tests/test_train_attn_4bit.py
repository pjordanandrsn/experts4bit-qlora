# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""TRAIN_ATTN_4BIT: the frozen attention projections stored in bnb NF4
before the LoRA wrap. The conversion must be structural (any module with
q/k/v/o Linears), refuse biased projections rather than dropping the
bias, and compose with LoRALinear (bnb Linear4bit IS an nn.Linear, so
add_attention_lora's detector still matches the wrapped result)."""
import pytest

torch = pytest.importorskip("torch")
bnb = pytest.importorskip("bitsandbytes", reason="needs bitsandbytes")

from torch import nn  # noqa: E402

from experts4bit_qlora.lora import (  # noqa: E402
    LoRALinear,
    add_attention_lora,
    quantize_attention_projections_4bit,
)


class _Attn(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        for p, (i, o) in dict(q_proj=(64, 128), k_proj=(64, 64),
                              v_proj=(64, 64), o_proj=(128, 64)).items():
            setattr(self, p, nn.Linear(i, o, bias=bias))


class _Model(nn.Module):
    def __init__(self, n=2, bias=False):
        super().__init__()
        self.layers = nn.ModuleList(_Attn(bias) for _ in range(n))
        self.other = nn.Linear(8, 8)      # non-attention: must NOT convert


def test_converts_structurally_and_counts():
    m = _Model(n=3)
    n = quantize_attention_projections_4bit(m)
    assert n == 12
    for lyr in m.layers:
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert isinstance(getattr(lyr, p), bnb.nn.Linear4bit)
    assert type(m.other) is nn.Linear     # untouched


def test_biased_projection_refused():
    m = _Model(bias=True)
    with pytest.raises(SystemExit, match="bias"):
        quantize_attention_projections_4bit(m)


def test_lora_wraps_the_4bit_base():
    m = _Model(n=1)
    quantize_attention_projections_4bit(m)
    n = add_attention_lora(m, r=4, alpha=8, dtype=torch.bfloat16)
    assert n == 4
    q = m.layers[0].q_proj
    assert isinstance(q, LoRALinear)
    assert isinstance(q.base, bnb.nn.Linear4bit)
    assert q.lora_A.requires_grad and q.lora_B.requires_grad
    assert not any(p.requires_grad for p in q.base.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_forward_close_to_bf16_base():
    torch.manual_seed(0)
    m = _Model(n=1).cuda().to(torch.bfloat16)
    ref = {p: getattr(m.layers[0], p).weight.detach().clone()
           for p in ("q_proj",)}
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    want = m.layers[0].q_proj(x)
    quantize_attention_projections_4bit(m)
    got = m.layers[0].q_proj(x.clone())
    err = (got.float() - want.float()).abs().max()
    scale = ref["q_proj"].float().abs().max()
    assert err <= scale, (err, scale)     # 4-bit-error class, not garbage
