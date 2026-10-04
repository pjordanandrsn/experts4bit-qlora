"""`TRAIN_FROZEN_4BIT` (lora.quantize_frozen_linears_4bit): which frozen linears it stores in NF4.

The selection is the contract and runs on CPU: exact frozen bias-free bf16/fp16 ``nn.Linear`` modules, minus the output head, the
routers and the attention projections (those belong to ``TRAIN_ATTN_4BIT``). The conversion itself needs bitsandbytes on CUDA.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from experts4bit_qlora.lora import frozen_linear_4bit_candidates, quantize_frozen_linears_4bit  # noqa: E402

H = 64


def _lin(i, o, bias=False):
    return nn.Linear(i, o, bias=bias)


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = _lin(H, H), _lin(H, H), _lin(H, H), _lin(H, H)


class _LinearAttn(nn.Module):           # Qwen3.6's gated linear attention: dense projections, no q/k/o names
    def __init__(self):
        super().__init__()
        self.in_proj_qkvz, self.in_proj_ba, self.out_proj = _lin(H, 2 * H), _lin(H, 8), _lin(H, H)
        self.conv1d = nn.Conv1d(H, H, 4, groups=H)


class _Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = _lin(H, 2 * H), _lin(H, 2 * H), _lin(2 * H, H)


class _Moe(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _lin(H, 8)                      # the router
        self.shared_expert = _Shared()
        self.shared_expert_gate = _lin(H, 1)        # the shared expert's 1-wide gate


class _Layer(nn.Module):
    def __init__(self, full):
        super().__init__()
        if full:
            self.self_attn = _Attn()
        else:
            self.linear_attn = _LinearAttn()
        self.mlp = _Moe()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(full=True), _Layer(full=False)])
        self.lm_head = _lin(H, 100)
        self.biased = _lin(H, H, bias=True)
        self.trainable = _lin(H, H)
        self.fp32 = _lin(H, H)

    def get_output_embeddings(self):
        return self.lm_head


def _model():
    m = _Model().to(torch.bfloat16)
    for p in m.parameters():
        p.requires_grad_(False)
    m.trainable.weight.requires_grad_(True)
    m.fp32.float()
    return m


def test_candidates_are_the_dense_frozen_projections_only():
    got = sorted(full for _p, _n, full in frozen_linear_4bit_candidates(_model()))
    assert got == sorted([
        "layers.1.linear_attn.in_proj_qkvz", "layers.1.linear_attn.in_proj_ba", "layers.1.linear_attn.out_proj",
        "layers.0.mlp.shared_expert.gate_proj", "layers.0.mlp.shared_expert.up_proj", "layers.0.mlp.shared_expert.down_proj",
        "layers.1.mlp.shared_expert.gate_proj", "layers.1.mlp.shared_expert.up_proj", "layers.1.mlp.shared_expert.down_proj",
    ]), got


def test_routers_head_attention_bias_trainable_and_fp32_are_never_candidates():
    names = {full for _p, _n, full in frozen_linear_4bit_candidates(_model())}
    for never in ("layers.0.mlp.gate", "layers.1.mlp.gate", "layers.0.mlp.shared_expert_gate", "lm_head", "biased", "trainable", "fp32",
                  "layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj", "layers.0.self_attn.o_proj"):
        assert never not in names, never


def test_a_qwen3_like_model_has_no_candidates():
    """Qwen3-30B-A3B / Mixtral: the only frozen bf16 linears outside the attention are routers and lm_head -- the switch is a no-op."""
    m = _model()
    for layer in m.layers:
        layer.mlp.shared_expert = nn.Identity()
        layer.mlp.shared_expert_gate = nn.Identity()
    m.layers[1].linear_attn = _Attn()
    del m.biased, m.trainable, m.fp32
    assert frozen_linear_4bit_candidates(m) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes NF4 quantises on a CUDA transfer")
def test_quantize_converts_exactly_the_candidates():
    bnb = pytest.importorskip("bitsandbytes")
    m = _model().cuda()
    want = {full for _p, _n, full in frozen_linear_4bit_candidates(m)}
    assert quantize_frozen_linears_4bit(m) == len(want)
    four = {n for n, mod in m.named_modules() if isinstance(mod, bnb.nn.Linear4bit)}
    assert four == want and frozen_linear_4bit_candidates(m) == []
    assert type(m.layers[0].mlp.gate) is nn.Linear and type(m.lm_head) is nn.Linear
    x = torch.randn(3, H, device="cuda", dtype=torch.bfloat16)
    assert m.layers[1].linear_attn.out_proj(x).shape == (3, H)
