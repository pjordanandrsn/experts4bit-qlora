"""The LoRA delta skips its scaling multiply at exactly 1 -- bit-exact, and only there.

At alpha == r (the field recipe) ``scaling * delta`` was a full elementwise pass over every delta in the forward, the checkpoint
recompute and the backward. ``1.0 * x == x`` for every float, so dropping it changes no byte; any other scaling must still apply.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experts4bit_qlora.lora import LoRALinear, _scaled


def test_scaled_skips_only_exact_one():
    d = torch.randn(3, 4)
    assert _scaled(d, 1.0) is d and _scaled(d, 1) is d
    assert torch.equal(_scaled(d, 2.0), 2.0 * d) and torch.equal(_scaled(d, 0.5), 0.5 * d)
    assert _scaled(d, torch.tensor(1.0)) is not d                    # a tensor scaling is applied, never compared


@pytest.mark.parametrize("r,alpha", [(16, 16), (8, 16), (16, 32), (16, 8), (12, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lora_linear_forward_and_grads_equal_the_explicit_multiply(r, alpha, dtype):
    torch.manual_seed(0)
    base = nn.Linear(32, 24, bias=False)
    m = LoRALinear(base, r=r, alpha=alpha, dtype=dtype)
    with torch.no_grad():
        m.lora_B.normal_(0, 0.1)
    x = torch.randn(5, 32, requires_grad=True)
    y = m(x)
    gy = torch.randn_like(y)
    y.backward(gy)
    got = (y.detach(), x.grad.clone(), m.lora_A.grad.clone(), m.lora_B.grad.clone())
    for t in (x, m.lora_A, m.lora_B):
        t.grad = None
    ref = m.base(x) + (m.scaling * F.linear(F.linear(x.to(m.lora_A.dtype), m.lora_A), m.lora_B)).to(x.dtype)   # the previous body
    ref.backward(gy)
    want = (ref.detach(), x.grad, m.lora_A.grad, m.lora_B.grad)
    for name, g, w in zip(("y", "dx", "dA", "dB"), got, want):
        assert torch.equal(g, w), (name, r, alpha, dtype)
