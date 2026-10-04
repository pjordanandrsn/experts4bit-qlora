"""``_PermGather``: the fused training forward's routing-weight gather is the old two-index gather, bit for bit.

``fused_experts_train_forward`` used ``top_k_weights[order // k, order % k]``, whose backward radix-sorts its indices
(``index_put_(accumulate=True)``). ``order`` is a permutation of the [tokens, k] slots, so ``_PermGather`` (a 1-D
``index_select`` with a zero-fill + ``index_copy_`` backward) must give ``torch.equal`` values forward and backward.
"""
import pytest
import torch

from experts4bit_qlora.engines.fast import _PermGather

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("tokens,k", [(1, 8), (7, 2), (190, 8), (380, 8), (33, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_perm_gather_equals_the_two_index_gather(dev, tokens, k, dtype):
    g = torch.Generator().manual_seed(tokens * 31 + k)
    E = 16
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(tokens)]).to(dev)
    order = torch.argsort(idx.reshape(-1), stable=True)
    token_rows = order // k
    top_pos = order - token_rows * k
    base = torch.rand(tokens, k, generator=g).to(dtype).to(dev)
    gout = torch.randn(tokens * k, generator=g).to(torch.float32).to(dev)

    w_old = base.clone().requires_grad_(True)
    a = w_old[token_rows, top_pos].to(torch.float32)
    (a * gout).sum().backward()

    w_new = base.clone().requires_grad_(True)
    b = _PermGather.apply(w_new, order).to(torch.float32)
    (b * gout).sum().backward()

    assert torch.equal(a, b)
    assert w_new.grad.shape == w_old.grad.shape and w_new.grad.dtype == w_old.grad.dtype
    assert torch.equal(w_new.grad, w_old.grad)


def test_perm_gather_reads_a_non_contiguous_source_in_logical_order():
    base = torch.arange(24, dtype=torch.float32).view(4, 6).t()          # [6, 4], non-contiguous
    perm = torch.randperm(24, generator=torch.Generator().manual_seed(0))
    src = base.clone().requires_grad_(True)
    out = _PermGather.apply(src, perm)
    assert torch.equal(out, base.reshape(-1)[perm])
    out.sum().backward()
    assert torch.equal(src.grad, torch.ones_like(base))
