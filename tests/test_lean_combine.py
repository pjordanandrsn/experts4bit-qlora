"""E4B_TRAIN_LEAN_COMBINE=1: the training combine's glue in fewer launches (engines/combine_lean.py), bit for bit the composite's.

The forward, both gradients (each alone too) and ``_PermGather``'s gradient must be EQUAL with the knob on and off: the kernels
change only launches, dead zero fills and exact casts. On a plain CPU run the lean functions take their torch fallbacks (the
wiring); under ``TRITON_INTERPRET=1`` or on CUDA they run the Triton kernels. The narrowing case feeds round-to-nearest-even
ties, NaNs, infinities, subnormals and the overflow edge through the down gradient on the same device as the composite.
"""
import pytest
import torch

from experts4bit_qlora.engines import fast
from experts4bit_qlora.engines.combine_lean import grad_products, lean_combine_enabled
from experts4bit_qlora.engines.fast import _PermGather, _ScatterCombine

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
GRADS = ((True, True), (True, False), (False, True))


def _case(dev, tokens, k, hidden, down_dtype, out_dtype, seed):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(tokens * k, generator=g).to(dev)
    down0 = torch.randn(tokens * k, hidden, generator=g).to(down_dtype).to(dev)
    w0 = torch.rand(tokens * k, generator=g).to(dev)
    gout = torch.randn(tokens, hidden, generator=g).to(out_dtype).to(dev)
    return order, down0, w0, gout


def _run(monkeypatch, lean, case, tokens, k, out_dtype, grads):
    monkeypatch.setenv("E4B_TRAIN_LEAN_COMBINE", "1" if lean else "0")
    order, down0, w0, gout = case
    d, w = down0.clone().requires_grad_(grads[0]), w0.clone().requires_grad_(grads[1])
    out = _ScatterCombine.apply(d, w, order, tokens, k, out_dtype)
    (out.float() * gout.float()).sum().backward()
    return out, d.grad, w.grad


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("tokens,k,hidden", [(1, 8, 64), (7, 2, 32), (190, 8, 128), (33, 4, 96), (5, 8, 2048), (3, 2, 1100)])
@pytest.mark.parametrize("down_dtype,out_dtype", [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32),
                                                  (torch.bfloat16, torch.float32)])
def test_lean_combine_is_the_composite_bit_for_bit(monkeypatch, dev, tokens, k, hidden, down_dtype, out_dtype):
    case = _case(dev, tokens, k, hidden, down_dtype, out_dtype, tokens * 31 + k + hidden)
    for grads in GRADS:
        base = _run(monkeypatch, False, case, tokens, k, out_dtype, grads)
        lean = _run(monkeypatch, True, case, tokens, k, out_dtype, grads)
        for a, b in zip(base, lean):
            assert (a is None and b is None) or (a.dtype == b.dtype and torch.equal(a, b))


@pytest.mark.parametrize("dev", DEVICES)
def test_perm_gather_backward_is_unchanged(monkeypatch, dev):
    g = torch.Generator().manual_seed(5)
    src0 = torch.randn(37, 8, generator=g).to(dev)
    perm = torch.randperm(37 * 8, generator=g).to(dev)
    gout = torch.randn(37 * 8, generator=g).to(dev)
    grads = []
    for lean in (False, True):
        monkeypatch.setenv("E4B_TRAIN_LEAN_COMBINE", "1" if lean else "0")
        src = src0.clone().requires_grad_(True)
        _PermGather.apply(src, perm).backward(gout)
        grads.append(src.grad)
    assert torch.equal(*grads)


def _narrowing_edges():
    """fp32 bit patterns at every rounding edge of fp32 -> bf16, positive and negative."""
    bits = [0x00000000, 0x3F808000, 0x3F818000, 0x3F807FFF, 0x3F808001, 0x3F80FFFF, 0x3F817FFF,   # zero, ties (even, odd), around
            0x00000001, 0x00008000, 0x00018000, 0x007FFFFF, 0x00800000,                          # subnormals, smallest normal
            0x7F7F7FFF, 0x7F7F8000, 0x7F7FFFFF,                                                   # largest finite: stays, ties up, overflows
            0x7F800000, 0x7FC00000, 0x7F800001, 0x7FFFFFFF, 0x7FA00000]                           # inf, NaNs (quiet, signalling, payloads)
    bits = bits + [b | 0x80000000 for b in bits]
    return torch.tensor(bits, dtype=torch.int64).to(torch.int32)


@pytest.mark.parametrize("dev", DEVICES)
def test_down_gradient_narrows_as_the_composite_does(dev):
    """The down gradient ``bf16(fp32(g[order // k]) * w)`` at every narrowing edge, with ``w = 1`` (an exact multiply), on the
    composite's own device. On CUDA every bit, NaNs included, is torch's CUDA cast. On CPU torch has no one NaN pattern: its
    vectorized cast writes 0xFFFF and its scalar rule 0x7FC0 (measured: torch 2.14 CPU, 0xFFFF for every NaN here), so there
    a NaN must stay a NaN and every other value must match bit for bit."""
    edges = _narrowing_edges().view(torch.float32)
    k, H = 1, edges.numel()
    g = edges.view(1, H).to(dev)
    down = torch.zeros(1, H, dtype=torch.bfloat16, device=dev)
    w = torch.ones(1, dtype=torch.float32, device=dev)
    order = torch.zeros(1, dtype=torch.int64, device=dev)
    want = (g.to(torch.float32)[torch.div(order, k, rounding_mode="floor")] * w[:, None]).to(torch.bfloat16)
    _, got = grad_products(g, down, w, order, k, need_w=False, need_down=True)
    got_b, want_b = got.view(torch.int16).cpu(), want.view(torch.int16).cpu()
    if dev == "cuda":
        assert torch.equal(got_b, want_b)
        return
    nan = torch.isnan(want.float().cpu())
    assert int(nan.sum()) == 8 and torch.isnan(got.float().cpu()[nan]).all()
    assert torch.equal(got_b[~nan], want_b[~nan])


def test_chunked_images_keep_the_chunked_path(monkeypatch):
    """Above the chunk gate the lean knob steps aside: the chunked combine runs and the bytes are the whole-tensor path's."""
    case = _case("cpu", 190, 8, 128, torch.bfloat16, torch.bfloat16, 3)
    base = _run(monkeypatch, False, case, 190, 8, torch.bfloat16, (True, True))
    monkeypatch.setattr(fast, "COMBINE_CHUNK_MIN_BYTES", 0)
    monkeypatch.setattr(fast, "COMBINE_CHUNK_BYTES", 100 * 128 * 4)
    n0 = dict(fast.COMBINE_STATS)
    lean = _run(monkeypatch, True, case, 190, 8, torch.bfloat16, (True, True))
    assert fast.COMBINE_STATS["chunked_fwd"] - n0["chunked_fwd"] == 1
    assert fast.COMBINE_STATS["chunked_bwd"] - n0["chunked_bwd"] == 1
    for a, b in zip(base, lean):
        assert torch.equal(a, b)


def test_the_knob_is_off_unless_one(monkeypatch):
    for v, want in (("1", True), (" 1 ", True), ("0", False), ("", False), ("true", False), ("2", False)):
        monkeypatch.setenv("E4B_TRAIN_LEAN_COMBINE", v)
        assert lean_combine_enabled() is want
    monkeypatch.delenv("E4B_TRAIN_LEAN_COMBINE")
    assert lean_combine_enabled() is False
