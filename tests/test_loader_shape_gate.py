"""The streaming loader must not place a tensor whose shape the model disagrees with.

`_assign` REPLACES a meta parameter, so an unchecked mismatch does not raise -- it
makes the checkpoint's shape become the model's shape, and the disagreement resurfaces
later as a broadcast that happens to work. These tests pin the one disagreement that
is provably inert (Kimi K3's zero-padded `A_log`) as allowed, and every other shape of
disagreement as fatal.
"""
import pytest
import torch
from torch import nn

pytest.importorskip("bitsandbytes")          # loader imports it transitively

from experts4bit_qlora.loader import _assign, _fit

HEADS, HEAD_DIM = 96, 128                     # K3's KDA geometry


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(HEADS))
        self.q_proj = nn.Linear(64, 64, bias=False)
        self.register_buffer("scale", torch.zeros(HEADS))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()


def _meta_model():
    """A model whose tensors are meta -- what the loader is handed."""
    return Toy().to("meta")


# --------------------------------------------------------------------- _fit ---
def test_matching_shape_passes_through_untouched():
    t = torch.randn(HEADS)
    got, cut = _fit("attn.A_log", t, torch.zeros(HEADS, device="meta"))
    assert cut is False
    assert got.data_ptr() == t.data_ptr(), "a matching tensor must not be copied"


def test_k3_a_log_zero_padding_is_narrowed():
    """The real case: 96 trained values stored in 128 lanes, tail exact zeros.

    vLLM's `a_log_weight_loader` narrows to the head count; that is the reference
    behaviour, and this reproduces it through the loader's own gate.
    """
    full = torch.zeros(HEAD_DIM)
    full[:HEADS] = torch.randn(HEADS)
    got, cut = _fit("attn.A_log", full, torch.zeros(HEADS, device="meta"))
    assert cut is True
    assert got.shape == (HEADS,)
    assert torch.equal(got, full[:HEADS])
    assert got.data_ptr() != full.data_ptr(), (
        "must be an owned clone: a view would keep the 128-lane storage alive for the "
        "lifetime of the model, 69 times over")


def test_nonzero_padding_is_refused():
    """If the tail is not zero it is DATA, and narrowing it would be the silent
    wrongness the check exists to catch. `-exp(0) = -1` on the phantom lanes is a
    full-strength decay, so 'probably padding' is not good enough."""
    full = torch.randn(HEAD_DIM)                 # every lane populated
    with pytest.raises(ValueError, match="provably zero"):
        _fit("attn.A_log", full, torch.zeros(HEADS, device="meta"))


def test_shorter_than_the_parameter_is_refused():
    with pytest.raises(ValueError, match=r"holds \(64,\)"):
        _fit("attn.A_log", torch.zeros(64), torch.zeros(HEADS, device="meta"))


def test_same_numel_different_shape_is_refused():
    """The square-dim transpose class: 96x128 and 128x96 hold the same element count,
    so a numel-only check accepts a transposed tensor and computes garbage."""
    with pytest.raises(ValueError, match="but the model declares"):
        _fit("attn.q_proj.weight", torch.zeros(HEAD_DIM, HEADS),
             torch.zeros(HEADS, HEAD_DIM, device="meta"))


def test_padding_permit_is_one_dimensional_only():
    """A 2-D tensor with a zero-padded row block is not obviously padding -- the
    permit stays narrow rather than guessing which axis was padded."""
    big = torch.zeros(HEAD_DIM, 64)
    big[:HEADS] = torch.randn(HEADS, 64)
    with pytest.raises(ValueError):
        _fit("attn.q_proj.weight", big, torch.zeros(HEADS, 64, device="meta"))


def test_dtype_is_not_coerced():
    """K3 keeps A_log/dt_bias/o_norm in fp32 under a bf16 model. The checkpoint is the
    authority on dtype; quietly casting weights is what this package exists not to do."""
    t = torch.randn(HEADS, dtype=torch.float32)
    got, _cut = _fit("attn.A_log", t, torch.zeros(HEADS, dtype=torch.bfloat16,
                                                 device="meta"))
    assert got.dtype == torch.float32


# ------------------------------------------------------------------ _assign ---
def test_assign_checks_parameters():
    m = _meta_model()
    full = torch.zeros(HEAD_DIM)
    full[:HEADS] = torch.arange(HEADS, dtype=torch.float32)
    assert _assign(m, "attn.A_log", full) is True
    assert m.attn.A_log.shape == (HEADS,)
    assert not m.attn.A_log.is_meta
    assert torch.equal(m.attn.A_log.detach(), full[:HEADS])


def test_assign_checks_buffers_too():
    """Buffers were the unchecked half: `A_log` is a Parameter in K3, but the same
    padding could land on a registered buffer in the next architecture."""
    m = _meta_model()
    with pytest.raises(ValueError, match="but the model declares"):
        _assign(m, "attn.scale", torch.randn(HEAD_DIM))
    m2 = _meta_model()
    padded = torch.zeros(HEAD_DIM)
    padded[:HEADS] = 1.0
    assert _assign(m2, "attn.scale", padded) is True
    assert m2.attn.scale.shape == (HEADS,)


def test_assign_of_an_undeclared_attribute_is_not_checked():
    """Nothing to compare against, and nothing the model already believes."""
    m = _meta_model()
    assert _assign(m, "attn.something_new", torch.randn(3)) is False
    assert m.attn.something_new.shape == (3,)


def test_assign_reports_no_narrow_on_a_clean_load():
    m = _meta_model()
    assert _assign(m, "attn.A_log", torch.randn(HEADS)) is False
    assert _assign(m, "attn.q_proj.weight", torch.randn(64, 64)) is False
