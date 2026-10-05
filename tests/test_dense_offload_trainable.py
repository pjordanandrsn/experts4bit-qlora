"""Dense offload never streams a trainable parameter beside frozen ones, and says so; an unfrozen model is unchanged.

The crash this pins (found by the DQ3 rehearsal at Qwen3-32B width): ``enable_dense_offload`` on a PEFT/QLoRA model
selected the trainable ``lora_B`` matrices (1.6 MB at 25600 wide, over ``MIN_BYTES``) for streaming, eviction swapped
them for empty placeholders, and ``AdamW.step`` raised "The size of tensor a (0) must match the size of tensor b (16)".

The rule is decided over the call: if any streamable parameter (2-D, >= min_bytes) is frozen, trainable ones stay
resident (with a warning); if none is frozen (an unfrozen model, the common inference case), the selection is exactly
the old one (with a warning naming the trainable tensors streamed).
"""
from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

from experts4bit_qlora.engines.dense_offload import _DenseOffload, enable_dense_offload

H, INTER, NL = 256, 1024, 3          # o_proj / q_proj = 1 MB fp32: over MIN_BYTES


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, INTER, bias=False)
        self.o_proj = nn.Linear(INTER, H, bias=False)
        self.norm = nn.Parameter(torch.ones(H))

    def forward(self, x):
        return x + self.o_proj(torch.relu(self.q_proj(x * self.norm)))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(NL))

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x


@pytest.fixture(autouse=True)
def _clean_class_state():
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()
    yield
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()


def _toy(freeze: str):
    """freeze: 'none' (unfrozen), 'all', 'base+one' (frozen base, layer 1's o_proj trainable: the LoRA shape),
    'layer' (frozen base, layer 2 entirely trainable: a partial fine-tune)."""
    torch.manual_seed(5)
    m = Toy()
    if freeze != "none":
        m.requires_grad_(False)
    if freeze == "base+one":
        m.layers[1].o_proj.weight.requires_grad_(True)
    if freeze == "layer":
        m.layers[2].requires_grad_(True)
    return m


def _selected(handles, m):
    """(layer index, module name, attr) of every streamed slot."""
    names = {id(mod): (i, n) for i, lay in enumerate(m.layers) for n, mod in lay.named_modules()}
    return sorted((*names[id(mod)], attr) for h in handles for mod, attr, _p, _home in h.slots)


def _offload(m, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        hs = enable_dense_offload(m, "cpu", pin=False, prefetch=False, **kw)
    return hs, [str(x.message) for x in w if "dense offload" in str(x.message)]


# The rule as main has always applied it: every 2-D parameter over min_bytes streams, norms stay.
EVERY_PROJECTION = sorted((i, p, "weight") for i in range(NL) for p in ("q_proj", "o_proj"))


def test_a_frozen_model_streams_exactly_what_it_always_did_and_is_silent():
    m = _toy("all")
    hs, msgs = _offload(m)
    assert _selected(hs, m) == EVERY_PROJECTION
    assert msgs == []


def test_an_unfrozen_model_streams_exactly_what_it_always_did_and_warns():
    m = _toy("none")
    hs, msgs = _offload(m)
    assert _selected(hs, m) == EVERY_PROJECTION, "an unfrozen (inference) model's selection changed"
    assert len(msgs) == 1 and "streaming 6 trainable tensor(s)" in msgs[0] and "requires_grad_(False)" in msgs[0], msgs


def test_a_trainable_matrix_beside_frozen_ones_stays_resident_and_warns():
    m = _toy("base+one")
    hs, msgs = _offload(m)
    assert _selected(hs, m) == [s for s in EVERY_PROJECTION if s != (1, "o_proj", "weight")]
    assert m.layers[1].o_proj.weight.numel() == H * INTER, "the trainable weight must stay bound (not a placeholder)"
    assert len(msgs) == 1 and "kept 1 trainable tensor(s) resident" in msgs[0], msgs


def test_a_fully_trainable_layer_in_a_frozen_model_stays_resident():
    """Decided per call, not per layer: layer 2 alone would look 'unfrozen', but it is a partial fine-tune."""
    m = _toy("layer")
    hs, msgs = _offload(m)
    assert _selected(hs, m) == [s for s in EVERY_PROJECTION if s[0] != 2]
    assert len(msgs) == 1 and "kept 2 trainable tensor(s) resident" in msgs[0], msgs


def test_the_log_carries_the_warning_too():
    lines = []
    enable_dense_offload(_toy("base+one"), "cpu", pin=False, prefetch=False, log=lines.append)
    assert any("WARNING" in s and "kept 1 trainable" in s for s in lines), lines


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))])
@pytest.mark.parametrize("freeze", ["base+one", "layer"])
def test_an_optimizer_steps_through_the_offloaded_model_exactly(device, freeze):
    """The rehearsal's crash in miniature: AdamW over trainable matrices over MIN_BYTES, through the offloaded model,
    two steps. Before the fix AdamW raised on the evicted placeholder; now losses and trained weights match the
    un-offloaded model bit for bit."""
    def run(offload):
        _DenseOffload._staged_now.clear()
        _DenseOffload._resident.clear()
        m = _toy(freeze).to(device)
        if offload:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                enable_dense_offload(m, device, pin=False, prefetch=False)
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3)
        g = torch.Generator(device="cpu").manual_seed(9)
        losses = []
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss = m(torch.randn(8, H, generator=g).to(device)).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu())
        return losses, [p.detach().cpu().clone() for p in params]

    (la, wa), (lb, wb) = run(False), run(True)
    assert all(torch.equal(a, b) for a, b in zip(la, lb)), (la, lb)
    assert len(wa) == len(wb) and all(torch.equal(a, b) for a, b in zip(wa, wb)), "trained weights diverged"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("freeze", ["base+one", "layer"])
def test_an_optimizer_built_before_offload_still_trains(freeze):
    """Stage on CPU, build AdamW, THEN enable_dense_offload(m, "cuda"). Every trainable parameter that offload places
    on the GPU (a kept LoRA-shaped matrix, the 1-D norms) must be moved IN PLACE: a re-wrapped Parameter would leave
    the optimizer stepping a stale CPU copy and the model would silently not train. Two steps must move the weights
    and match a model that was on CUDA from the start, bit for bit."""
    def run(offload):
        _DenseOffload._staged_now.clear()
        _DenseOffload._resident.clear()
        m = _toy(freeze)
        if not offload:
            m = m.to("cuda")
        params = [p for p in m.parameters() if p.requires_grad]
        before = [p.detach().cpu().clone() for p in params]
        opt = torch.optim.AdamW(params, lr=1e-3)
        if offload:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                enable_dense_offload(m, "cuda", pin=False, prefetch=False)
        g = torch.Generator(device="cpu").manual_seed(9)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            m(torch.randn(8, H, generator=g).to("cuda")).pow(2).mean().backward()
            opt.step()
        return before, [p.detach().cpu().clone() for p in params]

    b0, w_ref = run(False)
    b1, w_off = run(True)
    assert all(torch.equal(a, b) for a, b in zip(b0, b1))
    assert any(not torch.equal(a, b) for a, b in zip(b1, w_off)), "no parameter moved: the optimizer stepped stale copies"
    assert len(w_ref) == len(w_off) and all(torch.equal(a, b) for a, b in zip(w_ref, w_off)), "diverged from no-offload"
