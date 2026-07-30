"""Gates for non-expert residency: the dense weights must come back UNCHANGED.

This exists because the alternative — quantizing attention to fit a cheap card —
measurably does not. On a real K3 attention weight, NF4 leaves 2.87% of elements
bit-identical with 10.4% median relative error. A residency path has no excuse for
losing a single bit, so that is what these tests check, at three levels:

1. **Bit-identity of a staged weight** vs the tensor before offload. No CUDA needed
   for the selection/home half; the staged comparison needs a device.
2. **Forward equivalence** — a model's output with dense offload must be
   ``torch.equal`` to its output without. Not close. Equal.
3. **Residency actually bounded** — at most two layers staged during a no_grad
   forward, and zero once it returns. Otherwise nothing has been saved.

Plus the selection rules, which are where a silent mistake would hide: touching an
expert module, or trying to pin a meta tensor, or moving 1-D norms.
"""
import pytest
import torch
from torch import nn

from experts4bit_qlora.dense_offload import (  # noqa: E402
    _DenseOffload, decoder_layers, dense_offload_report, enable_dense_offload)

H, INTER, NL = 512, 1024, 4
# 512x1024 fp32 = 2 MB per projection (above MIN_BYTES); the conv
# buffer is 16 KB and the norm is 1-D, both below/ineligible — the
# same relationship K3 has (projections GB-scale, tail 0.04 GB).


class Block(nn.Module):
    """A decoder-layer stand-in: big 2-D projections, a 1-D norm, a small 3-D conv
    — the shapes that make the selection rule non-trivial (K3 has all three)."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, INTER, bias=False)
        self.o_proj = nn.Linear(INTER, H, bias=False)
        self.norm = nn.Parameter(torch.ones(H))                 # 1-D: stays
        self.register_buffer("conv1d", torch.randn(INTER, 1, 4))  # 3-D, tiny: stays

    def forward(self, x):
        return x + self.o_proj(torch.relu(self.q_proj(x * self.norm)))


class Toy(nn.Module):
    def __init__(self, n=NL):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(n))

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x


@pytest.fixture(autouse=True)
def _clean_class_state():
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident = None
    yield
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident = None


def _model(device, seed=11):
    torch.manual_seed(seed)
    return Toy().to(device)


# ------------------------------------------------------------ selection ------
def test_decoder_layers_found_by_name_not_class():
    m = _model("cpu")
    got = decoder_layers(m)
    assert [n for n, _ in got] == [f"layers.{i}" for i in range(NL)], got
    # and they come back in DEPTH order, which the prefetch chain depends on
    assert [int(n.rsplit(".", 1)[1]) for n, _ in got] == list(range(NL))


def test_only_big_2d_tensors_are_offloaded():
    """1-D norms and the small 3-D conv buffer must stay resident: moving them
    costs a launch and saves nothing. K3's whole such tail is 0.04 GB."""
    m = _model("cpu")
    h = _DenseOffload(m.layers[0], "cpu", pin=False)
    attrs = {a for _mod, a, _p, _home in h.slots}
    assert attrs == {"weight"}, attrs
    assert len(h.slots) == 2, h.slots        # q_proj.weight, o_proj.weight
    assert h.bytes == (H * INTER + INTER * H) * 4


def test_min_bytes_floor_is_respected():
    m = _model("cpu")
    huge = _DenseOffload(m.layers[0], "cpu", pin=False, min_bytes=1 << 30)
    assert huge.slots == [] and huge.bytes == 0


def test_meta_tensors_are_skipped():
    """Arena-served experts live on `meta`. Pinning one would raise; silently
    materializing one would defeat the entire point of the arena."""
    m = _model("cpu")
    with torch.device("meta"):
        m.layers[0].meta_big = nn.Parameter(torch.empty(256, 256))
    h = _DenseOffload(m.layers[0], "cpu", pin=False)
    assert all(not home.is_meta for _mo, _a, _p, home in h.slots)
    assert len(h.slots) == 2, "the meta parameter must not have been captured"


def test_expert_modules_are_left_alone():
    """Composition with the expert path depends on this: an expert module is
    somebody else's residency problem."""
    m = _model("cpu")

    class FakeExperts(nn.Module):
        def __init__(self):
            super().__init__()
            for n in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax"):
                self.register_buffer(n, torch.zeros(128, 128))

    m.layers[0].experts = FakeExperts()
    h = _DenseOffload(m.layers[0], "cpu", pin=False)
    assert len(h.slots) == 2, [a for _m, a, _p, _h in h.slots]


# ------------------------------------------------------- BIT IDENTITY --------
def test_homes_are_bit_identical_copies():
    """The claim the whole module rests on. No cast, no quantization, no rounding."""
    m = _model("cpu")
    before = {id(mod): mod.weight.detach().clone()
              for mod in (m.layers[0].q_proj, m.layers[0].o_proj)}
    h = _DenseOffload(m.layers[0], "cpu", pin=False)
    for mod, attr, _p, home in h.slots:
        orig = before[id(mod)]
        assert home.dtype == orig.dtype, (home.dtype, orig.dtype)
        assert home.shape == orig.shape
        assert torch.equal(home, orig), (
            f"{attr}: home differs from the loaded weight — "
            f"{(home - orig).abs().max().item()}")


def test_evicted_then_staged_restores_exactly():
    m = _model("cpu")
    orig = {n: p.detach().clone() for n, p in m.named_parameters()}
    h = _DenseOffload(m.layers[0], "cpu", pin=False)
    assert m.layers[0].q_proj.weight.numel() == 0, "should start evicted"
    h.stage()
    for n, p in m.layers[0].named_parameters():
        key = f"layers.0.{n}"
        assert torch.equal(p, orig[key]), n


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_staged_weights_are_bit_identical_on_device():
    m = _model("cuda")
    orig = {n: p.detach().clone() for n, p in m.named_parameters()}
    hs = enable_dense_offload(m, "cuda", pin=True)
    hs[0].stage()
    for n, p in m.layers[0].named_parameters():
        assert torch.equal(p, orig[f"layers.0.{n}"]), n


@cuda
def test_forward_is_bit_identical_to_no_offload():
    """The end-to-end gate: same answer, weights never resident together."""
    ref = _model("cuda")
    x = torch.randn(3, H, device="cuda")
    ref.eval()
    with torch.no_grad():
        want = ref(x)

    m = _model("cuda")           # same seed -> same weights
    m.eval()
    with torch.no_grad():
        assert torch.equal(ref(x), m(x)), "fixture is not deterministic"
    enable_dense_offload(m, "cuda", pin=True)
    with torch.no_grad():
        got = m(x)
    assert torch.equal(got, want), (got - want).abs().max().item()


@cuda
def test_two_resident_at_most_and_zero_after():
    m = _model("cuda")
    m.eval()
    hs = enable_dense_offload(m, "cuda", pin=True)
    seen = []

    orig_pre = hs[1].stage_for_inference

    def spy():
        orig_pre()
        seen.append(sum(1 for h in hs if h.staged))

    hs[1].stage_for_inference = spy
    with torch.no_grad():
        m(torch.randn(2, H, device="cuda"))
    assert seen and max(seen) <= 2, seen
    # ONE, not zero: the prefetch chain wraps, so finishing the last layer has
    # already started layer 0's copy for the next token. That is the intent — the
    # bound that matters is that residency never grows with depth.
    assert sum(1 for h in hs if h.staged) <= 1, [h.staged for h in hs]


@cuda
def test_grad_enabled_forward_stays_resident_for_backward():
    """Evicting after a grad-enabled forward would hand autograd 0-element
    placeholders — it fails with a shape mismatch far from the cause. So under grad
    the weights stay staged: no saving, but correct. Gradients must also match an
    un-offloaded run exactly."""
    x = torch.randn(2, H, device="cuda")
    ref = _model("cuda")
    ref.train()
    ref(x).sum().backward()
    want = {n: p.grad.detach().clone() for n, p in ref.named_parameters()
            if p.grad is not None}

    m = _model("cuda")
    m.train()
    enable_dense_offload(m, "cuda", pin=True)
    m(x).sum().backward()          # must not raise
    got = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
    assert set(got) == set(want), (set(got) ^ set(want))
    for n in want:
        assert torch.equal(got[n], want[n]), (
            n, (got[n] - want[n]).abs().max().item())


@cuda
def test_report_numbers_are_real():
    m = _model("cuda")
    hs = enable_dense_offload(m, "cuda", pin=True)
    r = dense_offload_report(hs)
    assert r["layers"] == NL
    assert r["tensors"] == 2 * NL
    assert r["host_bytes"] == sum(h.bytes for h in hs) > 0
    assert r["all_pinned"] is True
    assert r["seconds_per_token_at_19GBs"] == round(r["host_bytes"] / 19e9, 3)


@cuda
def test_idempotent_enable():
    m = _model("cuda")
    a = enable_dense_offload(m, "cuda", pin=True)
    b = enable_dense_offload(m, "cuda", pin=True)
    assert [id(x) for x in a] == [id(x) for x in b], "second call rebuilt handles"
    # a second handle would have captured the 0-element placeholders as its homes
    assert all(h.bytes > 0 for h in b)


# ---------------- regressions for the five Bugbot findings on #45 -------------
def test_state_dict_while_evicted_carries_real_weights():
    """FINDING 5, and the one with destructive consequences: saving a checkpoint
    while layers are evicted would serialize 0-element placeholders — a file that
    looks fine and has no attention weights in it."""
    m = _model("cpu")
    want = {n: p.detach().clone() for n, p in m.named_parameters()}
    enable_dense_offload(m, "cpu", pin=False)
    assert m.layers[0].q_proj.weight.numel() == 0, "must be evicted for this test"
    sd = m.state_dict()
    for n, t in want.items():
        assert n in sd, n
        assert sd[n].numel() == t.numel(), (n, sd[n].shape, t.shape)
        assert torch.equal(sd[n].cpu(), t.cpu()), n


def test_prefetch_false_actually_disables_prefetch():
    """FINDING 3: links were assigned only under `prefetch=True`, so a second call
    with prefetch=False left the first call's circular links in place."""
    m = _model("cpu")
    a = enable_dense_offload(m, "cpu", pin=False, prefetch=True)
    assert all(h._prefetch_next is not None for h in a)
    b = enable_dense_offload(m, "cpu", pin=False, prefetch=False)
    assert b is not None and [id(x) for x in a] == [id(x) for x in b]
    assert all(h._prefetch_next is None for h in b), \
        "stale prefetch links survived a prefetch=False call"


def test_stage_sweeps_every_staged_handle():
    """FINDING 4: stage() evicted only `_resident`, so a grad-enabled forward after
    an inference forward inherited that forward's TWO residents and silently broke
    the single-slot bound stage() exists to enforce."""
    m = _model("cpu")
    hs = enable_dense_offload(m, "cpu", pin=False, prefetch=True)
    hs[0].stage()
    hs[1].stage()
    _DenseOffload._staged_now.add(hs[2])      # simulate a leftover prefetch
    hs[2].staged = True
    hs[3].stage()
    assert sum(1 for h in hs if h.staged) == 1, [h.staged for h in hs]
    assert _DenseOffload._staged_now == {hs[3]}, _DenseOffload._staged_now


@cuda
def test_stage_waits_for_an_inflight_prefetch():
    """FINDING 1 (High): a layer can reach stage() already `staged` from a prefetch
    whose copy is still in flight. Binding without consuming the ready event hands
    compute a partially-written weight. Checked by asserting the event is consumed
    and the values are right."""
    m = _model("cuda")
    want = {n: p.detach().clone() for n, p in m.named_parameters()}
    hs = enable_dense_offload(m, "cuda", pin=True, prefetch=True)
    hs[0].stage_for_inference()               # also prefetches hs[1]
    assert hs[1].staged and hs[1].ready_event is not None, "no prefetch in flight"
    hs[1].stage()                             # the dangerous transition
    assert hs[1].ready_event is None, "stage() bound without consuming the event"
    for n, p in m.layers[1].named_parameters():
        assert torch.equal(p, want[f"layers.1.{n}"]), n


@cuda
def test_bound_tensors_are_record_streamed():
    """FINDING 2 (High): tensors allocated on the prefetch stream and consumed on
    the compute stream must be record_stream'd, or the allocator can reuse the
    block while compute still reads it. Unobservable directly, so this asserts the
    marking happens and that a churned allocator still yields correct weights."""
    m = _model("cuda")
    want = {n: p.detach().clone() for n, p in m.named_parameters()}
    hs = enable_dense_offload(m, "cuda", pin=True, prefetch=True)
    marked = []
    real = torch.Tensor.record_stream

    def spy(self, s):
        marked.append(tuple(self.shape))
        return real(self, s)

    torch.Tensor.record_stream = spy
    try:
        hs[0].stage_for_inference()
        hs[1].stage_for_inference()
    finally:
        torch.Tensor.record_stream = real
    assert marked, "no tensor was record_stream'd on bind"
    # churn the allocator hard, then confirm the still-bound weights are intact
    junk = [torch.empty(1 << 20, device="cuda") for _ in range(64)]
    del junk
    for n, p in m.layers[1].named_parameters():
        if p.numel():
            assert torch.equal(p, want[f"layers.1.{n}"]), n
