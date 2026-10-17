"""Unit tests for :func:`experts4bit_qlora.engines.offload.hold_forward_residency`, the forward hold a layer-major
driver uses to stage each layer once across its micro-batches. CPU-only torch is enough; under CUDA the same tests run
the real host<->device path.

What the hold must do, and what it must not change:
- a layer-major forward under the hold stages each layer ONCE (M micro-batches, one copy). A whole held step stages
  exactly 2n from cold (n forward, n recompute; the exit evicts the last layer), and 2n - 1 in steady state, where the
  previous backward left layer 0 resident (Shape B: 187, as ``bench/q235/staged-bytes`` counts it);
- the single resident slot holds at every copy, held or not;
- the exit evicts and clears the flag, also on an exception;
- gradients are the same math as the as-is schedule (fp32 here, so summation order only);
- routed handles, inference policies, a nested hold and a call inside a backward are refused.
"""

import pytest

from quant_guard import require_quantize

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from torch.utils.checkpoint import checkpoint  # noqa: E402

from experts4bit_qlora import Experts4bit, ExpertsLoRA  # noqa: E402
from experts4bit_qlora import enable_expert_offload  # noqa: E402
from experts4bit_qlora.engines.offload import _ExpertOffload, hold_forward_residency  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
N_EXP, HIDDEN, INTER, TOP_K = 4, 64, 64, 2
N_LAYERS, TOKENS = 3, (5, 8, 3)  # three unequal micro-batches


@pytest.fixture(autouse=True)
def _reset_class_state():
    """The resident slot and the hold are class-level global state; reset them around each test."""
    _ExpertOffload._resident = None
    _ExpertOffload._hold = False
    yield
    _ExpertOffload._resident = None
    _ExpertOffload._hold = False


@pytest.fixture
def copies(monkeypatch):
    """Count host->device stagings and assert the single slot at each one: every stage() copy happens only after the
    previous resident layer was evicted."""
    log = []
    real = _ExpertOffload._copy_home_to_device

    def counted(self, policy="sync"):
        staged = [h for h in _HANDLES if h.staged]
        assert not staged, f"single slot broken: {len(staged)} layer(s) still staged at a new copy"
        log.append(self)
        return real(self, policy)

    monkeypatch.setattr(_ExpertOffload, "_copy_home_to_device", counted)
    return log


_HANDLES: list = []


def _stack(seed=0):
    require_quantize(DEVICE, "nf4")
    layers = []
    for i in range(N_LAYERS):
        torch.manual_seed(seed + i)
        gate_up = torch.randn(N_EXP, 2 * INTER, HIDDEN, dtype=DTYPE, device=DEVICE) * HIDDEN ** -0.5  # stays O(1) over
        down = torch.randn(N_EXP, HIDDEN, INTER, dtype=DTYPE, device=DEVICE) * INTER ** -0.5  # the stacked layers
        base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=DTYPE)
        lo = ExpertsLoRA(base, r=8, alpha=16, dtype=DTYPE).to(DEVICE)
        torch.nn.init.normal_(lo.gate_up_lora_B, std=0.01)
        torch.nn.init.normal_(lo.down_lora_B, std=0.01)
        layers.append(lo)
    return layers


def _offload(layers):
    _HANDLES[:] = [enable_expert_offload(lo, DEVICE, pin=False) for lo in layers]
    return list(_HANDLES)


def _micro_batches(seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for t in TOKENS:
        hs = torch.randn(t, HIDDEN, dtype=DTYPE, generator=g).to(DEVICE)
        idx = torch.randint(0, N_EXP, (t, TOP_K), generator=g).to(DEVICE)
        w = torch.rand(t, TOP_K, dtype=DTYPE, generator=g).to(DEVICE)
        out.append((hs.requires_grad_(True), idx, w))  # inputs require grad, as enable_input_require_grads makes them
    return out


def _weights(mbs):
    total = sum(hs.shape[0] for hs, _, _ in mbs)
    return [hs.shape[0] / total for hs, _, _ in mbs]


def _layer(lo, reentrant):
    return lambda x, idx, w: checkpoint(lambda a, b, c: lo(a, b, c), x, idx, w, use_reentrant=reentrant)


def _as_is(layers, mbs, reentrant):
    """Shape B's schedule: per micro-batch, a forward through every layer, then its weighted backward."""
    for (x, idx, w), wk in zip(mbs, _weights(mbs)):
        for lo in layers:
            x = _layer(lo, reentrant)(x, idx, w)
        (x.pow(2).mean() * wk).backward()


def _layer_major(layers, mbs, reentrant, handles=None):
    """Every micro-batch through layer L before L+1, then ONE backward of the weighted sum. ``handles`` -> held."""
    xs = [x for x, _, _ in mbs]
    ctx = hold_forward_residency(handles) if handles is not None else _null()
    with ctx:
        for lo in layers:
            xs = [_layer(lo, reentrant)(x, idx, w) for x, (_, idx, w) in zip(xs, mbs)]
    sum(x.pow(2).mean() * wk for x, wk in zip(xs, _weights(mbs))).backward()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _grads(layers):
    return [p.grad.clone() for lo in layers for p in (lo.gate_up_lora_A, lo.gate_up_lora_B, lo.down_lora_A, lo.down_lora_B)]


@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_held_layer_major_stages_each_layer_once_per_direction(copies, reentrant):
    """Under the hold, a layer-major step from cold stages exactly 2n: n in the forward (one per layer, all micro-batches)
    and n in the recompute, layer-major. The next step starts with layer 0 still resident from the recompute, so it
    stages 2n - 1. Without the hold the same schedule pays M per layer in the forward. Both checkpoint kinds."""
    layers = _stack()
    handles = _offload(layers)
    m, n = len(TOKENS), N_LAYERS

    _layer_major(layers, _micro_batches(), reentrant)  # unheld, from cold: the post-hook evicts after every micro-batch
    assert len(copies) == m * n + n
    assert handles[0].staged  # the recompute ended on layer 0, and the single slot keeps it

    for h in handles:
        h.evict()
    copies.clear()
    _layer_major(layers, _micro_batches(), reentrant, handles=handles)  # held, from cold
    assert len(copies) == 2 * n
    assert [handles.index(h) for h in copies[:n]] == list(range(n))  # the forward, in layer order
    assert [handles.index(h) for h in copies[n:]] == list(range(n - 1, -1, -1))  # the recompute, layer-major

    copies.clear()
    _layer_major(layers, _micro_batches(), reentrant, handles=handles)  # held, steady state: layer 0 already resident
    assert len(copies) == 2 * n - 1


@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_held_layer_major_gradients_match_as_is(reentrant):
    """The same weighted sum as the as-is schedule. fp32 here, so only the summation order differs (one backward of the
    sum against M accumulating backwards); the bf16 band is the layer-major driver's test, not this one."""
    layers = _stack()
    _offload(layers)
    _as_is(layers, _micro_batches(), reentrant)
    ref = _grads(layers)
    for lo in layers:
        lo.zero_grad(set_to_none=True)
    _layer_major(layers, _micro_batches(), reentrant, handles=_HANDLES)
    for got, want in zip(_grads(layers), ref):
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-7)


def test_exit_evicts_and_clears_the_flag_also_on_an_exception():
    layers = _stack()
    handles = _offload(layers)
    hs, idx, w = _micro_batches()[0]
    with pytest.raises(ValueError, match="boom"):
        with hold_forward_residency(handles):
            with torch.no_grad():
                layers[0](hs, idx, w)
            assert handles[0].staged  # held: the post-hook kept it
            raise ValueError("boom")
    assert _ExpertOffload._hold is False
    assert not any(h.staged for h in handles)
    assert _ExpertOffload._resident is None


def test_without_the_hold_the_post_hook_still_evicts():
    """Flag off is today's behaviour: the forward evicts its layer right after the call."""
    layers = _stack()
    handles = _offload(layers)
    hs, idx, w = _micro_batches()[0]
    with torch.no_grad():
        layers[0](hs, idx, w)
    assert not handles[0].staged


@pytest.mark.parametrize("attr, value, match", [
    ("_routed_only", True, "routed"),
    ("_routed_in_train", True, "routed"),
    ("_prefetch_next", "linked", "prefetch"),
    ("_superseded", True, "pipelined"),
])
def test_refuses_routed_handles_and_inference_policies(attr, value, match):
    layers = _stack()
    handles = _offload(layers)
    setattr(handles[1], attr, value)
    with pytest.raises(RuntimeError, match=match):
        with hold_forward_residency(handles):
            pass
    assert _ExpertOffload._hold is False  # refused before anything was held


def test_refuses_a_nested_hold():
    handles = _offload(_stack())
    with hold_forward_residency(handles):
        with pytest.raises(RuntimeError, match="nested"):
            with hold_forward_residency(handles):
                pass
    assert _ExpertOffload._hold is False


def test_refuses_a_call_inside_a_backward():
    handles = _offload(_stack())
    seen = {}

    class Probe(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, g):
            try:
                with hold_forward_residency(handles):
                    pass
            except RuntimeError as e:
                seen["err"] = str(e)
            return g

    x = torch.ones(2, requires_grad=True)
    Probe.apply(x).sum().backward()
    assert "inside a backward" in seen.get("err", "")
    assert _ExpertOffload._hold is False
