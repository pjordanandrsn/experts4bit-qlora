"""Workstream-A offload instrumentation: per-copy transfer stats (``E4B_OFFLOAD_STATS``) and copy
consolidation (``E4B_OFFLOAD_ARENA``).

Both are default-off diagnostics. The load-bearing property is that **arena mode changes only the
number of H2D copies, never the bytes that land** — an offloaded forward is bit-identical arena-on
vs off, and identical to a non-offloaded forward (same guarantee `test_offload.py` makes for the
non-arena path). Stats accounting (bytes, copy counts, cold misses) is asserted on CPU by stubbing
CUDA-event timing; the GB/s reduction itself is CUDA-only and lightly smoke-tested there.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import Experts4bit, ExpertsLoRA  # noqa: E402
from experts4bit_qlora import enable_expert_offload, enable_inference_prefetch  # noqa: E402
from experts4bit_qlora import offload_stats_report, reset_offload_stats  # noqa: E402
from experts4bit_qlora import offload as offload_mod  # noqa: E402
from experts4bit_qlora.offload import _ExpertOffload  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
_QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)
E, HID, INTER, TOP_K, N_TOK = 4, 64, 64, 2, 8


@pytest.fixture(autouse=True)
def _reset_offload_class_state():
    _ExpertOffload._resident = None
    _ExpertOffload._staged_now = set()
    reset_offload_stats()
    yield
    _ExpertOffload._resident = None
    _ExpertOffload._staged_now = set()
    reset_offload_stats()


def _build(seed=0):
    torch.manual_seed(seed)
    gate_up = (torch.randn(E, 2 * INTER, HID) * 0.1).to(DEVICE)
    down = (torch.randn(E, HID, INTER) * 0.1).to(DEVICE)
    try:
        base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=torch.float32)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")
    return ExpertsLoRA(base, r=4, alpha=8, dtype=torch.float32).to(DEVICE)


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(N_TOK, HID, dtype=torch.float32, generator=g).to(DEVICE)
    idx = torch.randint(0, E, (N_TOK, TOP_K), generator=g).to(DEVICE)
    wts = torch.rand(N_TOK, TOP_K, dtype=torch.float32, generator=g).to(DEVICE)
    return hs, idx, wts


# ---------------------------------------------------------------------------------------------------
# Arena: same math, fewer copies.
# ---------------------------------------------------------------------------------------------------
def test_arena_layout_packs_four_homes_into_two_dtype_arenas(monkeypatch):
    monkeypatch.setenv("E4B_OFFLOAD_ARENA", "1")
    lora = _build()
    handle = enable_expert_offload(lora, DEVICE, pin=False)
    # uint8 packed weights + float32 absmax -> exactly two arenas; a stage issues 2 copies, not 4.
    assert handle._arena_layout is not None and len(handle._arena_cpu) == 2
    assert handle._stage_ncopies == 2
    # Every home is a correctly-shaped view onto its arena (state_dict + readers see per-tensor shapes).
    for name in _ExpertOffload._names():
        dt, s, e, shape = handle._arena_layout[name]
        assert handle.home[name].shape == torch.Size(shape)
        assert handle.home[name].numel() == e - s


def test_arena_forward_bit_identical_to_nonarena_and_nonoffloaded(monkeypatch):
    lora = _build(seed=3)
    hs, idx, wts = _inputs()
    with torch.no_grad():
        ref = lora(hs, idx, wts)  # non-offloaded reference

    monkeypatch.setenv("E4B_OFFLOAD_ARENA", "1")
    enable_expert_offload(lora, DEVICE, pin=False)
    with torch.no_grad():
        out1 = lora(hs, idx, wts)
        out2 = lora(hs, idx, wts)  # stage->evict->stage round-trips through the arena
    assert torch.equal(ref, out1) and torch.equal(ref, out2)
    assert lora._offload.staged is False


def test_arena_full_state_dict_carries_real_weights(monkeypatch):
    """The state_dict hook must still substitute correctly-shaped CPU homes for the evicted
    placeholders — arena homes are views, so this checks the views serialize as full tensors."""
    monkeypatch.setenv("E4B_OFFLOAD_ARENA", "1")
    lora = _build(seed=5)
    enable_expert_offload(lora, DEVICE, pin=False)
    sd = lora.state_dict()
    for name in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax"):
        t = sd[f"base.{name}"]
        assert t.numel() > 0 and t.device.type == "cpu"


# ---------------------------------------------------------------------------------------------------
# Stats accounting — CPU-safe by stubbing CUDA-event timing.
# ---------------------------------------------------------------------------------------------------
class _FakeEvent:
    """Stand-in for torch.cuda.Event: monotonic fake clock so elapsed_time is deterministic on CPU."""

    _clock = 0.0

    def __init__(self, enable_timing=False):
        self.t = None

    def record(self, stream=None):
        _FakeEvent._clock += 1.0
        self.t = _FakeEvent._clock

    def elapsed_time(self, other):
        return other.t - self.t


def _install_fake_events(monkeypatch):
    _FakeEvent._clock = 0.0
    monkeypatch.setattr(offload_mod.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(offload_mod.torch.cuda, "synchronize", lambda *a, **k: None)
    # current_stream must be callable; its value is only passed to Event.record (stubbed).
    monkeypatch.setattr(offload_mod.torch.cuda, "current_stream", lambda *a, **k: None)


@pytest.mark.parametrize("arena", [False, True])
def test_stats_tally_bytes_and_copy_counts(monkeypatch, arena):
    monkeypatch.setenv("E4B_OFFLOAD_STATS", "1")
    if arena:
        monkeypatch.setenv("E4B_OFFLOAD_ARENA", "1")
    _install_fake_events(monkeypatch)

    lora = _build(seed=7)
    handle = enable_expert_offload(lora, DEVICE, pin=False)
    expect_bytes = sum(t.numel() * t.element_size() for t in handle.home.values())
    expect_copies_per_stage = 2 if arena else 4

    hs, idx, wts = _inputs()
    reset_offload_stats()  # ignore the enable-time evict; count only the forwards below
    n_forward = 3
    with torch.no_grad():
        for _ in range(n_forward):
            lora(hs, idx, wts)

    rep = offload_stats_report(log=None)
    assert rep is not None
    sync = rep["by_policy"]["sync"]
    assert sync["stages"] == n_forward
    assert sync["copies"] == n_forward * expect_copies_per_stage
    # Byte tally is exact regardless of arena (same tensors, packed differently).
    assert abs(sync["gb"] - n_forward * expect_bytes / 1e9) < 1e-12


@cuda
def test_stats_counts_cold_miss_on_prefetch_cold_start(monkeypatch):
    """A prefetch-policy forward whose layer wasn't pre-staged is a cold miss; the first layer of a
    fresh generate() is always cold. Two linked layers, one no_grad pass -> exactly one cold miss.
    CUDA-only: the prefetch path enters a real ``torch.cuda.stream()`` context (a side stream), so
    the fake-event stub can't drive it — real events, real reduction."""
    monkeypatch.setenv("E4B_OFFLOAD_STATS", "1")

    class _Two(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_build(seed=1), _build(seed=2)])
            self.eval()

        def forward(self, hs, idx, wts):
            for m in self.layers:
                hs = hs + m(hs, idx, wts)
            return hs

    model = _Two()
    handles = [enable_expert_offload(m, DEVICE, pin=False) for m in model.layers]
    enable_inference_prefetch(handles)
    reset_offload_stats()
    hs, idx, wts = _inputs()
    with torch.no_grad():
        model(hs, idx, wts)  # layer0 cold; layer0's pre-hook prefetches layer1 (a hit)

    rep = offload_stats_report(log=None)
    assert rep is not None and rep["cold_misses"] == 1
    assert "cold_miss" in rep["by_policy"] and "prefetch" in rep["by_policy"]


def test_stats_report_none_when_disabled():
    reset_offload_stats()
    assert offload_stats_report(log=None) is None  # nothing recorded, stats flag off


@cuda
def test_stats_report_reduces_on_real_cuda(monkeypatch):
    """Smoke: with real CUDA events, a few offloaded forwards produce a positive GB/s under 'sync'."""
    monkeypatch.setenv("E4B_OFFLOAD_STATS", "1")
    lora = _build(seed=11)
    enable_expert_offload(lora, DEVICE, pin=True)
    hs, idx, wts = _inputs()
    reset_offload_stats()
    with torch.no_grad():
        for _ in range(4):
            lora(hs, idx, wts)
    rep = offload_stats_report(log=None)
    assert rep is not None and rep["by_policy"]["sync"]["gbps"] > 0
