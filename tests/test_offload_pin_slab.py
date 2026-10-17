"""Exact-size pinned slabs for offloaded expert homes (``E4B_OFFLOAD_PIN_SLAB=1``, default off).

torch's pinned allocator rounds every request up to a power of two, so pinning a layer's four homes separately costs up to 1.33x
their bytes. In slab mode the four homes are views into one exact-size, page-aligned pageable buffer pinned in place with one
``cudaHostRegister`` per layer. The load-bearing properties, checked on CPU with the registration calls stood in for:
- the layout: one slab per layer, page-aligned, each home at a 512-byte-aligned offset, every home bitwise equal to its source;
- one registration per layer, and the pages unregistered before the slab's memory can go (on release, on collection, and when a
  copy fails after registration); nothing left registered after the model is gone;
- a refused registration falls back to per-tensor pinning, with the reason recorded;
- off by default, and exclusive with ``E4B_OFFLOAD_ARENA``.
On a CUDA device the real registration is checked: every view reads as pinned, and staging lands the same bytes as the default.
"""
import ctypes
import gc
import mmap
import weakref

import pytest
from quant_guard import require_quantize

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import Experts4bit, ExpertsLoRA, enable_expert_offload  # noqa: E402
from experts4bit_qlora.engines import offload as off  # noqa: E402
from experts4bit_qlora.engines.offload import _ExpertOffload  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
E, HID, INTER = 4, 64, 64


@pytest.fixture(autouse=True)
def _reset():
    _ExpertOffload._resident = None
    _ExpertOffload._staged_now = set()
    yield
    _ExpertOffload._resident = None
    _ExpertOffload._staged_now = set()
    gc.collect()


def _build(seed=0, device=DEVICE):
    torch.manual_seed(seed)
    gate_up = (torch.randn(E, 2 * INTER, HID) * 0.1).to(device)
    down = (torch.randn(E, HID, INTER) * 0.1).to(device)
    require_quantize(device)
    base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=torch.float32)
    return ExpertsLoRA(base, r=4, alpha=8, dtype=torch.float32).to(device)


class _FakeCudart:
    """Stands in for ``torch.cuda.cudart()``: records each call, and at unregister checks the slab's memory is still held."""

    def __init__(self, fail_register=False):
        self.registered, self.unregistered, self.fail_register = [], [], fail_register
        self.alive_at_unregister = []
        self.watch = {}                                   # ptr -> weakref to the slab's backing tensor (set by the test)

    def cudaHostRegister(self, ptr, size, flags):
        if self.fail_register:
            return 2                                      # cudaErrorMemoryAllocation
        self.registered.append((ptr, size, flags))
        return 0

    def cudaHostUnregister(self, ptr):
        self.unregistered.append(ptr)
        w = self.watch.get(ptr)
        self.alive_at_unregister.append(w is None or w() is not None)
        return 0


@pytest.fixture
def fake(monkeypatch):
    f = _FakeCudart()
    monkeypatch.setattr(off, "_cudart", lambda: f)
    monkeypatch.setattr(off, "_host_device_pointer", lambda ptr: (ptr, "fake"))   # unified addressing: the device reads the host address
    monkeypatch.setattr(off, "_REGISTERED", {})
    monkeypatch.setenv("E4B_OFFLOAD_PIN_SLAB", "1")
    monkeypatch.delenv("E4B_OFFLOAD_ARENA", raising=False)
    return f


def test_slab_layout_one_page_aligned_slab_with_aligned_bitwise_homes(fake):
    lora = _build()
    want = {n: getattr(lora.base, n).detach().to("cpu").clone() for n in _ExpertOffload._names()}
    h = enable_expert_offload(lora, DEVICE, pin=True)
    assert h.pin_mode == "slab" and h.slab_why is None and h._arena_layout is None
    slab = h._slab
    page = mmap.PAGESIZE
    assert slab.ptr % page == 0 and slab.nbytes % page == 0
    assert fake.registered == [(slab.ptr, slab.nbytes, off._HOST_REGISTER_FLAGS)]      # one portable, mapped registration per layer
    assert off._HOST_REGISTER_FLAGS == 3 and slab.device_readable
    total = 0
    for n in _ExpertOffload._names():
        v = h.home[n]
        assert v.untyped_storage().data_ptr() == slab.tensor.untyped_storage().data_ptr()   # all four in the one slab
        assert (v.data_ptr() - slab.ptr) % off._SLAB_ALIGN == 0
        assert v.dtype == want[n].dtype and v.shape == want[n].shape and torch.equal(v, want[n])
        total = -(-total // off._SLAB_ALIGN) * off._SLAB_ALIGN + v.numel() * v.element_size()
    assert slab.nbytes == max(page, -(-total // page) * page)              # exact: the bytes, aligned, to the page
    assert h._stage_nbytes == sum(want[n].numel() * want[n].element_size() for n in want)
    assert h._stage_ncopies == 4                                           # staging unchanged: one copy per home


def test_unregistered_before_the_memory_goes_and_nothing_left_registered(fake):
    lora = _build()
    h = enable_expert_offload(lora, DEVICE, pin=True)
    ptr = h._slab.ptr
    fake.watch[ptr] = weakref.ref(h._slab._finalizer.peek()[2][1])     # the backing tensor the finalizer holds (args[1])
    assert set(off._REGISTERED) == {ptr}
    del h, lora
    gc.collect()
    assert fake.unregistered == [ptr] and fake.alive_at_unregister == [True]
    assert fake.watch[ptr]() is None                                       # and then the memory did go
    assert off._REGISTERED == {}


def test_the_finalizer_does_not_run_at_interpreter_exit(fake):
    slab = off._PinnedSlab(4096)
    assert slab._finalizer.atexit is False                                 # the runtime may be gone at exit; the OS takes the pages
    slab.release()


def test_device_readability_is_recorded_per_slab(fake, monkeypatch):
    """A slab the device reads at its host address may be read in place; one it does not (a different device address, or no
    answer from the runtime) may not, and in-place readers must then refuse rather than copy."""
    ok = off._PinnedSlab(8192)
    view = ok.tensor[512:1024]
    assert ok.device_readable and off._slab_of(view)[2] and off._device_readable_in_place(view)
    for answer in (None, 0xDEAD000):
        monkeypatch.setattr(off, "_host_device_pointer", lambda ptr, a=answer: (a, None if a is None else "fake"))
        bad = off._PinnedSlab(8192)
        v = bad.tensor[:64]
        assert not bad.device_readable and off._slab_of(v) == (bad.ptr, bad.nbytes, False)
        assert off._is_pinned(v) and not off._device_readable_in_place(v)   # pinned, but not for a kernel's in-place read
        bad.release()
    ok.release()
    assert off._slab_of(view) is None and not off._device_readable_in_place(torch.empty(16, dtype=torch.uint8))


def test_release_is_idempotent(fake):
    slab = off._PinnedSlab(10_000)
    slab.release()
    slab.release()
    del slab
    gc.collect()
    assert len(fake.unregistered) == 1 and off._REGISTERED == {}


def test_a_copy_failing_after_registration_unregisters_then_raises(fake, monkeypatch):
    lora = _build()
    real_copy = torch.Tensor.copy_
    calls = {"n": 0}

    def flaky(self, src, *a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("copy failed")
        return real_copy(self, src, *a, **k)

    monkeypatch.setattr(torch.Tensor, "copy_", flaky)
    with pytest.raises(RuntimeError, match="copy failed"):
        _ExpertOffload._build_slab_homes(lora.base, _ExpertOffload._names())
    monkeypatch.setattr(torch.Tensor, "copy_", real_copy)
    assert len(fake.registered) == 1 and fake.unregistered == [fake.registered[0][0]] and off._REGISTERED == {}


def test_a_refused_registration_falls_back_to_per_tensor_pinning(fake):
    fake.fail_register = True
    lora = _build()
    h = enable_expert_offload(lora, DEVICE, pin=True)
    assert h.pin_mode == "per-tensor" and h._slab is None and "cudaError 2" in h.slab_why
    assert off._REGISTERED == {} and fake.unregistered == []
    assert all(h.home[n].untyped_storage().data_ptr() != h.home["gate_up_proj"].untyped_storage().data_ptr()
               for n in ("down_proj", "gate_up_absmax", "down_absmax"))                       # separate tensors, as before


def test_off_by_default_and_exclusive_with_the_arena(monkeypatch, fake):
    monkeypatch.delenv("E4B_OFFLOAD_PIN_SLAB")
    h = enable_expert_offload(_build(), DEVICE, pin=True)
    assert h.pin_mode == "per-tensor" and h._slab is None and fake.registered == []
    h2 = enable_expert_offload(_build(), DEVICE, pin=False)
    assert h2.pin_mode == "pageable" and fake.registered == []          # pin=False never takes the slab, flag or not
    monkeypatch.setenv("E4B_OFFLOAD_PIN_SLAB", "1")
    monkeypatch.setenv("E4B_OFFLOAD_ARENA", "1")
    with pytest.raises(ValueError, match="exclusive"):
        enable_expert_offload(_build(), DEVICE, pin=True)


@cuda
def test_on_cuda_the_views_are_pinned_and_staging_lands_the_same_bytes(monkeypatch):
    monkeypatch.delenv("E4B_OFFLOAD_ARENA", raising=False)
    monkeypatch.delenv("E4B_OFFLOAD_PIN_SLAB", raising=False)
    ref = enable_expert_offload(_build(), "cuda", pin=True)
    monkeypatch.setenv("E4B_OFFLOAD_PIN_SLAB", "1")
    h = enable_expert_offload(_build(), "cuda", pin=True)
    # torch's is_pinned knows only its own allocator's blocks; the registered views count through offload's _is_pinned
    assert h.pin_mode == "slab" and h.pinned and all(off._is_pinned(h.home[n]) for n in h.home)
    for handle in (ref, h):
        handle._copy_home_to_device()
    torch.cuda.synchronize()
    for n in _ExpertOffload._names():
        a = getattr(ref.base, n) if n in ref._param_names else ref.base._buffers[n]
        b = getattr(h.base, n) if n in h._param_names else h.base._buffers[n]
        assert a.is_cuda and b.is_cuda and torch.equal(a, b)
    for handle in (ref, h):
        handle.evict()
    ptr = h._slab.ptr
    assert ptr in off._REGISTERED
    del h, handle, a, b                                                    # every reference to the slab handle
    gc.collect()
    assert ptr not in off._REGISTERED


@cuda
def test_on_cuda_a_copy_from_a_slab_is_asynchronous():
    """Pinned in behaviour, not by flag: a 256 MiB non_blocking H2D from the slab is still in flight when the call returns, and
    its enqueue takes a small fraction of the copy. From pageable memory the enqueue lasts most of the copy (the control: the
    driver stages pageable memory through its own buffer before the call returns)."""
    import time
    n = 256 << 20
    slab = off._PinnedSlab(n)
    dev = torch.empty(n, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.current_stream()

    def enqueue_share(src):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dev.copy_(src, non_blocking=True)
        t_enq = time.perf_counter() - t0
        in_flight = not stream.query()
        torch.cuda.synchronize()
        return t_enq / (time.perf_counter() - t0), in_flight

    enqueue_share(slab.tensor[:n])                                         # warm-up
    share, in_flight = enqueue_share(slab.tensor[:n])
    pageable_share, _ = enqueue_share(torch.empty(n, dtype=torch.uint8))
    slab.release()
    assert in_flight and share < 0.2, (share, in_flight)
    assert pageable_share > 0.5, pageable_share


def _pipelined_mod(E_=8, H_=128, INTER_=64):
    from experts4bit_qlora import Experts4bit
    torch.manual_seed(0)
    gu = torch.randn(E_, 2 * INTER_, H_)
    dn = torch.randn(E_, H_, INTER_)
    return Experts4bit.from_float(gate_up_proj=gu.cuda(), down_proj=dn.cuda(), compute_dtype=torch.bfloat16, has_gate=True)


def _needs_pipelined():
    pytest.importorskip("triton")
    pytest.importorskip("nf4_grouped")
    import os
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("raw-pointer gather is compiled-only")


@cuda
def test_on_cuda_pipelined_reads_slab_homes_in_place_and_matches_the_arena(monkeypatch):
    """The flag exists to save host RAM: with slab homes the pipelined engine must take the in-place path (no second arena) and
    read the same bytes as the copied-arena control."""
    _needs_pipelined()
    from experts4bit_qlora.engines.pipelined import disable_pipelined_residency, enable_pipelined_residency
    monkeypatch.delenv("E4B_OFFLOAD_ARENA", raising=False)
    torch.manual_seed(1)
    E_, H_, K_ = 8, 128, 3
    hs = torch.randn(1, H_, dtype=torch.bfloat16, device="cuda")
    w, idx = torch.topk(torch.softmax(torch.randn(1, E_, device="cuda"), -1), k=K_, dim=-1)
    wts = w.to(torch.bfloat16)
    for hot in ([], [0, 2, 5]):
        monkeypatch.setenv("E4B_OFFLOAD_PIN_SLAB", "1")
        seg_mod = _pipelined_mod()
        handle = enable_expert_offload(seg_mod, "cuda", pin=True)
        assert handle.pin_mode == "slab" and handle._slab.device_readable and handle._slab.device_pointer_via
        assert enable_pipelined_residency(seg_mod, [torch.tensor(hot, dtype=torch.long)], device="cuda", k_slots=K_) == 1
        st = seg_mod._pipelined
        assert st.arena is None and st.seg_srcs is not None and len(st.seg_srcs) == 4, "slab homes fell back to a copied arena"
        monkeypatch.delenv("E4B_OFFLOAD_PIN_SLAB")
        arena_mod = _pipelined_mod()
        assert enable_pipelined_residency(arena_mod, [torch.tensor(hot, dtype=torch.long)], device="cuda", k_slots=K_) == 1
        assert arena_mod._pipelined.arena is not None
        with torch.no_grad():
            got, want = seg_mod(hs, idx, wts), arena_mod(hs, idx, wts)
        assert torch.equal(got, want), f"hot={hot}: the slab read differs from the arena"
        disable_pipelined_residency(seg_mod)
        disable_pipelined_residency(arena_mod)


@cuda
def test_on_cuda_pipelined_refuses_a_slab_the_device_cannot_read_in_place(monkeypatch):
    """Fail closed: never a silent second arena."""
    _needs_pipelined()
    from experts4bit_qlora.engines.pipelined import enable_pipelined_residency
    monkeypatch.delenv("E4B_OFFLOAD_ARENA", raising=False)
    monkeypatch.setenv("E4B_OFFLOAD_PIN_SLAB", "1")
    monkeypatch.setattr(off, "_host_device_pointer", lambda ptr: (None, None))   # neither the driver nor the runtime answers
    mod = _pipelined_mod()
    handle = enable_expert_offload(mod, "cuda", pin=True)
    assert handle.pin_mode == "slab" and not handle._slab.device_readable
    with pytest.raises(RuntimeError, match="E4B_OFFLOAD_PIN_SLAB"):
        enable_pipelined_residency(mod, [torch.tensor([], dtype=torch.long)], device="cuda", k_slots=3)



class _LookupFn:
    """A C entry point that writes the address it is asked about into its out-parameter, as a mapped registration answers
    under unified addressing (or returns ``rc`` without writing)."""

    def __init__(self, rc=0):
        self.rc = rc

    def __call__(self, out, ptr, flags):
        if self.rc == 0:
            out._obj.value = ptr.value
        return self.rc


def _fake_cdll(libs):
    """A stand-in for ``ctypes.CDLL``: ``libs`` maps a soname (``None`` = the process) to ``{symbol: fn}``; any other soname is
    not loadable, and any other symbol is missing."""
    class Lib:
        def __init__(self, name):
            if name not in libs:
                raise OSError(f"{name}: cannot open shared object file")
            self.syms = libs[name]

        def __getattr__(self, sym):
            if sym not in self.syms:
                raise AttributeError(sym)
            return self.syms[sym]
    return Lib


@pytest.mark.parametrize("libs, via", [
    # the driver API answers first, under the one soname every CUDA version ships
    ({"libcuda.so.1": {"cuMemHostGetDevicePointer_v2": _LookupFn()}}, "libcuda.so.1:cuMemHostGetDevicePointer_v2"),
    # a CUDA 13 torch with no driver library on the path: only libcudart.so.13 answers (the soname the first lookup missed)
    ({"libcudart.so.13": {"cudaHostGetDevicePointer": _LookupFn()}}, "libcudart.so.13:cudaHostGetDevicePointer"),
    # the driver refuses (CUDA_ERROR_INVALID_VALUE), so the runtime is asked next
    ({"libcuda.so.1": {"cuMemHostGetDevicePointer_v2": _LookupFn(rc=1)},
      "libcudart.so.12": {"cudaHostGetDevicePointer": _LookupFn()}}, "libcudart.so.12:cudaHostGetDevicePointer"),
    # nothing answers: no device pointer, and the slab reads not device-readable (pipelined residency refuses it)
    ({}, None),
])
def test_the_device_pointer_lookup_tries_the_driver_then_every_runtime_soname(monkeypatch, libs, via):
    monkeypatch.setattr(ctypes, "CDLL", _fake_cdll(libs))
    host = 0x7F0000001000
    dptr, got = off._host_device_pointer(host)
    assert got == via
    assert dptr == (host if via is not None else None)
