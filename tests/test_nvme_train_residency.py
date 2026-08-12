"""The gate on training through the NVMe tier: an arena-staged layer must be
BYTE-IDENTICAL to the same layer staged from pinned host RAM.

That is the whole correctness claim, and it decomposes cleanly. The fused
training path (`enable_fast_train` -> `nf4_qlora`) already has its own gates —
gradient tests, the 48-layer parity run — and it is unchanged here. What changed
is only where the frozen bytes come from. So if the bytes an arena stage hands
the kernel equal the bytes a host stage hands it, every downstream claim carries
over unaltered; and if they differ, no downstream test could rescue it.

Everything below runs on CPU with no CUDA and no triton, because the comparison
is over staged tensors rather than kernel output. A GPU forward/backward parity
arm belongs on a GPU box and is NOT in this file — it would skip silently here,
and a skipped gate reads as a passing one.

The second thing under test is the guard. Routed staging fills only the routed
rows of a full-shaped stack, so a backward that reads an unstaged row gets
uninitialized memory: finite, plausible, wrong. `assert_rows_staged` is what
turns that into an exception, and it is negative-controlled here — a gate that
cannot fire is not a gate.
"""
import json
import struct

import pytest
import torch

pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series modules")
pytest.importorskip("nvme_residency")
pytest.importorskip("bitsandbytes", reason="Experts4bit quantization needs bnb")

from nvme_arena import bake_expert_tensors, load_index  # noqa: E402
from nvme_residency import ColdTier  # noqa: E402

# The staging entry point this whole module depends on. Skip loudly on an older
# grouped-nf4-gemm rather than failing 16 tests with an AttributeError apiece.
pytest.importorskip("nvme_residency").segment_into

from experts4bit_qlora import Experts4bit  # noqa: E402
from experts4bit_qlora.lora import ExpertsLoRA  # noqa: E402
from experts4bit_qlora.engines.offload import _ExpertOffload, enable_expert_offload  # noqa: E402
from experts4bit_qlora.engines.nvme_train import (  # noqa: E402
    OFFLOAD_SEGMENTS,
    _ArenaExpertOffload,
    arena_train_stats,
    enable_nvme_train_residency,
)

# NF4 blocks must tile each expert exactly, so both contracted dims (gate_up's
# hidden, down's intermediate) must be multiples of blocksize 64.
E, INTER, H = 8, 64, 128
LAYER = 0
KINDS = tuple(OFFLOAD_SEGMENTS.values())
ROUTED = [3, 1]                     # out of order, a strict subset


def _st_bytes(tensors: dict) -> bytes:
    """Minimal safetensors writer (the format nvme_arena's header reader parses)."""
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def _module(seed=1689, inter=INTER, hidden=H, n_exp=E):
    g = torch.Generator().manual_seed(seed)
    gate_up = torch.randn(n_exp, 2 * inter, hidden, generator=g, dtype=torch.float32) * 0.05
    down = torch.randn(n_exp, hidden, inter, generator=g, dtype=torch.float32) * 0.05
    return Experts4bit.from_float(gate_up.to(torch.bfloat16), down.to(torch.bfloat16),
                                  has_gate=True, activation=torch.nn.functional.silu,
                                  quant_type="nf4", compute_dtype=torch.bfloat16)


def _per_expert_stacks(mod):
    """The module's four packed tensors, reshaped per expert exactly as the arena
    segments carry them. Sized from the MODULE, not the file-level ``E``, so a
    fixture with a different expert count reshapes correctly."""
    e = mod.num_experts
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    return {"gate_up_proj": mod.gate_up_proj.view(e, n1, k1 // 2),
            "gate_up_absmax": mod.gate_up_absmax.view(e, n1, k1 // 64),
            "down_proj": mod.down_proj.view(e, n2, k2 // 2),
            "down_absmax": mod.down_absmax.view(e, n2, k2 // 64)}


def _bake(mod, tmp_path, name="m.arena", n_layers=1):
    """Relocate the module's OWN quantized stacks into an arena, per expert, so
    the arena bytes are the module's bytes and any difference later is tiering.

    ``n_layers`` repeats the same experts across that many arena layers — enough
    for tests that need a model with several MoE modules, where layer ``i`` must
    exist for module ``i``.
    """
    dt = {torch.uint8: "U8", torch.float32: "F32"}
    tensors = {}
    for attr, stack in _per_expert_stacks(mod).items():
        kind = OFFLOAD_SEGMENTS[attr]
        for lay in range(n_layers):
            for e in range(mod.num_experts):
                t = stack[e].contiguous().cpu()
                tensors[f"model.layers.{lay}.mlp.experts.{e}.{kind}"] = (
                    tuple(t.shape), dt[t.dtype], t.numpy().tobytes())
    snap = tmp_path / f"snap-{name}"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(_st_bytes(tensors))
    path = str(tmp_path / name)
    bake_expert_tensors(
        str(snap), path,
        name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
        kinds=KINDS, align=4096, log=lambda *a: None)
    return path, load_index(path)


@pytest.fixture()
def arena(tmp_path):
    mod = _module()
    path, index = _bake(mod, tmp_path)
    return mod, path, index


def _arena_handle(mod, tier, layer=LAYER, device="cpu"):
    """An ExpertsLoRA over `mod` with an arena-backed offload handle installed."""
    mod._e4b_cold_tier = tier
    mod._e4b_arena_layer = layer
    wrapper = ExpertsLoRA(mod, r=4, alpha=8)
    handle = enable_expert_offload(wrapper, device, pin=False,
                                   handle_cls=_ArenaExpertOffload)
    return wrapper, handle


# --------------------------------------------------------------- THE claim --

@pytest.mark.parametrize("hot_rows", [E, 2])
def test_arena_stage_is_bitwise_identical_to_a_host_stage(arena, hot_rows):
    """Stage the same experts both ways; every byte must match.

    hot_rows=2 forces eviction and re-read between requests, so a stale or
    partially-filled slot shows up here as a bitwise difference rather than as a
    plausible number some later test would have to catch. It also means the
    experts have to be staged in chunks that FIT: a request whose unique rows
    exceed hot_rows is refused by the tier, since every slot in one request is
    protected from eviction by the others.
    """
    mod, path, index = arena
    host_mod = _module()                       # same seed -> same quantized bytes
    host = enable_expert_offload(ExpertsLoRA(host_mod, r=4, alpha=8), "cpu", pin=False)

    # Only ONE layer may be GPU-resident at a time, so staging the arena arm
    # evicts the host arm. Snapshot the host's staged bytes before comparing --
    # reading them afterwards reads placeholders, which is how the first cut of
    # this test "passed" a shape check against a 0-element tensor.
    host.stage()
    ref = {n: getattr(host.base, n).clone()
           for n in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")}
    assert all(t.numel() > 0 for t in ref.values()), "host arm staged nothing"

    with ColdTier(path, hot_rows=hot_rows, pinned=False, index=index) as tier:
        _w, arena_h = _arena_handle(mod, tier)
        for lo in range(0, E, hot_rows):
            chunk = list(range(lo, min(lo + hot_rows, E)))
            arena_h.evict()                    # force a cold re-stage each round
            arena_h.stage_routed(chunk)
            for n, want in ref.items():
                got = getattr(arena_h.base, n)
                assert got.dtype == want.dtype and got.shape == want.shape, n
                for e in chunk:
                    assert torch.equal(got[e], want[e]), (
                        f"{n}[{e}]: arena-staged bytes differ from host-staged "
                        f"(hot_rows={hot_rows})")


def test_routed_stage_matches_the_host_on_the_routed_rows(arena):
    """Routed staging is the path training actually takes. The rows it claims to
    have staged must equal the host's; the rows it did not are out of contract
    and deliberately not asserted (they hold whatever the allocator left)."""
    mod, path, index = arena
    host_mod = _module()
    host = enable_expert_offload(ExpertsLoRA(host_mod, r=4, alpha=8), "cpu", pin=False)
    host.stage()
    ref = {n: getattr(host.base, n).clone()
           for n in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")}
    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        _w, arena_h = _arena_handle(mod, tier)
        arena_h.stage_routed(ROUTED)
        assert arena_h._staged_ids == frozenset(ROUTED)
        for n, want in ref.items():
            for e in ROUTED:
                assert torch.equal(getattr(arena_h.base, n)[e], want[e]), f"{n}[{e}]"


def test_routed_stage_reads_only_the_routed_rows_from_disk(arena):
    """Measure the quantity that matters. Staging the right BYTES while reading
    every row off the device would pass every equality test above and defeat the
    entire point, so count the tier's reads rather than trusting the intent."""
    mod, path, index = arena
    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        _w, arena_h = _arena_handle(mod, tier)
        before = tier.stats()["disk_reads"]
        arena_h.stage_routed(ROUTED)
        after = tier.stats()["disk_reads"]
    # One aligned row read per unique routed expert -- not per segment (the four
    # segments share a row) and not per expert in the layer.
    assert after - before == len(set(ROUTED)), (
        f"{after - before} disk reads for {len(set(ROUTED))} routed experts")


def test_a_request_larger_than_the_pinned_budget_is_refused(arena):
    """The hard floor `hot_rows` imposes, surfaced where a caller meets it. Every
    slot in one request is protected from eviction by the others, so a forward
    routing more unique experts than the budget cannot be served -- and raising
    is the right answer, since the alternative is thrashing that looks like
    working. A training batch routes far more experts than decode's top-k, which
    is exactly when this bites."""
    mod, path, index = arena
    with ColdTier(path, hot_rows=2, pinned=False, index=index) as tier:
        _w, handle = _arena_handle(mod, tier)
        with pytest.raises(ValueError, match="exceeds hot_rows"):
            handle.stage_routed(list(range(E)))


def test_a_second_stage_of_the_same_experts_re_reads_nothing(arena):
    """Residency has to actually hold: the tier is the thing making repeat access
    cheap, and a handle that bypassed it would look correct and be slow."""
    mod, path, index = arena
    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        _w, arena_h = _arena_handle(mod, tier)
        arena_h.stage_routed(ROUTED)
        mid = tier.stats()["disk_reads"]
        arena_h.evict()
        arena_h.stage_routed(ROUTED)
        assert tier.stats()["disk_reads"] == mid
        assert tier.stats()["hits"] >= len(set(ROUTED))


# ------------------------------------------------------- the host-RAM floor --

def test_a_meta_base_never_materializes_expert_storage(arena):
    """The reason this exists. A model whose experts exceed host RAM has nowhere
    to put an [E, ...] home, so the handle must work over a base built on `meta`
    and must not allocate one itself."""
    _mod, path, index = arena
    from experts4bit_qlora import build_meta_experts

    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        meta_base = build_meta_experts(index, E, has_gate=True,
                                       compute_dtype=torch.bfloat16)
        assert meta_base.gate_up_proj.is_meta
        _w, handle = _arena_handle(meta_base, tier)
        # Homes carry shape and dtype only -- no storage anywhere.
        for n, t in handle.home.items():
            assert t.is_meta, f"home {n} materialized {t.numel()} elements"
        assert handle.home["gate_up_proj"].shape == (E, 2 * INTER * H // 2)
        # And it can still serve: staging allocates only the destination.
        handle.stage_routed(ROUTED)
        assert not handle.base.gate_up_proj.is_meta
        assert handle.base.gate_up_proj.shape == (E, 2 * INTER * H // 2)


def test_bulk_stage_goes_through_the_same_read_path(arena):
    """`stage()` (no routing visible) must stage every row and say so. Two read
    implementations is how `_staged_ids` and reality drift apart, which would
    make the backward guard unsound in exactly the case it exists for."""
    mod, path, index = arena
    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        _w, handle = _arena_handle(mod, tier)
        handle.stage()
        assert handle._staged_ids == frozenset(range(E))
        assert handle.staged and handle._last_stage_nbytes > 0
        handle.assert_rows_staged(range(E))       # covers everything


# ------------------------------------------------------------- the guard --

def test_guard_refuses_an_expert_that_was_not_staged(arena):
    """NEGATIVE CONTROL. This is the failure the design can still have: a
    recompute that routes differently reads a row nothing filled. Uninitialized
    memory is finite and plausible, so nothing downstream would notice."""
    mod, path, index = arena
    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        _w, handle = _arena_handle(mod, tier)
        handle.stage_routed([0, 1])
        handle.assert_rows_staged([0, 1])         # the staged set is accepted
        with pytest.raises(RuntimeError, match=r"experts \[5\].*not staged"):
            handle.assert_rows_staged([0, 5])


def test_guard_refuses_a_read_while_evicted(arena):
    """The no-gradient-checkpointing case: the evict post-hook fired when the
    forward returned and nothing re-staged the layer, so backward is reading
    0-element placeholders. The message has to name the cause."""
    mod, path, index = arena
    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        _w, handle = _arena_handle(mod, tier)
        handle.stage_routed(ROUTED)
        handle.evict()
        assert handle._staged_ids == frozenset()
        with pytest.raises(RuntimeError, match="gradient checkpointing"):
            handle.assert_rows_staged(ROUTED)


def test_the_guard_is_installed_where_backward_will_call_it(arena):
    """The guard is only worth anything if `fast.py`'s weights_fn closures find
    it. They look it up as `base._e4b_stage_guard`, so enable must set exactly
    that name on exactly the base the closures capture."""
    mod, path, index = arena

    class _Model(torch.nn.Module):
        def __init__(self, wrapper):
            super().__init__()
            self.block = wrapper

    del index
    model = _Model(ExpertsLoRA(mod, r=4, alpha=8))
    n = enable_nvme_train_residency(model, path, hot_rows=E, device="cpu",
                                    pinned=False)
    assert n == 1
    guard = getattr(mod, "_e4b_stage_guard", None)
    assert guard is not None, "fast.py's closures would find nothing"
    assert guard == mod._e4b_arena_offload.assert_rows_staged
    with pytest.raises(RuntimeError, match="gradient checkpointing"):
        guard([0])                                # evicted at install
    stats = arena_train_stats(model)
    assert stats is not None and stats["modules"] == 1
    mod._e4b_cold_tier.close()


# ------------------------------------------------------------- refusals --

def test_an_arena_from_another_model_is_refused(tmp_path):
    """A geometry cross-check, not a guess. The two byte counts are close enough
    that the copies would succeed and produce garbage, so the mismatch has to be
    caught at attach time and named."""
    other = _module(seed=99, inter=INTER * 2)     # different expert geometry
    path, index = _bake(other, tmp_path, name="other.arena")
    mod = _module()                               # the model we mean to train
    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        with pytest.raises(ValueError, match="does not match this model"):
            _arena_handle(mod, tier)


def test_an_arena_with_a_different_expert_count_is_refused(tmp_path):
    """The per-expert width check does NOT cover this, and the silent direction
    is the dangerous one.

    Two variants of one architecture that differ only in expert count have
    IDENTICAL per-expert geometry, so every other check passes. If the module has
    FEWER experts than the arena, every id resolves and expert `e` reads another
    model's expert `e` — real bytes, right shapes, wrong weights, no error. (The
    other direction eventually KeyErrors in `row_offset`, but only once a
    high-numbered expert is routed, which can be many steps into a run.)

    Found by Cursor Bugbot on #117.
    """
    wide = _module(n_exp=E * 2)                 # same per-expert shape, 2x experts
    path, index = _bake(wide, tmp_path, name="wide.arena")
    narrow = _module()                          # the model we mean to train
    # The dangerous direction, and the one every other check passes.
    assert narrow.num_experts < index["n_experts_per_layer"], "test is not testing"
    assert (narrow.gate_up_proj.shape[1] == wide.gate_up_proj.shape[1]), \
        "per-expert width must MATCH, or the existing geometry check catches it"
    with ColdTier(path, hot_rows=4, pinned=False, index=index) as tier:
        with pytest.raises(ValueError, match="expert count mismatch"):
            _arena_handle(narrow, tier)


def test_an_arena_layer_out_of_range_is_refused(arena):
    """Same class: `row_offset` would KeyError at the first stage rather than at
    attach, and only for the layers that are actually routed."""
    mod, path, index = arena
    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        with pytest.raises(ValueError, match="out of range"):
            _arena_handle(mod, tier, layer=index["n_layers"] + 5)


def test_a_second_enable_is_refused_and_leaks_no_tier(arena):
    """`enable_expert_offload` is idempotent and keeps the tier it was built
    with, so a second enable would construct a fresh ColdTier, hand back the OLD
    handle, and orphan the new tier — while appearing to retune `hot_rows`.

    Found by Cursor Bugbot on #117.
    """
    mod, path, _index = arena

    class _Model(torch.nn.Module):
        def __init__(self, wrapper):
            super().__init__()
            self.block = wrapper

    model = _Model(ExpertsLoRA(mod, r=4, alpha=8))
    assert enable_nvme_train_residency(model, path, hot_rows=E, device="cpu",
                                       pinned=False) == 1
    first_tier = mod._e4b_cold_tier
    with pytest.raises(RuntimeError, match="already have arena-backed"):
        enable_nvme_train_residency(model, path, hot_rows=E * 2, device="cpu",
                                    pinned=False)
    # The refusal must happen BEFORE a second tier is opened, so the module still
    # points at the original and nothing was allocated to be cleaned up.
    assert mod._e4b_cold_tier is first_tier
    assert mod._e4b_arena_offload._tier is first_tier
    first_tier.close()


def _model_of(*mods):
    class _Model(torch.nn.Module):
        def __init__(self, wrappers):
            super().__init__()
            self.blocks = torch.nn.ModuleList(wrappers)
    return _Model([ExpertsLoRA(m, r=4, alpha=8) for m in mods])


def _pool_is_shutdown(tier) -> bool:
    """Whether ``tier``'s reader was closed, by BEHAVIOUR rather than a flag: a
    shut-down ThreadPoolExecutor refuses new work.

    Two earlier cuts of this check were vacuous, each confirmed by removing the
    close and watching the test still pass — a `_closed` attribute that does not
    exist (`hasattr`-guarded, i.e. `assert True`), then `_fds == []`, which is
    equally true of a FRESH reader because fds open lazily per worker on the
    first read.
    """
    try:
        tier.reader._pool.submit(int).result()
        return False
    except RuntimeError:
        return True


def test_a_geometry_refusal_opens_no_tier_at_all(tmp_path):
    """Everything the geometry check needs is in the arena INDEX, which reads
    without fds or pinned memory. So a mismatched arena must be refused before a
    ColdTier exists — there is then nothing to leak and nothing to unwind."""
    other = _module(seed=99, inter=INTER * 2)
    path, _index = _bake(other, tmp_path, name="mismatch.arena")
    mod = _module()
    model = _model_of(mod)
    with pytest.raises((ValueError, TypeError), match="does not match this model"):
        enable_nvme_train_residency(model, path, hot_rows=E, device="cpu",
                                    pinned=False)
    assert not hasattr(mod, "_e4b_cold_tier"), \
        "a tier was opened (and stamped) for an arena that was going to be refused"


def test_a_failure_before_any_module_attaches_closes_the_tier(arena, monkeypatch):
    """Past the pre-flight, a failure on the FIRST module leaves a tier nobody is
    using. That one must be closed."""
    mod, path, _index = arena
    model = _model_of(mod)
    import experts4bit_qlora.engines.nvme_train as nt
    monkeypatch.setattr(nt, "enable_expert_offload",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        enable_nvme_train_residency(model, path, hot_rows=E, device="cpu",
                                    pinned=False)
    assert _pool_is_shutdown(mod._e4b_cold_tier), "tier leaked with nobody holding it"


def test_a_partial_attach_leaves_the_tier_open_for_live_modules(tmp_path, monkeypatch):
    """The regression Bugbot caught in the first fix: closing on ANY failure
    shut the tier under modules that had already attached and were holding it.

    Closing a live tier is strictly worse than leaking one — the earlier modules
    get a shut-down reader on their first stage, and the re-enable guard blocks
    recovery on that model. So past the first success the tier stays OPEN and the
    error says the model must be discarded.
    """
    mod_a, mod_b = _module(), _module(seed=4242)
    path, _index = _bake(mod_a, tmp_path, name="two-layer.arena", n_layers=2)
    model = _model_of(mod_a, mod_b)

    import experts4bit_qlora.engines.nvme_train as nt
    real = nt.enable_expert_offload
    calls = {"n": 0}

    def _flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("boom on the second module")
        return real(*a, **k)

    monkeypatch.setattr(nt, "enable_expert_offload", _flaky)
    with pytest.raises(RuntimeError, match="partially-attached"):
        enable_nvme_train_residency(model, path, hot_rows=E, device="cpu",
                                    pinned=False)
    tier = mod_a._e4b_cold_tier
    assert not _pool_is_shutdown(tier), \
        "the tier was closed under a module that had already attached to it"
    # ...and the attached module can still actually use it.
    mod_a._e4b_arena_offload.stage_routed(ROUTED)
    assert mod_a._e4b_arena_offload.staged
    tier.close()


def test_a_partial_attach_chains_the_real_cause(tmp_path, monkeypatch):
    """The partial-attach message describes CLEANUP, not the failure. Raising it
    `from None` would leave the caller with a note about tier lifetime and no idea
    why the attach died — an OOM, a driver error, a bad handle class.

    Found by Cursor Bugbot on #117.
    """
    mod_a, mod_b = _module(), _module(seed=4242)
    path, _index = _bake(mod_a, tmp_path, name="chain.arena", n_layers=2)
    model = _model_of(mod_a, mod_b)

    import experts4bit_qlora.engines.nvme_train as nt
    real, calls = nt.enable_expert_offload, {"n": 0}

    def _flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1:
            raise MemoryError("CUDA out of memory (the real cause)")
        return real(*a, **k)

    monkeypatch.setattr(nt, "enable_expert_offload", _flaky)
    with pytest.raises(RuntimeError, match="partially-attached") as ei:
        enable_nvme_train_residency(model, path, hot_rows=E, device="cpu",
                                    pinned=False)
    assert isinstance(ei.value.__cause__, MemoryError), \
        "the real failure was suppressed; the caller sees only the cleanup note"
    assert "real cause" in str(ei.value.__cause__)
    mod_a._e4b_cold_tier.close()


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_a_partial_attach_does_not_swallow_control_flow(tmp_path, monkeypatch, exc):
    """KeyboardInterrupt and SystemExit are control flow, not attach failures.
    Wrapping them in a RuntimeError makes the interpreter refuse to exit and reads
    as a bug in this function — so they pass through untouched, while the tier
    still stays open for the modules already holding it."""
    mod_a, mod_b = _module(), _module(seed=4242)
    path, _index = _bake(mod_a, tmp_path, name=f"ctrlc-{exc.__name__}.arena",
                         n_layers=2)
    model = _model_of(mod_a, mod_b)

    import experts4bit_qlora.engines.nvme_train as nt
    real, calls = nt.enable_expert_offload, {"n": 0}

    def _flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1:
            raise exc()
        return real(*a, **k)

    monkeypatch.setattr(nt, "enable_expert_offload", _flaky)
    with pytest.raises(exc):
        enable_nvme_train_residency(model, path, hot_rows=E, device="cpu",
                                    pinned=False)
    assert not _pool_is_shutdown(mod_a._e4b_cold_tier)
    mod_a._e4b_cold_tier.close()


def test_a_bare_module_with_no_adapter_is_refused(arena):
    """Without a LoRA wrapper there is nothing to train, and the caller almost
    certainly wanted the serving path. Say which one."""
    mod, path, index = arena

    class _Model(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.block = m

    with ColdTier(path, hot_rows=E, pinned=False, index=index):
        with pytest.raises(RuntimeError, match="enable_nvme_residency"):
            enable_nvme_train_residency(_Model(mod), path, hot_rows=E,
                                        device="cpu", pinned=False)


def test_the_device_expert_cache_is_refused_rather_than_ignored(arena):
    """`enable_expert_cache` fills its rows from a pinned host home this handle
    does not have. Silently ignoring the pool would make it dead code — the exact
    bug `_copy_routed_to_device` already shipped once."""
    mod, path, index = arena
    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        _w, handle = _arena_handle(mod, tier)
        handle._pool = object()                   # any non-None pool
        with pytest.raises(RuntimeError, match="expert cache"):
            handle.stage_routed(ROUTED)


def test_the_resident_slot_is_shared_across_handle_classes(arena):
    """At most ONE layer may be device-resident, and that has to hold when a model
    mixes handle classes. The slot is a class attribute, so writing it through
    `type(self)` binds a SECOND slot on the subclass and leaves the base class
    pointing at a handle nothing will evict — two resident layers, silently, which
    is the memory bound this policy exists to hold.
    """
    mod, path, index = arena
    host_mod = _module()
    host = enable_expert_offload(ExpertsLoRA(host_mod, r=4, alpha=8), "cpu", pin=False)
    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        _w, arena_h = _arena_handle(mod, tier)
        host.stage()
        assert host.staged
        arena_h.stage_routed(ROUTED)             # must evict the host handle
        assert arena_h.staged
        assert not host.staged, "two layers resident at once"
        assert host.base.gate_up_proj.numel() == 0, "evicted handle still holds storage"
        # ...and back the other way.
        host.stage()
        assert not arena_h.staged, "two layers resident at once (arena not evicted)"


def test_host_offload_still_takes_the_conservative_path(arena):
    """The shared pre-hook gained a `_routed_in_train` escape. A host-RAM handle
    must not have taken it: its bulk stage is a cheap whole-layer copy and it has
    no guard installed, so routed staging in training would be unguarded."""
    _mod, _path, _index = arena
    assert _ExpertOffload._routed_in_train is False
    assert _ArenaExpertOffload._routed_in_train is True


def test_segment_into_is_required_and_named(monkeypatch, arena):
    """An older grouped-nf4-gemm has the tier but not the staging entry point.
    Fail at enable, naming the missing symbol, rather than inside the first
    backward -- and name the SYMBOL rather than a version, which would be a
    forward reference to a release that may not be cut."""
    _mod, path, _index = arena
    import nvme_residency

    monkeypatch.delattr(nvme_residency, "segment_into")
    with pytest.raises(ImportError, match="segment_into"):
        enable_nvme_train_residency(torch.nn.Module(), path, hot_rows=E,
                                    device="cpu", pinned=False)
