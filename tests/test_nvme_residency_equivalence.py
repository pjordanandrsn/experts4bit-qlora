"""The gate that decides whether NVMe residency is real: serving the cold tail
from disk must produce the SAME answer as serving it from RAM.

Two levels, weakest assumption first:

1. **Stack equivalence (bitwise).** ``_TieredStack.index_select`` must return
   tensors bit-identical to the fully-materialized cold stack's. Needs no CUDA,
   so it runs everywhere and is the claim everything else rests on.
2. **Forward equivalence.** A tier-backed `_HotResidency` forward must match a
   fully-resident one. Needs the fused kernel (CUDA + [fast]), so it is skipped
   where unavailable — but the skip is loud, never silent.

The arena here is built by RELOCATION from the module's own quantized stacks, so
the bytes on disk are bit-identical to the bytes the engine would have held in
RAM. That keeps the provenance claim intact end to end: any divergence below is a
bug in the tiering, not an artifact of a re-quantization step.
"""
import json
import struct

import pytest
import torch

pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series modules")
pytest.importorskip("nvme_residency")

from nvme_arena import bake_expert_tensors, load_index, verify  # noqa: E402
from nvme_residency import ColdTier  # noqa: E402

from experts4bit_qlora import Experts4bit  # noqa: E402
from experts4bit_qlora.engines.nvme_experts import NF4_SEGMENTS, _TieredStack  # noqa: E402

# NF4 blocks must tile each expert exactly, so BOTH in_features dims
# (gate_up's hidden, down's intermediate) must be multiples of blocksize 64.
E, INTER, H = 8, 64, 128         # experts, intermediate, hidden
LAYER = 0
KINDS = tuple(NF4_SEGMENTS.values())


def _st_bytes(tensors: dict) -> bytes:
    """Minimal safetensors writer (same format nvme_arena's header reader parses)."""
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def _module():
    """A real Experts4bit built from deterministic bf16 weights."""
    g = torch.Generator().manual_seed(1689)
    gate_up = torch.randn(E, 2 * INTER, H, generator=g, dtype=torch.float32) * 0.05
    down = torch.randn(E, H, INTER, generator=g, dtype=torch.float32) * 0.05
    return Experts4bit.from_float(gate_up.to(torch.bfloat16), down.to(torch.bfloat16),
                                  has_gate=True, activation=torch.nn.functional.silu,
                                  quant_type="nf4", compute_dtype=torch.bfloat16)


def _stacks(mod):
    """The four per-expert views _HotResidency derives from a module."""
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    return (mod.gate_up_proj.view(E, n1, k1 // 2),
            mod.gate_up_absmax.view(E, n1, k1 // 64).float(),
            mod.down_proj.view(E, n2, k2 // 2),
            mod.down_absmax.view(E, n2, k2 // 64).float())


@pytest.fixture()
def arena(tmp_path):
    """Relocate the module's OWN quantized stacks into an arena, per expert."""
    mod = _module()
    gu_p, gu_a, dn_p, dn_a = _stacks(mod)
    payload = {"nf4.gate_up_blocks": gu_p, "nf4.gate_up_absmax": gu_a,
               "nf4.down_blocks": dn_p, "nf4.down_absmax": dn_a}
    dt = {torch.uint8: "U8", torch.float32: "F32"}
    tensors = {}
    for kind, stack in payload.items():
        for e in range(E):
            t = stack[e].contiguous().cpu()
            tensors[f"model.layers.{LAYER}.mlp.experts.{e}.{kind}"] = (
                tuple(t.shape), dt[t.dtype], t.numpy().tobytes())
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(_st_bytes(tensors))
    arena_path = str(tmp_path / "m.arena")
    bake_expert_tensors(
        str(snap), arena_path,
        name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
        kinds=KINDS, align=4096, log=lambda *a: None)
    return mod, arena_path, load_index(arena_path)


def test_arena_relocation_is_verifiable(arena):
    """The bake's own gate: arena bytes match the source byte ranges."""
    _mod, path, _idx = arena
    rep = verify(path, log=lambda *a: None)
    assert rep["ok"] and rep["rows_checked"] == E


@pytest.mark.parametrize("hot_rows", [E, 2])
def test_tiered_stack_is_bitwise_equal_to_the_resident_stack(arena, hot_rows):
    """THE claim. hot_rows=2 forces eviction and re-read between requests, so a
    stale or partially-filled slot would show up as a bitwise difference."""
    mod, path, index = arena
    resident = dict(zip(NF4_SEGMENTS, _stacks(mod)))
    cold_ids = torch.arange(E)                     # treat every expert as cold
    with ColdTier(path, hot_rows=hot_rows, pinned=False, index=index) as tier:
        for attr, suffix in NF4_SEGMENTS.items():
            ts = _TieredStack(tier, index, LAYER, cold_ids, suffix)
            full = resident[attr]
            assert ts.shape == tuple(full.shape), (attr, ts.shape, full.shape)
            for routed in (torch.tensor([0]), torch.tensor([E - 1, 0]),
                           torch.arange(E) if hot_rows >= E else torch.tensor([1, 2])):
                got = ts.index_select(0, routed)
                ref = full.index_select(0, routed)
                assert got.dtype == ref.dtype, attr
                assert got.shape == ref.shape, attr
                assert torch.equal(got, ref), (
                    f"{attr}: tiered bytes differ from the resident stack "
                    f"(routed={routed.tolist()}, hot_rows={hot_rows})")


def test_absmax_arrives_as_float32_not_reinterpreted_bytes(arena):
    """The trap this path had to avoid: the engine holds absmax as float32. If a
    tiered stack handed back raw scale bytes the shapes would still look right."""
    mod, path, index = arena
    _gu_p, gu_a, _dn_p, _dn_a = _stacks(mod)
    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        ts = _TieredStack(tier, index, LAYER, torch.arange(E),
                          NF4_SEGMENTS["c_gu_a"])
        got = ts.index_select(0, torch.arange(E))
        assert got.dtype == torch.float32
        assert torch.equal(got, gu_a)
        assert got.abs().max() > 0, "absmax should not be all zeros"


def test_cold_gather_lands_the_right_expert_on_a_hybrid_partition(arena):
    """The cold branch's id algebra, on the shape the hybrid tier gives it.

    ``_HotResidency`` splits experts VRAM/not-VRAM, so on a three-tier manifest
    the cold stack spans DRAM experts too and the NVMe rows sit at cold-local
    ids with DRAM-shaped holes between them. This replays the exact chain a
    cold step walks — ``g2c_cpu`` -> ``torch.unique(..., return_inverse=True)``
    -> ``_TieredStack.index_select`` -> ``compact`` -> the kernel's row order —
    and checks each ROW against the expert it was routed to.

    Needs no CUDA and no kernel, which is the point: it is the cheap half of
    #171's question. A mis-indexed cold gather (the hypothesis that issue names)
    would land here, in CI, rather than only on a box with an arena and a GPU.
    """
    mod, path, index = arena
    resident = dict(zip(NF4_SEGMENTS, _stacks(mod)))
    # deliberately interleaved so a cold-local id is never its global id and
    # never its position among the NVMe experts either
    vram, nvme = [0, 3], [2, 4, 6, 7]                 # the rest, {1, 5}, is DRAM
    cold_ids = torch.tensor([e for e in range(E) if e not in vram])   # dram + nvme
    g2c = torch.full((E,), -1, dtype=torch.long)
    g2c[cold_ids] = torch.arange(cold_ids.numel())

    g = torch.Generator().manual_seed(171)
    with ColdTier(path, hot_rows=E, pinned=False, index=index) as tier:
        for attr, suffix in NF4_SEGMENTS.items():
            ts = _TieredStack(tier, index, LAYER, cold_ids, suffix)
            for _ in range(32):
                # one step's cold rows: NVMe globals, with repeats, unsorted
                pick = torch.randint(0, len(nvme), (int(torch.randint(1, 7, (1,), generator=g)),),
                                     generator=g)
                cold_glob = torch.tensor([nvme[int(i)] for i in pick])
                routed, compact = torch.unique(g2c.index_select(0, cold_glob),
                                               return_inverse=True)
                got = ts.index_select(0, routed)
                for row, e in enumerate(cold_glob.tolist()):
                    assert torch.equal(got[compact[row]], resident[attr][e]), (
                        f"{attr}: row {row} routed to expert {e} got another "
                        f"expert's bytes (cold_glob={cold_glob.tolist()}, "
                        f"routed={routed.tolist()})")


def test_wrong_arena_names_the_missing_segment(arena, tmp_path):
    """A relocation arena of the WRONG kinds must fail loudly, not serve noise."""
    mod, _path, _index = arena
    gu_p, *_ = _stacks(mod)
    tensors = {}
    for e in range(E):
        t = gu_p[e].contiguous().cpu()
        tensors[f"model.layers.0.mlp.experts.{e}.other.blocks"] = (
            tuple(t.shape), "U8", t.numpy().tobytes())
    snap = tmp_path / "snap2"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(_st_bytes(tensors))
    p2 = str(tmp_path / "b.arena")
    bake_expert_tensors(str(snap), p2,
                        name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
                        kinds=("other.blocks",), align=4096, log=lambda *a: None)
    idx2 = load_index(p2)
    with ColdTier(p2, hot_rows=E, pinned=False, index=idx2) as tier:
        with pytest.raises(KeyError, match="no segment"):
            _TieredStack(tier, idx2, 0, torch.arange(E), "nf4.gate_up_blocks")


# ----------------------------------------------------- forward equivalence --
@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel needs CUDA")
def test_forward_matches_fully_resident(arena):
    """End to end: identical partition, identical inputs, cold tail from NVMe."""
    from experts4bit_qlora.engines.hot_residency import _HotResidency, hot_residency_available
    if not hot_residency_available():
        pytest.skip("grouped-nf4-gemm [fast] kernel unavailable")
    # `hot_residency_available()` can be True while the kernel still cannot JIT:
    # grouped-nf4-gemm needs triton>=3.4 for `tl.gather`, and an older triton
    # fails inside the dependency walker with an AttributeError rather than a
    # clean feature error. Probe explicitly so the skip names the real reason
    # instead of surfacing as a mystery failure.
    import triton.language as _tl
    if not hasattr(_tl, "gather"):
        import triton
        pytest.skip(f"fused kernel needs triton>=3.4 for tl.gather; "
                    f"this box has {triton.__version__}")
    from experts4bit_qlora.engines.nvme_experts import _NvmeResidency

    mod, path, index = arena
    mod = mod.to("cuda")
    hot = torch.tensor([0, 1])                     # 2 hot, 6 cold
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(5, H, generator=g, dtype=torch.float32) * 0.1).to(
        "cuda", torch.bfloat16)
    top_idx = torch.tensor([[2, 3], [4, 5], [0, 6], [7, 1], [3, 7]], device="cuda")
    top_w = torch.full((5, 2), 0.5, device="cuda", dtype=torch.bfloat16)

    ref = _HotResidency(mod, hot, "cuda").forward(x, top_idx, top_w)

    # hot_rows must be >= the unique cold experts routed in ONE forward:
    # _cold_contrib fetches them all in a single index_select. Here that is 6
    # ({2,3,4,5,6,7}). Eviction correctness is covered by the stack-level test
    # at hot_rows=2; this test is about the forward answer.
    with ColdTier(path, hot_rows=6, pinned=True, index=index) as tier:
        mod._e4b_cold_tier = tier
        mod._e4b_arena_layer = LAYER
        got = _NvmeResidency(mod, hot, "cuda").forward(x, top_idx, top_w)
        st = tier.stats()
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert torch.equal(got, ref), (
        f"NVMe-backed forward diverges from fully-resident "
        f"(max abs diff {(got.float() - ref.float()).abs().max().item():.3e})")
    assert st["misses"] > 0, "test never exercised a disk read"


# ------------------- zero expert RAM: the module holds only shapes -----------
def _meta_module():
    """Same geometry as `_module()` but on `meta`: expert buffers are unallocated.
    This is the shape a K3-scale load must take — 1.446 TB of experts cannot be
    materialized, so nothing may index the module's own [E, ...] storage."""
    m = Experts4bit(num_experts=E, hidden_dim=H, intermediate_dim=INTER,
                    has_gate=True, activation=torch.nn.functional.silu,
                    quant_type="nf4", compute_dtype=torch.bfloat16, device="meta")
    assert m.gate_up_proj.is_meta, "expert buffers must not be allocated"
    return m


def test_meta_module_allocates_no_expert_storage(arena):
    """Guard the premise: a meta module's expert buffers have no backing storage.

    `is_meta` is the real signal — a meta storage still REPORTS its nominal size
    (`untyped_storage().nbytes()` returns the would-be byte count), so that number
    is not a memory measurement and must not be used as one.
    """
    m = _meta_module()
    for name in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax"):
        buf = getattr(m, name)
        assert buf.is_meta, f"{name} is materialized"
        assert buf.device.type == "meta", f"{name} on {buf.device}"
    # and the shapes are still real, which is all the engine needs from it
    assert m._gate_up_shape == (2 * INTER, H) and m._down_shape == (H, INTER)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel needs CUDA")
def test_forward_from_meta_module_matches_materialized(arena):
    """The K3-shaped claim: with the module on `meta` and BOTH partitions served
    from the arena, the forward must equal a fully-materialized reference.

    If this holds, expert storage is no longer a function of model size — only
    the hot set and the routed working set are.
    """
    from experts4bit_qlora.engines.hot_residency import _HotResidency, hot_residency_available
    if not hot_residency_available():
        pytest.skip("grouped-nf4-gemm [fast] kernel unavailable")
    import triton.language as _tl
    if not hasattr(_tl, "gather"):
        import triton
        pytest.skip(f"needs triton>=3.4 for tl.gather; box has {triton.__version__}")
    from experts4bit_qlora.engines.nvme_experts import _NvmeResidency

    mod, path, index = arena
    hot = torch.tensor([0, 1])
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(5, H, generator=g, dtype=torch.float32) * 0.1).to(
        "cuda", torch.bfloat16)
    top_idx = torch.tensor([[2, 3], [4, 5], [0, 6], [7, 1], [3, 7]], device="cuda")
    top_w = torch.full((5, 2), 0.5, device="cuda", dtype=torch.bfloat16)

    ref = _HotResidency(mod.to("cuda"), hot, "cuda").forward(x, top_idx, top_w)

    meta = _meta_module()                      # zero expert bytes
    with ColdTier(path, hot_rows=6, pinned=True, index=index) as tier:
        meta._e4b_cold_tier = tier
        meta._e4b_arena_layer = LAYER
        st = _NvmeResidency(meta, hot, "cuda")
        # hot came from the arena, not from the (meta) module
        assert st.h_gu_p.shape[0] == 2 and not st.h_gu_p.is_meta
        got = st.forward(x, top_idx, top_w)
        stats = tier.stats()
    assert torch.equal(got, ref), (
        "meta-module NVMe forward diverges from the materialized reference "
        f"(max abs diff {(got.float() - ref.float()).abs().max().item():.3e})")
    assert stats["misses"] > 0, "never read from disk"


# --------------- geometry from the arena, not from the config ---------------
def test_expert_geometry_recovered_from_the_arena(arena):
    """The arena is the authority on expert shape. For a LATENT MoE (Kimi K3) the
    expert input width is the latent (3584), not hidden_size (7168) — reading the
    config would size every expert twice too wide."""
    from experts4bit_qlora.engines.nvme_experts import expert_geometry_from_arena
    _mod, _path, index = arena
    assert expert_geometry_from_arena(index) == (INTER, H)


def test_inconsistent_arena_geometry_is_rejected():
    """gate_up and down over-determine (I, H); a mismatch must raise rather than
    silently pick one and build wrongly-shaped experts."""
    from experts4bit_qlora.engines.nvme_experts import expert_geometry_from_arena
    bad = {"segments": [
        {"suffix": "nf4.gate_up_blocks", "shape_per_expert": [2 * INTER, H // 2]},
        {"suffix": "nf4.down_blocks", "shape_per_expert": [H, (INTER // 2) + 8]},
    ]}
    with pytest.raises(ValueError, match="inconsistent"):
        expert_geometry_from_arena(bad)
    with pytest.raises(KeyError, match="arena lacks"):
        expert_geometry_from_arena({"segments": []})


def test_build_meta_experts_matches_the_real_module_shapes(arena):
    """A shapes-only module must be interchangeable with the materialized one as
    far as the engine's shape algebra is concerned."""
    from experts4bit_qlora.engines.nvme_experts import build_meta_experts
    mod, _path, index = arena
    meta = build_meta_experts(index, E, compute_dtype=torch.bfloat16)
    assert meta._gate_up_shape == mod._gate_up_shape
    assert meta._down_shape == mod._down_shape
    assert meta.num_experts == mod.num_experts
    assert meta.gate_up_proj.is_meta and meta.down_proj.is_meta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel needs CUDA")
def test_end_to_end_shapes_only_module_forward(arena):
    """The assembled path: geometry from the arena, module from shapes alone, both
    partitions served from disk, answer identical to the materialized reference."""
    from experts4bit_qlora.engines.hot_residency import _HotResidency, hot_residency_available
    if not hot_residency_available():
        pytest.skip("grouped-nf4-gemm [fast] kernel unavailable")
    import triton.language as _tl
    if not hasattr(_tl, "gather"):
        import triton
        pytest.skip(f"needs triton>=3.4 for tl.gather; box has {triton.__version__}")
    from experts4bit_qlora.engines.nvme_experts import _NvmeResidency, build_meta_experts

    mod, path, index = arena
    hot = torch.tensor([0, 1])
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(5, H, generator=g, dtype=torch.float32) * 0.1).to(
        "cuda", torch.bfloat16)
    top_idx = torch.tensor([[2, 3], [4, 5], [0, 6], [7, 1], [3, 7]], device="cuda")
    top_w = torch.full((5, 2), 0.5, device="cuda", dtype=torch.bfloat16)
    ref = _HotResidency(mod.to("cuda"), hot, "cuda").forward(x, top_idx, top_w)

    meta = build_meta_experts(index, E, compute_dtype=torch.bfloat16)
    with ColdTier(path, hot_rows=6, pinned=True, index=index) as tier:
        meta._e4b_cold_tier = tier
        meta._e4b_arena_layer = LAYER
        got = _NvmeResidency(meta, hot, "cuda").forward(x, top_idx, top_w)
        assert tier.stats()["misses"] > 0
    assert torch.equal(got, ref), "assembled shapes-only path diverges"


# ----------------- Bugbot #42: module selection must be SHARED ---------------
def test_target_modules_includes_lora_bases_and_keeps_order(arena):
    """Regression: `enable_nvme_residency` used to walk modules itself, which
    disagreed with `enable_hot_residency` whenever LoRA-wrapped and bare modules
    were interleaved — stamping real MoE layers with the wrong arena layer index.
    Both must derive the target list from one implementation.

    That shared list now INCLUDES ExpertsLoRA bases. It excluded them originally
    because `ExpertsLoRA.forward` never calls `base.forward`, making a patch dead
    code — but `ExpertsLoRA._delegate_to_base` does hand the forward to the base
    once an engine is attached and the adapter is provably zero, and excluding
    them left the pipelined engine unreachable for every model the streaming
    loader returns. Membership means "targetable and index-bearing"; an engine
    that is not delegated to (v0 hot residency, the NVMe engines) skips wrapped
    bases itself, consuming the entry so `hot_sets[i]` keeps its meaning.
    """
    from experts4bit_qlora.lora import ExpertsLoRA
    from experts4bit_qlora.engines.hot_residency import target_modules, wrapped_bases
    mod, _path, index = arena

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # interleaved on purpose: bare, wrapped, bare
            self.a = _meta_module()
            self.b = ExpertsLoRA(_module(), r=4, alpha=8, dtype=torch.bfloat16)
            self.c = _meta_module()

    net = Net()
    targets = target_modules(net)
    naive = [m for m in net.modules() if isinstance(m, type(mod))]
    # one shared enumeration: it now agrees with the naive sweep on membership...
    assert len(naive) == len(targets), (len(naive), len(targets))
    bases = wrapped_bases(net)
    assert bases, "fixture no longer has a wrapped base — the test would be vacuous"
    assert any(id(t) in bases for t in targets), "wrapped base must be targetable"
    # ...and the ordering guarantee is what hot_sets[i] actually rides on
    assert targets[0] is net.a and targets[-1] is net.c
    assert targets[1] is net.b.base


def test_enable_nvme_residency_refuses_a_partial_stamp(arena):
    """More hot_sets than targetable modules must raise, not stamp a prefix and
    serve the remaining layers from whatever rows happen to be there."""
    from experts4bit_qlora.engines.nvme_experts import enable_nvme_residency
    _mod, path, _index = arena

    class One(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _meta_module()

    with pytest.raises(ValueError, match="targetable MoE module"):
        enable_nvme_residency(One(), path, [torch.tensor([0]), torch.tensor([1])],
                              hot_rows=4, device="cpu")


def test_the_partial_stamp_refusal_allocates_no_tier(arena, monkeypatch):
    """The refusal must fire before a ColdTier exists, asserted the only way that
    cannot be faked: make constructing one an error.

    Two things went wrong while it did not. On a host with no accelerator
    `ColdTier` pins its landing buffer and raised "Cannot access accelerator
    device" FIRST, so a caller who passed too many hot_sets got an allocator
    error instead of the message naming their mistake — and the test above failed
    for a reason unrelated to what it checks. On a host that does pin, the
    refusal leaked the tier: constructed, never closed.

    It went unnoticed because this whole module was `importorskip`ped on CI until
    `grouped-nf4-gemm` was added to `[test]`. It had only ever run on a laptop,
    where the failure was misread as a macOS quirk rather than the ordering bug
    it is.
    """
    import nvme_residency
    from experts4bit_qlora.engines.nvme_experts import enable_nvme_residency
    _mod, path, _index = arena

    class One(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _meta_module()

    def _boom(*_a, **_k):
        raise AssertionError("a ColdTier was allocated before the refusal fired")

    monkeypatch.setattr(nvme_residency, "ColdTier", _boom)
    with pytest.raises(ValueError, match="targetable MoE module"):
        enable_nvme_residency(One(), path, [torch.tensor([0]), torch.tensor([1])],
                              hot_rows=4, device="cpu")


def _pool_is_shutdown(tier) -> bool:
    """Whether the tier's reader was closed, by behaviour: a shut-down pool
    refuses new work. Asserted this way rather than via a flag because the
    obvious `_fds == []` is equally true of a fresh reader — fds open lazily on
    the first read."""
    try:
        tier.reader._pool.submit(int).result()
        return False
    except RuntimeError:
        return True


def _capture_tier(monkeypatch, seen):
    """Record the ColdTier `enable_nvme_residency` builds.

    Patched on `nvme_residency`, NOT on the engine module: the engine imports
    ColdTier INSIDE the function, so a name bound on the engine is never read.
    (An earlier cut patched the engine and the capture simply never ran — the
    test failed with KeyError rather than passing vacuously, which is the good
    outcome, but the seam still has to be the one the code uses.) The real class
    is captured before patching so the wrapper does not recurse into itself.
    """
    import nvme_residency
    real = nvme_residency.ColdTier

    def _capture(*a, **k):
        seen["tier"] = real(*a, **k)
        return seen["tier"]

    monkeypatch.setattr(nvme_residency, "ColdTier", _capture)


def test_a_failure_before_anything_is_patched_closes_the_tier(arena, monkeypatch):
    """Nobody holds the tier, so it must not be leaked."""
    import experts4bit_qlora.engines.nvme_experts as ne
    from experts4bit_qlora.engines.nvme_experts import enable_nvme_residency
    _mod, path, _index = arena
    seen = {}

    class One(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _meta_module()

    _capture_tier(monkeypatch, seen)
    monkeypatch.setattr(ne, "enable_hot_residency",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        enable_nvme_residency(One(), path, [torch.tensor([0])], hot_rows=4,
                              device="cpu", pinned=False)
    assert _pool_is_shutdown(seen["tier"]), "tier leaked with nobody holding it"


def test_a_partial_patch_leaves_the_tier_open_for_live_modules(arena, monkeypatch):
    """The regression Bugbot caught: `enable_hot_residency` patches one module at
    a time, so a later failure leaves earlier modules serving from this shared
    tier. Closing it under them is worse than leaking it.

    Identical policy to `nvme_train`'s attach loop, where Bugbot made the same
    finding first.
    """
    import experts4bit_qlora.engines.nvme_experts as ne
    from experts4bit_qlora.engines.nvme_experts import enable_nvme_residency
    _mod, path, _index = arena
    seen = {}

    class Two(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = _meta_module(), _meta_module()

    def _patch_one_then_fail(model, hot_sets, **k):
        # Mimic the real thing: mark the first module live, then die.
        mods = [m for m in model.modules() if hasattr(m, "_e4b_cold_tier")]
        mods[0]._e4b_hot_ref = mods[0].forward
        raise MemoryError("CUDA out of memory (the real cause)")

    _capture_tier(monkeypatch, seen)
    monkeypatch.setattr(ne, "enable_hot_residency", _patch_one_then_fail)
    with pytest.raises(RuntimeError, match="partially attached") as ei:
        enable_nvme_residency(Two(), path, [torch.tensor([0]), torch.tensor([1])],
                              hot_rows=4, device="cpu", pinned=False)
    assert not _pool_is_shutdown(seen["tier"]), \
        "the tier was closed under a module already serving from it"
    assert isinstance(ei.value.__cause__, MemoryError), \
        "the real failure was suppressed behind the cleanup note"
    seen["tier"].close()


def test_a_stale_hot_ref_does_not_pass_for_a_live_holder(arena, monkeypatch):
    """`_e4b_hot_ref` is STICKY — it survives an earlier enable. Reading it bare
    answers "has this module ever been patched", not "does it hold the tier this
    call just built", and that difference is a leak: a stale marker reads as a
    live holder, the close is skipped, and the new pinned arena is left with
    nothing serving from it.

    Found by Cursor Bugbot on #120, which also named the fix — `nvme_train` counts
    what it attached rather than trusting a persistent attribute.
    """
    import experts4bit_qlora.engines.nvme_experts as ne
    from experts4bit_qlora.engines.nvme_experts import enable_nvme_residency
    _mod, path, _index = arena
    seen = {}

    class One(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _meta_module()

    model = One()
    # A marker left over from some EARLIER enable, before this tier existed.
    model.a._e4b_hot_ref = model.a.forward

    _capture_tier(monkeypatch, seen)
    monkeypatch.setattr(ne, "enable_hot_residency",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    # The real cause must survive: nothing of THIS call is live, so this is the
    # plain re-raise path, not the partial-attach wrap.
    with pytest.raises(RuntimeError, match="boom"):
        enable_nvme_residency(model, path, [torch.tensor([0])], hot_rows=4,
                              device="cpu", pinned=False)
    assert _pool_is_shutdown(seen["tier"]), \
        "a stale hot-ref was mistaken for a live holder and the new tier leaked"


def test_a_short_layers_list_is_refused_before_any_tier(arena, monkeypatch):
    """The sibling refusal on the same path, held to the same rule."""
    import nvme_residency
    from experts4bit_qlora.engines.nvme_experts import enable_nvme_residency
    _mod, path, _index = arena

    class One(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _meta_module()

    def _boom(*_a, **_k):
        raise AssertionError("a ColdTier was allocated before the refusal fired")

    monkeypatch.setattr(nvme_residency, "ColdTier", _boom)
    with pytest.raises(ValueError, match="layers has"):
        enable_nvme_residency(One(), path, [torch.tensor([0])],
                              hot_rows=4, device="cpu", layers=[])


# ---- Bugbot: enable_mxfp4_nvme_residency guards (the MXFP4 lane had NO coverage) ----
# The NF4 lane above is guarded and tested; its MXFP4 counterpart shipped with neither,
# which is how both bugs below got in. These need no arena reads and no CUDA — every guard
# fires before an engine is built — so a stub engine makes them CPU tests.
class _StubEngine:
    """Accepts the real call signature and does nothing. `tier`/`store` are read back
    off `engines[0]` to be shared across layers, so they must exist."""
    def __init__(self, *a, **k):
        self.tier = None
        self.store = None

    def forward(self, hidden, top_k_index, top_k_weights):   # pragma: no cover
        return hidden


def _two_layer_net():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _meta_module()
            self.b = _meta_module()
    return Net()


@pytest.mark.parametrize("n_sets", [1, 3, 0])
def test_mxfp4_hot_sets_length_must_match(arena, n_sets):
    """A SHORT list raises IndexError deep in the loop; a LONG one is worse — it silently
    applies each layer's set to the wrong layer, and an informed hot set on the wrong layer
    is a uniform random draw (the by-index row of the V4 table: 4.4 GiB of VRAM for +0%).
    Both must be refused up front, like the NF4 lane and `enable_hot_residency`."""
    pytest.importorskip("mxfp4_residency")
    from experts4bit_qlora.engines.nvme_experts import enable_mxfp4_nvme_residency
    _mod, path, _index = arena
    with pytest.raises(ValueError, match="exactly one entry per MoE layer"):
        enable_mxfp4_nvme_residency(
            _two_layer_net(), path, k_slots=2, hot_rows=4, limit=10.0, device="cpu",
            engine_cls=_StubEngine, hot_sets=[()] * n_sets)


@pytest.mark.parametrize("ref,name", [
    ("_e4b_hot_ref", "hot residency"), ("_e4b_pipe_ref", "pipelined residency"),
    ("_e4b_cold_ref", "cold engine"), ("_e4b_fast_ref", r"\[fast\]"),
])
def test_mxfp4_refuses_to_patch_over_another_engine(arena, ref, name):
    """All five engines replace `mod.forward`. Patching over a live one strands its state
    and makes the OTHER engine's `disable_*` restore a forward that is still patched."""
    pytest.importorskip("mxfp4_residency")
    from experts4bit_qlora.engines.nvme_experts import enable_mxfp4_nvme_residency
    _mod, path, _index = arena
    net = _two_layer_net()
    setattr(net.b, ref, net.b.forward)          # pretend that engine is enabled
    with pytest.raises(RuntimeError, match=name):
        enable_mxfp4_nvme_residency(net, path, k_slots=2, hot_rows=4, limit=10.0,
                                    device="cpu", engine_cls=_StubEngine)


def test_mxfp4_reenable_then_disable_fully_restores(arena):
    """Re-enabling (new hot_sets, reloaded checkpoint) must not re-capture the ALREADY
    PATCHED forward as the restore point — otherwise one disable leaves the module wrapped
    around a stale engine forever, and it looks fine until the arena is closed."""
    pytest.importorskip("mxfp4_residency")
    from experts4bit_qlora.engines.nvme_experts import (
        disable_mxfp4_nvme_residency, enable_mxfp4_nvme_residency)
    _mod, path, _index = arena
    net = _two_layer_net()
    pristine = [net.a.forward, net.b.forward]

    for hot in ([(), ()], [(0,), (1,)]):        # enable, then re-enable with new hot sets
        assert enable_mxfp4_nvme_residency(
            net, path, k_slots=2, hot_rows=4, limit=10.0, device="cpu",
            engine_cls=_StubEngine, hot_sets=hot) == 2

    assert disable_mxfp4_nvme_residency(net) == 2
    for mod, orig in zip((net.a, net.b), pristine):
        assert mod.forward == orig, "restored a PATCHED forward, not the original"
        assert not hasattr(mod, "_e4b_mxfp4_ref")
        assert not hasattr(mod, "_e4b_mxfp4_engine")


@pytest.mark.parametrize("enabler,mod_name,kwargs", [
    ("hot_residency", "enable_hot_residency", {}),
    ("pipelined", "enable_pipelined_residency", {"k_slots": 2}),
    ("cold_engine", "enable_cold_engine", {}),
])
def test_other_engines_refuse_a_module_the_mxfp4_engine_owns(arena, enabler, mod_name, kwargs):
    """The other direction of the same matrix: every engine already refused the other
    three, but none knew about `_e4b_mxfp4_ref`, so they would patch straight over it."""
    pytest.importorskip("nf4_grouped")
    import importlib
    enable = getattr(importlib.import_module(f"experts4bit_qlora.{enabler}"), mod_name)
    net = _two_layer_net()
    for m in (net.a, net.b):
        m._e4b_mxfp4_ref = m.forward            # the mxfp4 engine owns these
    patched = enable(net, [torch.tensor([0]), torch.tensor([0])], device="cpu", **kwargs)
    assert patched == 0, f"{mod_name} patched over a module the mxfp4 engine owns"
