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
import os
import struct
import sys

import pytest
import torch

pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series modules")
pytest.importorskip("nvme_residency")

from nvme_arena import bake_expert_tensors, load_index, verify  # noqa: E402
from nvme_residency import ColdTier  # noqa: E402

from experts4bit_qlora import Experts4bit  # noqa: E402
from experts4bit_qlora.nvme_experts import NF4_SEGMENTS, _TieredStack  # noqa: E402

# NF4 blocks must tile each expert exactly, so BOTH in_features dims
# (gate_up's hidden, down's intermediate) must be multiples of blocksize 64.
E, I, H = 8, 64, 128         # experts, intermediate, hidden
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
    gate_up = torch.randn(E, 2 * I, H, generator=g, dtype=torch.float32) * 0.05
    down = torch.randn(E, H, I, generator=g, dtype=torch.float32) * 0.05
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


def test_wrong_arena_names_the_missing_segment(arena, tmp_path):
    """A relocation arena of the WRONG kinds must fail loudly, not serve noise."""
    mod, _path, _index = arena
    gu_p, *_ = _stacks(mod)
    tensors = {}
    for e in range(E):
        t = gu_p[e].contiguous().cpu()
        tensors[f"model.layers.0.mlp.experts.{e}.other.blocks"] = (
            tuple(t.shape), "U8", t.numpy().tobytes())
    snap = tmp_path / "snap2"; snap.mkdir()
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
    from experts4bit_qlora.hot_residency import _HotResidency, hot_residency_available
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
    from experts4bit_qlora.nvme_experts import _NvmeResidency

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
