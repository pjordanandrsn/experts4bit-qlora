# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Hybrid three-tier executor (Phase 3): forward correctness against the
fully-resident reference across a real vram/dram/nvme split, the
disjoint-bus law (warm experts never touch the tier; DRAM stacks never
pinned), enable/disable hygiene, and delegation when a tier is empty."""

import json
import struct

import pytest
import torch

pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series")
pytest.importorskip("nvme_residency")
cpu_grouped = pytest.importorskip("cpu_grouped")

from nvme_arena import bake_expert_tensors, load_index  # noqa: E402

from experts4bit_qlora import Experts4bit  # noqa: E402
from experts4bit_qlora.engines.hybrid import (  # noqa: E402
    _split_oversize_groups,
    disable_hybrid_tier,
    enable_hybrid_tier,
    hybrid_available,
)
from experts4bit_qlora.engines.nvme_experts import NF4_SEGMENTS  # noqa: E402

E, INTER, H, K = 8, 64, 128, 2
LAYER = 0

needs_stack = pytest.mark.skipif(
    not hybrid_available(), reason="needs CUDA + gnf4_native CPU kernels"
)


def _st_bytes(tensors):
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


class _Wrap(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.experts = mod


def _module():
    g = torch.Generator().manual_seed(20260816)
    gate_up = torch.randn(E, 2 * INTER, H, generator=g) * 0.05
    down = torch.randn(E, H, INTER, generator=g) * 0.05
    return Experts4bit.from_float(gate_up.to(torch.bfloat16),
                                  down.to(torch.bfloat16), has_gate=True,
                                  activation=torch.nn.functional.silu,
                                  quant_type="nf4",
                                  compute_dtype=torch.bfloat16)


@pytest.fixture()
def arena(tmp_path):
    mod = _module()
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    payload = {
        "nf4.gate_up_blocks": mod.gate_up_proj.view(E, n1, k1 // 2),
        "nf4.gate_up_absmax": mod.gate_up_absmax.view(E, n1, k1 // 64).float(),
        "nf4.down_blocks": mod.down_proj.view(E, n2, k2 // 2),
        "nf4.down_absmax": mod.down_absmax.view(E, n2, k2 // 64).float(),
    }
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
        kinds=tuple(NF4_SEGMENTS.values()), align=4096, log=lambda *a: None)
    return mod, arena_path, load_index(arena_path)


def _manifest(vram, dram, nvme):
    tiers = {"vram": [[LAYER, e] for e in vram],
             "dram": [[LAYER, e] for e in dram],
             "nvme": [[LAYER, e] for e in nvme]}
    return {"schema": "e4b-placement/1",
            "tiers": tiers,
            "masses": {"vram_frac": 0.5, "dram_frac": 0.4, "nvme_frac": 0.1}}


def test_split_oversize_groups():
    sizes, eids = _split_oversize_groups([3, 19, 8], [4, 1, 7])
    assert sizes == [3, 8, 8, 3, 8]
    assert eids == [4, 1, 1, 1, 7]


@needs_stack
def test_forward_matches_fully_resident_reference(arena):
    mod, path, _ = arena
    ref_mod = _module().to("cuda")
    model = _Wrap(_module().to("cuda"))
    n = enable_hybrid_tier(model, path,
                           _manifest(vram=[0, 3], dram=[1, 4, 6], nvme=[2, 5, 7]),
                           hot_rows=E, verbose=False)
    assert n == 1
    try:
        torch.manual_seed(7)
        T = 5
        hidden = (torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.3)
        idx = torch.stack([torch.randperm(E, device="cuda")[:K]
                           for _ in range(T)])
        wts = torch.rand(T, K, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            got = model.experts(hidden, idx, wts)
            want = ref_mod(hidden, idx, wts)
        assert got.shape == want.shape and got.dtype == want.dtype
        # three rounding paths meet here (GPU-resident, GPU-streamed, CPU
        # fp32); agreement is tolerance-level by design and the tolerance is
        # the documented cross-placement one
        torch.testing.assert_close(got.float(), want.float(),
                                   atol=5e-2, rtol=5e-2)
    finally:
        disable_hybrid_tier(model)


@needs_stack
def test_buses_are_disjoint(arena):
    """Warm experts must NEVER be fetched from the tier, and the DRAM stacks
    must never be pinned — the two halves of the bus law."""
    mod, path, _ = arena
    model = _Wrap(_module().to("cuda"))
    enable_hybrid_tier(model, path,
                       _manifest(vram=[0], dram=[1, 4, 6],
                                 nvme=[2, 3, 5, 7]),
                       hot_rows=E)
    try:
        st = model.experts._hot_residency
        assert not st.d_gu_p.is_pinned() and not st.d_dn_p.is_pinned()
        base = st.tier_stats()
        # route ONLY dram + vram experts: the tier must see zero new traffic
        hidden = torch.randn(2, H, dtype=torch.bfloat16, device="cuda")
        idx = torch.tensor([[0, 1], [4, 6]], device="cuda")
        wts = torch.rand(2, K, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            model.experts(hidden, idx, wts)
        after = st.tier_stats()
        assert after.get("misses", 0) == base.get("misses", 0), (
            "a DRAM-tier expert reached the NVMe tier — bus law violated")
        # and routing an nvme expert DOES reach the tier (the check can fail)
        idx2 = torch.tensor([[2, 5]], device="cuda")
        with torch.no_grad():
            model.experts(hidden[:1], idx2, wts[:1])
        assert st.tier_stats().get("misses", 0) > after.get("misses", 0)
    finally:
        disable_hybrid_tier(model)


@needs_stack
@needs_stack
def test_enable_failure_leaves_no_stamps_or_pool(arena):
    """A vram/dram overlap raises from construction; every stamp must come
    off and no pool may be left running (the checker can fail: a clean
    enable right after must succeed)."""
    mod, path, _ = arena
    model = _Wrap(_module().to("cuda"))
    bad = _manifest(vram=[0, 1], dram=[1, 4], nvme=[2, 3, 5, 6, 7])
    with pytest.raises(ValueError, match="BOTH"):
        enable_hybrid_tier(model, path, bad, hot_rows=E)
    for attr in ("_e4b_cold_tier", "_e4b_arena_layer",
                 "_e4b_hybrid_dram_ids", "_e4b_hybrid_owns_pool"):
        assert not hasattr(model.experts, attr), attr
    assert cpu_grouped.pool_start(2) == 2      # pool free to start fresh
    cpu_grouped.pool_stop()
    n = enable_hybrid_tier(model, path,
                           _manifest(vram=[0], dram=[1], nvme=[2, 3, 4, 5, 6, 7]),
                           hot_rows=E)
    assert n == 1
    disable_hybrid_tier(model)


@needs_stack
def test_short_layers_refused(arena):
    mod, path, _ = arena
    model = _Wrap(_module().to("cuda"))
    with pytest.raises(ValueError, match="partial map"):
        enable_hybrid_tier(model, path,
                           _manifest(vram=[0], dram=[1], nvme=[2, 3, 4, 5, 6, 7]),
                           hot_rows=E, layers=[])


@needs_stack
def test_empty_dram_degrades_to_parent_and_teardown_is_clean(arena):
    mod, path, _ = arena
    model = _Wrap(_module().to("cuda"))
    enable_hybrid_tier(model, path,
                       _manifest(vram=[0, 1], dram=[], nvme=[2, 3, 4, 5, 6, 7]),
                       hot_rows=E)
    hidden = torch.randn(2, H, dtype=torch.bfloat16, device="cuda")
    idx = torch.tensor([[2, 0], [7, 1]], device="cuda")
    wts = torch.rand(2, K, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.experts(hidden, idx, wts)
    assert out.isfinite().all()
    disable_hybrid_tier(model)
    for attr in ("_e4b_hybrid", "_e4b_cold_tier", "_e4b_arena_layer",
                 "_e4b_hybrid_dram_ids", "_e4b_hybrid_threads",
                 "_e4b_hybrid_owns_pool"):
        assert not hasattr(model.experts, attr), attr
    assert cpu_grouped.pool_start(0) >= 1     # pool restartable after stop
    cpu_grouped.pool_stop()


@needs_stack
def test_fused_ffn_path_matches_two_call_within_silu_ulps(arena):
    """The fused kernel replaces two grouped GEMVs + torch silu with one
    call; the ONLY numeric difference is the locked-polynomial silu vs
    torch's sleef (last-ulp level). A wiring error — swapped gate/up
    halves, wrong stack, wrong group order — shows up at 1e-1 scale, so
    the tight tolerance here is a real oracle, not decoration."""
    import cpu_grouped
    if not hasattr(cpu_grouped, "gemm_nf4_ffn_grouped_cpu"):
        pytest.skip("gnf4 without the fused FFN entry")
    mod, path, _ = arena
    man = _manifest(vram=[0], dram=[1, 2, 3, 4, 5, 6], nvme=[7])
    torch.manual_seed(11)
    T = 6
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.3
    idx = torch.stack([torch.randperm(E, device="cuda")[:K]
                       for _ in range(T)])
    wts = torch.rand(T, K, device="cuda", dtype=torch.bfloat16)
    outs = {}
    for fused in (False, True):
        model = _Wrap(_module().to("cuda"))
        n = enable_hybrid_tier(model, path, man, hot_rows=E,
                               fused_ffn=fused)
        assert n == 1
        st = model.experts._hot_residency
        assert st.fused_ffn is fused
        try:
            with torch.no_grad():
                outs[fused] = model.experts(hidden, idx, wts).float()
        finally:
            disable_hybrid_tier(model)
    torch.testing.assert_close(outs[True], outs[False],
                               atol=1e-4, rtol=1e-4)


@needs_stack
def test_fused_ffn_stamp_cleared_on_disable(arena):
    mod, path, _ = arena
    model = _Wrap(_module().to("cuda"))
    enable_hybrid_tier(model, path, _manifest(vram=[0], dram=[1], nvme=[]),
                       hot_rows=E, fused_ffn=False)
    assert model.experts._e4b_hybrid_fused_ffn is False
    disable_hybrid_tier(model)
    assert not hasattr(model.experts, "_e4b_hybrid_fused_ffn")


@needs_stack
def test_thin_layer_routes_dram_to_gpu_statically(arena):
    """A layer whose DRAM population is <= offload_thin_uniq must serve
    every DRAM activation through the GPU path (thin_steps advances) and
    still match the fully-resident reference; above the threshold the
    CPU tier serves as usual (thin_steps stays 0). The decision is
    static — no per-call device sync."""
    mod, path, _ = arena
    ref_mod = _module().to("cuda")
    torch.manual_seed(23)
    T = 6
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.3
    idx = torch.stack([torch.randperm(E, device="cuda")[:K]
                       for _ in range(T)])
    wts = torch.rand(T, K, device="cuda", dtype=torch.bfloat16)
    man = _manifest(vram=[0, 3], dram=[1, 4, 6], nvme=[2, 5, 7])
    for thin, expect_thin in ((3, True), (2, False), (None, False)):
        model = _Wrap(_module().to("cuda"))
        n = enable_hybrid_tier(model, path, man, hot_rows=E,
                               offload_thin_uniq=thin)
        assert n == 1
        st = model.experts._hot_residency
        assert st.dram_thin is expect_thin
        try:
            with torch.no_grad():
                got = model.experts(hidden, idx, wts)
                want = ref_mod(hidden, idx, wts)
            torch.testing.assert_close(got.float(), want.float(),
                                       atol=5e-2, rtol=5e-2)
            if expect_thin:
                assert st.thin_steps > 0, "thin layer never took the GPU path"
            else:
                assert st.thin_steps == 0
        finally:
            disable_hybrid_tier(model)
        assert not hasattr(model.experts, "_e4b_hybrid_thin_uniq")
