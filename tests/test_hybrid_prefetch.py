# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Speculative prefetch (hybrid Phase 4): predicted NVMe experts are warm
before the demand path asks (zero new disk reads), outputs are bit-equal
with the feature on or off, and the feature is structurally free when a
layer has no NVMe-resident experts."""

import json
import struct

import pytest
import torch

pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series")
pytest.importorskip("nvme_residency")
cpu_grouped = pytest.importorskip("cpu_grouped")

from nvme_arena import bake_expert_tensors, load_index  # noqa: E402

from experts4bit_qlora import Experts4bit  # noqa: E402
from experts4bit_qlora.engines import hybrid as hy  # noqa: E402
from experts4bit_qlora.engines.nvme_experts import NF4_SEGMENTS  # noqa: E402

E, INTER, H, K = 8, 64, 128, 2

needs_stack = pytest.mark.skipif(
    not hy.hybrid_available(), reason="needs CUDA + gnf4_native CPU kernels"
)


class OlmoeTopKRouter(torch.nn.Module):
    """Class NAME drives the prefetch chain discovery."""

    def __init__(self, seed):
        super().__init__()
        self.top_k, self.num_experts, self.hidden_dim = K, E, H
        self.norm_topk_prob = False
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(E, H, generator=g) * 0.3)


class _Block(torch.nn.Module):
    def __init__(self, experts, seed):
        super().__init__()
        self.router = OlmoeTopKRouter(seed)
        self.experts = experts


def _st_bytes(tensors):
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def _module(seed):
    g = torch.Generator().manual_seed(seed)
    gate_up = torch.randn(E, 2 * INTER, H, generator=g) * 0.05
    down = torch.randn(E, H, INTER, generator=g) * 0.05
    return Experts4bit.from_float(gate_up.to(torch.bfloat16),
                                  down.to(torch.bfloat16), has_gate=True,
                                  activation=torch.nn.functional.silu,
                                  quant_type="nf4",
                                  compute_dtype=torch.bfloat16)


@pytest.fixture()
def two_layer(tmp_path):
    mods = [_module(11), _module(12)]
    dt = {torch.uint8: "U8", torch.float32: "F32"}
    tensors = {}
    for layer, mod in enumerate(mods):
        n1, k1 = mod._gate_up_shape
        n2, k2 = mod._down_shape
        payload = {
            "nf4.gate_up_blocks": mod.gate_up_proj.view(E, n1, k1 // 2),
            "nf4.gate_up_absmax": mod.gate_up_absmax.view(E, n1, k1 // 64).float(),
            "nf4.down_blocks": mod.down_proj.view(E, n2, k2 // 2),
            "nf4.down_absmax": mod.down_absmax.view(E, n2, k2 // 64).float(),
        }
        for kind, stack in payload.items():
            for e in range(E):
                t = stack[e].contiguous().cpu()
                tensors[f"model.layers.{layer}.mlp.experts.{e}.{kind}"] = (
                    tuple(t.shape), dt[t.dtype], t.numpy().tobytes())
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(_st_bytes(tensors))
    arena_path = str(tmp_path / "m.arena")
    bake_expert_tensors(
        str(snap), arena_path,
        name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
        kinds=tuple(NF4_SEGMENTS.values()), align=4096, log=lambda *a: None)
    model = torch.nn.ModuleList(
        [_Block(m.to("cuda"), seed=21 + i) for i, m in enumerate(mods)])
    return model, arena_path, load_index(arena_path)


def _manifest(per_layer):
    tiers = {"vram": [], "dram": [], "nvme": []}
    for layer, spec in enumerate(per_layer):
        for tier, ids in spec.items():
            tiers[tier] += [[layer, e] for e in ids]
    return {"schema": "e4b-placement/1", "tiers": tiers,
            "masses": {"vram_frac": 0, "dram_frac": 0, "nvme_frac": 0}}


def _barrier():
    hy._PF_POOL.submit(lambda: None).result()


@needs_stack
def test_prefetch_warms_predicted_experts_and_changes_no_bits(two_layer):
    model, path, _ = two_layer
    man = _manifest([
        {"vram": [0], "dram": [1], "nvme": [2, 3, 4, 5, 6, 7]},
        {"vram": [0], "dram": [1], "nvme": [2, 3, 4, 5, 6, 7]},
    ])
    n = hy.enable_hybrid_tier(model, path, man, hot_rows=E,
                              prefetch=True, verbose=True)
    assert n == 2
    st0 = model[0].experts._hot_residency
    st1 = model[1].experts._hot_residency
    assert st0.pf is not None and st0.pf_enabled
    assert st1.pf is None                        # last layer has no L+1
    try:
        torch.manual_seed(3)
        hidden = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
        wts = torch.rand(1, K, device="cuda", dtype=torch.bfloat16)
        # the ids layer 1 will REALLY route, from its own router — the
        # predictor uses the same weights, so prediction == truth here
        r1 = model[1].router
        logits1 = torch.nn.functional.linear(
            hidden.float().cpu(), r1.weight.detach().float().cpu())
        idx1 = torch.topk(logits1, K, dim=-1).indices.to("cuda")

        # control: with prefetch OFF, a cold layer-1 demand fetch MISSES
        hy.set_prefetch(model, False)
        base = st1.tier_stats()["misses"]
        with torch.no_grad():
            off_out = model[1].experts(hidden, idx1, wts)
        m_ctrl = st1.tier_stats()["misses"]
        assert m_ctrl > base, "control: cold demand fetch must miss"

        # rebuild cold state for the real arm: route DIFFERENT experts via
        # a shifted hidden so the slots aren't already resident
        torch.manual_seed(9)
        hidden2 = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
        logits2 = torch.nn.functional.linear(
            hidden2.float().cpu(), r1.weight.detach().float().cpu())
        idx2 = torch.topk(logits2, K, dim=-1).indices.to("cuda")
        if set(idx2.reshape(-1).tolist()) <= set(idx1.reshape(-1).tolist()):
            pytest.skip("routing draw collided; seeds need retuning")

        hy.set_prefetch(model, True)
        idx0 = torch.tensor([[2, 3]], device="cuda")
        with torch.no_grad():
            model[0].experts(hidden2, idx0, wts)   # layer 0 fires prefetch
        _barrier()
        assert hy.prefetch_stats(model)["prefetch_rows"] > 0
        m_before = st1.tier_stats()["misses"]
        with torch.no_grad():
            on_out = model[1].experts(hidden2, idx2, wts)
        m_after = st1.tier_stats()["misses"]
        assert m_after == m_before, (
            f"prefetched experts still missed ({m_before}->{m_after})")

        # bits: the feature changes WHEN bytes load, never their values
        hy.set_prefetch(model, False)
        with torch.no_grad():
            off2 = model[1].experts(hidden2, idx2, wts)
        assert torch.equal(on_out, off2)
        assert off_out.isfinite().all()
    finally:
        hy.disable_hybrid_tier(model)


@needs_stack
def test_free_when_no_nvme_mass(two_layer):
    model, path, _ = two_layer
    man = _manifest([
        {"vram": [0, 2, 4, 6], "dram": [1, 3, 5, 7], "nvme": []},
        {"vram": [0, 2, 4, 6], "dram": [1, 3, 5, 7], "nvme": []},
    ])
    hy.enable_hybrid_tier(model, path, man, hot_rows=E, prefetch=True)
    try:
        hidden = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
        idx = torch.tensor([[0, 1]], device="cuda")
        wts = torch.rand(1, K, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            model[0].experts(hidden, idx, wts)
        _barrier()
        # next layer has no NVMe set: the submit short-circuits entirely
        assert hy.prefetch_stats(model)["prefetch_submitted"] == 0
    finally:
        hy.disable_hybrid_tier(model)


@needs_stack
def test_concurrent_prefetch_never_evicts_a_demand_row(two_layer):
    """The 235B crash class: a prefetch ensure racing a demand
    ensure→read window must never evict the demand rows (KeyError
    'not resident'). Tiny hot_rows + a hammering loop reproduce the
    pressure; the transaction lock is what makes this pass."""
    model, path, _ = two_layer
    man = _manifest([
        {"vram": [], "dram": [], "nvme": list(range(E))},
        {"vram": [], "dram": [], "nvme": list(range(E))},
    ])
    hy.enable_hybrid_tier(model, path, man, hot_rows=4, prefetch=True)
    try:
        hy.set_prefetch(model, True)
        wts = torch.rand(1, K, device="cuda", dtype=torch.bfloat16)
        torch.manual_seed(31)
        for i in range(40):
            hidden = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
            idx0 = torch.tensor([[(i * 3) % E, (i * 5 + 1) % E]],
                                device="cuda")
            idx1 = torch.tensor([[(i * 7 + 2) % E, (i * 11 + 3) % E]],
                                device="cuda")
            with torch.no_grad():
                model[0].experts(hidden, idx0, wts)   # fires prefetch
                model[1].experts(hidden, idx1, wts)   # races the worker
        _barrier()
    finally:
        hy.disable_hybrid_tier(model)
