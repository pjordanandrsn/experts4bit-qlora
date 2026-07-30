"""The composition: NVMe-served experts AND host-resident dense weights, together.

Each half has its own gate — `test_nvme_residency_equivalence.py` for the experts,
`test_dense_offload.py` for the dense weights. Neither says anything about the two
running at once, and that is the configuration a >VRAM MoE actually needs:

    experts    -> `meta` module, rows read from an NVMe arena on demand
    dense      -> pinned host RAM, streamed per layer with prefetch
    resident   -> only what a layer needs right now

The two are meant to be blind to each other: `dense_offload` skips expert modules
and `meta` tensors, `nvme_experts` never touches a non-expert parameter. "Meant to"
is the part under test. Both must also be simultaneously true about `state_dict()`,
which each fixes separately by substituting its own host copies.

The forward half needs the fused kernel (CUDA + triton>=3.4 for ``tl.gather``); the
structural half runs anywhere and is where a composition mistake would actually
show up.
"""
import json
import struct

import pytest
import torch
from torch import nn

pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series modules")

from nvme_arena import bake_expert_tensors, load_index  # noqa: E402

from experts4bit_qlora import Experts4bit  # noqa: E402
from experts4bit_qlora.dense_offload import (  # noqa: E402
    _DenseOffload, enable_dense_offload)
from experts4bit_qlora.nvme_experts import NF4_SEGMENTS  # noqa: E402

E, INTER, H = 8, 64, 128          # NF4 blocksize 64 must tile both in_features
LAYER = 0
KINDS = tuple(NF4_SEGMENTS.values())


@pytest.fixture(autouse=True)
def _clean():
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()
    yield
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()


def _st_bytes(tensors: dict) -> bytes:
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


class MoEBlock(nn.Module):
    """A layer with BOTH kinds of weight: dense projections that dense_offload
    should claim, and an experts module it must leave entirely alone."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        self.q_proj = nn.Linear(H, 2048, bias=False)      # 1 MB, above the floor
        self.o_proj = nn.Linear(2048, H, bias=False)
        self.norm = nn.Parameter(torch.ones(H))
        gate_up = torch.randn(E, 2 * INTER, H, generator=g, dtype=torch.float32) * 0.05
        down = torch.randn(E, H, INTER, generator=g, dtype=torch.float32) * 0.05
        self.experts = Experts4bit.from_float(
            gate_up.to(torch.bfloat16), down.to(torch.bfloat16), has_gate=True,
            activation=torch.nn.functional.silu, quant_type="nf4",
            compute_dtype=torch.bfloat16)

    def forward(self, x):
        return x + self.o_proj(torch.relu(self.q_proj(x * self.norm)))


class MoEToy(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.layers = nn.ModuleList(MoEBlock() for _ in range(n))

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x


@pytest.fixture()
def arena(tmp_path):
    """An arena relocated from one block's OWN quantized expert stacks."""
    m = MoEToy()
    mod = m.layers[0].experts
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    stacks = {"nf4.gate_up_blocks": mod.gate_up_proj.view(E, n1, k1 // 2),
              "nf4.gate_up_absmax": mod.gate_up_absmax.view(E, n1, k1 // 64).float(),
              "nf4.down_blocks": mod.down_proj.view(E, n2, k2 // 2),
              "nf4.down_absmax": mod.down_absmax.view(E, n2, k2 // 64).float()}
    dt = {torch.uint8: "U8", torch.float32: "F32"}
    tensors = {}
    for kind, stack in stacks.items():
        for e in range(E):
            t = stack[e].contiguous().cpu()
            tensors[f"model.layers.{LAYER}.mlp.experts.{e}.{kind}"] = (
                tuple(t.shape), dt[t.dtype], t.numpy().tobytes())
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(_st_bytes(tensors))
    path = str(tmp_path / "m.arena")
    bake_expert_tensors(
        str(snap), path,
        name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
        kinds=KINDS, align=4096, log=lambda *a: None)
    return m, path, load_index(path)


def _meta_experts(model, index):
    """Replace the experts module with one BUILT on meta, via the real
    `build_meta_experts` — which is how the arena path actually gets there.

    Converting an existing module is not just unfaithful, PyTorch forbids it:
    `Parameter.data = <meta tensor>` raises "incompatible tensor type". That
    mismatch between how I imagined the state arises and how it actually arises is
    exactly what this test should be built on.
    """
    from experts4bit_qlora.nvme_experts import build_meta_experts
    for lay in model.layers:
        lay.experts = build_meta_experts(
            index, E, has_gate=True,
            activation=torch.nn.functional.silu, compute_dtype=torch.bfloat16)
    return model


# ------------------------------------------------------------- structural ----
def test_dense_offload_ignores_the_experts_module(arena):
    """The load-bearing composition property: dense_offload must not capture a
    single expert tensor, or it would pin in host RAM exactly the bytes the arena
    exists to keep OUT of host RAM."""
    m, _p, _i = arena
    hs = enable_dense_offload(m, "cpu", pin=False)
    for h in hs:
        for mod, attr, _is_param, _home in h.slots:
            assert not isinstance(mod, Experts4bit), (type(mod).__name__, attr)
            assert attr not in ("gate_up_proj", "down_proj",
                                "gate_up_absmax", "down_absmax"), attr
        assert len(h.slots) == 2, [a for _m, a, _p2, _h in h.slots]


def test_dense_offload_skips_meta_experts(arena, tmp_path):
    """With the arena path the expert buffers are on `meta`. Capturing one would
    either raise on pin or silently materialize the thing we refused to store."""
    m, _p, _i = arena
    _meta_experts(m, _i)
    hs = enable_dense_offload(m, "cpu", pin=False)
    assert all(len(h.slots) == 2 for h in hs)
    for h in hs:
        assert all(not home.is_meta for _m, _a, _p, home in h.slots)


def test_state_dict_carries_dense_homes_with_meta_experts(arena):
    """Both halves must be true of state_dict() at once: dense entries come from
    the pinned homes, and meta expert entries are still meta (their own path owns
    them). A saved checkpoint must not silently contain empty attention."""
    m, _p, _i = arena
    # only the DENSE parameters — the expert stacks are deliberately made meta
    want = {n: p.detach().clone() for n, p in m.named_parameters()
            if ".experts." not in n}
    assert want, "fixture has no dense parameters"
    _meta_experts(m, _i)
    enable_dense_offload(m, "cpu", pin=False)
    sd = m.state_dict()
    for n, t in want.items():
        assert sd[n].numel() == t.numel(), (n, sd[n].shape, t.shape)
        assert torch.equal(sd[n].cpu(), t.cpu()), n
    # and the arena-served experts are still meta: their own path owns them, and
    # dense_offload must not have quietly materialized them into host RAM
    # Experts4bit puts non-tensor quant state in its state_dict too, so filter.
    exp = {k: v for k, v in sd.items() if ".experts." in k and torch.is_tensor(v)}
    assert exp, list(sd)
    metas = [k for k, v in exp.items() if v.is_meta]
    assert metas, {k: (tuple(v.shape), str(v.device)) for k, v in exp.items()}


def test_residency_bound_holds_with_experts_present(arena):
    """dense_offload's single-slot bound must not be affected by the experts
    module sitting in the same layer."""
    m, _p, _i = arena
    hs = enable_dense_offload(m, "cpu", pin=False, prefetch=False)
    for h in hs:
        h.stage()
    assert sum(1 for h in hs if h.staged) == 1, [h.staged for h in hs]


# ---------------------------------------------------------------- forward ----
def _needs_fused():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("triton")
    import triton
    import triton.language as tl
    if not hasattr(tl, "gather"):
        pytest.skip(f"grouped-nf4 gather needs triton>=3.4 for tl.gather; "
                    f"have {triton.__version__}")


def test_dense_offload_forward_is_unaffected_by_expert_residency(arena):
    """End to end for the composition: turning on BOTH must give the same answer as
    turning on neither. The dense path is what this forward exercises; the expert
    path's own equivalence is gated in test_nvme_residency_equivalence.py."""
    _needs_fused()
    from experts4bit_qlora.nvme_experts import enable_nvme_residency

    m, path, index = arena
    m = m.cuda().eval()
    x = torch.randn(2, H, device="cuda")
    with torch.no_grad():
        want = m(x)

    # One entry per MoE module, and `layers` maps module -> arena layer. Only
    # layer 0's experts are in this arena, so only layer 0 is served from it.
    n = enable_nvme_residency(m.layers[0], path, hot_sets=[()], hot_rows=E,
                              device="cuda", layers=[LAYER])
    assert n == 1, f"expected 1 module patched, got {n}"
    enable_dense_offload(m, "cuda", pin=True)
    with torch.no_grad():
        got = m(x)
    assert torch.equal(got, want), (got - want).abs().max().item()
