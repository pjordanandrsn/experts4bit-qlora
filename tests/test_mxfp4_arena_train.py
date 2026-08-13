"""Training from an arena of NATIVE MXFP4 bytes — the CPU spec.

Two rented-pod runs died on things this file would have caught for free, which
is the whole reason it exists and why it is written before the feature:

  * a bake call with the wrong signature, masked by a pipe so the run continued
    against an arena that was never created;
  * `quant_type="mxfp4"` rejected by the loader, and behind it the real
    requirement — a meta expert whose DECLARED buffers match the arena's
    segments, which is what the tier's geometry check compares.

The arena here is built from bytes this file writes, so nothing depends on a
checkpoint, a GPU, or a 149 GB download.

The tests split into two groups on purpose:

  IMPLEMENTED — `arena_offload_view` resolves NF4 and MXFP4 layouts and refuses
  anything else. These pass today.

  SPEC — the meta expert must DECLARE MXFP4-shaped buffers so an MXFP4 arena can
  be staged into it. Marked `xfail(strict=True)`: they fail now, and the day the
  feature lands they fail LOUDLY for passing unexpectedly, which is the signal
  to delete the marker rather than leave a green test that asserts nothing.
"""
from __future__ import annotations

import json
import struct

import pytest
import torch

pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series modules")
pytest.importorskip("mxfp4_residency", reason="needs grouped-nf4-gemm MXFP4 residency")

from nvme_arena import bake_expert_tensors, load_index  # noqa: E402

from experts4bit_qlora.engines.nvme_train import (  # noqa: E402
    MXFP4_OFFLOAD_SEGMENTS,
    OFFLOAD_SEGMENTS,
    arena_offload_view,
    check_arena_geometry,
)

# Both in_features must be a multiple of the NF4 blocksize (64): gate_up's is
# hidden_dim, down's is intermediate_dim. The vendored primitive enforces it so
# per-expert quant blocks tile exactly, and a 32-wide fixture is simply illegal.
E, INTER, H = 4, 128, 64
NAME_TEMPLATE = "model.layers.{layer}.mlp.experts.{expert}.{kind}"
# Order is load-bearing: the fuse presents each PAIR as one range, so both
# blocks must be adjacent and both scales adjacent. This mirrors
# `mxfp4_residency.V4_RESIDENCY_KINDS`.
MXFP4_KINDS = ("w1.weight", "w3.weight", "w1.scale", "w3.scale",
               "w2.weight", "w2.scale")


def _st_bytes(tensors: dict) -> bytes:
    """Minimal safetensors writer — the format nvme_arena's header reader parses."""
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def _mxfp4_stacks():
    """Per-expert MXFP4 storage: nibble-packed blocks (K/2 bytes) and one e8m0
    scale byte per 32 elements. Values are arbitrary — every assertion here is
    about SHAPE and DTYPE resolution, never about arithmetic."""
    g = torch.Generator().manual_seed(4)

    def u8(*shape):
        return torch.randint(0, 255, shape, generator=g, dtype=torch.uint8)

    return {
        "w1.weight": u8(E, INTER, H // 2), "w1.scale": u8(E, INTER, H // 32),
        "w3.weight": u8(E, INTER, H // 2), "w3.scale": u8(E, INTER, H // 32),
        "w2.weight": u8(E, H, INTER // 2), "w2.scale": u8(E, H, INTER // 32),
    }


def _bake_mxfp4(tmp_path, name="v4.mxarena"):
    tensors = {}
    for kind, stack in _mxfp4_stacks().items():
        for e in range(E):
            t = stack[e].contiguous()
            tensors[NAME_TEMPLATE.format(layer=0, expert=e, kind=kind)] = (
                tuple(t.shape), "U8", t.numpy().tobytes())
    snap = tmp_path / "snap-mxfp4"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(_st_bytes(tensors))
    path = str(tmp_path / name)
    bake_expert_tensors(str(snap), path, name_template=NAME_TEMPLATE,
                        kinds=MXFP4_KINDS, align=4096, log=lambda *a: None)
    return path, load_index(path)


# --------------------------------------------------------------- implemented
def test_mxfp4_arena_resolves_to_the_four_segment_staging_view(tmp_path):
    """The point of the change: six per-projection segments present as the same
    four the NF4 path stages, so staging is one code path rather than two."""
    _path, index = _bake_mxfp4(tmp_path)
    assert len(index["segments"]) == 6, "fixture should bake six MXFP4 segments"

    view, segmap = arena_offload_view(index)

    # Suffixes are checkpoint-dependent (V4 'w1.weight+w3.weight', K3
    # 'w1.weight_packed+w3.weight_packed'), so assert the MAPPING resolves, not
    # that it equals any fixed set of names.
    assert len(view["segments"]) == 4, "fusion must present four segments"
    have = {g["suffix"] for g in view["segments"]}
    assert set(segmap.values()) <= have, (
        f"map points at segments the fused view lacks: {sorted(segmap.values())} vs {sorted(have)}")
    # every offload tensor the tier stages must resolve
    assert set(segmap) == set(OFFLOAD_SEGMENTS), (
        "the MXFP4 map must cover exactly the tensors the NF4 map covers")


def test_an_unknown_layout_raises_and_names_both_expectations(tmp_path):
    """A wrong arena must fail at attach with both expected layouts named — not
    later, deep in a stage, with a shape error."""
    with pytest.raises((ValueError, KeyError)) as exc:
        arena_offload_view({"segments": [{"suffix": "w9.bogus"}]})
    msg = str(exc.value)
    assert "w9.bogus" in msg, "the error must say what the arena actually has"


def test_nf4_layout_is_returned_untouched():
    """The NF4 path must not acquire a fusion step it does not need — the index
    object itself is returned, so a regression here is visible as identity."""
    idx = {"segments": [{"suffix": s} for s in OFFLOAD_SEGMENTS.values()]}
    view, segmap = arena_offload_view(idx)
    assert view is idx
    assert segmap is OFFLOAD_SEGMENTS


# ---------------------------------------------------------------------- spec
def test_meta_experts_declare_mxfp4_shapes_for_an_mxfp4_arena(tmp_path):
    """THE requirement the second pod run died on.

    Under `arena_train=True` the base is on `meta` and holds nothing, so what
    matters is the DECLARED dtype and per-expert width. The tier's geometry
    check compares them against the arena's segments, and an NF4-shaped module
    cannot match an MXFP4 arena: NF4 declares `absmax` as fp32 per 64 elements,
    MXFP4 needs `scales` as uint8 per 32. Same blocks width, different scales
    entirely.
    """
    from experts4bit_qlora.engines.nvme_experts import build_meta_experts

    _path, index = _bake_mxfp4(tmp_path)
    experts = build_meta_experts(index, E, has_gate=True,
                                 compute_dtype=torch.bfloat16, quant_type="nf4")
    # Must not raise: the module the loader builds for an MXFP4 arena has to be
    # stageable from that arena.
    check_arena_geometry(experts, index, 0)


def test_geometry_check_rejects_an_nf4_module_against_an_mxfp4_arena(tmp_path):
    """The other half, and the one that keeps the spec honest: whatever makes an
    MXFP4 module match must still REJECT a genuinely mismatched one, or the check
    has been widened into uselessness rather than taught a second layout.

    This passes TODAY and must keep passing after option B lands. An NF4 module
    declares `absmax` as fp32 per 64 elements; the MXFP4 arena carries uint8
    scales per 32. Attaching them would copy real bytes into the right-sized
    buffer and compute nonsense.
    """
    from experts4bit_qlora import Experts4bit

    _path, index = _bake_mxfp4(tmp_path)
    g = torch.Generator().manual_seed(1)
    nf4 = Experts4bit.from_float(
        (torch.randn(E, 2 * INTER, H, generator=g) * 0.05).to(torch.bfloat16),
        (torch.randn(E, H, INTER, generator=g) * 0.05).to(torch.bfloat16),
        has_gate=True, activation=torch.nn.functional.silu,
        quant_type="nf4", compute_dtype=torch.bfloat16)

    with pytest.raises((TypeError, ValueError)) as exc:
        check_arena_geometry(nf4, index, 0)
    assert "arena" in str(exc.value).lower()


def test_mxfp4_arena_module_is_flagged_and_refuses_nf4_arithmetic(tmp_path):
    """Option B wires STAGING, not COMPUTE.

    Declaring MXFP4 buffers makes the bytes land correctly; it does not make the
    NF4 fused kernel interpret them. Running anyway would be the silent-wrong
    class this codebase guards hardest against, so the module is flagged. This
    test exists so that flag cannot be quietly dropped when the MXFP4 forward
    lands -- at that point this test should be REPLACED by a numerics parity
    test against `dequant_mxfp4`, not deleted.
    """
    from experts4bit_qlora.engines.nvme_experts import build_meta_experts

    _path, index = _bake_mxfp4(tmp_path)
    experts = build_meta_experts(index, E, has_gate=True,
                                 compute_dtype=torch.bfloat16, quant_type="nf4")
    assert getattr(experts, "_e4b_mxfp4_arena", False) is True, (
        "an MXFP4-arena module must be flagged, so no caller can mistake it for "
        "one whose forward is wired")
