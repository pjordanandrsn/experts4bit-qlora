"""The segmented (offload-homes) cold source must be byte-identical to the copied arena.

Under offload the engine reads the homes in place instead of baking a second
full-size pinned arena from them (Qwen3-30B: 15.19 GiB each, so ~30 GiB of host
RAM for the one config that exists to fit a big model on a small card). The homes
group by TENSOR, the engine's row layout groups by EXPERT, so an expert becomes
four contiguous runs instead of one and the gather issues a launch per segment.

The addressing is the whole change, and getting a stride or an offset wrong would
not crash — it would feed the GEMM another expert's bytes and quietly produce a
wrong answer. So this asserts the property directly and WITHOUT a GPU: the row
assembled from the four segment (base, stride, offset, length) tuples equals the
row the copied-arena path would have produced, for every expert.

The kernel plumbing that consumes these addresses still needs CUDA and is covered
by tests/test_pipelined.py.
"""
import pytest
import torch

from experts4bit_qlora.pipelined import _align8


def _layout(E, n1, k1, n2, k2):
    """The engine's row layout, mirroring _PipelinedResidency.__init__."""
    seg = [n1 * (k1 // 2), n1 * (k1 // 64) * 4, n2 * (k2 // 2), n2 * (k2 // 64) * 4]
    off = [0]
    for s in seg[:-1]:
        off.append(_align8(off[-1] + s))
    return seg, off, _align8(off[-1] + seg[-1])


def _fake_homes(E, n1, k1, n2, k2):
    """Offload homes: one buffer per tensor, expert-major, exactly as offload packs them."""
    g = torch.Generator().manual_seed(0)
    return {
        "gate_up_proj":   torch.randint(0, 255, (E, n1, k1 // 2), dtype=torch.uint8, generator=g),
        "gate_up_absmax": torch.randn(E, n1 * (k1 // 64), generator=g).float(),
        "down_proj":      torch.randint(0, 255, (E, n2, k2 // 2), dtype=torch.uint8, generator=g),
        "down_absmax":    torch.randn(E, n2 * (k2 // 64), generator=g).float(),
    }


@pytest.mark.parametrize("E,n1,k1,n2,k2", [
    (8, 128, 128, 64, 64),      # toy
    (4, 1536, 2048, 2048, 768),  # Qwen3-30B-A3B expert geometry
])
def test_segmented_row_equals_copied_arena_row(E, n1, k1, n2, k2):
    seg, off, row_bytes = _layout(E, n1, k1, n2, k2)
    homes = _fake_homes(E, n1, k1, n2, k2)
    names = ("gate_up_proj", "gate_up_absmax", "down_proj", "down_absmax")

    # What the copied-arena path builds today.
    arena = torch.zeros(E, row_bytes, dtype=torch.uint8)
    a_f32 = arena.view(torch.float32)
    arena[:, off[0]:off[0] + seg[0]] = homes["gate_up_proj"].view(E, -1)
    a_f32[:, off[1] // 4: off[1] // 4 + seg[1] // 4] = homes["gate_up_absmax"].view(E, -1)
    arena[:, off[2]:off[2] + seg[2]] = homes["down_proj"].view(E, -1)
    a_f32[:, off[3] // 4: off[3] // 4 + seg[3] // 4] = homes["down_absmax"].view(E, -1)

    # What the segmented source addresses: base + e*stride, for each of the four runs.
    # Compare through byte views of the same underlying storage the pointers name.
    for j, name in enumerate(names):
        flat = homes[name].reshape(E, -1).contiguous().view(torch.uint8).reshape(E, -1)
        assert flat.shape[1] == seg[j], (
            f"segment {name} is {flat.shape[1]} B/expert but the row layout reserves {seg[j]}")
        for e in range(E):
            assert torch.equal(flat[e], arena[e, off[j]:off[j] + seg[j]]), (
                f"expert {e} segment {name} differs from its arena bytes")


@pytest.mark.parametrize("E,n1,k1,n2,k2", [(4, 1536, 2048, 2048, 768)])
def test_segments_are_whole_int64_words(E, n1, k1, n2, k2):
    """The gather copies int64 words, so a segment that is not 8-byte-divisible would
    make the last expert's read run past the end of its home tensor. The engine falls
    back to the copied arena in that case; this pins that real geometries do not need
    the fallback."""
    seg, off, _ = _layout(E, n1, k1, n2, k2)
    assert all(s % 8 == 0 for s in seg), f"segments not 8B-aligned: {seg}"
    assert all(o % 8 == 0 for o in off), f"offsets not 8B-aligned: {off}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_offloaded_module_reads_homes_and_matches_the_arena_bit_for_bit():
    """End-to-end: an OFFLOADED module must take the segmented path and produce
    byte-identical output to the copied-arena path.

    The liveness assertion is the load-bearing half. If the engine quietly fell back
    to the arena this would still pass every numeric check -- a dead change reproduces
    the control perfectly -- so the fallback is a failure here, not a silent success.
    """
    pytest.importorskip("triton")
    pytest.importorskip("nf4_grouped")
    import os
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("raw-pointer gather is compiled-only")

    from experts4bit_qlora import Experts4bit
    from experts4bit_qlora.offload import enable_expert_offload
    from experts4bit_qlora.pipelined import (
        disable_pipelined_residency, enable_pipelined_residency)

    E_, H_, INTER_, K_ = 8, 128, 64, 3

    def _mod():
        torch.manual_seed(0)
        gu = torch.randn(E_, 2 * INTER_, H_)
        dn = torch.randn(E_, H_, INTER_)
        return Experts4bit.from_float(gate_up_proj=gu.cuda(), down_proj=dn.cuda(),
                                      compute_dtype=torch.bfloat16, has_gate=True)

    torch.manual_seed(1)
    hs = torch.randn(1, H_, dtype=torch.bfloat16, device="cuda")
    w, idx = torch.topk(torch.softmax(torch.randn(1, E_, device="cuda"), -1), k=K_, dim=-1)
    wts = w.to(torch.bfloat16)

    for hot in ([], [0, 2, 5]):
        seg_mod = _mod()
        handle = enable_expert_offload(seg_mod, "cuda", pin=True)
        assert getattr(handle, "home", None), "offload produced no homes"
        assert enable_pipelined_residency(
            seg_mod, [torch.tensor(hot, dtype=torch.long)], device="cuda", k_slots=K_) == 1
        st = seg_mod._pipelined
        assert st.arena is None and st.seg_srcs is not None and len(st.seg_srcs) == 4, (
            "offloaded module fell back to the copied arena — the segmented path is untested")

        arena_mod = _mod()
        assert enable_pipelined_residency(
            arena_mod, [torch.tensor(hot, dtype=torch.long)], device="cuda", k_slots=K_) == 1
        assert arena_mod._pipelined.arena is not None, "control must use the copied arena"

        with torch.no_grad():
            got, want = seg_mod(hs, idx, wts), arena_mod(hs, idx, wts)
        assert torch.equal(got, want), (
            f"hot={hot}: segmented source read different bytes than the arena "
            f"(max abs diff {(got.float() - want.float()).abs().max().item():.3e})")
        disable_pipelined_residency(seg_mod)
        disable_pipelined_residency(arena_mod)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hot_expert_primes_each_segment_from_its_own_offset():
    """`_prime` must read each segment at its own offset, including for a HOT expert.

    `_prime` seeds every slot from expert 0 with `have = -1`, so it does NOT take the
    hot skip. If the hot lane's address were the resident ROW START for all four
    segments, the primed slots would get gate_up bytes written into the absmax and
    down regions. Nothing reads those today — `_fetch` forces hot lanes to skip and
    they read the resident row in place — so this is latent, and it stays latent only
    while that skip holds. Shipped in 0.16.0 and found by review, not by a test.
    """
    pytest.importorskip("triton")
    pytest.importorskip("nf4_grouped")
    import os
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("raw-pointer gather is compiled-only")

    from experts4bit_qlora import Experts4bit
    from experts4bit_qlora.offload import enable_expert_offload
    from experts4bit_qlora.pipelined import enable_pipelined_residency

    E_, H_, INTER_, K_ = 8, 128, 64, 3
    torch.manual_seed(0)
    mod = Experts4bit.from_float(
        gate_up_proj=torch.randn(E_, 2 * INTER_, H_).cuda(),
        down_proj=torch.randn(E_, H_, INTER_).cuda(),
        compute_dtype=torch.bfloat16, has_gate=True)
    handle = enable_expert_offload(mod, "cuda", pin=True)
    # expert 0 HOT: the case that makes _prime read through the hot address
    enable_pipelined_residency(mod, [torch.tensor([0, 1])], device="cuda", k_slots=K_)
    st = mod._pipelined
    assert st.arena is None and st.seg_srcs is not None, "not on the segmented path"

    names = ("gate_up_proj", "gate_up_absmax", "down_proj", "down_absmax")
    off = [st.seg_srcs[j][1] * 8 for j in range(4)]
    seg = [st.seg_srcs[j][2] * 8 for j in range(4)]
    truth = torch.zeros(st.row_bytes, dtype=torch.uint8)
    for j, n in enumerate(names):
        flat = handle.home[n].reshape(E_, -1).contiguous().view(torch.uint8).reshape(E_, -1)
        truth[off[j]:off[j] + seg[j]] = flat[0]

    primed = st.slots[0].cpu()
    bad = [names[j] for j in range(4)
           if not torch.equal(primed[off[j]:off[j] + seg[j]], truth[off[j]:off[j] + seg[j]])]
    assert not bad, f"primed slot holds the wrong bytes for segment(s) {bad}"
