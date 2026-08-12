# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Unit suite for the address-gather kernel itself (operator note: the new
fetching instruction is trusted on its own tests, not just the end-to-end
b_rel gate).

Covers, at the byte level against plain torch copies: pinned-host (UVA) and
device (D2D) sources — mixed in ONE launch; the have-skip discipline proven
by mutation (a skipped slot must retain its stale bytes, not silently
re-copy); multi-chunk rows and tail masking on non-BLOCK-multiple sizes with
neighbor-row integrity; duplicate addresses across slots; and the engine's
traffic counters against hand-counted expectations. Skips unless CUDA +
triton are present.
"""
import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

@pytest.fixture(autouse=True)
def _no_triton_interpreter():
    """Runtime (order-proof) guard: the address-gather is compiled-only — raw
    device/UVA pointers segfault the host-side Triton interpreter. When an
    interpreter-contract suite has set TRITON_INTERPRET=1 in this process
    (it does so at import), these tests skip; run them in separate pytest
    invocations to execute both."""
    import os
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("Triton interpreter mode active (raw-pointer gather is compiled-only)")


from experts4bit_qlora.pipelined import _align8, _gather_kernel  # noqa: E402


def _mk_store(E, row_bytes, pattern0, pinned):
    """[E, row_bytes] uint8 whose row e is filled with byte (pattern0+e)%251."""
    t = torch.empty(E, row_bytes, dtype=torch.uint8)
    for e in range(E):
        t[e].fill_((pattern0 + e) % 251)
    if pinned:
        t = t.pin_memory()
        assert t.is_pinned()
    else:
        t = t.cuda()
    return t


def _addrs(store, ids):
    rb = store.shape[1]
    return store.data_ptr() + torch.as_tensor(ids, dtype=torch.long, device="cuda") * rb


def _launch(slots, src, have, block=16, ident=None, dst_word_off=0, n_words=None):
    """Whole-row launch by default: the read address doubles as the lane identity,
    the segment starts at word 0 and spans the row — i.e. exactly the pre-segmented
    behaviour. ``ident``/``dst_word_off``/``n_words`` drive one segment instead."""
    rb = slots.shape[1]
    assert rb % 8 == 0
    rw = rb // 8
    kern = _gather_kernel()
    nw = rw if n_words is None else n_words
    grid = (slots.shape[0], -(-nw // block))
    kern[grid](slots.view(torch.int64), src, src if ident is None else ident, have,
               rw, dst_word_off, nw, BLOCK=block, num_warps=1)


def test_copies_bytes_exactly_host_and_device_mixed():
    rb = _align8(296)  # 37 int64 words: 3 chunks of BLOCK=16 with a 5-word tail
    host = _mk_store(6, rb, 10, pinned=True)
    dev = _mk_store(4, rb, 100, pinned=False)
    slots = torch.zeros(4, rb, dtype=torch.uint8, device="cuda")
    have = torch.full((4,), -1, dtype=torch.long, device="cuda")
    # slot0 <- host row 2, slot1 <- DEVICE row 3, slot2 <- host row 5, slot3 <- device row 0
    src = torch.stack([_addrs(host, [2])[0], _addrs(dev, [3])[0],
                       _addrs(host, [5])[0], _addrs(dev, [0])[0]])
    _launch(slots, src, have)
    torch.cuda.synchronize()
    exp = [12, 103, 15, 100]
    for j, v in enumerate(exp):
        assert (slots[j] == v).all(), (j, v, slots[j][:8].tolist())


def test_tail_mask_no_neighbor_corruption():
    rb = _align8(296)
    host = _mk_store(3, rb, 40, pinned=True)
    slots = torch.zeros(3, rb, dtype=torch.uint8, device="cuda")
    have = torch.full((3,), -1, dtype=torch.long, device="cuda")
    src = _addrs(host, [0, 1, 2])
    _launch(slots, src, have, block=16)   # forces masked tail chunk per row
    torch.cuda.synchronize()
    for j in range(3):
        assert (slots[j] == 40 + j).all()   # full row incl. final tail bytes
    # rows are exact => no cross-row (OOB) writes occurred


def test_have_skip_is_a_real_skip():
    # prove the skip by mutation: matching have must leave STALE bytes in
    # place even though the source now holds different bytes
    rb = _align8(64)
    host = _mk_store(2, rb, 7, pinned=True)
    slots = torch.zeros(1, rb, dtype=torch.uint8, device="cuda")
    have = torch.full((1,), -1, dtype=torch.long, device="cuda")
    src = _addrs(host, [1])
    _launch(slots, src, have)
    torch.cuda.synchronize()
    assert (slots[0] == 8).all()
    have.copy_(src)                    # engine discipline: have := src after fetch
    host[1].fill_(99)                  # mutate the source
    _launch(slots, src, have)          # want == have -> must NOT copy
    torch.cuda.synchronize()
    assert (slots[0] == 8).all(), "skip re-copied: have discipline broken"
    have.fill_(-1)                     # invalidate -> must copy the new bytes
    _launch(slots, src, have)
    torch.cuda.synchronize()
    assert (slots[0] == 99).all()


def test_duplicate_sources_across_slots():
    rb = _align8(128)
    host = _mk_store(2, rb, 30, pinned=True)
    slots = torch.zeros(3, rb, dtype=torch.uint8, device="cuda")
    have = torch.full((3,), -1, dtype=torch.long, device="cuda")
    src = _addrs(host, [1, 1, 0])      # two slots want the same row
    _launch(slots, src, have)
    torch.cuda.synchronize()
    assert (slots[0] == 31).all() and (slots[1] == 31).all() and (slots[2] == 30).all()


def test_partial_skip_mixed_launch():
    # one launch where some slots skip and some fetch: only the misses move
    rb = _align8(64)
    host = _mk_store(4, rb, 50, pinned=True)
    slots = torch.zeros(2, rb, dtype=torch.uint8, device="cuda")
    have = torch.full((2,), -1, dtype=torch.long, device="cuda")
    src = _addrs(host, [0, 1])
    _launch(slots, src, have)
    torch.cuda.synchronize()
    have.copy_(src)
    host[0].fill_(200)
    host[1].fill_(201)
    src2 = torch.stack([src[0], _addrs(host, [3])[0]])   # slot0 same, slot1 new
    _launch(slots, src2, have)
    torch.cuda.synchronize()
    assert (slots[0] == 50).all()      # skipped (stale by design)
    assert (slots[1] == 53).all()      # fetched the new row


def test_engine_traffic_counters_hand_counted():
    pytest.importorskip("nf4_grouped")
    from experts4bit_qlora import Experts4bit
    from experts4bit_qlora.pipelined import (
        disable_pipelined_residency, enable_pipelined_residency)

    torch.manual_seed(0)
    E, H, inter, k = 8, 128, 64, 3
    gate_up = torch.randn(E, 2 * inter, H)
    down = torch.randn(E, H, inter)
    mod = Experts4bit.from_float(gate_up_proj=gate_up.cuda(), down_proj=down.cuda(),
                                 compute_dtype=torch.bfloat16, has_gate=True)
    enable_pipelined_residency(mod, [torch.tensor([0, 1])], device="cuda", k_slots=k)
    st = mod._pipelined
    # Counting is off by default (the two reductions cost ~8.5% of the fetch), and
    # `traffic()` RAISES rather than reporting zeros when it is off — so this must
    # opt in before any fetch, or the witness below would be asserting on nothing.
    st.count_traffic = True
    rb = st.row_bytes
    hs = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")

    def go(ids):
        ti = torch.tensor([ids], device="cuda")
        tw = torch.full((1, k), 1.0 / k, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            mod(hs, ti, tw)

    # prime left every slot holding expert 0 (have == addr(0)).
    go([0, 1, 5])   # slots 0,1 are HOT -> read in place, no copy; slot2 cold miss
    go([0, 1, 5])   # all match -> zero new traffic
    go([2, 3, 6])   # three cold misses (2,3 are cold: hot set is {0,1})
    t = st.traffic()
    # hot_d2d is 0 BY CONSTRUCTION since the in-place hot path: a hot expert is
    # read from its resident row in the shared store, so no row is ever copied
    # for it. This assertion is the regression witness -- if it moves off zero, a
    # hot lane is being gathered into a slot again. The cold accounting is
    # unchanged (4 misses), which is what shows the change is a re-copy removal
    # and not a change to what gets streamed.
    assert t["hot_d2d_bytes"] == 0, t
    assert t["cold_pcie_bytes"] == 4 * rb, t
    # The final call routed 2,3,6 — all cold against hot set {0,1} — so every lane
    # must name a gathered slot. (The hot-lane side of the dispatch is asserted in
    # tests/test_pipelined.py, where the routing is held hot on purpose.)
    assert bool((st.row_idx >= st.n_hot).all()), st.row_idx
    disable_pipelined_residency(mod)


def test_interpreter_mode_refused_loudly(monkeypatch):
    # the engine must raise a pointed error, never reach the kernel (which
    # would segfault the interpreter on a raw device pointer)
    pytest.importorskip("nf4_grouped")
    from experts4bit_qlora import Experts4bit
    from experts4bit_qlora.pipelined import enable_pipelined_residency

    torch.manual_seed(0)
    mod = Experts4bit.from_float(gate_up_proj=torch.randn(4, 128, 128).cuda(),
                                 down_proj=torch.randn(4, 128, 64).cuda(),
                                 compute_dtype=torch.bfloat16, has_gate=True)
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    with pytest.raises(RuntimeError, match="interpreter"):
        enable_pipelined_residency(mod, [torch.tensor([0])], device="cuda", k_slots=2)


def test_segment_writes_only_its_slice_of_the_row():
    """A segmented source writes at a word offset and must not touch its neighbours.

    This is what lets the engine read offload's homes in place (they group by TENSOR)
    instead of baking a second full-size arena from them: an expert becomes four
    contiguous runs written into one row. An off-by-one in the offset or the length
    would corrupt the adjacent segment -- another expert's real bytes, so the GEMM
    would still run and simply be wrong.
    """
    rb = _align8(8 * 16)                      # 16 int64 words
    host = _mk_store(4, rb, 60, pinned=True)  # row e is filled with byte 60+e
    slots = torch.full((2, rb), 0xAB, dtype=torch.uint8, device="cuda")
    have = torch.full((2,), -1, dtype=torch.long, device="cuda")

    seg_off, seg_len = 4, 5                   # words: write [4,9), leave the rest
    ident = _addrs(host, [1, 2])
    src = ident + seg_off * 8                 # read the matching slice of the source
    _launch(slots, src, have, block=16, ident=ident,
            dst_word_off=seg_off, n_words=seg_len)
    torch.cuda.synchronize()

    lo, hi = seg_off * 8, (seg_off + seg_len) * 8
    for lane, e in enumerate((1, 2)):
        assert (slots[lane, lo:hi] == 60 + e).all(), f"lane {lane} segment wrong"
    assert (slots[:, :lo] == 0xAB).all(), "wrote before the segment"
    assert (slots[:, hi:] == 0xAB).all(), "wrote past the segment"


def test_identity_drives_the_skip_not_the_read_address():
    """All segments of one expert must skip or copy together, so the skip test keys on
    IDENTITY rather than the read address: with a segmented source those differ."""
    rb = _align8(64)
    host = _mk_store(4, rb, 70, pinned=True)
    slots = torch.zeros(2, rb, dtype=torch.uint8, device="cuda")
    ident = _addrs(host, [1, 2])
    src = _addrs(host, [3, 3])                # a DIFFERENT but valid read address

    _launch(slots, src, ident.clone(), block=16, ident=ident)   # have == ident -> skip
    torch.cuda.synchronize()
    assert (slots == 0).all(), "copied despite identity == have"

    have = torch.full((2,), -1, dtype=torch.long, device="cuda")
    _launch(slots, src, have, block=16, ident=ident)            # now it must copy
    torch.cuda.synchronize()
    assert (slots == 73).all(), "did not copy from the read address when identity differed"


def test_traffic_refuses_when_counting_is_off():
    """Disabled counters must RAISE, not report zeros.

    `hot_d2d_bytes == 0` is a regression witness. If `traffic()` returned zeros
    when counting was off, that assertion would pass while measuring nothing —
    the counters would read perfect precisely because they were never
    incremented. This is the arm on that failure mode.
    """
    pytest.importorskip("nf4_grouped")
    from experts4bit_qlora import Experts4bit
    from experts4bit_qlora.pipelined import enable_pipelined_residency

    torch.manual_seed(0)
    E, H, inter, k = 8, 128, 64, 3
    mod = Experts4bit.from_float(
        gate_up_proj=torch.randn(E, 2 * inter, H).cuda(),
        down_proj=torch.randn(E, H, inter).cuda(),
        compute_dtype=torch.bfloat16, has_gate=True)
    enable_pipelined_residency(mod, [torch.tensor([0, 1])], device="cuda", k_slots=k)
    st = mod._pipelined

    assert st.count_traffic is False, "traffic counting must default OFF"
    with pytest.raises(RuntimeError, match="traffic counting is disabled"):
        st.traffic()

    st.count_traffic = True
    hs = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    ti = torch.tensor([[2, 3, 6]], device="cuda")
    tw = torch.full((1, k), 1.0 / k, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        mod(hs, ti, tw)
    t = st.traffic()                       # now it reports
    assert t["cold_pcie_bytes"] > 0, "enabled counters must see the cold misses"
