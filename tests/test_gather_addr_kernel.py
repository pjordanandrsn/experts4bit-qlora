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


def _launch(slots, src, have, block=16):
    rb = slots.shape[1]
    assert rb % 8 == 0
    rw = rb // 8
    kern = _gather_kernel()
    grid = (slots.shape[0], -(-rw // block))
    kern[grid](slots.view(torch.int64), src, have, rw, BLOCK=block, num_warps=1)


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
    rb = st.row_bytes
    hs = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")

    def go(ids):
        ti = torch.tensor([ids], device="cuda")
        tw = torch.full((1, k), 1.0 / k, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            mod(hs, ti, tw)

    # prime left every slot holding expert 0 (have == addr(0)).
    go([0, 1, 5])   # slot0 skip; slot1 hot miss; slot2 cold miss
    go([0, 1, 5])   # all match -> zero new traffic
    go([2, 3, 6])   # three cold misses (2,3 are cold: hot set is {0,1})
    t = st.traffic()
    assert t["hot_d2d_bytes"] == 1 * rb, t
    assert t["cold_pcie_bytes"] == 4 * rb, t
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
