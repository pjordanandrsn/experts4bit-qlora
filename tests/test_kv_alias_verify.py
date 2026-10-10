# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Alias slots for a speculative verify (lane SD2, ``bench/sd2/PREREG-sd2.md`` B1 and B2).

A verify of ``k + 1`` rows runs as a decode bucket over ``[slot] + aliases``. :meth:`Fp8PagedKV.alias_bind` copies the
slot's block-table row into each alias and sets alias ``i``'s length to the slot's plus ``i + 1``, so the batch append
writes row ``i`` into the slot's own blocks at its position plus ``i``. After the step,
:meth:`Fp8PagedKV.set_len_device` sets the slot's length to ``base + accepted + 1``, the step's last length write.

The maintainer's four conditions (bus, 2026-10-10T06:55:22Z) are tested here at the pool. The fourth (bucket 3 is for
the verify only) belongs to the runner and is tested with it.
1. the alias slots sit outside ``scratch``, and that is asserted;
2. the order of the length writes: the publish, then the rollback, last; the alias lengths re-set every step;
3. a verify that straddles a block boundary writes into the right block.

The addressing reference is the pool's per-row graph append (:meth:`Fp8PagedKV.append_graph_t1`), which the fused
batch append is bitwise against (grouped-nf4-gemm's ``test_fp8_kv_append.py``). It runs on CPU.
"""
import pytest
import torch

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV

H, D = 2, 32


def _kv(monkeypatch=None, *, step_select=True, L=3, batch=2, scratch=4, alias=3, max_tokens=64):
    if monkeypatch is not None:
        monkeypatch.setenv("E4B_KV_STEP_SELECT", "1" if step_select else "0")
    return Fp8PagedKV(L, H, D, batch=batch, max_tokens_per_seq=max_tokens, device="cpu",
                      scratch_slots=scratch, alias_slots=alias)


# ---------------------------------------------------------------------------- condition 1: outside scratch --

def test_alias_slots_follow_the_scratch_slots_and_own_no_rows():
    kv = _kv()
    assert kv.scratch == [2, 3, 4, 5] and kv.alias == [6, 7, 8]
    assert not set(kv.alias) & set(kv.scratch)
    assert all((layer, a) not in kv._rows for layer in range(kv.L) for a in kv.alias)
    # every pool row is a real slot's or a scratch slot's: aliases add none
    assert kv._n_rows == kv.B * kv.blocks_per_seq + kv.n_scratch
    assert kv.seq_lens.shape[1] == kv.B + kv.n_scratch + kv.n_alias


def test_resetting_the_scratch_lengths_leaves_the_aliases_alone():
    kv = _kv()
    kv.seq_lens[:, kv.alias] = 9
    kv.reset_scratch_lens()
    assert int(kv.seq_lens[:, kv.alias].min()) == 9 and int(kv.seq_lens[:, kv.scratch].max()) == 0


def test_no_alias_slots_by_default():
    kv = Fp8PagedKV(2, H, D, batch=2, max_tokens_per_seq=32, device="cpu", scratch_slots=2)
    assert kv.alias == [] and kv.n_alias == 0 and kv.seq_lens.shape[1] == 4
    assert kv.alias_bind(0, 0) == []


def test_alias_bind_refuses_too_many_rows_and_a_slot_that_is_not_real():
    kv = _kv()
    with pytest.raises(ValueError, match="alias rows"):
        kv.alias_bind(0, 4)
    for bad in (kv.scratch[0], kv.alias[0], -1):
        with pytest.raises(ValueError, match="real slot"):
            kv.alias_bind(bad, 1)


def test_alias_bind_copies_the_table_and_staggers_the_lengths():
    kv = _kv()
    for layer in range(kv.L):
        kv._ensure_blocks(layer, 1, 2)
    kv.seq_lens[:, 1] = torch.tensor([20, 21, 22], dtype=kv.seq_lens.dtype)   # per-layer lengths may differ
    got = kv.alias_bind(1, 3)
    assert got == kv.alias
    for layer in range(kv.L):
        for i, a in enumerate(got):
            assert torch.equal(kv.block_table[layer][a], kv.block_table[layer][1])
            assert int(kv.seq_lens[layer, a]) == int(kv.seq_lens[layer, 1]) + i + 1
    assert kv.alias_bind(1, 1) == kv.alias[:1]


# ------------------------------------------------------------------------- condition 2: the order of writes --

def test_the_rollback_is_the_last_length_write(monkeypatch):
    kv = _kv(monkeypatch, step_select=True)
    slot, base, k = 0, 10, 3
    for layer in range(kv.L):
        kv._ensure_blocks(layer, slot, kv.blocks_per_seq - 1)
    kv.seq_lens[:, slot] = base
    rows = [slot] + kv.alias_bind(slot, k)
    st = kv.graph_bucket(k + 1)
    kv.graph_bucket_load(st, rows)
    # attention reads row i at base + i + 1: the past plus draft tokens 0..i (S2-lite's stagger)
    for layer in range(kv.L):
        assert st["lens"][layer].tolist() == [base + i + 1 for i in range(k + 1)]
        assert all(torch.equal(st["tbl"][layer][i], kv.block_table[layer][slot]) for i in range(k + 1))
    kv.graph_bucket_publish(st)                       # +1 on every bound row
    assert int(kv.seq_lens[0, slot]) == base + 1
    accepted = 2
    kv.set_len_device(slot, torch.tensor(base + accepted + 1))
    assert kv.seq_lens[:, slot].tolist() == [base + accepted + 1] * kv.L
    kv.note_len(slot, base + accepted + 1)
    assert [kv._seen[layer][slot] for layer in range(kv.L)] == [base + accepted + 1] * kv.L
    # the wrong order would be off by one: the publish after the rollback pushes the slot past its accepted length
    kv.graph_bucket_publish(st)
    assert int(kv.seq_lens[0, slot]) == base + accepted + 2


def test_the_alias_lengths_are_re_set_each_step(monkeypatch):
    kv = _kv(monkeypatch, step_select=True)
    slot = 1
    for layer in range(kv.L):
        kv._ensure_blocks(layer, slot, kv.blocks_per_seq - 1)
    kv.seq_lens[:, slot] = 5
    kv.alias_bind(slot, 2)
    kv.seq_lens[:, kv.alias] += 40                    # whatever the last step left
    kv.set_len_device(slot, torch.tensor(7))
    kv.alias_bind(slot, 2)
    assert kv.seq_lens[0, kv.alias[:2]].tolist() == [8, 9]


def test_without_step_select_the_appends_bump_the_bound_rows(monkeypatch):
    kv = _kv(monkeypatch, step_select=False)
    slot, base = 0, 6
    for layer in range(kv.L):
        kv._ensure_blocks(layer, slot, kv.blocks_per_seq - 1)
    kv.seq_lens[:, slot] = base
    rows = [slot] + kv.alias_bind(slot, 2)
    st = kv.graph_bucket(3)
    kv.graph_bucket_load(st, rows)
    assert "lens" not in st
    kv.graph_bucket_publish(st)                       # a no-op without the switch
    assert int(kv.seq_lens[0, slot]) == base


# ----------------------------------------------------------------------- condition 3: the block boundary --

def _rows(n, layer):
    g = torch.Generator().manual_seed(1000 + layer)
    return torch.randn(n, H, D, generator=g), torch.randn(n, H, D, generator=g)


@pytest.mark.parametrize("base,k", [(14, 3), (15, 1), (16, 2), (31, 3)])
def test_a_verify_that_straddles_a_block_writes_into_the_right_block(base, k):
    """Pool A takes the verify's k + 1 rows through the slot and its aliases, each at its device-computed position.
    Pool B appends the same rows to the slot directly. Their bytes must match exactly."""
    slot = 1
    a, b = _kv(), _kv()
    past_k, past_v = torch.randn(base, H, D), torch.randn(base, H, D)
    for kv in (a, b):
        for layer in range(kv.L):
            kv.append(layer, slot, past_k, past_v)
    # A: blocks claimed up front, as a graphed slot has them, then one row per bound slot through the graph append
    for layer in range(a.L):
        a._ensure_blocks(layer, slot, a.blocks_per_seq - 1)
    a.graph_mode_init(seq=slot)
    bound = [slot] + a.alias_bind(slot, k)
    assert [int(a.seq_lens[0, s_]) for s_ in bound] == [base + i for i in range(k + 1)]
    for layer in range(a.L):
        rk, rv = _rows(k + 1, layer)
        for i, s_ in enumerate(bound):
            a._g_seq = s_
            a.append_graph_t1(layer, rk[i:i + 1], rv[i:i + 1])
    a.set_len_device(slot, torch.tensor(base + k + 1))
    a.note_len(slot, base + k + 1)
    # B: the same rows appended to the slot directly, then the same blocks claimed
    for layer in range(b.L):
        rk, rv = _rows(k + 1, layer)
        b.append(layer, slot, rk, rv)
        b._ensure_blocks(layer, slot, b.blocks_per_seq - 1)
    assert a._rows == b._rows
    for layer in range(a.L):
        assert torch.equal(a.kp.dev[layer], b.kp.dev[layer]), f"K bytes differ at layer {layer}"
        assert torch.equal(a.vp.dev[layer], b.vp.dev[layer]), f"V bytes differ at layer {layer}"
        assert int(a.seq_lens[layer, slot]) == int(b.seq_lens[layer, slot]) == base + k + 1


def test_an_alias_write_lands_in_the_slots_second_block():
    """At base 15 with k = 2, rows 1 and 2 (positions 16 and 17) belong to the slot's block 1, through the aliased
    table, and nothing lands in another slot's rows."""
    kv = _kv(batch=2)
    slot, other = 0, 1
    for s_ in (slot, other):
        for layer in range(kv.L):
            kv._ensure_blocks(layer, s_, kv.blocks_per_seq - 1)
    kv.seq_lens[:, slot] = 15
    kv.graph_mode_init(seq=slot)
    before = kv.kp.dev.clone()
    bound = [slot] + kv.alias_bind(slot, 2)
    rk, rv = _rows(3, 0)
    for i, s_ in enumerate(bound):
        kv._g_seq = s_
        kv.append_graph_t1(0, rk[i:i + 1], rv[i:i + 1])
    changed = {int(r) for r in torch.nonzero((kv.kp.dev[0] != before[0]).any(dim=1)).flatten()}
    blk0, blk1 = int(kv.block_table[0][slot, 0]), int(kv.block_table[0][slot, 1])
    assert changed == {blk0, blk1}
    assert not changed & set(kv._rows[(0, other)])
