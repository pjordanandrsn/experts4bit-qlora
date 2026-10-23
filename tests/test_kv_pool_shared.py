# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Incremental block claims under the decode graphs (lever G1, ``bench/kv-pool-g/DESIGN.md``), on the real KV pool and
buckets with a stand-in for the captured graph.

Pinned: with ``incremental_claims`` every real row of every replay owns the block its write position lands in, under
the synchronous path and the lookahead; the streams equal claim-all's; a slot holds only the blocks it has reached, so
fewer rows are held than under claim-all, and every row is back on the free lists at idle; the claim arithmetic,
including a write at a block boundary; the speculative claim reaches base + k_eff and no further than the budget allows;
and the server's ``E4B_PAGED_KV_POOL`` parses, refuses nonsense and defaults to claim-all.
"""
import types

import pytest
import torch

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
from experts4bit_qlora.engines.paged_runner import PagedModelRunner
from experts4bit_qlora.engines.scheduler import ContinuousScheduler

V = 97
PROMPTS = [[3, 17, 42, 9, 11, 5, 88, 23, 7], [5, 1, 99, 64, 2], [120, 7, 7, 31, 2, 9, 14, 60, 3, 3, 8, 1],
           [15, 250, 4, 4, 81, 6, 12], list(range(1, 33))]          # the last one is a 32-token prompt (a block edge)
MAX_NEW = [40, 22, 9, 3, 17]
MAX_TOKENS = 256                                                  # 16 blocks a slot


class OwnershipCheckingDecode:
    """The decode graph's contract (as ``tests/test_decode_lookahead.py``'s stand-in), plus the G1 check: at replay,
    every real row's slot owns the block its position writes, in every pool layer."""

    def __init__(self, runner, b, hist, seen):
        self.r, self.buf, self.hist, self.seen = runner, runner._bufs[b], hist, seen

    def replay(self):
        kv, buf = self.r.kv, self.buf
        st = buf["st"]
        sl = st["slot_l"]
        for i, slot in enumerate(sl.tolist()):
            if slot in kv.scratch:
                continue
            pos = int(buf["pos"].view(-1)[i])
            for layer in self.r.pool_layers:
                owned = len(kv._rows.get((layer, slot), ()))
                assert owned > pos // kv.bt, f"slot {slot} writes position {pos} but owns {owned} blocks in layer {layer}"
            self.seen.append(sum(len(v) for (_, seq), v in kv._rows.items() if seq not in kv.scratch))
        lens = st["lens"][0].to(torch.long)
        h = self.hist.index_select(0, sl)
        h = (h * 1_000_003 + buf["ids"].view(-1) * 7_919 + buf["pos"].view(-1) * 104_729 + lens * 13) % 2_147_483_647
        self.hist.index_copy_(0, sl, h)
        buf["tok"].copy_(h % V)


def _runner(monkeypatch, incremental, bulk=True):
    monkeypatch.setenv("E4B_KV_STEP_SELECT", "1")
    kv = Fp8PagedKV(2, 2, 32, batch=4, max_tokens_per_seq=MAX_TOKENS, device="cpu", scratch_slots=4)
    r = PagedModelRunner(torch.nn.Module(), kv, device="cpu", bulk_kv=bulk)
    r.incremental_claims = incremental
    r.enable_decode_graphs((1, 2, 4), capture=False, verbose=False)
    hist = torch.zeros(kv.B + kv.n_scratch, dtype=torch.long)
    seen = []
    for b in (1, 2, 4):
        r._graphs[b] = OwnershipCheckingDecode(r, b, hist, seen)

    def prefill(chunks):
        first = {}
        for rid, start, take in chunks:
            slot, n = r.slot_of[rid], len(r.tokens[rid])
            if start + take >= n:
                kv.seq_lens[:, slot] = n
                hist[slot] = sum((i + 1) * t for i, t in enumerate(r.tokens[rid])) % 2_147_483_647
                first[rid] = int(hist[slot]) % V
                r.tokens[rid].append(first[rid])
                r.pos_of[rid] = n + 1
        return first

    r.run_prefill = prefill
    return r, kv, seen


def _serve(monkeypatch, incremental, lookahead, bulk=True):
    r, kv, seen = _runner(monkeypatch, incremental, bulk)
    s = ContinuousScheduler(runner=r, max_seqs=4, kv_slots=4, chunk_tokens=64, lookahead=lookahead)
    rids = [s.add_request(p, m) for p, m in zip(PROMPTS, MAX_NEW)]
    s.run_until_idle(max_steps=2000)
    done = {q.rid: q.out for q in s.done}
    return [done[i] for i in rids], kv, seen


@pytest.mark.parametrize("lookahead", [False, True])
@pytest.mark.parametrize("bulk", [True, False])
def test_incremental_claims_cover_every_write_and_keep_the_streams(monkeypatch, lookahead, bulk):
    out_all, kv_all, seen_all = _serve(monkeypatch, False, lookahead, bulk)
    out_inc, kv_inc, seen_inc = _serve(monkeypatch, True, lookahead, bulk)
    assert out_inc == out_all and [len(o) for o in out_inc] == MAX_NEW
    assert seen_inc and max(seen_inc) < max(seen_all)          # a slot holds what it reached, not its whole ceiling
    for kv in (kv_all, kv_inc):                                # at idle only the scratch slots hold rows (one block each)
        assert {seq for (_layer, seq) in kv._rows} <= set(kv.scratch)
        held = {layer: sum(len(v) for (lyr, _), v in kv._rows.items() if lyr == layer) for layer in range(kv.L)}
        assert all(len(kv._free[layer]) + held[layer] == kv._n_rows for layer in range(kv.L))


@pytest.mark.parametrize("lookahead", [False, True])
def test_the_ownership_check_fails_a_step_that_reads_an_unclaimed_block(monkeypatch, lookahead):
    """The negative control: with incremental claims on and the per-step claim removed, the stand-in graph's ownership
    check trips on the first replay, so the passing test above is not vacuous."""
    monkeypatch.setattr(PagedModelRunner, "_claim_through", lambda self, slot, write_pos: None)
    monkeypatch.setattr(PagedModelRunner, "_claim_step", lambda self, write_pos: None)
    with pytest.raises(AssertionError, match="owns 0 blocks"):
        _serve(monkeypatch, True, lookahead)


@pytest.mark.parametrize("prompt_len", [15, 16, 17, 32, 33])
@pytest.mark.parametrize("incremental", [True, False])
def test_the_flush_claims_the_prompt_and_its_first_decode_block(monkeypatch, prompt_len, incremental):
    """Incremental: the prompt flush claims ceil((L + 1) / 16) blocks, the prompt's positions 0..L-1 and the first
    decode's write at L (at L = 16 or 32 that opens a block). Claim-all: every block the slot can reach."""
    r, kv, _ = _runner(monkeypatch, incremental)
    r.bind(0, 0, list(range(1, prompt_len + 1)))
    for layer in r.pool_layers:
        k = torch.randn(prompt_len, kv.Hs[layer], kv.Ds[layer], dtype=torch.bfloat16)
        r.ctx.staging[(layer, 0)] = ([k], [k.clone()])
    r._flush_bulk(0, 0)
    want = prompt_len // kv.bt + 1 if incremental else kv.blocks_per_seq
    assert all(len(kv._rows[(layer, 0)]) == want for layer in r.pool_layers), (prompt_len, want)
    assert int(kv.seq_lens[0, 0]) == prompt_len


def test_claim_many_gives_the_rows_consecutive_claims_give_in_one_write(monkeypatch):
    wants = {0: 2, 1: 0, 2: 5}
    _, kv_a, _ = _runner(monkeypatch, True)
    for seq, upto in wants.items():
        kv_a.claim_blocks(seq, upto)
    _, kv_b, _ = _runner(monkeypatch, True)
    assert kv_b.claim_many(wants) == kv_b.L * sum(u + 1 for u in wants.values())
    assert kv_a._rows == kv_b._rows and kv_a._free == kv_b._free
    assert torch.equal(kv_a._bt_all, kv_b._bt_all)
    assert kv_b.claim_many(wants) == 0                         # already owned: nothing written
    assert kv_b.claim_many({0: 3}) == kv_b.L                    # one more block, every layer


def test_claim_many_pops_nothing_when_the_step_does_not_fit(monkeypatch):
    _, kv, _ = _runner(monkeypatch, True)
    kv.claim_many({0: 1})                                       # something already owned, so "unchanged" is not all zeros
    free_before, rows_before, bt_before = [list(f) for f in kv._free], dict(kv._rows), kv._bt_all.clone()
    with pytest.raises(RuntimeError, match="out of KV blocks"):
        kv.claim_many({seq: kv.blocks_per_seq for seq in range(kv.B)})   # one block more per slot than the pool has
    assert kv._free == free_before and kv._rows == rows_before and torch.equal(kv._bt_all, bt_before)


def test_claim_many_never_writes_one_table_entry_twice(monkeypatch):
    """index_put_ with repeated indices has no defined write order on CUDA: one step's entries must be unique."""
    _, kv, _ = _runner(monkeypatch, True)
    seen = []
    real = torch.Tensor.index_put_

    def spy(t, indices, values, accumulate=False):
        assert not accumulate
        seen.append(list(zip(*(i.tolist() for i in indices))))
        return real(t, indices, values, accumulate)
    monkeypatch.setattr(torch.Tensor, "index_put_", spy)
    kv.claim_many({0: 3, 2: 1, 1: 0})
    kv.claim_many({0: 5, 1: 2, 3: 0})
    assert seen and all(len(step) == len(set(step)) for step in seen)
    assert all(kv._bt_all[layer, seq, blk].item() == kv._rows[(layer, seq)][blk]
               for step in seen for layer, seq, blk in step)


def test_the_claim_reaches_the_write_positions_block_and_no_further(monkeypatch):
    r, kv, _ = _runner(monkeypatch, True)
    r.bind(0, 0, [1] * 32)
    r._ensure_graph_ready(0)
    assert not kv._rows.get((0, 0))                            # incremental: the claim-all path claims nothing
    r._claim_through(0, 31)
    assert len(kv._rows[(0, 0)]) == 2                          # positions 0..31: blocks 0, 1
    r._claim_through(0, 32)
    assert len(kv._rows[(0, 0)]) == 3                          # position 32 opens block 2 (a 32-token prompt's first decode)
    r._claim_through(0, 40)
    assert len(kv._rows[(0, 0)]) == 3                          # still block 2: idempotent
    r._claim_through(0, 10_000)
    assert len(kv._rows[(0, 0)]) == kv.blocks_per_seq          # capped at the slot's ceiling
    assert all(len(kv._rows[(layer, 0)]) == kv.blocks_per_seq for layer in r.pool_layers)


def test_claim_all_is_unchanged(monkeypatch):
    r, kv, _ = _runner(monkeypatch, False)
    r.bind(0, 0, [1] * 5)
    r._ensure_graph_ready(0)
    assert len(kv._rows[(0, 0)]) == kv.blocks_per_seq          # the first graphed decode claims every reachable block


@pytest.mark.parametrize("budget,want_last", [(1 << 30, (40 - 1 + 3) // 16), (2, (40 - 1 + 1) // 16), (1, (40 - 1) // 16)])
def test_the_speculative_claim_is_bounded_by_the_budget(monkeypatch, budget, want_last):
    r, kv, _ = _runner(monkeypatch, True)
    r.bind(0, 0, [1] * 39)
    r.pos_of[0] = 40
    r.spec = types.SimpleNamespace(k=3, step=lambda rid, b: None)
    r._budgets = {0: budget}
    assert r._run_decode_spec(0) is None
    assert len(kv._rows[(0, 0)]) == want_last + 1


def test_the_server_switch(monkeypatch):
    from experts4bit_qlora import serve_paged
    from experts4bit_qlora.serve_paged import PagedServeConfig
    monkeypatch.setattr(serve_paged, "_capability", lambda device: None)
    monkeypatch.setenv("E4B_PAGED_MODEL", "Qwen/Qwen3-30B-A3B")
    monkeypatch.delenv("E4B_PAGED_KV_POOL", raising=False)
    assert PagedServeConfig.from_env().kv_pool == "claim-all"
    monkeypatch.setenv("E4B_PAGED_KV_POOL", " Shared ")
    assert PagedServeConfig.from_env().kv_pool == "shared"
    for bad in ("on", "1", "auto"):
        monkeypatch.setenv("E4B_PAGED_KV_POOL", bad)
        with pytest.raises(ValueError, match="E4B_PAGED_KV_POOL"):
            PagedServeConfig.from_env().validate()
