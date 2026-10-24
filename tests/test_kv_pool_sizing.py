# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lever G3 (``bench/kv-pool-g/DESIGN.md``): under ``E4B_PAGED_KV_POOL=shared`` the KV pool holds the rows free memory
leaves, which can be fewer than its slots could ever reach.

Pinned: ``Fp8PagedKV(pool_rows=...)`` allocates that many rows and refuses a pool that cannot hold the scratch rows plus
one full sequence, or one above the claim-all size; ``paged_kv_pool_bytes(pool_rows=...)`` is what it allocates; a
claim-all runner refuses a short pool. The design's test 2 on the real pool and buckets with the stand-in graph: 4 slots
and room for 2.5 sequences' blocks, so admission waits, nothing raises, every request finishes with the tokens it gets
under claim-all, no step ever reads an unclaimed block, and at idle only the scratch slots hold rows.
``serve_recipe.size_shared_pool``: the most rows that fit, the widest width that holds the floor and one full sequence,
the claim-all size when it fits whole, and the claim-all pool when free memory is unknown or the solver places the
experts. ``serve_paged.resolve_kv_pool``: claim-all never comes there; an explicit width is kept.
"""
import pytest
import torch

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
from experts4bit_qlora.engines.paged_runner import PagedModelRunner
from experts4bit_qlora.engines.scheduler import ContinuousScheduler
from test_kv_pool_shared import V, OwnershipCheckingDecode

GIB = 2**30
FREE_5090 = int(31.3 * GIB)
FREE_4090 = int(23.4 * GIB)
MAX_TOKENS = 256                                     # 16 blocks a slot
PROMPTS = [[(7 * i + j) % 90 + 3 for j in range(n)] for i, n in enumerate([150, 40, 170, 9, 120, 200, 33])]
MAX_NEW = [50, 30, 60, 20, 80, 40, 100]              # needs 13, 5, 15, 2, 13, 15, 9 blocks; 4 slots
POOL_ROWS = 4 + 40                                   # 4 scratch rows + 2.5 sequences' blocks


def _runner(monkeypatch, incremental, pool_rows=None):
    monkeypatch.setenv("E4B_KV_STEP_SELECT", "1")
    kv = Fp8PagedKV(2, 2, 32, batch=4, max_tokens_per_seq=MAX_TOKENS, device="cpu", scratch_slots=4,
                    pool_rows=pool_rows)
    r = PagedModelRunner(torch.nn.Module(), kv, device="cpu", bulk_kv=True)
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


def _serve(monkeypatch, shared, lookahead, pool_rows=None):
    r, kv, seen = _runner(monkeypatch, shared, pool_rows)
    s = ContinuousScheduler(runner=r, max_seqs=4, kv_slots=4, chunk_tokens=64, lookahead=lookahead,
                            kv_blocks=(kv._n_rows - kv.n_scratch) if shared else None, block_tokens=kv.bt)
    rids = [s.add_request(p, m) for p, m in zip(PROMPTS, MAX_NEW)]
    peak = 0
    while s.queue or s.active:
        s.step()
        peak = max(peak, max(sum(len(v) for (lyr, seq), v in kv._rows.items() if lyr == layer and seq not in kv.scratch)
                             for layer in range(kv.L)))
    done = {q.rid: q.out for q in s.done}
    return [done[i] for i in rids], kv, s, peak


def test_a_pool_of_pool_rows_holds_that_many_rows_and_refuses_what_cannot_serve():
    kv = Fp8PagedKV(2, 2, 32, batch=4, max_tokens_per_seq=MAX_TOKENS, device="cpu", scratch_slots=4, pool_rows=POOL_ROWS)
    assert kv._n_rows == POOL_ROWS and kv.claim_all_rows == 4 * 16 + 4 and kv.shared_pool
    assert all(len(kv._free[layer]) == POOL_ROWS - kv.n_scratch for layer in range(kv.L))   # scratch rows are held
    full = Fp8PagedKV(2, 2, 32, batch=4, max_tokens_per_seq=MAX_TOKENS, device="cpu", scratch_slots=4,
                      pool_rows=4 * 16 + 4)
    assert not full.shared_pool
    for bad in (4 + 15, 4 * 16 + 5):                 # below scratch + one sequence; above the claim-all size
        with pytest.raises(ValueError, match="pool_rows"):
            Fp8PagedKV(2, 2, 32, batch=4, max_tokens_per_seq=MAX_TOKENS, device="cpu", scratch_slots=4, pool_rows=bad)


@pytest.mark.parametrize("rows", [None, 4 + 16, POOL_ROWS])
def test_pool_bytes_are_what_a_shared_pool_allocates(rows):
    pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm")
    from experts4bit_qlora.serve_recipe import paged_kv_pool_bytes
    kv = Fp8PagedKV(3, 2, 64, batch=4, max_tokens_per_seq=MAX_TOKENS, device="cpu", scratch_slots=4, pool_rows=rows)
    allocated = sum(t.numel() * t.element_size() for t in (kv.kp.dev, kv.vp.dev, kv._bt_all, kv.seq_lens))
    assert paged_kv_pool_bytes(3, 2, 64, batch=4, max_tokens_per_seq=MAX_TOKENS, scratch_slots=4,
                               pool_rows=rows) == allocated


def test_a_claim_all_runner_refuses_a_short_pool(monkeypatch):
    r, kv, _ = _runner(monkeypatch, False, POOL_ROWS)
    with pytest.raises(RuntimeError, match="needs incremental_claims"):
        r._ensure_graph_ready(0)


@pytest.mark.parametrize("lookahead", [False, True])
def test_a_pool_smaller_than_its_slots_can_reach_waits_and_every_request_finishes(monkeypatch, lookahead):
    """The design's test 2: every request finishes with exactly the tokens claim-all gives it (the stand-in's tokens
    depend on the request, not the batch), admission waited on blocks, no replay read an unclaimed block (the stand-in
    asserts it), the pool never held more than its 40 rows for the slots, and at idle only the scratch slots hold rows."""
    ref, _, _, ref_peak = _serve(monkeypatch, False, lookahead)
    out, kv, s, peak = _serve(monkeypatch, True, lookahead, POOL_ROWS)
    assert out == ref and [len(o) for o in out] == MAX_NEW
    assert s.kv_blocks == 40 and s.kv_blocks_waits > 0 and s.kv_blocks_reserved == 0
    assert peak <= 40 < ref_peak
    assert {seq for (_layer, seq) in kv._rows} <= set(kv.scratch)
    assert all(len(kv._free[layer]) == 40 for layer in range(kv.L))


def test_without_the_reservation_a_short_pool_runs_dry(monkeypatch):
    """The negative control: the same short pool admitted by slots alone (no kv_blocks) claims past its rows."""
    r, kv, _ = _runner(monkeypatch, True, POOL_ROWS)
    s = ContinuousScheduler(runner=r, max_seqs=4, kv_slots=4, chunk_tokens=64)
    for p, m in zip(PROMPTS, MAX_NEW):
        s.add_request(p, m)
    with pytest.raises(RuntimeError, match="out of KV blocks|does not fit|room"):
        s.run_until_idle(max_steps=5000)


# ---- sizing (serve_recipe.size_shared_pool, serve_paged.resolve_kv_pool)

def _qwen3_30b():
    tr = pytest.importorskip("transformers")
    pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm")
    from experts4bit_qlora.arch.topology import describe_moe
    return describe_moe(tr.Qwen3MoeConfig(
        hidden_size=2048, intermediate_size=6144, moe_intermediate_size=768, num_experts=128, num_experts_per_tok=8,
        num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=4, head_dim=128, vocab_size=151936,
        max_position_embeddings=40960, decoder_sparse_step=1, tie_word_embeddings=False))


def _need(topo, setup, rows, c):
    from dataclasses import replace

    from experts4bit_qlora.serve_recipe import MAX_SEQS_AUTO_MARGIN_BYTES, estimate_serve_footprint
    fp = estimate_serve_footprint(topo, replace(setup, max_seqs=c["max_seqs"], kv_pool_rows=rows))
    return fp.device_bytes + c["prefill_graph_reserve_bytes"] + c["prefill_graph_headroom_bytes"] + MAX_SEQS_AUTO_MARGIN_BYTES


def test_at_4096_tokens_a_slot_the_shared_pool_keeps_64_slots_where_claim_all_cannot():
    """On Qwen3-30B-A3B int4 at 4,096 tokens a slot, claim-all's 64 slots need ~34 GiB, so `auto` keeps 16 on the
    default list. The shared pool gives 64 slots the rows that fit: the most rows whose need fits, at least the floor."""
    from experts4bit_qlora.serve_recipe import (MAX_SEQS_AUTO_WIDTHS, SHARED_POOL_MIN_TOKENS_PER_SLOT, ServeSetup,
                                                 choose_max_seqs, size_shared_pool)
    topo = _qwen3_30b()
    st = ServeSetup(max_tokens_per_seq=4096, exp_int4=True, attn_int4=True)
    assert choose_max_seqs(topo, st, FREE_5090)["max_seqs"] == 16
    r = size_shared_pool(topo, st, FREE_5090, widths=MAX_SEQS_AUTO_WIDTHS)
    c = r["candidates"][0]
    assert r["max_seqs"] == 64 and r["pool_rows"] == c["pool_rows"] < c["claim_all_rows"]
    assert c["mean_tokens_per_slot"] >= SHARED_POOL_MIN_TOKENS_PER_SLOT and c["pool_rows"] >= c["least_rows"]
    assert c["need_bytes"] == _need(topo, st, c["pool_rows"], c) <= FREE_5090 < _need(topo, st, c["pool_rows"] + 1, c)
    assert "rows" in r["why"] and r["free_bytes"] == FREE_5090


def test_when_claim_all_fits_whole_the_shared_pool_is_the_claim_all_size():
    from experts4bit_qlora.serve_recipe import MAX_SEQS_AUTO_WIDTHS, ServeSetup, size_shared_pool
    topo = _qwen3_30b()
    r = size_shared_pool(topo, ServeSetup(max_tokens_per_seq=2048, exp_int4=True, attn_int4=True), FREE_5090,
                         widths=MAX_SEQS_AUTO_WIDTHS)
    assert r["max_seqs"] == 64 and r["pool_rows"] is None
    assert r["candidates"][0]["pool_rows"] == r["candidates"][0]["claim_all_rows"]


def test_a_width_that_cannot_hold_the_floor_is_not_offered():
    """On a 24 GB card at 4,096 tokens a slot, 64 slots cannot have 1,024 tokens each, so a narrower width is taken;
    and a free memory too small for any width keeps the narrowest width's claim-all pool."""
    from experts4bit_qlora.serve_recipe import MAX_SEQS_AUTO_WIDTHS_AUTO_BUCKETS, ServeSetup, size_shared_pool
    topo = _qwen3_30b()
    st = ServeSetup(max_tokens_per_seq=4096, exp_int4=True, attn_int4=True)
    r = size_shared_pool(topo, st, FREE_4090, widths=MAX_SEQS_AUTO_WIDTHS_AUTO_BUCKETS)
    offered = [c for c in r["candidates"] if c.get("fits")]
    assert all(c["mean_tokens_per_slot"] >= 1024 for c in offered)
    assert all(not c.get("fits") for c in r["candidates"] if c["max_seqs"] > r["max_seqs"])
    none = size_shared_pool(topo, st, 4 * GIB, widths=MAX_SEQS_AUTO_WIDTHS_AUTO_BUCKETS)
    assert none["max_seqs"] == 16 and none["pool_rows"] is None and "claim-all" in none["why"]


def test_a_refused_estimate_above_the_floor_counts_as_not_fitting(monkeypatch):
    """The bisection treats an estimate that refuses (no device total) as not fitting, rather than comparing None."""
    from experts4bit_qlora import serve_recipe as sr
    real = sr.estimate_serve_footprint
    topo = _qwen3_30b()
    st = sr.ServeSetup(max_tokens_per_seq=4096, exp_int4=True, attn_int4=True)
    ok = sr.size_shared_pool(topo, st, FREE_5090, widths=(64,))["candidates"][0]
    cap = ok["least_rows"] + 100                      # every pool above this refuses

    def fake(topology, setup):
        fp = real(topology, setup)
        if setup.kv_pool_rows is not None and setup.kv_pool_rows > cap:
            return sr.Footprint(items=(), refusals=("a refusal above the cap",))
        return fp
    monkeypatch.setattr(sr, "estimate_serve_footprint", fake)
    r = sr.size_shared_pool(topo, st, FREE_5090, widths=(64,))
    assert r["max_seqs"] == 64 and r["pool_rows"] == cap


def test_an_estimate_refused_even_at_the_floor_keeps_the_claim_all_pool_and_names_why(monkeypatch):
    """No estimate at all (every width refused at its floor): no crash, the claim-all pool, the refusal in the record."""
    from experts4bit_qlora import serve_recipe as sr
    monkeypatch.setattr(sr, "estimate_serve_footprint",
                        lambda topology, setup: sr.Footprint(items=(), refusals=("no estimate for this setup",)))
    r = sr.size_shared_pool(_qwen3_30b(), sr.ServeSetup(max_tokens_per_seq=4096), FREE_5090, widths=(64, 16))
    assert r["max_seqs"] == 16 and r["pool_rows"] is None
    assert [c["refused"] for c in r["candidates"]] == [["no estimate for this setup"]] * 2
    assert "refused every width" in r["why"] and "no estimate for this setup" in r["why"]


@pytest.mark.parametrize("free,placement,why", [(None, "all-vram", "unknown"), (FREE_5090, "solver", "solver")])
def test_unknown_memory_or_the_solver_keeps_the_claim_all_pool(free, placement, why):
    from experts4bit_qlora.serve_recipe import MAX_SEQS_AUTO_WIDTHS, ServeSetup, size_shared_pool
    r = size_shared_pool(_qwen3_30b(), ServeSetup(placement=placement, max_tokens_per_seq=4096), free,
                         widths=MAX_SEQS_AUTO_WIDTHS)
    assert r["max_seqs"] == 16 and r["pool_rows"] is None and why in r["why"]


def _cfg(monkeypatch, **env):
    from experts4bit_qlora.serve_paged import PagedServeConfig
    for k in [k for k in __import__("os").environ if k.startswith("E4B_")]:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return PagedServeConfig.from_env()


def test_resolve_kv_pool_leaves_claim_all_alone_and_keeps_an_explicit_width(monkeypatch):
    tr = pytest.importorskip("transformers")
    pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm")
    from experts4bit_qlora.serve_paged import resolve_kv_pool
    mc = tr.Qwen3MoeConfig(hidden_size=2048, intermediate_size=6144, moe_intermediate_size=768, num_experts=128,
                           num_experts_per_tok=8, num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=4,
                           head_dim=128, vocab_size=151936, max_position_embeddings=40960, decoder_sparse_step=1,
                           tie_word_embeddings=False)
    base = dict(E4B_PAGED_MODEL="x", E4B_PAGED_ARENA="a", E4B_PAGED_CALIB="c", E4B_PAGED_MAX_TOKENS_PER_SEQ="4096",
                E4B_SERVE_EXP_INT4="1", E4B_SERVE_ATTN_INT4="1", E4B_PAGED_GRAPHS="1", E4B_PAGED_DEVICE="cuda")
    ca = _cfg(monkeypatch, **base, E4B_PAGED_MAX_SEQS="auto")
    assert resolve_kv_pool(ca, mc, FREE_5090) == {} and ca.kv_pool_rows is None
    fixed = _cfg(monkeypatch, **base, E4B_PAGED_MAX_SEQS="64", E4B_PAGED_KV_POOL="shared")
    res = resolve_kv_pool(fixed, mc, FREE_5090)
    assert fixed.max_seqs == 64 and fixed.kv_pool_rows == res["pool_rows"] and fixed.kv_pool_rows is not None
    assert [c["max_seqs"] for c in res["candidates"]] == [64] and fixed.kv_pool_resolution is res
    auto = _cfg(monkeypatch, **base, E4B_PAGED_MAX_SEQS="auto", E4B_PAGED_KV_POOL="shared")
    res = resolve_kv_pool(auto, mc, FREE_5090)
    assert auto.max_seqs == 64 and auto.max_seqs_resolution is res and auto.kv_pool_rows is not None


# ---- the prefill graph's runtime headroom, held back by the shared pool (sc2j-5090-1's refusal)
# Recorded on that box (Qwen3-30B-A3B int4, 64 slots, 4,096 tokens a slot, buckets auto, RTX 5090; its /health and the
# refusal's own words): free before load 33,138,933,760 B; the shared pool 11,156 rows of 847,872 B (claim-all 16,448);
# the reserve 805,306,368 B; extra 311,772,672 B (the int8 head); the prefill graph refused with "the graph's private
# pool holds 426 MiB ... 724 MiB is free after capture ... the bulk KV flush up to 319 MiB more". The MiB figures are
# rounded to whole MiB, so the checks take their worst side (pool and flush + 0.5, free - 0.5).
BOX = {"free": 33138933760, "rows": 11156, "row_bytes": 847872, "claim_all_rows": 16448, "reserve": 805306368,
       "extra": 311772672, "pool_mib": 426, "free_after_mib": 724, "flush_mib": 319}
MIB = 2 ** 20


def _box_setup():
    from experts4bit_qlora.serve_recipe import ServeSetup
    return ServeSetup(max_tokens_per_seq=4096, exp_int4=True, attn_int4=True, buckets="auto")


def _rows_freed(headroom=None):
    """The rows the shared pool gives back on the box's inputs because of the hold-back: the pool sized with
    ``headroom`` (the shipped rule when None) against the pool sized with none. That is what the box's free memory
    after capture would gain."""
    import experts4bit_qlora.serve_recipe as sr
    topo, st = _qwen3_30b(), _box_setup()

    def sized(rule):
        with pytest.MonkeyPatch.context() as mp:
            if rule is not None:
                mp.setattr(sr, "prefill_graph_headroom_bytes", rule)
            return sr.size_shared_pool(topo, st, BOX["free"], widths=sr.MAX_SEQS_AUTO_WIDTHS, extra_bytes=BOX["extra"])
    on, off = sized(headroom), sized(lambda pool, flush: 0)
    assert on["max_seqs"] == off["max_seqs"] == 64
    return off["pool_rows"] - on["pool_rows"], on


def _gate_passes(rows_freed):
    """The runtime gate (the one rule, serve_recipe.prefill_graph_headroom_bytes) on the box's recorded numbers, with
    the free memory after capture raised by the rows the hold-back gives back."""
    from experts4bit_qlora.serve_recipe import prefill_graph_headroom_bytes
    free_after = (BOX["free_after_mib"] - 0.5) * MIB + rows_freed * BOX["row_bytes"]
    return free_after >= prefill_graph_headroom_bytes((BOX["pool_mib"] + 0.5) * MIB, (BOX["flush_mib"] + 0.5) * MIB)


def test_the_fixture_is_the_box():
    """The CPU fixture prices the box's quantities: the same row bytes, reserve, flush bound and claim-all rows."""
    import experts4bit_qlora.serve_recipe as sr
    topo, st = _qwen3_30b(), _box_setup()
    r = sr.size_shared_pool(topo, st, BOX["free"], widths=sr.MAX_SEQS_AUTO_WIDTHS, extra_bytes=BOX["extra"])
    c = r["candidates"][0]
    assert (c["row_bytes"], c["prefill_graph_reserve_bytes"], c["claim_all_rows"]) == (
        BOX["row_bytes"], BOX["reserve"], BOX["claim_all_rows"])
    assert abs(sr.bulk_flush_bytes(topo, replace_seqs(st, 64)) / MIB - BOX["flush_mib"]) <= 0.5


def replace_seqs(st, w):
    from dataclasses import replace
    return replace(st, max_seqs=w)


def test_the_box_as_recorded_fails_the_gate():
    """No hold-back (the box's sizing): 724 MiB free after capture, against 426 + 319 MiB: the refusal it recorded."""
    assert not _gate_passes(0)


def test_the_hold_back_lets_the_box_pass_the_gate():
    """With the hold-back, the pool gives back rows on the box's inputs, and the box's free memory after capture then
    clears the gate's 745 MiB."""
    freed, on = _rows_freed()
    assert freed > 0 and _gate_passes(freed)
    c = on["candidates"][0]
    from experts4bit_qlora.serve_recipe import bulk_flush_bytes, prefill_graph_headroom_bytes
    assert c["prefill_graph_headroom_bytes"] == prefill_graph_headroom_bytes(
        c["prefill_graph_reserve_bytes"], bulk_flush_bytes(_qwen3_30b(), replace_seqs(_box_setup(), 64)))


def test_a_zero_hold_back_reproduces_the_refusal():
    freed, _ = _rows_freed(lambda pool, flush: 0)
    assert freed == 0 and not _gate_passes(freed)


def test_the_runtime_gate_and_the_sizing_share_one_rule():
    import inspect

    from experts4bit_qlora import serve_recipe as sr
    from experts4bit_qlora.engines import paged_runner
    src = inspect.getsource(paged_runner)
    assert 'prefill_graph_headroom_bytes(pg["pool_bytes"], pg["bulk_flush_bytes"])' in src
    assert 'pg["free_after_bytes"] < pg["pool_bytes"] + pg["bulk_flush_bytes"]' not in src
    assert "prefill_graph_headroom_bytes(reserve, bulk_flush_bytes(topology, s))" in inspect.getsource(sr.size_shared_pool)
    assert sr.prefill_graph_headroom_bytes(3, 4) == 7


def test_claim_all_sizing_does_not_move():
    """choose_max_seqs (claim-all) prices no hold-back: each candidate's need is the estimate, the reserve, the extra and
    the margin, exactly as before; its claim-all rows at 32 and 64 slots are the box's."""
    from dataclasses import replace

    import experts4bit_qlora.serve_recipe as sr
    topo, st = _qwen3_30b(), _box_setup()
    out = sr.choose_max_seqs(topo, st, BOX["free"], widths=(64, 32, 16), extra_bytes=BOX["extra"])
    for c in out["candidates"]:
        s = replace(st, max_seqs=c["max_seqs"])
        fp = sr.estimate_serve_footprint(topo, s)
        want = fp.device_bytes + sr.prefill_graph_reserve_bytes(topo, s) + BOX["extra"] + sr.MAX_SEQS_AUTO_MARGIN_BYTES
        assert c.get("need_bytes", want) == want and "prefill_graph_headroom_bytes" not in c
    sh = sr.size_shared_pool(topo, st, BOX["free"], widths=(64, 32), extra_bytes=BOX["extra"])
    assert {c["max_seqs"]: c["claim_all_rows"] for c in sh["candidates"]} == {64: 16448, 32: 8224}
