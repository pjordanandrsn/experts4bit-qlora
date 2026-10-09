# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P126's box (``bench/p126/p126_box.py``) on CPU, before any GPU is rented.

The arms need CUDA graphs, the fp8 paged KV and grouped-nf4-gemm's kernels, so this stands in the RUNNER, the pool, the
profiler and the kernel module: ``_StandIn`` decodes with full-sequence forwards of a tiny Qwen3-MoE, pads each piece
to its bucket as ``PagedModelRunner`` does, counts replays or eager steps, and routes every step through
``int4_b32.build_group_tiles_fused`` with the ``programs`` its capture saw, so a wrong tile table moves its tokens.
What it checks:
- every profile and every runner runs under its own ``E4B_INT4_TILE_PROGRAMS``, set before its capture, and the box
  leaves the knob unset;
- each block holds a P = 1 runner and a candidate runner, both alive, decoded in strict alternation in the block's order;
  one bucket-64 piece a step, every token recorded, the traced steps and the memory;
- the mutant shifts live tiles at P = 8, keeps the builder's signature (so ``programs=`` still reads as supported), and
  moves the tokens; the reducer reads the record (only the step counts lowered) without VOID.
Speeds are read on the rental.
"""
import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("p126", "p124", "p120", "p119", "p117", "p108", "p97")]
P, ROWS, STEPS, BUSY = 5, 33, 2, 3
IDS = torch.tensor([0, 0, 1, 3, 3, 3], dtype=torch.int32)


def _load(name):
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(name, LANES[0] / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tables(ids, n_exp, block_m):
    ids = ids.long()
    counts = torch.bincount(ids, minlength=n_exp)
    order = torch.argsort(ids, stable=True)
    budget = -(-ids.numel() // block_m) + n_exp
    row0, rows, grp = (torch.zeros(budget, dtype=torch.int32) for _ in range(3))
    t, start = 0, 0
    for e in range(n_exp):
        c = int(counts[e])
        for i in range(-(-c // block_m)):
            row0[t], rows[t], grp[t] = start + i * block_m, min(block_m, c - i * block_m), e
            t += 1
        start += c
    return row0, rows, grp, order, counts


def _fake_gnf4():
    mod = types.ModuleType("int4_b32")

    def build_group_tiles_fused(ids, n_exp, block_m, tiles_budget=None, *, lean=False, sorted_ids=False, warps=4,
                                rank="pairwise", rchunk=None, programs=None):
        mod.calls.append(programs)
        return _tables(ids, n_exp, block_m)
    mod.build_group_tiles_fused, mod.calls = build_group_tiles_fused, []
    return mod


class _StandIn:
    """``PagedModelRunner``'s surface the arms drive; a captured step uses the ``programs`` its capture saw."""
    made, log = [], []

    def __init__(self, model, kv, device="cpu", bulk_kv=False, **_):
        self.model, self.tokens, self.graph_stats, self.bulk_kv, self.tracer = model, {}, {}, bulk_kv, None
        _StandIn.made.append(self)

    def enable_decode_graphs(self, buckets, *, capture=True, warmup=2, verbose=True):
        self.buckets, self.capture = tuple(sorted(int(b) for b in buckets)), capture
        self.programs_at_capture = os.environ.get("E4B_INT4_TILE_PROGRAMS")
        self.graph_stats = {b: {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in self.buckets}
        return {b: ("graph" if capture else "eager: capture=False") for b in self.buckets}

    def disable_decode_graphs(self):
        self.disabled = True

    def bind(self, rid, slot, prompt):
        self.tokens[rid] = list(prompt)

    def run_prefill(self, chunks):
        for rid, s, n in chunks:
            o = self.model(input_ids=torch.tensor([self.tokens[rid][:s + n]]))
            self.tokens[rid].append(int(o.logits[0, -1].argmax()))

    def run_decode(self, rids):
        import int4_b32
        from experts4bit_qlora.engines.paged_runner import bucket_for, chunk_rows
        _StandIn.log.append(self)
        progs = self.programs_at_capture if self.capture else os.environ.get("E4B_INT4_TILE_PROGRAMS")
        got = {}
        for piece in chunk_rows(rids, self.buckets[-1]):
            b = bucket_for(len(piece), self.buckets)
            if self.tracer is not None:
                self.tracer.mark("dec_prep", event=True)
            ids = torch.tensor([self.tokens[r] for r in piece] + [self.tokens[piece[0]]] * (b - len(piece)))
            logits = self.model(input_ids=ids).logits
            kw = {} if progs in (None, "1") else {"programs": int(progs)}
            grp = int4_b32.build_group_tiles_fused(IDS, 4, 2, rank="cumsum", **kw)[2]
            wrong = 0 if torch.equal(grp, _tables(IDS, 4, 2)[2]) else 1
            if self.tracer is not None:
                self.tracer.mark("dec_issue", event=True)
            st = self.graph_stats[b]
            st["replays" if self.capture else "eager_steps"] += 1
            st["rows"] += len(piece)
            st["pad_rows"] += b - len(piece)
            for j, r in enumerate(piece):
                got[r] = (int(logits[j, -1].argmax()) + wrong) % 256
                self.tokens[r].append(got[r])
        return got


class _Avg:
    def __init__(self, key, count, us):
        self.key, self.count, self.self_device_time_total = key, count, us


def _fake_profile(fn, n, device):
    for _ in range(n):
        fn()
    p = os.environ.get("E4B_INT4_TILE_PROGRAMS")
    ks = [("_tile_table_r1", 2, 50.0)] if p in (None, "1") else [("_tile_table_cumsum_mp", 2, 50.0 / int(p) * 2)]
    ks.append(("_gemm_int4_b32_grouped_kernel", 4, 900.0))
    return [_Avg(k, c * n, us * n) for k, c, us in ks], 1.0


def _tiny():
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=4, num_experts_per_tok=2,
                         max_position_embeddings=128, decoder_sparse_step=1, mlp_only_layers=[])
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    pytest.importorskip("transformers.models.qwen3_moe", reason="needs transformers with Qwen3-MoE")
    box = _load("p126_box")
    from experts4bit_qlora.engines import paged_runner
    fake = _fake_gnf4()
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(paged_runner, "PagedModelRunner", _StandIn)
        mp.setattr(box.p119_box, "_pool", lambda model, rows, tokens, scratch, device: (object(), 2))
        mp.setattr(box.p119_box, "_profile", _fake_profile)
        mp.setitem(sys.modules, "int4_b32", fake)
        mp.setattr(box, "BUSY", BUSY)
        mp.setattr(box.p120_box, "WARM", box.WARM)
        mp.setattr(box.p124_box._Clock, "PERIOD", 0.01)
        mp.setattr(box.p124_box.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="2800, 13801, 500.0, 65, P1\n"))
        traced = {}

        def busy(path):
            traced[path] = sum(1 for _ in open(path, encoding="utf-8"))
            return [[0.9, 1.0]] * traced[path]
        mp.setattr(box.p124_box, "_busy", busy)
        mp.setattr(box.p124_box, "_mem", lambda device: {"max_allocated_mib": 100.0, "max_reserved_mib": 200.0,
                                                          "wide_workspace_mib": 0.0})
        mp.delenv("E4B_INT4_TILE_PROGRAMS", raising=False)
        _StandIn.made.clear()
        _StandIn.log.clear()
        g = torch.Generator().manual_seed(1)
        ws = [torch.randint(0, 256, (P,), generator=g).tolist() for _ in range(ROWS)]
        rec = box.measure(_tiny(), ws, prompt=P, device="cpu", rows=ROWS, steps=STEPS, bulk_kv=True,
                          trace_dir=str(tmp_path_factory.mktemp("trace")))
        env_after = (os.environ.get("E4B_INT4_TILE_PROGRAMS"), os.environ.get("E4B_INT4_WIDE_TILES"))
        restored = sys.modules["int4_b32"].build_group_tiles_fused is fake.build_group_tiles_fused
        made, log = list(_StandIn.made), list(_StandIn.log)
    finally:
        mp.undo()
    return box, rec, fake, restored, env_after, made, log, traced


def test_every_arm_runs_under_its_own_setting_and_the_knob_is_left_unset(run):
    box, rec, _fake, _r, env_after, made, *_ = run
    assert env_after == (None, None)
    assert box.SETTINGS == ("1", "4", "8") and box.CANDIDATES == ("4", "8") and box.BLOCKS == ("a", "b")
    profiled = [m for m in made if m.capture is False]
    assert [m.programs_at_capture for m in profiled] == ["1", "4", "8"]
    served = [m for m in made if m.capture and m.bulk_kv and not hasattr(m, "mutant_marker")]
    assert [m.programs_at_capture for m in served] == ["1", "4", "4", "1", "1", "8", "8", "1", "8"]   # 4 blocks, mutant
    assert all(getattr(m, "disabled", False) for m in served)
    assert [rec["profile"][p]["programs"] for p in box.SETTINGS] == ["1", "4", "8"]


def test_the_profiles_read_the_table_they_ran(run):
    _box, rec, *_ = run
    assert rec["profile"]["1"]["table_one_calls"] == 2.0 and rec["profile"]["1"]["table_mp_calls"] == 0.0
    assert rec["profile"]["8"]["table_mp_calls"] == 2.0 and rec["profile"]["8"]["table_one_calls"] == 0.0
    assert rec["profile"]["1"]["profiled"]["64"] == {"replays": 0, "eager_steps": 8, "rows": ROWS * 8,
                                                     "pad_rows": (64 - ROWS) * 8}


def test_each_block_alternates_its_two_runners_in_lockstep(run):
    box, rec, _f, _r, _e, made, log, traced = run
    n = box.WARM + STEPS + BUSY
    for c in box.CANDIDATES:
        for b in box.BLOCKS:
            blk = rec["served"][c][b]
            assert blk["order"] == (["1", c] if b == "a" else [c, "1"])
            assert blk["clock"] and blk["memory"]["max_allocated_mib"] > 0
            for st in blk["order"]:
                s = blk["arms"][st]
                assert s["programs"] == st
                assert s["graph_stats"]["64"] == {"replays": n, "eager_steps": 0, "rows": ROWS * n,
                                                  "pad_rows": (64 - ROWS) * n}
                assert len(s["step_ms"]) == STEPS and len(s["tokens"]) == n and len(s["busy"]) == BUSY
    served = [m for m in made if m.capture and m.bulk_kv][:8]
    for first, second in zip(served[0::2], served[1::2]):
        assert [m for m in log if m is first or m is second] == [first, second] * n
    assert sorted(traced.values()) == [BUSY] * 8
    assert rec["served"]["8"]["a"]["arms"]["8"]["tokens"] == rec["served"]["8"]["a"]["arms"]["1"]["tokens"]


def test_the_mutant_moves_the_tokens_and_is_restored(run, monkeypatch):
    box, rec, fake, restored, *_ = run
    assert restored
    assert rec["mutant"]["programs"] == "8" and rec["mutant"]["mutant"] is True
    assert rec["mutant"]["tokens"] != rec["served"]["4"]["a"]["arms"]["1"]["tokens"][:len(rec["mutant"]["tokens"])]
    monkeypatch.setitem(sys.modules, "int4_b32", fake)
    with box.p120_box._Mutant():
        assert "programs" in inspect.signature(fake.build_group_tiles_fused).parameters   # capability still reads true


def test_the_reducer_reads_the_record_with_only_the_step_counts_lowered(run, monkeypatch):
    box, rec, *_ = run
    red = _load("p126_reduce")
    monkeypatch.setattr(red, "STEPS", STEPS)
    monkeypatch.setattr(red, "BUSY", BUSY)
    model = "ibm-granite/granite-3.1-3b-a800m-instruct"
    full = {"model": model, "revision": red.REVS[model], "e4b_sha": "a" * 40, "gnf4_sha": red.GNF4_SHA,
            "tile_programs_env": None, "wide_tiles_env": None, "census_build": {"moe_layers": 2}, **rec}
    assert red.faults(full, "a" * 40) == []
    assert red.engagement(full) == []
    v = red.reduce_obj(full, "a" * 40)
    assert v["verdict"] not in ("VOID", "NO_READING", "TOKENS_DIFFER"), v.get("reasons")
    assert v["tables"]["mutant_differ_share"] > 0
    assert box.SETTINGS == red.SETTINGS and box.CANDIDATES == red.CANDIDATES and box.BLOCKS == red.BLOCKS
    assert box.WARM == red.WARM and box.MUTANT_STEPS == red.MUTANT_STEPS and tuple(box.B64) == red.B64
    assert box.TABLE_ONE == red.TABLE_ONE and box.TABLE_MP == red.TABLE_MP
