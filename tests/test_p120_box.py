# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P120's speed box (``bench/p120/p120_box.py``) on CPU, before any GPU is rented.

The arms need CUDA graphs, the fp8 paged KV and grouped-nf4-gemm's kernels, so this stands in the RUNNER, the pool, the
profiler and the kernel module: ``_StandIn`` decodes with full-sequence forwards of a tiny Qwen3-MoE, pads each piece
to its bucket as ``PagedModelRunner`` does, counts replays (captured) or eager steps (``capture=False``), and routes
every step through ``int4_b32.build_group_tiles_fused`` so a wrong tile table moves its tokens. What it checks:
- each arm runs under its own ``E4B_INT4_WIDE_TILES``, set before the capture (served) or the profile (eager twin);
- the served arms replay one bucket-64 piece per step, ``warm`` untimed then ``steps`` timed, and record every token;
- the mutant shifts live tiles' expert ids only, keeps the builder's signature (so e4b's ``rank=`` check passes), is
  restored afterwards, and moves the tokens; the reducer reads the record (only the step count differs) without VOID.
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
LANES = [ROOT / "bench" / d for d in ("p120", "p119", "p117", "p108", "p97")]


def _load(name):
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(name, LANES[0] / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tables(ids, n_exp, block_m):
    """The chained builder's five tables, small and on CPU (row0, rows, grp, order, counts)."""
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
                                rank="pairwise"):
        mod.calls.append(rank)
        return _tables(ids, n_exp, block_m)
    mod.build_group_tiles_fused, mod.calls = build_group_tiles_fused, []
    return mod


IDS = torch.tensor([0, 0, 1, 3, 3, 3], dtype=torch.int32)


class _StandIn:
    """``PagedModelRunner``'s surface the arms drive."""
    made = []

    def __init__(self, model, kv, device="cpu", bulk_kv=False, **_):
        self.model, self.tokens, self.graph_stats, self.bulk_kv = model, {}, {}, bulk_kv
        _StandIn.made.append(self)

    def enable_decode_graphs(self, buckets, *, capture=True, warmup=2, verbose=True):
        self.buckets, self.capture = tuple(sorted(int(b) for b in buckets)), capture
        self.wide_at_capture = os.environ.get("E4B_INT4_WIDE_TILES")
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
        got = {}
        for piece in chunk_rows(rids, self.buckets[-1]):
            b = bucket_for(len(piece), self.buckets)
            ids = torch.tensor([self.tokens[r] for r in piece] + [self.tokens[piece[0]]] * (b - len(piece)))
            logits = self.model(input_ids=ids).logits
            grp = int4_b32.build_group_tiles_fused(IDS, 4, 2, rank="cumsum")[2]       # the step's tile table
            wrong = 0 if torch.equal(grp, _tables(IDS, 4, 2)[2]) else 1               # a wrong table moves the tokens
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
    """The profiler, stood in: runs the steps, and reports the kernels each setting launches per step (x n)."""
    for _ in range(n):
        fn()
    if os.environ.get("E4B_INT4_WIDE_TILES") == "1":
        ks = [("_tile_table_r1", 2, 40.0)]
    else:
        ks = [("void at::native::radixSortKVInPlace<-2, -1, 32, 32, long, long, unsigned int>", 2, 50.0),
              ("void at_cuda_detail::cub::DeviceScanKernel<x>", 4, 10.0)]
    ks.append(("_gemm_int4_b32_grouped_kernel", 4, 900.0))
    return [_Avg(k, c * n, us * n) for k, c, us in ks], 1.0


P, ROWS, STEPS = 5, 33, 2


def _tiny():
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=4, num_experts_per_tok=2,
                         max_position_embeddings=128, decoder_sparse_step=1, mlp_only_layers=[])
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def run():
    pytest.importorskip("transformers.models.qwen3_moe", reason="needs transformers with Qwen3-MoE")
    box = _load("p120_box")
    from experts4bit_qlora.engines import paged_runner
    fake = _fake_gnf4()
    saved = (paged_runner.PagedModelRunner, box.p119_box._pool, box.p119_box._profile, sys.modules.get("int4_b32"),
             os.environ.get("E4B_INT4_WIDE_TILES"))
    paged_runner.PagedModelRunner = _StandIn
    box.p119_box._pool = lambda model, rows, tokens, scratch, device: (object(), 2)
    box.p119_box._profile = _fake_profile
    sys.modules["int4_b32"] = fake
    _StandIn.made.clear()
    try:
        g = torch.Generator().manual_seed(1)
        ws = [torch.randint(0, 256, (P,), generator=g).tolist() for _ in range(ROWS)]
        rec = box.measure(_tiny(), ws, prompt=P, device="cpu", rows=ROWS, steps=STEPS, bulk_kv=True)
        restored = fake.build_group_tiles_fused
        env_after = os.environ.get("E4B_INT4_WIDE_TILES")
    finally:
        paged_runner.PagedModelRunner, box.p119_box._pool, box.p119_box._profile = saved[:3]
        if saved[3] is None:
            sys.modules.pop("int4_b32", None)
        else:
            sys.modules["int4_b32"] = saved[3]
        if saved[4] is not None:
            os.environ["E4B_INT4_WIDE_TILES"] = saved[4]
    return box, rec, fake, restored, env_after


def test_each_arm_runs_under_its_own_setting(run):
    box, rec, _fake, _r, env_after = run
    assert box.SERVED == ("OFF_a", "ON_a", "ON_b", "OFF_b")                         # ABBA
    served = [m for m in _StandIn.made if getattr(m, "capture", False)]
    assert [m.wide_at_capture for m in served] == ["0", "1", "1", "0", "1"]          # the four arms, then the mutant
    assert all(getattr(m, "disabled", False) for m in served)                       # graphs released between arms
    profiled = [m for m in _StandIn.made if getattr(m, "capture", True) is False]
    assert [m.wide_at_capture for m in profiled] == ["0", "1"]
    assert env_after is None                                                       # the box leaves the knob unset
    assert [rec["served"][k]["wide"] for k in box.SERVED] == [False, True, True, False]
    assert rec["served"]["MUTANT"]["wide"] is True and rec["served"]["MUTANT"]["mutant"] is True


def test_the_profile_arms_read_the_builder_and_the_table(run):
    _box, rec, *_ = run
    off, on = rec["profile"]["off"], rec["profile"]["on"]
    assert (off["radix_sort_calls"], off["table_calls"], off["builder_ms"]) == (2.0, 0.0, 0.06)
    assert (on["radix_sort_calls"], on["table_calls"], on["table_ms"], on["builder_ms"]) == (0.0, 2.0, 0.04, 0.0)
    assert off["profiled"]["64"] == {"replays": 0, "eager_steps": 8, "rows": ROWS * 8, "pad_rows": (64 - ROWS) * 8}


def test_the_served_arms_replay_one_bucket_64_piece_per_step_and_record_every_token(run):
    box, rec, *_ = run
    n = box.WARM + STEPS
    for label in box.SERVED + ("MUTANT",):
        s = rec["served"][label]
        assert s["graph_stats"]["64"] == {"replays": n, "eager_steps": 0, "rows": ROWS * n, "pad_rows": (64 - ROWS) * n}
        assert all(v == {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0}
                   for b, v in s["graph_stats"].items() if b != "64")
        assert len(s["step_ms"]) == STEPS and len(s["tokens"]) == n and all(len(t) == ROWS for t in s["tokens"])
        assert s["bulk_kv"] is True and set(s["graph_status"].values()) == {"graph"}
    assert rec["served"]["ON_a"]["tokens"] == rec["served"]["OFF_a"]["tokens"] == rec["served"]["OFF_b"]["tokens"]


def test_the_mutant_shifts_live_tiles_keeps_the_signature_and_is_restored(run, monkeypatch):
    box, rec, fake, restored, _e = run
    assert restored is fake.build_group_tiles_fused and "_Mutant" not in repr(restored)
    assert rec["served"]["MUTANT"]["tokens"] != rec["served"]["OFF_a"]["tokens"]     # the token gate can fail
    monkeypatch.setitem(sys.modules, "int4_b32", fake)
    with box._Mutant():
        f = fake.build_group_tiles_fused
        assert "rank" in inspect.signature(f).parameters
        row0, rows, grp, order, counts = f(IDS, 4, 2, rank="cumsum")
    ref = _tables(IDS, 4, 2)
    live = ref[1] > 0
    assert torch.equal(grp[live], (ref[2][live] + 1) % 4) and torch.equal(grp[~live], ref[2][~live])
    assert all(torch.equal(x, y) for x, y in zip((row0, rows, order, counts), (ref[0], ref[1], ref[3], ref[4])))
    assert fake.build_group_tiles_fused is restored


def test_the_reducer_reads_the_record_with_only_the_step_count_changed(run, monkeypatch):
    box, rec, *_ = run
    red = _load("p120_reduce")
    monkeypatch.setattr(red, "STEPS", STEPS)
    model = "ibm-granite/granite-3.1-3b-a800m-instruct"
    full = {"model": model, "revision": red.REVS[model], "e4b_sha": "a" * 40, "gnf4_sha": red.GNF4_SHA,
            "census_build": {"moe_layers": 2}, **rec}
    assert red.faults(full, "a" * 40) == []
    v = red.reduce_obj(full, "a" * 40)
    assert v["verdict"] not in ("VOID", "NO_READING", "TOKENS_DIFFER"), v
    assert v["tables"]["tokens"]["on_vs_off_differing"] == 0 and v["tables"]["tokens"]["mutant_vs_off_differing"] > 0
    assert box.BUILDER == red.BUILDER and box.TABLE == red.TABLE and box.WARM == red.WARM
    assert box.B64 == red.B64 and box.SERVED == red.SERVED
