# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P124's box (``bench/p124/p124_box.py``) on CPU, before any GPU is rented.

The arms need CUDA graphs, the fp8 paged KV and grouped-nf4-gemm's kernels, so this stands in the RUNNER, the pool, the
profiler, P117's teacher-forced pass and the kernel package (``tests/test_int4_attn_wide.py``'s stubs, whose ``block_m=``
GEMM rounds a little differently above 16 rows, as the real route does). The model is a tiny one whose attention
projections are real ``Int4Linear`` modules, so :class:`Route` runs through the shipped forward. ``_Runner`` pads each
piece to its bucket and calls every projection on exactly the bucket's rows, eagerly or at capture and replay. Checks:
- each arm sets the route before its captures, profile or pass, and the modules are restored afterwards;
- the route counts are the ones the reducer registers (decode calls by bucket; prefill rows above 64);
- ``mutant_wide`` moves only the wide calls; the served arms replay one piece per step and record every token, the
  traced steps and the memory; the reducer reads the record (only the step counts lowered) without VOID.
Speeds and quality are read on the rental.
"""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("p124", "p119", "p117", "p108", "p97")]
sys.path.insert(0, os.path.dirname(__file__))
wide_tests = pytest.importorskip("test_int4_attn_wide")      # its kernel stubs (skips where int4_pack_ref is absent)

P, C, ROWS, STEPS, BUSY = 80, 4, 33, 2, 3                    # prefill rows above 64; one bucket-64 piece of 33 rows
V = 256
PHASE = 1.0      # how far a step's score turns the stand-in log-probs: the wide mutant moves them, rounding does not


def _load(name):
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(name, LANES[0] / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Attention(nn.Module):                                   # structural: the name ends in Attention
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 96, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(96, 64, bias=False, dtype=torch.bfloat16)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(num_hidden_layers=2)
        torch.manual_seed(0)
        self.layers = nn.ModuleList(_Attention() for _ in range(2))


def _x(tok, K):
    return [(((int(tok) + 1) * (k + 3)) % 23) / 23.0 - 0.5 for k in range(K)]


class _Runner:
    """``PagedModelRunner``'s surface the arms drive: a padded step of bucket b runs every projection on b rows."""
    made = []

    def __init__(self, model, kv=None, device="cpu", bulk_kv=False, **_):
        from experts4bit_qlora.engines.int4_attn import Int4Linear
        self.lins = [m for m in model.modules() if isinstance(m, Int4Linear)]
        self.tokens, self.graph_stats, self.bulk_kv, self.tracer = {}, {}, bulk_kv, None
        _Runner.made.append(self)

    def _step(self, toks):
        s = torch.zeros(len(toks))
        for m in self.lins:
            y = m(torch.tensor([_x(t, m.K) for t in toks], dtype=torch.bfloat16))
            s += y.float().sum(-1)
        return s                                                 # per row: a scalar the tokens and log-probs derive from

    def enable_decode_graphs(self, buckets, *, capture=True, warmup=2, verbose=True):
        self.buckets, self.capture = tuple(sorted(int(b) for b in buckets)), capture
        self.wide_at_capture = [m._wide for m in self.lins]
        self.graph_stats = {b: {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in self.buckets}
        if capture:
            for b in self.buckets:
                for _ in range(warmup + 1):                      # the warm-ups and the capture run the forward
                    self._step([0] * b)
        return {b: ("graph" if capture else "eager: capture=False") for b in self.buckets}

    def disable_decode_graphs(self):
        self.disabled = True

    def bind(self, rid, slot, prompt):
        self.tokens[rid] = list(prompt)

    def run_prefill(self, chunks):
        for rid, s, n in chunks:
            sc = self._step(self.tokens[rid][s:s + n])
            self.last = sc[-1]
            if s + n >= len(self.tokens[rid]):
                self.tokens[rid].append(int(abs(float(sc[-1])) * 997) % V)

    def run_decode(self, rids):
        from experts4bit_qlora.engines.paged_runner import bucket_for, chunk_rows
        got, self.scores = {}, {}
        for piece in chunk_rows(rids, self.buckets[-1]):
            b = bucket_for(len(piece), self.buckets)
            if self.tracer is not None:
                self.tracer.mark("dec_prep", event=True)
            sc = self._step([self.tokens[r][-1] for r in piece] + [0] * (b - len(piece)))
            if self.tracer is not None:
                self.tracer.mark("dec_issue", event=True)
            st = self.graph_stats[b]
            st["replays" if self.capture else "eager_steps"] += 1
            st["rows"] += len(piece)
            st["pad_rows"] += b - len(piece)
            for j, r in enumerate(piece):
                got[r] = int(abs(float(sc[j])) * 997) % V
                self.scores[r] = sc[j]
                self.tokens[r].append(got[r])
        return got


def _lp(score, mutant=None):
    v = torch.arange(V, dtype=torch.float32)
    logits = torch.zeros(V) if mutant == "scale" else 3 * torch.cos(v * 0.37 + PHASE * float(score))
    return logits.log_softmax(-1)


def _fake_paged_pass(model, ws, P_, C_, chunk, device, *, buckets, reverse=False, mutant=None, capture=False,
                     stand_in=False):
    """P117's pass, stood in: the same split, prefill and teacher forcing, with log-probs derived from the step."""
    from experts4bit_qlora.engines.paged_runner import chunk_rows
    runner = _Runner(model)
    gs = runner.enable_decode_graphs(buckets, capture=capture)
    n = len(ws)
    lps, tk = [[] for _ in ws], [[] for _ in ws]
    for rid in range(n):
        runner.bind(rid, rid, ws[rid][:P_])
        for s in range(0, P_, chunk):
            runner.run_prefill([(rid, s, min(chunk, P_ - s))])
        lps[rid].append(_lp(runner.last, mutant))
        runner.tokens[rid][-1] = ws[rid][P_]
    order = list(range(n))
    for t in range(C_ - 1):
        got = runner.run_decode(order)
        for rid in order:
            tk[rid].append(int(got[rid]))
            lps[rid].append(_lp(runner.scores[rid], mutant))
            runner.tokens[rid][-1] = ws[rid][P_ + t + 1]
    pieces = len(list(chunk_rows(order, max(buckets))))
    eng = {"buckets": list(buckets), "reverse": reverse, "capture": capture, "windows": n,
           "graph_status": {str(k): v for k, v in gs.items()},
           "decode_calls": 0 if capture else (C_ - 1) * 2 * pieces, "step_ms": 0.0,
           "grouping_flags_in_pass": {"device_grouping": True, "force_singleton_groups": False},
           "graph_stats": {str(k): dict(v) for k, v in runner.graph_stats.items()}}
    return (None if capture else [torch.stack(x) for x in lps]), tk, eng


class _Avg:
    def __init__(self, key, count, us):
        self.key, self.count, self.self_device_time_total = key, count, us


def _fake_profile_for(model):
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    lins = [m for m in model.modules() if isinstance(m, Int4Linear)]

    def fake(fn, n, device):
        for _ in range(n):
            fn()
        ks = [("nvjet_tst_192x192_64x3_1x2_h_bz_TNT (head)", 1, 120.0)]
        if lins[0]._wide:
            ks.append(("_gemm_int4_b32_smallm", len(lins), 20.0))
        else:
            ks.append(("nvjet_tst_64x64_64x8_1x1_v_bz_TNT", len(lins), 45.0))
        ks.append(("_gemm_int4_b32_grouped_smallm_kernel", 4, 900.0))
        return [_Avg(k, c * n, us * n) for k, c, us in ks], 1.0
    return fake


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    box = _load("p124_box")
    from experts4bit_qlora.engines import paged_attention, paged_runner
    mp = pytest.MonkeyPatch()
    calls = []
    try:
        ia = wide_tests._stubs(mp, calls)
        model = _Model()
        assert ia.enable_serve_attn_int4(model, smallm=True, wide=True) == 4
        mp.setattr(paged_runner, "PagedModelRunner", _Runner)
        mp.setattr(box.p119_box, "_pool", lambda model, rows, tokens, scratch, device: (object(), 2))
        mp.setattr(box.p119_box, "_profile", _fake_profile_for(model))
        mp.setattr(box.p117_box, "paged_pass", _fake_paged_pass)
        mp.setattr(paged_attention, "register", lambda m: None)
        mp.setattr(box, "BUSY", BUSY)
        real_busy, real_mem = box._busy, box._mem
        traced = {}

        def busy(path):
            traced[path] = sum(1 for _ in open(path, encoding="utf-8"))
            assert real_busy(path) == []                                  # on CPU the trace has no device events
            return [[0.9, 1.0]] * traced[path]

        def mem(device):
            m = real_mem(device)
            return {**m, "max_allocated_mib": 100.0 + m["wide_workspace_mib"], "max_reserved_mib": 200.0}
        mp.setattr(box, "_busy", busy)
        mp.setattr(box, "_mem", mem)
        _Runner.made.clear()
        g = torch.Generator().manual_seed(1)
        ws = [torch.randint(0, V, (P + C,), generator=g).tolist() for _ in range(ROWS)]
        rec = box.measure(model, ws, prompt=P, cont=C, device="cpu", rows=ROWS, steps=STEPS, bulk_kv=True,
                          trace_dir=str(tmp_path_factory.mktemp("trace")))
        lins = box.int4_linears(model)
        after = [(m._wide, "forward" in m.__dict__, "_bf16_weight" in m.__dict__) for m in lins]
        made = list(_Runner.made)
    finally:
        mp.undo()
    return box, rec, after, traced, made


def test_each_arm_sets_the_route_before_it_runs_and_the_modules_are_restored(run):
    box, rec, after, _t, made = run
    assert after == [(True, False, False)] * 4                       # built with wide=True; no wrapper left behind
    served = [m for m in made if getattr(m, "capture", False) and hasattr(m, "bulk_kv") and m.bulk_kv]
    assert [all(m.wide_at_capture) for m in served] == [False, True, True, False] * 2    # ABBA at 64, then at 32
    assert all(getattr(m, "disabled", False) for m in served)
    assert [rec["profile"][k]["wide"] for k in ("off64", "on64", "off32", "on32")] == [False, True, False, True]
    assert [rec["quality"]["engagement"][a]["wide"] for a in box.QUALITY] == \
        [False, False, False, False, True, True, True, True, True]


def test_the_route_counts_are_the_registered_ones(run):
    box, rec, *_ = run
    red = _load("p124_reduce")
    q = rec["quality"]
    for arm in box.QUALITY:
        if arm == "G64on":
            continue
        r = q["engagement"][arm]["route"]
        want = red.expected_decode_route(ROWS, red.Q_BUCKETS[arm], red.Q_WIDE[arm], C - 1, 4)
        assert {k: v for k, v in r.items() if int(k.split(":")[1]) <= 64} == want, arm
        assert r[f"bf16:{P}"] == ROWS * 4, arm                          # the prefill: one 80-row forward per window
    on, off = rec["profile"]["on64"]["route"], rec["profile"]["off64"]["route"]
    assert on["wide:64"] == off["bf16:64"] == (3 + 8) * 4 and "bf16:64" not in on and "wide:64" not in off
    cap = rec["served"]["64"]["ON_a"]["capture_route"]
    assert cap["wide:64"] == cap["wide:32"] == 3 * 4 and cap["k16:16"] == 3 * 4 and cap["gemv:1"] == 3 * 4
    assert "bf16:64" not in cap and "wide:64" not in rec["served"]["64"]["OFF_a"]["capture_route"]


def test_mutant_wide_moves_only_the_wide_calls(monkeypatch):
    box = _load("p124_box")
    calls = []
    ia = wide_tests._stubs(monkeypatch, calls)
    m = ia.Int4Linear(nn.Linear(64, 96, bias=False, dtype=torch.bfloat16), smallm=True, wide=True)
    model = nn.Sequential(m)
    x16, x40 = torch.randn(16, 64, dtype=torch.bfloat16), torch.randn(40, 64, dtype=torch.bfloat16)
    y16, y40 = m(x16), m(x40)
    with box.Route(model, True, mutant_wide=True) as r:
        z16, z40 = m(x16), m(x40)
    assert torch.equal(z16, y16) and not torch.equal(z40, y40)
    assert r.counts == {"k16:16": 1, "wide:40": 1}
    with box.Route(model, False) as r:
        m(x40)
    assert r.counts == {"bf16:40": 1} and m._wide is True and "forward" not in m.__dict__


def test_the_served_arms_replay_one_piece_per_step_and_record_tokens_trace_and_memory(run):
    box, rec, _a, traced, _m = run
    n = box.WARM + STEPS + BUSY
    for d, rows in (("64", ROWS), ("32", 32)):
        for label in box.SERVED:
            s = rec["served"][d][label]
            assert s["graph_stats"][d] == {"replays": n, "eager_steps": 0, "rows": rows * n, "pad_rows": (int(d) - rows) * n}
            assert all(v["replays"] == 0 for b, v in s["graph_stats"].items() if b != d)
            assert len(s["step_ms"]) == STEPS and len(s["tokens"]) == n and all(len(t) == rows for t in s["tokens"])
            assert len(s["busy"]) == BUSY and s["memory"]["max_allocated_mib"] > 0
    assert sorted(traced.values()) == [BUSY] * 8
    on, off = rec["served"]["64"]["ON_a"]["memory"], rec["served"]["64"]["OFF_a"]["memory"]
    assert on["wide_workspace_mib"] > 0 and on["max_allocated_mib"] > off["max_allocated_mib"] - 1


def test_the_busy_fraction_reads_the_trace_events(tmp_path):
    box = _load("p124_box")
    path = tmp_path / "t.jsonl"
    rows = [{"step": 0, "gpu": {"dec_prep": 0.1, "dec_issue": 15.1, "dispatch": 15.3}, "step_ms": 16.0},
            {"step": 1, "gpu": {"pf_prep": 0.1}, "step_ms": 3.0}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert box._busy(str(path)) == [[15.0, 16.0]]


def test_the_reducer_reads_the_record_with_only_the_step_counts_lowered(run, monkeypatch):
    box, rec, *_ = run
    red = _load("p124_reduce")
    monkeypatch.setattr(red, "STEPS", STEPS)
    monkeypatch.setattr(red, "BUSY", BUSY)
    model = "ibm-granite/granite-3.1-3b-a800m-instruct"
    full = {"model": model, "revision": red.REVS[model], "e4b_sha": "a" * 40, "gnf4_sha": red.GNF4_SHA,
            "attn_int4_wide_env": "1", "census_build": {"moe_layers": 2}, **rec}
    assert red.faults(full, "a" * 40) == []
    assert red.engagement(full) == []
    v = red.reduce_obj(full, "a" * 40)
    assert v["verdict"] not in ("VOID", "NO_READING"), v.get("reasons")
    q = v["tables"]["quality"]
    assert q["passes"]["ON64"] and q["passes"]["ON32"] and not q["passes"]["mutant_wide"] and not q["passes"]["mutant_scale"]
    assert q["rep_identical"] is True and q["function"]["differ"] == 0
    assert box.SERVED == red.SERVED and box.WARM == red.WARM and box.QUALITY == red.QUALITY
    assert tuple(box.B64) == red.B64 and tuple(box.B32) == red.B32 and box.DEPTHS == red.DEPTHS
    assert {a: tuple(k["buckets"]) for a, k in box.Q_KW.items()} == {a: tuple(b) for a, b in red.Q_BUCKETS.items()}
    assert {a: k["wide"] for a, k in box.Q_KW.items()} == red.Q_WIDE
