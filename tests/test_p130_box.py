# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P130's measurement (``bench/p130/p130_box.py``) on CPU, before any GPU is rented.

Phase A captures CUDA graphs of the served prefill forward, and Phase B decodes through the bucketed path with the fused
KV append, so this test stands the RUNNERS in:
- ``_PrefillStandIn`` has the prefill-graph surface the box drives (``enable_prefill_graph``, ``_prefill_forward``,
  ``_prefill_scope``). Its "capture" runs the five eager forwards the server's capture runs (two warm-ups, the capture,
  two startup checks), and its "replay" runs the model with no Python counters, as a replay does. Each eager forward
  bumps the engine's own counters as the int4 K19 route would: ``ROUTE_SEEN`` and ``K19_DISPATCH_SEEN`` (``lean`` under
  ``E4B_PREFILL_LEAN_DISPATCH=1``) once a layer, and ``PREFILL_FOLD_SEEN`` by the fold table under
  ``E4B_FUSE_PREFILL_GLUE=1``;
- ``_StandIn`` is P117's: it decodes each piece with a full-sequence forward of a tiny Qwen3-MoE, split and padded as
  ``PagedModelRunner`` does.
What it checks is the box's own bookkeeping: the deltas per capture, FUNCTION and its digests, every timed replay, the
eager forwards, the knob and grouping restored afterwards, a refused capture; and Phase B's arms, R's saved log-probs
and their digests, the subject scored against them. The kernels, the graphs and the timing run on the proving rental.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("p130", "p117", "p108", "p97")]
P1_ENV, P2_ENV = "E4B_FUSE_PREFILL_GLUE", "E4B_PREFILL_LEAN_DISPATCH"


def _load():
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("p130_box", LANES[0] / "p130_box.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny():
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=4, num_experts_per_tok=2,
                         max_position_embeddings=256, decoder_sparse_step=1, mlp_only_layers=[])
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).eval()


class _Graph:
    def __init__(self, runner, pg, lean):
        self.runner, self.pg, self.lean = runner, pg, lean

    def replay(self):                               # no Python counters: a replay runs none
        with torch.no_grad():
            o = type(self.runner.model).forward(self.runner.model, input_ids=self.pg["ids"])
        self.pg["logits"].copy_(o.logits)
        for lay, (k, v) in self.pg["staged"].items():
            k.copy_(o.logits[0, :, :4] * (lay + 1))
            v.copy_(o.logits[0, :, 4:8] + (1e-3 if (self.lean and self.runner.perturb_p2 and lay == 1) else 0.0))


class _PrefillStandIn:
    """``PagedModelRunner``'s prefill-graph surface, on CPU."""
    _PG_KEY = -1

    def __init__(self, model, perturb_p2=False, refuse=()):
        self.model, self.perturb_p2, self.refuse = model, perturb_p2, refuse
        self.ctx = types.SimpleNamespace(drop=lambda slot: None)
        self._prefill_graph = None
        self.layers = int(model.config.num_hidden_layers)

    def _prefill_forward(self, ids, positions):
        from experts4bit_qlora.engines import glue_fuse
        from experts4bit_qlora.engines import hot_residency as hr
        lean = os.environ.get(P2_ENV) == "1"
        for _ in range(self.layers):
            hr._seen_route("int4_k19", 4096)
            hr._seen_k19_dispatch("chained", "lean" if lean else "gather", 4096)
            if os.environ.get(P1_ENV) == "1":
                glue_fuse._seen_prefill("layer")
                glue_fuse._seen_prefill("attention")
                glue_fuse._seen_prefill("norm")
        if os.environ.get(P1_ENV) == "1":
            glue_fuse._seen_prefill("norm")              # the final norm
        with torch.no_grad():
            return self.model(input_ids=ids)

    def _prefill_scope(self):
        return lambda: None

    def enable_prefill_graph(self, T):
        from experts4bit_qlora.engines.paged_runner import PrefillGraphRefused
        lean = os.environ.get(P2_ENV) == "1"
        ids = torch.randint(0, 256, (1, T), generator=torch.Generator().manual_seed(1689))
        pos = torch.arange(T)[None]
        if ("p2_on" if lean else "p2_off") in self.refuse:
            for _ in range(3):
                self._prefill_forward(ids, pos)
            raise PrefillGraphRefused(f"the {T}-token prefill forward did not capture (RuntimeError: a sync)")
        for _ in range(5):                               # two warm-ups, the capture, two startup checks
            out = self._prefill_forward(ids, pos)
        staged = {lay: (torch.zeros(T, 4), torch.zeros(T, 4)) for lay in range(self.layers)}
        pg = {"T": T, "ids": ids.clone(), "logits": out.logits.clone(), "staged": staged}
        pg["graph"] = _Graph(self, pg, lean)
        self._prefill_graph = pg
        return {"status": "on", "T": T, "replays": 0, "pool_mib": 0}

    def disable_prefill_graph(self):
        self._prefill_graph = None


T, NW = 24, 3


@pytest.fixture(scope="module")
def box():
    pytest.importorskip("transformers.models.qwen3_moe", reason="needs transformers with Qwen3-MoE")
    return _load()


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(P1_ENV, raising=False)
    monkeypatch.delenv(P2_ENV, raising=False)


def _prompts():
    g = torch.Generator().manual_seed(2)
    return [torch.randint(0, 256, (T + 5,), generator=g).tolist() for _ in range(NW)]


def _speed(box, runner, prompts=None):
    return box.speed(runner.model, prompts or _prompts(), T, "cpu", rounds=2, warm=1, eager_rounds=1, eager_windows=2,
                     runner=runner)


def test_each_capture_records_its_own_route_dispatch_and_folds(box, clean_env):
    rec = _speed(box, _PrefillStandIn(_tiny()))
    L = 2
    for g, dispatch in (("p2_off", "gather"), ("p2_on", "lean")):
        gr = rec["graphs"][g]
        assert gr["status"] == "on" and gr["T"] == T and gr["forwards"] == 5, gr
        assert gr["seen"] == {"route": {"int4_k19|gt256": 5 * L}, "k19": {f"chained|{dispatch}|gt256": 5 * L}, "folds": {}}


def test_with_p1_set_the_fold_deltas_follow_the_table(box, clean_env, monkeypatch):
    monkeypatch.setenv(P1_ENV, "1")
    rec = _speed(box, _PrefillStandIn(_tiny()))
    want = {k: 5 * v for k, v in box.fold_table(2).items()}
    assert box.fold_table(48) == {"norm": 49, "layer": 48, "attention": 48}
    assert rec["graphs"]["p2_off"]["seen"]["folds"] == want and rec["graphs"]["p2_on"]["seen"]["folds"] == want


def test_function_digests_and_every_timed_replay(box, clean_env):
    model = _tiny()
    rec = _speed(box, _PrefillStandIn(model))
    assert rec["function_p2"] == {"windows": NW, "differ": []}
    assert len(rec["digests_p2_off"]) == NW and len(set(rec["digests_p2_off"])) == NW
    assert {g: len(v) for g, v in rec["replay_ms"].items()} == {"p2_off": 2 * NW, "p2_on": 2 * NW}
    assert set(rec["median_ms"]) == {"p2_off", "p2_on"} and all(v > 0 for v in rec["median_ms"].values())
    again = _speed(box, _PrefillStandIn(model))
    assert again["digests_p2_off"] == rec["digests_p2_off"]           # the cross-process gate's comparison


def test_the_eager_forwards_are_counted_and_both_settings_timed(box, clean_env):
    rec = _speed(box, _PrefillStandIn(_tiny()))
    e = rec["eager"]
    assert e["forwards"] == 1 * 2 * 2 and {g: len(v) for g, v in e["ms"].items()} == {"p2_off": 2, "p2_on": 2}
    assert e["seen"]["k19"] == {"chained|gather|gt256": 2 * 2, "chained|lean|gt256": 2 * 2}


def test_the_knob_and_the_grouping_are_restored(box, clean_env, monkeypatch):
    from experts4bit_qlora.engines import hot_residency as hr
    _speed(box, _PrefillStandIn(_tiny()))
    assert P2_ENV not in os.environ and hr.DEVICE_GROUPING[0] is False
    monkeypatch.setenv(P2_ENV, "0")
    _speed(box, _PrefillStandIn(_tiny()))
    assert os.environ[P2_ENV] == "0"


def test_function_catches_a_kv_only_difference(box, clean_env):
    rec = _speed(box, _PrefillStandIn(_tiny(), perturb_p2=True))
    assert [d["window"] for d in rec["function_p2"]["differ"]] == list(range(NW))
    assert all(d["logits_equal"] and d["kv_layers_differ"] == [1] for d in rec["function_p2"]["differ"])


def test_a_refused_p2_capture_still_times_the_default(box, clean_env):
    rec = _speed(box, _PrefillStandIn(_tiny(), refuse=("p2_on",)))
    assert rec["graphs"]["p2_on"]["status"] == "refused" and "did not capture" in rec["graphs"]["p2_on"]["why"]
    assert rec["function_p2"] is None and set(rec["replay_ms"]) == {"p2_off"} and len(rec["digests_p2_off"]) == NW


def test_a_refused_default_capture_times_nothing(box, clean_env):
    rec = _speed(box, _PrefillStandIn(_tiny(), refuse=("p2_off",)))
    assert rec["graphs"]["p2_off"]["status"] == "refused" and "replay_ms" not in rec and "function_p2" not in rec


# ---- Phase B: P117's stand-in runner (decodes by full-sequence forwards; splits and pads as the runner does)

class _StandIn:
    def __init__(self, model, kv, device="cpu", **_):
        self.model, self.tokens, self.graph_stats, self.prompt_len = model, {}, {}, {}

    def enable_decode_graphs(self, buckets, *, capture=True, warmup=2, verbose=True):
        self.buckets, self.capture = tuple(sorted(int(b) for b in buckets)), capture
        self.graph_stats = {b: {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in self.buckets}
        return {b: ("graph" if capture else "eager: capture=False") for b in self.buckets}

    def bind(self, rid, slot, prompt):
        self.tokens[rid], self.prompt_len[rid] = list(prompt), len(prompt)

    def run_prefill(self, chunks):
        out = {}
        for rid, s, n in chunks:
            o = self.model(input_ids=torch.tensor([self.tokens[rid][:s + n]]))
            if s + n >= self.prompt_len[rid]:
                out[rid] = int(o.logits[0, -1].argmax())
                self.tokens[rid].append(out[rid])
        return out

    def run_decode(self, rids):
        from experts4bit_qlora.engines.paged_runner import bucket_for, chunk_rows
        got = {}
        for piece in chunk_rows(rids, self.buckets[-1]):
            b = bucket_for(len(piece), self.buckets)
            ids = torch.tensor([self.tokens[r] for r in piece] + [self.tokens[piece[0]]] * (b - len(piece)))
            st = self.graph_stats[b]
            logits = self.model(input_ids=ids).logits
            st["eager_steps"] += 1
            st["rows"] += len(piece)
            st["pad_rows"] += b - len(piece)
            for j, r in enumerate(piece):
                got[r] = int(logits[j, -1].argmax())
                self.tokens[r].append(got[r])
        return got


P, C, N = 12, 5, 20


@pytest.fixture(scope="module")
def quality(box, tmp_path_factory):
    pytest.importorskip("fp8_kv", reason="Phase B's pool is grouped-nf4-gemm's fp8 KV")
    from experts4bit_qlora.engines import paged_runner
    real = paged_runner.PagedModelRunner
    paged_runner.PagedModelRunner = _StandIn
    ref = tmp_path_factory.mktemp("ref")
    try:
        g = torch.Generator().manual_seed(1)
        ws = [torch.randint(0, 256, (P + C,), generator=g).tolist() for _ in range(N)]
        model = _tiny()
        off = box.quality_off(model, ws, prompt=P, cont=C, chunk=P, floor_chunk=4, device="cpu", ref_dir=str(ref),
                              stand_in=True)
        on = box.quality_on(model, ws, prompt=P, cont=C, chunk=P, device="cpu", ref_dir=str(ref), stand_in=True)
    finally:
        paged_runner.PagedModelRunner = real
    return off, on, ws, ref


def test_phase_b_off_scores_every_arm_and_saves_r(quality, box):
    off, _on, _ws, ref = quality
    assert list(off["per_window"]) == list(box.OFF_ARMS) and all(len(v) == N for v in off["per_window"].values())
    assert off["rep_identical"] is True and len(off["ref_digests"]) == N
    assert sorted(p.name for p in ref.iterdir()) == ["digests.json"] + [f"r{i:03d}.pt" for i in range(N)]
    r = {x["window"]: x["nll"] for x in off["per_window"]["R"]}
    for arm in ("half", "rev"):
        assert all(abs(x["nll"] - r[x["window"]]) < 1e-4 and x["argmax_agree"] == 1.0 for x in off["per_window"][arm])
    e = {a: off["engagement"][a][0] for a in box.OFF_ARMS}
    assert e["R"]["graph_stats"]["16"]["eager_steps"] == C - 1 and e["R"]["graph_stats"]["4"]["eager_steps"] == C - 1
    assert e["half"]["graph_stats"]["8"]["eager_steps"] == 2 * (C - 1) and e["chunk"]["chunk"] == 4
    assert all(set(e[a]["seen"]) == {"route", "k19", "folds"} for a in box.OFF_ARMS)


def test_phase_b_on_scores_the_subject_against_rs_saved_log_probs(quality, box):
    off, on, _ws, _ref = quality
    assert on["ref_ok"] is True and list(on["per_window"]) == ["P1"] and len(on["per_window"]["P1"]) == N
    r = {x["window"]: x["nll"] for x in off["per_window"]["R"]}
    for x in on["per_window"]["P1"]:                 # the same model: the subject reproduces R exactly
        assert abs(x["nll"] - r[x["window"]]) < 1e-6 and x["argmax_agree"] == 1.0 and abs(x["kl"]) < 1e-6
    assert on["engagement"]["P1"][0]["buckets"] == [1, 2, 4, 8, 16]


def test_a_corrupted_reference_is_caught_on_load(quality, box):
    _off, _on, ws, ref = quality
    from experts4bit_qlora.engines import paged_runner
    t = torch.load(ref / "r003.pt")
    torch.save(t + 1e-6, ref / "r003.pt")
    real = paged_runner.PagedModelRunner
    paged_runner.PagedModelRunner = _StandIn
    try:
        on = box.quality_on(_tiny(), ws, prompt=P, cont=C, chunk=P, device="cpu", ref_dir=str(ref), stand_in=True)
    finally:
        paged_runner.PagedModelRunner = real
        torch.save(t, ref / "r003.pt")
    assert on["ref_ok"] is False and on["ref_bad"] == [3]
