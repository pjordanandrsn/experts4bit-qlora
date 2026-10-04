# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The first-chunk prefill graph on a GPU (``E4B_PAGED_PREFILL_GRAPH``): capture, the startup check, replay, refusals.

Two runners over one tiny dense Qwen3 with the paged attention, each with its own FP8 pool. One prefills eagerly and
one serves first chunks of ``T`` tokens from the graph. They drive the same interleaving: a prompt that completes in
its replayed chunk, one that continues past it with two eager chunks, and another first-chunk replay BETWEEN the
continuing prompt's chunks (which overwrites the graph's K/V outputs), then a short first chunk. The first tokens and
every slot's FP8 pool contents must be equal bit for bit. The mutation arm stages the graph's K/V without the copy and
must be caught by the same comparison.

The model is dense on purpose: transformers' own MoE block routes experts with data-dependent indexing, which no CUDA
graph can hold. That e4b's MoE engine makes no host sync in the prefill forward is the A2000 census's finding
(``bench/prefill-graph-census-2026-10-04``), and the startup check re-verifies it on every engaged server. Prefill
needs no fp8 attention kernel, so any CUDA card runs these.
"""
import types

import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device for graph capture")
T = 16
PROMPTS = {"A": 16, "B": 40, "D": 16, "C": 10}           # prompt lengths
SLOTS = {"A": 0, "B": 1, "D": 2, "C": 3}
ORDER = [("A", 0, 16), ("B", 0, 16), ("D", 0, 16), ("B", 16, 16), ("C", 0, 10), ("B", 32, 8)]


def _model():
    pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm's fp8 KV")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    from experts4bit_qlora.engines import paged_attention
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=64, vocab_size=256, max_position_embeddings=256)
    model = Qwen3ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
    paged_attention.register(model)
    return model


def _runner(model):
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    cfg = model.config
    kv = Fp8PagedKV(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, batch=4, max_tokens_per_seq=64)
    return PagedModelRunner(model, kv)


def _prompts():
    g = torch.Generator().manual_seed(7)
    return {n: torch.randint(0, 256, (L,), generator=g).tolist() for n, L in PROMPTS.items()}


def _drive(runner, prompts):
    rid = {n: i for i, n in enumerate(prompts)}
    for n, p in prompts.items():
        runner.bind(rid[n], SLOTS[n], p)
    first = {}
    for n, start, take in ORDER:
        for _, tok in runner.run_prefill([(rid[n], start, take)]).items():
            first[n] = tok
    pool = {n: [runner.kv.reference_kv(layer, SLOTS[n]) for layer in runner.pool_layers] for n in prompts}
    return first, pool


def _same_pool(a, b):
    return all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(a, b))


@pytest.fixture
def grouping(monkeypatch):
    from experts4bit_qlora.engines import hot_residency
    monkeypatch.setattr(hot_residency, "DEVICE_GROUPING", [True])


@needs_cuda
def test_replayed_first_chunks_prefill_exactly_as_eager(grouping):
    model, prompts = _model(), _prompts()
    eager_first, eager_pool = _drive(_runner(model), prompts)
    r = _runner(model)
    st = r.enable_prefill_graph(T)
    assert {k: st[k] for k in ("status", "T", "replays", "eager_chunks", "eager_reasons")} == {
        "status": "on", "T": T, "replays": 0, "eager_chunks": 0,
        "eager_reasons": {"later_chunk": 0, "short_chunk": 0}}                  # the startup check is not counted
    assert st["pool_mib"] >= 0 and st["free_after_mib"] > 0
    first, pool = _drive(r, prompts)
    assert first == eager_first
    for n in prompts:
        assert _same_pool(pool[n], eager_pool[n]), n
    # the prompts differ, so equal pools are not a coincidence of identical inputs
    assert not _same_pool(eager_pool["A"], eager_pool["D"])
    st = r.prefill_graph_stats()
    assert {k: st[k] for k in ("status", "T", "replays", "eager_chunks", "eager_reasons")} == {
        "status": "on", "T": T, "replays": 3, "eager_chunks": 3, "eager_reasons": {"later_chunk": 2, "short_chunk": 1}}


@needs_cuda
def test_without_the_copy_the_continuing_prompts_pool_is_wrong(grouping):
    """The mutation arm: D's replay overwrites the graph's K/V before B's flush."""
    model, prompts = _model(), _prompts()
    _, eager_pool = _drive(_runner(model), prompts)
    r = _runner(model)
    r.enable_prefill_graph(T)

    def no_copy(self, rid, slot, take, done):
        pg = self._prefill_graph
        pg["ids"].copy_(torch.tensor(self.tokens[rid][:take], dtype=torch.long)[None])
        pg["graph"].replay()
        self.ctx.drop(slot)
        for layer, (k, v) in pg["staged"].items():
            self.ctx.staging[(layer, slot)] = ([k], [v])
        return pg["logits"]

    r._replay_prefill_graph = types.MethodType(no_copy, r)
    _, pool = _drive(r, prompts)
    assert not _same_pool(pool["B"], eager_pool["B"])
    assert _same_pool(pool["A"], eager_pool["A"])           # a prompt that completes in its replay is unaffected


@needs_cuda
def test_refuses_without_device_grouping(monkeypatch):
    from experts4bit_qlora.engines import hot_residency
    from experts4bit_qlora.engines.paged_runner import PrefillGraphRefused
    monkeypatch.setattr(hot_residency, "DEVICE_GROUPING", [False])
    r = _runner(_model())
    with pytest.raises(PrefillGraphRefused, match="device grouping is off"):
        r.enable_prefill_graph(T)
    assert r.prefill_graph_stats() == {"status": "off"}


@needs_cuda
def test_the_startup_check_refuses_a_graph_that_differs_from_eager(grouping):
    """A Python scalar that changes per call is baked into the capture: eager and replay then differ."""
    from experts4bit_qlora.engines.paged_runner import PrefillGraphRefused
    model, calls = _model(), [0]

    def drift(m, a):
        calls[0] += 1
        return (a[0] + 0.01 * calls[0],) + tuple(a[1:])

    model.model.layers[0].register_forward_pre_hook(drift)
    r = _runner(model)
    with pytest.raises(PrefillGraphRefused, match="differs from the eager forward"):
        r.enable_prefill_graph(T)


@needs_cuda
def test_the_startup_check_refuses_when_its_prompts_cannot_tell_graphs_apart(grouping):
    from experts4bit_qlora.engines.paged_runner import PrefillGraphRefused
    model = _model()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    r = _runner(model)
    with pytest.raises(PrefillGraphRefused, match="identical logits"):
        r.enable_prefill_graph(T)


@needs_cuda
def test_the_startup_check_refuses_a_graph_reading_a_tensor_nothing_keeps(grouping):
    """The lifetime mutation arm: the positions the graph reads are dropped after capture (as a first version did,
    with the check run while they were still alive, so it passed and every served replay read freed memory). The
    check runs after the capture's scope and an allocator churn, so it must see it."""
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, PrefillGraphRefused
    r = _runner(_model())
    orig = PagedModelRunner._capture_prefill_graph

    def drops_pos(self, T, first_ids, warmup):
        pg = orig(self, T, first_ids, warmup)
        pg["pos"] = pg["pos"].clone()          # the graph still reads the original block, now freed
        return pg

    r._capture_prefill_graph = types.MethodType(drops_pos, r)
    with pytest.raises(PrefillGraphRefused, match="differs from the eager forward"):
        r.enable_prefill_graph(T)


@needs_cuda
def test_auto_stands_down_when_the_pool_leaves_too_little_memory(grouping, monkeypatch):
    """With require_headroom (``auto``), free memory after capture below the graph's pool is a refusal, and the
    graph is released. The same runner engages when the device reports room."""
    from experts4bit_qlora.engines.paged_runner import PrefillGraphRefused
    r = _runner(_model())
    total = torch.cuda.mem_get_info()[1]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (0, total))
    with pytest.raises(PrefillGraphRefused, match="memory: the graph's private pool"):
        r.enable_prefill_graph(T, require_headroom=True)
    assert r._prefill_graph is None
    monkeypatch.undo()
    st = r.enable_prefill_graph(T, require_headroom=True)
    assert st["status"] == "on"


@needs_cuda
def test_refuses_a_forward_that_syncs(grouping):
    """Last in the file: a capture invalidated by a sync is the one failure that could disturb the context."""
    from experts4bit_qlora.engines.paged_runner import PrefillGraphRefused
    model = _model()
    model.model.layers[0].register_forward_pre_hook(lambda m, a: float(a[0].float().sum().item()) and None)
    r = _runner(model)
    with pytest.raises(PrefillGraphRefused, match="did not capture"):
        r.enable_prefill_graph(T)
    assert r.prefill_graph_stats() == {"status": "off"}
