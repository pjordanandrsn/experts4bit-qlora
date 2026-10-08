"""Opt-in prefill LM-head row reduction; no performance or quality licence."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experts4bit_qlora import serve_paged
from experts4bit_qlora.engines.paged_runner import PagedModelRunner, _last_logits_kwargs
from experts4bit_qlora.serve_paged import PagedServeConfig, _last_logits_env


def _routing_fixture():
    spec = importlib.util.spec_from_file_location("prefill_fixture", Path(__file__).with_name("test_prefill_graph.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExplicitModel:
    def forward(self, input_ids, *, logits_to_keep=0):
        pass


class LegacyModel:
    def forward(self, input_ids, num_logits_to_keep=0):
        pass


class KwargsOnly:
    def forward(self, **kwargs):
        pass


class PositionalOnly:
    def forward(self, logits_to_keep, /):
        pass


@pytest.mark.parametrize("model,key", [(ExplicitModel(), "logits_to_keep"), (LegacyModel(), "num_logits_to_keep")])
def test_only_an_explicit_keyword_can_enable_the_mode(model, key):
    assert _last_logits_kwargs(model, True) == {key: 1}


@pytest.mark.parametrize("model", [KwargsOnly(), PositionalOnly()])
def test_catch_all_and_positional_only_signatures_refuse(model):
    with pytest.raises(ValueError, match="explicit"):
        _last_logits_kwargs(model, True)
    assert _last_logits_kwargs(model, False) == {}


@pytest.mark.parametrize("value,expected", [("", False), ("0", False), ("1", True), (" 1 ", True)])
def test_only_the_explicit_opt_in_changes_the_default(value, expected):
    assert _last_logits_env(value) is expected
    assert not PagedServeConfig().last_logits


@pytest.mark.parametrize("value", ["auto", "true", "2"])
def test_unread_modes_are_refused(value):
    with pytest.raises(ValueError, match="E4B_PAGED_LAST_LOGITS"):
        _last_logits_env(value)


def test_startup_reads_the_opt_in_and_health_reports_the_engaged_runner(monkeypatch):
    monkeypatch.setattr(serve_paged, "_capability", lambda _device: None)
    monkeypatch.setenv("E4B_PAGED_DEVICE", "cpu")
    monkeypatch.setenv("E4B_PAGED_LAST_LOGITS", "1")
    cfg = PagedServeConfig.from_env()
    assert cfg.last_logits
    fixture = _routing_fixture()

    class SlicedModel(fixture.StubModel):
        def forward(self, input_ids, position_ids=None, use_cache=False, logits_to_keep=0):
            result = super().forward(input_ids, position_ids, use_cache)
            if logits_to_keep:
                result.logits = result.logits[:, -logits_to_keep:]
            return result

    runner = PagedModelRunner(SlicedModel(), fixture.RecordingKV(), device="cpu", last_logits=True)
    fixture._drive(runner)
    spec = importlib.util.spec_from_file_location("serve_fixture", Path(__file__).with_name("test_serve_paged.py"))
    server_fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_fixture)
    client, engine = server_fixture._client(server_fixture.ScriptedRunner(), last_logits=True)
    engine.parts.runner = runner
    with client as http:
        report = http.get("/health").json()["last_logits"]
    assert report == {"status": "on", "requested": True, "keyword": "logits_to_keep", "prefill_forward_calls": 6}


def test_eager_interleaved_chunks_keep_all_kv_and_the_same_first_tokens():
    fixture = _routing_fixture()

    class SlicedModel(fixture.StubModel):
        def forward(self, input_ids, position_ids=None, use_cache=False, logits_to_keep=0):
            result = super().forward(input_ids, position_ids, use_cache)
            if logits_to_keep:
                result.logits = result.logits[:, -logits_to_keep:]
            return result

    full = PagedModelRunner(SlicedModel(), fixture.RecordingKV(), device="cpu")
    last = PagedModelRunner(SlicedModel(), fixture.RecordingKV(), device="cpu", last_logits=True)
    assert fixture._drive(full) == fixture._drive(last)
    for slot, (k, v) in full.kv.appended.items():
        lk, lv = last.kv.appended[slot]
        assert torch.equal(k, lk) and torch.equal(v, lv)
    assert last.last_logits_stats() == {"status": "on", "keyword": "logits_to_keep", "prefill_forward_calls": 6}
    assert full.last_logits_stats()["status"] == "off"


def test_a_model_that_ignores_the_keyword_is_caught():
    fixture = _routing_fixture()

    class IgnoringModel(fixture.StubModel):
        def forward(self, input_ids, position_ids=None, use_cache=False, logits_to_keep=0):
            return super().forward(input_ids, position_ids, use_cache)

    runner = PagedModelRunner(IgnoringModel(), fixture.RecordingKV(), device="cpu", last_logits=True)
    with pytest.raises(RuntimeError, match="requested one position"):
        fixture._drive(runner)


def test_capture_control_flow_uses_the_mode_for_every_warmup_and_capture(monkeypatch):
    """CPU stand-ins check routing/counts, not CUDA capture correctness."""
    fixture = _routing_fixture()

    class SlicedModel(fixture.StubModel):
        def forward(self, input_ids, position_ids=None, use_cache=False, logits_to_keep=0):
            assert not torch.is_grad_enabled()
            assert logits_to_keep == 1
            result = super().forward(input_ids, position_ids, use_cache)
            result.logits = result.logits[:, -1:]
            return result

    runner = PagedModelRunner(SlicedModel(), fixture.RecordingKV(), device="cpu", last_logits=True)
    stream = SimpleNamespace(wait_stream=lambda _other: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda _device: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: object())
    monkeypatch.setattr(torch.cuda, "graph", lambda _graph: nullcontext())
    monkeypatch.setattr(runner, "_private_pool_bytes", lambda _graph: 0)
    captured = runner._capture_prefill_graph(4, torch.tensor([[1, 2, 3, 4]]), warmup=3)
    assert runner.last_logits_stats()["prefill_forward_calls"] == 4
    assert captured["logits"].shape == (1, 1, fixture.V)
    assert captured["staged"][0][0].shape[0] == 4
    assert runner.ctx.mode == "decode" and not runner.ctx.staging


def test_real_qwen_head_projects_one_row_while_the_decoder_processes_the_prompt():
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.manual_seed(41)
    config = transformers.Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                      vocab_size=128, max_position_embeddings=64)
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 16))
    head_shapes, layer_shapes = [], []
    head = model.lm_head.register_forward_pre_hook(lambda _m, args: head_shapes.append(tuple(args[0].shape)))
    layer = model.model.layers[0].register_forward_pre_hook(lambda _m, args: layer_shapes.append(tuple(args[0].shape)))
    try:
        with torch.no_grad():
            full = model(input_ids=ids, use_cache=False).logits
            last = model(input_ids=ids, use_cache=False, **_last_logits_kwargs(model, True)).logits
    finally:
        head.remove()
        layer.remove()
    assert head_shapes == [(1, 16, 32), (1, 1, 32)]
    assert layer_shapes == [(1, 16, 32), (1, 16, 32)]
    assert last.shape == (1, 1, 128)
    torch.testing.assert_close(last, full[:, -1:], rtol=1e-5, atol=1e-6)
    assert torch.equal(last.argmax(-1), full[:, -1:].argmax(-1))
