# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The paged server's speculative-decode knobs (lane SD2, ``bench/sd2/PREREG-sd2.md`` B6).

``E4B_PAGED_SPEC`` is ``off`` by default, or ``eagle3`` with ``E4B_PAGED_SPEC_K`` (1-3) and ``E4B_PAGED_SPEC_HEAD``. It
needs the decode graphs and refuses the decode lookahead. Off, nothing is installed. ``_enable_spec`` checks the head
against the target and sets the runner up before any graph; ``/health`` reports what was asked, the build and the
census. ``load_head``'s pin is tested in ``tests/test_eagle3_draft.py``; it is replaced by a fake here so a tiny model can
stand in for the target.
"""
import types

import pytest
import torch

from experts4bit_qlora import serve_paged as sp
from experts4bit_qlora.serve_paged import PagedServeConfig


def test_the_spec_knobs_parse_and_refuse():
    assert sp._spec_env(None) == "off" and sp._spec_env("") == "off" and sp._spec_env(" EAGLE3 ") == "eagle3"
    with pytest.raises(ValueError, match="E4B_PAGED_SPEC"):
        sp._spec_env("ngram")
    assert sp._spec_k_env(None) == 0 and sp._spec_k_env(" 2 ") == 2
    with pytest.raises(ValueError, match="E4B_PAGED_SPEC_K"):
        sp._spec_k_env("two")


def _cfg(**kw):
    base = dict(graphs=True, buckets=(1, 2, 4, 8, 16), max_seqs=16, spec="eagle3", spec_k=2, spec_head="/heads/e3")
    base.update(kw)
    return PagedServeConfig(**base)


def test_spec_is_off_by_default(monkeypatch):
    monkeypatch.setattr(sp, "_capability", lambda device: None)
    for k in ("E4B_PAGED_SPEC", "E4B_PAGED_SPEC_K", "E4B_PAGED_SPEC_HEAD"):
        monkeypatch.delenv(k, raising=False)
    cfg = PagedServeConfig.from_env()
    assert cfg.spec == "off" and cfg.spec_k == 0 and cfg.spec_head == ""


def test_a_valid_spec_config_passes():
    _cfg().validate()


@pytest.mark.parametrize("kw,why", [
    (dict(spec_k=0), "E4B_PAGED_SPEC_K of 1, 2 or 3"),
    (dict(spec_k=4), "E4B_PAGED_SPEC_K of 1, 2 or 3"),
    (dict(spec_head=""), "E4B_PAGED_SPEC_HEAD"),
    (dict(graphs=False), "bucketed decode graphs"),
    (dict(decode_lookahead=True), "do not combine"),
    (dict(spec="off"), "E4B_PAGED_SPEC_K is set"),
])
def test_spec_configs_that_cannot_run_are_refused(kw, why):
    with pytest.raises(ValueError, match=why):
        _cfg(**kw).validate()


def test_the_env_reaches_the_config(monkeypatch):
    monkeypatch.setattr(sp, "_capability", lambda device: None)
    monkeypatch.setenv("E4B_PAGED_GRAPHS", "1")
    monkeypatch.setenv("E4B_PAGED_SPEC", "eagle3")
    monkeypatch.setenv("E4B_PAGED_SPEC_K", "3")
    monkeypatch.setenv("E4B_PAGED_SPEC_HEAD", "/h")
    cfg = PagedServeConfig.from_env()
    assert (cfg.spec, cfg.spec_k, cfg.spec_head) == ("eagle3", 3, "/h")
    monkeypatch.delenv("E4B_PAGED_SPEC_K")
    with pytest.raises(ValueError, match="E4B_PAGED_SPEC_K"):
        PagedServeConfig.from_env()


# ------------------------------------------------------------------------------------------ the build's part --

HID, V, L = 32, 64, 8


def _tiny():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(hidden_size=HID, intermediate_size=48, num_hidden_layers=L, num_attention_heads=4,
                                        num_key_value_heads=2, head_dim=8, vocab_size=V,
                                        max_position_embeddings=128)).eval()


def _fake_head(hidden=HID, vocab=V, nh=4, nkv=2, hd=8, inter=48):
    from experts4bit_qlora.engines.eagle3_draft import NAMES
    shapes = {"d2t": (vocab,), "embed_tokens.weight": (vocab, hidden), "fc.weight": (hidden, 3 * hidden),
              "layers.0.hidden_norm.weight": (hidden,), "layers.0.input_layernorm.weight": (hidden,),
              "layers.0.mlp.down_proj.weight": (hidden, inter), "layers.0.mlp.gate_proj.weight": (inter, hidden),
              "layers.0.mlp.up_proj.weight": (inter, hidden), "layers.0.post_attention_layernorm.weight": (hidden,),
              "layers.0.self_attn.k_proj.weight": (nkv * hd, 2 * hidden), "layers.0.self_attn.o_proj.weight": (hidden, nh * hd),
              "layers.0.self_attn.q_proj.weight": (nh * hd, 2 * hidden), "layers.0.self_attn.v_proj.weight": (nkv * hd, 2 * hidden),
              "lm_head.weight": (vocab, hidden), "norm.weight": (hidden,)}
    assert set(shapes) == set(NAMES)
    t = {n: (torch.zeros(s, dtype=torch.long) if n == "d2t" else torch.randn(s) / 8) for n, s in shapes.items()}
    geo = {"n_heads": nh, "n_kv": nkv, "head_dim": hd, "rope_theta": 10000.0, "eps": 1e-6, "hidden": hidden,
           "draft_vocab": vocab}
    return t, geo


def _parts(k=2, buckets=(1, 2, 4)):
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    model = _tiny()
    kv = Fp8PagedKV(L, 2, 8, batch=2, max_tokens_per_seq=64, device="cpu", scratch_slots=max(buckets), alias_slots=k)
    return model, kv, PagedModelRunner(model, kv, device="cpu")


def test_enable_spec_sets_the_runner_up_before_any_graph(monkeypatch):
    from experts4bit_qlora.engines import eagle3_draft
    monkeypatch.setattr(eagle3_draft, "load_head", lambda path: _fake_head())
    model, kv, runner = _parts(k=2)
    rec = sp._enable_spec(model, kv, runner, _cfg(spec_k=2, buckets=(1, 2, 4), device="cpu"))
    assert rec["mode"] == "eagle3" and rec["k"] == 2 and rec["hooks"] == 3
    assert rec["aux_layers"] == [2, 4, 5] and rec["verify_buckets"] == [2, 3]
    assert rec["draft_max_positions"] == kv.blocks_per_seq * kv.bt + 5
    assert runner.speculative and runner.spec.k == 2 and runner.spec.aux.mode == "decode"


@pytest.mark.parametrize("head,why", [(dict(hidden=16), "wide"), (dict(vocab=32), "embeds")])
def test_enable_spec_refuses_a_head_that_does_not_fit_the_target(monkeypatch, head, why):
    from experts4bit_qlora.engines import eagle3_draft
    monkeypatch.setattr(eagle3_draft, "load_head", lambda path: _fake_head(**head))
    model, kv, runner = _parts(k=1)
    with pytest.raises(ValueError, match=why):
        sp._enable_spec(model, kv, runner, _cfg(spec_k=1, buckets=(1, 2, 4), device="cpu"))
    assert runner.spec is None


def test_health_reports_what_was_asked_and_once_built_the_census(monkeypatch):
    cfg = _cfg()
    assert sp.spec_report(cfg, types.SimpleNamespace(parts=None)) == {"requested": "eagle3", "k": 2}
    off = PagedServeConfig()
    assert sp.spec_report(off, types.SimpleNamespace(parts=None)) == {"requested": "off", "k": None}
    from experts4bit_qlora.engines import eagle3_draft
    monkeypatch.setattr(eagle3_draft, "load_head", lambda path: _fake_head())
    model, kv, runner = _parts(k=2)
    rec = sp._enable_spec(model, kv, runner, _cfg(spec_k=2, buckets=(1, 2, 4), device="cpu"))
    runner.spec_graph_status = {2: "graph", 3: "graph"}
    parts = types.SimpleNamespace(runner=runner, info={"spec": rec})
    rep = sp.spec_report(cfg, types.SimpleNamespace(parts=parts))
    assert rep["build"] == rec and rep["post_graphs"] == {"2": "graph", "3": "graph"}
    assert rep["census"]["steps"] == 0 and rep["census"]["k"] == 2 and rep["census"]["tau_live"] is None
