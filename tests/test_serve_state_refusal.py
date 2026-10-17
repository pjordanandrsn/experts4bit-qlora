# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""serve_paged refuses a model whose state-carrying layers the paged runner cannot keep BEFORE any weight is read.

The serve estimate, serve_paged's refusal before the load (``config_state_refusal``) and the runner's ``layer_plan``
all ask one verdict, ``paged_runner.paged_state_refusal``. Real records first: the committed config fixtures
(``tests/fixtures/topology_controls``, digests checked against SOURCES.json) through the transformers config class,
then raw configs that carry no ``layer_types`` (the no-layer_types guard), then synthetic departures."""
import hashlib
import json
import pathlib
import types

import pytest

torch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from experts4bit_qlora.engines import paged_runner  # noqa: E402
from experts4bit_qlora.engines.paged_runner import (config_state_refusal, layer_plan,  # noqa: E402
                                                    undeclared_mixer_refusal)

REPO = pathlib.Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "topology_controls"
SOURCES = json.loads((FIX / "SOURCES.json").read_text(encoding="utf-8"))
REFUSED = {                                   # fixture -> the runner's reason, in its own words
    "nemotron-3.5-lightning-30b-a3b": "Gated DeltaNet only",
    "nemotron-3-super-120b-a12b": "Gated DeltaNet only",
    "granite-4.0-h-tiny": "Gated DeltaNet only",
    "lfm2-8b-a1b": "['conv']",
}
SERVED = ("qwen3-30b-a3b", "qwen3.6-35b-a3b", "olmoe-1b-7b-0924-instruct", "granite-3.1-3b-a800m-instruct",
          "mixtral-8x7b-instruct-v0.1", "gpt-oss-20b", "gemma-4-26b-a4b-it")


def _raw(name):
    raw = (FIX / name / "config.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCES[name]["sha256"], f"{name} is not the committed record"
    return json.loads(raw)


def _hf(name):
    """The fixture through its transformers config class (no download): what AutoConfig hands serve_paged."""
    d = _raw(name)
    return tr.CONFIG_MAPPING[d["model_type"]].from_dict(d)


def _ns(d):
    """A raw config.json as attributes, with no transformers class to derive anything: a config that reaches the runner
    exactly as written."""
    return types.SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def test_the_fixtures_are_the_committed_records():
    for name in (*REFUSED, *SERVED):
        _raw(name)


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_hybrids_are_refused_before_any_weight_is_read(name):
    why = config_state_refusal(_hf(name))
    assert why and REFUSED[name] in why, (name, why)


@pytest.mark.parametrize("name", SERVED)
def test_served_families_are_admitted(name):
    assert config_state_refusal(_hf(name)) is None


@pytest.mark.parametrize("name,key", [("nemotron-3.5-lightning-30b-a3b", "layers_block_type"),
                                      ("nemotron-3-super-120b-a12b", "hybrid_override_pattern")])
def test_a_hybrid_without_layer_types_is_refused_not_planned_as_attention(name, key):
    """The two Nemotron-H fixtures name their mixer list under DIFFERENT keys; the raw config carries no layer_types."""
    raw = _ns(_raw(name))
    assert not getattr(raw, "layer_types", None)
    why = undeclared_mixer_refusal(raw)
    assert why and key in why and "mamba_" in why and "no layer_types" in why, why
    stub = torch.nn.Linear(2, 2)
    stub.config = raw
    with pytest.raises(NotImplementedError, match="no layer_types"):
        layer_plan(stub, raw.num_hidden_layers, 2)


def test_a_plain_config_without_layer_types_is_still_all_attention():
    raw = _ns(_raw("qwen3-30b-a3b"))
    assert undeclared_mixer_refusal(raw) is None
    stub = torch.nn.Linear(2, 2)
    stub.config = raw
    assert layer_plan(stub, 4, 2) == ([0, 1, 2, 3], None)


@pytest.mark.parametrize("extra,refused", [({"attn_layer_period": 8, "attn_layer_offset": 4}, True),
                                           ({"ssm_state_size": 128}, True),
                                           ({"mamba_d_state": 16}, True),
                                           ({"mamba_d_state": 16, "layer_types": ["full_attention"] * 4}, False)])
def test_synthetic_departures(extra, refused):
    """A Jamba-style period, a bare SSM geometry: refused without layer_types. With layer_types the layer-type rules
    decide, not the guard."""
    raw = _ns({**_raw("qwen3-30b-a3b"), **extra})
    assert (undeclared_mixer_refusal(raw) is not None) is refused


def test_serve_paged_asks_the_one_verdict_before_the_load():
    src = (REPO / "experts4bit_qlora" / "serve_paged.py").read_text(encoding="utf-8")
    i_verdict = src.index("why = config_state_refusal(model_config)")
    i_refuse = src.index('raise RuntimeError(f"serve_paged refuses {cfg.model}: {why}")')
    i_load = src.index("load_moe_4bit_streaming(", i_refuse)
    assert i_verdict < i_refuse < i_load
    assert "kv_layout_refusal(model_config)" not in src          # MLA rides inside the one verdict now


def test_one_verdict_three_callers():
    """The estimate asks paged_state_refusal on its meta tree; config_state_refusal and layer_plan ask it too."""
    import inspect
    assert "paged_state_refusal(tree)" in inspect.getsource(paged_runner.config_state_refusal)
    assert "paged_state_refusal(model)" in inspect.getsource(paged_runner.layer_plan)
    topo = (REPO / "experts4bit_qlora" / "arch" / "topology.py").read_text(encoding="utf-8")
    assert "paged_state_refusal(tree)" in topo


@pytest.mark.parametrize("name,refused", [("nemotron-3-super-120b-a12b", True), ("qwen3-30b-a3b", False)])
def test_no_tree_falls_back_to_the_config_rules_and_says_so(name, refused):
    """A config transformers cannot build a tree from (here a raw namespace) warns, then the config-only rules answer."""
    with pytest.warns(RuntimeWarning, match="no empty-weight tree"):
        why = config_state_refusal(_ns(_raw(name)))
    assert (why is not None) is refused, why
