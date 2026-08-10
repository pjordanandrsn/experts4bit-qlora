"""The Mixtral-convention key map, checked in both directions against a real
released checkpoint's key list and the built module tree.

The hazard specific to this family: released checkpoints are per-expert under
`block_sparse_moe` while transformers >= 5 builds a fused `mlp.experts` tree,
and **w1/w3 are shape-identical**, so nothing about the tensors themselves says
which is the gate. Getting that backwards computes `act(up) * gate` — a wrong
activation with every shape agreeing and nothing raising. The orientation test
below pins it against transformers' own forward.
"""
import inspect
import json
import os
import re
import urllib.request

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.mixtral import (  # noqa: E402
    MIXTRAL_CONVENTION_MODEL_TYPES,
    MixtralKeymapError,
    classify_key,
    expected_param_names,
    fuse_experts,
)

REPO = "mistralai/Mixtral-8x7B-Instruct-v0.1"
N_LAYERS = 32
N_EXPERTS = 8


def _c(key, n=N_LAYERS):
    return classify_key(key, num_hidden_layers=n)


def test_expert_roles_are_pinned_not_guessed():
    """w1=gate, w3=up, w2=down. w1 and w3 are shape-identical, so this can only
    come from the forward/converter — never from inference."""
    assert _c("model.layers.0.block_sparse_moe.experts.5.w1.weight")["role"] == "gate"
    assert _c("model.layers.0.block_sparse_moe.experts.5.w3.weight")["role"] == "up"
    assert _c("model.layers.0.block_sparse_moe.experts.5.w2.weight")["role"] == "down"


def test_router_and_passthrough_rename():
    got = _c("model.layers.7.block_sparse_moe.gate.weight")
    assert got == {"kind": "passthrough", "param": "model.layers.7.mlp.gate.weight"}
    assert _c("model.layers.7.self_attn.q_proj.weight")["kind"] == "passthrough"
    assert _c("lm_head.weight")["kind"] == "passthrough"


def test_unknown_and_out_of_range_raise():
    for bad in ["model.layers.0.mlp.experts.0.w1.weight",   # already-converted spelling
                "model.layers.0.block_sparse_moe.experts.0.w4.weight",
                "model.layers.0.self_attn.rotary_emb.inv_freq"]:
        with pytest.raises(MixtralKeymapError):
            _c(bad)
    with pytest.raises(MixtralKeymapError, match="but the model builds"):
        _c("model.layers.99.self_attn.q_proj.weight")


def test_fuse_puts_gate_first():
    E, inter, hidden = 3, 4, 5
    gate = [torch.full((inter, hidden), float(e)) for e in range(E)]
    up = [torch.full((inter, hidden), 10.0 + e) for e in range(E)]
    down = [torch.full((hidden, inter), 100.0 + e) for e in range(E)]
    gu, dn = fuse_experts(gate, up, down)
    assert gu.shape == (E, 2 * inter, hidden) and dn.shape == (E, hidden, inter)
    for e in range(E):
        assert torch.equal(gu[e, :inter], gate[e]), "gate must occupy the FIRST block"
        assert torch.equal(gu[e, inter:], up[e])


def test_fuse_refuses_partial():
    with pytest.raises(MixtralKeymapError):
        fuse_experts([torch.zeros(2, 2)], [None], [torch.zeros(2, 2)])
    with pytest.raises(MixtralKeymapError, match="count mismatch"):
        fuse_experts([torch.zeros(2, 2)], [], [])


def test_gate_first_matches_transformers_forward():
    """Independent pin: transformers' own expert forward chunks the linear
    output and applies the activation to the FIRST half."""
    M = pytest.importorskip("transformers.models.mixtral.modeling_mixtral")
    src = inspect.getsource(M.MixtralExperts.forward)
    assert re.search(r"gate,\s*up\s*=.*chunk\(2", src), \
        "Mixtral expert forward no longer chunks into (gate, up) — re-adjudicate"
    assert re.search(r"act_fn\(\s*gate\s*\)", src), \
        "activation no longer applied to the gate half — re-adjudicate the map"


def _released_keys():
    url = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors.index.json"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return list(json.load(r)["weight_map"])
    except Exception as e:
        pytest.skip(f"released index unavailable (hermetic CI): {e}")


def _meta_model(n_layers):
    pytest.importorskip("transformers")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(REPO)
        cfg.num_hidden_layers = n_layers      # tiny build, real structure
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg)
    except Exception as e:
        pytest.skip(f"cannot build Mixtral tree (hermetic CI): {e}")
    return model, cfg


def test_every_released_key_classifies():
    keys = _released_keys()
    kinds = {"passthrough": 0, "expert": 0}
    for k in keys:
        kinds[_c(k)["kind"]] += 1
    assert kinds["expert"] == N_EXPERTS * 3 * N_LAYERS, kinds
    assert sum(kinds.values()) == len(keys)


def test_forward_and_reverse_coverage_on_meta_model():
    """A 2-layer build carries the full per-layer structure; the map is
    layer-indexed so coverage on it generalizes."""
    n = 2
    model, _ = _meta_model(n)
    params = {p for p, _ in model.named_parameters()}
    claimed = set()
    for key in _released_keys():
        m = re.match(r"^model\.layers\.(\d+)\.", key)
        if m and int(m.group(1)) >= n:
            continue                     # outside the tiny build
        got = _c(key, n)
        if got["kind"] == "passthrough":
            assert got["param"] in params, f"forward: {key} -> not in model"
            claimed.add(got["param"])
        else:
            base = f"model.layers.{got['layer']}.mlp.experts."
            claimed.add(base + ("down_proj" if got["role"] == "down" else "gate_up_proj"))
    missing = expected_param_names(n) - params
    assert not missing, f"map targets absent from model: {sorted(missing)[:5]}"
    uncovered = params - claimed
    assert not uncovered, f"reverse: params no checkpoint key claims: {sorted(uncovered)[:5]}"


def test_convention_covers_the_aliased_families():
    """transformers aliases minimax/minimax_m2 onto the mixtral converter, so
    one map serves all three — the reason this module is convention-named."""
    assert MIXTRAL_CONVENTION_MODEL_TYPES == {"mixtral", "minimax", "minimax_m2"}
    import transformers.conversion_mapping as CM
    src = inspect.getsource(CM)
    for alias in ("minimax", "minimax_m2"):
        assert re.search(rf'"{alias}":\s*"mixtral"', src), \
            f"{alias} no longer aliases to mixtral — re-check its convention"
