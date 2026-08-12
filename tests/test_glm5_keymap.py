"""The GLM-5 (glm_moe_dsa) checkpoint key map, checked in BOTH directions
against the real released key list and the built module tree.

Same shape as test_deepseek_v4_keys / test_glimmer_keymap, because the same
class of bug is in play: a map that is merely plausible loads with every shape
agreeing and computes nonsense. The GLM-5-specific hazards under test:

* the checkpoint carries ONE MORE layer than the model builds (an MTP head) —
  it must be classified as a skip, never mapped;
* experts are per-expert on disk and FUSED in the tree, with gate/up as blocks
  (not interleaved — the GPT-OSS convention would silently mis-activate);
* the same ``mlp.*_proj`` spelling means a dense MLP below
  first_k_dense_replace and something else above it.

The pure-map arms always run. The coverage arms need the transformers tree and
the released index, and SKIP (never fail) when either is unavailable so CI
stays hermetic.
"""
import json
import os
import urllib.request

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.arch.glm5 import (  # noqa: E402
    Glm5KeymapError,
    classify_key,
    expected_param_names,
    fuse_experts,
)

REPO = "zai-org/GLM-5"
N_LAYERS = 78            # transformers builds 0..77
FIRST_K_DENSE = 3
N_EXPERTS = 256


def _c(key):
    return classify_key(key, num_hidden_layers=N_LAYERS,
                        first_k_dense_replace=FIRST_K_DENSE)


def test_global_and_passthrough():
    assert _c("lm_head.weight")["kind"] == "passthrough"
    assert _c("model.layers.5.self_attn.kv_b_proj.weight")["param"].endswith("kv_b_proj.weight")
    # DSA indexer is a plain passthrough on every built layer.
    for s in ["wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias",
              "weights_proj.weight"]:
        assert _c(f"model.layers.7.self_attn.indexer.{s}")["kind"] == "passthrough"


def test_expert_keys_classify_with_index_and_proj():
    got = _c("model.layers.3.mlp.experts.17.gate_proj.weight")
    assert got == {"kind": "expert", "layer": 3, "expert": 17, "proj": "gate"}
    assert _c("model.layers.77.mlp.experts.255.down_proj.weight")["proj"] == "down"


def test_mtp_layer_is_skipped_not_mapped():
    # Layer 78 exists on disk, is not built, and carries the MTP signature.
    for suffix in ["eh_proj.weight", "enorm.weight", "hnorm.weight",
                   "mlp.experts.0.gate_proj.weight", "input_layernorm.weight"]:
        got = _c(f"model.layers.78.{suffix}")
        assert got["kind"] == "skip_mtp", (suffix, got)


def test_mtp_marker_on_built_layer_raises():
    # If our layer math were wrong, MTP markers would land on a built layer.
    with pytest.raises(Glm5KeymapError, match="layer-count"):
        _c("model.layers.10.eh_proj.weight")


def test_dense_vs_moe_boundary_is_enforced():
    # Dense MLP is legal below the boundary, illegal above it.
    assert _c("model.layers.0.mlp.gate_proj.weight")["kind"] == "passthrough"
    with pytest.raises(Glm5KeymapError, match="dense-MLP tensor on MoE layer"):
        _c("model.layers.9.mlp.gate_proj.weight")
    # Router/shared expert is legal above, illegal below.
    assert _c("model.layers.9.mlp.gate.weight")["kind"] == "passthrough"
    with pytest.raises(Glm5KeymapError, match="MoE tensor on dense layer"):
        _c("model.layers.1.mlp.shared_experts.up_proj.weight")
    with pytest.raises(Glm5KeymapError, match="expert tensor on dense layer"):
        _c("model.layers.1.mlp.experts.0.gate_proj.weight")


def test_unknown_key_raises():
    for bad in ["model.layers.4.self_attn.rotary_emb.inv_freq",
                "model.layers.4.mlp.experts.3.w1.weight",
                "some.other.tensor"]:
        with pytest.raises(Glm5KeymapError):
            _c(bad)


def test_fuse_experts_blocks_gate_then_up():
    E, inter, hidden = 3, 4, 5
    gate = [torch.full((inter, hidden), float(e)) for e in range(E)]
    up = [torch.full((inter, hidden), 10.0 + e) for e in range(E)]
    down = [torch.full((hidden, inter), 100.0 + e) for e in range(E)]
    gu, dn = fuse_experts(gate, up, down)
    assert gu.shape == (E, 2 * inter, hidden) and dn.shape == (E, hidden, inter)
    for e in range(E):
        # rows [0:inter] MUST be the gate — the forward chunks the linear output.
        assert torch.equal(gu[e, :inter], gate[e])
        assert torch.equal(gu[e, inter:], up[e])
        assert torch.equal(dn[e], down[e])


def test_fuse_experts_refuses_missing():
    with pytest.raises(Glm5KeymapError):
        fuse_experts([torch.zeros(2, 2)], [None], [torch.zeros(2, 2)])
    with pytest.raises(Glm5KeymapError, match="count mismatch"):
        fuse_experts([torch.zeros(2, 2)], [], [])


def _released_keys():
    url = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors.index.json"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return list(json.load(r)["weight_map"])
    except Exception as e:
        pytest.skip(f"released index unavailable (hermetic CI): {e}")


def _meta_model():
    pytest.importorskip("transformers")
    try:
        from transformers.models import glm_moe_dsa  # noqa: F401
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(REPO)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg)
    except Exception as e:
        pytest.skip(f"cannot build GLM-5 tree (hermetic CI): {e}")
    return model, cfg


def test_every_released_key_classifies():
    keys = _released_keys()
    kinds = {"passthrough": 0, "expert": 0, "skip_mtp": 0}
    for k in keys:
        kinds[_c(k)["kind"]] += 1
    assert kinds["expert"] == N_EXPERTS * 3 * (N_LAYERS - FIRST_K_DENSE), kinds
    assert kinds["skip_mtp"] == 791, kinds        # the whole MTP layer
    assert sum(kinds.values()) == len(keys)


def test_forward_and_reverse_coverage_on_meta_model():
    model, cfg = _meta_model()
    keys = _released_keys()
    # state_dict, not named_parameters: the router's e_score_correction_bias is
    # a BUFFER, and a params-only walk silently cannot see missing buffers.
    # rotary inv_freq is computed at build time and never shipped — excluded.
    params = {n for n in model.state_dict()
              if not n.startswith("model.rotary_emb.")}
    claimed = set()
    for k in keys:
        got = classify_key(k, num_hidden_layers=cfg.num_hidden_layers,
                           first_k_dense_replace=cfg.first_k_dense_replace)
        if got["kind"] == "passthrough":
            assert got["param"] in params, f"forward: {k} -> not in model"
            claimed.add(got["param"])
        elif got["kind"] == "expert":
            base = f"model.layers.{got['layer']}.mlp.experts."
            claimed.add(base + ("down_proj" if got["proj"] == "down" else "gate_up_proj"))
    expected = expected_param_names(cfg.num_hidden_layers, cfg.first_k_dense_replace)
    assert not (expected - params), f"map targets absent: {sorted(expected - params)[:5]}"
    uncovered = params - claimed
    assert not uncovered, f"reverse: params no checkpoint key claims: {sorted(uncovered)[:5]}"
