"""The DeepSeek-V4 reference->transformers key map, checked against reality in BOTH
directions on the real checkpoint's own key list.

transformers ships no `_checkpoint_conversion_mapping` for `deepseek_v4`, so this map
is the only thing standing between the published shards and the module tree. A map that
is merely *plausible* produces a model that loads with every shape agreeing and computes
nonsense, so it is not enough to assert the table looks right:

* forward — every on-disk key must land on a name the built model actually has;
* reverse — every name the model has must be claimed by some on-disk key.

The reverse direction is the one that catches a silently-dropped tensor.
"""
import os
import re

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.deepseek_v4 import (  # noqa: E402
    DEEPSEEK_V4_RENAMES,
    rename_checkpoint_key,
)

# Both written by the scratchpad tooling; skipped when absent so CI stays hermetic.
KEYS = os.environ.get("E4B_V4_KEYS", "/workspace/v4_keys_l0_5.txt")
CFG = os.environ.get("E4B_V4_CONFIG", "/workspace/v4cfg")
N_LAYERS = 6  # the slice the key dump covers

EXPERT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.w[123]\.(weight|scale)$")
# Recomputed by the loader after placement, never read from a checkpoint.
ROTARY_RE = re.compile(r"(^|\.)rotary_emb\.")


def _ckpt_keys():
    with open(KEYS) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _is_fp8_companion(key, all_keys):
    """`X.scale` is an FP8 scale only when `X.weight` sits beside it on disk."""
    return key.endswith(".scale") and key[: -len(".scale")] + ".weight" in all_keys


def _model_names():
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(CFG)
    cfg.num_hidden_layers = N_LAYERS
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    return set(m.state_dict())


needs_files = pytest.mark.skipif(
    not (os.path.exists(KEYS) and os.path.exists(CFG)),
    reason="V4 key dump / config not present",
)


def test_mtp_block_is_dropped():
    """The text model does not build the multi-token-prediction layer."""
    assert rename_checkpoint_key("mtp.0.attn.wq_a.weight") is None
    assert rename_checkpoint_key("mtp.0.ffn.experts.3.w1.weight") is None


def test_indexer_nesting_is_inverted():
    """On disk: attn.indexer.compressor.*  In the tree: self_attn.compressor.indexer.*"""
    got = rename_checkpoint_key("layers.3.attn.indexer.compressor.wkv.weight")
    assert got == "model.layers.3.self_attn.compressor.indexer.kv_proj.weight"
    # and the bare compressor rule must not have swallowed it first
    assert rename_checkpoint_key("layers.3.attn.compressor.wkv.weight") == (
        "model.layers.3.self_attn.compressor.kv_proj.weight")


def test_shared_expert_uses_mlp_spelling_not_w123():
    assert rename_checkpoint_key("layers.4.ffn.shared_experts.w1.weight") == (
        "model.layers.4.mlp.shared_experts.gate_proj.weight")
    assert rename_checkpoint_key("layers.4.ffn.shared_experts.w3.weight") == (
        "model.layers.4.mlp.shared_experts.up_proj.weight")
    assert rename_checkpoint_key("layers.4.ffn.shared_experts.w2.weight") == (
        "model.layers.4.mlp.shared_experts.down_proj.weight")
    # ...while the routed siblings in the same block keep w1/w3/w2
    assert rename_checkpoint_key("layers.4.ffn.experts.7.w1.weight") == (
        "model.layers.4.mlp.experts.7.w1.weight")


def test_hc_head_keeps_its_prefix_but_layer_hc_does_not():
    assert rename_checkpoint_key("hc_head_fn") == "model.hc_head.hc_fn"
    assert rename_checkpoint_key("layers.2.hc_ffn_fn") == "model.layers.2.ffn_hc.fn"
    assert rename_checkpoint_key("layers.2.hc_attn_scale") == "model.layers.2.attn_hc.scale"


def test_router_correction_bias_is_renamed():
    assert rename_checkpoint_key("layers.9.ffn.gate.bias") == (
        "model.layers.9.mlp.gate.e_score_correction_bias")
    assert rename_checkpoint_key("layers.0.ffn.gate.tid2eid") == (
        "model.layers.0.mlp.gate.tid2eid")


def test_no_rule_is_shadowed_by_an_earlier_prefix():
    """An EARLIER rule that prefixes a later one consumes its keys first, so the later
    rule can never fire. (The reverse — a later rule prefixing an earlier one, e.g. the
    catch-all `.ffn.` after `.ffn.shared_experts.w1` — is exactly the intended shape.)"""
    olds = [o for o, _ in DEEPSEEK_V4_RENAMES]
    for i, a in enumerate(olds):
        for b in olds[i + 1:]:
            assert not b.startswith(a), (
                f"{a!r} precedes {b!r} and is a prefix of it, so {b!r} is unreachable")


def test_scale_suffix_is_overloaded_and_needs_a_sibling_weight():
    """`X.scale` is an FP8 companion ONLY when `X.weight` exists beside it.

    Hyper-connections ship a standalone parameter literally named `scale`
    (`attn_hc.scale`, `ffn_hc.scale`, `hc_head.hc_scale`) with no `weight` sibling.
    Treating every `.scale` key as an FP8 companion silently loses them.
    """
    assert rename_checkpoint_key("layers.2.hc_attn_scale") == "model.layers.2.attn_hc.scale"
    assert rename_checkpoint_key("layers.2.attn.wq_a.scale") == (
        "model.layers.2.self_attn.q_a_proj.scale")
    # the discriminator: only the second has a `.weight` sibling on disk
    assert rename_checkpoint_key("layers.2.attn.wq_a.weight") == (
        "model.layers.2.self_attn.q_a_proj.weight")


@needs_files
def test_every_checkpoint_key_lands_on_a_real_name():
    names = _model_names()
    keys = _ckpt_keys()
    all_keys = set(keys)
    unmapped, missing = [], []
    for k in keys:
        mapped = rename_checkpoint_key(k)
        if mapped is None:
            if not k.startswith("mtp."):
                unmapped.append(k)
            continue
        if EXPERT_RE.match(mapped):
            continue                       # fused into mlp.experts.{gate_up,down}_proj
        if _is_fp8_companion(k, all_keys):
            continue                       # consumed by Fp8BlockLinear, no module of its own
        if mapped not in names:
            missing.append((k, mapped))
    assert not unmapped, f"{len(unmapped)} keys mapped to None, e.g. {unmapped[:5]}"
    assert not missing, f"{len(missing)} keys hit no module, e.g. {missing[:5]}"


@needs_files
def test_every_model_name_is_claimed_by_the_checkpoint():
    """The direction that catches a tensor being silently dropped."""
    names = {n for n in _model_names() if not ROTARY_RE.search(n)}
    keys = _ckpt_keys()
    all_keys = set(keys)
    claimed = set()
    for k in keys:
        mapped = rename_checkpoint_key(k)
        if mapped is None:
            continue
        if EXPERT_RE.match(mapped):
            layer = mapped.split(".")[2]
            claimed.add(f"model.layers.{layer}.mlp.experts.gate_up_proj")
            claimed.add(f"model.layers.{layer}.mlp.experts.down_proj")
            continue
        if _is_fp8_companion(k, all_keys):
            continue
        claimed.add(mapped)
    orphans = sorted(names - claimed)
    assert not orphans, f"{len(orphans)} model tensors have no checkpoint source: {orphans[:8]}"
