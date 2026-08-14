"""The DFlash drafter GGUF key map, checked in both directions against the
real released tensor list and the built module tree.

The specific hazard here is a map that LOOKS like Glimmer's. The drafter shares
tensor names with the 30B but means different things by two of them, so the
tests below pin exactly the divergences: `attn_q_norm`/`attn_k_norm` are real
learned parameters (the 30B drops them as scalars), and the norms are not
centered (the 30B's need `gguf - 1`). Either mistake corrupts weights while
everything still loads.
"""
import re

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.arch.glimmer_draft import (  # noqa: E402
    GlimmerDraftKeymapError,
    expected_param_names,
    map_draft_key,
)

REPO = "meta-models/Muse-Glimmer-30B-assistant"
N_LAYERS = 5

# The real released surface (parsed from dflash-kquant.gguf's header).
_GLOBAL_GGUF = ["fc.weight", "enc.output_norm.weight", "output_norm.weight"]
_PER_LAYER_GGUF = [
    "attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_k.weight",
    "attn_v.weight", "attn_output.weight", "attn_q_norm.weight",
    "attn_k_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
]


def _gguf_names(n=N_LAYERS):
    return _GLOBAL_GGUF + [f"blk.{i}.{s}" for i in range(n) for s in _PER_LAYER_GGUF]


def test_released_tensor_count_reconciles():
    # The released GGUF holds 58 tensors; the map must claim all of them.
    names = _gguf_names()
    assert len(names) == 3 + 11 * N_LAYERS == 58
    assert len({map_draft_key(n) for n in names}) == 58


def test_qk_norms_are_mapped_here_not_dropped():
    """The 30B drops these as scalars; the drafter's are real parameters."""
    assert map_draft_key("blk.3.attn_q_norm.weight") == "layers.3.self_attn.q_norm.weight"
    assert map_draft_key("blk.3.attn_k_norm.weight") == "layers.3.self_attn.k_norm.weight"


def test_encoder_tensors_map():
    assert map_draft_key("fc.weight") == "encoder.fc.weight"
    assert map_draft_key("enc.output_norm.weight") == "encoder.output_norm_enc.weight"
    assert map_draft_key("output_norm.weight") == "norm.weight"


def test_thirty_b_only_tensors_are_refused():
    # attn_gate / sandwich norms / embeddings exist in the 30B, not the drafter.
    for bad in ["blk.0.attn_gate.weight", "blk.0.post_attention_norm.weight",
                "blk.0.post_ffw_norm.weight", "token_embd.weight", "output.weight"]:
        with pytest.raises(GlimmerDraftKeymapError):
            map_draft_key(bad)


def _meta_model():
    pytest.importorskip("transformers")
    try:
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(REPO)
        with torch.device("meta"):
            model = AutoModel.from_config(cfg)
    except Exception as e:
        pytest.skip(f"cannot build drafter tree (hermetic CI): {e}")
    return model, cfg


def test_forward_and_reverse_coverage_on_meta_model():
    model, cfg = _meta_model()
    n = cfg.num_hidden_layers
    params = {p for p, _ in model.named_parameters()}
    claimed = set()
    for name in _gguf_names(n):
        target = map_draft_key(name)
        assert target in params, f"forward: {name} -> {target} not in model"
        claimed.add(target)
    missing = expected_param_names(n) - params
    assert not missing, f"map targets absent from model: {sorted(missing)[:5]}"
    uncovered = params - claimed
    assert not uncovered, f"reverse: params no GGUF tensor claims: {sorted(uncovered)[:5]}"


def test_norms_are_not_centered_here():
    """Guards the divergence from the 30B: applying `gguf - 1` here would shift
    every norm by one. The drafter's RMSNorm is `weight * normed`, plain."""
    import inspect
    m = pytest.importorskip(
        "transformers.models.muse_glimmer_assistant.modeling_muse_glimmer_assistant")
    src = inspect.getsource(m.MuseGlimmerAssistantRMSNorm.forward)
    assert "1.0 + self.weight" not in src and "1 + self.weight" not in src, \
        "drafter norm became centered — the loader must then subtract 1"
    assert re.search(r"self\.weight\s*\*", src), "unexpected drafter norm forward"

# --- drafter load ------------------------------------------------------------

def _expected(n):
    from experts4bit_qlora.arch.glimmer_draft import expected_param_names
    return expected_param_names(n)


def _gguf_names_for(n):
    """The GGUF spelling of every parameter, i.e. map_draft_key's inverse."""
    from experts4bit_qlora.arch.glimmer_draft import _GLOBAL, _PER_LAYER
    names = [g for g in _GLOBAL]
    for i in range(n):
        names += [f"blk.{i}.{g}" for g in _PER_LAYER]
    return names


def test_both_released_spellings_build_the_same_parameter_set():
    """The safetensors drafter already spells its tensors as the module does
    (verified 58/58 against meta-models/Muse-Glimmer-30B-assistant), while the
    GGUF one uses blk.N.*. Both must converge on the same parameters."""
    import torch
    from experts4bit_qlora.arch.glimmer_draft import load_draft_state_dict
    exp = _expected(2)
    st = load_draft_state_dict(lambda k: torch.zeros(1), sorted(exp), 2,
                               source="safetensors")
    gg = load_draft_state_dict(lambda k: torch.zeros(1), _gguf_names_for(2), 2,
                               source="gguf")
    assert set(st) == set(gg) == exp


def test_a_missing_drafter_weight_raises_instead_of_degrading_silently():
    """This is the whole point of reconciling coverage: a dropped drafter
    weight does not crash, it just lowers the acceptance rate — which reads as
    'speculation isn't helping' rather than as a bug."""
    import torch
    import pytest
    from experts4bit_qlora.arch.glimmer_draft import (
        GlimmerDraftKeymapError, load_draft_state_dict)
    keys = sorted(_expected(2))
    with pytest.raises(GlimmerDraftKeymapError, match="missing"):
        load_draft_state_dict(lambda k: torch.zeros(1), keys[:-1], 2)


def test_an_unexpected_drafter_tensor_also_raises():
    import torch
    import pytest
    from experts4bit_qlora.arch.glimmer_draft import (
        GlimmerDraftKeymapError, load_draft_state_dict)
    keys = sorted(_expected(2)) + ["layers.0.self_attn.rotary_emb.inv_freq"]
    with pytest.raises(GlimmerDraftKeymapError, match="unexpected"):
        load_draft_state_dict(lambda k: torch.zeros(1), keys, 2)
