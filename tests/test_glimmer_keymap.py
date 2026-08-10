"""The Muse-Glimmer GGUF -> transformers text-tower key map, checked in BOTH
directions, plus the norm value-transform arithmetic.

A name-only map that is merely plausible loads with every shape agreeing and
computes nonsense (untied head aliased to embed, a centered norm left
un-decremented). So, mirroring test_deepseek_v4_keys:

* forward — every GGUF text tensor lands on a param the built model has, or is
  an explicitly-asserted drop;
* reverse — every text-tower param the model has is claimed by exactly one GGUF
  tensor (the direction that catches a silently dropped weight);
* transforms — sub1 subtracts, drops assert the scalar identity and refuse a
  learned qk-norm.

The forward/reverse arms need the transformers module tree, so they build the
model on ``meta`` (no weights, no GPU) and skip if muse_glimmer is absent. The
GGUF name surface is synthesized from the observed per-layer schema — the
transform arithmetic arm is pure and always runs.
"""
import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.glimmer import (  # noqa: E402
    GlimmerKeymapError,
    expected_param_names,
    map_gguf_key,
    transform_weight,
)

# The real per-layer GGUF schema (observed by parsing both providers' headers).
_PER_LAYER_GGUF = [
    "attn_norm.weight", "post_attention_norm.weight", "ffn_norm.weight",
    "post_ffw_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_output.weight", "attn_gate.weight", "ffn_gate.weight", "ffn_up.weight",
    "ffn_down.weight", "attn_q_norm.weight", "attn_k_norm.weight",
]
_GLOBAL_GGUF = ["token_embd.weight", "output.weight", "output_norm.weight"]


def _gguf_names(n_layers):
    names = list(_GLOBAL_GGUF)
    for i in range(n_layers):
        names += [f"blk.{i}.{s}" for s in _PER_LAYER_GGUF]
    return names


def test_every_gguf_tensor_maps_or_drops():
    for name in _gguf_names(4):
        param, transform = map_gguf_key(name)
        assert transform in {"dequant", "asis", "sub1", "drop_q", "drop_k"}
        assert (param is None) == transform.startswith("drop")


def test_unknown_tensor_raises_not_warns():
    for bad in ["blk.0.attn_rot_embd", "blk.0.ffn_act.weight", "rope_freqs.weight"]:
        with pytest.raises(GlimmerKeymapError):
            map_gguf_key(bad)


def test_count_reconciles_exactly():
    # 627 mapped + 104 dropped qk-norms == 731 for the real 52-layer file.
    n = 52
    mapped = dropped = 0
    for name in _gguf_names(n):
        param, transform = map_gguf_key(name)
        if transform.startswith("drop"):
            dropped += 1
        else:
            mapped += 1
    assert dropped == 2 * n
    assert mapped == 3 + 12 * n
    assert mapped + dropped == 3 + 14 * n == 731


def test_sub1_and_asis_arithmetic():
    w = torch.full((6656,), 1.34)
    out = transform_weight(w, "sub1", qk_scale_factor=3.87, name="attn_norm")
    assert torch.allclose(out, torch.full((6656,), 0.34), atol=1e-6)
    passthrough = transform_weight(w, "asis", qk_scale_factor=3.87)
    assert passthrough is w


def test_drop_asserts_scalar_identity():
    q = torch.full((128,), 3.87)
    assert transform_weight(q, "drop_q", qk_scale_factor=3.87, name="q") is None
    k = torch.full((128,), 1.0)
    assert transform_weight(k, "drop_k", qk_scale_factor=3.87, name="k") is None
    # A learned (non-uniform) qk-norm must fail loudly, not be dropped.
    learned = torch.linspace(0.5, 4.0, 128)
    with pytest.raises(GlimmerKeymapError, match="learned qk-norm"):
        transform_weight(learned, "drop_q", qk_scale_factor=3.87, name="q")
    # Right shape, wrong constant (e.g. qk_scale_factor mismatch) also fails.
    with pytest.raises(GlimmerKeymapError):
        transform_weight(torch.full((128,), 2.0), "drop_q", qk_scale_factor=3.87)


def _build_meta_model():
    """Instantiate Glimmer on meta for the coverage arms. Hermetic-safe: any
    reason the tree can't be built here (transformers too old, muse_glimmer
    absent, no network for the config) is a SKIP, not a failure — the pure
    map/arithmetic arms above carry correctness in CI, and this deeper arm runs
    wherever transformers+config are reachable (it is green locally)."""
    pytest.importorskip("transformers")
    try:
        from transformers.models import muse_glimmer  # noqa: F401
    except Exception:
        pytest.skip("transformers build lacks muse_glimmer")
    try:
        from transformers import AutoConfig, AutoModelForImageTextToText
        cfg = AutoConfig.from_pretrained("meta-models/Muse-Glimmer-30B")
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(cfg)
    except Exception as e:
        pytest.skip(f"cannot obtain Glimmer config/tree (hermetic CI): {e}")
    return model, cfg


def test_forward_and_reverse_coverage_on_meta_model():
    model, cfg = _build_meta_model()
    n_layers = cfg.text_config.num_hidden_layers
    param_names = {n for n, _ in model.named_parameters()}
    text_params = {
        n for n in param_names
        if n.startswith("model.language_model.") or n == "lm_head.weight"
    }
    mapped_targets = set()
    for name in _gguf_names(n_layers):
        param, transform = map_gguf_key(name)
        if param is not None:
            assert param in text_params, f"forward: {name} -> {param} not in model"
            mapped_targets.add(param)
    missing = expected_param_names(n_layers) - text_params
    assert not missing, f"map targets absent from model: {sorted(missing)[:5]}"
    uncovered = text_params - mapped_targets
    assert not uncovered, f"reverse: model params no GGUF tensor claims: {sorted(uncovered)[:5]}"


def test_qk_scale_factor_matches_dropped_constant():
    _, cfg = _build_meta_model()
    assert abs(cfg.text_config.qk_scale_factor - 3.87) < 1e-6, \
        "the dropped attn_q_norm constant is pinned to this scalar"
