"""Plan -> execute round trip on a real (tiny) MoE, values checked.

Builds a genuine small Qwen3-MoE, invents a checkpoint for it, plans, executes,
and then asserts the VALUES landed where the plan said — in particular that the
gate half of each fused stack really came from the gate tensors. A load that
merely runs proves nothing; this proves placement.
"""
import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.arch.moe_conventions import MoEConventionError  # noqa: E402
from experts4bit_qlora.arch.moe_load import execute_moe_plan  # noqa: E402
from experts4bit_qlora.arch.moe_plan import plan_moe_checkpoint  # noqa: E402


def _tiny_moe():
    """A real Qwen3-MoE module tree, small enough to materialize on CPU."""
    pytest.importorskip("transformers")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.for_model(
            "qwen3_moe", hidden_size=32, intermediate_size=64,
            moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, vocab_size=64,
            num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
            tie_word_embeddings=False,
        )
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg)
    except Exception as e:
        pytest.skip(f"cannot build a tiny qwen3_moe (hermetic CI): {e}")
    return model, cfg


def _fake_checkpoint(model, cfg):
    """Invent a checkpoint matching the tree: per-expert tensors, everything
    else spelled as the tree spells it. Values are index-coded so placement is
    verifiable after the fact."""
    keys, store = [], {}
    n_exp = cfg.num_experts
    for name, t in model.state_dict().items():
        if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.down_proj"):
            continue                                  # fused: supplied per-expert
        if "rotary_emb" in name or name.endswith("inv_freq"):
            continue                                  # computed, never shipped
        keys.append(name)
        store[name] = torch.full(tuple(t.shape), 0.5)
    E, hid = n_exp, cfg.hidden_size
    inter = cfg.moe_intermediate_size
    for layer in range(cfg.num_hidden_layers):
        for e in range(E):
            for proj, shape, val in (
                ("gate_proj", (inter, hid), 1.0 + e),
                ("up_proj", (inter, hid), 100.0 + e),
                ("down_proj", (hid, inter), 1000.0 + e),
            ):
                k = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
                keys.append(k)
                store[k] = torch.full(shape, float(val))
    return keys, store


def test_plan_then_execute_places_values_correctly():
    model, cfg = _tiny_moe()
    keys, store = _fake_checkpoint(model, cfg)
    plan = plan_moe_checkpoint(keys, model, "qwen3_moe")
    report = execute_moe_plan(plan, model, store.__getitem__,
                              device="cpu", dtype=torch.float32)
    assert report["still_meta"] == []
    assert report["fused_stacks"] == 2 * cfg.num_hidden_layers
    assert report["assigned"] == len(plan.passthrough)

    sd = dict(model.state_dict())
    inter = cfg.moe_intermediate_size
    gu = sd["model.layers.0.mlp.experts.gate_up_proj"]
    dn = sd["model.layers.0.mlp.experts.down_proj"]
    for e in range(cfg.num_experts):
        # GATE occupies the first block — the orientation everything hinges on.
        assert torch.allclose(gu[e, :inter], torch.full_like(gu[e, :inter], 1.0 + e)), \
            "gate half of the fused stack did not come from the gate tensors"
        assert torch.allclose(gu[e, inter:], torch.full_like(gu[e, inter:], 100.0 + e))
        assert torch.allclose(dn[e], torch.full_like(dn[e], 1000.0 + e))
    # A plain weight landed too, and rotary was rebuilt rather than left on meta.
    assert torch.allclose(sd["model.layers.0.self_attn.q_proj.weight"],
                          torch.full_like(sd["model.layers.0.self_attn.q_proj.weight"], 0.5))
    assert report["rebuilt_buffers"] >= 1


def test_loaded_model_actually_runs_a_forward():
    """The end-to-end claim: a model loaded this way computes without touching
    a meta tensor."""
    model, cfg = _tiny_moe()
    keys, store = _fake_checkpoint(model, cfg)
    plan = plan_moe_checkpoint(keys, model, "qwen3_moe")
    execute_moe_plan(plan, model, store.__getitem__, device="cpu", dtype=torch.float32)
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        out = model(ids)
    assert out.logits.shape == (1, 4, cfg.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_missing_tensor_is_refused_at_execution():
    model, cfg = _tiny_moe()
    keys, store = _fake_checkpoint(model, cfg)
    plan = plan_moe_checkpoint(keys, model, "qwen3_moe")
    victim = next(k for k in plan.passthrough if k.endswith("q_proj.weight"))
    plan.passthrough.pop(victim)               # simulate a plan/checkpoint drift
    with pytest.raises(MoEConventionError, match="still on meta"):
        execute_moe_plan(plan, model, store.__getitem__,
                         device="cpu", dtype=torch.float32)


def test_mis_shaped_tensor_is_refused():
    model, cfg = _tiny_moe()
    keys, store = _fake_checkpoint(model, cfg)
    plan = plan_moe_checkpoint(keys, model, "qwen3_moe")
    bad = next(k for k in plan.passthrough if k.endswith("q_proj.weight"))
    store[bad] = torch.zeros(3, 3)
    with pytest.raises(MoEConventionError, match="refusing to place a mis-shaped"):
        execute_moe_plan(plan, model, store.__getitem__,
                         device="cpu", dtype=torch.float32)
