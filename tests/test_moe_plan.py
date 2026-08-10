"""End-to-end load planning across REAL MoE checkpoints from several families.

This is the test that decides whether "conventions" actually buys end-to-end
loading or just looked like it would. For each family it fetches the real
released key list, builds the real module tree (shrunk to a couple of layers,
on ``meta``, so it costs nothing), and demands a complete plan: every
checkpoint key lands, every model parameter is claimed, every expert stack is
whole. A family that needs a per-model special case fails here loudly instead
of silently loading wrong.

Network- and transformers-dependent arms SKIP rather than fail so CI stays
hermetic; the synthetic arms below always run and cover the failure modes.
"""
import json
import os
import re
import urllib.request

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.moe_conventions import MoEConventionError  # noqa: E402
from experts4bit_qlora.moe_plan import plan_moe_checkpoint  # noqa: E402

# (repo, model_type, layers to build) — one per convention shape we claim.
FAMILIES = [
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "mixtral", 2),
    ("allenai/OLMoE-1B-7B-0924", "olmoe", 2),
    ("Qwen/Qwen3-30B-A3B", "qwen3_moe", 2),
    ("microsoft/Phi-3.5-MoE-instruct", "phimoe", 2),
]


def _keys(repo):
    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return list(json.load(r)["weight_map"])
    except Exception as e:
        pytest.skip(f"{repo}: index unavailable (hermetic CI): {e}")


def _model(repo, n_layers):
    pytest.importorskip("transformers")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=False)
        inner = getattr(cfg, "text_config", cfg)
        inner.num_hidden_layers = n_layers
        for attr in ("first_k_dense_replace",):
            if getattr(inner, attr, 0) and getattr(inner, attr) >= n_layers:
                setattr(inner, attr, 0)   # keep at least one MoE layer in the build
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg)
    except Exception as e:
        pytest.skip(f"{repo}: cannot build tree (hermetic CI): {e}")
    return model, cfg


@pytest.mark.parametrize("repo,model_type,n", FAMILIES, ids=[f[1] for f in FAMILIES])
def test_real_checkpoint_plans_completely(repo, model_type, n):
    keys = _keys(repo)
    model, _ = _model(repo, n)
    # Restrict to layers the shrunk build actually has; non-layer keys all stay.
    kept = [k for k in keys
            if not (m := re.match(r"^model\.layers\.(\d+)\.", k)) or int(m.group(1)) < n]
    plan = plan_moe_checkpoint(kept, model, model_type)
    assert plan.passthrough, f"{model_type}: nothing passed through"
    assert plan.n_expert_stacks >= 2 * 1, f"{model_type}: no expert stacks planned"
    # Every stack is square: same expert count across gate/up/down.
    for layer, roles in plan.experts.items():
        counts = {r: len(v) for r, v in roles.items()}
        assert set(roles) == {"gate", "up", "down"}, (layer, sorted(roles))
        assert len(set(counts.values())) == 1, (layer, counts)
    print("\n  " + plan.summary())


def _toy(model_type="qwen3_moe", n_experts=2):
    """A minimal fake tree + key list exercising the planner's guarantees."""
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.ParameterDict()
        def state_dict(self, *a, **k):
            return dict(self.p)
    t = Toy()
    names = ["model.embed_tokens.weight",
             "model.layers.0.self_attn.q_proj.weight",
             "model.layers.0.mlp.experts.gate_up_proj",
             "model.layers.0.mlp.experts.down_proj"]
    for nm in names:
        t.p[nm.replace(".", "|")] = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    t.state_dict = lambda *a, **k: {nm: torch.zeros(1) for nm in names}  # noqa: E731
    keys = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"]
    for e in range(n_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            keys.append(f"model.layers.0.mlp.experts.{e}.{proj}.weight")
    return t, keys


def test_plan_succeeds_on_a_complete_toy():
    t, keys = _toy()
    plan = plan_moe_checkpoint(keys, t, "qwen3_moe")
    assert len(plan.passthrough) == 2
    assert plan.n_expert_stacks == 2          # gate_up_proj + down_proj
    assert set(plan.experts[0]) == {"gate", "up", "down"}


def test_unmapped_checkpoint_key_raises():
    t, keys = _toy()
    keys.append("model.layers.0.self_attn.NOT_A_THING.weight")
    with pytest.raises(MoEConventionError, match="do not map"):
        plan_moe_checkpoint(keys, t, "qwen3_moe")


def test_unclaimed_model_parameter_raises():
    """The direction that catches a weight the checkpoint never supplies."""
    t, keys = _toy()
    keys.remove("model.layers.0.self_attn.q_proj.weight")
    with pytest.raises(MoEConventionError, match="no checkpoint key supplies"):
        plan_moe_checkpoint(keys, t, "qwen3_moe")


def test_ragged_expert_stack_raises():
    t, keys = _toy(n_experts=3)
    keys.remove("model.layers.0.mlp.experts.1.up_proj.weight")
    with pytest.raises(MoEConventionError, match="ragged expert stack|missing from a"):
        plan_moe_checkpoint(keys, t, "qwen3_moe")


def test_unknown_model_type_refuses():
    t, keys = _toy()
    with pytest.raises(MoEConventionError, match="cannot be inferred from shapes"):
        plan_moe_checkpoint(keys, t, "brand_new_moe")
