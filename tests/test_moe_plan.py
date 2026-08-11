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

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.moe_conventions import MoEConventionError  # noqa: E402
from experts4bit_qlora.moe_load import execute_moe_plan  # noqa: E402
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


DENSE_FAMILIES = [
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    ("Qwen/Qwen3-8B", "qwen3"),
    ("microsoft/Phi-4", "phi3"),
]


@pytest.mark.parametrize("repo,model_type", DENSE_FAMILIES, ids=[f[1] for f in DENSE_FAMILIES])
def test_dense_open_weight_models_plan_completely(repo, model_type):
    """The non-expert path was never MoE-specific: with `dense_ok` the same
    planner serves plain open-weight causal LMs, with the same guarantees."""
    keys = _keys(repo)
    model, _ = _model(repo, 2)
    kept = [k for k in keys
            if not (m := re.match(r"^model\.layers\.(\d+)\.", k)) or int(m.group(1)) < 2]
    plan = plan_moe_checkpoint(kept, model, model_type, dense_ok=True)
    assert plan.convention == "dense"
    assert plan.n_expert_stacks == 0, "a dense model must plan no expert stacks"
    assert len(plan.passthrough) == len(kept)
    print("\n  " + plan.summary())


def test_dense_ok_still_refuses_a_real_moe():
    """dense_ok is opt-in precisely because mislabelling an MoE as dense would
    load its expert tensors as mystery passthroughs. The tree lookup catches
    it: fused stacks exist, per-expert keys do not resolve."""
    t, keys = _toy()
    with pytest.raises(MoEConventionError, match="do not map"):
        plan_moe_checkpoint(keys, t, "actually_an_moe", dense_ok=True)


# --- tied heads -------------------------------------------------------------
# A checkpoint that ties its head does not ship `lm_head.weight` at all. Before
# this was handled the planner rejected every such checkpoint, which is most
# small models. The pair of tests below pins BOTH directions, because the
# tempting one-line fix ("just excuse lm_head") reintroduces a defect this
# project already shipped once: an UNTIED head silently tied to the embedding,
# which loads, runs, and is quietly wrong.

class _TiedHead(torch.nn.Module):
    def __init__(self, tied: bool):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tied)
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}


def test_tied_head_absent_from_checkpoint_is_supplied_by_its_source():
    m = _TiedHead(tied=True)
    plan = plan_moe_checkpoint(["model.embed_tokens.weight"], m, "llama", dense_ok=True)
    assert plan.tied_params == {"lm_head.weight": "model.embed_tokens.weight"}


def test_untied_head_absent_from_checkpoint_still_raises():
    """The config, not the class attribute, decides. `_tied_weights_keys` is set
    on the class either way, so trusting it alone would tie an untied head."""
    m = _TiedHead(tied=False)
    with pytest.raises(MoEConventionError, match="no checkpoint key supplies"):
        plan_moe_checkpoint(["model.embed_tokens.weight"], m, "llama", dense_ok=True)


def test_tie_to_an_unloaded_source_is_not_a_free_pass():
    """Excusing a head because it is tied only makes sense if the SOURCE itself
    got real values; tying to a skeleton propagates skeleton values."""
    m = _TiedHead(tied=True)
    with pytest.raises(MoEConventionError, match="no checkpoint key supplies"):
        plan_moe_checkpoint([], m, "llama", dense_ok=True)


def test_executor_binds_the_same_parameter_object_not_a_copy():
    """A copied tie reads identically and then diverges under training."""
    m = _TiedHead(tied=True)
    plan = plan_moe_checkpoint(["model.embed_tokens.weight"], m, "llama", dense_ok=True)
    w = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    execute_moe_plan(plan, m, lambda k: w, device="cpu", dtype=torch.float32)
    assert m.lm_head.weight is m.model.embed_tokens.weight


def test_tied_head_that_the_checkpoint_ALSO_ships_is_still_tied():
    """The subtle half. granite-3.0-3b declares tie_word_embeddings=True AND
    ships lm_head.weight, bitwise equal to the embedding. Loading the shipped
    copy instead of tying leaves two separate tensors that read identically, so
    inference looks perfect and training silently diverges — the halves take
    independent gradient steps on a weight meant to move as one. It also wastes
    a full vocab x hidden tensor that from_pretrained shares."""
    m = _TiedHead(tied=True)
    plan = plan_moe_checkpoint(
        ["model.embed_tokens.weight", "lm_head.weight"], m, "llama", dense_ok=True)
    assert plan.tied_params == {"lm_head.weight": "model.embed_tokens.weight"}
    w = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    execute_moe_plan(plan, m, lambda k: w, device="cpu", dtype=torch.float32)
    assert m.lm_head.weight is m.model.embed_tokens.weight


def test_shipped_head_contradicting_a_tied_config_refuses_to_guess():
    """If the config says tied but the shipped head DIFFERS, the config and the
    checkpoint contradict each other. Tying discards weights the publisher
    shipped; not tying ignores the config. Both are guesses, so refuse."""
    m = _TiedHead(tied=True)
    plan = plan_moe_checkpoint(
        ["model.embed_tokens.weight", "lm_head.weight"], m, "llama", dense_ok=True)

    def read(key):
        base = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        return base + (1.0 if key == "lm_head.weight" else 0.0)

    with pytest.raises(MoEConventionError, match="refusing to guess"):
        execute_moe_plan(plan, m, read, device="cpu", dtype=torch.float32)


# --- block-FP8 checkpoints --------------------------------------------------

def test_block_fp8_scales_are_paired_and_dequantized_not_dropped():
    """DeepSeek-V3-class checkpoints ship a `*_scale_inv` companion beside every
    quantized matrix. The weight keys match a convention PERFECTLY, which is
    what makes ignoring the scales so dangerous: the model would load clean and
    compute garbage. This asserts the executor actually dequantizes."""
    from experts4bit_qlora.fp8_blocks import dequantize_fp8_blocks

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False,
                                          num_hidden_layers=1)
            self.proj = torch.nn.Linear(256, 128, bias=False)

    m = M()
    torch.manual_seed(0)
    q = (torch.randn(128, 256) * 8).to(torch.float8_e4m3fn)
    sc = torch.rand(1, 2) + 0.5                      # [tiles_out, tiles_in]
    store = {"proj.weight": q, "proj.weight_scale_inv": sc}

    plan = plan_moe_checkpoint(list(store), m, "llama", dense_ok=True)
    assert plan.scales == {"proj.weight": ("fp8", "proj.weight", "proj.weight_scale_inv")}
    # The scale key is NOT a parameter of its own.
    assert "proj.weight_scale_inv" not in plan.passthrough

    rep = execute_moe_plan(plan, m, store.__getitem__,
                           device="cpu", dtype=torch.float32)
    assert rep["fp8_dequantized"] == 1
    expected = dequantize_fp8_blocks(q, sc, dtype=torch.float32)
    assert torch.equal(m.proj.weight, expected)
    # And it is genuinely different from the raw bytes reinterpreted.
    assert not torch.equal(m.proj.weight, q.to(torch.float32))


def test_mismatched_block_scale_shape_refuses_to_dequantize():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False,
                                          num_hidden_layers=1)
            self.proj = torch.nn.Linear(256, 128, bias=False)

    m = M()
    store = {"proj.weight": torch.zeros(128, 256, dtype=torch.float8_e4m3fn),
             "proj.weight_scale_inv": torch.ones(9, 9)}   # wrong tiling
    plan = plan_moe_checkpoint(list(store), m, "llama", dense_ok=True)
    with pytest.raises(MoEConventionError, match="mismatched scale"):
        execute_moe_plan(plan, m, store.__getitem__, device="cpu",
                         dtype=torch.float32)


def test_extra_prediction_head_layers_are_never_dropped_silently():
    """DeepSeek-V3 ships an MTP head as layer 61 of 0..60; GLM-5 ships layer 78
    of 0..77. The base model builds neither. Dropping unmapped keys by default
    is the exact failure this planner exists to prevent, so it raises and names
    the opt-in instead."""
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False,
                                          num_hidden_layers=2)
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4, bias=False) for _ in range(2)])

    m = M()
    keys = ["layers.0.weight", "layers.1.weight", "layers.2.weight"]
    with pytest.raises(MoEConventionError, match="past this model's depth"):
        plan_moe_checkpoint(keys, m, "llama", dense_ok=True)

    plan = plan_moe_checkpoint(keys, m, "llama", dense_ok=True,
                               skip_extra_layers=True)
    assert plan.skipped_keys == ("layers.2.weight",)
    assert "layers.2.weight" not in plan.passthrough


def test_mxfp4_blocks_and_scales_pair_to_a_synthesized_base():
    """gpt-oss ships NO plain `experts.gate_up_proj` — only `_blocks`+`_scales`.
    The planner must synthesize the base name so the pre-fused stack maps to the
    tree target, and record the pair for the executor to dequantize. Placing the
    packed blocks as if dense would be a confidently-wrong load."""
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False,
                                          num_hidden_layers=1)
            # the dense target the two companions dequantize into
            self.gate_up_proj = torch.nn.Parameter(
                torch.zeros(2, 8, 4), requires_grad=False)

    m = M()
    keys = ["gate_up_proj_blocks", "gate_up_proj_scales"]
    plan = plan_moe_checkpoint(keys, m, "llama", dense_ok=True)
    # The base is synthesized and recorded as an mxfp4 pair; the companions are
    # NOT placed as parameters of their own.
    assert plan.scales == {
        "gate_up_proj": ("mxfp4", "gate_up_proj_blocks", "gate_up_proj_scales")}
    assert "gate_up_proj_blocks" not in plan.passthrough
    assert "gate_up_proj_scales" not in plan.passthrough
    assert plan.passthrough.get("gate_up_proj") == "gate_up_proj"


def test_a_lone_mxfp4_scales_key_without_blocks_is_not_paired():
    """A companion only counts when its primary is present — a stray `_scales`
    must stay a normal (here: unmapped) key, not silently vanish."""
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False,
                                          num_hidden_layers=1)
            self.w = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)

    m = M()
    # only a scales key, no matching _blocks
    with pytest.raises(MoEConventionError):
        plan_moe_checkpoint(["w", "orphan_scales"], m, "llama", dense_ok=True)
