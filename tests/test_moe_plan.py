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

from experts4bit_qlora.arch.moe_conventions import MoEConventionError  # noqa: E402
from experts4bit_qlora.arch.moe_load import execute_moe_plan  # noqa: E402
from experts4bit_qlora.arch.moe_plan import plan_moe_checkpoint  # noqa: E402

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
    from experts4bit_qlora.formats.fp8_blocks import dequantize_fp8_blocks

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
    assert plan.scales == {"proj.weight": ("fp8", "proj.weight", "proj.weight_scale_inv", None)}
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
        "gate_up_proj": ("mxfp4", "gate_up_proj_blocks", "gate_up_proj_scales", None)}
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


def test_prefused_transpose_places_the_transposed_stack():
    """qwen3_vl_moe ships experts pre-fused as [E, in, out]; the module declares
    [E, out, in]. The executor must transpose at load, or _assign rejects the
    mis-shaped tensor. Verified end-to-end: the placed parameter equals the
    source's last-two-axis transpose (which for a 3-D stack is Transpose(1,2),
    exactly transformers' converter op)."""
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False,
                                          num_hidden_layers=1)
            self.mlp = torch.nn.Module()
            self.mlp.experts = torch.nn.Module()
            self.mlp.experts.gate_up_proj = torch.nn.Parameter(
                torch.zeros(2, 3, 4), requires_grad=False)   # [E, out, in]

    m = M()
    src = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)  # [E, in, out]
    store = {"mlp.experts.gate_up_proj": src}
    plan = plan_moe_checkpoint(list(store), m, "qwen3_vl_moe")
    assert plan.transforms == {"mlp.experts.gate_up_proj": "transpose_last2"}
    execute_moe_plan(plan, m, store.__getitem__, device="cpu", dtype=torch.float32)
    placed = m.mlp.experts.gate_up_proj
    assert tuple(placed.shape) == (2, 3, 4)
    assert torch.equal(placed, src.transpose(-1, -2))
    assert torch.equal(placed, torch.transpose(src, 1, 2))   # == transformers op


def test_dim_theta_rotary_buffer_is_rebuilt_from_the_standard_formula():
    """Vision/audio rotary towers (Qwen3-VL) compute inv_freq from plain
    dim/theta attributes with no config or rope_type, so the config-driven
    materializer skips them. The dim/theta fallback must rebuild the exact same
    tensor the module's own __init__ would, or the first vision forward dies on
    a meta buffer."""
    from experts4bit_qlora.arch.moe_load import _materialize_computed_buffers

    class VisionRotary(torch.nn.Module):
        def __init__(self, dim=36, theta=10000.0):
            super().__init__()
            self.dim = dim
            self.theta = theta
            self.register_buffer("inv_freq", torch.empty(dim // 2, device="meta"),
                                 persistent=False)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary = VisionRotary()

    m = M()
    rebuilt = _materialize_computed_buffers(m, device="cpu")
    assert rebuilt == ["rotary.inv_freq"]
    assert not m.rotary.inv_freq.is_meta
    expected = 1.0 / (10000.0 ** (torch.arange(0, 36, 2, dtype=torch.float) / 36))
    assert torch.equal(m.rotary.inv_freq, expected)


def test_a_meta_inv_freq_with_no_way_to_rebuild_it_still_raises():
    """The fallback must not become a silent catch-all: a rotary buffer with
    neither a config initializer nor dim/theta is a real gap and must raise."""
    from experts4bit_qlora.arch.moe_load import _materialize_computed_buffers
    from experts4bit_qlora.arch.moe_conventions import MoEConventionError

    class Mystery(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.empty(4, device="meta"),
                                 persistent=False)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.r = Mystery()

    with pytest.raises(MoEConventionError, match="neither a rope initializer"):
        _materialize_computed_buffers(M(), device="cpu")


def test_vl_prefix_expert_fusion_targets_inherit_the_checkpoint_prefix():
    """A multimodal composite (MiniMax-M3-VL) carries its experts under a
    `language_model.model.` prefix. The planner's prefix-aware fused_names must
    place the fused target under the SAME prefix, not a hardcoded `model.` — or
    the target would miss the tree. Mirrors the real minimax_m3_vl key shape."""
    from experts4bit_qlora.arch.moe_conventions import MIXTRAL
    pfx = "language_model.model."
    # MIXTRAL matches the w1/w3/w2 suffix; fused_names carries the prefix.
    assert MIXTRAL.match("block_sparse_moe.experts.5.w1.weight") == (5, "gate")
    gu, dn = MIXTRAL.fused_names(3, pfx)
    assert gu == f"{pfx}layers.3.mlp.experts.gate_up_proj"
    assert dn == f"{pfx}layers.3.mlp.experts.down_proj"
    # a plain model keeps the default prefix (backward compatible)
    gu0, _ = MIXTRAL.fused_names(3)
    assert gu0 == "model.layers.3.mlp.experts.gate_up_proj"


def test_executor_stacks_match_the_quantizer_input_contract():
    """The quantized-fusion bridge (#21/#27) feeds the generic executor's fused
    stacks straight into ExpertsNbit.from_float. That only works if the shapes
    the executor produces are exactly the shapes from_float documents:
    gate_up_proj [E, 2*inter, hidden] when gated, up_proj [E, inter, hidden]
    when not, and down_proj [E, hidden, inter] either way. Pin it on CPU so a
    change to fuse_experts/stack_experts can't silently break the bridge before
    anyone runs it on a GPU. (The nf4 quantize + forward-parity half needs CUDA.)"""
    from experts4bit_qlora.arch.moe_conventions import fuse_experts, stack_experts
    E, inter, hidden = 4, 8, 16
    g = [torch.randn(inter, hidden) for _ in range(E)]
    u = [torch.randn(inter, hidden) for _ in range(E)]
    d = [torch.randn(hidden, inter) for _ in range(E)]

    # gated -> from_float(has_gate=True) wants [E, 2*inter, hidden]
    gate_up, down = fuse_experts(g, u, d)
    assert tuple(gate_up.shape) == (E, 2 * inter, hidden)
    assert tuple(down.shape) == (E, hidden, inter)

    # non-gated -> from_float(has_gate=False) wants [E, inter, hidden]
    up_proj, down2 = stack_experts(u, d)
    assert tuple(up_proj.shape) == (E, inter, hidden)
    assert tuple(down2.shape) == (E, hidden, inter)

    # ExpertsNbit.from_float must accept exactly these as its first two args.
    from experts4bit_qlora import ExpertsNbit
    import inspect
    params = list(inspect.signature(ExpertsNbit.from_float).parameters)
    assert params[:3] == ["gate_up_proj", "down_proj", "has_gate"], (
        "from_float's contract moved — the bridge feeds these positionally")


def test_axk1_keymap_drops_dense_post_mlp_norm_but_renames_moe_layers():
    """A.X-K1 stores post_mlp_layernorm at the layer level. The MoE mlp declares
    it (rename to mlp.post_mlp_layernorm); the dense mlp does not (drop it —
    transformers treats it as an unexpected key). The split is
    layer_idx >= first_k_dense_replace."""
    from experts4bit_qlora.arch.axk1 import rewrite_axk1_keys
    keys = [
        "model.layers.0.post_mlp_layernorm.weight",   # dense (fkd=1): drop
        "model.layers.1.post_mlp_layernorm.weight",   # MoE: rename
        "model.layers.1.mlp.experts.gate_up_proj",    # untouched
        "model.layers.0.input_layernorm.weight",      # untouched
    ]
    out, dropped = rewrite_axk1_keys(keys, first_k_dense_replace=1)
    assert dropped == ["model.layers.0.post_mlp_layernorm.weight"]
    assert "model.layers.1.mlp.post_mlp_layernorm.weight" in out
    assert "model.layers.1.post_mlp_layernorm.weight" not in out
    assert "model.layers.1.mlp.experts.gate_up_proj" in out
    assert "model.layers.0.input_layernorm.weight" in out


def test_axk1_is_native_prefused_and_ignores_the_unshipped_router_buffer():
    from experts4bit_qlora.arch.moe_conventions import convention_for, AXK1
    from experts4bit_qlora.arch.axk1 import AXK1_IGNORE_PARAM_PATTERNS
    assert convention_for("axk1") is AXK1
    assert not AXK1.roles                                 # native, never per-expert
    assert AXK1.match("mlp.experts.gate_up_proj") is None
    # the e_score buffer the checkpoint never ships is excluded, not "missing"
    import re
    assert any(re.search(p, "model.layers.5.mlp.gate.e_score_correction_bias")
               for p in AXK1_IGNORE_PARAM_PATTERNS)


def test_compressed_int_triple_is_paired_dequantized_and_synthesized():
    """compressed-tensors pack-quantized ships weight_packed + weight_scale +
    weight_shape and NO plain weight. The planner must synthesize the .weight
    base, pair all three, and the executor must dequantize it to the exact dense
    tensor compressed_tensors would — placing the packed int32 as if dense would
    load clean and compute garbage."""
    ct = pytest.importorskip("compressed_tensors.compressors.pack_quantized.helpers")
    from experts4bit_qlora.formats.compressed_int import dequantize_compressed_int

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False,
                                          num_hidden_layers=1)
            self.proj = torch.nn.Linear(64, 8, bias=False)   # [out=8, in=64]

    m = M()
    torch.manual_seed(0)
    q = torch.randint(-8, 8, (8, 64), dtype=torch.int8)
    packed = ct.pack_to_int32(q, 4, packed_dim=1)
    scale = (torch.rand(8, 64 // 32) * 0.1 + 0.01)          # group_size 32
    shape = torch.tensor([8, 64])
    store = {"proj.weight_packed": packed, "proj.weight_scale": scale,
             "proj.weight_shape": shape}

    plan = plan_moe_checkpoint(list(store), m, "llama", dense_ok=True)
    assert plan.scales == {"proj.weight": (
        "compressed_int", "proj.weight_packed", "proj.weight_scale",
        "proj.weight_shape")}
    assert "proj.weight_packed" not in plan.passthrough      # not placed as-is

    rep = execute_moe_plan(plan, m, store.__getitem__, device="cpu",
                           dtype=torch.float32)
    assert rep["compressed_int_dequantized"] == 1
    expected = dequantize_compressed_int(packed, scale, shape, dtype=torch.float32)
    assert torch.equal(m.proj.weight, expected)
    assert not torch.equal(m.proj.weight, packed.to(torch.float32))   # really decoded


def test_nvfp4_triple_is_distinguished_from_int4_and_dequantized():
    """NVFP4 and int pack-quantized SHARE weight_packed + weight_scale; only the
    third companion differs (weight_global_scale vs weight_shape). The planner
    must route each to the right decoder — an fp4-as-int4 mixup would be a silent
    wrong load — and the executor must match compressed_tensors' nvfp4 output."""
    nv = pytest.importorskip("compressed_tensors.compressors.nvfp4.helpers")
    from experts4bit_qlora.formats.nvfp4 import dequantize_nvfp4

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False,
                                          num_hidden_layers=1)
            self.proj = torch.nn.Linear(64, 8, bias=False)   # [out=8, in=64]

    m = M()
    torch.manual_seed(0)
    cb = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    vals = (cb[torch.randint(0, 8, (8, 64))]
            * (torch.randint(0, 2, (8, 64)) * 2 - 1)).to(torch.bfloat16)
    packed = nv.pack_fp4_to_uint8(vals)
    scale = (torch.rand(8, 64 // 16) + 0.5).to(torch.float32)     # group 16
    gscale = torch.tensor(0.03125)
    store = {"proj.weight_packed": packed, "proj.weight_scale": scale,
             "proj.weight_global_scale": gscale}

    plan = plan_moe_checkpoint(list(store), m, "llama", dense_ok=True)
    assert plan.scales["proj.weight"][0] == "nvfp4"           # NOT compressed_int
    rep = execute_moe_plan(plan, m, store.__getitem__, device="cpu",
                           dtype=torch.float32)
    assert rep["nvfp4_dequantized"] == 1
    expected = dequantize_nvfp4(packed, scale, gscale, dtype=torch.float32)
    assert torch.equal(m.proj.weight, expected)


def test_modelopt_fp4_is_recognized_and_reuses_the_nvfp4_decoder():
    """NVIDIA ModelOpt FP4 (nvidia/DeepSeek-R1-FP4, Llama-FP4) is the SAME E2M1
    as compressed-tensors nvfp4 but spells its keys differently: the packed
    bytes sit under the ordinary `weight` name, with weight_scale (per group)
    and weight_scale_2 (per tensor). input_scale is an ACTIVATION scale and must
    not be mistaken for part of the weight. Verified bit-exact against
    modelopt's own NVFP4QTensor.dequantize, so it reuses that decoder."""
    from experts4bit_qlora.formats.nvfp4 import dequantize_nvfp4

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False,
                                          num_hidden_layers=1)
            self.proj = torch.nn.Linear(128, 8, bias=False)

    m = M()
    torch.manual_seed(0)
    nib = torch.randint(0, 16, (8, 128), dtype=torch.uint8)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    scale = (torch.rand(8, 128 // 16) + 0.5)
    scale2 = torch.tensor(0.0123)
    store = {"proj.weight": packed, "proj.weight_scale": scale,
             "proj.weight_scale_2": scale2,
             "proj.input_scale": torch.tensor(1.5)}      # activation: ignored

    plan = plan_moe_checkpoint(list(store), m, "llama", dense_ok=True)
    assert plan.scales["proj.weight"] == (
        "nvfp4", "proj.weight", "proj.weight_scale", "proj.weight_scale_2")
    # the activation scale is consumed, never placed as a parameter
    assert "proj.input_scale" not in plan.passthrough

    rep = execute_moe_plan(plan, m, store.__getitem__, device="cpu",
                           dtype=torch.float32)
    assert rep["nvfp4_dequantized"] == 1
    assert torch.equal(m.proj.weight,
                       dequantize_nvfp4(packed, scale, scale2, dtype=torch.float32))


def test_awq_triple_is_recognized_and_uses_the_asymmetric_decoder():
    """AWQ is the first ASYMMETRIC format here (a zero-point per group) and its
    nibbles are interleaved [0,4,1,5,2,6,3,7] — an order no shape inspection
    reveals, so a from-scratch guess yields right-shaped scrambled weights.
    Pinned against autoawq's own constant."""
    from experts4bit_qlora.formats.awq import AWQ_REVERSE_ORDER, dequantize_awq
    assert AWQ_REVERSE_ORDER == [0, 4, 1, 5, 2, 6, 3, 7]   # verbatim autoawq

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False,
                                          num_hidden_layers=1)
            self.proj = torch.nn.Linear(256, 64, bias=False)   # [out=64, in=256]

    m = M()
    torch.manual_seed(0)
    qw = torch.randint(-(2**31), 2**31 - 1, (256, 8), dtype=torch.int32)
    qz = torch.randint(-(2**31), 2**31 - 1, (2, 8), dtype=torch.int32)
    sc = (torch.rand(2, 64) * 0.05 + 0.01)
    store = {"proj.qweight": qw, "proj.qzeros": qz, "proj.scales": sc}

    plan = plan_moe_checkpoint(list(store), m, "llama", dense_ok=True)
    assert plan.scales["proj.weight"] == (
        "awq", "proj.qweight", "proj.scales", "proj.qzeros")
    for companion in ("proj.qweight", "proj.qzeros", "proj.scales"):
        assert companion not in plan.passthrough      # never placed as-is

    rep = execute_moe_plan(plan, m, store.__getitem__, device="cpu",
                           dtype=torch.float32)
    assert rep["awq_dequantized"] == 1
    # dequant yields [out, in] — the orientation a Linear declares
    assert tuple(m.proj.weight.shape) == (64, 256)
    assert torch.equal(m.proj.weight,
                       dequantize_awq(qw, qz, sc, dtype=torch.float32))


def test_gptq_and_awq_are_told_apart_by_g_idx():
    """GPTQ ships the SAME qweight/qzeros/scales names as AWQ but packs along a
    different axis, in a different bit order, with a +1 zero offset and a g_idx
    row permutation. Decoding either as the other gives correctly-shaped
    SCRAMBLED weights that load clean — measured: the AWQ branch once claimed
    all 18624 experts of Qwen3-30B-A3B-GPTQ-Int4. The g_idx sibling is the
    name-level discriminator."""
    from experts4bit_qlora.arch.moe_plan import _split_block_scales

    _w, dq = _split_block_scales(["m.qweight", "m.qzeros", "m.scales", "m.g_idx"])
    assert dq["m.weight"] == ("gptq", "m.qweight", "m.scales",
                              ("m.qzeros", "m.g_idx"))

    # the same triple WITHOUT g_idx is AWQ
    _w2, dq2 = _split_block_scales(["m.qweight", "m.qzeros", "m.scales"])
    assert dq2["m.weight"][0] == "awq"


def test_gptq_dequant_matches_gptqmodel_including_desc_act():
    """Pinned against gptqmodel's dequantize_weight: scales[g_idx] * (w -
    (zeros+1)[g_idx]), sequential bit order, packed along IN. The desc_act case
    (a permuted g_idx) must go through the same path."""
    from experts4bit_qlora.formats.gptq import dequantize_gptq
    torch.manual_seed(0)
    BITS, IN, OUT, G = 4, 256, 64, 128
    PER, groups = 32 // BITS, IN // 128
    iw = torch.randint(0, 16, (IN, OUT), dtype=torch.int32)
    iz = torch.randint(0, 16, (groups, OUT), dtype=torch.int32)
    shifts = torch.arange(0, 32, BITS, dtype=torch.int32)
    qw = (iw.view(IN // PER, PER, OUT) << shifts.view(1, PER, 1)).sum(1).to(torch.int32)
    qz = (iz.view(groups, OUT // PER, PER) << shifts.view(1, 1, PER)).sum(-1).to(torch.int32)
    scales = torch.rand(groups, OUT) * 0.05 + 0.01

    for g_idx in (torch.arange(IN, dtype=torch.int32) // G,          # plain
                  torch.randperm(groups).repeat_interleave(G).to(torch.int32)):  # desc_act
        gi = g_idx.long()
        ref = (scales[gi] * (iw.float() - (iz + 1).float()[gi])).t()
        got = dequantize_gptq(qw, qz, scales, g_idx, bits=BITS, dtype=torch.float32)
        assert tuple(got.shape) == (OUT, IN)
        assert torch.equal(got, ref)


# --- quant-format dispatch matrix -------------------------------------------
# Seven packed formats now share one recognition path, and several share key
# NAMES (AWQ/GPTQ: qweight+qzeros+scales; compressed-int/NVFP4:
# weight_packed+weight_scale). Adding AWQ once made the planner claim every GPTQ
# tensor. These two tests pin the whole dispatch so an eighth format cannot
# quietly steal an existing one's keys.

_FORMAT_KEYS = {
    "fp8": (["m.weight", "m.weight_scale_inv"], "fp8"),
    "mxfp4": (["m_blocks", "m_scales"], "mxfp4"),
    "compressed_int": (["m.weight_packed", "m.weight_scale", "m.weight_shape"],
                       "compressed_int"),
    "nvfp4": (["m.weight_packed", "m.weight_scale", "m.weight_global_scale"],
              "nvfp4"),
    # ModelOpt FP4 is the same E2M1 decoder under different key names
    "modelopt": (["m.weight", "m.weight_scale", "m.weight_scale_2",
                  "m.input_scale"], "nvfp4"),
    "awq": (["m.qweight", "m.qzeros", "m.scales"], "awq"),
    "gptq": (["m.qweight", "m.qzeros", "m.scales", "m.g_idx"], "gptq"),
}


@pytest.mark.parametrize("fmt", sorted(_FORMAT_KEYS))
def test_each_quant_format_is_claimed_by_exactly_its_own_decoder(fmt):
    """No format may be claimed by another's branch, and every companion must be
    consumed (a leftover companion would later be placed as if it were a dense
    parameter)."""
    from experts4bit_qlora.arch.moe_plan import _split_block_scales
    keys, expected = _FORMAT_KEYS[fmt]
    weights, dq = _split_block_scales(keys)
    assert {v[0] for v in dq.values()} == {expected}, f"{fmt} misrouted"
    companions = ("weight_scale_inv", "_blocks", "_scales", "weight_packed",
                  "weight_scale", "weight_shape", "weight_global_scale",
                  "weight_scale_2", "input_scale", "qweight", "qzeros",
                  "scales", "g_idx")
    leftover = [w for w in weights if any(w.endswith(c) for c in companions)]
    assert not leftover, f"{fmt}: unconsumed companions {leftover}"


@pytest.mark.parametrize("keys", [
    ["proj.qweight", "proj.scales"],                       # AWQ missing qzeros
    ["proj.qweight", "proj.qzeros"],                       # AWQ missing scales
    ["proj.qweight", "proj.scales", "proj.g_idx"],         # GPTQ missing qzeros
    ["proj.weight_packed", "proj.weight_scale"],           # ambiguous: no 3rd
])
def test_a_partial_quant_triple_raises_instead_of_falling_through(keys):
    """A missing companion must NOT let the key fall through to a different
    decoder, and must not vanish: it stays unmapped so the planner's
    no-unmapped-key invariant raises. Half a quantized matrix is not loadable."""
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False,
                                          num_hidden_layers=1)
            self.proj = torch.nn.Linear(64, 8, bias=False)

    with pytest.raises(MoEConventionError, match="do not map"):
        plan_moe_checkpoint(keys, M(), "llama", dense_ok=True)


@pytest.mark.parametrize("g_idx, why", [
    (torch.full((64,), -1, dtype=torch.int32), "negative"),
    (torch.full((64,), 5, dtype=torch.int32), "out of range"),
    (torch.zeros(4, dtype=torch.int32), "wrong length"),
])
def test_gptq_rejects_a_g_idx_that_would_load_the_wrong_scales(g_idx, why):
    """Found by stress-testing before the 0.15.0 cut. A NEGATIVE g_idx is the
    dangerous one: Python indexing wraps -1 to the LAST group, so a corrupt or
    misparsed g_idx produced a full-shaped tensor built from the wrong scales,
    with no error at all. A wrong-length g_idx broadcast into a truncated
    weight. Both are silent-wrongness; both now raise."""
    from experts4bit_qlora.formats.gptq import dequantize_gptq
    qw = torch.zeros(8, 8, dtype=torch.int32)
    qz = torch.zeros(2, 1, dtype=torch.int32)
    sc = torch.ones(2, 8)
    with pytest.raises(ValueError):
        dequantize_gptq(qw, qz, sc, g_idx, dtype=torch.float32)

    # a valid g_idx over the same tensors still works
    good = torch.arange(64, dtype=torch.int32) // 32
    assert tuple(dequantize_gptq(qw, qz, sc, good,
                                 dtype=torch.float32).shape) == (8, 64)
