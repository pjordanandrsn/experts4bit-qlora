"""What a MoE checkpoint IS, from its config alone: :func:`describe_moe` -> :class:`MoETopology`.

Planning a load -- will it fit, which layers carry experts, how big is one expert, how many adapters will
train -- used to mean loading it, or re-deriving the answer from config attributes whose names differ per
family (the routed top-k alone has seven spellings, and two call sites here carried different lists of them).

This module answers from the same two sources the loader itself trusts, and reads no weight:

* the **config** (``AutoConfig``; one small file), for the scalars and for :func:`admission_refusal`;
* the **module tree** transformers builds from that config on ``meta`` (no allocation), for structure: which
  decoder layers have an expert stack at the loader's expert path, each stack's shape, every non-expert
  parameter, the attention projections the LoRA wrapper and the 4-bit attention store will touch.

Structure therefore comes from upstream's own modeling code, not from a per-family table here: a family the
loader admits is described by the same walk. Family knowledge stays where it already lives -- the expert path
and gating in :mod:`~experts4bit_qlora.arch.moe_conventions` (via ``expert_layout_for``), the admission gates
in the loader.

Needs the ``[train]`` extra (transformers, accelerate). Never touches a GPU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

#: Every spelling of "routed experts per token" a supported config uses, in lookup order. The union of the two
#: lists serve_paged and the int4 lane carried separately (each lacked one the other had).
ROUTED_TOP_K_KEYS = ("num_experts_per_tok", "num_experts_per_token", "moe_top_k", "moe_topk",
                     "n_routed_experts_per_tok", "top_k_experts", "top_k")


def routed_top_k(config):
    """Routed experts per token, from ``config`` or its ``text_config``; ``None`` if no known key holds one."""
    for c in (config, getattr(config, "text_config", None)):
        if c is None:
            continue
        for key in ROUTED_TOP_K_KEYS:
            v = getattr(c, key, None)
            if isinstance(v, int) and v > 0:
                return v
    return None


@dataclass(frozen=True)
class ExpertStack:
    """One decoder layer's fused expert stack, as the module tree declares it."""

    layer: int
    n_experts: int
    #: ``gate_up_proj`` for gated experts, ``up_proj`` for non-gated (nemotron_h)
    first_name: str
    first_shape: tuple
    down_shape: tuple
    #: per-expert input width of the first projection (the model's hidden size) and of ``down_proj``
    hidden: int
    intermediate: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.first_shape:
            n *= d
        m = 1
        for d in self.down_shape:
            m *= d
        return n + m


@dataclass(frozen=True)
class AttentionProjections:
    """The q/k/v/o projections :func:`~experts4bit_qlora.lora.detect_attention_projections` finds, summarized."""

    count: int
    layers: int
    #: sum of ``in_features + out_features``: a rank-``r`` LoRA over all of them has ``r`` times this many params
    in_plus_out: int
    numel: int
    any_bias: bool
    #: K and V output width summed over attention layers (V counted as K where a layer has no ``v_proj``): the KV
    #: cache holds this many elements per token. Sliding windows and latent (MLA) caches are NOT modelled here.
    kv_elements_per_token: int
    #: q/k/v weight elements in the attention classes the fused training projection takes (``engines.train_qkv_fuse``:
    #: Qwen3-MoE attention); the estimate prices that projection's fp32 absmax over them
    fused_qkv_numel: int = 0


@dataclass(frozen=True)
class MoETopology:
    """A MoE checkpoint's structure, without its weights. Every field is data; :meth:`to_dict` is JSON-ready."""

    model: str
    model_type: str
    #: the config's ``architectures[0]``, or ``None``
    architecture: str | None
    #: the hub commit the config resolved to (``None`` for a local directory or a config object)
    revision: str | None
    #: why the loader would refuse this config (:func:`admission_refusal`), or ``None`` if it is admitted
    loader_refusal: str | None
    convention: str | None
    gated: bool | None
    n_layers: int
    hidden_size: int
    vocab_size: int
    tied_embeddings: bool
    top_k: int | None
    expert_stacks: tuple = ()
    #: non-expert parameters (embeddings, attention, norms, routers, dense MLPs, shared experts), tied counted once
    dense_numel: int = 0
    embedding_numel: int = 0
    #: ``0`` when tied to the input embedding
    lm_head_numel: int = 0
    attention: AttentionProjections | None = None
    #: the paged KV pool's geometry, by the paged server's own rules (``serve_paged._kv_geometry`` and
    #: ``paged_runner.kv_layers`` over ``decoder_layers``, as ``serve_paged.build_engine`` calls them): KV heads and
    #: head dim (scalars, or per-layer lists for per-layer configs) and the number of pool layers. ``None`` if undescribable.
    kv_heads: object = None
    kv_head_dims: object = None
    kv_layers: int | None = None
    #: per-expert tensors the module carries that the generic adapter's epilogue does not (``*bias*``); the
    #: adapter refuses such a stack (:func:`~experts4bit_qlora.lora.assert_stock_epilogue`)
    expert_bias_tensors: tuple = ()
    #: ``(out_features, in_features)`` of every projection the serving int4-attention swap would store on the int4-b32
    #: grid (``engines.int4_attn.attention_linears``, the swap's own rule), in module order
    int4_attention_linears: tuple = ()
    #: why the paged server refuses this model's state-carrying layers (``engines.paged_runner.paged_state_refusal`` on
    #: the meta tree), or ``None``
    paged_state_refusal: str | None = None
    #: ``(layer, conv_dim, conv_kernel, v_heads, head_k_dim, head_v_dim)`` per linear-attention layer the paged server's
    #: per-slot state pool drives (``engines.linear_state.state_geometry`` on the meta tree); empty for a non-hybrid model
    linear_state_layers: tuple = ()
    #: where each fact came from
    provenance: dict = field(default_factory=dict)

    @property
    def moe_layers(self) -> tuple:
        return tuple(s.layer for s in self.expert_stacks)

    @property
    def n_experts(self) -> int:
        return max((s.n_experts for s in self.expert_stacks), default=0)

    @property
    def expert_numel(self) -> int:
        return sum(s.numel for s in self.expert_stacks)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["moe_layers"] = list(self.moe_layers)
        d["n_experts"] = self.n_experts
        d["expert_numel"] = self.expert_numel
        return d

    def summary(self) -> str:
        if self.loader_refusal:
            return f"{self.model} ({self.model_type}): refused by the loader"
        s0 = self.expert_stacks[0] if self.expert_stacks else None
        shape = f"E={s0.n_experts} H={s0.hidden} I={s0.intermediate}" if s0 else "no expert stacks"
        return (f"{self.model} ({self.model_type}, {self.convention}): {len(self.expert_stacks)}/{self.n_layers} "
                f"MoE layers, {shape}, top-{self.top_k}; {self.expert_numel / 1e9:.2f}B expert + "
                f"{self.dense_numel / 1e9:.2f}B dense params")


def _shape(t) -> tuple:
    return tuple(int(d) for d in t.shape)


def describe_moe(model, *, revision=None, trust_remote_code=False) -> MoETopology:
    """Describe ``model`` -- a hub id, a local snapshot directory or a ``PretrainedConfig`` -- without its weights.

    Fetches at most ``config.json`` (and the remote modeling code when ``trust_remote_code``, which stays the
    caller's decision as in the loader). Builds the text tower on ``meta`` exactly as
    :func:`~experts4bit_qlora.loader.load_moe_4bit_streaming` does, then walks it. A config the loader refuses is
    still described as far as its config allows, with ``loader_refusal`` set and no expert stacks: refusal is an
    answer, not an exception.
    """
    import torch
    import transformers
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from ..loader import admission_refusal, expert_layout_for
    from ..lora import describe_epilogue_structure, detect_attention_projections
    from .moe_conventions import MoEConventionError, convention_for

    if isinstance(model, transformers.PretrainedConfig):
        config, name = model, getattr(model, "_name_or_path", None) or model.model_type
    else:
        config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code, revision=revision)
        name = str(model)
    model_type = getattr(config, "model_type", None)
    lm_config = getattr(config, "text_config", None) or config
    archs = getattr(config, "architectures", None) or [None]
    prov = {
        "config": name,
        "transformers": transformers.__version__,
        "structure": "module tree built on meta from the config (no weights read)",
        "top_k": "config: first of ROUTED_TOP_K_KEYS present",
    }
    common = dict(
        model=name, model_type=model_type, architecture=archs[0],
        revision=getattr(config, "_commit_hash", None),
        n_layers=int(getattr(lm_config, "num_hidden_layers", 0) or 0),
        hidden_size=int(getattr(lm_config, "hidden_size", 0) or 0),
        vocab_size=int(getattr(lm_config, "vocab_size", 0) or 0),
        tied_embeddings=bool(getattr(lm_config, "tie_word_embeddings", False)),
        top_k=routed_top_k(config),
    )
    refusal = admission_refusal(config)
    if refusal is not None:
        return MoETopology(loader_refusal=refusal, convention=None, gated=None, provenance=prov, **common)
    expert_rel, has_gate = expert_layout_for(model_type)
    try:
        conv = convention_for(model_type).name
    except MoEConventionError:
        conv = None

    with init_empty_weights():
        tree = AutoModelForCausalLM.from_config(lm_config, dtype=torch.bfloat16, trust_remote_code=trust_remote_code)

    stacks, expert_param_ids, bias_tensors = [], set(), set()
    first_name = "gate_up_proj" if has_gate else "up_proj"
    for i in range(common["n_layers"]):
        try:
            mod = tree.get_submodule(f"model.layers.{i}.{expert_rel}")
        except AttributeError:
            continue                                   # a dense layer: no expert stack at the loader's path
        first, down = getattr(mod, first_name, None), getattr(mod, "down_proj", None)
        if not (isinstance(first, torch.Tensor) and isinstance(down, torch.Tensor) and first.dim() == 3):
            continue
        E = int(down.shape[0])
        H = common["hidden_size"]
        inter = down.numel() // (E * H) if E and H else 0
        stacks.append(ExpertStack(layer=i, n_experts=E, first_name=first_name, first_shape=_shape(first),
                                  down_shape=_shape(down), hidden=H, intermediate=int(inter)))
        expert_param_ids.update(id(p) for p in mod.parameters())
        bias_tensors.update(describe_epilogue_structure(mod)["bias_tensors"])

    dense = sum(p.numel() for p in tree.parameters() if id(p) not in expert_param_ids)
    emb = tree.get_input_embeddings()
    head = tree.get_output_embeddings()
    # The loader ties the head to the embedding when the config says so and the checkpoint ships no
    # ``lm_head.weight`` (load_moe_4bit_streaming, "Tie lm_head"). A tree built on meta may not be tied yet, so
    # the config decides here; a tied checkpoint that nevertheless ships a head would load it untied.
    tied = bool(getattr(lm_config, "tie_word_embeddings", True)) and head is not None and emb is not None
    if tied and head.weight is not emb.weight:
        dense -= head.weight.numel()
    prov["tied_embeddings"] = "config tie_word_embeddings (the loader's rule when no lm_head.weight ships)"
    try:
        census = detect_attention_projections(tree, exact_linear=False)
    except SystemExit as e:                            # the detector refuses cross-layer KV reuse by exiting
        census = None
        prov["attention"] = f"not described: {e}"
    from ..engines.train_qkv_fuse import FUSED_ATTENTION_CLASSES   # the classes the fused training projection takes (P129)
    attn_mods, in_out, numel, bias, kv, fqkv = set(), 0, 0, False, 0, 0
    for mod, pname in (census.candidates if census else ()):
        lin = getattr(mod, pname)
        attn_mods.add(id(mod))
        in_out += lin.in_features + lin.out_features
        numel += lin.weight.numel()
        if pname in ("q_proj", "k_proj", "v_proj") and type(mod).__name__ in FUSED_ATTENTION_CLASSES:
            fqkv += lin.weight.numel()
        bias = bias or lin.bias is not None
        if pname == "k_proj":
            kv += lin.out_features * (1 if isinstance(getattr(mod, "v_proj", None), torch.nn.Linear) else 2)
        elif pname == "v_proj":
            kv += lin.out_features
    common["tied_embeddings"] = tied
    from ..engines.int4_attn import attention_linears
    int4_attn = tuple((int(lin.out_features), int(lin.in_features)) for _m, _n, lin in attention_linears(tree))
    prov["int4_attention_linears"] = "engines.int4_attn.attention_linears (the serving swap's rule) on the meta tree"
    kv_geo = {}
    try:
        from ..engines.paged_runner import decoder_layers, kv_layers
        from ..serve_paged import _kv_geometry

        heads, dims = _kv_geometry(config)                 # it reads text_config first, as build_engine does
        kv_geo = {"kv_heads": tuple(heads) if isinstance(heads, list) else int(heads),
                  "kv_head_dims": tuple(dims) if isinstance(dims, list) else int(dims),
                  "kv_layers": int(kv_layers(tree, decoder_layers(tree.config)))}
        prov["kv"] = "serve_paged._kv_geometry + paged_runner.kv_layers over decoder_layers (the paged server's rules)"
    except Exception as e:  # noqa: BLE001 - an undescribable KV geometry is an answer, recorded
        prov["kv"] = f"not described: {type(e).__name__}: {e}"[:300]
    from ..engines.linear_state import state_geometry
    from ..engines.paged_runner import paged_state_refusal
    kv_geo["paged_state_refusal"] = paged_state_refusal(tree)
    kv_geo["linear_state_layers"] = tuple(state_geometry(tree))
    prov["paged_state_refusal"] = "engines.paged_runner.paged_state_refusal (the runner's own rules) on the meta tree"
    return MoETopology(
        **kv_geo, loader_refusal=None, convention=conv, gated=has_gate, expert_stacks=tuple(stacks), dense_numel=int(dense),
        embedding_numel=int(emb.weight.numel()) if emb is not None else 0,
        lm_head_numel=0 if tied or head is None else int(head.weight.numel()),
        attention=None if census is None else AttentionProjections(count=census.expected_count, layers=len(attn_mods), in_plus_out=int(in_out),
                                       numel=int(numel), any_bias=bias, kv_elements_per_token=int(kv),
                                       fused_qkv_numel=int(fqkv)),
        expert_bias_tensors=tuple(sorted(bias_tensors)), int4_attention_linears=int4_attn, provenance=prov, **common)
