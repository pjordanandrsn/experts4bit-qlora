"""Generic MoE checkpoint -> module-tree load plan.

:mod:`experts4bit_qlora.arch.moe_conventions` says how a family stores its EXPERTS.
This module turns that into a complete, validated plan for a specific model:
which checkpoint key feeds which parameter, which per-expert tensors fuse into
which stack, and — the part that matters — proof that the two sides actually
agree before a single byte is read.

The non-expert surface needs no per-model table. Upstream's converters rewrite
only the expert tensors plus a couple of documented substring renames
(``mixtral``: ``.block_sparse_moe.`` -> ``.mlp.``; ``phimoe`` adds
``.gate.weight`` -> ``.router.weight``); everything else — attention, norms,
embeddings, shared experts, routers, dense-layer MLPs — is already spelled the
way the module tree spells it. So the plan resolves those by LOOKING THEM UP in
the real tree rather than guessing from a hand-written list, and raises on any
key that does not land.

Three properties are enforced, each because its absence is a silent bug:

* **No unmapped checkpoint key.** A key that matches nothing is a weight that
  would be dropped. Raise, never warn.
* **No unclaimed model parameter.** The reverse direction is what catches a
  tensor the checkpoint never supplies — the model would keep whatever the
  skeleton was built with and compute confidently wrong numbers.
* **No partial expert stack.** Every expert index in a layer must contribute
  all three projections, or routing sends tokens into uninitialized memory.

What this module does NOT do is decide expert ORIENTATION — that is settled in
:mod:`~experts4bit_qlora.arch.moe_conventions` against upstream's own converter
spec, because gate and up are shape-identical and cannot be told apart here.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from .moe_conventions import MoEConventionError, convention_for

# Capture the prefix before `layers.N.` so a fused target inherits it. A plain
# model gives prefix "model."; a multimodal composite gives "language_model.model."
# (MiniMax-M3-VL) or similar. Non-greedy so the FIRST layers.N wins.
_LAYER = re.compile(r"^(.*?)layers\.(\d+)\.(.+)$")


@dataclass
class MoELoadPlan:
    """A validated mapping from checkpoint keys to module parameters."""

    model_type: str
    convention: str
    #: checkpoint key -> module parameter name (non-expert weights)
    passthrough: dict = field(default_factory=dict)
    #: layer index -> {"gate"|"up"|"down": {expert_index: checkpoint key}}.
    #: Keyed by LAYER, not by fused target: gate+up feed gate_up_proj while
    #: down feeds down_proj, so the "all three projections present, same expert
    #: count" invariant only holds per layer.
    experts: dict = field(default_factory=dict)
    #: layer index -> (gate_up_proj name, down_proj name)
    expert_targets: dict = field(default_factory=dict)
    #: target param -> source param, for heads a tied checkpoint omits.
    tied_params: dict = field(default_factory=dict)
    #: mapped key -> (kind, primary_ckpt_key, scale_ckpt_key, extra_ckpt_key).
    #: The mapped key is what the convention/passthrough sees; reading it means
    #: dequantizing the tuple. kind is "fp8" (block-FP8, DeepSeek-V3), "mxfp4"
    #: (gpt-oss / Kimi-K3), or "compressed_int" (llm-compressor pack-quantized,
    #: Kimi-K2.5) whose extra key is the weight_shape tensor. Named ``scales``
    #: for back-compat; extra is None for fp8/mxfp4.
    scales: dict = field(default_factory=dict)
    #: mapped-key -> load-time transform name (e.g. "transpose_last2") applied
    #: after any dequant, for pre-fused families the module stores transposed.
    transforms: dict = field(default_factory=dict)
    #: checkpoint keys deliberately not loaded (extra prediction heads).
    skipped_keys: tuple = ()
    #: model params a checkpoint legitimately never supplies (computed buffers)
    ignored_params: tuple = ()

    @property
    def n_expert_stacks(self) -> int:
        return 2 * len(self.experts)      # gate_up_proj + down_proj per layer

    def summary(self) -> str:
        n_exp_keys = sum(len(v) for st in self.experts.values() for v in st.values())
        return (f"{self.model_type} via {self.convention}: "
                f"{len(self.passthrough)} passthrough + {n_exp_keys} expert tensors "
                f"-> {self.n_expert_stacks} fused stacks")




#: Suffix upstream uses for a block-FP8 weight's companion scale tensor.
FP8_SCALE_SUFFIX = "_scale_inv"
#: MXFP4 (gpt-oss lineage) ships a matrix as two tensors: ``X_blocks`` (packed
#: fp4 nibbles) and ``X_scales`` (e8m0 block exponents). The dense parameter is
#: ``X`` — neither companion is a parameter of its own.
MXFP4_BLOCKS_SUFFIX = "_blocks"
MXFP4_SCALES_SUFFIX = "_scales"
#: compressed-tensors pack-quantized (llm-compressor / vLLM): a matrix ships as
#: ``X.weight_packed`` (int32 densely-packed low-bit values) + ``X.weight_scale``
#: (per-group scales) + ``X.weight_shape`` (the logical [out, in]). The module
#: parameter is ``X.weight`` — a name the checkpoint never spells.
CT_PACKED_SUFFIX = "weight_packed"
CT_SCALE_SUFFIX = "weight_scale"
CT_SHAPE_SUFFIX = "weight_shape"
#: NVFP4 (compressed-tensors nvfp4-pack-quantized): same weight_packed +
#: weight_scale, but E2M1 fp4 with a per-tensor global scale instead of an int
#: weight_shape. The companion that tells the two apart.
CT_GLOBAL_SCALE_SUFFIX = "weight_global_scale"
#: NVIDIA ModelOpt FP4 (nvidia/DeepSeek-R1-FP4, Llama-FP4, ...): the SAME E2M1
#: fp4 as nvfp4, but the packed bytes live under the ordinary ``weight`` name
#: with a per-group ``weight_scale`` and a per-tensor ``weight_scale_2``.
#: ``input_scale`` is an ACTIVATION scale and is not part of the weight.
MODELOPT_SCALE2_SUFFIX = "weight_scale_2"
MODELOPT_INPUT_SCALE_SUFFIX = "input_scale"
#: AWQ / GPTQ: qweight + qzeros + scales, ASYMMETRIC (zero-point) and packed
#: along the OUT axis. The module parameter is ``X.weight``, a name these
#: checkpoints never spell.
AWQ_QWEIGHT_SUFFIX = "qweight"
AWQ_QZEROS_SUFFIX = "qzeros"
AWQ_SCALES_SUFFIX = "scales"
#: GPTQ ships the SAME qweight/qzeros/scales names as AWQ but packs along a
#: different axis, in a different bit order, and adds this act-order column
#: permutation. Its presence is the name-level tell that a triple is GPTQ.
GPTQ_GIDX_SUFFIX = "g_idx"


def _split_block_scales(checkpoint_keys):
    """Partition keys into ``(weights, dequant)``.

    A quantized checkpoint ships a matrix as more than one tensor, only one of
    which corresponds to a module parameter:

    * **block-FP8** (DeepSeek-V3 and friends): ``X`` plus ``X_scale_inv``. The
      module parameter is ``X``; the scale is extra.
    * **MXFP4** (gpt-oss / Kimi-K3): ``X_blocks`` plus ``X_scales`` and NO plain
      ``X``. The module parameter is ``X`` — a name the checkpoint never spells.
    * **compressed-tensors** (Kimi-K2.5 and any llm-compressor pack-quantized
      release): ``X.weight_packed`` + ``X.weight_scale`` + ``X.weight_shape``,
      no plain ``X.weight``. The module parameter is ``X.weight``.

    Returned as ``dequant[mapped_key] = (kind, primary_key, scale_key, extra_key)``
    where ``mapped_key`` is the name that flows through the convention — the real
    weight key for FP8, a SYNTHESIZED base for MXFP4/compressed-tensors — and
    ``extra_key`` is the shape tensor for compressed-tensors, ``None`` otherwise.
    The companions are removed from ``weights`` so nothing tries to place them as
    parameters.

    Absorbing these silently would be the worst outcome: the weight keys match a
    convention perfectly, so ignoring the packing loads clean and computes
    garbage. A companion only counts when its primary is present, so a lone
    suffix cannot orphan a key.
    """
    keys = set(checkpoint_keys)
    dequant, weights = {}, []
    consumed = set()
    for k in checkpoint_keys:
        if k.endswith(CT_PACKED_SUFFIX):
            stem = k[: -len(CT_PACKED_SUFFIX)]           # "...up_proj."
            scale = stem + CT_SCALE_SUFFIX
            shape = stem + CT_SHAPE_SUFFIX
            gscale = stem + CT_GLOBAL_SCALE_SUFFIX
            base = stem + "weight"                       # "...up_proj.weight"
            # Same weight_packed + weight_scale; the third companion decides:
            # a weight_shape means int pack-quantized, a weight_global_scale
            # means NVFP4 (E2M1 with a per-tensor global scale).
            if scale in keys and shape in keys:
                dequant[base] = ("compressed_int", k, scale, shape)
                consumed.update((k, scale, shape))
                continue
            if scale in keys and gscale in keys:
                dequant[base] = ("nvfp4", k, scale, gscale)
                consumed.update((k, scale, gscale))
                continue
        if k.endswith(AWQ_QWEIGHT_SUFFIX):
            # AWQ: qweight + qzeros + scales -> the dense X.weight. Asymmetric,
            # and stored in AWQ's interleaved nibble order (see awq.py).
            stem = k[: -len(AWQ_QWEIGHT_SUFFIX)]
            qz, sc = stem + AWQ_QZEROS_SUFFIX, stem + AWQ_SCALES_SUFFIX
            if qz in keys and sc in keys:
                gidx = stem + GPTQ_GIDX_SUFFIX
                if gidx in keys:
                    # GPTQ: same names as AWQ, different bit order + g_idx.
                    # Verified bit-exact against gptqmodel's dequantize_weight,
                    # including the desc_act (permuted g_idx) case.
                    base = stem + "weight"
                    dequant[base] = ("gptq", k, sc, (qz, gidx))
                    consumed.update((k, qz, sc, gidx))
                    continue
                base = stem + "weight"
                dequant[base] = ("awq", k, sc, qz)
                consumed.update((k, qz, sc))
                continue
        if k.endswith(MODELOPT_SCALE2_SUFFIX):
            # ModelOpt FP4: X.weight (packed uint8) + X.weight_scale (per group)
            # + X.weight_scale_2 (per tensor). Verified bit-exact against
            # modelopt's own NVFP4QTensor.dequantize, so it reuses that decoder;
            # only the key spelling differs from compressed-tensors nvfp4.
            stem = k[: -len(MODELOPT_SCALE2_SUFFIX)]
            base, scale = stem + "weight", stem + CT_SCALE_SUFFIX
            if base in keys and scale in keys:
                dequant[base] = ("nvfp4", base, scale, k)
                consumed.update((k, scale))          # base stays as its own key
                # the activation scale is not a weight; drop it if present
                inp = stem + MODELOPT_INPUT_SCALE_SUFFIX
                if inp in keys:
                    consumed.add(inp)
                continue
        if k.endswith(MXFP4_SCALES_SUFFIX):
            base = k[: -len(MXFP4_SCALES_SUFFIX)]
            blocks = base + MXFP4_BLOCKS_SUFFIX
            if blocks in keys:
                dequant[base] = ("mxfp4", blocks, k, None)
                consumed.update((k, blocks))
                continue
        if k.endswith(FP8_SCALE_SUFFIX) and k[: -len(FP8_SCALE_SUFFIX)] in keys:
            base = k[: -len(FP8_SCALE_SUFFIX)]
            dequant[base] = ("fp8", base, k, None)
            consumed.add(k)          # the primary X stays in weights as itself
    for k in checkpoint_keys:
        if k in consumed:
            continue
        weights.append(k)
    # Synthesized bases (MXFP4, compressed-tensors) are not literal checkpoint
    # keys; add them so the convention has a key to map to the dense target.
    for base, (kind, _p, _s, _x) in dequant.items():
        if kind in ("mxfp4", "compressed_int", "nvfp4", "awq", "gptq"):
            weights.append(base)
    return weights, dequant



_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _extra_head_keys(checkpoint_keys, model):
    """Keys belonging to layers past the model's declared depth.

    Several MoE releases append a multi-token-prediction head as one more
    "layer" beyond ``num_hidden_layers``: DeepSeek-V3 ships layer 61 of 0..60,
    GLM-5 ships layer 78 of 0..77. The base causal-LM does not build those
    modules, so their keys have no home in the tree.

    They are returned rather than dropped. Dropping unmapped keys by default is
    exactly the behaviour this planner exists to prevent — it is indistinguishable
    from dropping keys that matter. The caller has to opt in, and the plan then
    records what was left behind so a speculative-decoding user can see that the
    draft head was skipped rather than discovering it missing at run time.
    """
    depth = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if depth is None:
        return ()
    out = []
    for k in checkpoint_keys:
        m = _LAYER_RE.search(k)
        if m and int(m.group(1)) >= depth:
            out.append(k)
    return tuple(sorted(out))


def _tied_targets(model):
    """Parameters a checkpoint may legitimately omit because the model ties them
    to another parameter, as ``{target: source}``.

    Gated on ``config.tie_word_embeddings`` being TRUE. That gate is the whole
    point: transformers exposes ``_tied_weights_keys`` on the CLASS, so it is
    present even on models whose config declares the head untied. Trusting it
    unconditionally would silently tie an untied head to the embedding — a real
    defect this project shipped once before (#37/PR#69), and one that produces a
    model that loads, runs, and is quietly wrong. Reading the config means an
    untied head stays required, so its absence still raises.
    """
    if not getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        return {}
    keys = getattr(model, "_tied_weights_keys", None) or {}
    if isinstance(keys, dict):
        return dict(keys)
    # Older transformers spells it as a bare list of tied target names; the
    # source is the input embedding by construction.
    src = "model.embed_tokens.weight"
    return {k: src for k in keys if k != src}


def plan_moe_checkpoint(
    checkpoint_keys,
    model,
    model_type: str,
    *,
    ignore_param_patterns=(r"\.rotary_emb\.", r"\.inv_freq$"),
    dense_ok: bool = False,
    skip_extra_layers: bool = False,
) -> MoELoadPlan:
    """Build and validate a load plan. Raises rather than returning a partial one.

    ``checkpoint_keys`` is the released key list (e.g. the safetensors index's
    ``weight_map`` keys). ``model`` is the built module tree — typically on
    ``meta``, which is free. Validation is against that tree's ``state_dict``
    (not ``named_parameters``): buffers such as a router's correction bias are
    real weights a checkpoint supplies, and a params-only walk cannot see one
    go missing.

    ``ignore_param_patterns`` names parameters no checkpoint ships because they
    are computed at build time (rotary ``inv_freq``). They are excluded from the
    "everything must be claimed" check — and only those.

    ``dense_ok`` admits architectures with no experts (plain Llama/Mistral/Qwen
    /Phi-style models) instead of raising on an unknown model_type. It is opt-in
    because silently treating an unrecognised MoE as dense would load its expert
    tensors as mystery passthroughs — with ``dense_ok`` the expert keys simply
    fail to resolve against the tree and the plan raises, which is the point.
    """
    conv = convention_for(model_type, dense_ok=dense_ok)
    checkpoint_keys, block_scales = _split_block_scales(checkpoint_keys)
    extra = _extra_head_keys(checkpoint_keys, model)
    if extra and skip_extra_layers:
        drop = set(extra)
        checkpoint_keys = [k for k in checkpoint_keys if k not in drop]
        block_scales = {mk: v for mk, v in block_scales.items()
                        if mk not in drop and not (set(v[1:]) & drop)}
    elif extra:
        depth = model.config.num_hidden_layers
        raise MoEConventionError(
            f"{model_type}: {len(extra)} checkpoint keys live in layers >= "
            f"{depth}, past this model's depth — e.g. {list(extra[:3])}. That is "
            f"usually a multi-token-prediction head the base model does not "
            f"build. Pass skip_extra_layers=True to load without it; they are "
            f"NOT dropped silently")
    tree = set(model.state_dict())
    ignore_re = [re.compile(p) for p in ignore_param_patterns]
    ignored = tuple(sorted(n for n in tree if any(r.search(n) for r in ignore_re)))
    claimable = tree - set(ignored)

    plan = MoELoadPlan(model_type=model_type, convention=conv.name,
                       ignored_params=ignored, scales=block_scales,
                       skipped_keys=extra if skip_extra_layers else ())
    # layer -> role -> {expert_idx: ckpt key}
    experts = defaultdict(lambda: defaultdict(dict))
    targets = {}
    unmapped = []

    for key in checkpoint_keys:
        m = _LAYER.match(key)
        if m:
            prefix, layer, suffix = m.group(1), int(m.group(2)), m.group(3)
            hit = conv.match(suffix)
            if hit is not None:
                idx, role = hit
                gu, dn = conv.fused_names(layer, prefix)
                target = dn if role == "down" else gu
                if target not in claimable:
                    unmapped.append((key, f"fused target {target} absent from the model"))
                    continue
                experts[layer][role][idx] = key
                targets[layer] = (gu, dn)
                continue
        renamed = conv.rename(key)
        if renamed in claimable:
            plan.passthrough[key] = renamed
            if conv.needs_transpose(key):
                # Pre-fused-but-transposed families (qwen3_vl_moe): the tensor
                # passes through by NAME, but its last two axes are swapped
                # relative to the module. Record it (keyed on the checkpoint key
                # the executor reads) so it transposes at load — placing it as-is
                # would mis-shape, and _assign would then reject it, which is the
                # safety net, not the plan.
                plan.transforms[key] = "transpose_last2"
            continue
        unmapped.append((key, f"no parameter {renamed!r} in the model"))

    if unmapped:
        head = "; ".join(f"{k} ({why})" for k, why in unmapped[:4])
        raise MoEConventionError(
            f"{model_type}: {len(unmapped)} checkpoint keys do not map — {head}")

    # Expert stacks must be complete and consistent, PER LAYER. The required
    # role set is the convention's own — gated families need gate+up+down, a
    # non-gated one (nemotron_h) needs exactly up+down and must NOT be failed
    # for a gate it never has.
    required_roles = set(conv.roles.values())
    for layer, roles in experts.items():
        missing_roles = required_roles - set(roles)
        if missing_roles:
            raise MoEConventionError(
                f"layer {layer}: expert stack missing {sorted(missing_roles)} entirely")
        sizes = {r: len(v) for r, v in roles.items()}
        if len(set(sizes.values())) != 1:
            raise MoEConventionError(
                f"layer {layer}: ragged expert stack {sizes} — some experts lack "
                f"a projection")
        n = next(iter(sizes.values()))
        for role, byidx in roles.items():
            gaps = sorted(set(range(n)) - set(byidx))
            if gaps:
                raise MoEConventionError(
                    f"layer {layer} {role}: expert indices {gaps[:5]} missing from "
                    f"a {n}-expert stack — routing would hit uninitialized weights")
    plan.experts = {k: {r: dict(v) for r, v in roles.items()}
                    for k, roles in experts.items()}
    plan.expert_targets = targets

    claimed = set(plan.passthrough.values())
    for gu, dn in targets.values():
        claimed.update((gu, dn))
    # A tied head is supplied by its source, not by a key of its own. Honour
    # that only when the SOURCE is itself claimed — a tie to a parameter nothing
    # loaded would propagate skeleton values, not fix them.
    # NOTE the absence of a `t not in claimed` guard. A checkpoint may declare
    # tie_word_embeddings=True and ALSO ship lm_head.weight — granite-3.0-3b
    # does, bitwise equal to the embedding. Skipping the tie there leaves two
    # separate tensors that read identically, so inference looks perfect while
    # training silently diverges: the halves take independent gradient steps on
    # a weight that is supposed to move as one. Tying regardless is also what
    # from_pretrained does, so this keeps the two loaders in agreement.
    tied = {t: srcn for t, srcn in _tied_targets(model).items()
            if t in claimable and srcn in claimed}
    plan.tied_params = tied
    claimed |= set(tied)

    unclaimed = sorted(claimable - claimed)
    if unclaimed:
        raise MoEConventionError(
            f"{model_type}: {len(unclaimed)} model parameters no checkpoint key "
            f"supplies, e.g. {unclaimed[:4]} — the model would keep skeleton "
            f"values and compute confidently wrong numbers")
    return plan
