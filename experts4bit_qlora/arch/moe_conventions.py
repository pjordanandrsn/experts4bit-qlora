"""MoE storage *conventions* — the small set of on-disk shapes that the large
set of MoE models actually use.

There are far fewer MoE conventions than MoE models. transformers encodes this
directly: its ``conversion_mapping`` defines a handful of converter specs and
aliases dozens of ``model_type``s onto them. Mapping a convention therefore
lights up every model that follows it, and — more importantly — means the
error-prone part (expert fusion orientation) is adjudicated ONCE per
convention instead of once per model.

Two conventions are implemented here. Both fuse per-expert checkpoint tensors
into the stacked tensors transformers >= 5 builds, and both put the **gate
block first**; they differ only in the on-disk spelling:

All of them vary on only two axes — the CONTAINER the experts live under, and
how the projections are SPELLED:

===============  =================  ======================  ==================
convention       container          projection spelling     model_types
===============  =================  ======================  ==================
``QWEN2_MOE``    ``mlp``            ``{gate,up,down}_proj``  22
``MIXTRAL``      ``block_sparse_moe``  ``w1``/``w3``/``w2``  3
``PHIMOE``       ``block_sparse_moe``  ``w1``/``w3``/``w2``  1
``JAMBA``        ``feed_forward``   ``{gate,up,down}_proj``  1
``LFM2_MOE``     ``feed_forward``   ``w1``/``w3``/``w2``     1
===============  =================  ======================  ==================

``QWEN2_MOE`` alone covers deepseek_v2/v3/v32, every GLM MoE (glm4_moe,
glm4_moe_lite, glm4v_moe, glm_moe_dsa), olmoe, qwen3_moe, qwen3_next, dots1,
ernie4_5_moe, hunyuan_v1_moe, longcat_flash, exaone_moe, cohere2_moe,
flex_olmo, afmoe, mellum and solar_open.

``w1``=gate, ``w3``=up, ``w2``=down in every ``w``-spelled family — pinned to
upstream, never inferred.

``GRANITEMOE`` is the odd one out: it ships **already fused**, so it needs
renames only and its expert pattern deliberately matches nothing. Applying a
fusion to it would be wrong.

**Why the orientation is the whole point.** In both conventions the gate and up
tensors are SHAPE-IDENTICAL, so no amount of shape inspection can tell them
apart. Ordering them wrong yields ``act(up) * gate`` — a wrong activation with
every shape agreeing, nothing raising, and only a quality regression to show
for it. The orientation below is taken from transformers' own converter specs
(source order + ``Concatenate(dim=1)``) and cross-checked against each family's
expert ``forward``; :mod:`tests.test_moe_conventions` asserts both so an
upstream change fails loudly instead of silently swapping halves.

This module owns the EXPERT surface only. Each model's non-expert surface
(attention flavour, shared experts, router bias, dense-layer prefix, extra
heads) still needs its own map — see :mod:`experts4bit_qlora.arch.glm5` for a worked
example that layers a model's quirks on top of the QWEN2_MOE expert
convention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MoEConvention:
    """How one family stores per-expert weights on disk.

    ``expert_re`` captures (expert_index, projection_token) from the part of a
    checkpoint key that follows ``model.layers.{L}.``. ``roles`` maps the
    projection token to its role. ``fused_prefix`` is where the stacked
    tensors live in the module tree.
    """

    name: str
    expert_re: re.Pattern
    roles: dict
    fused_prefix: str
    model_types: frozenset = field(default_factory=frozenset)
    # Substring rewrites upstream applies to NON-expert keys before matching
    # them against the module tree, in order. Taken verbatim from the
    # convention's WeightRenaming entries; empty for families whose non-expert
    # spelling already matches (qwen2_moe, jamba, lfm2_moe).
    renames: tuple = ()
    # Whether experts are SwiGLU-gated (gate+up fused into gate_up_proj) or
    # plain up/down (nemotron_h: no gate, up_proj and down_proj stacked
    # separately). Non-gated conventions declare roles {up, down} only.
    gated: bool = True
    # Keys whose SUFFIX matches this get their last two axes transposed at load.
    # Some pre-fused families (qwen3_vl_moe) ship experts as [E, in, out] and
    # the module declares [E, out, in]; upstream's converter is a Transpose(1,2)
    # with nothing else. None means no key is ever transposed. The orientation
    # is adjudicated against the real checkpoint's shapes, never inferred.
    transpose_re: "re.Pattern | None" = None

    def rename(self, key: str) -> str:
        for src, dst in self.renames:
            key = key.replace(src, dst)
        return key

    def needs_transpose(self, key: str) -> bool:
        return self.transpose_re is not None and bool(self.transpose_re.search(key))

    def match(self, layer_suffix: str):
        """-> (expert_index, role) or None."""
        m = self.expert_re.match(layer_suffix)
        if not m:
            return None
        token = m.group(2)
        if token not in self.roles:
            return None
        return int(m.group(1)), self.roles[token]

    def fused_names(self, layer: int, prefix: str = "model.") -> tuple[str, str]:
        """(first_target, down_target) for a layer. ``prefix`` is everything
        before ``layers.N.`` in the checkpoint key that matched — "model." for a
        plain model, "language_model.model." for a multimodal composite — so the
        fused target lands under the SAME prefix as its per-expert sources. The
        first target is the fused gate_up_proj for gated experts, or a plain
        up_proj when the convention has no gate (nemotron_h)."""
        base = f"{prefix}layers.{layer}.{self.fused_prefix}"
        first = "gate_up_proj" if self.gated else "up_proj"
        return f"{base}.{first}", f"{base}.down_proj"


QWEN2_MOE = MoEConvention(
    name="qwen2_moe",
    expert_re=re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"),
    roles={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    fused_prefix="mlp.experts",
    model_types=frozenset({
        "qwen2_moe", "qwen3_moe", "qwen3_next", "qwen3_omni_moe",
        "qwen3_omni_moe_thinker", "olmoe", "flex_olmo", "afmoe", "cohere2_moe",
        "deepseek_v2", "deepseek_v3", "deepseek_v32", "dots1", "ernie4_5_moe",
        "exaone_moe", "glm4_moe", "glm4_moe_lite", "glm4v_moe", "glm_moe_dsa",
        "hunyuan_v1_moe", "longcat_flash", "mellum", "solar_open",
        # Newest releases (adjudicated 2026-08-11 against real indexes AND the
        # converter API): per-expert gate/up/down + block-FP8, same fusion.
        # A.X-K2 (SKT, 256 experts) and MiMo-V2-Flash (Xiaomi, 256) — both ship
        # _scale_inv companions the FP8 path already handles. kimi_k2
        # (Kimi-K2-Instruct, 384 experts) is the same shape at scale: 69120
        # per-expert keys, balanced 23040/23040/23040, + FP8. Adjudicated from
        # the released index, not the converter API (which is EMPTY for kimi_k2
        # in current transformers — a support gap, not a layout signal).
        "axk2", "mimo_v2_flash", "kimi_k2",
        # kimi_k25 (Kimi-K2.5): per-expert qwen2_moe layout under a
        # language_model.model. prefix, weights in compressed-tensors int4
        # pack-quantized — handled by the read path, not the convention.
        "kimi_k25",
    }),
)

MIXTRAL = MoEConvention(
    name="mixtral",
    # w1 = gate, w3 = up, w2 = down — pinned by transformers' converter source
    # order and MixtralExperts.forward, never inferred (w1/w3 are same-shaped).
    expert_re=re.compile(r"^block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$"),
    roles={"w1": "gate", "w3": "up", "w2": "down"},
    fused_prefix="mlp.experts",
    # minimax_m3_vl is the VL composite: same w1/w3/w2 block_sparse_moe experts,
    # but its keys carry a `language_model.model.` prefix — handled by the
    # planner's prefix-aware fused_names (validated against the 23416-key
    # MiniMax-M3 index: 57 MoE layers, complete 128-expert gate/up/down stacks).
    model_types=frozenset({"mixtral", "minimax", "minimax_m2", "minimax_m3_vl"}),
    renames=((".block_sparse_moe.", ".mlp."),),
)

PHIMOE = MoEConvention(
    name="phimoe",
    expert_re=re.compile(r"^block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$"),
    roles={"w1": "gate", "w3": "up", "w2": "down"},
    fused_prefix="mlp.experts",
    model_types=frozenset({"phimoe"}),
    renames=((".block_sparse_moe.", ".mlp."), (".gate.weight", ".router.weight")),
)

JAMBA = MoEConvention(
    name="jamba",
    expert_re=re.compile(
        r"^feed_forward\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"),
    roles={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    fused_prefix="feed_forward.experts",
    model_types=frozenset({"jamba"}),
)

LFM2_MOE = MoEConvention(
    name="lfm2_moe",
    expert_re=re.compile(r"^feed_forward\.experts\.(\d+)\.(w1|w2|w3)\.weight$"),
    roles={"w1": "gate", "w3": "up", "w2": "down"},
    fused_prefix="feed_forward.experts",
    model_types=frozenset({"lfm2_moe"}),
)

#: Dense models are the degenerate case: no experts at all. Giving them a
#: convention whose expert pattern can never match lets the same planner serve
#: every open-weight causal LM — the non-expert path (look the key up in the
#: real module tree, raise if it is not there) was never MoE-specific. Verified
#: end-to-end on Mistral-7B, Qwen3-8B and Phi-4.
DENSE = MoEConvention(
    name="dense",
    expert_re=re.compile(r"(?!)"),      # matches nothing, by construction
    roles={},
    fused_prefix="mlp.experts",         # unused; no expert key can reach it
    model_types=frozenset(),            # selected explicitly, never by lookup
)

#: GraniteMoE ships its experts **ALREADY FUSED** — the checkpoint carries
#: `input_linear` [E, 2*inter, hidden] and `output_linear` [E, hidden, inter]
#: directly, so there is nothing to gather or concatenate. It therefore needs
#: renames ONLY, and applying an expert fusion to it would be wrong. That is
#: expressible with the existing machinery: a never-matching expert pattern
#: (so no key is ever treated as per-expert) plus the rename table upstream
#: uses. Verified against the released checkpoint AND the built tree.
#: Covers granitemoe + its granitemoehybrid / granitemoeshared aliases.
GRANITEMOE = MoEConvention(
    name="granitemoe",
    expert_re=re.compile(r"(?!)"),      # matches nothing: never per-expert
    roles={},
    fused_prefix="block_sparse_moe.experts",
    model_types=frozenset({"granitemoe", "granitemoehybrid", "granitemoeshared"}),
    renames=(
        ("block_sparse_moe.input_linear.weight", "block_sparse_moe.experts.gate_up_proj"),
        ("block_sparse_moe.output_linear.weight", "block_sparse_moe.experts.down_proj"),
        ("block_sparse_moe.router.layer.weight", "block_sparse_moe.router.weight"),
    ),
)

#: gpt-oss ships its experts PRE-FUSED (``experts.gate_up_proj`` is one stacked
#: [E, hidden, 2*inter] tensor, not per-expert) AND MXFP4-quantized on disk as
#: ``experts.gate_up_proj_blocks`` + ``experts.gate_up_proj_scales``. The planner
#: pairs those and dequantizes in the read path, so by the time the convention
#: runs each dense stack is a single synthesized ``experts.gate_up_proj`` key
#: that passes straight through to the tree. Like GraniteMoE, nothing is ever
#: per-expert here, so the expert pattern matches nothing; the expert/router
#: biases pass through unrenamed. The clamped-SwiGLU activation is the model's
#: OWN forward — a loading convention only places weights.
GPTOSS = MoEConvention(
    name="gptoss",
    expert_re=re.compile(r"(?!)"),      # matches nothing: never per-expert
    roles={},
    fused_prefix="mlp.experts",
    model_types=frozenset({"gpt_oss"}),
    renames=(),
)

#: qwen3_vl_moe (and its qwen3_vl_moe_text tower) ship experts PRE-FUSED as a
#: single stacked tensor, but in [E, in, out] where the module declares
#: [E, out, in]. Upstream's converter for it is exactly a Transpose(1, 2) on
#: ``experts.gate_up_proj`` and ``experts.down_proj`` with no other operation —
#: NOT qwen2_moe's per-expert MergeModulelist. Adjudicated against the released
#: Qwen3-VL-30B-A3B index: checkpoint gate_up_proj [128, 2048, 1536] and
#: down_proj [128, 768, 2048]; the built tree wants [128, 1536, 2048] and
#: [128, 2048, 768] respectively — the last two axes swapped, both projections.
#: Never per-expert, so the expert pattern matches nothing.
QWEN3_VL_MOE = MoEConvention(
    name="qwen3_vl_moe",
    expert_re=re.compile(r"(?!)"),      # matches nothing: never per-expert
    roles={},
    fused_prefix="mlp.experts",
    model_types=frozenset({"qwen3_vl_moe", "qwen3_vl_moe_text"}),
    renames=(),
    transpose_re=re.compile(r"mlp\.experts\.(gate_up_proj|down_proj)$"),
)

#: JetMoE is a DUAL MoE — mixture-of-experts in BOTH the MLP (``mlp.input_linear``
#: / ``output_linear`` / ``router.layer``) AND the attention block
#: (``self_attention.experts.*``). The obvious worry is that an MLP-expert
#: planner fuses the MLP experts and silently drops the attention ones. It does
#: not, because JetMoE's converter is EMPTY/native: every expert stack ships
#: PRE-FUSED with the exact name the module declares, so there is nothing to
#: fuse or reorder — both MoEs pass straight through, and placing a native
#: tensor makes no orientation choice to get wrong. Adjudicated against the
#: released jetmoe-8b index: all 266 checkpoint keys map 1:1 (96 of them
#: attention-expert tensors), and the only tree param the checkpoint omits is
#: the tied ``lm_head`` — 267/267 covered. Never per-expert; no renames.
JETMOE = MoEConvention(
    name="jetmoe",
    expert_re=re.compile(r"(?!)"),      # matches nothing: native, never per-expert
    roles={},
    fused_prefix="mlp.experts",
    model_types=frozenset({"jetmoe"}),
    renames=(),
)

#: DBRX stores each expert projection as ONE flat ``[E*ffn_hidden, hidden]``
#: tensor — ``ffn.experts.mlp.w1`` / ``v1`` / ``w2`` — and the transformers
#: ``DbrxExpertGLU`` module declares them the SAME flat way, so the checkpoint
#: matches the tree natively (empty converter). Loading is pure passthrough;
#: nothing is gathered or reshaped. Adjudicated against katuni4ka/tiny-random-dbrx:
#: 19/19 keys map 1:1, ckpt == tree exactly, logits bit-identical to
#: from_pretrained. (A community re-layout — v2ray/dbrx-base-fixed — splits the
#: flat stacks into per-expert ``ffn.experts.mlp_experts.N.w1.weight``; that is
#: NOT the canonical transformers format and is out of scope for this passthrough.)
#:
#: The roles are pinned from ``DbrxExpertGLU.forward`` for the eventual
#: quantized-fusion path (which would reshape the flat stacks into e4b's
#: [E, 2*inter, hidden]): ``gate = x @ w1.T``, ``up = x @ v1.T``,
#: ``down = (act(gate) * up) @ w2`` — so **w1 = gate, v1 = up, w2 = down**,
#: SwiGLU gate-first. They are NOT used by the passthrough loader (which places
#: the flat tensors as-is) but are recorded here so the orientation is never
#: re-guessed. The expert pattern matches nothing: dbrx is never per-expert.
DBRX = MoEConvention(
    name="dbrx",
    expert_re=re.compile(r"(?!)"),      # matches nothing: flat native stacks
    roles={},                           # passthrough; w1=gate/v1=up/w2=down documented above
    fused_prefix="ffn.experts.mlp",
    model_types=frozenset({"dbrx"}),
    renames=(),
)

#: qwen3_5_moe ships experts PRE-FUSED with an EMPTY/native converter — unlike
#: its sibling qwen3_vl_moe, there is no Transpose, so the stacked
#: ``mlp.experts.gate_up_proj`` [E, 2*inter, hidden] and ``down_proj``
#: [E, hidden, inter] match the module tree as-is. Pure passthrough, never
#: per-expert; the per-layer shared_expert passes through too. Adjudicated from
#: the converter API (empty) + the built tree; awaiting a canonical released
#: checkpoint to confirm end-to-end (none published in standard form yet), but
#: native placement makes no orientation choice, so there is nothing to get
#: wrong once the format is confirmed native.
QWEN3_5_MOE = MoEConvention(
    name="qwen3_5_moe",
    expert_re=re.compile(r"(?!)"),      # matches nothing: pre-fused native
    roles={},
    fused_prefix="mlp.experts",
    model_types=frozenset({"qwen3_5_moe"}),
    renames=(),
)

#: Nemotron-H is a hybrid Mamba/attention model whose MoE lives in a ``mixer``
#: block and — unlike every gated family here — has NO gate: each expert is a
#: plain up_proj + down_proj, run as down(act(up(x))). transformers stacks the
#: per-expert up/down into ``mixer.experts.up_proj`` [E, inter, hidden] and
#: ``mixer.experts.down_proj`` [E, hidden, inter] with MergeModulelist and no
#: Concatenate. Declared non-gated so the executor stacks up_proj on its own
#: instead of fusing a (nonexistent) gate. The per-layer shared_expert passes
#: through. Adjudicated against the converter (up_proj/down_proj MergeModulelist,
#: no gate token) and the built tree.
NEMOTRON_H = MoEConvention(
    name="nemotron_h",
    expert_re=re.compile(r"^mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$"),
    roles={"up_proj": "up", "down_proj": "down"},
    fused_prefix="mixer.experts",
    model_types=frozenset({"nemotron_h"}),
    gated=False,
)

#: A.X-K1 (SKT): a DeepSeek-V3 MoE whose released checkpoint ships experts
#: PRE-FUSED (mlp.experts.gate_up_proj [E, 2*inter, hidden] matching the tree,
#: no transpose), so the expert surface is plain native passthrough. Its two
#: non-expert quirks — a layer-conditional post_mlp_layernorm and an unshipped
#: e_score_correction_bias buffer — need per-layer knowledge and live in the
#: dedicated keymap :mod:`experts4bit_qlora.arch.axk1`. Never per-expert.
AXK1 = MoEConvention(
    name="axk1",
    expert_re=re.compile(r"(?!)"),
    roles={},
    fused_prefix="mlp.experts",
    model_types=frozenset({"axk1"}),
    renames=(),
)

CONVENTIONS = (QWEN2_MOE, MIXTRAL, PHIMOE, JAMBA, LFM2_MOE, GRANITEMOE, GPTOSS,
               QWEN3_VL_MOE, JETMOE, DBRX, QWEN3_5_MOE, NEMOTRON_H, AXK1)
_BY_MODEL_TYPE = {mt: c for c in CONVENTIONS for mt in c.model_types}


class MoEConventionError(ValueError):
    """An expert stack could not be built. Never downgrade to a warning: a
    partial or mis-ordered stack computes wrong numbers without raising."""


def convention_for(model_type: str, *, dense_ok: bool = False) -> MoEConvention:
    """The storage convention for a model_type, or raise.

    Raising on an unknown type is deliberate. Guessing a convention is exactly
    the failure this module exists to prevent — a wrong guess still loads.
    """
    try:
        return _BY_MODEL_TYPE[model_type]
    except KeyError:
        if dense_ok:
            # The caller has asserted this architecture has no experts. The
            # planner still validates every key against the real tree, so a
            # model that DOES have experts fails loudly on its expert keys
            # rather than loading them as mystery passthroughs.
            return DENSE
        raise MoEConventionError(
            f"no adjudicated MoE convention for model_type {model_type!r}. Add "
            f"one only after reading its converter spec in transformers' "
            f"conversion_mapping AND its expert forward — the gate/up "
            f"orientation cannot be inferred from shapes."
        ) from None


def fuse_experts(gate: list, up: list, down: list):
    """Per-expert lists -> ``(gate_up_proj [E, 2*inter, hidden],
    down_proj [E, hidden, inter])``.

    ``gate[e]``/``up[e]`` are ``[inter, hidden]``; ``down[e]`` is
    ``[hidden, inter]``. Equivalent to transformers'
    ``MergeModulelist(dim=0)`` + ``Concatenate(dim=1)``: stack each projection
    across experts, then join gate and up along the intermediate axis with the
    **gate first**. Missing experts raise rather than silently shrinking the
    stack — a short stack would route some tokens to the wrong expert.
    """
    import torch

    n = len(gate)
    if n == 0 or not (len(up) == len(down) == n):
        raise MoEConventionError(
            f"expert count mismatch: gate={len(gate)} up={len(up)} down={len(down)}")
    missing = [i for i, t in enumerate(gate) if t is None]
    missing += [i for i, t in enumerate(up) if t is None]
    missing += [i for i, t in enumerate(down) if t is None]
    if missing:
        raise MoEConventionError(
            f"missing expert tensors at indices {sorted(set(missing))[:5]} — "
            f"refusing to build a partial expert stack")
    gate_up = torch.stack([torch.cat([gate[e], up[e]], dim=0) for e in range(n)])
    return gate_up, torch.stack(list(down))


def stack_experts(up: list, down: list):
    """Per-expert lists -> ``(up_proj [E, inter, hidden], down_proj
    [E, hidden, inter])`` for NON-gated MoEs (nemotron_h). Same MergeModulelist
    stacking as :func:`fuse_experts` but with no gate to concatenate — the
    module runs a plain ``down(act(up(x)))`` with no SwiGLU gate. Missing
    experts raise rather than shrinking the stack."""
    import torch

    n = len(up)
    if n == 0 or len(down) != n:
        raise MoEConventionError(
            f"expert count mismatch: up={len(up)} down={len(down)}")
    missing = [i for i, t in enumerate(up) if t is None]
    missing += [i for i, t in enumerate(down) if t is None]
    if missing:
        raise MoEConventionError(
            f"missing expert tensors at indices {sorted(set(missing))[:5]} — "
            f"refusing to build a partial expert stack")
    return torch.stack(list(up)), torch.stack(list(down))
