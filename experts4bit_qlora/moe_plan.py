"""Generic MoE checkpoint -> module-tree load plan.

:mod:`experts4bit_qlora.moe_conventions` says how a family stores its EXPERTS.
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
:mod:`~experts4bit_qlora.moe_conventions` against upstream's own converter
spec, because gate and up are shape-identical and cannot be told apart here.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from .moe_conventions import MoEConventionError, convention_for

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


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
    tree = set(model.state_dict())
    ignore_re = [re.compile(p) for p in ignore_param_patterns]
    ignored = tuple(sorted(n for n in tree if any(r.search(n) for r in ignore_re)))
    claimable = tree - set(ignored)

    plan = MoELoadPlan(model_type=model_type, convention=conv.name,
                       ignored_params=ignored)
    # layer -> role -> {expert_idx: ckpt key}
    experts = defaultdict(lambda: defaultdict(dict))
    targets = {}
    unmapped = []

    for key in checkpoint_keys:
        m = _LAYER.match(key)
        if m:
            layer, suffix = int(m.group(1)), m.group(2)
            hit = conv.match(suffix)
            if hit is not None:
                idx, role = hit
                gu, dn = conv.fused_names(layer)
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
            continue
        unmapped.append((key, f"no parameter {renamed!r} in the model"))

    if unmapped:
        head = "; ".join(f"{k} ({why})" for k, why in unmapped[:4])
        raise MoEConventionError(
            f"{model_type}: {len(unmapped)} checkpoint keys do not map — {head}")

    # Expert stacks must be complete and consistent, PER LAYER.
    for layer, roles in experts.items():
        missing_roles = {"gate", "up", "down"} - set(roles)
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
    tied = {t: srcn for t, srcn in _tied_targets(model).items()
            if t in claimable and t not in claimed and srcn in claimed}
    plan.tied_params = tied
    claimed |= set(tied)

    unclaimed = sorted(claimable - claimed)
    if unclaimed:
        raise MoEConventionError(
            f"{model_type}: {len(unclaimed)} model parameters no checkpoint key "
            f"supplies, e.g. {unclaimed[:4]} — the model would keep skeleton "
            f"values and compute confidently wrong numbers")
    return plan
