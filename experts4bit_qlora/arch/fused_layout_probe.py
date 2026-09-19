"""Derive a family's fused gate/up layout by EXECUTING upstream's expert forward.

e4b#515. A fused ``gate_up_proj`` packs two shape-identical projections into one
tensor. Nothing in the tensor says which is which, so for a natively pre-fused
family -- ``roles={}``, unmatchable ``expert_re``, no per-expert key names left
to recover the order from -- the order has always been a human adjudication
carried in prose, and since #518 in a field. Getting it wrong is silent:
``up * act(gate)`` computes a wrong activation with every shape agreeing and
every structural gate still passing.

**What this module establishes, and what it cannot.** The bytes in a checkpoint
carry no label. What makes one stripe "the gate" is the arithmetic that consumes
it, so for a checkpoint released against transformers the ground truth for the
order IS upstream's expert ``forward``. This module reads that ground truth by
running the real upstream module. It therefore answers "does e4b split the way
the model author's own code splits", which is the whole correctness question.
It does NOT, and no procedure can, answer "were these particular released bytes
trained in the order upstream's code assumes" -- an unlabeled tensor cannot
testify about its own training, and a family that shipped weights contradicting
its own modeling code would be an upstream bug, not something a loader can see.

**Why the test is activation-agnostic.** In every gated expert the output is
AFFINE in ``up`` and NONLINEAR in ``gate``::

    granitemoe / gemma4 / qwen*  act(gate) * up          linear in up
    gpt_oss                      (up + 1) * glu(gate)    affine in up

So scale one index set of the fused axis by ``t`` and take the second difference
``y(0) - 2*y(1) + y(2)``. It vanishes exactly for the ``up`` set and does not for
the ``gate`` set. The probe never needs to know whether the activation is silu,
gelu_pytorch_tanh or a clamped sigmoid-GLU, so it does not go stale when a family
picks a different one -- which a regex over upstream source text does.

**Why it tests four index sets, not two.** The layout space is not two-valued.
gpt-oss packs gate and up INTERLEAVED (``gate_up[..., ::2]`` / ``[..., 1::2]``),
not as contiguous halves, so a probe that only asked "first half or second half"
would have to answer one of them and would be wrong either way. Asking all four
lets the probe return ``interleaved`` -- or refuse, when no set is affine.

Pure CPU, float64, seeded, no checkpoint and no GPU: the upstream module is built
at toy size (2 experts, hidden 6, intermediate 4) with random weights, because the
layout is a property of the CODE, not of any particular weights.
"""
from __future__ import annotations

import importlib
import types
from dataclasses import dataclass

import torch

#: Toy dimensions. Small enough to be free, big enough that the four index sets
#: over the fused axis are all distinct (intermediate 4 -> width 8).
_E, _I, _H, _T = 2, 4, 6, 5
#: Seeded so this can never be flaky (#341's lesson).
_SEED = 20260919
#: Weight scale. Deliberately small so the clamps in gpt-oss's and DeepSeek-V4's
#: gates (limit 7.0) never engage: inside the clamp ``up`` is affine, at the clamp
#: it is not, and a probe run in the clamped regime would find NO affine set and
#: refuse. Refusing would be safe but useless, so stay well clear.
_SCALE = 0.15

#: How much smaller the winning residual must be than the runner-up before the
#: verdict is believed. The true ``up`` set cancels to rounding (~1e-16 relative);
#: every other set sits at ~1e-1. Twelve orders of magnitude is a chasm, so this
#: margin refuses long before it could mislead.
_MARGIN = 1e6
#: Absolute ceiling on the winning residual, relative to the output scale.
_AFFINE_TOL = 1e-10


class FusedLayoutUndetermined(RuntimeError):
    """The probe could not name the layout, so it refuses to guess one.

    Raised when no index set is affine (an unexpected gate shape, or a clamp
    engaged), when more than one is (a degenerate fixture -- a linear activation
    makes EVERY set affine and the probe would otherwise pick arbitrarily), or
    when upstream exposes no probeable expert module for the family. A guess here
    would be indistinguishable from an answer, which is the exact failure mode
    #515 exists to remove.
    """


@dataclass(frozen=True)
class FusedLayout:
    """Where the gate lives in a fused ``gate_up_proj``, as measured.

    ``packing`` is ``"contiguous"`` (two blocks) or ``"interleaved"`` (alternating
    stripes). ``order`` is ``("gate", "up")`` when the gate occupies the FIRST
    block / the even stripes. ``residuals`` is the full evidence, kept so a
    failure message can show its work, and ``margin`` is how decisive it was.
    """

    packing: str
    order: tuple
    residuals: dict
    margin: float

    @property
    def gate_first(self) -> bool:
        return tuple(self.order) == ("gate", "up")

    def __str__(self) -> str:
        return f"{self.packing}/{'gate-first' if self.gate_first else 'up-first'}"


def _index_sets(n: int) -> dict:
    """The four ways two equal-width projections can share one axis of width n.

    Keyed by the (packing, order) the set implies IF it turns out to be the
    affine -- that is, the ``up`` -- one.
    """
    h = n // 2
    return {
        ("contiguous", ("up", "gate")): torch.arange(0, h),      # up occupies [0:I]
        ("contiguous", ("gate", "up")): torch.arange(h, n),      # up occupies [I:2I]
        ("interleaved", ("up", "gate")): torch.arange(0, n, 2),  # up on even stripes
        ("interleaved", ("gate", "up")): torch.arange(1, n, 2),  # up on odd stripes
    }


def _toy_config(**overrides) -> types.SimpleNamespace:
    """The attributes upstream expert ``__init__``s read, at toy size.

    A namespace rather than a real config class on purpose: the real classes
    validate cross-field invariants a 6-wide model does not satisfy, and the
    expert module only ever reads these names. The module and its forward are the
    genuine upstream objects -- only the sizes are ours.
    """
    cfg = dict(
        num_experts=_E, num_local_experts=_E, moe_num_experts=_E,
        hidden_size=_H, intermediate_size=_I, moe_intermediate_size=_I,
        ffn_hidden_size=_I, num_experts_per_tok=1, moe_top_k=1,
        hidden_act="silu", hidden_activation="gelu_pytorch_tanh",
        activation_function="silu", swiglu_limit=7.0,
        # transformers >= 5 dispatches expert forwards through ExpertsInterface;
        # "eager" selects the model's own forward, which is what we are measuring.
        _experts_implementation="eager",
    )
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def _standard_run(module, x):
    """``forward(hidden_states, top_k_index, top_k_weights)`` -- all tokens to expert 0."""
    return module(x, torch.zeros(x.shape[0], 1, dtype=torch.long),
                  torch.ones(x.shape[0], 1, dtype=torch.float64))


def _gptoss_run(module, x):
    """gpt-oss spells the routing arguments differently and takes them by keyword."""
    return module(x, router_indices=torch.zeros(x.shape[0], 1, dtype=torch.long),
                  routing_weights=torch.ones(x.shape[0], 1, dtype=torch.float64))


def _jetmoe_run(module, x):
    """JetMoe fuses inside ``JetMoeMoE``, which runs its own router over [B, T, H]."""
    return module(x.unsqueeze(0))[0]


#: model_type -> (transformers module, expert class, driver, fused parameter,
#: axis of that parameter -- AFTER the expert axis -- that carries 2*intermediate).
#:
#: The axis is adjudicated per family rather than inferred, because at toy size
#: more than one axis could coincidentally measure 2*intermediate. gpt-oss
#: declares its stack [E, hidden, 2*inter] (``x @ W``, fused axis LAST); every
#: other family declares [E, 2*inter, hidden] (``linear(x, W)``, fused axis FIRST).
#:
#: dbrx is absent deliberately: it stores w1/v1/w2 as three separately NAMED flat
#: tensors and never fuses them, so it has no fused axis and no order to get
#: wrong -- see ``test_dbrx_has_no_fused_tensor_and_so_no_order_to_get_wrong``.
#: nemotron_h is absent because it is non-gated (up/down only, no gate at all).
UPSTREAM_EXPERT_DRIVERS = {
    # per-expert on disk, fused in the module tree (e4b builds the fusion itself)
    "qwen3_moe":    ("qwen3_moe",    "Qwen3MoeExperts",       _standard_run, "gate_up_proj", 0),
    "mixtral":      ("mixtral",      "MixtralExperts",        _standard_run, "gate_up_proj", 0),
    "phimoe":       ("phimoe",       "PhimoeExperts",         _standard_run, "gate_up_proj", 0),
    "jamba":        ("jamba",        "JambaExperts",          _standard_run, "gate_up_proj", 0),
    "lfm2_moe":     ("lfm2_moe",     "Lfm2MoeExperts",        _standard_run, "gate_up_proj", 0),
    # natively pre-fused: the exposed set, with no key names to recover the order from
    "granitemoe":   ("granitemoe",   "GraniteMoeExperts",     _standard_run, "gate_up_proj", 0),
    "gemma4_text":  ("gemma4",       "Gemma4TextExperts",     _standard_run, "gate_up_proj", 0),
    "qwen3_vl_moe": ("qwen3_vl_moe", "Qwen3VLMoeTextExperts", _standard_run, "gate_up_proj", 0),
    "qwen3_5_moe":  ("qwen3_5_moe",  "Qwen3_5MoeExperts",     _standard_run, "gate_up_proj", 0),
    "axk1":         ("axk1",         "AXK1Experts",           _standard_run, "gate_up_proj", 0),
    "jetmoe":       ("jetmoe",       "JetMoeMoE",             _jetmoe_run,   "input_linear.weight", 0),
    "gpt_oss":      ("gpt_oss",      "GptOssExperts",         _gptoss_run,   "gate_up_proj", 1),
    # dedicated e4b keymap that also splits a fused stack itself
    "deepseek_v4":  ("deepseek_v4",  "DeepseekV4Experts",     _standard_run, "gate_up_proj", 0),
}


def _resolve(module, dotted: str):
    obj = module
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def build_upstream_expert(model_type: str):
    """The real upstream expert module for ``model_type``, at toy size, float64.

    Raises :class:`FusedLayoutUndetermined` if transformers does not ship the
    family or no longer exposes the class -- never a silent fallback, because a
    skipped probe and a passing probe must not look the same.
    """
    try:
        mod_name, cls_name, run, param, axis = UPSTREAM_EXPERT_DRIVERS[model_type]
    except KeyError:
        raise FusedLayoutUndetermined(
            f"{model_type!r} has no upstream expert driver registered in "
            f"UPSTREAM_EXPERT_DRIVERS, so its fused layout cannot be measured. "
            f"Add one rather than assuming the default (e4b#515)."
        ) from None
    try:
        mod = importlib.import_module(f"transformers.models.{mod_name}.modeling_{mod_name}")
        cls = getattr(mod, cls_name)
    except (ImportError, AttributeError) as exc:
        raise FusedLayoutUndetermined(
            f"{model_type!r}: transformers does not expose "
            f"{mod_name}.{cls_name} ({exc}); the layout cannot be measured here."
        ) from exc
    return cls(_toy_config()).to(torch.float64), run, param, axis


def _fill(module, generator):
    for name, p in module.named_parameters():
        with torch.no_grad():
            p.copy_(torch.zeros(p.shape, dtype=torch.float64) if "bias" in name
                    else torch.randn(p.shape, generator=generator, dtype=torch.float64) * _SCALE)


def measure_fused_layout(model_type: str) -> FusedLayout:
    """Run upstream's own expert forward and report where the gate actually is.

    Raises :class:`FusedLayoutUndetermined` rather than returning a guess.
    """
    module, run, param_path, axis = build_upstream_expert(model_type)
    return measure_module_fused_layout(module, run, param_path, axis, label=model_type)


def measure_module_fused_layout(module, run, param_path="gate_up_proj", axis=0, *, label="<module>"):
    """The measurement itself, over any module exposing a fused gate/up parameter.

    Split out from :func:`measure_fused_layout` so the tests can drive it with a
    DELIBERATELY swapped or interleaved forward and prove the probe catches it --
    a detector never run against a known-wrong input has not been calibrated.
    """
    with torch.no_grad():
        g = torch.Generator().manual_seed(_SEED)
        _fill(module, g)
        x = torch.randn(_T, _H, generator=g, dtype=torch.float64) * 0.5
        param = _resolve(module, param_path)
        pristine = param.detach().clone()
        width = pristine.shape[1 + axis]
        if width % 2:
            raise FusedLayoutUndetermined(
                f"{label}: fused axis {axis} of {param_path} has odd width {width}; "
                f"a gate/up pack is even by construction."
            )

        def out_with(selection, t):
            scaled = pristine.clone()
            if axis == 0:
                scaled[:, selection, :] *= t
            else:
                scaled[:, :, selection] *= t
            param.copy_(scaled)
            try:
                return run(module, x)
            finally:
                param.copy_(pristine)

        residuals, scale = {}, None
        for key, selection in _index_sets(width).items():
            y0, y1, y2 = out_with(selection, 0.0), out_with(selection, 1.0), out_with(selection, 2.0)
            if scale is None:
                scale = max(y1.abs().max().item(), 1e-30)
            # Second difference: exactly zero iff the output is AFFINE in this set.
            residuals[key] = (y0 - 2 * y1 + y2).abs().max().item() / scale

    ranked = sorted(residuals, key=residuals.get)
    best, runner_up = ranked[0], ranked[1]
    r_best, r_next = residuals[best], residuals[runner_up]
    shown = ", ".join(f"{p}/{o[0]}-first={v:.3e}" for (p, o), v in residuals.items())

    if r_best > _AFFINE_TOL:
        raise FusedLayoutUndetermined(
            f"{label}: no index set of the fused axis is affine in the output, so "
            f"nothing behaves like an `up` projection and the gate cannot be located. "
            f"Residuals [{shown}]. This is a REFUSAL, not a default: the layout is "
            f"neither contiguous nor interleaved, or a clamp engaged (e4b#515)."
        )
    if r_next <= r_best * _MARGIN:
        raise FusedLayoutUndetermined(
            f"{label}: more than one index set is affine ({best} at {r_best:.3e}, "
            f"{runner_up} at {r_next:.3e}; the {_MARGIN:g}x margin is not met), so the "
            f"probe cannot tell them apart and refuses to pick. A LINEAR activation "
            f"does this -- with act(g)*u linear in both halves there is no gate to "
            f"find, and a verdict here would prove nothing. Residuals [{shown}]."
        )

    packing, order = best
    return FusedLayout(packing=packing, order=order, residuals=residuals,
                       margin=(r_next / r_best if r_best else float("inf")))
