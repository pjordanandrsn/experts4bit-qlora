"""e4b#515 -- the fused gate/up layout, VERIFIED against upstream's own forward.

What was already in place before this file, and why it was not enough:

* #518 made the order a declared field (``MoEConvention.fused_order``). A field
  records an adjudication; it cannot tell you the adjudication was wrong.
* #519 added a numerical detector that packs with ``fuse_experts`` and splits with
  the consumers' ``chunk(2, dim=-1)``. That closes OUR packer against OUR splitter
  -- but the natively pre-fused families, which are the ones #515 is about, never
  go through ``fuse_experts`` at all. Their stack arrives already fused from the
  checkpoint, so the composition #519 checks is not the composition they use.
* Three older tests (``test_gate_precedes_up_in_upstream_spec`` and the two
  ``inspect.getsource`` forward tests) are regexes over upstream SOURCE TEXT, and
  they cover only the per-expert ``FUSING`` conventions.

So nothing measured what the pre-fused families actually do. This file does, by
running upstream's real expert module and reading the layout out of its
behaviour -- see :mod:`experts4bit_qlora.arch.fused_layout_probe` for why that is
the right ground truth and what it still cannot establish.

It found a real defect: ``gptoss`` declared contiguous gate-first and is
interleaved.

CPU, float64, seeded, no checkpoint and no GPU.
"""
from __future__ import annotations

import pathlib
import re
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from experts4bit_qlora.arch.fused_layout_probe import (  # noqa: E402
    UPSTREAM_EXPERT_DRIVERS,
    FusedLayoutUndetermined,
    measure_fused_layout,
    measure_module_fused_layout,
)
from experts4bit_qlora.arch.moe_conventions import (  # noqa: E402
    CONVENTIONS,
    MoEConvention,
    MoEConventionError,
    convention_for,
)

_ROOT = pathlib.Path(__file__).resolve().parents[1] / "experts4bit_qlora"


# ─────────────────────────────────────────────────────────────────────────────
# The verification itself: what each family DECLARES vs what upstream DOES.
# ─────────────────────────────────────────────────────────────────────────────

#: One representative model_type per gated convention, chosen so the probe builds
#: the class that convention's checkpoints are actually consumed by.
_REPRESENTATIVES = {
    "qwen2_moe": "qwen3_moe",
    "mixtral": "mixtral",
    "phimoe": "phimoe",
    "jamba": "jamba",
    "lfm2_moe": "lfm2_moe",
    "granitemoe": "granitemoe",
    "gptoss": "gpt_oss",
    "gemma4": "gemma4_text",
    "qwen3_vl_moe": "qwen3_vl_moe",
    "jetmoe": "jetmoe",
    "qwen3_5_moe": "qwen3_5_moe",
    "axk1": "axk1",
}

#: Gated conventions that nonetheless have NO fused gate/up tensor, so there is no
#: order to measure. Only dbrx: it stores w1 / v1 / w2 as three separately NAMED
#: flat tensors and never fuses them, which is why the issue's ``roles={}``
#: enumeration over-counted the exposed set. Excluded by name and by reason, and
#: the reason is itself tested -- see
#: ``test_dbrx_has_no_fused_tensor_and_so_no_order_to_get_wrong``.
_NO_FUSED_TENSOR = frozenset({"dbrx"})

_GATED = [c for c in CONVENTIONS
          if c.gated and c.model_types and c.name not in _NO_FUSED_TENSOR]


@pytest.mark.parametrize("conv", _GATED, ids=lambda c: c.name)
def test_declared_layout_matches_what_upstream_actually_computes(conv):
    """The gap #518 and #519 left: is the declared layout TRUE for this family?

    A mismatch here is the silent-wrong-numbers case. It is a hard failure, not a
    warning, because a run with the halves swapped produces an admitted receipt.
    """
    model_type = _REPRESENTATIVES.get(conv.name)
    assert model_type is not None, (
        f"{conv.name} is gated and reachable but has no representative model_type here. "
        f"Add one and let the probe measure it -- a family that is not measured is not "
        f"verified, and #515 is about exactly the families nobody thought to check."
    )
    measured = measure_fused_layout(model_type)
    assert measured.packing == conv.ckpt_gate_up_packing, (
        f"{conv.name}: declares ckpt_gate_up_packing={conv.ckpt_gate_up_packing!r}, but "
        f"upstream's {model_type} expert forward packs gate/up {measured.packing!r} "
        f"(margin {measured.margin:.1e}). Loading it as declared computes a wrong "
        f"activation and raises nothing."
    )
    assert tuple(measured.order) == tuple(conv.fused_order), (
        f"{conv.name}: declares fused_order={tuple(conv.fused_order)!r}, but upstream's "
        f"{model_type} expert forward treats the layout as {tuple(measured.order)!r} "
        f"(margin {measured.margin:.1e})."
    )


def test_gptoss_is_interleaved_and_the_record_says_so():
    """The defect this file found, pinned as its own test.

    Before #515's packing field the record declared contiguous gate-first for
    gpt-oss, which is false about the released checkpoint. It was harmless only
    because ``from_gptoss`` de-interleaves on the single path that loads gpt-oss.
    A field stating something untrue is a hazard even while unused, because the
    next consumer reads the field rather than the family.
    """
    conv = convention_for("gpt_oss")
    assert conv.ckpt_gate_up_packing == "interleaved"
    assert conv.splits_by_chunk2 is False
    measured = measure_fused_layout("gpt_oss")
    assert measured.packing == "interleaved" and measured.gate_first
    # ...and it is genuinely NOT contiguous: both contiguous hypotheses must be
    # far from affine, or "interleaved" would just be the winner of a weak field.
    contiguous = [v for (p, _), v in measured.residuals.items() if p == "contiguous"]
    assert min(contiguous) > 1e-6, (
        f"a contiguous split of gpt-oss looks affine ({min(contiguous):.3e}); the "
        f"probe's verdict would not be safe"
    )


def test_every_gated_convention_is_actually_measured():
    """A guard on coverage, so an unmeasured family cannot look verified.

    Without this, adding a convention and forgetting a representative would leave
    a green suite that never probed it -- the #515 failure in miniature.
    """
    missing = [c.name for c in _GATED if c.name not in _REPRESENTATIVES]
    assert not missing, f"gated conventions with no probe representative: {missing}"
    # Every exclusion must be deliberate and named, not merely absent.
    excluded = [c.name for c in CONVENTIONS
                if c.gated and c.model_types and c.name not in _REPRESENTATIVES]
    assert set(excluded) == set(_NO_FUSED_TENSOR), (
        f"gated conventions neither measured nor listed in _NO_FUSED_TENSOR: "
        f"{sorted(set(excluded) - set(_NO_FUSED_TENSOR))}"
    )
    unknown = [m for m in _REPRESENTATIVES.values() if m not in UPSTREAM_EXPERT_DRIVERS]
    assert not unknown, f"representatives with no upstream driver: {unknown}"


def test_the_exposed_set_is_derived_from_the_records_not_asserted():
    """Which shipped families are actually exposed -- pinned, and re-derived here.

    ``NATIVELY_PREFUSED`` is a hand-written frozenset, so on its own it is another
    unverified declaration. Re-derive it from the records: a convention is exposed
    when it is gated, reachable by lookup, never per-expert (so no key names can
    recover the order), and actually fuses a gate/up pair into one tensor.

    Two earlier counts of this set -- six in the issue, nine in the correction --
    both used ``roles={} and unmatchable expert_re`` as the predicate. That admits
    ``dense`` (no model_types, never reached by lookup) and ``dbrx`` (three
    separately named flat tensors, nothing fused). The answer is seven.
    """
    from experts4bit_qlora.arch.moe_conventions import NATIVELY_PREFUSED

    derived = {
        c.name for c in CONVENTIONS
        if c.gated and c.model_types and not c.roles and c.name not in _NO_FUSED_TENSOR
    }
    assert derived == set(NATIVELY_PREFUSED), (
        f"NATIVELY_PREFUSED disagrees with the records: only in the set "
        f"{sorted(set(NATIVELY_PREFUSED) - derived)}, only derived "
        f"{sorted(derived - set(NATIVELY_PREFUSED))}"
    )
    assert len(derived) == 7, f"expected seven exposed conventions, got {sorted(derived)}"
    # Every exposed convention is measured against upstream by this file.
    assert set(NATIVELY_PREFUSED) <= set(_REPRESENTATIVES), (
        f"exposed but never probed: "
        f"{sorted(set(NATIVELY_PREFUSED) - set(_REPRESENTATIVES))}"
    )


def test_dbrx_has_no_fused_tensor_and_so_no_order_to_get_wrong():
    """A scope correction: dbrx is NOT in the exposed set, despite roles={}.

    The issue's exposed set was enumerated by ``roles={}`` + unmatchable
    ``expert_re``. That over-counts. The defect needs a tensor whose two halves
    are UNLABELLED; dbrx stores w1 / v1 / w2 as three separately NAMED flat
    tensors and never fuses them, so the order is recoverable from names exactly
    as it is for a per-expert family.
    """
    M = pytest.importorskip("transformers.models.dbrx.modeling_dbrx")
    params = dict(M.DbrxExpertGLU(types.SimpleNamespace(
        hidden_size=6, ffn_hidden_size=4, moe_num_experts=2,
        ffn_act_fn={"name": "silu"},
    )).named_parameters())
    assert "gate_up_proj" not in params, "dbrx now fuses gate/up -- it joins the exposed set"
    assert {"w1", "v1", "w2"} <= set(params), f"dbrx expert params changed: {sorted(params)}"


# ─────────────────────────────────────────────────────────────────────────────
# Calibration. A detector never run against a KNOWN-WRONG input proves nothing,
# so each of these hands the probe a deliberately mis-built forward.
# ─────────────────────────────────────────────────────────────────────────────

class _FusedMLP(torch.nn.Module):
    """A minimal gated expert whose split is ours to choose.

    ``layout`` selects how ``forward`` reads the fused stack, so a test can build
    one that is deliberately swapped or interleaved and ask whether the probe
    notices.
    """

    def __init__(self, layout, act=torch.nn.functional.silu, experts=2, inter=4, hidden=6):
        super().__init__()
        self.layout, self.act, self.inter = layout, act, inter
        self.gate_up_proj = torch.nn.Parameter(torch.empty(experts, 2 * inter, hidden, dtype=torch.float64))
        self.down_proj = torch.nn.Parameter(torch.empty(experts, hidden, inter, dtype=torch.float64))

    def forward(self, x, top_k_index, top_k_weights):
        proj = torch.nn.functional.linear(x, self.gate_up_proj[0])
        if self.layout == "contiguous_gate_first":
            gate, up = proj[..., :self.inter], proj[..., self.inter:]
        elif self.layout == "contiguous_up_first":          # the swap
            up, gate = proj[..., :self.inter], proj[..., self.inter:]
        elif self.layout == "interleaved_gate_first":
            gate, up = proj[..., 0::2], proj[..., 1::2]
        elif self.layout == "interleaved_up_first":
            up, gate = proj[..., 0::2], proj[..., 1::2]
        else:  # pragma: no cover - guarded by the parametrisation
            raise AssertionError(self.layout)
        return torch.nn.functional.linear(self.act(gate) * up, self.down_proj[0])


def _run(module, x):
    return module(x, torch.zeros(x.shape[0], 1, dtype=torch.long),
                  torch.ones(x.shape[0], 1, dtype=torch.float64))


@pytest.mark.parametrize("layout,packing,order", [
    ("contiguous_gate_first", "contiguous", ("gate", "up")),
    ("contiguous_up_first", "contiguous", ("up", "gate")),
    ("interleaved_gate_first", "interleaved", ("gate", "up")),
    ("interleaved_up_first", "interleaved", ("up", "gate")),
])
def test_the_probe_reads_back_every_layout_including_a_deliberate_swap(layout, packing, order):
    """Construct a stack that IS swapped, and prove the probe reports the swap.

    This is the false-accept half of the detector. Without it, the agreement
    tests above could all be passing for the wrong reason -- a probe that always
    answered "contiguous gate-first" would satisfy twelve of the thirteen
    families and nobody would know.
    """
    measured = measure_module_fused_layout(_FusedMLP(layout), _run, label=layout)
    assert measured.packing == packing and tuple(measured.order) == order, (
        f"built a {layout} expert, probe read it as {measured}"
    )
    assert measured.margin > 1e6


def test_a_swap_is_invisible_to_every_structural_check():
    """Why this has to be numerical at all.

    A swapped stack has identical shape, dtype, expert count and split widths, and
    the same multiset of values -- only the order moved. The attention census,
    ``n_patched == n_layers``, the kernel-call floor and C1 bit-exactness are all
    indifferent to it. Only the arithmetic differs.
    """
    g = torch.Generator().manual_seed(515)
    correct = _FusedMLP("contiguous_gate_first")
    swapped = _FusedMLP("contiguous_up_first")
    with torch.no_grad():
        for p, q in zip(correct.parameters(), swapped.parameters()):
            v = torch.randn(p.shape, generator=g, dtype=torch.float64) * 0.15
            p.copy_(v)
            q.copy_(v)
    x = torch.randn(5, 6, generator=g, dtype=torch.float64) * 0.5
    a, b = _run(correct, x), _run(swapped, x)
    assert correct.gate_up_proj.shape == swapped.gate_up_proj.shape
    assert correct.gate_up_proj.dtype == swapped.gate_up_proj.dtype
    assert torch.equal(torch.sort(correct.gate_up_proj.flatten()).values,
                       torch.sort(swapped.gate_up_proj.flatten()).values)
    assert not torch.allclose(a, b), "the fixture is degenerate; the swap changed nothing"


def test_the_probe_refuses_a_degenerate_fixture_instead_of_picking_one():
    """A LINEAR activation makes every index set affine, so there is no gate to find.

    The probe must refuse. If it returned a verdict here it would be returning
    noise, and a verdict indistinguishable from noise is worse than none -- it is
    the false confidence #515 is about.
    """
    module = _FusedMLP("contiguous_gate_first", act=lambda t: 2.0 * t)
    with pytest.raises(FusedLayoutUndetermined, match="more than one index set is affine"):
        measure_module_fused_layout(module, _run, label="linear-activation")


def test_the_probe_refuses_when_no_set_is_affine():
    """A forward that is nonlinear in BOTH halves has no `up` to find either."""
    class _BothNonlinear(_FusedMLP):
        def forward(self, x, top_k_index, top_k_weights):
            proj = torch.nn.functional.linear(x, self.gate_up_proj[0])
            gate, up = proj[..., :self.inter], proj[..., self.inter:]
            return torch.nn.functional.linear(self.act(gate) * self.act(up), self.down_proj[0])

    with pytest.raises(FusedLayoutUndetermined, match="no index set .* is affine"):
        measure_module_fused_layout(_BothNonlinear("contiguous_gate_first"), _run, label="both-nonlinear")


def test_an_unregistered_family_is_refused_not_defaulted():
    """An unmeasurable family must not inherit the default silently."""
    with pytest.raises(FusedLayoutUndetermined, match="no upstream expert driver"):
        measure_fused_layout("a_family_nobody_registered")


# ─────────────────────────────────────────────────────────────────────────────
# The load-time gate.
# ─────────────────────────────────────────────────────────────────────────────

def test_an_interleaved_convention_is_refused_at_the_loader_funnel(monkeypatch):
    """Loud beats wrong, on the packing axis as well as the order axis.

    ``chunk(2, dim=-1)`` recovers gate and up only from contiguous blocks, so an
    interleaved stack placed unchanged pairs gate stripe k with up stripe k+1 --
    a wrong activation with every shape agreeing. The single funnel from the
    convention system into the loader refuses it instead.
    """
    from experts4bit_qlora import loader

    interleaved = MoEConvention(
        name="interleaved-probe", expert_re=re.compile(r"(?!)"), roles={},
        fused_prefix="mlp.experts", model_types=frozenset({"interleaved_probe"}),
        ckpt_gate_up_packing="interleaved",
    )
    assert interleaved.gate_first is True, "the ORDER is fine here; only the packing is not"
    assert interleaved.splits_by_chunk2 is False
    monkeypatch.setattr(loader, "convention_for", lambda mt, **k: interleaved, raising=False)
    monkeypatch.setattr("experts4bit_qlora.arch.moe_conventions.convention_for",
                        lambda mt, **k: interleaved)
    with pytest.raises(MoEConventionError, match="split a fused gate_up_proj with chunk"):
        loader.expert_layout_for("interleaved_probe")


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_text", "granitemoe", "qwen3_5_moe"])
def test_a_refusal_is_not_swallowed_by_the_unknown_family_fallback(monkeypatch, model_type):
    """The refusals must fire for families that ALSO have a loader-map entry.

    ``expert_layout_for`` ends in ``except MoEConventionError: -> fall back to
    SUPPORTED_ARCHITECTURES``, which exists for one case: no convention at all.
    While the refusals were raised INSIDE that ``try`` it also caught them, so
    every refusal was downgraded to a successful return for any model_type with a
    map entry -- gemma4, gemma4_text, granitemoe, gpt_oss, qwen3_5_moe, olmoe,
    qwen3_moe, deepseek_v4, kimi_k3. That is every reachable natively pre-fused
    family, i.e. precisely the set the guard was written for: it read as a guard
    and protected nobody.

    Each model_type here is in ``SUPPORTED_ARCHITECTURES``, so each one takes the
    fallback path if the refusal is ever moved back inside the ``try``.
    """
    from experts4bit_qlora import loader

    assert model_type in loader.SUPPORTED_ARCHITECTURES, (
        f"{model_type} is no longer in SUPPORTED_ARCHITECTURES, so it no longer "
        f"exercises the fallback this test is about -- pick another"
    )
    upfirst = MoEConvention(
        name=f"upfirst-{model_type}", expert_re=re.compile(r"(?!)"), roles={},
        fused_prefix="experts", model_types=frozenset({model_type}),
        fused_order=("up", "gate"),
    )
    monkeypatch.setattr("experts4bit_qlora.arch.moe_conventions.convention_for",
                        lambda mt, **k: upfirst)
    with pytest.raises(MoEConventionError, match="assume the gate occupies rows"):
        loader.expert_layout_for(model_type)


def test_gptoss_still_loads_because_its_path_de_interleaves():
    """The exemption is real code, not a carve-out for a green suite."""
    from experts4bit_qlora import loader

    assert "gpt_oss" in loader.DEINTERLEAVING_LOADERS
    assert loader.expert_layout_for("gpt_oss") == ("mlp.experts", True)


def test_every_deinterleaving_loader_names_real_code():
    """An exemption whose code has gone is a hole, so pin it to the source.

    If ``from_gptoss`` ever stops de-interleaving, this fails and the exemption
    must be withdrawn -- rather than continuing to admit a stack nothing rewrites.
    """
    from experts4bit_qlora import loader

    assert loader.DEINTERLEAVING_LOADERS == frozenset({"gpt_oss"}), (
        "a new de-interleaving exemption was added; point this test at its code too"
    )
    src = (_ROOT / "arch" / "gptoss.py").read_text()
    assert "from_gptoss" in src
    assert "0::2" in src and "1::2" in src, (
        "arch/gptoss.py no longer de-interleaves the gate_up rows, so gpt_oss must not "
        "be exempt from the chunk(2) refusal in expert_layout_for (e4b#515)"
    )


def test_a_bogus_packing_is_refused_at_construction():
    """A frozen record cannot repair itself, so it refuses rather than coerces."""
    for bad in ("Contiguous", "striped", "", None, 0):
        with pytest.raises(MoEConventionError, match="ckpt_gate_up_packing must be"):
            MoEConvention(name="probe", expert_re=re.compile(r"(?!)"), roles={},
                          fused_prefix="p", ckpt_gate_up_packing=bad)


def test_a_non_gated_convention_may_not_declare_a_packing():
    """nemotron_h stacks {up, down}; there is no gate to share an axis with."""
    with pytest.raises(MoEConventionError, match="no gate/up pair"):
        MoEConvention(name="ungated-probe", expert_re=re.compile(r"(?!)"), roles={},
                      fused_prefix="p", gated=False, ckpt_gate_up_packing="interleaved")
