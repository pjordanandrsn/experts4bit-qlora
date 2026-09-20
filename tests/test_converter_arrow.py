"""e4b#637 -- the checkpoint -> module arrow, measured against upstream's converter.

#515/#630 measured the MODULE-LAYOUT arrow: given a fused stack already placed,
which stripe does upstream's ``forward`` treat as the gate? This file measures the
other one: getting a tensor from the checkpoint INTO the module is a
transformation -- rename, merge, concatenate, transpose -- and e4b re-implements
upstream's converter in ``MoEConvention`` (``renames``, ``transpose_re``,
``expert_re`` + ``roles``). Nothing compared that re-implementation to the
converter it mirrors.

What WAS asserted before this file, from ``tests/test_moe_conventions.py``:

* ``Transpose`` and ``MergeModulelist`` appeared only inside DOCSTRINGS -- never
  in an assertion.
* ``Concatenate(dim=1)`` was asserted, but as a substring of
  ``inspect.getsource(...)``, and only for the ``FUSING`` conventions.
* ``test_prefused_transpose_family_has_its_own_convention_not_qwen2_moe``
  asserted only that qwen3_vl_moe's converter SIGNATURE differs from qwen2_moe's,
  plus that e4b's own ``transpose_re`` matches e4b's own key strings. Both sides
  of that are e4b.

``get_checkpoint_conversion_mapping`` returns first-class objects carrying their
operations, so the comparison can be made by EXECUTION rather than by reading:
run upstream's ops and e4b's plan over the same synthetic checkpoint tensors and
require the resulting module tensors to be identical. That catches direction and
axis errors an attribute comparison can only restate.

It found a real divergence: upstream's qwen3_vl_moe transpose is
``Transpose(dim0=1, dim1=2, check_dims=True)`` -- conditional -- and e4b's was
unconditional. They agree on every non-square stack and disagree, silently, on
every square one.

CPU, seeded, no checkpoint and no GPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from experts4bit_qlora.arch.moe_conventions import (  # noqa: E402
    CONVENTIONS,
    MoEConventionError,
)
from experts4bit_qlora.arch.moe_load import make_plan_reader  # noqa: E402

_SEED = 20260920


def _unescape(pattern: str) -> str:
    """Upstream spells some rename sources as regex (``block_sparse_moe\\.input_linear``)
    and others literally. e4b's renames are plain substrings, so compare on the
    literal form rather than on the spelling."""
    return pattern.replace("\\.", ".")


def _mapping(model_type):
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    return get_checkpoint_conversion_mapping(model_type) or []


def _ops_for(model_type, target_suffix):
    """Upstream's operation list for the converter whose target ends in ``suffix``."""
    for conv in _mapping(model_type):
        if any(t.endswith(target_suffix) for t in conv.target_patterns):
            return conv, list(getattr(conv, "operations", None) or [])
    return None, []


class _Stub(torch.nn.Module):
    """A module exposing one parameter of a chosen shape, for check_dims."""

    def __init__(self, shape):
        super().__init__()
        self.p = torch.nn.Parameter(torch.empty(*shape))


def _upstream_convert(op, tensor, module_shape):
    """Run one upstream ConversionOps the way the real loader would."""
    return op.convert({"src": tensor}, ["src"], ["tgt"],
                      model=_Stub(module_shape), full_layer_name="p")["tgt"]


# ─────────────────────────────────────────────────────────────────────────────
# The divergence this file was written for.
# ─────────────────────────────────────────────────────────────────────────────

def test_upstreams_transpose_is_conditional_not_unconditional():
    """Pin the property that made e4b's unconditional transpose wrong.

    If upstream ever drops check_dims this test fails and e4b's conditional
    becomes the divergence -- which is exactly when someone should look.
    """
    conv, ops = _ops_for("qwen3_vl_moe", "mlp.experts.gate_up_proj")
    assert conv is not None, "qwen3_vl_moe no longer has a gate_up_proj converter"
    assert len(ops) == 1 and type(ops[0]).__name__ == "Transpose", \
        f"qwen3_vl_moe's gate_up converter is no longer a single Transpose: {ops}"
    t = ops[0]
    assert (t.dim0, t.dim1) == (1, 2), \
        f"upstream transposes dims {(t.dim0, t.dim1)}, not the last two of [E, a, b]"
    assert t.check_dims is True, (
        "upstream's Transpose is no longer conditional; e4b now replicates a "
        "condition upstream does not apply (e4b#637)"
    )


@pytest.mark.parametrize("module_shape,ckpt_shape,should_transpose", [
    ((2, 4, 6), (2, 6, 4), True),    # the ordinary released layout: shapes differ
    ((2, 8, 8), (2, 8, 8), False),   # SQUARE: upstream leaves it alone
])
def test_e4b_matches_upstream_including_the_square_case(module_shape, ckpt_shape, should_transpose):
    """e4b's read path and upstream's converter must produce the SAME tensor.

    The square row is the one that regressed: both layouts have identical shape,
    so ``_assign`` accepts either and nothing raises -- only the numbers differ.
    """
    op = _ops_for("qwen3_vl_moe", "mlp.experts.gate_up_proj")[1][0]
    g = torch.Generator().manual_seed(_SEED)
    ckpt = torch.randn(*ckpt_shape, generator=g, dtype=torch.float64)

    want = _upstream_convert(op, ckpt, module_shape)
    got = _e4b_read(ckpt, module_shape)

    assert tuple(got.shape) == tuple(want.shape)
    assert torch.equal(got, want), (
        "e4b's load-time transform disagrees with upstream's converter on a "
        f"{ckpt_shape} checkpoint into a {module_shape} module"
    )
    # ...and the test is not vacuous: assert it actually did (or skipped) the flip.
    assert torch.equal(got, ckpt.transpose(-1, -2).contiguous()) is should_transpose or \
        torch.equal(got, ckpt) is (not should_transpose)


class _Plan:
    """The minimum of a plan the reader touches."""

    def __init__(self, key, param, transform):
        self.passthrough = {key: param}
        self.transforms = {key: transform}
        self.scales = {}


def _e4b_read(ckpt, module_shape=None):
    """e4b's own read path for a transposed pre-fused stack."""
    key, param = "mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.gate_up_proj"
    plan = _Plan(key, param, "transpose_last2")
    shape_of = (lambda name: torch.Size(module_shape)) if module_shape else None
    read = make_plan_reader(plan, lambda k: ckpt, torch.float64, param_shape=shape_of)
    return read(key)


def test_a_square_stack_without_an_expected_shape_is_refused_not_guessed():
    """When upstream's condition cannot be evaluated, refuse.

    A caller with no model to consult (the bare ``make_plan_reader`` contract)
    cannot know whether a square stack wants the flip. Both answers load and
    compute different functions, so this must raise rather than pick.
    """
    g = torch.Generator().manual_seed(_SEED)
    square = torch.randn(2, 8, 8, generator=g, dtype=torch.float64)
    with pytest.raises(MoEConventionError, match="square on its last two axes"):
        _e4b_read(square, module_shape=None)


def test_a_non_square_stack_without_an_expected_shape_still_loads():
    """The refusal is scoped to the ambiguous case only.

    A non-square stack is unambiguous -- upstream's condition would be true --
    so the bare contract keeps working and existing callers are unaffected.
    """
    g = torch.Generator().manual_seed(_SEED)
    tall = torch.randn(2, 6, 4, generator=g, dtype=torch.float64)
    assert tuple(_e4b_read(tall, module_shape=None).shape) == (2, 4, 6)


def test_the_old_unconditional_transform_really_did_diverge():
    """Calibration: prove the fix was fixing something.

    Without this the tests above could pass against a transform that never
    differed from the old one.
    """
    g = torch.Generator().manual_seed(_SEED)
    square = torch.randn(2, 8, 8, generator=g, dtype=torch.float64)
    op = _ops_for("qwen3_vl_moe", "mlp.experts.gate_up_proj")[1][0]
    upstream = _upstream_convert(op, square, (2, 8, 8))
    unconditional = square.transpose(-1, -2).contiguous()      # what e4b used to do
    assert upstream.shape == unconditional.shape, "the shapes agree -- that is the hazard"
    assert not torch.equal(upstream, unconditional), (
        "upstream and an unconditional transpose agree on this fixture, so it "
        "cannot demonstrate the divergence"
    )
    assert torch.equal(_e4b_read(square, (2, 8, 8)), upstream)


# ─────────────────────────────────────────────────────────────────────────────
# The declaration, checked against upstream's operation list.
# ─────────────────────────────────────────────────────────────────────────────

#: convention name -> a model_type upstream actually maps to a converter.
_WITH_UPSTREAM_CONVERTER = {
    "qwen2_moe": "qwen3_moe",
    "mixtral": "mixtral",
    "phimoe": "phimoe",
    "jamba": "jamba",
    "lfm2_moe": "lfm2_moe",
    "granitemoe": "granitemoe",
    "qwen3_vl_moe": "qwen3_vl_moe",
    "nemotron_h": "nemotron_h",
}

#: Conventions upstream gives NO converter entry, so there is nothing to compare
#: against and the adjudication stays prose plus a released index. Stated rather
#: than silently skipped -- an unverifiable family must not look verified.
_NO_UPSTREAM_CONVERTER = {"gemma4", "qwen3_5_moe", "gptoss", "jetmoe", "dbrx", "dense"}

#: Conventions whose shape DIFFERS from upstream's converter without either being
#: wrong. A converter whose source patterns do not match a checkpoint is a no-op --
#: granitemoe's own record says as much ("a checkpoint already saved with the
#: current names matches nothing and passes through unchanged") -- so "upstream
#: merges per-expert keys" does NOT imply "this release ships per-expert keys".
#:
#: Recorded with the evidence that settled each one, and asserted below to still be
#: exactly this, so a real contradiction cannot hide behind an explained one.
_CONVERTER_COVERS_ANOTHER_SPELLING = {
    "axk1": (
        "Upstream's converter merges per-expert keys "
        "(mlp.experts.*.{gate,up}_proj.weight -> MergeModulelist + Concatenate), "
        "but the RELEASED skt/A.X-K1 checkpoint is pre-fused and e4b's record is "
        "right: its model.safetensors.index.json has 976 keys, of which 120 are "
        "expert keys and ZERO are per-expert -- 60 mlp.experts.gate_up_proj + 60 "
        "mlp.experts.down_proj over layers 1..60. The converter's source patterns "
        "simply never match, so it does nothing. Checked against the index rather "
        "than the module tree, which is what the earlier adjudication had cited."
    ),
}


#: One key from the released skt/A.X-K1 index, in its real spelling.
_AXK1_RELEASED_EXPERT_KEY = "model.layers.1.mlp.experts.gate_up_proj"


def test_axk1s_upstream_converter_cannot_match_its_released_spelling():
    """Turn the recorded explanation into a check.

    The claim is that upstream's per-expert converter is a NO-OP on the released
    A.X-K1 checkpoint because its source patterns never match. Assert that rather
    than trusting the prose: if upstream ever rewrites those patterns so they DO
    match a pre-fused key, the explanation stops holding and e4b's pre-fused
    convention becomes a real contradiction.
    """
    import fnmatch

    sources = [sp for c in _mapping("axk1") for sp in c.source_patterns]
    assert any("experts.*" in sp for sp in sources), \
        "axk1's converter no longer has per-expert source patterns — re-read the record"
    for sp in sources:
        assert not fnmatch.fnmatch(_AXK1_RELEASED_EXPERT_KEY, f"*{sp}*"), (
            f"upstream's axk1 pattern {sp!r} now matches the RELEASED pre-fused key "
            f"{_AXK1_RELEASED_EXPERT_KEY!r}; the converter is no longer a no-op and "
            f"e4b's pre-fused convention must be re-adjudicated (e4b#637)"
        )
    # ...and e4b does claim that key, which is the other half of "no contradiction".
    from experts4bit_qlora.arch.moe_conventions import convention_for
    conv = convention_for("axk1")
    assert not conv.roles, "axk1 is declared pre-fused"
    assert conv.match("mlp.experts.gate_up_proj") is None, \
        "a pre-fused convention must not treat the fused stack as a per-expert key"


def _first_target_suffix(conv):
    """The suffix of this convention's FIRST fused target.

    Gated families land on ``experts.gate_up_proj``; nemotron_h has no gate and
    lands on ``experts.up_proj``. Derived from the convention rather than hard-coded
    so a non-gated family is not silently compared against a target it never has.
    """
    first, _ = conv.fused_names(0)
    # Last two segments only. Upstream states its targets AFTER its own renames
    # (phimoe's converter targets ".experts.gate_up_proj" because it has already
    # rewritten block_sparse_moe -> mlp), so matching the CONTAINER would compare
    # two different spellings of the same place.
    return ".".join(first.split(".")[-2:])


def test_the_no_converter_set_is_true_of_upstream():
    """The exemption list must describe upstream, not just excuse a gap."""
    for conv in CONVENTIONS:
        name = conv.name
        mt = _WITH_UPSTREAM_CONVERTER.get(name) or next(iter(conv.model_types), None)
        if mt is None:
            continue
        has = bool(_mapping(mt))
        if name in _CONVERTER_COVERS_ANOTHER_SPELLING:
            continue
        if name in _WITH_UPSTREAM_CONVERTER:
            assert has, f"{name}: expected an upstream converter for {mt}, found none"
        elif name in _NO_UPSTREAM_CONVERTER:
            assert not has, (
                f"{name}: upstream now DOES ship a converter for {mt} -- move it into "
                f"_WITH_UPSTREAM_CONVERTER so its declaration gets checked (e4b#637)"
            )


def test_every_convention_is_either_checked_or_named_unverifiable():
    """No convention may fall through the gap between the two sets."""
    covered = (set(_WITH_UPSTREAM_CONVERTER) | _NO_UPSTREAM_CONVERTER
               | set(_CONVERTER_COVERS_ANOTHER_SPELLING))
    missing = [c.name for c in CONVENTIONS if c.name not in covered]
    assert not missing, f"conventions in neither set: {missing}"


@pytest.mark.parametrize("name,model_type", sorted(_WITH_UPSTREAM_CONVERTER.items()))
def test_transpose_declaration_matches_upstreams_operations(name, model_type):
    """``transpose_re`` set iff upstream actually has a Transpose, on the last two axes.

    Before this, e4b declaring a transpose upstream did not have (or missing one
    upstream did have) passed every test. On a non-square stack that is a loud
    shape error; on a square one it is silent.
    """
    conv = next(c for c in CONVENTIONS if c.name == name)
    _, ops = _ops_for(model_type, _first_target_suffix(conv))
    upstream_transposes = [o for o in ops if type(o).__name__ == "Transpose"]
    if conv.transpose_re is not None:
        assert upstream_transposes, (
            f"{name}: e4b transposes experts.gate_up_proj at load, but upstream's "
            f"converter for {model_type} has no Transpose ({ops}) — e4b would flip a "
            f"tensor upstream leaves alone"
        )
        for t in upstream_transposes:
            assert {t.dim0, t.dim1} == {1, 2}, (
                f"{name}: upstream transposes dims {(t.dim0, t.dim1)}, but e4b's "
                f"transpose_last2 swaps the last two of a 3-D [E, a, b] stack"
            )
    else:
        assert not upstream_transposes, (
            f"{name}: upstream's converter for {model_type} has a Transpose "
            f"({upstream_transposes}) but e4b's convention declares transpose_re=None, "
            f"so the stack would be placed unflipped"
        )


@pytest.mark.parametrize("name,model_type", sorted(_WITH_UPSTREAM_CONVERTER.items()))
def test_per_expert_declaration_matches_upstreams_merge_and_concatenate(name, model_type):
    """``roles``/``expert_re`` set iff upstream MERGES per-expert tensors.

    ``MergeModulelist`` is what makes a family per-expert on disk; a pre-fused one
    has none. e4b encodes the same distinction as a matchable-vs-unmatchable
    ``expert_re``, and the two must not drift: treating a pre-fused stack as
    per-expert gathers keys that are already stacked, and the reverse silently
    reads a layer as dense.
    """
    conv = next(c for c in CONVENTIONS if c.name == name)
    _, ops = _ops_for(model_type, _first_target_suffix(conv))
    names = [type(o).__name__ for o in ops]
    if conv.roles:
        assert "MergeModulelist" in names, (
            f"{name}: e4b gathers per-expert keys, but upstream's converter for "
            f"{model_type} does not merge a modulelist ({names})"
        )
        if conv.gated:
            assert "Concatenate" in names, (
                f"{name}: gated per-expert families concatenate gate+up; upstream's "
                f"ops are {names}"
            )
            cat = next(o for o in ops if type(o).__name__ == "Concatenate")
            assert cat.dim == 1, (
                f"{name}: upstream concatenates gate/up on dim {cat.dim}; fuse_experts "
                f"joins on the intermediate axis of a per-expert [inter, hidden] pair"
            )
        else:
            # nemotron_h stacks up_proj alone -- there is no gate to concatenate.
            assert "Concatenate" not in names, (
                f"{name}: declared non-gated, but upstream concatenates ({names})"
            )
    else:
        assert "MergeModulelist" not in names, (
            f"{name}: upstream MERGES per-expert tensors for {model_type} ({names}), "
            f"but e4b's convention is pre-fused (unmatchable expert_re) and would "
            f"never gather them"
        )


#: Renames upstream applies that e4b does NOT, with the consequence of each.
#: Recorded rather than asserted away, and asserted to be exactly this set: the
#: missing direction fails LOUDLY (keys land on no parameter and the planner
#: raises "do not map"), so it is a support gap, not a silent-wrong-numbers bug
#: -- which is why it is listed here instead of holding up this change.
_MISSING_RENAMES = {
    # NVIDIA's Nemotron-H modeling nests the decoder under ``backbone.``; upstream
    # rewrites that to ``model.`` on load and e4b does not, so a checkpoint using
    # the backbone spelling raises "no parameter ... in the model" rather than
    # mis-loading. A support gap with a loud failure, tracked here.
    "nemotron_h": {("backbone.", "model.")},
}


@pytest.mark.parametrize("name,model_type", sorted(_WITH_UPSTREAM_CONVERTER.items()))
def test_e4b_never_invents_a_rename_upstream_does_not_have(name, model_type):
    """The rename direction that could be SILENT.

    A rename e4b invents can redirect a checkpoint key onto a real-but-wrong
    parameter, which loads and computes the wrong thing. A rename e4b LACKS
    leaves keys unmapped and raises. So this direction is the hard assertion and
    the other is a recorded census.
    """
    conv = next(c for c in CONVENTIONS if c.name == name)
    upstream = {(_unescape(s0), t0)
                for c in _mapping(model_type) if type(c).__name__ == "WeightRenaming"
                for s0, t0 in zip(c.source_patterns, c.target_patterns)}
    for src, dst in conv.renames:
        assert (src, dst) in upstream, (
            f"{name}: e4b rewrites {src!r} -> {dst!r}, which upstream's converter for "
            f"{model_type} does not do ({sorted(upstream)}). An invented rename can "
            f"land a key on a real but wrong parameter — that loads and is silent"
        )
        assert src != dst, f"{name}: a no-op rename {src!r} hides whether it is needed"


def _upstream_rename_pairs(model_type):
    """Upstream's renames as a set of literal ``(source, target)`` pairs."""
    return {(_unescape(s0), t0)
            for c in _mapping(model_type) if type(c).__name__ == "WeightRenaming"
            for s0, t0 in zip(c.source_patterns, c.target_patterns)}


@pytest.mark.parametrize("name,model_type", sorted(_WITH_UPSTREAM_CONVERTER.items()))
def test_renames_e4b_lacks_are_exactly_the_recorded_ones(name, model_type):
    """The loud direction, pinned so it cannot widen unnoticed.

    A new entry here means upstream started rewriting a key e4b does not, so some
    spelling of that family's checkpoint stops loading entirely.
    """
    conv = next(c for c in CONVENTIONS if c.name == name)
    missing = _upstream_rename_pairs(model_type) - {(src, dst) for src, dst in conv.renames}
    assert missing == _MISSING_RENAMES.get(name, set()), (
        f"{name}: renames upstream has and e4b lacks changed.\n"
        f"  found:    {sorted(missing)}\n"
        f"  recorded: {sorted(_MISSING_RENAMES.get(name, set()))}\n"
        f"Each is a checkpoint spelling e4b cannot map — it raises rather than "
        f"mis-loading. Implement it, or record it with its consequence."
    )


# ─────────────────────────────────────────────────────────────────────────────
# The census the issue asked for: who is one config revision from square?
# ─────────────────────────────────────────────────────────────────────────────

#: model -> (hidden_size, moe_intermediate_size), from the ADJUDICATED released
#: configs recorded in the convention comments. ``gate_up`` goes square when
#: 2*inter == hidden; ``down`` when inter == hidden.
#:
#: This is a WATCHLIST, not a guarantee. It covers the configs e4b has actually
#: adjudicated, so a variant nobody here has looked at is simply absent -- which
#: is why the load-time condition above is the real protection and this is a
#: record of how much margin each family has.
ADJUDICATED_DIMS = {
    # qwen3_vl_moe is the ONLY family e4b transposes today, so it is the only one
    # where a square stack changes behaviour. Qwen3-VL-30B-A3B: checkpoint
    # gate_up [128, 2048, 1536] -> hidden 2048, 2*inter 1536, inter 768.
    "Qwen3-VL-30B-A3B": (2048, 768),
    # gemma-4-26B-A4B-it: gate_up [128, 1408, 2816] -> hidden 2816, inter 704.
    "gemma-4-26B-A4B-it": (2816, 704),
}


@pytest.mark.parametrize("model,dims", sorted(ADJUDICATED_DIMS.items()))
def test_adjudicated_configs_are_not_square(model, dims):
    """None of the adjudicated releases sits on the ambiguous point today.

    If one ever does, the conditional transform above is what keeps it correct —
    but the margin is worth knowing, because "not square" is an accident of the
    config, not a property anybody guaranteed.
    """
    hidden, inter = dims
    assert 2 * inter != hidden, (
        f"{model}: 2*moe_intermediate ({2 * inter}) == hidden ({hidden}), so its "
        f"gate_up stack is SQUARE and a transpose decision is invisible to shape"
    )
    assert inter != hidden, (
        f"{model}: moe_intermediate == hidden ({hidden}), so its down stack is SQUARE"
    )
