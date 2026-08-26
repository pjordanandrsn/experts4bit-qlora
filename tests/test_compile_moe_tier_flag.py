# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k12 instrument, checked WITHOUT a GPU.

The flag lifts ONE of two dynamo exclusions. Two things must hold or
the arm it enables measures the wrong thing:

  1. It must not touch the ATTENTION disable, whose cause is known and
     specific (F1 Stage B: inductor re-emits the paged-decode kernel
     and it dies on a loop-carried m_i typed fp32 then fp64). That
     exclusion is lifted only by its OWN flag, --compile-attn-tier,
     which is arm 3's control -- and that flag must be honoured at
     EVERY site that re-applies the disable, not just the first.
  2. It must refuse outside --placement-override all-vram, because the
     exclusion it lifts exists for CPU tier dispatch and only all-vram
     is known to avoid it.

And the receipt must record which way the flag was set, so an arm
cannot be labelled with a flag it did not actually use.
"""
import pathlib
import re

_SRC = (pathlib.Path(__file__).resolve().parents[1]
        / "bench" / "hybrid-g9" / "step_decomp.py").read_text()


def _compile_block() -> str:
    """The --compile-layers site specifically.

    `ALL_ATTENTION_FUNCTIONS[IMPL_NAME] = dynamo.disable(` appears at
    MORE THAN ONE site in this file (the BV3 stage unwraps and
    re-disables it too), and the first match is not this one. Anchor on
    the comment unique to the compile-layers block instead.
    """
    i = _SRC.index("clean graph breaks:")
    # The window must reach past the MoE guard's else-branch. It was
    # 1800, which stopped short once the arm-3 amendment lengthened
    # the block -- and a too-short window makes these tests fail with
    # "substring not found" rather than say what is wrong. Anchor on
    # the end of the block instead of guessing a length.
    end = _SRC.index("n_c = 0", i)
    return _SRC[i:end]


def test_moe_flag_never_touches_the_attention_exclusion():
    """--compile-moe-tier lifts ONE exclusion, and not the other.

    This was `test_attention_disable_is_unconditional`. The attention
    disable is no longer unconditional -- PREREG-k12's arm-3
    amendment lifts it under its OWN flag -- but the invariant that
    matters is unchanged and is what this now states: the MoE flag
    must not reach attention. Only --compile-attn-tier may.
    """
    blk = _compile_block()
    g = blk.index("if a.compile_moe_tier:")
    tail = blk[g:]
    assert "ALL_ATTENTION_FUNCTIONS" not in tail, \
        "nothing about attention may live under the MoE guard"
    assert "m.forward = dynamo.disable(m.forward)" in tail, \
        "the MoE disable must live under the guard's else branch"
    # Test the CONDITION that gates the attention disable, not whether
    # the flag's name appears anywhere above it. A bare substring scan
    # fires on the arm-3 block's own `assert a.compile_moe_tier`
    # precondition, which is a legitimate mention -- the same
    # code-versus-prose confusion that made an earlier gnf4 test fire
    # on the comment explaining its own fix.
    attn = blk.index("ALL_ATTENTION_FUNCTIONS[IMPL_NAME] = dynamo.disable(")
    gate = blk.rfind("if ", 0, attn)
    cond = blk[gate:blk.index(":", gate)]
    assert "compile_moe_tier" not in cond, \
        f"the attention disable is gated on the MoE flag: {cond!r}"
    assert "compile_attn_tier" in cond, \
        f"the attention disable must be gated on its own flag: {cond!r}"


def test_attention_disable_is_lifted_ONLY_by_its_own_flag():
    blk = _compile_block()
    a = blk.index("if a.compile_attn_tier:")
    body = blk[a:blk.index("else:", a)]
    assert "assert a.compile_moe_tier" in body, \
        ("--compile-attn-tier alone is an arm PREREG-k12 never "
         "registered; it must require --compile-moe-tier")
    assert "ALL_ATTENTION_FUNCTIONS[IMPL_NAME] = dynamo.disable(" \
        in blk[blk.index("else:", a):], \
        "the default path must still apply the attention disable"


def test_every_rewrap_site_honours_the_attention_flag():
    """The finding that made arm 3 a lie (review, e4b#289).

    `dynamo.disable` on the attention shim is applied at MORE THAN
    ONE site: the compile-layers block sets it, and the b1d and batch
    stages unwrap to the raw `_orig` and RE-apply it. Skipping only
    the first one leaves those re-wraps to silently restore the
    exclusion on the very graph loops arm 3 measures -- so arm 3 runs
    as arm 2 while its receipt records compile_attn_tier=True.

    This is the disable-wrappers-get-unwrapped trap in mirror image,
    and the docstring of `_compile_block` above already warned that
    the site is not unique. A grep-for-one-site test is what let it
    through, so this one counts them.
    """
    sites = [i for i in range(len(_SRC))
             if _SRC.startswith("_dyn.disable(_orig)", i)
             or _SRC.startswith("dynamo.disable(_orig)", i)]
    assert sites, "no re-wrap site found -- has the shim changed?"
    for i in sites:
        # walk back to the enclosing `if` that gates this re-wrap
        head = _SRC[max(0, i - 400):i]
        gate = head.rfind("if ")
        assert gate != -1, f"re-wrap at {i} has no enclosing guard"
        cond = head[gate:]
        assert "compile_attn_tier" in cond, (
            f"re-wrap site at offset {i} re-applies the attention "
            "disable without honouring --compile-attn-tier; arm 3 "
            "would silently be arm 2")


def test_flag_refuses_outside_all_vram():
    blk = _compile_block()
    g = blk.index("if a.compile_moe_tier:")
    body = blk[g:blk.index("else:", g)]
    assert "assert" in body and "all-vram" in body, \
        "the flag must refuse outside the placement it is registered for"


def test_flags_default_off_and_are_store_true():
    for flag in ("--compile-moe-tier", "--compile-attn-tier"):
        m = re.search(r'add_argument\("' + flag + r'"[^)]*\)', _SRC, re.S)
        assert m, f"{flag} not registered"
        assert "store_true" in m.group(0), f"{flag} must default OFF"


def test_receipts_record_the_flag():
    """An arm must not be able to claim a flag it did not use."""
    for flag in ("compile_moe_tier", "compile_attn_tier"):
        assert _SRC.count(f'"{flag}": bool(a.{flag})') >= 2, \
            f"both receipt shapes must record {flag}'s value"
