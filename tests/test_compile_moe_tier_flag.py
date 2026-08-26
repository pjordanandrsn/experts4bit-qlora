# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k12 instrument, checked WITHOUT a GPU.

The flag lifts ONE of two dynamo exclusions. Two things must hold or
the arm it enables measures the wrong thing:

  1. It must not touch the ATTENTION disable, whose cause is known and
     specific (F1 Stage B: inductor re-emits the paged-decode kernel
     and it dies on a loop-carried m_i typed fp32 then fp64).
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
    return _SRC[i:i + 1800]


def test_attention_disable_is_unconditional():
    """The flag must never reach the attention exclusion."""
    blk = _compile_block()
    attn = blk.index("ALL_ATTENTION_FUNCTIONS[IMPL_NAME] = dynamo.disable(")
    guard = blk.index("if a.compile_moe_tier:")
    assert attn < guard, "the attention disable must precede the guard"
    head = blk[:guard]
    assert "compile_moe_tier" not in head, \
        "the attention disable must not be conditioned on the flag"


def test_moe_disable_is_the_only_thing_guarded():
    blk = _compile_block()
    g = blk.index("if a.compile_moe_tier:")
    tail = blk[g:]
    assert "m.forward = dynamo.disable(m.forward)" in tail, \
        "the MoE disable must live under the guard's else branch"
    assert "ALL_ATTENTION_FUNCTIONS" not in tail, \
        "nothing about attention may live under the guard"


def test_flag_refuses_outside_all_vram():
    blk = _compile_block()
    g = blk.index("if a.compile_moe_tier:")
    body = blk[g:blk.index("else:", g)]
    assert "assert" in body and "all-vram" in body, \
        "the flag must refuse outside the placement it is registered for"


def test_flag_defaults_off_and_is_store_true():
    m = re.search(r'add_argument\("--compile-moe-tier"[^)]*\)', _SRC, re.S)
    assert m, "flag not registered"
    assert "store_true" in m.group(0), "must default OFF"


def test_receipts_record_the_flag():
    """An arm must not be able to claim a flag it did not use."""
    assert _SRC.count('"compile_moe_tier": bool(a.compile_moe_tier)') >= 2, \
        "both receipt shapes must record the flag's value"
