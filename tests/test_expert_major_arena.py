"""Expert-major arena: fewer copies, identical bytes.

`_copy_rows_into` issues one `copy_` per (routed expert x tensor) -- ~32 copies
of ~2.7 MB per layer at 235B shapes, measured at 0.59x the pinned H2D ceiling
(finding #52). The stock arena (`E4B_OFFLOAD_ARENA=1`) is NAME-major, so
`home[n]` stays a strided view and that loop survives it. `E4B_OFFLOAD_ARENA=expert`
lays each expert's slabs contiguously so a routed stage is one copy per
(expert, dtype).

Two properties, and the first outranks the second:

1. **Bit-identity.** The rows that were copied must be exactly the rows the
   name-major path copies. A mis-strided view fails SILENTLY -- the kernel reads
   whatever is in the destination buffer -- so equality is the whole gate.
2. **The copy count must actually fall.** A coalescer that never engages passes
   every correctness check and reports "no measurable change" (the `enable_fast`
   failure: dead on every offloaded model until #22).
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import (  # noqa: E402
    Experts4bit,
    ExpertsLoRA,
    enable_expert_offload,
    enable_routed_staging,
)
from experts4bit_qlora import offload as offload_mod  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
N_EXP, HIDDEN, INTER = 16, 128, 192
_UNAVAIL = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="offload needs CUDA")


def _module(seed=0):
    torch.manual_seed(seed)
    gate_up = torch.randn(N_EXP, 2 * INTER, HIDDEN, dtype=DTYPE, device=DEVICE)
    down = torch.randn(N_EXP, HIDDEN, INTER, dtype=DTYPE, device=DEVICE)
    try:
        base = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=DTYPE)
    except _UNAVAIL as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable: {e}")
    return ExpertsLoRA(base, r=8, alpha=16, dtype=DTYPE).to(DEVICE)


def _staged_rows(arena_mode, ids):
    """Stage `ids` under the given arena mode; return {name: cpu copy of those rows}."""
    prev = os.environ.get("E4B_OFFLOAD_ARENA")
    os.environ["E4B_OFFLOAD_ARENA"] = arena_mode
    try:
        mod = _module()
        h = enable_expert_offload(mod, DEVICE, pin=True)
        enable_routed_staging([h])
        h._stage_ncopies_routed = 0
        h.stage_routed(ids)
        torch.cuda.synchronize()
        names = h._param_names + h._buffer_names
        out = {n: getattr(mod.base, n, None) for n in names}
        got = {}
        for n in names:
            t = out[n]
            if t is None:
                t = dict(mod.base.named_buffers()).get(n)
            got[n] = torch.stack([t[int(e)].detach().cpu().clone() for e in ids])
        return got, h._stage_ncopies_routed, len(names)
    finally:
        if prev is None:
            os.environ.pop("E4B_OFFLOAD_ARENA", None)
        else:
            os.environ["E4B_OFFLOAD_ARENA"] = prev


def test_expert_major_rows_are_bit_identical_to_name_major():
    ids = [1, 4, 9]
    ref, ref_copies, n_names = _staged_rows("1", ids)          # name-major
    got, got_copies, _ = _staged_rows("expert", ids)           # expert-major
    assert set(ref) == set(got)
    for n in ref:
        assert torch.equal(ref[n], got[n]), (
            f"{n}: expert-major staged different bytes than name-major. A "
            "mis-strided as_strided view fails silently here -- the kernel would "
            "read whatever was already in the destination buffer."
        )


def test_copy_count_actually_falls():
    """Both arms must be instrumented, and BOTH counts must be non-zero.

    The first version of this test read `expert < name or name == 0`. The control
    was uninstrumented, so it reported 0 copies and the `or` clause passed the
    test vacuously -- exactly the no-op-fast-path failure it was written to catch.
    """
    ids = [1, 4, 9]
    _, name_copies, n_names = _staged_rows("1", ids)
    _, expert_copies, _ = _staged_rows("expert", ids)
    assert name_copies > 0, "control arm is uninstrumented; the comparison is meaningless"
    assert expert_copies > 0, "expert-major arm issued no copies at all"
    # name-major: one copy per (expert x tensor); expert-major: per (expert x dtype)
    assert expert_copies == len(ids) * 2, f"expected {len(ids) * 2}, got {expert_copies}"
    assert name_copies == len(ids) * n_names, f"expected {len(ids) * n_names}, got {name_copies}"
    assert expert_copies < name_copies, (
        f"expert-major issued {expert_copies} copies vs name-major {name_copies}: "
        "the coalescer did not engage. A fast path that never fires passes every "
        "correctness check and reports no measurable change."
    )


def test_default_is_still_off():
    """Opt-in only: an unset env must not change the layout."""
    prev = os.environ.pop("E4B_OFFLOAD_ARENA", None)
    try:
        assert offload_mod._arena_enabled() is False
    finally:
        if prev is not None:
            os.environ["E4B_OFFLOAD_ARENA"] = prev


@pytest.mark.parametrize("val,expected", [("0", False), ("1", "name"), ("name", "name"),
                                          ("expert", "expert"), ("EXPERT", "expert")])
def test_arena_flag_tristate(val, expected):
    prev = os.environ.get("E4B_OFFLOAD_ARENA")
    os.environ["E4B_OFFLOAD_ARENA"] = val
    try:
        assert offload_mod._arena_enabled() == expected
    finally:
        if prev is None:
            os.environ.pop("E4B_OFFLOAD_ARENA", None)
        else:
            os.environ["E4B_OFFLOAD_ARENA"] = prev
