"""An arena that stores absmax as bf16 must stage BYTE-IDENTICAL fp32.

gnf4 can now bake absmax as bf16 — 11.1% of a Qwen3-30B row down to 5.6%, and
bitwise lossless for a bf16 checkpoint because absmax is the max of ``|w|`` over
a block and is therefore one of the source magnitudes. This is the consumer half:
``check_arena_geometry`` used to refuse ANY dtype difference between the module's
home and the arena segment, which would reject such an arena outright.

The relaxation is narrow on purpose and this file is mostly about its edges:

  * a WIDENING (bf16 segment -> fp32 home) is accepted and converts, so the
    kernel still receives the fp32 absmax its contract specifies;
  * every other mismatch is still refused, because "this arena was not baked
    from this model" is the far more common cause of a dtype difference and
    silently reinterpreting those bytes is the failure this check exists for.

The equivalence assertion is BITWISE. "Close" is what a silent precision loss
looks like, and the whole justification for bf16 storage is that there is none.
"""
import pytest
import torch

pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series modules")
pytest.importorskip("nvme_residency")
pytest.importorskip("bitsandbytes", reason="Experts4bit quantization needs bnb")

_wc = getattr(pytest.importorskip("nvme_residency"), "widening_casts", None)
if _wc is None:
    pytest.skip("grouped-nf4-gemm predates bf16 absmax (no widening_casts)",
                allow_module_level=True)

from nvme_arena import bake_expert_tensors, load_index  # noqa: E402
from nvme_residency import ColdTier, segment_into  # noqa: E402

from experts4bit_qlora.engines.nvme_train import (  # noqa: E402
    OFFLOAD_SEGMENTS,
    check_arena_geometry,
)
from test_nvme_train_residency import (  # noqa: E402
    E,
    KINDS,
    _module,
    _per_expert_stacks,
    _st_bytes,
)

_DT = {torch.uint8: "U8", torch.float32: "F32", torch.bfloat16: "BF16"}


def _bake_absmax_as(mod, tmp_path, name, absmax_dtype):
    """Relocation-bake the module's own stacks, storing absmax at the given dtype.

    Same shape as the suite's ``_bake``; the only difference is the cast on the
    absmax stacks, which is exactly the variable under test.
    """
    tensors = {}
    for attr, stack in _per_expert_stacks(mod).items():
        kind = OFFLOAD_SEGMENTS[attr]
        for e in range(mod.num_experts):
            t = stack[e].contiguous().cpu()
            if "absmax" in kind:
                t = t.to(absmax_dtype)
            raw = t.contiguous().view(torch.uint8).numpy().tobytes()
            tensors[f"model.layers.0.mlp.experts.{e}.{kind}"] = (
                tuple(t.shape), _DT[t.dtype], raw)
    snap = tmp_path / f"snap-{name}"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(_st_bytes(tensors))
    path = str(tmp_path / name)
    bake_expert_tensors(
        str(snap), path,
        name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
        kinds=KINDS, align=4096, log=lambda *a: None)
    return path, load_index(path)


def test_bf16_absmax_arena_is_accepted_by_the_geometry_check(tmp_path):
    mod = _module()
    _p, index = _bake_absmax_as(mod, tmp_path, "bf16.arena", torch.bfloat16)
    # Would have raised TypeError before the widening whitelist.
    geo = check_arena_geometry(mod, index, 0)
    assert geo, "geometry check returned nothing"


def test_bf16_absmax_arena_is_smaller(tmp_path):
    mod = _module()
    _p32, i32 = _bake_absmax_as(mod, tmp_path, "w32.arena", torch.float32)
    _p16, i16 = _bake_absmax_as(mod, tmp_path, "w16.arena", torch.bfloat16)
    am32 = sum(s["length"] for s in i32["segments"] if "absmax" in s["suffix"])
    am16 = sum(s["length"] for s in i16["segments"] if "absmax" in s["suffix"])
    assert am16 * 2 == am32
    assert i16["row_bytes"] < i32["row_bytes"]


def test_staged_absmax_is_bitwise_identical(tmp_path):
    """The claim that justifies the whole change."""
    mod = _module()
    p32, i32 = _bake_absmax_as(mod, tmp_path, "s32.arena", torch.float32)
    p16, i16 = _bake_absmax_as(mod, tmp_path, "s16.arena", torch.bfloat16)
    t32 = ColdTier(p32, hot_rows=E, pinned=False)
    t16 = ColdTier(p16, hot_rows=E, pinned=False)
    ids = list(range(E))
    checked = 0
    for attr, kind in OFFLOAD_SEGMENTS.items():
        if "absmax" not in kind:
            continue
        shape = tuple(_per_expert_stacks(mod)[attr][0].shape)
        a = torch.zeros(E, *shape, dtype=torch.float32)
        b = torch.zeros(E, *shape, dtype=torch.float32)
        segment_into(t32, i32, 0, ids, kind, a)
        segment_into(t16, i16, 0, ids, kind, b)
        assert torch.equal(a, b), f"{kind} diverged between f32 and bf16 storage"
        assert b.abs().sum() > 0, "staged nothing — the comparison would be vacuous"
        checked += 1
    assert checked == 2, f"expected both absmax segments, checked {checked}"


def test_homes_keep_the_MODULE_dtype_not_the_arenas(tmp_path):
    """The route test, and the one that matters most.

    `check_arena_geometry`'s return feeds `_build_homes`, and a home's dtype
    becomes the staging DESTINATION's dtype. Returning the arena's dtype would
    allocate a bf16 destination for a bf16 segment — at which point
    `segment_into` sees matching dtypes, takes its memcpy path, converts
    nothing, and the kernel is handed bf16 absmax where its contract says fp32.
    Wrong scales, finite numbers, no error anywhere.

    The earlier tests in this file all passed while that was broken, because
    they called `segment_into` with a destination THEY allocated as fp32 —
    testing the mechanism instead of the route.
    """
    mod = _module()
    _p, index = _bake_absmax_as(mod, tmp_path, "homes.arena", torch.bfloat16)
    geo = check_arena_geometry(mod, index, 0)
    for attr, kind in OFFLOAD_SEGMENTS.items():
        _shape, dtype = geo[attr]
        want = getattr(mod, attr).dtype
        assert dtype is want, (
            f"{attr}: geometry reports {dtype} but the module holds {want}; "
            "a destination built from this would skip the conversion")
        if "absmax" in kind:
            assert dtype is torch.float32, f"{attr} must stage as fp32"


def test_a_real_dtype_mismatch_is_still_refused(tmp_path):
    """Negative control. The widening must not have turned the dtype check off:
    a u8 segment feeding an fp32 home is still 'not baked from this model'."""
    mod = _module()
    _p, index = _bake_absmax_as(mod, tmp_path, "bad.arena", torch.float32)
    for seg in index["segments"]:
        if "absmax" in seg["suffix"]:
            seg["dtype"] = "U8"          # a mismatch nothing may widen
    with pytest.raises(TypeError, match="was not baked from this model"):
        check_arena_geometry(mod, index, 0)


def test_narrowing_is_never_allowed(tmp_path):
    """An fp32 segment into a bf16 destination would ROUND. The staging path
    promises the bytes it serves are the bytes that were baked, so it refuses."""
    mod = _module()
    p32, i32 = _bake_absmax_as(mod, tmp_path, "n32.arena", torch.float32)
    t32 = ColdTier(p32, hot_rows=E, pinned=False)
    attr = next(a for a, k in OFFLOAD_SEGMENTS.items() if "absmax" in k)
    shape = tuple(_per_expert_stacks(mod)[attr][0].shape)
    narrow = torch.zeros(E, *shape, dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        segment_into(t32, i32, 0, list(range(E)), OFFLOAD_SEGMENTS[attr], narrow)
