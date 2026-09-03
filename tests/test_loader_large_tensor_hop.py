# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Large tensors are read through host memory, not as one device-side
copy: `safe_open(device="cuda").get_tensor` on a multi-GB slice fails
with `CUDA error: invalid argument` on some driver/host pairs
(Gemma-4-26B-A4B's ~4 GB per-layer embedding, e4b#344)."""
import importlib.util
import os

import torch

# by PATH, not through the package: `experts4bit_qlora/__init__` imports
# the modelling stack, and a test that skipped when that is absent would
# be a test that never runs where it matters
_spec = importlib.util.spec_from_file_location(
    "e4b_shard_read",
    os.path.join(os.path.dirname(__file__), "..", "experts4bit_qlora",
                 "_shard_read.py"))
_sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sr)
CPU_HOP_BYTES = _sr.CPU_HOP_BYTES
too_big_for_one_copy = _sr.too_big_for_one_copy


class _Slice:
    def __init__(self, shape, dtype):
        self._shape, self._dtype = shape, dtype

    def get_shape(self):
        return self._shape

    def get_dtype(self):
        return self._dtype


class _Handle:
    def __init__(self, slices):
        self._slices = slices

    def get_slice(self, key):
        return self._slices[key]


def test_threshold_picks_the_host_hop_only_for_huge_tensors():
    h = _Handle({
        "embed": _Slice([262144, 7680], "BF16"),          # 4.0 GB -> hop
        "norm": _Slice([2880], "F32"),                    # 11 KB  -> direct
        "big_i8": _Slice([3 * 1024 ** 3], "U8"),          # 3 GB   -> hop
        "just_under": _Slice([1024 ** 3], "U8"),          # 1 GB   -> direct
    })
    assert too_big_for_one_copy(h, "embed", "cuda") is True
    assert too_big_for_one_copy(h, "big_i8", "cuda") is True
    assert too_big_for_one_copy(h, "norm", "cuda") is False
    assert too_big_for_one_copy(h, "just_under", "cuda") is False


def test_cpu_destination_never_hops():
    h = _Handle({"embed": _Slice([262144, 7680], "BF16")})
    assert too_big_for_one_copy(h, "embed", "cpu") is False
    assert too_big_for_one_copy(h, "embed", torch.device("cpu")) is False


def test_unreadable_metadata_keeps_the_existing_path():
    """A handle that cannot report shape/dtype must not change behaviour:
    answering True on ignorance would route every tensor through the host."""

    class _Broken:
        def get_slice(self, key):
            raise RuntimeError("no slice API in this safetensors build")

    assert too_big_for_one_copy(_Broken(), "anything", "cuda") is False


def test_unknown_dtype_underestimates_rather_than_hops():
    h = _Handle({"weird": _Slice([1024 ** 3], "F8_E4M3")})   # unknown spelling
    # 1 byte/element assumed -> 1 GB -> under the threshold -> fast path
    assert too_big_for_one_copy(h, "weird", "cuda") is False


def test_threshold_is_two_gibibytes():
    assert CPU_HOP_BYTES == 2 * 1024 ** 3
