# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""How a checkpoint slice reaches the device.

`safe_open(..., device="cuda").get_tensor(key)` copies the whole slice to
the device in one operation. For a multi-gigabyte tensor that has been
observed to fail with ``CUDA error: invalid argument`` on some
driver/host pairs -- two of four rented RTX 5090 hosts, loading
Gemma-4-26B-A4B, always in the non-expert weight load (its per-layer
embedding is about 4 GB) and never in the expert quantisation that runs
first (e4b#344). Reading such a tensor into host memory and moving it
with a normal ``.to(device)`` lets torch chunk the transfer.

This module holds only the size decision, with no torch-adjacent imports
beyond torch itself, so it can be tested without loading the modelling
stack.
"""
from __future__ import annotations

import torch

__all__ = ["CPU_HOP_BYTES", "DTYPE_OF_STR", "too_big_for_one_copy"]

#: Read through host memory at or above this size. 2 GiB sits above every
#: tensor the families that already load produce, so their path is
#: unchanged, and below the ~4 GB tensor that fails.
CPU_HOP_BYTES = 2 * 1024 ** 3

#: safetensors dtype spellings -> torch dtypes, for the size estimate.
#: An unknown spelling falls back to one byte per element, which
#: UNDER-estimates and so keeps the fast path -- the conservative
#: direction, since the hop is the unusual behaviour.
DTYPE_OF_STR = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool,
}


def too_big_for_one_copy(handle, key: str, device) -> bool:
    """Whether ``key`` in this shard should be read via host memory.

    False for CPU destinations (there is no device copy to split) and for
    anything under :data:`CPU_HOP_BYTES`. A handle that cannot report a
    slice's shape or dtype also answers False: on ignorance the caller's
    existing path must run, so a metadata gap can never quietly reroute
    every tensor.
    """
    if getattr(device, "type", str(device)) == "cpu":
        return False
    try:
        sl = handle.get_slice(key)
        n = 1
        for d in sl.get_shape():
            n *= int(d)
        dt = DTYPE_OF_STR.get(sl.get_dtype(), torch.uint8)
        return n * torch.empty(0, dtype=dt).element_size() >= CPU_HOP_BYTES
    except Exception:                                    # noqa: BLE001
        return False
