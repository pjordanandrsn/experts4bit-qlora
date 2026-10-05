# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Hand the C heap's freed pages back to the operating system.

glibc serves allocations below its mmap threshold (dynamic, up to 32 MiB on 64-bit) from the heap and keeps them after
``free``. A load that churns through gigabytes of 8-16 MiB host tensors therefore leaves that much resident for the
life of the process, although nothing references it. The int4 serving levers do exactly that: the expert repack reads
every expert projection in fp32, and the attention swap copies every projection to the host in fp32. Measured on
OLMoE-1B-7B (RTX A2000 host, both levers): 4.5 GB of anonymous memory after the build, 0.6 GB after one
``malloc_trim(0)``.
"""
from __future__ import annotations

import ctypes

_TRIM: list = []


def release_freed_host_heap() -> bool:
    """``malloc_trim(0)`` where the C library has it: True if pages were returned. False where it does not (macOS,
    musl), where there is nothing to do."""
    if not _TRIM:
        try:
            _TRIM.append(getattr(ctypes.CDLL("libc.so.6"), "malloc_trim", None))
        except OSError:
            _TRIM.append(None)
    fn = _TRIM[0]
    return bool(fn(0)) if fn is not None else False
