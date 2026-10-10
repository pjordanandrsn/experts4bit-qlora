"""Vendor first: what a ROCm build of torch means for the capability-gated paths.

A ROCm build of torch puts an AMD GPU behind ``torch.cuda`` and answers ``torch.cuda.get_device_capability`` with gfx
numbers, not sm numbers: MI300X (gfx942) is ``(9, 4)``. Every rule written for sm numbers passes there, such as
``>= (8, 9)`` for the fused FP8 KV append, although gfx942's fp8 is FNUZ rather than the OCP e4m3 the pool stores and
RDNA3 has no fp8 at all. No AMD card has run this package's suites, so each such gate asks :func:`rocm_refusal` first
and refuses in words that name AMD. The same helper lives in grouped-nf4-gemm's ``_triton_shim``; this copy keeps the
gates working without the optional kernel package.
"""

from __future__ import annotations


def rocm_version() -> str | None:
    """``torch.version.hip`` when this torch is a ROCm build, else ``None``."""
    import torch

    return getattr(torch.version, "hip", None) or None


def rocm_refusal(what: str) -> str | None:
    """Why ``what`` is refused on an AMD GPU, or ``None`` on any other torch build."""
    hip = rocm_version()
    if not hip:
        return None
    return (f"AMD GPU (ROCm {hip}): {what} is not yet supported on AMD; no AMD card has run the suites, and ROCm's "
            "compute capability is a gfx number, not an sm number")


def vendor_of(vendor: str | None) -> str:
    """``vendor`` checked (``"nvidia"`` or ``"amd"``), or the running torch's when ``None``."""
    if vendor is None:
        return "amd" if rocm_version() else "nvidia"
    if vendor not in ("nvidia", "amd"):
        raise ValueError(f"vendor must be 'nvidia', 'amd' or None, got {vendor!r}")
    return vendor
