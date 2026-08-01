"""Block-scaled FP8 (e4m3) dequantization — the DENSE half of a DeepSeek-V4 checkpoint.

MoE checkpoints in this family split their storage two ways. The routed experts are
4-bit (:mod:`experts4bit_qlora.mxfp4`); *everything else* — attention projections, the
shared expert, the MTP block — ships as ``float8_e4m3fn`` weights carrying one scale per
``[128, 128]`` tile of the matrix. DeepSeek-V4-Flash is 140.25 GiB of routed experts and
8.40 GiB of this, so the dense side is what has to stay resident while the experts stream.

Two storage details differ from the MXFP4 lane and are easy to get backwards:

* The scale here is used as a **value**, not an exponent. ``dequantize_mxfp4`` wants the
  raw e8m0 *byte* (it feeds ``ldexp``); this function wants ``2**(byte-127)`` as a
  multiplier. Reading either one the other way is silent, not loud.
* Tiles are 2-D (``[128, 128]``), not 1-D groups of 32 along K, so the scale tensor is
  ``[ceil(out/128), ceil(in/128)]`` — one entry per tile, not per row.

The arithmetic matches the checkpoint's own ``inference/convert.py``::

    weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() * scale[:, None, :, None].float()

which is what :func:`dequantize_fp8_blocks` reproduces (see ``tests/test_fp8_blocks.py``,
which checks it against that expression on real V4 tensors).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "dequantize_fp8_blocks", "fp8_block_scale_shape", "Fp8BlockLinear",
    "convert_to_fp8_blocks",
]

BLOCK = 128


def fp8_block_scale_shape(weight_shape, block_size: int = BLOCK):
    """The ``[tiles_out, tiles_in]`` scale shape a ``weight_shape`` matrix implies."""
    out, inn = weight_shape
    return ((out + block_size - 1) // block_size, (inn + block_size - 1) // block_size)


def _scale_values(scale: torch.Tensor) -> torch.Tensor:
    """e8m0 -> fp32 multipliers, whichever way the tensor was stored.

    safetensors labels these ``F8_E8M0``; torch only grew ``float8_e8m0fnu`` in 2.7, and
    a checkpoint may equally hand over the raw exponent bytes as ``uint8``. Both reach
    the same fp32 values, but by different routes: the float8 dtype already *is* the
    value (so ``.float()`` is the conversion), while a uint8 holds the biased exponent
    (so ``2**(b-127)`` is). Casting the wrong one gives a scale that is off by 2**127
    with nothing raised.
    """
    if scale.dtype == torch.uint8:
        return torch.exp2((scale.to(torch.int32) - 127).float())
    if scale.dtype == torch.int8:  # same bytes, signed container
        return torch.exp2((scale.view(torch.uint8).to(torch.int32) - 127).float())
    return scale.float()


def dequantize_fp8_blocks(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int = BLOCK,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a ``[out, in]`` block-scaled FP8 matrix to a dense ``dtype`` tensor.

    Args:
        weight: ``[out, in]`` ``float8_e4m3fn`` (or the equivalent uint8/int8 storage).
        scale: ``[ceil(out/block), ceil(in/block)]`` — e8m0 as ``float8_e8m0fnu``, or the
            raw biased-exponent bytes as ``uint8``/``int8``. One entry per tile.
        block_size: tile edge; 128 for every DeepSeek-V4 tensor.
        dtype: output dtype.

    Returns:
        ``[out, in]`` in ``dtype``.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D [out, in]; got {tuple(weight.shape)}")
    if scale.ndim != 2:
        raise ValueError(f"scale must be 2-D [tiles_out, tiles_in]; got {tuple(scale.shape)}")
    want = fp8_block_scale_shape(weight.shape, block_size)
    if tuple(scale.shape) != want:
        raise ValueError(
            f"scale {tuple(scale.shape)} does not tile weight {tuple(weight.shape)} at "
            f"block_size={block_size}; expected {want}. A scale that merely BROADCASTS "
            "against the weight would dequantize silently and wrongly, so this is checked "
            "rather than left to broadcasting."
        )

    if weight.dtype in (torch.uint8, torch.int8):
        weight = weight.view(torch.float8_e4m3fn)
    out, inn = weight.shape
    s = _scale_values(scale.to(weight.device))
    # repeat_interleave rather than unflatten so a ragged final tile (out or in not a
    # multiple of block_size) stays correct; the trailing narrow is a no-op when they are.
    s = s.repeat_interleave(block_size, 0)[:out].repeat_interleave(block_size, 1)[:, :inn]
    return (weight.float() * s).to(dtype)


class Fp8BlockLinear(nn.Module):
    """A frozen ``nn.Linear`` whose weight stays FP8 in memory and is decoded per call.

    This is the *resident* form of a V4 dense weight. Holding e4m3 bytes plus one e8m0
    scale per ``[128, 128]`` tile costs ~1 byte per parameter; materializing the same
    weight in bf16 costs 2. Across DeepSeek-V4-Flash's dense side that is the difference
    between ~8.4 GiB and ~14 GiB resident — i.e. between fitting a 12 GB card and not.

    Decoding on every call sounds wasteful and is not: each of these projections is used
    exactly once per forward pass, so "dequantize on use" performs exactly one dequant
    per use. The transient bf16 copy is one weight at a time (~67 MB for V4's largest,
    ``wq_b`` at ``[32768, 1024]``), not the whole dense side at once.

    There is deliberately no FP8 matmul here. The A2000 this was built against is sm_86,
    which has no FP8 tensor cores; decoding to ``compute_dtype`` and running a normal
    matmul is what actually happens on that hardware, so it is what this module says it
    does. On sm_89+/Hopper a scaled-FP8 GEMM would be the faster path and would replace
    :meth:`forward`, not the storage.

    The weight is a non-trainable buffer: QLoRA trains adapters, never the frozen base.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        block_size: int = BLOCK,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        want = fp8_block_scale_shape(weight.shape, block_size)
        if tuple(scale.shape) != want:
            raise ValueError(
                f"scale {tuple(scale.shape)} does not tile weight {tuple(weight.shape)} "
                f"at block_size={block_size}; expected {want}."
            )
        if weight.dtype in (torch.uint8, torch.int8):
            weight = weight.view(torch.float8_e4m3fn)
        # e8m0 storage is normalized to raw exponent bytes at construction: uint8 moves
        # across devices and dtypes without incident, and `_scale_values` reads it back
        # unambiguously. Keeping the float8 dtype here would make every later `.to()`
        # a numeric conversion rather than a move.
        if scale.dtype not in (torch.uint8, torch.int8):
            scale = scale.view(torch.uint8) if scale.element_size() == 1 else scale
        self.out_features, self.in_features = weight.shape
        self.block_size = int(block_size)
        self.compute_dtype = compute_dtype
        self.register_buffer("weight", weight, persistent=True)
        self.register_buffer("scale", scale, persistent=True)
        self.register_buffer("bias", None if bias is None else bias.to(compute_dtype),
                             persistent=True)

    @classmethod
    def from_checkpoint(cls, weight, scale, bias=None, **kw) -> "Fp8BlockLinear":
        """Build straight from the two tensors a V4 shard holds for one projection."""
        return cls(weight, scale, bias, **kw)

    def dense(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Materialize the weight — for tests, probes, and the bf16-resident variant."""
        return dequantize_fp8_blocks(
            self.weight, self.scale, block_size=self.block_size,
            dtype=dtype or self.compute_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cd = self.compute_dtype if self.compute_dtype is not None else x.dtype
        w = dequantize_fp8_blocks(
            self.weight, self.scale, block_size=self.block_size, dtype=cd)
        return F.linear(x.to(cd), w, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, block_size={self.block_size}, "
                f"storage=float8_e4m3fn")


class _Fp8WeightMixin:
    """Serves ``.weight`` by decoding FP8 blocks, leaving the owner's forward untouched."""

    @property
    def weight(self):
        return dequantize_fp8_blocks(
            self.fp8_weight, self.fp8_scale,
            block_size=self.fp8_block_size, dtype=self.fp8_compute_dtype,
        )


_FP8_SUBCLASS_CACHE = {}


def convert_to_fp8_blocks(
    module: nn.Module,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int = BLOCK,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Give an existing module an FP8-resident weight **in place**, preserving its forward.

    Prefer this over swapping in an :class:`Fp8BlockLinear` whenever a module already
    exists, because a module's *class* can carry semantics its shape does not.
    DeepSeek-V4's ``self_attn.o_a_proj`` is the case that proves it: it is a
    ``DeepseekV4GroupedLinear``, which **subclasses ``nn.Linear``** but whose forward is
    block-diagonal — it views the weight as ``n_groups`` blocks and ``bmm``s them. It
    therefore looks exactly like a Linear by every duck-typed check, and replacing it with
    a real Linear silently turns a grouped projection into a dense one. (Observed: a
    `[5, 65536] @ [8192, 4096]` shape error one layer downstream, which is the *lucky*
    outcome — with different group counts the shapes can agree and the answer is just
    wrong.)

    So instead of replacing the module, this rebinds its class to a subclass whose
    ``weight`` is a decoding property. Any forward that reads ``self.weight`` — the stock
    ``nn.Linear`` one, the grouped one, anything else — keeps working unchanged, and the
    stored bytes stay FP8.
    """
    want = fp8_block_scale_shape(weight.shape, block_size)
    if tuple(scale.shape) != want:
        raise ValueError(
            f"scale {tuple(scale.shape)} does not tile weight {tuple(weight.shape)} at "
            f"block_size={block_size}; expected {want}."
        )
    if weight.dtype in (torch.uint8, torch.int8):
        weight = weight.view(torch.float8_e4m3fn)
    if scale.dtype not in (torch.uint8, torch.int8) and scale.element_size() == 1:
        scale = scale.view(torch.uint8)

    # Drop the dense parameter first: a data descriptor on the class shadows
    # `nn.Module.__getattr__`, so leaving it would keep a full-precision copy alive and
    # unreachable rather than freeing it.
    module._parameters.pop("weight", None)
    module.register_buffer("fp8_weight", weight, persistent=True)
    module.register_buffer("fp8_scale", scale, persistent=True)
    module.fp8_block_size = int(block_size)
    module.fp8_compute_dtype = compute_dtype

    base = type(module)
    cls = _FP8_SUBCLASS_CACHE.get(base)
    if cls is None:
        cls = type(f"Fp8{base.__name__}", (_Fp8WeightMixin, base), {})
        _FP8_SUBCLASS_CACHE[base] = cls
    module.__class__ = cls
    return module
