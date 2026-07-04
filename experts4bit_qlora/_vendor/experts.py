# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Callable
import functools
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F_nn

import bitsandbytes.functional as F
from bitsandbytes.functional import QuantState

# Storage bit-width per supported quantization scheme. The 4-bit schemes pack two values per
# byte via quantize_4bit; the 8-bit schemes store one codebook index per byte via
# quantize_blockwise ("int8" = the blockwise dynamic map, NOT the LLM.int8() vectorwise
# scheme; "fp8" = an e4m3 float codebook). The 16-bit entries are unquantized passthrough
# storage — no codebook, no absmax — useful as a reference baseline or a per-layer opt-out.
_SCHEME_BITS = {
    "nf4": 4,
    "fp4": 4,
    "int8": 8,
    "fp8": 8,
    "bf16": 16,
    "fp16": 16,
}

_PASSTHROUGH_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _build_code(quant_type: str, device) -> Optional[torch.Tensor]:
    """The shared per-scheme codebook (`None` for 16-bit passthrough)."""
    if quant_type in ("nf4", "fp4"):
        return F.get_4bit_type(quant_type, device=device)
    if quant_type == "int8":
        code = F.create_dynamic_map()
    elif quant_type == "fp8":
        code = F.create_fp8_map(signed=True, exponent_bits=4, precision_bits=3, total_bits=8)
    else:
        return None
    return code.to(device) if device is not None else code


class _FrozenLinearRecomputeBackward(torch.autograd.Function):
    """``F.linear`` against a frozen dequantized weight, re-dequantizing it in backward.

    The weight produced by ``dequant_fn`` (a closure over the packed buffers) is an
    intermediate, not a Parameter, so a plain ``F.linear`` would stash it as a saved
    activation for the whole forward-to-backward window — one full-precision expert
    weight per projection per layer. Because the base is frozen, backward needs no
    gradient for the weight and only computes ``grad_output @ weight``; the weight can
    therefore be dropped after the forward matmul and re-dequantized on demand, keeping
    training memory independent of the number of experts held between forward and
    backward. Numerically identical to dequantize-then-``linear`` by construction — the
    forward *is* dequantize-then-linear; recomputation only changes what is saved, never
    what is computed.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, dequant_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        ctx.dequant_fn = dequant_fn
        return F_nn.linear(x, dequant_fn())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        if ctx.needs_input_grad[0]:
            try:
                weight = ctx.dequant_fn()
            except Exception as e:
                # The closure is a functools.partial over (packed, absmax, shape, expert_idx, ...);
                # a 0-element packed buffer means the expert was offload-evicted after the forward.
                packed = ctx.dequant_fn.args[0] if getattr(ctx.dequant_fn, "args", None) else None
                if isinstance(packed, torch.Tensor) and packed.numel() == 0:
                    raise RuntimeError(
                        "backward re-dequantization read an offload-evicted expert (0-element "
                        "placeholder). Offloaded training requires gradient checkpointing "
                        "(use_reentrant=False) so the recompute re-stages the layer before its "
                        "backward runs — non-checkpointed offload training is unsupported; see "
                        "experts4bit_qlora/offload.py."
                    ) from e
                raise
            grad_x = grad_output @ weight
        return grad_x, None


class ExpertsNbit(nn.Module):
    """Low-bit quantized storage for fused Mixture-of-Experts expert weights.

    A growing number of models in the Hugging Face ecosystem store their MoE expert
    weights as a single 3D ``nn.Parameter`` of shape ``[num_experts, out_features,
    in_features]`` (e.g. ``OlmoeExperts``, ``Qwen3MoeExperts``) rather than as a
    collection of ``nn.Linear`` layers. The default quantization walkers only replace
    ``nn.Linear`` modules, so these fused experts are silently skipped and stay in full
    precision — the dominant contribution to the model's memory footprint
    (see https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849).

    ``ExpertsNbit`` holds the two expert projections (``gate_up_proj`` and ``down_proj``)
    at a configurable storage precision selected by ``quant_type``:

      * ``"nf4"`` / ``"fp4"`` — 4-bit blockwise (two values per byte, 4x compression).
        :class:`Experts4bit` is the fixed-4-bit specialization of this class.
      * ``"int8"`` / ``"fp8"`` — 8-bit blockwise (one codebook index per byte, 2x
        compression at substantially higher fidelity than 4-bit). ``"int8"`` uses the
        blockwise dynamic map (`quantize_blockwise`'s default); it is *not* the
        LLM.int8() vectorwise scheme. ``"fp8"`` uses an e4m3 float codebook.
      * ``"bf16"`` / ``"fp16"`` — unquantized passthrough storage (no compression). A
        reference baseline and a per-layer opt-out inside an otherwise-quantized model.

    The packed weights are kept as plain ``nn.Parameter`` buffers and the per-expert
    quantization statistics (``absmax``) live on the module as ordinary buffers. This
    avoids bending :class:`Params4bit`'s tensor-subclass and device-movement machinery
    around a 3D stack, and it means the module serializes through the standard
    ``state_dict`` mechanism with no custom save/load hooks.

    The forward pass dequantizes a single expert at a time (a per-expert loop), mirroring
    the reference fused-experts forward. In training, the dequantized weight is not kept
    as a saved activation: it is re-dequantized on demand in backward (see
    :class:`_FrozenLinearRecomputeBackward`), for every storage scheme, so activation
    memory stays independent of the number of experts. Grouped-GEMM is intentionally left
    for future work.

    <Tip warning={true}>This feature is experimental and may change in future releases.</Tip>

    Args:
        num_experts (`int`): Number of experts in the layer.
        hidden_dim (`int`): Model hidden size (the ``in_features`` of ``gate_up_proj`` and
            the ``out_features`` of ``down_proj``).
        intermediate_dim (`int`): Expert intermediate size (the ``in_features`` of
            ``down_proj``).
        has_gate (`bool`, *optional*, defaults to `True`): Whether ``gate_up_proj`` packs a
            gate and an up projection (SwiGLU-style). When `False`, the projection is a
            plain up projection of size ``intermediate_dim``.
        activation (`Callable`, *optional*): The activation applied to the gate. Defaults
            to ``torch.nn.functional.silu`` (SwiGLU), matching OLMoE / Qwen3-MoE.
        compute_dtype (`torch.dtype`, *optional*): The dtype expert weights are
            dequantized to for the matmul. When `None`, the input's dtype is used.
        quant_type (`str`, *optional*, defaults to `"nf4"`): The storage scheme — one of
            ``nf4``, ``fp4``, ``int8``, ``fp8``, ``bf16``, ``fp16``.
        blocksize (`int`, *optional*, defaults to `64`): The quantization block size.
            Ignored for the 16-bit passthrough schemes.
        device (*optional*): The device for the (empty) packed buffers.

    Raises:
        ValueError: If ``quant_type`` is invalid, or if a quantized scheme is selected and
            ``hidden_dim`` / ``intermediate_dim`` is not divisible by ``blocksize``
            (required so per-expert quantization blocks never straddle an expert boundary).
    """

    _ALLOWED_QUANT_TYPES: tuple = tuple(_SCHEME_BITS)

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        has_gate: bool = True,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        compute_dtype: Optional[torch.dtype] = None,
        quant_type: str = "nf4",
        blocksize: int = 64,
        device=None,
    ):
        super().__init__()

        allowed = type(self)._ALLOWED_QUANT_TYPES
        if quant_type not in allowed:
            raise ValueError(f"quant_type must be one of {allowed}, got {quant_type!r}")

        self.bits = _SCHEME_BITS[quant_type]

        # Each expert is quantized independently, so an expert occupies a contiguous
        # `out_features * in_features` run of elements. Requiring the in_features dim to
        # be a multiple of the blocksize guarantees `out_features * in_features` is too,
        # so blocks tile each expert exactly and absmax reshapes cleanly to
        # [num_experts, blocks_per_expert]. (gate_up in_features is hidden_dim; down_proj
        # in_features is intermediate_dim.) Passthrough storage has no blocks.
        if self.bits < 16:
            for name, in_features in (("hidden_dim", hidden_dim), ("intermediate_dim", intermediate_dim)):
                if in_features % blocksize != 0:
                    raise ValueError(
                        f"{name} ({in_features}) must be divisible by blocksize ({blocksize}) "
                        "so per-expert quantization blocks align with expert boundaries"
                    )

        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.has_gate = has_gate
        self.act_fn = activation if activation is not None else F_nn.silu
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
        self.blocksize = blocksize

        gate_up_out = 2 * intermediate_dim if has_gate else intermediate_dim
        self._gate_up_shape = (gate_up_out, hidden_dim)
        self._down_shape = (hidden_dim, intermediate_dim)

        gate_up_numel = gate_up_out * hidden_dim
        down_numel = hidden_dim * intermediate_dim

        # Packed weights as plain (frozen) parameters. 4-bit: two values per byte; 8-bit:
        # one codebook index per byte; 16-bit: the weights themselves in the storage dtype.
        if self.bits == 16:
            storage_dtype = _PASSTHROUGH_DTYPES[quant_type]
            gate_up_storage = torch.empty(num_experts, gate_up_numel, dtype=storage_dtype, device=device)
            down_storage = torch.empty(num_experts, down_numel, dtype=storage_dtype, device=device)
        else:
            packed_per_value = 2 if self.bits == 4 else 1
            gate_up_storage = torch.empty(
                num_experts, gate_up_numel // packed_per_value, dtype=torch.uint8, device=device
            )
            down_storage = torch.empty(num_experts, down_numel // packed_per_value, dtype=torch.uint8, device=device)
        self.gate_up_proj = nn.Parameter(gate_up_storage, requires_grad=False)
        self.down_proj = nn.Parameter(down_storage, requires_grad=False)

        # Per-expert quantization scales (absent for passthrough storage — registering None
        # keeps attribute access uniform while leaving state_dict free of the keys).
        if self.bits < 16:
            self.register_buffer(
                "gate_up_absmax",
                torch.empty(num_experts, gate_up_numel // blocksize, dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "down_absmax",
                torch.empty(num_experts, down_numel // blocksize, dtype=torch.float32, device=device),
            )
        else:
            self.register_buffer("gate_up_absmax", None)
            self.register_buffer("down_absmax", None)

        # The codebook is identical for every expert and fully determined by quant_type,
        # so it is reconstructed at init rather than serialized.
        self.register_buffer("code", _build_code(quant_type, device), persistent=False)

    @classmethod
    def from_float(
        cls,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        has_gate: bool = True,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        compute_dtype: Optional[torch.dtype] = None,
        quant_type: str = "nf4",
        blocksize: int = 64,
    ) -> "ExpertsNbit":
        """Build an :class:`ExpertsNbit` by quantizing full-precision expert weights.

        Args:
            gate_up_proj (`torch.Tensor`): Shape ``[num_experts, gate_up_out, hidden_dim]``,
                where ``gate_up_out`` is ``2 * intermediate_dim`` when ``has_gate`` else
                ``intermediate_dim``.
            down_proj (`torch.Tensor`): Shape ``[num_experts, hidden_dim, intermediate_dim]``.

        Returns:
            `ExpertsNbit`: A module holding the quantized weights on the inputs' device.
        """
        if gate_up_proj.dim() != 3 or down_proj.dim() != 3:
            raise ValueError("gate_up_proj and down_proj must be 3D [num_experts, out, in] tensors")

        num_experts, gate_up_out, hidden_dim = gate_up_proj.shape
        intermediate_dim = down_proj.shape[2]

        # Cross-check the two stacks against the documented [num_experts, out, in] layout. Without
        # this, numel-preserving mistakes — a transposed down_proj, or the grouped-GEMM
        # [num_experts, in, out] convention some transformers checkpoints use — quantize cleanly
        # and only surface as a scrambled forward (or a cryptic reshape error inside dequantize).
        # A transposed stack whose 2D expert weight is square is inherently invisible to a shape
        # check; the value-level orientation check in tests/test_reference_parity.py covers that.
        expected_gate_up_out = 2 * intermediate_dim if has_gate else intermediate_dim
        if (
            down_proj.shape[0] != num_experts
            or down_proj.shape[1] != hidden_dim
            or gate_up_out != expected_gate_up_out
        ):
            raise ValueError(
                f"inconsistent expert stacks for has_gate={has_gate}: expected gate_up_proj "
                f"[num_experts, {'2*intermediate' if has_gate else 'intermediate'}, hidden] and "
                f"down_proj [num_experts, hidden, intermediate] (layout [num_experts, out, in]), "
                f"got gate_up_proj {tuple(gate_up_proj.shape)} vs down_proj {tuple(down_proj.shape)} "
                "— is one of them transposed / stored [num_experts, in, out]?"
            )

        module = cls(
            num_experts,
            hidden_dim,
            intermediate_dim,
            has_gate=has_gate,
            activation=activation,
            compute_dtype=compute_dtype if compute_dtype is not None else gate_up_proj.dtype,
            quant_type=quant_type,
            blocksize=blocksize,
            device=gate_up_proj.device,
        )

        gate_up_packed, gate_up_absmax = module._quantize_stack(gate_up_proj)
        down_packed, down_absmax = module._quantize_stack(down_proj)

        module.gate_up_proj = nn.Parameter(gate_up_packed, requires_grad=False)
        module.down_proj = nn.Parameter(down_packed, requires_grad=False)
        module.gate_up_absmax = gate_up_absmax
        module.down_absmax = down_absmax
        return module

    def _quantize_stack(self, weights: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Quantize a ``[num_experts, out, in]`` stack to packed storage + per-expert absmax.

        For 16-bit passthrough the stack is stored as-is (flattened per expert) and the
        returned absmax is `None`.
        """
        if self.bits == 16:
            storage_dtype = _PASSTHROUGH_DTYPES[self.quant_type]
            return weights.reshape(weights.shape[0], -1).to(storage_dtype).contiguous(), None

        packed = []
        absmax = []
        for e in range(weights.shape[0]):
            if self.bits == 4:
                q, state = F.quantize_4bit(
                    weights[e].contiguous(),
                    blocksize=self.blocksize,
                    compress_statistics=False,
                    quant_type=self.quant_type,
                )
            else:
                q, state = F.quantize_blockwise(
                    weights[e].contiguous(),
                    code=self.code,
                    blocksize=self.blocksize,
                )
            packed.append(q.reshape(-1))
            absmax.append(state.absmax.reshape(-1))
        return torch.stack(packed), torch.stack(absmax)

    def _dequantize_expert(
        self,
        packed: torch.Tensor,
        absmax: Optional[torch.Tensor],
        shape: tuple[int, int],
        expert_idx: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize a single expert's 2D weight ``[out, in]`` for the matmul."""
        if self.bits == 16:
            return packed[expert_idx].reshape(shape).to(dtype)
        if self.bits == 8:
            quant_state = QuantState(
                absmax=absmax[expert_idx],
                code=self.code,
                blocksize=self.blocksize,
                dtype=dtype,
            )
            return F.dequantize_blockwise(packed[expert_idx].reshape(shape), quant_state=quant_state)
        quant_state = QuantState(
            absmax=absmax[expert_idx],
            shape=torch.Size(shape),
            code=self.code,
            blocksize=self.blocksize,
            quant_type=self.quant_type,
            dtype=dtype,
        )
        # Restore the [packed, 1] layout quantize_4bit emits (and which keeps the
        # transpose back-compat shim — keyed on A.shape[0] == 1 — from firing).
        return F.dequantize_4bit(packed[expert_idx].reshape(-1, 1), quant_state=quant_state)

    def _project(self, packed, absmax, shape, expert_idx, x, compute_dtype):
        """One expert projection: dequantize + ``linear``, re-dequantizing in backward.

        Works identically for every storage scheme — the recompute closure is just
        :meth:`_dequantize_expert` — and never produces a gradient for the frozen storage.
        """
        dequant_fn = functools.partial(self._dequantize_expert, packed, absmax, shape, expert_idx, compute_dtype)
        return _FrozenLinearRecomputeBackward.apply(x, dequant_fn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        compute_dtype = self.compute_dtype if self.compute_dtype is not None else hidden_states.dtype
        hidden_states = hidden_states.to(compute_dtype)

        # Accumulate in float32 for numerical stability with bf16/fp16 routing weights.
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)

        with torch.no_grad():
            expert_mask = F_nn.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

        for expert_idx in expert_hit:
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            proj = self._project(
                self.gate_up_proj,
                self.gate_up_absmax,
                self._gate_up_shape,
                expert_idx,
                current_state,
                compute_dtype,
            )
            if self.has_gate:
                gate, up = proj.chunk(2, dim=-1)
                current_hidden = self.act_fn(gate) * up
            else:
                current_hidden = self.act_fn(proj)

            current_hidden = self._project(
                self.down_proj,
                self.down_absmax,
                self._down_shape,
                expert_idx,
                current_hidden,
                compute_dtype,
            )
            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden.to(final_hidden_states.dtype))

        # Return in the caller's dtype, not compute_dtype (hidden_states was rebound to the cast).
        return final_hidden_states.to(input_dtype)


class Experts4bit(ExpertsNbit):
    """4-bit quantized storage for fused Mixture-of-Experts expert weights.

    The fixed-4-bit (``nf4`` / ``fp4``) specialization of :class:`ExpertsNbit` — see the
    parent class for the storage design, constructor arguments, and forward semantics.
    Kept as a distinct class so the 4-bit contract (two packed values per byte, 4x
    compression) has a stable name.

    <Tip warning={true}>This feature is experimental and may change in future releases.</Tip>
    """

    _ALLOWED_QUANT_TYPES = ("nf4", "fp4")
