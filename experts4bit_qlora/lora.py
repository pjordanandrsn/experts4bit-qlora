"""Trainable LoRA adapters over frozen bases.

* :class:`ExpertsLoRA` — per-expert low-rank adapters over a frozen :class:`Experts4bit` base
  (the QLoRA-on-fused-MoE piece).
* :class:`LoRALinear` — the usual per-projection LoRA over a frozen ``nn.Linear`` (attention).
* :func:`add_attention_lora` — wrap an OLMoE model's attention q/k/v/o projections in-place.

In both, ``B`` is zero-initialised so the adapted module is identical to the frozen base at
step 0 and only departs as the adapters train (standard LoRA initialisation).

The frozen base may be any :class:`ExpertsNbit` storage scheme (nf4/fp4 4-bit, int8/fp8 blockwise,
or bf16/fp16 passthrough); the adapters and forward are storage-agnostic. Training routes the base
through its recompute-in-backward projection (``ExpertsNbit._project``) for every scheme.

Inference (``no_grad``) additions, both default-on with env kill-switches for A/B:

* **Decode fast-path** (``E4B_DECODE_FASTPATH=0`` disables): a single-token forward skips the
  one-hot expert-mask machinery and the per-hit ``torch.where`` host syncs, iterating the token's
  ``top_k`` experts directly with 0-d tensor indices (no device->host transfer in the loop).
* **Fused 4-bit GEMV** (``E4B_INFER_GEMV=0`` disables): single-row base projections route through
  ``bnb.matmul_4bit``'s GEMV kernel, which reads the packed 4-bit weight directly instead of
  materializing the full dequantized expert — ~4x less memory traffic per expert at decode. Only
  under ``no_grad`` (so it is safe under offload too: no backward ever re-reads an evicted
  weight), only for **4-bit** bases (``int8``/``fp8``/``bf16``/``fp16`` have no GEMV kernel and use
  the dequantize path), and only when the kernel passes a correctness probe for our ``[packed, 1]``
  layout (:func:`_gemv_4bit_matches_dequant`).
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from . import Experts4bit


@functools.lru_cache(maxsize=None)  # tiny domain: the (quant_type, blocksize, dtype) combos actually used
def _gemv_4bit_matches_dequant(
    quant_type: str = "nf4", blocksize: int = 64, dtype: torch.dtype = torch.bfloat16
) -> bool:
    """Whether ``bnb.matmul_4bit``'s single-row GEMV branch is correct for the ``[packed, 1]``
    layout :class:`Experts4bit` stores, in exactly this quantization/compute configuration.

    The training-side probe (:func:`_matmul_4bit_supported`) uses a multi-row input, which takes
    ``matmul_4bit``'s dequantize-based ``MatMul4Bit.apply`` branch — it says nothing about the
    fused ``gemv_4bit`` kernel that a 1-row, ``requires_grad=False`` input dispatches to. This
    probe exercises exactly that decode shape, on a deliberately **non-square** weight so an
    orientation/transpose bug shows up as a shape mismatch or garbage rather than passing by
    symmetry, and is parameterized (bnb instantiates a separate GEMV kernel per compute dtype, and
    the codebook/blocksize are per-module) so the verdict covers exactly the configuration it
    gates. Probed once per configuration, lazily, on the first decode forward — GPU-only, so
    returns ``False`` without CUDA (the dequantize path is used there, as before).
    """
    try:
        if not torch.cuda.is_available():
            return False
        import bitsandbytes as bnb
        import bitsandbytes.functional as F_bnb

        dev = "cuda"
        out_f, in_f = 2 * blocksize, blocksize  # non-square [out, in], in divisible by blocksize
        w = torch.randn(out_f, in_f, dtype=dtype, device=dev)
        x = torch.randn(1, in_f, dtype=dtype, device=dev)  # the decode GEMV shape
        assert not x.requires_grad  # gemv_4bit dispatch requires a non-grad single-row input
        q, st = F_bnb.quantize_4bit(
            w.contiguous(), blocksize=blocksize, compress_statistics=False, quant_type=quant_type
        )
        qs = F_bnb.QuantState(
            absmax=st.absmax.reshape(-1),
            shape=torch.Size((out_f, in_f)),
            code=F_bnb.get_4bit_type(quant_type, device=dev),
            blocksize=blocksize,
            quant_type=quant_type,
            dtype=dtype,
        )
        ref = F.linear(x, F_bnb.dequantize_4bit(q.reshape(-1, 1), quant_state=qs))
        got = bnb.matmul_4bit(x, q.reshape(-1, 1), quant_state=qs)
        if got.shape != ref.shape:
            return False
        # Accept kernel-vs-dequantize rounding (a few %); reject a wrong-kernel result (error >> 1).
        return bool((got - ref).abs().max() / ref.abs().max().clamp_min(1e-6) < 0.05)
    except Exception:
        return False


def _decode_fastpath_enabled() -> bool:
    """Single-token decode fast-path kill-switch (``E4B_DECODE_FASTPATH=0``); default on."""
    return os.environ.get("E4B_DECODE_FASTPATH", "1") != "0"


def _infer_gemv_enabled() -> bool:
    """Inference 4-bit-GEMV route kill-switch (``E4B_INFER_GEMV=0``); default on."""
    return os.environ.get("E4B_INFER_GEMV", "1") != "0"


class ExpertsLoRA(nn.Module):
    """Per-expert LoRA adapters over a frozen :class:`Experts4bit` base.

    For each expert ``e``, the two frozen 4-bit projections are augmented with a trainable
    low-rank term ``scaling * (x @ A[e].T) @ B[e].T``:

      * ``gate_up``: ``A[e]`` is ``[r, hidden]``, ``B[e]`` is ``[gate_up_out, r]``
      * ``down``:    ``A[e]`` is ``[r, intermediate]``, ``B[e]`` is ``[hidden, r]``
    """

    def __init__(
        self,
        base: "Experts4bit",
        r: int = 8,
        alpha: int = 16,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r

        num_experts = base.num_experts
        gate_up_out, hidden = base._gate_up_shape  # [2*intermediate (or intermediate), hidden]
        _, intermediate = base._down_shape  # [hidden, intermediate]

        self.gate_up_lora_A = nn.Parameter(torch.empty(num_experts, r, hidden, dtype=dtype))
        self.gate_up_lora_B = nn.Parameter(torch.zeros(num_experts, gate_up_out, r, dtype=dtype))
        self.down_lora_A = nn.Parameter(torch.empty(num_experts, r, intermediate, dtype=dtype))
        self.down_lora_B = nn.Parameter(torch.zeros(num_experts, hidden, r, dtype=dtype))

        # A ~ small random, B = 0  =>  the initial LoRA delta is exactly zero.
        nn.init.normal_(self.gate_up_lora_A, std=1.0 / r)
        nn.init.normal_(self.down_lora_A, std=1.0 / r)

    def _lora(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # x: [n, in]; A: [r, in]; B: [out, r]  ->  [n, out]. Adapters may deliberately sit in a
        # different (typically higher, e.g. fp32) precision than the compute dtype; matmul requires
        # matching dtypes, so run the low-rank path in the adapter dtype and cast the delta back.
        # No-ops (no copies) when the dtypes already match.
        return (self.scaling * F.linear(F.linear(x.to(A.dtype), A), B)).to(x.dtype)

    def _use_infer_gemv(self, hidden_states: torch.Tensor) -> bool:
        """Whether *single-row* base projections in this forward may route through
        ``bnb.matmul_4bit`` (which dispatches them to the fused 4-bit GEMV kernel) instead of the
        base's dequantize path.

        Inference-only (``no_grad``): with no backward, nothing ever re-reads the packed weight
        after the forward, so this is safe **under offload too** (the eviction hazard
        :mod:`experts4bit_qlora.offload` guards against is a *backward* construct). The win is
        memory traffic: the GEMV reads the packed 4-bit weight directly instead of materializing
        the full dequantized expert for a 1-row matmul. Gated on the decode-shape correctness probe
        (:func:`_gemv_4bit_matches_dequant`) for this module's exact configuration; multi-row
        inference projections keep the dequantize path either way (equal cost, no kernel exposure).

        Bail-outs:

        * ``base.bits != 4`` — the fused GEMV kernel is 4-bit-only; ``int8``/``fp8`` blockwise and
          ``bf16``/``fp16`` passthrough have no ``matmul_4bit`` analogue, so they always decode via
          the base's dequantize path.
        * ``hidden_states.requires_grad`` — bnb dispatches to ``gemv_4bit`` only for a
          ``requires_grad=False`` input; a grad-carrying tensor (legal under ``no_grad``) would
          silently fall into ``MatMul4Bit.apply``'s multi-row branch, which is *wrong* for our
          ``[packed, 1]`` layout on bnb<0.50. Gate it to the always-correct dequantize path.
        * ``self.training`` — reentrant gradient checkpointing (``use_reentrant=True``) runs the
          *initial training forward* under ``torch.no_grad()``; requiring eval mode keeps that
          forward on the same kernels as its grad-enabled recompute."""
        if torch.is_grad_enabled() or self.training or not hidden_states.is_cuda or hidden_states.requires_grad:
            return False
        base = self.base
        if getattr(base, "bits", 4) != 4:
            return False  # gemv_4bit is 4-bit only; N-bit schemes decode via the dequantize path
        return _infer_gemv_enabled() and _gemv_4bit_matches_dequant(
            base.quant_type, base.blocksize, hidden_states.dtype
        )

    def _gemv_project(self, packed, absmax, shape, expert_idx, x, compute_dtype):
        """One expert's 4-bit projection via bnb's fused GEMV — the inference analogue of
        :meth:`Experts4bit._project`, reading the packed weight directly rather than dequantizing.

        Mirrors the base's ``[packed, 1]`` ``QuantState`` construction (see
        ``Experts4bit._dequantize_expert``) so the result matches the dequantize path within
        kernel rounding. Only reached under :meth:`_use_infer_gemv` (no_grad, eval, CUDA, 4-bit,
        single-row, probe-validated), so it never participates in autograd."""
        import bitsandbytes as bnb
        from bitsandbytes.functional import QuantState

        base = self.base
        quant_state = QuantState(
            absmax=absmax[expert_idx],
            shape=torch.Size(shape),
            code=base.code,
            blocksize=base.blocksize,
            quant_type=base.quant_type,
            dtype=compute_dtype,
        )
        return bnb.matmul_4bit(x, packed[expert_idx].reshape(-1, 1), quant_state=quant_state)

    def _base_project(self, packed, absmax, shape, expert_idx, x, compute_dtype, use_gemv):
        """Route one expert projection: fused GEMV when ``use_gemv`` (inference 4-bit fast path),
        else the base's dequantize+recompute path (correct for training and for non-gemv decode)."""
        if use_gemv:
            return self._gemv_project(packed, absmax, shape, expert_idx, x, compute_dtype)
        return self.base._project(packed, absmax, shape, expert_idx, x, compute_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base
        input_dtype = hidden_states.dtype
        compute_dtype = base.compute_dtype if base.compute_dtype is not None else hidden_states.dtype
        hidden_states = hidden_states.to(compute_dtype)

        use_infer_gemv = self._use_infer_gemv(hidden_states)
        # Decode fast-path: a single token routes to exactly its top_k experts, so the expert-mask
        # machinery below (one-hot over num_experts + a host-syncing torch.where per hit expert)
        # is pure overhead — iterate the token's experts directly instead. Requires eval mode so a
        # reentrant-checkpoint initial forward (no_grad, but training) keeps the mask path — and
        # with it the exact summation order — of its grad-enabled recompute.
        if (
            hidden_states.shape[0] == 1
            and not torch.is_grad_enabled()
            and not self.training
            and _decode_fastpath_enabled()
        ):
            return self._forward_decode(
                hidden_states, top_k_index, top_k_weights, compute_dtype, use_infer_gemv, input_dtype
            )

        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=base.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

        for expert_idx in expert_hit:
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            x = hidden_states[token_idx]

            # Inference 4-bit GEMV for any expert that received exactly one token (only 1-row inputs
            # dispatch to gemv_4bit); otherwise the base's dequantize+recompute path.
            use_gemv = use_infer_gemv and x.shape[0] == 1

            # Frozen base projection (dequantize/recompute, or fused GEMV) + trainable low-rank delta.
            proj = self._base_project(
                base.gate_up_proj,
                base.gate_up_absmax,
                base._gate_up_shape,
                expert_idx,
                x,
                compute_dtype,
                use_gemv,
            ) + self._lora(x, self.gate_up_lora_A[expert_idx], self.gate_up_lora_B[expert_idx])

            if base.has_gate:
                gate, up = proj.chunk(2, dim=-1)
                current_hidden = base.act_fn(gate) * up
            else:
                current_hidden = base.act_fn(proj)

            current_hidden = self._base_project(
                base.down_proj,
                base.down_absmax,
                base._down_shape,
                expert_idx,
                current_hidden,
                compute_dtype,
                use_gemv,
            ) + self._lora(
                current_hidden,
                self.down_lora_A[expert_idx],
                self.down_lora_B[expert_idx],
            )

            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden.to(final_hidden_states.dtype))

        # Same contract as the base primitive: output in the caller's dtype, not compute_dtype.
        return final_hidden_states.to(input_dtype)

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        compute_dtype: torch.dtype,
        use_infer_gemv: bool,
        input_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Single-token (``no_grad``) forward: loop the token's ``top_k`` experts directly.

        Semantically identical to the expert-mask loop in :meth:`forward` (same projections, same
        fp32 accumulation; only the summation *order* differs — routing order here vs. ascending
        expert id there, an ulp-level float difference). What it avoids per layer per token: the
        ``[n, k, E]`` one-hot mask, the ``num_experts``-sized hit scan, and ``2k`` host syncs —
        every index below stays a 0-d device tensor, so the loop enqueues kernels without ever
        blocking on a device->host transfer. With ``use_infer_gemv`` the two base projections are
        1-row by construction and take bnb's fused 4-bit GEMV (packed weight read directly,
        no dequantized-expert materialization).
        """
        base = self.base
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)

        for j in range(top_k_index.shape[1]):
            expert_idx = top_k_index[0, j]  # 0-d device tensor: indexes below without a host sync

            proj = self._base_project(
                base.gate_up_proj,
                base.gate_up_absmax,
                base._gate_up_shape,
                expert_idx,
                hidden_states,
                compute_dtype,
                use_infer_gemv,
            ) + self._lora(hidden_states, self.gate_up_lora_A[expert_idx], self.gate_up_lora_B[expert_idx])

            if base.has_gate:
                gate, up = proj.chunk(2, dim=-1)
                current_hidden = base.act_fn(gate) * up
            else:
                current_hidden = base.act_fn(proj)

            current_hidden = self._base_project(
                base.down_proj,
                base.down_absmax,
                base._down_shape,
                expert_idx,
                current_hidden,
                compute_dtype,
                use_infer_gemv,
            ) + self._lora(current_hidden, self.down_lora_A[expert_idx], self.down_lora_B[expert_idx])

            final_hidden_states += (current_hidden * top_k_weights[0, j]).to(final_hidden_states.dtype)

        # ``hidden_states`` here is already the compute-dtype cast; return the caller's dtype.
        return final_hidden_states.to(input_dtype)


class LoRALinear(nn.Module):
    """Frozen base ``nn.Linear`` + trainable low-rank adapter (for the attention projections)."""

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=dtype, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=dtype, device=dev))
        nn.init.normal_(self.lora_A, std=1.0 / r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same dtype rule as ExpertsLoRA._lora: adapters may sit in a different precision than the
        # activations; compute the delta in the adapter dtype, cast back (no-op when they match).
        delta = self.scaling * F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return self.base(x) + delta.to(x.dtype)


def add_attention_lora(model, r: int, alpha: int, dtype: torch.dtype) -> int:
    """Wrap each attention q/k/v/o projection with a trainable LoRA adapter (base stays frozen).

    Detects attention blocks **structurally** — any module exposing ``q_proj``/``k_proj``/``v_proj``/
    ``o_proj`` as ``nn.Linear`` — so it is architecture-agnostic (OLMoE, Qwen3-MoE, ...). Idempotent:
    once wrapped, a projection is a ``LoRALinear`` (not ``nn.Linear``), so it is not re-wrapped.
    """
    projs = ("q_proj", "k_proj", "v_proj", "o_proj")
    n = 0
    for mod in model.modules():
        if all(isinstance(getattr(mod, p, None), nn.Linear) for p in projs):
            for name in projs:
                setattr(mod, name, LoRALinear(getattr(mod, name), r, alpha, dtype))
                n += 1
    return n
