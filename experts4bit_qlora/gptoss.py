"""GPT-OSS fused experts in 4-bit (NF4) — a faithful `GptOssExperts` replacement.

GPT-OSS's MoE experts differ from OLMoE/Qwen3 fused experts in ways that the
generic :class:`ExpertsNbit` forward does not model, so it gets its own subclass
rather than an activation swap:

* **per-projection biases** — ``gate_up_proj_bias`` and ``down_proj_bias``;
* **interleaved gate/up** — ``gate_up[..., ::2]`` is the gate, ``[..., 1::2]`` the up
  (vs. e4b's first-half/second-half convention);
* **clamped GLU** — ``gate.clamp(max=limit)``, ``up.clamp(±limit)``, then
  ``(up + 1) * (gate * sigmoid(alpha * gate))`` (vs. ``silu(gate) * up``);
* **input-major weight layout** on disk — ``gate_up_proj[e]`` is ``[hidden, 2*inter]``
  (used as ``x @ W``), ``down_proj[e]`` is ``[inter, hidden]``.

:meth:`GptOssExperts4bit.from_gptoss` takes the *dequantized* dense weights (see
:func:`experts4bit_qlora.mxfp4.dequantize_mxfp4`, verified bit-identical to the
released MXFP4 bytes) and applies the two load-time transforms — transpose to
e4b's ``[E, 2*inter, hidden]`` / ``[E, hidden, inter]`` and de-interleave the
gate/up rows into gate-block-then-up-block — before NF4-quantizing via
:meth:`Experts4bit.from_float`. The biases and the ``alpha``/``limit`` scalars
ride along untouched.

The loaded experts are **NF4** (a re-quantization of the exact released bytes);
the "exact released bytes" provenance lives one step earlier, at the dequant.
Trainable LoRA over this stack needs a GPT-OSS-aware adapter (the generic
:class:`ExpertsLoRA` assumes standard SwiGLU) — that is a separate change; this
module is the inference/probe/kernel-facing frozen expert.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ._vendor.experts import Experts4bit, ExpertsNbit


class _GptOssForwardMixin:
    """GPT-OSS expert forward (biases + clamped GLU) + the load-time builder.

    Applied over either storage base: :class:`Experts4bit` (nf4/fp4 — the
    production probe/kernel path) or :class:`ExpertsNbit` (bf16/fp16/int8/fp8 —
    e.g. the bf16-passthrough structural parity test). ``from_gptoss`` dispatches
    the base class by ``quant_type`` and rebinds to the matching subclass below.
    """

    @classmethod
    def from_gptoss(
        cls,
        gate_up_dense: torch.Tensor,   # [E, hidden, 2*inter] (input-major, interleaved gate/up)
        gate_up_bias: torch.Tensor,    # [E, 2*inter]        (interleaved)
        down_dense: torch.Tensor,      # [E, inter, hidden]  (input-major)
        down_bias: torch.Tensor,       # [E, hidden]
        *,
        alpha: float = 1.702,
        limit: float = 7.0,
        quant_type: str = "nf4",
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> "_GptOssForwardMixin":
        if gate_up_dense.ndim != 3 or down_dense.ndim != 3:
            raise ValueError("expected [E, *, *] dense stacks")
        E, H, twoI = gate_up_dense.shape
        if twoI % 2:
            raise ValueError(f"gate_up last dim {twoI} not even")

        # -> e4b layout [E, 2*inter, hidden], gate-block then up-block (so chunk(2) recovers them)
        gu = gate_up_dense.transpose(1, 2).contiguous()          # [E, 2I, H]
        gu = torch.cat([gu[:, 0::2, :], gu[:, 1::2, :]], dim=1)   # de-interleave rows
        gub = torch.cat([gate_up_bias[:, 0::2], gate_up_bias[:, 1::2]], dim=1)  # [E, 2I]
        dn = down_dense.transpose(1, 2).contiguous()             # [E, H, I]

        base_cls = Experts4bit if quant_type in ("nf4", "fp4") else ExpertsNbit
        obj = base_cls.from_float(
            gu, dn, has_gate=True, quant_type=quant_type, compute_dtype=compute_dtype
        )
        # same slots + storage; rebind to the GPT-OSS forward over the matching base
        obj.__class__ = GptOssExperts4bit if base_cls is Experts4bit else GptOssExpertsNbit
        obj.register_buffer("gate_up_bias", gub.to(compute_dtype), persistent=True)
        obj.register_buffer("down_bias", down_bias.to(compute_dtype), persistent=True)
        obj.alpha = float(alpha)
        obj.limit = float(limit)
        return obj

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor,   # [num_tokens, top_k]
        router_scores: torch.Tensor,    # [num_tokens, top_k]
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        cd = self.compute_dtype if self.compute_dtype is not None else input_dtype
        x = hidden_states.to(cd)
        out = torch.zeros_like(x, dtype=torch.float32)

        with torch.no_grad():
            mask = F.one_hot(router_indices, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

        for e in hit:
            pos, tok = torch.where(mask[e])
            cur = x[tok]
            gate_up = self._project(
                self.gate_up_proj, self.gate_up_absmax, self._gate_up_shape, e, cur, cd
            ) + self.gate_up_bias[e]
            gate, up = gate_up.chunk(2, dim=-1)
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            gated = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
            h = self._project(
                self.down_proj, self.down_absmax, self._down_shape, e, gated, cd
            ) + self.down_bias[e]
            h = h * router_scores[tok, pos, None]
            out.index_add_(0, tok, h.to(out.dtype))

        return out.to(input_dtype)


class GptOssExpertsNbit(_GptOssForwardMixin, ExpertsNbit):
    """GPT-OSS experts over the general N-bit storage base (bf16/fp16/int8/fp8)."""


class GptOssExperts4bit(_GptOssForwardMixin, Experts4bit):
    """GPT-OSS experts over the NF4/FP4 base — the production probe/kernel path."""
