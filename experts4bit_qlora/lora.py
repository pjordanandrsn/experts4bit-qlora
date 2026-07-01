"""Trainable LoRA adapters over frozen bases.

* :class:`ExpertsLoRA` — per-expert low-rank adapters over a frozen :class:`Experts4bit` base
  (the QLoRA-on-fused-MoE piece).
* :class:`LoRALinear` — the usual per-projection LoRA over a frozen ``nn.Linear`` (attention).
* :func:`add_attention_lora` — wrap an OLMoE model's attention q/k/v/o projections in-place.

In both, ``B`` is zero-initialised so the adapted module is identical to the frozen base at
step 0 and only departs as the adapters train (standard LoRA initialisation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from . import Experts4bit


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
        # x: [n, in]; A: [r, in]; B: [out, r]  ->  [n, out]
        return self.scaling * F.linear(F.linear(x, A), B)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base
        compute_dtype = base.compute_dtype if base.compute_dtype is not None else hidden_states.dtype
        hidden_states = hidden_states.to(compute_dtype)

        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=base.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

        for expert_idx in expert_hit:
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            x = hidden_states[token_idx]

            # Frozen 4-bit base projection + trainable low-rank delta.
            gate_up_w = base._dequantize_expert(
                base.gate_up_proj,
                base.gate_up_absmax,
                base._gate_up_shape,
                expert_idx,
                compute_dtype,
            )
            proj = F.linear(x, gate_up_w) + self._lora(
                x, self.gate_up_lora_A[expert_idx], self.gate_up_lora_B[expert_idx]
            )

            if base.has_gate:
                gate, up = proj.chunk(2, dim=-1)
                current_hidden = base.act_fn(gate) * up
            else:
                current_hidden = base.act_fn(proj)

            down_w = base._dequantize_expert(
                base.down_proj,
                base.down_absmax,
                base._down_shape,
                expert_idx,
                compute_dtype,
            )
            current_hidden = F.linear(current_hidden, down_w) + self._lora(
                current_hidden,
                self.down_lora_A[expert_idx],
                self.down_lora_B[expert_idx],
            )

            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)


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
        return self.base(x) + self.scaling * F.linear(F.linear(x, self.lora_A), self.lora_B)


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
