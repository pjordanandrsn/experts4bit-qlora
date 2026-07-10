#!/usr/bin/env python3
"""Phase-0 access-pattern counter — COUNT distinct experts, do not time.

The access-pattern axis is router behavior, not storage physics: how fast the gathered-expert
union climbs toward ``n_experts_total`` as effective tokens grow is identical wherever the bytes
live. So it is characterized on owned hardware by counting distinct expert IDs — no bandwidth,
no rented metal.

Model-agnostic seam: hook the router **gate** ``nn.Linear`` (the one whose ``out_features`` ==
``n_experts_total``), capture its logits, take top-k, and record the distinct expert union per
layer per forward. Works uniformly for OLMoE / Qwen3-MoE / Gemma-4 / GraniteMoe without coupling
to any modeling-class internal tensor name. No e4b offload machinery required — this is the base
router only.
"""

from __future__ import annotations

import json
from collections import defaultdict

import torch


class ExpertAccessCounter:
    """Per-forward distinct-expert union per layer, tagged by (batch, seq) at close_step."""

    def __init__(self, n_experts_total: int, top_k: int):
        self.E = n_experts_total
        self.k = top_k
        self.per_layer: dict[int, set] = defaultdict(set)
        self.records: list[dict] = []
        self._layer_of: dict[int, int] = {}  # module id -> stable layer index

    def note_logits(self, module_id: int, logits: torch.Tensor) -> None:
        """logits: [..., E]. Take top-k over the last dim, add the distinct ids to this layer."""
        if module_id not in self._layer_of:
            self._layer_of[module_id] = len(self._layer_of)
        L = self._layer_of[module_id]
        flat = logits.reshape(-1, logits.shape[-1])
        topk = torch.topk(flat, self.k, dim=-1).indices  # [tokens, k]
        self.per_layer[L].update(int(e) for e in torch.unique(topk).tolist())

    def note_indices(self, module_id: int, top_k_index: torch.Tensor) -> None:
        """Alternative seam: caller already has the selected-expert index tensor."""
        if module_id not in self._layer_of:
            self._layer_of[module_id] = len(self._layer_of)
        L = self._layer_of[module_id]
        self.per_layer[L].update(int(e) for e in torch.unique(top_k_index).tolist())

    def close_step(self, step: int, batch: int, seq: int) -> None:
        eff = batch * seq
        for L, ids in self.per_layer.items():
            self.records.append(
                {
                    "step": step,
                    "batch": batch,
                    "seq": seq,
                    "eff_tokens": eff,
                    "layer": L,
                    "n_distinct": len(ids),
                    "read_fraction": len(ids) / self.E,
                }
            )
        self.per_layer.clear()

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")


def attach(model, counter: ExpertAccessCounter) -> int:
    """Register a forward hook on every router gate Linear (out_features == counter.E).

    Returns the number of gates hooked (== number of MoE layers). A gate is any ``nn.Linear``
    with no bias whose output dimension equals the total expert count — the standard MoE router
    across the four target families.
    """
    n = 0
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Linear)
            and module.out_features == counter.E
            and module.in_features != counter.E  # exclude odd square layers
        ):
            mid = id(module)

            def _hook(mod, inp, out, mid=mid):
                logits = out[0] if isinstance(out, tuple) else out
                counter.note_logits(mid, logits.detach())

            module.register_forward_hook(_hook)
            n += 1
    return n
