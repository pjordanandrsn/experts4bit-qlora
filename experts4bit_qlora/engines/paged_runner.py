# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Model-driving :class:`~.scheduler.StepRunner` — hybrid Stage 2, Phase 9.

The scheduler decides what runs; this runs it. Everything structural
lives elsewhere by design — paging and attention in
:mod:`.paged_attention`, expert placement in the hybrid tier — so what
remains here is bookkeeping with three jobs the gate depends on:

* bind a sequence to a KV slot and keep its position clock, since the
  paged attention path owns the KV and the model is run with
  ``use_cache=False``;
* flip mixed mode around each regime — prefill chunks compute-bound on
  the GPU, decode bandwidth-bound on the hybrid tier;
* flush a completed prompt's staged K/V into the FP8 pool exactly once,
  which is the moment a sequence stops being a prefill and becomes a
  resident decoder.

Greedy sampling in v1, stated rather than implied: G9 measures
throughput and latency, and a sampler would add variance to both without
changing either mechanism.
"""
from __future__ import annotations

import torch

from .paged_attention import PagedAttentionContext, set_context
from .scheduler import StepRunner


class PagedModelRunner(StepRunner):
    def __init__(self, model, kv, *, device="cuda", eos_id: int | None = None,
                 gpu_only_prefill: bool = True):
        self.model = model
        self.kv = kv
        self.device = torch.device(device)
        self.eos_id = eos_id
        self.gpu_only_prefill = gpu_only_prefill
        self.ctx = PagedAttentionContext(kv=kv, slots=[], mode="decode")
        self.slot_of: dict[int, int] = {}
        self.pos_of: dict[int, int] = {}
        self.tokens: dict[int, list[int]] = {}
        self.tiers = [m._hot_residency for m in model.modules()
                      if hasattr(m, "_hot_residency")
                      and hasattr(m._hot_residency, "prefill_gpu_only")]
        self.n_layers = kv.L

    # ------------------------------------------------------------ intake --
    def bind(self, rid: int, slot: int, prompt) -> None:
        self.slot_of[rid] = slot
        self.pos_of[rid] = 0
        self.tokens[rid] = list(prompt)
        self.kv.reset(slot)          # a recycled slot carries no history

    def _mode(self, prefill: bool) -> None:
        if self.gpu_only_prefill:
            for t in self.tiers:
                t.prefill_gpu_only(prefill)

    # -------------------------------------------------------- StepRunner --
    @torch.no_grad()
    def run_prefill(self, chunks):
        first: dict[int, int] = {}
        self._mode(True)
        self.ctx.mode = "prefill"
        try:
            for rid, start, take in chunks:
                slot = self.slot_of[rid]
                self.ctx.slots = [slot]
                ids = torch.tensor(self.tokens[rid][start:start + take],
                                   dtype=torch.long, device=self.device)
                pos = torch.arange(start, start + take, device=self.device)
                prev = set_context(self.ctx)
                try:
                    out = self.model(input_ids=ids[None],
                                     position_ids=pos[None], use_cache=False)
                finally:
                    set_context(prev)
                self.pos_of[rid] = start + take
                if start + take >= len(self.tokens[rid]):
                    # prompt complete: the staged bf16 K/V become the
                    # sequence's FP8 residency, once, here
                    for layer in range(self.n_layers):
                        staged = self.ctx.flush(layer, slot)
                        if staged is None:
                            raise RuntimeError(
                                f"layer {layer} staged no K/V for rid {rid} "
                                f"— the attention implementation was not "
                                f"bound for this forward")
                        k, v = staged
                        self.kv.append(layer, slot, k.contiguous(),
                                       v.contiguous())
                    tok = int(out.logits[0, -1].argmax(-1))
                    first[rid] = tok
                    self.tokens[rid].append(tok)
                    self.pos_of[rid] += 1
        finally:
            self.ctx.mode = "decode"
            self._mode(False)
        return first

    @torch.no_grad()
    def run_decode(self, rids):
        if not rids:
            return {}
        self.ctx.mode = "decode"
        self.ctx.slots = [self.slot_of[r] for r in rids]
        ids = torch.tensor([[self.tokens[r][-1]] for r in rids],
                           dtype=torch.long, device=self.device)
        pos = torch.tensor([[self.pos_of[r] - 1] for r in rids],
                           dtype=torch.long, device=self.device)
        prev = set_context(self.ctx)
        try:
            out = self.model(input_ids=ids, position_ids=pos,
                             use_cache=False)
        finally:
            set_context(prev)
        got: dict[int, int] = {}
        for rid, tok in zip(rids, out.logits[:, -1].argmax(-1).tolist()):
            got[rid] = int(tok)
            self.tokens[rid].append(int(tok))
            self.pos_of[rid] += 1
        return got

    def free_slot(self, rid: int) -> None:
        slot = self.slot_of.pop(rid, None)
        self.pos_of.pop(rid, None)
        self.tokens.pop(rid, None)
        if slot is not None:
            self.ctx.drop(slot)      # any staging from an aborted prefill
            self.kv.reset(slot)
