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

**Bucketed decode graphs (#511, opt-in).** :meth:`PagedModelRunner.enable_decode_graphs`
captures one CUDA graph per batch bucket (default ``{1, 2, 4, 8, 16}``). A decode step pads
its active set up to the next bucket with the KV cache's scratch slots, replays that
bucket's graph and discards the padded rows. Each bucket is captured on SCRATCH slots only,
so neither the warm-up forwards nor the capture can advance a live sequence. A bucket whose
capture fails runs the SAME padded step eagerly, and says so once. The oracle for a replay
is that padded eager step, bitwise: a bf16 GEMM can round differently at a different row
count, so an unpadded eager step is not the same function.
"""
from __future__ import annotations

import torch

from . import linear_state
from .paged_attention import PagedAttentionContext, set_context
from .scheduler import StepRunner


DEFAULT_BUCKETS = (1, 2, 4, 8, 16)

#: ``config.layer_types`` values whose layers own K/V in the paged pool, and values that carry no per-sequence state
ATTENTION_LAYER_TYPES = frozenset({"full_attention", "sliding_attention", "attention"})
STATELESS_LAYER_TYPES = frozenset({"mlp", "moe"})


def layer_plan(model, n_layers: int, n_slots: int):
    """``(attention layer indices, linear-state pool or None)`` for ``model``.

    A model without ``layer_types`` is all attention (every layer flushes K/V). A hybrid model's linear-attention
    layers keep per-slot state in :mod:`.linear_state` (installed here); any other state-carrying layer type is
    refused rather than run without its state."""
    types = linear_state.layer_types(getattr(model, "config", None))
    if not types:
        return list(range(n_layers)), None
    known = ATTENTION_LAYER_TYPES | STATELESS_LAYER_TYPES | {linear_state.LINEAR_LAYER_TYPE}
    unknown = sorted(set(types) - known)
    if unknown:
        raise NotImplementedError(f"layer types {unknown} carry state the paged runner does not keep; refusing")
    attn = [i for i, t in enumerate(types) if t in ATTENTION_LAYER_TYPES]
    pool = linear_state.install(model, n_slots) if linear_state.LINEAR_LAYER_TYPE in types else None
    return attn, pool


def bucket_for(n: int, buckets) -> int:
    """The smallest bucket that holds ``n`` rows (``n`` <= the largest)."""
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"{n} rows exceed the largest bucket {buckets[-1]}")


def chunk_rows(rids, max_bucket: int):
    """Split an active set into consecutive chunks of at most ``max_bucket``."""
    rids = list(rids)
    return [rids[i:i + max_bucket] for i in range(0, len(rids), max_bucket)]


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
        # the layers whose K/V the pool holds, and the per-slot linear-attention state of a hybrid model
        n_slots = int(getattr(kv, "B", 0)) + len(getattr(kv, "scratch", ()) or ())
        self.attn_layers, self.linear_state = layer_plan(model, self.n_layers, n_slots)
        self._graphs = None          # bucket -> CUDAGraph | None (eager); see enable_decode_graphs

    # ------------------------------------------------------------ intake --
    def bind(self, rid: int, slot: int, prompt) -> None:
        if slot in set(getattr(self.kv, "scratch", ()) or ()):
            raise ValueError(f"slot {slot} is a scratch slot (padding rows for the decode "
                             f"graphs); give the scheduler kv_slots = {self.kv.B}")
        self.slot_of[rid] = slot
        self.pos_of[rid] = 0
        self.tokens[rid] = list(prompt)
        self.kv.reset(slot)          # a recycled slot carries no history
        if self.linear_state is not None:
            self.linear_state.reset(slot)

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
                if self.linear_state is not None:
                    self.linear_state.mark([slot])     # its linear layers now carry this prompt's state
                self.pos_of[rid] = start + take
                if start + take >= len(self.tokens[rid]):
                    # prompt complete: the staged bf16 K/V become the
                    # sequence's FP8 residency, once, here (attention layers only)
                    for layer in self.attn_layers:
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
        if self._graphs is not None:
            return self._run_decode_bucketed(rids)
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
        # production hook for the slot controller (engines.slot_controller):
        # fires between decode forwards, never concurrently with one
        ctrl = getattr(self, "slot_controller", None)
        if ctrl is not None:
            ctrl.on_decode_step()
        return got

    # ------------------------------------------ bucketed decode graphs --
    def enable_decode_graphs(self, buckets=DEFAULT_BUCKETS, *, capture: bool = True,
                             warmup: int = 2, verbose: bool = True) -> dict:
        """Capture one decode graph per bucket (#511); see the module docstring.

        Needs the KV cache built with ``scratch_slots >= max(buckets)`` and the
        fused batch KV append. The MoE engine must be capture-safe for a
        ``[b, 1]`` step (e.g. hot residency with device grouping, or the
        pipelined engine); a bucket whose capture raises is recorded, runs its
        padded step eagerly, and prints one ``DECODE_GRAPH ... EAGER`` line.
        ``capture=False`` runs every bucket eagerly on the same padded layout:
        the bitwise oracle for the replays. Returns ``{bucket: "graph" |
        "eager: <reason>"}``."""
        if self.linear_state is not None:
            raise NotImplementedError("decode graphs for a model with linear-attention layers: the per-slot state "
                                      "gather / scatter (engines/linear_state.py) is not captured yet; run eagerly")
        buckets = tuple(sorted({int(b) for b in buckets}))
        if not buckets or buckets[0] < 1:
            raise ValueError(f"buckets must be positive, got {buckets}")
        scratch = list(getattr(self.kv, "scratch", ()) or ())
        if len(scratch) < buckets[-1]:
            raise ValueError(
                f"decode graphs need scratch_slots >= the largest bucket ({buckets[-1]}); "
                f"the KV cache has {len(scratch)} -- build Fp8PagedKV(..., scratch_slots="
                f"{buckets[-1]})")
        kv, dev = self.kv, self.device
        kv.graph_mode_init(seq=scratch[0], upto_tokens=kv.bt)
        self._buckets = buckets
        self._bufs, self._graphs, self.graph_status = {}, {}, {}
        self.graph_stats = {b: {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0}
                            for b in buckets}
        self._graph_ready = set()
        for b in buckets:
            st = kv.graph_bucket(b)
            buf = {"ids": torch.zeros(b, 1, dtype=torch.long, device=dev),
                   "pos": torch.zeros(b, 1, dtype=torch.long, device=dev),
                   "tok": torch.zeros(b, dtype=torch.long, device=dev),
                   "st": st, "slots": scratch[:b]}
            self._bufs[b] = buf
            self._graphs[b] = None
            if not capture:
                self.graph_status[b] = "eager: capture=False"
                continue
            try:
                self._graphs[b] = self._capture_bucket(b, warmup)
                self.graph_status[b] = "graph"
            except Exception as e:           # a failed capture is data, not a crash
                torch.cuda.synchronize(dev)
                self.graph_status[b] = f"eager: {type(e).__name__}: {str(e)[:160]}"
            if verbose:
                s = self.graph_status[b]
                print(f"DECODE_GRAPH bucket={b} " + ("captured" if s == "graph" else f"EAGER ({s})"),
                      flush=True)
        kv.reset_scratch_lens()
        return dict(self.graph_status)

    def disable_decode_graphs(self) -> None:
        self._graphs = None
        self.kv.graph_bucket_unbind()

    def _padded_step(self, b: int) -> None:
        """The step a bucket's graph captures, on its static buffers."""
        buf = self._bufs[b]
        out = self.model(input_ids=buf["ids"], position_ids=buf["pos"], use_cache=False)
        buf["tok"].copy_(out.logits[:, -1].argmax(-1))

    def _bind(self, b: int, slots) -> None:
        self.kv.graph_bucket_bind(self._bufs[b]["st"], slots)
        self.ctx.mode = "decode"
        self.ctx.slots = list(slots)

    def _capture_bucket(self, b: int, warmup: int):
        buf, kv = self._bufs[b], self.kv
        slots = buf["slots"]                  # scratch only: nothing live moves
        buf["ids"].zero_()
        buf["pos"].zero_()
        kv.graph_bucket_load(buf["st"], slots)
        self._bind(b, slots)
        prev = set_context(self.ctx)
        try:
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(max(1, warmup)):
                    kv.reset_scratch_lens()
                    self._padded_step(b)
            torch.cuda.current_stream(self.device).wait_stream(side)
            torch.cuda.synchronize(self.device)
            kv.reset_scratch_lens()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._padded_step(b)
            torch.cuda.synchronize(self.device)
            return g
        finally:
            set_context(prev)
            kv.graph_bucket_unbind()
            kv.reset_scratch_lens()

    def _ensure_graph_ready(self, slot: int) -> None:
        """A graph step addresses KV blocks from the device table and never
        allocates, so every block a decoding slot can reach is claimed once,
        when the slot first decodes (released again by ``free_slot``)."""
        if slot in self._graph_ready:
            return
        last = self.kv.blocks_per_seq - 1
        for layer in range(self.n_layers):
            self.kv._ensure_blocks(layer, slot, last)
        self._graph_ready.add(slot)

    def _run_decode_bucketed(self, rids):
        got: dict[int, int] = {}
        kv = self.kv
        for chunk in chunk_rows(rids, self._buckets[-1]):
            n = len(chunk)
            b = bucket_for(n, self._buckets)
            buf = self._bufs[b]
            slots = [self.slot_of[r] for r in chunk]
            for s_ in slots:
                self._ensure_graph_ready(s_)
            pad = b - n
            all_slots = slots + kv.scratch[:pad]
            ids = [self.tokens[r][-1] for r in chunk] + [0] * pad
            pos = [self.pos_of[r] - 1 for r in chunk] + [0] * pad
            buf["ids"].copy_(torch.tensor(ids, dtype=torch.long).view(b, 1), non_blocking=True)
            buf["pos"].copy_(torch.tensor(pos, dtype=torch.long).view(b, 1), non_blocking=True)
            kv.graph_bucket_load(buf["st"], all_slots)
            g = self._graphs[b]
            stats = self.graph_stats[b]
            if g is not None:
                g.replay()
                stats["replays"] += 1
            else:
                self._bind(b, all_slots)
                prev = set_context(self.ctx)
                try:
                    self._padded_step(b)
                finally:
                    set_context(prev)
                    kv.graph_bucket_unbind()
                stats["eager_steps"] += 1
            stats["rows"] += n
            stats["pad_rows"] += pad
            toks = buf["tok"][:n].tolist()
            for rid, slot, tok in zip(chunk, slots, toks):
                got[rid] = int(tok)
                self.tokens[rid].append(int(tok))
                self.pos_of[rid] += 1
                for layer in range(self.n_layers):
                    kv._seen[layer][slot] += 1     # the host mirror of the device append
        ctrl = getattr(self, "slot_controller", None)
        if ctrl is not None:
            ctrl.on_decode_step()
        return got

    def free_slot(self, rid: int) -> None:
        slot = self.slot_of.pop(rid, None)
        if self._graphs is not None:
            self._graph_ready.discard(slot)
        self.pos_of.pop(rid, None)
        self.tokens.pop(rid, None)
        if slot is not None:
            self.ctx.drop(slot)      # any staging from an aborted prefill
            self.kv.reset(slot)
            if self.linear_state is not None:
                self.linear_state.reset(slot)
