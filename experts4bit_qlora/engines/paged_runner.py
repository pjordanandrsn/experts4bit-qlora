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

**First-chunk prefill graph (opt-in, ``E4B_PAGED_PREFILL_GRAPH``).** :meth:`PagedModelRunner.enable_prefill_graph`
captures one CUDA graph of a ``T``-token prefill forward at positions ``0..T-1`` and serves every first chunk of
exactly ``T`` tokens from it. A first chunk reads no history: its attention runs over its own K/V, which
:meth:`.paged_attention.PagedAttentionContext.stage` keys by slot in Python. So one graph serves every slot, and after
a replay the graph's K/V outputs are staged for the request's slot. Later chunks, and first chunks of any other
length, run eagerly and are counted by reason. The A2000 census (``bench/prefill-graph-census-2026-10-04``) found no
host sync in the prefill forward under device grouping.
"""
from __future__ import annotations

import torch

from . import linear_state
from .paged_attention import PagedAttentionContext, set_context
from .scheduler import StepRunner


DEFAULT_BUCKETS = (1, 2, 4, 8, 16)


class PrefillGraphRefused(ValueError):
    """:meth:`PagedModelRunner.enable_prefill_graph` would not engage; ``why`` is the reason."""

    def __init__(self, why: str):
        super().__init__(f"prefill graph refused: {why}")
        self.why = why


#: ``config.layer_types`` values whose layers own K/V in the paged pool, and values that carry no per-sequence state
ATTENTION_LAYER_TYPES = frozenset({"full_attention", "sliding_attention", "attention"})
STATELESS_LAYER_TYPES = frozenset({"mlp", "moe"})
#: every layer type the runner can serve: attention (K/V in the paged pool), stateless, and linear attention (per-slot
#: state in :mod:`.linear_state`)
KEPT_LAYER_TYPES = ATTENTION_LAYER_TYPES | STATELESS_LAYER_TYPES | {linear_state.LINEAR_LAYER_TYPE}


def kv_layout_refusal(config) -> str | None:
    """Why the FP8 paged pool cannot hold this model's K/V, or None. ``Fp8PagedKV`` keeps one head dim per layer for
    both. Multi-head latent attention (DeepSeek-V2 / V3, Kimi: ``kv_lora_rank``) hands attention keys of
    ``qk_nope_head_dim + qk_rope_head_dim`` and values of ``v_head_dim``, against a ``head_dim`` that is the rotary
    width: DeepSeek-V2-Lite's are 192 and 128 against 64, and the first prompt's append refused them."""
    c = getattr(config, "text_config", None) or config
    if not getattr(c, "kv_lora_rank", None):
        return None
    k = (getattr(c, "qk_nope_head_dim", 0) or 0) + (getattr(c, "qk_rope_head_dim", 0) or 0)
    return (f"multi-head latent attention (kv_lora_rank {c.kv_lora_rank}): keys {k} and values "
            f"{getattr(c, 'v_head_dim', None)} wide per head, and the FP8 paged pool keeps one head dim for both")


def paged_state_refusal(model) -> str | None:
    """Why the paged server refuses ``model``, or None: :func:`kv_layout_refusal`, then the runner's rules for
    state-carrying layers, which it applies when it is built:
    - :func:`layer_plan`: a layer type it keeps no state for;
    - :func:`.linear_state.install`: linear-attention layers its per-slot pool cannot drive. transformers labels
      Mamba-style layers ``linear_attention`` too, and the pool drives Gated DeltaNet only.

    It reads the config and module classes only, so a planner can ask on a meta tree."""
    layout = kv_layout_refusal(getattr(model, "config", None))
    if layout:
        return layout
    types = linear_state.layer_types(getattr(model, "config", None))
    if not types:
        return None
    unknown = sorted(set(types) - KEPT_LAYER_TYPES)
    if unknown:
        return f"layer types {unknown} carry state the paged runner does not keep"
    want, driven = linear_state.driven_linear_layers(model)
    if want and want != driven:
        return (f"config.layer_types names linear-attention layers {want}, but the per-slot state pool drives "
                f"{driven} (Gated DeltaNet only); the others would run without their state")
    return None


def layer_plan(model, n_layers: int, n_slots: int):
    """``(attention layer indices, linear-state pool or None)`` for ``model``.

    A model without ``layer_types`` is all attention (every layer flushes K/V). A hybrid model's linear-attention
    layers keep per-slot state in :mod:`.linear_state` (installed here); any other state-carrying layer type is
    refused rather than run without its state."""
    types = linear_state.layer_types(getattr(model, "config", None))
    if not types:
        return list(range(n_layers)), None
    unknown = sorted(set(types) - KEPT_LAYER_TYPES)
    if unknown:
        raise NotImplementedError(f"layer types {unknown} carry state the paged runner does not keep; refusing")
    attn = [i for i, t in enumerate(types) if t in ATTENTION_LAYER_TYPES]
    pool = linear_state.install(model, n_slots) if linear_state.LINEAR_LAYER_TYPE in types else None
    return attn, pool


def decoder_layers(config) -> int:
    """The decoder's layer count (a composite's text config). It is the paged KV pool's layer count unless
    ``config.layer_types`` says some layers keep no K/V. It is NOT the MoE layer count: a model whose leading layers
    are dense (ERNIE-4.5's layer 0, DeepSeek-V2's first_k_dense_replace) still has attention in them, and a pool sized
    by its MoE layers indexed past its end at the first dense layer's append."""
    c = getattr(config, "text_config", None) or config
    return int(c.num_hidden_layers)


def kv_layers(model, n_layers: int) -> int:
    """How many layers a paged KV pool needs for ``model``: its attention layers when ``config.layer_types`` names
    them (a hybrid model's linear layers keep no K/V), else ``n_layers``."""
    types = linear_state.layer_types(getattr(model, "config", None))
    attn = [t for t in types if t in ATTENTION_LAYER_TYPES]
    return len(attn) if types and len(attn) < len(types) else n_layers


def kv_layer_map(attn_layers, pool_layers: int) -> dict:
    """``{model attention layer: KV pool layer}``. Empty (identity) when the pool has a layer for every index up to
    the last attention layer; compact when the pool holds exactly the attention layers; refused otherwise."""
    if not attn_layers or pool_layers > max(attn_layers):
        return {}
    if pool_layers == len(attn_layers):
        return {a: i for i, a in enumerate(attn_layers)}
    raise ValueError(f"the KV pool has {pool_layers} layers; the model's attention layers are {list(attn_layers)}: size "
                     f"it to {len(attn_layers)} (compact) or to at least {max(attn_layers) + 1} (one per layer index)")


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
                 gpu_only_prefill: bool = True, bulk_kv: bool = False):
        self.model = model
        self.kv = kv
        # E4B_PAGED_BULK_KV (serve_paged): a request's KV bookkeeping -- the slot reset at bind and free, the prompt's
        # flush into the pool, and (with decode graphs) the claim of every block the slot can reach -- in a launch count
        # independent of layers and blocks (Fp8PagedKV.reset_all_layers / append_prompt / claim_blocks). The pool, the
        # tables and the lengths it leaves are the per-layer forms'; off keeps the per-layer forms.
        self.bulk_kv = bool(bulk_kv)
        # which path each request's bookkeeping took (/health's kv_bookkeeping: a registered lane's engagement gate)
        self._kv_counts = {"flush_layers": 0, "flush_bulk": 0, "flush_bulk_fallback": 0, "ready_layers": 0, "ready_bulk": 0,
                           "ready_at_flush": 0}
        self.tracer = None           # engines.step_trace.StepTrace, set by serve_paged under E4B_PAGED_STEP_TRACE
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
        # a hybrid model's pool may hold its attention layers only: paged attention maps model layer -> pool layer
        self.ctx.layer_map = kv_layer_map(self.attn_layers, self.n_layers)
        # the pool layers attention appends to (all of them, unless a hybrid's pool keeps a row per model layer)
        self.pool_layers = [self.ctx.layer_map.get(a, a) for a in self.attn_layers]
        self._graphs = None          # bucket -> CUDAGraph | None (eager); see enable_decode_graphs
        self._prefill_graph = None   # see enable_prefill_graph
        self._pg_refused = None      # why an `auto` prefill graph stood down (see note_prefill_graph_refused)
        self._pg_stats = {"replays": 0, "eager_chunks": 0, "eager_reasons": {"later_chunk": 0, "short_chunk": 0}}
        self._la = None              # decode lookahead state (issue_decode / collect_decode), built at first use

    # ------------------------------------------------------------ intake --
    def bind(self, rid: int, slot: int, prompt) -> None:
        if slot in set(getattr(self.kv, "scratch", ()) or ()):
            raise ValueError(f"slot {slot} is a scratch slot (padding rows for the decode "
                             f"graphs); give the scheduler kv_slots = {self.kv.B}")
        self.slot_of[rid] = slot
        self.pos_of[rid] = 0
        self.tokens[rid] = list(prompt)
        if self._la is not None:
            self._la["dev_len"].pop(slot, None)     # the slot's device token is the previous tenant's
        self._reset(slot)            # a recycled slot carries no history
        if self.linear_state is not None:
            self.linear_state.reset(slot)

    def _reset(self, slot: int) -> None:
        if self.bulk_kv:
            self.kv.reset_all_layers(slot)
        else:
            self.kv.reset(slot)

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
                done = start + take >= len(self.tokens[rid])
                pg = self._prefill_graph
                if self.tracer is not None:
                    self.tracer.count("prefill_chunks")
                    self.tracer.count("prefill_tokens", take)
                    self.tracer.count("prefill_replays", int(pg is not None and start == 0 and take == pg["T"]))
                    self.tracer.mark("pf_prep", event=True)   # the GPU is idle here: forward device time starts
                if pg is not None and start == 0 and take == pg["T"]:
                    logits = self._replay_prefill_graph(rid, slot, take, done)
                else:
                    if pg is not None:
                        self._note_eager("later_chunk" if start else "short_chunk")
                    ids = torch.tensor(self.tokens[rid][start:start + take],
                                       dtype=torch.long, device=self.device)
                    pos = torch.arange(start, start + take, device=self.device)
                    prev = set_context(self.ctx)
                    try:
                        out = self.model(input_ids=ids[None],
                                         position_ids=pos[None], use_cache=False)
                    finally:
                        set_context(prev)
                    logits = out.logits
                if self.linear_state is not None:
                    self.linear_state.mark([slot])     # its linear layers now carry this prompt's state
                self.pos_of[rid] = start + take
                tr = self.tracer
                if tr is not None:
                    tr.mark("pf_forward", event=True)
                if start + take >= len(self.tokens[rid]):
                    # prompt complete: the staged bf16 K/V become the
                    # sequence's FP8 residency, once, here (attention layers only)
                    if self.bulk_kv:
                        self._flush_bulk(rid, slot)
                    else:
                        self._kv_counts["flush_layers"] += 1
                        for layer in self.pool_layers:
                            staged = self.ctx.flush(layer, slot)
                            if staged is None:
                                raise RuntimeError(
                                    f"layer {layer} staged no K/V for rid {rid} "
                                    f"— the attention implementation was not "
                                    f"bound for this forward")
                            k, v = staged
                            self.kv.append(layer, slot, k.contiguous(),
                                           v.contiguous())
                    if tr is not None:
                        tr.mark("pf_flush", event=True)
                    tok = int(logits[0, -1].argmax(-1))
                    if tr is not None:
                        tr.mark("pf_sync")
                    first[rid] = tok
                    self.tokens[rid].append(tok)
                    self.pos_of[rid] += 1
        finally:
            self.ctx.mode = "decode"
            self._mode(False)
        return first

    def _flush_bulk(self, rid: int, slot: int) -> None:
        """The prompt's staged K/V into the pool for every pool layer at once (``Fp8PagedKV.append_prompt``). With
        decode graphs the slot's every reachable block is claimed first, in the same single table write, so its first
        graphed decode claims nothing (:meth:`_ensure_graph_ready`)."""
        ks, vs = [], []
        for layer in self.pool_layers:
            buf = self.ctx.staging.pop((layer, slot), None)
            if buf is None:
                raise RuntimeError(f"layer {layer} staged no K/V for rid {rid} — the attention implementation was "
                                   f"not bound for this forward")
            ks.append(buf[0][0] if len(buf[0]) == 1 else torch.cat(buf[0]))
            vs.append(buf[1][0] if len(buf[1]) == 1 else torch.cat(buf[1]))
        if self._graphs is not None and slot not in self._graph_ready:
            self.kv.claim_blocks(slot, self.kv.blocks_per_seq - 1, self.pool_layers)
            self._graph_ready.add(slot)
            self._kv_counts["ready_at_flush"] += 1
        if not self.kv.append_prompt(slot, self.pool_layers, ks, vs):
            self._kv_counts["flush_bulk_fallback"] += 1     # append_prompt took its per-layer path (same bytes)
        self._kv_counts["flush_bulk"] += 1

    @torch.no_grad()
    def run_decode(self, rids):
        if not rids:
            return {}
        if self._la is not None and any(r["open"] for r in self._la["ring"]):
            raise RuntimeError("run_decode while a lookahead decode step is queued: collect_decode it first")
        if self._graphs is not None:
            return self._run_decode_bucketed(rids)
        tr = self.tracer
        self.ctx.mode = "decode"
        self.ctx.slots = [self.slot_of[r] for r in rids]
        ids = torch.tensor([[self.tokens[r][-1]] for r in rids],
                           dtype=torch.long, device=self.device)
        pos = torch.tensor([[self.pos_of[r] - 1] for r in rids],
                           dtype=torch.long, device=self.device)
        if tr is not None:
            tr.mark("dec_prep", event=True)
        prev = set_context(self.ctx)
        try:
            out = self.model(input_ids=ids, position_ids=pos,
                             use_cache=False)
        finally:
            set_context(prev)
        if tr is not None:
            tr.note(decode_rows=len(rids), bucket=None)
            tr.mark("dec_issue", event=True)
        toks = out.logits[:, -1].argmax(-1).tolist()
        if tr is not None:
            tr.mark("dec_sync")
        got: dict[int, int] = {}
        for rid, tok in zip(rids, toks):
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
        if self.linear_state is not None:
            # a hybrid's per-slot linear state is gathered and scattered through the bucket's device selector
            # (linear_state._bucket_selector); a graph cannot allocate the pool, so warm it first, then freeze it
            self._warm_linear_state(scratch[0])
            self.linear_state.frozen = True
        kv.graph_mode_init(seq=scratch[0], upto_tokens=kv.bt)
        self._la = None
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

    @torch.no_grad()
    def _warm_linear_state(self, slot: int) -> None:
        """Allocate every linear layer's pooled state before a graph is captured, by one eager one-token prefill on a
        scratch slot; its staged K/V and its state are discarded."""
        from experts4bit_qlora.engines.linear_state import linear_layers
        pool = self.linear_state
        want = linear_layers(self.model.config)
        if pool.allocated(want):
            return
        self._mode(True)
        self.ctx.mode, self.ctx.slots = "prefill", [slot]
        prev = set_context(self.ctx)
        try:
            self.model(input_ids=torch.zeros(1, 1, dtype=torch.long, device=self.device),
                       position_ids=torch.zeros(1, 1, dtype=torch.long, device=self.device), use_cache=False)
        finally:
            set_context(prev)
            self.ctx.drop(slot)
            self.ctx.mode, self.ctx.slots = "decode", []
            self._mode(False)
        pool.reset(slot)
        if not pool.allocated(want):
            raise RuntimeError(f"warming the linear-state pool left layers {sorted(set(want) - set(pool.conv))} "
                               "unallocated")

    def disable_decode_graphs(self) -> None:
        if self._la is not None and any(r["open"] for r in self._la["ring"]):
            raise RuntimeError("disable_decode_graphs while a lookahead decode step is queued")
        self._la = None
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
        if self.bulk_kv:
            self.kv.claim_blocks(slot, last, self.pool_layers)
            self._kv_counts["ready_bulk"] += 1
        else:
            for layer in self.pool_layers:
                self.kv._ensure_blocks(layer, slot, last)
            self._kv_counts["ready_layers"] += 1
        self._graph_ready.add(slot)

    def _run_decode_bucketed(self, rids):
        got: dict[int, int] = {}
        kv = self.kv
        tr = self.tracer
        for chunk in chunk_rows(rids, self._buckets[-1]):
            n = len(chunk)
            b = bucket_for(n, self._buckets)
            buf = self._bufs[b]
            slots = [self.slot_of[r] for r in chunk]
            if tr is not None:
                tr.count("first_decodes", sum(1 for s_ in slots if s_ not in self._graph_ready))
                tr.count("dec_pieces")             # > 1: the step ran as consecutive replays of the largest bucket
            for s_ in slots:
                self._ensure_graph_ready(s_)
            if tr is not None:
                tr.mark("dec_ready")
            pad = b - n
            all_slots = slots + kv.scratch[:pad]
            ids = [self.tokens[r][-1] for r in chunk] + [0] * pad
            pos = [self.pos_of[r] - 1 for r in chunk] + [0] * pad
            buf["ids"].copy_(torch.tensor(ids, dtype=torch.long).view(b, 1), non_blocking=True)
            buf["pos"].copy_(torch.tensor(pos, dtype=torch.long).view(b, 1), non_blocking=True)
            kv.graph_bucket_load(buf["st"], all_slots)
            if tr is not None:
                tr.mark("dec_prep", event=True)
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
            kv.graph_bucket_publish(buf["st"])     # E4B_KV_STEP_SELECT: the step's +1, every layer at once
            stats["rows"] += n
            stats["pad_rows"] += pad
            if tr is not None:
                tr.count("decode_rows", n)
                tr.note(bucket=b)
                tr.mark("dec_issue", event=True)
            toks = buf["tok"][:n].tolist()
            if tr is not None:
                tr.mark("dec_sync")
            for rid, slot, tok in zip(chunk, slots, toks):
                got[rid] = int(tok)
                self.tokens[rid].append(int(tok))
                self.pos_of[rid] += 1
                for layer in self.pool_layers:
                    kv._seen[layer][slot] += 1     # the host mirror of the device append
            if tr is not None:
                tr.mark("dec_mirror")
        ctrl = getattr(self, "slot_controller", None)
        if ctrl is not None:
            ctrl.on_decode_step()
        return got

    # ------------------------------------------------ decode lookahead (P118) --
    # run_decode reads a step's tokens back before it returns, so the GPU idles while the host mirrors them, the
    # scheduler emits and retires, and the next step is planned and its inputs copied in. The two entry points below
    # split that step for the serving scheduler (ContinuousScheduler(lookahead=True), E4B_PAGED_DECODE_LOOKAHEAD):
    # issue_decode enqueues a step without waiting, its input ids gathered on the device from a table of each slot's
    # newest token (which the previous issue wrote), and collect_decode reads a queued step's tokens back through a
    # pinned buffer and an event. Issuing step t+1 before collecting step t keeps a step queued while the host works.
    # The device inputs are run_decode's (same ids, positions, slots and padding), so the tokens are too.
    # run_decode keeps its contract and refuses while a lookahead step is queued.

    def _lookahead_state(self) -> dict:
        la = self._la
        if la is None:
            if self._graphs is None:
                raise RuntimeError("the decode lookahead needs bucketed decode graphs (enable_decode_graphs)")
            kv, dev = self.kv, self.device
            cuda = dev.type == "cuda"
            rows = int(kv.B) + self._buckets[-1]     # every piece of one step, padded, at disjoint offsets

            def host(n):
                t = torch.zeros(n, dtype=torch.long)
                return t.pin_memory() if cuda else t

            la = self._la = {
                # each slot's newest token on the device; the scratch slots' entries stay 0, the padding rows' ids
                "tok": torch.zeros(int(kv.B) + len(kv.scratch), dtype=torch.long, device=dev),
                # slot -> the request's token count with the token in "tok" counted (a mismatch resyncs from the host)
                "dev_len": {},
                "inflight": {},      # rid -> steps issued and not yet collected
                # host staging for two queued steps: slot ids, positions, tokens read back; reused only once collected
                "ring": [{"slots": host(rows), "pos": host(rows), "out": host(rows), "event": None, "open": False}
                         for _ in range(2)],
                "next": 0, "cuda": cuda,
            }
        return la

    @torch.no_grad()
    def issue_decode(self, rids) -> dict:
        """Enqueue one decode step for ``rids`` and return without waiting for it; :meth:`collect_decode` reads its
        tokens. A row's input is its newest token, which may still be in flight: its position counts the queued steps
        and its id is gathered on the device. A row whose newest token is known only on the host (its first decode, or
        after a host edit of ``tokens`` that changed its length) is written into the device table first. At most two
        steps may be queued. ``tokens``, ``pos_of`` and the KV host mirrors advance when a step is collected, as
        :meth:`run_decode` advances them."""
        if not rids:
            raise ValueError("issue_decode: no rows")
        if getattr(self, "slot_controller", None) is not None:
            raise RuntimeError("the decode lookahead does not run with a slot controller: it acts between decode "
                               "forwards, and under the lookahead one is always queued")
        la = self._lookahead_state()
        ring = la["ring"][la["next"]]
        if ring["open"]:
            raise RuntimeError("issue_decode: two decode steps are already queued; collect_decode the older one first")
        kv, tr = self.kv, self.tracer
        tok, dev_len, inflight = la["tok"], la["dev_len"], la["inflight"]
        for r in rids:                         # refuse before anything is queued
            k = inflight.get(r, 0)
            s_ = self.slot_of[r]
            if k and dev_len.get(s_) != len(self.tokens[r]) + k:
                raise RuntimeError(f"rid {r} has a decode step in flight but slot {s_}'s device token is not its "
                                   f"newest (tokens edited while a step was queued?)")
        if ring["event"] is not None:
            ring["event"].synchronize()        # collected already, so its copies have run: this never waits
        pieces, off = [], 0
        for chunk in chunk_rows(rids, self._buckets[-1]):
            n = len(chunk)
            b = bucket_for(n, self._buckets)
            buf = self._bufs[b]
            st = buf["st"]
            slots = [self.slot_of[r] for r in chunk]
            if tr is not None:
                tr.count("first_decodes", sum(1 for s_ in slots if s_ not in self._graph_ready))
                tr.count("dec_pieces")
            for s_ in slots:
                self._ensure_graph_ready(s_)
            if tr is not None:
                tr.mark("dec_ready")
            pad = b - n
            pos = []
            for r, s_ in zip(chunk, slots):
                k = inflight.get(r, 0)
                have = len(self.tokens[r]) + k      # the row's tokens once its queued steps land
                if dev_len.get(s_) != have:         # known on the host only (k == 0, checked above)
                    tok.narrow(0, s_, 1).fill_(self.tokens[r][-1])
                    dev_len[s_] = have
                pos.append(self.pos_of[r] - 1 + k)
            ph = ring["pos"][off:off + b]
            ph.copy_(torch.as_tensor(pos + [0] * pad, dtype=torch.long))
            kv.graph_bucket_load(st, slots + kv.scratch[:pad], staging=ring["slots"][off:off + b])
            torch.index_select(tok, 0, st["slot_l"], out=buf["ids"].view(b))
            buf["pos"].copy_(ph.view(b, 1), non_blocking=True)
            if tr is not None:
                tr.mark("dec_prep", event=True)
            g = self._graphs[b]
            stats = self.graph_stats[b]
            if g is not None:
                g.replay()
                stats["replays"] += 1
            else:
                self._bind(b, slots + kv.scratch[:pad])
                prev = set_context(self.ctx)
                try:
                    self._padded_step(b)
                finally:
                    set_context(prev)
                    kv.graph_bucket_unbind()
                stats["eager_steps"] += 1
            kv.graph_bucket_publish(st)
            tok.index_copy_(0, st["slot_l"][:n], buf["tok"][:n])     # the next step's ids, never via the host
            ring["out"][off:off + n].copy_(buf["tok"][:n], non_blocking=True)
            stats["rows"] += n
            stats["pad_rows"] += pad
            for r, s_ in zip(chunk, slots):
                inflight[r] = inflight.get(r, 0) + 1
                dev_len[s_] = len(self.tokens[r]) + inflight[r]
            if tr is not None:
                tr.count("decode_rows", n)
                tr.note(bucket=b, lookahead=1)
                tr.mark("dec_issue", event=True)
            pieces.append((list(chunk), slots, off, n))
            off += b
        if la["cuda"]:
            ev = torch.cuda.Event()
            ev.record()
            ring["event"] = ev
        ring["open"] = True
        la["next"] ^= 1
        return {"ring": ring, "pieces": pieces}

    def collect_decode(self, handle) -> dict:
        """Wait for a step :meth:`issue_decode` queued and return ``{rid: token}``, advancing ``tokens``, ``pos_of`` and
        the KV host mirrors as :meth:`run_decode` does. Steps are collected in the order they were issued."""
        ring = handle["ring"]
        if self._la is None or not ring["open"]:
            raise RuntimeError("collect_decode: this step was already collected")
        older = self._la["ring"][self._la["next"]]
        if older is not ring and older["open"]:
            raise RuntimeError("collect_decode: an older queued step must be collected first")
        tr = self.tracer
        if ring["event"] is not None:
            ring["event"].synchronize()
        if tr is not None:
            tr.mark("dec_sync")
        kv, inflight, got = self.kv, self._la["inflight"], {}
        for chunk, slots, off, n in handle["pieces"]:
            for rid, slot, t in zip(chunk, slots, ring["out"][off:off + n].tolist()):
                left = inflight[rid] - 1
                if left:
                    inflight[rid] = left
                else:
                    del inflight[rid]
                got[rid] = int(t)
                self.tokens[rid].append(int(t))
                self.pos_of[rid] += 1
                for layer in self.pool_layers:
                    kv._seen[layer][slot] += 1     # the host mirror of the device append
        ring["open"] = False
        if tr is not None:
            tr.mark("dec_mirror")
        return got

    # ------------------------------------------ first-chunk prefill graph --
    def enable_prefill_graph(self, T: int, *, warmup: int = 2, seed: int = 1689,
                             require_headroom: bool = False) -> dict:
        """Capture one CUDA graph of a ``T``-token first-chunk prefill forward and serve first chunks of exactly
        ``T`` tokens from it (see the module docstring). Refuses with :class:`PrefillGraphRefused` naming the reason:

        * no CUDA device;
        * ``hot_residency.DEVICE_GROUPING`` off: host grouping syncs inside the forward;
        * a linear-attention (hybrid) model: its per-slot state is mutated by prefill;
        * a capture that raises (a host sync inside the forward invalidates it);
        * a capture that does not stage K/V for every pool layer;
        * a failed startup check (:meth:`_check_prefill_graph`). The check runs only after the capture's scope has
          returned and the allocator has been churned, so a tensor the graph reads but nothing keeps fails it;
        * with ``require_headroom`` (``E4B_PAGED_PREFILL_GRAPH=auto``), too little memory left. The graph keeps its
          forward's working set in a private pool for its life (SC2b measured +3.3 GiB on Qwen3-30B-A3B int4), where
          an eager prefill only borrows it, and a later chunk still runs eagerly and needs that working set again. So
          ``auto`` stands down when the device's free memory after capture is below the pool's size. With ``bulk_kv``
          the bulk flush's bound (``Fp8PagedKV.append_prompt_peak_bytes`` at the slot's capacity) is added: it
          allocates on top of the pool.

        Returns :meth:`prefill_graph_stats`."""
        from . import hot_residency

        T = int(T)
        why = None
        if self.device.type != "cuda":
            why = f"needs a CUDA device, not {self.device}"
        elif T < 1:
            why = f"T must be positive, got {T}"
        elif not hot_residency.DEVICE_GROUPING[0]:
            why = ("device grouping is off: host grouping syncs inside the forward (serve_paged turns it on with "
                   "decode graphs at max_seqs > 1 on the all-vram placement)")
        elif self.linear_state is not None:
            why = "a hybrid model's linear-attention state is per slot, and prefill mutates it"
        if why:
            raise PrefillGraphRefused(why)
        vocab = int(self.model.get_output_embeddings().weight.shape[0])
        gen = torch.Generator().manual_seed(seed)
        prompts = [torch.randint(0, vocab, (1, T), generator=gen) for _ in range(2)]
        pg = self._capture_prefill_graph(T, prompts[0], warmup)
        self._check_prefill_graph(pg, prompts)
        pg["free_after_bytes"] = int(torch.cuda.mem_get_info(self.device)[0])
        if require_headroom and pg["pool_bytes"] <= 0:
            del pg
            torch.cuda.empty_cache()
            raise PrefillGraphRefused("memory: the graph's private pool could not be measured (no allocator segment "
                                      "carries its pool id), so the headroom rule cannot be applied")
        # with bulk KV bookkeeping, a prompt's flush allocates on top of the graph's private pool (nothing the
        # forward returned is reusable while the pool holds it): count its bound, at the slot's capacity
        pg["bulk_flush_bytes"] = (self.kv.append_prompt_peak_bytes(self.kv.blocks_per_seq * self.kv.bt, self.pool_layers)
                                  if self.bulk_kv and hasattr(self.kv, "append_prompt_peak_bytes") else 0)
        if require_headroom and pg["free_after_bytes"] < pg["pool_bytes"] + pg["bulk_flush_bytes"]:
            pool_mib, free_mib = pg["pool_bytes"] / 2**20, pg["free_after_bytes"] / 2**20
            flush_mib = pg["bulk_flush_bytes"] / 2**20
            del pg
            torch.cuda.empty_cache()
            raise PrefillGraphRefused(
                f"memory: the graph's private pool holds {pool_mib:.0f} MiB for its life and {free_mib:.0f} MiB is "
                f"free after capture; a later chunk's eager forward needs about that working set again"
                + (f", and the bulk KV flush up to {flush_mib:.0f} MiB more" if flush_mib else ""))
        self._prefill_graph = pg
        self._pg_refused = None
        self._pg_stats = {"replays": 0, "eager_chunks": 0, "eager_reasons": {"later_chunk": 0, "short_chunk": 0}}
        return self.prefill_graph_stats()

    _PG_KEY = -1      # the staging slot of the capture and the startup check: a first chunk reads no slot

    def _staged_under_key(self) -> dict:
        return {lay: (buf[0][-1], buf[1][-1]) for (lay, s), buf in self.ctx.staging.items() if s == self._PG_KEY}

    def _prefill_scope(self):
        """Enter prefill mode on the capture key; returns the restore callable."""
        ctx, saved_slots = self.ctx, self.ctx.slots
        self._mode(True)
        ctx.mode, ctx.slots = "prefill", [self._PG_KEY]
        prev = set_context(ctx)

        def restore():
            ctx.drop(self._PG_KEY)
            set_context(prev)
            ctx.slots, ctx.mode = saved_slots, "decode"
            self._mode(False)
        return restore

    def _capture_prefill_graph(self, T: int, first_ids, warmup: int) -> dict:
        """Warm, then capture. Everything the graph reads that was allocated outside the capture -- the input ids and
        the positions -- is returned with it and kept: a CUDA graph holds no reference to what it reads."""
        dev, ctx = self.device, self.ctx
        ids = first_ids.to(dev).clone()
        pos = torch.arange(T, device=dev)[None]
        restore = self._prefill_scope()
        try:
            with torch.no_grad():
                side = torch.cuda.Stream(dev)
                side.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(side):
                    for _ in range(max(1, warmup)):
                        ctx.drop(self._PG_KEY)
                        self.model(input_ids=ids, position_ids=pos, use_cache=False)
                torch.cuda.current_stream(dev).wait_stream(side)
                torch.cuda.synchronize(dev)
                ctx.drop(self._PG_KEY)
                g = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(g):
                        out = self.model(input_ids=ids, position_ids=pos, use_cache=False)
                except Exception as e:  # noqa: BLE001 -- any capture failure is a refusal, with its reason
                    raise PrefillGraphRefused(f"the {T}-token prefill forward did not capture "
                                              f"({type(e).__name__}: {str(e)[:300]})") from e
                staged = self._staged_under_key()
        finally:
            restore()
        if sorted(staged) != sorted(self.pool_layers):
            raise PrefillGraphRefused(f"the capture staged K/V for layers {sorted(staged)}, not the pool's "
                                      f"{sorted(self.pool_layers)}")
        return {"T": T, "graph": g, "ids": ids, "pos": pos, "logits": out.logits, "staged": staged,
                "pool_bytes": self._private_pool_bytes(g)}

    @staticmethod
    def _private_pool_bytes(graph) -> int:
        """Bytes of the device segments in ``graph``'s private memory pool, from the allocator's own snapshot. The
        growth of ``memory_reserved`` across a capture undercounts it -- to 0 once an earlier graph's freed segments
        are recycled (seen on the A2000) -- and an undercount would let the headroom rule wave through a graph that
        should stand down."""
        pid = tuple(graph.pool())
        return sum(int(s["total_size"]) for s in torch.cuda.memory_snapshot()
                   if tuple(s.get("segment_pool_id") or ()) == pid)

    @staticmethod
    def _churn_allocator(dev) -> None:
        """Hand freed blocks of many sizes out again, filled with a sentinel, then free them: a block the graph reads
        but nothing keeps now holds garbage."""
        held = [torch.full((n,), 1 << 40, dtype=torch.long, device=dev)
                for n in (16, 64, 128, 256, 512, 1024, 4096, 16384, 65536) for _ in range(48)]
        del held

    def _check_prefill_graph(self, pg: dict, prompts) -> None:
        """The startup check, run after the capture's scope has returned and the allocator has been churned. On two
        seeded prompts each replay must equal an eager forward bit for bit, in the logits and in every pool layer's
        staged K/V; the two prompts' eager logits must differ (a check whose prompts gave identical outputs could
        not tell a stale graph from a live one)."""
        dev, ctx = self.device, self.ctx
        self._churn_allocator(dev)
        refs = []
        restore = self._prefill_scope()
        try:
            with torch.no_grad():
                for p in prompts:
                    ctx.drop(self._PG_KEY)
                    o = self.model(input_ids=p.to(dev), position_ids=torch.arange(pg["T"], device=dev)[None],
                                   use_cache=False)
                    refs.append((o.logits.clone(),
                                 {lay: (k.clone(), v.clone()) for lay, (k, v) in self._staged_under_key().items()}))
        finally:
            restore()
        if torch.equal(refs[0][0], refs[1][0]):
            raise PrefillGraphRefused("the startup check's two prompts gave identical logits, so it could not tell a "
                                      "stale graph from a live one")
        self._churn_allocator(dev)
        for i in (1, 0):          # the second prompt first: the graph last saw the first
            pg["ids"].copy_(prompts[i])
            pg["graph"].replay()
            rl, rkv = refs[i]
            bad = [lay for lay in rkv if not (torch.equal(pg["staged"][lay][0], rkv[lay][0])
                                              and torch.equal(pg["staged"][lay][1], rkv[lay][1]))]
            if not torch.equal(pg["logits"], rl) or bad:
                d = (pg["logits"].float() - rl.float()).abs().max().item()
                raise PrefillGraphRefused(f"the startup check's replay of prompt {i} differs from the eager forward "
                                          f"(logits max abs {d:.3g}; K/V layers {bad})")

    def _replay_prefill_graph(self, rid: int, slot: int, take: int, done: bool):
        pg = self._prefill_graph
        # the same host-to-device copy the eager path makes, into the graph's input
        pg["ids"].copy_(torch.tensor(self.tokens[rid][:take], dtype=torch.long)[None])
        pg["graph"].replay()
        self.ctx.drop(slot)
        for layer, (k, v) in pg["staged"].items():
            # the next replay overwrites the graph's outputs: a prompt that continues keeps a copy. A prompt that
            # completes is flushed into the pool in this call, before any other replay.
            self.ctx.staging[(layer, slot)] = ([k], [v]) if done else ([k.clone()], [v.clone()])
        self._pg_stats["replays"] += 1
        return pg["logits"]

    def _note_eager(self, reason: str) -> None:
        s = self._pg_stats
        s["eager_chunks"] += 1
        s["eager_reasons"][reason] += 1

    def prefill_graph_stats(self) -> dict:
        """``{"status": "off"}``; ``{"status": "refused", "why"}`` when an ``auto`` graph stood down
        (:meth:`note_prefill_graph_refused`); or ``{"status": "on", "T", "replays", "eager_chunks", "eager_reasons",
        "pool_mib", "free_after_mib"}``: first chunks served from the graph, chunks that ran eagerly with the reason
        (``later_chunk``: a chunk after the first; ``short_chunk``: a first chunk of another length), the graph's
        private pool, and the device's free memory after capture. The startup check's replays are not counted."""
        pg = self._prefill_graph
        if pg is None:
            return {"status": "refused", "why": self._pg_refused} if self._pg_refused else {"status": "off"}
        s = self._pg_stats
        return {"status": "on", "T": pg["T"], "replays": s["replays"],
                "eager_chunks": s["eager_chunks"], "eager_reasons": dict(s["eager_reasons"]),
                "pool_mib": round(pg.get("pool_bytes", 0) / 2**20), "free_after_mib": round(pg.get("free_after_bytes", 0) / 2**20),
                "bulk_flush_mib": round(pg.get("bulk_flush_bytes", 0) / 2**20)}

    def note_prefill_graph_refused(self, why: str) -> None:
        """Record that an ``auto`` prefill graph stood down, and why: prefill stays eager, and the stats say so."""
        self._prefill_graph = None
        self._pg_refused = str(why)

    def disable_prefill_graph(self) -> None:
        self._prefill_graph = None

    def kv_bookkeeping_stats(self) -> dict:
        """``bulk`` (``E4B_PAGED_BULK_KV``) and how many requests took each path: prompt flushes per layer
        (``flush_layers``) or in bulk (``flush_bulk``), of which ``flush_bulk_fallback`` went through the bulk call but
        were written per layer inside it (:meth:`Fp8PagedKV.append_prompt`'s fallback); a graphed slot's block claims
        at its first decode, per layer (``ready_layers``) or in bulk (``ready_bulk``), or already made at its flush
        (``ready_at_flush``)."""
        return {"bulk": self.bulk_kv, **self._kv_counts}

    def free_slot(self, rid: int) -> None:
        if self._la is not None:
            if self._la["inflight"].get(rid):
                raise RuntimeError(f"free_slot: rid {rid} has a decode step in flight; collect it first")
            self._la["dev_len"].pop(self.slot_of.get(rid), None)
        slot = self.slot_of.pop(rid, None)
        if self._graphs is not None:
            self._graph_ready.discard(slot)
        self.pos_of.pop(rid, None)
        self.tokens.pop(rid, None)
        if slot is not None:
            self.ctx.drop(slot)      # any staging from an aborted prefill
            self._reset(slot)
            if self.linear_state is not None:
                self.linear_state.reset(slot)
