"""Optional 4-bit KV cache — ``pip install "experts4bit-qlora[fast]"``.

The residency engines budget weights; this budgets *context*. Measured KV cost
(grouped-nf4-gemm ``docs/context-budgets.md``): Qwen3-235B spends **188.0
KB/token**, so the stamped "235B on <=16 GB" figure — 15.2 GB at seq-512 — covers
roughly 5K tokens of context and nothing longer. Storing K/V as NF4 instead of
bf16 cuts that to 52.8 KB/token (3.56x; the fp32 blockwise absmax is the missing
0.44x), taking the 32K case from 5.88 GB to 1.65 GB.

Design, and why it is a cache object rather than an attention patch. Every
architecture's attention differs in ways that have nothing to do with storage
(Qwen3 QK-norm, gpt-oss attention sinks, Gemma-4's per-layer-type geometry), so
patching forwards would multiply arch-specific surface for a storage change.
Instead this stores the cache packed and hands back a dequantized view of **one
layer** when that layer runs. Peak VRAM is then
``packed_total + one_layer_bf16`` — for 94-layer Qwen3-235B at 32K that is
1.65 GB + 62 MB against 5.88 GB, i.e. essentially the full saving with stock
attention doing the maths.

What this deliberately does NOT do: fuse the dequant into the attention matmuls.
``grouped-nf4-gemm``'s ``nf4_kv`` kernels do that (never materializing even one
layer), but they are a decode-only path and cost 2.5-3x attention latency in
their v1, so wiring them in is a per-architecture performance decision, not the
default. This module is the memory dial.

Fidelity is measured at model level, not inferred from a fixture. Teacher-forced
on OLMoE-1B-7B over 1024 tokens of wikitext (4-bit weights held constant, so the
cache is the only variable):

===================  =======  ==============
config               ppl      argmax agree
===================  =======  ==============
fp16 cache            5.978   100%
K4 V4  (3.56x)        6.102   93.2%
K4 V16 (keys only)    6.061   94.6%
K16 V4 (values only)  5.991   97.3%
===================  =======  ==============

**Keys are the sensitive tensor.** Quantizing K alone costs +0.083 ppl; V alone
costs +0.013 — a ~6x asymmetry the other way round. Real keys carry per-channel
outliers that blow up a 64-element block's shared absmax; values do not. An
earlier iid fixture said the opposite (V-dominant) precisely because Gaussian
noise has no outliers to expose, which is why this table, not that fixture, is
what the dials below are set from.

So the useful operating point is the knee: **values-only** is 1.56x off the
cache for +0.2% perplexity, while the remaining 2x costs six times more. This
is NOT greedy-identical at any setting — see ``docs/`` for the exactness tier.

Usage::

    from experts4bit_qlora import NF4KVCache, kv_nf4_available
    assert kv_nf4_available()
    cache = NF4KVCache()
    out = model.generate(**inputs, past_key_values=cache, use_cache=True)
    print(cache.memory_bytes(), "bytes of KV")     # vs cache.memory_bytes(fp16=True)
"""
from __future__ import annotations

from typing import Any, Optional

import torch


def kv_nf4_available() -> bool:
    """True iff the packing/dequant primitives are importable and CUDA is up."""
    try:
        import nf4_grouped  # noqa: F401
        from nf4_kv import dequant_kv_ref, quantize_kv  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


def _kv_ops():
    from nf4_kv import dequant_kv_ref, quantize_kv

    return quantize_kv, dequant_kv_ref


class NF4KVCache:
    """A ``transformers``-compatible cache that stores keys/values as NF4.

    Implements the ``update``/``get_seq_length`` protocol the generation loop
    uses. Layers whose ``head_dim`` the 64-element quant blocksize cannot tile
    fall back to bf16 storage for that layer rather than failing the run — the
    per-layer decision is recorded in :attr:`layer_modes` so a caller can see
    exactly what was quantized.
    """

    def __init__(self, quantize_keys: bool = True, quantize_values: bool = True,
                 key_scaling: str = "per_token", group: int = 64,
                 keep_sink: int = 0, keep_recent: Optional[int] = None,
                 residence: str = "gpu", max_context: Optional[int] = None):
        # Separate switches because the error is NOT symmetric, and the
        # asymmetry favours keeping KEYS in bf16: measured +0.083 ppl for
        # K-only vs +0.013 for V-only. A caller tuning fidelity should reach
        # for quantize_keys=False first (1.56x, near-free) before paying for
        # the full 3.56x.
        self.quantize_keys = quantize_keys
        self.quantize_values = quantize_values
        if key_scaling not in ("per_token", "per_channel"):
            raise ValueError("key_scaling must be 'per_token' or 'per_channel'")
        # "per_channel" gives every channel its own scale, grouped over `group`
        # tokens. Costs exactly the same bytes (both store one fp32 scale per 64
        # quantized values), but MEASURED WORSE on OLMoE: +0.275 ppl against
        # per-token's +0.083, degrading monotonically as the group grows,
        # because key magnitude varies strongly across tokens and one loud token
        # then spoils the 63 sharing its group. Kept because it is correct, free
        # in bytes, and may land differently on an architecture that does not
        # rotate its keys -- not because it is recommended here. Default off.
        self.key_scaling = key_scaling
        self.group = group
        # Decode hands us one token per step, but per-channel scaling needs a
        # GROUP of tokens before its scales are meaningful — quantizing a lone
        # token per-channel would store one fp32 per value, worse than bf16. So
        # keys accumulate in a bf16 tail and flush a group at a time. The tail
        # is at most `group - 1` tokens, which is why it does not show up in the
        # footprint in any meaningful way.
        self._ktail: dict[int, torch.Tensor] = {}
        # Token-axis sparsity: retain the first `keep_sink` tokens plus the most
        # recent `keep_recent`, dropping the middle. This is the second,
        # independent axis of KV reduction -- quantization shrinks each token,
        # eviction removes tokens -- and unlike the rejected low-rank and
        # per-channel schemes it does not group across tokens, so it composes
        # with NF4 multiplicatively instead of competing for the same
        # information. Slicing the packed store along tokens is exactly the
        # access pattern the kernels' contiguity guard was written to preserve.
        # Eviction is EXPLICIT (call evict() between forwards), not automatic
        # inside update(). transformers builds the attention mask before the
        # layers run, so a cache that shrinks mid-forward contradicts the mask
        # it was already measured for -- concretely, trimming 640 to 516 during
        # the forward raises a shape error, and silently would be worse.
        self.keep_sink = keep_sink
        self.keep_recent = keep_recent
        # Tokens ever seen, tracked separately from tokens HELD. Positions must
        # follow the true count or evicted-cache RoPE goes wrong: new keys would
        # be rotated for position `held` while retained keys carry rotations
        # from their original positions, corrupting every relative distance.
        self._true: dict[int, int] = {}
        self._k: dict[int, Any] = {}
        self._v: dict[int, Any] = {}
        self._seen: dict[int, int] = {}
        self.layer_modes: dict[int, str] = {}
        # Third axis, after quantization (shrink each token) and eviction (drop
        # tokens): move the cache OFF the GPU entirely and stream one layer's
        # slice back per forward. GPU residency becomes one layer instead of the
        # whole cache -- for a 94-layer model that is ~47x less VRAM at the cost
        # of KV(context) bytes of PCIe traffic per decode step, which does NOT
        # amortize over batch the way streamed weights do (see
        # grouped-nf4-gemm docs/context-budgets.md finding #14 for when that
        # trade is worth taking; at batch 1 it usually is not).
        #
        # This is a RESIDENCE change, not a fidelity one: the same bytes are
        # produced, just held somewhere else. Quantization still runs on the
        # GPU, so the packed bytes are bit-identical to the resident path and
        # the property suite asserts exactly that.
        if residence not in ("gpu", "host"):
            raise ValueError("residence must be 'gpu' or 'host'")
        if residence == "host" and not max_context:
            raise ValueError(
                "residence='host' needs max_context: the pinned arena is sized "
                "up front because appending by cat would make prefill O(T^2) in "
                "host memcpy, and pinning a fresh chunk per step would cost an "
                "mlock syscall per layer per token.")
        self.residence = residence
        self.max_context = max_context
        self._arena: dict[tuple[str, int], Any] = {}
        self._device: Optional[torch.device] = None

    # ---- storage helpers -------------------------------------------------
    def _store(self, x: torch.Tensor, quantize: bool):
        """``x [B, H, T, D]`` -> packed tuple, or the raw tensor if not eligible."""
        from nf4_grouped import BLOCKSIZE

        if not quantize or x.shape[-1] % BLOCKSIZE != 0 or not x.is_cuda:
            return ("raw", x)
        quantize_kv, _ = _kv_ops()
        b, h, t, d = x.shape
        if b != 1:
            return ("raw", x)          # batch>1 not in the v1 layout
        p, a = quantize_kv(x[0].transpose(0, 1).contiguous())     # [T, H, D]
        return ("nf4", p, a, d)

    def _load(self, slot, dtype) -> torch.Tensor:
        """Materialize one layer. Under host residence this is the STREAM: the
        packed slice crosses PCIe here, and nowhere else."""
        if slot[0] == "raw":
            x = slot[1]
            return x if x.is_cuda or self._device is None else x.to(
                self._device, non_blocking=True)
        _, p, a, d = slot
        if not p.is_cuda and self._device is not None:
            # Pinned prefix views, so this is a real async DMA rather than a
            # bounce through a staging buffer. Dequant then runs on device, from
            # bytes identical to what the resident path holds.
            p = p.to(self._device, non_blocking=True)
            a = None if a is None else a.to(self._device, non_blocking=True)
        if a is None:                                   # "rawh": nothing to dequant
            return p.transpose(0, 1).unsqueeze(0).contiguous()
        _, dequant_kv_ref = _kv_ops()
        x = dequant_kv_ref(p, a, d, dtype=dtype)                  # [T, H, D]
        return x.transpose(0, 1).unsqueeze(0).contiguous()        # [1, H, T, D]

    # ---- host residence --------------------------------------------------
    def _arena_for(self, key, slot):
        """Pinned host buffers for one layer's K or V, allocated on first write.

        Sized to ``max_context`` rows and written at an offset. The alternative
        -- growing by ``torch.cat`` -- is O(T^2) host memcpy over a prefill, and
        pinning each new chunk instead costs one page-locking syscall per layer
        per step. The arena pays both costs once.
        """
        got = self._arena.get(key)
        if got is not None:
            return got
        cap = self.max_context
        if slot[0] == "raw":
            # TOKEN-MAJOR [cap, H, D], not the [1, H, cap, D] the resident path
            # uses. A prefix view of the latter slices dim 2, which is NOT
            # contiguous -- and ``is_pinned()`` still returns True for it, so the
            # only symptom is that the DMA falls off a cliff: measured
            # 0.09 GB/s against 0.95 on the same device, and a 17x cliff on the
            # full cache at 8K. Matching the packed store's token-major axis
            # makes the prefix contiguous and the copy a real DMA.
            x = slot[1]
            got = ("rawh",
                   torch.empty((cap, x.shape[1], x.shape[3]),
                               dtype=x.dtype).pin_memory(), None, x.shape[3])
        else:
            _, p, a, d = slot
            got = ("nf4",
                   torch.empty((cap,) + tuple(p.shape[1:]), dtype=p.dtype).pin_memory(),
                   torch.empty((cap,) + tuple(a.shape[1:]), dtype=a.dtype).pin_memory(),
                   d)
        self._arena[key] = got
        return got

    def _host_append(self, key, slot, used):
        """Copy a freshly quantized device-side slot into the arena at ``used``
        and return a VIEW of the occupied prefix.

        Returning a view is what keeps every other method residence-agnostic:
        the slot's row count still IS the held length, so ``_resync_held``,
        ``memory_bytes``, ``_evict_slot`` and ``_index_slot`` need no host
        branch at all.
        """
        arena = self._arena_for(key, slot)
        n = slot[1].shape[2] if slot[0] == "raw" else slot[1].shape[0]
        if used + n > self.max_context:
            raise ValueError(
                f"KV arena overflow: {used}+{n} tokens exceeds "
                f"max_context={self.max_context}. Size it for the longest "
                f"context this cache will see; it is not grown on demand.")
        if slot[0] == "raw":
            arena[1][used:used + n].copy_(slot[1][0].transpose(0, 1))  # -> [n,H,D]
            return ("rawh", arena[1][:used + n], None, arena[3])
        arena[1][used:used + n].copy_(slot[1])
        arena[2][used:used + n].copy_(slot[2])
        return ("nf4", arena[1][:used + n], arena[2][:used + n], slot[3])

    def _host_recompact(self) -> None:
        """Realign the arena after an eviction.

        Eviction builds a fresh tensor of the retained rows, which is no longer
        a view into the arena -- so the next append would write at an offset the
        data no longer occupies. Writing the survivors back to the front and
        re-viewing restores the invariant that the slot IS the arena prefix.
        """
        for kind, store in (("k", self._k), ("v", self._v)):
            for li, slot in list(store.items()):
                arena = self._arena.get((kind, li))
                if arena is None or slot[1].is_cuda:
                    continue
                # Both host kinds are token-major, so one path serves both.
                n = slot[1].shape[0]
                if slot[1].data_ptr() != arena[1].data_ptr():
                    arena[1][:n].copy_(slot[1])
                    if slot[2] is not None:
                        arena[2][:n].copy_(slot[2])
                store[li] = (slot[0], arena[1][:n],
                             None if slot[2] is None else arena[2][:n], slot[3])

    @staticmethod
    def _cat(slot_a, slot_b_tensor, quantize: bool, cache: "NF4KVCache"):
        """Append new tokens. Packed slots concatenate along the token axis with
        no dequantization of the existing cache — appending must stay O(new
        tokens), or a 32K cache would be re-quantized on every step."""
        if slot_a is None:
            return cache._store(slot_b_tensor, quantize)
        new = cache._store(slot_b_tensor, quantize)
        if slot_a[0] == "raw" or new[0] == "raw":
            old = slot_a[1] if slot_a[0] == "raw" else cache._load(slot_a, slot_b_tensor.dtype)
            nw = new[1] if new[0] == "raw" else cache._load(new, slot_b_tensor.dtype)
            return ("raw", torch.cat([old, nw], dim=2))
        return ("nf4", torch.cat([slot_a[1], new[1]], 0),
                torch.cat([slot_a[2], new[2]], 0), slot_a[3])

    def _store_perchannel(self, layer_idx: int, new_keys: torch.Tensor):
        """Append keys, flushing full groups into per-channel-scaled slots."""
        from nf4_kv import quantize_kv_perchannel

        tail = self._ktail.get(layer_idx)
        tail = new_keys if tail is None else torch.cat([tail, new_keys], dim=2)
        n_full = (tail.shape[2] // self.group) * self.group
        if n_full:
            flush = tail[:, :, :n_full]                      # [1, H, n_full, D]
            p, a = quantize_kv_perchannel(
                flush[0].transpose(0, 1).contiguous(), self.group)
            prev = self._k.get(layer_idx)
            if prev is None:
                self._k[layer_idx] = ("nf4pc", p, a, flush.shape[-1])
            else:
                self._k[layer_idx] = ("nf4pc", torch.cat([prev[1], p], 0),
                                      torch.cat([prev[2], a], 0), prev[3])
            tail = tail[:, :, n_full:]
        self._ktail[layer_idx] = tail

    def _load_keys(self, layer_idx: int, dtype) -> torch.Tensor:
        from nf4_kv import dequant_kv_ref

        parts = []
        slot = self._k.get(layer_idx)
        if slot is not None:
            x = dequant_kv_ref(slot[1], slot[2], slot[3], dtype=dtype,
                               token_group=self.group)
            parts.append(x.transpose(0, 1).unsqueeze(0).contiguous())
        tail = self._ktail.get(layer_idx)
        if tail is not None and tail.shape[2]:
            parts.append(tail)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)

    # ---- transformers Cache protocol ------------------------------------
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int, cache_kwargs: Optional[dict] = None):
        """Append this step's K/V and return the FULL cache for this layer."""
        dtype = key_states.dtype
        if self._device is None and key_states.is_cuda:
            self._device = key_states.device
        per_channel = (self.key_scaling == "per_channel" and self.quantize_keys
                       and key_states.is_cuda and key_states.shape[0] == 1
                       and key_states.shape[-1] % 64 == 0)
        if self.residence == "host":
            if per_channel:
                raise NotImplementedError(
                    "residence='host' does not support key_scaling='per_channel': "
                    "its bf16 tail is appended per step and flushed a group at a "
                    "time, which the fixed-offset arena has no representation "
                    "for. per_token keys (the default) stream fine.")
            used = self._seen.get(layer_idx, 0)
            # Quantize on the GPU, THEN move the packed bytes host-side -- doing
            # it the other way round would produce different bytes from the
            # resident path and quietly turn a residence change into a fidelity
            # change.
            self._k[layer_idx] = self._host_append(
                ("k", layer_idx), self._store(key_states, self.quantize_keys), used)
            self._v[layer_idx] = self._host_append(
                ("v", layer_idx), self._store(value_states, self.quantize_values), used)
        elif per_channel:
            self._store_perchannel(layer_idx, key_states)
            self._v[layer_idx] = self._cat(self._v.get(layer_idx), value_states,
                                           self.quantize_values, self)
        else:
            self._k[layer_idx] = self._cat(self._k.get(layer_idx), key_states,
                                           self.quantize_keys, self)
            self._v[layer_idx] = self._cat(self._v.get(layer_idx), value_states,
                                           self.quantize_values, self)
        # held reads back from the store so it cannot drift from reality;
        # true counts every token ever appended, including evicted ones
        vs = self._v[layer_idx]
        self._seen[layer_idx] = (vs[1].shape[2] if vs[0] == "raw" else vs[1].shape[0])
        self._true[layer_idx] = self._true.get(layer_idx, 0) + key_states.shape[2]
        self.layer_modes[layer_idx] = (
            f"K={'nf4pc' if per_channel else self._k[layer_idx][0]},"
            f"V={self._v[layer_idx][0]}")
        keys = (self._load_keys(layer_idx, dtype) if per_channel
                else self._load(self._k[layer_idx], dtype))
        return keys, self._load(self._v[layer_idx], dtype)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Tokens ever SEEN. transformers derives ``cache_position`` from this,
        and positions must be absolute so RoPE stays consistent between newly
        written keys and retained ones. The mask, which needs the tokens
        actually HELD, reads :meth:`get_query_offset` / :meth:`get_mask_sizes`
        instead — the two diverge exactly when eviction is on."""
        return self._true.get(layer_idx, self._seen.get(layer_idx, 0))

    def held_length(self, layer_idx: int = 0) -> int:
        """Tokens actually in the cache — what the mask must describe."""
        return self._seen.get(layer_idx, 0)

    @staticmethod
    def _evict_slot(slot, sink: int, recent: int):
        """Keep [:sink] + [-recent:] along the token axis."""
        if slot[0] == "raw":
            x = slot[1]
            if x.shape[2] <= sink + recent:
                return slot
            return ("raw", torch.cat([x[:, :, :sink], x[:, :, -recent:]], dim=2))
        kind, p, a, d = slot
        if p.shape[0] <= sink + recent:
            return slot
        return (kind, torch.cat([p[:sink], p[-recent:]], 0),
                None if a is None else torch.cat([a[:sink], a[-recent:]], 0), d)

    def evict(self) -> None:
        """Drop the middle of the cache, keeping sinks + recent. Call BETWEEN
        forwards — never during one, see __init__."""
        if self.keep_recent is None:
            return
        for store in (self._k, self._v):
            for li, slot in list(store.items()):
                store[li] = self._evict_slot(slot, self.keep_sink, self.keep_recent)
        if self.residence == "host":
            self._host_recompact()
        self._resync_held()

    @staticmethod
    def _index_slot(slot, idx: torch.Tensor):
        """Keep an arbitrary set of token slots, in the given order."""
        if slot[0] == "raw":
            return ("raw", slot[1].index_select(2, idx).contiguous())
        kind, p, a, d = slot
        return (kind, p.index_select(0, idx).contiguous(),
                None if a is None else a.index_select(0, idx).contiguous(), d)

    def evict_index(self, keep) -> None:
        """Retain an arbitrary set of token slots. Call BETWEEN forwards.

        ``keep`` is either a 1-D index tensor applied to every layer, or a
        mapping ``layer_idx -> 1-D index tensor``; indices address the CURRENT
        held axis, so a policy that just ran a forward can select straight from
        what it observed. Ascending order is expected — the token axis carries
        no positions of its own (RoPE is already baked into the stored keys, and
        ``get_seq_length`` counts what was SEEN), so reordering here would only
        scramble the relationship between the cache and the scores a caller
        holds alongside it.

        :meth:`evict` is the sink+recent special case. This is the general one,
        which is what an importance-based policy needs: H2O-style selection by
        accumulated attention keeps a set that is neither a prefix nor a suffix.
        Mechanism only — no policy lives here, deliberately, because the one
        policy measured so far (recency) lost to quantization by 28x at matched
        bytes and this class should not imply otherwise.

        Per-channel keys are refused rather than silently mishandled: their
        absmax is grouped over *runs* of ``group`` tokens, so dropping tokens
        out of the middle would leave scales describing groups that no longer
        exist. Sink+recent has the same problem and predates the guard; see
        ``key_scaling`` in ``__init__`` for why that path is off by default.
        """
        if self.key_scaling == "per_channel" and self.quantize_keys:
            raise NotImplementedError(
                "evict_index does not support key_scaling='per_channel': the "
                "key absmax is grouped over runs of tokens, so an arbitrary "
                "keep-set would leave scales that describe groups which no "
                "longer exist. Use per_token keys (the default) for eviction.")
        per_layer = hasattr(keep, "keys")            # mapping vs a single index tensor
        for li in list(self._v):
            idx = (keep[li] if per_layer else keep)
            idx = idx.to(device=self._v[li][1].device, dtype=torch.long)
            for store in (self._k, self._v):
                slot = store.get(li)
                if slot is not None:
                    store[li] = self._index_slot(slot, idx)
        if self.residence == "host":
            self._host_recompact()
        self._resync_held()

    def _resync_held(self) -> None:
        """Re-read held length from the store so it cannot drift from reality."""
        for li in list(self._seen):
            vs = self._v.get(li)
            if vs is not None:
                self._seen[li] = (vs[1].shape[2] if vs[0] == "raw" else vs[1].shape[0])

    def get_max_cache_shape(self) -> Optional[int]:
        return None                      # grows without a preset bound

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """Where this step's queries sit relative to the cache — i.e. the
        number of tokens already held.

        This returned 0 originally, on the reasoning that a cache which never
        pre-allocates has queries "at the front". That reasoning was wrong: the
        offset positions the query rows against the kv axis when the causal
        mask is built, so with a non-empty cache a 0 offset masks chunk N's
        queries as though they were the first tokens of the sequence. Feeding
        1024 tokens in 128-token chunks scored ppl 330 against 5.97 for the
        same cache in one shot.

        It hid because every single-forward test starts from an empty cache,
        where 0 is the correct answer — so four architectures' worth of control
        arms matched their fp16 reference EXACTLY and still said nothing about
        the accumulating path. See test_chunked_prefill_matches_single_forward.
        """
        return self.held_length(layer_idx)

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        """(kv_length, kv_offset) — the shape transformers' mask builder wants.

        kv_length must count the tokens about to be written, not just those
        already stored: the mask is built BEFORE the layers run, so at that
        moment ``get_seq_length`` is still the PREVIOUS length (0 on the first
        forward). Returning it alone yields a zero-width mask.

        This surfaced only on gpt-oss, which uses eager attention with an
        explicit additive mask because of its attention sinks; OLMoE's path
        tolerates a None mask and hid the bug entirely. Hence the rule: a cache
        cannot be called correct against one architecture's attention path.
        """
        # cache_position is not always a tensor: transformers' generate path
        # passes a bare int (observed: 5 on a 5-token prompt against an empty
        # cache, so it is the QUERY LENGTH, not a position — a position would
        # have been 0..4). Handle tensor, sequence, and scalar alike.
        if hasattr(cache_position, "shape"):
            q_len = int(cache_position.shape[0])
        elif hasattr(cache_position, "__len__"):
            q_len = len(cache_position)
        else:
            q_len = int(cache_position)
        return q_len + self.held_length(layer_idx), 0

    @property
    def is_compileable(self) -> bool:
        return False                     # ragged growth; do not let compile assume shapes

    def __len__(self) -> int:
        return len(self._seen)

    # ---- the budget number ----------------------------------------------
    def device_bytes(self) -> int:
        """Bytes of the cache resident on the GPU.

        Equals :meth:`memory_bytes` under ``residence='gpu'``. Under
        ``'host'`` it is 0 for the stored cache -- the whole point -- and the
        real GPU cost is the transient one-layer slice materialized inside
        :meth:`_load`, which lives and dies within a single layer's forward and
        so is an activation, not cache residency. Reported separately from
        ``memory_bytes`` because conflating "how big is the cache" with "how
        much VRAM does it hold" is exactly the confusion streaming introduces.
        """
        total = 0
        for t in self._ktail.values():
            if t.is_cuda:
                total += t.numel() * 2
        for store in (self._k, self._v):
            for slot in store.values():
                if slot[0] == "raw":
                    if slot[1].is_cuda:
                        total += slot[1].numel() * slot[1].element_size()
                elif slot[1].is_cuda:
                    total += (slot[1].numel() * slot[1].element_size()
                              if slot[2] is None
                              else slot[1].numel() + slot[2].numel() * 4)
        return total

    def memory_bytes(self, fp16: bool = False) -> int:
        """Bytes actually held. ``fp16=True`` reports what bf16 storage of the
        same tokens would have cost — the comparison the context budget uses."""
        total = 0
        for t in self._ktail.values():                # bf16 tail, < group tokens
            total += t.numel() * 2
        for store in (self._k, self._v):
            for slot in store.values():
                if slot[0] == "raw":
                    t = slot[1]
                    total += t.numel() * (2 if fp16 else t.element_size())
                else:
                    _, p, a, d = slot     # "nf4"/"nf4pc" packed, or "rawh"
                    n_tok, n_head = p.shape[0], p.shape[1]
                    if fp16:
                        total += n_tok * n_head * d * 2
                    elif a is None:               # host-resident, unquantized
                        total += p.numel() * p.element_size()
                    else:
                        total += p.numel() + a.numel() * 4
        return total
