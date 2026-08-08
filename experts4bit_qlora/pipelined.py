# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Pipelined hot-expert residency — one cold path, zero host knowledge of ids.

The v0 partition (:mod:`experts4bit_qlora.hot_residency`) is correct but its
token loop pays a per-layer host round-trip: ``nonzero`` syncs, a ``.cpu()``
id hop, CPU ``index_select`` gathers whose outputs are unpinned (so the
"non_blocking" H2D is silently synchronous), and ``tolist()`` group
descriptors per GEMM call. Measured on an H200 (receipt
``bench/homelab/RESULT-hybrid-curve-v0-h200.md`` in grouped-nf4-gemm): 250
ms/token at K=0 against a ~34 ms transfer floor, and 61 ms/token at K=128
with the cold branch never firing — the loop, not the link.

This module replaces that loop with the flagship shape (the design law of
``kernel/host_gather.py``: *"No CPU knows the ids; nothing synchronizes"*):

- **RAMStore arena**: per layer, ONE pinned host tensor ``[E, row_bytes]``
  whose row-block packs the expert's four segments (gate_up packed | gate_up
  absmax | down packed | down absmax, 8-byte aligned) — the same
  pinned-arena pattern as ``offload._ExpertOffload`` (E4B_OFFLOAD_ARENA),
  expert-granular instead of layer-granular.
- **Residency filter as an address table**: ``src_of_expert[e]`` holds the
  absolute source address of expert ``e``'s row-block — into the resident
  GPU hot stack when ``e`` is hot, into the pinned host arena (UVA) when
  cold. Changing K rebuilds a table; it never changes the code path.
- **One gather kernel per layer per token** (:func:`_gather_rows_addr`, a
  per-slot-address sibling of the flagship ``gather_expert_rows``): each
  slot fetches its expert's row-block from wherever the table points — HBM
  D2D for hot (≈free), UVA PCIe reads for cold — and slots whose ``have``
  address already matches are skipped (residency short-circuit AND
  slot-level caching in the same mechanism). Launch geometry is
  id-independent; the ids flow through device memory only.
- **Device-id GEMV**: the fused grouped kernel's decode route takes expert
  ids as a device tensor with an id-independent grid, and its ``sizes``
  argument is a Python *constant* (``[1]*k``), so both projections are
  plain enqueues — no ``tolist``, no data-dependent launch parameters.

Per token per layer the host enqueues a fixed, id-independent sequence:
table lookup → gather → ``have`` update → GEMV(gate_up) → epilogue glue →
GEMV(down) → weighted scatter-add. Nothing synchronizes; nothing
data-dependent reaches Python. That makes the step CUDA-graph-capturable
(Phase 3) — every buffer here is persistent and the launch parameters are
static.

Scope: inference decode (T==1) on CUDA with bf16/fp16 compute. Prefill
(T>1), grad-enabled forwards, and unsupported dtypes fall back to the
module's reference forward — correct, sync-tolerant, one-time cost.
Memory model matches the v0 note: the module's original packed weights are
kept (reference fallback stays valid), so resident-expert GB is the
*computed* constrained-card figure; the realized-VRAM path is the
streaming-loader increment, unchanged here.

Hot sets: 1-D LongTensor of hot expert ids per MoE layer, in module order —
derive from stamped router receipts via the committed reducer (cite receipt
paths in the enabling commit).
"""
from __future__ import annotations

from typing import Sequence

import torch

from .hot_residency import _eligible


def _align8(n: int) -> int:
    return (n + 7) & ~7


def pipelined_available() -> bool:
    try:
        from nf4_grouped import gemm_4bit_grouped  # noqa: F401
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


# --------------------------------------------------------------------------
# The gather kernel: per-slot absolute source address, have-skip discipline.
# Sibling of kernel/host_gather.py::_gather_rows (flagship, untouched); the
# address indirection is what lets ONE launch serve hot (device) and cold
# (pinned-host UVA) sources — the residency filter happens in the table that
# produced the addresses, on device, before this kernel runs.
# --------------------------------------------------------------------------
_KERNEL = None


def _gather_kernel():
    global _KERNEL
    if _KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _gather_rows_addr(
            dst_ptr,        # cuda [k, row_words] int64-viewed slot store
            src_ptr,        # cuda int64 [k] — absolute row-block address per slot
            have_ptr,       # cuda int64 [k] — address whose bytes the slot holds
            row_words,      # int64 words per row-block
            BLOCK: tl.constexpr,
        ):
            slot = tl.program_id(0)
            chunk = tl.program_id(1)
            want = tl.load(src_ptr + slot)
            have = tl.load(have_ptr + slot)
            if want == have:
                return
            offs = chunk * BLOCK + tl.arange(0, BLOCK)
            mask = offs < row_words
            src = tl.cast(want, tl.pointer_type(tl.int64))
            vals = tl.load(src + offs, mask=mask)
            tl.store(dst_ptr + slot.to(tl.int64) * row_words + offs, vals, mask=mask)

        _KERNEL = _gather_rows_addr
    return _KERNEL


class _PipelinedResidency:
    """Per-module engine: pinned host arena + resident hot stack + k-slot
    store, dispatched by an address table. All state is persistent (the
    address table bakes ``data_ptr()``s — the arena, hot stack, and slot
    tensors are owned here and never reallocated)."""

    def __init__(self, mod, hot_ids, device, k_slots: int, homes=None):
        import os
        if os.environ.get("TRITON_INTERPRET") == "1":
            raise RuntimeError(
                "pipelined residency cannot run under the Triton interpreter "
                "(TRITON_INTERPRET=1): the address-gather dereferences raw device/UVA "
                "pointers, which the host-side interpreter segfaults on. Run "
                "interpreter-mode suites in their own process.")
        if k_slots < 1:
            raise ValueError(f"k_slots must be >= 1, got {k_slots}")
        self.mod = mod
        self.device = torch.device(device)
        self.k = int(k_slots)
        E = mod.num_experts
        hot_ids = torch.as_tensor(hot_ids, dtype=torch.long).unique()
        if hot_ids.numel() and (hot_ids.min() < 0 or hot_ids.max() >= E):
            raise ValueError(
                f"hot ids must lie in [0, {E}); got range "
                f"[{int(hot_ids.min())}, {int(hot_ids.max())}]"
            )
        self.hot_ids = hot_ids
        n1, k1 = mod._gate_up_shape
        n2, k2 = mod._down_shape
        self.shapes = (n1, k1, n2, k2)
        self.has_gate = mod.has_gate
        self.act_fn = mod.act_fn

        # --- row-block layout: four segments, each 8-byte aligned ---------
        seg = [n1 * (k1 // 2), n1 * (k1 // 64) * 4, n2 * (k2 // 2), n2 * (k2 // 64) * 4]
        off = [0]
        for s in seg[:-1]:
            off.append(_align8(off[-1] + s))
        row_bytes = _align8(off[-1] + seg[-1])
        self.row_bytes, self.off = row_bytes, off
        self.row_words = row_bytes // 8

        # --- RAMStore arena: pinned [E, row_bytes], filled segment-wise ---
        # (cross-device copy_ moves each GPU segment straight into the pinned
        # rows; the arena is the UVA-addressable cold source thereafter)
        #
        # The pinned ALLOCATION is reusable across re-enables of the same
        # module (layout signature checked); pinning tens of GB is the
        # expensive part of enable(), so a K-ladder costs one pinning, not
        # one per rung. The CONTENT is still refilled from the module's
        # current weights on every enable (the copy_ block below always
        # runs) — the always-refresh invariant is untouched; only the
        # allocation is cached.
        cached = getattr(mod, "_e4b_arena_cache", None)
        sig = (E, row_bytes, tuple(off))
        arena = None
        if cached is not None and cached[0] == sig:
            arena = cached[1]
            self.pinned = bool(cached[2])
        if arena is None:
            arena = torch.zeros(E, row_bytes, dtype=torch.uint8)
            try:
                arena = arena.pin_memory()
                self.pinned = arena.is_pinned()
            except (RuntimeError, AssertionError):
                self.pinned = False  # pageable fallback: correct, but UVA reads
                # from pageable memory are not guaranteed — enable() refuses below
        a_f32 = arena.view(torch.float32)
        # Source the arena from the OFFLOAD HOMES when the module is offloaded. Under
        # offload the base's expert tensors are 0-element GPU placeholders and the real
        # (pinned, correctly shaped) weights live in handle.home — copying the placeholders
        # produced "size of tensor a (N) must match tensor b (0)" and made residency and
        # offload mutually exclusive, which is backwards: residency IS an offload mechanism,
        # so it should TAKE OVER from the staging hooks, not refuse to coexist with them.
        # This is what lets a model larger than VRAM use the dial at all (Qwen3-30B NF4
        # needs 19.84 GiB resident vs a 12 GB A2000).
        src = homes if homes is not None else {
            "gate_up_proj": mod.gate_up_proj, "gate_up_absmax": mod.gate_up_absmax,
            "down_proj": mod.down_proj, "down_absmax": mod.down_absmax}
        gu_p = src["gate_up_proj"].view(E, -1)
        gu_a = src["gate_up_absmax"].view(E, -1).float()
        dn_p = src["down_proj"].view(E, -1)
        dn_a = src["down_absmax"].view(E, -1).float()
        arena[:, off[0]:off[0] + seg[0]].copy_(gu_p.view(torch.uint8) if gu_p.dtype != torch.uint8 else gu_p)
        a_f32[:, off[1] // 4: off[1] // 4 + seg[1] // 4].copy_(gu_a)
        arena[:, off[2]:off[2] + seg[2]].copy_(dn_p.view(torch.uint8) if dn_p.dtype != torch.uint8 else dn_p)
        a_f32[:, off[3] // 4: off[3] // 4 + seg[3] // 4].copy_(dn_a)
        self.arena = arena
        mod._e4b_arena_cache = (sig, arena, self.pinned)

        # --- ONE device store: [hot rows | k slots], same row-block layout --
        # The hot stack and the slot store used to be separate allocations, which
        # forced every routed expert through a slot: a hot hit still paid a full
        # device-to-device row copy (resident stack -> slot) before the GEMM could
        # read it. Measured on granite/A2000, that re-copy was 48.5% of ALL gather
        # traffic and the total bytes moved were INVARIANT across hot-set sizes —
        # a hot set relocated traffic from PCIe to d2d rather than removing it
        # (bench/receipts-hotsets-granite-20260804/).
        #
        # Sharing one allocation lets the GEMM address a hot row IN PLACE: the row
        # index it reads is data, so a hot expert points at its resident row and
        # only a cold one is gathered into a slot. `sizes` stays a host constant
        # and there is still exactly one GEMM launch, so the fixed-shape,
        # zero-host-sync decode contract is untouched.
        k = self.k
        n_hot = int(hot_ids.numel())
        self.n_hot = n_hot
        store = torch.empty(n_hot + k, row_bytes, dtype=torch.uint8, device=self.device)
        if n_hot:
            store[:n_hot].copy_(arena.index_select(0, hot_ids).to(self.device))
        self.store = store
        self.hot_stack = store[:n_hot]          # view — hot rows, never copied again
        slots = store[n_hot:]                   # view — cold landing rows
        self.slots = slots
        self.slots64 = slots.view(torch.int64)

        # --- the residency filter: absolute source address per expert -----
        is_hot = torch.zeros(E, dtype=torch.bool, device=self.device)
        is_hot[hot_ids.to(self.device)] = True
        h_row = torch.zeros(E, dtype=torch.long, device=self.device)
        h_row[hot_ids.to(self.device)] = torch.arange(hot_ids.numel(), device=self.device)
        host_addr = self.arena.data_ptr() + torch.arange(E, device=self.device, dtype=torch.long) * row_bytes
        hot_addr = self.hot_stack.data_ptr() + h_row * row_bytes
        self.src_of_expert = torch.where(is_hot, hot_addr, host_addr)  # [E] int64
        self.is_hot = is_hot
        self.h_row = h_row                      # [E] -> row within the hot segment

        # --- GEMM views span the WHOLE store, so a row index may name either
        # a resident hot row or a gathered slot ------------------------------
        rows = n_hot + k
        s_f32 = store.view(torch.float32)
        self.gu_p_v = torch.as_strided(store, (rows, n1, k1 // 2), (row_bytes, k1 // 2, 1), off[0])
        self.gu_a_v = torch.as_strided(s_f32, (rows, n1, k1 // 64), (row_bytes // 4, k1 // 64, 1), off[1] // 4)
        self.dn_p_v = torch.as_strided(store, (rows, n2, k2 // 2), (row_bytes, k2 // 2, 1), off[2])
        self.dn_a_v = torch.as_strided(s_f32, (rows, n2, k2 // 64), (row_bytes // 4, k2 // 64, 1), off[3] // 4)

        # persistent step state: fixed sizes list (Python constant — only
        # sum()/max() ever touch it), row-index dispatch, have table, input buf
        self.sizes = [1] * k
        # Row the GEMM reads for each of the k routed slots. Recomputed device-side
        # every fetch; initialised to the slot rows so it is valid before one runs.
        self.slot_rows = n_hot + torch.arange(k, dtype=torch.long, device=self.device)
        self.row_idx_buf = self.slot_rows.clone()
        self.row_idx = self.row_idx_buf.to(torch.int32)
        self.have = torch.full((k,), -1, dtype=torch.long, device=self.device)
        self.a_buf = None  # lazy: dtype follows live compute_dtype
        self.want_buf = torch.zeros(k, dtype=torch.long, device=self.device)

        # traffic accounting (device scalars, accumulated with enqueued tensor
        # ops — never read in the loop; .traffic() syncs once at report time).
        # hot_d2d_bytes measures the ACCEPTED re-copy inefficiency: hot rows are
        # copied a short distance (resident stack -> slot) instead of computed
        # in place. Harmless at HBM bandwidth (~5 us/expert), worth watching on
        # small-bandwidth cards (~45 us/expert on GDDR6) — if a small-hardware
        # profile shows this term, the known fix is an in-place hot GEMM path.
        self.hot_d2d_bytes = torch.zeros((), dtype=torch.long, device=self.device)
        self.cold_pcie_bytes = torch.zeros((), dtype=torch.long, device=self.device)

        # prime the slots with a valid row (expert 0) so a skipped slot can
        # never feed the GEMM uninitialized bytes (any *valid* stale row is
        # harmless: its lane weight is exactly the router weight it earns,
        # and slots only ever hold rows the gather placed there)
        self._prime()

    def _prime(self):
        kern = _gather_kernel()
        src0 = self.src_of_expert[0].expand(self.k).contiguous()
        grid = (self.k, -(-self.row_words // 2048))
        kern[grid](self.slots64, src0, self.have, self.row_words, BLOCK=2048, num_warps=4)
        self.have.copy_(src0)

    # ---- lead-time routing (flag-shaped: nothing calls this unless the
    # harness opts in). Issue the gather for PREDICTED ids early so the copy
    # overlaps upstream compute; the forward's own gather then corrects any
    # mispredicted slot via the have-skip discipline — predicted hits cost
    # nothing, misses re-fetch. Correctness is invariant: the GEMM only ever
    # sees rows the (real-id) gather placed. Mechanism only — untuned. ------
    def hint(self, pred_ids):
        pw = pred_ids.reshape(-1)
        if pw.numel() != self.k:
            return
        self._fetch(pw.to(device=self.device, dtype=torch.long))
        # bytes a hint moves are counted at the fetch site like any other
        # traffic; a perfect hint just shifts them earlier (the forward's own
        # fetch then skips and counts zero), a wrong hint shows up as the
        # extra traffic it really is.

    def _fetch(self, want):
        """The one fetch site: copy want_buf, resolve each routed slot to the row
        the GEMM will read, gather ONLY the cold ones, count traffic, advance
        ``have``. All enqueued; nothing reads back."""
        self.want_buf.copy_(want)
        src = self.src_of_expert.index_select(0, self.want_buf)
        hot = self.is_hot.index_select(0, self.want_buf)

        # A hot expert is read where it already lives, so it needs no slot and no
        # copy. Forcing its src to the slot's current `have` makes the gather's own
        # skip test fail for that lane — the kernel is untouched, and the decision
        # stays device-side (a host-visible branch here would break the zero-sync
        # decode contract that tests/test_pipelined.py enforces).
        src_eff = torch.where(hot, self.have, src)
        miss = src_eff != self.have
        # hot_d2d is now 0 by construction and kept as a REGRESSION WITNESS: if it
        # ever moves again, a hot row is being copied instead of read in place.
        self.hot_d2d_bytes += (miss & hot).sum() * self.row_bytes
        self.cold_pcie_bytes += (miss & ~hot).sum() * self.row_bytes

        kern = _gather_kernel()
        grid = (self.k, -(-self.row_words // 2048))
        kern[grid](self.slots64, src_eff, self.have, self.row_words, BLOCK=2048, num_warps=4)
        self.have.copy_(src_eff)
        # hot lane -> its resident row; cold lane -> the slot just gathered into.
        # With an EMPTY hot set every lane is cold, so the dispatch is the constant
        # slot_rows and recomputing it is pure overhead on the pure-streaming
        # config -- measured at -0.7% (p=0.013) on OLMoE/A2000 before this guard,
        # which is a real regression on the one config that cannot benefit.
        if self.n_hot:
            torch.where(hot, self.h_row.index_select(0, self.want_buf), self.slot_rows,
                        out=self.row_idx_buf)
            self.row_idx = self.row_idx_buf.to(torch.int32)

    def traffic(self) -> dict:
        """Report accumulated fetch traffic. SYNCHRONIZES (two .item() reads) —
        call outside any timed loop."""
        return {"hot_d2d_bytes": int(self.hot_d2d_bytes.item()),
                "cold_pcie_bytes": int(self.cold_pcie_bytes.item())}

    # ---- the per-token step: fixed, id-independent enqueues only ---------
    def step(self, x_row, want, cd):
        """x_row [1,H] (device, cd), want [k] long (device). Returns dn [k, n2]
        fp32-accurate bf16 plus nothing else — caller applies weights/epilogue."""
        from nf4_grouped import gemm_4bit_grouped

        k = self.k
        self._fetch(want)
        if self.a_buf is None or self.a_buf.dtype != cd:
            self.a_buf = torch.empty(k, x_row.shape[-1], dtype=cd, device=self.device)
        self.a_buf.copy_(x_row.expand(k, -1))
        from .lora import _epilogue

        gu = gemm_4bit_grouped(self.a_buf, self.gu_p_v, self.gu_a_v, self.sizes, self.row_idx)
        # The module's OWN epilogue (`_epilogue` -> `base._apply_gate` when it has one),
        # not an assumed SwiGLU. gpt-oss needs the subclass below because it also adds
        # per-expert biases; a custom ACTIVATION alone is handled right here, which is
        # what lets DeepSeek-V4 run on this engine instead of only on the deprecated one.
        h = _epilogue(self.mod, gu)
        dn = gemm_4bit_grouped(h.contiguous(), self.dn_p_v, self.dn_a_v, self.sizes, self.row_idx)
        return dn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        cd = self.mod.compute_dtype if self.mod.compute_dtype is not None else hidden_states.dtype
        in_dtype, in_dev = hidden_states.dtype, hidden_states.device
        x = hidden_states.to(device=self.device, dtype=cd)
        want = top_k_index.reshape(-1).to(device=self.device, dtype=torch.long)
        dn = self.step(x, want, cd)
        w = top_k_weights.reshape(-1).to(device=self.device, dtype=torch.float32)
        out = (dn.to(torch.float32) * w[:, None]).sum(0, keepdim=True)
        return out.to(device=in_dev, dtype=in_dtype)


class _GptOssPipelined(_PipelinedResidency):
    """gpt-oss epilogue on the same engine: clamped GLU
    ``(up+1)*(gate*sigmoid(gate*alpha))`` with per-expert biases indexed by
    the routed global ids (device index_select — enqueued, never read)."""

    def __init__(self, mod, hot_ids, device, k_slots, homes=None):
        super().__init__(mod, hot_ids, device, k_slots, homes=homes)
        self.gate_up_bias = mod.gate_up_bias.to(self.device)
        self.down_bias = mod.down_bias.to(self.device)
        self.alpha = float(mod.alpha)
        self.limit = float(mod.limit)

    def forward(self, hidden_states, router_indices, router_scores):
        from nf4_grouped import gemm_4bit_grouped

        cd = self.mod.compute_dtype if self.mod.compute_dtype is not None else hidden_states.dtype
        in_dtype, in_dev = hidden_states.dtype, hidden_states.device
        x = hidden_states.to(device=self.device, dtype=cd)
        want = router_indices.reshape(-1).to(device=self.device, dtype=torch.long)
        k = self.k
        self._fetch(want)
        if self.a_buf is None or self.a_buf.dtype != cd:
            self.a_buf = torch.empty(k, x.shape[-1], dtype=cd, device=self.device)
        self.a_buf.copy_(x.expand(k, -1))
        gu = gemm_4bit_grouped(self.a_buf, self.gu_p_v, self.gu_a_v, self.sizes, self.row_idx)
        gu = gu + self.gate_up_bias.index_select(0, self.want_buf)
        gate, up = gu.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        h = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
        dn = gemm_4bit_grouped(h.contiguous(), self.dn_p_v, self.dn_a_v, self.sizes, self.row_idx)
        dn = dn.to(torch.float32) + self.down_bias.index_select(0, self.want_buf).to(torch.float32)
        w = router_scores.reshape(-1).to(device=self.device, dtype=torch.float32)
        out = (dn * w[:, None]).sum(0, keepdim=True)
        return out.to(device=in_dev, dtype=in_dtype)


def enable_pipelined_residency(model, hot_sets: Sequence, device: str = "cuda",
                               k_slots: int | None = None,
                               verbose: bool = False) -> int:
    """Partition every eligible experts module under ``model`` into the
    pipelined engine: resident hot stack + pinned-arena cold source, one
    address-dispatched gather, device-id GEMV. ``hot_sets`` carries exactly
    one entry per targeted module in module order (skipped modules still
    consume their entry). ``k_slots`` is the model's routed top-k (required —
    it sizes the slot store; a forward with a different k falls back to the
    reference path). K (the hot count) is data: pass a 0-length set for pure
    streaming, all experts for fully resident — same code path.

    **The engine, not K, is what buys the speed.** Measured 2026-08-08 on an A100
    (``offload=False``, 64 new tokens, medians over fresh processes): empty hot sets already
    give 3.51x on granite-3.0-1b, 2.74x on OLMoE-1B-7B and 2.15x on Qwen3-30B-A3B against the
    reference forward, and raising K on top was flat-to-slightly-negative on all three while
    costing VRAM (Qwen: +0.77 GiB at K=8, +3.7 GiB at K=32). Empty hot sets are therefore a
    perfectly good production configuration, not a degenerate one.

    ``ExpertsLoRA``-wrapped bases — the experts of every model
    ``load_moe_4bit_streaming`` returns — are targeted as well: the wrapper
    delegates the whole forward to the patched base under eval + ``no_grad``
    with a provably-zero adapter (``ExpertsLoRA._delegate_to_base``). Outside
    those conditions the patch installs but can never run, and this warns
    rather than returning a count that implies work.

    Mutually exclusive with the v0 hot-residency and the [fast] patch on the
    same module (disable those first). Grad-enabled forwards, T>1 (prefill),
    and non-bf16/fp16 compute run the saved reference forward.
    """
    from experts4bit_qlora import Experts4bit, ExpertsNbit
    from experts4bit_qlora.gptoss import GptOssExperts4bit

    if k_slots is None:
        raise ValueError("k_slots (the model's routed top-k) is required")
    if hasattr(model, "modules"):
        from experts4bit_qlora.hot_residency import target_modules
        mods = target_modules(model)
        # An `ExpertsLoRA.base` IS a valid target. This used to raise
        # NotImplementedError, which made the engine unreachable for every model
        # `load_moe_4bit_streaming` returns (its experts are always wrapped) — i.e.
        # for the path most callers take. `ExpertsLoRA.forward` calls `self.base(...)`
        # — reaching the patch installed below — whenever `_delegate_to_base()` holds:
        # eval mode, no_grad, and an adapter that provably contributes nothing.
        #
        # Those conditions are why the wrapper is mapped here rather than assumed
        # away: when they do NOT hold the patch installs and never runs, so the
        # warnings at the end of this function tell the caller instead of letting a
        # healthy-looking return value imply work that will not happen.
        try:
            from experts4bit_qlora.lora import ExpertsLoRA
            lora_parent = {id(m.base): m for m in model.modules()
                           if isinstance(m, ExpertsLoRA) and hasattr(m, "base")}
        except ImportError:
            lora_parent = {}
    else:
        mods = [model]
        lora_parent = {}
    if len(hot_sets) != len(mods):
        raise ValueError(
            f"hot_sets has {len(hot_sets)} entries but the model has {len(mods)} "
            f"ExpertsNbit modules — exactly one entry per MoE layer in module order")

    offloaded_taken_over = []   # handles whose staging the engine supersedes
    stock_forwards = {ExpertsNbit.forward, Experts4bit.forward}
    try:
        stock_forwards.add(GptOssExperts4bit.forward)
        from experts4bit_qlora.gptoss import GptOssExpertsNbit
        stock_forwards.add(GptOssExpertsNbit.forward)
        # V4 overrides `forward` for its CLAMPED SwiGLU and nothing else -- no biases --
        # and `_PipelinedResidency.step` now reproduces that through `_apply_gate`. Without
        # this it was skipped as "custom forward", so V4 residency worked only on
        # `enable_hot_residency`, the engine this one deprecates.
        from experts4bit_qlora.deepseek_v4 import (
            DeepseekV4Experts4bit, DeepseekV4ExpertsNbit)
        stock_forwards |= {DeepseekV4Experts4bit.forward, DeepseekV4ExpertsNbit.forward}
    except ImportError:
        pass

    patched = 0
    unreachable = []      # patched, but the parent adapter will not delegate to it
    wrapped_patched = 0   # patched modules that are reached via an ExpertsLoRA wrapper
    for i, mod in enumerate(mods):
        if (hasattr(mod, "_e4b_fast_ref") or hasattr(mod, "_e4b_hot_ref")
                or hasattr(mod, "_e4b_cold_ref") or hasattr(mod, "_e4b_mxfp4_ref")):
            if verbose:
                print(f"[pipelined] skip {type(mod).__name__}: another forward patch is active")
            continue
        if type(mod).forward not in stock_forwards and not hasattr(mod, "_e4b_pipe_ref"):
            if verbose:
                print(f"[pipelined] skip {type(mod).__name__}: custom forward")
            continue
        reason = _eligible(mod)
        if reason is not None:
            if verbose:
                print(f"[pipelined] skip {type(mod).__name__}: {reason}")
            continue
        parent = lora_parent.get(id(mod))
        if parent is not None:
            wrapped_patched += 1
            # The data question only — `_delegate_to_base` additionally requires
            # eval + no_grad, which are properties of the call site, not of the
            # adapter, and are warned about separately below.
            if not parent._adapter_is_zero():
                unreachable.append(type(mod).__name__)
        # If this module is OFFLOADED, its expert tensors are 0-element placeholders and the
        # real pinned weights are in handle.home. Hand those to the engine as the arena
        # source, then take the staging hooks OUT of the loop: the engine now owns expert
        # movement (hot resident + cold gathered from its own pinned arena), so leaving
        # offload's per-forward stage/evict active would re-stage the whole layer every
        # token — the exact traffic the engine exists to remove.
        homes = None
        handle = getattr(parent, "_offload", None) if parent is not None else None
        if handle is not None and getattr(handle, "home", None):
            homes = handle.home
            offloaded_taken_over.append(handle)
        cls = _GptOssPipelined if isinstance(mod, GptOssExperts4bit) else _PipelinedResidency
        state = cls(mod, hot_sets[i], device, k_slots, homes=homes)
        if not state.pinned:
            raise RuntimeError(
                "pipelined residency requires pinned host memory (UVA-addressable) for the cold "
                "arena; pin_memory() fell back to pageable on this host")
        if hasattr(mod, "_e4b_pipe_ref"):
            mod._pipelined = state  # re-enable: rebuild from current weights
            patched += 1
            continue
        mod._e4b_pipe_ref = mod.forward
        mod._pipelined = state

        def _fwd(hidden, idx, wts, _m=mod):
            st = _m._pipelined
            cd = _m.compute_dtype if _m.compute_dtype is not None else hidden.dtype
            if (hidden.shape[0] != 1 or idx.numel() != st.k
                    or cd not in (torch.bfloat16, torch.float16)):
                return _m._e4b_pipe_ref(hidden, idx, wts)
            if torch.is_grad_enabled() and (
                hidden.requires_grad or any(p.requires_grad for p in _m.parameters())
            ):
                return _m._e4b_pipe_ref(hidden, idx, wts)
            return st.forward(hidden, idx, wts)

        mod.forward = _fwd
        patched += 1

    # Two ways a patch on an ExpertsLoRA base installs and then never runs. Both are
    # silent, and both are worst precisely where this engine gets measured: a residency
    # split that never executes trivially reproduces the unsplit reference, so a DEAD
    # patch scores a perfect zero divergence and reads as a PASS. Say so instead of
    # returning a count that implies work.
    if wrapped_patched and getattr(model, "training", False):
        import warnings

        warnings.warn(
            f"[pipelined] model is in TRAINING mode: ExpertsLoRA only hands off to the "
            f"patched base under eval + no_grad, so {wrapped_patched} patch(es) on wrapped "
            "bases will be bypassed and the residency engine will not run. Call "
            "model.eval() before inference.",
            RuntimeWarning,
            stacklevel=2,
        )
    if unreachable:
        import warnings

        warnings.warn(
            f"[pipelined] {len(unreachable)} of {patched} patched module(s) are ExpertsLoRA "
            "bases with a non-zero adapter: ExpertsLoRA injects its low-rank delta before the "
            "activation, so it does not call base.forward and these patches will never run. "
            "The engine cannot serve a trained per-expert adapter today; merge or drop the "
            "adapter for residency inference.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Take over expert movement from offload for the calls this engine serves (decode).
    # Conditional by design — see the takeover block in offload._stage_pre_hook: prefill,
    # grad and odd-dtype forwards still fall back to the reference path and still need the
    # layer staged, so this flag suppresses staging ONLY where the engine is authoritative.
    for _h in offloaded_taken_over:
        _h._superseded = True
    return patched


def disable_pipelined_residency(model) -> int:
    """Undo :func:`enable_pipelined_residency`; returns modules restored."""
    mods = list(model.modules()) if hasattr(model, "modules") else [model]
    restored = 0
    for mod in mods:
        if hasattr(mod, "_e4b_pipe_ref") and hasattr(mod, "_pipelined"):
            mod.forward = mod._e4b_pipe_ref
            del mod._e4b_pipe_ref, mod._pipelined
            restored += 1
    # Hand expert movement BACK to offload. Leaving _superseded set would make the staging
    # hooks skip decode forever while nothing serves them — the module's tensors are
    # placeholders, so that reads as a shape error long after residency was turned off.
    for m in mods:
        h = getattr(m, "_offload", None)
        if h is not None and getattr(h, "_superseded", False):
            h._superseded = False
    return restored
