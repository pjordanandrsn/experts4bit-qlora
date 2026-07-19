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

    def __init__(self, mod, hot_ids, device, k_slots: int):
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
        arena = torch.zeros(E, row_bytes, dtype=torch.uint8)
        try:
            arena = arena.pin_memory()
            self.pinned = arena.is_pinned()
        except (RuntimeError, AssertionError):
            self.pinned = False  # pageable fallback: correct, but UVA reads
            # from pageable memory are not guaranteed — enable() refuses below
        a_f32 = arena.view(torch.float32)
        gu_p = mod.gate_up_proj.view(E, -1)
        gu_a = mod.gate_up_absmax.view(E, -1).float()
        dn_p = mod.down_proj.view(E, -1)
        dn_a = mod.down_absmax.view(E, -1).float()
        arena[:, off[0]:off[0] + seg[0]].copy_(gu_p.view(torch.uint8) if gu_p.dtype != torch.uint8 else gu_p)
        a_f32[:, off[1] // 4: off[1] // 4 + seg[1] // 4].copy_(gu_a)
        arena[:, off[2]:off[2] + seg[2]].copy_(dn_p.view(torch.uint8) if dn_p.dtype != torch.uint8 else dn_p)
        a_f32[:, off[3] // 4: off[3] // 4 + seg[3] // 4].copy_(dn_a)
        self.arena = arena

        # --- hot stack: same row-block layout, resident on device ---------
        if hot_ids.numel():
            self.hot_stack = arena.index_select(0, hot_ids).to(self.device)
        else:
            self.hot_stack = torch.empty(0, row_bytes, dtype=torch.uint8, device=self.device)

        # --- the residency filter: absolute source address per expert -----
        is_hot = torch.zeros(E, dtype=torch.bool, device=self.device)
        is_hot[hot_ids.to(self.device)] = True
        h_row = torch.zeros(E, dtype=torch.long, device=self.device)
        h_row[hot_ids.to(self.device)] = torch.arange(hot_ids.numel(), device=self.device)
        host_addr = self.arena.data_ptr() + torch.arange(E, device=self.device, dtype=torch.long) * row_bytes
        hot_addr = self.hot_stack.data_ptr() + h_row * row_bytes
        self.src_of_expert = torch.where(is_hot, hot_addr, host_addr)  # [E] int64
        self.is_hot = is_hot

        # --- k-slot store + GEMM views (as_strided into the same bytes) ---
        k = self.k
        slots = torch.empty(k, row_bytes, dtype=torch.uint8, device=self.device)
        self.slots = slots
        self.slots64 = slots.view(torch.int64)
        s_f32 = slots.view(torch.float32)
        self.gu_p_v = torch.as_strided(slots, (k, n1, k1 // 2), (row_bytes, k1 // 2, 1), off[0])
        self.gu_a_v = torch.as_strided(s_f32, (k, n1, k1 // 64), (row_bytes // 4, k1 // 64, 1), off[1] // 4)
        self.dn_p_v = torch.as_strided(slots, (k, n2, k2 // 2), (row_bytes, k2 // 2, 1), off[2])
        self.dn_a_v = torch.as_strided(s_f32, (k, n2, k2 // 64), (row_bytes // 4, k2 // 64, 1), off[3] // 4)

        # persistent step state: fixed sizes list (Python constant — only
        # sum()/max() ever touch it), device slot ids, have table, input buf
        self.sizes = [1] * k
        self.slot_eids = torch.arange(k, dtype=torch.int32, device=self.device)
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
        """The one fetch site: copy want_buf, dispatch the address-gather, count
        traffic, advance ``have``. All enqueued; nothing reads back."""
        self.want_buf.copy_(want)
        src = self.src_of_expert.index_select(0, self.want_buf)
        miss = src != self.have
        hot = self.is_hot.index_select(0, self.want_buf)
        self.hot_d2d_bytes += (miss & hot).sum() * self.row_bytes
        self.cold_pcie_bytes += (miss & ~hot).sum() * self.row_bytes
        kern = _gather_kernel()
        grid = (self.k, -(-self.row_words // 2048))
        kern[grid](self.slots64, src, self.have, self.row_words, BLOCK=2048, num_warps=4)
        self.have.copy_(src)

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
        gu = gemm_4bit_grouped(self.a_buf, self.gu_p_v, self.gu_a_v, self.sizes, self.slot_eids)
        if self.has_gate:
            gate, up = gu.chunk(2, dim=-1)
            h = self.act_fn(gate) * up
        else:
            h = self.act_fn(gu)
        dn = gemm_4bit_grouped(h.contiguous(), self.dn_p_v, self.dn_a_v, self.sizes, self.slot_eids)
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

    def __init__(self, mod, hot_ids, device, k_slots):
        super().__init__(mod, hot_ids, device, k_slots)
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
        gu = gemm_4bit_grouped(self.a_buf, self.gu_p_v, self.gu_a_v, self.sizes, self.slot_eids)
        gu = gu + self.gate_up_bias.index_select(0, self.want_buf)
        gate, up = gu.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        h = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
        dn = gemm_4bit_grouped(h.contiguous(), self.dn_p_v, self.dn_a_v, self.sizes, self.slot_eids)
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

    Mutually exclusive with the v0 hot-residency and the [fast] patch on the
    same module (disable those first). Grad-enabled forwards, T>1 (prefill),
    and non-bf16/fp16 compute run the saved reference forward.
    """
    from experts4bit_qlora import Experts4bit, ExpertsNbit
    from experts4bit_qlora.gptoss import GptOssExperts4bit

    if k_slots is None:
        raise ValueError("k_slots (the model's routed top-k) is required")
    if hasattr(model, "modules"):
        try:
            from experts4bit_qlora.lora import ExpertsLoRA
            wrapped = {id(m.base) for m in model.modules()
                       if isinstance(m, ExpertsLoRA) and hasattr(m, "base")}
        except ImportError:
            wrapped = set()
        all_nbit = [m for m in model.modules() if isinstance(m, ExpertsNbit)]
        mods = [m for m in all_nbit if id(m) not in wrapped]
        if wrapped and not mods:
            raise NotImplementedError(
                "every ExpertsNbit here is an ExpertsLoRA.base (the streaming-loader/offload "
                "path); pipelined residency supports standalone Experts4bit modules")
    else:
        mods = [model]
    if len(hot_sets) != len(mods):
        raise ValueError(
            f"hot_sets has {len(hot_sets)} entries but the model has {len(mods)} "
            f"ExpertsNbit modules — exactly one entry per MoE layer in module order")

    stock_forwards = {ExpertsNbit.forward, Experts4bit.forward}
    try:
        stock_forwards.add(GptOssExperts4bit.forward)
        from experts4bit_qlora.gptoss import GptOssExpertsNbit
        stock_forwards.add(GptOssExpertsNbit.forward)
    except ImportError:
        pass

    patched = 0
    for i, mod in enumerate(mods):
        if hasattr(mod, "_e4b_fast_ref") or hasattr(mod, "_e4b_hot_ref"):
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
        cls = _GptOssPipelined if isinstance(mod, GptOssExperts4bit) else _PipelinedResidency
        state = cls(mod, hot_sets[i], device, k_slots)
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
    return patched


def disable_pipelined_residency(model) -> int:
    """Undo :func:`enable_pipelined_residency`; returns modules restored."""
    mods = model.modules() if hasattr(model, "modules") else [model]
    restored = 0
    for mod in mods:
        if hasattr(mod, "_e4b_pipe_ref") and hasattr(mod, "_pipelined"):
            mod.forward = mod._e4b_pipe_ref
            del mod._e4b_pipe_ref, mod._pipelined
            restored += 1
    return restored
