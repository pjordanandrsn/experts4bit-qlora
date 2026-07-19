"""Hot-expert residency — the constrained-card MoE serving path.

Pins each MoE layer's *hottest* experts permanently in VRAM (computed by the
fused ``grouped-nf4-gemm`` kernel, zero per-token transfer) and streams only
the *cold* tail from pinned host RAM on demand. This is finer-grained than the
whole-layer residency that GGUF runtimes place at (they cannot split experts
within a fused per-layer tensor), and it exploits the empirical fact that MoE
routing, while near-uniform globally, concentrates per layer: on gpt-oss-120b
a per-layer top-16 (12% of experts) captures ~30% of that layer's hits
out-of-sample, so ~12% of the expert VRAM carries ~30% of the traffic at zero
transfer cost.

Regime: this wins where the host CPU is weak and VRAM is small but the link is
not the sole bottleneck (edge boxes, small discrete GPUs). On a strong-CPU
server, computing the cold experts on the host (a GGUF runtime's path) is
faster — that is a different instrument, not this one.

The math is identical to the reference ``ExpertsNbit`` forward: the hot and
cold experts are the same NF4 values, merely partitioned by residence, and
both paths decode through the fused kernel with fp32 accumulation. Inference
only (no backward); training forwards fall back to the reference path.

Usage::

    from experts4bit_qlora import enable_hot_residency
    # hot_sets[i] = 1-D LongTensor of hot expert ids for the i-th MoE layer,
    # e.g. derived from a routing-frequency histogram.
    n = enable_hot_residency(model, hot_sets, device="cuda")
"""
from __future__ import annotations

from typing import Sequence

import torch


def _fused_over_stack(x_rows, local_ids, gu_p, gu_a, dn_p, dn_a, shapes, has_gate, act_fn):
    """Down-projection outputs for each (token,slot) row, computed on the device
    the packed stack lives on. ``local_ids`` index into the G-expert stack
    (``gu_p`` is ``[G, n1, k1//2]`` etc.). Returns ``[R, H]`` in the input row
    order (unweighted; the caller applies router scores and scatters)."""
    from nf4_grouped import gemm_4bit_grouped

    n1, k1, n2, k2 = shapes
    order = torch.argsort(local_ids)
    sorted_ids = local_ids.index_select(0, order)
    x_sorted = x_rows.index_select(0, order).contiguous()
    uniq, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    sizes = counts.tolist()
    eids = uniq.tolist()
    gu = gemm_4bit_grouped(x_sorted, gu_p, gu_a, sizes, eids)
    if has_gate:
        gate, up = gu.chunk(2, dim=-1)
        h = act_fn(gate) * up
    else:
        h = act_fn(gu)
    dn = gemm_4bit_grouped(h.contiguous(), dn_p, dn_a, sizes, eids)
    out = torch.empty_like(dn)
    out.index_copy_(0, order, dn)  # unsort back to caller's row order
    return out


class _HotResidency:
    """Per-module state: a resident GPU hot-stack + a pinned-CPU cold-stack, and
    the global<->local id maps needed to dispatch each routed expert."""

    def __init__(self, mod, hot_ids, device):
        self.mod = mod
        self.device = torch.device(device)
        E = mod.num_experts
        hot_ids = torch.as_tensor(hot_ids, dtype=torch.long).unique()
        cold_ids = torch.tensor([e for e in range(E) if e not in set(hot_ids.tolist())],
                                dtype=torch.long)
        self.hot_ids, self.cold_ids = hot_ids, cold_ids
        n1, k1 = mod._gate_up_shape
        n2, k2 = mod._down_shape
        self.shapes = (n1, k1, n2, k2)
        self.has_gate = mod.has_gate
        self.act_fn = mod.act_fn
        self.compute_dtype = mod.compute_dtype

        # per-expert flattened packed storage -> [E, n, k/2] / [E, n, k/64]
        gu_p = mod.gate_up_proj.view(E, n1, k1 // 2)
        gu_a = mod.gate_up_absmax.view(E, n1, k1 // 64).float()
        dn_p = mod.down_proj.view(E, n2, k2 // 2)
        dn_a = mod.down_absmax.view(E, n2, k2 // 64).float()

        # index on the weights' own device (the model is typically CUDA-resident)
        hi = hot_ids.to(gu_p.device)
        ci = cold_ids.to(gu_p.device)
        # HOT: resident on the GPU (never transferred again)
        self.h_gu_p = gu_p.index_select(0, hi).contiguous().to(self.device)
        self.h_gu_a = gu_a.index_select(0, hi).contiguous().to(self.device)
        self.h_dn_p = dn_p.index_select(0, hi).contiguous().to(self.device)
        self.h_dn_a = dn_a.index_select(0, hi).contiguous().to(self.device)
        # COLD: pinned host RAM, streamed per token (only the routed subset)
        self.c_gu_p = gu_p.index_select(0, ci).contiguous().cpu()
        self.c_gu_a = gu_a.index_select(0, ci).contiguous().cpu()
        self.c_dn_p = dn_p.index_select(0, ci).contiguous().cpu()
        self.c_dn_a = dn_a.index_select(0, ci).contiguous().cpu()
        try:
            self.c_gu_p = self.c_gu_p.pin_memory()
            self.c_gu_a = self.c_gu_a.pin_memory()
            self.c_dn_p = self.c_dn_p.pin_memory()
            self.c_dn_a = self.c_dn_a.pin_memory()
        except (RuntimeError, AssertionError):
            pass  # pageable fallback is correct, just synchronous H2D

        # global expert id -> (is_hot, local index within its stack)
        g2h = torch.full((E,), -1, dtype=torch.long)
        g2h[hot_ids] = torch.arange(hot_ids.numel())
        g2c = torch.full((E,), -1, dtype=torch.long)
        g2c[cold_ids] = torch.arange(cold_ids.numel())
        self.is_hot = torch.zeros(E, dtype=torch.bool, device=self.device)
        self.is_hot[hot_ids.to(self.device)] = True
        self.g2h = g2h.to(self.device)
        self.g2c_cpu = g2c  # cold local ids resolved on CPU (stack is on CPU)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        input_dtype = hidden_states.dtype
        cd = self.compute_dtype if self.compute_dtype is not None else input_dtype
        x = hidden_states.to(cd)
        T, H = x.shape
        k = top_k_index.shape[1]
        dev = self.device

        flat = top_k_index.reshape(-1).to(dev)                 # [T*k] global expert per assignment
        row_token = torch.arange(T * k, device=dev) // k
        row_slot = torch.arange(T * k, device=dev) - row_token * k
        hot_row = self.is_hot[flat]                            # [T*k] bool
        out = torch.zeros(T, H, dtype=torch.float32, device=dev)

        # --- HOT: resident GPU stack, zero transfer ---
        hr = hot_row.nonzero(as_tuple=False).view(-1)
        if hr.numel():
            local = self.g2h[flat.index_select(0, hr)]
            xr = x.index_select(0, row_token.index_select(0, hr))
            dn = _fused_over_stack(xr, local, self.h_gu_p, self.h_gu_a, self.h_dn_p,
                                   self.h_dn_a, self.shapes, self.has_gate, self.act_fn)
            w = top_k_weights[row_token.index_select(0, hr), row_slot.index_select(0, hr)].to(torch.float32)
            out.index_add_(0, row_token.index_select(0, hr), dn.to(torch.float32) * w[:, None])

        # --- COLD: stream ONLY the routed cold experts from pinned host RAM ---
        cr = (~hot_row).nonzero(as_tuple=False).view(-1)
        if cr.numel():
            cold_glob = flat.index_select(0, cr).cpu()
            cold_local_full = self.g2c_cpu.index_select(0, cold_glob)     # local id in the full cold stack
            routed, compact = torch.unique(cold_local_full, return_inverse=True)  # only the ones used now
            # gather + stream the routed cold experts' NF4 to the GPU
            gu_p = self.c_gu_p.index_select(0, routed).to(dev, non_blocking=True)
            gu_a = self.c_gu_a.index_select(0, routed).to(dev, non_blocking=True)
            dn_p = self.c_dn_p.index_select(0, routed).to(dev, non_blocking=True)
            dn_a = self.c_dn_a.index_select(0, routed).to(dev, non_blocking=True)
            xr = x.index_select(0, row_token.index_select(0, cr))
            dn = _fused_over_stack(xr, compact.to(dev), gu_p, gu_a, dn_p, dn_a,
                                   self.shapes, self.has_gate, self.act_fn)
            w = top_k_weights[row_token.index_select(0, cr), row_slot.index_select(0, cr)].to(torch.float32)
            out.index_add_(0, row_token.index_select(0, cr), dn.to(torch.float32) * w[:, None])

        return out.to(input_dtype)


def hot_residency_available() -> bool:
    try:
        from nf4_grouped import gemm_4bit_grouped  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _eligible(mod):
    if getattr(mod, "bits", None) != 4 or getattr(mod, "quant_type", None) != "nf4":
        return "storage is not nf4-4bit"
    if getattr(mod, "blocksize", None) != 64:
        return f"blocksize {getattr(mod, 'blocksize', None)} != 64"
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    if k1 % 64 or k2 % 64:
        return "K not divisible by 64"
    return None


def enable_hot_residency(model, hot_sets: Sequence, device: str = "cuda",
                         verbose: bool = False) -> int:
    """Partition every eligible ``ExpertsNbit`` under ``model`` into a resident
    GPU hot-stack + a streamed CPU cold-stack, in MoE-layer order.

    ``hot_sets`` is a sequence with one entry per MoE layer (module order): a
    1-D array/list of hot expert ids for that layer. Returns the number of
    modules converted. Modules that override ``forward`` (custom-activation
    experts, e.g. gpt-oss) or have ineligible storage are skipped."""
    from experts4bit_qlora import Experts4bit, ExpertsNbit

    stock_forwards = {ExpertsNbit.forward, Experts4bit.forward}
    mods = [m for m in model.modules() if isinstance(m, ExpertsNbit)] if hasattr(model, "modules") else [model]
    n = 0
    for mod in mods:
        if type(mod).forward not in stock_forwards:
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: custom forward")
            continue
        reason = _eligible(mod)
        if reason is not None:
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: {reason}")
            continue
        if n >= len(hot_sets):
            break
        state = _HotResidency(mod, hot_sets[n], device)
        if not hasattr(mod, "_e4b_reference_forward"):
            mod._e4b_reference_forward = mod.forward
        mod._hot_residency = state

        def _fwd(hidden, top_k_index, top_k_weights, _m=mod):
            if torch.is_grad_enabled() and (
                hidden.requires_grad or any(p.requires_grad for p in _m.parameters())
            ):
                return _m._e4b_reference_forward(hidden, top_k_index, top_k_weights)
            return _m._hot_residency.forward(hidden, top_k_index, top_k_weights)

        mod.forward = _fwd
        n += 1
    return n


def disable_hot_residency(model) -> int:
    """Undo :func:`enable_hot_residency`; returns the number of modules restored."""
    mods = model.modules() if hasattr(model, "modules") else [model]
    restored = 0
    for mod in mods:
        if hasattr(mod, "_e4b_reference_forward") and hasattr(mod, "_hot_residency"):
            mod.forward = mod._e4b_reference_forward
            del mod._e4b_reference_forward, mod._hot_residency
            restored += 1
    return restored
