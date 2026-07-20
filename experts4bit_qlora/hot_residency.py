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


def _fused_over_stack(x_rows, local_ids, gu_p, gu_a, dn_p, dn_a, shapes, has_gate,
                      act_fn, gptoss=None):
    """Down-projection outputs for each (token,slot) row, computed on the device
    the packed stack lives on. ``local_ids`` index into the G-expert stack
    (``gu_p`` is ``[G, n1, k1//2]`` etc.). Returns ``[R, H]`` in the input row
    order (unweighted; the caller applies router scores and scatters).

    ``gptoss``, when given, is ``(gu_bias, dn_bias, alpha, limit)`` where the
    bias stacks are aligned to the SAME local indexing as ``gu_p``/``dn_p``.
    It selects the gpt-oss expert epilogue — per-expert biases + the clamped
    GLU ``(up+1)*(gate*sigmoid(alpha*gate))`` — instead of the plain
    ``act_fn(gate)*up``. gpt-oss weights are de-interleaved to a contiguous
    ``[gate; up]`` layout at load (``gptoss.py``), so ``chunk(2)`` is the
    correct split here (NOT ``[...::2]``). Mirrors ``_GptOssForwardMixin.forward``
    exactly (the correctness oracle)."""
    from nf4_grouped import gemm_4bit_grouped

    n1, k1, n2, k2 = shapes
    order = torch.argsort(local_ids)
    sorted_ids = local_ids.index_select(0, order)
    x_sorted = x_rows.index_select(0, order).contiguous()
    uniq, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    sizes = counts.tolist()
    eids = uniq.tolist()
    gu = gemm_4bit_grouped(x_sorted, gu_p, gu_a, sizes, eids)
    if gptoss is not None:
        gu_bias, dn_bias, alpha, limit = gptoss
        gu = gu + gu_bias.index_select(0, sorted_ids).to(gu.dtype)  # per-expert bias by local id
        gate, up = gu.chunk(2, dim=-1)                              # de-interleaved at load
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        h = (up + 1) * (gate * torch.sigmoid(gate * alpha))
        dn = gemm_4bit_grouped(h.contiguous(), dn_p, dn_a, sizes, eids)
        dn = dn + dn_bias.index_select(0, sorted_ids).to(dn.dtype)
    elif has_gate:
        gate, up = gu.chunk(2, dim=-1)
        h = act_fn(gate) * up
        dn = gemm_4bit_grouped(h.contiguous(), dn_p, dn_a, sizes, eids)
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
        if hot_ids.numel() and (hot_ids.min() < 0 or hot_ids.max() >= E):
            raise ValueError(f"hot ids must lie in [0, {E}); got range "
                             f"[{int(hot_ids.min())}, {int(hot_ids.max())}]")
        cold_ids = torch.tensor([e for e in range(E) if e not in set(hot_ids.tolist())],
                                dtype=torch.long)
        self.hot_ids, self.cold_ids = hot_ids, cold_ids
        n1, k1 = mod._gate_up_shape
        n2, k2 = mod._down_shape
        self.shapes = (n1, k1, n2, k2)
        self.has_gate = mod.has_gate
        self.act_fn = mod.act_fn
        self.compute_dtype = mod.compute_dtype

        # gpt-oss epilogue: per-expert biases (de-interleaved to contiguous
        # [gate;up] at load) + clamped GLU. Biases are tiny — keep the hot AND
        # cold sub-stacks resident on the compute device, aligned to the same
        # local id order as the packed hot/cold stacks below.
        self.gptoss = getattr(mod, "alpha", None) is not None and hasattr(mod, "gate_up_bias")
        if self.gptoss:
            self.alpha, self.limit = float(mod.alpha), float(mod.limit)

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
        if self.gptoss:
            gub, dnb = mod.gate_up_bias, mod.down_bias           # [E, 2I] / [E, H], contiguous
            self.h_gu_b = gub.index_select(0, hi).contiguous().to(self.device)
            self.h_dn_b = dnb.index_select(0, hi).contiguous().to(self.device)
            self.c_gu_b = gub.index_select(0, ci).contiguous().to(self.device)
            self.c_dn_b = dnb.index_select(0, ci).contiguous().to(self.device)
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
        input_dev = hidden_states.device
        # read compute_dtype LIVE off the module (a later change must be honored)
        cd = self.mod.compute_dtype if self.mod.compute_dtype is not None else input_dtype
        dev = self.device
        x = hidden_states.to(device=dev, dtype=cd)
        top_k_weights = top_k_weights.to(dev)
        T, H = x.shape
        k = top_k_index.shape[1]

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
            gptoss = ((self.h_gu_b, self.h_dn_b, self.alpha, self.limit)
                      if self.gptoss else None)
            dn = _fused_over_stack(xr, local, self.h_gu_p, self.h_gu_a, self.h_dn_p,
                                   self.h_dn_a, self.shapes, self.has_gate, self.act_fn,
                                   gptoss=gptoss)
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
            # bias sub-stack aligned to the streamed `routed` subset (compact indexes into
            # it); `routed` is a CPU index, the bias stacks are device-resident.
            gptoss = None
            if self.gptoss:
                r_dev = routed.to(dev)
                gptoss = (self.c_gu_b.index_select(0, r_dev), self.c_dn_b.index_select(0, r_dev),
                          self.alpha, self.limit)
            dn = _fused_over_stack(xr, compact.to(dev), gu_p, gu_a, dn_p, dn_a,
                                   self.shapes, self.has_gate, self.act_fn, gptoss=gptoss)
            w = top_k_weights[row_token.index_select(0, cr), row_slot.index_select(0, cr)].to(torch.float32)
            out.index_add_(0, row_token.index_select(0, cr), dn.to(torch.float32) * w[:, None])

        return out.to(device=input_dev, dtype=input_dtype)


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

    ``hot_sets`` must carry exactly one entry per targeted ``ExpertsNbit`` module
    in module order (a 1-D array/list of hot expert ids each); a wrong length
    raises. Re-enabling rebuilds the partition from the module's *current*
    weights (never a stale cache). **gpt-oss experts (custom clamped-GLU +
    per-expert biases) ARE supported** — the hot path reproduces their
    epilogue. Modules with OTHER custom ``forward`` overrides, ineligible
    storage, ``[fast]``-enabled, or an ``ExpertsLoRA`` base (the
    streaming-loader path — not yet supported) are skipped; ids outside
    ``[0, num_experts)`` raise.

    Memory model: on a fully-resident module the hot (GPU) and cold (CPU) stacks
    are *added* — the module keeps its original packed weights so the reference
    fallback still works, so VRAM is not reduced in that configuration. The VRAM
    win is realized when the base experts are offloaded (streaming loader): the
    resident stack is then the only GPU copy. Standalone Experts4bit is the
    correctness-supported path today."""
    from experts4bit_qlora import Experts4bit, ExpertsNbit

    stock_forwards = {ExpertsNbit.forward, Experts4bit.forward}
    # gpt-oss experts override forward (clamped-GLU + biases); the hot path now
    # reproduces that epilogue (_fused_over_stack gptoss branch), so treat their
    # forwards as supported rather than skipping them as "custom".
    try:
        from experts4bit_qlora.gptoss import GptOssExperts4bit, GptOssExpertsNbit
        stock_forwards |= {GptOssExperts4bit.forward, GptOssExpertsNbit.forward}
    except ImportError:
        pass
    if hasattr(model, "modules"):
        # ExpertsLoRA.forward bypasses base.forward (it calls base._project), so the
        # frozen base is NEVER dispatched — patching it is dead code that only
        # duplicates weights. Exclude any Experts4bit that is an ExpertsLoRA.base.
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
                "every ExpertsNbit here is an ExpertsLoRA.base (the streaming-loader / "
                "offload path). ExpertsLoRA.forward bypasses base.forward, so residency "
                "must hook the wrapper + its offload homes — a separate increment. "
                "enable_hot_residency currently supports standalone Experts4bit modules.")
    else:
        mods = [model]
    if len(hot_sets) != len(mods):
        raise ValueError(
            f"hot_sets has {len(hot_sets)} entries but the model has {len(mods)} "
            f"ExpertsNbit modules — exactly one entry per MoE layer in module "
            f"order is required (skipped layers still consume their entry, so "
            f"alignment never silently shifts and trailing entries are never "
            f"dropped)")
    patched = 0
    for i, mod in enumerate(mods):  # hot_sets[i] belongs to mods[i], patched or not
        if type(mod).forward not in stock_forwards and not hasattr(mod, "_e4b_hot_ref"):
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: custom forward")
            continue
        reason = _eligible(mod)
        if reason is not None:
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: {reason}")
            continue
        if hasattr(mod, "_e4b_fast_ref"):
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: [fast] enabled — disable it first")
            continue
        if hasattr(mod, "_e4b_pipe_ref"):
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: pipelined residency enabled — disable it first")
            continue
        if hasattr(mod, "_hot_residency"):
            # rebuild every time — the base weights are frozen NF4, but a caller may
            # have reloaded a checkpoint; a cached partition must never go stale.
            mod._hot_residency = _HotResidency(mod, hot_sets[i], device)
            patched += 1
            continue
        state = _HotResidency(mod, hot_sets[i], device)
        mod._e4b_hot_ref = mod.forward
        mod._hot_residency = state

        def _fwd(hidden, top_k_index, top_k_weights, _m=mod):
            st = _m._hot_residency
            cd = _m.compute_dtype if _m.compute_dtype is not None else hidden.dtype
            if cd not in (torch.bfloat16, torch.float16):
                return _m._e4b_hot_ref(hidden, top_k_index, top_k_weights)
            if torch.is_grad_enabled() and (
                hidden.requires_grad or any(p.requires_grad for p in _m.parameters())
            ):
                return _m._e4b_hot_ref(hidden, top_k_index, top_k_weights)
            return st.forward(hidden, top_k_index, top_k_weights)

        mod.forward = _fwd
        patched += 1
    return patched


def disable_hot_residency(model) -> int:
    """Undo :func:`enable_hot_residency`; returns the number of modules restored."""
    mods = model.modules() if hasattr(model, "modules") else [model]
    restored = 0
    for mod in mods:
        if hasattr(mod, "_e4b_hot_ref") and hasattr(mod, "_hot_residency"):
            mod.forward = mod._e4b_hot_ref
            del mod._e4b_hot_ref, mod._hot_residency
            restored += 1
    return restored
