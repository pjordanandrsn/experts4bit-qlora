# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Hybrid backward (Phase 5): QLoRA training over the three-tier engine.

The frozen quantized base takes no weight gradients, so its backward is one
thing — ``grad_in = grad_out @ W`` per routed expert, on whatever bus holds
that expert's bytes. Everything else (GLU, LoRA deltas, routing weights,
token scatter) is ordinary differentiable torch, so autograd owns it.

Structure per MoE layer (mirrors the ``[fast]`` train lane's shape, with
hybrid storage under the frozen projections):

    x_rows  = hidden[token per assignment]           # autograd gather
    y_gu    = _HybridProjFn(x_rows, "gu")            # frozen, three buses
            + LoRA gate_up delta                     # BEFORE the GLU —
    h       = GLU(y_gu)                              # act(Wx+BAx) != act(Wx)+d
    y_dn    = _HybridProjFn(h, "dn") + LoRA down delta
    out     = scatter-add(router_weight * y_dn)

``_HybridProjFn.forward`` routes each bus's rows through that bus's grouped
kernel — DRAM: ``cpu_grouped.gemv_nf4_grouped_cpu`` on the packed host
stacks; VRAM / NVMe(staged): ``nf4_grouped.gemm_4bit_grouped`` on the
device stacks. Its backward runs the matching dgrad — DRAM:
``cpu_grouped.dgrad_nf4_grouped_cpu`` (Phase 5's transposed-access kernel,
L1 tile scratch, no transposed copy ever exists); VRAM / NVMe:
``nf4_grouped.dgrad_4bit_grouped``. Only the projection INPUT is saved;
NVMe sub-stacks re-stage from the tier in backward (checkpoint semantics —
the tier is the recompute cache).

LoRA adapter math stays GPU-side by construction: deltas are plain torch on
the device, outside the Functions entirely. This seam patches
``ExpertsLoRA.forward`` (which deliberately never calls ``base.forward`` in
training — the delta must land before the nonlinearity, see
``lora._delegate_to_base``), and bare ``Experts4bit`` modules get the same
path with zero deltas.

Gradient checkpointing: default-on per the directive means the TRAINER
wraps decoder layers (HF ``gradient_checkpointing_enable`` composes — this
path is recompute-safe). The Functions themselves already save only their
projection inputs.

Known limits (v1): NF4 storage (the shipped hybrid arena format); the
gpt-oss biased epilogue is refused at enable time rather than shipped
unexercised. Gradient scatters share the serve path's batch-collision
atomics nondeterminism on CUDA (the open defect filed with G3 formal).
"""
from __future__ import annotations

import torch

from .hot_residency import target_modules

_MARKER = "_e4b_hybrid_train"


# --------------------------------------------------------------------------- #
# routing plan: computed once per forward, shared by both projections
# --------------------------------------------------------------------------- #

class _Plan:
    __slots__ = ("st", "buses", "n_rows")

    def __init__(self, st, buses, n_rows):
        self.st = st
        self.buses = buses
        self.n_rows = n_rows


def _grouped(local):
    order = torch.argsort(local)
    sl = local.index_select(0, order)
    uniq, counts = torch.unique_consecutive(sl, return_counts=True)
    return order, sl, uniq, counts


def _make_plan(st, flat):
    buses = []
    hot_row = st.is_hot[flat]
    hr = hot_row.nonzero(as_tuple=False).view(-1)
    cr = (~hot_row).nonzero(as_tuple=False).view(-1)
    if hr.numel():
        local = st.g2h[flat.index_select(0, hr)]
        order, _sl, uniq, counts = _grouped(local)
        buses.append({"bus": "hot", "rr": hr, "order": order,
                      "sizes": counts.tolist(),
                      "eids": uniq.to(torch.int32)})
    if cr.numel():
        dmask = st.is_dram[flat.index_select(0, cr)]
        nr = cr[~dmask]
        dr = cr[dmask]
        if nr.numel():
            glob = flat.index_select(0, nr).cpu()
            local = st.g2c_cpu.index_select(0, glob)
            order, _sl, uniq, counts = _grouped(local)
            # staged sub-stack is exactly `uniq` in order, so kernel eids
            # are positions 0..G-1 into it
            buses.append({"bus": "cold", "rr": nr, "order": order.to(nr.device),
                          "sizes": counts.tolist(), "routed": uniq,
                          "eids": torch.arange(uniq.numel(),
                                               dtype=torch.int32,
                                               device=nr.device)})
        if dr.numel():
            from .hybrid import _split_oversize_groups
            glob = flat.index_select(0, dr).cpu()
            local = st.g2d_cpu.index_select(0, glob)
            order, _sl, uniq, counts = _grouped(local)
            # the CPU FORWARD gemv carries the decode contract (sizes 1..8),
            # so oversize groups split; the dgrad kernel has no cap and the
            # split is chain-identical for it (rows are independent)
            sizes, eids = _split_oversize_groups(counts.tolist(),
                                                 uniq.tolist())
            buses.append({"bus": "dram", "rr": dr, "order": order.to(dr.device),
                          "sizes": sizes, "eids": eids})
    return _Plan(st, buses, int(flat.numel()))


# --------------------------------------------------------------------------- #
# the frozen grouped projection over hybrid storage
# --------------------------------------------------------------------------- #

def _stacks(st, b, which, dev):
    """(packed, absmax) for one bus and one projection, staged if NVMe."""
    if b["bus"] == "hot":
        return ((st.h_gu_p, st.h_gu_a) if which == "gu"
                else (st.h_dn_p, st.h_dn_a))
    if b["bus"] == "cold":
        routed = b["routed"]
        if which == "gu":
            p = st.c_gu_p.index_select(0, routed)
            a = st.c_gu_a.index_select(0, routed)
        else:
            p = st.c_dn_p.index_select(0, routed)
            a = st.c_dn_a.index_select(0, routed)
        return p.to(dev, non_blocking=True), a.to(dev, non_blocking=True)
    return ((st.d_gu_p, st.d_gu_a) if which == "gu"
            else (st.d_dn_p, st.d_dn_a))


class _HybridProjFn(torch.autograd.Function):
    """``y[row] = W[e(row)] @ x[row]`` with frozen packed W, per bus."""

    @staticmethod
    def forward(ctx, x_rows, plan, which):
        st = plan.st
        dev = x_rows.device
        n_out = st.shapes[0] if which == "gu" else st.shapes[2]
        y = torch.zeros(x_rows.shape[0], n_out, dtype=torch.float32,
                        device=dev)
        for b in plan.buses:
            xb = x_rows.index_select(0, b["rr"]).index_select(0, b["order"])
            packed, absmax = _stacks(st, b, which, dev)
            if b["bus"] == "dram":
                import cpu_grouped
                ys = cpu_grouped.gemv_nf4_grouped_cpu(
                    xb.to("cpu", torch.float32).contiguous(), packed, absmax,
                    b["sizes"], b["eids"], threads=st._threads).to(dev)
            else:
                from nf4_grouped import gemm_4bit_grouped
                ys = gemm_4bit_grouped(
                    xb.to(torch.bfloat16).contiguous(), packed, absmax,
                    b["sizes"], b["eids"]).to(torch.float32)
            yb = torch.empty_like(ys)
            yb.index_copy_(0, b["order"], ys)
            y.index_copy_(0, b["rr"], yb)
        ctx.plan = plan                # indices only; inputs are the caller's
        ctx.which = which
        return y

    @staticmethod
    def backward(ctx, grad_y):
        plan, which = ctx.plan, ctx.which
        st = plan.st
        dev = grad_y.device
        k_in = st.shapes[1] if which == "gu" else st.shapes[3]
        gx = torch.zeros(grad_y.shape[0], k_in, dtype=torch.float32,
                         device=dev)
        for b in plan.buses:
            gb = grad_y.index_select(0, b["rr"]).index_select(0, b["order"])
            packed, absmax = _stacks(st, b, which, dev)
            if b["bus"] == "dram":
                import cpu_grouped
                gs = cpu_grouped.dgrad_nf4_grouped_cpu(
                    gb.to("cpu", torch.float32).contiguous(), packed, absmax,
                    b["sizes"], b["eids"], threads=st._threads).to(dev)
            else:
                from nf4_grouped import dgrad_4bit_grouped
                gs = dgrad_4bit_grouped(
                    gb.to(torch.bfloat16).contiguous(), packed, absmax,
                    b["sizes"], b["eids"]).to(torch.float32)
            gxb = torch.empty_like(gs)
            gxb.index_copy_(0, b["order"], gs)
            gx.index_copy_(0, b["rr"], gxb)
        return gx, None, None


# --------------------------------------------------------------------------- #
# the training forward (module seam)
# --------------------------------------------------------------------------- #

def _glu(st, gu):
    if st.has_gate:
        gate, up = gu.chunk(2, dim=-1)
        if st.clamp_limit is not None:
            gate = gate.clamp(max=st.clamp_limit)
            up = up.clamp(min=-st.clamp_limit, max=st.clamp_limit)
        return st.act_fn(gate) * up
    return st.act_fn(gu)


def _lora_delta(x_rows, flat, lora_A, lora_B, scaling):
    """Per-assignment low-rank delta, grouped-padded like the fast lane
    (per-row adapter gathers at hidden width would cost [R, r, in])."""
    from .batched import _lora_delta_padded
    order = torch.argsort(flat)
    sf = flat.index_select(0, order)
    uniq, counts = torch.unique_consecutive(sf, return_counts=True)
    maxn = int(counts.max())
    g = uniq.numel()
    xs = x_rows.index_select(0, order)
    x_pad = x_rows.new_zeros(g, maxn, x_rows.shape[-1])
    starts = torch.cumsum(torch.tensor([0] + counts.tolist()[:-1]), 0)
    pos = torch.arange(sf.numel(), device=x_rows.device)
    seg = torch.repeat_interleave(
        torch.arange(g, device=x_rows.device), counts.to(x_rows.device))
    off = pos - starts.to(x_rows.device).index_select(0, seg)
    x_pad[seg, off] = xs
    d_pad = _lora_delta_padded(x_pad, lora_A, lora_B, uniq, scaling)
    ds = d_pad[seg, off]
    d = torch.empty_like(ds)
    d.index_copy_(0, order, ds)
    return d


def _train_forward(st, lora, hidden, top_k_index, top_k_weights):
    dev = st.device
    cd = (st.mod.compute_dtype if st.mod.compute_dtype is not None
          else hidden.dtype)
    t, k = top_k_index.shape
    x = hidden.to(device=dev, dtype=cd)
    wts = top_k_weights.to(dev)
    flat = top_k_index.reshape(-1).to(dev)
    row_token = torch.arange(t * k, device=dev) // k
    plan = _make_plan(st, flat)

    x_rows = x.index_select(0, row_token).to(torch.float32)
    y_gu = _HybridProjFn.apply(x_rows, plan, "gu")
    if lora is not None:
        y_gu = y_gu + _lora_delta(x_rows.to(cd), flat, lora.gate_up_lora_A,
                                  lora.gate_up_lora_B, lora.scaling
                                  ).to(torch.float32)
    h = _glu(st, y_gu)
    y_dn = _HybridProjFn.apply(h, plan, "dn")
    if lora is not None:
        y_dn = y_dn + _lora_delta(h.to(cd), flat, lora.down_lora_A,
                                  lora.down_lora_B, lora.scaling
                                  ).to(torch.float32)

    w_rows = wts.reshape(-1).to(torch.float32)
    # rows are dense token-major/slot-minor by construction, so the weighted
    # combine is a reshape + fixed-order sum — no scatter, no atomics,
    # deterministic and differentiable (index_add with duplicate token rows
    # was the same nondeterminism the serve path carried)
    out = (y_dn * w_rows[:, None]).view(t, k, -1).sum(dim=1)
    return out.to(device=hidden.device, dtype=hidden.dtype)


# --------------------------------------------------------------------------- #
# enable / disable
# --------------------------------------------------------------------------- #

def enable_hybrid_train(model, arena_path: str, manifest, **kw) -> int:
    """``enable_hybrid_tier`` + the training seam. Modules wrapped in
    ``ExpertsLoRA`` get the seam on the WRAPPER (its inline forward is the
    only place the gate_up delta can land before the GLU); bare base
    modules get it with zero deltas. Inference calls (no grad, or nothing
    requiring grad) pass through untouched. Returns the tier patch count."""
    from .hybrid import enable_hybrid_tier
    from ..lora import ExpertsLoRA

    n = enable_hybrid_tier(model, arena_path, manifest, **kw)
    targets = set(map(id, target_modules(model)))
    lora_of = {id(m.base): m for _, m in model.named_modules()
               if isinstance(m, ExpertsLoRA) and id(m.base) in targets}

    for mod in target_modules(model):
        st = mod._hot_residency
        if st.gptoss:
            from .hybrid import disable_hybrid_tier
            disable_hybrid_tier(model)
            raise NotImplementedError(
                "hybrid train does not ship the gpt-oss biased epilogue "
                "yet — refusing rather than running it unexercised")
        lora = lora_of.get(id(mod))
        host = lora if lora is not None else mod
        serve = host.forward
        base_serve = mod.forward       # the tier-installed serve wrapper

        def _fwd(hidden, top_k_index, top_k_weights,
                 _st=st, _lora=lora, _serve=serve, _base_serve=base_serve):
            if torch.is_grad_enabled() and (
                    hidden.requires_grad or top_k_weights.requires_grad
                    or (_lora is not None and any(
                        p.requires_grad for p in _lora.parameters()
                        if p is not None))):
                return _train_forward(_st, _lora, hidden, top_k_index,
                                      top_k_weights)
            # No-grad routing must not pass through ExpertsLoRA.forward:
            # its delegation predicate refuses under model.training, and
            # the inline fallback reads base storage the streaming loader
            # never materializes — reentrant checkpointing's no-grad outer
            # pass and any train()+no_grad validation forward would leave
            # the tier and crash at scale (Bugbot, HIGH). Route on the
            # adapter's own state instead:
            if _lora is not None:
                # FRESH zero test, never the cached _adapter_is_zero: an
                # optimizer step mutates B without invalidating that cache,
                # so a True cached during warm-up (B still zero) would
                # silently drop the trained deltas from every later no-grad
                # forward. Two device reductions per call — noise against a
                # validation forward.
                if not (bool(_lora.gate_up_lora_B.any())
                        or bool(_lora.down_lora_B.any())):
                    # delta identically zero: the fused tier serve path,
                    # immune to the training flag
                    return _base_serve(hidden, top_k_index, top_k_weights)
                # trained adapter under no-grad: deltas must still apply and
                # the fused serve path cannot inject them — run the train
                # forward's math grad-free (buses + deltas, no base-storage
                # reads); slower than fused serve, correct at any scale
                return _train_forward(_st, _lora, hidden, top_k_index,
                                      top_k_weights)
            return _serve(hidden, top_k_index, top_k_weights)

        host.forward = _fwd
        setattr(host, _MARKER, serve)
    return n


def disable_hybrid_train(model) -> int:
    """Remove the seam (restoring the forward it wrapped), then tear down
    the tier."""
    from .hybrid import disable_hybrid_tier

    for _, m in model.named_modules():
        if hasattr(m, _MARKER):
            m.forward = getattr(m, _MARKER)
            delattr(m, _MARKER)
    return disable_hybrid_tier(model)
