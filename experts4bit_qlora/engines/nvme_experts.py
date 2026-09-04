"""NVMe-backed cold tail — serving MoE experts that do not fit host RAM.

:mod:`.hot_residency` pins each layer's hottest experts in VRAM and streams the
cold tail from four fully-materialized pinned host stacks. That assumption breaks
at scale: Kimi K3's experts total **1.446 TB** (93 layers x 896 experts x
~17.5 MB, measured against the real checkpoint) while a rented pod exposes
**503 GB** — a per-container ceiling that renting more GPUs does not raise. Yet
top-16-of-896 routing touches only ~26 GB per token, so the cold tail never needs
to be resident, only *reachable*.

This module makes it reachable. The four cold stacks become
:class:`_TieredStack` views over ``grouped-nf4-gemm``'s N-series arena
(``nvme_arena`` bakes it, ``nvme_reader`` reads at ~99% of the device link,
``nvme_residency.ColdTier`` decides what stays in pinned DRAM). Nothing in
``forward()`` or ``_cold_contrib()`` changes: they only ever call
``.index_select(0, routed)`` then ``.to(dev)``, which is exactly the contract a
``_TieredStack`` honours.

**Provenance.** A relocation bake (``bake``/``bake_expert_tensors``) is
single-source and hash-preserving, so the bytes served here are bit-identical to
the bytes in the checkpoint; ``nvme_arena.verify(against_source=...)`` re-checks
that, and ``nvme_residency``'s own gates check that the tier and the tensor
reconstruction do not perturb them. A *quantize-at-bake* arena
(``nvme_bake_nf4``) is a documented two-hop transform instead — bit-identical to
the quantizer's output, not to a bf16 release. Which claim applies is a property
of the arena, and its manifest records it as ``bake_mode``.

Usage::

    from experts4bit_qlora.engines.nvme_experts import enable_nvme_residency
    n = enable_nvme_residency(model, arena_path, hot_sets, hot_rows=20_000)
"""
from __future__ import annotations

import types
from typing import Sequence

import torch

from .hot_residency import _HotResidency, enable_hot_residency
from ..util import log

# Segment suffixes written by grouped-nf4-gemm's NF4 quantize-at-bake, mapped to
# the four cold attributes they replace. Geometry (per-expert shape + dtype) is
# read from the arena index, never assumed here: gate_up blocks are
# ``[2I, H/2]`` u8 and absmax ``[2I, H/64]`` f32, which already match
# ``[n1, k1//2]`` / ``[n1, k1//64]`` float32 — so no cast is needed on this path.
NF4_SEGMENTS = {
    "c_gu_p": "nf4.gate_up_blocks",
    "c_gu_a": "nf4.gate_up_absmax",
    "c_dn_p": "nf4.down_blocks",
    "c_dn_a": "nf4.down_absmax",
}


class _TieredStack:
    """One arena segment, exposed as the slice of a cold stack the engine wants.

    Duck-types the *only* two operations ``_cold_contrib`` performs on a cold
    stack — ``.index_select(0, routed)`` then ``.to(dev)`` — so the forward path
    is untouched. ``routed`` indexes the cold stack locally; this maps it back to
    global expert ids, makes those rows resident (reading from NVMe on a miss),
    and reconstructs the segment as a ``[R, *shape]`` tensor.
    """

    def __init__(self, tier, index: dict, layer: int, cold_ids: torch.Tensor,
                 suffix: str, *, cast=None):
        self.tier, self.index, self.layer = tier, index, layer
        self.cold_ids = cold_ids.cpu()
        self.suffix, self.cast = suffix, cast
        geo = next((g for g in index["segments"] if g["suffix"] == suffix), None)
        if geo is None:
            raise KeyError(
                f"arena has no segment {suffix!r}; present: "
                f"{[g['suffix'] for g in index['segments']]}. A quantize-at-bake "
                f"arena (nvme_bake_nf4) is required for the NF4 serving path.")
        self.shape_per_expert = tuple(geo["shape_per_expert"])

    def _view_stack(self):
        """``(view, stack)`` when handing back rows of the cold view's stack is
        byte-identical to rebuilding them, else ``(None, None)``.

        ``None`` — meaning take the rebuild path — when there is no view, when
        it does not materialize this segment, or when its stack dtype differs
        from this segment's contract. That last case is real: the view widens
        narrow absmax to f32 for the CPU kernel, and a caller that asked for
        the stored dtype must not silently receive f32. The dtype is compared
        against the ARENA's own geometry rather than assumed, because the
        whole point of this class is that it does not reinterpret bytes.
        """
        view = getattr(self.tier, "_e4b_cold_view", None)
        if view is None or self.suffix not in getattr(view, "segments", ()):
            return None, None
        # The A/B control, in the shape gnf4#133 set for DevRowCache: turning
        # this off is not a degraded mode, it is the engine that shipped
        # before this change, byte for byte, so the two can be compared
        # without a second code path to trust. The CPU destination keeps
        # using the view either way -- this switch governs only whether the
        # GPU stack reads it.
        if not getattr(view, "e4b_serve_gpu_stacks", True):
            return None, None
        try:
            stack = view.stack(self.suffix)
        except KeyError:
            return None, None
        from nvme_residency import segment_geometry
        native, _shape, _off, _ln = segment_geometry(self.index, self.suffix)
        want = self.cast if self.cast is not None else native
        return (view, stack) if stack.dtype == want else (None, None)

    def index_select(self, dim: int, idx: torch.Tensor) -> torch.Tensor:
        if dim != 0:
            raise ValueError(f"_TieredStack indexes experts on dim 0, got {dim}")
        globals_ = self.cold_ids.index_select(0, idx.cpu().to(torch.long))
        # Safe against the hybrid prefetch worker with no locking here: the
        # tier's DEMAND WINDOW (nvme_residency) keeps every row of the latest
        # demand ensure unevictable until the next demand ensure, which spans
        # this whole ensure -> row-reads sequence. (Before that contract, a
        # concurrent speculative ensure could evict between the two — measured
        # at 235B as KeyError '(layer 81, expert 73) not resident'.)
        view, stack = self._view_stack()
        if view is not None:
            # The view already holds kernel-shaped rows and knows which slots
            # are still current, so an expert materialized on an earlier step
            # costs nothing here — where segment_tensor pays a host copy per
            # row per call regardless. Gate 1 attributes ~98% of cold cost to
            # exactly this staging, not to disk.
            #
            # ensure() is the tier's demand ensure plus the view's own
            # (key, generation) check, so this is not a weaker residency
            # guarantee than the rebuild path -- it is the same one, with the
            # copy skipped when the slot is unchanged.
            slots = view.ensure(self.layer, globals_.tolist())
            return stack.index_select(
                0, torch.as_tensor(slots, dtype=torch.long))
        from nvme_residency import segment_tensor
        return segment_tensor(self.tier, self.index, self.layer,
                              globals_.tolist(), self.suffix, cast=self.cast)

    # a cold stack is only ever indexed, but keep these honest rather than absent
    def __len__(self):
        return int(self.cold_ids.numel())

    @property
    def shape(self):
        return (len(self),) + self.shape_per_expert

    def __repr__(self):
        return (f"_TieredStack({self.suffix!r}, layer={self.layer}, "
                f"cold={len(self)}, per_expert={self.shape_per_expert})")


class _NvmeResidency(_HotResidency):
    """Hot residency whose cold tail lives on NVMe instead of in host RAM.

    The hot partition, the id algebra, the gpt-oss bias epilogue and the fused
    kernel path are all inherited unchanged — only where the cold bytes come
    from differs.
    """

    _PIN_COLD = False           # nothing to pin: the tier owns its pinned arena

    def _build_hot(self, gu_p, gu_a, dn_p, dn_a, hi):
        """Read the hot partition from the arena instead of from ``mod``.

        This is what lets the module itself be built on ``meta``: nothing ever
        indexes its ``[E, ...]`` expert buffers, so they need not be allocated.
        For Kimi K3 that is the difference between 1.446 TB of expert storage and
        none — the hot experts land on the GPU, the cold tail stays on NVMe, and
        the module holds only shapes.

        Hot rows transit the tier in chunks of at most ``hot_rows`` so a hot set
        larger than the pinned arena still works. Once copied to the device they
        are never requested again, so LFU reclaims their slots naturally.
        """
        from nvme_residency import segment_tensor

        # SETUP tier, not the serving one. These are one-shot bulk reads that
        # materialize the resident stacks; once copied to the device they are
        # never requested again. Reading them through the serving tier filled
        # its slots with experts that then live permanently in VRAM/DRAM, and
        # -- since an external-landing tier cannot serve row() at all -- made
        # the direct cold landing impossible to attach. enable_hybrid_tier
        # stamps _e4b_setup_tier for the duration of the patch and drops it
        # after; without one this falls back to the serving tier, which is the
        # pre-existing behaviour.
        tier = getattr(self.mod, "_e4b_setup_tier", None) or self.mod._e4b_cold_tier
        index = tier.reader.index
        layer = int(getattr(self.mod, "_e4b_arena_layer", 0))
        ids = [int(e) for e in self.hot_ids.tolist()]
        chunk = max(1, min(len(ids) or 1, tier.hot_rows))
        for attr, suffix in (("h_gu_p", NF4_SEGMENTS["c_gu_p"]),
                             ("h_gu_a", NF4_SEGMENTS["c_gu_a"]),
                             ("h_dn_p", NF4_SEGMENTS["c_dn_p"]),
                             ("h_dn_a", NF4_SEGMENTS["c_dn_a"])):
            if not ids:
                # An all-cold layer: keep an empty tensor of the right rank so the
                # fused path's shape algebra still holds.
                geo = next(g for g in index["segments"] if g["suffix"] == suffix)
                shp = (0,) + tuple(geo["shape_per_expert"])
                dt = torch.float32 if geo["dtype"] == "F32" else torch.uint8
                setattr(self, attr, torch.empty(shp, dtype=dt, device=self.device))
                continue
            parts = [segment_tensor(tier, index, layer, ids[i:i + chunk], suffix)
                     for i in range(0, len(ids), chunk)]
            stacked = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            setattr(self, attr, stacked.contiguous().to(self.device))

    def _build_cold(self, gu_p, gu_a, dn_p, dn_a, ci):
        tier = self.mod._e4b_cold_tier
        index = tier.reader.index
        layer = int(getattr(self.mod, "_e4b_arena_layer", 0))
        for attr, suffix in NF4_SEGMENTS.items():
            setattr(self, attr, _TieredStack(tier, index, layer,
                                             self.cold_ids, suffix))
        self._tier = tier

    def tier_stats(self) -> dict:
        return self._tier.stats()


def expert_geometry_from_arena(index: dict) -> tuple:
    """Recover ``(intermediate_dim, hidden_dim)`` from an arena's index.

    Geometry comes from the ARENA, not from the model config, and that is
    deliberate. For a plain MoE the expert input width equals ``hidden_size``, but
    for a **latent** MoE it does not: Kimi K3's experts sit behind
    ``routed_expert_{down,up}_proj`` and operate on a 3584-wide latent, while
    ``hidden_size`` is 7168. Reading the config would size every expert twice too
    wide. The bake already recorded the true per-expert shapes, so use those.

    ``nf4.gate_up_blocks`` is ``[2*I, H/2]`` and ``nf4.down_blocks`` is
    ``[H, I/2]``; the pair over-determines (I, H), so they are cross-checked
    rather than trusted individually.
    """
    segs = index.get("segments") or []
    suffixes = [s["suffix"] for s in segs]
    # Detect the MXFP4 arena POSITIVELY. "no nf4.* suffixes" is not evidence of one --
    # an empty or malformed segment list satisfies it too, and routing that to the MXFP4
    # reader replaces this function's own precise error ("arena lacks nf4.gate_up_blocks")
    # with a confusing one from two layers down ("expected 4 fused or 6 split, got 0").
    # A real MXFP4 arena has exactly 4 (fused) or 6 (split w1/w3) projection segments.
    if len(segs) in (4, 6) and not any(x.startswith("nf4.") for x in suffixes):
        # A NATIVE MXFP4 arena (relocation bake) rather than a quantize-bake. Its
        # segments carry the checkpoint's own projection names, so the NF4 suffixes
        # are simply absent; the geometry is still recoverable, just from the packed
        # shapes. Defer to the engine's own reader so there is one definition of how
        # an MXFP4 arena is measured rather than a second, drifting copy here.
        from mxfp4_residency import engine_segment_map, fuse_gate_up_segments
        _groups, geo = engine_segment_map(fuse_gate_up_segments(index))
        _E, n1, half1, _nb1, n2, half2, _nb2 = geo
        intermediate, hidden = n1 // 2, half1 * 2
        if n1 % 2 or hidden != n2 or intermediate != half2 * 2:
            raise ValueError(
                f"mxfp4 arena geometry is inconsistent: gate_up {(n1, half1)} "
                f"implies (I={intermediate}, H={hidden}) but down {(n2, half2)} "
                f"implies (I={half2 * 2}, H={n2})")
        return intermediate, hidden

    def seg(suffix):
        g = next((s for s in index["segments"] if s["suffix"] == suffix), None)
        if g is None:
            raise KeyError(f"arena lacks {suffix!r}; segments: "
                           f"{[s['suffix'] for s in index['segments']]}")
        return tuple(g["shape_per_expert"])

    two_i, half_h = seg("nf4.gate_up_blocks")
    h_rows, half_i = seg("nf4.down_blocks")
    intermediate, hidden = two_i // 2, half_h * 2
    if two_i % 2 or intermediate != half_i * 2 or hidden != h_rows:
        raise ValueError(
            f"arena expert geometry is inconsistent: gate_up {(two_i, half_h)} "
            f"implies (I={intermediate}, H={hidden}) but down {(h_rows, half_i)} "
            f"implies (I={half_i * 2}, H={h_rows})")
    return intermediate, hidden


def build_meta_experts(index: dict, num_experts: int, *, has_gate: bool = True,
                       activation=None, compute_dtype=None, quant_type: str = "nf4",
                       cls=None):
    """An expert module carrying SHAPES ONLY — no expert storage anywhere.

    Built on ``meta``, so the ``[E, ...]`` packed buffers are never allocated.
    Both partitions are later served from the arena by :class:`_NvmeResidency`,
    which is what makes expert storage independent of model size: K3's 1.446 TB
    of experts would otherwise have to exist somewhere just to be indexed.

    Pair with :func:`enable_nvme_residency` on the same arena.
    """
    from .. import Experts4bit, ExpertsNbit
    intermediate, hidden = expert_geometry_from_arena(index)
    # `cls` lets a caller keep an architecture-specific EPILOGUE on the arena path.
    # Without it the arena branch would build a plain-SwiGLU `Experts4bit` even for a
    # model whose experts clamp (DeepSeek-V4) or use gpt-oss's GLU — the storage would
    # be right and the arithmetic quietly wrong, which is the whole failure class the
    # resident path already guards against.
    if cls is None:
        cls = Experts4bit if quant_type in ("nf4", "fp4") else ExpertsNbit
    mod = cls(num_experts=num_experts, hidden_dim=hidden,
              intermediate_dim=intermediate, has_gate=has_gate,
              activation=activation or torch.nn.functional.silu,
              compute_dtype=compute_dtype or torch.bfloat16,
              quant_type=quant_type, device="meta")
    _redeclare_for_mxfp4_arena(mod, index)
    return mod


def _redeclare_for_mxfp4_arena(mod, index) -> bool:
    """Re-declare a meta expert's buffers to an MXFP4 arena's geometry.

    An NF4 module declares `absmax` as fp32 per 64 elements; an MXFP4 arena
    carries uint8 e8m0 scales per 32. Same blocks width, different scales
    entirely — so the tier's geometry check (rightly) refuses to stage one into
    the other, and a natively-MXFP4 checkpoint had no way into arena TRAINING
    without being re-quantized to NF4 first.

    Nothing is allocated: the base is on `meta`, so this only changes DECLARED
    dtype and per-expert width, which is exactly what the check compares and
    what `segment_into` sizes its copies from.

    Shapes are taken from the arena's own fused segments rather than recomputed,
    for the same reason the segment map is positional: the suffixes are
    checkpoint-dependent, the ORDER is guaranteed.
    """
    from .nvme_train import arena_offload_view, OFFLOAD_SEGMENTS
    try:
        view, segmap = arena_offload_view(index)
    except Exception:
        return False                      # not an arena this tier can stage
    if segmap is OFFLOAD_SEGMENTS:
        return False                      # NF4: leave the module exactly as built

    geo = {g["suffix"]: g for g in view["segments"]}
    for name, suffix in segmap.items():
        g = geo[suffix]
        shape = tuple(g["shape_per_expert"])
        per_expert = 1
        for d in shape:
            per_expert *= d
        dt = _ST_TO_TORCH_DTYPE(g["dtype"])
        t = torch.empty(mod.num_experts, per_expert, dtype=dt, device="meta")
        # These four are a mix of parameters and buffers on the primitive, and a
        # module rejects a bare Tensor assigned over a Parameter. Re-declare in
        # kind: frozen storage either way (requires_grad=False), since the base
        # is never trained -- only the adapter is.
        if name in mod._parameters:
            mod._parameters[name] = torch.nn.Parameter(t, requires_grad=False)
        else:
            mod._buffers[name] = t
    # Staging is only half of it. Bytes landing correctly does not make the NF4
    # arithmetic interpret them, so the COMPUTE half is wired here too, in the
    # two places that read the packed buffers:
    #
    #   * `_dequantize_expert` — the per-expert unit every reference forward and
    #     `ExpertsLoRA._base_project` funnels through. Overriding it makes the
    #     module's OWN forward correct, whichever one it is, so the arch's
    #     epilogue (V4's clamped SwiGLU) is used by construction rather than
    #     re-derived here.
    #   * `forward` — routed to the grouped MXFP4 kernel when that is both
    #     available and legal, and to the module's own forward otherwise.
    mod._e4b_mxfp4_arena = True
    mod._dequantize_expert = types.MethodType(_mxfp4_dequantize_expert, mod)
    # Capture the PRISTINE forward once. A second `build_meta_experts` over the
    # same module would otherwise save our own router as the reference and make
    # the reference lane recurse.
    if not hasattr(mod, "_e4b_mxfp4_arena_ref"):
        mod._e4b_mxfp4_arena_ref = mod.forward
    mod.forward = types.MethodType(mxfp4_experts_forward, mod)
    return True


def _ST_TO_TORCH_DTYPE(name: str):
    import torch as _t
    return {"U8": _t.uint8, "I8": _t.int8, "F8_E8M0": _t.uint8,
            "F32": _t.float32, "BF16": _t.bfloat16}[name]


def enable_nvme_residency(model, arena_path: str, hot_sets: Sequence,
                          *, hot_rows: int, device: str = "cuda",
                          pinned: bool = True, qd: int = 4,
                          layers: Sequence[int] | None = None,
                          verbose: bool = False) -> int:
    """Serve each MoE layer's cold experts from ``arena_path``.

    Args:
        arena_path: arena baked by ``grouped-nf4-gemm``'s ``nvme_bake_nf4``
            (NF4 serving path) — its manifest records which provenance claim it
            supports.
        hot_sets: as :func:`enable_hot_residency` — one entry per MoE module.
        hot_rows: expert rows kept in pinned DRAM. Size from MEASURED free RAM
            (``nvme_residency.capacity_for_bytes``), never a declared figure.

            **Hard floor:** ``hot_rows`` must be >= the number of UNIQUE cold
            experts a single forward routes, because ``_cold_contrib`` fetches
            them in one ``index_select`` and every slot in that request is
            protected from eviction. For decode (one token, top-k) that is at
            most ``k``; for a prefill batch of ``T`` tokens it approaches
            ``min(T*k, num_experts - len(hot))``. Undersizing raises rather than
            thrashing, which is the right failure — but it means large-batch
            prefill needs either a big hot set or chunked prefill.
        layers: arena layer index per module, defaulting to ``range(len(hot_sets))``.
            Pass explicitly when the model's MoE modules are not arena layers
            0..N-1 (dense layers interleaved, or a partial bake).

    Returns the number of modules patched.
    Use it for serving when the NF4 experts do not fit host RAM: the cold experts live in an
    arena on NVMe, ``hot_rows`` expert rows stay pinned in DRAM. Expects an arena from
    grouped-nf4-gemm's ``nvme_bake_nf4`` and a model whose experts are plain (not
    adapter-wrapped) modules. Returns the number of layers attached (assert > 0). Needs the
    ``[fast]`` extra, a CUDA device, local NVMe. See
    ``docs/solutions/offload-moe-experts-to-cpu-or-nvme.md``.
    """
    try:
        from nvme_residency import ColdTier
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "NVMe residency needs grouped-nf4-gemm's N-series modules "
            "(nvme_residency/nvme_reader/nvme_arena) on the import path"
        ) from exc

    # VALIDATE BEFORE ALLOCATING. Everything below needs only `model` and the
    # caller's lists, so none of it has any business running after a ColdTier
    # exists — and running it after had two costs, one of them invisible until
    # this test finally executed on a runner:
    #
    #   * The refusals were UNREACHABLE without an accelerator. `ColdTier`
    #     pins its landing buffer, so on a CPU-only host it raised "Cannot
    #     access accelerator device when none is available" first, and a caller
    #     who passed too many hot_sets got an allocator error instead of the
    #     message naming their mistake.
    #   * On a host that DOES pin, a refusal leaked the tier: constructed, never
    #     closed. Same bug Bugbot caught in `nvme_train`'s attach loop.
    #
    # Use the SHARED selection, not an independent walk: a plain `isinstance` sweep
    # yields a different list AND a different order whenever wrapped and bare modules
    # are interleaved — silently stamping real MoE layers with the wrong arena layer,
    # or leaving them unstamped. Everything keying off hot_sets[i] must agree on what
    # i means. `target_modules` includes ExpertsLoRA bases; `enable_hot_residency`
    # below skips them (they consume their entry, so the indices stamped here stay
    # aligned) and raises outright if every module is wrapped.
    from .hot_residency import target_modules
    mods = target_modules(model)
    lay = list(layers) if layers is not None else list(range(len(hot_sets)))
    if len(lay) < len(hot_sets):
        raise ValueError(f"layers has {len(lay)} entries for {len(hot_sets)} hot_sets")
    if len(mods) < len(hot_sets):
        raise ValueError(
            f"hot_sets has {len(hot_sets)} entries but the model exposes "
            f"{len(mods)} targetable MoE module(s) — refusing to stamp a partial "
            f"set, which would serve some layers from the wrong arena rows")

    tier = ColdTier(arena_path, hot_rows=hot_rows, pinned=pinned, qd=qd)
    idx = tier.reader.index
    log(f"  nvme residency: arena {arena_path} rows={idx['n_layers']}x"
        f"{idx['n_experts_per_layer']} row_stride={idx['row_stride']} "
        f"hot_rows={hot_rows} ({hot_rows * idx['row_stride'] / 1e9:.1f} GB pinned)")

    # Which modules were ALREADY patched before this call. `_e4b_hot_ref` is
    # sticky — it survives an earlier `enable_hot_residency` or NVMe enable — so
    # reading it bare answers "has this module ever been patched", not "does this
    # module hold the tier THIS call just built". Getting that wrong leaks the new
    # pinned arena: a stale marker reads as a live holder, the close is skipped,
    # and nothing is actually serving from it. Snapshot first, and count only what
    # this call added. (Cursor Bugbot, #120: "Sticky hot-ref leaks new ColdTier" —
    # `nvme_train` avoids it with a call-local attach count, and this is the same
    # idea where the patching happens inside a callee.)
    targets = mods[:len(hot_sets)]
    already = {id(m) for m in targets if hasattr(m, "_e4b_hot_ref")}

    # Stamp each module with its arena layer BEFORE construction: _build_cold and
    # _build_hot both run inside _HotResidency.__init__ and need it.
    try:
        for i, m in enumerate(targets):
            m._e4b_cold_tier = tier
            m._e4b_arena_layer = lay[i]

        n = enable_hot_residency(model, hot_sets, device=device, verbose=verbose,
                                 state_cls=_NvmeResidency)
    except BaseException as exc:
        # `enable_hot_residency` patches modules ONE AT A TIME, so "we failed" and
        # "nothing is live" are different claims. A later-layer failure — OOM, a
        # bad layers index, an interrupt — leaves earlier modules already patched
        # and serving from this shared tier, and closing it under them hands them
        # a shut-down reader. Closing a live tier is strictly worse than leaking
        # one.
        #
        # (Cursor Bugbot, #120 — the SAME finding it made on `nvme_train`'s attach
        # loop earlier, which is why the policy below is identical: close only
        # when nothing is holding the tier, never convert control flow into an
        # error, and chain the real cause rather than replacing it.)
        live = [m for m in targets
                if hasattr(m, "_e4b_hot_ref") and id(m) not in already]
        if not live:
            tier.close()
            raise
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError(
            f"NVMe residency failed after {len(live)} of {len(hot_sets)} module(s) "
            "were already patched and are serving from this tier, so it is left "
            "OPEN rather than closed under them. The model is partially attached "
            "and must be discarded, not re-enabled; the tier is released with it. "
            "The failure that caused this is chained below."
        ) from exc
    log(f"  nvme residency active on {n} module(s)")
    return n


def enable_mxfp4_nvme_residency(model, arena_path: str, *, k_slots: int,
                                hot_rows: int, limit=None, device: str = "cuda",
                                qd: int = 4, layers: Sequence[int] | None = None,
                                engine_cls=None, hot_sets: Sequence | None = None) -> int:
    """Serve each MoE layer's experts from a **native MXFP4** arena.

    The counterpart to :func:`enable_nvme_residency`, which serves an NF4 arena.
    The difference is provenance, not just format: an NF4 arena is baked by
    re-quantizing, while ``nvme_arena.bake_expert_tensors`` relocates the released
    bytes verbatim — so this path computes on the checkpoint's own expert weights.
    It is also smaller (DeepSeek-V4-Flash: ~140 GiB against ~156 GB) and bakes far
    faster, since nothing is quantized.

    Unlike the NF4 lane, ``grouped-nf4-gemm``'s MXFP4 engines are standalone
    per-layer objects rather than ``nn.Module`` patches, so this binds one engine
    per MoE module. ``tier`` and ``store`` are shared across every layer — one
    pinned host buffer and one set of device slots, not one per layer, which is
    the difference between a few hundred MB and tens of GB at real depth.

    Args:
        k_slots: the model's routed top-k; it sizes the device slot store.
        hot_sets: per-module expert ids to keep RESIDENT in VRAM, never read again.
            This is the cost/speed dial: ``None`` (the default) is pure streaming —
            minimum VRAM, every routed expert fetched — while a hot set of H experts
            costs ``H * row_stride`` of VRAM per layer and removes those experts from
            the read path. It is a Sequence so a caller can size H per layer from a
            routing histogram rather than uniformly.
        hot_rows: pinned-DRAM rows the shared tier may hold. Same hard floor as
            :func:`enable_nvme_residency` — at least the number of DISTINCT cold
            experts one fetch can want.
        limit: the clamped-GLU bound. Defaults to each module's own ``.limit``
            (DeepSeek-V4 carries ``swiglu_limit`` there), so a mixed model cannot
            silently get one layer's bound applied to another's.
        engine_cls: the engine, which owns the EPILOGUE. Defaults to
            ``Mxfp4NvmeResidencyV4``. Passing the wrong one loads correct bytes
            and computes a different activation, so it is explicit rather than
            inferred.

    Returns the number of modules bound.
    Use it when the experts are native MXFP4 (gpt-oss, DeepSeek-V4) and must be served from
    the checkpoint's own bytes relocated into an arena (``nvme_arena.bake_expert_tensors``),
    never re-quantised. Returns the number of layers attached (assert > 0); pair with
    :func:`disable_mxfp4_nvme_residency`. Needs ``[fast]``, a CUDA device, local NVMe. See
    ``docs/solutions/mxfp4-moe-training-and-residency.md``.
    """
    try:
        from nvme_arena import load_index
        from mxfp4_residency import Mxfp4NvmeResidencyV4
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "MXFP4 NVMe residency needs grouped-nf4-gemm's mxfp4_residency / "
            "nvme_arena on the import path") from exc
    from .hot_residency import target_modules

    engine_cls = engine_cls or Mxfp4NvmeResidencyV4
    index = load_index(arena_path)
    mods = target_modules(model)
    if not mods:
        raise RuntimeError(
            "no targetable MoE modules found — an ExpertsLoRA-wrapped base is "
            "never dispatched, so there would be nothing for the engine to serve")
    lay = list(layers) if layers is not None else list(range(len(mods)))
    if len(lay) < len(mods):
        raise ValueError(f"layers has {len(lay)} entries for {len(mods)} modules")

    log(f"  mxfp4 nvme residency: arena {arena_path} rows={index['n_layers']}x"
        f"{index['n_experts_per_layer']} row_stride={index['row_stride']} "
        f"k_slots={k_slots} hot_rows={hot_rows}")

    # An ExpertsLoRA is a TRAINING target, not a frozen stack: binding the engine over it
    # would replace the adapter's forward outright, silently discarding the delta. And under
    # the arena loader its base buffers are on `meta`, so the adapter could not run anyway.
    # Refuse rather than pick one of those two wrong answers.
    # Test membership, not type: `target_modules` returns `ExpertsNbit` instances, and
    # `ExpertsLoRA` is NOT an `ExpertsNbit` subclass — so `isinstance(m, ExpertsLoRA)`
    # over this list was never true and this refusal could not fire. It went unnoticed
    # while `target_modules` filtered wrapped bases out on its own; now that it returns
    # them (the pipelined engine reaches them through the wrapper, this one does not),
    # the check has to actually work or a wrapped base gets a dead patch and the engine
    # reports success while serving nothing.
    from .hot_residency import wrapped_bases
    _wrapped_ids = wrapped_bases(model)
    wrapped = [i for i, m in enumerate(mods) if id(m) in _wrapped_ids]
    if wrapped:
        raise RuntimeError(
            f"{len(wrapped)} of {len(mods)} expert modules are ExpertsLoRA-wrapped "
            f"(first at index {wrapped[0]}). The NVMe engine serves FROZEN experts and "
            "would bypass the adapter entirely. Load without `arena=` to train, or drop "
            "the adapters to serve.")

    # Exactly one entry per MoE module, like `enable_hot_residency`. Without this a short
    # list raises IndexError somewhere down the loop and a LONG one is worse than an error:
    # trailing layers silently keep `()` while every earlier layer takes its neighbour's
    # set. That is precisely the failure `hot_sets_from_profile` cannot survive — an
    # informed hot set applied to the wrong layer is a uniform random draw, which the
    # +37.1% measurement showed is worth nothing (`docs/RESIDENCY-ENGINES.md`). It would
    # cost VRAM and read as "informed hot sets don't help here".
    if hot_sets is not None and len(hot_sets) != len(mods):
        raise ValueError(
            f"hot_sets has {len(hot_sets)} entries but the model has {len(mods)} expert "
            f"modules — exactly one entry per MoE layer in module order is required, so "
            f"alignment never silently shifts. Pass `()` for layers that should hold "
            f"nothing resident.")

    # The four other engines all replace `mod.forward`, and each already refuses the other
    # three; the NVMe engine was added without joining that set in either direction. Patching
    # over a live engine would strand its state and make `disable_*` restore a PATCHED
    # forward, leaving the module permanently wrapped.
    for ref, name in (("_e4b_fast_ref", "[fast]"), ("_e4b_hot_ref", "hot residency"),
                      ("_e4b_pipe_ref", "pipelined residency"),
                      ("_e4b_cold_ref", "cold engine")):
        busy = [i for i, m in enumerate(mods) if hasattr(m, ref)]
        if busy:
            raise RuntimeError(
                f"{len(busy)} of {len(mods)} expert modules already have {name} enabled "
                f"(first at index {busy[0]}). The engines are mutually exclusive — "
                f"disable it before enabling mxfp4 NVMe residency.")

    engines = []
    for i, mod in enumerate(mods):
        lim = limit if limit is not None else getattr(mod, "limit", None)
        if lim is None:
            raise ValueError(
                f"module {i} carries no `.limit` and none was passed — the "
                "clamped-GLU bound is not guessable, and a wrong one is silent")
        hot = () if hot_sets is None else hot_sets[i]
        eng = engine_cls(arena_path, lay[i], k_slots=k_slots, hot_rows=hot_rows,
                         hot_ids=hot, limit=float(lim), device=device, index=index, qd=qd,
                         tier=engines[0].tier if engines else None,
                         store=engines[0].store if engines else None)
        engines.append(eng)

        mod._e4b_mxfp4_engine = eng
        # Re-enabling (new hot_sets, reloaded checkpoint) must keep the PRISTINE forward:
        # capturing `mod.forward` a second time would save the previous `_fwd`, and then
        # `disable_mxfp4_nvme_residency` would "restore" a closure over the OLD engine —
        # the module stays patched forever, one disable short of every enable.
        if not hasattr(mod, "_e4b_mxfp4_ref"):
            mod._e4b_mxfp4_ref = mod.forward

        def _fwd(hidden, top_k_index, top_k_weights, _m=mod, _e=eng):
            # Same guard the v0 residency patch uses, and deliberately NOT
            # `_m.training`: a model left in train mode still runs plenty of
            # no-grad eval forwards, and gating on the mode sends those to a
            # reference path whose buffers are on `meta` under the arena loader —
            # i.e. NaNs, not a slow answer. The real condition is whether autograd
            # actually needs a graph.
            cd = _m.compute_dtype if _m.compute_dtype is not None else hidden.dtype
            if cd not in (torch.bfloat16, torch.float16):
                return _m._e4b_mxfp4_ref(hidden, top_k_index, top_k_weights)
            if torch.is_grad_enabled() and (
                hidden.requires_grad or any(p.requires_grad for p in _m.parameters())
            ):
                return _m._e4b_mxfp4_ref(hidden, top_k_index, top_k_weights)
            return _e.forward(hidden, top_k_index, top_k_weights)

        mod.forward = _fwd

    log(f"  mxfp4 nvme residency active on {len(engines)} module(s)")
    return len(engines)


def disable_mxfp4_nvme_residency(model) -> int:
    """Restore the saved forwards; the engines and their shared tier are dropped."""
    n = 0
    for mod in model.modules():
        if hasattr(mod, "_e4b_mxfp4_ref"):
            mod.forward = mod._e4b_mxfp4_ref
            del mod._e4b_mxfp4_ref, mod._e4b_mxfp4_engine
            n += 1
    return n


def _decode_mxfp4_rows(blocks, scales, rows: int, k: int, expert, dtype):
    """One expert's projection, decoded from the buffers PASSED IN.

    Takes the tensors rather than reading them off the module, because that is
    the contract the recompute backward relies on: ``_project`` closes over the
    buffers it saw in FORWARD, and under arena staging the module's attribute may
    already point at a different tensor by the time backward re-dequantizes. The
    NF4 ``_dequantize_expert`` has always worked this way; this matches it.

    Returns ``[k, rows]`` — the oracle's own orientation (``dequantize_mxfp4``
    transposes the trailing pair), i.e. ``[in, out]``.
    """
    from ..formats.mxfp4 import dequantize_mxfp4

    groups = k // 32                       # one e8m0 scale per 32 values
    return dequantize_mxfp4(blocks[expert].view(rows, groups, 16),
                            scales[expert].view(rows, groups), dtype=dtype)


def mxfp4_expert_weight(mod, name: str, expert: int, *, dtype=None):
    """Decode ONE expert's projection from staged MXFP4 bytes.

    This is the unit the MXFP4 forward is built on, and it is separated so it can
    be pinned against the format's own oracle on CPU: `dequantize_mxfp4` is pure
    torch, so parity here is checkable without a GPU, a kernel, or a checkpoint.

    The staged buffers are flat ``[E, per_expert]`` because every consumer indexes
    by global expert id; MXFP4 wants ``[rows, G, B]`` with B=16 bytes per 32-value
    group. The reshape is derived from the module's own declared geometry rather
    than passed in, so a mismatch surfaces here instead of as silent nonsense.

    Returns ``[in, out]`` (``x @ W`` orientation), which is what
    ``dequantize_mxfp4`` produces and what ``GptOssExperts`` expects of its
    ``{gate_up,down}_proj``. The ``F.linear`` orientation ``[out, in]`` is the
    transpose — see :func:`_mxfp4_dequantize_expert`.
    """
    import torch as _t

    if not getattr(mod, "_e4b_mxfp4_arena", False):
        raise TypeError(
            f"{type(mod).__name__} is not an MXFP4-arena module; its bytes are "
            "NF4 and decoding them as MXFP4 would return nonsense")
    if name == "gate_up":
        blocks, scales = mod.gate_up_proj, mod.gate_up_absmax
        rows, k = mod._gate_up_shape
    elif name == "down":
        blocks, scales = mod.down_proj, mod.down_absmax
        rows, k = mod._down_shape
    else:
        raise ValueError(f"name must be 'gate_up' or 'down', got {name!r}")
    return _decode_mxfp4_rows(blocks, scales, rows, k, expert,
                              dtype or _t.bfloat16)


def _mxfp4_dequantize_expert(mod, packed, absmax, shape, expert_idx, dtype):
    """``ExpertsNbit._dequantize_expert`` for MXFP4 bytes — the REFERENCE path.

    Bound per instance by :func:`_redeclare_for_mxfp4_arena`, which is what makes
    the module's own forward correct without this file knowing which forward that
    is. Every reference lane funnels through here:
    ``ExpertsNbit.forward`` -> ``_project``, ``_DeepseekV4ForwardMixin.forward``
    -> ``_project``, and ``ExpertsLoRA._base_project`` -> ``base._project``. So
    the arch's epilogue (V4's clamped SwiGLU via ``_apply_gate``) is applied by
    the arch's own code, not re-derived here — the failure mode ``lora._epilogue``
    exists to prevent.

    ``_project`` wraps this in ``_FrozenLinearRecomputeBackward``, so the training
    path keeps its property that the dequantized expert is dropped after the
    forward matmul and recomputed in backward. Nothing about that changes: only
    what "dequantize" means does.

    Returns ``[out, in]``, the layout ``F.linear`` wants — the transpose of the
    oracle's output. The round trip (``dequantize_mxfp4`` makes its result
    contiguous, then ``.t()`` makes it a view again) is left as-is deliberately:
    the decode is pinned to the oracle bit-for-bit, and re-deriving the untransposed
    form here to save a copy would be a second implementation of the format.
    """
    rows, k = shape
    return _decode_mxfp4_rows(packed, absmax, rows, k, expert_idx, dtype).t()


def _mxfp4_grouped_available(dev) -> bool:
    """Whether the fused grouped MXFP4 kernel can run at all here.

    Import and device are separate questions and both are cheap to get wrong:
    ``grouped-nf4-gemm`` publishes ``mxfp4_grouped`` on every platform but its
    Triton dependency is Linux-only, so the import succeeds on a laptop and the
    launch does not.
    """
    if dev.type != "cuda":
        return False
    try:
        import mxfp4_grouped  # noqa: F401
    except Exception:
        return False
    return True


def mxfp4_experts_forward(mod, hidden_states, top_k_index, top_k_weights):
    """Forward for an MXFP4-arena expert module.

    Two lanes, and the router between them is a correctness gate rather than a
    speed one:

    * **fused** — ``mxfp4_grouped.gemm_mxfp4_grouped``, one launch per projection
      over all routed tokens. Requires CUDA, bf16 compute (the kernel returns
      bf16 unconditionally, so an fp16 module would get a silent dtype swap), and
      **no autograd graph**: the kernel is raw Triton with no ``autograd.Function``
      behind it, so it produces no ``dL/dx`` and a training step routed here would
      simply stop learning below this layer. That is the same condition
      :func:`enable_mxfp4_nvme_residency`'s patch tests, and for the same reason —
      not ``mod.training``, because a model left in train mode still runs plenty
      of no-grad eval forwards.
    * **reference** — the module's own pristine forward, whose per-expert
      dequantize is :func:`_mxfp4_dequantize_expert`. This is the lane that
      carries gradients, and it is what the fused lane is graded against.

    The epilogue is the base's own in both lanes: the reference lane reaches it by
    being the base's forward, the fused lane through ``lora._epilogue`` — the same
    hook, so the two cannot drift onto different activations.
    """
    ref = mod._e4b_mxfp4_arena_ref
    cd = mod.compute_dtype if mod.compute_dtype is not None else hidden_states.dtype
    if cd is not torch.bfloat16:
        return ref(hidden_states, top_k_index, top_k_weights)
    if torch.is_grad_enabled() and (
        hidden_states.requires_grad or any(p.requires_grad for p in mod.parameters())
    ):
        return ref(hidden_states, top_k_index, top_k_weights)
    if not _mxfp4_grouped_available(hidden_states.device):
        return ref(hidden_states, top_k_index, top_k_weights)
    return _mxfp4_fused_forward(mod, hidden_states, top_k_index, top_k_weights)


def _mxfp4_fused_forward(mod, hidden_states, top_k_index, top_k_weights):
    """The grouped-kernel lane. Group-sorted layout, exactly as ``fast.py`` builds it.

    **Routing weights are applied AFTER the down projection**, which is the
    vendored ``ExpertsNbit`` contract. ``_DeepseekV4ForwardMixin.forward`` applies
    them before, in fp32, and the two agree in exact arithmetic — the down
    projection is linear and these modules carry no down bias, so scaling by a
    positive scalar commutes with it. What does NOT commute is the activation, and
    that is taken from the base rather than assumed.
    """
    from mxfp4_grouped import gemm_mxfp4_grouped

    from ..lora import _epilogue

    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.bfloat16)
    tokens, hidden = x.shape
    E = mod.num_experts
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape

    k = top_k_index.shape[1]
    # unique (token, slot) landing + one fixed-order sum — never index_add_
    # with duplicate token rows (CUDA atomics order = nondeterministic bits;
    # same fix as hot_residency.forward, see the comment there)
    out = torch.zeros(tokens, k, hidden, dtype=torch.float32, device=x.device)
    flat = top_k_index.reshape(-1)
    counts = torch.bincount(flat, minlength=E)
    active = torch.nonzero(counts, as_tuple=False).view(-1)
    if active.numel() == 0:                       # nothing routed anywhere
        return out.sum(dim=1).to(input_dtype)

    order = torch.argsort(flat, stable=True)
    token_rows = order // k
    top_pos = order - token_rows * k
    sizes = counts[active].tolist()               # every entry > 0, by construction
    expert_ids = active.to(torch.int32).tolist()
    a_cat = x.index_select(0, token_rows).contiguous()

    # REINTERPRET, never convert. A DeepSeek-V4 arena labels its blocks `I8` and
    # its scales `F8_E8M0`, so the staged buffers can arrive as int8 even though
    # the bytes are exactly what the kernel wants; `.to(uint8)` of an e8m0 scale
    # would yield the VALUE, not the exponent byte. Same trap `formats.mxfp4`
    # documents, one layer up.
    def _u8(t):
        return t if t.dtype == torch.uint8 else t.view(torch.uint8)

    proj = gemm_mxfp4_grouped(
        a_cat,
        _u8(mod.gate_up_proj).view(E, n1, k1 // 2),
        _u8(mod.gate_up_absmax).view(E, n1, k1 // 32),
        sizes, expert_ids)
    # The base's OWN epilogue, resolved through the same hook the reference lane
    # and `ExpertsLoRA` use. A plain SwiGLU here would silently drop V4's clamps.
    h = _epilogue(mod, proj)
    down = gemm_mxfp4_grouped(
        h.to(torch.bfloat16).contiguous(),
        _u8(mod.down_proj).view(E, n2, k2 // 2),
        _u8(mod.down_absmax).view(E, n2, k2 // 32),
        sizes, expert_ids)

    weighted = down.float() * top_k_weights[token_rows, top_pos, None].float()
    out.index_put_((token_rows, top_pos), weighted)
    return out.sum(dim=1).to(input_dtype)
