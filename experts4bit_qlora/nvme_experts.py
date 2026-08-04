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

    from experts4bit_qlora.nvme_experts import enable_nvme_residency
    n = enable_nvme_residency(model, arena_path, hot_sets, hot_rows=20_000)
"""
from __future__ import annotations

from typing import Sequence

import torch

from .hot_residency import _HotResidency, enable_hot_residency
from .util import log

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

    def index_select(self, dim: int, idx: torch.Tensor) -> torch.Tensor:
        if dim != 0:
            raise ValueError(f"_TieredStack indexes experts on dim 0, got {dim}")
        from nvme_residency import segment_tensor
        globals_ = self.cold_ids.index_select(0, idx.cpu().to(torch.long))
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

        tier = self.mod._e4b_cold_tier
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
    from . import Experts4bit, ExpertsNbit
    intermediate, hidden = expert_geometry_from_arena(index)
    # `cls` lets a caller keep an architecture-specific EPILOGUE on the arena path.
    # Without it the arena branch would build a plain-SwiGLU `Experts4bit` even for a
    # model whose experts clamp (DeepSeek-V4) or use gpt-oss's GLU — the storage would
    # be right and the arithmetic quietly wrong, which is the whole failure class the
    # resident path already guards against.
    if cls is None:
        cls = Experts4bit if quant_type in ("nf4", "fp4") else ExpertsNbit
    return cls(num_experts=num_experts, hidden_dim=hidden,
               intermediate_dim=intermediate, has_gate=has_gate,
               activation=activation or torch.nn.functional.silu,
               compute_dtype=compute_dtype or torch.bfloat16,
               quant_type=quant_type, device="meta")


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
    """
    try:
        from nvme_residency import ColdTier
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "NVMe residency needs grouped-nf4-gemm's N-series modules "
            "(nvme_residency/nvme_reader/nvme_arena) on the import path"
        ) from exc

    tier = ColdTier(arena_path, hot_rows=hot_rows, pinned=pinned, qd=qd)
    idx = tier.reader.index
    log(f"  nvme residency: arena {arena_path} rows={idx['n_layers']}x"
        f"{idx['n_experts_per_layer']} row_stride={idx['row_stride']} "
        f"hot_rows={hot_rows} ({hot_rows * idx['row_stride'] / 1e9:.1f} GB pinned)")

    # Stamp each module with its arena layer BEFORE construction: _build_cold and
    # _build_hot both run inside _HotResidency.__init__ and need it.
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
    for i, m in enumerate(mods[:len(hot_sets)]):
        m._e4b_cold_tier = tier
        m._e4b_arena_layer = lay[i]

    n = enable_hot_residency(model, hot_sets, device=device, verbose=verbose,
                             state_cls=_NvmeResidency)
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
