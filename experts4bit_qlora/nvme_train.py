"""Training whose frozen experts live on NVMe — the arena as a *training* tier.

:mod:`.nvme_experts` made a baked arena serve inference: expert modules build on
``meta`` and the residency engine reads rows off the device. It refuses to bind
over an ``ExpertsLoRA``, and correctly — that engine replaces the module's
forward outright, which would discard the adapter's delta. So today the two
capabilities are exclusive, and the refusal says so in as many words: *"Load
without `arena=` to train, or drop the adapters to serve."*

That leaves training's ceiling where it has always been. ``offload._ExpertOffload``
streams one layer at a time to the GPU, which bounds **VRAM** beautifully and
does nothing at all for **host RAM**: its homes are the full ``[E, ...]`` packed
stacks, pinned, for every layer at once. Qwen3-30B-A3B in NF4 is ~16 GB of that
and fits anywhere; the models the tier exists for do not.

This module removes that floor without touching the adapter. The frozen base's
home stops being host RAM and becomes the arena:

    disk row  ->  ColdTier pinned slot  ->  [E, ...] device stack  ->  kernel

Nothing about the *training* path changes. ``enable_fast_train`` still patches
the ``ExpertsLoRA`` wrapper, the delta is still added pre-activation, and
:class:`~nf4_qlora.FusedGroupedNf4` still re-reads the staged weights in backward
through its ``weights_fn`` closure — which is precisely the seam that makes this
work: the closure reads *whatever is staged when backward runs*, so an
arena-staged layer is as valid to it as a DRAM-staged one.

**Gradient checkpointing is required**, and it already was. The evict post-hook
fires when a forward returns, so without a recompute to re-stage it, backward
reads 0-element placeholders — true of the host-RAM path too, just never fatal
enough to notice. Here it is guarded: see :meth:`_ArenaExpertOffload.assert_rows_staged`.

**What this does NOT bound: VRAM.** The staged destination keeps the full
``[E, ...]`` shape so every consumer keeps indexing by global expert id — the
kernel's ``expert_ids``, ``ExpertsLoRA``'s adapter stacks, the pool. So one
layer's whole expert stack is resident on the device even though only the routed
rows were read from disk. At 128 experts/layer that is a few hundred MB and it is
what the offload path already did; at K3's 896 experts/layer it is ~15.7 GB, and
fitting that on a small card needs a *compacted* stack (``[R, ...]`` plus an id
remap), which splits the kernel's id space from the adapter's. Deliberately not
done here: it changes the kernel contract, and this change does not.

Usage::

    from experts4bit_qlora.nvme_train import enable_nvme_train_residency
    n = enable_nvme_train_residency(model, arena_path, hot_rows=20_000)
    enable_fast_train(model)          # unchanged
"""
from __future__ import annotations

import os
from typing import Sequence

import torch

from .offload import _ExpertOffload, enable_expert_offload
from .util import log

#: Offload tensor name -> the arena segment carrying it. These are the four
#: tensors ``_ExpertOffload`` homes, and an NF4 quantize-at-bake arena
#: (``nvme_bake_nf4``) ships exactly these four segments per expert row.
OFFLOAD_SEGMENTS = {
    "gate_up_proj": "nf4.gate_up_blocks",
    "down_proj": "nf4.down_blocks",
    "gate_up_absmax": "nf4.gate_up_absmax",
    "down_absmax": "nf4.down_absmax",
}


def _poison_enabled() -> bool:
    """``E4B_ARENA_POISON=1``: fill each staged destination before writing the
    routed rows, so reading an *unrouted* row yields obvious nonsense instead of
    whatever the allocator last left there.

    Off by default because it is a full ``[E, ...]`` memset per stage — real cost
    on the path whose whole point is not touching every expert. Worth turning on
    for a bring-up run: it converts the one failure this design can still have
    (reading a row nothing staged) from plausible numbers into visible ones.
    """
    return os.environ.get("E4B_ARENA_POISON", "") == "1"


class _ArenaExpertOffload(_ExpertOffload):
    """An offload handle whose homes are an NVMe arena instead of pinned DRAM.

    Everything about *residency* is inherited: the single-GPU-resident-layer
    rule, the stage/evict hooks, the backward-recompute eviction subtlety. Only
    the answer to "where do this layer's bytes come from" changes.

    The homes are ``meta`` tensors — shape and dtype, no storage. That is not a
    trick to satisfy the parent class, it is the literal claim: for a model whose
    experts exceed host RAM there is nowhere for an ``[E, ...]`` home to live, and
    saying so with ``meta`` keeps every shape-reading consumer (``_alloc_dest``,
    ``evict``'s placeholder dtypes, the state_dict hook) working off the truth.
    """

    #: Routed staging must run on grad-enabled forwards here — the bulk path
    #: reads every expert row off the device. See ``offload._stage_pre_hook``.
    _routed_in_train = True

    @classmethod
    def _build_homes(cls, base, names, pin: bool, arena: bool):
        """Homes as ``meta`` tensors, shaped from the MODULE and dtyped from the
        ARENA — then cross-checked against each other.

        Both halves have to agree or the staging path writes real bytes into the
        wrong shape. The module knows ``[E, flat_numel]`` per tensor; the arena
        index knows each segment's per-expert shape and dtype. Their product must
        match exactly, and a mismatch means this arena was not baked from this
        model — which is silent otherwise, because the byte counts are close
        enough that the copies would still "work".
        """
        from nvme_residency import segment_geometry

        tier = base._e4b_cold_tier
        index = tier.reader.index
        home = {}
        for n in names:
            suffix = OFFLOAD_SEGMENTS.get(n)
            if suffix is None:
                raise KeyError(
                    f"no arena segment is defined for offload tensor {n!r}; "
                    f"known: {sorted(OFFLOAD_SEGMENTS)}")
            dt, shape, _off, _ln = segment_geometry(index, suffix)
            cur = getattr(base, n)
            per_expert = 1
            for s in shape:
                per_expert *= s
            if cur.dtype != dt:
                raise TypeError(
                    f"{n}: module holds {cur.dtype} but arena segment {suffix!r} "
                    f"is {dt} — this arena was not baked from this model")
            if cur.dim() != 2 or cur.shape[1] != per_expert:
                raise ValueError(
                    f"{n}: module storage is {tuple(cur.shape)} but arena segment "
                    f"{suffix!r} carries {shape} = {per_expert} values per expert. "
                    "Expected [num_experts, per_expert]; the arena does not match "
                    "this model's expert geometry.")
            home[n] = torch.empty(cur.shape, dtype=dt, device="meta")
        return home, None, None

    def __init__(self, base, device, pin: bool = True):
        # Set before super().__init__: it calls evict(), which clears these.
        self._tier = base._e4b_cold_tier
        self._arena_layer = int(base._e4b_arena_layer)
        self._staged_ids: frozenset = frozenset()
        super().__init__(base, device, pin=pin)
        # Routed staging is not an optimisation here, it is the design: a bulk
        # stage means reading the entire layer off the device. `_routed_max` is
        # the crossover above which a single bulk copy beats N strided ones —
        # true for a DRAM home, false for a disk one, where each expert row is
        # its own aligned read either way. So it never fires: routed is weakly
        # better at every width, and identical at full width.
        self._routed_only = True
        self._routed_max = int(base.num_experts)
        # The homes are meta, so the parent's `all(_is_pinned(...))` is False and
        # would misreport the transfer. The bytes actually cross the link from
        # the TIER's landing buffer, so that is what decides whether the H2D is
        # genuinely async.
        self.pinned = bool(self._tier.pinned)

    # ------------------------------------------------------------- staging --
    def _copy_rows_into(self, dest, ids) -> int:
        """Read rows ``ids`` off the arena straight into ``dest``.

        Replaces the parent's pinned-home row copy. The destination keeps its
        full ``[E, ...]`` shape and rows land at their global expert index, so
        every consumer downstream is unchanged.
        """
        from nvme_residency import segment_geometry, segment_into

        if getattr(self, "_pool", None) is not None:
            raise RuntimeError(
                "the device-side expert cache (enable_expert_cache) is not wired "
                "to the arena path: its rows are keyed by (layer, expert) but "
                "filled from a pinned host home this handle does not have. "
                "Disable the cache, or use a host-RAM offload handle.")

        ids = [int(e) for e in ids]
        index = self._tier.reader.index
        E = int(self.base.num_experts)
        nbytes = 0
        for n in self._param_names + self._buffer_names:
            suffix = OFFLOAD_SEGMENTS[n]
            _dt, shape, _off, ln = segment_geometry(index, suffix)
            if _poison_enabled():
                dest[n].fill_(0xA5 if dest[n].dtype == torch.uint8 else float("nan"))
            # dest[n] is [E, per_expert]; segment_into wants [E, *shape_per_expert].
            segment_into(self._tier, index, self._arena_layer, ids, suffix,
                         dest[n].view(E, *shape), rows=ids, non_blocking=True)
            nbytes += ln * len(ids)
        self._staged_ids = frozenset(ids)
        return nbytes

    def _copy_home_to_device(self, policy: str = "sync") -> None:
        """Bulk stage — every expert in the layer, read from the arena.

        Routed staging covers every real training forward, so this is the
        fallback for a call whose routing was not visible to the pre-hook. It is
        deliberately the SAME read path rather than a second one: a whole-layer
        stage is just a routed stage over ``range(E)``, which keeps
        ``_staged_ids`` truthful (and therefore the backward guard sound) with no
        second implementation to drift.
        """
        dest = self._alloc_dest()
        nbytes = self._copy_rows_into(dest, range(int(self.base.num_experts)))
        self._bind(dest)
        self._last_stage_policy = policy
        self._last_stage_nbytes = nbytes

    def evict(self) -> None:
        # A base built by `build_meta_experts` holds META parameters, and the
        # parent evicts by assigning `.data` — which raises "incompatible tensor
        # type" from meta to a real device. Replace the tensor OBJECT once, on
        # the first evict, so the parent's `.data` assignment (and every later
        # stage) has something real to write through. Shapes are already captured
        # in `self.home`, so nothing is lost by dropping them here: an evicted
        # tensor is a 0-element placeholder either way.
        b = self.base
        for n in self._param_names:
            p = b._parameters.get(n)
            if p is not None and p.is_meta:
                b._parameters[n] = torch.nn.Parameter(
                    torch.empty(0, dtype=p.dtype, device=self.device),
                    requires_grad=False)
        for n in self._buffer_names:
            t = b._buffers.get(n)
            if t is not None and t.is_meta:
                b._buffers[n] = torch.empty(0, dtype=t.dtype, device=self.device)
        super().evict()
        # Nothing is staged, so nothing is covered. The guard checks `staged`
        # first and would refuse anyway; clearing keeps the two from ever
        # disagreeing about what this handle is holding.
        self._staged_ids = frozenset()

    # --------------------------------------------------------------- guard --
    def assert_rows_staged(self, expert_ids) -> None:
        """Refuse to read expert rows this handle did not stage.

        Called from ``fast.py``'s ``weights_fn`` closures — that is, **in
        backward**, which is the only place this can go wrong. Routed staging
        fills the routed rows of a full-shaped destination and leaves the rest
        untouched, so a backward that asks for a row the *recompute* did not
        re-stage reads whatever the allocator left there: finite, plausible,
        wrong. Two ways to get there, both real:

        * **No gradient checkpointing.** The evict post-hook fires when the
          forward returns and there is no recompute to re-stage, so backward
          finds placeholders.
        * **A recompute that routed differently.** ``use_reentrant=False``
          restores RNG state by default, so a jitter-noise router reproduces its
          draw; ``preserve_rng_state=False`` does not, and the recompute can pick
          a different top-k.

        Cost is a set membership test per expert group, once per projection.
        """
        if not self.staged:
            raise RuntimeError(
                f"arena layer {self._arena_layer}: expert weights are not staged "
                "at the moment they are being read. Training through the NVMe "
                "tier requires gradient checkpointing — the checkpoint recompute "
                "is what re-stages a layer for its own backward. Enable it "
                "(model.gradient_checkpointing_enable()), or keep the experts in "
                "host RAM with the ordinary offload path.")
        missing = sorted({int(e) for e in expert_ids} - self._staged_ids)
        if missing:
            raise RuntimeError(
                f"arena layer {self._arena_layer}: experts {missing} are being "
                f"read but were not staged (staged {len(self._staged_ids)} rows). "
                "A gradient-checkpoint recompute routed differently than its "
                "forward did — pass preserve_rng_state=True (the default) so a "
                "noisy router reproduces its draw. Reading these rows would "
                "return uninitialized memory, not an error.")

    def tier_stats(self) -> dict:
        return self._tier.stats()


def _engine_conflicts(mods) -> None:
    """Refuse when another residency engine already owns these modules.

    The engines each replace ``forward``; stacking one over another strands the
    inner engine's state and makes ``disable_*`` restore a patched forward, so
    the module stays wrapped forever. Every other engine already refuses the
    rest — this joins that set rather than being the one that does not.
    """
    for ref, name in (("_e4b_hot_ref", "hot residency"),
                      ("_e4b_pipe_ref", "pipelined residency"),
                      ("_e4b_cold_ref", "cold engine"),
                      ("_e4b_mxfp4_ref", "mxfp4 NVMe residency")):
        busy = [i for i, m in enumerate(mods) if hasattr(m, ref)]
        if busy:
            raise RuntimeError(
                f"{len(busy)} of {len(mods)} expert modules already have {name} "
                f"enabled (first at index {busy[0]}). Those engines serve FROZEN "
                "experts and replace the module forward; arena-backed training "
                "keeps the adapter's forward and changes only where the frozen "
                "bytes come from. Disable the engine first.")


def enable_nvme_train_residency(model, arena_path: str, *, hot_rows: int,
                                device: str = "cuda", qd: int = 4,
                                pinned: bool = True,
                                layers: Sequence[int] | None = None,
                                verbose: bool = False) -> int:
    """Serve every ``ExpertsLoRA``'s frozen base from ``arena_path`` during training.

    Args:
        arena_path: an arena baked by ``grouped-nf4-gemm``'s ``nvme_bake_nf4``
            (the NF4 quantize-at-bake path — it ships the four segments this
            offloads). Its manifest records ``bake_mode``, which is what decides
            the provenance claim: bit-identical to the quantizer's output, not to
            a bf16 release.
        hot_rows: expert rows held in the tier's pinned DRAM. Size from MEASURED
            free RAM (``nvme_residency.capacity_for_bytes``), never a declared
            figure.

            **Hard floor:** at least the number of unique experts one forward
            routes, since a stage requests them together and every slot in that
            request is protected from eviction. For a training batch of ``T``
            tokens at top-``k`` that approaches ``min(T*k, num_experts)`` — much
            larger than decode's ``k``. Undersizing raises rather than thrashing.
        layers: arena layer index per MoE module, defaulting to ``0..N-1``. Pass
            explicitly when the model's MoE modules are not arena layers in order.

    Returns the number of modules moved onto the arena.

    Composes with ``enable_fast_train`` in either order, and requires gradient
    checkpointing — see :meth:`_ArenaExpertOffload.assert_rows_staged`.
    """
    # Named by SYMBOL, not by version number: the staging entry point ships with
    # this feature, so any version assertion here would be a forward reference to
    # a release that may not have been cut yet — and a wrong one is worse than
    # none, since it sends people to upgrade past a version that has it.
    try:
        from nvme_residency import ColdTier
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "arena-backed training needs grouped-nf4-gemm's N-series modules "
            "(nvme_residency/nvme_reader/nvme_arena) on the import path") from exc
    try:
        from nvme_residency import segment_into  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the installed grouped-nf4-gemm has no `nvme_residency.segment_into`; "
            "arena staging needs it to fill a caller-owned destination, and the "
            "serving-side `segment_tensor` cannot substitute (it allocates its "
            "own result). Upgrade grouped-nf4-gemm to a build that exports it."
        ) from exc

    from .hot_residency import target_modules
    from .lora import ExpertsLoRA

    mods = target_modules(model)
    if not mods:
        raise RuntimeError(
            "no MoE expert modules found under this model — nothing to move onto "
            "the arena")

    # The training target is the WRAPPER, not the base: its forward is what runs,
    # and the stage/evict hooks have to sit where the gradient-checkpoint
    # recompute will re-enter. Index off the shared `target_modules` list so
    # `layers[i]` means the same `i` here as everywhere else in the package.
    wrapper_of = {id(m.base): m for m in model.modules()
                  if isinstance(m, ExpertsLoRA) and hasattr(m, "base")}
    bare = [i for i, m in enumerate(mods) if id(m) not in wrapper_of]
    if bare:
        raise RuntimeError(
            f"{len(bare)} of {len(mods)} expert modules are not ExpertsLoRA-wrapped "
            f"(first at index {bare[0]}). This path exists to train an adapter over "
            "arena-resident frozen experts; with no adapter there is nothing to "
            "train, and `enable_nvme_residency` is the serving equivalent.")

    _engine_conflicts(mods)

    lay = list(layers) if layers is not None else list(range(len(mods)))
    if len(lay) < len(mods):
        raise ValueError(
            f"layers has {len(lay)} entries for {len(mods)} MoE modules — one per "
            "module in dispatch order is required, so a layer's rows are never "
            "silently read from its neighbour's arena row")

    tier = ColdTier(arena_path, hot_rows=hot_rows, pinned=pinned, qd=qd)
    idx = tier.reader.index
    log(f"  nvme TRAIN residency: arena {arena_path} rows={idx['n_layers']}x"
        f"{idx['n_experts_per_layer']} row_stride={idx['row_stride']} "
        f"hot_rows={hot_rows} ({hot_rows * idx['row_stride'] / 1e9:.1f} GB pinned)")

    n = 0
    for i, base in enumerate(mods):
        # Stamp BEFORE construction: _build_homes runs inside __init__ and reads
        # both of these off the base.
        base._e4b_cold_tier = tier
        base._e4b_arena_layer = lay[i]
        wrapper = wrapper_of[id(base)]
        handle = enable_expert_offload(wrapper, device, pin=False,
                                       handle_cls=_ArenaExpertOffload)
        if not isinstance(handle, _ArenaExpertOffload):
            raise RuntimeError(
                f"module {i} already carries a host-RAM offload handle "
                f"({type(handle).__name__}). enable_expert_offload is idempotent "
                "and returns the existing handle, so the arena one was never "
                "installed — enable arena residency before offloading, not after.")
        # fast.py's weights_fn closures call this at BACKWARD time.
        base._e4b_stage_guard = handle.assert_rows_staged
        base._e4b_arena_offload = handle
        n += 1

    log(f"  nvme TRAIN residency active on {n} module(s); "
        f"gradient checkpointing is REQUIRED (backward re-reads staged rows)")
    if verbose:
        print(f"[e4b.nvme_train] {n} module(s) on arena {arena_path}")
    return n


def arena_train_stats(model) -> dict | None:
    """Tier counters for an arena-backed training run, or ``None`` if not enabled.

    Reports the SHARED tier once (every module reads the same one) plus how many
    modules are attached — a per-module sum would multiply-count the same reads.
    """
    handles = [m._e4b_arena_offload for m in model.modules()
               if hasattr(m, "_e4b_arena_offload")]
    if not handles:
        return None
    s = dict(handles[0].tier_stats())
    s["modules"] = len(handles)
    s["last_stage_rows"] = [len(h._staged_ids) for h in handles]
    return s
