"""Optional CPU-offloading of the frozen 4-bit expert weights.

When enabled (``OFFLOAD_EXPERTS=1``), each :class:`Experts4bit` base keeps its four big tensors —
the packed 4-bit ``gate_up_proj`` / ``down_proj`` weights and their ``gate_up_absmax`` /
``down_absmax`` scales — in **pinned CPU RAM** instead of on the GPU. A forward *pre*-hook on the
enclosing :class:`~experts4bit_qlora.lora.ExpertsLoRA` copies that layer's experts onto the GPU just
before its experts forward, and a forward *post*-hook drops the GPU copy right after. Because
training uses gradient checkpointing (``use_reentrant=False``), each decoder layer's forward is
recomputed in the backward pass — the *same* pre-hook re-stages the experts for the recompute.
(PyTorch stops that recompute *early*, as soon as the saved tensors are regenerated, so the evict
post-hook does **not** fire on the recompute; eviction during backward is instead driven by a
single-resident-slot policy — staging a layer first evicts the previously-staged one.) So only
**one layer's** experts are GPU-resident at any instant, in forward and backward alike.

This lets a fused-MoE whose 4-bit experts exceed VRAM (Qwen3-30B-A3B ~15 GB, Gemma-4-26B-A4B
~13 GB) QLoRA-train on a 12 GB card, at the cost of one host->device expert transfer per layer per
pass (a memory-for-compute trade — see ``docs/METHODOLOGY.md`` §11).

Why this is correct (and why the hook goes on ``ExpertsLoRA``, not ``Experts4bit``):

* ``ExpertsLoRA.forward`` never calls ``base.forward()``; it reads ``base.gate_up_proj`` etc.
  directly and calls ``base._dequantize_expert(...)``. A pre-hook on the base would never fire in
  training, so the hook must sit on the module whose ``__call__`` actually runs — ``ExpertsLoRA``.
* ``ExpertsLoRA`` uses the **dequantize path**: ``w = base._dequantize_expert(...)`` then
  ``F.linear(x, w)``. The packed weight has ``requires_grad=False``, so the dequantized ``w`` is a
  non-grad constant and the dequant op is not on the autograd tape. ``F.linear`` saves the
  *dequantized* ``w`` (for ``grad_x``), never the packed weight. The packed weight is therefore only
  read during the forward (and recompute-forward) to *produce* ``w`` — never needed by backward — so
  evicting it in the post-hook is safe in both the initial forward and the checkpoint recompute.

  **Invariant:** this safety holds only while ``ExpertsLoRA`` uses ``_dequantize_expert`` + ``F.linear``.
  Do **not** route ``ExpertsLoRA`` through ``bnb.matmul_4bit`` (whose autograd ``Function``
  re-dequantizes in backward and would need the evicted packed weight) while offloading.

The tiny NF4 ``code`` buffer and the trainable LoRA adapters stay GPU-resident throughout.
"""

from __future__ import annotations

import torch

# 0-element GPU placeholders that an evicted base's parameters/buffers point at, shared across all
# offloaded layers (reads never mutate them, so sharing is safe) and cached per device. Keeping the
# real "home" data OFF the module — only these placeholders are registered while evicted — means
# ``model.to(device)`` / ``state_dict()`` / ``save`` never drag the big expert tensors around.
_PLACEHOLDERS: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _placeholders(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ph = _PLACEHOLDERS.get(device)
    if ph is None:
        ph = (
            torch.empty(0, dtype=torch.uint8, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )
        _PLACEHOLDERS[device] = ph
    return ph


def _is_pinned(t: torch.Tensor) -> bool:
    """Whether ``t`` is in pinned memory (so ``non_blocking`` H2D is truly async). Robust on hosts
    where ``is_pinned`` is unavailable/raises without CUDA."""
    try:
        return bool(t.is_pinned())
    except Exception:
        return False


class _ExpertOffload:
    """Owns the pinned-CPU home copies of one :class:`Experts4bit` base's four big tensors and
    streams them to ``device`` for the duration of each forward / gradient-checkpoint recompute.

    The base's own ``gate_up_proj`` / ``down_proj`` parameters and ``gate_up_absmax`` /
    ``down_absmax`` buffers hold shared 0-element GPU placeholders while evicted, so nothing that
    walks the module tree (``.to()``, ``state_dict()``, checkpoint-save) touches the offloaded data.
    """

    _NAMES_PARAM = ("gate_up_proj", "down_proj")  # nn.Parameter (uint8, packed 4-bit, frozen)
    _NAMES_BUFFER = ("gate_up_absmax", "down_absmax")  # float32 quantization scales (buffers)

    # The single handle whose experts are currently GPU-staged. Under use_reentrant=False gradient
    # checkpointing the backward RECOMPUTE re-runs the layer forward but PyTorch stops it early (once
    # the saved tensors are regenerated), so the evict *post*-hook does NOT fire on the recompute.
    # Relying on the post-hook alone would leave every recomputed layer's experts staged through the
    # whole backward — accumulating to the full footprint offload exists to avoid. So staging a new
    # layer first evicts this previously-staged one: at most one layer is GPU-resident at any instant,
    # in forward AND backward, regardless of whether the post-hook fired.
    _resident = None

    def __init__(self, base, device, pin: bool = True):
        self.base = base
        self.device = torch.device(device)
        # Capture the homes as SEPARATE (pinned) CPU tensors BEFORE any placeholder swap. Capturing
        # the live Parameter object instead would alias the very tensor we later overwrite with a
        # placeholder — losing the weights. The source is on the GPU at load time, so ``.to("cpu")``
        # is a real device->host copy that decouples the home from the module.
        self.home = {n: self._to_home(getattr(base, n).detach(), pin) for n in self._names()}
        # True iff every home landed in pinned memory (so staging's non_blocking H2D is real); False
        # when pin=False or pin_memory fell back to pageable. Surfaced by the loader's summary log.
        self.pinned = all(_is_pinned(t) for t in self.home.values())
        self.staged = False
        self.evict()  # start evicted: base holds placeholders, ~0 GPU footprint (frees the load copy)

    @classmethod
    def _names(cls):
        return cls._NAMES_PARAM + cls._NAMES_BUFFER

    @staticmethod
    def _to_home(t: torch.Tensor, pin: bool) -> torch.Tensor:
        cpu = t.to("cpu")
        if pin:
            try:
                return cpu.pin_memory()
            except (RuntimeError, AssertionError):
                pass  # pinning is best-effort; pageable fallback is correct, just no async H2D
        return cpu

    def stage(self) -> None:
        """Copy the four big tensors onto ``device`` (idempotent), first evicting the previously
        staged layer so at most one layer's experts are GPU-resident (holds through the backward
        recompute, where the evict post-hook does not fire). The H2D copy is enqueued on the current
        stream, so the immediately-following dequant kernels are correctly ordered after it."""
        if self.staged:
            return
        cls = type(self)
        if cls._resident is not None and cls._resident is not self:
            cls._resident.evict()  # single-slot: free the prior layer before staging this one
        b = self.base
        for n in self._NAMES_PARAM:
            b._parameters[n].data = self.home[n].to(self.device, non_blocking=True)
        for n in self._NAMES_BUFFER:
            b._buffers[n] = self.home[n].to(self.device, non_blocking=True)
        self.staged = True
        cls._resident = self

    def evict(self) -> None:
        """Point the four big tensors back at shared 0-element placeholders (idempotent), dropping
        the GPU copies so the caching allocator can reuse the memory for the next layer."""
        ph_u8, ph_f32 = _placeholders(self.device)
        b = self.base
        for n in self._NAMES_PARAM:
            b._parameters[n].data = ph_u8
        for n in self._NAMES_BUFFER:
            b._buffers[n] = ph_f32
        self.staged = False
        cls = type(self)
        if cls._resident is self:
            cls._resident = None


def enable_expert_offload(experts_lora, device, pin: bool = True) -> _ExpertOffload:
    """Offload one :class:`ExpertsLoRA`'s frozen 4-bit base to (pinned) CPU RAM and install the
    stream-in/evict hooks.

    Registers a forward pre-hook (stage the base onto ``device``) and a forward post-hook (evict it)
    on ``experts_lora`` — the module whose ``__call__`` runs on every forward *and* on the
    gradient-checkpoint recompute — and stashes the handle on ``experts_lora._offload`` so it stays
    alive with the module. Returns the handle.
    """
    base = getattr(experts_lora, "base", None)
    if base is None or not all(hasattr(base, n) for n in _ExpertOffload._names()):
        raise TypeError(
            "enable_expert_offload expects an ExpertsLoRA wrapping an Experts4bit base "
            f"(gate_up_proj/down_proj/gate_up_absmax/down_absmax); got {type(experts_lora).__name__}"
        )
    handle = _ExpertOffload(base, device, pin=pin)
    experts_lora._offload = handle
    experts_lora.register_forward_pre_hook(lambda module, args: handle.stage())
    experts_lora.register_forward_hook(lambda module, args, output: handle.evict())
    return handle


def offload_model_experts(model, device=None, pin: bool = True) -> list[_ExpertOffload]:
    """Offload every :class:`ExpertsLoRA` in ``model`` to (pinned) CPU RAM.

    Convenience for the already-loaded / test path. The streaming loader does **not** use this — it
    offloads each layer inside its per-layer loop so the experts never all sit on the GPU at once
    (a post-load pass would require every layer GPU-resident first, defeating the purpose). ``device``
    defaults to the device of the first offloadable base found.
    """
    from .lora import ExpertsLoRA

    handles = []
    for module in model.modules():
        if isinstance(module, ExpertsLoRA):
            dev = device if device is not None else module.base.gate_up_proj.device
            handles.append(enable_expert_offload(module, dev, pin=pin))
    return handles
