"""The QLoRA training recipe as one typed setup, its memory cost before loading, and the call that builds it.

Three things that used to live apart, which is how an estimate and a run drift:

* :class:`QLoRASetup` names every mechanism choice the recipe makes -- storage scheme, adapters, where the
  frozen experts live, which expert kernel trains, how many MoE layers keep their activations. The same
  choices were env vars read at import (``train.py``), keyword arguments spread over five calls (the
  documented fast path), or both.
* :func:`estimate_qlora_footprint` prices that setup for a :class:`~experts4bit_qlora.arch.topology.MoETopology`
  without loading a weight. Each line is a :class:`FootprintItem` that says what it is and how it was obtained:
  ``derived`` (sized by constructing this package's own storage and adapter modules on ``meta`` -- the same
  classes the load builds, so the arithmetic cannot drift from them) or ``heuristic`` (a stated formula,
  never presented as a measurement). What it does not model is listed in :attr:`Footprint.unmodelled`.
* :func:`prepare_qlora_training` builds exactly that setup: the documented fast path
  (``docs/solutions/qlora-fused-moe-experts.md``) as one call that asserts every accelerated path engaged.

The training LOOP stays the caller's: ``train.py`` keeps its own, benchmark harnesses theirs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

#: Bytes per logit live at a training step's loss: three fp32 logits-sized tensors together (one allocated in
#: transformers' ``fixed_cross_entropy``, two from frames the replay could not name), measured by allocator replays on
#: an RTX A2000: granite-3.1-3b-a800m (3 x 192.0 MiB at T = 1024, V = 49,155; loggetta#49) and OLMoE-1B-7B (3 x 196.5 MiB,
#: V = 50,304; loggetta#44). It was 10 (bf16 + fp32 + fp32), which left those peaks short by 2 x T x V bytes. The chunked
#: LM loss (``enable_fast_train`` past 1 GiB of fp32 logits) does not materialize them whole; this term does not model it.
LOGITS_LOSS_BYTES = 12
#: storage width per scheme, mirroring ``_vendor.experts._SCHEME_BITS`` without importing a private table
_ADAPTER_BYTES = {"bf16": 2, "fp32": 4}
_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
EXPERT_RESIDENCIES = ("device", "host")
EXPERT_KERNELS = ("grouped_nf4", "reference")


@dataclass(frozen=True)
class QLoRASetup:
    """Every mechanism choice the recipe makes. Defaults are the documented fast path on one GPU."""

    quant_type: str = "nf4"
    blocksize: int = 64
    r: int = 8
    alpha: int = 16
    #: dtype of every LoRA adapter, expert and attention ("bf16" is ``train.py``'s; TC1 trains "fp32")
    adapter_dtype: str = "bf16"
    train_experts: bool = True
    train_attention: bool = True
    #: frozen attention q/k/v/o stored in bitsandbytes NF4 (``TRAIN_ATTN_4BIT``)
    attn_4bit: bool = False
    #: "device": every frozen expert stack resident on the GPU. "host": stacks live in (pinned) host RAM and
    #: stream to the GPU one layer at a time (``OFFLOAD_EXPERTS``; :mod:`~experts4bit_qlora.engines.offload`).
    expert_residency: str = "device"
    pin: bool = True
    #: "grouped_nf4": grouped-nf4-gemm's fused grouped kernel (:func:`~experts4bit_qlora.enable_fast_train`).
    #: "reference": the per-expert bitsandbytes loop -- what ``python -m experts4bit_qlora.train`` runs.
    expert_kernel: str = "grouped_nf4"
    #: route the backward through the single-launch dgrad kernel (``enable_fast_train(dgrad=True)``)
    dgrad: bool = True
    #: MoE layers (counted from the last) whose activations are kept instead of recomputed in backward
    #: (:func:`~experts4bit_qlora.engines.moe_keep.keep_moe_activations`); 0 recomputes every layer
    keep_moe_layers: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FootprintItem:
    name: str
    #: "device" or "host" (memory held), or "link" (bytes moved host to device per micro-batch)
    where: str
    bytes: int
    #: "derived" (constructed from this package's own modules / exact arithmetic) or "heuristic" (a stated formula)
    basis: str
    detail: str = ""


@dataclass(frozen=True)
class Footprint:
    """An itemized memory estimate. Totals are sums of items; nothing outside :attr:`items` is counted."""

    items: tuple
    #: what this estimate leaves out, in words (each one a reason the real peak can be higher)
    unmodelled: tuple = ()
    #: setup choices this topology cannot take, in words; empty when the setup is valid for it
    refusals: tuple = ()

    def total(self, where: str, basis: str | None = None) -> int:
        return sum(i.bytes for i in self.items if i.where == where and (basis is None or i.basis == basis))

    @property
    def device_bytes(self) -> int:
        return self.total("device")

    @property
    def host_bytes(self) -> int:
        return self.total("host")

    def to_dict(self) -> dict:
        return {"items": [asdict(i) for i in self.items], "unmodelled": list(self.unmodelled),
                "refusals": list(self.refusals), "device_bytes": self.device_bytes, "host_bytes": self.host_bytes}


def setup_refusals(topology, setup: QLoRASetup) -> tuple:
    """Why ``setup`` cannot run on ``topology``, in words; ``()`` when it can. Decided from structure only."""
    out = []
    if topology.loader_refusal:
        out.append(f"the loader refuses this model: {topology.loader_refusal}")
        return tuple(out)
    if not topology.expert_stacks:
        out.append("no fused expert stack at the loader's expert path: nothing for this recipe to quantize")
    if setup.expert_residency not in EXPERT_RESIDENCIES:
        out.append(f"expert_residency must be one of {EXPERT_RESIDENCIES}, got {setup.expert_residency!r}")
    if setup.expert_kernel not in EXPERT_KERNELS:
        out.append(f"expert_kernel must be one of {EXPERT_KERNELS}, got {setup.expert_kernel!r}")
    if setup.adapter_dtype not in _ADAPTER_BYTES:
        out.append(f"adapter_dtype must be one of {tuple(_ADAPTER_BYTES)}, got {setup.adapter_dtype!r}")
    if setup.train_experts and topology.expert_bias_tensors:
        out.append(f"train_experts: the expert stacks carry {list(topology.expert_bias_tensors)}, an epilogue "
                   "ExpertsLoRA cannot represent (assert_stock_epilogue); the loader builds them bare")
    if setup.expert_kernel == "grouped_nf4" and setup.quant_type != "nf4":
        out.append(f"expert_kernel='grouped_nf4' reads NF4 stacks only; quant_type is {setup.quant_type!r}")
    if setup.expert_kernel == "grouped_nf4" and not setup.train_experts:
        out.append("expert_kernel='grouped_nf4' patches the ExpertsLoRA wrappers; with train_experts=False "
                   "there is nothing for it to train through")
    if setup.train_attention or setup.attn_4bit:
        what = [n for n, on in (("train_attention", setup.train_attention), ("attn_4bit", setup.attn_4bit)) if on]
        if topology.attention is None:
            out.append(f"{'/'.join(what)}: this model's attention could not be described "
                       f"({topology.provenance.get('attention', 'detector refused it')}); add_attention_lora would refuse it too")
        elif topology.attention.count == 0:
            out.append(f"{'/'.join(what)}: no attention projection the adapter can wrap (q_proj/k_proj/o_proj as "
                       "linears); the setup would train or store nothing there")
    if setup.attn_4bit and topology.attention is not None and topology.attention.any_bias:
        out.append("attn_4bit: an attention projection carries a bias; the NF4 attention store is weight-only")
    if setup.keep_moe_layers and setup.expert_kernel != "grouped_nf4":
        out.append("keep_moe_layers applies through the fused training path only (enable_fast_train)")
    if not (setup.train_experts or setup.train_attention):
        out.append("nothing trains: train_experts and train_attention are both off")
    return tuple(out)


def _module_bytes(mod) -> int:
    return sum(t.numel() * t.element_size() for t in list(mod.parameters()) + list(mod.buffers()) if t is not None)


def _stack_modules(stack, setup):
    """The base and adapter modules the loader builds for one stack, on meta (shapes only, no allocation)."""
    from . import Experts4bit, ExpertsNbit
    from .lora import ExpertsLoRA

    cls = Experts4bit if setup.quant_type in ("nf4", "fp4") else ExpertsNbit
    base = cls(stack.n_experts, stack.hidden, stack.intermediate, has_gate=stack.first_name == "gate_up_proj",
               quant_type=setup.quant_type, blocksize=setup.blocksize, device="meta")
    lora = ExpertsLoRA(base, r=setup.r, alpha=setup.alpha, dtype=_DTYPES.get(setup.adapter_dtype, torch.bfloat16))
    return base, lora


def _first_out(stack) -> int:
    """Per-expert output width of the first projection (2I gated, I not), whatever the stack's orientation."""
    n = 1
    for d in stack.first_shape:
        n *= d
    return n // (stack.n_experts * stack.hidden)


def _pinned_cost(n: int) -> int:
    """PyTorch's caching host allocator rounds each pinned request up to a power of two (grouped-nf4-gemm#71)."""
    return 1 << (int(n) - 1).bit_length() if n > 0 else 0


#: optimizer -> (bytes of state per trainable parameter as a function of the adapter's bytes, how it is known)
OPTIMIZERS = {
    "adamw": (lambda ab: 2 * ab, "exp_avg + exp_avg_sq per trainable parameter, in its dtype (torch.optim.AdamW)"),
    # bitsandbytes AdamW8bit: two uint8 states plus an fp32 absmax per 256-element block each (blockwise 8-bit);
    # tensors under min_8bit_size (4096) keep 32-bit state, which no LoRA tensor here is
    "adamw_8bit": (lambda ab: 2 + 8 / 256, "two uint8 states + an fp32 absmax per 256 values each "
                                            "(bitsandbytes.optim.AdamW8bit)"),
}


def estimate_env() -> dict:
    """The environment switches :func:`estimate_qlora_footprint` reads, with their values in this process (``None`` when
    unset): today ``E4B_CHUNKED_LM_LOSS``, which decides whether the loss branch is priced chunked or whole.

    When to use it: a planner that prices in one process and runs in another records this beside the plan and compares it
    before the run, since the run builds what its own process's switches say. The names come from the modules that read
    them, so the list grows with the estimate rather than being kept twice."""
    import os

    from .engines.chunked_lm_loss import _ENV as chunked_loss_switch

    return {chunked_loss_switch: os.environ.get(chunked_loss_switch)}


def _loss_chunk(topology, setup: QLoRASetup, tokens: int, vocab: int):
    """The chunk size the run's training loss will use, or None for the stock loss: what ``enable_fast_train`` decides
    through :mod:`~experts4bit_qlora.engines.chunked_lm_loss` (its table of supported architectures, its switch and its
    ``auto`` size gate), read the same way here so the estimate prices the loss the run takes."""
    if setup.expert_kernel != "grouped_nf4":
        return None                                        # the reference loop never calls enable_fast_train
    from .engines.chunked_lm_loss import SUPPORTED, chunked_lm_loss_min_bytes, chunked_lm_loss_requested

    if topology.architecture not in SUPPORTED:
        return None
    chunk = chunked_lm_loss_requested()
    gate = chunked_lm_loss_min_bytes()
    if chunk is None or (gate is not None and tokens * vocab * 4 < gate):
        return None
    return chunk


def estimate_qlora_footprint(topology, setup: QLoRASetup, *, tokens_per_microbatch: int,
                             optimizer: str = "adamw") -> Footprint:
    """Price ``setup`` on ``topology`` for micro-batches of ``tokens_per_microbatch`` tokens (padding included),
    trained with ``optimizer`` (a key of :data:`OPTIMIZERS`).

    Covers what the PyTorch allocator holds for this process at the training step's peak, by item, plus the
    host RAM the setup pins. It does not cover the CUDA context, the allocator's reserved-but-unallocated
    blocks, or anything another process holds -- those belong to whoever compares the estimate with a device.

    The loss branch of the activation item follows what the run will do. With ``expert_kernel="grouped_nf4"``,
    ``enable_fast_train`` routes a supported architecture's training forward through the chunked LM loss
    (:mod:`~experts4bit_qlora.engines.chunked_lm_loss`) under ``E4B_CHUNKED_LM_LOSS``: by default (``auto``) exactly when
    the stock fp32 logits, ``T x V x 4`` bytes, reach ``AUTO_MIN_LOGITS_BYTES`` (1 GiB -- e.g. ``T >= 1,767`` at
    Qwen3's 151,936-token vocabulary, ``T >= 5,462`` at granite's 49,155). Then the branch is
    :func:`~experts4bit_qlora.engines.chunked_lm_loss.chunked_loss_bytes` (one chunk's logits plus the gathered hidden
    rows); otherwise it is the whole logits at :data:`LOGITS_LOSS_BYTES` per logit.

    ``E4B_CHUNKED_LM_LOSS`` is read here, at estimate time, so a plan prices the loss the run takes only if both see the
    same environment. In one process (a plan followed by its run) they do. A plan written in one process and run in
    another prices the planning process's setting.
    """
    refusals = setup_refusals(topology, setup)
    if refusals:
        return Footprint(items=(), refusals=refusals)
    T = int(tokens_per_microbatch)
    H, V = topology.hidden_size, topology.vocab_size
    ab = _ADAPTER_BYTES[setup.adapter_dtype]
    items, unmodelled = [], []

    # --- frozen expert storage and expert adapters (built on meta: the same classes the load builds) ------
    slab, slab_max_layer, host_pinned, lora_numel, keep_bytes_per_layer = 0, 0, 0, 0, 0
    for st in topology.expert_stacks:
        base, lora = _stack_modules(st, setup)
        b = _module_bytes(base)
        slab += b
        slab_max_layer = max(slab_max_layer, b)
        if setup.expert_residency == "host":
            host_pinned += sum((_pinned_cost if setup.pin else int)(t.numel() * t.element_size())
                               for t in (base.gate_up_proj, base.down_proj, base.gate_up_absmax, base.down_absmax)
                               if t is not None)
        if setup.train_experts:
            lora_numel += sum(p.numel() for n, p in lora.named_parameters() if "lora" in n)
        first_out = _first_out(st)
        keep_bytes_per_layer = max(keep_bytes_per_layer,
                                   T * (topology.top_k or 0) * (first_out + st.intermediate + st.hidden) * 2)
    qdesc = f"{setup.quant_type}, blocksize {setup.blocksize}"
    if setup.expert_residency == "device":
        items.append(FootprintItem("frozen expert stacks", "device", slab, "derived",
                                   f"{len(topology.expert_stacks)} stacks in {qdesc} (packed + absmax)"))
    else:
        items.append(FootprintItem("frozen expert stacks, one layer staged", "device", slab_max_layer, "derived",
                                   "offload streams one layer's stack to the GPU at a time"))
        items.append(FootprintItem("frozen expert stacks, host homes" + (" (pinned)" if setup.pin else ""), "host",
                                   host_pinned, "derived",
                                   f"{slab / 1e9:.2f} GB of {qdesc}"
                                   + (", each pinned tensor rounded up to a power of two" if setup.pin else "")))
        # engines/offload.py: the pre-hook stages a layer's stack for its forward and again for the checkpoint
        # recompute in backward (single resident slot), so every micro-batch moves the whole slab twice
        items.append(FootprintItem("expert staging, host to device per micro-batch", "link", 2 * slab, "derived",
                                   "each layer's stack is copied once for its forward and once for the "
                                   "checkpoint recompute (engines/offload.py)"))
        if setup.expert_kernel == "reference":
            # docs/OFFLOAD_MEMORY_FACTS.md: a constant ~0.64 GB over the shape model on OLMoE (reference loop,
            # every quantized scheme), ~79% of one layer's bf16 dequantized stack. Its mechanism is open.
            w = int(0.79 * 2 * max(s.numel for s in topology.expert_stacks))
            items.append(FootprintItem("offload staging transient", "device", w, "heuristic",
                                       "0.79 x one layer's bf16 dequantized experts (measured ratio on OLMoE, "
                                       "reference loop; mechanism unexplained)"))
        else:
            unmodelled.append("offload staging transient on the fused kernel path (unmeasured)")

    # --- dense (non-expert) weights ----------------------------------------------------------------------
    dense = topology.dense_numel
    attn = topology.attention
    if setup.attn_4bit and attn is not None:
        dense -= attn.numel
        # bitsandbytes Linear4bit, compress_statistics=True: 4-bit codes, an 8-bit absmax per 64 values and an
        # fp32 second-level scale per 256 absmax blocks
        q4 = attn.numel // 2 + attn.numel // 64 + (attn.numel // (64 * 256)) * 4
        items.append(FootprintItem("attention projections (NF4)", "device", q4, "derived",
                                   f"{attn.count} projections, bitsandbytes nested statistics"))
    items.append(FootprintItem("dense weights (bf16)", "device", 2 * dense, "derived",
                               "embeddings, attention, norms, routers, dense/shared MLPs; tied head counted once"))

    # --- adapters, gradients, optimizer ------------------------------------------------------------------
    attn_lora = setup.r * attn.in_plus_out if (setup.train_attention and attn is not None) else 0
    trainable = lora_numel + attn_lora
    if lora_numel:
        items.append(FootprintItem("expert LoRA adapters", "device", lora_numel * ab, "derived",
                                   f"ExpertsLoRA r={setup.r}, {setup.adapter_dtype}"))
    if attn_lora:
        items.append(FootprintItem("attention LoRA adapters", "device", attn_lora * ab, "derived",
                                   f"{attn.count} projections, r={setup.r}, {setup.adapter_dtype}"))
    items.append(FootprintItem("adapter gradients", "device", trainable * ab, "derived",
                               "one gradient per trainable parameter, in its dtype"))
    per_param, how = OPTIMIZERS[optimizer]
    items.append(FootprintItem(f"optimizer state ({optimizer})", "device", int(per_param(ab) * trainable), "derived",
                               how))

    # --- activations (gradient checkpointing on; a stated formula, not a measurement) ---------------------
    n_layers = topology.n_layers
    boundaries = n_layers * T * H * 2
    chunk = _loss_chunk(topology, setup, T, V)
    if chunk:
        from .engines.chunked_lm_loss import chunked_loss_bytes

        logits = chunked_loss_bytes(T, V, hidden=H, chunk=chunk)
        loss_how = f"the chunked LM loss in {chunk}-token chunks = {logits / 1e9:.2f} GB"
    else:
        logits = T * V * LOGITS_LOSS_BYTES
        loss_how = f"logits and loss, three fp32 logits-sized tensors = {logits / 1e9:.2f} GB"
    st0 = topology.expert_stacks[0]
    first_out0 = _first_out(st0)
    kv = (attn.kv_elements_per_token // max(attn.layers, 1)) if attn else 0
    layer = T * ((H + kv + H) * 2 + (topology.top_k or 0) * (first_out0 + st0.intermediate + H) * 2
                 + st0.n_experts * 4)
    items.append(FootprintItem("activations", "device", boundaries + max(logits, 2 * layer), "heuristic",
                               f"{n_layers} saved layer inputs (T x H bf16) + max({loss_how}, 2 x one layer's recompute = "
                               f"{2 * layer / 1e9:.2f} GB) at T={T}"))
    if setup.keep_moe_layers:
        n_keep = min(setup.keep_moe_layers, len(topology.expert_stacks))
        items.append(FootprintItem("kept MoE activations", "device", n_keep * keep_bytes_per_layer, "heuristic",
                                   f"{n_keep} layers x T x top_k x (first_out + I + H) bf16"))
    if attn is not None and attn.layers < n_layers:
        unmodelled.append(f"{n_layers - attn.layers} of {n_layers} layers mix tokens without q/k/v attention "
                          "(state-space, convolution or linear attention): their recompute working set")
    unmodelled += ["CUDA context, cuBLAS/Triton workspaces and allocator fragmentation (the caller's to add)",
                   "load-time transients (the streaming loader's per-tensor quantize; host page cache)"]
    return Footprint(items=tuple(items), unmodelled=tuple(unmodelled))


@dataclass
class PreparedQLoRA:
    model: object
    trainable: list
    #: what was actually engaged, for a receipt: counts the setup implies, and every accelerated path asserted
    report: dict = field(default_factory=dict)


def prepare_qlora_training(model_id, setup: QLoRASetup, *, device="cuda", revision=None) -> PreparedQLoRA:
    """Load ``model_id`` and build ``setup`` on it: the documented fast path as one call.

    Order: :func:`~experts4bit_qlora.load_moe_4bit_streaming` (experts quantized on the way in; host-resident
    when ``expert_residency="host"``), the optional NF4 attention store, gradient checkpointing, attention LoRA,
    then :func:`~experts4bit_qlora.enable_fast_train` for ``expert_kernel="grouped_nf4"``. Every requested
    accelerated path must engage or this raises ``RuntimeError`` -- never a silent fallback. Returns the model,
    the trainable parameters (every LoRA tensor; nothing else requires grad) and a report of what engaged.
    Refuses (``ValueError``) a setup :func:`setup_refusals` rejects for this model, before loading anything.
    """
    from .arch.topology import describe_moe
    from .loader import load_moe_4bit_streaming
    from .lora import ExpertsLoRA, LoRALinear, add_attention_lora, quantize_attention_projections_4bit

    refusals = setup_refusals(describe_moe(model_id, revision=revision), setup)
    if refusals:
        raise ValueError("QLoRASetup refused for this model: " + "; ".join(refusals))
    host = setup.expert_residency == "host"
    model, config = load_moe_4bit_streaming(model_id, device, torch.bfloat16, setup.r, setup.alpha, offload=host,
                                            pin=setup.pin, quant_type=setup.quant_type, revision=revision,
                                            blocksize=setup.blocksize)
    if not host:
        model.to(device)
    report = {"setup": setup.to_dict(), "commit": getattr(config, "_commit_hash", None)}
    if setup.attn_4bit:
        report["attn_4bit_projections"] = quantize_attention_projections_4bit(model)
        if not report["attn_4bit_projections"]:
            raise RuntimeError("attn_4bit: no attention projection was converted")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    dtype = _DTYPES[setup.adapter_dtype]
    if setup.train_attention:
        report["attention_lora_projections"] = add_attention_lora(model, setup.r, setup.alpha, dtype)
    if setup.train_experts and dtype != torch.bfloat16:
        for mod in model.modules():
            if isinstance(mod, ExpertsLoRA):
                for n, p in mod.named_parameters(recurse=False):
                    if "lora" in n:
                        p.data = p.data.to(dtype)
    if setup.expert_kernel == "grouped_nf4":
        from .engines.fast import enable_fast_train
        from .engines.rmsnorm_train import fused_rmsnorm_requested
        from .engines.rope_train import fused_rope_requested

        from .engines.moe_keep import moe_keep_layers_requested

        # enable_fast_train also reads E4B_MOE_KEEP_LAYERS; say so if the environment set it behind the setup
        report["env_moe_keep_layers"] = moe_keep_layers_requested()
        n = enable_fast_train(model, dgrad=setup.dgrad)
        if n == 0:
            raise RuntimeError("expert_kernel='grouped_nf4': enable_fast_train patched 0 modules "
                               "(is grouped-nf4-gemm installed?)")
        report.update(fused_expert_modules=n, dgrad=setup.dgrad, fused_rope=fused_rope_requested(),
                      fused_rmsnorm=fused_rmsnorm_requested())
        try:
            import nf4_route

            report["train_gemm_route"] = nf4_route.train_gemm_route(torch.device(device))
        except ImportError:
            report["train_gemm_route"] = "fused (grouped-nf4-gemm without nf4_route)"
        if setup.keep_moe_layers:
            from .engines.moe_keep import keep_moe_activations

            report["kept_moe_layers"] = keep_moe_activations(model, setup.keep_moe_layers)
    trainable = []
    for mod in model.modules():
        if isinstance(mod, (ExpertsLoRA, LoRALinear)):
            for n, p in mod.named_parameters(recurse=False):
                if "lora" in n and (setup.train_attention if isinstance(mod, LoRALinear) else setup.train_experts):
                    trainable.append(p)
    keep = {id(p) for p in trainable}
    for p in model.parameters():
        p.requires_grad_(id(p) in keep)
    report["trainable_numel"] = sum(p.numel() for p in trainable)
    return PreparedQLoRA(model=model, trainable=trainable, report=report)
