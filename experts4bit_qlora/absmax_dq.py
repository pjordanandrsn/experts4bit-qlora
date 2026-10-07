"""Double-quantized ("nested") storage for the frozen expert absmax (the CLI trainer's default for resident training).

An :class:`~experts4bit_qlora.ExpertsNbit` NF4 stack keeps one fp32 absmax per 64 weights,
``gate_up_absmax`` / ``down_absmax`` of shape ``[E, N*K/64]``. On a resident training run that is
real memory -- about 2.8 GB on Mixtral-8x7B, 1.8 GB on Qwen3-30B-A3B and 2.1 GB on
Qwen3.6-35B-A3B -- and it is what separated e4b's 31.07 GB Mixtral peak on a 32 GB RTX 5090 from
Unsloth's 29.12 GB (Unsloth trains on bitsandbytes' double-quantized statistics), and part of why
Qwen3.6-35B-A3B did not fit at all.

:func:`compress_expert_absmax_` replaces each stack's fp32 absmax with bitsandbytes' own nested
form, computed exactly as ``bitsandbytes.functional.quantize_4bit(W, compress_statistics=True)``
does it for the same stack (bitsandbytes 0.50, ``functional.py``)::

    offset = absmax.mean()
    qabsmax, state2 = quantize_blockwise(absmax - offset, blocksize=256)

stored as four buffers per projection -- ``<which>_absmax_q`` (uint8 ``[E, N*K/64]``),
``<which>_absmax_s`` (fp32, one per 256), ``<which>_absmax_off`` (fp32 ``[1]``) and
``<which>_absmax_code`` (bitsandbytes' 256-entry dynamic map, ``state2.code``) -- 1.016 bytes per
64 weights instead of 4. Dequantization is bitsandbytes' too: ``dequantize_blockwise(q, state2) +
offset``. The VALUES are therefore lossy against the fp32 absmax (they are what Unsloth and
``bnb_4bit_use_double_quant`` train on); with the switch off nothing here runs and every value is
unchanged.

What reads the compressed form. grouped-nf4-gemm's kernels take only a plain fp32 ``[E, N, K//64]``
absmax, and no kernel changes for this. :func:`expert_absmax_fp32` expands ONE projection of ONE
layer just in time, where the training path hands it to a kernel (a transient buffer, freed after
the call; the backward's ``weights_fn`` expands it again rather than holding it), and
:func:`expert_absmax_rows` gives the per-expert reference loop the same values one expert at a
time. The paths routed through them: ``enable_fast_train``'s fused training forward (and
``enable_fast``'s inference forward over an ``ExpertsLoRA``), and the ``ExpertsLoRA`` reference
loop -- the parity control. EVERY other reader of the absmax -- expert offload, the batched,
hot/cold/pipelined/hybrid residency and NVMe engines, the gpt-oss / DeepSeek-V4 per-expert
forwards -- refuses a compressed module by name (:class:`AbsmaxCompressedError`), and the old
attribute names hold a guard object that raises the same error on any use, so a reader that was
missed fails loudly instead of reading a wrong buffer.

The switch: ``compress_expert_absmax_(model)`` in code. ``python -m experts4bit_qlora.train`` applies it by
default for resident training (after load, before training; off under ``OFFLOAD_EXPERTS=1`` / ``TRAIN_ARENA``;
a model the compressor refuses keeps its fp32 absmax), since TC1 amendments 28 / 31 read 1.4 % of the step for
1.34 GB on Qwen3-30B-A3B and 2.3 % for 2.04 GB on Mixtral-8x7B; ``E4B_ABSMAX_DQ=0`` turns it off and ``=1``
requires it (refused with ``OFFLOAD_EXPERTS=1``). ``enable_fast_train`` applies the same default since TC1
amendment 56 (packed 4,096-token rows: 1.002 of the step for 1.35 GB, torch 2.12 on one RTX 5090), with the
same guards and switch, plus ``absmax_dq=False`` / ``True`` in code. The TC1 harness's ``--absmax-dq`` stays explicit.
"""
from __future__ import annotations

import os

import torch

ENV_SWITCH = "E4B_ABSMAX_DQ"
NESTED_BLOCKSIZE = 256          # bitsandbytes' quantize_4bit(compress_statistics=True) nests in blocks of 256
_KERNEL_BLOCKSIZE = 64          # grouped-nf4-gemm's absmax contract: one fp32 per 64 weights
_WHICH = ("gate_up", "down")
_MARK = "_e4b_absmax_dq"
_SUFFIXES = ("_q", "_s", "_off", "_code")


class AbsmaxCompressedError(RuntimeError):
    """A path that reads the fp32 expert absmax met a module whose absmax is double-quantized."""


def absmax_dq_requested() -> bool:
    """Whether ``E4B_ABSMAX_DQ=1`` is set (the trainer's and the TC1 harness's switch)."""
    return os.environ.get(ENV_SWITCH, "0").strip() == "1"


def _refusal(entry: str) -> str:
    return (f"{entry}: this expert stack's absmax is stored double-quantized (compress_expert_absmax_: "
            f"enable_fast_train's and the CLI trainer's default for resident training, or {ENV_SWITCH}=1), and {entry} "
            "reads the fp32 absmax buffer, which no longer exists. Only the resident training paths read the compressed "
            "form -- enable_fast_train's fused forward and the ExpertsLoRA reference loop, through "
            f"experts4bit_qlora.expert_absmax_fp32. To use {entry}, keep the fp32 absmax: set {ENV_SWITCH}=0 before "
            "enable_fast_train (or pass absmax_dq=False to it) and do not call compress_expert_absmax_.")


class _CompressedAbsmaxGuard:
    """Sits under the old attribute name (``gate_up_absmax`` / ``down_absmax``) of a compressed module.

    Not a tensor and not None: ``base.gate_up_absmax is None`` (the passthrough test) stays False, and any
    use -- ``.view``, ``.numel``, ``[e]``, ``.to`` -- raises :class:`AbsmaxCompressedError` naming the switch,
    so a reader that was never routed through the accessor fails loudly rather than reading another buffer.
    Dunder lookups raise AttributeError as usual, so copy / pickle / repr keep working."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def _refuse(self):
        raise AbsmaxCompressedError(_refusal(f"reading {self.name}"))

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        self._refuse()

    def __getitem__(self, idx):
        self._refuse()

    def __iter__(self):
        self._refuse()

    def __len__(self):
        self._refuse()

    def __reduce__(self):
        return (type(self), (self.name,))

    def __repr__(self):
        return (f"<{self.name}: double-quantized by compress_expert_absmax_ ({ENV_SWITCH}=1); "
                "read it through experts4bit_qlora.expert_absmax_fp32>")


def is_absmax_compressed(mod) -> bool:
    """True iff :func:`compress_expert_absmax_` compressed this module's absmax."""
    return _MARK in getattr(mod, "__dict__", {})


def assert_absmax_uncompressed(mod, entry: str) -> None:
    """Raise :class:`AbsmaxCompressedError` if ``mod`` (one module) holds a compressed absmax. Cheap: one dict lookup."""
    if _MARK in getattr(mod, "__dict__", {}):
        raise AbsmaxCompressedError(_refusal(entry))


def refuse_compressed_absmax(obj, entry: str) -> None:
    """Raise :class:`AbsmaxCompressedError` if ``obj`` -- a model, an ``ExpertsLoRA`` or an expert stack -- holds
    any module whose absmax is double-quantized. The enable-time refusal of every engine that reads the fp32
    absmax (offload, batched, residency, NVMe, hybrid)."""
    mods = obj.modules() if hasattr(obj, "modules") else [obj]
    for m in mods:
        assert_absmax_uncompressed(m, entry)


def _shape(mod, which: str):
    if which == "gate_up":
        return mod._gate_up_shape
    if which == "down":
        return mod._down_shape
    raise ValueError(f"which must be one of {_WHICH}, got {which!r}")


def _buffers(mod, which: str):
    return tuple(getattr(mod, f"{which}_absmax{s}") for s in _SUFFIXES)


def expert_absmax_fp32(mod, which: str) -> torch.Tensor:
    """The fp32 absmax of one projection of one expert stack, kernel-shaped ``[E, N, K//64]``.

    ``which`` is ``"gate_up"`` or ``"down"``. Uncompressed: exactly what the fused paths passed before
    (``<which>_absmax.view(E, N, K // 64).float()`` -- a view of the stored buffer, no copy). Compressed:
    a fresh fp32 tensor, ``bitsandbytes.functional.dequantize_blockwise(q, state2) + offset`` -- the
    statement bitsandbytes' ``dequantize_4bit`` runs for nested statistics -- which the caller hands to a
    kernel and drops (one layer's projection at a time; nothing holds it across layers).
    """
    E = mod.num_experts
    n, k = _shape(mod, which)
    if _MARK not in mod.__dict__:
        return getattr(mod, f"{which}_absmax").view(E, n, k // _KERNEL_BLOCKSIZE).float()
    import bitsandbytes.functional as F
    from bitsandbytes.functional import QuantState

    q, s, off, code = _buffers(mod, which)
    state2 = QuantState(absmax=s, code=code, blocksize=NESTED_BLOCKSIZE, dtype=torch.float32)
    flat = F.dequantize_blockwise(q.reshape(-1), state2)
    flat += off
    return flat.view(E, n, k // _KERNEL_BLOCKSIZE)


class _CompressedAbsmaxRows:
    """``rows[e]`` -> expert ``e``'s fp32 absmax ``[N*K/64]``, dequantized from the nested form.

    What the per-expert loop indexes in place of the stored buffer (``ExpertsNbit._dequantize_expert``
    reads ``absmax[expert_idx]``; ``ExpertsLoRA._gemv_project`` likewise). The same arithmetic as
    bitsandbytes' blockwise dequantize -- ``code[q] * absmax2[block]`` in fp32, then ``+ offset`` -- on one
    expert's slice only, so the values are identical to the matching rows of :func:`expert_absmax_fp32`.
    ``e`` may be a Python int or a 0-d device tensor (the decode loop indexes without a host sync)."""

    __slots__ = ("q", "s", "off", "code", "E", "M", "B", "aligned")

    def __init__(self, mod, which: str):
        self.q, self.s, self.off, self.code = _buffers(mod, which)
        self.E, self.M = self.q.shape
        self.aligned = self.M % NESTED_BLOCKSIZE == 0
        self.B = self.M // NESTED_BLOCKSIZE

    def __getitem__(self, e):
        v = self.code[self.q[e].long()]
        if self.aligned:
            # Expert rows start on a nested-block boundary: expert e owns blocks [e*B, (e+1)*B).
            s = self.s.view(self.E, self.B)[e]
            out = (v.view(self.B, NESTED_BLOCKSIZE) * s.view(self.B, 1)).view(self.M)
        else:
            pos = e * self.M + torch.arange(self.M, device=self.q.device)
            out = v * self.s[torch.div(pos, NESTED_BLOCKSIZE, rounding_mode="floor")]
        return out + self.off


def expert_absmax_rows(mod, which: str):
    """What the per-expert reference loop indexes as ``absmax[e]``: the stored ``<which>_absmax`` buffer
    itself when uncompressed (the same object as before -- nothing changes), else a light view whose
    ``[e]`` dequantizes expert ``e``'s row of the nested form (see :class:`_CompressedAbsmaxRows`)."""
    if _MARK not in mod.__dict__:
        return getattr(mod, f"{which}_absmax")
    return _CompressedAbsmaxRows(mod, which)


def _module_absmax_bytes(mod) -> int:
    total = 0
    for which in _WHICH:
        if is_absmax_compressed(mod):
            total += sum(t.numel() * t.element_size() for t in _buffers(mod, which))
        else:
            t = getattr(mod, f"{which}_absmax", None)
            if isinstance(t, torch.Tensor):
                total += t.numel() * t.element_size()
    return total


def expert_absmax_bytes(model) -> int:
    """Bytes the expert absmax occupies under ``model``: the fp32 buffers of uncompressed stacks plus the four
    nested buffers of compressed ones. Read it before and after :func:`compress_expert_absmax_` for the saving."""
    from . import ExpertsNbit

    mods = model.modules() if hasattr(model, "modules") else [model]
    return sum(_module_absmax_bytes(m) for m in mods if isinstance(m, ExpertsNbit))


def _why_not(mod, wrapper) -> str | None:
    """None if ``mod`` (an ExpertsNbit holding an absmax) can be compressed, else the refusal reason."""
    if wrapper is None:
        return ("a BARE expert stack (no ExpertsLoRA wrapper): its own forward reads the fp32 absmax buffer "
                "directly, so compressing it would break it; only ExpertsLoRA-wrapped stacks train through "
                "the accessor")
    if getattr(mod, "_e4b_mxfp4_arena", False):
        return "storage is MXFP4 (arena): e8m0 scales, not an fp32 absmax"
    if getattr(mod, "bits", None) != 4:
        return f"storage is {getattr(mod, 'quant_type', '?')} ({getattr(mod, 'bits', '?')}-bit); only 4-bit stacks are compressed"
    if getattr(mod, "blocksize", None) != _KERNEL_BLOCKSIZE:
        return f"blocksize {getattr(mod, 'blocksize', None)} != {_KERNEL_BLOCKSIZE}"
    if getattr(wrapper, "_offload", None) is not None:
        return ("expert offload is enabled on this layer (its absmax lives in the offload handle's host home and "
                "is staged by name); expert offload cannot carry the compressed form")
    if "forward" in mod.__dict__:
        return ("an engine rebound this stack's forward (enable_fast / hot / cold / pipelined / NVMe residency); "
                "those read the fp32 absmax -- compress before attaching nothing but enable_fast_train")
    if "forward" in wrapper.__dict__ and not ("_e4b_train_ref" in wrapper.__dict__ or "_e4b_fast_ref" in wrapper.__dict__):
        return ("an engine other than enable_fast_train / enable_fast patched this ExpertsLoRA's forward "
                "(enable_batched_train, hybrid / NVMe training); it reads the fp32 absmax")
    n_gu = mod._gate_up_shape[0] * mod._gate_up_shape[1] // _KERNEL_BLOCKSIZE
    n_dn = mod._down_shape[0] * mod._down_shape[1] // _KERNEL_BLOCKSIZE
    for which, m in (("gate_up", n_gu), ("down", n_dn)):
        t = mod._buffers.get(f"{which}_absmax")
        if not isinstance(t, torch.Tensor):
            return f"{which}_absmax is not a registered buffer ({type(t).__name__})"
        if t.dtype != torch.float32:
            return f"{which}_absmax is {t.dtype}, not float32"
        if t.is_meta or t.numel() != mod.num_experts * m:
            return (f"{which}_absmax holds {t.numel()} elements on {t.device} (expected {mod.num_experts * m}): "
                    "evicted, offloaded or not materialized")
    return None


@torch.no_grad()
def _compress_one(mod) -> None:
    import bitsandbytes.functional as F

    for which in _WHICH:
        absmax = mod._buffers[f"{which}_absmax"]
        flat = absmax.reshape(-1)
        # bitsandbytes quantize_4bit(compress_statistics=True), statement for statement.
        offset = flat.mean()
        qabsmax, state2 = F.quantize_blockwise(flat - offset, blocksize=NESTED_BLOCKSIZE)
        # Remove the fp32 buffer BEFORE registering the replacements, so the old name stops resolving to a
        # tensor; the guard takes the name as a plain attribute.
        delattr(mod, f"{which}_absmax")
        mod.register_buffer(f"{which}_absmax_q", qabsmax.reshape(absmax.shape).contiguous())
        mod.register_buffer(f"{which}_absmax_s", state2.absmax.contiguous())
        mod.register_buffer(f"{which}_absmax_off", offset.reshape(1).to(torch.float32))
        mod.register_buffer(f"{which}_absmax_code", state2.code.to(torch.float32).contiguous())
        object.__setattr__(mod, f"{which}_absmax", _CompressedAbsmaxGuard(f"{type(mod).__name__}.{which}_absmax"))
        del absmax, flat
    object.__setattr__(mod, _MARK, {"nested_blocksize": NESTED_BLOCKSIZE})


def compress_expert_absmax_(model) -> int:
    """Store every ExpertsLoRA-wrapped NF4 expert stack's absmax double-quantized, in place (the CLI trainer's and
    ``enable_fast_train``'s default).

    When to use it: resident QLoRA training where the fp32 absmax is the margin between fitting and not
    (one fp32 per 64 weights: ~2.8 GB on Mixtral-8x7B, ~1.8 GB on Qwen3-30B-A3B). It stores bitsandbytes'
    nested statistics -- what ``quantize_4bit(W, compress_statistics=True)`` produces for the same stack, at
    ~3.94x fewer bytes -- so the absmax VALUES change (lossy against fp32, identical to bitsandbytes' own
    double-quant). Call it after loading (``load_moe_4bit_streaming``, which installs the ExpertsLoRA
    wrappers) and before training; ``enable_fast_train`` may be applied before or after it.

    Expected layout: ExpertsNbit stacks (nf4/fp4, blocksize 64, fp32 absmax resident on one device) each
    wrapped by an ExpertsLoRA. Passthrough (bf16/fp16) stacks carry no absmax and are skipped.

    Returns the number of stacks compressed by THIS call. Idempotent: a stack already compressed is left as
    it is and not counted, so a second call returns 0 -- assert the first call's count is non-zero.

    Refuses (``ValueError``, before touching any module, so a refusal leaves the model unchanged): a bare
    stack with no ExpertsLoRA wrapper (its own forward reads the fp32 buffer), MXFP4-arena storage, 8-bit
    storage, a blocksize other than 64, a layer under expert offload, a stack whose forward an engine rebound
    or whose wrapper an engine other than enable_fast_train / enable_fast patched, and an absmax that is
    evicted, meta or not fp32. Afterwards those engines refuse the model in turn (:class:`AbsmaxCompressedError`).

    Platform: bitsandbytes' ``quantize_blockwise`` / ``dequantize_blockwise`` on the absmax's device (CUDA
    for training; the CPU backend works too). See ``docs/solutions/qlora-fused-moe-experts.md``.
    """
    from . import ExpertsNbit
    from .lora import ExpertsLoRA

    mods = list(model.modules()) if hasattr(model, "modules") else [model]
    wrapper_of = {id(m.base): m for m in mods if isinstance(m, ExpertsLoRA) and hasattr(m, "base")}
    todo, refused = [], []
    for name_mod in (model.named_modules() if hasattr(model, "named_modules") else [("", model)]):
        name, mod = name_mod
        if not isinstance(mod, ExpertsNbit) or is_absmax_compressed(mod):
            continue
        if getattr(mod, "bits", 16) >= 16:
            continue                                     # passthrough storage: no absmax
        why = _why_not(mod, wrapper_of.get(id(mod)))
        if why is not None:
            refused.append(f"{name or type(mod).__name__}: {why}")
        else:
            todo.append(mod)
    if refused:
        more = f" (and {len(refused) - 3} more)" if len(refused) > 3 else ""
        raise ValueError("compress_expert_absmax_ refuses -- nothing was compressed: "
                         + "; ".join(refused[:3]) + more)
    for mod in todo:
        _compress_one(mod)
    return len(todo)
