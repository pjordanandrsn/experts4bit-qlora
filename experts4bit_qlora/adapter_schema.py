# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``backend-adapter/2``: what a trained adapter's manifest records so a server can reproduce and VERIFY the function
it was trained to correct (docs/ADAPTER_PORTABILITY.md, section 6). One schema and one fingerprint, used by both sides:
Loggetta's writer and e4b's reader.

* :func:`quant_fingerprint` hashes the quantized base in a CANONICAL form, the one dequantize computes, never the
  storage form. Training's fused q/k/v path stores the same NF4 weights as a non-nested fp32 absmax, the unfused path
  as nested 8-bit statistics, and both dequantize bit for bit the same. So the same base gives the same fingerprint
  whichever form stored it, while a lossy double-quantized absmax gives a different one from an fp32 absmax (F1).
  Scheme ``qfp-v1`` is pinned below. A change to the canonical form is a new scheme, and a reader refuses a scheme it
  does not know.
* :func:`lora_scale_applied` is the one accessor for the LoRA scale the modules hold. The writer records it and never
  recomputes alpha/r.
* :func:`resolved_training_facts` is what ``prepare_qlora_training`` resolved: the expert absmax form, the fused-q/k/v
  count, the attention quantization as built, and the versions. Each value is read from the runtime, never from the
  environment.

``backend-adapter/2`` is defined for ``backend="experts4bit"`` only. A dense ``/2`` manifest is refused by name until a
dense section is added.

**qfp-v1, the canonicalization:**
- One record per module in scope, sorted by module name (UTF-8 code points).
- Each record is a line ``name TAB kind TAB shape TAB blocksize TAB sha256(data) TAB sha256(absmax) TAB sha256(code)``,
  and the fingerprint is the sha256 of the lines joined by ``\\n`` with a trailing ``\\n``.
- ``kind``:
  - ``bnb4:<quant_type>``: a bitsandbytes 4-bit linear. ``data`` is its packed uint8 bytes, ``absmax`` the fp32
    absmax as ``dequantize_4bit`` computes it (``dequantize_blockwise(absmax, state2) + offset`` when nested), ``code``
    its fp32 code table.
  - ``dense:<dtype>``: a plain linear. ``data`` is its weight's bytes; ``absmax`` and ``code`` are empty.
  - ``experts<bits>:<quant_type>:<gate_up|down>``: one projection of an expert stack. ``absmax`` is the stack's fp32
    absmax (``absmax_dq.expert_absmax_fp32``: the stored buffer, or the nested form dequantized). ``data`` is a FIXED
    subset of the packed bytes, the first and last ``EXPERT_SUBSET_BYTES`` of the stack's flattened storage (all of
    it when smaller), and ``code`` the fp32 code table.
- ``shape`` is the logical weight shape, decimal integers joined by ``x`` (an expert stack is ``E x N x K``).
  ``blocksize`` is a decimal integer, ``0`` for dense. Every tensor is hashed as its contiguous little-endian bytes.

**What each fingerprint proves:**
- ``attention``: every attention projection's base, in full. Two models with the same attention fingerprint dequantize
  every attention weight to the same values.
- ``experts``: the full absmax of every stack and a fixed byte sample. That is a RECIPE check: it catches a
  quant_type, blocksize or absmax-form mismatch anywhere (an fp32 and a double-quantized absmax differ in every
  block). It is NOT a content proof of the packed bytes outside the sample.
"""
from __future__ import annotations

import hashlib
import sys
import time

SCHEMA_V2 = "backend-adapter/2"
SCHEME_ID = "qfp-v1"
KNOWN_SCHEMES = (SCHEME_ID,)
SCOPES = ("attention", "experts")
#: bytes taken from each end of an expert stack's flattened packed storage (qfp-v1)
EXPERT_SUBSET_BYTES = 64 * 1024

#: the fields backend-adapter/2 adds to v1's, each with what it proves and what it does not (docs section 6)
V2_FIELDS = {
    "expert_absmax": "the expert absmax form training resolved (fp32 | double-quantized), from the runtime's record; "
                     "proves which form the adapter was trained against, not that a server's arena matches it",
    "train_fused_qkv": "how many attention modules training fused (q/k/v as one projection); equal-to-rounding "
                       "numerics, recorded so a reproduction can match it",
    "attn_quant": "the attention base as built: form (bf16 | bnb4), quant_type, blocksize, nested statistics",
    "versions": "bitsandbytes, torch, grouped-nf4-gemm, transformers and experts4bit-qlora as imported at training",
    "lora_scale_applied": "the LoRA scale the modules held (attention, experts), read from them, never recomputed",
    "fingerprints": "quant_fingerprint per scope (attention, experts): scheme_id, sha256, module count and the seconds "
                    "it took; see 'What each fingerprint proves'",
    "run_id": "the run that produced the adapter (writer-side)",
    "created_at": "UTC time the manifest was written (writer-side)",
}


def _bytes(t) -> bytes:
    """A tensor's contiguous little-endian bytes (qfp-v1); the CPU copy is taken here, one tensor at a time."""
    import torch

    if sys.byteorder != "little":                       # pragma: no cover - every supported host is little-endian
        raise RuntimeError("qfp-v1 hashes little-endian bytes; this host is big-endian")
    t = t.detach().contiguous().cpu()
    return t.view(torch.uint8).numpy().tobytes() if t.numel() else b""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _linear_record(name, lin) -> tuple:
    """``(line, bytes_hashed)`` for one attention projection's base (a LoRA wrapper is unwrapped)."""
    base = getattr(lin, "base", lin)
    w = base.weight
    qs = getattr(w, "quant_state", None)
    if qs is not None:
        from .engines.train_qkv_fuse import _expanded_absmax

        data, absmax, code = _bytes(w.data), _bytes(_expanded_absmax(qs)), _bytes(qs.code.float())
        kind, shape, bs = f"bnb4:{qs.quant_type}", "x".join(str(int(d)) for d in qs.shape), int(qs.blocksize)
    else:
        data, absmax, code = _bytes(w), b"", b""
        kind, shape, bs = f"dense:{str(w.dtype).replace('torch.', '')}", "x".join(str(int(d)) for d in w.shape), 0
    line = "\t".join((name, kind, shape, str(bs), _sha(data), _sha(absmax), _sha(code)))
    return line, len(data) + len(absmax) + len(code)


def _expert_records(name, mod) -> list:
    """``[(line, bytes_hashed)]`` for one expert stack's gate_up and down (qfp-v1's fixed subset of the packed bytes)."""
    from .absmax_dq import expert_absmax_fp32

    out = []
    for which in ("gate_up", "down"):
        packed = getattr(mod, f"{which}_proj")
        flat = packed.detach().reshape(-1)
        n = EXPERT_SUBSET_BYTES
        sample = _bytes(flat) if flat.numel() <= 2 * n else _bytes(flat[:n]) + _bytes(flat[-n:])
        absmax = _bytes(expert_absmax_fp32(mod, which))
        code = _bytes(mod.code.float()) if getattr(mod, "code", None) is not None else b""
        shape = getattr(mod, f"_{which}_shape")
        kind = f"experts{int(mod.bits)}:{mod.quant_type}:{which}"
        line = "\t".join((f"{name}.{which}", kind, "x".join(str(int(d)) for d in (mod.num_experts, *shape)),
                          str(int(mod.blocksize)), _sha(sample), _sha(absmax), _sha(code)))
        out.append((line, len(sample) + len(absmax) + len(code)))
    return out


_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj")


def _unwrap(mod):
    """The base linear under a LoRA wrapper (``lora.LoRALinear.base``), else the module itself."""
    from .lora import LoRALinear

    return mod.base if isinstance(mod, LoRALinear) else mod


def _attention_projections(model) -> list:
    """``[(name, module)]`` of every attention projection: a module carrying ``q_proj`` and an output projection
    (``o_proj``, or ``out_proj`` beside ``q_proj`` and ``k_proj``), each an ``nn.Linear`` once a LoRA wrapper is
    unwrapped. The NAME is the projection's path in the model, the same with or without the wrapper, so a training
    model and a served model fingerprint under the same names."""
    from torch import nn

    out = []
    for parent, mod in model.named_modules():
        have = {p: _unwrap(getattr(mod, p)) for p in _PROJ if isinstance(getattr(mod, p, None), nn.Module)}
        lin = {p: m for p, m in have.items() if isinstance(m, nn.Linear)}
        if "q_proj" not in lin or not ("o_proj" in lin or ("out_proj" in lin and "k_proj" in lin)):
            continue
        for proj in sorted(lin):
            out.append((f"{parent}.{proj}" if parent else proj, getattr(mod, proj)))
    return out


def _expert_stacks(model) -> list:
    """``[(name, stack)]`` of every quantized expert stack (an :class:`ExpertsLoRA` is unwrapped to its base)."""
    from ._vendor.experts import Experts4bit

    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, Experts4bit):
            # an ExpertsLoRA holds its stack as ``.base``: the stack is named as the served model names it
            out.append((name[: -len(".base")] if name.endswith(".base") else name, mod))
    return out


def quant_fingerprint(model, scope: str) -> dict:
    """``{"scheme_id", "scope", "modules", "sha256", "bytes_hashed", "seconds"}`` for ``scope`` (``"attention"`` or
    ``"experts"``), by qfp-v1 (module docstring). An empty scope is refused: nothing to fingerprint is not a match."""
    if scope not in SCOPES:
        raise ValueError(f"quant_fingerprint scope must be one of {SCOPES}, got {scope!r}")
    t0 = time.perf_counter()
    records = []
    if scope == "attention":
        for name, lin in _attention_projections(model):
            records.append(_linear_record(name, lin))
    else:
        for name, mod in _expert_stacks(model):
            records.extend(_expert_records(name, mod))
    if not records:
        raise ValueError(f"quant_fingerprint: no {scope} module in this model")
    lines = sorted(line for line, _ in records)
    digest = _sha(("\n".join(lines) + "\n").encode("utf-8"))
    return {"scheme_id": SCHEME_ID, "scope": scope, "modules": len(lines), "sha256": digest,
            "bytes_hashed": sum(n for _, n in records), "seconds": round(time.perf_counter() - t0, 3)}


def lora_scale_applied(model) -> dict:
    """``{"attention": s | None, "experts": s | None}``: the scale every LoRA module of each kind holds. Modules of one
    kind that disagree are an error, never averaged."""
    from .lora import ExpertsLoRA, LoRALinear

    seen = {"attention": set(), "experts": set()}
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            seen["attention"].add(float(mod.scaling))
        elif isinstance(mod, ExpertsLoRA):
            seen["experts"].add(float(mod.scaling))
    out = {}
    for kind, vals in seen.items():
        if len(vals) > 1:
            raise ValueError(f"{kind} LoRA modules hold different scales {sorted(vals)}")
        out[kind] = next(iter(vals)) if vals else None
    return out


def resolved_training_facts(model, setup, *, fast_stats: dict | None, qkv_stats: dict | None) -> dict:
    """What training resolved, read from the runtime: the fields ``prepare_qlora_training`` adds to its report AFTER
    ``enable_fast_train`` has run (it resolves the absmax form and fuses q/k/v)."""
    from importlib.metadata import PackageNotFoundError, version

    rec = (fast_stats or {}).get("absmax_dq")
    if rec is None:
        absmax = {"form": "fp32", "stacks": 0,
                  "record": {"reason": "enable_fast_train did not run: the absmax stays fp32 by construction"}}
    else:
        n = int(rec.get("compressed") or 0)
        absmax = {"form": "double-quantized" if n else "fp32", "stacks": n, "record": dict(rec)}
    fused = int((qkv_stats or {}).get("fused") or 0) if rec is not None else 0
    attn = {"form": "bf16", "quant_type": None, "blocksize": None, "nested": None}
    for _, lin in _attention_projections(model):
        qs = getattr(getattr(getattr(lin, "base", lin), "weight", None), "quant_state", None)
        if qs is not None:
            attn = {"form": "bnb4", "quant_type": qs.quant_type, "blocksize": int(qs.blocksize),
                    "nested": bool(qs.nested)}
        break
    versions = {}
    for dist in ("bitsandbytes", "torch", "grouped-nf4-gemm", "transformers", "experts4bit-qlora"):
        try:
            versions[dist] = version(dist)
        except PackageNotFoundError:
            versions[dist] = None
    return {"expert_absmax": absmax, "train_fused_qkv": fused, "attn_quant": attn, "versions": versions,
            "lora_scale_applied": lora_scale_applied(model)}
