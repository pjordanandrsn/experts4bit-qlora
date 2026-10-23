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
- ``experts.fp32_source`` (present when ``compress_expert_absmax_`` compressed a stack): the same digest with each
  compressed stack's fp32 absmax, hashed in full before compression. It proves the BASE matches an fp32 arena; it never
  proves the served absmax equals the trained (double-quantized) one.
"""
from __future__ import annotations

import hashlib
import sys
import time

SCHEMA_V2 = "backend-adapter/2"
SCHEME_ID = "qfp-v1"
KNOWN_SCHEMES = (SCHEME_ID,)
SCOPES = ("attention", "experts")
#: where the tensors are looked up (the canonical lines are the same function whatever the source):
#: ``resident`` the module's own tensors (an evicted placeholder is refused); ``offload_homes`` each stack where its bytes
#: live (resident when staged, its expert-offload host home when evicted); ``arena`` a served model's arena rows (PA-1)
SOURCES = ("resident", "offload_homes", "arena")
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


def _shape(dims) -> str:
    return "x".join(str(int(d)) for d in dims)


def linear_line(name: str, *, kind: str, shape, blocksize: int, data, absmax=None, code=None) -> tuple:
    """THE canonical qfp-v1 line for one linear: every writer and every reader (PA-1's arena reader included) MUST
    compute its lines through this function, never re-derive the format. ``data`` is the packed bytes (``bnb4``) or the
    weight (``dense``), ``absmax`` the fp32 absmax as dequantize computes it, ``code`` the fp32 code table. Returns
    ``(line, bytes_hashed)``."""
    d = _bytes(data)
    a = _bytes(absmax.float()) if absmax is not None else b""
    c = _bytes(code.float()) if code is not None else b""
    return "\t".join((name, kind, _shape(shape), str(int(blocksize)), _sha(d), _sha(a), _sha(c))), len(d) + len(a) + len(c)


def absmax_sha256(absmax) -> str:
    """The qfp-v1 hash of an fp32 absmax's bytes: the absmax field of :func:`expert_line`. ``compress_expert_absmax_``
    records it for the fp32 absmax it compresses, before compressing (the fp32 source fingerprint)."""
    return _sha(_bytes(absmax.float()))


def expert_line(name: str, which: str, *, bits: int, quant_type: str, num_experts: int, shape, blocksize: int,
                packed, absmax=None, code=None, absmax_sha=None) -> tuple:
    """THE canonical qfp-v1 line for one projection of an expert stack: every writer and every reader (PA-1's arena
    reader included) MUST compute its lines through this function, from tensors found anywhere (a resident stack, an
    offload host home, an arena's rows), never re-derive the format. ``packed`` is the stack's packed storage (its
    BYTES are sampled, whatever the dtype), ``absmax`` the fp32 absmax ``[E, N, K/blocksize]`` as dequantize computes
    it, or, instead, ``absmax_sha`` its :func:`absmax_sha256` (already taken, or recorded before compression). Returns
    ``(line, bytes_hashed)``."""
    import torch

    if (absmax is None) == (absmax_sha is None):
        raise ValueError("expert_line takes the absmax or its absmax_sha256, exactly one")
    flat = packed.detach().contiguous().reshape(-1).view(torch.uint8)       # bytes, whatever the storage dtype
    n = EXPERT_SUBSET_BYTES
    sample = _bytes(flat) if flat.numel() <= 2 * n else _bytes(flat[:n]) + _bytes(flat[-n:])
    a = _bytes(absmax.float()) if absmax is not None else b""
    c = _bytes(code.float()) if code is not None else b""
    line = "\t".join((f"{name}.{which}", f"experts{int(bits)}:{quant_type}:{which}", _shape((num_experts, *shape)),
                      str(int(blocksize)), _sha(sample), _sha(a) if absmax is not None else absmax_sha, _sha(c)))
    return line, len(sample) + len(a) + len(c)


def _linear_record(name, lin) -> tuple:
    """One attention projection's base (a LoRA wrapper is unwrapped), through :func:`linear_line`."""
    w = _unwrap(lin).weight
    qs = getattr(w, "quant_state", None)
    if qs is not None:
        from .engines.train_qkv_fuse import _expanded_absmax

        return linear_line(name, kind=f"bnb4:{qs.quant_type}", shape=qs.shape, blocksize=int(qs.blocksize),
                           data=w.data, absmax=_expanded_absmax(qs), code=qs.code)
    return linear_line(name, kind=f"dense:{str(w.dtype).replace('torch.', '')}", shape=w.shape, blocksize=0, data=w)


def _resolved(name, mod, handle, attr):
    """``mod.<attr>``, or, while expert offload has it evicted (a 0-element placeholder), its HOST home: the same
    substitution ``state_dict`` makes, so the canonical form is the same wherever the bytes live. A placeholder with no
    readable host copy is refused by name, never hashed as empty."""
    t = getattr(mod, attr)
    if t is None or t.numel() > 0:
        return t
    home = getattr(handle, "home", None) if handle is not None else None
    h = home.get(attr) if isinstance(home, dict) else None
    if h is None or h.is_meta or h.numel() == 0 or h.device.type != "cpu":
        raise ValueError(f"quant_fingerprint: expert stack {name} is evicted and its host copy of {attr} is "
                         f"unavailable ({'no offload handle' if handle is None else 'no readable CPU home'})")
    return h


def _expert_record(name, which, mod, *, shape, packed, absmax, code, source_sha=None) -> dict:
    """One projection's record: ``line`` (through :func:`expert_line`), ``bytes``, ``key`` (``<stack>.<which>``),
    ``absmax`` (the line's absmax hash) and, for a stack ``compress_expert_absmax_`` compressed, ``source`` (the
    hash of the fp32 absmax it recorded before compressing; None if it recorded none) and ``source_line`` (the line
    with that absmax). A stack whose absmax is fp32 has no ``source``: its own line already reads the fp32 absmax."""
    from .absmax_dq import is_absmax_compressed

    common = dict(bits=int(mod.bits), quant_type=mod.quant_type, num_experts=mod.num_experts, shape=shape,
                  blocksize=int(mod.blocksize), packed=packed, code=code)
    a_sha = absmax_sha256(absmax)
    line, n = expert_line(name, which, absmax_sha=a_sha, **common)
    rec = {"line": line, "bytes": n + absmax.numel() * 4, "key": f"{name}.{which}", "absmax": a_sha}
    if is_absmax_compressed(mod):
        rec["source"] = source_sha
        rec["source_line"] = expert_line(name, which, absmax_sha=source_sha, **common)[0] if source_sha else None
    return rec


def _expert_records(name, mod, handle=None, source="offload_homes") -> list:
    """One :func:`_expert_record` for each of one expert stack's gate_up and down (qfp-v1's fixed subset of the packed
    bytes). ``handle`` is the stack's expert-offload handle, when one is installed (its homes hold the evicted
    tensors)."""
    from .absmax_dq import absmax_source_sha256, expert_absmax_fp32

    out = []
    for which in ("gate_up", "down"):
        shape = getattr(mod, f"_{which}_shape")
        evicted = getattr(mod, f"{which}_proj").numel() == 0
        if evicted and source == "resident":
            raise ValueError(f"quant_fingerprint: expert stack {name} is evicted by expert offload; source='resident' "
                             f"reads only resident tensors: fingerprint it with source='offload_homes'")
        packed = _resolved(name, mod, handle, f"{which}_proj")
        if evicted:
            # evicted: the host home holds the fp32 absmax (offload refuses a double-quantized absmax, so the module's
            # absmax attribute, which raises once compressed, is never read here)
            home = _resolved(name, mod, handle, f"{which}_absmax")
            absmax = home.view(mod.num_experts, shape[0], shape[1] // int(mod.blocksize))
        else:
            absmax = expert_absmax_fp32(mod, which)
        out.append(_expert_record(name, which, mod, shape=shape, packed=packed, absmax=absmax,
                                  code=getattr(mod, "code", None), source_sha=absmax_source_sha256(mod, which)))
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
    """``[(name, stack, offload_handle)]`` of every quantized expert stack (an :class:`ExpertsLoRA` is unwrapped to its
    base). The handle is the expert-offload handle on the wrapper (or on a bare stack), else None."""
    from ._vendor.experts import Experts4bit

    handles = {}
    for mod in model.modules():
        h = getattr(mod, "_offload", None)
        if h is not None and getattr(h, "base", None) is not None:
            handles[id(h.base)] = h
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, Experts4bit):
            # an ExpertsLoRA holds its stack as ``.base``: the stack is named as the served model names it
            out.append((name[: -len(".base")] if name.endswith(".base") else name, mod, handles.get(id(mod))))
    return out


_ARENA_DTYPES = {"U8": "uint8", "F32": "float32", "BF16": "bfloat16", "F16": "float16"}


def _arena_segment(fh, index: dict, offsets: dict, layer: int, num_experts: int, suffix: str):
    """One segment of every expert of ``layer``, ``[E, *shape_per_expert]``, read from the arena file at the offsets
    its index records (``rows``: ``[layer, expert, offset]``; ``segments``: ``seg_off`` / ``length`` inside a row). A plain
    seek and read, the same bytes grouped-nf4-gemm's ``nvme_residency.segment_tensor`` hands the server (a test checks
    it where that reader runs)."""
    import torch

    geo = next((g for g in index["segments"] if g["suffix"] == suffix), None)
    if geo is None or geo.get("dtype") not in _ARENA_DTYPES:
        raise ValueError(f"PA-1: the arena has no readable segment {suffix!r} "
                         f"(have {[g.get('suffix') for g in index.get('segments', ())]})")
    dt = getattr(torch, _ARENA_DTYPES[geo["dtype"]])
    out = []
    for e in range(num_experts):
        if (layer, e) not in offsets:
            raise ValueError(f"PA-1: the arena index has no row for layer {layer} expert {e}")
        fh.seek(offsets[(layer, e)] + int(geo["seg_off"]))
        raw = bytearray(fh.read(int(geo["length"])))
        if len(raw) != int(geo["length"]):
            raise ValueError(f"PA-1: a short read of layer {layer} expert {e} segment {suffix!r}")
        out.append(torch.frombuffer(raw, dtype=dt).reshape(tuple(geo["shape_per_expert"])))
    return torch.stack(out)


def _arena_expert_records(model, arena: str) -> list:
    """``[(line, bytes_hashed)]`` for every expert stack of a SERVED model, from the arena it serves (PA-1). The served
    stacks are meta: their names, shapes and recipe come from the modules, their bytes from the arena's rows, the k-th
    stack being arena layer k (the loader's order). Each line goes through :func:`expert_line`, the function every other
    source uses, and the code table through the stack class's own ``_build_code``."""
    import json

    from ._vendor.experts import _build_code
    from .engines.nvme_train import arena_offload_view

    with open(arena + ".index.json", encoding="utf-8") as f:
        index = json.load(f)
    view, segs = arena_offload_view(index)
    stacks = _expert_stacks(model)
    if len(stacks) != int(index["n_layers"]):
        raise ValueError(f"PA-1: the model has {len(stacks)} expert stacks, the arena {index['n_layers']} layers")
    offsets = {(int(r[0]), int(r[1])): int(r[2]) for r in view["rows"]}
    out = []
    with open(arena, "rb") as fh:
        for layer, (name, mod, _handle) in enumerate(stacks):
            if int(mod.num_experts) != int(index["n_experts_per_layer"]):
                raise ValueError(f"PA-1: {name} has {mod.num_experts} experts, the arena {index['n_experts_per_layer']}")
            code = _build_code(mod.quant_type, "cpu")
            for which in ("gate_up", "down"):
                shape = getattr(mod, f"_{which}_shape")
                packed = _arena_segment(fh, view, offsets, layer, mod.num_experts, segs[f"{which}_proj"])
                absmax = _arena_segment(fh, view, offsets, layer, mod.num_experts, segs[f"{which}_absmax"]).float()
                absmax = absmax.reshape(mod.num_experts, shape[0], shape[1] // int(mod.blocksize))
                out.append(_expert_record(name, which, mod, shape=shape, packed=packed, absmax=absmax, code=code))
    return out


def quant_fingerprint(model, scope: str, *, source: str = "resident", arena: str | None = None) -> dict:
    """``{"scheme_id", "scope", "source", "modules", "sha256", "bytes_hashed", "seconds"}`` for ``scope``
    (``"attention"`` or ``"experts"``), by qfp-v1 (module docstring), looking the tensors up in ``source`` (``SOURCES``).
    A training run with expert offload fingerprints with ``source="offload_homes"``; a served model's experts with
    ``source="arena"`` and ``arena=`` the arena it serves (PA-1). An empty scope is refused: nothing to fingerprint is
    not a match.

    The experts scope adds ``absmax_by_stack`` (``{"<stack>.<which>": absmax hash}``) and, when
    ``compress_expert_absmax_`` compressed any stack, ``fp32_source``: ``{"sha256", "absmax_by_stack"}``, the same
    qfp-v1 digest taken with each compressed stack's fp32 absmax as recorded BEFORE compression (every other line
    unchanged). It is the fingerprint the base had in fp32, so a server compares it with its fp32 arena's ``sha256``
    exactly: it proves the BASE is the trained one, never that the served absmax equals the trained (double-quantized)
    one. It is absent when a compressed stack recorded no fp32 hash."""
    if scope not in SCOPES:
        raise ValueError(f"quant_fingerprint scope must be one of {SCOPES}, got {scope!r}")
    if source not in SOURCES:
        raise ValueError(f"quant_fingerprint source must be one of {SOURCES}, got {source!r}")
    if source == "arena" and (scope != "experts" or not arena):
        raise ValueError("quant_fingerprint source='arena' fingerprints the experts scope of a served model and needs "
                         "arena=<path>; a served model's attention is resident (source='resident')")
    t0 = time.perf_counter()
    records = []
    if scope == "attention":
        for name, lin in _attention_projections(model):
            line, n = _linear_record(name, lin)
            records.append({"line": line, "bytes": n})
    elif source == "arena":
        records.extend(_arena_expert_records(model, arena))
    else:
        for name, mod, handle in _expert_stacks(model):
            records.extend(_expert_records(name, mod, handle, source))
    if not records:
        raise ValueError(f"quant_fingerprint: no {scope} module in this model")
    out = {"scheme_id": SCHEME_ID, "scope": scope, "source": source, "modules": len(records),
           "sha256": _digest(r["line"] for r in records), "bytes_hashed": sum(r["bytes"] for r in records)}
    if scope == "experts":
        out["absmax_by_stack"] = {r["key"]: r["absmax"] for r in records}
        compressed = [r for r in records if "source" in r]
        if compressed and all(r["source"] for r in compressed):
            out["fp32_source"] = {"sha256": _digest(r.get("source_line") or r["line"] for r in records),
                                  "absmax_by_stack": {r["key"]: r.get("source") or r["absmax"] for r in records}}
    out["seconds"] = round(time.perf_counter() - t0, 3)
    return out


def _digest(lines) -> str:
    """qfp-v1's digest of a scope's lines: sorted, newline-joined, newline-terminated."""
    return _sha(("\n".join(sorted(lines)) + "\n").encode("utf-8"))


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
    fused = int(qkv_stats.get("fused") or 0) if qkv_stats is not None else 0
    forms = set()
    for _, lin in _attention_projections(model):
        qs = getattr(getattr(_unwrap(lin), "weight", None), "quant_state", None)
        forms.add(("bnb4", qs.quant_type, int(qs.blocksize), bool(qs.nested)) if qs is not None
                  else ("bf16", None, None, None))
    keys = ("form", "quant_type", "blocksize", "nested")
    if len(forms) > 1:                                   # every projection read: a mix has no one recipe to record
        raise ValueError(f"attention projections are quantized differently {sorted(forms, key=str)}: "
                         f"backend-adapter/2 records one attention recipe, so this model is refused")
    attn = (dict(zip(keys, next(iter(forms)))) if forms else
            {"form": "none", "quant_type": None, "blocksize": None, "nested": None})
    versions = {}
    for dist in ("bitsandbytes", "torch", "grouped-nf4-gemm", "transformers", "experts4bit-qlora"):
        try:
            versions[dist] = version(dist)
        except PackageNotFoundError:
            versions[dist] = None
    return {"expert_absmax": absmax, "train_fused_qkv": fused, "attn_quant": attn, "versions": versions,
            "lora_scale_applied": lora_scale_applied(model)}
