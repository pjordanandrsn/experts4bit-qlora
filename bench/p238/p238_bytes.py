#!/usr/bin/env python3
"""p238_bytes.py -- P238's per-load premise table: the NF4 bytes each loaded model holds, per layer and projection, in one canonical
form, so 40 loads in 40 processes (and the op process) can be held equal by the reducer.

Per (layer, projection) it records sha256 of:
- ``packed``: the 4-bit payload, flattened in storage order (both sides store bitsandbytes' flat packing of the [E, N, K] stack:
  e4b's ``ExpertsNbit._quantize_stack``, Unsloth's ``Params4bit``; the CPU premise found them byte-equal flattened);
- ``absmax_q``: the double-quantized absmax (uint8), ``absmax_s``: its fp32 second-level scales, ``absmax_off``: the fp32 offset,
  ``absmax_code``: the 256-entry second-level code;
- ``code``: the 16-entry NF4 code.

Where each side keeps them:
- e4b (``experts4bit_qlora/_vendor/experts.py`` ``ExpertsNbit``, inside an ``ExpertsLoRA`` as ``.base``): ``gate_up_proj`` /
  ``down_proj`` (the packed parameters), the nested buffers ``<which>_absmax_q`` / ``_s`` / ``_off`` / ``_code``
  (``absmax_dq.install_compressed_absmax_``, absmax_dq.py:316-319), and ``code``. A stack still on its fp32 absmax (no nested
  buffers) is recorded as ``absmax_fp32`` and can never equal an Unsloth table, which is the point: P238 runs ``E4B_ABSMAX_DQ=1``.
- Unsloth (the experts module of each layer, ``gate_up_proj`` / ``down_proj`` unwrapped through ``base_layer`` to the
  ``Params4bit``): ``.data``, ``quant_state.absmax`` (uint8 when nested), ``quant_state.state2.absmax``, ``quant_state.offset``,
  ``quant_state.state2.code``, ``quant_state.code``.

The table never decides a verdict here: :func:`compare` returns the first difference, and the reducer and the box's stop rule use it.
Standard library plus torch, imported by the caller's process.
"""
from __future__ import annotations

import hashlib
import re

PROJ = ("gate_up", "down")
FIELDS = ("packed", "absmax_q", "absmax_s", "absmax_off", "absmax_code", "code")
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _sha(t):
    import torch
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    t = t.contiguous().reshape(-1)
    if t.dtype != torch.uint8:
        t = t.view(torch.uint8)              # the raw bytes, whatever the dtype
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def _layer(name):
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _base_param(p):
    while hasattr(p, "base_layer"):
        p = p.base_layer
    return p


def e4b_table(model):
    """{layer: {proj: {field: sha256}}} from every ExpertsNbit in ``model`` (an ExpertsLoRA's ``.base`` is reached by modules())."""
    from experts4bit_qlora import ExpertsNbit
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, ExpertsNbit):
            continue
        L = _layer(name)
        row = {}
        for which in PROJ:
            f = {"packed": _sha(getattr(mod, f"{which}_proj")), "code": _sha(mod.code)}
            q = getattr(mod, f"{which}_absmax_q", None)
            if q is None:
                f["absmax_fp32"] = _sha(getattr(mod, f"{which}_absmax"))
            else:
                for suf in ("q", "s", "off", "code"):
                    f[f"absmax_{suf}"] = _sha(getattr(mod, f"{which}_absmax_{suf}"))
            row[which] = f
        out[L if L is not None else name] = row
    return out


def unsloth_table(model):
    """The same table from every module holding both expert stacks as bitsandbytes ``Params4bit`` (after unwrapping PEFT)."""
    from bitsandbytes.nn import Params4bit
    out = {}
    for name, mod in model.named_modules():
        ps = {w: _base_param(getattr(mod, f"{w}_proj", None)) for w in PROJ}
        if not all(isinstance(p, Params4bit) for p in ps.values()):
            continue
        L = _layer(name)
        row = {}
        for which, p in ps.items():
            qs = p.quant_state
            f = {"packed": _sha(p.data), "code": _sha(qs.code)}
            if getattr(qs, "nested", False):
                import torch
                f.update(absmax_q=_sha(qs.absmax), absmax_s=_sha(qs.state2.absmax),
                         absmax_off=_sha(torch.as_tensor(qs.offset, dtype=torch.float32).reshape(1)), absmax_code=_sha(qs.state2.code))
            else:
                f["absmax_fp32"] = _sha(qs.absmax)
            row[which] = f
        out[L if L is not None else name] = row
    return out


def compare(a, b, n_layers=None):
    """(ok, why): two tables equal on every layer, projection and field. ``n_layers``, when given, requires exactly layers
    0..n_layers-1 on both sides (a table that lost a layer must not compare equal on the layers it kept)."""
    if not a or not b:
        return False, f"an empty table ({len(a or {})} vs {len(b or {})} layers)"
    if n_layers is not None:
        want = set(range(n_layers))
        for tag, t in (("first", a), ("second", b)):
            if set(t) != want:
                return False, f"the {tag} table has layers {sorted(t, key=str)[:5]}... not 0..{n_layers - 1}"
    if set(a) != set(b):
        return False, f"layer sets differ: {sorted(set(a) ^ set(b), key=str)[:5]}"
    for L in sorted(a, key=str):
        for which in PROJ:
            fa, fb = a[L].get(which), b[L].get(which)
            if fa is None or fb is None:
                return False, f"layer {L} {which}: missing"
            for k in sorted(set(fa) | set(fb)):
                if fa.get(k) != fb.get(k):
                    return False, f"layer {L} {which} {k}: {str(fa.get(k))[:12]} != {str(fb.get(k))[:12]}"
    return True, ""
