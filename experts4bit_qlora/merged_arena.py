# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""A trained adapter's expert delta, baked offline into a NEW NF4 arena (path (a); bench/d0/ARENA-DESIGN.md).

:func:`write` reads a base arena that grouped-nf4-gemm's ``bake_nf4`` wrote and a backend-adapter/2 directory, and writes
an arena with the base's geometry. Each (layer, expert) row is re-quantized from ``S + s * B[e] @ A[e]``, the delta
computed in fp32 with the scale the trained modules held (``lora_scale_applied.experts``, which must equal alpha / r):

* ``source="arena"`` (the default): S is the base arena's own rows, dequantized in fp32 (codebook times fp32 absmax,
  never through bf16). ``source="checkpoint"``: S is the raw checkpoint in fp32, read by the bake itself (``bake_nf4``
  with a quantizer that adds the delta), so the reader, the layout and the geometry are the bake's own.
* S is proven to be the base before anything is written. The base arena's qfp-v1 must equal the adapter's recorded
  expert fingerprint (its ``fp32_source`` for a double-quantized trained base), and every row of S, re-quantized with
  no delta, must reproduce the base arena's bytes exactly (the self-check; for the checkpoint, the proof that it is the
  base). A tie the writer's device would round differently from the bake's is caught here, never written.
* The quantizer is the bake's call: ``quantize_4bit``, nf4, blocksize 64, no compression, fp32 absmax.
* The index keeps the base's geometry, sets ``bake_mode`` to :data:`MERGED_BAKE_MODE` and adds a ``merge`` block, the
  record a server's merged-arena rules read, the merged arena's own fingerprint included. A retention receipt (D0's
  Step-1 sums per layer and projection) is written beside it: a record, never a gate.

Nothing serves a merged arena yet. :func:`refuse_merged` runs wherever this package opens an arena index (the loader and
both NVMe residencies), so a merged arena is never read as a base: served without its adapter, or trained from.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import types

MERGED_BAKE_MODE = "nf4-merged"
BASE_BAKE_MODE = "nf4-quantize"
WRITER = "experts4bit_qlora.merged_arena"
WRITER_VERSION = 1
BLOCK = 64
SOURCES = ("arena", "checkpoint")
WHICH = ("gate_up", "down")
#: the sidecars beside a merged arena, after the bake's two
RETENTION = ".retention.json"
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


class MergeRefused(ValueError):
    """The writer will not write this merge; the message names why."""


def is_merged(index: dict) -> bool:
    return index.get("bake_mode") == MERGED_BAKE_MODE or "merge" in index


def refuse_merged(index: dict, arena: str) -> None:
    """Every arena this package opens passes here. A merged arena holds a trained function, so it is never read as a
    base: neither served under the base's name nor trained from. Serving one with its own adapter is the server's
    opt-in (bench/d0/ARENA-DESIGN.md, point 4), not yet supported. Raises ValueError."""
    if is_merged(index):
        adapter = str((index.get("merge") or {}).get("adapter_id") or "unrecorded")[:12]
        raise ValueError(f"{arena} is a merged arena (bake_mode {index.get('bake_mode')!r}, adapter {adapter}): it is "
                         f"never read as a base, and serving it with its adapter is not supported yet "
                         f"(bench/d0/ARENA-DESIGN.md)")


def expert_tensors_sha256(ad) -> str:
    """One hash over the adapter's expert LoRA tensors: each name, dtype, shape and the sha256 of its bytes, sorted."""
    import torch

    lines = []
    for name in sorted(ad.expert_tensors):
        t = ad.tensors[name].detach().contiguous().reshape(-1).view(torch.uint8)
        lines.append(f"{name}\t{ad.tensors[name].dtype}\t{list(ad.tensors[name].shape)}\t"
                     f"{hashlib.sha256(t.numpy().tobytes()).hexdigest()}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scale(ad) -> float:
    applied = ad.manifest.get("lora_scale_applied")
    held = applied.get("experts") if isinstance(applied, dict) else None
    if held is None:
        raise MergeRefused("the adapter records no lora_scale_applied.experts: the scale its expert modules held is "
                           "unknown, and it is never recomputed")
    if abs(float(held) - ad.scale) > 1e-12:
        raise MergeRefused(f"lora_scale_applied.experts {held!r} is not the setup's alpha/r {ad.scale!r}")
    return float(held)


def _trained_base(ad) -> tuple:
    """``(qfp-v1 sha256 the base arena must have, trained-base form)`` from the adapter's recorded expert fingerprint."""
    from .adapter_dir import SCHEMA_V2, _dq_trained

    fp = (ad.manifest.get("fingerprints") or {}).get("experts") if ad.manifest.get("schema") == SCHEMA_V2 else None
    if not isinstance(fp, dict) or not isinstance(fp.get("absmax_by_stack"), dict):
        raise MergeRefused("the adapter records no expert fingerprint by stack (a backend-adapter/2 manifest's "
                           "fingerprints.experts): the writer cannot prove the arena is its base")
    if _dq_trained(ad):
        src = fp.get("fp32_source")
        if not isinstance(src, dict) or not src.get("sha256"):
            raise MergeRefused("a double-quantized trained base needs the fingerprint's fp32_source to prove the fp32 "
                               "arena is its base; this manifest records none")
        return src["sha256"], "double-quantized"
    return fp["sha256"], "fp32-absmax"


def _stack_names(ad, layer_ids) -> dict:
    """``{arena layer id: served stack name}``, from the adapter's recorded expert fingerprint keys
    (``<stack>.<which>``). Every arena layer has exactly one stack, and every key is a known projection."""
    keys = ad.manifest["fingerprints"]["experts"]["absmax_by_stack"]
    stacks = {k.rsplit(".", 1)[0] for k in keys}
    if set(keys) != {f"{s}.{w}" for s in stacks for w in WHICH}:
        raise MergeRefused(f"the fingerprint's stacks do not each carry exactly {list(WHICH)}")
    by_layer = {}
    for s in stacks:
        ids = _LAYER.findall(s + ".")
        if not ids:
            raise MergeRefused(f"stack {s!r} names no layer")
        by_layer.setdefault(int(ids[-1]), []).append(s)
    if sorted(by_layer) != sorted(layer_ids) or any(len(v) != 1 for v in by_layer.values()):
        raise MergeRefused(f"the adapter's stacks (layers {sorted(by_layer)}) are not the arena's layers "
                           f"{sorted(layer_ids)}")
    return {lay: v[0] for lay, v in by_layer.items()}


def _check_tensors(ad, names: dict, geo: dict, n_experts: int) -> None:
    """The adapter's expert LoRA tensors are exactly one A and one B per stack and projection, in ExpertsLoRA's layout
    (``A`` ``[E, r, in]``, ``B`` ``[E, out, r]``), so no delta is dropped and none is read transposed."""
    r = int(ad.setup["r"])
    want = {}
    for name in names.values():
        for which in WHICH:
            n, k = geo[which]["shape"]
            want[f"{name}.{which}_lora_A"] = (n_experts, r, k)
            want[f"{name}.{which}_lora_B"] = (n_experts, n, r)
    have = set(ad.expert_tensors)
    if have != set(want):
        extra, missing = sorted(have - set(want)), sorted(set(want) - have)
        raise MergeRefused(f"the adapter's expert LoRA tensors are not this arena's stacks: missing {missing[:2]}, "
                           f"not merged {extra[:2]}")
    for name, shape in want.items():
        if tuple(ad.tensors[name].shape) != shape:
            raise MergeRefused(f"{name} is {list(ad.tensors[name].shape)}, not {list(shape)} (ExpertsLoRA's layout)")


def _check_base(index: dict, arena: str) -> dict:
    """The base arena's geometry, or :class:`MergeRefused`: only a complete fp32-absmax ``bake_nf4`` NF4 bake."""
    if is_merged(index) or index.get("bake_mode") != BASE_BAKE_MODE:
        raise MergeRefused(f"{arena}: bake_mode {index.get('bake_mode')!r}; the writer reads only a bake_nf4 "
                           f"{BASE_BAKE_MODE!r} base (a merged arena is never a base)")
    q = index.get("quantizer") or {}
    if q.get("quant_type") != "nf4" or int(q.get("blocksize") or 0) != BLOCK or q.get("absmax_dtype", "f32") != "f32":
        raise MergeRefused(f"{arena}: quantizer {q!r}; the writer reads nf4, blocksize {BLOCK}, fp32 absmax")
    dims = index.get("model_dims") or {}
    if int(index["n_experts_per_layer"]) != int(dims.get("E", -1)):
        raise MergeRefused(f"{arena}: a partial bake ({index['n_experts_per_layer']} of {dims.get('E')} experts)")
    inter, hidden = int(dims["I"]), int(dims["H"])
    shapes = {"gate_up": (2 * inter, hidden), "down": (hidden, inter)}
    geo = {g["suffix"]: g for g in index["segments"]}
    out = {}
    for which, (n, k) in shapes.items():
        b, a = geo.get(f"nf4.{which}_blocks"), geo.get(f"nf4.{which}_absmax")
        if not b or not a or b.get("dtype") != "U8" or a.get("dtype") != "F32" \
                or int(b["length"]) != n * k // 2 or int(a["length"]) != n * k // BLOCK * 4:
            raise MergeRefused(f"{arena}: the {which} segments are not nf4 blocks + fp32 absmax of {n} x {k}")
        out[which] = {"shape": (n, k), "blocks": b, "absmax": a}
    return out


def _stand_ins(names: dict, index: dict) -> list:
    """The served stacks as adapter_schema's arena reader reads them, without a model: what it takes from each module
    (name, expert count, recipe and shapes), in layer order."""
    inter, hidden = int(index["model_dims"]["I"]), int(index["model_dims"]["H"])
    return [(names[lay], types.SimpleNamespace(num_experts=int(index["n_experts_per_layer"]), quant_type="nf4", bits=4,
                                               blocksize=BLOCK, _gate_up_shape=(2 * inter, hidden),
                                               _down_shape=(hidden, inter)), None)
            for lay in sorted(names)]


def _code():
    from ._vendor.experts import _build_code
    return _build_code("nf4", "cpu").float()


def _deq(blocks: bytes, absmax: bytes, n: int, k: int, code):
    """fp32 ``[n, k]``: codebook[code] times the fp32 absmax, high nibble first (bitsandbytes' packing)."""
    import torch

    p = torch.frombuffer(bytearray(blocks), dtype=torch.uint8)
    a = torch.frombuffer(bytearray(absmax), dtype=torch.float32)
    codes = torch.stack([(p >> 4).long(), (p & 0xF).long()], -1).reshape(-1)
    return (code[codes].view(-1, BLOCK) * a.view(-1, 1)).view(n, k)


def _quantize(x) -> tuple:
    """``(blocks, absmax)`` bytes: the bake's call on fp32 ``x``."""
    from bitsandbytes import functional as F

    q, st = F.quantize_4bit(x.float().contiguous(), blocksize=BLOCK, quant_type="nf4", compress_statistics=False)
    return q.reshape(-1).contiguous().numpy().tobytes(), st.absmax.float().contiguous().numpy().tobytes()


def _read(fh, off: int, seg: dict) -> bytes:
    fh.seek(off + int(seg["seg_off"]))
    raw = fh.read(int(seg["length"]))
    if len(raw) != int(seg["length"]):
        raise MergeRefused(f"a short read at offset {off} segment {seg['suffix']!r}")
    return raw


class _Merge:
    """The delta, the base rows and the retention sums for one write."""

    def __init__(self, ad, names: dict, geo: dict, base_fh, offsets: dict, scale: float):
        self.ad, self.names, self.geo, self.fh, self.offsets, self.scale = ad, names, geo, base_fh, offsets, scale
        self.code = _code()
        self.sums = {}

    def delta(self, lay: int, e: int, which: str):
        """fp32 ``s * B[e] @ A[e]``, ``[out, in]`` as the weight is (ExpertsLoRA: ``A[e]`` is ``[r, in]``, ``B[e]``
        ``[out, r]``)."""
        a_name, b_name = (f"{self.names[lay]}.{which}_lora_{ab}" for ab in ("A", "B"))
        A, B = self.ad.tensors[a_name][e].float(), self.ad.tensors[b_name][e].float()
        return self.scale * (B @ A)

    def base(self, lay: int, e: int, which: str) -> tuple:
        g = self.geo[which]
        off = self.offsets[(lay, e)]
        return _read(self.fh, off, g["blocks"]), _read(self.fh, off, g["absmax"])

    def row(self, lay: int, e: int, which: str, S, what: str) -> tuple:
        """The merged ``(blocks, absmax)`` of one projection from fp32 ``S``, after proving S re-quantizes to the base
        row exactly; the retention sums accumulate D = deq(merged) - deq(base)."""
        import torch

        n, k = self.geo[which]["shape"]
        if tuple(S.shape) != (n, k):
            raise MergeRefused(f"layer {lay} expert {e} {which}: {what} is {tuple(S.shape)}, the arena's {(n, k)}")
        rb, ra = self.base(lay, e, which)
        if _quantize(S) != (rb, ra):
            raise MergeRefused(f"layer {lay} expert {e} {which}: {what}, re-quantized with no delta, does not "
                               f"reproduce the base arena's row; the writer will not merge into a base it cannot "
                               f"reproduce")
        d = self.delta(lay, e, which)
        qb, qa = _quantize(S + d)
        D = (_deq(qb, qa, n, k, self.code) - _deq(rb, ra, n, k, self.code)).double()
        d = d.double()
        acc = self.sums.setdefault((lay, which), {"dot": 0.0, "dd": 0.0, "DD": 0.0, "ee": 0.0, "changed": 0,
                                                  "numel": 0})
        acc["dot"] += float((D * d).sum())
        acc["dd"] += float(d.pow(2).sum())
        acc["DD"] += float(D.pow(2).sum())
        acc["ee"] += float((D - d).pow(2).sum())
        acc["changed"] += int(torch.count_nonzero(D))
        acc["numel"] += D.numel()
        return qb, qa


def _summary(rows) -> dict:
    """D0's Step-1 summary over ``rows``: rho_bar = sum<D,dW> / sum|dW|^2 (energy-weighted), c_bar the cosine, f_bar
    the share of changed values."""
    s = {k: sum(r[k] for r in rows) for k in ("dot", "dd", "DD", "ee", "changed", "numel")}
    return {"rho_bar": s["dot"] / s["dd"] if s["dd"] > 0 else None,
            "c_bar": s["dot"] / math.sqrt(s["DD"] * s["dd"]) if s["DD"] > 0 and s["dd"] > 0 else None,
            "f_bar": s["changed"] / s["numel"] if s["numel"] else None,
            "e_bar": math.sqrt(s["ee"] / s["dd"]) if s["dd"] > 0 else None}


def _retention(merge: _Merge, source: str) -> dict:
    rows = [{"layer": lay, "which": which, **v} for (lay, which), v in sorted(merge.sums.items())]
    return {"scheme": "d0-step1/1", "source": source,
            "definition": "D = deq(merged row) - deq(base row), per projection; rho_bar = sum<D,dW>/sum|dW|^2, "
                          "c_bar = sum<D,dW>/sqrt(sum|D|^2 sum|dW|^2), f_bar = changed/numel (PREREG-d0 Step 1)",
            "summary": {"all": _summary(rows), **{w: _summary([r for r in rows if r["which"] == w]) for w in WHICH}},
            "rows": rows}


def write(base_arena: str, adapter_dir: str, out: str, *, source: str = "arena", checkpoint: str | None = None,
          log=print) -> dict:
    """Write the merged arena ``out`` (with ``out.index.json``, ``out.manifest.json`` and ``out.retention.json``) from
    ``base_arena`` and the adapter at ``adapter_dir``; return its ``merge`` block. ``source="checkpoint"`` needs
    ``checkpoint``, the HF snapshot directory the base was baked from (its commit must be the base's). Refuses
    (:class:`MergeRefused`, or the verifier's refusal 7) before writing anything it cannot prove."""
    from .adapter_dir import arena_source, verify_adapter_dir
    from .adapter_schema import arena_expert_fingerprint

    if source not in SOURCES:
        raise MergeRefused(f"source must be one of {SOURCES}, got {source!r}")
    if (source == "checkpoint") != bool(checkpoint):
        raise MergeRefused("source='checkpoint' needs checkpoint=<the HF snapshot>, and only it takes one")
    finals = [out + s for s in ("", ".index.json", ".manifest.json", RETENTION)]
    if any(os.path.lexists(p) for p in finals):
        raise MergeRefused(f"{out}: refusing to overwrite an existing arena or sidecar")
    ad = verify_adapter_dir(adapter_dir)
    expected_fp, trained_base = _trained_base(ad)
    scale = _scale(ad)
    index_path = base_arena + ".index.json"
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    geo = _check_base(index, base_arena)
    _repo, base_rev = arena_source(index)
    if base_rev and ad.revision and base_rev != ad.revision:
        raise MergeRefused(f"the base arena was baked from {base_rev}, not the adapter's base {ad.revision}")
    offsets = {(int(lay), int(e)): int(off) for lay, e, off in index["rows"]}
    names = _stack_names(ad, {lay for lay, _ in offsets})
    _check_tensors(ad, names, geo, int(index["n_experts_per_layer"]))
    t0 = time.time()
    base_fp = arena_expert_fingerprint(base_arena, _stand_ins(names, index))
    if base_fp["sha256"] != expected_fp:
        raise MergeRefused(f"the base arena's qfp-v1 {base_fp['sha256'][:16]} is not the adapter's recorded expert "
                           f"fingerprint {expected_fp[:16]} ({trained_base}): it is not the base this adapter was "
                           f"trained on")
    log(f"merged arena: base verified (qfp-v1 {base_fp['sha256'][:16]}, {trained_base}), source={source}, "
        f"scale {scale:g}, {len(offsets)} rows")
    tmp = out + ".tmp"
    tmps = [tmp + s for s in ("", ".index.json", ".manifest.json")] + [out + RETENTION + ".tmp"]
    for p in tmps:
        if os.path.lexists(p):
            os.remove(p)
    try:
        block = _write(base_arena, index, index_path, geo, offsets, names, ad, scale, base_fp, trained_base,
                       base_rev, source, checkpoint, tmp, tmps, log)
    except BaseException:
        for p in tmps:
            if os.path.lexists(p):
                os.remove(p)
        raise
    for src, dst in zip(tmps, finals):
        os.replace(src, dst)
    log(f"merged arena written: {out} (merged qfp-v1 {block['merged_qfp_v1'][:16]}, rho_bar "
        f"{block['retention_summary']['all']['rho_bar']}, {time.time() - t0:.0f} s)")
    return block


def _write(base_arena, index, index_path, geo, offsets, names, ad, scale, base_fp, trained_base, base_rev, source,
           checkpoint, tmp, tmps, log) -> dict:
    """The rows, the index, the manifest and the receipt, at the temporary paths ``tmps``; the caller renames them."""
    from .adapter_dir import arena_source
    from .adapter_schema import arena_expert_fingerprint

    with open(base_arena, "rb") as base_fh:
        merge = _Merge(ad, names, geo, base_fh, offsets, scale)
        if source == "arena":
            man_rows = []
            with open(tmp, "wb") as dst:
                for lay, e, off in index["rows"]:
                    segs = []
                    for which in WHICH:
                        n, k = geo[which]["shape"]
                        rb, ra = merge.base(lay, e, which)
                        qb, qa = merge.row(lay, e, which, _deq(rb, ra, n, k, merge.code), "the base arena's row")
                        for seg, raw in ((geo[which]["blocks"], qb), (geo[which]["absmax"], qa)):
                            dst.seek(off + int(seg["seg_off"]))
                            dst.write(raw)
                            segs.append({"suffix": seg["suffix"], "sha256": hashlib.sha256(raw).hexdigest()})
                    man_rows.append({"layer": lay, "expert": e, "offset": off, "segments": segs})
                dst.flush()
                dst.truncate(int(index["arena_bytes"]))
                os.fsync(dst.fileno())
            new_index = dict(index)
            manifest = {"magic": index.get("magic"), "version": index.get("version"), "algo": "sha256",
                        "snapshot": index.get("snapshot"), "rows": man_rows}
        else:
            new_index, manifest = _bake_from_checkpoint(checkpoint, tmp, index, merge, log)
            _src_repo, src_rev = arena_source(new_index)
            if src_rev != base_rev:
                raise MergeRefused(f"the checkpoint {checkpoint} is commit {src_rev}, the base arena's {base_rev}")
    import bitsandbytes
    import torch

    retention = _retention(merge, source)
    blob = json.dumps(retention, indent=1, sort_keys=True).encode("utf-8")
    block = {"writer": WRITER, "writer_version": WRITER_VERSION, "source": source,
             "source_revision": base_rev, "trained_base": trained_base,
             "base_index_sha256": _sha256_file(index_path), "base_qfp_v1": base_fp["sha256"], "merged_qfp_v1": None,
             "adapter_id": ad.adapter_id, "adapter_manifest_sha256": ad.manifest_sha256,
             "expert_tensors_sha256": expert_tensors_sha256(ad), "scale": scale,
             "quantizer": {"kind": "bnb.quantize_4bit", "quant_type": "nf4", "blocksize": BLOCK,
                           "compress_statistics": False, "dequantize": "fp32 codebook x fp32 absmax",
                           "bnb": bitsandbytes.__version__, "torch": torch.__version__},
             "retention_sha256": hashlib.sha256(blob).hexdigest(),
             "retention_summary": retention["summary"],
             "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    new_index.update(bake_mode=MERGED_BAKE_MODE, merge=block)
    with open(tmps[1], "w", encoding="utf-8") as f:
        json.dump(new_index, f)
    block["merged_qfp_v1"] = arena_expert_fingerprint(tmp, _stand_ins(names, index))["sha256"]
    new_index["merge"] = block
    manifest.update(bake_mode=MERGED_BAKE_MODE, merge=block)
    with open(tmps[1], "w", encoding="utf-8") as f:
        json.dump(new_index, f)
    with open(tmps[2], "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(tmps[3], "wb") as f:
        f.write(blob)
    return block


def _bake_from_checkpoint(checkpoint: str, tmp: str, index: dict, merge: _Merge, log) -> tuple:
    """``source="checkpoint"``: grouped-nf4-gemm's own ``bake_nf4`` reads the checkpoint and lays out the rows; its
    quantizer is this writer's (fp32 W, the proof against the base row, plus the delta). The bake calls it per row in
    its row order, gate_up then down, and its geometry must come out equal to the base's."""
    import nvme_bake_nf4 as nb
    import torch

    calls = iter([(int(lay), int(e), which) for lay, e, _off in index["rows"] for which in WHICH])

    def quantize(w_bf16):
        nxt = next(calls, None)
        if nxt is None:
            raise MergeRefused("the bake asked for more projections than the base arena holds")
        lay, e, which = nxt
        n, k = merge.geo[which]["shape"]
        qb, qa = merge.row(lay, e, which, w_bf16.float(), "the checkpoint's weight")
        return (torch.frombuffer(bytearray(qb), dtype=torch.uint8).view(n, k // 2),
                torch.frombuffer(bytearray(qa), dtype=torch.float32).view(n, k // BLOCK))

    new_index = nb.bake_nf4(checkpoint, tmp, prefix=index["prefix"], align=int(index["align"]), quantize_fn=quantize,
                            log=lambda *a, **k: None)
    if next(calls, None) is not None:
        raise MergeRefused("the bake wrote fewer projections than the base arena holds")
    for key in ("prefix", "align", "row_bytes", "row_stride", "n_layers", "n_experts_per_layer", "segments", "rows",
                "arena_bytes", "model_dims"):
        if new_index.get(key) != index.get(key):
            raise MergeRefused(f"the checkpoint bake's {key} differs from the base arena's")
    with open(tmp + ".manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    return new_index, manifest
