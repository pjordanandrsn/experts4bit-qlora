# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Hash-pinned calibrated int4 pack artifacts (#405).

A licence is a property of pack BYTES, not of the calibration recipe.
This module serialises the packed expert tensors + scales, writes a
canonical JSON manifest, and computes the root ``pack_fingerprint`` as
``sha256:<64 hex>`` over the ordered ``(path, size, sha256)`` payload
tuples. A licensed load that is given an expected fingerprint verifies
every payload and **refuses** a mismatch; it never rebuilds from the
recipe.

Counts (GPTQ vs RTN) stay diagnostics. Identity is the fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
LAYOUT = "int4_b32.gate_first.v1"
MANIFEST_NAME = "manifest.json"
PAYLOAD_DIR = "payloads"
FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
PROVENANCE_ATTR = "_e4b_pack_provenance"


class PackManifestError(RuntimeError):
    """Artifact missing, corrupt, or identity mismatch -- never a silent rebuild."""


def format_fingerprint(digest: str) -> str:
    d = digest.lower()
    if len(d) != 64 or any(c not in "0123456789abcdef" for c in d):
        raise PackManifestError(f"not a sha256 hex digest: {digest!r}")
    return f"sha256:{d}"


def parse_fingerprint(value: str) -> str:
    if not isinstance(value, str) or not FINGERPRINT_RE.fullmatch(value):
        raise PackManifestError(
            f"pack_fingerprint {value!r} is not sha256:<64 lowercase hex>")
    return value[7:]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_canonical(obj: Any) -> str:
    blob = json.dumps(obj, separators=(",", ":"), sort_keys=True,
                      ensure_ascii=True).encode("utf-8")
    return format_fingerprint(sha256_bytes(blob))


def tensor_payload_bytes(t) -> bytes:
    """Canonical bytes for a CPU tensor: JSON header + native tobytes().

    Independent of ``torch.save`` pickle, so a fingerprint is stable across
    torch versions for the same dtype/shape/values.
    """
    import torch
    t = t.detach().cpu().contiguous()
    dtype = str(t.dtype).replace("torch.", "")
    header = json.dumps({"dtype": dtype, "shape": list(t.shape)},
                        separators=(",", ":")).encode("utf-8")
    if t.dtype == torch.bfloat16:
        body = t.view(torch.uint8).numpy().tobytes()
    else:
        body = t.numpy().tobytes()
    return struct.pack("<I", len(header)) + header + body


def tensor_from_payload_bytes(data: bytes):
    import torch
    import numpy as np
    if len(data) < 4:
        raise PackManifestError("tensor payload truncated")
    (n,) = struct.unpack_from("<I", data, 0)
    header = json.loads(data[4:4 + n].decode("utf-8"))
    body = data[4 + n:]
    dtype_name = header["dtype"]
    shape = tuple(header["shape"])
    if dtype_name == "bfloat16":
        t = torch.empty(shape, dtype=torch.bfloat16)
        t.view(torch.uint8).view(-1).copy_(
            torch.frombuffer(bytearray(body), dtype=torch.uint8))
        return t
    np_dtype = getattr(np, dtype_name, None)
    torch_dtype = getattr(torch, dtype_name, None)
    if np_dtype is None or torch_dtype is None:
        raise PackManifestError(f"unsupported tensor dtype {dtype_name!r}")
    arr = np.frombuffer(body, dtype=np_dtype).reshape(shape)
    return torch.from_numpy(arr.copy())


def payload_entry(relpath: str, data: bytes) -> dict:
    return {"path": relpath.replace("\\", "/"),
            "size": len(data),
            "sha256": sha256_bytes(data)}


def ordered_payload_tuples(payloads: Sequence[Mapping]) -> list[tuple]:
    rows = []
    for p in payloads:
        path, size, digest = p["path"], int(p["size"]), str(p["sha256"])
        if (not path or path.startswith("/") or "\\" in path
                or any(part in ("", ".", "..") for part in path.split("/"))):
            raise PackManifestError(f"payload path is not a bare relative file: {path!r}")
        rows.append((path, size, digest.lower()))
    rows.sort(key=lambda r: r[0])
    return rows


def compute_pack_fingerprint(payloads: Sequence[Mapping]) -> str:
    """Root identity: sha256 of the canonical ordered (path, size, sha256) list."""
    blob = json.dumps(ordered_payload_tuples(payloads),
                      separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return format_fingerprint(sha256_bytes(blob))


def require_artifact_for_licensed_load(artifact_dir, expected_fingerprint) -> None:
    """A licensed path given an expected fingerprint never falls back to the recipe."""
    if expected_fingerprint is None:
        return
    parse_fingerprint(expected_fingerprint)
    if not artifact_dir:
        raise PackManifestError(
            "expected_fingerprint is set; refusing to rebuild the pack from the "
            "calibration recipe -- pass artifact_dir of the licensed bytes")


def write_json(path: str | os.PathLike, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def load_manifest(artifact_dir: str | os.PathLike) -> dict:
    root = Path(artifact_dir)
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise PackManifestError(f"missing {MANIFEST_NAME} under {root}")
    try:
        man = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PackManifestError(f"manifest is not JSON: {e}") from e
    if not isinstance(man, dict):
        raise PackManifestError("manifest is not an object")
    return man


def verify_artifact(artifact_dir: str | os.PathLike, *,
                    expected_fingerprint: str | None = None,
                    expected_model_revision: str | None = None,
                    expected_layout: str | None = LAYOUT) -> dict:
    """Hash every payload; refuse corruption, missing files, and identity drift."""
    root = Path(artifact_dir)
    man = load_manifest(root)
    if int(man.get("schema_version", -1)) != SCHEMA_VERSION:
        raise PackManifestError(
            f"unsupported pack-manifest schema_version {man.get('schema_version')!r} "
            f"(this code reads {SCHEMA_VERSION})")
    payloads = man.get("payloads")
    if not isinstance(payloads, list) or not payloads:
        raise PackManifestError("manifest payloads[] is empty")
    for p in payloads:
        rel = p.get("path")
        if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in rel.split("/"):
            raise PackManifestError(f"payload path is not a bare relative file: {rel!r}")
        fp = root / rel
        if not fp.is_file():
            raise PackManifestError(f"missing payload {rel}")
        size = fp.stat().st_size
        digest = sha256_file(fp)
        if size != int(p["size"]) or digest != str(p["sha256"]).lower():
            raise PackManifestError(
                f"payload {rel} does not match the manifest "
                f"(size {size} vs {p['size']}, sha256 {digest} vs {p['sha256']})")
    computed = compute_pack_fingerprint(payloads)
    recorded = man.get("pack_fingerprint")
    if recorded is not None and recorded != computed:
        raise PackManifestError(
            f"manifest pack_fingerprint {recorded} != recomputed {computed}")
    if expected_fingerprint is not None:
        parse_fingerprint(expected_fingerprint)
        if computed != expected_fingerprint:
            raise PackManifestError(
                f"pack_fingerprint {computed} != expected {expected_fingerprint} "
                "-- refusing to load; not rebuilding from the recipe")
    if expected_layout is not None and man.get("layout") != expected_layout:
        raise PackManifestError(
            f"layout {man.get('layout')!r} != expected {expected_layout!r}")
    if expected_model_revision is not None:
        got = man.get("model_revision")
        if got != expected_model_revision:
            raise PackManifestError(
                f"model_revision {got!r} != expected {expected_model_revision!r}")
    man = dict(man)
    man["pack_fingerprint"] = computed
    return man


def write_artifact(artifact_dir: str | os.PathLike, *,
                   tensors: Mapping[tuple, Any],
                   meta: Mapping[str, Any]) -> dict:
    """Write payload files + manifest. ``tensors`` keys are (layer, role, kind)."""
    root = Path(artifact_dir)
    pay = root / PAYLOAD_DIR
    pay.mkdir(parents=True, exist_ok=True)
    payloads = []
    for layer, role, kind in sorted(tensors):
        rel = f"{PAYLOAD_DIR}/layer_{int(layer):04d}_{role}_{kind}.bin"
        data = tensor_payload_bytes(tensors[(layer, role, kind)])
        (root / rel).write_bytes(data)
        payloads.append(payload_entry(rel, data))
    fp = compute_pack_fingerprint(payloads)
    man = {
        "schema_version": SCHEMA_VERSION,
        "layout": LAYOUT,
        "pack_fingerprint": fp,
        "payloads": payloads,
        **{k: v for k, v in meta.items() if v is not None},
    }
    write_json(root / MANIFEST_NAME, man)
    return man


def load_payload_tensors(artifact_dir: str | os.PathLike,
                         manifest: Mapping[str, Any]) -> dict[tuple, Any]:
    root = Path(artifact_dir)
    out = {}
    for p in manifest["payloads"]:
        rel = p["path"]
        name = Path(rel).name
        # layer_0007_gu_packed.bin
        m = re.fullmatch(r"layer_(\d+)_(gu|dn)_(packed|scales)\.bin", name)
        if not m:
            raise PackManifestError(f"payload name is not layer_NNNN_{{gu,dn}}_{{packed,scales}}.bin: {rel}")
        layer, role, kind = int(m.group(1)), m.group(2), m.group(3)
        out[(layer, role, kind)] = tensor_from_payload_bytes((root / rel).read_bytes())
    return out


def method_map_hash(entries: Sequence[Mapping]) -> str:
    rows = [{"layer": int(e["layer"]), "expert": int(e["expert"]),
             "role": e["role"], "method": e["method"]} for e in entries]
    rows.sort(key=lambda r: (r["layer"], r["role"], r["expert"]))
    return hash_canonical(rows)


def row_count_vector_hash(entries: Sequence[Mapping]) -> str:
    rows = [{"layer": int(e["layer"]), "expert": int(e["expert"]),
             "rows": int(e["rows"])} for e in entries]
    rows.sort(key=lambda r: (r["layer"], r["expert"]))
    return hash_canonical(rows)


def token_stream_sha(batches: Iterable) -> str | None:
    h = hashlib.sha256()
    n = 0
    for ids in batches:
        n += 1
        t = ids.detach().cpu().contiguous()
        h.update(tensor_payload_bytes(t))
    if n == 0:
        return None
    return format_fingerprint(h.hexdigest())


def toolchain_record() -> dict:
    rec = {}
    try:
        import torch
        rec["torch"] = getattr(torch, "__version__", None)
        rec["cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        rec["torch"] = None
        rec["cuda"] = None
    try:
        import triton
        rec["triton"] = getattr(triton, "__version__", None)
    except Exception:
        rec["triton"] = None
    rec["e4b_int4_gptq_device"] = os.environ.get("E4B_INT4_GPTQ_DEVICE")
    rec["e4b_int4_gptq_damp"] = os.environ.get("E4B_INT4_GPTQ_DAMP", "0.01")
    rec["e4b_int4_hessian_budget_gb"] = os.environ.get("E4B_INT4_HESSIAN_BUDGET_GB")
    return rec


def provenance_record(*, pack_fingerprint: str, component_hashes: Sequence[Mapping],
                      method_map_hash_value: str | None = None,
                      row_count_vector_hash_value: str | None = None,
                      calibration_token_stream_sha: str | None = None,
                      model_id: str | None = None,
                      model_revision: str | None = None,
                      min_rows: int | None = None,
                      damping: float | None = None,
                      solve_device: str | None = None,
                      layout: str = LAYOUT,
                      extra: Mapping | None = None) -> dict:
    rec = {
        "pack_fingerprint": pack_fingerprint,
        "pack_layout": layout,
        "component_hashes": list(component_hashes),
        "method_map_hash": method_map_hash_value,
        "row_count_vector_hash": row_count_vector_hash_value,
        "calibration_token_stream_sha": calibration_token_stream_sha,
        "model": model_id,
        "model_revision": model_revision,
        "min_rows": min_rows,
        "damping": damping,
        "solve_device": solve_device,
        "toolchain": toolchain_record(),
    }
    if extra:
        rec.update(extra)
    return rec


def attach_provenance(model, record: Mapping) -> None:
    setattr(model, PROVENANCE_ATTR, dict(record))


def provenance_from_model(model) -> dict | None:
    rec = getattr(model, PROVENANCE_ATTR, None)
    return dict(rec) if rec else None


def merge_provenance_into_receipt(rep: dict, model) -> dict:
    rec = provenance_from_model(model)
    if rec:
        for k, v in rec.items():
            if k not in rep:
                rep[k] = v
    return rep


def fingerprint_mismatch_reason(observed, expected) -> str | None:
    """Reducer helper: exact identity, or None when there is no expected hash.

    Legacy receipts without ``pack_fingerprint`` keep their count-banner VOID
    rule; this only fires when the lane names an expected fingerprint.
    """
    if not expected:
        return None
    if not observed:
        return "no pack_fingerprint in receipt (expected licensed bytes)"
    if observed != expected:
        return f"pack_fingerprint {observed} != expected {expected}"
    return None
