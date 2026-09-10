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

Since #530 a calibrated artifact also carries the gptq/rtn ASSIGNMENT
per (layer, expert, role) plus the routed-row counts, as a hashed
payload (``payloads/assignment.json``). ``read_assignment`` returns it;
``int4_experts.enable_serve_experts_int4(assignment=...)`` honours it.
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
IDENTITY_PATH = f"{PAYLOAD_DIR}/identity.json"
# The per-(layer, expert, role) gptq/rtn decision and the routed-row counts it was made from,
# written as a HASHED payload so the root fingerprint covers them (#530). The decision is a
# threshold (rows >= min_rows) on a quantity at the router-flip noise floor, so it does not
# reproduce across boxes; recording it lets a re-pack HONOUR the licensed split instead of
# re-litigating it. It fixes the classification only -- GPTQ output still depends on the
# Hessian, which routing also perturbs -- so bytes reproduce only via the artifact.
ASSIGNMENT_PATH = f"{PAYLOAD_DIR}/assignment.json"
ROLES = ("gu", "dn")
METHODS = ("gptq", "rtn")
# The manifest fields that name WHICH checkpoint and layout the bytes belong to. They are
# written a second time as a hashed payload (IDENTITY_PATH) so the root fingerprint covers
# them; verify_artifact refuses a manifest whose top-level copy disagrees with the hashed one.
IDENTITY_KEYS = ("schema_version", "layout", "model_id", "model_revision", "layers")
BLOCK = 32


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
            "calibration recipe -- pass artifact_dir of the licensed bytes "
            "(None and the empty string are both refused)")


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
    # Identity is hashed: the manifest's top-level copy must equal the payload the fingerprint covers.
    identity = read_identity_payload(root, payloads)
    for key in IDENTITY_KEYS:
        if man.get(key) != identity.get(key):
            raise PackManifestError(
                f"manifest {key!r} = {man.get(key)!r} differs from the hashed identity payload "
                f"({identity.get(key)!r}) -- the manifest was edited after the bytes were fingerprinted; refusing")
    if any(p.get("path") == ASSIGNMENT_PATH for p in payloads):
        rec = read_assignment(root)
        if man.get("method_map_hash") not in (None, rec["method_map_hash"]):
            raise PackManifestError(
                f"manifest method_map_hash {man.get('method_map_hash')} != the hashed assignment "
                f"payload's {rec['method_map_hash']} -- the manifest was edited after the decision "
                "was fingerprinted; refusing")
    man = dict(man)
    man["pack_fingerprint"] = computed
    return man


def identity_payload_bytes(man: Mapping[str, Any]) -> bytes:
    """Canonical bytes of the identity fields, exactly as the manifest carries them (absent stays absent)."""
    ident = {k: man[k] for k in IDENTITY_KEYS if k in man and man[k] is not None}
    return json.dumps(ident, separators=(",", ":"), sort_keys=True, ensure_ascii=True).encode("utf-8")


def canonical_method_map(entries: Sequence[Mapping]) -> list[dict]:
    """Sorted, typed, validated rows -- the one shape method_map_hash and the payload share."""
    rows = []
    for e in entries:
        role, method = e["role"], e["method"]
        if role not in ROLES or method not in METHODS:
            raise PackManifestError(f"assignment row has role={role!r} method={method!r}")
        rows.append({"layer": int(e["layer"]), "expert": int(e["expert"]),
                     "role": role, "method": method})
    rows.sort(key=lambda r: (r["layer"], r["role"], r["expert"]))
    return rows


def canonical_row_counts(entries: Sequence[Mapping]) -> list[dict]:
    rows = [{"layer": int(e["layer"]), "expert": int(e["expert"]), "rows": int(e["rows"])}
            for e in entries]
    rows.sort(key=lambda r: (r["layer"], r["expert"]))
    return rows


def assignment_payload_bytes(method_map: Sequence[Mapping],
                             row_counts: Sequence[Mapping] | None,
                             min_rows: int | None) -> bytes:
    """Canonical bytes of the recorded decision. ``min_rows`` rides along as the rule that
    CREATED the assignment; a reader never re-applies it."""
    obj = {"method_map": canonical_method_map(method_map),
           "row_counts": canonical_row_counts(row_counts or []),
           "min_rows": None if min_rows is None else int(min_rows)}
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=True).encode("utf-8")


def read_assignment(source: str | os.PathLike) -> dict:
    """The recorded assignment from an artifact dir (its hashed payload) or a bare
    ``assignment.json``. Returns ``{"method_map", "row_counts", "min_rows",
    "method_map_hash"}``; refuses an empty or malformed one. Reading does NOT verify the
    artifact's fingerprint -- callers that need the licensed bytes use verify_artifact."""
    path = Path(source)
    if path.is_dir():
        path = path / ASSIGNMENT_PATH
    if not path.is_file():
        raise PackManifestError(f"no assignment payload at {path}")
    try:
        obj = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise PackManifestError(f"assignment payload unreadable: {e}") from e
    if not isinstance(obj, dict) or not isinstance(obj.get("method_map"), list) or not obj["method_map"]:
        raise PackManifestError("assignment payload has no method_map rows")
    mm = canonical_method_map(obj["method_map"])
    rc = canonical_row_counts(obj.get("row_counts") or [])
    return {"method_map": mm, "row_counts": rc, "min_rows": obj.get("min_rows"),
            "method_map_hash": method_map_hash(mm)}


def assignment_index(method_map: Sequence[Mapping]) -> dict[tuple[int, int, str], str]:
    """``(layer, expert, role) -> method``; a duplicate key is a refusal, not a last-wins."""
    idx: dict[tuple[int, int, str], str] = {}
    for r in canonical_method_map(method_map):
        key = (r["layer"], r["expert"], r["role"])
        if key in idx and idx[key] != r["method"]:
            raise PackManifestError(f"assignment names {key} twice with different methods")
        idx[key] = r["method"]
    return idx


def read_identity_payload(root: Path, payloads: Sequence[Mapping]) -> dict:
    """The hashed identity payload. Its absence is a refusal: no artifact this code reads was written without one."""
    if not any(p.get("path") == IDENTITY_PATH for p in payloads):
        raise PackManifestError(
            f"artifact has no hashed identity payload ({IDENTITY_PATH}); the root fingerprint "
            "does not cover model_revision / layout / layers -- refusing")
    try:
        ident = json.loads((root / IDENTITY_PATH).read_bytes().decode("utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise PackManifestError(f"identity payload unreadable: {e}") from e
    if not isinstance(ident, dict):
        raise PackManifestError("identity payload is not an object")
    return ident


def int4_store_dims(packed, scales) -> tuple[int, int]:
    """(N, K) from the bytes of one int4_b32 store: ``packed [..., N, K//2] uint8``,
    ``scales [..., N, K//32]`` (leading dim = experts). The manifest never decides N/K;
    it may only agree with the payload."""
    if getattr(packed, "dim", lambda: 0)() < 2 or getattr(scales, "dim", lambda: 0)() < 2:
        raise PackManifestError("packed/scales payloads must be at least 2-D ([..., N, K//2] / [..., N, K//32])")
    n, k_half = int(packed.shape[-2]), int(packed.shape[-1])
    k = k_half * 2
    n_s, k_blocks = int(scales.shape[-2]), int(scales.shape[-1])
    if n_s != n or k_blocks * BLOCK != k:
        raise PackManifestError(
            f"packed and scales disagree: packed implies N={n} K={k}, scales imply N={n_s} K={k_blocks * BLOCK}")
    if k % BLOCK:
        raise PackManifestError(f"K={k} from the payload is not a multiple of {BLOCK}")
    return n, k


def check_manifest_dims(layer, role: str, meta: Mapping[str, Any], n: int, k: int) -> None:
    """``layers[]`` is a cross-check of the hashed bytes, never their source of truth."""
    try:
        mn, mk = int(meta[f"N{role}"]), int(meta[f"K{role}"])
    except (KeyError, TypeError, ValueError) as e:
        raise PackManifestError(f"layer {layer} {role}: manifest layers[] lacks N/K: {e}") from e
    if (mn, mk) != (n, k):
        raise PackManifestError(
            f"layer {layer} {role}: manifest layers[] says N={mn} K={mk} but the hashed payload "
            f"bytes are N={n} K={k} -- refusing (layers[] is outside the tensor bytes)")


def require_model_revision(revision, *, allow_unknown: bool = False) -> tuple[str | None, bool]:
    """A pack without a pinned checkpoint revision can never be a licensed load.

    Returns ``(revision, missing)``. Refuses when unknown unless ``allow_unknown``; then the
    caller must record ``model_revision_missing: true`` so the observation pack says so itself.
    """
    if revision:
        return str(revision), False
    if not allow_unknown:
        raise PackManifestError(
            "model_revision is unknown (config._commit_hash missing): an artifact without a pinned "
            "checkpoint revision can never become a licensed load -- refusing to dump it. Pass "
            "allow_unknown_revision=True (or E4B_INT4_DUMP_ALLOW_UNKNOWN_REVISION=1) to write an "
            "observation pack marked model_revision_missing.")
    return None, True


def write_artifact(artifact_dir: str | os.PathLike, *,
                   tensors: Mapping[tuple, Any],
                   meta: Mapping[str, Any],
                   assignment: Mapping[str, Any] | None = None) -> dict:
    """Write payload files + manifest. ``tensors`` keys are (layer, role, kind).

    ``assignment`` = ``{"method_map": [...], "row_counts": [...]}`` is written as a hashed
    payload (ASSIGNMENT_PATH) so the root fingerprint covers the recorded gptq/rtn decision;
    the manifest's ``method_map_hash`` is then derived from it, never taken on trust."""
    root = Path(artifact_dir)
    pay = root / PAYLOAD_DIR
    pay.mkdir(parents=True, exist_ok=True)
    payloads = []
    for layer, role, kind in sorted(tensors):
        rel = f"{PAYLOAD_DIR}/layer_{int(layer):04d}_{role}_{kind}.bin"
        data = tensor_payload_bytes(tensors[(layer, role, kind)])
        (root / rel).write_bytes(data)
        payloads.append(payload_entry(rel, data))
    meta = dict(meta)
    if assignment is not None:
        mm = assignment.get("method_map") or []
        if not mm:
            raise PackManifestError("assignment given but its method_map is empty")
        blob = assignment_payload_bytes(mm, assignment.get("row_counts"), meta.get("min_rows"))
        (root / ASSIGNMENT_PATH).write_bytes(blob)
        payloads.append(payload_entry(ASSIGNMENT_PATH, blob))
        # the manifest copy is DERIVED from the hashed payload
        meta["method_map_hash"] = method_map_hash(mm)
        if assignment.get("row_counts"):
            meta["row_count_vector_hash"] = row_count_vector_hash(assignment["row_counts"])
    head = {"schema_version": SCHEMA_VERSION, "layout": LAYOUT,
            **{k: v for k, v in meta.items() if v is not None}}
    ident = identity_payload_bytes(head)
    (root / IDENTITY_PATH).write_bytes(ident)
    payloads.append(payload_entry(IDENTITY_PATH, ident))
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
        if rel in (IDENTITY_PATH, ASSIGNMENT_PATH):
            continue  # hashed identity / recorded decision, not tensors
        name = Path(rel).name
        # layer_0007_gu_packed.bin
        m = re.fullmatch(r"layer_(\d+)_(gu|dn)_(packed|scales)\.bin", name)
        if not m:
            raise PackManifestError(f"payload name is not layer_NNNN_{{gu,dn}}_{{packed,scales}}.bin: {rel}")
        layer, role, kind = int(m.group(1)), m.group(2), m.group(3)
        out[(layer, role, kind)] = tensor_from_payload_bytes((root / rel).read_bytes())
    return out


def method_map_hash(entries: Sequence[Mapping]) -> str:
    return hash_canonical(canonical_method_map(entries))


def row_count_vector_hash(entries: Sequence[Mapping]) -> str:
    return hash_canonical(canonical_row_counts(entries))


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
