"""Shared, torch-free helpers for the train/query storage-mode matrix scripts.

Vocabulary (see docs/MODE_DECOUPLED_ADAPTERS.md): a *mode label* names a storage scheme plus an
offload flag (``nf4``, ``nf4-offload``, ``int8``, ...). An adapter is trained under a *train mode*
and evaluated under a *query mode*; when the two storage schemes differ that is a *storage-mode
mismatch* — allowed by the validation scripts, recorded explicitly, never silent.

Everything here is importable without torch/CUDA so the unit tests run on any host.
"""

import csv
import hashlib
import json
import os
from datetime import datetime, timezone

# Canonical storage schemes — mirrors experts4bit_qlora._SCHEME_BITS (kept literal here so this
# module stays torch-free; the scripts that build modules re-validate through the package).
STORAGE_MODES = ("nf4", "fp4", "int8", "fp8", "bf16", "fp16")
OFFLOAD_SUFFIX = "-offload"
METADATA_FILENAME = "expertsnbit_adapter_metadata.json"
METADATA_SCHEMA = 1

# Every result row carries at least these keys (validate_row enforces it, tests pin it).
REQUIRED_ROW_FIELDS = (
    "run_id",
    "base_model",
    "train_mode_label",
    "train_storage_mode",
    "train_offload",
    "query_mode_label",
    "query_storage_mode",
    "query_offload",
    "storage_mode_mismatch",
    "offload_mismatch",
    "status",
    "timestamp",
)
ROW_STATUSES = ("pass", "fail", "skip")


def parse_mode_label(label):
    """``'nf4-offload'`` -> ``('nf4', True)``; ``'int8'`` -> ``('int8', False)``.

    Case/whitespace-insensitive. Anything that does not normalize to a canonical storage scheme
    plus an optional ``-offload`` suffix raises — ambiguous names are not allowed.
    """
    if not isinstance(label, str):
        raise ValueError(f"mode label must be a string, got {type(label).__name__}")
    q = label.strip().lower()
    offload = q.endswith(OFFLOAD_SUFFIX)
    if offload:
        q = q[: -len(OFFLOAD_SUFFIX)]
    if q not in STORAGE_MODES:
        raise ValueError(
            f"unknown mode label {label!r}: expected one of {STORAGE_MODES} with an optional "
            f"'{OFFLOAD_SUFFIX}' suffix (e.g. 'nf4', 'nf4-offload')"
        )
    return q, offload


def mode_label(storage_mode: str, offload: bool) -> str:
    """Inverse of :func:`parse_mode_label` (canonical spelling)."""
    storage_mode, _ = parse_mode_label(storage_mode)
    return storage_mode + (OFFLOAD_SUFFIX if offload else "")


def parse_mode_list(csv_labels):
    """``'nf4,nf4-offload,int8'`` -> list of parsed ``(label, storage, offload)`` triples,
    de-duplicated in order, every label validated up front (a typo fails before any work)."""
    out, seen = [], set()
    for raw in csv_labels.split(","):
        raw = raw.strip()
        if not raw:
            continue
        storage, offload = parse_mode_label(raw)
        label = mode_label(storage, offload)
        if label not in seen:
            seen.add(label)
            out.append((label, storage, offload))
    if not out:
        raise ValueError(f"no modes parsed from {csv_labels!r}")
    return out


def compute_mismatch(train_meta, query_storage: str, query_offload: bool):
    """Compare an adapter's train-mode provenance against a query mode.

    Returns ``{"storage_mode_mismatch", "offload_mismatch", "warnings"}``. Unknown provenance
    (missing/blank metadata) is itself a warning — storage mode is part of adapter provenance,
    and an adapter without it cannot be assumed same-mode.
    """
    warnings = []
    t_storage = (train_meta or {}).get("train_storage_mode")
    t_offload = (train_meta or {}).get("train_offload")
    if t_storage is None:
        warnings.append(
            "adapter has no train_storage_mode metadata: storage provenance unknown — treating "
            "as a storage-mode mismatch is not possible, but same-mode cannot be assumed either"
        )
        return {"storage_mode_mismatch": None, "offload_mismatch": None, "warnings": warnings}
    storage_mismatch = t_storage != query_storage
    offload_mismatch = bool(t_offload) != bool(query_offload)
    if storage_mismatch:
        warnings.append(
            f"storage-mode mismatch: adapter trained under {t_storage!r}, querying under "
            f"{query_storage!r} — cross-mode query is an empirical path, not a contract"
        )
    if offload_mismatch:
        warnings.append(
            f"offload mismatch: trained with offload={bool(t_offload)}, querying with "
            f"offload={query_offload} (same math by design; recorded for provenance)"
        )
    return {
        "storage_mode_mismatch": storage_mismatch,
        "offload_mismatch": offload_mismatch,
        "warnings": warnings,
    }


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_metadata(adapter_dir: str, meta: dict) -> str:
    """Write the sidecar metadata JSON beside the adapter files; returns its path."""
    meta = dict(meta)
    meta.setdefault("metadata_schema", METADATA_SCHEMA)
    meta.setdefault("timestamp", utc_now())
    path = os.path.join(adapter_dir, METADATA_FILENAME)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def read_metadata(adapter_dir: str):
    """Read the sidecar; returns ``None`` when absent (callers must warn, not crash)."""
    path = os.path.join(adapter_dir, METADATA_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def validate_row(row: dict) -> dict:
    """Minimal schema check for a result row; raises on a missing field or bad status."""
    missing = [k for k in REQUIRED_ROW_FIELDS if k not in row]
    if missing:
        raise ValueError(f"result row missing fields: {missing}")
    if row["status"] not in ROW_STATUSES:
        raise ValueError(f"result row status must be one of {ROW_STATUSES}, got {row['status']!r}")
    if row["status"] != "pass" and not row.get("skip_or_fail_reason"):
        raise ValueError(f"non-pass row must carry skip_or_fail_reason: {row}")
    return row


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------------------------
# Summarizer core (pure functions over result rows; the CLI wrapper lives in
# summarize_train_query_matrix.py). Language discipline: everything reported is "observed in
# this run" — these tables do not prove universal portability.
# ---------------------------------------------------------------------------------------------
def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def matrix_table(rows, value_key, nd=4):
    """Markdown table: train mode (rows) x query mode (columns), cell = ``value_key`` for pass
    rows, ``fail``/``skip`` markers otherwise."""
    trains = sorted({r["train_mode_label"] for r in rows})
    queries = sorted({r["query_mode_label"] for r in rows})
    cell = {}
    for r in rows:
        key = (r["train_mode_label"], r["query_mode_label"])
        cell[key] = _fmt(r.get(value_key), nd) if r["status"] == "pass" else r["status"].upper()
    lines = ["| train \\ query | " + " | ".join(queries) + " |", "|---" * (len(queries) + 1) + "|"]
    for t in trains:
        lines.append("| " + t + " | " + " | ".join(cell.get((t, q), "-") for q in queries) + " |")
    return "\n".join(lines)


def best_per(rows, group_key, value_key="eval_loss_with_adapter"):
    """For each value of ``group_key`` ('train_mode_label' or 'query_mode_label'), the pass row
    with the lowest ``value_key``."""
    best = {}
    for r in rows:
        if r["status"] != "pass" or r.get(value_key) is None:
            continue
        g = r[group_key]
        if g not in best or r[value_key] < best[g][value_key]:
            best[g] = r
    return best


def transfer_summary(rows):
    """Same-mode / upward / downward / offload-transfer / symmetry observations for this run.

    'Upward' = trained under a lower-fidelity storage scheme, queried under a higher-fidelity
    one (fidelity order per the test-pinned reconstruction chain); 'downward' the reverse.
    Returns a list of plain-English lines, each prefixed with its category.
    """
    fidelity = {"fp4": 0, "nf4": 1, "fp8": 2, "int8": 3, "bf16": 4, "fp16": 5}
    passed = [r for r in rows if r["status"] == "pass" and r.get("eval_loss_with_adapter") is not None]
    by_pair = {(r["train_mode_label"], r["query_mode_label"]): r for r in passed}

    def same_mode(train_label):
        # The same-mode row for an offload-trained adapter is the same storage RESIDENT query
        # unless an exact-label query exists (query offload legs are optional in the grid).
        if (train_label, train_label) in by_pair:
            return by_pair[(train_label, train_label)]
        storage, _ = parse_mode_label(train_label)
        return by_pair.get((train_label, storage))

    lines = []
    for t in sorted({r["train_mode_label"] for r in passed}):
        base = same_mode(t)
        if base is None:
            continue
        lines.append(
            f"same-mode: {t} -> {base['query_mode_label']}: eval {_fmt(base['eval_loss_with_adapter'])} "
            f"(base-no-adapter {_fmt(base.get('eval_loss_base_query_mode_no_adapter'))})"
        )
        t_storage, _ = parse_mode_label(t)
        for r in passed:
            if r["train_mode_label"] != t or r is base:
                continue
            q_storage, _ = parse_mode_label(r["query_mode_label"])
            if q_storage == t_storage:
                kind = "offload-transfer"
            elif fidelity[q_storage] > fidelity[t_storage]:
                kind = "upward"
            else:
                kind = "downward"
            delta = (
                r["eval_loss_with_adapter"] - base["eval_loss_with_adapter"]
                if base.get("eval_loss_with_adapter") is not None
                else None
            )
            lines.append(
                f"{kind}: {t} -> {r['query_mode_label']}: eval {_fmt(r['eval_loss_with_adapter'])} "
                f"({'+' if delta is not None and delta >= 0 else ''}{_fmt(delta)} vs same-mode)"
            )
    # Symmetry: A-trained-on-X-queried-on-Y vs A-trained-on-Y-queried-on-X (storage only,
    # resident labels), observed in this run.
    storages = sorted({parse_mode_label(r["train_mode_label"])[0] for r in passed})
    for i, a in enumerate(storages):
        for b in storages[i + 1 :]:
            ab, ba = by_pair.get((a, b)), by_pair.get((b, a))
            if ab and ba:
                lines.append(
                    f"symmetry: {a}->{b} eval {_fmt(ab['eval_loss_with_adapter'])} vs "
                    f"{b}->{a} eval {_fmt(ba['eval_loss_with_adapter'])} (observed in this run)"
                )
    return lines


def write_csv(path: str, rows) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
