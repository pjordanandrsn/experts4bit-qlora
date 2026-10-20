#!/usr/bin/env python3
"""HBD's reducer (bench/hbd/PREREG-hbd.md): the registered outcome table, applied in its order, to the eight cells' receipts.

    python hbd_reduce.py hbd_spec.json --out hbd/result.json

The spec names, per cell (E0a E0b E2 E1 U0a U0b U1 U2), its receipt (the arm's JSON), its in-cell record (``--hbd-record``)
and its sibling probe (``hbd_probe.py``), as paths relative to the spec. Reporting only beyond the table: no parity claim is
made in any outcome.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

CELLS = ("E0a", "E0b", "E2", "E1", "U0a", "U0b", "U1", "U2")
REPEATS = (("E0a", "E0b"), ("U0a", "U0b"))
BUILD_PAIRS = (("E2", "E1", "e4b"), ("U0a", "U1", "unsloth"))     # across builds at fixed transformers (PREREG outcome 3)
VENV_PAIR = ("E0a", "E2")                                          # e4b across venvs at a fixed build (outcome 4)
REPORTED = (("E0a", "E1"), ("U1", "U2"), ("E0a", "U0a"), ("E1", "U1"))


def _load(base, rel):
    with open(os.path.join(base, rel)) as f:
        return json.load(f)


def cell_view(base, files):
    """The fields the table reads from one cell, or the reason the cell cannot be read."""
    try:
        rec = _load(base, files["receipt"])
    except (OSError, ValueError) as e:
        return {"error": f"receipt unreadable: {type(e).__name__}"}
    hp1 = rec.get("hp1") or {}
    v = {"status": rec.get("status"), "train_base_error": hp1.get("train_base_error"), "hbd_error": hp1.get("hbd_error"),
         "digest": (hp1.get("train_base") or {}).get("digest"), "dq_n_slots": (hp1.get("train_base") or {}).get("n_slots")}
    hbd = hp1.get("hbd") or {}
    v.update(stored=hbd.get("stored_digest"), n_slots=hbd.get("n_slots"), keys=hbd.get("keys_sha256"),
             versions=hbd.get("versions") or {}, libs=((hbd.get("bnb") or {}).get("libraries")) or [])
    try:
        v["per_slot"] = _load(base, files["record"]).get("per_slot") or {}
    except (OSError, ValueError, KeyError):
        v["per_slot"] = None
    try:
        v["probe_libs"] = _load(base, files["probe"]).get("libraries") or []
    except (OSError, ValueError, KeyError):
        v["probe_libs"] = None
    return v


def _lib(libs):
    shas = {x.get("sha256") for x in libs or [] if x.get("sha256")}
    return shas.pop() if len(shas) == 1 and len(libs) == 1 else None


def void_reasons(c):
    why = []
    for name in CELLS:
        x = c.get(name)
        if x is None or "error" in x:
            why.append(f"{name}: {'missing' if x is None else x['error']}")
            continue
        if x["status"] != "ok":
            why.append(f"{name}: status {x['status']}")
        if x["train_base_error"]:
            why.append(f"{name}: train_base_error")
        if x["hbd_error"] or not x["stored"] or not x["digest"]:
            why.append(f"{name}: no stored-state or dequantized digest")
        if _lib(x["libs"]) is None:
            why.append(f"{name}: the cell mapped {len(x['libs'])} libbitsandbytes files, not one")
        elif x["probe_libs"] is None or _lib(x["probe_libs"]) != _lib(x["libs"]):
            why.append(f"{name}: the sibling probe disagrees with the cell's own library")
    if why:
        return why
    for a, b in REPEATS:
        if (c[a]["digest"], c[a]["stored"]) != (c[b]["digest"], c[b]["stored"]):
            why.append(f"determinism: {a} and {b} differ")
    for a, b, fw in BUILD_PAIRS:
        if _lib(c[a]["libs"]) == _lib(c[b]["libs"]):
            why.append(f"{fw}: {a} and {b} loaded the same libbitsandbytes file, so the build did not change the kernel")
    return why


def _kind(key):
    k = key.split("|")[1]
    return "experts" if k in ("gate_up", "down") else "attention" if k in ("q", "k", "v", "o") else "other"


def compare(c, a, b):
    """One pair's reading: STRUCTURE, SAME, DIFFERENT_QUANTIZED or DIFFERENT_DEQUANT, with the versions that differ and, for a
    DIFFERENT, the slot kinds whose stored state differs."""
    x, y = c[a], c[b]
    vers = sorted(k for k in set(x["versions"]) | set(y["versions"]) if x["versions"].get(k) != y["versions"].get(k))
    out = {"pair": [a, b], "versions_differ": {k: [x["versions"].get(k), y["versions"].get(k)] for k in vers}}
    if (x["n_slots"], x["keys"]) != (y["n_slots"], y["keys"]):
        px, py = set(x["per_slot"] or {}), set(y["per_slot"] or {})
        return dict(out, reading="STRUCTURE", only_in=[sorted(px - py)[:20], sorted(py - px)[:20]])
    if x["stored"] == y["stored"] and x["digest"] == y["digest"]:
        return dict(out, reading="SAME")
    reading = "DIFFERENT_QUANTIZED" if x["stored"] != y["stored"] else "DIFFERENT_DEQUANT"
    kinds = {}
    if x["per_slot"] is not None and y["per_slot"] is not None:
        for k, h in x["per_slot"].items():
            if y["per_slot"].get(k) != h:
                kinds[_kind(k)] = kinds.get(_kind(k), 0) + 1
    return dict(out, reading=reading, stored_slots_differ_by_kind=kinds, dequant_equal=x["digest"] == y["digest"])


def reduce(spec, base):
    cells = {name: cell_view(base, spec["cells"][name]) for name in CELLS if name in spec.get("cells", {})}
    out = {"reducer": "bench/hbd/hbd_reduce.py", "prereg": "bench/hbd/PREREG-hbd.md", "claim": "none (no parity claim in any outcome)",
           "cells": {k: {f: v.get(f) for f in ("status", "digest", "stored", "n_slots")} | {"library": _lib(v.get("libs"))}
                     for k, v in cells.items() if "error" not in v}}
    why = void_reasons(cells)
    if why:
        return dict(out, outcome="VOID", reasons=why)
    out["build"] = {fw: compare(cells, a, b) for a, b, fw in BUILD_PAIRS}
    out["venv_at_fixed_build"] = compare(cells, *VENV_PAIR)
    out["reported"] = [compare(cells, a, b) for a, b in REPORTED]
    reads = {r["reading"] for r in out["build"].values()}
    out["outcome"] = "READ"
    out["summary"] = {fw: r["reading"] for fw, r in out["build"].items()}
    out["depth_note"] = ("SAME at 2 layers says nothing about depth: HP2's gap at 48 layers stays open"
                         if reads == {"SAME"} else "a DIFFERENT here makes HP2's base term at least partly a build effect"
                         if reads & {"DIFFERENT_QUANTIZED", "DIFFERENT_DEQUANT"} else "see the per-pair readings")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="HBD's registered outcome table")
    ap.add_argument("spec")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec))
    res = reduce(spec, os.path.dirname(os.path.abspath(a.spec)))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(f"HBD={res['outcome']} " + json.dumps(res.get("summary") or res.get("reasons")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
