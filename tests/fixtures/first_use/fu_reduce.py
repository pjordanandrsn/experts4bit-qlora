#!/usr/bin/env python3
"""T1 step 3, the first-use matrix (bench/first_use/PREREG-first_use.md): one verdict per cell from the cell's own
``first_run.json`` (``scripts/first_run.sh``'s summary), graded against the expectations registered before any data.

A cell is graded by the driver the BOX read (``first_run.json``'s ``gpu`` field: nvidia-smi's ``name, driver_version``),
never by the offer listing. A box outside its cell's band reads CELL_MISMATCH, which is neither a pass nor a fail, and
both values are kept. Standard library only: the box runs this with the image's python, outside any environment the
first run builds.

    fu_reduce.py --dir CELLS_DIR --run-id ID --out verdict.json
    fu_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

#: driver-major bands, [lo, hi). The default PyPI torch (cu130) needs driver 580; cu126/cu129 run from 525.
BANDS = {"below": (525, 580), "at": (580, 590), "current": (590, None)}

#: The registered cells. ``torch`` is how the cell sets TORCH_INDEX: "pypi" (the default PyPI torch), "auto" (unset:
#: ``loggetta inspect --torch-index`` names the build), or "cpu" (the CPU-only build on a box whose GPU is visible).
#: ``band`` "none" is a host with no NVIDIA GPU at all; None means any GPU band.
CELLS = {
    "below-default":   {"band": "below",   "torch": "pypi", "exit": {4},    "index": {"pypi"}},
    "below-matched":   {"band": "below",   "torch": "auto", "exit": {0, 7}, "index": {"cu126", "cu129"}},
    "at-default":      {"band": "at",      "torch": "pypi", "exit": {0, 7}, "index": {"pypi"}},
    "at-matched":      {"band": "at",      "torch": "auto", "exit": {0, 7}, "index": {"cu130"}},
    "current-default": {"band": "current", "torch": "pypi", "exit": {0, 7}, "index": {"pypi"}},
    "current-matched": {"band": "current", "torch": "auto", "exit": {0, 7}, "index": {"cu130"}},
    "cpu-torch":       {"band": None,      "torch": "cpu",  "exit": {4},    "index": {"cpu"}},
    "no-gpu":          {"band": "none",    "torch": "auto", "exit": {3},    "index": {None}},
}

#: first_run.sh's exit codes. Every one but 2 (usage) is named; 7 is a trained run whose device check did not run.
EXIT_MEANING = {0: "TRAINED, device check passed", 7: "TRAINED, device check not run (not a pass of it)",
                3: "refused: no torch build fits this machine", 4: "refused: the plan was refused",
                5: "refused: an install step failed", 6: "refused: the run did not train"}
NAMED_REFUSALS = {3, 4, 5, 6}
TRACEBACK = "Traceback (most recent call last)"


def band_of(driver: str | None) -> str | None:
    """The band a driver version falls in, or None outside every band (or unreadable)."""
    try:
        major = int(str(driver).split(".")[0])
    except (TypeError, ValueError):
        return None
    for name, (lo, hi) in BANDS.items():
        if major >= lo and (hi is None or major < hi):
            return name
    return None


def box_driver(record: dict) -> str | None:
    """The driver the box read: the last comma field of first_run.json's ``gpu`` (``name, driver_version``)."""
    gpu = record.get("gpu")
    if not gpu:
        return None
    return str(gpu).splitlines()[0].rsplit(",", 1)[-1].strip() or None


def named(record: dict) -> bool:
    """The gate: a trained exit, or a named refusal (a known code with its reason recorded), never a traceback."""
    code = record.get("exit")
    if code in (0, 7):
        return True
    why = record.get("refusal") or ""
    return code in NAMED_REFUSALS and bool(why.strip()) and TRACEBACK not in why


def traceback_files(cell_dir: str, logs_dir: str | None, cell: str) -> list:
    """The cell's own outputs that hold a traceback header. first_run.sh builds a refusal from the LAST lines of a step's
    output, which never include the header, so the gate reads the files themselves: the run's step outputs, the
    driver read's stderr, and the runner's logs of the cell."""
    names = []
    run = os.path.join(cell_dir, "run")
    if os.path.isdir(run):
        names += [os.path.join(run, f) for f in sorted(os.listdir(run)) if f.endswith((".txt", ".err"))]
    if logs_dir:
        names += [os.path.join(logs_dir, f"{cell}.{ext}") for ext in ("out", "err")]
    hits = []
    for n in names:
        try:
            with open(n, errors="replace") as f:
                if TRACEBACK in f.read():
                    hits.append(os.path.relpath(n, os.path.dirname(cell_dir.rstrip("/"))))
        except OSError:
            continue
    return hits


def grade(cell: str, record: dict | None, listed_driver: str | None = None, tracebacks: list = (),
          skipped: str | None = None) -> dict:
    spec = CELLS[cell]
    row = {"cell": cell, "expected_exit": sorted(spec["exit"]), "listed_driver": listed_driver}
    if record is None:
        return {**row, "verdict": "MISSING", "gate": None, **({"skipped": skipped} if skipped else {})}
    driver = box_driver(record)
    platform = record.get("platform") or ""
    row.update(exit=record.get("exit"), meaning=EXIT_MEANING.get(record.get("exit"), "not a first_run.sh exit"),
               refusal=record.get("refusal"), torch_index=record.get("torch_index"), box_driver=driver,
               box_band=band_of(driver), gpu=record.get("gpu"), platform=platform,
               seconds_to_trained=record.get("seconds_to_trained"),
               run_id=(record.get("receipt") or {}).get("run_id"),
               device_check=(record.get("receipt") or {}).get("device_check"),
               versions=record.get("versions"), tracebacks=list(tracebacks),
               gate=named(record) and not tracebacks)
    if spec["band"] == "none":
        in_band = record.get("gpu") is None and platform.startswith("Linux")
        why = "" if in_band else f"the no-GPU cell needs a Linux host with no NVIDIA GPU (gpu {record.get('gpu')!r}, platform {platform!r})"
    elif spec["band"] is None:
        in_band = row["box_band"] is not None
        why = "" if in_band else f"the box read driver {driver!r}, in no registered band"
    else:
        in_band = row["box_band"] == spec["band"]
        why = "" if in_band else f"the box read driver {driver!r} (band {row['box_band']}), the cell registers {spec['band']}"
    if not in_band:
        return {**row, "verdict": "CELL_MISMATCH", "why": why}
    met = record.get("exit") in spec["exit"] and record.get("torch_index") in spec["index"]
    return {**row, "verdict": "MET" if met else "NOT_MET",
            **({} if met else {"why": f"expected exit {sorted(spec['exit'])} with torch index in "
                                      f"{sorted(spec['index'], key=str)}"})}


def reduce(cells_dir: str, run_id: str, logs_dir: str | None = None) -> dict:
    """Grades every cell directory the runner made. The runner makes each one BEFORE its deadline check, so a skipped
    cell is present and reads MISSING (the run INCOMPLETE), never silently absent."""
    if logs_dir is None:
        guess = os.path.join(os.path.dirname(os.path.abspath(cells_dir)), "logs")
        logs_dir = guess if os.path.isdir(guess) else None
    rows = []
    for cell in CELLS:
        d = os.path.join(cells_dir, cell)
        if not os.path.isdir(d):
            continue                                   # a box runs only its own cells; absent cells are not this run's
        record = None
        p = os.path.join(d, "first_run.json")
        if os.path.exists(p):
            with open(p) as f:
                record = json.load(f)
        meta = {}
        cp = os.path.join(d, "cell.json")
        if os.path.exists(cp):
            with open(cp) as f:
                meta = json.load(f)
        rows.append(grade(cell, record, meta.get("listed_driver") or None, traceback_files(d, logs_dir, cell),
                          meta.get("skipped") or None))
    graded = [r for r in rows if r["verdict"] in ("MET", "NOT_MET")]
    gate = "PASS" if graded and all(r["gate"] for r in graded) else ("FAIL" if graded else "NONE_GRADED")
    if any(r["verdict"] == "MISSING" for r in rows):
        gate = "FAIL" if gate == "FAIL" else "INCOMPLETE"
    return {"schema": "first-use-verdict/1", "run_id": run_id, "gate": gate, "cells": rows,
            "met": sum(r["verdict"] == "MET" for r in rows), "not_met": sum(r["verdict"] == "NOT_MET" for r in rows),
            "mismatch": sum(r["verdict"] == "CELL_MISMATCH" for r in rows),
            "times": [{"cell": r["cell"], "seconds_to_trained": r["seconds_to_trained"], "run_id": r["run_id"],
                       "exit": r["exit"]} for r in graded if r.get("exit") in (0, 7)]}


def self_test() -> None:
    assert band_of("575.64.05") == "below" and band_of("580.82.07") == "at" and band_of("595.1") == "current"
    assert band_of("520.0") is None and band_of(None) is None and band_of("x") is None
    assert box_driver({"gpu": "NVIDIA RTX A4000, 575.64.05"}) == "575.64.05"
    ok = {"exit": 0, "gpu": "NVIDIA GeForce RTX 4090, 580.82.07", "torch_index": "cu130", "platform": "Linux-6.8",
          "refusal": None, "receipt": {"run_id": "r", "device_check": "checked"}, "seconds_to_trained": 600}
    assert grade("at-matched", ok)["verdict"] == "MET"
    assert grade("current-matched", ok)["verdict"] == "CELL_MISMATCH"
    assert grade("at-matched", {**ok, "exit": 2, "refusal": "usage"})["gate"] is False
    tb = {**ok, "exit": 6, "refusal": "Traceback (most recent call last): ..."}
    assert grade("at-matched", tb)["gate"] is False
    assert grade("at-matched", {**ok, "exit": 6, "refusal": "ValueError: x"}, tracebacks=["run/execute.txt"])["gate"] is False
    print("fu_reduce self-test OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        self_test()
        return 0
    if not a.dir:
        ap.error("--dir is required")
    v = reduce(a.dir, a.run_id)
    text = json.dumps(v, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    for r in v["cells"]:
        print(f"{r['cell']:16s} {r['verdict']:13s} exit={r.get('exit')} driver={r.get('box_driver')} "
              f"index={r.get('torch_index')} {r.get('why', '')}".rstrip())
    print(f"GATE {v['gate']}: {v['met']} met, {v['not_met']} not met, {v['mismatch']} mismatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
