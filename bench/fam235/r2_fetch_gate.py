# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""R2's pre-rental fetch gate: Q235's (bench/q235/q235_fetch_gate.py at the commit R2 cites), run unchanged, with R2's lock, closure
and registration in place of Q235's. Every check is Q235's: the hub listing against the lock row for row, every locked wheel's
one-byte probe through the box's own opener, the git pins, and the scan for unfilled values. Only the files it reads differ: R2's
``model_files.json`` and, for the scan, R2's closure, manifest, drive, stage and registration.

    python r2_fetch_gate.py --e4b-sha SHA --out GATE.json            (exit 0 passed, 1 refused, 2 usage)
    python r2_fetch_gate.py --e4b-sha SHA --check-report GATE.json   (r2_drive.sh's check: passed, this commit, under a day old)
"""
from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
Q235 = REPO / "bench" / "q235"
SCHEMA = "r2-fetch-gate/1"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=None)
def cited():
    """Q235's tree at the commit R2 cites (``r2_drift.cited_tree``)."""
    return _load("r2_drift_for_gate", HERE / "r2_drift.py").cited_tree()


class _Here:
    """Q235's ``HERE`` with R2's own files where R2 has them: the lock is R2's; the wheels and the bootstrap are Q235's, as cited."""
    own = {"model_files.json": HERE / "model_files.json"}

    def __truediv__(self, name):
        return self.own.get(name, cited() / "bench" / "q235" / name)

    def glob(self, pattern):
        return HERE.glob(pattern)


def _r2_stage():
    return _load("r2_stage_for_gate", HERE / "r2_stage.py")


def placeholder_files() -> list:
    files = [REPO / p for p in _r2_stage().CLOSURE]
    files += [HERE / n for n in ("r2.sha256", "r2_drive.sh", "r2_stage.py", "PREREG-r2-mixtral.md")] + [Q235 / "locked_inputs.json"]
    return [f for f in files if f.name != "tc1_arm.py"]       # TC1's instrument, registered and reviewed in its own lane


def gate_module():
    """Q235's gate at the cited commit as a private module instance, reading R2's files. Its scan reports paths relative to this
    repository, and its listing helper (P130's) is read at the same cited commit."""
    mod = _load("q235_fetch_gate_for_r2", cited() / "bench" / "q235" / "q235_fetch_gate.py")
    mod.HERE, mod.REPO, mod.SCHEMA, mod.placeholder_files = _Here(), REPO, SCHEMA, placeholder_files
    mod._p130 = lambda: _load("p130_fetch_gate_for_r2", cited() / "bench" / "p130" / "p130_fetch_gate.py")
    return mod


if __name__ == "__main__":
    gate_module().main()
