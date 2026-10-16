# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""R2's pre-rental fetch gate: Q235's (bench/q235/q235_fetch_gate.py), run unchanged, with R2's lock, closure and registration in
place of Q235's. Every check is Q235's: the hub listing against the lock row for row, every locked wheel's one-byte probe through
the box's own opener, the git pins, and the scan for unfilled values. Only the files it reads differ: R2's ``model_files.json``
and, for the scan, R2's closure, manifest, drive, stage and registration.

    python r2_fetch_gate.py --e4b-sha SHA --out GATE.json            (exit 0 passed, 1 refused, 2 usage)
    python r2_fetch_gate.py --e4b-sha SHA --check-report GATE.json   (r2_drive.sh's check: passed, this commit, under a day old)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
Q235 = REPO / "bench" / "q235"
SCHEMA = "r2-fetch-gate/1"


class _Here:
    """Q235's ``HERE`` with R2's own files where R2 has them: the lock is R2's, the wheels and the bootstrap are Q235's."""
    own = {"model_files.json": HERE / "model_files.json"}

    def __truediv__(self, name):
        return self.own.get(name, Q235 / name)

    def glob(self, pattern):
        return HERE.glob(pattern)


def _r2_stage():
    spec = importlib.util.spec_from_file_location("r2_stage_for_gate", HERE / "r2_stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def placeholder_files() -> list:
    files = [REPO / p for p in _r2_stage().CLOSURE]
    files += [HERE / n for n in ("r2.sha256", "r2_drive.sh", "r2_stage.py", "PREREG-r2-mixtral.md")] + [Q235 / "locked_inputs.json"]
    return [f for f in files if f.name != "tc1_arm.py"]       # TC1's instrument, registered and reviewed in its own lane


def gate_module():
    """Q235's gate as a private module instance, reading R2's files."""
    spec = importlib.util.spec_from_file_location("q235_fetch_gate_for_r2", Q235 / "q235_fetch_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.HERE, mod.SCHEMA, mod.placeholder_files = _Here(), SCHEMA, placeholder_files
    return mod


if __name__ == "__main__":
    gate_module().main()
