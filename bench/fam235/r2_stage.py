# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Seal R2's transport closure with Q235's sealer (bench/q235/q235_stage.py), run unchanged with R2's closure and manifest in
place of Q235's. The locked input assets are Q235's (bench/q235/locked_inputs.json): the same data, the same bytes.

    python r2_stage.py --assets DIR        (the flat path list r2_drive.sh stages, or a refusal)
    python r2_stage.py --write             (rewrite r2.sha256 from CLOSURE after a reviewed change)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Every file the box runs or reads: Q235's tools unchanged, with R2's runner, lock and tiny model in place of Q235's.
CLOSURE = ("bench/fam235/r2_run.sh", "bench/q235/q235_bootstrap.py", "bench/q235/q235_fetch.py", "bench/fam235/r2_tiny.py",
           "bench/q235/q235_hostrate.py",
           "bench/q235/wheels.json", "bench/fam235/model_files.json", "bench/q235/instrument/q235_box.py",
           "bench/q235/instrument/q235_tokens.py", "bench/q235/instrument/mk_slice.py", "bench/tc1/tc1_arm.py",
           "bench/dq11/dq11_require_git.sh")
MANIFEST = HERE / "r2.sha256"


def q235_stage():
    """Q235's sealer as a private module instance, with R2's closure and manifest set on it."""
    spec = importlib.util.spec_from_file_location("q235_stage_for_r2", REPO / "bench" / "q235" / "q235_stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CLOSURE, mod.MANIFEST = CLOSURE, MANIFEST
    return mod


def write() -> str:
    return q235_stage().write()


def stage(assets: Path) -> tuple:
    return q235_stage().stage(assets)


if __name__ == "__main__":
    q235_stage().main()
