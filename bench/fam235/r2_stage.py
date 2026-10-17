# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Seal R2's transport closure with Q235's sealer (bench/q235/q235_stage.py at the commit R2 cites), run unchanged with R2's closure
and manifest in place of Q235's. The locked input assets are Q235's (its locked_inputs.json at that commit): the same data, the
same bytes.

- **Sealing** (``--write``) hashes R2's own files from the working tree and Q235's (and the shared ones) at ``r2_drift.CITED``, so
  ``r2.sha256`` is what R2 cites, whatever Q235's working tree holds.
- **Staging** (``--assets``, at launch) hashes the working tree. Once Q235 moves past the cited commit, the closure no longer
  matches ``r2.sha256`` and staging refuses: R2 never launches on Q235 bytes it has not cited. It takes a Q235 change by re-citing
  (``CITED``, the copies and the re-seal together).

    python r2_stage.py --assets DIR        (the flat path list r2_drive.sh stages, or a refusal)
    python r2_stage.py --write             (rewrite r2.sha256 after a reviewed change or a re-cite)
"""
from __future__ import annotations

import argparse
import atexit
import functools
import importlib.util
import shutil
import tempfile
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
OWN = "bench/fam235/"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=None)
def drift():
    return _load("r2_drift_for_stage", HERE / "r2_drift.py")


def q235_stage(repo: Path = REPO):
    """Q235's sealer at the cited commit, as a private module instance, with R2's closure and manifest, hashing under ``repo``."""
    mod = _load("q235_stage_for_r2", drift().cited_tree() / "bench" / "q235" / "q235_stage.py")
    mod.CLOSURE, mod.MANIFEST, mod.REPO = CLOSURE, MANIFEST, repo
    return mod


def sealing_tree() -> Path:
    """The closure as R2 seals it: R2's own files from the working tree, the rest at the cited commit."""
    cited = drift().cited_tree()
    root = Path(tempfile.mkdtemp(prefix="r2-seal-"))
    atexit.register(shutil.rmtree, root, True)
    for p in CLOSURE:
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / p if p.startswith(OWN) else cited / p, root / p)
    return root


def write() -> str:
    return q235_stage(sealing_tree()).write()


def stage(assets: Path) -> tuple:
    return q235_stage(REPO).stage(assets)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--assets", type=Path)
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        print(write(), end="")
    else:
        print(" ".join(str(p) for p in stage(a.assets)))


if __name__ == "__main__":
    main()
