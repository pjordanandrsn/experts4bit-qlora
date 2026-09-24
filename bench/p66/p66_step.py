# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P66 Level M: the served decode step, through ``bench/hybrid-g9/step_decomp.py`` UNCHANGED.

step_decomp's ``--engine pipelined`` arm (PREREG-b1 R1) hard-codes every expert hot. This shim runs the
staged step_decomp byte for byte (``runpy``, argv passed through) and changes exactly one thing, only when
``P66_HOT_FRAC`` is set: the hot sets handed to ``enable_pipelined_residency`` become the first
``round(P66_HOT_FRAC * E)`` of the same seeded per-layer permutation ``p66_census.hot_sets_for`` uses, so a
Level M arm and a Level L arm at one hot fraction hold the same experts.

step_decomp imports ``enable_pipelined_residency`` inside ``main`` at call time, so replacing the module
attribute before ``main`` runs is sufficient and nothing in step_decomp is edited. The swap is recorded:
``P66_HOT_SETS_OUT`` receives the sets actually passed, and the shim refuses if step_decomp never calls
the engine (a hot fraction that silently did not apply would read as the all-hot arm)."""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path


def _hot_sets(L: int, E: int, n_hot: int, seed: int) -> list:
    import torch
    out = []
    for li in range(L):
        g = torch.Generator().manual_seed(seed * 1000 + li)
        out.append(sorted(torch.randperm(E, generator=g)[:n_hot].tolist()))
    return out


def main() -> int:
    step_decomp = os.environ.get("P66_STEP_DECOMP", str(Path(__file__).resolve().parent / "step_decomp.py"))
    frac = os.environ.get("P66_HOT_FRAC")
    called = []
    if frac is not None:
        import experts4bit_qlora.engines.pipelined as P
        orig = P.enable_pipelined_residency
        f, seed = float(frac), int(os.environ.get("P66_HOT_SEED", "66"))

        def patched(model, hot_sets, *a, **kw):
            L = len(hot_sets)
            E = len(hot_sets[0])                       # step_decomp passes range(E) per layer
            sets = _hot_sets(L, E, int(round(f * E)), seed)
            called.append({"layers": L, "experts": E, "hot_frac": f, "n_hot": len(sets[0]), "seed": seed})
            out = os.environ.get("P66_HOT_SETS_OUT")
            if out:
                Path(out).write_text(json.dumps({"meta": called[-1], "hot_sets": sets}))
            print(f"[p66_step] pipelined hot sets: {len(sets[0])}/{E} per layer (hot_frac {f}, seed {seed})",
                  flush=True)
            return orig(model, sets, *a, **kw)
        P.enable_pipelined_residency = patched
    sys.argv = [step_decomp] + sys.argv[1:]
    rc = 0
    try:
        runpy.run_path(step_decomp, run_name="__main__")
    except SystemExit as e:                  # step_decomp's own exit code is the arm's, unless it is None
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    if frac is not None and not called:
        print("[p66_step] REFUSE: P66_HOT_FRAC was set but step_decomp never enabled the pipelined engine",
              flush=True)
        return 3
    return rc


if __name__ == "__main__":
    sys.exit(main())
