# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""R2's time-left fit: Q235's (bench/q235/q235_fit.py), run unchanged, with R2's depth and checkpoint in place of Q235's. Every phase's
budget is the A2000 rehearsal's measured time scaled to the registered model, times the margin; each sealed cap in ``r2_run.sh``
must cover its budget, and the sealed host-rate floors must equal the rehearsal's own.

The constants that differ (Mixtral-8x22B-Instruct-v0.1 at cc88a6cc, from its config): 56 decoder layers against the rehearsal's 4;
per layer, bf16, q and o (6,144 x 6,144 each), k and v (1,024 x 6,144 each), the router (8 x 6,144), the 8 experts' w1, w3 and w2
(3 x 16,384 x 6,144 each) and two norms (2 x 6,144); outside the layers the embedding and the head (32,768 x 6,144 each) and the
final norm (6,144). 56 layers and the outside sum to 281,260,142,592 B, the checkpoint index's ``total_size``.

Its reads: the lock and the runner are R2's (``model_files.json``, ``r2_run.sh``), the wheels Q235's.

    python r2_fit.py --rehearsal DIR --warm-install-s S --bandwidth-mbps M --usd-per-hour R [--out FIT.json]
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
Q235 = REPO / "bench" / "q235"
LAYERS, SLICE_LAYERS = 56, 4
H, FF, V, E, KV = 6144, 16384, 32768, 8, 1024           # hidden, expert intermediate, vocabulary, experts, k/v width
LAYER_BYTES = 2 * (H * H + 2 * KV * H + H * H + E * H + E * 3 * FF * H + 2 * H)
OUTSIDE_BYTES = 2 * (2 * V * H + H)


class _Here:
    """Q235's ``HERE`` with R2's own files where R2 has them: the lock and the runner are R2's, the wheels Q235's."""
    own = {"model_files.json": HERE / "model_files.json", "q235_run.sh": HERE / "r2_run.sh"}

    def __truediv__(self, name):
        return self.own.get(name, Q235 / name)


def fit_module():
    """Q235's fit as a private module instance, with R2's depth, checkpoint bytes and files."""
    spec = importlib.util.spec_from_file_location("q235_fit_for_r2", Q235 / "q235_fit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.HERE, mod.LAYERS, mod.SLICE_LAYERS = _Here(), LAYERS, SLICE_LAYERS
    mod.LAYER_BYTES, mod.OUTSIDE_BYTES = LAYER_BYTES, OUTSIDE_BYTES
    return mod


if __name__ == "__main__":
    fit_module().main()
