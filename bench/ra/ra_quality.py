"""Keep P115's R scores unchanged while retaining their per-position argmax IDs."""
from __future__ import annotations

import functools


def measure_r(instrument, model, windows, **kwargs):
    """Run the frozen helper; restore its score function even after failure.

    This is process-local instrumentation. RA components run in fresh processes.
    No fusion flags are set here; the executor supplies the release-default model.
    """
    if set(windows) != {"wikitext"} or kwargs.get("phase") != "off":
        raise ValueError("RA quality requires its own wikitext R")
    original = instrument.p108_box._score
    samples = []

    @functools.wraps(original)
    def score(*args, **kw):
        nll, am = original(*args, **kw)
        samples.append((list(nll), list(am)))
        return nll, am

    instrument.p108_box._score = score
    try:
        result = instrument.measure_phase(model, windows, arms=("R",), **kwargs)
    finally:
        instrument.p108_box._score = original
    rows = result["per_window"]["wikitext"]["R"]
    if len(samples) != len(rows) or len(rows) != len(windows["wikitext"]):
        raise ValueError("RA score capture/window count mismatch")
    for row, (nll, am) in zip(rows, samples):
        if not nll or row["nll"] != sum(nll) / len(nll) or len(am) != kwargs["cont"]:
            raise ValueError("RA score capture changed scores/shapes")
        row["argmax_ids"] = am
    return result
