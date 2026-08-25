# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-tr1-census phase brackets for the SHIPPED training loop.

The census instruments ``experts4bit_qlora.train``'s own step loop
rather than a replica: a replica is a simulation of the trainer, and
this campaign has already had a simulation disagree with the shipped
component on every trace (S-series). Default-inert -- nothing here
runs unless ``TR1_CENSUS=1``.

Brackets are CUDA-event-fenced on GPU (phase cost includes the kernels
it launched, not just host time) and degrade to ``perf_counter`` on
CPU so the accounting logic is testable without a GPU. Per step, phase
totals accumulate; ``write()`` emits one JSON with per-step rows and
meta. Closure/steady-state/A-A gates live in the composer
(``bench/tr1-census/tr1_compose.py``), not here -- the trainer should
never refuse a run over instrumentation.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch


def enabled() -> bool:
    return os.environ.get("TR1_CENSUS") == "1"


class PhaseClock:
    """Accumulates named phase durations within a step.

    GPU mode records a CUDA event pair per bracket and resolves them
    lazily at ``step_end`` (one sync per step, at a point the trainer
    already syncs: after ``loss.item()``/``opt.step``). CPU mode uses
    perf_counter directly.
    """

    def __init__(self, use_cuda: bool | None = None):
        self.use_cuda = (torch.cuda.is_available()
                         if use_cuda is None else use_cuda)
        self.steps: list[dict] = []
        self._open: str | None = None
        self._pairs: list[tuple[str, object, object]] = []
        self._t0 = 0.0
        self._wall0 = 0.0

    def step_start(self):
        self._pairs = []
        self._wall0 = time.perf_counter()

    def start(self, phase: str):
        assert self._open is None, \
            f"bracket {self._open!r} still open when starting {phase!r}"
        self._open = phase
        if self.use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pending = ev
        else:
            self._t0 = time.perf_counter()

    def stop(self):
        assert self._open is not None, "stop() with no open bracket"
        phase = self._open
        self._open = None
        if self.use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pairs.append((phase, self._pending, ev))
        else:
            self._pairs.append(
                (phase, self._t0, time.perf_counter()))

    def step_end(self):
        assert self._open is None, \
            f"bracket {self._open!r} left open at step end"
        wall_ms = (time.perf_counter() - self._wall0) * 1e3
        row: dict = {}
        if self.use_cuda:
            torch.cuda.synchronize()
            for phase, a, b in self._pairs:
                row[phase] = row.get(phase, 0.0) + a.elapsed_time(b)
        else:
            for phase, a, b in self._pairs:
                row[phase] = row.get(phase, 0.0) + (b - a) * 1e3
        row["step_wall_ms"] = wall_ms
        self.steps.append(row)

    def write(self, path: str, meta: dict):
        out = {"meta": meta, "steps": self.steps}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(out, indent=1))
        return out
