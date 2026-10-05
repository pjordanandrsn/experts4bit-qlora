# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Per-step timeline of the serving loop (``E4B_PAGED_STEP_TRACE=<path>``, opt-in; serve_paged).

``E4B_PAGED_TRACE`` writes one row per finished REQUEST, which is enough to fit a stall per prefill but not to say
where it goes (SC2's fit could not even separate it from batch growth). This writes one row per engine STEP:

* what the step carried: prefill chunks and tokens, decode rows and their bucket, slots decoding for the first time,
  admissions, the queue and the active set;
* where its host time went: ``seg`` holds the time between consecutive marks on the engine thread, named for the work
  that ended at each mark. In order they are ``ops`` (requests and aborts handed in), ``plan`` (admission, including
  the slot reset), ``pf_prep`` (a chunk's inputs), ``pf_forward`` (the prefill forward, or its graph replay, issued),
  ``pf_flush`` (the prompt's K/V into the FP8 pool), ``pf_sync`` (waiting for the first token), ``pf_emit``,
  ``dec_ready`` (blocks claimed for slots decoding for the first time), ``dec_prep`` (the bucket's inputs),
  ``dec_issue`` (the decode graph replayed, or the eager step issued), ``dec_sync`` (waiting for the tokens),
  ``dec_mirror`` (host length mirrors), ``dec_emit``, ``retire`` (finished slots freed) and ``dispatch`` (tokens
  handed to the HTTP side);
* when the GPU finished: ``gpu`` holds, for each mark that records an event (``pf_prep``, ``pf_forward``,
  ``pf_flush``, ``dec_prep``, ``dec_issue``), the milliseconds from the step's first event to that one. The GPU is
  idle at ``pf_prep`` and ``dec_prep`` (the previous step synced), so ``pf_forward - pf_prep`` is the forward's device
  time and ``dec_issue - dec_prep`` the decode's. These are read only after the step's own syncs, so the instrument
  adds none.

The cost per step is a few ``perf_counter`` calls and up to six CUDA events. Rows are buffered and appended every
``flush_every`` steps, and on close. A step whose events the GPU has not finished yet (one with no sync after its
last event) is held until a later step's sync has passed it, never waited for.
"""
from __future__ import annotations

import json
import os
import time

import torch


class StepTrace:
    def __init__(self, path: str, *, cuda: bool = True, flush_every: int = 64):
        self.path = path
        self.cuda = bool(cuda) and torch.cuda.is_available()
        self.flush_every = int(flush_every)
        self.rows: list = []
        self._pending: list = []         # (row, events) not yet readable without a sync; see end()
        self.n = 0
        self.cur = None
        self._last = 0.0
        self._ev: list = []
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    # ----------------------------------------------------------- per step --
    def begin(self, **info) -> None:
        t = time.perf_counter()
        self.cur = {"step": self.n, "t": t, "seg": {}, **info}
        self._last = t
        self._ev = []
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._ev.append(("begin", e))

    def mark(self, name: str, *, event: bool = False) -> None:
        """Close the segment ``name``: the host time since the previous mark, summed if ``name`` recurs."""
        cur = self.cur
        if cur is None:
            return
        t = time.perf_counter()
        seg = cur["seg"]
        seg[name] = seg.get(name, 0.0) + (t - self._last) * 1e3
        self._last = t
        if event and self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._ev.append((name, e))

    def note(self, **kw) -> None:
        """Set fields on the current row."""
        if self.cur is not None:
            self.cur.update(kw)

    def count(self, name: str, n: int = 1) -> None:
        """Add ``n`` to a counter on the current row."""
        if self.cur is not None:
            self.cur[name] = self.cur.get(name, 0) + n

    def discard(self) -> None:
        """Drop the current row (a step that found nothing to run)."""
        self.cur = None
        self._ev = []

    def end(self, **info) -> None:
        cur = self.cur
        if cur is None:
            return
        self.mark("dispatch")
        cur.update(info)
        cur["step_ms"] = round((self._last - cur["t"]) * 1e3, 4)
        cur["seg"] = {k: round(v, 4) for k, v in cur["seg"].items()}
        # A step's events are read once its LAST event has completed, which a step that synced (a first token, a
        # decode's tokens) guarantees. A step with no sync after its last event (a prefill chunk that does not complete
        # its prompt, with nothing decoding) waits in `_pending` until a later step's sync has passed it, so the trace
        # never synchronizes on its own.
        self._pending.append((cur, self._ev))
        self.n += 1
        self.cur = None
        self._ev = []
        self._resolve(block=False)
        if len(self.rows) >= self.flush_every:
            self.flush()

    def _resolve(self, block: bool) -> None:
        """Move finished steps from ``_pending`` to ``rows``, in step order (``block`` waits: for close only)."""
        while self._pending:
            cur, evs = self._pending[0]
            if evs:
                last = evs[-1][1]
                try:
                    if not block and not last.query():
                        return
                    last.synchronize()
                    base = evs[0][1]
                    cur["gpu"] = {name: round(base.elapsed_time(e), 4) for name, e in evs[1:]}
                except RuntimeError as exc:             # a trace must never take serving down
                    cur["gpu_error"] = str(exc)[:200]
            self.rows.append(cur)
            self._pending.pop(0)

    def flush(self) -> None:
        if not self.rows:
            return
        rows, self.rows = self.rows, []
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
        except OSError as exc:
            print(f"[step_trace] append failed ({type(exc).__name__}: {exc})", flush=True)

    def close(self) -> None:
        """Resolve every pending step (waiting on the GPU if it must) and write what is buffered."""
        self._resolve(block=True)
        self.flush()
