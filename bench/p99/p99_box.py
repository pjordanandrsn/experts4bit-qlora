#!/usr/bin/env python3
"""Lane P99's measurement (bench/p99/PREREG-p99.md; e4b#913): ``bench/p98/p98_box.py`` unchanged (its registered
bytes are staged beside this file), with one line printed and flushed BEFORE every ``run_decode`` call. When a replay
faults, the process aborts asynchronously and writes no record; the last ``P99_STEP`` line in its log names the step
and the row count -- so the bucket -- the fault surfaced on.

    python p99_box.py <p98_box.py's arguments>
"""
from __future__ import annotations

import sys

import p98_box


def _logged(timer_call):
    def call(self, rids):
        n = len(rids)
        print(f"P99_STEP call={len(self.calls)} rows={n}", flush=True)
        return timer_call(self, rids)
    return call


p98_box.DecodeTimer.__call__ = _logged(p98_box.DecodeTimer.__call__)

if __name__ == "__main__":
    sys.exit(p98_box.main())
