"""PGX: the tier counters' own cost and their real producer check (PREREG-pgx.md, Amendment 3).

Run with ``bench/pgx/hook_e4b`` on PYTHONPATH and ``PGX_DRAM_TIME_OUT`` set, in an interpreter that imports e4b:

    PYTHONPATH=bench/pgx/hook_e4b PGX_DRAM_TIME_OUT=/tmp/d.jsonl python bench/pgx/pgx_dram_bench.py [--calls N]

It imports e4b's real ``experts4bit_qlora.engines.hybrid`` through the hook and checks:
- ``wrap``: each of the hook's ``ROUTES`` on ``_HybridTier`` is the hook's wrapper, and its ``__wrapped__`` is e4b's own
  method of that name;
- ``split``: every wrapper counts what it should, decode (rows <= ``PGX_MAX_DECODE_ROWS``, 4) apart from prefill. Each
  wrapper is called on a ``_HybridTier`` built without ``__init__``. ``_dram_contrib``'s real body runs (its
  ``amort is None`` branch, the production shape) into a counting stub of ``_dram_contrib_inner``. The other four real
  bodies are entered and stop at their first tensor op on the stand-in arguments, which is caught: their kernels need
  CUDA, which this $0 path does not have.
- ``per_call_ns``: each wrapper's added cost per call, the median over 5 timed loops of the wrapper with its original
  swapped for a no-op, minus the no-op called directly;
- ``append_ns``: one append of the cumulative counters (the daemon does one a second);
- ``bound``: per decode step, 48 MoE layers x (forward + _cold_contrib + _dram_contrib), plus append_ns a second, as a
  fraction of the shakedown's measured E-hyb decode step (``STEP_S``).
Prints one JSON line. Exit 0 when the wrap and the split are right and the bound is under 0.1 %, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

STEP_S = 0.190672          # pgx-shake-r1's E-hyb median TPOT at 8 GB (bench/pgx/receipts/pgx-shake-r1)
LAYERS = 48
BOUND = 0.001
ROUTES = ("forward", "_cold_contrib", "_dram_contrib", "_dram_on_gpu", "_cold_on_cpu")


class _T:
    """A stand-in tensor: ``shape`` (rows first) and ``numel``; no data, so a real body stops at its first tensor op."""
    def __init__(self, rows, numel=None):
        self.shape = (rows, 2048)
        self._n = rows if numel is None else numel

    def numel(self):
        return self._n


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    calls = int(argv[argv.index("--calls") + 1]) if "--calls" in argv else 200_000
    if not os.environ.get("PGX_DRAM_TIME_OUT"):
        print(json.dumps({"ok": False, "why": "PGX_DRAM_TIME_OUT is not set: the hook is inert"}))
        return 1
    import experts4bit_qlora.engines.hybrid as h
    cls = h._HybridTier

    def _wrapped(name):
        w = cls.__dict__.get(name)
        orig = getattr(w, "__wrapped__", None)
        return (w is not None and getattr(w, "__qualname__", "") == f"pgx_hook.{name}" and orig is not None
                and getattr(orig, "__module__", None) == h.__name__ and orig.__qualname__ == f"_HybridTier.{name}")
    out = {"wrap": {r: _wrapped(r) for r in ROUTES}}
    c = getattr(h, "_pgx_dram_counters", None)
    if not all(out["wrap"].values()) or c is None:
        out.update(ok=False, why="the hook did not wrap every route", python=sys.version.split()[0])
        print(json.dumps(out))
        return 1
    t = object.__new__(cls)
    t.amort = None
    inner = []
    t._dram_contrib_inner = lambda x, *a: inner.append(x.shape[0])
    base = dict(c)

    def enter(fn, *a):
        try:
            fn(*a)
            return "ran"
        except Exception:  # noqa: BLE001 -- the real body stopped on a stand-in argument, after the wrapper counted
            return "entered"
    states = {
        "forward": [enter(t.forward, _T(2), _T(2, 16), None), enter(t.forward, _T(512), _T(512, 4096), None)],
        "_cold_contrib": [enter(t._cold_contrib, _T(3), None, None, None, _T(9)),
                          enter(t._cold_contrib, _T(512), None, None, None, _T(700))],
        "_dram_contrib": [enter(t._dram_contrib, _T(4), None, None, None, _T(5), None, None, None),
                          enter(t._dram_contrib, _T(5), None, None, None, _T(7), None, None, None)],
        "_dram_on_gpu": [enter(t._dram_on_gpu, _T(1), None, None, None, _T(11), None, None, None),
                         enter(t._dram_on_gpu, _T(512), None, None, None, _T(13), None, None, None)],
        "_cold_on_cpu": [enter(t._cold_on_cpu, _T(1), None, None, None, _T(0), None, None, None)],
    }
    d = {k: c[k] - base[k] for k in c}
    want = {"decode_routed": 16, "prefill_routed": 4096, "decode_nonvram": 9, "prefill_nonvram": 700,
            "decode_dram_rows": 5 + 11, "prefill_dram_rows": 7 + 13, "decode_calls": 1, "prefill_calls": 1,
            "decode_nvme_cpu_rows": 0}
    out["split"] = {k: d[k] for k in want}
    out["entered"] = states
    ok_split = (out["split"] == want and inner == [4, 5] and states["_dram_contrib"] == ["ran", "ran"]
                and all(v == "entered" for k, vs in states.items() if k != "_dram_contrib" for v in vs))
    # each wrapper's own cost: the real wrapper around a no-op original, minus the no-op itself
    o = h._pgx_dram_originals

    def noop(*a, **k):
        return None
    args = {"forward": (_T(1), _T(1, 8), None), "_cold_contrib": (_T(1), None, None, None, _T(3)),
            "_dram_contrib": (_T(1), None, None, None, _T(3), None, None, None)}
    out["per_call_ns"] = {}
    for r, a in args.items():
        keep, o[r] = o[r], noop
        try:
            diffs = []
            for _ in range(5):
                t0 = time.perf_counter_ns()
                for _ in range(calls):
                    getattr(t, r)(*a)
                t1 = time.perf_counter_ns()
                for _ in range(calls):
                    noop(t, *a)
                t2 = time.perf_counter_ns()
                diffs.append(((t1 - t0) - (t2 - t1)) / calls)
        finally:
            o[r] = keep
        out["per_call_ns"][r] = max(0.0, statistics.median(diffs))
    ta = time.perf_counter_ns()
    for _ in range(50):
        h._pgx_dram_append()
    out["append_ns"] = (time.perf_counter_ns() - ta) / 50
    per_step = LAYERS * sum(out["per_call_ns"].values()) / 1e9
    out["bound"] = per_step / STEP_S + out["append_ns"] / 1e9
    out["ok"] = bool(ok_split and out["bound"] < BOUND)
    out["python"] = sys.version.split()[0]
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
