#!/usr/bin/env python3
"""Lane P100's instruments (bench/p100/PREREG-p100.md; e4b#916). Each mode runs as its own process, started by
``p100_run.sh``:

  python p100_box.py ttft
      SC1's TTFT arm, ``bench/sc1/sc1_e4b_sched.py --ttft`` at its registered bytes (staged beside this file), unchanged:
      one warm and three timed ``max_new_tokens=1`` requests through ``ContinuousScheduler``, chunked at the env's
      ``E4B_PAGED_CHUNK_TOKENS``. The engine ``build_engine`` returns is kept. After the arm has written its record, the
      counters are installed and ONE more request on the same row runs with them on (the dispatch census): the timed
      requests run the library's functions unwrapped. With ``P100_LAUNCH_CENSUS=1``,
      one more runs under ``torch.profiler`` (CUDA activity) to count device kernels. Neither request is in the TTFT.
  python p100_box.py prof STEP_DECOMP_ARGS...
      ``bench/p39/step_decomp.py`` run as ``__main__`` with those arguments (its registered bytes; the caller passes
      ``--cprofile-out``), with the counters on for the whole process. The counters are snapshotted when step_decomp's
      cProfile window opens and closes, so the census can be compared with the profile's own ``ncalls``.
  python p100_box.py --self-test
      The pure helpers on CPU (no torch, no GPU).

The census (``P100_CENSUS_OUT``, JSON) lists every ``hot_residency._fused_over_stack`` call with:
- its routed rows R (``local_ids.numel()``) and distinct experts;
- the grouping flags it was given (``singleton_groups``, ``device_grouping``);
- whether an int4 store rode along;
- the number of ``int4_pack_ref.dequant_int4_ref`` calls made inside it.

The counters replace those two module attributes for the process. Both are looked up at call time
(``hot_residency`` calls ``_fused_over_stack`` by its global name, ``hybrid`` imports it inside the function, and the
int4 branch imports ``dequant_int4_ref`` inside the branch), so every call is seen. The library itself is not edited.
"""
from __future__ import annotations

import inspect
import json
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
#: the device-event names that are copies/fills rather than launched kernels
_COPY_PREFIXES = ("Memcpy", "Memset")


class Census:
    """Counts ``_fused_over_stack`` calls and the ``dequant_int4_ref`` calls inside them, while ``active``."""

    def __init__(self):
        self.active = False
        self.calls = []                  # [R, distinct, singleton_groups, device_grouping, store_kind, dequant_n]
        self.dequant_total = 0
        self.windows = {}

    def snapshot(self) -> dict:
        return {"calls": len(self.calls), "dequant_total": self.dequant_total}

    def install(self) -> None:
        import int4_pack_ref

        from experts4bit_qlora.engines import hot_residency as hr
        orig_dq = int4_pack_ref.dequant_int4_ref
        orig_fos = hr._fused_over_stack
        sig = inspect.signature(orig_fos)
        census = self

        def _counted_dequant_int4_ref(*a, **k):
            if census.active:
                census.dequant_total += 1
            return orig_dq(*a, **k)

        def _counted_fused_over_stack(*a, **k):
            if not census.active:
                return orig_fos(*a, **k)
            b = sig.bind(*a, **k)
            b.apply_defaults()
            ids = b.arguments["local_ids"]
            st = b.arguments["int4_stores"]
            n0 = census.dequant_total
            out = orig_fos(*a, **k)
            distinct = int(ids.unique().numel())        # a sync; the counted requests are never the timed ones
            kind = None if st is None else str(st.get("kind") or "int4_b32")
            census.calls.append([int(ids.numel()), distinct, bool(b.arguments["singleton_groups"]),
                                 bool(b.arguments["device_grouping"]), kind, census.dequant_total - n0])
            return out

        int4_pack_ref.dequant_int4_ref = _counted_dequant_int4_ref
        hr._fused_over_stack = _counted_fused_over_stack

    def watch_cprofile(self) -> None:
        """Snapshot the counters when a ``cProfile.Profile`` enables and disables (step_decomp's window)."""
        import cProfile
        census = self
        base = cProfile.Profile

        class Profile(base):
            def enable(self, *a, **k):
                census.windows["cprofile_open"] = census.snapshot()
                return super().enable(*a, **k)

            def disable(self, *a, **k):
                r = super().disable(*a, **k)
                census.windows["cprofile_close"] = census.snapshot()
                return r

        cProfile.Profile = Profile

    def record(self) -> dict:
        return {"calls": self.calls, "dequant_total": self.dequant_total, "windows": self.windows,
                "summary": summarize(self.calls, self.dequant_total)}


def summarize(calls, dequant_total) -> dict:
    """Totals by call shape. A ``loop`` call is a grouped call (not singleton-grouped) that was NOT device-grouped and
    carried an int4 store: the host-grouped int4 branch this lane is about."""
    multi = [c for c in calls if not c[2]]            # not singleton-grouped: the T > 1 calls
    single = [c for c in calls if c[2]]
    loop = [c for c in multi if not c[3] and c[4] is not None]
    inside = sum(c[5] for c in calls)
    return {
        "calls": len(calls),
        "singleton_calls": len(single),
        "singleton_dequant": sum(c[5] for c in single),
        "grouped_calls": len(multi),
        "grouped_device_calls": sum(1 for c in multi if c[3]),
        "grouped_no_store_calls": sum(1 for c in multi if c[4] is None),
        "loop_calls": len(loop),
        "loop_rows": sorted({c[0] for c in loop}),
        "loop_distinct_mean": (round(sum(c[1] for c in loop) / len(loop), 2) if loop else None),
        "loop_dequant": sum(c[5] for c in loop),
        "loop_dequant_is_two_per_distinct": all(c[5] == 2 * c[1] for c in loop),
        "dequant_total": dequant_total,
        "dequant_outside_calls": dequant_total - inside,
    }


def kernel_census(events) -> dict:
    """Device events in a torch.profiler event list: launched kernels, copies/fills, and the 12 most frequent names."""
    names = {}
    kernels = copies = 0
    for e in events:
        if str(getattr(e, "device_type", "")).split(".")[-1] != "CUDA":
            continue
        n = e.name
        if n.startswith(_COPY_PREFIXES):
            copies += 1
            continue
        kernels += 1
        names[n] = names.get(n, 0) + 1
    top = sorted(names.items(), key=lambda kv: -kv[1])[:12]
    return {"device_kernels": kernels, "device_copies": copies, "top_kernels": [[n[:120], c] for n, c in top]}


def _ttft() -> int:
    import sc1_e4b_sched as s

    from experts4bit_qlora import serve_paged
    kept = {}
    orig_build = serve_paged.build_engine

    def build_engine(cfg):
        parts = orig_build(cfg)
        kept["parts"] = parts
        return parts

    serve_paged.build_engine = build_engine
    rc = s.main(["--ttft"])
    if rc != 0 or "parts" not in kept:
        return rc or 3
    import torch
    parts = kept["parts"]
    census = Census()
    census.install()                     # AFTER the timed requests: they ran the library's code unwrapped
    rows = s.load_prompts(os.environ["SC1_PROMPTS"], 1, prompt_len=None)["prompts"]
    plen = len(rows[0])
    rec = {"arm": os.environ.get("SC1_ARM"), "prompt_tokens": plen,
           "chunk_tokens": int(os.environ["E4B_PAGED_CHUNK_TOKENS"])}
    census.active = True
    t0 = time.perf_counter()
    s.run_batch(parts, torch, rows, 1, prompt_len=plen)
    rec["census_request_wall_s"] = round(time.perf_counter() - t0, 4)
    census.active = False
    rec.update(census.record())
    if os.environ.get("P100_LAUNCH_CENSUS", "0") == "1":
        # descriptive only (the registered rule reads the counters above): a profiler failure is recorded, never fatal
        try:
            from torch.profiler import ProfilerActivity, profile
            t0 = time.perf_counter()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                s.run_batch(parts, torch, rows, 1, prompt_len=plen)
            rec["launch_request_wall_s"] = round(time.perf_counter() - t0, 4)
            rec["launches"] = kernel_census(prof.events())
        except Exception as e:  # noqa: BLE001
            rec["launches"] = {"error": repr(e)[:400]}
    json.dump(rec, open(os.environ["P100_CENSUS_OUT"], "w"), indent=1, default=str)
    sm = rec["summary"]
    print("P100_CENSUS " + json.dumps({"arm": rec["arm"], "loop_calls": sm["loop_calls"],
                                       "loop_dequant": sm["loop_dequant"], "grouped_device_calls": sm["grouped_device_calls"],
                                       "kernels": rec.get("launches", {}).get("device_kernels")}), flush=True)
    return 0


def _prof(args) -> int:
    census = Census()
    census.install()
    census.watch_cprofile()
    census.active = True
    sys.argv = [os.path.join(HERE, "step_decomp.py")] + list(args)
    rc = 0
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    census.active = False
    rec = {"arm": "prof", "argv": list(args), "rc": rc}
    rec.update(census.record())
    w = rec["windows"]
    if "cprofile_open" in w and "cprofile_close" in w:
        rec["cprofile_window_dequant"] = w["cprofile_close"]["dequant_total"] - w["cprofile_open"]["dequant_total"]
        rec["summary_cprofile_window"] = summarize(
            rec["calls"][w["cprofile_open"]["calls"]:w["cprofile_close"]["calls"]],
            rec["cprofile_window_dequant"])
    json.dump(rec, open(os.environ["P100_CENSUS_OUT"], "w"), indent=1, default=str)
    print("P100_CENSUS " + json.dumps({"arm": "prof", "rc": rc, "cprofile_window_dequant": rec.get("cprofile_window_dequant"),
                                       "loop_calls": rec["summary"]["loop_calls"]}), flush=True)
    return rc


class _Ev:
    def __init__(self, name, device_type):
        self.name, self.device_type = name, device_type


def selftest() -> int:
    calls = [[4096, 128, False, False, "int4_b32", 256], [4096, 120, False, False, "int4_b32", 240],
             [8, 8, True, False, "int4_b32", 0], [128, 64, False, True, "int4_b32", 0], [64, 30, False, False, None, 0]]
    s = summarize(calls, 497)
    assert s["calls"] == 5 and s["singleton_calls"] == 1 and s["singleton_dequant"] == 0
    assert s["grouped_calls"] == 4 and s["grouped_device_calls"] == 1 and s["grouped_no_store_calls"] == 1
    assert s["loop_calls"] == 2 and s["loop_rows"] == [4096] and s["loop_distinct_mean"] == 124.0
    assert s["loop_dequant"] == 496 and s["loop_dequant_is_two_per_distinct"] and s["dequant_outside_calls"] == 1
    calls[1][5] = 239
    assert not summarize(calls, 496)["loop_dequant_is_two_per_distinct"]
    assert summarize([], 0)["loop_distinct_mean"] is None
    ev = [_Ev("void gemv<...>", "DeviceType.CUDA"), _Ev("Memcpy DtoH", "DeviceType.CUDA"),
          _Ev("aten::mul", "DeviceType.CPU"), _Ev("void gemv<...>", "DeviceType.CUDA"), _Ev("elementwise", "DeviceType.CUDA")]
    k = kernel_census(ev)
    assert k["device_kernels"] == 3 and k["device_copies"] == 1 and k["top_kernels"][0] == ["void gemv<...>", 2], k
    print("self-test OK (3 cases)")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--self-test":
        return selftest()
    if argv and argv[0] == "ttft" and len(argv) == 1:
        sys.path.insert(0, HERE)
        return _ttft()
    if argv and argv[0] == "prof":
        sys.path.insert(0, HERE)
        return _prof(argv[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
