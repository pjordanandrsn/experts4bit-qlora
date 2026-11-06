"""PGX placement proof for e4b's ``serve_paged`` (E-hyb, E-hyb-u): the tiers the engine was BUILT with, read in-process.

The registration VOIDs an E-hyb lifetime without "the placement proof (per-layer tier byte and expert counts read after
the build)", and ``serve_paged`` reports the placement profile at ``/health`` but not the tiers. This hook is loaded as
``sitecustomize`` (so it loads in a venv too) by the server process when this directory is on PYTHONPATH. It is inert
unless ``PGX_PLACEMENT_OUT`` names a file. When set, after ``experts4bit_qlora.engines.hybrid`` imports, it wraps
``enable_hybrid_tier``: on the call ``build_engine`` makes, it writes the manifest it was handed (tier lists keyed by
the arena's layer ids, after the all-vram override) as per-layer expert counts per tier, with the arena path, then
calls through unchanged. ``build_engine`` imports the function inside its body, so the wrapped attribute is the one
it calls. Fails loud: if enabled and the function is missing, it raises at import.

Amendment 3: the tier counters. The registration read E-hyb's DRAM-tier host time from "the step trace's host-tier time"
and its VRAM-tier hit rate from "the step trace and the tier counters"; e4b's step trace carries neither (a decode step's
``dec_issue`` covers both tiers) and ``/health`` has no tier counters. When ``PGX_DRAM_TIME_OUT`` names a file, the hook
wraps the five ``_HybridTier`` routes a routed expert row can take (``ROUTES``), each counting by shape only, with no
sync (every count is ``numel()`` of an index the method already holds on the host):
- ``forward`` (the per-layer entry): ``routed``, every routed slot of the call (``top_k_index.numel()``);
- ``_cold_contrib`` (the dispatcher of every non-VRAM row): ``nonvram`` (``cr.numel()``);
- ``_dram_contrib`` (the DRAM bus, computed in place on the host): ``dram_rows``, and its wall as ``ns`` / ``calls``
  (a ``perf_counter_ns`` pair; the method's own host copy is the only sync, as in e4b's own ``amort["dram_ns"]``);
- ``_dram_on_gpu`` (the thin / offload route: DRAM experts computed on the GPU): ``dram_rows``;
- ``_cold_on_cpu`` (NVMe experts on the host; the NVMe tier is empty in stage 1): ``nvme_cpu_rows``.
A call with at most ``PGX_MAX_DECODE_ROWS`` (default 4) token rows counts as decode, the rest as prefill (``decode_*`` /
``prefill_*`` keys). A daemon thread appends the cumulative counters, with a wall epoch ``t``, to that file once a
second. ``pgx_box.e_tiers`` takes per-pass deltas over the runner's pass windows: the DRAM-tier host seconds per decode
token, and the VRAM-tier hit rate 1 - nonvram / routed, each behind its own invariant (routed = the pass's decode tokens x
MoE layers x top_k, read from the box's ``/health``). It loads only in the e4b arms.
"""
import os

_OUT = os.environ.get("PGX_PLACEMENT_OUT")
_DRAM = os.environ.get("PGX_DRAM_TIME_OUT")
if _OUT or _DRAM:
    import importlib.abc
    import importlib.util
    import json
    import sys

    _HYBRID = "experts4bit_qlora.engines.hybrid"

    def _record(arena, man) -> dict:
        tiers = man.get("tiers") or {}
        per = {}
        for tier in ("vram", "dram", "nvme"):
            for lay, _e in tiers.get(tier) or []:
                row = per.setdefault(str(int(lay)), {"vram": 0, "dram": 0, "nvme": 0})
                row[tier] += 1
        return {"arena": str(arena), "per_layer": per,
                "experts": {t: sum(r[t] for r in per.values()) for t in ("vram", "dram", "nvme")},
                "masses": man.get("masses"), "profile": man.get("profile")}

    ROUTES = ("forward", "_cold_contrib", "_dram_contrib", "_dram_on_gpu", "_cold_on_cpu")
    KEYS = ("ns", "calls", "routed", "nonvram", "dram_rows", "nvme_cpu_rows")

    def _patch_dram_timer(mod) -> None:
        import threading
        import time
        cls = getattr(mod, "_HybridTier", None)
        missing = [r for r in ROUTES if cls is None or r not in cls.__dict__]
        if missing:
            raise RuntimeError(f"PGX hook: {_HYBRID}._HybridTier lacks {missing}; the tier counters cannot be wired")
        ns = time.perf_counter_ns
        max_decode = int(os.environ.get("PGX_MAX_DECODE_ROWS", "4"))
        c = {f"{ph}_{k}": 0 for ph in ("decode", "prefill") for k in KEYS}
        o = {r: cls.__dict__[r] for r in ROUTES}

        def _ph(rows):
            return "decode_" if rows <= max_decode else "prefill_"

        def forward(self, hidden_states, top_k_index, *a, **kw):
            c[_ph(hidden_states.shape[0]) + "routed"] += top_k_index.numel()
            return o["forward"](self, hidden_states, top_k_index, *a, **kw)

        def _cold_contrib(self, x, flat, row_token, row_slot, cr, *a, **kw):
            c[_ph(x.shape[0]) + "nonvram"] += cr.numel()
            return o["_cold_contrib"](self, x, flat, row_token, row_slot, cr, *a, **kw)

        def _dram_contrib(self, x, flat, row_token, row_slot, dr, *a, **kw):
            t0 = ns()
            try:
                return o["_dram_contrib"](self, x, flat, row_token, row_slot, dr, *a, **kw)
            finally:
                ph = _ph(x.shape[0])
                c[ph + "ns"] += ns() - t0
                c[ph + "calls"] += 1
                c[ph + "dram_rows"] += dr.numel()

        def _dram_on_gpu(self, x, flat, row_token, row_slot, dr, *a, **kw):
            c[_ph(x.shape[0]) + "dram_rows"] += dr.numel()
            return o["_dram_on_gpu"](self, x, flat, row_token, row_slot, dr, *a, **kw)

        def _cold_on_cpu(self, x, flat, row_token, row_slot, nr, *a, **kw):
            c[_ph(x.shape[0]) + "nvme_cpu_rows"] += nr.numel()
            return o["_cold_on_cpu"](self, x, flat, row_token, row_slot, nr, *a, **kw)
        for r, fn in (("forward", forward), ("_cold_contrib", _cold_contrib), ("_dram_contrib", _dram_contrib),
                      ("_dram_on_gpu", _dram_on_gpu), ("_cold_on_cpu", _cold_on_cpu)):
            fn.__wrapped__ = o[r]
            fn.__qualname__ = f"pgx_hook.{r}"
            setattr(cls, r, fn)
        mod._pgx_dram_counters = c
        mod._pgx_dram_originals = o      # the bench swaps an original for a no-op to time the wrapper itself

        def _append():
            row = dict(c, t=time.time())
            with open(_DRAM, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        def _loop():
            while True:
                _append()
                time.sleep(float(os.environ.get("PGX_DRAM_TIME_PERIOD_S", "1")))
        mod._pgx_dram_append = _append
        threading.Thread(target=_loop, name="pgx-dram-time", daemon=True).start()

    def _patch(mod) -> None:
        if _DRAM:
            _patch_dram_timer(mod)
        if not _OUT:
            return
        orig = getattr(mod, "enable_hybrid_tier", None)
        if orig is None:
            raise RuntimeError(f"PGX hook: {_HYBRID}.enable_hybrid_tier not found; the placement proof cannot be read")

        def enable_hybrid_tier(model, arena, man, *a, **kw):
            rec = _record(arena, man)
            tmp = _OUT + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1, default=str)
            os.replace(tmp, _OUT)
            return orig(model, arena, man, *a, **kw)
        mod.enable_hybrid_tier = enable_hybrid_tier

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _HYBRID:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                _patch(module)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
