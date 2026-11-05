"""PGX placement proof for e4b's ``serve_paged`` (E-hyb, E-hyb-u): the tiers the engine was BUILT with, read in-process.

The registration VOIDs an E-hyb lifetime without "the placement proof (per-layer tier byte and expert counts read after
the build)", and ``serve_paged`` reports the placement profile at ``/health`` but not the tiers. This hook is loaded as
``sitecustomize`` (so it loads in a venv too) by the server process when this directory is on PYTHONPATH. It is inert
unless ``PGX_PLACEMENT_OUT`` names a file. When set, after ``experts4bit_qlora.engines.hybrid`` imports, it wraps
``enable_hybrid_tier``: on the call ``build_engine`` makes, it writes the manifest it was handed (tier lists keyed by
the arena's layer ids, after the all-vram override) as per-layer expert counts per tier, with the arena path, then
calls through unchanged. ``build_engine`` imports the function inside its body, so the wrapped attribute is the one
it calls. Fails loud: if enabled and the function is missing, it raises at import.
"""
import os

_OUT = os.environ.get("PGX_PLACEMENT_OUT")
if _OUT:
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

    def _patch(mod) -> None:
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
