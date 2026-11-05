"""PGX routing dump for SGLang Paged Experts (the instrument pass only; never on a timed pass).

Loaded by every Python process of the SGLang server when this directory is on PYTHONPATH (as ``usercustomize``).
Inert unless ``PGX_ROUTE_DUMP`` names a directory. When set, after SGLang's paged-experts modules import, it wraps:

- ``PagedExpertsMoEMethod.apply``: one JSON line per MoE call to ``<dir>/route-<pid>.jsonl``, carrying the layer, the
  token count, ``topk_ids`` (a device -> host copy: a sync, which is why this runs only on the untimed instrument
  pass, with CUDA graphs off), and whether the call is decode (``tokens <= PGX_MAX_DECODE_ROWS``, default 4: stage 1
  runs c <= 4 and every prompt is one 1,024-token prefill chunk, so no prefill call has that few rows);
- ``executor.plan_waves`` (the host path's planner, imported by name into the executor): the call's own slot loads,
  recorded as ``host_loads``, the dump's self-check (``pgx_residency.host_path_check``).

The pass id comes from ``PGX_PASS`` re-read on each call from ``<dir>/PASS`` (the driver writes the id there before a
pass), so one server lifetime can hold several passes. Fails loud: if the hook is enabled and the modules it patches
do not have the expected names, it raises at import.
"""
import os

_DUMP = os.environ.get("PGX_ROUTE_DUMP")
if _DUMP:
    import importlib.abc
    import importlib.util
    import json
    import sys
    import threading

    _METHOD = "sglang.srt.layers.moe.paged_experts.method"
    _EXEC = "sglang.srt.layers.moe.paged_experts.executor"
    _MAX_DECODE = int(os.environ.get("PGX_MAX_DECODE_ROWS", "4"))
    _local = threading.local()
    _lock = threading.Lock()
    _state = {"call": 0, "fh": None, "layer_ids": {}}

    def _pass_id() -> str:
        try:
            with open(os.path.join(_DUMP, "PASS"), encoding="utf-8") as f:
                return f.read().strip() or "unset"
        except FileNotFoundError:
            return "unset"

    def _write(row: dict) -> None:
        with _lock:
            if _state["fh"] is None:
                os.makedirs(_DUMP, exist_ok=True)
                _state["fh"] = open(os.path.join(_DUMP, f"route-{os.getpid()}.jsonl"), "a", encoding="utf-8")
            _state["call"] += 1
            row["call"] = _state["call"]
            _state["fh"].write(json.dumps(row) + "\n")
            _state["fh"].flush()

    def _layer_id(method, layer) -> int:
        lid = getattr(layer, "layer_id", None)
        if isinstance(lid, int):
            return lid
        ids = _state["layer_ids"]                 # fallback: order of first call, which is model order
        return ids.setdefault(id(method), len(ids))

    def _patch_executor(mod) -> None:
        orig = getattr(mod, "plan_waves", None)
        if orig is None:
            raise RuntimeError(f"PGX hook: {_EXEC}.plan_waves not found; the dump's self-check cannot be wired")

        def plan_waves(*a, **kw):
            waves = orig(*a, **kw)
            _local.host_loads = (getattr(_local, "host_loads", None) or 0) + sum(len(w.loads) for w in waves)
            return waves
        mod.plan_waves = plan_waves

    def _patch_method(mod) -> None:
        cls = getattr(mod, "PagedExpertsMoEMethod", None)
        if cls is None or not hasattr(cls, "apply"):
            raise RuntimeError(f"PGX hook: {_METHOD}.PagedExpertsMoEMethod.apply not found")
        orig = cls.apply

        def apply(self, layer, dispatch_output):
            ids = dispatch_output.topk_output.topk_ids
            _local.host_loads = None
            out = orig(self, layer, dispatch_output)
            host = getattr(_local, "host_loads", None)
            rows = ids.reshape(ids.shape[0], -1).tolist()
            _write({"pass": _pass_id(), "layer": _layer_id(self, layer), "tokens": len(rows),
                    "ids": rows, "decode": len(rows) <= _MAX_DECODE, "host_loads": host})
            return out
        cls.apply = apply

    class _Finder(importlib.abc.MetaPathFinder):
        _patches = {_METHOD: _patch_method, _EXEC: _patch_executor}

        def find_spec(self, name, path, target=None):
            patch = self._patches.get(name)
            if patch is None:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module):
                orig_exec(module)
                patch(module)
            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
