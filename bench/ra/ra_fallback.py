"""Observe release glue fallbacks; changed/unknown forward bodies fail closed.

AST adapters bind the complete forward body, not a name-based family license.
Tensor metadata only: no device values, synchronizations or profilers. Python
counts cover eager/capture calls; graph replay needs separate native evidence.
"""
from __future__ import annotations

import ast
import copy
import functools
import hashlib
import inspect
import json
import textwrap
import types
from pathlib import Path

FEATURES = ("rms_glue", "residual_glue", "rope_glue", "router_epilogue")
MODULES = {"experts4bit_qlora.engines." + m for m in ("glue_fuse", "glue_r2", "router_epilogue")}


def fingerprint(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("unsupported forward source")
    return hashlib.sha256(ast.dump(tree.body[0], include_attributes=False).encode()).hexdigest()


def rows(args, kwargs):
    x = args[0] if args else kwargs.get("hidden_states")
    if x is None or not x.shape or x.shape[-1] <= 0 or x.numel() % x.shape[-1]:
        raise ValueError("unresolved fallback input shape")
    n = x.numel() // x.shape[-1]
    if n <= 0:
        raise ValueError("empty fallback input")
    return n


def clone_orig(fn, observer):
    """Replace the retained original in a private function clone only."""
    defaults = list(fn.__defaults__ or ())
    names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    suffix = names[len(names) - len(defaults):] if defaults else ()
    kw = dict(fn.__kwdefaults__ or {})
    places = int("_orig" in suffix) + int("_orig" in kw)
    if places != 1:
        raise ValueError("one retained original required")
    orig = defaults[suffix.index("_orig")] if "_orig" in suffix else kw["_orig"]
    if not callable(orig):
        raise ValueError("retained original is not callable")

    @functools.wraps(orig)
    def fallback(*args, **kwargs):
        observer["small_fallbacks" if rows(args, kwargs) <= 64 else "large_fallbacks"] += 1
        return orig(*args, **kwargs)

    if "_orig" in suffix:
        defaults[suffix.index("_orig")] = fallback
    else:
        kw["_orig"] = fallback
    clone = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, tuple(defaults), fn.__closure__)
    clone.__kwdefaults__ = kw
    clone.__dict__.update(fn.__dict__)
    clone.__module__, clone.__qualname__ = fn.__module__, fn.__qualname__
    clone.__annotations__ = fn.__annotations__
    return clone


class GlueObserver:
    def __init__(self, adapters=None):
        data = adapters or json.loads(Path(__file__).with_name("fallback-adapters.json").read_text())
        if data["schema"] != 1:
            raise ValueError("unknown fallback adapter schema")
        self.adapters = {(r["module"], r["qualname"], r["ast_sha256"]): r["feature"] for r in data["adapters"]}
        if len(self.adapters) != len(data["adapters"]) or any(role not in FEATURES for role in self.adapters.values()):
            raise ValueError("invalid or duplicate fallback adapter")
        self.records = {}
        self.installed = False

    def install(self, model):
        if self.installed:
            raise ValueError("fallback observer already installed")
        self.installed = True
        for path, mod in model.named_modules():
            fn = mod.forward
            # Other model/module forwards are outside this adapter's claimed coverage.
            if getattr(fn, "__module__", None) not in MODULES:
                continue
            if not isinstance(fn, types.FunctionType):
                raise ValueError("unsupported glue forward binding")
            key = (fn.__module__, fn.__qualname__, fingerprint(fn))
            if key not in self.adapters:
                raise ValueError("unreviewed glue forward body")
            role = self.adapters[key]
            rec = {"feature": role, "ast_sha256": key[2], "small_calls": 0, "large_calls": 0,
                   "small_fallbacks": 0, "large_fallbacks": 0}
            self.records[path] = rec
            cloned = clone_orig(fn, rec)

            @functools.wraps(fn)
            def counted(*args, _fn=cloned, _rec=rec, **kwargs):
                _rec["small_calls" if rows(args, kwargs) <= 64 else "large_calls"] += 1
                return _fn(*args, **kwargs)

            mod.forward = counted
        return self

    def snapshot(self):
        if not self.installed:
            raise ValueError("uninstalled fallback observer")
        return copy.deepcopy(self.records)

    def evidence(self, census, before=None):
        if set(census) != set(FEATURES) or any(type(v) is not int or v < 0 for v in census.values()):
            raise ValueError("invalid native census")
        current = self.snapshot()
        before = before or {}
        if before and set(before) != set(current):
            raise ValueError("fallback module coverage changed")
        for p, rec in before.items():
            if set(rec) != set(current[p]) or any(rec[k] != current[p][k] for k in ("feature", "ast_sha256")):
                raise ValueError("fallback snapshot identity changed")
        result = {}
        for role in FEATURES:
            modules = {p: r for p, r in current.items() if r["feature"] == role}
            if len(modules) != census[role]:
                raise ValueError("native census differs from observed forward coverage")
            keys = ("small_calls", "large_calls", "small_fallbacks", "large_fallbacks")
            for path, rec in modules.items():
                old = before.get(path, dict.fromkeys(keys, 0))
                if any(type(rec[k]) is not int or type(old[k]) is not int or not 0 <= old[k] <= rec[k] for k in keys):
                    raise ValueError("contradictory fallback accounting")
                for kind in ("small", "large"):
                    if rec[kind + "_fallbacks"] - old[kind + "_fallbacks"] > rec[kind + "_calls"] - old[kind + "_calls"]:
                        raise ValueError("contradictory per-module fallback accounting")
            totals = {k: sum(r[k] - before.get(p, {}).get(k, 0) for p, r in modules.items())
                      for k in ("small_calls", "large_calls", "small_fallbacks", "large_fallbacks")}
            if any(type(v) is not int or v < 0 for v in totals.values()) or any(
                    totals[f"{kind}_fallbacks"] > totals[f"{kind}_calls"] for kind in ("small", "large")):
                raise ValueError("contradictory fallback accounting")
            result[role] = {"observed_modules": len(modules), **totals,
                            "fallback_calls": totals["small_fallbacks"],
                            "scope": "Python eager/capture; large-row prefill reported separately"}
        return result


def resolved_defaults(server, cfg):
    """Reconstruct release defaults independently of assembly's reported modes."""
    fresh = server.PagedServeConfig.from_env()
    actual = {k: v for k, v in vars(cfg).items() if k != "token"}
    expected = {k: v for k, v in vars(fresh).items() if k != "token"}
    if actual != expected:
        raise ValueError("configuration differs from independently reconstructed release defaults")
    required = {"E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI"}
    modes = actual.get("fusion_modes")
    if not isinstance(modes, dict) or set(modes) != required or any(v not in ("auto", "0", "1") for v in modes.values()):
        raise ValueError("unresolved release fusion defaults")
    return copy.deepcopy(actual)


def check_fold_reports(info):
    """Auto kernel gaps are failures, not an inapplicable family or an off mode."""
    fields = {"E4B_FUSE_T1_GLUE": info["fuse_t1_glue_n"],
              "E4B_FUSE_T1_GLUE_R2": list(info["fuse_t1_glue_r2_n"]),
              "E4B_FUSE_ROUTER_EPI": info["fuse_router_epilogue_n"]}
    reports = info["fusion_report"]
    if set(reports) != set(fields):
        raise ValueError("missing native fold reports")
    for name, patched in fields.items():
        report = reports[name]
        if report.get("mode") != info["fusion_modes"][name]:
            raise ValueError("fold report mode contradiction")
        if "skipped" in report or any(report.get(k, 0) for k in ("kernel_gaps", "no_kernel_mode")):
            raise ValueError("fold kernel coverage gap")
        if report["mode"] == "0" and patched not in (0, [0, 0]):
            raise ValueError("off fold has patched modules")
        if report["mode"] != "0" and report.get("patched") != patched:
            raise ValueError("fold report census contradiction")
