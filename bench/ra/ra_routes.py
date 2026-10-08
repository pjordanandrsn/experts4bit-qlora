"""Source-bound QKV and NF4 GEMV coverage for the registered all-VRAM battery.

Python eager/capture observations only. A successful wrapper return is not an
on-card premise or proof of graph replay. Unknown native module bodies refuse.
"""
from __future__ import annotations

import ast
import copy
import functools
import inspect
import json
import sys
from pathlib import Path

import ra_fallback

PRIMARY = ("dotpad", "dotpad_splitk", "scalar", "scalar_splitk", "bw_tree", "bw_prmt32")
COUNTERS = set(PRIMARY) | {"bw_splitk"}


def module_digest(module):
    return ra_fallback.ast_digest(ast.parse(inspect.getsource(module)))


def counters(nf4):
    out = nf4.dispatch_counts()
    if set(out) != COUNTERS or any(type(v) is not int or v < 0 for v in out.values()):
        raise ValueError("unknown GEMV counter schema")
    return out


class RouteObserver:
    def __init__(self, nf4, hot, qkv, *, adapters=None):
        data = adapters or json.loads(Path(__file__).with_name("route-adapters.json").read_text())
        if data["schema"] != 2:
            raise ValueError("unknown route adapter schema")
        for name, mod in (("nf4", nf4), ("hot", hot), ("qkv", qkv)):
            if not any(r["module"] == name and r["ast_sha256"] == module_digest(mod) for r in data["modules"]):
                raise ValueError("unreviewed route module body")
        counters(nf4)
        self.nf4, self.hot, self.qkv = nf4, hot, qkv
        self.active, self.modules, self.projections = None, {}, {}
        self.saved = []
        self.unscoped_calls = 0
        self.model_saved, self.hooks = [], []

    @classmethod
    def native(cls, nf4):
        from experts4bit_qlora.engines import hot_residency, qkv_fuse
        return cls(nf4, hot_residency, qkv_fuse)

    def start(self):
        self.replace(self.nf4, "gemm_4bit_grouped", self.observe_gemm)
        self.replace(self.hot, "_seen_route", self.observe_route)
        return self

    def replace(self, obj, name, wrapper):
        original = getattr(obj, name)
        if getattr(original, "__globals__", None) is not vars(obj) or original.__name__ != name or \
                Path(original.__code__.co_filename).resolve() != Path(obj.__file__).resolve():
            raise ValueError("foreign route callable")
        self.saved.append((obj, name, original))
        setattr(obj, name, functools.wraps(original)(lambda *a, **k: wrapper(original, *a, **k)))

    def close(self):
        for obj, name, original in reversed(self.saved):
            setattr(obj, name, original)
        self.saved.clear()
        for obj, name, original in reversed(self.model_saved):
            setattr(obj, name, original)
        self.model_saved.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def candidates(self, model):
        paths = []
        for path, mod in model.named_modules():
            children = {n for n, _ in mod.named_children()}
            if children == {"q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"} and \
                    hasattr(mod, "head_dim") and all(getattr(mod, n).weight.numel() == mod.head_dim for n in ("q_norm", "k_norm")):
                paths.append(path)
        return paths

    def assemble(self, model, candidates, modes):
        for path, mod in model.named_modules():
            if hasattr(mod, "qkv_proj"):
                if path not in candidates or any(hasattr(mod, n) for n in ("q_proj", "k_proj", "v_proj")):
                    raise ValueError("unbound or half-fused QKV module")
                bound = mod.forward
                if getattr(bound, "__func__", None) is not self.qkv._fused_forward:
                    if getattr(bound, "__qualname__", None) != "_patch_attention.<locals>._fwd":
                        raise ValueError("unknown fused QKV forward")
                    retained = inspect.signature(bound).parameters["_orig"].default
                    if getattr(retained, "__func__", None) is not self.qkv._fused_forward:
                        raise ValueError("QKV retains an unfused original")
                self.projections[path] = 0

                def called(module, args, output, _path=path):
                    self.projections[_path] += 1

                self.hooks.append(mod.qkv_proj.register_forward_hook(called))
            if hasattr(mod, "_hot_residency"):
                state = mod._hot_residency
                if not hasattr(mod, "_e4b_hot_ref") or state.device.type != "cuda" or \
                        state.hot_ids.numel() != mod.num_experts or getattr(state, "_int4_stores", None) is not None:
                    raise ValueError("unregistered MoE placement/store")
                rec = dict(single_calls=0, batch_calls=0, reference_single=0, reference_batch=0,
                           gemv_calls=0, route_mismatches=0, selected_routes={}, gemv_routes={})
                self.modules[path] = rec
                original, reference = mod.forward, mod._e4b_hot_ref
                if original.__globals__ is not vars(self.hot) or original.__qualname__ != "enable_hot_residency.<locals>._fwd":
                    raise ValueError("unknown MoE forward binding")
                self.model_saved.extend(((mod, "forward", original), (mod, "_e4b_hot_ref", reference)))

                def ref(*a, _orig=reference, _rec=rec, **k):
                    if self.active is None or self.active[0] is not _rec:
                        raise ValueError("unscoped reference fallback")
                    _rec["reference_single" if self.active[1] == 1 else "reference_batch"] += 1
                    return _orig(*a, **k)

                def forward(hidden, *a, _orig=original, _rec=rec, **k):
                    if self.active is not None or len(hidden.shape) != 2 or hidden.shape[0] < 1:
                        raise ValueError("nested/unknown MoE observation")
                    n = hidden.shape[0]
                    _rec["single_calls" if n == 1 else "batch_calls"] += 1
                    before = _rec["gemv_calls"]
                    ref_before = _rec["reference_single"]
                    self.active = (_rec, n)
                    try:
                        out = _orig(hidden, *a, **k)
                        if n == 1 and _rec["reference_single"] == ref_before and _rec["gemv_calls"] - before != 2:
                            raise ValueError("singleton MoE lacks two observed GEMV projections")
                        return out
                    finally:
                        self.active = None

                mod._e4b_hot_ref, mod.forward = ref, forward
        expected = 0 if modes["E4B_PAGED_FUSE_QKV"] == "0" else len(candidates)
        if len(self.projections) != expected:
            raise ValueError("QKV candidate coverage incomplete")

    def observe_route(self, original, route, rows):
        if self.active is not None:
            rec, _ = self.active
            rec["selected_routes"][route] = rec["selected_routes"].get(route, 0) + 1
            if route not in ("nf4_singleton", "nf4_mtile_host", "nf4_mtile_captured", "nf4_k25"):
                raise ValueError("unregistered all-VRAM NF4 route")
        return original(route, rows)

    def expected(self, N, K, T, device):
        n = self.nf4
        sm = n._sm_count(device)
        if n._bw() == "1" or (n._bw() == "auto" and (N, K) in n._BW_SHAPES and sm >= 160):
            # _bw_decode's reviewed default: compiled NVIDIA, no interpreter.
            return "bw_prmt32", int(n._bw_plan(N, K)[3] > 1)
        if n._dotpad() and (N, K) in n._DOTPAD_CONFIGS and sm >= 160:
            sk = n._splitk_plan_sk(N, K)
            return "dotpad_splitk" if sk is not None and sk > 1 else "dotpad", 0
        return "scalar_splitk" if n._decode_plan(N, K, T, sm)[2] > 1 else "scalar", 0

    def observe_gemm(self, original, *args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        v = bound.arguments
        if self.active is None:
            self.unscoped_calls += 1
            return original(*args, **kwargs)
        if any(v.get(k) is not None for k in ("decode_config", "split_k", "bw_config")):
            raise ValueError("GEMV plan override")
        rec, tokens = self.active
        sizes = v["sizes"]
        if not isinstance(sizes, (list, tuple)) or not sizes or any(type(x) is not int or x < 1 for x in sizes):
            raise ValueError("unresolved native group sizes")
        expected, split = None, None
        if max(sizes) == 1:
            T, K = v["a_cat"].shape
            expected, split = self.expected(v["B"].shape[1], K, T, v["a_cat"].device)
        before = counters(self.nf4)
        out = original(*args, **kwargs)
        changes = {k: counters(self.nf4)[k] - before[k] for k in before}
        if any(x < 0 for x in changes.values()):
            raise ValueError("GEMV counters reset")
        if max(sizes) == 1:
            if sum(changes[k] for k in PRIMARY) != 1 or changes["bw_splitk"] != split:
                raise ValueError("missing/duplicate GEMV dispatch")
            actual = next(k for k in PRIMARY if changes[k])
            rec["gemv_calls"] += 1
            rec["gemv_routes"][actual] = rec["gemv_routes"].get(actual, 0) + 1
            rec["route_mismatches"] += int(actual != expected)
            if tokens != 1:
                raise ValueError("batched MoE unexpectedly dispatched singleton GEMV")
        elif any(changes.values()):
            raise ValueError("prefill unexpectedly dispatched GEMV")
        return out

    def snapshot(self):
        return dict(modules=copy.deepcopy(self.modules), qkv=copy.deepcopy(self.projections))

    def evidence(self, info, qkv_calls, before=None):
        now = self.snapshot()
        before = before or dict(modules={}, qkv={})
        if len(now["modules"]) != info["moe_layers"] or len(now["qkv"]) != info["fuse_qkv_n"]:
            raise ValueError("native MoE/QKV census differs from observed coverage")
        if before["modules"] and set(before["modules"]) != set(now["modules"]) or \
                before["qkv"] and set(before["qkv"]) != set(now["qkv"]):
            raise ValueError("route snapshot coverage changed")
        qcalls = sum(v - before["qkv"].get(p, 0) for p, v in now["qkv"].items())
        if qcalls != qkv_calls or any(v <= before["qkv"].get(p, 0) for p, v in now["qkv"].items()):
            raise ValueError("QKV call accounting disagreement")
        fields = ("single_calls", "batch_calls", "reference_single", "reference_batch", "gemv_calls", "route_mismatches")
        for path, rec in now["modules"].items():
            old = before["modules"].get(path, dict.fromkeys(fields, 0))
            if any(type(rec[k]) is not int or type(old[k]) is not int or not 0 <= old[k] <= rec[k] for k in fields):
                raise ValueError("route counters reset")
            if sum(rec[k] - old[k] for k in ("single_calls", "batch_calls")) <= 0:
                raise ValueError("unobserved MoE module")
            if any(rec["reference_" + k] - old["reference_" + k] > rec[k + "_calls"] - old[k + "_calls"] for k in ("single", "batch")):
                raise ValueError("reference fallback accounting disagreement")
        totals = {k: sum(r[k] - before["modules"].get(p, {}).get(k, 0) for p, r in now["modules"].items()) for k in fields}
        if any(v < 0 for v in totals.values()):
            raise ValueError("route counters reset")
        return dict(qkv=dict(fallback_calls=0, calls=qcalls, observed_modules=len(now["qkv"]),
                             basis="complete source-bound fusion; no unfused projections retained"),
                    decode_gemv=dict(fallback_calls=totals["reference_single"] + totals["reference_batch"] + totals["route_mismatches"],
                                     calls=totals["gemv_calls"], observed_modules=len(now["modules"]), **totals))


def observed(fn):
    """One component owns/restores every temporary route hook, even on failure."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        bound = inspect.signature(fn).bind(*args, **kwargs)
        if "routes" in bound.arguments:
            raise ValueError("external route observer override")
        routes = RouteObserver.native(bound.arguments["nf4"])
        try:
            routes.start()
            bound.arguments["routes"] = routes
            return fn(*bound.args, **bound.kwargs)
        except BaseException:
            print(json.dumps({"status": "FAILED_ROUTE_OBSERVATION", "partial_route_snapshot": routes.snapshot(),
                              "proves_gpu_engagement": False}, allow_nan=False), file=sys.stderr)
            raise
        finally:
            routes.close()
    return run
