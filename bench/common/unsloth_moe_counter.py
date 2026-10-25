#!/usr/bin/env python3
"""unsloth_moe_counter.py -- counts the expert GEMMs Unsloth's MoE QLoRA actually dispatches, on both of its grouped paths.

Shared by every lane that times Unsloth's MoE training (P237's offload head-to-head imports it unchanged; P238's NF4-fused
head-to-head too). One implementation, so the two lanes cannot count differently.

What it counts, in the release-proven unsloth_zoo 2026.10.8 (wheel sha256 db8c283b..., the pos79 box's release proof):
- The generic Triton grouped GEMM: ``unsloth_zoo.temporary_patches.moe_grouped_fp16.generic_grouped_mm`` (line 741: forward and dX
  of every expert GEMM the NF4-fused path does not take) and ``generic_grouped_wgrad`` (754: the LoRA dW). Every caller resolves the
  name through the module at call time (moe_utils.py:602 and 1408 ``mg.generic_grouped_mm``, 615 ``mg.generic_grouped_wgrad``), so
  wrapping the module attribute catches them all.
- The library's own self-check calls both functions through module globals (moe_grouped_fp16.py:788-790, from
  ``_self_check_generic`` at 771, run once per device by ``triton_grouped_available`` at 819). Those calls are counted SEPARATELY
  (``selfcheck_gemm`` / ``selfcheck_wgrad``), never as training GEMMs. The library restores its own ``GENERIC_CALLS`` around the
  self-check (816, 834) for the same reason.
- The NF4-fused grouped GEMM (unsloth-zoo#1935, merge commit 4fafdb26, ``UNSLOTH_MOE_NF4_FUSED``): ``maybe_nf4_grouped_mm``
  (moe_nf4_grouped.py:521), imported at call time by moe_utils.py:1188. A call that returns a tensor took the fused path
  (``nf4_fused_calls``); one that returns None kept the dequant path (``nf4_declined``). The library's ``NF4_FUSED_STATS``
  (moe_nf4_grouped.py:50) is read as a delta: ``launches`` (+= 1 at 269 inside the custom op, -= 1 at 275 when a launch raises and
  the call decodes eagerly, uncounted) is the compile-safe counter; ``calls`` (546) and ``fallbacks`` (531) count only in eager mode.

What binds, and what doesn't:
- A fused arm binds on ``NF4_FUSED_STATS.launches`` == the expected number of fused base GEMMs, exact (forward + dX per projection;
  the lane computes the number). That equality catches the silent paths: a shape mismatch (moe_nf4_grouped.py:534-537) and a
  ``FAILED_SHAPES`` entry (538-542), which return None with no counter, and the failed-launch decrement (275), whose eager decode is
  uncounted.
- ``auto`` declining above its row cap (523) is that arm's DESIGN, not a fault. So the expected count of an auto arm admits only the
  calls the LIBRARY's own predicate admits: on every ``maybe_nf4_grouped_mm`` call under mode ``auto``, the wrapper evaluates the
  library's ``nf4_auto_fused(inputs.shape[0], offsets.shape[0])`` (rows <= ``nf4_auto_max_rows()`` x E, lines 69-81) on that call's
  own shapes and counts ``nf4_auto_eligible`` / ``nf4_auto_ineligible``. :func:`fused_calls_expected` is the eligible count for auto
  and every call for ``1``; the lane's own count from its routed rows must agree (:func:`check_equal`). At a fixed op-level shape
  every call has the same rows, so auto is all-or-none (rows = tokens x 8 for top-8: fused at <= 4,096 tokens, 0 launches above).
- A ``UNSLOTH_MOE_NF4_FUSED=0`` arm binds on ``launches == 0`` and ``calls == 0``.
- Every Unsloth MoE knob an arm depends on is set explicitly to its registered value, and :func:`check_env` refuses any other value
  or any unregistered ``UNSLOTH_MOE_*`` key in the arm's environment (an ambient ``UNSLOTH_MOE_NF4_AUTO_MAX_ROWS`` moves auto's cap;
  ``UNSLOTH_MOE_NF4_AUTOTUNE=1`` retunes tiles per shape, line 316).
- The wrapper counts are binding only BESIDE a counterpart that runs per call even when Python doesn't: the profiler's kernel counts
  (F15 v2) or ``NF4_FUSED_STATS.launches``. The two are required equal (:func:`check_equal`). Why: a Python wrapper runs only when
  Python runs. Under ``torch.compile`` (dynamo, torch 2.14) the dict increments are replayed on every call, so the wrapper DOES count
  per call there (measured on CPU, backend eager and aot_eager). But a traced or graph-captured block (``torch.jit.trace``, CUDA-graph
  replay, or a ``torch.export`` program) runs the wrapper once, at capture, and never again, while the library's in-op counter
  keeps counting. The tests show it on an exported program.
- A caller that bound the function to a local name BEFORE :func:`install` bypasses the wrapper. The cross-check against the
  library's own ``GENERIC_CALLS`` catches that (:func:`check_library`), and the tests show it.
- ``generic_grouped_mm``, ``generic_grouped_wgrad`` and ``maybe_nf4_grouped_mm`` resolve by absolute import, so the compile-cache copy
  of moe_utils (``_load_cached_moe_utils_module``, moe_utils.py:297) reaches the same wrapped objects.

The per-call Triton-or-torch dispatch (moe_utils.py:572 ``_triton_grouped_mm_wanted``; the table at 503-508, the policy at 521-558):
- Every grouped expert GEMM asks it at moe_utils.py:378 (``_grouped_mm_with_backward_fix``). ``_lora_rank_pad_needed`` (3004) asks it
  too, as a padding probe, not a GEMM; the wrapper tells the two apart by the calling function and counts only 378 as a decision.
- It is a moe_utils GLOBAL, so it is NOT resolved by absolute import: ``get_forward_moe_backend`` (348-356) returns the compile-cache
  copy's ``forward_moe_backend`` when one is present and current, and that copy calls its OWN global. :func:`install` therefore patches
  the predicate in the package module AND in the globals of the function ``get_forward_moe_backend()`` returns, and records which
  object served (``served``: its ``__name__``, ``__file__``, and whether it is the package module).
- Each decision is recorded by site, class (the library's own ``_triton_grouped_mm_kind``) and outcome, with whether grad was enabled
  and whether the input / weight required grad; the resolved limit per class (``_triton_grouped_mm_max_rows``) is recorded on first
  sight. :func:`implied_generic` turns the decisions into the generic counts they imply (a Triton forward; its backward gx when
  grad was on and the input required it; its LoRA dW when the weight required it), and :func:`check_dispatch` holds the wrapper
  counts to them, with every LoRA-class call deciding Triton and the library's ``GENERIC_CALLS["fallback"]`` delta 0.
- :func:`check_env` works by SOURCE SCAN: every ``UNSLOTH_*`` key the three modules read from the environment (moe_utils,
  moe_grouped_fp16, moe_nf4_grouped; scanned at :func:`install` from their files) is unset or at its registered value. A prefix test
  would miss ``UNSLOTH_DISABLE_MOE_TRITON``, ``UNSLOTH_MOE_BACKEND``, ``UNSLOTH_COMPILE_LOCATION`` and others. The policy is cached
  on the environment at first use (521-533), so every pin is set before the zoo is imported.

Standard library plus nothing else at import time; the target modules are imported only by :func:`install`.
"""
from __future__ import annotations

import importlib
import re
import sys
import threading

FP16 = "unsloth_zoo.temporary_patches.moe_grouped_fp16"
NF4 = "unsloth_zoo.temporary_patches.moe_nf4_grouped"
#: (module, attribute) -> the counter it feeds. The self-check is wrapped only to mark its own calls.
TARGETS = {
    "generic_gemm": (FP16, "generic_grouped_mm"),
    "generic_wgrad": (FP16, "generic_grouped_wgrad"),
    "nf4_maybe": (NF4, "maybe_nf4_grouped_mm"),
}
SELF_CHECK = (FP16, "_self_check_generic")
LIBRARY_STATS = {"generic": (FP16, "GENERIC_CALLS"), "nf4": (NF4, "NF4_FUSED_STATS")}
COUNTS = ("generic_gemm", "generic_wgrad", "selfcheck_gemm", "selfcheck_wgrad", "nf4_maybe_calls", "nf4_fused_calls", "nf4_declined",
          "nf4_auto_eligible", "nf4_auto_ineligible", "nf4_mode_unread")
#: the library's own predicate functions, read at call time from the NF4 module (never re-typed here)
NF4_MODE_FN, NF4_AUTO_FN = "nf4_fused_mode", "nf4_auto_fused"
MOE_UTILS = "unsloth_zoo.temporary_patches.moe_utils"
DISPATCH, KIND_FN, LIMIT_FN, SERVER_FN = "_triton_grouped_mm_wanted", "_triton_grouped_mm_kind", "_triton_grouped_mm_max_rows", "get_forward_moe_backend"
GEMM_SITE = "_grouped_mm_with_backward_fix"
ENV_SOURCES = (MOE_UTILS, FP16, NF4)
ENV_RE = re.compile(r"""(?:environ|getenv)[^\n]{0,20}?["'](UNSLOTH_[A-Z0-9_]+)["']""")


class UnslothMoeCounter:
    """Install once per process before the model's first forward; read :meth:`snapshot` at the window's edges."""

    def __init__(self):
        self.counts = dict.fromkeys(COUNTS, 0)
        self.absent = []                     # "module.attr: why" for every target that could not be wrapped
        self._saved = []                     # (module, attr, original) to restore
        self._lib0 = {}
        self._local = threading.local()
        self.dispatch = {}                   # "site:class:decision" -> calls
        self.dispatch_grad = {"triton_grad_input": 0, "triton_grad_lora_weight": 0}
        self.dispatch_served = {}            # "package" / "cache_copy" -> decisions counted through that module object
        self.dispatch_absent = []
        self.served = None
        self.limits = {}
        self.env_keys = []
        self.env_unscanned = []
        self._saved_globals = []

    # -- install / uninstall -------------------------------------------------------------------------------------------------
    def install(self):
        mods = {}
        for mod in {m for m, _ in list(TARGETS.values()) + [SELF_CHECK] + list(LIBRARY_STATS.values())}:
            try:
                mods[mod] = importlib.import_module(mod)
            except Exception as e:  # noqa: BLE001 -- recorded, never fatal: a missing module is a finding the check fails on
                self.absent.append(f"{mod}: import failed: {type(e).__name__}")
        for key, (mod, attr) in TARGETS.items():
            m = mods.get(mod)
            fn = getattr(m, attr, None) if m is not None else None
            if not callable(fn):
                self.absent.append(f"{mod}.{attr}: absent")
                continue
            self._saved.append((m, attr, fn))
            setattr(m, attr, self._wrap(key, fn))
        m = mods.get(SELF_CHECK[0])
        sc = getattr(m, SELF_CHECK[1], None) if m is not None else None
        if callable(sc):
            self._saved.append((m, SELF_CHECK[1], sc))
            setattr(m, SELF_CHECK[1], self._wrap_selfcheck(sc))
        else:
            self.absent.append(f"{SELF_CHECK[0]}.{SELF_CHECK[1]}: absent")
        self._install_dispatch()
        self._scan_env()
        self._lib0 = self._library()
        return self

    def _install_dispatch(self):
        try:
            mu = importlib.import_module(MOE_UTILS)
        except Exception as e:  # noqa: BLE001
            self.dispatch_absent.append(f"{MOE_UTILS}: import failed: {type(e).__name__}")
            return
        globs = [("package", vars(mu))]
        try:
            fwd = getattr(mu, SERVER_FN)()
            g = getattr(fwd, "__globals__", None)
            if g is None:
                self.dispatch_absent.append(f"{SERVER_FN}() returned {type(fwd).__name__} without __globals__")
            else:
                self.served = {"module": g.get("__name__"), "file": g.get("__file__"), "is_package": g is vars(mu)}
                if g is not vars(mu):
                    globs.append(("cache_copy", g))
        except Exception as e:  # noqa: BLE001
            self.dispatch_absent.append(f"{MOE_UTILS}.{SERVER_FN}: {type(e).__name__}")
        for tag, g in globs:
            fn = g.get(DISPATCH)
            if not callable(fn):
                self.dispatch_absent.append(f"{tag} {DISPATCH}: absent")
                continue
            self._saved_globals.append((g, DISPATCH, fn))
            g[DISPATCH] = self._wrap_dispatch(tag, g, fn)

    def _wrap_dispatch(self, tag, g, fn):
        disp, grad, served, limits = self.dispatch, self.dispatch_grad, self.dispatch_served, self.limits

        def wrapper(inputs, weight, *args, **kwargs):
            decided = fn(inputs, weight, *args, **kwargs)
            site = "gemm" if sys._getframe(1).f_code.co_name == GEMM_SITE else "probe"
            try:
                E, K, N = weight.shape
                kind = g[KIND_FN](K, N, E)
            except Exception:  # noqa: BLE001 -- a weight the library could not classify reads as "unknown", which VOIDs
                kind = "unknown"
            key = f"{site}:{kind}:{'triton' if decided else 'torch'}"
            disp[key] = disp.get(key, 0) + 1
            served[tag] = served.get(tag, 0) + 1
            if site == "gemm" and decided:
                try:
                    import torch
                    on = torch.is_grad_enabled()
                    if on and getattr(inputs, "requires_grad", False):
                        grad["triton_grad_input"] += 1
                    if on and kind == "lora" and getattr(weight, "requires_grad", False):
                        grad["triton_grad_lora_weight"] += 1
                except Exception:  # noqa: BLE001
                    pass
            try:
                lk = f"{tag}:{inputs.device.index}:{kind}"
                if lk not in limits:
                    limits[lk] = g[LIMIT_FN](inputs.device.index, kind)
            except Exception:  # noqa: BLE001
                pass
            return decided

        wrapper.__wrapped__ = fn
        return wrapper

    def _scan_env(self):
        keys = set()
        for mod in ENV_SOURCES:
            try:
                path = importlib.import_module(mod).__file__
                keys |= set(ENV_RE.findall(open(path, encoding="utf-8", errors="replace").read()))
            except Exception as e:  # noqa: BLE001 -- an unscanned module fails check_env
                self.env_unscanned.append(f"{mod}: {type(e).__name__}")
        self.env_keys = sorted(keys)

    def uninstall(self):
        while self._saved:
            m, attr, fn = self._saved.pop()
            setattr(m, attr, fn)
        while self._saved_globals:
            g, attr, fn = self._saved_globals.pop()
            g[attr] = fn

    def _wrap(self, key, fn):
        counts, local = self.counts, self._local

        def wrapper(*args, **kwargs):
            if key == "nf4_maybe":
                self._auto_eligibility(args, kwargs)
            out = fn(*args, **kwargs)
            if key == "nf4_maybe":
                counts["nf4_maybe_calls"] += 1
                counts["nf4_fused_calls" if out is not None else "nf4_declined"] += 1
            elif getattr(local, "in_selfcheck", 0):
                counts["selfcheck_" + key.split("_", 1)[1]] += 1
            else:
                counts[key] += 1
            return out

        wrapper.__wrapped__ = fn
        wrapper.__name__ = getattr(fn, "__name__", "wrapper")
        return wrapper

    def _auto_eligibility(self, args, kwargs):
        """Under mode auto, the library's own predicate on this call's shapes. An unreadable mode or predicate is counted as
        ``nf4_mode_unread``, which every check fails on."""
        try:
            m = importlib.import_module(NF4)
            mode = getattr(m, NF4_MODE_FN)()
            if mode != "auto":
                return
            inputs = kwargs.get("inputs", args[0] if args else None)
            offsets = kwargs.get("offsets", args[1] if len(args) > 1 else None)
            ok = getattr(m, NF4_AUTO_FN)(inputs.shape[0], offsets.shape[0])
        except Exception:  # noqa: BLE001 -- recorded, never fatal to the arm
            self.counts["nf4_mode_unread"] += 1
            return
        self.counts["nf4_auto_eligible" if ok else "nf4_auto_ineligible"] += 1

    def _wrap_selfcheck(self, fn):
        local = self._local

        def wrapper(*args, **kwargs):
            local.in_selfcheck = getattr(local, "in_selfcheck", 0) + 1
            try:
                return fn(*args, **kwargs)
            finally:
                local.in_selfcheck -= 1

        wrapper.__wrapped__ = fn
        return wrapper

    # -- reading -------------------------------------------------------------------------------------------------------------
    def _library(self):
        out = {}
        for key, (mod, attr) in LIBRARY_STATS.items():
            try:
                stats = getattr(importlib.import_module(mod), attr)
                out[key] = {k: int(v) for k, v in stats.items()}
            except Exception:  # noqa: BLE001 -- an unreadable library counter reads as None, which every check fails on
                out[key] = None
        return out

    def snapshot(self):
        """The wrapper counts, the library counters' deltas since :meth:`install`, and what could not be wrapped."""
        lib = self._library()
        delta = {}
        for key, now in lib.items():
            base = self._lib0.get(key)
            delta[key] = None if (now is None or base is None) else {k: now[k] - base.get(k, 0) for k in now}
        return {"counts": dict(self.counts), "library_delta": delta, "absent": list(self.absent),
                "dispatch": dict(self.dispatch), "dispatch_grad": dict(self.dispatch_grad), "dispatch_served": dict(self.dispatch_served),
                "dispatch_absent": list(self.dispatch_absent), "served": self.served, "limits": dict(self.limits),
                "env_keys": list(self.env_keys), "env_unscanned": list(self.env_unscanned)}


def check_library(snap):
    """(ok, why): every wrapper count agrees with the library's own counter for the same calls. A disagreement means a caller
    reached the library past the wrapper (a local alias bound before install) or the wrapper counted something the library
    didn't."""
    why = []
    if snap["absent"]:
        why.append(f"not wrapped: {snap['absent']}")
    g = snap["library_delta"].get("generic")
    if g is None:
        why.append("GENERIC_CALLS unreadable")
    else:
        for k in ("gemm", "wgrad"):
            if snap["counts"][f"generic_{k}"] != g.get(k):
                why.append(f"generic_{k}: wrapper {snap['counts'][f'generic_{k}']} != library GENERIC_CALLS {g.get(k)}")
    n = snap["library_delta"].get("nf4")
    if n is None:
        why.append("NF4_FUSED_STATS unreadable")
    elif snap["counts"]["nf4_fused_calls"] != n.get("calls"):
        why.append(f"nf4_fused_calls: wrapper {snap['counts']['nf4_fused_calls']} != library NF4_FUSED_STATS calls {n.get('calls')}")
    return (not why), "; ".join(why)


def check_equal(name, wrapper_count, counterpart_count, counterpart_name):
    """(ok, why): a wrapper count binds only when it equals a compile-safe counterpart that counted the same calls."""
    if wrapper_count is None or counterpart_count is None:
        return False, f"{name}: unread (wrapper {wrapper_count}, {counterpart_name} {counterpart_count})"
    if wrapper_count != counterpart_count:
        return False, f"{name}: wrapper {wrapper_count} != {counterpart_name} {counterpart_count} (a traced or captured block?)"
    return True, ""


def check_fused(snap, expected_launches):
    """(ok, why) for a fused arm (UNSLOTH_MOE_NF4_FUSED 1 / auto): NF4_FUSED_STATS.launches == the expected fused base GEMMs,
    exact. The lane computes ``expected_launches`` from the arm's own expert-forward count (forward + dX per projection)."""
    n = snap["library_delta"].get("nf4")
    if n is None or expected_launches is None:
        return False, f"NF4_FUSED_STATS unreadable or no expected count (delta {n}, expected {expected_launches})"
    if n.get("launches") != expected_launches:
        return False, f"NF4 fused launches {n.get('launches')} != expected {expected_launches}"
    return True, ""


def check_unfused(snap):
    """(ok, why) for an UNSLOTH_MOE_NF4_FUSED=0 arm: no fused launch and no fused call."""
    n = snap["library_delta"].get("nf4")
    if n is None:
        return False, "NF4_FUSED_STATS unreadable"
    bad = [f"{k} {n.get(k)}" for k in ("launches", "calls") if n.get(k) != 0]
    if snap["counts"]["nf4_fused_calls"]:
        bad.append(f"wrapper nf4_fused_calls {snap['counts']['nf4_fused_calls']}")
    return (not bad), ("an unfused arm took the fused path: " + ", ".join(bad)) if bad else ""


def fused_calls_expected(snap, mode):
    """The fused calls a ``mode`` arm must make: every ``maybe_nf4_grouped_mm`` call under ``1``, the calls the library's own auto
    predicate admitted under ``auto``, none under ``0``. None when it cannot be read (an unread mode or predicate)."""
    c = snap["counts"]
    if c["nf4_mode_unread"]:
        return None
    if mode == "1":
        return c["nf4_maybe_calls"]
    if mode == "auto":
        return c["nf4_auto_eligible"]
    if mode == "0":
        return 0
    return None


def check_auto(snap):
    """(ok, why) for an auto arm, independent of the lane's launch arithmetic: every admitted call fused and every refused call
    declined. A forced decline at an eligible shape, or a fused call at an ineligible one, VOIDs."""
    c = snap["counts"]
    if c["nf4_mode_unread"]:
        return False, f"the NF4 mode or the auto predicate was unreadable on {c['nf4_mode_unread']} calls"
    if c["nf4_auto_eligible"] + c["nf4_auto_ineligible"] != c["nf4_maybe_calls"]:
        return False, (f"auto eligibility read on {c['nf4_auto_eligible'] + c['nf4_auto_ineligible']} of {c['nf4_maybe_calls']} calls "
                       "(was the mode auto on every call?)")
    if c["nf4_fused_calls"] != c["nf4_auto_eligible"]:
        return False, f"auto: {c['nf4_fused_calls']} fused calls != {c['nf4_auto_eligible']} the library's predicate admitted"
    return True, ""


def check_env(snap, environ, registered):
    """(ok, why): every ``UNSLOTH_*`` key the three library modules read from the environment (scanned from their source at install)
    is unset or at its registered value, and every registered key is one of them and set. ``registered`` maps key -> value: the
    arm's explicit pins (the NF4 mode, AUTO_MAX_ROWS, AUTOTUNE, the grouped-Triton mode, ...)."""
    keys = snap.get("env_keys") or []
    if snap.get("env_unscanned") or not keys:
        return False, f"the library's environment reads are unscanned ({snap.get('env_unscanned') or 'no keys found'})"
    bad = [f"{k}={environ.get(k)!r} (registered {registered[k]!r})" if k in registered else f"ambient {k}={environ.get(k)!r}"
           for k in keys if environ.get(k) is not None and environ.get(k) != registered.get(k)]
    bad += [f"registered {k} is not a key the library reads" for k in registered if k not in keys]
    bad += [f"registered {k}={v!r} is not set" for k, v in registered.items() if k in keys and environ.get(k) is None]
    return (not bad), ("arm environment: " + "; ".join(bad)) if bad else ""


def implied_generic(snap):
    """The generic GEMM and dW calls the recorded dispatch decisions imply: each Triton decision at the GEMM site is one forward
    generic_grouped_mm; one taken with grad on and an input that required grad adds its backward gx (the Triton autograd's backward,
    moe_utils.py:634-636, which asks no predicate); one on a LoRA-class weight that required grad adds its dW (generic_grouped_wgrad,
    611-615). A decision under no_grad (the dequant path's recompute forward, and its own dX decision at 1110) adds no backward."""
    d = snap["dispatch"]
    fwd = sum(v for k, v in d.items() if k.startswith("gemm:") and k.endswith(":triton"))
    return {"generic_gemm": fwd + snap["dispatch_grad"]["triton_grad_input"], "generic_wgrad": snap["dispatch_grad"]["triton_grad_lora_weight"]}


def check_dispatch(snap, base_must_be_triton=False):
    """(ok, why): the per-call dispatch is as the library decided it. Fails when:
    - the predicate could not be wrapped, or the serving module object counted no decision while generic calls ran;
    - any LoRA-class GEMM decided torch (the predicate returns False silently on a dtype, a tensor subclass or a dim);
    - any GEMM's class is unknown;
    - the library's GENERIC_CALLS["fallback"] moved (a Triton-decided call that ran eager, moe_utils.py:600 / 608);
    - the wrapper's generic counts differ from :func:`implied_generic`;
    - ``base_must_be_triton`` (a lane whose sweep sits inside every class limit, as P238 on sm_120) and a base GEMM decided torch."""
    why = []
    if snap["dispatch_absent"]:
        why.append(f"dispatch not wrapped: {snap['dispatch_absent']}")
    served = snap.get("served") or {}
    tag = "package" if served.get("is_package") else "cache_copy"
    if not served:
        why.append("the serving forward was not identified")
    elif (snap["counts"]["generic_gemm"] or snap["dispatch"]) and not snap["dispatch_served"].get(tag):
        why.append(f"no decision counted through the serving module ({tag}: {served.get('module')})")
    d = snap["dispatch"]
    lora_torch = d.get("gemm:lora:torch", 0)
    if lora_torch:
        why.append(f"{lora_torch} LoRA-class GEMMs decided torch")
    unknown = sum(v for k, v in d.items() if k.startswith("gemm:unknown:"))
    if unknown:
        why.append(f"{unknown} GEMMs of unknown class")
    if base_must_be_triton:
        base_torch = sum(v for k, v in d.items() if k.startswith("gemm:") and k.endswith(":torch") and ":lora:" not in k)
        if base_torch:
            why.append(f"{base_torch} base GEMMs decided torch where the registered sweep is inside every limit")
    g = snap["library_delta"].get("generic")
    if g is None:
        why.append("GENERIC_CALLS unreadable")
    elif g.get("fallback"):
        why.append(f"GENERIC_CALLS fallback moved by {g.get('fallback')} (Triton-decided calls ran eager)")
    imp = implied_generic(snap)
    for k in ("generic_gemm", "generic_wgrad"):
        if snap["counts"][k] != imp[k]:
            why.append(f"{k}: wrapper {snap['counts'][k]} != {imp[k]} the decisions imply")
    return (not why), "; ".join(why)
