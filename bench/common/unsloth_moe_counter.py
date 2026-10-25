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
- A fused arm binds on ``NF4_FUSED_STATS.launches`` == :func:`expected_launches`, exact: one forward launch per fused call (recompute
  and evals included), plus the dX launches BY STRUCTURE (:meth:`UnslothMoeCounter.structural`: every expert module is backwarded
  once per micro-batch per step; the custom op's backward is dX only). Never from "fused calls under grad": HF non-reentrant
  checkpointing runs both passes under grad and backwards one, and an eval runs with grad on and backwards none (that split,
  ``nf4_fused_calls_grad``, is kept as information). That equality catches the silent paths: a shape mismatch (moe_nf4_grouped.py:534-537) and a
  ``FAILED_SHAPES`` entry (538-542), which return None with no counter, and the failed-launch decrement (275), whose eager decode is
  uncounted.
- ``auto`` declining above its row cap (523) is that arm's DESIGN, not a fault. So the expected count of an auto arm admits only the
  calls the LIBRARY's own predicate admits: on every ``maybe_nf4_grouped_mm`` call under mode ``auto``, the wrapper evaluates the
  library's ``nf4_auto_fused(inputs.shape[0], offsets.shape[0])`` (rows <= ``nf4_auto_max_rows()`` x E, lines 69-81) on that call's
  own shapes and counts ``nf4_auto_eligible`` / ``nf4_auto_ineligible``. :func:`fused_calls_expected` is the eligible count for auto
  and every call for ``1``; the lane's own count from its routed rows must agree (:func:`check_equal`). At a fixed op-level shape
  every call has the same rows, so auto is all-or-none (rows = tokens x 8 for top-8: fused at <= 4,096 tokens, 0 launches above).
- A ``UNSLOTH_MOE_NF4_FUSED=0`` arm binds on ``launches == 0`` and ``calls == 0``.
- Every Unsloth knob an arm depends on is set explicitly to its registered value, and :func:`check_env` refuses any other value
  and any unregistered ``UNSLOTH_*`` key SET in the arm's environment (an ambient ``UNSLOTH_MOE_NF4_AUTO_MAX_ROWS`` moves auto's cap;
  ``UNSLOTH_MOE_NF4_AUTOTUNE=1`` retunes tiles per shape, line 316; ``UNSLOTH_MOE_FUSED_ROUTER`` switches the router in the step).
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
  copy's ``forward_moe_backend`` when one is present and current, and that copy calls its OWN global. The zoo binds that answer AT
  PATCH TIME as the experts' class forward (qwen3_5_moe.py:53-60, qwen3_moe.py:84-85), and the cache copy can appear during the
  model load. So :func:`install` (before the load) patches the package module and the current answer, and
  :meth:`UnslothMoeCounter.patch_loaded` (after the load and the framework's patching, before the first forward) patches the globals
  of the forward each LOADED expert module runs (``type(module).forward``) and serves from those. EXCEPT where the library hands the
  module off: ``forward_moe_backend`` asks ``moe_utils_bnb4bit._moe_uses_bnb4bit_expert_weights`` and then the fp8 predicate
  (moe_utils.py:2065-2087), and those modules import ``forward_native_grouped_mm`` RELATIVELY (moe_utils_bnb4bit.py:367-376), which
  resolves to the PACKAGE moe_utils, never the copy. So each module's serving object is chosen by the library's own predicates, in its
  own order: bnb 4-bit (P237's and P238's NF4 QLoRA) and fp8 experts are served by the package. ``served`` records each serving
  object (its tag, ``__name__``, ``__file__``, whether it is the package module); ``expert_forwards`` records each expert type's
  distinct forwards and how many modules hold each, so a mixed model shows; ``served_fresh`` (a fresh
  ``get_forward_moe_backend()`` call) is information only.
- Each decision is recorded by site, class (the library's own ``_triton_grouped_mm_kind``) and outcome, with whether grad was enabled
  and whether the input / weight required grad, and is ATTRIBUTED to the expert-module forward it ran in (forward pre/post hooks
  ``patch_loaded`` puts on the passed modules; they return None) or to "outside" (the backward's own decisions, as the dequant
  path's dX at 1110). The lane marks each optimizer step (:meth:`UnslothMoeCounter.mark_step`). The resolved limit per class
  (``_triton_grouped_mm_max_rows``) is recorded on first sight. :meth:`UnslothMoeCounter.structural` gives the backward BY STRUCTURE
  (each module backwarded once per micro-batch per step; its training passes in that step must agree, else VOID), and
  :func:`implied_generic` adds it to the Triton forwards; :func:`check_dispatch` holds the wrapper counts to that, with every
  LoRA-class call deciding Triton and the library's ``GENERIC_CALLS["fallback"]`` delta 0.
- :func:`check_env` refuses any ``UNSLOTH_*`` key SET in the arm's environment that is not registered at exactly its value. The
  registered names are held to a SOURCE SCAN of every file of the installed ``unsloth_zoo`` and ``unsloth`` packages (no hand-kept
  module list: the module nobody named is the one that bites), with the count per module recorded. The policy is cached on the
  environment at first use (521-533), so every pin is set before the zoo is imported.

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
          "nf4_auto_eligible", "nf4_auto_ineligible", "nf4_mode_unread", "nf4_fused_calls_grad")
#: the library's own predicate functions, read at call time from the NF4 module (never re-typed here)
NF4_MODE_FN, NF4_AUTO_FN = "nf4_fused_mode", "nf4_auto_fused"
MOE_UTILS = "unsloth_zoo.temporary_patches.moe_utils"
DISPATCH, KIND_FN, LIMIT_FN, SERVER_FN = "_triton_grouped_mm_wanted", "_triton_grouped_mm_kind", "_triton_grouped_mm_max_rows", "get_forward_moe_backend"
GEMM_SITE = "_grouped_mm_with_backward_fix"
#: the library's own hand-off predicates, in the order forward_moe_backend asks them (moe_utils.py:2065-2087): a module they claim is
#: served by the module object their relative ``from .moe_utils import ...`` resolves to (moe_utils_bnb4bit.py:367-376), the PACKAGE
HANDOFF = (("bnb4bit", "unsloth_zoo.temporary_patches.moe_utils_bnb4bit", "_moe_uses_bnb4bit_expert_weights"),
           ("fp8", "unsloth_zoo.temporary_patches.moe_utils_fp8", "_moe_uses_fp8_expert_weights"))
#: every source file of these installed packages is scanned for UNSLOTH_* names (no hand-kept module list)
ENV_PACKAGES = ("unsloth_zoo", "unsloth")
ENV_RE = re.compile(r"""["'](UNSLOTH_[A-Z0-9_]+)["']""")


class UnslothMoeCounter:
    """:meth:`install` once per process before the model load; :meth:`patch_loaded` with the loaded expert modules after the load,
    before the first forward; read :meth:`snapshot` at the window's edges."""

    def __init__(self):
        self.counts = dict.fromkeys(COUNTS, 0)
        self.absent = []                     # "module.attr: why" for every target that could not be wrapped
        self._saved = []                     # (module, attr, original) to restore
        self._lib0 = {}
        self._local = threading.local()
        self.dispatch = {}                   # "site:class:decision" -> calls
        self.dispatch_grad = {"triton_grad_input": 0, "triton_grad_lora_weight": 0}
        self.dispatch_served = {}            # "package" / "copyN" -> {"gemm", "probe"}: decisions counted through that object
        self.dispatch_absent = []
        self.served = []                     # the module objects whose forward the model runs (see patch_loaded)
        self.served_fresh = None             # get_forward_moe_backend()'s answer at the last patch: information only
        self.expert_forwards = {}            # expert type -> [{forward, globals_module, tag, modules}] (patch_loaded)
        self._patched = {}                   # id(globals) -> (globals, tag); the dict is held so its id stays unique
        self.limits = {}
        self.env_keys = []
        self.env_unscanned = []
        self._saved_globals = []
        self.env_by_module = {}              # package-relative path -> the UNSLOTH_* names its source quotes
        self.passes = []                     # one record per expert-module forward (patch_loaded's hooks): decisions attributed
        self.outside = []                    # decisions made outside any expert forward (the backward's own dX decisions)
        self.n_modules = 0
        self._step = None                    # set by mark_step (the tc1_arm hook's step_edge)
        self._hooks = []

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

    def patch_loaded(self, experts):
        """After the model load and the framework's patching, before the first forward (the tc1_arm hook's post_load): patch the
        predicate in the globals of the forward each loaded expert module runs (``type(module).forward``), and serve from THOSE
        objects. The compile-cache copy of moe_utils can appear during the load, after :meth:`install`; the zoo binds
        ``get_forward_moe_backend()``'s answer AT PATCH TIME as the experts' class forward (qwen3_5_moe.py:53-60 via
        qwen3_moe.py:84-85), so the class attribute, not a fresh call, is the object the model runs. Records each expert type's
        distinct forwards and how many modules hold each, so a mixed model shows. ``experts``: the lane's expert modules."""
        self._install_dispatch(experts)
        return self

    def _install_dispatch(self, experts=None):
        try:
            mu = importlib.import_module(MOE_UTILS)
        except Exception as e:  # noqa: BLE001
            self.dispatch_absent.append(f"{MOE_UTILS}: import failed: {type(e).__name__}")
            return
        self._patch_globals(vars(mu), mu)
        fresh = None
        try:
            fwd = getattr(mu, SERVER_FN)()
            fresh = getattr(fwd, "__globals__", None)
            if fresh is None:
                self.dispatch_absent.append(f"{SERVER_FN}() returned {type(fwd).__name__} without __globals__")
            else:
                self.served_fresh = {"module": fresh.get("__name__"), "file": fresh.get("__file__"), "is_package": fresh is vars(mu)}
        except Exception as e:  # noqa: BLE001
            self.dispatch_absent.append(f"{MOE_UTILS}.{SERVER_FN}: {type(e).__name__}")
        if experts is None:                  # install() before the load: the fresh answer serves until patch_loaded replaces it
            serving = [(fresh, "fresh")] if fresh is not None else []
        else:
            experts = list(experts)
            serving = self._expert_globals(experts)
            if not experts:
                self.dispatch_absent.append("patch_loaded: no expert modules passed")
            self._track_passes(experts)
        self.served = []
        for g, source in serving:
            tag = self._patch_globals(g, mu)
            if tag is not None and all(s["tag"] != tag for s in self.served):
                self.served.append({"tag": tag, "module": g.get("__name__"), "file": g.get("__file__"), "is_package": g is vars(mu),
                                    "source": source})

    def _rule(self, m):
        """Which path serves this module's GEMM decisions, by the library's OWN hand-off predicates in its own order: "bnb4bit", "fp8",
        else "dispatcher" (the class forward). A predicate that cannot be imported is skipped, as the library skips it (2066-2072)."""
        for rule, mod, fn in HANDOFF:
            try:
                pred = getattr(importlib.import_module(mod), fn)
            except (ImportError, AttributeError):
                continue
            if pred(m):
                return rule
        return "dispatcher"

    def _expert_globals(self, experts):
        """[(globals, source)] serving the passed expert modules. A module a hand-off predicate claims is served by the package
        moe_utils (its relative import); any other by its class forward, unwrapped through ``__wrapped__`` to the first function whose
        globals hold the predicate. A forward with no such function is recorded as dispatch-absent. ``expert_forwards`` records each
        expert type's distinct (forward, rule) with its module count."""
        self.expert_forwards = {}
        groups, out = {}, []
        for m in experts:
            t = type(m)
            f = getattr(t, "forward", None)
            try:
                rule = self._rule(m)
            except Exception as e:  # noqa: BLE001 -- the library would raise here too; recorded, and it VOIDs
                rule = "unreadable"
                self.dispatch_absent.append(f"{t.__qualname__}: the hand-off predicate raised {type(e).__name__}")
            groups.setdefault((f"{t.__module__}.{t.__qualname__}", id(f), rule), [f, 0])[1] += 1
        for (tname, _, rule), (f, n) in groups.items():
            name = f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', type(f).__name__)}"
            g, hop = None, None
            if rule in {r for r, _, _ in HANDOFF}:
                mod = importlib.import_module(next(m_ for r, m_, _ in HANDOFF if r == rule))
                g = vars(importlib.import_module(f"{mod.__package__}.moe_utils"))
                if not callable(g.get(DISPATCH)):
                    self.dispatch_absent.append(f"{tname} ({rule}): no {DISPATCH} in {mod.__package__}.moe_utils")
                    g = None
            elif rule == "dispatcher":
                hop, cur = 0, f
                while cur is not None and hop <= 8:
                    cg = getattr(getattr(cur, "__func__", cur), "__globals__", None)
                    if cg is not None and callable(cg.get(DISPATCH)):
                        g = cg
                        break
                    cur, hop = getattr(cur, "__wrapped__", None), hop + 1
                if g is None:
                    self.dispatch_absent.append(f"{tname}.forward ({name}): no {DISPATCH} in its globals ({n} modules)")
            self.expert_forwards.setdefault(tname, []).append({"forward": name, "rule": rule, "modules": n,
                                                               "globals_module": g.get("__name__") if g else None,
                                                               "unwrapped": hop if g is not None and rule == "dispatcher" else None})
            if g is not None:
                src = "loaded" if rule == "dispatcher" else rule
                if all(x is not g for x, _ in out):
                    out.append((g, src))
        return out

    def _track_passes(self, experts):
        """Forward pre/post hooks on each passed expert module (observe only: both return None; ``always_call`` so a raising forward
        still closes its pass). Each forward opens a pass record; the wrappers attribute every decision to the innermost open pass."""
        local, passes = self._local, self.passes
        self.n_modules = len(experts)
        for idx, m in enumerate(experts):
            def pre(mod, inputs, _i=idx):
                import torch
                rec = {"m": _i, "grad": bool(torch.is_grad_enabled()), "training": bool(mod.training), "step": self._step,
                       "gemm": [], "nf4": []}
                if getattr(local, "stack", None) is None:
                    local.stack = []
                local.stack.append(rec)
                passes.append(rec)
                return None

            def post(mod, inputs, output):
                if getattr(local, "stack", None):
                    local.stack.pop()
                return None

            self._hooks.append(m.register_forward_pre_hook(pre))
            self._hooks.append(m.register_forward_hook(post, always_call=True))

    def _current(self):
        st = getattr(self._local, "stack", None)
        return st[-1] if st else None

    def mark_step(self, i):
        """The boundary before optimizer step ``i`` (and once after the last, i = N): passes until the next mark belong to step i."""
        self._step = int(i)

    def _patch_globals(self, g, mu):
        """Wrap the predicate in one module object's globals, once; returns its tag ("package", or "copyN" in patch order)."""
        if id(g) in self._patched:
            return self._patched[id(g)][1]
        tag = "package" if g is vars(mu) else f"copy{sum(t != 'package' for _, t in self._patched.values()) + 1}"
        fn = g.get(DISPATCH)
        if not callable(fn):
            self.dispatch_absent.append(f"{tag} {DISPATCH}: absent")
            return None
        self._patched[id(g)] = (g, tag)
        self._saved_globals.append((g, DISPATCH, fn))
        g[DISPATCH] = self._wrap_dispatch(tag, g, fn)
        return tag

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
            by = served.setdefault(tag, {"gemm": 0, "probe": 0})
            by[site] += 1
            gin = gw = False
            try:
                import torch
                on = torch.is_grad_enabled()
                gin = bool(on and getattr(inputs, "requires_grad", False))
                gw = bool(on and kind == "lora" and getattr(weight, "requires_grad", False))
            except Exception:  # noqa: BLE001
                pass
            if site == "gemm" and decided:   # the under-grad split: INFORMATION and a cross-check, never the binding count
                grad["triton_grad_input"] += gin
                grad["triton_grad_lora_weight"] += gw
            cur = self._current()
            entry = (site, kind, bool(decided), gin, gw)
            if cur is not None:
                cur["gemm"].append(entry)
            else:
                self.outside.append({"step": self._step, "entry": entry})
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
        """Every source file of the installed ``unsloth_zoo`` and ``unsloth`` packages, located without importing them, scanned for
        quoted ``UNSLOTH_*`` names. A closed module list misses the module nobody named (``UNSLOTH_MOE_FUSED_ROUTER`` in
        moe_router_triton switches the router inside the timed step)."""
        import importlib.util
        import pathlib
        keys = set()
        for pkg in ENV_PACKAGES:
            try:
                spec = importlib.util.find_spec(pkg)
                roots = list(spec.submodule_search_locations or [])
                if not roots:
                    raise ImportError("no package directory")
            except Exception as e:  # noqa: BLE001 -- an unscanned package fails check_env
                self.env_unscanned.append(f"{pkg}: {type(e).__name__}")
                continue
            for root in roots:
                for path in sorted(pathlib.Path(root).rglob("*.py")):
                    found = set(ENV_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
                    if found:
                        self.env_by_module[f"{pkg}/{path.relative_to(root).as_posix()}"] = sorted(found)
                    keys |= found
        self.env_keys = sorted(keys)

    def uninstall(self):
        while self._saved:
            m, attr, fn = self._saved.pop()
            setattr(m, attr, fn)
        while self._saved_globals:
            g, attr, fn = self._saved_globals.pop()
            g[attr] = fn
        self._patched.clear()
        while self._hooks:
            self._hooks.pop().remove()

    def _wrap(self, key, fn):
        counts, local = self.counts, self._local

        def wrapper(*args, **kwargs):
            if key == "nf4_maybe":
                self._auto_eligibility(args, kwargs)
            out = fn(*args, **kwargs)
            if key == "nf4_maybe":
                counts["nf4_maybe_calls"] += 1
                counts["nf4_fused_calls" if out is not None else "nf4_declined"] += 1
                if out is not None and _grad_call(args, kwargs):
                    counts["nf4_fused_calls_grad"] += 1     # INFORMATION: the under-grad split, cross-checked, never binding
                cur = self._current()
                if cur is not None:
                    cur["nf4"].append(out is not None)
                elif out is not None:
                    self.outside.append({"step": self._step, "nf4": True})
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

    # -- the structural backward count ---------------------------------------------------------------------------------------
    def structural(self, micro_batches, n_steps):
        """The backward launches a run implies, BY STRUCTURE, independent of the checkpointer: each optimizer step backwards every
        expert module once per micro-batch. For each step 0..n_steps-1 and each tracked module, the TRAINING passes (``mod.training``;
        evals run in eval mode) must agree on every GEMM decision (site, class, Triton/torch) and every NF4 fused/declined outcome:
        decisions are shape-determined, so a disagreement VOIDs. A training pass under grad gives the backward: per micro-batch, one
        NF4 dX per fused call (the custom op's backward is dX only), one generic gx per Triton GEMM whose input required grad, and one
        LoRA dW per Triton LoRA GEMM whose weight required grad. Under HF non-reentrant checkpointing both passes run under grad and
        only one is backwarded, so the per-call under-grad split would double those; it is kept as INFORMATION (``split``).

        Returns {"expected": {"dx_nf4", "gx", "wgrad"}, "faults": [...], "split": {...}, "steps", "modules", "micro_batches"}."""
        faults, exp = [], {"dx_nf4": 0, "gx": 0, "wgrad": 0}
        if not self.n_modules:
            faults.append("no expert module tracked (patch_loaded never saw the experts)")
        by = {}
        for p in self.passes:
            if not p["training"]:
                continue
            s_ = p["step"]
            if s_ is None or not 0 <= s_ < n_steps:
                faults.append(f"a training pass of expert module {p['m']} outside a marked step (step {s_})")
                continue
            by.setdefault((s_, p["m"]), []).append(p)
        gemm = lambda p: tuple(e[:3] for e in p["gemm"] if e[0] == "gemm")         # noqa: E731
        for st in range(n_steps):
            for m in range(self.n_modules):
                T = by.get((st, m))
                if not T:
                    faults.append(f"step {st}: expert module {m} ran no training pass")
                    continue
                if len({(gemm(p), tuple(p["nf4"])) for p in T}) != 1:
                    faults.append(f"step {st}: expert module {m}'s {len(T)} training passes decided differently")
                    continue
                G = [p for p in T if p["grad"]]
                flags = {tuple(e[3:] for e in p["gemm"] if e[0] == "gemm") for p in G}
                if not G or len(flags) != 1:
                    faults.append(f"step {st}: expert module {m} has {len(G)} passes under grad with {len(flags)} grad patterns")
                    continue
                rep_ = G[0]
                exp["dx_nf4"] += micro_batches * sum(rep_["nf4"])
                exp["gx"] += micro_batches * sum(1 for e in rep_["gemm"] if e[0] == "gemm" and e[2] and e[3])
                exp["wgrad"] += micro_batches * sum(1 for e in rep_["gemm"] if e[0] == "gemm" and e[2] and e[4])
        split = {"gx_under_grad": self.dispatch_grad["triton_grad_input"], "wgrad_under_grad": self.dispatch_grad["triton_grad_lora_weight"],
                 "dx_nf4_under_grad": self.counts["nf4_fused_calls_grad"]}
        split["agrees"] = (split["gx_under_grad"], split["wgrad_under_grad"], split["dx_nf4_under_grad"]) == (exp["gx"], exp["wgrad"], exp["dx_nf4"])
        return {"expected": exp, "faults": faults, "split": split, "steps": n_steps, "modules": self.n_modules,
                "micro_batches": micro_batches}

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
                "dispatch": dict(self.dispatch), "dispatch_grad": dict(self.dispatch_grad), "dispatch_served": {k: dict(v) for k, v in self.dispatch_served.items()},
                "dispatch_absent": list(self.dispatch_absent), "served": [dict(x) for x in self.served],
                "served_fresh": self.served_fresh, "expert_forwards": {k: [dict(x) for x in v] for k, v in self.expert_forwards.items()},
                "limits": dict(self.limits),
                "env_keys": list(self.env_keys), "env_unscanned": list(self.env_unscanned),
                "env_counts": {k: len(v) for k, v in self.env_by_module.items()}, "passes_n": len(self.passes)}


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
    exact: ``expected_launches`` (the fused calls plus the dX launches by structure)."""
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


def _grad_call(args, kwargs):
    """Whether a fused call ran with grad on and an input that requires grad: the under-grad split (INFORMATION only)."""
    try:
        import torch
        inputs = kwargs.get("inputs", args[0] if args else None)
        return bool(torch.is_grad_enabled() and getattr(inputs, "requires_grad", False))
    except Exception:  # noqa: BLE001
        return False


def expected_launches(snap, structural):
    """The fused launches a run implies: one forward launch per fused call, recompute and evals included (269), plus the structural
    dX launches (:meth:`UnslothMoeCounter.structural`). None when the mode or predicate was unread, or the structure faulted."""
    c = snap["counts"]
    if c["nf4_mode_unread"] or not structural or structural.get("faults"):
        return None
    return c["nf4_fused_calls"] + structural["expected"]["dx_nf4"]


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
    """(ok, why): the arm's environment holds no UNSLOTH_* key it did not register. Every ``UNSLOTH_*`` key SET in ``environ`` is
    registered at exactly its value, scanned or not (unset keys cost nothing); every registered key is set, and is a name the
    installed packages' source quotes (the whole-package scan at install: a registered name nothing reads is a typo). ``registered``
    maps key -> value: the arm's explicit pins."""
    keys = set(snap.get("env_keys") or [])
    if snap.get("env_unscanned") or not keys:
        return False, f"the library's environment reads are unscanned ({snap.get('env_unscanned') or 'no keys found'})"
    bad = [f"{k}={v!r} (registered {registered[k]!r})" if k in registered else f"ambient {k}={v!r}"
           for k, v in sorted(environ.items()) if k.startswith("UNSLOTH_") and v != registered.get(k)]
    bad += [f"registered {k} is not a key the library reads" for k in registered if k not in keys]
    bad += [f"registered {k}={v!r} is not set" for k, v in registered.items() if environ.get(k) is None]
    return (not bad), ("arm environment: " + "; ".join(bad)) if bad else ""


def implied_generic(snap, structural):
    """The generic GEMM and dW calls the decisions imply: every Triton decision at the GEMM site is one forward ``generic_grouped_mm``
    (inside any pass or outside one, as the dequant path's own dX decision at 1110 in backward); plus, BY STRUCTURE, the backward gx
    and LoRA dW of :meth:`UnslothMoeCounter.structural`. None when the structure is missing or faulted."""
    if not structural or structural.get("faults"):
        return None
    d = snap["dispatch"]
    fwd = sum(v for k, v in d.items() if k.startswith("gemm:") and k.endswith(":triton"))
    return {"generic_gemm": fwd + structural["expected"]["gx"], "generic_wgrad": structural["expected"]["wgrad"]}


def check_dispatch(snap, structural, base_must_be_triton=False):
    """(ok, why): the per-call dispatch is as the library decided it. Fails when:
    - the predicate could not be wrapped (or a loaded expert's forward holds none), or a serving module object counted no
      GEMM decision while generic calls ran, or a GEMM decision was counted through a module object that serves no forward;
    - any LoRA-class GEMM decided torch (the predicate returns False silently on a dtype, a tensor subclass or a dim);
    - any GEMM's class is unknown;
    - the library's GENERIC_CALLS["fallback"] moved (a Triton-decided call that ran eager, moe_utils.py:600 / 608);
    - the wrapper's generic counts differ from :func:`implied_generic`;
    - ``base_must_be_triton`` (a lane whose sweep sits inside every class limit, as P238 on sm_120) and a base GEMM decided torch."""
    why = []
    if snap["dispatch_absent"]:
        why.append(f"dispatch not wrapped: {snap['dispatch_absent']}")
    served = snap.get("served") or []
    ran = snap["counts"]["generic_gemm"] or snap["dispatch"]
    if not served:
        why.append("the serving forward was not identified")
    elif ran:
        for s in served:
            if not snap["dispatch_served"].get(s["tag"], {}).get("gemm"):
                why.append(f"no decision counted through the serving module ({s['tag']}: {s['module']})")
        stray = {t: v["gemm"] for t, v in snap["dispatch_served"].items() if v.get("gemm") and t not in {s["tag"] for s in served}}
        if stray:
            why.append(f"GEMM decisions counted through a module object that serves no forward: {stray}")
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
    imp = implied_generic(snap, structural)
    if imp is None:
        why.append(f"no structural count ({(structural or {}).get('faults') or 'missing'})")
    else:
        for k in ("generic_gemm", "generic_wgrad"):
            if snap["counts"][k] != imp[k]:
                why.append(f"{k}: wrapper {snap['counts'][k]} != {imp[k]} the decisions imply")
    return (not why), "; ".join(why)
