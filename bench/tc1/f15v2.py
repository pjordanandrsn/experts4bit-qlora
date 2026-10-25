"""F15 v2: Unsloth's expert-GEMM engagement counted on the dispatch the pinned unsloth_zoo actually takes, plus a profiled arm's
kernels-only device time. TC1 amendment 80; e4b#2129 deliverable 1. Standard library only; reads a cell receipt and its _profile.json.

Why. F15 v1 (tc1_arm.py's ``_install_torch_grouped_mm`` wrapper) counts ``torch._grouped_mm``. unsloth_zoo 2026.9.x's grouped_mm backend
called it. 2026.10.8's ``_grouped_mm_with_backward_fix`` (temporary_patches/moe_utils.py:366) first asks ``_triton_grouped_mm_wanted``
(378-379). On sm_120 under the default ``UNSLOTH_MOE_GROUPED_TRITON=auto`` (``_TRITON_GROUPED_MM_AUTO_ROWS[(12, 0)]``, 503-508) that
routes every Qwen3-30B-A3B expert GEMM to ``moe_grouped_fp16.generic_grouped_mm`` (741). Compiled, that's the custom op
``unsloth_zoo::grouped_mm_triton`` (656-709); eager, the ``_GroupedMMTriton`` autograd.Function (618-639). Both launch
``_grouped_gemm_kernel`` from its only launch site, moe_grouped_fp16.py:522, and dW from ``_grouped_wgrad_kernel`` (563). v1 therefore
reads 0 on a backend that engaged (TC1 amendment 79, both Unsloth draws VOID). Source: the release-proven wheel unsloth_zoo 2026.10.8,
sha256 db8c283bfab4c250e429cab35047bbde20687a445ccd592e1fcf7887b544d82a.

The bars, per profiled step, EXACT (``==``): the clean records sit exactly at them, and a double dispatch is a defect too. E is the arm's
own expert-module forward count (the selected backend's wrapper, recompute included; E == 2*L*A). The backwards number E/2.
- Triton path (``_grouped_gemm_kernel`` in top_device): grouped GEMM launches == 9E (6E forward: two projections x {base, LoRA A, LoRA
  B}; plus 3E backward dX, the same six per backward) and ``_grouped_wgrad_kernel`` == 2E (LoRA A and B dW for two projections; the frozen
  base has no dW). Derived from the source above.
- torch path (``aten::_grouped_mm`` in top_cpu, no Triton kernel): == 11E, OBSERVED on unsloth_zoo 2026.9.9 (TC1 amendment 69's box,
  exactly 4224 = 11E). It's registered for that zoo version only. On any other version a torch-path dispatch has no registered bar and
  VOIDs.
- Both paths in one arm (a mixed dispatch): VOID, because no bar is registered for it.
- Always: E == 2*L*A, and v1's manual per-group fallback counter == 0.

Fallbacks only these counts catch (they have no counter of their own): moe_utils.py:928-929 (``_grouped_matmul_loop`` when grad is off)
and moe_grouped_modulelist.py:180-185 (the inline per-group loop). v1's manual counter wraps ``_ManualGroupedMM`` on the imported
moe_utils module and is blind to its compile-cache copy (``_load_cached_moe_utils_module``, moe_utils.py:297). The device-kernel count
doesn't depend on the module. The one-time probes (moe_utils.py:1637, 1717, 1807-1808; moe_grouped_modulelist.py:97;
moe_grouped_fp16.py:788-789) run at first use, before the profiled window. The profiler starts after ``profile_warm`` steps (a 0-based
loop index: warm W profiles steps W+1..W+3, counting from 1); W is 3 at amendments 69 and 79 and 20 at amendment 80.

Truncation. A profile file keeps the top 80 rows of each list (tc1_arm.py:3600-3601). An absent row counts 0, so a dispatch ranked
below the kept rows changes an exact count, and the arm VOIDs. It is never a pass.

Kernels-only device time. tc1_arm.summarize_profile sums every CUDA-typed event into ``device_ms``, user-annotation spans included. The
``Optimizer.step#…`` row is such a span: it overlaps the optimizer kernels it encloses, gaps included. torch 2.8.0's own device-time
sum excludes them (profiler_util.py:1010-1020, ``not evt.is_user_annotation``). The stored profile keeps no such flag, so this subtracts
the rows whose NAME is a known annotation (``ANNOTATION_RE``) and records what it removed. It fails CLOSED: exactly one
``Optimizer.step#`` row with count == profiled_steps is required (it is in all 120 of TC1's stored profiles), and anything else raises,
so a missing or truncated annotation is never read as kernels-only. TC1 amendment 81 fixes the instrument at the source and marks its
profiles ``summary.device_ms_method == "kernels_only_at_source"``. This function REFUSES such a profile: subtracting the row again would
correct it twice. A legacy profile carries no field, or ``"all_cuda"``.
"""
from __future__ import annotations

import re

CPU_GROUPED = "aten::_grouped_mm"
TRITON_GEMM = "_grouped_gemm_kernel"
TRITON_WGRAD = "_grouped_wgrad_kernel"
TRITON_GEMM_PER_E, TRITON_WGRAD_PER_E = 9, 2
#: the torch path's bar, by unsloth_zoo version: observed, never derived (that version's source was not read)
TORCH_GEMM_PER_E = {"2026.9.9": 11}
BACKEND_KEY = {"grouped_mm": "unsloth_grouped_mm", "unsloth_triton": "unsloth_triton", "native_torch": "unsloth_loop"}
#: user-annotation rows seen in TC1's stored profiles (all 120 carry ``Optimizer.step#AdamW8bit.step``)
ANNOTATION_RE = re.compile(r"^(Optimizer\.step#|ProfilerStep#)")


def _per_step(rows, name, steps):
    return sum(r["count"] for r in rows if r["name"] == name) / steps


def counts(profile) -> dict:
    """Per-step counts of the three counted dispatches in one profile."""
    ps = profile["summary"]["profiled_steps"]
    return {"torch_grouped": _per_step(profile["top_cpu"], CPU_GROUPED, ps),
            "triton_gemm": _per_step(profile["top_device"], TRITON_GEMM, ps),
            "triton_wgrad": _per_step(profile["top_device"], TRITON_WGRAD, ps)}


def engagement(cell, profile):
    """("ENGAGED" | "VOID", why, detail) for one Unsloth arm."""
    L, A = cell["n_layers"], cell["accum"]
    key = BACKEND_KEY.get(cell.get("moe_backend_selected"))
    detail = {"path": None}
    if key is None:
        return "VOID", f"moe_backend_selected {cell.get('moe_backend_selected')!r} is not a registered backend", detail
    if profile is None:
        return "VOID", "no profile: F15 v2 reads a profiled arm", detail
    E = (cell.get("unsloth_backend_calls_per_step_min") or {}).get(key, 0)
    zoo = (cell.get("env") or {}).get("unsloth_zoo")
    c = counts(profile)
    detail.update(c, E=E, zoo=zoo)
    why = []
    if E != 2 * L * A:
        why.append(f"expert forwards {E}/step != 2*L*A = {2 * L * A}")
    tri, tor = c["triton_gemm"] > 0 or c["triton_wgrad"] > 0, c["torch_grouped"] > 0
    if tri and tor:
        detail["path"] = "mixed"
        why.append(f"mixed dispatch (Triton {c['triton_gemm']:.0f}, torch {c['torch_grouped']:.0f}/step): no bar is registered for it")
    elif tri:
        detail["path"] = "triton"
        if c["triton_gemm"] != TRITON_GEMM_PER_E * E:
            why.append(f"Triton grouped GEMMs {c['triton_gemm']:.0f}/step != {TRITON_GEMM_PER_E}E = {TRITON_GEMM_PER_E * E}")
        if c["triton_wgrad"] != TRITON_WGRAD_PER_E * E:
            why.append(f"Triton dW {c['triton_wgrad']:.0f}/step != {TRITON_WGRAD_PER_E}E = {TRITON_WGRAD_PER_E * E}")
    elif tor:
        detail["path"] = "torch"
        per = TORCH_GEMM_PER_E.get(zoo)
        if per is None:
            why.append(f"torch-path dispatch on unsloth_zoo {zoo!r}: no bar is registered for that version")
        elif c["torch_grouped"] != per * E:
            why.append(f"torch grouped GEMMs {c['torch_grouped']:.0f}/step != {per}E = {per * E}")
    else:
        detail["path"] = "none"
        why.append("no grouped GEMM op or kernel in the profile's kept rows")
    manual = cell.get("unsloth_manual_grouped_mm_calls_per_step_max")
    if manual != 0:
        why.append(f"manual per-group fallback {manual!r}/step (must be 0)")
    return ("VOID", "; ".join(why), detail) if why else ("ENGAGED", f"E {E}/step on the {detail['path']} path", detail)


def v1_line(cell) -> str:
    """F15 v1's own reading, printed beside v2 and labelled: reported, not binding."""
    return (f"F15 v1 (reported, not binding): torch._grouped_mm {cell.get('unsloth_grouped_mm_calls_per_step_min')}/step "
            f"against 6*L*A = {6 * cell['n_layers'] * cell['accum']}")


class AnnotationError(ValueError):
    """The profile's annotation rows are not the one the kernels-only figure can subtract."""


def kernels_only_device_ms(profile):
    """(kernel ms per profiled step, annotation ms per profiled step, annotation row names): recorded device_ms minus the rows whose
    name is a known user annotation. Raises AnnotationError unless exactly one ``Optimizer.step#`` row is present with count ==
    profiled_steps. A missing row would leave the annotation span inside the figure, unnoticed."""
    method = profile["summary"].get("device_ms_method", "all_cuda")
    if method != "all_cuda":
        raise AnnotationError(f"device_ms_method {method!r}: only a legacy all-CUDA device_ms (no field, or 'all_cuda') is corrected here; "
                              "an at-source kernels-only figure would be corrected twice")
    ps = profile["summary"]["profiled_steps"]
    ann = [r for r in profile["top_device"] if ANNOTATION_RE.match(r["name"])]
    opt = [r for r in ann if r["name"].startswith("Optimizer.step#")]
    if len(opt) != 1 or int(opt[0].get("count") or 0) != int(ps):
        raise AnnotationError(f"{len(opt)} Optimizer.step# annotation rows (counts {[r.get('count') for r in opt]}) where exactly one with "
                              f"count {ps} (one a profiled step) is required: the span cannot be removed, so kernels-only is unread")
    a_ms = sum(r["self_device_ms"] for r in ann) / ps
    return profile["summary"]["device_ms"] / ps - a_ms, a_ms, [r["name"] for r in ann]
