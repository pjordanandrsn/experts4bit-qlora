#!/usr/bin/env python3
"""p238_reduce.py -- P238's rule (bench/p238/PREREG-p238.md "Validity, quality and the rule"), over normalized rows.

A step-load ROW is one tc1_arm process (one point, one arm, one draw); an OP ROW is one op process (p238_op.py). ``normalize_*``
builds them from the receipts. Every check here FAILS CLOSED: a missing, None or empty input is a fault, never a pass, and the
self-test (``--selftest``) removes each piece of evidence in turn to show it.

Row-level checks (any fault VOIDs that row): presence and status; the pins; the card; the card floor; the premise table equal to
the reference table on 40 layers; the absmax bind (e4b nested, required; Unsloth loaded nested); the named Unsloth loader; the
adapter census equal across every load and op process; the engagement checks per arm; the step contract; amendment 81's device-time
marker. Pair-level: quality (within / across framework) and stability (the two draws of a (point, arm)).

Standard library only, plus bench/common/unsloth_moe_counter.py for the Unsloth checks (the same functions the lane's counter
module defines, so the reducer cannot count differently).
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import unsloth_moe_counter as C  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "tc1"))
import f15v2  # noqa: E402  -- TC1 amendment 80's merged F15 v2 bars, read, never re-typed

N_LAYERS, STEPS, TIMED = 40, 60, (10, 60)            # timed steps 11..60 (0-based 10..59)
POINT_TOKENS = {"2k": 2048, "4k": 4096, "8k": 8192, "field": None}
ARMS = {"E_F": "e4b", "E_A": "e4b", "U0": "unsloth", "U1": "unsloth", "UA": "unsloth"}
PINS = {"experts4bit-qlora": "0.58.1", "grouped-nf4-gemm": "0.51.0", "unsloth": "2026.10.8", "unsloth_zoo": "2026.10.8",
        "peft": "0.21.2", "bitsandbytes": "0.50.2"}
TORCH_PREFIX = "2.12.1"
CLOCK_FLOOR_MHZ, SWPC_CLEAR = 2700, 0.95
DENSE_AUTO_MAX_GROUPS = 16                           # gnf4 0.51.0 nf4_route: auto on sm_120 is dense at 1..16 present groups
UNSLOTH_PINS = {"UNSLOTH_MOE_NF4_AUTO_MAX_ROWS": "128", "UNSLOTH_MOE_NF4_AUTOTUNE": "0", "UNSLOTH_MOE_GROUPED_TRITON": "auto",
                "UNSLOTH_MOE_FUSED_ROUTER": "0", "UNSLOTH_ENABLE_LOGGING": "1"}
NF4_MODE = {"U0": "0", "U1": "1", "UA": "auto"}
# The box's environment: ONE list for three places, bound by the kit test (a mutant widening any one fails it): the runner's
# ambient refusal (any key under these prefixes but the registered ones refuses, 78, before STARTED), its env_box.txt grep (these
# prefixes plus the exact names), and check_env_box's registered set. Any other PYTORCH_* key is information (env_image.txt, never
# a fault): it reaches both frameworks on the box alike, and the image is pinned.
ENV_PREFIXES = ("UNSLOTH", "E4B", "GNF4", "NF4_QLORA")
ENV_EXACT = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
ENV_BOX = {**UNSLOTH_PINS, **ENV_EXACT, "E4B_SHA": None}      # None: any 40-hex commit (the driver's clean HEAD)
ENV_REFUSE_GREP = "^(" + "|".join(ENV_PREFIXES) + ")_[A-Za-z0-9_]+"
ENV_BOX_GREP = "^(" + "|".join(p + "_" for p in ENV_PREFIXES) + ")|" + "|".join(f"^{k}=" for k in ENV_EXACT)
DEVICE_MS_METHOD = "kernels_only_at_source"         # amendment 81's marker (f15v2.py's docstring names it; the kit test pins it)
FUSED_KERNEL = "_gemm_nf4_grouped"                   # grouped-nf4-gemm's fused grouped forward kernel
PROFILED = (20, 23)                                  # --profile-warm 20 --profile-steps 3: loop steps 20..22, edges 20 and 23
ENERGY_SPANS = ((10, PROFILED[0]), (PROFILED[1], 59))  # steps 11-20 and 24-59 (46): the profile window and the closing eval left out
MIN_ENERGY_S, MIN_POWER_SAMPLES = 10.0, 40
GAP_FLOOR_S, GAP_FRAC = 0.2, 0.2                     # non-step wall inside one window edge pair: > max(0.2 s, 0.2 x the step) is an eval
E_F_PROFILED_FORWARDS = 2 * 2 * N_LAYERS * 3         # 2 passes (HF non-reentrant recompute) x 2 projections x 40 x 3 = 480
OP_ROWS, OP_ROUTINGS, OP_REPS, OP_TOL = (32, 64, 128, 256, 512), ("uniform", "skew"), 20, 2e-2
OP_ARMS = {"e4b": ("E_F", "E_A"), "unsloth": ("U0", "U1", "UA")}
WITHIN_STEP0, WITHIN_HELD = 1e-3, 0.01
ACROSS_STEP0, ACROSS_HELD = 0.01, 0.05               # TC1-PREREG.md:63 (step 0) and TC1's COMPARABLE (held-out)
STABLE = {"e4b": 0.05, "unsloth": 0.10}
OOM_EXIT = 5                                         # tc1_arm's exit for its OOM stub (classify_load_exception, the train loop)
READ_PAIRS = (("E_F", "U1"), ("E_A", "UA"), ("U1", "U0"), ("E_F", "U0"), ("E_A", "E_F"))   # the headline two first (PREREG)


# -- row checks ----------------------------------------------------------------------------------------------------------------------
def check_present(row):
    why = []
    if not row:
        return ["no record"]
    if row.get("dry"):
        why.append("a dry run")
    if row.get("status") != "ok":
        why.append(f"status {row.get('status')!r}")
    return why


def check_pins(row):
    v = row.get("versions") or {}
    why = [f"{k} {v.get(k)!r} != {want}" for k, want in PINS.items() if v.get(k) != want]
    if not str(v.get("torch") or "").startswith(TORCH_PREFIX):
        why.append(f"torch {v.get('torch')!r} is not {TORCH_PREFIX}")
    if not v.get("release_proof_ok"):
        why.append("the release proof is missing or refused")
    return why


def check_env_box(row):
    """The box's env_box.txt: its key set must EQUAL ENV_BOX (the five Unsloth pins, PYTORCH_CUDA_ALLOC_CONF, E4B_SHA), at the
    registered values. The per-arm GNF4_TRAIN_GEMM, E4B_ABSMAX_DQ and UNSLOTH_MOE_NF4_FUSED are set only on each arm's own command
    line, after the dump. No file, an unregistered key or a missing one VOIDs."""
    env = row.get("env_box")
    if env is None:
        return ["no env_box.txt"]
    why = [f"env_box: an unregistered {k}" for k in sorted(set(env) - set(ENV_BOX))]
    why += [f"env_box: {k} missing" for k in sorted(set(ENV_BOX) - set(env))]
    why += [f"env_box: {k}={env[k]!r} != {v!r}" for k, v in ENV_BOX.items() if v is not None and k in env and env[k] != v]
    if "E4B_SHA" in env and not re.fullmatch(r"[0-9a-f]{40}", env["E4B_SHA"] or ""):
        why.append(f"env_box: E4B_SHA {env['E4B_SHA']!r} is not a commit")
    return why


def check_card(row):
    c = row.get("card") or {}
    why = []
    if "5090" not in str(c.get("name") or ""):
        why.append(f"card {c.get('name')!r}")
    if list(c.get("cap") or []) != [12, 0]:
        why.append(f"cc {c.get('cap')}")
    if (c.get("sm_count") or 0) < 160:
        why.append(f"SMs {c.get('sm_count')}")
    k = row.get("clock") or {}
    if (k.get("busy_samples") or 0) < 10:
        why.append(f"{k.get('busy_samples') or 0} busy clock samples < 10 (the card was not read under load)")
    if k.get("busy_median_mhz") is None or k["busy_median_mhz"] < CLOCK_FLOOR_MHZ:
        why.append(f"busy SM median {k.get('busy_median_mhz')} < {CLOCK_FLOOR_MHZ} MHz")
    if k.get("hard_reason_any") is not False:
        why.append(f"a hard clock reason set (or unread): {k.get('hard_reason_any')}")
    if k.get("swpowercap_clear_frac") is None or k["swpowercap_clear_frac"] < SWPC_CLEAR:
        why.append(f"SwPowerCap clear on {k.get('swpowercap_clear_frac')} of busy samples < {SWPC_CLEAR}")
    return why


def check_premise(row, ref_table):
    sys.path.insert(0, HERE)
    import p238_bytes
    ok, why = p238_bytes.compare(row.get("premise_table") or {}, ref_table or {}, n_layers=N_LAYERS)
    return [] if ok else [f"premise: {why}"]


def check_absmax(row):
    a = row.get("absmax") or {}
    why = []
    if row.get("framework") == "e4b":
        if a.get("tc1_modules") != N_LAYERS:
            why.append(f"--absmax-dq compressed {a.get('tc1_modules')} stacks, not {N_LAYERS}")
        if a.get("compressed_now") != N_LAYERS:
            why.append(f"{a.get('compressed_now')} of {N_LAYERS} stacks read compressed")
        ft = a.get("fast_train") or {}
        if ft.get("required") is not True or "kept_fp32" not in ft or ft.get("kept_fp32") is not None:
            why.append(f"FAST_TRAIN_STATS absmax_dq {ft}")
    elif row.get("framework") == "unsloth":
        if a.get("loaded_nested") is not True:
            why.append(f"Unsloth's loaded absmax nested = {a.get('loaded_nested')}")
        if row.get("loader_used") != "FastModel":
            why.append(f"loader_used {row.get('loader_used')!r}, not FastModel")
    else:
        why.append(f"framework {row.get('framework')!r}")
    return why


def census_key(census):
    """The adapter census as a comparable value (every (layer, projection) with its counts, shapes, scaling and A bytes)."""
    return json.dumps(census or {}, sort_keys=True)


def check_census(row, ref_census):
    c = row.get("adapters") or {}
    why = []
    want_keys = {f"{L}:{p}" for L in range(N_LAYERS) for p in ("gate_up_proj", "down_proj")}
    if not want_keys <= set(c):
        why.append(f"adapter census lacks {len(want_keys - set(c))} expert (layer, projection) entries")
    if not ref_census or census_key(c) != census_key(ref_census):
        why.append("adapter census differs from the reference (counts, shapes, scaling or A bytes)")
    return why


def _delta(first, last, *path):
    a, b = first, last
    for p in path:
        a, b = (a or {}).get(p), (b or {}).get(p)
    return None if a is None or b is None else b - a


def check_e4b_engagement(row):
    h, arm = row.get("hook") or {}, row.get("arm")
    f, edges = h.get("facts") or {}, h.get("edges") or []
    why = []
    if len(edges) != STEPS + 1:
        return [f"{len(edges)} step edges, not {STEPS + 1}"]
    if f.get("fast_train_patched") != N_LAYERS or f.get("fast_train_skipped") != {}:
        why.append(f"fast_train patched {f.get('fast_train_patched')}, skipped {f.get('fast_train_skipped')}")
    first, last = edges[0], edges[-1]
    kernel = _delta(first, last, "dgrad", "kernel")
    dense_dgrad = _delta(first, last, "route", "dense_dgrad")
    want = 2 * N_LAYERS * STEPS                        # one backward per layer per step, a dX per projection
    log = {}
    for k, v in (last.get("route_log") or {}).items():
        log[k] = v - ((first.get("route_log") or {}).get(k, 0))
    if kernel is None or dense_dgrad is None or not log:
        return why + ["the dgrad, route or route-log counters are unread"]
    bad = {"grouped_mm": ("fwd", "dgrad"), "decoded": ("decoded_fwd", "decoded_dgrad")}
    for name, keys in bad.items():
        n = sum(_delta(first, last, "route", k) or 0 for k in keys)
        if n:
            why.append(f"{n} {name} route calls")
    prof = row.get("profile") or {}
    kernels = sum(r.get("count", 0) for r in prof.get("top_device") or [] if r.get("name") == FUSED_KERNEL)
    plog_a, plog_b = (edges[PROFILED[0]].get("route_log") or {}), (edges[PROFILED[1]].get("route_log") or {})
    fused_logged = sum(v - plog_a.get(k, 0) for k, v in plog_b.items() if k.endswith(":fused"))
    if not prof.get("top_device"):
        why.append("no profiled device kernels")
    elif kernels != fused_logged:
        why.append(f"profiled {FUSED_KERNEL} {kernels} != {fused_logged} fused routes logged in the profiled steps")
    if arm == "E_F":
        if kernels != E_F_PROFILED_FORWARDS:
            why.append(f"profiled {FUSED_KERNEL} {kernels} != {E_F_PROFILED_FORWARDS}")
        if any(not k.endswith(":fused") for k, v in log.items() if v):
            why.append(f"a non-fused route at GNF4_TRAIN_GEMM=fused: {log}")
        if kernel != want or dense_dgrad:
            why.append(f"dgrad kernel {kernel} (dense {dense_dgrad}) != {want}")
    elif arm == "E_A":
        for k, v in log.items():
            try:
                n, route = k.split(":")
                expect = "dense" if n not in ("None", "") and 1 <= int(n) <= DENSE_AUTO_MAX_GROUPS else "fused"
            except ValueError:
                why.append(f"an unreadable route-log key {k!r}")
                continue
            if v and route != expect:
                why.append(f"{v} calls at n_groups {n} took {route}, route_for says {expect}")
        dense_logged = sum(v for k, v in log.items() if k.endswith(":dense"))
        if _delta(first, last, "route", "dense_fwd") != dense_logged:
            why.append(f"ROUTE_STATS dense_fwd {_delta(first, last, 'route', 'dense_fwd')} != {dense_logged} logged")
        if kernel + dense_dgrad != want:
            why.append(f"dgrad kernel {kernel} + dense {dense_dgrad} != {want}")
    else:
        why.append(f"arm {arm!r} is not an e4b arm")
    return why


def _safe(fn, *args, **kw):
    """(ok, why) from a counter check; a snapshot it cannot read is a FAULT, never a crash and never a pass."""
    try:
        return fn(*args, **kw)
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        return False, f"{fn.__name__}: unreadable snapshot ({type(e).__name__}: {e})"


def check_unsloth_engagement(row):
    h, arm = row.get("hook") or {}, row.get("arm")
    edges = h.get("edges") or []
    if len(edges) != STEPS + 1:
        return [f"{len(edges)} step edges, not {STEPS + 1}"]
    snap = edges[-1].get("unsloth")
    if not snap:
        return ["no counter snapshot"]
    st = h.get("structural")
    if not st or st.get("steps") != STEPS:
        return [f"no structural count over {STEPS} steps ({(st or {}).get('steps')})"]
    why = [f"structure: {f}" for f in st.get("faults") or []]
    env = dict(row.get("unsloth_env") or {})
    pins = {**UNSLOTH_PINS, "UNSLOTH_MOE_NF4_FUSED": NF4_MODE.get(arm)}
    for ok, w in (_safe(C.check_env, snap, env, pins), _safe(C.check_library, snap),
                  _safe(C.check_dispatch, snap, st, base_must_be_triton=True)):
        if not ok:
            why.append(w)
    if arm == "U0":
        ok, w = _safe(C.check_unfused, snap)
        if not ok:
            why.append(w)
        try:
            verdict, w, _ = f15v2.engagement(row.get("cell") or {}, row.get("profile") or None)
        except (KeyError, TypeError, ZeroDivisionError) as e:
            verdict, w = "VOID", f"unreadable cell or profile ({type(e).__name__}: {e})"
        if verdict != "ENGAGED":
            why.append(f"F15 v2: {w}")
    elif arm in ("U1", "UA"):
        try:
            exp = C.expected_launches(snap, st)
        except (KeyError, TypeError) as e:
            exp, why = None, why + [f"expected_launches: unreadable snapshot ({type(e).__name__})"]
        ok, w = _safe(C.check_fused, snap, exp)
        if not ok:
            why.append(w)
        if arm == "U1" and (snap.get("counts") or {}).get("nf4_declined", 1):
            why.append(f"{(snap.get('counts') or {}).get('nf4_declined')} calls declined (or unread) at UNSLOTH_MOE_NF4_FUSED=1")
        if arm == "UA":
            ok, w = _safe(C.check_auto, snap)
            if not ok:
                why.append(w)
    else:
        why.append(f"arm {arm!r} is not an Unsloth arm")
    return why


def check_steps(row):
    why = []
    steps = row.get("tokens_per_step") or []
    if len(steps) != STEPS or len(row.get("step_ms") or []) != STEPS:
        return [f"{len(steps)} steps recorded, not {STEPS}"]
    if row.get("micro_batches_per_step") != 1:
        why.append(f"{row.get('micro_batches_per_step')} micro-batches a step, not 1")
    want = POINT_TOKENS.get(row.get("point"), "?")
    if want == "?":
        why.append(f"point {row.get('point')!r}")
    elif want is not None and any(t != want for t in steps):
        why.append(f"a step is not {want} tokens: {sorted(set(steps))[:4]}")
    m = ((row.get("profile") or {}).get("summary") or {}).get("device_ms_method")
    if m != DEVICE_MS_METHOD:
        why.append(f"profile device_ms_method {m!r}, not {DEVICE_MS_METHOD!r}")
    return why


def row_faults(row, ref_table, ref_census):
    why = check_present(row)
    if why:
        return why
    why += check_pins(row) + check_env_box(row) + check_card(row) + check_premise(row, ref_table) + check_absmax(row)
    why += check_census(row, ref_census)
    why += (check_e4b_engagement(row) if row.get("framework") == "e4b" else check_unsloth_engagement(row))
    why += check_steps(row)
    return why


# -- the readings --------------------------------------------------------------------------------------------------------------------
def s_per_step(row):
    ms = (row.get("step_ms") or [])[TIMED[0]:TIMED[1]]
    return statistics.median(ms) / 1e3 if ms else None


def quality(a, b):
    """(ok, why) for two READ rows at the same point: within-framework 1e-3 / 0.01, across 0.01 / 0.05."""
    within = ARMS.get(a.get("arm")) == ARMS.get(b.get("arm"))
    s0, ho = (WITHIN_STEP0, WITHIN_HELD) if within else (ACROSS_STEP0, ACROSS_HELD)
    try:
        d0 = abs(a["step0_loss"] - b["step0_loss"])
        dh = abs(a["heldout_loss"] - b["heldout_loss"])
    except (KeyError, TypeError):
        return False, "a step-0 or held-out loss is missing"
    why = []
    if d0 > s0:
        why.append(f"|d step-0| {d0:.4g} > {s0}")
    if dh > ho:
        why.append(f"|d held-out| {dh:.4g} > {ho}")
    return (not why), "; ".join(why)


def stable(d1, d2):
    """(ok, why): a (point, arm)'s two draws within 5 % (e4b) / 10 % (Unsloth) on wall s/step and device time."""
    fw = ARMS.get(d1.get("arm"))
    bar = STABLE.get(fw)
    why = []
    for name, get in (("wall", s_per_step), ("device", lambda r: ((r.get("profile") or {}).get("summary") or {}).get("device_ms"))):
        x, y = get(d1), get(d2)
        if x is None or y is None or bar is None:
            why.append(f"{name}: unread")
        elif abs(x - y) / ((x + y) / 2) > bar:
            why.append(f"{name}: draws {x:.4g} / {y:.4g} apart by more than {bar:.0%}")
    return (not why), "; ".join(why)


def op_faults(o):
    """An op process's record (p238_op.py): the registered cells, every arm's parity (the non-zero-B layout check; a miss is the
    PREREG's STOP), B restored, the visit count, and every timing sample positive and finite."""
    import math
    why = []
    if not o or o.get("p238_op") != 1:
        return ["no op record"]
    if o.get("dry"):
        why.append("a dry run")
    if o.get("capture") != "eager":
        why.append(f"capture {o.get('capture')!r}, not eager")
    arms = OP_ARMS.get(o.get("framework"))
    if not arms or list(o.get("arms") or []) != list(arms):
        return why + [f"framework {o.get('framework')!r} with arms {o.get('arms')}"]
    cells = {(c.get("rows"), c.get("routing")): c for c in o.get("cells") or []}
    want = {(r, k) for r in OP_ROWS for k in OP_ROUTINGS}
    if set(cells) != want:
        why.append(f"cells {sorted(set(cells) ^ want, key=str)[:4]} missing or unregistered")
    for key, c in sorted(cells.items(), key=str):
        par = c.get("parity") or {}
        for arm in arms:
            p = par.get(arm) or {}
            errs = p.get("errs") or {}
            if not errs or not all(isinstance(v, (int, float)) and v <= OP_TOL for v in errs.values()):
                why.append(f"LAYOUT/PARITY STOP: {key} {arm} errs {errs}")
        if c.get("b_restored_zero") is not True:
            why.append(f"{key}: B not restored to zero")
        v = c.get("visits") or []
        if len(v) != OP_REPS * 2 * len(arms):
            why.append(f"{key}: {len(v)} visits, not {OP_REPS * 2 * len(arms)}")
        bad = [x for x in v for t in ("device_ms_per_call", "wall_ms_per_call")
               if not isinstance(x.get(t), (int, float)) or not math.isfinite(x[t]) or x[t] <= 0]
        if bad:
            why.append(f"{key}: {len(bad)} timing samples missing, non-positive or not finite")
    return why


def energy(row):
    """The step energy (PREREG: NVML's total-energy counter, read by the hook at every step edge): J per step and J per token over
    two spans that leave out the profile window, edges 10..20 and 23..59 (steps 11-20 and 24-59, 46 steps):
    J = (E20 - E10) + (E59 - E23). tc1_arm starts the profiler after edge 20 and stops it (and writes the profile) before edge
    23, outside step_ms, and loop steps 20-22 (steps 21-23) run under CUPTI tracing, so the excluded pairs 20..23 carry profiling the two frameworks pay
    differently. The span is taken from the row's own profile_warm / profile_steps, which must equal PROFILED, so a flag change can
    never fold the profiler back in. No eval sits in a span: the runner passes --eval-every 60, so the only eval is the held-out
    after the last step, after edge 59. Edges are taken by their own index (``i``), every edge of a span must be there, and each
    included edge pair's wall may exceed tc1_arm's step time (which excludes evals) by at most max(0.2 s, 0.2 x the step): a pair
    holding more reads UNREAD, so non-step work can never move the number. UNREAD, never 0, also when an edge or its counter is
    missing, or the spans hold under 10 s. The cross-check: the mean of power.draw.instant (0.25 s samples) over the same two spans,
    read with >= 40 samples."""
    out = {"method": "nvml_total_energy", "spans": [list(s) for s in ENERGY_SPANS], "j_per_step": None, "j_per_token": None,
           "status": "UNREAD"}
    warm, nprof = row.get("profile_warm"), row.get("profile_steps")
    if not (isinstance(warm, int) and isinstance(nprof, int)) or (warm, warm + nprof) != PROFILED:
        out["why"] = f"the row's profile window ({warm}, {nprof}) is not the registered edges {PROFILED}"
        return out
    by_i = {}
    for e in ((row.get("hook") or {}).get("edges")) or []:
        if isinstance(e, dict) and isinstance(e.get("i"), int):
            by_i[e["i"]] = e
    missing = [i for a, b in ENERGY_SPANS for i in range(a, b + 1) if i not in by_i]
    if missing:
        out["why"] = f"edge(s) {missing[:5]} missing from the spans {ENERGY_SPANS}"
        return out
    step_ms, tps = row.get("step_ms") or [], row.get("tokens_per_step") or []
    mj = secs = toks = steps = 0
    spans = []
    for a, b in ENERGY_SPANS:
        ea, eb, ta, tb = by_i[a].get("energy_mj"), by_i[b].get("energy_mj"), by_i[a].get("t"), by_i[b].get("t")
        if None in (ea, eb, ta, tb):                # a backwards counter is the pair check's (every included pair)
            out["why"] = f"span {a}..{b}: energy {ea}->{eb} mJ, t {ta}->{tb}"
            return out
        for i in range(a, b):
            t0, t1 = by_i[i].get("t"), by_i[i + 1].get("t")
            e0, e1 = by_i[i].get("energy_mj"), by_i[i + 1].get("energy_mj")
            ms = step_ms[i] if i < len(step_ms) else None
            if None in (t0, t1, ms):
                out["why"] = f"edges {i}..{i + 1}: no wall or no step time"
                return out
            if None in (e0, e1) or e1 < e0:      # a counter reset inside a span understates it even when the span ends higher
                out["why"] = f"edges {i}..{i + 1}: the energy counter went {e0}->{e1} mJ"
                return out
            gap = (t1 - t0) - ms / 1e3
            if gap > max(GAP_FLOOR_S, GAP_FRAC * ms / 1e3):
                out["why"] = f"edges {i}..{i + 1} hold {gap:.2f} s of non-step wall (an eval in a span)"
                return out
        mj, secs, toks, steps = mj + eb - ea, secs + tb - ta, toks + sum(tps[a:b]), steps + b - a
        spans.append((ta, tb))
    if secs < MIN_ENERGY_S or not toks:
        out["why"] = f"spans {secs:.1f} s (needs >= {MIN_ENERGY_S}) or {toks} tokens"
        return out
    joules = mj / 1e3
    out.update(status="READ", steps=steps, seconds=round(secs, 3), joules=round(joules, 3), j_per_step=round(joules / steps, 4),
               j_per_token=round(joules / toks, 6))
    samples = [(t, w) for t, w in (row.get("power_samples") or []) if any(ta <= t <= tb for ta, tb in spans)]
    if len(samples) >= MIN_POWER_SAMPLES:
        sampled = 0.0
        for ta, tb in spans:
            ws = [w for t, w in samples if ta <= t <= tb]
            if not ws:
                break
            sampled += sum(ws) / len(ws) * (tb - ta)
        else:
            out["sampled_j_per_step"] = round(sampled / steps, 4)
    out["sampled_n"] = len(samples)
    return out


def is_oom_row(row):
    return bool(row) and row.get("status") == "oom"


def oom_faults(row):
    """A step row that is tc1_arm's OOM stub (status "oom", exit 5, with its phase) is an OOM outcome, a READ result (the PREREG:
    "An OOM is a result"), when it ran on the registered box: the release proof, env_box.txt and the card. A row that is a
    harness_error, a hook fault (exit 10) or anything else not "ok" stays VOID (check_present)."""
    why = ["a dry run"] if row.get("dry") else []
    if row.get("rc") != OOM_EXIT:
        why.append(f"exit {row.get('rc')!r}, not tc1_arm's OOM exit {OOM_EXIT}")
    if not (isinstance(row.get("phase"), str) and row["phase"]):
        why.append("an OOM stub with no phase")
    if not (row.get("versions") or {}).get("release_proof_ok"):
        why.append("the release proof is missing or refused")
    return why + check_env_box(row) + check_card(row)


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def readings(out_rows, raw):
    """The PREREG's readings per step point and pair (the headline E_F/U1 and E_A/UA first): a ratio of the two arms' mean wall
    s/step and mean device time over the draws (a/b, < 1 means a is faster) when both arms READ; BOTH_OOM when every draw of both
    arms OOMs (that point leaves the step read); OOM_DIFFERENCE when one arm's draws all OOM and the other's all READ (the READ
    memory difference, with the fitting arm's peak VRAM); VOID when any draw is VOID. An OOM row is never in a speed ratio."""
    by = {}
    for o in out_rows:
        by.setdefault((o["point"], o["arm"]), []).append(o)
    res = {}
    for point in POINT_TOKENS:
        for a, b in READ_PAIRS:
            ga, gb = by.get((point, a), []), by.get((point, b), [])
            va, vb = {o["verdict"] for o in ga}, {o["verdict"] for o in gb}
            r = {"point": point, "a": a, "b": b, "headline": (a, b) in READ_PAIRS[:2]}
            if not ga or not gb:
                r["status"] = "NOT_RUN"
            elif "VOID" in va | vb:
                r["status"] = "VOID"
            elif va == vb == {"OOM"}:
                r["status"] = "BOTH_OOM"
            elif {frozenset(va), frozenset(vb)} == {frozenset({"OOM"}), frozenset({"READ"})}:
                fits = b if va == {"OOM"} else a
                r.update(status="OOM_DIFFERENCE", oom=a if fits == b else b, fits=fits,
                         fits_peak_vram_gb=_mean(raw[o["index"]].get("peak_vram_gb") for o in by[(point, fits)]))
            elif va == vb == {"READ"}:
                dev = lambda g: _mean(((raw[o["index"]].get("profile") or {}).get("summary") or {}).get("device_ms") for o in g)  # noqa: E731
                wa, wb, da, db = _mean(o["s_per_step"] for o in ga), _mean(o["s_per_step"] for o in gb), dev(ga), dev(gb)
                r.update(status="RATIO", wall_ratio=round(wa / wb, 4) if wa and wb else None,
                         device_ratio=round(da / db, 4) if da and db else None)
            else:
                r["status"] = "VOID"
            res[f"{point}:{a}/{b}"] = r
    return res


def reduce(rows, op_rows):
    """The verdict per row, the reference tables, the readings, and the box verdict. The reference premise table and census are
    the first OP row's (an op process loads exactly as a step load does); every other table must equal them. A step row is READ,
    OOM (tc1_arm's OOM stub on the registered box) or VOID; an OOM must REPRODUCE: every draw of that (point, arm) OOMs, else
    every draw of it is VOID (an unstable memory reading). Op rows have no OOM outcome. The box is READ when every op row READs
    and every step row is READ or a reproduced OOM (each fault is reported on its row; a VOID row is never dropped)."""
    ref = (op_rows or [{}])[0]
    ref_table, ref_census = ref.get("premise_table"), ref.get("adapters")
    out = {"rows": [], "op_rows": []}
    for o in op_rows or []:
        why = op_faults(o.get("record")) + check_premise(o, ref_table) + check_census(o, ref_census)
        out["op_rows"].append({"framework": o.get("framework"), "draw": o.get("draw"), "verdict": "VOID" if why else "READ", "why": why})
    for r in rows:
        if is_oom_row(r):
            why = oom_faults(r)
            entry = {"verdict": "VOID" if why else "OOM", "why": why, "phase": r.get("phase"), "s_per_step": None, "energy": None}
        else:
            why = row_faults(r, ref_table, ref_census)
            entry = {"verdict": "VOID" if why else "READ", "why": why, "s_per_step": s_per_step(r), "energy": energy(r)}
        out["rows"].append({k: r.get(k) for k in ("index", "draw", "point", "arm")} | entry)
    groups = {}
    for o in out["rows"]:
        groups.setdefault((o["point"], o["arm"]), []).append(o)
    for (point, arm), g in groups.items():
        if any(o["verdict"] == "OOM" for o in g) and not (len(g) >= 2 and all(o["verdict"] == "OOM" for o in g)):
            for o in g:
                if o["verdict"] != "VOID":
                    o["verdict"] = "VOID"
                    o["why"] = o["why"] + [f"{arm} at {point}: an OOM that did not reproduce over its {len(g)} draw(s)"]
    out["readings"] = readings(out["rows"], {r.get("index"): r for r in rows})
    bad = [o for o in out["op_rows"] if o["verdict"] != "READ"] + [o for o in out["rows"] if o["verdict"] not in ("READ", "OOM")]
    out["verdict"] = "READ" if (out["rows"] or out["op_rows"]) and not bad else "VOID"
    out["n_rows"], out["n_op_rows"], out["n_void"] = len(out["rows"]), len(out["op_rows"]), len(bad)
    out["n_oom"] = sum(o["verdict"] == "OOM" for o in out["rows"])
    return out


# -- the receipts on disk -> rows ------------------------------------------------------------------------------------------------
RELEASES = ("experts4bit-qlora", "grouped-nf4-gemm", "unsloth", "unsloth_zoo", "peft", "bitsandbytes", "transformers")


def _int_keys(table):
    """The premise table after a JSON round trip: layer keys come back as strings; p238_bytes.compare wants the layer ints."""
    return {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in (table or {}).items()}


def _json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def release_proof_ok(path):
    """release_proof.py prints one JSON object per distribution; every registered one must say record_equal."""
    seen = {}
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("{"):
                d = json.loads(line)
                seen[d.get("dist")] = bool(d.get("record_equal"))
    except (OSError, ValueError):
        return False
    return all(seen.get(d) is True for d in RELEASES)


def clock_facts(path):
    """A load's gpuclk log through card_clock.py's own summary (the same parser and busy threshold as the burn)."""
    import card_clock
    try:
        sm = card_clock.summarize_log(path)
    except OSError:
        return {}
    return {"busy_median_mhz": sm.get("sm_median"), "hard_reason_any": bool(sm.get("hard_reasons_any_sample")),
            "swpowercap_clear_frac": sm.get("busy_soft_clean_fraction"), "busy_samples": sm.get("busy_samples")}


def power_samples(path):
    """nvidia-smi's ``timestamp, power.draw.instant`` lines (csv, noheader, nounits) -> [(epoch seconds, watts)]."""
    import datetime
    out = []
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            ts, w = (x.strip() for x in line.split(","))
            t = datetime.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
            out.append((t, float(w)))
        except ValueError:
            continue
    return out


def env_file(path):
    """``env``'s KEY=VALUE lines -> {key: value}; None when the file is missing. A line with no ``=`` keeps its text as a key, so it
    reads as an unregistered key (VOID), never as nothing."""
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return None
    out = {}
    for line in lines:
        if line.strip():
            k, sep, v = line.partition("=")
            out[k if sep else line] = v if sep else None
    return out


def proc_rcs(path):
    """The runner's summary.txt ``PROC <idx> step <arm> <point> draw=<d> pos=<p> rc=<rc>`` lines -> {idx: rc} (the process's exit,
    which tc1_arm's OOM stub carries only as its exit code)."""
    out = {}
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return out
    for line in lines:
        m = re.fullmatch(r"PROC (\d+) step \S+ \S+ draw=\S+ pos=\S+ rc=(\d+)", line.strip())
        if m:
            out[int(m.group(1))] = int(m.group(2))
    return out


def normalize_load(work, sched, box, proof_ok, env_box=None, rc=None):
    """One step load (a schedule row) from its tc1_arm receipt, its full profile, its clock log and the hook's record."""
    idx, arm, point = sched["index"], sched["arm"], sched["point"]
    fw = ARMS[arm]
    base = os.path.join(work, "loads", str(idx), f"qwen3_5_{fw}_p238_{idx}_{arm}_{point}")
    cell = _json(base + ".json") or {}
    prof = _json(base + "_profile.json")
    rec = ((cell.get("arm_hook") or {}).get("record")) or {}
    facts = rec.get("facts") or {}
    env = cell.get("env") or {}
    row = {"index": idx, "draw": sched["draw"], "point": point, "arm": arm, "framework": fw, "position": sched["position"],
           "status": cell.get("status"), "dry": False, "phase": cell.get("phase"), "rc": rc,
           "versions": {**{k: env.get(k) for k in PINS}, "torch": env.get("torch"), "release_proof_ok": proof_ok},
           "card": box, "env_box": env_box, "clock": clock_facts(os.path.join(work, "loads", f"gpuclk_{idx}.txt")),
           "premise_table": _int_keys(facts.get("premise_table")), "adapters": facts.get("adapters"),
           "loader_used": cell.get("loader_used"), "unsloth_env": env.get("unsloth_env"),
           "hook": {"facts": facts, "edges": rec.get("edges"), "structural": rec.get("structural")},
           "tokens_per_step": cell.get("tokens_per_step"), "step_ms": cell.get("step_ms"),
           "profile_warm": cell.get("profile_warm"), "profile_steps": cell.get("profile_steps"),
           "micro_batches_per_step": rec.get("micro_batches"), "profile": prof, "cell": cell,
           "step0_loss": cell.get("loss_first"), "heldout_loss": cell.get("eval_loss_final"), "peak_vram_gb": cell.get("peak_vram_gb"),
           "power_samples": power_samples(os.path.join(work, "loads", f"power_{idx}.txt"))}
    if fw == "e4b":
        row["absmax"] = {"tc1_modules": cell.get("absmax_dq_modules"), "compressed_now": facts.get("absmax_compressed"),
                         "fast_train": facts.get("fast_train_absmax_dq")}
    else:
        row["absmax"] = {"loaded_nested": (cell.get("unsloth_double_quant") or {}).get("loaded_nested")}
    return row


def normalize_op(work, sched):
    """One op process from its p238_op.py record (its premise table and adapter census come from the same load path)."""
    path = os.path.join(work, "ops", f"op_{sched['index']}_{sched['framework']}.json")
    rec = _json(path)
    facts = (rec or {}).get("facts") or {}
    return {"record": rec, "framework": sched["framework"], "draw": sched["draw"], "index": sched["index"],
            "premise_table": _int_keys(facts.get("premise_table")), "adapters": facts.get("adapters")}


def reduce_work(work):
    """The box: the schedule it ran, every row and op row, the readings and the verdict (reduce's: READ only when every op row
    READs and every step row READs or is a reproduced OOM; each fault is reported on its row, a VOID row never dropped)."""
    sys.path.insert(0, HERE)
    import p238_schedule
    sched_txt = os.path.join(work, "schedule.txt")
    want = None
    if os.path.isfile(sched_txt):
        want = {int(line.split()[0]) for line in open(sched_txt, encoding="utf-8") if line.strip()}
    sched = [r for r in p238_schedule.schedule() if want is None or r["index"] in want]
    box = _json(os.path.join(work, "box.json")) or {}
    proof = release_proof_ok(os.path.join(work, "release_proof.json"))
    env_box = env_file(os.path.join(work, "env_box.txt"))
    rcs = proc_rcs(os.path.join(work, "summary.txt"))
    rows = [normalize_load(work, r, box, proof, env_box, rcs.get(r["index"])) for r in sched if r["kind"] == "step"]
    ops = [normalize_op(work, r) for r in sched if r["kind"] == "op"]
    out = reduce(rows, ops)
    out["image_env"] = env_file(os.path.join(work, "env_image.txt"))      # information only: never a fault
    return out


# -- the self-test ---------------------------------------------------------------------------------------------------------------
def _good_rows():
    table = {L: {p: {f: f"{L}{p}{f}" for f in ("packed", "absmax_q", "absmax_s", "absmax_off", "absmax_code", "code")}
                 for p in ("gate_up", "down")} for L in range(N_LAYERS)}
    census = {f"{L}:{p}": {"trainable": 1, "A_sha": f"a{L}{p}"} for L in range(N_LAYERS) for p in ("gate_up_proj", "down_proj")}
    base = {"status": "ok", "dry": False, "versions": {**PINS, "torch": "2.12.1+cu130", "release_proof_ok": True},
            "env_box": {k: "0" * 40 if v is None else v for k, v in ENV_BOX.items()},
            "card": {"name": "NVIDIA GeForce RTX 5090", "cap": [12, 0], "sm_count": 170},
            "clock": {"busy_median_mhz": 2800, "hard_reason_any": False, "swpowercap_clear_frac": 1.0, "busy_samples": 200},
            "premise_table": table, "adapters": census, "point": "2k", "tokens_per_step": [2048] * STEPS,
            "step_ms": [1000.0] * STEPS, "micro_batches_per_step": 1, "profile_warm": PROFILED[0],
            "profile_steps": PROFILED[1] - PROFILED[0],
            "profile": {"summary": {"device_ms_method": DEVICE_MS_METHOD, "device_ms": 900.0, "profiled_steps": 3},
                        "top_device": [{"name": FUSED_KERNEL, "count": E_F_PROFILED_FORWARDS}], "top_cpu": []},
            "step0_loss": 2.0, "heldout_loss": 1.5}
    e_edges = [{"dgrad": {"kernel": 2 * N_LAYERS * i}, "route": {"fwd": 0, "dgrad": 0, "dense_fwd": 0, "dense_dgrad": 0,
                                                                 "decoded_fwd": 0, "decoded_dgrad": 0},
                "route_log": {"256:fused": 2 * 2 * N_LAYERS * i}} for i in range(STEPS + 1)]   # 2 passes x 2 projections
    e = dict(base, framework="e4b", arm="E_F",
             absmax={"tc1_modules": N_LAYERS, "compressed_now": N_LAYERS, "fast_train": {"required": True, "kept_fp32": None}},
             hook={"facts": {"fast_train_patched": N_LAYERS, "fast_train_skipped": {}}, "edges": e_edges})
    return e, table, census


def selftest():
    import copy
    cases = 0
    e, table, census = _good_rows()
    assert row_faults(e, table, census) == [], row_faults(e, table, census)
    cases += 1
    removals = [
        ("versions", "the pins"), ("card", "the card"), ("clock", "the clock"), ("premise_table", "the premise"),
        ("absmax", "the absmax"), ("adapters", "the census"), ("hook", "the engagement"), ("profile", "the device marker"),
        ("tokens_per_step", "the steps"), ("env_box", "the box env"),
    ]
    for key, what in removals:                       # RULE 7: every piece of evidence removed -> a fault
        r = copy.deepcopy(e)
        r.pop(key)
        assert row_faults(r, table, census), f"{what} removed still reads clean"
        cases += 1
    for mutate, label in [
        (lambda r: r["absmax"]["fast_train"].update(kept_fp32="off"), "an fp32 e4b absmax"),
        (lambda r: r["absmax"].update(tc1_modules=0), "--absmax-dq compressed nothing"),
        (lambda r: r["hook"]["edges"][-1]["route_log"].update({"8:dense": 3}), "a dense route at fused"),
        (lambda r: r["hook"]["edges"][-1]["dgrad"].update(kernel=1), "a short dgrad count"),
        (lambda r: r["hook"]["edges"].pop(), "a missing closing edge"),
        (lambda r: r["premise_table"][3]["down"].update(packed="x"), "one differing premise field"),
        (lambda r: r["adapters"]["7:down_proj"].update(A_sha="x"), "one differing A"),
        (lambda r: r["tokens_per_step"].__setitem__(5, 2047), "a short step"),
        (lambda r: r["clock"].update(busy_median_mhz=2400), "a capped card"),
        (lambda r: r["env_box"].update(E4B_TRAIN_DETERMINISTIC="1"), "an ambient e4b knob in env_box"),
        (lambda r: r["env_box"].pop("PYTORCH_CUDA_ALLOC_CONF"), "env_box without the allocator conf"),
        (lambda r: r["env_box"].update(UNSLOTH_MOE_NF4_AUTOTUNE="1"), "env_box with another Unsloth pin value"),
        (lambda r: r["env_box"].update(E4B_SHA="main"), "env_box with a branch, not a commit"),
        (lambda r: r.update(dry=True), "a dry record"),
        (lambda r: r["versions"].update(peft="0.22.0"), "another PEFT"),
        (lambda r: r["profile"]["top_device"][0].update(count=E_F_PROFILED_FORWARDS - 1), "one profiled fused forward short"),
        (lambda r: r["profile"].update(top_device=[]), "no profiled kernels"),
        (lambda r: r["profile"]["summary"].update(device_ms_method="legacy"), "another device-time method"),
        (lambda r: r["hook"]["edges"][PROFILED[1]]["route_log"].update({"256:fused": 0}), "a window whose routes disagree"),
    ]:
        r = copy.deepcopy(e)
        mutate(r)
        assert row_faults(r, table, census), f"{label} still reads clean"
        cases += 1
    # E_A: the route log must agree with route_for per n_groups
    a = copy.deepcopy(e)
    a["arm"] = "E_A"
    a["hook"]["edges"][-1]["route_log"] = {"256:fused": 2 * N_LAYERS * STEPS - 4, "8:dense": 4}
    a["hook"]["edges"][-1]["route"]["dense_fwd"] = 4
    a["hook"]["edges"][-1]["route"]["dense_dgrad"] = 2
    a["hook"]["edges"][-1]["dgrad"]["kernel"] = 2 * N_LAYERS * STEPS - 2
    assert row_faults(a, table, census) == [], row_faults(a, table, census)
    a["hook"]["edges"][-1]["route_log"]["20:dense"] = 1        # 20 groups took dense: route_for says fused
    assert any("route_for says fused" in w for w in row_faults(a, table, census))
    a["hook"]["edges"][-1]["route_log"]["x:fused:y"] = 1        # an unreadable key VOIDs, never a crash
    assert any("unreadable route-log key" in w for w in row_faults(a, table, census))
    cases += 3
    # U0: F15 v2 (TC1 amendment 80's merged bars) on the cell and its profile
    E = 2 * N_LAYERS
    cell = {"n_layers": N_LAYERS, "accum": 1, "moe_backend_selected": "grouped_mm", "env": {"unsloth_zoo": "2026.10.8"},
            "unsloth_backend_calls_per_step_min": {"unsloth_grouped_mm": E}, "unsloth_manual_grouped_mm_calls_per_step_max": 0}
    prof = {"summary": {"profiled_steps": 3}, "top_cpu": [],
            "top_device": [{"name": f15v2.TRITON_GEMM, "count": 9 * E * 3}, {"name": f15v2.TRITON_WGRAD, "count": 2 * E * 3}]}
    assert f15v2.engagement(cell, prof)[0] == "ENGAGED"
    prof_short = copy.deepcopy(prof)
    prof_short["top_device"][0]["count"] -= 1
    assert f15v2.engagement(cell, prof_short)[0] == "VOID"
    cases += 2
    # Unsloth: no snapshot, no structure, a faulted structure, and a wrong loader
    u = dict(copy.deepcopy(e), framework="unsloth", arm="U1", absmax={"loaded_nested": True}, loader_used="FastModel",
             hook={"edges": [{}] * (STEPS + 1)})
    assert any("no counter snapshot" in w for w in row_faults(u, table, census))
    u["hook"]["edges"] = [{"unsloth": {"counts": {}}}] * (STEPS + 1)
    assert any("no structural count" in w for w in row_faults(u, table, census))
    u["hook"]["structural"] = {"steps": STEPS - 1, "faults": [], "expected": {}}
    assert any("no structural count" in w for w in row_faults(u, table, census))
    u["hook"]["structural"] = {"steps": STEPS, "faults": ["step 3: expert module 7's 2 training passes decided differently"],
                               "expected": {"dx_nf4": 0, "gx": 0, "wgrad": 0}}
    assert any("structure: step 3" in w for w in row_faults(u, table, census))
    cases += 3
    u["loader_used"] = "FastLanguageModel"
    assert any("not FastModel" in w for w in row_faults(u, table, census))
    u["absmax"]["loaded_nested"] = False
    assert any("nested = False" in w for w in row_faults(u, table, census))
    cases += 3
    # quality and stability
    b = dict(e, arm="U1", step0_loss=2.005, heldout_loss=1.53)
    assert quality(e, b)[0] is True and quality(e, dict(b, step0_loss=2.02))[0] is False
    assert quality(e, dict(e, arm="E_A", step0_loss=2.002))[0] is False
    assert quality(e, {"arm": "U1"})[0] is False
    assert stable(e, e)[0] is True and stable(e, dict(e, step_ms=[1100.0] * STEPS))[0] is False
    assert stable(e, dict(e, profile={}))[0] is False
    assert stable(e, dict(e, profile={"summary": {"device_ms": 920.0}}))[0] is True
    cases += 6
    # op records: a clean one reads, and every rule removes it
    good_op = {"p238_op": 1, "dry": False, "capture": "eager", "framework": "e4b", "arms": ["E_F", "E_A"], "cells": [
        {"rows": r, "routing": k, "b_restored_zero": True,
         "parity": {a_: {"errs": {"out": 1e-3, "dx": 1e-3}, "ok": True} for a_ in ("E_F", "E_A")},
         "visits": [{"device_ms_per_call": 1.0, "wall_ms_per_call": 1.1}] * (OP_REPS * 2 * 2)}
        for r in OP_ROWS for k in OP_ROUTINGS]}
    assert op_faults(good_op) == [], op_faults(good_op)
    for mutate, label in [
        (lambda o: o.update(dry=True), "a dry op record"),
        (lambda o: o.update(capture="graph"), "another capture mode"),
        (lambda o: o["cells"].pop(), "a missing cell"),
        (lambda o: o["cells"][3]["parity"]["E_A"]["errs"].update(dx=0.5), "a parity miss (the layout STOP)"),
        (lambda o: o["cells"][3]["parity"].pop("E_F"), "an arm with no parity record"),
        (lambda o: o["cells"][0].update(b_restored_zero=False), "B not restored"),
        (lambda o: o["cells"][1]["visits"].pop(), "a visit short"),
        (lambda o: o["cells"][2].update(visits=[{"device_ms_per_call": 0.0, "wall_ms_per_call": 1.0}] * (OP_REPS * 4)),
         "a zero timing sample"),
        (lambda o: o.update(framework="unsloth"), "arms that are not the framework's"),
    ]:
        o = copy.deepcopy(good_op)
        mutate(o)
        assert op_faults(o), f"{label} still reads clean"
        cases += 1
    assert op_faults(None) and op_faults({})
    # the reduce wiring: an op row is the reference; a step row equal to it reads READ
    op = {"record": good_op, "premise_table": table, "adapters": census, "framework": "e4b", "draw": 1}
    out = reduce([e], [op])
    assert out["rows"][0]["verdict"] == "READ" and out["op_rows"][0]["verdict"] == "READ", out
    assert reduce([e], [])["rows"][0]["verdict"] == "VOID"      # no op row -> no reference -> fails closed
    assert reduce([e], [dict(op, record=None)])["op_rows"][0]["verdict"] == "VOID"
    cases += 4
    return cases


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--work", help="a box's run directory (/root/p238 on the box, or a fetched copy)")
    ap.add_argument("--out", help="where the verdict JSON goes")
    a = ap.parse_args(argv)
    if a.selftest:
        print(f"P238 REDUCE SELFTEST OK cases={selftest()}")
        return 0
    if not a.work or not a.out:
        ap.error("--work and --out, or --selftest")
    out = reduce_work(a.work)
    data = (json.dumps(out, indent=1, sort_keys=True, default=str) + "\n").encode()
    with open(a.out, "wb") as f:
        f.write(data)
    print(f"P238 {out['verdict']}: {out['n_rows']} step rows, {out['n_op_rows']} op rows, {out['n_void']} VOID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
