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
DEVICE_MS_METHOD = "kernels_only_at_source"         # amendment 81's marker (f15v2.py's docstring names it; the kit test pins it)
FUSED_KERNEL = "_gemm_nf4_grouped"                   # grouped-nf4-gemm's fused grouped forward kernel
PROFILED = (20, 23)                                  # --profile-warm 20 --profile-steps 3: loop steps 20..22, edges 20 and 23
E_F_PROFILED_FORWARDS = 2 * 2 * N_LAYERS * 3         # 2 passes (HF non-reentrant recompute) x 2 projections x 40 x 3 = 480
OP_ROWS, OP_ROUTINGS, OP_REPS, OP_TOL = (32, 64, 128, 256, 512), ("uniform", "skew"), 20, 2e-2
OP_ARMS = {"e4b": ("E_F", "E_A"), "unsloth": ("U0", "U1", "UA")}
WITHIN_STEP0, WITHIN_HELD = 1e-3, 0.01
ACROSS_STEP0, ACROSS_HELD = 0.01, 0.05               # TC1-PREREG.md:63 (step 0) and TC1's COMPARABLE (held-out)
STABLE = {"e4b": 0.05, "unsloth": 0.10}


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
    why += check_pins(row) + check_card(row) + check_premise(row, ref_table) + check_absmax(row) + check_census(row, ref_census)
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


def reduce(rows, op_rows):
    """The verdict per row, the reference tables, and the pair flags. The reference premise table and census are the first OP
    row's (an op process loads exactly as a step load does); every other table must equal them."""
    ref = (op_rows or [{}])[0]
    ref_table, ref_census = ref.get("premise_table"), ref.get("adapters")
    out = {"rows": [], "op_rows": []}
    for o in op_rows or []:
        why = op_faults(o.get("record")) + check_premise(o, ref_table) + check_census(o, ref_census)
        out["op_rows"].append({"framework": o.get("framework"), "draw": o.get("draw"), "verdict": "VOID" if why else "READ", "why": why})
    for r in rows:
        why = row_faults(r, ref_table, ref_census)
        out["rows"].append({k: r.get(k) for k in ("index", "draw", "point", "arm")} | {"verdict": "VOID" if why else "READ", "why": why,
                                                                                       "s_per_step": s_per_step(r)})
    return out


# -- the self-test ---------------------------------------------------------------------------------------------------------------
def _good_rows():
    table = {L: {p: {f: f"{L}{p}{f}" for f in ("packed", "absmax_q", "absmax_s", "absmax_off", "absmax_code", "code")}
                 for p in ("gate_up", "down")} for L in range(N_LAYERS)}
    census = {f"{L}:{p}": {"trainable": 1, "A_sha": f"a{L}{p}"} for L in range(N_LAYERS) for p in ("gate_up_proj", "down_proj")}
    base = {"status": "ok", "dry": False, "versions": {**PINS, "torch": "2.12.1+cu130", "release_proof_ok": True},
            "card": {"name": "NVIDIA GeForce RTX 5090", "cap": [12, 0], "sm_count": 170},
            "clock": {"busy_median_mhz": 2800, "hard_reason_any": False, "swpowercap_clear_frac": 1.0},
            "premise_table": table, "adapters": census, "point": "2k", "tokens_per_step": [2048] * STEPS,
            "step_ms": [1000.0] * STEPS, "micro_batches_per_step": 1,
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
        ("tokens_per_step", "the steps"),
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
    a = ap.parse_args(argv)
    if a.selftest:
        print(f"P238 REDUCE SELFTEST OK cases={selftest()}")
        return 0
    ap.error("the receipt adapter lands with the tc1_arm hook PR's receipt layout; only --selftest runs today")


if __name__ == "__main__":
    sys.exit(main())
