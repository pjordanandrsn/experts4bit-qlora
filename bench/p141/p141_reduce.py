#!/usr/bin/env python3
"""p141_reduce.py -- lane P141's registered rule (bench/p141/PREREG-p141.md), derived from bench/p134/p134_reduce.py by
named substitutions: one family, Mixtral-8x7B (``--family mixtral``; the dry run's ``tiny``); no proof mode; every record
at P136's server size (16 slots x 768 tokens); and phase 1's selection (``select.json``, ``p141_select.py``) read and
checked: its own rule re-derived from its own record, and B1's ``GNF4_GEMV_BW_PLAN`` exactly its plan string. The
rungs, bars, integrity checks and the dry mode are P134's.

Records (``p141_box.py``), family F: ``select.json`` (phase 1); ``fit_F.json``; ``arm_{B0a,B1a,B1b,B0b}_F.json`` (speed,
W1 and the control) and ``quality_off_F.json`` / ``quality_on_F.json`` (P110's teacher-forced instrument at one window
per pass, T == 1). B0 = the shipped default (``GNF4_GEMV_BW`` unset: the scalar GEMV at Mixtral's shapes); B1 =
``GNF4_GEMV_BW=1`` at phase 1's plans.

The rule, first rung that applies:
  FUNCTION_FAIL  phase 1 found no plan meeting K33's numerics contract at one of the two shapes (the box stops before
                 the fetch; no arm runs).
  VOID           the selection record is missing, its choice is not its own rule's (the lowest median among its
                 contract-exact plans), or a selected plan misses the contract; the fit check failed (no rerun); a
                 record is missing or not ok; another e4b / grouped-nf4-gemm commit, model revision or server size; a
                 B1 record whose plan string is not the selection's; a bucket not captured (a reading); the arms differ
                 in prompts, lengths, fusion census, width or buckets; a void slope; ENGAGEMENT off: B1 must dispatch
                 every single-row decode GEMV to ``bw_prmt32`` (``bw_splitk`` beside it only under a selected split-K
                 plan) and B0 none to ``bw_*`` and some to the scalar GEMV; the quality integrity fails; or the scale
                 mutant passes the quality bar.
  NOISY          a self-pair (B0b/B0a, B1b/B1a) outside [0.96, 1.04] at the control, or outside [0.985, 1.015] at W1.
  FUNCTION_FAIL  B0b != B0a or B1b != B1a on any row, workload or length; timed reps of one arm digest differently.
  QUALITY_FAIL   on either text bias_ON > B_floor + 0.01 or spread_ON > 2 * max(S_floor, 0.005); on wikitext
                 |K8 delta ppl| > 0.05.
  SLOWER         g1 = min(B1a/B0a, B1b/B0b) at W1 < 1.03, or gW (the same at the control, the W16 no-regression check)
                 < 0.99.
  LICENSED       otherwise: Mixtral's two (N, K) shapes at the selected plans, on parts with >= 160 SMs.

    python p141_reduce.py --dir RUN_DIR --family mixtral --out verdict_mixtral.json [--e4b-sha SHA] [--dry]
    python p141_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p115_reduce as pb  # noqa: E402  (Phase B's statistics, staged at its registered bytes)

TAGS = ("B0a", "B1a", "B1b", "B0b")
SELF_LO, SELF_HI = 0.96, 1.04
SELF_W1_LO, SELF_W1_HI = 0.985, 1.015   # W1's self-pairs: half the 1.03 gain bar (P111 read 0.997-1.002); added in review
GAIN_MIN_W1, GAIN_MIN_WCTL = 1.03, 0.99
TOL, SPREAD_X, SPREAD_MIN, K8_BUDGET, K8_GATED = pb.TOL, pb.SPREAD_X, pb.SPREAD_MIN, pb.K8_BUDGET, pb.K8_GATED
TEXTS = pb.TEXTS
GNF4_SHA = "70bc03cb7a61b03c4e801e7423c58c103a1b6d7d"     # grouped-nf4-gemm 0.46.1, exact (K33 as 0.46.0 shipped it)
MIXTRAL, TINY = "mistralai/Mixtral-8x7B-Instruct-v0.1", "p133-tiny"
REVS = {MIXTRAL: "eba92302a2861cdc0098cc54bc9f17cb2c47eb61", TINY: ""}      # FAM Amendment 5's pin (P135's)
FAMILIES = {"mixtral": MIXTRAL, "tiny": TINY}                                   # the dry run's family: P133's fixture
LAYERS = {MIXTRAL: 32, TINY: 2}
WINDOWS = {"reading": 24, "dry": 4}    # per text
INCUMBENT = {MIXTRAL: "scalar", TINY: "scalar"}   # Mixtral's shapes are in neither _BW_SHAPES nor dot-pad's table
SERVER_SIZE = (16, 768)                # P136's Mixtral server: E4B_PAGED_MAX_SEQS=16, E4B_PAGED_MAX_TOKENS_PER_SEQ=768
SELECT_PROJ = ("gate_up", "down")
OFF_ARMS = ("R", "rep", "chunk", "mutant_scale")
def _instrument_buckets() -> tuple:
    """The buckets P110's instrument enables at capture=False in every quality pass: ``p110_box.BUCKETS``, read from that
    file's own source (staged beside this reducer on the box; ``bench/p110`` in the repository) without importing it (it
    imports torch), so the rule cannot drift from the instrument. Amendment 1: the pass integrity reads these, as P116's
    reducer did, not the server's buckets from the record, which follow its own max_seqs."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, "p110_box.py"), os.path.join(here, "..", "p110", "p110_box.py")):
        if os.path.exists(path):
            for node in ast.parse(open(path, encoding="utf-8").read()).body:
                if isinstance(node, ast.Assign) and any(getattr(x, "id", None) == "BUCKETS" for x in node.targets):
                    return tuple(ast.literal_eval(node.value))
    raise SystemExit("REFUSED: p110_box.py (P110's instrument, where its BUCKETS live) is not beside the reducer")


INSTRUMENT_BUCKETS = _instrument_buckets()
FLOORS = ("chunk",)
GEMV = ("dotpad", "dotpad_splitk", "scalar", "scalar_splitk")
BW = ("bw_tree", "bw_prmt32", "bw_splitk")


def _rate(r, w):
    return (r.get("workloads", {}).get(w) or {}).get("decode_tok_s")


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def engagement_faults(name, arm, model, d, splitk_ok=False) -> list:
    """The dispatch rule on a tally delta ``d``: B1 all ``bw_prmt32`` (and ``bw_splitk`` beside it only when a selected
    plan splits K); B0 none to ``bw_*``, some to the incumbent."""
    d = d or {}
    inc = INCUMBENT.get(model, "dotpad")
    other = [k for k in GEMV if not k.startswith(inc)]
    if arm == "B1":
        banned = ("bw_tree",) + (() if splitk_ok else ("bw_splitk",)) + GEMV
        if d.get("bw_prmt32", 0) <= 0 or any(d.get(k, 0) for k in banned):
            return [f"{name}: B1 dispatched {d}, registered bw_prmt32{' (+ bw_splitk)' if splitk_ok else ''} only"]
    elif any(d.get(k, 0) for k in BW) or d.get(inc, 0) + d.get(inc + "_splitk", 0) <= 0 \
            or any(d.get(k, 0) for k in other):
        return [f"{name}: B0 dispatched {d}, registered {inc} only"]
    return []


def _model_of(family: str, dry: bool) -> str:
    if family not in FAMILIES or (FAMILIES[family] == TINY) != dry:
        raise ValueError(f"family {family!r} with dry={dry}")
    return FAMILIES[family]


def _mode(dry: bool) -> str:
    return "dry" if dry else "reading"


def _model_name(rec) -> str:
    """The record's model, with a dry run's local fixture path read as :data:`TINY`."""
    m = rec.get("model") or ""
    return TINY if rec.get("dry") and m.rstrip("/").endswith(TINY) else m


def selection_rule(sel: dict) -> dict:
    """Phase 1's choice re-derived from its own record: per projection, the lowest median among the plans whose
    numerics met the contract, ties to the earlier plan in its plan list."""
    plans = [",".join(map(str, p)) for p in sel.get("plans") or []]
    out = {}
    for proj in SELECT_PROJ:
        ms = (sel.get("plan_ms") or {}).get(proj) or {}
        ok = [p for p in plans if ((sel.get("numerics") or {}).get(proj) or {}).get(p, {}).get("contract") and p in ms]
        if ok:
            out[proj] = min(ok, key=lambda p: (ms[p], plans.index(p)))
    return out


def splitk_selected(sel: dict) -> bool:
    return any(int(p[3]) > 1 for p in (sel.get("selected") or {}).values())


def selection_faults(sel, family: str, dry: bool) -> list:
    """The selection record's integrity: present, the family's, its choice its own rule's, its plan string its choice."""
    if not sel:
        return ["phase 1: no select.json"]
    void = []
    if sel.get("family") != ("tiny" if dry else "mixtral") or (family == "mixtral") == dry:
        void.append(f"phase 1 ran family {sel.get('family')!r}, the reduction is {family!r} (dry={dry})")
    want = selection_rule(sel)
    got = {p: ",".join(map(str, v)) for p, v in (sel.get("selected") or {}).items()}
    if got != want:
        void.append(f"phase 1 selected {got}, its own rule chooses {want}")
    spec = sel.get("spec") or {}
    try:
        env = ";".join(f"{spec[p][0]},{spec[p][1]}=" + got[p] for p in SELECT_PROJ)
    except (KeyError, IndexError, TypeError):
        env = None
    if sel.get("env") != env:
        void.append(f"phase 1's plan string {sel.get('env')!r} is not its selection ({env!r})")
    return void


def speed_faults(arms: dict, e4b_sha: str, family: str, dry: bool = False, sel: dict | None = None) -> list:
    void = []
    want_model = _model_of(family, dry)
    sel = sel or {}
    for t, r in arms.items():
        if r.get("selection_env") != sel.get("env"):
            void.append(f"{t}: ran under selection {r.get('selection_env')!r}, phase 1 chose {sel.get('env')!r}")
        if t.startswith("B1") and (r.get("knobs") or {}).get("GNF4_GEMV_BW_PLAN") != sel.get("env"):
            void.append(f"{t}: GNF4_GEMV_BW_PLAN {(r.get('knobs') or {}).get('GNF4_GEMV_BW_PLAN')!r} is not phase 1's")
        if not dry and (r.get("max_seqs"), r.get("max_tokens_per_seq")) != SERVER_SIZE:
            void.append(f"{t}: server {r.get('max_seqs')} x {r.get('max_tokens_per_seq')}, registered {SERVER_SIZE}")
        if r.get("e4b_sha") != e4b_sha:
            void.append(f"{t}: e4b {r.get('e4b_sha')} != {e4b_sha}")
        if r.get("gnf4_sha") != GNF4_SHA:
            void.append(f"{t}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
        if bool(r.get("dry")) != dry:
            void.append(f"{t}: dry={r.get('dry')}, the reduction is dry={dry}")
        if _model_name(r) != want_model or (REVS[want_model] or "") != (r.get("revision") or ""):
            void.append(f"{t}: model {r.get('model')}@{r.get('revision')} is not {want_model}@{REVS[want_model][:8]}")
            continue
        if r.get("arm") != t[:2]:
            void.append(f"{t}: the record says arm {r.get('arm')}")
        g, bk = r.get("graph_status") or {}, r.get("buckets") or []
        if not dry and (not bk or [g.get(str(b)) for b in bk] != ["graph"] * len(bk)):
            void.append(f"{t}: graph_status {g} over buckets {bk}")
        void += engagement_faults(t, t[:2], want_model, r.get("dispatch_total"), splitk_selected(sel))
    if len({json.dumps(r.get("fusions"), sort_keys=True) for r in arms.values()}) != 1:
        void.append("the arms report different fusion censuses (only the decode GEMV switch may differ)")
    if len({r.get("prompts_sha256") for r in arms.values()}) != 1:
        void.append("the arms read different prompts")
    if len({(r.get("short"), r.get("long"), r.get("reps")) for r in arms.values()}) != 1:
        void.append("the arms ran different lengths")
    if len({(r.get("control"), r.get("max_seqs"), tuple(r.get("buckets") or ())) for r in arms.values()}) != 1:
        void.append("the arms served different widths or buckets")
    if any(_rate(arms[t], w) is None for t in TAGS for w in _workloads(arms) if t in arms):
        void.append("a decode slope is void")
    return void


def _workloads(arms: dict) -> tuple:
    """W1, then the control when the arms ran one (their shared ``control``)."""
    c = next(iter(arms.values())).get("control") if arms else None
    return ("W1",) + ((c,) if c else ())


def _t1_pass_faults(e, C, L, dry=False) -> list:
    """One window, one pass: bucket 1 eager (capture=False), C - 1 decode steps, no padding, no replay. Device grouping
    is required except in a dry run (sm_86 groups on the host: its first execution is the paid reading)."""
    out = []
    if e.get("decode_calls") != (C - 1) * L:
        out.append(f"{e.get('decode_calls')} decode attention calls, expected {(C - 1) * L}")
    if not dry and not (e.get("grouping_flags_in_pass") or {}).get("device_grouping"):
        out.append("device grouping off")
    st = e.get("graph_status") or {}
    if [st.get(str(b)) for b in INSTRUMENT_BUCKETS] != ["eager: capture=False"] * len(INSTRUMENT_BUCKETS):
        out.append(f"graph_status {st}")
    gs = e.get("graph_stats") or {}
    got = gs.get("1", {})
    if (got.get("eager_steps"), got.get("pad_rows")) != (C - 1, 0):
        out.append(f"bucket 1 stats {got}, expected {C - 1} eager steps and no padding")
    if any(v.get("replays", 0) for v in gs.values()) or any(v.get("eager_steps", 0) for b, v in gs.items() if b != "1"):
        out.append(f"another bucket or a replay in a one-window pass: {gs}")
    return out


def quality_faults(off: dict, on: dict, e4b_sha: str, family: str, dry: bool = False, sel: dict | None = None) -> list:
    void = []
    want_model = _model_of(family, dry)
    for name, rec in (("quality_off", off), ("quality_on", on)):
        if rec.get("status") != "ok":
            void.append(f"{name}: status {rec.get('status')}")
        if rec.get("e4b_sha") != e4b_sha:
            void.append(f"{name}: e4b {rec.get('e4b_sha')} != {e4b_sha}")
        if rec.get("gnf4_sha") != GNF4_SHA:
            void.append(f"{name}: grouped-nf4-gemm {rec.get('gnf4_sha')} != {GNF4_SHA}")
        if _model_name(rec) != want_model or (REVS[want_model] or "") != (rec.get("revision") or ""):
            void.append(f"{name}: model {rec.get('model')}@{rec.get('revision')} is not {want_model}")
        if rec.get("group") != 1:
            void.append(f"{name}: group {rec.get('group')}, registered 1 (T == 1)")
        if bool(rec.get("dry")) != dry:
            void.append(f"{name}: dry={rec.get('dry')}, the reduction is dry={dry}")
        if not dry and (rec.get("max_seqs"), rec.get("max_tokens_per_seq")) != SERVER_SIZE:
            void.append(f"{name}: server {rec.get('max_seqs')} x {rec.get('max_tokens_per_seq')}, registered {SERVER_SIZE}")
        if rec.get("selection_env") != (sel or {}).get("env"):
            void.append(f"{name}: ran under selection {rec.get('selection_env')!r}, phase 1 chose {(sel or {}).get('env')!r}")
    if (on.get("knobs") or {}).get("GNF4_GEMV_BW_PLAN") != (sel or {}).get("env"):
        void.append(f"quality_on: GNF4_GEMV_BW_PLAN {(on.get('knobs') or {}).get('GNF4_GEMV_BW_PLAN')!r} is not phase 1's")
    if void:
        return void
    L, want_n, C = LAYERS[want_model], WINDOWS[_mode(dry)], off["cont"]
    for k in ("cont", "prompt", "chunk", "floor_chunk", "layers", "windows_sha256", "fusions", "buckets"):
        if off.get(k) != on.get(k):
            void.append(f"the phases differ in {k}: {off.get(k)} / {on.get(k)}")
    if off.get("layers") != L:
        void.append(f"{off.get('layers')} layers, the family registers {L}")
    void += engagement_faults("quality_off", "B0", want_model, off.get("dispatch_measure"))
    void += engagement_faults("quality_on", "B1", want_model, on.get("dispatch_measure"), splitk_selected(sel or {}))
    for t in TEXTS:
        for name, rec, arms in (("quality_off", off, OFF_ARMS), ("quality_on", on, ("ON",))):
            per = (rec.get("per_window") or {}).get(t, {})
            eng = (rec.get("engagement") or {}).get(t, {})
            for arm in arms:
                want = 1 if arm == "rep" else want_n
                if len(per.get(arm, [])) != want:
                    void.append(f"{name} {t}: arm {arm} has {len(per.get(arm, []))} windows, expected {want}")
                passes = eng.get(arm, [])
                if len(passes) != want:
                    void.append(f"{name} {t} {arm}: {len(passes)} passes, expected {want}")
                    continue
                for k, e in enumerate(passes):
                    void += [f"{name} {t} {arm} pass {k}: {m}" for m in _t1_pass_faults(e, C, L, dry)]
        r_passes = (off.get("engagement") or {}).get(t, {}).get("R", [])
        on_passes = (on.get("engagement") or {}).get(t, {}).get("ON", [])
        for k, (a, b) in enumerate(zip(r_passes, on_passes)):
            if (a.get("kernels"), a.get("qkv_calls")) != (b.get("kernels"), b.get("qkv_calls")):
                void.append(f"{t} window {k}: glue kernels / q/k/v calls differ between R ({a.get('kernels')}, "
                            f"{a.get('qkv_calls')}) and ON ({b.get('kernels')}, {b.get('qkv_calls')})")
    return void


def quality_read(off: dict, on: dict) -> dict:
    out = {}
    for t in TEXTS:
        ref = {x["window"]: x["nll"] for x in off["per_window"][t]["R"]}
        st = {a: pb._stats(off["per_window"][t], a, ref) for a in OFF_ARMS if a != "R"}
        st["ON"] = pb._stats(on["per_window"][t], "ON", ref)
        draws = list(FLOORS) + ([] if (off.get("rep_identical") or {}).get(t) else ["rep"])
        fl = {"draws": draws, "B_floor": max(abs(st[f]["bias"]) for f in draws),
              "S_floor": max(st[f]["spread"] for f in draws)}
        ppl_r, ppl_on = pb._ppl(off["per_window"][t]["R"]), pb._ppl(on["per_window"][t]["ON"])
        out[t] = {"stats": st, "floor": fl, "bias_bar": fl["B_floor"] + TOL,
                  "spread_bar": SPREAD_X * max(fl["S_floor"], SPREAD_MIN),
                  "mutant_passes": pb._passes(st["mutant_scale"], fl), "on_passes": pb._passes(st["ON"], fl),
                  "k8": {"ppl_R": ppl_r, "ppl_ON": ppl_on, "delta": ppl_on - ppl_r, "gated": t in K8_GATED,
                         "passes": abs(ppl_on - ppl_r) <= K8_BUDGET}}
    return out


def fit_faults(fit: dict | None, dry: bool) -> list:
    """The registration's fit rule: W1 must build and decode on the default server before any arm; no rerun."""
    if not fit:
        return ["fit: no fit record (the fit check runs before any arm)"]
    if fit.get("status") != "ok":
        return [f"fit: W1 does not fit the default server ({fit.get('error')}); no arm ran and no rerun is registered"]
    if bool(fit.get("dry")) != dry:
        return [f"fit: dry={fit.get('dry')}, the reduction is dry={dry}"]
    return []


def reduce(arms: dict, off: dict | None, on: dict | None, e4b_sha: str, *, family: str, fit=None, sel=None,
           dry=False) -> dict:
    out = {"lane": "P141", "family": family, "model": _model_of(family, dry), "dry": dry,
           "verdict": None, "reasons": [],
           "bars": {"self_pair": [SELF_LO, SELF_HI], "self_pair_w1": [SELF_W1_LO, SELF_W1_HI], "gain_min_w1": GAIN_MIN_W1,
                    "gain_min_control": GAIN_MIN_WCTL, "tol": TOL, "spread_x": SPREAD_X, "spread_min": SPREAD_MIN,
                    "k8_budget": K8_BUDGET, "k8_gated": list(K8_GATED), "floors": list(FLOORS)}}
    if sel and sel.get("status") != "ok":                                # phase 1: K33 missed its contract at a shape
        out.update(verdict="FUNCTION_FAIL", rerun=False, selection=sel.get("status"),
                   reasons=[f"phase 1 {sel.get('status')}: no plan met K33's numerics contract there "
                            f"(refused: {json.dumps(sel.get('plan_refused'))[:600]})"])
        return out
    void = selection_faults(sel, family, dry)
    if void:
        out.update(verdict="VOID", reasons=void, rerun=True)
        return out
    out["selection"] = {"selected": sel.get("selected"), "env": sel.get("env"),
                        "reported_never_ruled": {"plan_ms": {p: (sel.get("plan_ms") or {}).get(p, {}).get(
                            ",".join(map(str, sel["selected"][p]))) for p in SELECT_PROJ},
                            "incumbent_ms": sel.get("incumbent_ms"), "floor_ms": sel.get("floor_ms")}}
    void = fit_faults(fit, dry)
    if void:
        out.update(verdict="VOID", reasons=void, fit=fit, rerun=False)   # the fit rule: no rerun
        return out
    out["rerun"] = True                                                  # any later VOID: the one registered rerun
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    missing += [n for n, r in (("quality_off", off), ("quality_on", on)) if not r]
    if missing:
        out.update(verdict="VOID", reasons=[f"record(s) missing or not ok: {missing}"])
        return out
    void = speed_faults(arms, e4b_sha, family, dry, sel) + quality_faults(off, on, e4b_sha, family, dry, sel)
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    q = quality_read(off, on)
    out["quality"] = q
    void = [f"{t}: mutant_scale passes the bar (the gate cannot fail)" for t in TEXTS if q[t]["mutant_passes"]]
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    B0a, B1a, B1b, B0b = (arms[t] for t in TAGS)
    short, long_ = B0a["short"], B0a["long"]
    wls = _workloads(arms)
    ctl = wls[1] if len(wls) > 1 else None
    out.update(control=ctl, max_seqs=B0a.get("max_seqs"), buckets=B0a.get("buckets"),
               box={k: B0a.get(k) for k in ("gpu", "capability", "sm_count", "graphs", "placement", "grouping")})
    rates = {t: {w: _rate(arms[t], w) for w in wls} for t in TAGS}
    out["decode_tok_s"] = rates
    out["ms_per_step"] = {t: {w: arms[t]["workloads"][w].get("decode_ms_per_step") for w in wls} for t in TAGS}
    selfp = {f"{a}/{b} {w}": rates[a][w] / rates[b][w] for a, b in (("B0b", "B0a"), ("B1b", "B1a")) for w in wls}
    out["self_pairs"] = {k: round(v, 4) for k, v in selfp.items()}
    pairs = {w: [rates["B1a"][w] / rates["B0a"][w], rates["B1b"][w] / rates["B0b"][w]] for w in wls}
    out["pair_ratios"] = {w: [round(x, 4) for x in v] for w, v in pairs.items()}
    out["g1"] = round(min(pairs["W1"]), 4)
    out["gW"] = round(min(pairs[ctl]), 4) if ctl else None
    if not ctl:
        out["control_note"] = f"max_seqs {B0a.get('max_seqs')} serves no width >= 2: the control is void; W1 reads alone"
    out["geomean"] = {w: round(math.sqrt(v[0] * v[1]), 4) for w, v in pairs.items()}
    out["b1_vs_b0_tokens"] = {f"{w}@{n}": pb._agree(_tok(B1a, w, n), _tok(B0a, w, n)) for w in wls
                              for n in (short, long_)}
    out["dispatch"] = {t: arms[t].get("dispatch_total") for t in TAGS}
    fn = []
    for a, b in (("B0b", "B0a"), ("B1b", "B1a")):
        for w in wls:
            for n in (short, long_):
                if _tok(arms[a], w, n) != _tok(arms[b], w, n):
                    rows = [i for i, (x, y) in enumerate(zip(_tok(arms[a], w, n), _tok(arms[b], w, n))) if x != y]
                    fn.append(f"{a} != {b} on {w} at {n} tokens, rows {rows}")
    for t in TAGS:
        for w in wls:
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fn.append(f"{t} {w} at {n} tokens: timed reps differ")
    band = lambda k: (SELF_W1_LO, SELF_W1_HI) if k.endswith(" W1") else (SELF_LO, SELF_HI)  # noqa: E731
    noisy = [f"{k} = {v:.4f} (band {band(k)})" for k, v in selfp.items() if not band(k)[0] <= v <= band(k)[1]]
    qfail = []
    for t in TEXTS:
        s = q[t]["stats"]["ON"]
        if not q[t]["on_passes"]:
            qfail.append(f"{t}: ON bias {s['bias']:+.5f} (bar {q[t]['bias_bar']:.5f}), spread {s['spread']:.5f} "
                         f"(bar {q[t]['spread_bar']:.5f})")
        if q[t]["k8"]["gated"] and not q[t]["k8"]["passes"]:
            qfail.append(f"{t}: K8 perplexity {q[t]['k8']['ppl_R']:.5f} -> {q[t]['k8']['ppl_ON']:.5f} (|delta| > {K8_BUDGET})")
    gw = f"g{ctl[1:]} = {out['gW']}" if ctl else "no control"
    if noisy:
        out.update(verdict="NOISY", reasons=noisy)
    elif fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif qfail:
        out.update(verdict="QUALITY_FAIL", reasons=qfail)
    elif out["g1"] < GAIN_MIN_W1 or (ctl and out["gW"] < GAIN_MIN_WCTL):
        out.update(verdict="SLOWER", reasons=[f"g1 = {out['g1']} (bar {GAIN_MIN_W1}), {gw} (bar {GAIN_MIN_WCTL})"])
    else:
        out.update(verdict="LICENSED", reasons=[
            f"g1 = {out['g1']}, {gw} (geomean {out['geomean']}); "
            + "; ".join(f"{t} ON bias {q[t]['stats']['ON']['bias']:+.5f} (bar {q[t]['bias_bar']:.5f}), K8 "
                        f"{q[t]['k8']['delta']:+.4f}" for t in TEXTS)])
    return out


# ------------------------------------------------------------------ self-test --
E = "a" * 40
BUCKETS16 = [1, 2, 4, 8, 16]
FIT = {"mode": "fit", "status": "ok", "dry": False}
PLAN_LIST = [[16, 256, 4, 1], [16, 1024, 4, 1], [16, 256, 4, 2]]


def _fake_select(*, family="mixtral", gu="16,1024,4,1", dn="16,256,4,1", status="ok", bad_contract=None, env=None,
                 selected=None):
    """A phase-1 record whose plan_ms put ``gu`` / ``dn`` lowest among contract-exact plans."""
    spec = {"mixtral": {"gate_up": [28672, 4096], "down": [4096, 14336]},
            "tiny": {"gate_up": [512, 512], "down": [512, 256]}}[family]
    keys = [",".join(map(str, q)) for q in PLAN_LIST]
    ms = {"gate_up": {k: (1.0 if k == gu else 2.0) for k in keys}, "down": {k: (1.0 if k == dn else 2.0) for k in keys}}
    num = {pj: {k: {"contract": not (bad_contract and bad_contract == (pj, k))} for k in keys} for pj in SELECT_PROJ}
    sel = selected or {"gate_up": [int(x) for x in gu.split(",")], "down": [int(x) for x in dn.split(",")]}
    rec = {"family": family, "spec": spec, "plans": PLAN_LIST, "plan_ms": ms, "numerics": num, "status": status,
           "selected": sel, "plan_refused": {}, "incumbent_ms": {"gate_up": 9.0, "down": 5.0},
           "floor_ms": {"gate_up": 2.6, "down": 1.3}}
    if status == "ok":
        rec["env"] = env if env is not None else ";".join(
            f"{spec[pj][0]},{spec[pj][1]}=" + ",".join(map(str, sel[pj])) for pj in SELECT_PROJ)
    return rec


SEL = _fake_select()


def _fake_arm(tag, rate_c, rate1, toks, *, model=MIXTRAL, e4b=E, graphs=None, dispatch=None, rep_ok=True, width=16,
              dry=False, buckets=None, ms1=None, sel=SEL, plan=None, size=SERVER_SIZE):
    arm = tag[:2]
    inc = INCUMBENT[model]
    L = LAYERS[model]
    bk = buckets if buckets is not None else BUCKETS16
    d = dispatch or ({**{k: 0 for k in GEMV + BW}, "bw_prmt32": 2 * L} if arm == "B1"
                     else {**{k: 0 for k in GEMV + BW}, inc: 2 * L})
    w = {}
    for name, b, rate in (((f"W{width}", width, rate_c),) if width >= 2 else ()) + (("W1", 1, rate1),):
        w[name] = {"batch": b, "decode_tok_s": rate,
                   "decode_ms_per_step": ms1 if (ms1 and b == 1) else 1000.0 / rate * b,
                   "tokens": {"32": toks[:b], "160": toks[:b]},
                   "rep_digests": {"32": ["x", "x", "x" if rep_ok else "y"], "160": ["z"] * 3}}
    knobs = {"GNF4_GEMV_BW": "1", "GNF4_GEMV_BW_PLAN": plan if plan is not None else sel.get("env"),
             "GNF4_GEMV_BW_DECODE": None} if arm == "B1" else {k: None for k in ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN",
                                                                                 "GNF4_GEMV_BW_DECODE")}
    return {"arm": arm, "tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "dry": dry,
            "model": ("/root/p141/fam/tiny/" + TINY) if model == TINY else model, "revision": REVS[model],
            "graph_status": graphs or {str(b): "graph" for b in bk}, "buckets": bk, "max_seqs": size[0],
            "max_tokens_per_seq": size[1], "control": f"W{width}" if width >= 2 else None,
            "knobs": knobs, "selection_env": sel.get("env"),
            "fusions": {"fuse_qkv_n": 0, "fuse_t1_glue_n": 65, "fuse_t1_glue_r2_n": [32, 32],
                        "fuse_router_epilogue_n": 32},
            "prompts_sha256": "p", "short": 32, "long": 160, "reps": 3, "workloads": w, "dispatch_total": d}


def _fake_pass(C, L, buckets, device_grouping=True):
    gs = {str(b): {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in buckets}
    gs["1"]["eager_steps"] = C - 1
    return {"decode_calls": (C - 1) * L, "grouping_flags_in_pass": {"device_grouping": device_grouping},
            "graph_status": {str(b): "eager: capture=False" for b in buckets}, "graph_stats": gs,
            "kernels": {k: 0 for k in pb.KERNELS}, "forwards": C, "qkv_calls": 0}


def _fake_quality(phase, *, model=MIXTRAL, mode="reading", bias=None, mutant=0.4, chunk=-0.0004, rep_identical=True,
                  dispatch=None, n=None, group=1, dry=False, device_grouping=True, buckets=None, sel=SEL, plan=None,
                  size=SERVER_SIZE):
    L, C = LAYERS[model], (128 if mode == "reading" else 32)
    n = n if n is not None else WINDOWS[mode]
    bk = buckets if buckets is not None else BUCKETS16
    bias = bias or {}
    arms = OFF_ARMS if phase == "off" else ("ON",)
    inc = INCUMBENT[model]
    d = dispatch or ({**{k: 0 for k in GEMV + BW}, inc: 2 * L * n} if phase == "off"
                     else {**{k: 0 for k in GEMV + BW}, "bw_prmt32": 2 * L * n})
    per, eng = {}, {}
    for t in TEXTS:
        base = {w: 2.0 + 0.01 * w + (0.5 if t == "c4val1" else 0.0) for w in range(n)}
        per[t], eng[t] = {}, {}
        for arm in arms:
            off_by = {"R": 0.0, "rep": 0.0, "chunk": chunk, "mutant_scale": mutant}.get(arm, bias.get(t, 0.0))
            ws = range(1) if arm == "rep" else range(n)
            per[t][arm] = [{"window": w, "nll": base[w] + off_by + (0.001 if w % 2 else -0.001) * (arm != "R"),
                            "argmax_agree": 1.0, "kl": 0.001} for w in ws]
            eng[t][arm] = [_fake_pass(C, L, INSTRUMENT_BUCKETS, device_grouping) for _ in ws]
    knobs = ({"GNF4_GEMV_BW": "1", "GNF4_GEMV_BW_PLAN": plan if plan is not None else sel.get("env")}
             if phase == "on" else {"GNF4_GEMV_BW": None, "GNF4_GEMV_BW_PLAN": None})
    return {"mode": "quality", "arm": "B0" if phase == "off" else "B1", "status": "ok", "dry": dry,
            "model": ("/root/p141/fam/tiny/" + TINY) if model == TINY else model, "revision": REVS[model],
            "e4b_sha": E, "gnf4_sha": GNF4_SHA, "group": group, "prompt": 512, "cont": C, "buckets": bk,
            "chunk": 512, "floor_chunk": 256, "layers": L, "windows_sha256": {t: t for t in TEXTS},
            "fusions": {"fuse_qkv_n": 0}, "rep_identical": {t: rep_identical for t in TEXTS}, "per_window": per,
            "engagement": eng, "dispatch_measure": d, "knobs": knobs, "selection_env": sel.get("env"),
            "max_seqs": size[0], "max_tokens_per_seq": size[1]}


def _bucket_missing(rec: dict) -> dict:
    """``rec`` with its first R pass reporting no bucket 8: the instrument genuinely missed one of its buckets."""
    t = TEXTS[0]
    del rec["engagement"][t]["R"][0]["graph_status"]["8"]
    return rec


def _bucket_captured(rec: dict) -> dict:
    """``rec`` with its first R pass's bucket 16 reading ``graph``: the instrument must stay eager in every bucket."""
    t = TEXTS[0]
    rec["engagement"][t]["R"][0]["graph_status"]["16"] = "graph"
    return rec


def self_test() -> int:
    toks = [[i, i + 1] for i in range(16)]
    other = [[i, i + 2] for i in range(16)]

    def arms(**over):
        base = {"B0a": (800.0, 52.0), "B1a": (801.0, 80.0), "B1b": (800.5, 79.8), "B0b": (799.0, 52.1)}
        shared = over.pop("all", {})
        out = {}
        for t, (rc, r1) in base.items():
            kw = {**shared, **over.get(t, {})}
            out[t] = _fake_arm(t, kw.pop("rc", rc), kw.pop("r1", r1), kw.pop("toks", toks), **kw)
        return out
    off, on = _fake_quality("off"), _fake_quality("on", bias={"wikitext": 0.001, "c4val1": 0.001})
    v = lambda a, o=off, n=on, f=FIT, fam="mixtral", sel=SEL, **kw: reduce(a, o, n, E, family=fam, fit=f, sel=sel,  # noqa: E731
                                                                           **kw)["verdict"]
    dsel = _fake_select(family="tiny")
    dry_q = dict(model=TINY, mode="dry", dry=True, device_grouping=False, buckets=[1, 2, 4, 8], sel=dsel, size=(8, 4096))
    dry_a = dict(model=TINY, dry=True, graphs={"1": "eager: graphs off"}, buckets=[1, 2, 4, 8], width=8, sel=dsel,
                 size=(8, 4096))
    sk = _fake_select(dn="16,256,4,2")
    sk_d = {**{k: 0 for k in GEMV + BW}, "bw_prmt32": 64, "bw_splitk": 32}
    cases = [
        ("licensed", v(arms()) == "LICENSED"),
        ("slower at W1", v(arms(B1a={"r1": 53.0}, B1b={"r1": 53.0})) == "SLOWER"),
        ("W16 regression (the no-regression check)", v(arms(B1a={"rc": 780.0}, B1b={"rc": 780.0})) == "SLOWER"),
        ("W16 inside its bar is not a regression", v(arms(B1a={"rc": 795.0}, B1b={"rc": 795.0})) == "LICENSED"),
        ("noisy", v(arms(B0b={"r1": 48.0})) == "NOISY"),
        ("W1 drift inside the old band is NOISY", v(arms(B0b={"r1": 53.2})) == "NOISY"),
        ("function fail", v(arms(B1b={"toks": other})) == "FUNCTION_FAIL"),
        ("reps differ", v(arms(B0a={"rep_ok": False})) == "FUNCTION_FAIL"),
        ("B1 != B0 is reported, not failed", v(arms(B1a={"toks": other}, B1b={"toks": other})) == "LICENSED"),
        ("B1 reached the scalar", v(arms(B1a={"dispatch": {"bw_prmt32": 96, "scalar": 2}})) == "VOID"),
        ("B0 reached bw", v(arms(B0a={"dispatch": {"scalar": 96, "bw_prmt32": 2}})) == "VOID"),
        ("B0 reached dot-pad", v(arms(B0b={"dispatch": {"dotpad": 96}})) == "VOID"),
        ("bw_splitk without a split-K selection is void", v(arms(B1a={"dispatch": sk_d})) == "VOID"),
        ("bw_splitk under a split-K selection", v(arms(all={"sel": sk}, B1a={"dispatch": sk_d}, B1b={"dispatch": sk_d}),
                                                  _fake_quality("off", sel=sk),
                                                  _fake_quality("on", sel=sk, dispatch=sk_d), sel=sk) == "LICENSED"),
        ("a bucket not captured", v(arms(B1a={"graphs": {"1": "eager"}})) == "VOID"),
        ("wrong commit", v(arms(B0a={"e4b": "b" * 40})) == "VOID"),
        ("arms at different widths", v(arms(B1a={"width": 8})) == "VOID"),
        ("another server size", v(arms(all={"size": (16, 4096)})) == "VOID"),
        ("a quality server of another size", v(arms(), o=_fake_quality("off", size=(24, 768))) == "VOID"),
        ("missing arm", v({k: x for k, x in arms().items() if k != "B1b"}) == "VOID"),
        ("fit failed is void, no rerun", reduce({}, None, None, E, family="mixtral", sel=SEL,
                                                fit={"status": "error", "error": "x"})["rerun"] is False),
        ("quality bias", v(arms(), n=_fake_quality("on", bias={"c4val1": 0.02})) == "QUALITY_FAIL"),
        ("quality K8 wikitext", v(arms(), n=_fake_quality("on", bias={"wikitext": 0.009})) == "QUALITY_FAIL"),
        ("mutant passes", v(arms(), o=_fake_quality("off", mutant=0.0001)) == "VOID"),
        ("a quality pass whose bucket captured is void", v(arms(), o=_bucket_captured(_fake_quality("off"))) == "VOID"),
        ("a quality pass missing instrument bucket 8 is void",
         v(arms(all={"buckets": [1, 2, 4, 8, 16, 32, 64]}),
           _bucket_missing(_fake_quality("off", buckets=[1, 2, 4, 8, 16, 32, 64])),
           _fake_quality("on", buckets=[1, 2, 4, 8, 16, 32, 64])) == "VOID"),
        ("reading windows are 24", v(arms(), n=_fake_quality("on", n=12)) == "VOID"),
        # phase 1
        ("no selection is void", v(arms(), sel=None) == "VOID"),
        ("no plan met the contract is FUNCTION_FAIL, no rerun",
         reduce({}, None, None, E, family="mixtral", sel=_fake_select(status="no_plan:down"))["verdict"] == "FUNCTION_FAIL"),
        ("a choice that is not its rule's is void",
         v(arms(), sel=_fake_select(selected={"gate_up": [16, 256, 4, 1], "down": [16, 256, 4, 1]})) == "VOID"),
        ("a selected plan off the contract is void",
         v(arms(), sel=_fake_select(bad_contract=("gate_up", "16,1024,4,1"))) == "VOID"),
        ("a plan string that is not the selection is void", v(arms(), sel=_fake_select(env="28672,4096=16,256,4,1"))
         == "VOID"),
        ("B1 at another plan string is void", v(arms(B1a={"plan": "28672,4096=16,256,4,1;4096,14336=16,256,4,1"}))
         == "VOID"),
        ("B1 at the shipped table is void", v(arms(B1b={"plan": ""})) == "VOID"),
        ("quality ON at another plan is void", v(arms(), n=_fake_quality("on", plan="x")) == "VOID"),
        ("a tiny selection under mixtral is void", v(arms(), sel=_fake_select(family="tiny")) == "VOID"),
        ("the rule re-derived", selection_rule(SEL) == {"gate_up": "16,1024,4,1", "down": "16,256,4,1"}),
        ("selection reported, never ruled", reduce(arms(), off, on, E, family="mixtral", fit=FIT, sel=SEL)
         ["selection"]["reported_never_ruled"]["plan_ms"] == {"gate_up": 1.0, "down": 1.0}),
        # dry
        ("dry family", v(arms(all=dry_a), _fake_quality("off", **dry_q), _fake_quality("on", **dry_q),
                         f={**FIT, "dry": True}, fam="tiny", dry=True, sel=dsel) == "LICENSED"),
        ("a dry family in a reading refuses", _raises(lambda: v(arms(), fam="tiny"))),
        ("a dry record in a reading is void", v(arms(all={**dry_a, "model": MIXTRAL, "sel": SEL, "size": SERVER_SIZE}))
         == "VOID"),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"p141_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--family", choices=sorted(FAMILIES))
    p.add_argument("--out")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.family:
        raise SystemExit("REFUSED: --family")
    f = a.family

    def load(name):
        q = os.path.join(a.dir, name)
        return json.load(open(q)) if os.path.exists(q) else None
    arms = {t: r for t in TAGS if (r := load(f"arm_{t}_{f}.json"))}
    v = reduce(arms, load(f"quality_off_{f}.json"), load(f"quality_on_{f}.json"), a.e4b_sha, family=f,
               fit=load(f"fit_{f}.json"), sel=load("select.json"), dry=a.dry)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P141_VERDICT {f}{' (DRY: never a reading)' if a.dry else ''} {v['verdict']} "
          f"{json.dumps(v['reasons'], default=str)[:1500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
