#!/usr/bin/env python3
"""p133_reduce.py -- lane P133's registered rule (bench/p133/PREREG-p133.md), derived from bench/p116/p116_reduce.py by
named substitutions: the RTX 4090's incumbent is the scalar GEMV for every model (dot-pad and auto both need >= 160 SMs);
the control is ``Wn`` at the box's own width, read from the records, not P116's fixed W16; the captured buckets are the
records' own; grouped-nf4-gemm is pinned at 0.46.0; and a DRY run (``P133_DRY=1`` on a tiny fixture, any card,
eager allowed) reduces through the same rungs but is never a reading.

Records (``p133_box.py``): ``fit.json``; ``arm_{B0a,B1a,B1b,B0b}.json`` (speed, W1 and the control) and
``quality_off.json`` / ``quality_on.json`` (P110's teacher-forced instrument at one window per pass, T == 1). B0 = the
shipped default (``GNF4_GEMV_BW`` unset); B1 = ``GNF4_GEMV_BW=1`` at the shipped plan table.

The rule, first rung that applies:
  VOID           the fit check failed (W1 does not fit: no arm ran, no rerun); a record is missing or not ok; another
                 e4b / grouped-nf4-gemm commit or model revision; a bucket not captured (a reading or proof); the arms read different prompts or lengths or fusion censuses; a void slope; ENGAGEMENT off:
                 B1 must dispatch every single-row decode GEMV to ``bw_prmt32`` (no dot-pad, no scalar, no tree, no
                 split-K) and B0 none to ``bw_*`` and some to the scalar GEMV (the incumbent at 128 SMs); the quality records fail Phase B's integrity checks at one window per pass (window counts,
                 decode attention calls, bucket-1 eager steps, no replays, glue-kernel and q/k/v calls equal between R
                 and ON, the same dispatch rule); or the scale mutant passes the quality bar (the gate cannot fail).
  NOISY          a self-pair (B0b/B0a, B1b/B1a) outside [0.96, 1.04] at the control, or outside [0.985, 1.015] at W1
                 (half the W1 gain bar, P116's rule).
  FUNCTION_FAIL  B0b != B0a or B1b != B1a on any row, workload or length; timed reps of one arm digest differently.
                 (B1 != B0 is expected -- another reduction order -- and reported.)
  QUALITY_FAIL   on either text bias_ON > B_floor + 0.01 or spread_ON > 2 * max(S_floor, 0.005), the floor drawn from
                 ``chunk`` (and ``rep`` when R did not repeat bit for bit); on wikitext |K8 delta ppl| > 0.05.
  SLOWER         g1 = min(B1a/B0a, B1b/B0b) at W1 < 1.03, or gW (the same at the control) < 0.99. No control
                 (width < 2): gW is reported void and W1 reads alone.
  LICENSED       otherwise: the measured configuration (sm_89, 128 SMs) only, never sm_89 in general.

    python p133_reduce.py --dir RUN_DIR --out verdict.json [--e4b-sha SHA] [--proof | --dry]
    python p133_reduce.py --self-test
"""
from __future__ import annotations

import argparse
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
GNF4_SHA = "6b7c2e7474d20c8843814dd46dad6cfc3d94b440"     # grouped-nf4-gemm 0.46.0, exact (its _BW_PLANS: K33's plans)
QWEN, GRAN, TINY = "Qwen/Qwen3-30B-A3B", "ibm-granite/granite-3.1-3b-a800m-instruct", "p133-tiny"
REVS = {QWEN: "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", GRAN: "a02780686e08a03fe0d2679a293b5c74a90efa89", TINY: ""}
LAYERS = {QWEN: 48, GRAN: 32, TINY: 2}
WINDOWS = {QWEN: 24, GRAN: 12, TINY: 4}    # per text: the reading, the proof, the dry run
INCUMBENT = {QWEN: "scalar", GRAN: "scalar", TINY: "scalar"}   # the single-row route at 128 SMs (and on sm_86)
OFF_ARMS = ("R", "rep", "chunk", "mutant_scale")
#: The buckets P110's instrument enables at capture=False in every quality pass (``p110_box.BUCKETS``; P116's BUCKETS). The
#: pass integrity reads these, not the server's buckets, which follow its own max_seqs (Amendment 1: p133-4090-prove-1 VOIDed
#: comparing passes against the server's 1-64).
INSTRUMENT_BUCKETS = (1, 2, 4, 8, 16)
FLOORS = ("chunk",)
GEMV = ("dotpad", "dotpad_splitk", "scalar", "scalar_splitk")
BW = ("bw_tree", "bw_prmt32", "bw_splitk")


def _rate(r, w):
    return (r.get("workloads", {}).get(w) or {}).get("decode_tok_s")


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def engagement_faults(name, arm, model, d) -> list:
    """The dispatch rule on a tally delta ``d``: B1 all ``bw_prmt32``; B0 none to ``bw_*``, some to the incumbent."""
    d = d or {}
    inc = INCUMBENT.get(model, "dotpad")
    other = [k for k in GEMV if not k.startswith(inc)]
    if arm == "B1":
        if d.get("bw_prmt32", 0) <= 0 or any(d.get(k, 0) for k in ("bw_tree", "bw_splitk") + GEMV):
            return [f"{name}: B1 dispatched {d}, registered bw_prmt32 only"]
    elif any(d.get(k, 0) for k in BW) or d.get(inc, 0) + d.get(inc + "_splitk", 0) <= 0 \
            or any(d.get(k, 0) for k in other):
        return [f"{name}: B0 dispatched {d}, registered {inc} only"]
    return []


def _model_of(proof: bool, dry: bool) -> str:
    return TINY if dry else GRAN if proof else QWEN


def _model_name(rec) -> str:
    """The record's model, with a dry run's local fixture path read as :data:`TINY`."""
    m = rec.get("model") or ""
    return TINY if rec.get("dry") and m.rstrip("/").endswith(TINY) else m


def speed_faults(arms: dict, e4b_sha: str, proof: bool, dry: bool = False) -> list:
    void = []
    want_model = _model_of(proof, dry)
    for t, r in arms.items():
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
        void += engagement_faults(t, t[:2], want_model, r.get("dispatch_total"))
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
    is required except in a dry run (sm_86 groups on the host: its first execution is the paid proof)."""
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


def quality_faults(off: dict, on: dict, e4b_sha: str, proof: bool, dry: bool = False) -> list:
    void = []
    want_model = _model_of(proof, dry)
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
    if void:
        return void
    L, want_n, C = LAYERS[want_model], WINDOWS[want_model], off["cont"]
    for k in ("cont", "prompt", "chunk", "floor_chunk", "layers", "windows_sha256", "fusions", "buckets"):
        if off.get(k) != on.get(k):
            void.append(f"the phases differ in {k}: {off.get(k)} / {on.get(k)}")
    if off.get("layers") != L:
        void.append(f"{off.get('layers')} layers, the family registers {L}")
    void += engagement_faults("quality_off", "B0", want_model, off.get("dispatch_measure"))
    void += engagement_faults("quality_on", "B1", want_model, on.get("dispatch_measure"))
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


def reduce(arms: dict, off: dict | None, on: dict | None, e4b_sha: str, *, fit=None, proof=False, dry=False) -> dict:
    out = {"lane": "P133", "proof": proof, "dry": dry, "verdict": None, "reasons": [],
           "bars": {"self_pair": [SELF_LO, SELF_HI], "self_pair_w1": [SELF_W1_LO, SELF_W1_HI], "gain_min_w1": GAIN_MIN_W1,
                    "gain_min_control": GAIN_MIN_WCTL, "tol": TOL, "spread_x": SPREAD_X, "spread_min": SPREAD_MIN,
                    "k8_budget": K8_BUDGET, "k8_gated": list(K8_GATED), "floors": list(FLOORS)}}
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
    void = speed_faults(arms, e4b_sha, proof, dry) + quality_faults(off, on, e4b_sha, proof, dry)
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


def _fake_arm(tag, rate_c, rate1, toks, *, model=QWEN, e4b=E, graphs=None, dispatch=None, rep_ok=True, width=16,
              dry=False, buckets=None):
    arm = tag[:2]
    inc = INCUMBENT[model]
    L = LAYERS[model]
    bk = buckets if buckets is not None else BUCKETS16
    d = dispatch or ({**{k: 0 for k in GEMV + BW}, "bw_prmt32": 2 * L} if arm == "B1"
                     else {**{k: 0 for k in GEMV + BW}, inc: 2 * L})
    w = {}
    for name, b, rate in (((f"W{width}", width, rate_c),) if width >= 2 else ()) + (("W1", 1, rate1),):
        w[name] = {"batch": b, "decode_tok_s": rate, "decode_ms_per_step": 1000.0 / rate * b,
                   "tokens": {"32": toks[:b], "160": toks[:b]},
                   "rep_digests": {"32": ["x", "x", "x" if rep_ok else "y"], "160": ["z"] * 3}}
    return {"arm": arm, "tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "dry": dry,
            "model": ("/workspace/p133-dry/" + TINY) if model == TINY else model, "revision": REVS[model],
            "graph_status": graphs or {str(b): "graph" for b in bk}, "buckets": bk, "max_seqs": max(width, 1),
            "control": f"W{width}" if width >= 2 else None,
            "fusions": {"fuse_qkv_n": 0, "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": [0, 0], "fuse_router_epilogue_n": 0},
            "prompts_sha256": "p", "short": 32, "long": 160, "reps": 3, "workloads": w, "dispatch_total": d}


def _fake_pass(C, L, buckets, device_grouping=True):
    gs = {str(b): {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in buckets}
    gs["1"]["eager_steps"] = C - 1
    return {"decode_calls": (C - 1) * L, "grouping_flags_in_pass": {"device_grouping": device_grouping},
            "graph_status": {str(b): "eager: capture=False" for b in buckets}, "graph_stats": gs,
            "kernels": {k: 0 for k in pb.KERNELS}, "forwards": C, "qkv_calls": 0}


def _fake_quality(phase, *, model=QWEN, bias=None, mutant=0.4, chunk=-0.0004, rep_identical=True, dispatch=None,
                  n=None, group=1, dry=False, device_grouping=True, buckets=None):
    L, C = LAYERS[model], (128 if model == QWEN else 32)
    n = n if n is not None else WINDOWS[model]
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
    return {"mode": "quality", "arm": "B0" if phase == "off" else "B1", "status": "ok", "dry": dry,
            "model": ("/workspace/p133-dry/" + TINY) if model == TINY else model, "revision": REVS[model],
            "e4b_sha": E, "gnf4_sha": GNF4_SHA, "group": group, "prompt": 512, "cont": C, "buckets": bk,
            "chunk": 512, "floor_chunk": 256, "layers": L, "windows_sha256": {t: t for t in TEXTS},
            "fusions": {"fuse_qkv_n": 0}, "rep_identical": {t: rep_identical for t in TEXTS}, "per_window": per,
            "engagement": eng, "dispatch_measure": d}


def _bucket_captured(rec: dict) -> dict:
    """``rec`` with its first R pass's bucket 16 reading ``graph``: the instrument must stay eager in every bucket."""
    t = TEXTS[0]
    rec["engagement"][t]["R"][0]["graph_status"]["16"] = "graph"
    return rec


def self_test() -> int:
    toks = [[i, i + 1] for i in range(16)]
    other = [[i, i + 2] for i in range(16)]

    def arms(**over):
        base = {"B0a": (800.0, 110.0), "B1a": (801.0, 125.0), "B1b": (800.5, 124.8), "B0b": (799.0, 110.2)}
        shared = over.pop("all", {})
        out = {}
        for t, (rc, r1) in base.items():
            kw = {**shared, **over.get(t, {})}
            out[t] = _fake_arm(t, kw.pop("rc", rc), kw.pop("r1", r1), kw.pop("toks", toks), **kw)
        return out
    off, on = _fake_quality("off"), _fake_quality("on", bias={"wikitext": 0.001, "c4val1": 0.001})
    v = lambda a, o=off, n=on, f=FIT, **kw: reduce(a, o, n, E, fit=f, **kw)["verdict"]  # noqa: E731
    dry_q = dict(model=TINY, dry=True, device_grouping=False, buckets=[1, 2, 4, 8])
    dry_a = dict(model=TINY, dry=True, graphs={"1": "eager: graphs off"}, buckets=[1, 2, 4, 8], width=8)
    cases = [
        ("licensed", v(arms()) == "LICENSED"),
        ("slower at W1", v(arms(B1a={"r1": 111.0}, B1b={"r1": 111.0})) == "SLOWER"),
        ("control regression", v(arms(B1a={"rc": 780.0}, B1b={"rc": 780.0})) == "SLOWER"),
        ("noisy", v(arms(B0b={"r1": 100.0})) == "NOISY"),
        ("W1 drift inside the old band is NOISY", v(arms(B0b={"r1": 112.5})) == "NOISY"),
        ("control drift inside its band is not", v(arms(B0b={"rc": 820.0}, B1b={"rc": 821.0})) == "LICENSED"),
        ("function fail", v(arms(B1b={"toks": other})) == "FUNCTION_FAIL"),
        ("reps differ", v(arms(B0a={"rep_ok": False})) == "FUNCTION_FAIL"),
        ("B1 != B0 is reported, not failed", v(arms(B1a={"toks": other}, B1b={"toks": other})) == "LICENSED"),
        ("B1 reached the scalar", v(arms(B1a={"dispatch": {"bw_prmt32": 96, "scalar": 2}})) == "VOID"),
        ("B1 took the tree", v(arms(B1a={"dispatch": {"bw_prmt32": 0, "bw_tree": 96}})) == "VOID"),
        ("B0 reached bw", v(arms(B0a={"dispatch": {"scalar": 96, "bw_prmt32": 2}})) == "VOID"),
        ("B0 reached dot-pad (not the 128-SM route)", v(arms(B0b={"dispatch": {"dotpad": 96}})) == "VOID"),
        ("a bucket not captured", v(arms(B1a={"graphs": {"1": "eager"}})) == "VOID"),
        ("no buckets recorded is void", v(arms(all={"buckets": []})) == "VOID"),
        ("wrong commit", v(arms(B0a={"e4b": "b" * 40})) == "VOID"),
        ("arms at different widths", v(arms(B1a={"width": 8})) == "VOID"),
        ("control at width 8", v(arms(all={"width": 8})) == "LICENSED"
         and reduce(arms(all={"width": 8}), off, on, E, fit=FIT)["control"] == "W8"),
        ("no control: W1 reads alone", reduce(arms(all={"width": 1}), off, on, E, fit=FIT)["gW"] is None
         and v(arms(all={"width": 1})) == "LICENSED"),
        ("no control, W1 slower", v(arms(all={"width": 1}, B1a={"r1": 111.0}, B1b={"r1": 111.0})) == "SLOWER"),
        ("missing arm", v({k: x for k, x in arms().items() if k != "B1b"}) == "VOID"),
        ("fit failed is void before arms, no rerun", reduce({}, None, None, E, fit={"status": "error", "error": "OOM"})
         ["rerun"] is False),
        ("a later VOID keeps the rerun", reduce(arms(B0a={"e4b": "b" * 40}), off, on, E, fit=FIT)["rerun"] is True),
        ("no fit record is void", v(arms(), f=None) == "VOID"),
        ("quality bias", v(arms(), n=_fake_quality("on", bias={"c4val1": 0.02})) == "QUALITY_FAIL"),
        ("quality K8 wikitext", v(arms(), n=_fake_quality("on", bias={"wikitext": 0.009})) == "QUALITY_FAIL"),
        ("c4val1 K8 reported only", v(arms(), n=_fake_quality("on", bias={"c4val1": 0.009})) == "LICENSED"),
        ("mutant passes", v(arms(), o=_fake_quality("off", mutant=0.0001)) == "VOID"),
        ("quality group 12", v(arms(), o=_fake_quality("off", group=12)) == "VOID"),
        ("quality short", v(arms(), n=_fake_quality("on", n=23)) == "VOID"),
        ("quality on reached the scalar", v(arms(), n=_fake_quality("on", dispatch={"bw_prmt32": 9, "scalar": 1}))
         == "VOID"),
        ("server buckets 1-64, instrument passes at 1-16 (Amendment 1)",
         v(arms(all={"buckets": [1, 2, 4, 8, 16, 32, 64]}), _fake_quality("off", buckets=[1, 2, 4, 8, 16, 32, 64]),
           _fake_quality("on", buckets=[1, 2, 4, 8, 16, 32, 64])) == "LICENSED"),
        ("a quality pass whose bucket captured is void", v(arms(), o=_bucket_captured(_fake_quality("off"))) == "VOID"),
        ("quality host grouping is void", v(arms(), o=_fake_quality("off", device_grouping=False),
                                            n=_fake_quality("on", device_grouping=False)) == "VOID"),
        ("rep not identical joins the floor", reduce(arms(), _fake_quality("off", rep_identical=False), on, E, fit=FIT)
         ["quality"]["wikitext"]["floor"]["draws"] == ["chunk", "rep"]),
        ("proof", v({t: _fake_arm(t, 300.0, 50.0 if t[:2] == "B0" else 60.0, toks, model=GRAN) for t in TAGS},
                    _fake_quality("off", model=GRAN), _fake_quality("on", model=GRAN), proof=True) == "LICENSED"),
        ("proof on qwen is void", v(arms(), proof=True) == "VOID"),
        ("dry: eager, host grouping, width 8", v(arms(all=dry_a), _fake_quality("off", **dry_q),
                                                 _fake_quality("on", **dry_q), f={**FIT, "dry": True}, dry=True)
         == "LICENSED"),
        ("a dry record in a reading is void", v(arms(all=dry_a), _fake_quality("off", **dry_q),
                                                _fake_quality("on", **dry_q), f={**FIT, "dry": True}) == "VOID"),
        ("a reading record in a dry reduction is void", v(arms(), f={**FIT, "dry": True}, dry=True) == "VOID"),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"p133_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--proof", action="store_true")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()

    def load(name):
        f = os.path.join(a.dir, name)
        return json.load(open(f)) if os.path.exists(f) else None
    arms = {t: r for t in TAGS if (r := load(f"arm_{t}.json"))}
    if a.proof and a.dry:
        raise SystemExit("REFUSED: --proof and --dry")
    v = reduce(arms, load("quality_off.json"), load("quality_on.json"), a.e4b_sha, fit=load("fit.json"),
               proof=a.proof, dry=a.dry)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P133_VERDICT{' (DRY: never a reading)' if a.dry else ''} {v['verdict']} "
          f"{json.dumps(v['reasons'], default=str)[:1500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
