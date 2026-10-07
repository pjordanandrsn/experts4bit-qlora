#!/usr/bin/env python3
"""p115c_reduce.py -- lane P115 Phase C's registered rule (bench/p115/PREREG-p115.md, Amendment 2; e4b#1313).

Receipts, per model tag M (``p115c_box.py``): ``serve_M_off.json``, ``serve_M_on.json``, ``serve_M_explicit.json``; for
gpt-oss and Qwen3.6 ``sane_M_off.json`` and ``sane_M_on.json``; for Granite ``quality_granite_off.json`` and
``quality_granite_on.json`` (Phase B's full read; in the proof Granite carries both); ``harness_M.json`` when the runner
could not fetch, bake or prompt M.

Per model, the gates:
  SERVED          the off and on servers built and decoded every prompt to its length (off twice).
  ENGAGED         the on census is the registered prediction (``PREDICTED``); the off census is all zero.
  EXPLICIT_RAISE  all four knobs at 1 refused at build with one of the four knobs' own vacuous-enable refusals
                  (``EXPLICIT_RE``: the message names the knob at ``=1``; another feature's refusal does not count).
  DETERMINISM     the off server's two passes emit the same tokens.
  SANE            (gpt-oss, Qwen3.6) the maintainer's gross-error gate (59b79fb7): ON scored against R on
                  ``SANE_WINDOWS`` wikitext windows, |mean d_ON| <= 0.02 nats and mean argmax agreement >= 0.95. A model
                  the instrument cannot build (either SANE record missing or not ok) fails it.
  QUALITY         (Granite; Amendment 2, asked for in review) Phase B's rule unchanged, read with ``p115_reduce``'s own
                  functions: the records' integrity (census, windows, every pass's engagement -- else VOID), the mutant
                  must fail the bar (else VOID), then on both texts bias_ON <= B_floor + 0.01 and spread_ON <=
                  2 * max(S_floor, 0.005), and on wikitext |K8 delta| <= 0.05 (c4val1's K8 reported).
Reported, not gated: IDENTITY (on tokens against off's), the folds' reports, the SANE KL, c4val1's K8.

Verdict (gpt-oss and Qwen3.6; Granite's does not enter it):
  VOID           a record names another e4b / grouped-nf4-gemm commit or another model revision; or the harness could not
                 fetch, bake or prompt gpt-oss or Qwen3.6 (``harness_M.json``).
  FLIP_LICENSED  both models pass every gate.
  FLIP_HELD      otherwise (the failing gates named).
Granite's own verdict (``granite``): GRANITE_LICENSED iff it passes every gate; GRANITE_HELD otherwise; VOID when its
harness failed or its quality records fail Phase B's integrity checks. The flip PR's ``auto`` allowlist takes Granite
only on GRANITE_LICENSED.

The proof (``--proof``) runs Granite alone with every gate, SANE and QUALITY both, at the proof's sizes; its verdict is
FLIP_LICENSED / FLIP_HELD / VOID over Granite, and it is not a reading.

    python p115c_reduce.py --dir RUN_DIR --out verdict_c.json [--e4b-sha SHA] [--proof]
    python p115c_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p115_reduce as pb  # noqa: E402  (Phase B's rule, staged at its registered bytes)

GNF4_SHA = "b4f93f1c62d1e3436ed45bec8ccd608c90433737"
MODELS = {"gptoss": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
          "qw36": ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
          "granite": ("ibm-granite/granite-3.1-3b-a800m-instruct", "a02780686e08a03fe0d2679a293b5c74a90efa89")}
FLIP_MODELS, PROOF = ("gptoss", "qw36"), ("granite",)
READING = ("granite",) + FLIP_MODELS                       # the box's order: Granite first (cheapest), then gpt-oss
SANE_MODELS = {False: ("gptoss", "qw36"), True: ("granite",)}
QUALITY_MODELS = ("granite",)
QUALITY_WINDOWS = {False: 48, True: 12}                    # Phase B's registered 48 per text; the proof's 12
GRANITE_LAYERS = 32
PREDICTED = {"gptoss": {"fuse_qkv_n": 0, "fuse_t1_glue_n": 49, "fuse_t1_glue_r2_n": [24, 0], "fuse_router_epilogue_n": 24},
             "qw36": {"fuse_qkv_n": 0, "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": [0, 0], "fuse_router_epilogue_n": 40},
             "granite": {"fuse_qkv_n": 0, "fuse_t1_glue_n": 65, "fuse_t1_glue_r2_n": [32, 32], "fuse_router_epilogue_n": 32}}
#: one of the four knobs' own refusal of a vacuous enable: fused q/k/v matched no Qwen3-MoE attention
#: ("E4B_PAGED_FUSE_QKV=1 matched no attention module -- refusing a vacuous fusion"), or a fold patched nothing
#: ("E4B_FUSE_T1_GLUE=1 patched no RMSNorm modules ... -- refusing a vacuous enable"; Qwen3.6's centered norms make glue
#: round 1 refuse before the q/k/v check is reached: Amendment 2's correction). The knob must be named at =1, so the
#: same phrase from another feature (the int4 expert plan, the int4 attention) does not pass the gate.
EXPLICIT_RE = re.compile(r"\b(E4B_PAGED_FUSE_QKV|E4B_FUSE_T1_GLUE|E4B_FUSE_T1_GLUE_R2|E4B_FUSE_ROUTER_EPI)=1\b.*"
                         r"refusing a vacuous (enable|fusion)", re.S)
SANE_WINDOWS, SANE_BIAS, SANE_ARGMAX = 12, 0.02, 0.95
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
RECORD_PREFIXES = ("serve", "sane", "quality", "harness")


def _zero(v):
    return all(x == 0 for x in v) if isinstance(v, (list, tuple)) else v == 0


def quality_gate(off, on, *, proof):
    """Phase B's rule on Granite's records, with ``p115_reduce``'s own functions. Returns (verdict, reasons, read):
    VOID (integrity), QUALITY_FAIL or QUALITY_PASS."""
    if not (off and on and off.get("status") == "ok" and on.get("status") == "ok"):
        return "VOID", ["a quality record is missing or not ok"], None
    void, fam, L = [], "granite", GRANITE_LAYERS
    want_n, C, group = QUALITY_WINDOWS[proof], off.get("cont"), off.get("group")
    for k in ("cont", "group", "prompt", "chunk", "floor_chunk", "layers", "windows_sha256"):
        if off.get(k) != on.get(k):
            void.append(f"the phases differ in {k}: {off.get(k)} / {on.get(k)}")
    if off.get("layers") != L:
        void.append(f"{off.get('layers')} layers, Granite registers {L}")
    for name, rec, arms in (("quality_off", off, pb.OFF_ARMS), ("quality_on", on, ("ON",))):
        cen = rec.get("census") or {}
        if rec.get("fuse_qkv"):
            void.append(f"{name}: fused q/k/v engaged on Granite")
        if name == "quality_off":
            if not all(_zero(cen.get(k, 1)) for k in CENSUS_KEYS):
                void.append(f"quality_off: census {cen}")
        elif {k: cen.get(k) for k in CENSUS_KEYS} != pb.census_for(fam, L, False):
            void.append(f"quality_on: census {cen}, registered {pb.census_for(fam, L, False)}")
        for t in pb.TEXTS:
            per = (rec.get("per_window") or {}).get(t, {})
            for arm in arms:
                n = len(per.get(arm, []))
                want = group if arm == "rep" else want_n
                if n != want:
                    void.append(f"{name} {t}: arm {arm} has {n} windows, expected {want}")
            sizes = pb._groups(want_n, group) if group else []
            for arm in arms:
                passes = (rec.get("engagement") or {}).get(t, {}).get(arm, [])
                arm_sizes = sizes[:1] if arm == "rep" else sizes
                if len(passes) != len(arm_sizes):
                    void.append(f"{name} {t} {arm}: {len(passes)} passes, expected {len(arm_sizes)}")
                    continue
                for k, (e, n) in enumerate(zip(passes, arm_sizes)):
                    void += [f"{name} {t} {arm} pass {k}: {m}" for m in pb._pass_faults(e, arm, n, C, L, fam, False)]
    if void:
        return "VOID", void, None
    q = pb.quality_read(off, on)
    void = [f"{t}: mutant_scale passes the bar (the gate cannot fail)" for t in pb.TEXTS if q[t]["mutant_passes"]]
    if void:
        return "VOID", void, q
    fail = []
    for t in pb.TEXTS:
        s = q[t]["stats"]["ON"]
        if not q[t]["on_passes"]:
            fail.append(f"{t}: ON bias {s['bias']:+.5f} (bar {q[t]['bias_bar']:.5f}), spread {s['spread']:.5f} "
                        f"(bar {q[t]['spread_bar']:.5f})")
        if q[t]["k8"]["gated"] and not q[t]["k8"]["passes"]:
            fail.append(f"{t}: K8 perplexity {q[t]['k8']['ppl_R']:.5f} -> {q[t]['k8']['ppl_ON']:.5f} (|delta| > {pb.K8_BUDGET})")
    if fail:
        return "QUALITY_FAIL", fail, q
    return "QUALITY_PASS", [f"{t} ON bias {q[t]['stats']['ON']['bias']:+.5f} (bar {q[t]['bias_bar']:.5f}), K8 "
                            f"{q[t]['k8']['delta']:+.4f}" for t in pb.TEXTS], q


def model_gates(m, recs, *, new, proof=False):
    so, sn, sx = recs.get(f"serve_{m}_off"), recs.get(f"serve_{m}_on"), recs.get(f"serve_{m}_explicit")
    g, why, extra = {}, {}, {}
    ok_serve = all(r and r.get("status") == "ok" and r.get("tokens") for r in (so, sn))
    if ok_serve:
        ok_serve = (len(so["tokens"]) == 2 and len(sn["tokens"]) == 1
                    and all(len(t) == new for rows in so["tokens"] + sn["tokens"] for t in rows))
    g["SERVED"] = ok_serve
    if not ok_serve:
        why["SERVED"] = f"off {so and so.get('status')}, on {sn and sn.get('status')}"
    cen_on = {k: (sn or {}).get("census", {}).get(k) for k in CENSUS_KEYS}
    cen_off = (so or {}).get("census") or {}
    g["ENGAGED"] = bool(sn) and cen_on == PREDICTED[m] and bool(cen_off) and all(_zero(cen_off.get(k, 1)) for k in CENSUS_KEYS)
    if not g["ENGAGED"]:
        why["ENGAGED"] = f"on census {cen_on} (registered {PREDICTED[m]}), off census {cen_off}"
    msg = (sx or {}).get("exc_message") or ""
    g["EXPLICIT_RAISE"] = bool(sx) and sx.get("status") == "raised" and sx.get("exc_type") == "RuntimeError" \
        and bool(EXPLICIT_RE.search(msg))
    if not g["EXPLICIT_RAISE"]:
        why["EXPLICIT_RAISE"] = f"{sx and sx.get('status')}: {sx and sx.get('exc_type')}: {msg[:160]}"
    g["DETERMINISM"] = ok_serve and so["tokens"][0] == so["tokens"][1]
    if not g["DETERMINISM"]:
        why["DETERMINISM"] = "the off server's two passes differ" if ok_serve else "not served"
    stats = None
    if m in SANE_MODELS[proof]:
        qo, qn = recs.get(f"sane_{m}_off"), recs.get(f"sane_{m}_on")
        sane_ok = False
        if qo and qn and qo.get("status") == "ok" and qn.get("status") == "ok":
            ref = {x["window"]: x["nll"] for x in qo["per_window"]["wikitext"]["R"]}
            on = qn["per_window"]["wikitext"]["ON"]
            if len(ref) == SANE_WINDOWS and len(on) == SANE_WINDOWS:
                d = [x["nll"] - ref[x["window"]] for x in on]
                stats = {"bias": sum(d) / len(d), "argmax_agree": sum(x["argmax_agree"] for x in on) / len(on),
                         "mean_kl": sum(x.get("kl", 0.0) for x in on) / len(on), "n": len(on)}
                sane_ok = abs(stats["bias"]) <= SANE_BIAS and stats["argmax_agree"] >= SANE_ARGMAX
        g["SANE"] = sane_ok
        if not sane_ok:
            why["SANE"] = f"stats {stats}" if stats else "the instrument did not build or scored short"
    if m in QUALITY_MODELS:
        qv, reasons, read = quality_gate(recs.get(f"quality_{m}_off"), recs.get(f"quality_{m}_on"), proof=proof)
        g["QUALITY"] = qv == "QUALITY_PASS"
        extra["quality"] = {"verdict": qv, "reasons": reasons, "read": read}
        if not g["QUALITY"]:
            why["QUALITY"] = f"{qv}: {reasons}"
    ident = None
    if ok_serve:
        a, b = sn["tokens"][0], so["tokens"][0]
        ident = {"rows_identical": sum(x == y for x, y in zip(a, b)), "rows": len(a)}
    return g, why, {"sane": stats, "identity": ident, "fusion_report": (sn or {}).get("fusion_report"),
                    "explicit_message": msg[:300], **extra}


def reduce(recs: dict, e4b_sha: str, *, proof=False, new=32) -> dict:
    flip_models = PROOF if proof else FLIP_MODELS
    out = {"lane": "P115 Phase C", "proof": proof, "models": {}, "verdict": None, "reasons": [], "granite": None,
           "bars": {"sane_bias": SANE_BIAS, "sane_argmax": SANE_ARGMAX, "sane_windows": SANE_WINDOWS,
                    "quality": {"tol": pb.TOL, "spread_x": pb.SPREAD_X, "spread_min": pb.SPREAD_MIN,
                                "k8_budget": pb.K8_BUDGET, "k8_gated": list(pb.K8_GATED),
                                "windows": QUALITY_WINDOWS[proof]}}}
    void = []
    for m in flip_models:
        if f"harness_{m}" in recs:
            void.append(f"{m}: harness {recs[f'harness_{m}'].get('reason')}")
    for name, r in recs.items():
        if name.startswith("harness_"):
            continue
        m = r.get("model_tag")
        if r.get("e4b_sha") != e4b_sha:
            void.append(f"{name}: e4b {r.get('e4b_sha')} != {e4b_sha}")
        if r.get("gnf4_sha") != GNF4_SHA:
            void.append(f"{name}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
        if m in MODELS and r.get("model") and (r.get("model"), r.get("revision")) != MODELS[m]:
            void.append(f"{name}: model {r.get('model')}@{r.get('revision')} is not {m}'s registered pin")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    held = []
    for m in flip_models:
        g, why, rep = model_gates(m, recs, new=new, proof=proof)
        out["models"][m] = {"gates": g, "why": why, **rep}
        if proof and rep.get("quality", {}).get("verdict") == "VOID":
            out.update(verdict="VOID", reasons=[f"{m}: quality records fail Phase B's integrity checks: "
                                                f"{rep['quality']['reasons']}"])
            return out
        held += [f"{m}: {k} -- {why[k]}" for k, v in g.items() if not v]
    if held:
        out.update(verdict="FLIP_HELD", reasons=held)
    else:
        out.update(verdict="FLIP_LICENSED", reasons=[f"{m}: every gate passed (census {PREDICTED[m]}"
                                                     + (f"; SANE {out['models'][m]['sane']}" if out["models"][m]["sane"] else "")
                                                     + ")" for m in flip_models])
    if not proof:
        if "harness_granite" in recs:
            out["granite"] = {"verdict": "VOID", "reasons": [f"harness {recs['harness_granite'].get('reason')}"]}
        else:
            g, why, rep = model_gates("granite", recs, new=new, proof=False)
            out["models"]["granite"] = {"gates": g, "why": why, **rep}
            qv = rep["quality"]["verdict"]
            if qv == "VOID":
                out["granite"] = {"verdict": "VOID", "reasons": rep["quality"]["reasons"]}
            elif all(g.values()):
                out["granite"] = {"verdict": "GRANITE_LICENSED", "reasons": rep["quality"]["reasons"]}
            else:
                out["granite"] = {"verdict": "GRANITE_HELD", "reasons": [f"{k} -- {why[k]}" for k, v in g.items() if not v]}
    return out


# ------------------------------------------------------------------ self-test --
E = "a" * 40


def _fake_quality(phase, *, n, C, bias=None, mutant=0.4, census=None):
    """Granite's Phase B records in ``p115_quality``'s shape (``p115_reduce._fake_pass`` for every pass)."""
    fam, L, group = "granite", GRANITE_LAYERS, 12
    bias = bias or {}
    arms = pb.OFF_ARMS if phase == "off" else ("ON",)
    per, eng = {}, {}
    for t in pb.TEXTS:
        base = {w: 2.0 + 0.01 * w + (0.5 if t == "c4val1" else 0.0) for w in range(n)}
        per[t], eng[t] = {}, {}
        for arm in arms:
            off_by = {"R": 0.0, "rep": 0.0, "half": 0.0005, "chunk": -0.0004, "mutant_scale": mutant}.get(arm, bias.get(t, 0.0))
            ws = range(group) if arm == "rep" else range(n)
            per[t][arm] = [{"window": w, "nll": base[w] + off_by + (0.001 if w % 2 else -0.001) * (arm != "R"),
                            "argmax_agree": 1.0, "kl": 0.001} for w in ws]
            sizes = pb._groups(n, group)[:1] if arm == "rep" else pb._groups(n, group)
            eng[t][arm] = [pb._fake_pass(arm, s, C, L, fam, False) for s in sizes]
    on = phase == "on"
    cen = census or (pb.census_for(fam, L, False) if on else {k: (0 if k != "fuse_t1_glue_r2_n" else [0, 0]) for k in CENSUS_KEYS})
    model, rev = MODELS["granite"]
    return {"mode": "quality", "arm": phase, "model_tag": "granite", "model": model, "revision": rev, "e4b_sha": E,
            "gnf4_sha": GNF4_SHA, "status": "ok", "fuse_qkv": False, "census": cen, "phase": phase, "arms": list(arms),
            "group": group, "prompt": 512, "cont": C, "chunk": 512, "floor_chunk": 256, "layers": L,
            "windows_sha256": {t: t for t in pb.TEXTS}, "rep_identical": {t: True for t in pb.TEXTS}, "per_window": per,
            "engagement": eng}


def _fake(m, *, census=None, off_census=None, explicit="raised",
          msg="E4B_PAGED_FUSE_QKV=1 matched no attention module -- refusing a vacuous fusion",
          off_tokens2=None, bias=0.003, argmax=0.99, sane_missing=False, e4b=E, rev=None, new=4, sane=True,
          quality=False, qbias=None, qmutant=0.4, qwindows=48, qcont=128, qcensus=None):
    model, r0 = MODELS[m]
    rev = rev or r0
    toks = [[i * 10 + j for j in range(new)] for i in range(16)]
    base = {"model_tag": m, "model": model, "revision": rev, "e4b_sha": e4b, "gnf4_sha": GNF4_SHA}
    recs = {f"serve_{m}_off": {**base, "status": "ok", "census": off_census or {k: (0 if k != "fuse_t1_glue_r2_n" else [0, 0]) for k in CENSUS_KEYS},
                               "tokens": [toks, off_tokens2 or toks]},
            f"serve_{m}_on": {**base, "status": "ok", "census": census or dict(PREDICTED[m]), "tokens": [toks]},
            f"serve_{m}_explicit": {**base, "status": explicit, "exc_type": "RuntimeError" if explicit == "raised" else None,
                                    "exc_message": msg}}
    if sane and not sane_missing:
        recs[f"sane_{m}_off"] = {**base, "status": "ok", "per_window": {"wikitext": {"R": [{"window": w, "nll": 2.0} for w in range(12)]}}}
        recs[f"sane_{m}_on"] = {**base, "status": "ok", "per_window": {"wikitext": {"ON": [
            {"window": w, "nll": 2.0 + bias, "argmax_agree": argmax, "kl": 0.01} for w in range(12)]}}}
    if quality:
        recs[f"quality_{m}_off"] = _fake_quality("off", n=qwindows, C=qcont, mutant=qmutant)
        recs[f"quality_{m}_on"] = _fake_quality("on", n=qwindows, C=qcont, bias=qbias, census=qcensus)
    return recs


def self_test() -> int:
    def both(**over):
        r = {}
        r.update(_fake("gptoss", **over.get("gptoss", {})))
        r.update(_fake("qw36", **{"msg": "E4B_FUSE_T1_GLUE=1 patched no RMSNorm modules (40 name-matched but failed the "
                                         "semantic probe) -- refusing a vacuous enable", **over.get("qw36", {})}))
        r.update(_fake("granite", **{"sane": False, "quality": True, **over.get("granite", {})}))
        return r
    full = lambda recs, **kw: reduce(recs, E, new=4, **kw)  # noqa: E731
    v = lambda recs, **kw: full(recs, **kw)["verdict"]  # noqa: E731
    gv = lambda recs: full(recs)["granite"]["verdict"]  # noqa: E731
    proof = lambda **kw: reduce(_fake("granite", quality=True, qwindows=12, qcont=32, **kw), E, proof=True, new=4)  # noqa: E731
    c4 = both(granite={"qbias": {"c4val1": 0.008}})                 # inside the bias bar, |K8 delta| ~0.12 ppl
    cases = [
        ("licensed", v(both()) == "FLIP_LICENSED"),
        ("census off by one", v(both(gptoss={"census": {**PREDICTED["gptoss"], "fuse_t1_glue_n": 48}})) == "FLIP_HELD"),
        ("off engaged", v(both(qw36={"off_census": dict(PREDICTED["qw36"])})) == "FLIP_HELD"),
        ("explicit built", v(both(gptoss={"explicit": "built"})) == "FLIP_HELD"),
        ("explicit wrong message", v(both(gptoss={"msg": "CUDA out of memory"})) == "FLIP_HELD"),
        ("explicit another feature's refusal", v(both(gptoss={"msg": "enable_serve_experts_int4: the plan holds no expert "
                                                             "tensors -- refusing a vacuous enable"})) == "FLIP_HELD"),
        ("explicit the r2 fold's refusal", v(both(qw36={"msg": "E4B_FUSE_T1_GLUE_R2=1 patched nothing (no structurally "
                                                         "matched decoder layer or fused attention passed the probes) -- "
                                                         "refusing a vacuous enable"})) == "FLIP_LICENSED"),
        ("not deterministic", v(both(qw36={"off_tokens2": [[9] * 4] * 16})) == "FLIP_HELD"),
        ("sane bias", v(both(gptoss={"bias": 0.03})) == "FLIP_HELD"),
        ("sane bias negative", v(both(gptoss={"bias": -0.03})) == "FLIP_HELD"),
        ("sane argmax", v(both(qw36={"argmax": 0.90})) == "FLIP_HELD"),
        ("instrument did not build", v(both(qw36={"sane_missing": True})) == "FLIP_HELD"),
        ("wrong commit", v(both(gptoss={"e4b": "b" * 40})) == "VOID"),
        ("wrong revision", v(both(qw36={"rev": "c" * 40})) == "VOID"),
        ("harness", v({**both(), "harness_qw36": {"reason": "fetch failed"}}) == "VOID"),
        # Granite: Phase B's rule, its own verdict; it never moves the flip's
        ("granite licensed", gv(both()) == "GRANITE_LICENSED"),
        ("granite quality bias", gv(both(granite={"qbias": {"c4val1": 0.02}})) == "GRANITE_HELD"
         and v(both(granite={"qbias": {"c4val1": 0.02}})) == "FLIP_LICENSED"),
        ("granite wikitext K8", gv(both(granite={"qbias": {"wikitext": 0.008}})) == "GRANITE_HELD"
         and "K8" in " ".join(full(both(granite={"qbias": {"wikitext": 0.008}}))["granite"]["reasons"])),
        ("granite c4val1 K8 is reported, not gated", gv(c4) == "GRANITE_LICENSED"
         and abs(full(c4)["models"]["granite"]["quality"]["read"]["c4val1"]["k8"]["delta"]) > pb.K8_BUDGET),
        ("granite mutant passes", gv(both(granite={"qmutant": 0.0001})) == "VOID"),
        ("granite census", gv(both(granite={"qcensus": {**PREDICTED["granite"], "fuse_t1_glue_n": 64}})) == "VOID"),
        ("granite short", gv(both(granite={"qwindows": 36})) == "VOID"),
        ("granite harness", gv({**both(), "harness_granite": {"reason": "fetch failed"}}) == "VOID"
         and v({**both(), "harness_granite": {"reason": "fetch failed"}}) == "FLIP_LICENSED"),
        ("granite explicit built", gv(both(granite={"explicit": "built"})) == "GRANITE_HELD"),
        ("proof", proof()["verdict"] == "FLIP_LICENSED" and set(proof()["models"]["granite"]["gates"])
         == {"SERVED", "ENGAGED", "EXPLICIT_RAISE", "DETERMINISM", "SANE", "QUALITY"}),
        ("proof quality void", proof(qmutant=0.0001)["verdict"] == "VOID"),
        ("proof quality fail", proof(qbias={"wikitext": 0.03})["verdict"] == "FLIP_HELD"),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"p115c_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--proof", action="store_true")
    p.add_argument("--new", type=int, default=32)
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    recs = {}
    for f in sorted(os.listdir(a.dir)):
        if f.endswith(".json") and f.split("_")[0] in RECORD_PREFIXES:
            recs[f[:-5]] = json.load(open(os.path.join(a.dir, f)))
    v = reduce(recs, a.e4b_sha, proof=a.proof, new=a.new)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P115C_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)}")
    if v.get("granite"):
        print(f"P115C_GRANITE {v['granite']['verdict']} {json.dumps(v['granite']['reasons'], default=str)}")
    for m, r in v.get("models", {}).items():
        print(f"  {m}: gates {json.dumps(r['gates'])} sane {json.dumps(r['sane'])} identity {json.dumps(r['identity'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
