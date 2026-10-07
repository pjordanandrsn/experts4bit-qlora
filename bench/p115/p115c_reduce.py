#!/usr/bin/env python3
"""p115c_reduce.py -- lane P115 Phase C's registered rule (bench/p115/PREREG-p115.md, Amendment 2; e4b#1313).

Receipts, per model tag M: ``serve_M_off.json``, ``serve_M_on.json``, ``serve_M_explicit.json``, ``sane_M_off.json``,
``sane_M_on.json`` (``p115c_box.py``), and ``harness_M.json`` when the runner could not fetch, bake or prompt M.

Per model, the gates:
  SERVED          the off and on servers built and decoded every prompt to its length (off twice).
  ENGAGED         the on census is the registered prediction (``PREDICTED``); the off census is all zero.
  EXPLICIT_RAISE  all four knobs at 1 refused at build with one of the four knobs' own vacuous-enable refusals
                  (``EXPLICIT_RE``: the message names the knob at ``=1``; another feature's refusal does not count).
  DETERMINISM     the off server's two passes emit the same tokens.
  SANE            the maintainer's gross-error gate (59b79fb7): ON scored against R on ``SANE_WINDOWS`` wikitext windows,
                  |mean d_ON| <= 0.02 nats and mean argmax agreement >= 0.95. A model the instrument cannot build
                  (either SANE record missing or not ok) fails it.
Reported, not gated: IDENTITY (on tokens against off's), the folds' reports, the SANE KL.

Verdict:
  VOID           a record names another e4b / grouped-nf4-gemm commit or another model revision; or the harness could not
                 fetch, bake or prompt a model (``harness_M.json``).
  FLIP_LICENSED  every registered model passes every gate.
  FLIP_HELD      otherwise (the failing gates named).

    python p115c_reduce.py --dir RUN_DIR --out verdict_c.json [--e4b-sha SHA] [--proof]
    python p115c_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

GNF4_SHA = "b4f93f1c62d1e3436ed45bec8ccd608c90433737"
MODELS = {"gptoss": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
          "qw36": ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
          "granite": ("ibm-granite/granite-3.1-3b-a800m-instruct", "a02780686e08a03fe0d2679a293b5c74a90efa89")}
READING, PROOF = ("gptoss", "qw36"), ("granite",)
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


def _zero(v):
    return all(x == 0 for x in v) if isinstance(v, (list, tuple)) else v == 0


def model_gates(m, recs, *, new):
    so, sn, sx = recs.get(f"serve_{m}_off"), recs.get(f"serve_{m}_on"), recs.get(f"serve_{m}_explicit")
    qo, qn = recs.get(f"sane_{m}_off"), recs.get(f"sane_{m}_on")
    g, why = {}, {}
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
    sane_ok, stats = False, None
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
    ident = None
    if ok_serve:
        a, b = sn["tokens"][0], so["tokens"][0]
        ident = {"rows_identical": sum(x == y for x, y in zip(a, b)), "rows": len(a)}
    return g, why, {"sane": stats, "identity": ident, "fusion_report": (sn or {}).get("fusion_report"),
                    "explicit_message": msg[:300]}


def reduce(recs: dict, e4b_sha: str, *, proof=False, new=32) -> dict:
    models = PROOF if proof else READING
    out = {"lane": "P115 Phase C", "proof": proof, "models": {}, "verdict": None, "reasons": [],
           "bars": {"sane_bias": SANE_BIAS, "sane_argmax": SANE_ARGMAX, "sane_windows": SANE_WINDOWS}}
    void = []
    for m in models:
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
    for m in models:
        g, why, rep = model_gates(m, recs, new=new)
        out["models"][m] = {"gates": g, "why": why, **rep}
        held += [f"{m}: {k} -- {why[k]}" for k, v in g.items() if not v]
    if held:
        out.update(verdict="FLIP_HELD", reasons=held)
    else:
        out.update(verdict="FLIP_LICENSED", reasons=[f"{m}: every gate passed (census {PREDICTED[m]}; SANE "
                                                     f"{out['models'][m]['sane']})" for m in models])
    return out


# ------------------------------------------------------------------ self-test --

E = "a" * 40


def _fake(m, *, census=None, off_census=None, explicit="raised", msg="E4B_PAGED_FUSE_QKV=1 matched no attention module -- refusing a vacuous fusion",
          off_tokens2=None, bias=0.003, argmax=0.99, sane_missing=False, e4b=E, rev=None, new=4):
    model, r0 = MODELS[m]
    rev = rev or r0
    toks = [[i * 10 + j for j in range(new)] for i in range(16)]
    base = {"model_tag": m, "model": model, "revision": rev, "e4b_sha": e4b, "gnf4_sha": GNF4_SHA}
    recs = {f"serve_{m}_off": {**base, "status": "ok", "census": off_census or {k: (0 if k != "fuse_t1_glue_r2_n" else [0, 0]) for k in CENSUS_KEYS},
                               "tokens": [toks, off_tokens2 or toks]},
            f"serve_{m}_on": {**base, "status": "ok", "census": census or dict(PREDICTED[m]), "tokens": [toks]},
            f"serve_{m}_explicit": {**base, "status": explicit, "exc_type": "RuntimeError" if explicit == "raised" else None,
                                    "exc_message": msg}}
    if not sane_missing:
        recs[f"sane_{m}_off"] = {**base, "status": "ok", "per_window": {"wikitext": {"R": [{"window": w, "nll": 2.0} for w in range(12)]}}}
        recs[f"sane_{m}_on"] = {**base, "status": "ok", "per_window": {"wikitext": {"ON": [
            {"window": w, "nll": 2.0 + bias, "argmax_agree": argmax, "kl": 0.01} for w in range(12)]}}}
    return recs


def self_test() -> int:
    def both(**over):
        r = {}
        r.update(_fake("gptoss", **over.get("gptoss", {})))
        r.update(_fake("qw36", **{"msg": "E4B_FUSE_T1_GLUE=1 patched no RMSNorm modules (40 name-matched but failed the "
                                         "semantic probe) -- refusing a vacuous enable", **over.get("qw36", {})}))
        return r
    v = lambda recs, **kw: reduce(recs, E, new=4, **kw)["verdict"]  # noqa: E731
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
        ("proof", reduce(_fake("granite"), E, proof=True, new=4)["verdict"] == "FLIP_LICENSED"),
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
        if f.endswith(".json") and f.split("_")[0] in ("serve", "sane", "harness"):
            recs[f[:-5]] = json.load(open(os.path.join(a.dir, f)))
    v = reduce(recs, a.e4b_sha, proof=a.proof, new=a.new)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P115C_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)}")
    for m, r in v.get("models", {}).items():
        print(f"  {m}: gates {json.dumps(r['gates'])} sane {json.dumps(r['sane'])} identity {json.dumps(r['identity'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
