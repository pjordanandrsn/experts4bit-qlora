#!/usr/bin/env python3
"""Lane P82's reducer (bench/p82/PREREG-p82.md; e4b#511, e4b#674, e4b#777). Derived from bench/p81/p81_reduce.py.
Reads the five arm receipts (A1, B1, B2, A2 = eager/graph/graph/eager, P = the bucket step run eagerly), every arm on
the licensed int4 stack with both packs installed by fingerprint, the device grouping and the fp32 router
(E4B_ROUTER_EPI_CAST=0). The decisions are P81's plus two gates:
  - identity: the eager runner equals its bucket step (A1 == P) as well as the graph replay (B == P) -- after #777
    the graph path computes the eager function, so any difference is a bug, not rounding (B771b showed it on NF4);
  - router: every arm ran the fp32 router weights the licensed build used.
The build's wikitext K8 is read from build_ppl_wikitext.json and REPORTED against the licensed build's 6.36709 and
P81's 6.33015 (e4b#674); it is not decisive.

    python p82_reduce.py --dir <dir> --out verdict.json [--licensed-fp sha256:...]
    python p82_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BAND = 1.03                     # the p37 self-pair band the memo registered
ARMS = ("A1", "B1", "B2", "A2", "P")
MODES = {"A1": "eager", "A2": "eager", "B1": "graph", "B2": "graph", "P": "padded"}


def _first_divergence(x: dict, y: dict):
    for row in sorted(x, key=int):
        a, b = x[row], y.get(row, [])
        for i, (u, v) in enumerate(zip(a, b)):
            if u != v:
                return {"row": int(row), "token_index": i}
        if len(a) != len(b):
            return {"row": int(row), "token_index": min(len(a), len(b)), "length_differs": True}
    return None


def reduce(recs: dict) -> dict:
    """``recs`` maps arm -> receipt dict (or a dict with "status" when the arm did not produce one)."""
    v: dict = {"lane": "P82", "band": BAND, "reasons": [], "verdict": None}
    missing = [k for k in ARMS if not isinstance(recs.get(k), dict) or "timed_agg_tok_s" not in recs[k]]
    oom_b = [k for k in ("B1", "B2") if isinstance(recs.get(k), dict) and recs[k].get("oom")]
    if oom_b:
        v["verdict"] = "REFUTED"
        v["reasons"].append(f"candidate arm(s) {oom_b} OOM (the memo's failure criterion)")
        return v
    if missing:
        v["verdict"] = "VOID"
        v["reasons"].append(f"arm(s) without a receipt: {missing} (NOT_RUN/VOID, neither confirms nor refutes)")
        return v
    # engagement: each arm ran its declared path, over the registered trace, in both passes
    for k in ARMS:
        r = recs[k]
        if r.get("mode") != MODES[k]:
            v["verdict"] = "VOID"
            v["reasons"].append(f"{k} ran mode {r.get('mode')!r}, registered {MODES[k]!r}")
            return v
        if not r.get("warm_equals_timed_tokens"):
            v["reasons"].append(f"{k}: warm and timed passes decoded different tokens (non-determinism)")
    trace = recs["A1"]["trace"]
    if any(recs[k]["trace"] != trace for k in ARMS):
        v["verdict"] = "VOID"
        v["reasons"].append("arms disagree on the trace")
        return v
    # the pack gate: one expert pack and one attention pack, each installed from its artifact, in every arm
    fps = {k: (recs[k].get("pack_fingerprint"), recs[k].get("attn_pack_fingerprint"),
               recs[k].get("attn_pack_source")) for k in ARMS}
    v["packs"] = {"expert": fps["A1"][0], "attention": fps["A1"][1]}
    if any(f[0] is None or f[1] is None for f in fps.values()) or len({f[:2] for f in fps.values()}) != 1 \
            or any(f[2] != "artifact" for f in fps.values()):
        v["verdict"] = "VOID"
        v["reasons"].append(f"the arms did not all serve the same packs from their artifacts: {fps}")
        return v
    # every arm, the eager control included, served T > 1 through the device grouping (the licensed stack's
    # batched configuration; with it off the int4 store decodes T > 1 through the prefill branch)
    off = [k for k in ARMS if recs[k].get("device_grouping") is not True]
    if off:
        v["verdict"] = "VOID"
        v["reasons"].append(f"arm(s) {off} did not run the device grouping")
        return v
    # every arm ran the licensed build's fp32 router weights (E4B_ROUTER_EPI_CAST=0, stamped by the runner)
    notfp32 = [k for k in ARMS if recs[k].get("router_epi_cast_weights") is not False]
    if notfp32:
        v["verdict"] = "VOID"
        v["reasons"].append(f"arm(s) {notfp32} did not run the fp32 router weights (router_epi_cast_weights != False)")
        return v
    for k in ("B1", "B2"):
        st = recs[k].get("graph_status") or {}
        stats = recs[k].get("graph_stats") or {}
        bad = {b: s for b, s in st.items() if s != "graph"}
        eager = sum(int(s.get("eager_steps", 0)) for s in stats.values())
        if bad or eager or not st:
            v["verdict"] = "REFUTED"
            v["reasons"].append(f"{k}: a bucket fell back to eager (status {bad or st}, eager_steps {eager}) -- "
                                "a required eager fallback refutes (memo)")
            return v
    if recs["P"].get("graph_status") and any(not str(s).startswith("eager") for s in recs["P"]["graph_status"].values()):
        v["verdict"] = "VOID"
        v["reasons"].append("P captured graphs; it must run the padded step eagerly")
        return v
    for k in ("A1", "A2"):
        if recs[k].get("graph_status") is not None:
            v["verdict"] = "VOID"
            v["reasons"].append(f"{k} carried graph state; the control must be the unpadded eager path")
            return v
    tok = {k: recs[k]["passes"][1]["tokens"] for k in ARMS}
    # identity: the candidate against its bitwise oracle, and each self-pair
    v["identity"] = {
        "B1_vs_P": _first_divergence(tok["B1"], tok["P"]),
        "B2_vs_P": _first_divergence(tok["B2"], tok["P"]),
        "B1_vs_B2": _first_divergence(tok["B1"], tok["B2"]),
        "A1_vs_A2": _first_divergence(tok["A1"], tok["A2"]),
        # decisive in P82 (B771b): with #777 the eager runner and its bucket step compute one function
        "A1_vs_P": _first_divergence(tok["A1"], tok["P"]),
    }
    idn = v["identity"]
    if idn["A1_vs_A2"] is not None:
        v["verdict"] = "VOID"
        v["reasons"].append(f"the control does not repeat itself: A1 != A2 at {idn['A1_vs_A2']}")
        return v
    bad = {k: idn[k] for k in ("B1_vs_P", "B2_vs_P", "B1_vs_B2", "A1_vs_P") if idn[k] is not None}
    if bad:
        v["verdict"] = "REFUTED"
        v["reasons"].append(f"the eager runner, the bucket step and the graph replay do not decode one function: {bad}")
        return v
    agg = {k: float(recs[k]["timed_agg_tok_s"]) for k in ARMS}
    v["agg_tok_s"] = agg
    r = {"A2/A1": agg["A2"] / agg["A1"], "B2/B1": agg["B2"] / agg["B1"],
         "B1/A1": agg["B1"] / agg["A1"], "B2/A2": agg["B2"] / agg["A2"],
         "P/A1": agg["P"] / agg["A1"], "B1/P": agg["B1"] / agg["P"]}
    v["ratios"] = r
    v["step_ms_by_active"] = {k: recs[k]["timed_mean_step_ms_by_active"] for k in ARMS}
    for sp in ("A2/A1", "B2/B1"):
        if not (1 / BAND <= r[sp] <= BAND):
            v["verdict"] = "VOID"
            v["reasons"].append(f"self-pair {sp} = {r[sp]:.4f} outside [{1 / BAND:.4f}, {BAND}] (drift)")
            return v
    if r["B1/A1"] > BAND and r["B2/A2"] > BAND:
        v["verdict"] = "CONFIRMED"
        v["reasons"].append(f"both pairings exceed {BAND}: B1/A1 {r['B1/A1']:.4f}, B2/A2 {r['B2/A2']:.4f}")
    else:
        v["verdict"] = "REFUTED"
        v["reasons"].append(f"a pairing at or below {BAND}: B1/A1 {r['B1/A1']:.4f}, B2/A2 {r['B2/A2']:.4f}")
    return v


def _sidecar(f: Path) -> dict:
    """The runner's router stamp beside a receipt (<receipt>.router.json), or {} when there is none."""
    g = f.with_suffix(".router.json")
    try:
        return json.loads(g.read_text()) if g.is_file() else {}
    except json.JSONDecodeError:
        return {}


def _load(d: Path) -> dict:
    out = {}
    for k in ARMS:
        f = d / f"p82_{k}.json"
        if f.is_file():
            try:
                out[k] = json.loads(f.read_text())
            except json.JSONDecodeError as e:
                out[k] = {"status": f"unreadable: {e}"}
                continue
            if isinstance(out[k], dict):
                # the stamp never overrides a field the harness wrote
                out[k].update({x: y for x, y in _sidecar(f).items() if x not in out[k]})
    return out


def _synthetic(agg: dict, *, tokens_b=None, tokens_a=None, status_b="graph", eager_b=0):
    trace = [16] * 2 + [8] * 2
    base = {"0": [1, 2, 3], "1": [4, 5]}
    recs = {}
    for k in ARMS:
        mode = MODES[k]
        toks = (tokens_b if (k.startswith("B") and tokens_b) else tokens_a if (k.startswith("A") and tokens_a) else base)
        st = ({b: status_b for b in ("1", "2", "4")} if mode == "graph"
              else {b: "eager: capture=False" for b in ("1", "2", "4")} if mode == "padded" else None)
        stats = ({b: {"eager_steps": eager_b} for b in ("1", "2", "4")} if mode == "graph" else None)
        recs[k] = {"mode": mode, "trace": trace, "warm_equals_timed_tokens": True, "graph_status": st,
                   "pack_fingerprint": "sha256:" + "e" * 64, "attn_pack_fingerprint": "sha256:" + "a" * 64,
                   "attn_pack_source": "artifact", "device_grouping": True, "router_epi_cast_weights": False,
                   "graph_stats": stats, "timed_agg_tok_s": agg[k], "timed_mean_step_ms_by_active": {"16": 1.0},
                   "passes": [{"tokens": toks}, {"tokens": toks}]}
    return recs


def self_test() -> None:
    ok = {"A1": 100.0, "A2": 101.0, "B1": 150.0, "B2": 151.0, "P": 95.0}
    assert reduce(_synthetic(ok))["verdict"] == "CONFIRMED"
    slow = dict(ok, B1=102.0, B2=103.0)
    assert reduce(_synthetic(slow))["verdict"] == "REFUTED"
    drift = dict(ok, A2=110.0)
    assert reduce(_synthetic(drift))["verdict"] == "VOID"
    # the candidate must decode as its oracle: a B-only token change refutes even when faster
    diverge = _synthetic(ok, tokens_b={"0": [1, 2, 9], "1": [4, 5]})
    got = reduce(diverge)
    assert got["verdict"] == "REFUTED" and got["identity"]["B1_vs_P"] == {"row": 0, "token_index": 2}, got
    # a fallback refutes
    assert reduce(_synthetic(ok, status_b="eager: RuntimeError: x"))["verdict"] == "REFUTED"
    assert reduce(_synthetic(ok, eager_b=3))["verdict"] == "REFUTED"
    # a missing arm is VOID; a candidate OOM refutes
    m = _synthetic(ok)
    del m["P"]
    assert reduce(m)["verdict"] == "VOID"
    o = _synthetic(ok)
    o["B1"] = {"oom": True, "status": "error"}
    assert reduce(o)["verdict"] == "REFUTED"
    # P82's pack gate: a different attention pack in one arm, or a calibrated (not loaded) one, voids the read
    q = _synthetic(ok)
    q["B2"]["attn_pack_fingerprint"] = "sha256:" + "b" * 64
    assert reduce(q)["verdict"] == "VOID"
    q = _synthetic(ok)
    q["A1"]["attn_pack_source"] = "calibrated-and-dumped"
    assert reduce(q)["verdict"] == "VOID"
    q = _synthetic(ok)
    del q["P"]["pack_fingerprint"]
    assert reduce(q)["verdict"] == "VOID"
    # the eager control on the grouping-off path (the int4 store's prefill branch) is not the registered control
    q = _synthetic(ok)
    q["A2"]["device_grouping"] = False
    assert reduce(q)["verdict"] == "VOID"
    # P82: the eager runner must equal its bucket step, and every arm must run the fp32 router
    got = reduce(_synthetic(ok, tokens_a={"0": [1, 2, 7], "1": [4, 5]}))
    assert got["verdict"] == "REFUTED" and got["identity"]["A1_vs_P"] == {"row": 0, "token_index": 2}, got
    q = _synthetic(ok)
    q["B1"]["router_epi_cast_weights"] = None
    assert reduce(q)["verdict"] == "VOID"
    # the runner's router stamp is merged from the sidecar and never overrides a field the harness wrote
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        for k, rec in _synthetic(ok).items():
            rec.pop("router_epi_cast_weights")
            if k == "P":
                rec["router_epi_cast_weights"] = "harness"
            (d / f"p82_{k}.json").write_text(json.dumps(rec))
            (d / f"p82_{k}.router.json").write_text(json.dumps({"router_epi_cast_weights": False}))
        got = _load(d)
        assert got["A1"]["router_epi_cast_weights"] is False and got["P"]["router_epi_cast_weights"] == "harness", got
    # PREREG-p82's K8 table (e4b#674; reported): the four readings
    def k(ppl, cast_w, casts, sha=K8_WINDOW + "f" * 52):
        return {"ppl": ppl, "ppl_5dp": round(ppl, 5), "same_window": sha.startswith(K8_WINDOW),
                "router_epi_cast_weights": cast_w, "router_epi_casts_softmax_topk": casts}
    b32, k32, k16 = k(6.367091, False, False), k(6.367091, False, False), k(6.330147, None, True)
    assert k8_reading(b32, k32, k16).startswith("THE CAST IS THE GAP")
    assert k8_reading(b32, k32, k(6.34, None, True)).startswith("THE FP32 ROUTER READS THE LICENSED K8")
    assert k8_reading(k(6.35, False, False), k(6.35, False, False), k16).startswith("THE CAST IS NOT THE WHOLE GAP")
    assert k8_reading(b32, k(6.367091, None, True), k16).startswith("INCOMPLETE")
    assert k8_reading(b32, k32, k(6.330147, None, True, sha="0" * 64)).startswith("INCOMPLETE")
    assert k8_reading(k(6.367091, None, True), k32, k16).startswith("INCOMPLETE")
    print("p82_reduce self-test OK (21 cases)")


LICENSED_K8 = 6.36709    # P55x / P64 / P70's build, wikitext K8 at 2048 steps, window sha 9ef10d760ad9 (fp32 router)
P81_K8 = 6.33015         # P81's build on e4b 0.37.7, same window, the 0.37.5 router-cast default
K8_WINDOW = "9ef10d760ad9"
P81_ATTN_FP = "sha256:d7cfa1f496d75120ef50b72fae0a31928b6ffb04ac0c5ec49c522a673068a422"   # P81's attention pack


def _k8_one(f: Path) -> dict:
    if not f.is_file():
        return {"ppl": None, "note": f"no {f.name}"}
    try:
        r = json.loads(f.read_text())
        ppl = float(r["ppl"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return {"ppl": None, "note": f"{f.name} unreadable: {e}"}
    st = _sidecar(f)
    return {"ppl": ppl, "ppl_5dp": round(ppl, 5), "text_sha": r.get("text_sha"),
            "same_window": str(r.get("text_sha", "")).startswith(K8_WINDOW),
            "delta_vs_licensed": ppl - LICENSED_K8, "delta_vs_p81": ppl - P81_K8,
            "router_epi_cast_env": st.get("router_epi_cast_env"),
            "router_epi_cast_weights": st.get("router_epi_cast_weights"),
            "router_epi_casts_softmax_topk": st.get("router_epi_casts_softmax_topk")}


def k8_reading(b: dict, k32: dict, k16: dict) -> str:
    """PREREG-p82's table for e4b#674 (reported, never decisive). Five decimals, as every earlier K8 was reported."""
    if any(x.get("ppl") is None or not x.get("same_window") for x in (b, k32, k16)):
        return "INCOMPLETE: a K8 reading is missing or off the registered window"
    if (b.get("router_epi_cast_weights") is not False or k32.get("router_epi_cast_weights") is not False
            or k16.get("router_epi_casts_softmax_topk") is not True):
        return "INCOMPLETE: the build or K32 did not run the fp32 router, or K16 did not run the cast"
    lic = b["ppl_5dp"] == LICENSED_K8 and k32["ppl_5dp"] == LICENSED_K8
    if lic and k16["ppl_5dp"] == P81_K8:
        return "THE CAST IS THE GAP: the fp32 router reads the licensed 6.36709 and the cast P81's 6.33015, on the same packs"
    if lic:
        return "THE FP32 ROUTER READS THE LICENSED K8, but the cast does not read P81's here"
    return "THE CAST IS NOT THE WHOLE GAP: the fp32 router does not read the licensed 6.36709"


def k8_report(d: Path, attention_fp=None) -> dict:
    """The build's K8 and the two pack-loaded K8 arms against the licensed and P81 readings (e4b#674)."""
    b, k32, k16 = _k8_one(d / "build_ppl_wikitext.json"), _k8_one(d / "k8_K32.json"), _k8_one(d / "k8_K16.json")
    out = {"build": b, "K32": k32, "K16": k16, "reading": k8_reading(b, k32, k16)}
    if k32.get("ppl") is not None and b.get("ppl") is not None:
        out["K32_equals_build_exactly"] = k32["ppl"] == b["ppl"]
    if k32.get("ppl") is not None and k16.get("ppl") is not None:
        out["cast_effect_K16_minus_K32"] = k16["ppl"] - k32["ppl"]
    if attention_fp is not None:
        out["attention_pack"] = attention_fp
        out["attention_pack_equals_p81"] = attention_fp == P81_ATTN_FP
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--licensed-fp")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if not a.dir or not a.out:
        ap.error("--dir and --out are required")
    v = reduce(_load(Path(a.dir)))
    if a.licensed_fp and v.get("packs"):
        v["expert_pack_is_licensed"] = v["packs"]["expert"] == a.licensed_fp
    v["k8_report"] = k8_report(Path(a.dir), (v.get("packs") or {}).get("attention"))
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(f"P82_VERDICT {v['verdict']} licensed={v.get('expert_pack_is_licensed')} " + " | ".join(v["reasons"]))
    k8 = v["k8_report"]
    for name in ("build", "K32", "K16"):
        x = k8[name]
        if x.get("ppl") is not None:
            print(f"  K8 {name} (e4b#674, reported): {x['ppl']:.5f} on window {str(x.get('text_sha', ''))[:12]}, "
                  f"router cast env {x.get('router_epi_cast_env')!r}; licensed {LICENSED_K8} ({x['delta_vs_licensed']:+.5f}), "
                  f"P81 {P81_K8} ({x['delta_vs_p81']:+.5f})")
    print(f"  K8 reading: {k8['reading']}; attention pack = P81's: {k8.get('attention_pack_equals_p81')}")
    for k, x in (v.get("ratios") or {}).items():
        print(f"  {k} = {x:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
