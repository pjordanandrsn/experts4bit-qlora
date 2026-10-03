#!/usr/bin/env python3
"""Lane P107's reducer (bench/p107/PREREG-p107.md; e4b#960): paged prefill attention's route A/B and the default it
licenses.

Input in --dir: ``ab_routes.json`` (``p107_box.py ab``), and ``REHEARSAL`` if present.

**VOID** (each reason listed) when:
- the run is a REHEARSAL;
- the record is missing or off the registered shape: the build's census (48 int4 expert layers on ``int4_b32``),
  chunk 512, three rounds, SC1's prompt digests, both routes in every round at both lengths;
- a route did not run as registered in its 512-token kernel census: ``flash`` launched >= 48 flash kernels (one per
  layer per chunk); ``math`` launched none and >= 96 fp32 SGEMMs (its QK^T and PV per layer);
- an NLL window is missing for a route, the routes scored different text, or a window did not score 2,048 tokens.

**QUALITY**, ``flash`` against ``math``, on the SERVED-PREFILL NLL: the calibrated K8 rule (P95's windowed reading).
Per text, the mean over its fresh windows (c4val1 W = 8, wikitext W = 4) of ppl(flash) - ppl(math). PASS iff |mean|
<= 0.05 on both texts (``k8_gate.verdict``, uncalibrated regime). Per-window deltas and their SD are reported.

**SPEED:** each route's median TTFT over the three rounds, at 4096 and 512 tokens.

**DEFAULT:** ``DEFAULT=flash`` iff ``flash`` PASSES and its TTFT-4096 is at most 0.9 x ``math``'s (a >= 10 % gain);
otherwise ``NO_CHANGE``.

    python p107_reduce.py --dir RUN_DIR --out verdict.json
    python p107_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROUTES = ("math", "flash")
PROMPT_SHA = {"512": "a8e6ea1d7d140dbe726c94f0e6325eb11457a93f196419483de79507272f95e3",
              "4096": "cd70a142d533eb88a3d2bec79bc4b11539f3a08b30c20ba5fbba248fb9dbafdd"}
WINDOWS = {"c4val1": (9, 10, 11, 12, 13, 14, 15, 16), "wikitext": (9, 10, 11, 12)}
BUDGET, LAYERS, GAIN = 0.05, 48, 0.9


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _k8_verdict(pairs):
    try:
        from experts4bit_qlora import k8_gate
        arms = [(k8_gate.Arm(b, sha, 2048, src), k8_gate.Arm(c, sha, 2048, src)) for src, b, c, sha in pairs]
        return k8_gate.verdict(arms, calibrated=False, budget=BUDGET)
    except ImportError:
        ok = all(abs(c - b) <= BUDGET for _s, b, c, _h in pairs)
        return ok, [f"K8 {s}: base={b:.5f} cand={c:.5f} delta={c - b:+.5f}" for s, b, c, _h in pairs]


def reduce(d: Path) -> dict:
    void = []
    if (d / "REHEARSAL").exists():
        void.append("REHEARSAL")
    t = _load(d / "ab_routes.json")
    out = {"lane": "P107", "void_reasons": void}
    if t is None:
        void.append("no ab_routes.json")
        out["verdict"] = "VOID"
        return out
    cb = t.get("census_build") or {}
    if cb.get("int4_expert_layers") != LAYERS or cb.get("int4_store_kinds") != ["int4_b32"]:
        void.append(f"build: int4 store not engaged ({cb.get('int4_expert_layers')}, {cb.get('int4_store_kinds')})")
    if t.get("chunk_tokens") != 512 or t.get("rounds") != 3:
        void.append(f"ab: chunk {t.get('chunk_tokens')} rounds {t.get('rounds')}")
    if {str(k): v for k, v in (t.get("prompts_sha256") or {}).items()} != PROMPT_SHA:
        void.append("ab: prompts are not SC1's")
    draws = t.get("draws") or []
    for rnd in range(3):
        for rt in ROUTES:
            for L in (512, 4096):
                if not any(x["round"] == rnd and x["route"] == rt and x["L"] == L for x in draws):
                    void.append(f"ab: no draw round {rnd} {rt} {L}")
    k = t.get("kernels") or {}
    km, kf = k.get("math") or {}, k.get("flash") or {}
    if not (kf.get("flash", 0) >= LAYERS):
        void.append(f"flash: did not run as registered (kernels {kf})")
    if not (km.get("flash", 1) == 0 and km.get("sgemm", 0) >= 2 * LAYERS):
        void.append(f"math: did not run as registered (kernels {km})")
    by = {(w["source"], w["k"], w["route"]): w for w in t.get("windows") or []}
    for src, ks in WINDOWS.items():
        for kk in ks:
            ws = [by.get((src, kk, rt)) for rt in ROUTES]
            if None in ws:
                void.append(f"nll {src} k={kk}: a route is missing")
            elif len({w["text_sha"] for w in ws}) != 1 or any(w["steps"] != 2048 for w in ws):
                void.append(f"nll {src} k={kk}: routes scored different text or steps")
    if void:
        out["verdict"] = "VOID"
        return out
    med = t["ttft_s_median"]
    out["ttft_s_median"] = med
    out["int4_prefill_route"] = t.get("int4_prefill_route")
    pairs, per = [], {}
    for src, ks in WINDOWS.items():
        b = [by[(src, kk, "math")]["ppl"] for kk in ks]
        c = [by[(src, kk, "flash")]["ppl"] for kk in ks]
        deltas = [ci - bi for bi, ci in zip(b, c)]
        sha = hashlib.sha256("".join(by[(src, kk, "math")]["text_sha"] for kk in ks).encode()).hexdigest()
        pairs.append((src, statistics.mean(b), statistics.mean(c), sha))
        per[src] = {"deltas": [round(x, 5) for x in deltas], "mean": round(statistics.mean(deltas), 5),
                    "sd": round(statistics.stdev(deltas), 5) if len(deltas) > 1 else None,
                    "windows_over_budget": sum(abs(x) > BUDGET for x in deltas),
                    "mean_ppl_math": round(statistics.mean(b), 5)}
    ok, lines = _k8_verdict(pairs)
    out["quality"] = {"verdict": "PASS" if ok else "QUALITY_FAIL", "per_text": per, "k8_lines": lines}
    toks = {}
    for x in draws:
        toks.setdefault((x["round"], x["L"]), {})[x["route"]] = x["token"]
    out["first_tokens_identical"] = all(v.get("math") == v.get("flash") for v in toks.values())
    out["speedup_4096"] = round(med["math"]["4096"] / med["flash"]["4096"], 3)
    out["speedup_512"] = round(med["math"]["512"] / med["flash"]["512"], 3)
    out["verdict"] = "DEFAULT=flash" if ok and med["flash"]["4096"] <= GAIN * med["math"]["4096"] else "NO_CHANGE"
    return out


# ---------------------------------------------------------------------------------------------------- self-test
def _fixture(d: Path, ttft=None, delta=0.004, kern=None, drop=None):
    ttft = ttft or {"math": (0.113, 1.373), "flash": (0.095, 0.95)}
    kern = kern or {"math": {"flash": 0, "sgemm": 96}, "flash": {"flash": 48, "sgemm": 0}}
    draws = [{"route": rt, "L": L, "round": r, "wall_s": ttft[rt][i], "token": 7}
             for r in range(3) for rt in ROUTES for i, L in enumerate((512, 4096))]
    ws = []
    for src, ks in WINDOWS.items():
        base = 14.0 if src == "c4val1" else 9.0
        for kk in ks:
            for rt in ROUTES:
                v = base + (delta * (1 if kk % 2 else 0.5) if rt == "flash" else 0.0)
                ws.append({"source": src, "k": kk, "route": rt, "mean_nll": v, "ppl": v, "text_sha": f"{src}{kk}",
                           "steps": 2048})
    if drop:
        ws = [w for w in ws if not drop(w)]
    t = {"census_build": {"int4_expert_layers": 48, "int4_store_kinds": ["int4_b32"]}, "chunk_tokens": 512, "rounds": 3,
         "prompts_sha256": dict(PROMPT_SHA), "draws": draws, "kernels": kern, "windows": ws, "int4_prefill_route": "k19",
         "ttft_s_median": {rt: {"512": ttft[rt][0], "4096": ttft[rt][1]} for rt in ROUTES}}
    (d / "ab_routes.json").write_text(json.dumps(t))


def selftest() -> int:
    import tempfile
    cases = []

    def case(label, want, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            dd = Path(tmp)
            _fixture(dd, **kw)
            r = reduce(dd)
            assert r["verdict"] == want, (label, r)
            cases.append(label)
            return r

    case("flash passes and is 30 % faster", "DEFAULT=flash")
    case("flash fails quality", "NO_CHANGE", delta=0.09)
    case("flash only 5 % faster", "NO_CHANGE", ttft={"math": (0.113, 1.373), "flash": (0.11, 1.31)})
    case("flash never reached the flash kernel: VOID", "VOID",
         kern={"math": {"flash": 0, "sgemm": 96}, "flash": {"flash": 0, "sgemm": 96}})
    case("math reached a flash kernel: VOID", "VOID", kern={"math": {"flash": 48, "sgemm": 0}, "flash": {"flash": 48, "sgemm": 0}})
    case("a window missing: VOID", "VOID", drop=lambda w: w["source"] == "wikitext" and w["k"] == 11 and w["route"] == "flash")
    with tempfile.TemporaryDirectory() as tmp:
        dd = Path(tmp)
        _fixture(dd)
        (dd / "REHEARSAL").write_text("")
        assert reduce(dd)["verdict"] == "VOID"
        cases.append("rehearsal: VOID")
    print(f"self-test OK ({len(cases)} cases)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return selftest()
    r = reduce(Path(a.dir))
    Path(a.out).write_text(json.dumps(r, indent=1, default=str))
    print("P107_VERDICT " + json.dumps({k: r.get(k) for k in ("verdict", "ttft_s_median", "speedup_4096", "quality",
                                                              "void_reasons")}, default=str)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
