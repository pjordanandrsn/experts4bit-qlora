#!/usr/bin/env python3
"""Lane P102's reducer (bench/p102/PREREG-p102.md; e4b#916): the int4 store's prefill route A/B and the default it
licenses.

Inputs in --dir: ``ttft_routes.json`` and ``nll_routes.json`` (``p102_box.py``'s two modes), ``REHEARSAL`` if present.

**VOID** (each reason listed) when:
- the run is a REHEARSAL;
- a record is missing or off the registered shape (the build's census: 48 int4 expert layers on the int4_b32 store,
  ``device_grouping`` False; chunk 512; three rounds; SC1's prompt digests);
- a route did not run as registered in its dispatch census of one 512-token request:
  - ``loop``: 48 grouped calls, all the host loop, exactly two reference decodes per distinct expert per call, and at
    least 3,072 of them (>= 32 distinct experts per layer -- a floor for "the loop decoded experts"; P100 and P102's
    first reading both measured 8,390, 87 per layer. A2: the registered 9,600 came from a uniform-routing guess and
    voided p102-5090-5);
  - ``batched``: 48 host-grouped calls on the int4 store, 0 reference decodes;
  - ``k19``: 48 device-grouped calls, 0 reference decodes, >= 96 K19 launches, 0 M-tile launches;
  - ``mtile``: 48 device-grouped calls, 0 reference decodes, >= 96 M-tile launches, 0 K19 launches;
- an NLL window is missing for a route, its routes scored different text, or a step count is not 2048.

**QUALITY** per route against ``loop``:
- ``batched`` PASS iff its ``mean_nll`` equals loop's EXACTLY on all 12 windows and its first token equals loop's in every
  draw (the bit-identity claim); else QUALITY_FAIL.
- ``k19`` / ``mtile``: the calibrated K8 rule on the prefill-shaped NLL (P95/P96's windowed reading). Per text, the mean
  over its fresh windows (c4val1 W = 8, wikitext W = 4) of ppl(route) - ppl(loop); PASS iff |mean| <= 0.05 on both
  texts (``k8_gate.verdict``, uncalibrated regime: the RTN int4 store). Per-window deltas and their SD are reported.

**SPEED:** each route's median TTFT over three interleaved rounds, at 4096 and at 512 tokens (chunk 512).

**DEFAULT:** among the routes that PASS, the one with the lowest TTFT-4096; a PASSing route within 10 % of it with
better numerics (batched, then k19, then mtile) is taken instead. ``E4B_INT4_PREFILL``'s default becomes it iff its
TTFT-4096 is at most half of loop's (``DEFAULT=<route>``); otherwise ``NO_CHANGE`` (loop stays).

    python p102_reduce.py --dir RUN_DIR --out verdict.json
    python p102_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROUTES = ("loop", "batched", "k19", "mtile")
NUMERICS_RANK = {"batched": 0, "k19": 1, "mtile": 2}
PROMPT_SHA = {"512": "a8e6ea1d7d140dbe726c94f0e6325eb11457a93f196419483de79507272f95e3",
              "4096": "cd70a142d533eb88a3d2bec79bc4b11539f3a08b30c20ba5fbba248fb9dbafdd"}
WINDOWS = {"c4val1": (9, 10, 11, 12, 13, 14, 15, 16), "wikitext": (9, 10, 11, 12)}
BUDGET, LAYERS, MIN_KERNELS = 0.05, 48, 96
#: A2: the loop's engagement floor -- two decodes per distinct expert, >= 32 distinct experts per layer, 48 layers
MIN_LOOP_DEQUANT = 2 * 32 * 48


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _k8_verdict(pairs):
    """``k8_gate.verdict`` (uncalibrated) when the package is importable, else its rule restated (|delta| <= budget on
    every text) -- the box always has the package; the CI self-test may not."""
    try:
        from experts4bit_qlora import k8_gate
        arms = [(k8_gate.Arm(b, sha, 2048, src), k8_gate.Arm(c, sha, 2048, src)) for src, b, c, sha in pairs]
        ok, lines = k8_gate.verdict(arms, calibrated=False, budget=BUDGET)
        return ok, lines
    except ImportError:
        ok = all(abs(c - b) <= BUDGET for _s, b, c, _h in pairs)
        return ok, [f"K8 {s}: base={b:.5f} cand={c:.5f} delta={c - b:+.5f}" for s, b, c, _h in pairs]


def reduce(d: Path) -> dict:
    void = []
    if (d / "REHEARSAL").exists():
        void.append("REHEARSAL")
    t, n = _load(d / "ttft_routes.json"), _load(d / "nll_routes.json")
    if t is None:
        void.append("no ttft_routes.json")
    if n is None:
        void.append("no nll_routes.json")
    out = {"lane": "P102", "void_reasons": void}
    if t is not None:
        cb = t.get("census_build") or {}
        if cb.get("int4_expert_layers") != LAYERS or cb.get("int4_store_kinds") != ["int4_b32"]:
            void.append(f"build: int4 store not engaged ({cb.get('int4_expert_layers')}, {cb.get('int4_store_kinds')})")
        if (cb.get("grouping") or {}).get("device_grouping") is not False:
            void.append("build: device_grouping is not False (the max_seqs == 1 configuration)")
        if t.get("chunk_tokens") != 512 or t.get("rounds") != 3:
            void.append(f"ttft: chunk {t.get('chunk_tokens')} rounds {t.get('rounds')}")
        if {str(k): v for k, v in (t.get("prompts_sha256") or {}).items()} != PROMPT_SHA:
            void.append("ttft: prompts are not SC1's")
        for rt in ROUTES:
            c, k = (t.get("census") or {}).get(rt), (t.get("kernels") or {}).get(rt)
            if c is None or k is None:
                void.append(f"{rt}: no census")
                continue
            ok = {"loop": c["grouped_calls"] == c["loop_calls"] == LAYERS and c["dequant_total"] >= MIN_LOOP_DEQUANT
                  and c.get("loop_dequant_is_two_per_distinct") is True,
                  "batched": c["loop_calls"] == LAYERS and c["dequant_total"] == 0,
                  "k19": c["grouped_device_calls"] == LAYERS and c["dequant_total"] == 0
                  and k["k19"] >= MIN_KERNELS and k["mtile"] == 0,
                  "mtile": c["grouped_device_calls"] == LAYERS and c["dequant_total"] == 0
                  and k["mtile"] >= MIN_KERNELS and k["k19"] == 0}[rt]
            if not ok:
                void.append(f"{rt}: did not run as registered (census {c}, kernels {k})")
    if n is not None:
        by = {(w["source"], w["k"], w["route"]): w for w in n.get("windows", [])}
        for src, ks in WINDOWS.items():
            for k in ks:
                ws = [by.get((src, k, rt)) for rt in ROUTES]
                if None in ws:
                    void.append(f"nll {src} k={k}: a route is missing")
                elif len({w["text_sha"] for w in ws}) != 1 or any(w["steps"] != 2048 for w in ws):
                    void.append(f"nll {src} k={k}: routes scored different text or steps")
    if void:
        out["verdict"] = "VOID"
        return out
    med = t["ttft_s_median"]
    out["ttft_s_median"] = med
    q = {}
    # batched: bit identity
    same_nll = all(by[(s, k, "batched")]["mean_nll"] == by[(s, k, "loop")]["mean_nll"] for s, ks in WINDOWS.items() for k in ks)
    toks = {(dr["round"], dr["L"]): {} for dr in t["draws"]}
    for dr in t["draws"]:
        toks[(dr["round"], dr["L"])][dr["route"]] = dr["token"]
    same_tok = all(v.get("batched") == v.get("loop") for v in toks.values())
    q["batched"] = {"verdict": "PASS" if (same_nll and same_tok) else "QUALITY_FAIL", "nll_identical": same_nll,
                    "first_tokens_identical": same_tok}
    for rt in ("k19", "mtile"):
        pairs, per = [], {}
        for src, ks in WINDOWS.items():
            b = [by[(src, k, "loop")]["ppl"] for k in ks]
            c = [by[(src, k, rt)]["ppl"] for k in ks]
            deltas = [ci - bi for bi, ci in zip(b, c)]
            sha = hashlib.sha256("".join(by[(src, k, "loop")]["text_sha"] for k in ks).encode()).hexdigest()
            pairs.append((src, statistics.mean(b), statistics.mean(c), sha))
            per[src] = {"deltas": [round(x, 5) for x in deltas], "mean": round(statistics.mean(deltas), 5),
                        "sd": round(statistics.stdev(deltas), 5) if len(deltas) > 1 else None,
                        "windows_over_budget": sum(abs(x) > BUDGET for x in deltas)}
        ok, lines = _k8_verdict(pairs)
        q[rt] = {"verdict": "PASS" if ok else "QUALITY_FAIL", "per_text": per, "k8_lines": lines}
    out["quality"] = q
    passing = [rt for rt in ("batched", "k19", "mtile") if q[rt]["verdict"] == "PASS"]
    loop4096 = med["loop"]["4096"]
    pick = None
    if passing:
        fastest = min(passing, key=lambda r: med[r]["4096"])
        close = [r for r in passing if med[r]["4096"] <= 1.10 * med[fastest]["4096"]]
        pick = min(close, key=lambda r: NUMERICS_RANK[r])
    out["candidate"] = pick
    out["speedup_4096_vs_loop"] = {rt: round(loop4096 / med[rt]["4096"], 2) for rt in ROUTES}
    if pick is not None and med[pick]["4096"] <= 0.5 * loop4096:
        out["verdict"] = f"DEFAULT={pick}"
    else:
        out["verdict"] = "NO_CHANGE"
    return out


# ---------------------------------------------------------------------------------------------------- self-test
def _fixture(d: Path, ttft=None, nll_delta=None, census_fix=None, tok_fix=None):
    ttft = ttft or {"loop": (2.1, 16.6), "batched": (0.6, 4.0), "k19": (0.12, 0.6), "mtile": (0.12, 0.62)}
    nll_delta = nll_delta or {"batched": 0.0, "k19": 0.004, "mtile": 0.02}
    cen = {"loop": {"grouped_calls": 48, "loop_calls": 48, "grouped_device_calls": 0, "dequant_total": 8390,
                    "loop_dequant_is_two_per_distinct": True},
           "batched": {"grouped_calls": 48, "loop_calls": 48, "grouped_device_calls": 0, "dequant_total": 0},
           "k19": {"grouped_calls": 48, "loop_calls": 0, "grouped_device_calls": 48, "dequant_total": 0},
           "mtile": {"grouped_calls": 48, "loop_calls": 0, "grouped_device_calls": 48, "dequant_total": 0}}
    ker = {"loop": {"k19": 0, "mtile": 0}, "batched": {"k19": 0, "mtile": 0}, "k19": {"k19": 96, "mtile": 0},
           "mtile": {"k19": 0, "mtile": 96}}
    if census_fix:
        census_fix(cen, ker)
    draws = []
    for rnd in range(3):
        for rt in ROUTES:
            for i, L in enumerate((512, 4096)):
                draws.append({"route": rt, "L": L, "round": rnd, "wall_s": ttft[rt][i], "token": 100 + L})
    if tok_fix:
        tok_fix(draws)
    t = {"census_build": {"int4_expert_layers": 48, "int4_store_kinds": ["int4_b32"],
                          "grouping": {"device_grouping": False}}, "chunk_tokens": 512, "rounds": 3,
         "prompts_sha256": dict(PROMPT_SHA), "census": cen, "kernels": ker, "draws": draws,
         "ttft_s_median": {rt: {"512": ttft[rt][0], "4096": ttft[rt][1]} for rt in ROUTES}}
    (d / "ttft_routes.json").write_text(json.dumps(t))
    ws = []
    for src, ks in WINDOWS.items():
        for k in ks:
            base = 12.0 if src == "c4val1" else 6.0
            for rt in ROUTES:
                dl = 0.0 if rt == "loop" else nll_delta[rt] * (1 if k % 2 else 0.5)
                ws.append({"source": src, "k": k, "route": rt, "mean_nll": base + dl, "ppl": base + dl,
                           "text_sha": f"{src}{k}", "steps": 2048})
    (d / "nll_routes.json").write_text(json.dumps({"windows": ws}))


def selftest() -> int:
    import tempfile
    cases = []

    def case(label, want, **kw):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _fixture(d, **kw)
            r = reduce(d)
            assert r["verdict"] == want, (label, r)
            cases.append(label)
            return r

    case("k19 and mtile pass, tied: k19 by numerics", "DEFAULT=k19")
    case("mtile clearly faster: mtile", "DEFAULT=mtile",
         ttft={"loop": (2.1, 16.6), "batched": (0.6, 4.0), "k19": (0.2, 1.2), "mtile": (0.12, 0.6)})
    case("k19 fails quality: mtile", "DEFAULT=mtile", nll_delta={"batched": 0.0, "k19": 0.09, "mtile": 0.02})
    case("both fail, batched passes and is 4x: batched", "DEFAULT=batched",
         nll_delta={"batched": 0.0, "k19": 0.09, "mtile": 0.2})
    case("only batched passes, too slow: no change", "NO_CHANGE",
         nll_delta={"batched": 0.0, "k19": 0.09, "mtile": 0.2},
         ttft={"loop": (2.1, 16.6), "batched": (1.5, 12.0), "k19": (0.12, 0.6), "mtile": (0.12, 0.62)})
    r = case("batched not bit-identical: QUALITY_FAIL row", "DEFAULT=k19", nll_delta={"batched": 1e-6, "k19": 0.004, "mtile": 0.02})
    assert r["quality"]["batched"]["verdict"] == "QUALITY_FAIL"
    r = case("batched first token differs", "DEFAULT=k19",
             tok_fix=lambda dr: [x.update(token=7) for x in dr if x["route"] == "batched" and x["round"] == 2])
    assert r["quality"]["batched"]["first_tokens_identical"] is False
    case("k19 never launched K19: VOID", "VOID", census_fix=lambda c, k: k["k19"].update(k19=0, mtile=96))
    case("batched still decoded per expert: VOID", "VOID", census_fix=lambda c, k: c["batched"].update(dequant_total=8390))
    case("loop at the measured 8,390 decodes is engaged (A2)", "DEFAULT=k19")
    case("loop decodes not two per distinct: VOID", "VOID",
         census_fix=lambda c, k: c["loop"].update(loop_dequant_is_two_per_distinct=False))
    case("loop decoded almost nothing: VOID", "VOID", census_fix=lambda c, k: c["loop"].update(dequant_total=96))
    case("loop did not loop: VOID", "VOID", census_fix=lambda c, k: c["loop"].update(loop_calls=0))
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        _fixture(d)
        (d / "REHEARSAL").write_text("")
        assert reduce(d)["verdict"] == "VOID"
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
    print("P102_VERDICT " + json.dumps({k: r.get(k) for k in ("verdict", "candidate", "ttft_s_median",
                                                              "speedup_4096_vs_loop", "void_reasons")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
