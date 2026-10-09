#!/usr/bin/env python3
"""p127_reduce.py -- lane P127's registered rule (bench/p127/PREREG-p127.md; e4b#1313).

Reads arm_A1.json, arm_B1.json, arm_B2.json, arm_A2.json, arm_M1.json (p127_box.py's records) and audit.json (the
runner's served-path diff audit). The verdict is the first rung that applies: VOID, FUNCTION_FAIL, NOISY, then SLOWER,
FASTER or NO_MEASURABLE_GAIN. ``--self-test`` runs the rule on fixtures shaped as the box writes them.
"""
import argparse
import json
import math
import os
import sys

TIMED = ("A1", "B1", "B2", "A2")
ALL = TIMED + ("M1",)
WORKLOADS = ("W16", "W1")
BUCKETS = [1, 2, 4, 8, 16]
COUNTS = ("router_weights_dtype", "rope_norm_qk", "combine_residual", "gather_div", "int64_ids")
NEW_API = ("router_weights_dtype", "rope_norm_qk", "combine_residual", "gather_div")
NOISE = (0.97, 1.03)
SLOWER, FASTER = 0.98, 1.03
BLIND = 0.50


def _side(tag):
    return "A" if tag.startswith("A") else "B"


def positions(rec):
    """Every (workload, row, step) logits digest of the identity pass, in a fixed order."""
    out = []
    for w in WORKLOADS:
        for r, ds in enumerate(rec["identity"][w]["logits_digests"]):
            for s, d in enumerate(ds):
                out.append(((w, r, s), d))
    return out


def same_tokens(a, b):
    """Every token of every row of both workloads at both lengths, timed and identity."""
    for w in WORKLOADS:
        ta, tb = a["workloads"][w]["tokens"], b["workloads"][w]["tokens"]
        if ta != tb:
            return False
        if a["identity"][w]["tokens"] != b["identity"][w]["tokens"]:
            return False
    return True


def same_logits(a, b):
    return [d for _p, d in positions(a)] == [d for _p, d in positions(b)]


def reduce(recs, audit, want) -> dict:
    """``want``: {"e4b_a", "e4b_b", "gnf4_a", "gnf4_b", "revision"}."""
    out = {"verdict": None, "reasons": [], "report": {}}
    void = out["reasons"]
    # ---------------------------------------------------------------------------------------------- 1. VOID --
    for tag in ALL:
        r = recs.get(tag)
        if r is None or r.get("status") != "ok":
            void.append(f"arm {tag} missing or not ok")
    if void:
        out["verdict"] = "VOID"
        return out
    for tag in ALL:
        r, side = recs[tag], _side(tag)
        if r["e4b_sha"] != want[f"e4b_{side.lower()}"] or r["gnf4_sha"] != want[f"gnf4_{side.lower()}"]:
            void.append(f"{tag}: another commit (e4b {r['e4b_sha']}, gnf4 {r['gnf4_sha']})")
        if r.get("revision") != want["revision"]:
            void.append(f"{tag}: another model revision {r.get('revision')}")
        other = "B" if side == "A" else "A"
        if any(f"_{other}/" in str(v).replace("\\", "/") for v in r["paths"].values()):
            void.append(f"{tag}: imported arm {other}'s install ({r['paths']})")
        if r.get("max_seqs") != 16:
            void.append(f"{tag}: max_seqs {r.get('max_seqs')}, not 16")
    if len({recs[t]["prompts_sha256"] for t in ALL}) != 1:
        void.append("the arms read different prompts")
    if len({(recs[t]["short"], recs[t]["long"]) for t in TIMED}) != 1 or recs["M1"]["long"] != recs["A1"]["long"]:
        void.append("the arms ran different lengths")
    for t in TIMED:
        for w in WORKLOADS:
            if recs[t]["workloads"][w].get("decode_tok_s") is None:
                void.append(f"{t}/{w}: a void slope")
    for t in ("B1", "B2", "M1"):
        zero = [k for k in COUNTS if not recs[t]["counts"].get(k)]
        if zero:
            void.append(f"{t}: not engaged on {zero}")
    for t in ("B1", "B2"):
        mr = recs[t].get("moe_residual") or {}
        if mr.get("rows") != BUCKETS or mr.get("licensed") != recs[t].get("moe_layers") or not mr.get("licensed"):
            void.append(f"{t}: the residual licence does not cover every MoE layer at every bucket ({mr})")
    for t in ("A1", "A2"):
        hot = [k for k in NEW_API if recs[t]["counts"].get(k)]
        if hot:
            void.append(f"{t}: the before arm engaged {hot}")
    for x, y in (("A1", "A2"), ("B1", "B2")):
        if not same_tokens(recs[x], recs[y]) or not same_logits(recs[x], recs[y]):
            void.append(f"nondeterministic: {x} and {y} differ")
    if not audit or not audit.get("ok"):
        void.append(f"the served-path diff audit refused: {(audit or {}).get('unlisted')}")
    pa, pm = positions(recs["A1"]), positions(recs["M1"])
    differ = sum(1 for (p1, d1), (p2, d2) in zip(pa, pm) if p1 != p2 or d1 != d2) + abs(len(pa) - len(pm))
    frac = differ / max(len(pa), len(pm), 1)
    out["report"]["m_logits_differ_frac"] = round(frac, 4)
    if frac < BLIND:
        void.append(f"the instrument is blind: M differs from A1 at {frac:.1%} of positions (< {BLIND:.0%})")
    if void:
        out["verdict"] = "VOID"
        return out
    # ------------------------------------------------------------------------------------- 2. FUNCTION_FAIL --
    fails = out["reasons"]
    if not same_tokens(recs["B1"], recs["A1"]):
        fails.append("B1's tokens differ from A1's")
    if not same_logits(recs["B1"], recs["A1"]):
        first = next(((p1, p2) for (p1, d1), (p2, d2) in zip(positions(recs["A1"]), positions(recs["B1"]))
                      if d1 != d2), None)
        fails.append(f"B1's logits differ from A1's (first at {first})")
    for t in TIMED:
        for w in WORKLOADS:
            for n, ds in recs[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fails.append(f"{t}/{w}/{n}: the timed passes do not all digest the same")
    if fails:
        out["verdict"] = "FUNCTION_FAIL"
        return out
    # --------------------------------------------------------------------------------------------- 3. NOISY --
    tok = {t: {w: recs[t]["workloads"][w]["decode_tok_s"] for w in WORKLOADS} for t in TIMED}
    for x, y in (("A2", "A1"), ("B2", "B1")):
        for w in WORKLOADS:
            r = tok[x][w] / tok[y][w]
            out["report"][f"self_{x}/{y}_{w}"] = round(r, 4)
            if not NOISE[0] <= r <= NOISE[1]:
                out["reasons"].append(f"{x}/{y} on {w} = {r:.4f}, outside {list(NOISE)}")
    if out["reasons"]:
        out["verdict"] = "NOISY"
        return out
    # ---------------------------------------------------------------------------------------------- 4. speed --
    g = {}
    for w in WORKLOADS:
        r1, r2 = tok["B1"][w] / tok["A1"][w], tok["B2"][w] / tok["A2"][w]
        g[w] = math.fsum([r1, r2]) / 2
        out["report"][f"g_{w}"] = round(g[w], 4)
        out["report"][f"interval_{w}"] = [round(min(r1, r2), 4), round(max(r1, r2), 4)]
        sa = [recs[t]["workloads"][w]["decode_ms_per_step"] for t in ("A1", "A2")]
        sb = [recs[t]["workloads"][w]["decode_ms_per_step"] for t in ("B1", "B2")]
        out["report"][f"saving_us_per_step_{w}"] = round(1000 * (math.fsum(sa) - math.fsum(sb)) / 2, 1)
    if g["W1"] <= SLOWER or g["W16"] <= SLOWER:
        out["verdict"] = "SLOWER"
    elif g["W1"] >= FASTER and min(out["report"]["interval_W1"]) > 1:
        out["verdict"] = "FASTER"
    else:
        out["verdict"] = "NO_MEASURABLE_GAIN"
    return out


def predictions(recs, res) -> dict:
    rep = res.get("report", {})
    q = {}
    if "g_W1" in rep:
        q["Q1_g1_in_1.03_1.12"] = 1.03 <= rep["g_W1"] <= 1.12
        q["Q2_g16_in_0.99_1.04"] = 0.99 <= rep["g_W16"] <= 1.04
        q["Q4_self_pairs_within_1pct"] = all(0.99 <= v <= 1.01 for k, v in rep.items() if k.startswith("self_"))
        pa = [recs[t]["mem_after_runs"]["max_memory_allocated"] for t in ("A1", "A2")]
        pb = [recs[t]["mem_after_runs"]["max_memory_allocated"] for t in ("B1", "B2")]
        q["Q5_peak_mem_within_64MiB"] = abs(max(pb) - max(pa)) <= 64 * 2 ** 20
    if "m_logits_differ_frac" in rep:
        ta = recs["A1"]["identity"]["W16"]["tokens"]
        tm = recs["M1"]["identity"]["W16"]["tokens"]
        same = sum(x == y for ra, rm in zip(ta, tm) for x, y in zip(ra, rm))
        total = sum(len(ra) for ra in ta) or 1
        q["Q3_m_differs_ge_90pct_tokens_agree_ge_50pct"] = rep["m_logits_differ_frac"] >= 0.90 and same / total >= 0.5
    return q


def load(d):
    recs = {}
    for t in ALL:
        p = os.path.join(d, f"arm_{t}.json")
        if os.path.exists(p):
            recs[t] = json.load(open(p))
    ap = os.path.join(d, "audit.json")
    return recs, (json.load(open(ap)) if os.path.exists(ap) else None)


# ------------------------------------------------------------------------------------------------- self-test --
def _fake(tag, want, tok_s=(100.0, 1000.0), ident=None, counts=None):
    side = _side(tag)
    ident = ident or {"W16": {"tokens": [[1, 2, 3]] * 16, "logits_digests": [["a", "b"]] * 16},
                      "W1": {"tokens": [[1, 2, 3]], "logits_digests": [["a", "b"]]}}
    c = counts if counts is not None else ({k: 0 for k in COUNTS} | {"int64_ids": 5} if side == "A"
                                           else {k: 3 for k in COUNTS})
    return {"status": "ok", "arm": tag[0], "tag": tag, "e4b_sha": want[f"e4b_{side.lower()}"],
            "gnf4_sha": want[f"gnf4_{side.lower()}"], "revision": want["revision"], "max_seqs": 16,
            "paths": {"experts4bit_qlora": f"/root/p127/src/e4b_{side}/x", "int4_b32": f"/root/p127/src/gnf4_{side}/k"},
            "prompts_sha256": "p", "short": 32, "long": 160, "moe_layers": 48,
            "moe_residual": ({"rows": BUCKETS, "licensed": 48} if side == "B" else None),
            "counts": c, "identity": json.loads(json.dumps(ident)),
            "workloads": {"W16": {"tokens": {"32": [[1]], "160": [[1, 2]]}, "rep_digests": {"32": ["x"] * 3},
                                  "decode_tok_s": tok_s[1], "decode_ms_per_step": 16.0 / tok_s[1] * 1000},
                          "W1": {"tokens": {"32": [[1]], "160": [[1, 2]]}, "rep_digests": {"32": ["y"] * 3},
                                 "decode_tok_s": tok_s[0], "decode_ms_per_step": 1.0 / tok_s[0] * 1000}},
            "mem_after_runs": {"max_memory_allocated": 10}}


def _fixture(want, b_speed=1.06, b16=1.01):
    recs = {t: _fake(t, want, tok_s=((100 * b_speed, 1000 * b16) if t[0] == "B" else (100.0, 1000.0))) for t in TIMED}
    m = _fake("M1", want)
    m["identity"]["W16"]["logits_digests"] = [["m", "n"]] * 16
    m["identity"]["W1"]["logits_digests"] = [["m", "n"]]
    recs["M1"] = m
    return recs


def self_test() -> int:
    want = {"e4b_a": "a" * 40, "e4b_b": "b" * 40, "gnf4_a": "c" * 40, "gnf4_b": "d" * 40, "revision": "rev"}
    audit = {"ok": True, "unlisted": []}
    cases = []

    def case(name, recs, expect, aud=audit, needle=None):
        r = reduce(recs, aud, want)
        good = r["verdict"] == expect and (needle is None or any(needle in x for x in r["reasons"]))
        cases.append((name, good, r["verdict"], r["reasons"][:2]))

    case("faster", _fixture(want), "FASTER")
    case("no gain", _fixture(want, b_speed=1.01), "NO_MEASURABLE_GAIN")
    case("slower W1", _fixture(want, b_speed=0.97), "SLOWER")
    case("slower W16", _fixture(want, b16=0.975), "SLOWER")
    r = _fixture(want)
    del r["A2"]
    case("missing arm", r, "VOID", needle="missing")
    r = _fixture(want)
    r["B2"]["e4b_sha"] = "e" * 40
    case("wrong sha", r, "VOID", needle="another commit")
    r = _fixture(want)
    r["B1"]["paths"]["int4_b32"] = "/root/p127/src/gnf4_A/k"
    case("other install", r, "VOID", needle="imported arm A")
    r = _fixture(want)
    r["A1"]["max_seqs"] = 32
    case("max_seqs", r, "VOID", needle="max_seqs")
    r = _fixture(want)
    r["B1"]["counts"]["gather_div"] = 0
    case("not engaged", r, "VOID", needle="not engaged")
    r = _fixture(want)
    r["M1"]["counts"]["rope_norm_qk"] = 0
    case("M not engaged", r, "VOID", needle="not engaged")
    r = _fixture(want)
    r["A2"]["counts"]["rope_norm_qk"] = 1
    case("A engaged", r, "VOID", needle="before arm engaged")
    r = _fixture(want)
    r["B1"]["moe_residual"] = {"rows": BUCKETS, "licensed": 40}
    case("partial licence", r, "VOID", needle="residual licence")
    r = _fixture(want)
    r["A2"]["identity"]["W1"]["logits_digests"] = [["a", "z"]]
    case("nondeterministic", r, "VOID", needle="nondeterministic")
    r = _fixture(want)
    r["M1"]["identity"] = json.loads(json.dumps(r["A1"]["identity"]))
    case("blind M", r, "VOID", needle="blind")
    case("audit refused", _fixture(want), "VOID", aud={"ok": False, "unlisted": ["deadbeef"]}, needle="audit")
    r = _fixture(want)
    r["B1"]["prompts_sha256"] = "q"
    case("prompts", r, "VOID", needle="prompts")
    r = _fixture(want)
    r["A1"]["workloads"]["W1"]["decode_tok_s"] = None
    case("void slope", r, "VOID", needle="void slope")
    r = _fixture(want)
    for t in ("B1", "B2"):
        r[t]["identity"]["W16"]["tokens"] = [[1, 2, 4]] * 16
    case("tokens differ", r, "FUNCTION_FAIL", needle="tokens")
    r = _fixture(want)
    for t in ("B1", "B2"):
        r[t]["identity"]["W16"]["logits_digests"] = [["a", "c"]] * 16
    case("logits differ", r, "FUNCTION_FAIL", needle="logits")
    r = _fixture(want)
    r["A1"]["workloads"]["W1"]["rep_digests"] = {"32": ["y", "y", "z"]}
    case("reps differ", r, "FUNCTION_FAIL", needle="digest the same")
    r = _fixture(want)
    r["A2"]["workloads"]["W16"]["decode_tok_s"] = 1040.0
    case("noisy", r, "NOISY", needle="A2/A1")
    res = reduce(_fixture(want), audit, want)
    cases.append(("interval reported", res["report"]["interval_W1"] == [1.06, 1.06]
                  and res["report"]["m_logits_differ_frac"] == 1.0, res["verdict"], []))
    cases.append(("predictions evaluate", "Q1_g1_in_1.03_1.12" in predictions(_fixture(want), res), res["verdict"], []))
    bad = [c for c in cases if not c[1]]
    for c in bad:
        print("FAILED", c)
    print(f"p127_reduce self-test {'OK' if not bad else 'FAILED'} ({len(cases) - len(bad)}/{len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--e4b-a")
    p.add_argument("--e4b-b")
    p.add_argument("--gnf4-a")
    p.add_argument("--gnf4-b")
    p.add_argument("--revision")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    recs, audit = load(a.dir)
    want = {"e4b_a": a.e4b_a, "e4b_b": a.e4b_b, "gnf4_a": a.gnf4_a, "gnf4_b": a.gnf4_b, "revision": a.revision}
    res = reduce(recs, audit, want)
    res["predictions"] = predictions(recs, res) if res["verdict"] != "VOID" else {}
    res["counts"] = {t: recs[t].get("counts") for t in recs}
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    print("P127_VERDICT " + json.dumps({"verdict": res["verdict"], "reasons": res["reasons"][:4], **res["report"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
