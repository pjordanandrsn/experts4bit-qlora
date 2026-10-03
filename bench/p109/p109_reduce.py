#!/usr/bin/env python3
"""p109_reduce.py -- lane P109's registered rule (bench/p109/PREREG-p109.md; e4b#770), from the five arm receipts.

Receipts: arm_E1.json, arm_G1.json, arm_G2.json, arm_E2.json, arm_D1.json, arm_P1.json (run in that order), each from
p109_box.py.

The verdict is the first of these that applies:
  VOID           an arm is missing or not ok; a receipt names the wrong e4b / grouped-nf4-gemm commit or model revision; the
                 arms read different prompts; or engagement is wrong:
                   G: every bucket in (1, 2, 4, 8, 16) is "graph" and device grouping is on;
                   E: no graphs and device grouping off;
                   D: no graphs and device grouping on;
                   P: every bucket "eager: capture=False" and device grouping on (Amendment 2).
  NOISY          a self-pair (E2/E1 or G2/G1, decode tok/s, either workload) falls outside [0.93, 1.07].
  FUNCTION_FAIL  the graph replay is not its own eager function (Amendment 2: the PADDED eager step, P1):
                   G1 and G2 must emit tokens identical to P1's on every row of both workloads, at both lengths;
                   every timed rep of a G or P arm must digest the same.
  KEEP           speed: S16 = min(G1/E1, G2/E2) on W16 < 1.25, or S1 = the same on W1 < 0.97.
  DIVERGENT      sanity: fewer than 12 of W16's 16 rows have G1 agree with E1 for their first 16 LONG tokens.
  DEFAULT_GRAPHS otherwise.

Reported beside the verdict (no bar):
  - E1 = E2 tokens;
  - E-vs-G first-divergence positions;
  - G against D1, the UNPADDED eager step with the same grouping (the registration's original oracle): rows identical
    and first divergences, and whether P1 = D1;
  - G-minus-E peak memory and load time.
"""
import argparse
import json
import os
import statistics
import sys

TAGS = ("E1", "G1", "G2", "E2", "D1", "P1")
BUCKETS = ("1", "2", "4", "8", "16")
SELF_LO, SELF_HI = 0.93, 1.07
S16_MIN, S1_MIN = 1.25, 0.97
SANE_ROWS, SANE_TOKENS = 12, 16
GNF4_SHA = "51a49166ae7bc1a0f84188b7b5d1f42ecbc37e00"
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "ibm-granite/granite-3.1-3b-a800m-instruct": "a02780686e08a03fe0d2679a293b5c74a90efa89"}


def first_divergence(a, b):
    """Index of the first differing token between two token lists (len if equal over the shorter)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def reduce(arms: dict, e4b_sha: str) -> dict:
    out = {"verdict": None, "reasons": [], "bars": {"self_pair": [SELF_LO, SELF_HI], "s16_min": S16_MIN, "s1_min": S1_MIN,
                                                    "sanity": {"rows": SANE_ROWS, "tokens": SANE_TOKENS}}}
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    if missing:
        out.update(verdict="VOID", reasons=[f"arm(s) missing or not ok: {missing}"])
        return out
    void = []
    for t, r in arms.items():
        if r.get("e4b_sha") != e4b_sha:
            void.append(f"{t}: e4b {r.get('e4b_sha')} != {e4b_sha}")
        if r.get("gnf4_sha") != GNF4_SHA:
            void.append(f"{t}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
        if REVS.get(r.get("model")) != r.get("revision"):
            void.append(f"{t}: model {r.get('model')}@{r.get('revision')} is not a registered revision")
        g, flags = r.get("graph_status"), r.get("grouping_flags_at_run") or {}
        if t.startswith(("G", "P")):
            want = "graph" if t.startswith("G") else "eager: capture=False"
            if not g or [g.get(b) for b in BUCKETS] != [want] * 5:
                void.append(f"{t}: graph_status {g}")
            if not flags.get("device_grouping"):
                void.append(f"{t}: device grouping off")
        else:
            if g:
                void.append(f"{t}: graphs on in an eager arm")
            if bool(flags.get("device_grouping")) != t.startswith("D"):
                void.append(f"{t}: device_grouping={flags.get('device_grouping')}")
    if len({r["prompts_sha256"] for r in arms.values()}) != 1:
        void.append("the arms read different prompts")
    if len({(r["short"], r["long"], r["reps"]) for r in arms.values()}) != 1:
        void.append("the arms ran different lengths")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    E1, G1, G2, E2, D1, P1 = (arms[t] for t in TAGS)
    short, long_ = E1["short"], E1["long"]
    rates = {t: {w: _rate(arms[t], w) for w in ("W16", "W1")} for t in TAGS}
    out["decode_tok_s"] = rates
    if any(v is None for d in rates.values() for v in d.values()):
        out.update(verdict="VOID", reasons=["a decode slope is void"])
        return out
    selfp = {f"{a}/{b} {w}": rates[a][w] / rates[b][w] for a, b in (("E2", "E1"), ("G2", "G1")) for w in ("W16", "W1")}
    out["self_pairs"] = {k: round(v, 4) for k, v in selfp.items()}
    s16 = min(rates["G1"]["W16"] / rates["E1"]["W16"], rates["G2"]["W16"] / rates["E2"]["W16"])
    s1 = min(rates["G1"]["W1"] / rates["E1"]["W1"], rates["G2"]["W1"] / rates["E2"]["W1"])
    out["S16"], out["S1"] = round(s16, 4), round(s1, 4)
    fn = []
    for w in ("W16", "W1"):
        for n in (short, long_):
            for t in ("G1", "G2"):
                if _tok(arms[t], w, n) != _tok(P1, w, n):
                    rows = [i for i, (x, y) in enumerate(zip(_tok(arms[t], w, n), _tok(P1, w, n))) if x != y]
                    fn.append(f"{t} != P1 on {w} at {n} tokens, rows {rows}")
    for t in ("G1", "G2", "P1"):
        for w in ("W16", "W1"):
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fn.append(f"{t} {w} at {n} tokens: timed reps differ")
    fd = [first_divergence(a, b) for a, b in zip(_tok(E1, "W16", long_), _tok(G1, "W16", long_))]
    agree = sum(1 for d in fd if d >= min(SANE_TOKENS, long_))      # a row identical over all LONG tokens agrees
    out["report"] = {
        "E1_eq_E2": all(_tok(E1, w, n) == _tok(E2, w, n) for w in ("W16", "W1") for n in (short, long_)),
        "E_vs_G_W16_first_divergence": fd, "E_vs_G_W16_rows_identical": sum(1 for d in fd if d == long_),
        "E_vs_G_W16_median_first_divergence": statistics.median(fd),
        "E_vs_G_W1_first_divergence": first_divergence(_tok(E1, "W1", long_)[0], _tok(G1, "W1", long_)[0]),
        "sanity_rows_agreeing": agree,
        "G_vs_D_first_divergence": {w: [first_divergence(a, b) for a, b in zip(_tok(G1, w, long_), _tok(D1, w, long_))]
                                    for w in ("W16", "W1")},
        "G_eq_D": all(_tok(G1, w, n) == _tok(D1, w, n) for w in ("W16", "W1") for n in (short, long_)),
        "P_eq_D": all(_tok(P1, w, n) == _tok(D1, w, n) for w in ("W16", "W1") for n in (short, long_)),
        "peak_gib": {t: round(arms[t]["mem_after_runs"]["max_memory_allocated"] / 2**30, 3) for t in TAGS},
        "load_s": {t: arms[t]["load_s"] for t in TAGS}}
    out["report"]["G_minus_E_peak_gib"] = round(out["report"]["peak_gib"]["G1"] - out["report"]["peak_gib"]["E1"], 3)
    out["report"]["G_minus_E_load_s"] = round(arms["G1"]["load_s"] - arms["E1"]["load_s"], 2)
    noisy = [f"{k} = {v:.4f}" for k, v in selfp.items() if not SELF_LO <= v <= SELF_HI]
    if noisy:
        out.update(verdict="NOISY", reasons=noisy)
    elif fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif s16 < S16_MIN or s1 < S1_MIN:
        out.update(verdict="KEEP", reasons=[f"S16 = {s16:.4f} (bar {S16_MIN}), S1 = {s1:.4f} (bar {S1_MIN})"])
    elif agree < SANE_ROWS:
        out.update(verdict="DIVERGENT", reasons=[f"{agree} of 16 W16 rows agree for {SANE_TOKENS} tokens (bar {SANE_ROWS})"])
    else:
        out.update(verdict="DEFAULT_GRAPHS", reasons=[f"S16 = {s16:.4f}, S1 = {s1:.4f}; G = D on every row; "
                                                      f"{agree} of 16 rows agree with E for {SANE_TOKENS} tokens"])
    return out


# ------------------------------------------------------------------ self-test --

def _fake(tag, rate16, rate1, toks16, toks1, *, e4b="a" * 40, model="Qwen/Qwen3-30B-A3B", graphs=None, dg=None,
          digests=None, short=4, long_=8):
    arm = tag[0]
    if graphs is None:
        graphs = ({b: "graph" for b in BUCKETS} if arm == "G" else
                  {b: "eager: capture=False" for b in BUCKETS} if arm == "P" else None)
    if dg is None:
        dg = arm in ("G", "D", "P")

    def wl(rate, toks):
        return {"decode_tok_s": rate, "tokens": {str(short): [t[:short] for t in toks], str(long_): toks},
                "rep_digests": digests or {str(short): ["x"] * 3, str(long_): ["y"] * 3}}
    return {"arm": arm, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": model, "revision": REVS[model],
            "graph_status": graphs, "grouping_flags_at_run": {"device_grouping": dg}, "prompts_sha256": "p",
            "short": short, "long": long_, "reps": 3, "load_s": 60.0 + (5 if arm == "G" else 0),
            "mem_after_runs": {"max_memory_allocated": int((17.0 + (0.05 if arm == "G" else 0)) * 2**30)},
            "workloads": {"W16": wl(rate16, toks16), "W1": wl(rate1, toks1)}}


def self_test() -> int:
    base = [[r * 100 + i for i in range(8)] for r in range(16)]
    one = [base[0]]

    def arms(**over):
        a = {"E1": _fake("E1", 200.0, 40.0, base, one), "G1": _fake("G1", 900.0, 150.0, base, one),
             "G2": _fake("G2", 905.0, 151.0, base, one), "E2": _fake("E2", 198.0, 40.5, base, one),
             "D1": _fake("D1", 210.0, 41.0, base, one), "P1": _fake("P1", 230.0, 42.0, base, one)}
        a.update(over)
        return a
    E = "a" * 40
    cases = []
    cases.append(("default", reduce(arms(), E)["verdict"] == "DEFAULT_GRAPHS"))
    cases.append(("missing arm", reduce({k: v for k, v in arms().items() if k != "D1"}, E)["verdict"] == "VOID"))
    cases.append(("wrong e4b", reduce(arms(G2=_fake("G2", 905.0, 151.0, base, one, e4b="b" * 40)), E)["verdict"] == "VOID"))
    cases.append(("eager bucket in G", reduce(arms(G1=_fake("G1", 900.0, 150.0, base, one,
                                                             graphs={**{b: "graph" for b in BUCKETS}, "16": "eager: x"})), E)["verdict"] == "VOID"))
    cases.append(("E with device grouping", reduce(arms(E1=_fake("E1", 200.0, 40.0, base, one, dg=True)), E)["verdict"] == "VOID"))
    cases.append(("D without device grouping", reduce(arms(D1=_fake("D1", 210.0, 41.0, base, one, dg=False)), E)["verdict"] == "VOID"))
    cases.append(("noisy E", reduce(arms(E2=_fake("E2", 150.0, 40.5, base, one)), E)["verdict"] == "NOISY"))
    g_off = [list(r) for r in base]
    g_off[3][5] += 1
    cases.append(("G != P", reduce(arms(G1=_fake("G1", 900.0, 150.0, g_off, one)), E)["verdict"] == "FUNCTION_FAIL"))
    r_d = reduce(arms(D1=_fake("D1", 210.0, 41.0, g_off, one)), E)                 # unpadded eager differs: reported only
    cases.append(("G != D reported", r_d["verdict"] == "DEFAULT_GRAPHS" and not r_d["report"]["G_eq_D"]
                  and r_d["report"]["G_vs_D_first_divergence"]["W16"][3] == 5 and not r_d["report"]["P_eq_D"]))
    cases.append(("P captured graphs", reduce(arms(P1=_fake("P1", 230.0, 42.0, base, one,
                                                             graphs={b: "graph" for b in BUCKETS})), E)["verdict"] == "VOID"))
    cases.append(("G reps differ", reduce(arms(G2=_fake("G2", 905.0, 151.0, base, one, digests={"4": ["x"] * 3, "8": ["y", "y", "z"]})), E)["verdict"] == "FUNCTION_FAIL"))
    cases.append(("slow at B=16", reduce(arms(G1=_fake("G1", 240.0, 150.0, base, one), G2=_fake("G2", 241.0, 151.0, base, one)), E)["verdict"] == "KEEP"))
    cases.append(("regress at B=1", reduce(arms(G1=_fake("G1", 900.0, 38.0, base, one), G2=_fake("G2", 905.0, 38.2, base, one)), E)["verdict"] == "KEEP"))
    e_far = [[t + 1 if i >= 2 else t for i, t in enumerate(r)] for r in base]       # E diverges at token 2 on every row
    # 8-token fakes: the sanity bar is min(16, LONG) = 8 tokens, so a row diverging at token 2 does not agree
    cases.append(("divergent", reduce(arms(E1=_fake("E1", 200.0, 40.0, e_far, one)), E)["verdict"] == "DIVERGENT"))
    r = reduce(arms(), E)
    cases.append(("report", r["report"]["E1_eq_E2"] and r["report"]["G_minus_E_peak_gib"] == 0.05 and r["S16"] == round(900 / 200, 4)))
    cases.append(("first_divergence", first_divergence([1, 2, 3], [1, 2, 4]) == 2 and first_divergence([1, 2], [1, 2]) == 2))
    cases.append(("granite revision", reduce({t: dict(v, model="ibm-granite/granite-3.1-3b-a800m-instruct",
                                                      revision=REVS["ibm-granite/granite-3.1-3b-a800m-instruct"]) for t, v in arms().items()}, E)["verdict"] == "DEFAULT_GRAPHS"))
    cases.append(("wrong revision", reduce(arms(E1=dict(_fake("E1", 200.0, 40.0, base, one), revision="main")), E)["verdict"] == "VOID"))
    bad = [n for n, ok in cases if not ok]
    print(f"p109_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    arms = {}
    for t in TAGS:
        f = os.path.join(a.dir, f"arm_{t}.json")
        if os.path.exists(f):
            arms[t] = json.load(open(f))
    v = reduce(arms, a.e4b_sha)
    json.dump(v, open(a.out, "w"), indent=1)
    print(f"P109_VERDICT {v['verdict']} {json.dumps(v['reasons'])}")
    for k in ("S16", "S1", "self_pairs", "decode_tok_s"):
        if k in v:
            print(f"  {k}: {json.dumps(v[k])}")
    if "report" in v:
        print(f"  report: {json.dumps(v['report'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
