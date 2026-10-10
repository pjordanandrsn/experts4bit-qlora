#!/usr/bin/env python3
"""p130_reduce.py -- lane P130 (e4b#1313), the reducer: speculative decoding's acceptance on the shipped default's target
and its priced B = 1 speedup (bench/p130/PREREG-p130.md). It runs on CPU from the box's capture files; no GPU.

**Accounting.** It is the same for both routes. The prompt's prefill emits the first generated token x[P]. From then on,
every verify step that starts with x[t + 1] as its last emitted token proposes up to k drafts for x[t + 2 ..]. The step
accepts their longest matching prefix a, and emits a + 1 tokens (the drafts plus the verifier's own). tau = the generated
tokens after the first, divided by the verify steps.

**Routes.**
- EAGLE-3: chain t (saved by the box, K = 5) truncated to k.
- n-gram: vLLM's prompt-lookup rule on the context x[0 .. t + 1]. It takes the latest earlier occurrence of the last n
  tokens, for n from NMAX down to NMIN, and proposes the k tokens after it; with no match there are no drafts.

**Price.** verify(n) = T1 + PE * (D(n) - 8) + PR * (n - 1) ms. These are P123's census constants on today's B = 1 step
(PREREG "The verify-step model"). D(n) is the distinct experts of n positions: independent top-8 of 128, or
8 + (n - 1) * 8 * (1 - r) at reuse r. The draft costs DRAFT_MS[route] per drafted token. S = tau-weighted time ratio.
"""
from __future__ import annotations

import argparse
import json
import sys

T1, PE, PR = 3.9119, 0.15086, 0.13444               # ms: today's B=1 step; per distinct expert; per extra row
E, TOPK = 128, 8
REUSE = 0.444                                        # the measured co-routing reuse (research, #1313)
DRAFT_MS = {"eagle3": 0.3, "ngram": 0.02}            # per drafted token (EAGLE-3: about 0.4 GB bf16 a step; INFERENCE)
NMAX, NMIN = 4, 2
KS = (1, 2, 3, 4, 5)
PREMISE_TAU1 = 1.40                                  # EAGLE-3's tau(k=1) on C-think, the head's own training mode
BAR_EAGLE, BAR_NGRAM = 1.15, 1.05


def d_ind(n):
    return E * (1 - (1 - TOPK / E) ** n)


def d_reuse(n, r=REUSE):
    return TOPK + (n - 1) * TOPK * (1 - r)


def verify_ms(n, d):
    return T1 + PE * (d(n) - TOPK) + PR * (n - 1)


def ngram_propose(ctx, k, nmax=NMAX, nmin=NMIN):
    L = len(ctx)
    for n in range(nmax, nmin - 1, -1):
        if L <= n:
            continue
        pat = ctx[L - n:]
        for s in range(L - n - 1, -1, -1):
            if ctx[s:s + n] == pat:
                return ctx[s + n:s + n + k]
    return []


def simulate(rows, k, route, d_fn):
    """rows: [{"tokens", "prompt_len", "chains"?}]. Returns (tau, steps, accept_rate, S)."""
    steps = emitted = acc = drafted = 0
    t_spec = 0.0
    for r in rows:
        tok, P = r["tokens"], r["prompt_len"]
        L = len(tok)
        t = P - 1                                    # x[P] came from the prefill; the last emitted index is t + 1 = P
        while t + 1 < L - 1:
            if route == "eagle3":
                d = list(r["chains"][t][:k])
            else:
                d = ngram_propose(tok[:t + 2], k)
            d = d[:max(0, L - 1 - (t + 1))]          # never draft past the recorded continuation
            a = 0
            while a < len(d) and d[a] == tok[t + 2 + a]:
                a += 1
            adv = min(a + 1, L - 1 - (t + 1))
            t_spec += verify_ms(len(d) + 1, d_fn) + DRAFT_MS[route] * len(d)
            steps += 1
            emitted += adv
            acc += a
            drafted += len(d)
            t += adv
    if not steps:
        return None
    return {"tau": round(emitted / steps, 4), "steps": steps, "emitted": emitted,
            "accept_rate": round(acc / drafted, 4) if drafted else 0.0, "S": round(emitted * T1 / t_spec, 4)}


def reduce(caps, measured_dn=None):
    """caps: {workload: {"rows": [...]}} for R, C-think, C-nothink. measured_dn: {n: D(n)} from the #1469 census, which
    replaces the reuse arm when given (PREREG "The reducer"); independence stays the conservative, deciding arm."""
    d_second = d_reuse
    if measured_dn:
        dn = {int(k): float(v) for k, v in measured_dn.items()}
        d_second = lambda n: dn[n] if n in dn else d_reuse(n)  # noqa: E731
    out = {"constants": {"T1": T1, "PE": PE, "PR": PR, "reuse": REUSE, "draft_ms": DRAFT_MS, "nmax": NMAX, "nmin": NMIN,
                         "measured_dn": measured_dn}, "routes": {}, "reasons": []}
    for route in ("eagle3", "ngram"):
        out["routes"][route] = {}
        for w, cap in caps.items():
            res = {}
            for k in KS:
                ind = simulate(cap["rows"], k, route, d_ind)
                reu = simulate(cap["rows"], k, route, d_second)
                res[str(k)] = {"tau": ind["tau"], "accept_rate": ind["accept_rate"], "steps": ind["steps"],
                               "S_independent": ind["S"], "S_reuse": reu["S"]}
            out["routes"][route][w] = res
    eag = out["routes"]["eagle3"]
    tau1 = eag.get("C-think", {}).get("1", {}).get("tau")
    if tau1 is None or tau1 < PREMISE_TAU1:
        out["verdict"] = "VOID"
        out["reasons"].append(f"premise: EAGLE-3 tau(k=1) on C-think = {tau1}, below {PREMISE_TAU1}: the capture is suspect")
        return out
    best = {w: max(v["S_independent"] for v in res.values()) for w, res in eag.items()}
    best_reuse = {w: max(v["S_reuse"] for v in res.values()) for w, res in eag.items()}
    out["best_S_eagle3_independent"] = best
    out["best_S_eagle3_reuse"] = best_reuse
    clear = [w for w in best if best[w] >= BAR_EAGLE]
    ng_r = max(v["S_independent"] for v in out["routes"]["ngram"]["R"].values())
    out["best_S_ngram_R_independent"] = ng_r
    if len(clear) >= 2:
        out["verdict"] = "PROCEED_EAGLE3"
    elif ng_r >= BAR_NGRAM:
        out["verdict"] = "PROCEED_NGRAM"
    else:
        out["verdict"] = "STOP"
    out["eagle3_workloads_clearing"] = clear
    return out


# ------------------------------------------------------------------------------------------------- self-test --
def self_test() -> int:
    bad, n = [], [0]

    def check(name, cond):
        n[0] += 1
        print(f"{'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            bad.append(name)

    seq = list(range(10)) + [7, 8, 9] * 30              # prompt 10, then a repeating tail
    P = 10
    L = len(seq)
    perfect = [seq[t + 2:t + 7] + [0] * max(0, 5 - len(seq[t + 2:t + 7])) for t in range(L - 1)]
    wrong = [[-1] * 5 for _ in range(L - 1)]
    rows_p = [{"tokens": seq, "prompt_len": P, "chains": perfect}]
    rows_w = [{"tokens": seq, "prompt_len": P, "chains": wrong}]
    gen_after_first = L - 1 - P
    r = simulate(rows_w, 3, "eagle3", d_ind)
    check("no acceptance: tau 1, one step per token", r["tau"] == 1.0 and r["steps"] == gen_after_first and r["accept_rate"] == 0)
    for k in KS:
        r = simulate(rows_p, k, "eagle3", d_ind)
        check(f"perfect chains k={k}: tau = k+1 up to the tail", abs(r["tau"] - gen_after_first / -(-gen_after_first // (k + 1))) < 1e-3)
    r = simulate(rows_w, 3, "eagle3", d_ind)
    check("rejected drafts still pay the verify width (S < 1)", r["S"] < 1.0)
    r = simulate([{"tokens": seq, "prompt_len": P}], 3, "ngram", d_ind)
    check("n-gram finds the repeating tail (tau > 3)", r["tau"] > 3.0)
    check("n-gram proposer: latest earlier match", ngram_propose([1, 2, 3, 9, 1, 2, 3, 5, 1, 2, 3], 2) == [5, 1])
    check("n-gram proposer: no match -> no drafts", ngram_propose([1, 2, 3, 4], 3) == [])
    check("verify(1) is the plain step", abs(verify_ms(1, d_ind) - T1) < 1e-9)
    check("D independent 2 positions = 15.5", abs(d_ind(2) - 15.5) < 1e-9)
    caps = {w: {"rows": rows_p} for w in ("R", "C-think", "C-nothink")}
    check("all-perfect -> PROCEED_EAGLE3", reduce(caps)["verdict"] == "PROCEED_EAGLE3")
    caps_v = {w: {"rows": rows_w} for w in ("R", "C-think", "C-nothink")}
    check("C-think tau(1) below the premise -> VOID", reduce(caps_v)["verdict"] == "VOID")
    print(f"p130_reduce self-test {'OK' if not bad else 'FAILED'} ({n[0] - len(bad)}/{n[0]} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--caps", nargs="*", help="capture_<W>.json files")
    ap.add_argument("--out")
    ap.add_argument("--measured-dn", help="JSON {n: D(n)} from the #1469 census (optional)")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    caps = {}
    for f in a.caps or []:
        c = json.load(open(f))
        caps[c["workload"]] = c
    if set(caps) != {"R", "C-think", "C-nothink"}:
        print(f"REFUSED: workloads {sorted(caps)}", file=sys.stderr)
        return 22
    out = reduce(caps, json.load(open(a.measured_dn)) if a.measured_dn else None)
    json.dump(out, open(a.out, "w"), indent=1)
    print("P130_VERDICT " + json.dumps({"verdict": out["verdict"], "reasons": out["reasons"],
                                         "best_S_eagle3": out.get("best_S_eagle3_independent"),
                                         "best_S_ngram_R": out.get("best_S_ngram_R_independent")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
