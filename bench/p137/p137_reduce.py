#!/usr/bin/env python3
"""p137_reduce.py -- lane P137's reducer (bench/p137/PREREG-p137.md). Reads arm_{A0a,L1a,L1b,A0b}.json and writes the
verdict. The rule, first rung that applies:
  1. VOID: an arm record missing or OOM (an L1 OOM is VOID, never NO_GAIN); a record from another e4b commit than the
     pin, or the arms on different stacks or models; the fast path not patched or the experts not offloaded; a dry
     record in a reading or the reverse; a reading whose model path lacks the pinned revision; the patched and offloaded
     layer counts not both the registered n (48; the dry fixture's 2); step 0's per-micro-batch losses not
     bit-identical between every arm; the stagings per timed step not exactly ACCUM x (2n - 1) (A0) and 2n - 1 (L1)
     at that n; L1's peak device memory above A0's by more than MEM_TERM_GIB.
  2. NOISY: a self-pair (A0b/A0a, L1b/L1a, the median timed walls) outside [SELF_LO, SELF_HI].
  3. LICENSED iff S = min(A0a/L1a, A0b/L1b) >= S_MIN; otherwise NO_GAIN.
REPORTING (never changes the verdict): held-out |L1 - A0| <= HELD_TOL; S inside the registered band [BAND_LO, BAND_HI].

    python p137_reduce.py --dir RUN --out verdict.json --e4b-sha SHA [--dry]
    python p137_reduce.py --self-test
"""
import argparse
import json
import os
import sys

TAGS = ("A0a", "L1a", "L1b", "A0b")
SELF_LO, SELF_HI = 0.97, 1.03
S_MIN = 1.10
BAND_LO, BAND_HI = 1.25, 2.0
MEM_TERM_GIB = 4.25
HELD_TOL = 0.01
ACCUM = 4
QWEN = "Qwen/Qwen3-30B-A3B"
QWEN_REV = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
TINY = "p137-tiny"
N_LAYERS = {"reading": 48, "dry": 2}     # the registered n: Qwen3-30B-A3B's 48 MoE layers; the fixture's 2
GIB = 2 ** 30


def _ratio(a, b):
    return a / b if b else float("inf")


def reduce(recs, e4b_sha, dry=False):
    """recs: tag -> record dict, or None when the file is missing. Returns the verdict dict."""
    reasons = []
    missing = [t for t in TAGS if recs.get(t) is None]
    if missing:
        return {"verdict": "VOID", "reasons": [f"arm record missing: {t}" for t in missing]}
    oom = [t for t in TAGS if recs[t].get("oom")]
    if oom:
        return {"verdict": "VOID", "reasons": [f"{t}: OOM ({recs[t].get('oom')})" for t in oom]}
    r = recs
    n_reg = N_LAYERS["dry" if dry else "reading"]
    for t in TAGS:
        v = r[t].get("versions", {})
        if v.get("e4b_sha") != e4b_sha:
            reasons.append(f"{t}: e4b {v.get('e4b_sha')} is not the pin {e4b_sha}")
        if r[t].get("n_fast", 0) <= 0:
            reasons.append(f"{t}: enable_fast_train patched nothing")
        if r[t].get("n_offloaded", 0) <= 0:
            reasons.append(f"{t}: no offloaded expert layer")
        model = os.path.basename(str(r[t].get("model", "")).rstrip("/"))
        is_tiny = model == TINY
        if is_tiny != dry:
            reasons.append(f"{t}: a {'dry' if is_tiny else 'reading'} record in a {'dry run' if dry else 'reading'}")
        if not dry and QWEN_REV not in str(r[t].get("model", "")):
            reasons.append(f"{t}: the model path does not carry the pinned revision {QWEN_REV[:8]}")
        if not (r[t].get("n_fast") == r[t].get("n_offloaded") == n_reg):
            reasons.append(f"{t}: patched {r[t].get('n_fast')} / offloaded {r[t].get('n_offloaded')} layers, registered {n_reg}")
        if t[:2] == "A0" and r[t].get("loop") != "stock" or t[:2] == "L1" and r[t].get("loop") != "layer_major":
            reasons.append(f"{t}: loop {r[t].get('loop')} is not its arm's")
    stacks = {json.dumps({k: v for k, v in r[t].get("versions", {}).items() if k != "gpu"}, sort_keys=True) for t in TAGS}
    if len(stacks) != 1:
        reasons.append(f"the arms ran on {len(stacks)} different stacks")
    if len({r[t].get("model") for t in TAGS}) != 1 or len({json.dumps(r[t].get("recipe"), sort_keys=True) for t in TAGS}) != 1:
        reasons.append("the arms differ in model or recipe")
    s0 = {t: r[t].get("step0_hex") for t in TAGS}
    if len({json.dumps(v) for v in s0.values()}) != 1:
        reasons.append(f"step 0's per-micro-batch losses differ between arms: {s0}")
    for t in TAGS:
        want = ACCUM * (2 * n_reg - 1) if t[:2] == "A0" else 2 * n_reg - 1
        got = r[t].get("summary", {}).get("stagings_steady")
        if got != [want]:
            reasons.append(f"{t}: stagings per step after step 0 {got}, registered [{want}]")
    peak = {t: r[t].get("summary", {}).get("peak_alloc", 0) for t in TAGS}
    a0_peak = max(peak["A0a"], peak["A0b"])
    for t in ("L1a", "L1b"):
        if peak[t] > a0_peak + MEM_TERM_GIB * GIB:
            reasons.append(f"{t}: peak {peak[t] / GIB:.2f} GiB > A0's {a0_peak / GIB:.2f} + {MEM_TERM_GIB} GiB")
    if reasons:
        return {"verdict": "VOID", "reasons": reasons}

    med = {t: r[t]["summary"]["median_s"] for t in TAGS}
    selfp = {"A0b/A0a": _ratio(med["A0b"], med["A0a"]), "L1b/L1a": _ratio(med["L1b"], med["L1a"])}
    pairs = {"A0a/L1a": _ratio(med["A0a"], med["L1a"]), "A0b/L1b": _ratio(med["A0b"], med["L1b"])}
    s = min(pairs.values())
    held = {t: r[t].get("heldout") for t in TAGS}
    if any(h is None for h in held.values()):
        a0h = l1h = None                                       # REPORTING: a missing held-out never stops the verdict
    else:
        a0h = (held["A0a"] + held["A0b"]) / 2
        l1h = (held["L1a"] + held["L1b"]) / 2
    reporting = {"heldout_A0": a0h, "heldout_L1": l1h,
                 "heldout_ok": None if a0h is None else abs(l1h - a0h) <= HELD_TOL,
                 "in_band": BAND_LO <= s <= BAND_HI, "band": [BAND_LO, BAND_HI],
                 "peak_gib": {t: peak[t] / GIB for t in TAGS}}
    base = {"medians_s": med, "self_pairs": selfp, "pairs": pairs, "S": s, "reporting": reporting}
    noisy = [k for k, v in selfp.items() if not SELF_LO <= v <= SELF_HI]
    if noisy:
        return {"verdict": "NOISY", "reasons": [f"{k} = {selfp[k]:.4f} outside [{SELF_LO}, {SELF_HI}]" for k in noisy], **base}
    return {"verdict": "LICENSED" if s >= S_MIN else "NO_GAIN", "reasons": [], **base}


def _fake(tag, med, n=48, model=f"/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/{QWEN_REV}", e4b="e" * 40, peak=12 * GIB, step0=("0x1.0p+1", "0x1.1p+1"), loop=None, **kw):
    a0 = tag[:2] == "A0"
    steady = ACCUM * (2 * n - 1) if a0 else 2 * n - 1
    rec = {"tag": tag, "loop": loop or ("stock" if a0 else "layer_major"), "model": model, "n_fast": n, "n_offloaded": n,
           "step0_hex": list(step0), "heldout": 1.0, "recipe": {"seq": 2048},
           "versions": {"e4b_sha": e4b, "torch": "2.8.0+cu128", "gpu": "RTX 4090"},
           "summary": {"median_s": med, "stagings_steady": [steady], "peak_alloc": peak}}
    rec.update(kw)
    return rec


def self_test():
    E = "e" * 40
    ok = {"A0a": _fake("A0a", 10.5), "L1a": _fake("L1a", 6.0), "L1b": _fake("L1b", 6.05), "A0b": _fake("A0b", 10.4)}
    cases = []

    def case(name, recs, want, dry=False, e4b=E):
        v = reduce(recs, e4b, dry=dry)
        assert v["verdict"] == want, (name, v)
        cases.append(name)
        return v

    v = case("licensed", ok, "LICENSED")
    assert abs(v["S"] - 10.4 / 6.05) < 1e-12 and v["reporting"]["in_band"] and v["reporting"]["heldout_ok"]
    case("no gain", {**ok, "L1a": _fake("L1a", 9.8), "L1b": _fake("L1b", 9.75)}, "NO_GAIN")
    case("just licensed", {**ok, "L1a": _fake("L1a", 10.5 / 1.10), "L1b": _fake("L1b", 10.4 / 1.10)}, "LICENSED")
    case("just under", {**ok, "L1a": _fake("L1a", 10.5 / 1.0999), "L1b": _fake("L1b", 10.4 / 1.0999)}, "NO_GAIN")
    case("noisy A0", {**ok, "A0b": _fake("A0b", 11.0)}, "NOISY")
    case("noisy L1", {**ok, "L1b": _fake("L1b", 5.7)}, "NOISY")
    case("missing", {k: v for k, v in ok.items() if k != "L1b"}, "VOID")
    v = case("L1 OOM is VOID", {**ok, "L1a": {**ok["L1a"], "oom": "step 3"}}, "VOID")
    assert "OOM" in v["reasons"][0]
    case("pin", ok, "VOID", e4b="f" * 40)
    case("one arm off pin", {**ok, "A0b": _fake("A0b", 10.4, e4b="f" * 40)}, "VOID")
    case("no fast path", {**ok, "L1a": {**ok["L1a"], "n_fast": 0}}, "VOID")
    case("not offloaded", {**ok, "A0a": {**ok["A0a"], "n_offloaded": 0}}, "VOID")
    case("dry record in a reading", {**ok, "A0a": _fake("A0a", 10.5, model="/root/p137/p137-tiny")}, "VOID")
    case("reading record in a dry run", ok, "VOID", dry=True)
    dry = {t: _fake(t, m, model="/root/p137/p137-tiny", n=2) for t, m in
           (("A0a", 1.0), ("L1a", 0.8), ("L1b", 0.8), ("A0b", 1.0))}
    case("dry run reduces", dry, "LICENSED", dry=True)
    case("wrong loop", {**ok, "L1a": _fake("L1a", 6.0, loop="stock")}, "VOID")
    case("step-0 mismatch", {**ok, "L1b": _fake("L1b", 6.05, step0=("0x1.0p+1", "0x1.1000000000001p+1"))}, "VOID")
    case("stagings off (A0)", {**ok, "A0a": {**ok["A0a"], "summary": {**ok["A0a"]["summary"], "stagings_steady": [379, 380]}}}, "VOID")
    case("stagings off (L1)", {**ok, "L1a": {**ok["L1a"], "summary": {**ok["L1a"]["summary"], "stagings_steady": [96]}}}, "VOID")
    case("memory term", {**ok, "L1a": _fake("L1a", 6.0, peak=12 * GIB + int(4.26 * GIB))}, "VOID")
    case("memory inside the term", {**ok, "L1a": _fake("L1a", 6.0, peak=12 * GIB + int(4.24 * GIB))}, "LICENSED")
    case("stack differs", {**ok, "A0b": {**ok["A0b"], "versions": {**ok["A0b"]["versions"], "torch": "2.9.0"}}}, "VOID")
    case("recipe differs", {**ok, "A0b": {**ok["A0b"], "recipe": {"seq": 1024}}}, "VOID")
    v = case("held-out is reporting only", {**ok, "L1a": {**ok["L1a"], "heldout": 1.5}, "L1b": {**ok["L1b"], "heldout": 1.5}},
             "LICENSED")
    assert v["reporting"]["heldout_ok"] is False
    case("n = 47 (a layer not offloaded)", {t: _fake(t, m, n=47) for t, m in
                                             (("A0a", 10.5), ("L1a", 6.0), ("L1b", 6.05), ("A0b", 10.4))}, "VOID")
    case("n_fast != n_offloaded", {**ok, "L1a": {**ok["L1a"], "n_fast": 47}}, "VOID")
    case("unpinned revision", {**ok, "A0b": _fake("A0b", 10.4, model="/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/" + "0" * 40)}, "VOID")
    v = case("held-out missing is reporting only", {**ok, "A0a": {**ok["A0a"], "heldout": None}}, "LICENSED")
    assert v["reporting"]["heldout_ok"] is None
    v = case("out of band is reporting only", {**ok, "L1a": _fake("L1a", 4.0), "L1b": _fake("L1b", 4.0)}, "LICENSED")
    assert v["reporting"]["in_band"] is False
    print(f"p137_reduce self-test OK ({len(cases)} cases)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--e4b-sha")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    recs = {}
    for t in TAGS:
        f = os.path.join(a.dir, f"arm_{t}.json")
        recs[t] = json.load(open(f)) if os.path.exists(f) else None
    v = reduce(recs, a.e4b_sha, dry=a.dry)
    json.dump(v, open(a.out, "w"), indent=1)
    print("P137_VERDICT " + json.dumps({k: v.get(k) for k in ("verdict", "S", "self_pairs", "reasons")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
