#!/usr/bin/env python3
"""p144_reduce.py -- lane P144's reducer (bench/p144/PREREG-p144.md). Reads arm_{A0a,L1a,L1b,A0b}.json and writes the
verdict. The rule, first rung that applies:
  1. VOID: an arm record missing or OOM (an L1 OOM is VOID, never NO_GAIN); a record from another e4b commit than the
     pin, or the arms on different stacks or models; the fast path not patched or the experts not offloaded; a dry
     record in a reading or the reverse; a reading whose model path lacks the pinned revision; the patched and offloaded
     layer counts not both the registered n (48; the dry fixture's 2); step 0's per-micro-batch losses not
     bit-identical between every arm; the stagings per timed step not exactly ACCUM x (2n - 1) (A0) and 2n - 1 (L1)
     at that n; L1's peak device memory above A0's by more than MEM_TERM_GIB; (P137 Amendment 2, one process) an arm
     that did not start from the post-load adapter snapshot, or arms whose snapshots differ; an arm whose allocated memory
     at its start is more than MEM_START_TOL_MIB from the post-load baseline; step 0 not cold (A0a and A0b exactly
     M x (2n - 1) + 1, L1b exactly 2n, L1a above L1b: the drift guard's stock forwards run once, on L1a).
     P137 Amendment 3 replaces the L1a rung: the untimed warm-up runs one layer-major step after the snapshot (the adapters
     restored from it afterwards), so the guard runs there (its layer-major stagings above 2n, the same warm-up record
     in every arm) and L1a and L1b are both exactly 2n at step 0; every arm records exactly TIMED_REG timed steps.
  2. VOID (P137 Amendment 2): a self-pair outside [SELF_LO, SELF_HI] -- in one process it is the carry-over check.
  3. LICENSED iff S = min(A0a/L1a, A0b/L1b) >= S_MIN; otherwise NO_GAIN.
REPORTING (never changes the verdict): held-out |L1 - A0| <= HELD_TOL; S inside the registered band [BAND_LO, BAND_HI].

    python p144_reduce.py --dir RUN --out verdict.json --e4b-sha SHA [--dry]
    python p144_reduce.py --self-test
"""
import argparse
import json
import os
import sys

TAGS = ("A0a", "L1a", "L1b", "A0b")
SELF_LO, SELF_HI = 0.97, 1.03
S_MIN = 1.10
BAND_LO, BAND_HI = 1.5, 2.6
MEM_TERM_GIB = 4.25
HELD_TOL = 0.01
MEM_START_TOL_MIB = 16                   # P137 Amendment 2: arm-start allocated memory vs the post-load baseline
TIMED_REG = 32                           # P137 Amendment 3: timed steps per arm (was 8), fixed before any data
ACCUM = 4
QWEN = "mistralai/Mixtral-8x7B-Instruct-v0.1"
QWEN_REV = "fc7ac94680e38d7348cfa806e51218e6273104b0"
TINY = "p144-tiny"
N_LAYERS = {"reading": 32, "dry": 2}     # the registered n: Mixtral-8x7B's 32 MoE layers; the fixture's 2
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
    for t in TAGS:                                     # P137 Amendment 2: every arm from the same post-load adapters
        if r[t].get("adapter_sha_at_start") is None or r[t].get("adapter_sha_at_start") != r[t].get("adapter_sha_snapshot"):
            reasons.append(f"{t}: adapters at the arm's start do not match the post-load snapshot")
    if len({r[t].get("adapter_sha_snapshot") for t in TAGS}) != 1:
        reasons.append("the arms carry different adapter snapshots")
    for t in TAGS:
        m0, mb = r[t].get("mem_at_start"), r[t].get("mem_baseline")
        if m0 is None or mb is None or abs(m0 - mb) > MEM_START_TOL_MIB * 2 ** 20:
            reasons.append(f"{t}: allocated memory at the arm's start {m0} vs the post-load baseline {mb}")
    s0 = {t: r[t].get("summary", {}).get("stagings_step0") for t in TAGS}
    cold_a0, cold_l1 = ACCUM * (2 * n_reg - 1) + 1, 2 * n_reg
    if not (s0["A0a"] == s0["A0b"] == cold_a0 and s0["L1a"] == s0["L1b"] == cold_l1):
        reasons.append(f"step 0 not cold: stagings {s0}, registered A0 {cold_a0}, L1 {cold_l1}")
    wu = {t: json.dumps(r[t].get("warmup"), sort_keys=True) for t in TAGS}   # P137 Amendment 3: the guard ran in the warm-up
    lm = (r["A0a"].get("warmup") or {}).get("layer_major_stagings") or 0
    if len(set(wu.values())) != 1 or lm <= cold_l1:
        reasons.append(f"the warm-up did not run the layer-major step and its guard: {wu['A0a']} (needs above {cold_l1})")
    for t in TAGS:
        nt = len(r[t].get("summary", {}).get("timed_walls") or [])
        if nt != TIMED_REG or (r[t].get("recipe") or {}).get("timed") != TIMED_REG:
            reasons.append(f"{t}: {nt} timed steps (recipe {(r[t].get('recipe') or {}).get('timed')}), registered {TIMED_REG}")
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
    if noisy:                                        # P137 Amendment 2: in one process this is the carry-over check
        return {"verdict": "VOID", "reasons": [f"{k} = {selfp[k]:.4f} outside [{SELF_LO}, {SELF_HI}]" for k in noisy], **base}
    return {"verdict": "LICENSED" if s >= S_MIN else "NO_GAIN", "reasons": [], **base}


def _fake(tag, med, n=32, model=f"/root/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/{QWEN_REV}", e4b="e" * 40, peak=12 * GIB, step0=("0x1.0p+1", "0x1.1p+1"), loop=None, **kw):
    a0 = tag[:2] == "A0"
    steady = ACCUM * (2 * n - 1) if a0 else 2 * n - 1
    rec = {"tag": tag, "loop": loop or ("stock" if a0 else "layer_major"), "model": model, "n_fast": n, "n_offloaded": n,
           "adapter_sha_at_start": "a" * 64, "adapter_sha_snapshot": "a" * 64, "mem_at_start": 7 * GIB,
           "mem_baseline": 7 * GIB,
           "step0_hex": list(step0), "heldout": 1.0, "recipe": {"seq": 2048, "timed": TIMED_REG},
           "warmup": {"stock_stagings": 2 * n - 1, "layer_major_stagings": 2 * n + n * ACCUM},
           "versions": {"e4b_sha": e4b, "torch": "2.8.0+cu128", "gpu": "RTX 4090"},
           "summary": {"median_s": med, "stagings_steady": [steady], "peak_alloc": peak, "timed_walls": [med] * TIMED_REG,
                       "stagings_step0": (ACCUM * (2 * n - 1) + 1) if a0 else 2 * n}}
    rec.update(kw)
    return rec


def self_test():
    E = "e" * 40
    ok = {"A0a": _fake("A0a", 10.5), "L1a": _fake("L1a", 4.4), "L1b": _fake("L1b", 4.42), "A0b": _fake("A0b", 10.4)}
    cases = []

    def case(name, recs, want, dry=False, e4b=E):
        v = reduce(recs, e4b, dry=dry)
        assert v["verdict"] == want, (name, v)
        cases.append(name)
        return v

    v = case("licensed", ok, "LICENSED")
    assert abs(v["S"] - 10.4 / 4.42) < 1e-12 and v["reporting"]["in_band"] and v["reporting"]["heldout_ok"]
    case("no gain", {**ok, "L1a": _fake("L1a", 9.8), "L1b": _fake("L1b", 9.75)}, "NO_GAIN")
    case("just licensed", {**ok, "L1a": _fake("L1a", 10.5 / 1.10), "L1b": _fake("L1b", 10.4 / 1.10)}, "LICENSED")
    case("just under", {**ok, "L1a": _fake("L1a", 10.5 / 1.0999), "L1b": _fake("L1b", 10.4 / 1.0999)}, "NO_GAIN")
    case("A2: a self-pair off (A0) is VOID", {**ok, "A0b": _fake("A0b", 11.0)}, "VOID")
    case("A2: a self-pair off (L1) is VOID", {**ok, "L1b": _fake("L1b", 5.7)}, "VOID")
    case("A2: arm-start memory off the baseline", {**ok, "L1b": {**ok["L1b"], "mem_at_start": 7 * GIB + 17 * 2 ** 20}}, "VOID")
    case("A2: arm-start memory inside the tolerance", {**ok, "L1b": {**ok["L1b"], "mem_at_start": 7 * GIB + 15 * 2 ** 20}},
         "LICENSED")
    case("A2: a warm step 0 (L1b inherits residency)",
         {**ok, "L1b": {**ok["L1b"], "summary": {**ok["L1b"]["summary"], "stagings_step0": 63}}}, "VOID")
    case("A3: L1a carrying the guard's forwards (the warm-up skipped it)",
         {**ok, "L1a": {**ok["L1a"], "summary": {**ok["L1a"]["summary"], "stagings_step0": 192}}}, "VOID")
    case("A3: no layer-major warm-up", {t: {**v, "warmup": {"stock_stagings": 63}} for t, v in ok.items()}, "VOID")
    case("A3: warm-up without the guard", {t: {**v, "warmup": {**v["warmup"], "layer_major_stagings": 64}}
                                            for t, v in ok.items()}, "VOID")
    case("A3: arms carry different warm-ups", {**ok, "L1b": {**ok["L1b"], "warmup": {"stock_stagings": 1,
                                                 "layer_major_stagings": 192}}}, "VOID")
    case("A3: 8 timed steps (the old n)", {**ok, "A0b": {**ok["A0b"], "summary": {**ok["A0b"]["summary"],
                                         "timed_walls": [10.4] * 8}}}, "VOID")
    case("A3: the recipe's timed off", {t: {**v, "recipe": {"seq": 2048, "timed": 8}} for t, v in ok.items()}, "VOID")
    case("missing", {k: v for k, v in ok.items() if k != "L1b"}, "VOID")
    v = case("L1 OOM is VOID", {**ok, "L1a": {**ok["L1a"], "oom": "step 3"}}, "VOID")
    assert "OOM" in v["reasons"][0]
    case("pin", ok, "VOID", e4b="f" * 40)
    case("one arm off pin", {**ok, "A0b": _fake("A0b", 10.4, e4b="f" * 40)}, "VOID")
    case("no fast path", {**ok, "L1a": {**ok["L1a"], "n_fast": 0}}, "VOID")
    case("not offloaded", {**ok, "A0a": {**ok["A0a"], "n_offloaded": 0}}, "VOID")
    case("dry record in a reading", {**ok, "A0a": _fake("A0a", 10.5, model="/root/p144/p144-tiny")}, "VOID")
    case("reading record in a dry run", ok, "VOID", dry=True)
    dry = {t: _fake(t, m, model="/root/p144/p144-tiny", n=2) for t, m in
           (("A0a", 1.0), ("L1a", 0.8), ("L1b", 0.8), ("A0b", 1.0))}
    case("dry run reduces", dry, "LICENSED", dry=True)
    case("wrong loop", {**ok, "L1a": _fake("L1a", 4.4, loop="stock")}, "VOID")
    case("step-0 mismatch", {**ok, "L1b": _fake("L1b", 4.42, step0=("0x1.0p+1", "0x1.1000000000001p+1"))}, "VOID")
    case("stagings off (A0)", {**ok, "A0a": {**ok["A0a"], "summary": {**ok["A0a"]["summary"], "stagings_steady": [251, 252]}}}, "VOID")
    case("stagings off (L1)", {**ok, "L1a": {**ok["L1a"], "summary": {**ok["L1a"]["summary"], "stagings_steady": [64]}}}, "VOID")
    case("memory term", {**ok, "L1a": _fake("L1a", 4.4, peak=12 * GIB + int(4.26 * GIB))}, "VOID")
    case("memory inside the term", {**ok, "L1a": _fake("L1a", 4.4, peak=12 * GIB + int(4.24 * GIB))}, "LICENSED")
    case("stack differs", {**ok, "A0b": {**ok["A0b"], "versions": {**ok["A0b"]["versions"], "torch": "2.9.0"}}}, "VOID")
    case("recipe differs", {**ok, "A0b": {**ok["A0b"], "recipe": {"seq": 1024}}}, "VOID")
    v = case("held-out is reporting only", {**ok, "L1a": {**ok["L1a"], "heldout": 1.5}, "L1b": {**ok["L1b"], "heldout": 1.5}},
             "LICENSED")
    assert v["reporting"]["heldout_ok"] is False
    case("n = 31 (a layer not offloaded)", {t: _fake(t, m, n=31) for t, m in
                                             (("A0a", 10.5), ("L1a", 6.0), ("L1b", 6.05), ("A0b", 10.4))}, "VOID")
    case("n_fast != n_offloaded", {**ok, "L1a": {**ok["L1a"], "n_fast": 47}}, "VOID")
    case("unpinned revision", {**ok, "A0b": _fake("A0b", 10.4, model="/root/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/" + "0" * 40)}, "VOID")
    v = case("held-out missing is reporting only", {**ok, "A0a": {**ok["A0a"], "heldout": None}}, "LICENSED")
    assert v["reporting"]["heldout_ok"] is None
    case("A2: an arm off the snapshot", {**ok, "L1b": {**ok["L1b"], "adapter_sha_at_start": "b" * 64}}, "VOID")
    case("A2: snapshots differ", {**ok, "A0b": {**ok["A0b"], "adapter_sha_at_start": "c" * 64,
                                                 "adapter_sha_snapshot": "c" * 64}}, "VOID")
    case("A2: a record without the hash", {**ok, "A0a": {k: v for k, v in ok["A0a"].items() if k != "adapter_sha_at_start"}},
         "VOID")
    v = case("out of band is reporting only", {**ok, "L1a": _fake("L1a", 8.0), "L1b": _fake("L1b", 8.0)}, "LICENSED")
    assert v["reporting"]["in_band"] is False
    print(f"p144_reduce self-test OK ({len(cases)} cases)")
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
    print("P144_VERDICT " + json.dumps({k: v.get(k) for k in ("verdict", "S", "self_pairs", "reasons")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
