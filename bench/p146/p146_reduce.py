#!/usr/bin/env python3
"""p146_reduce.py -- lane P146's reducer (bench/p146/PREREG-p146.md). Reads arm_{A0a,L1a,L1b,A0b}.json (one fresh
``python -m experts4bit_qlora.train`` process each) and writes the verdict. The rule, first rung that applies:
  1. VOID: an arm record missing (a process that crashed, OOMed at the budget floor or timed out writes none); a record
     from another e4b commit than the pin, or the arms on different stacks or models; a dry record in a reading or the
     reverse; a reading whose model path lacks the pinned revision; the fused path not patched or the experts not
     offloaded on exactly the registered n layers (48; the dry fixture's 2); e4b's reentrant checkpoint not on all n;
     an arm's switches or recipe not the registered CLI recipe (RECIPE, with LAYER_MAJOR its arm's), or the token
     budget halved by train.py's OOM back-off; the alpaca rows not requested as train.py requests them; the arms not
     starting from the same adapters (sha256 at the optimizer's construction), or their BEFORE held-out losses not
     bit-identical; step 0's loss not bit-identical between every arm; step 0 not cold (A0a and A0b exactly
     STEP0_A0 = M x (2n - 1) + 1; L1a equal to L1b and above 2n: each fresh L1 process runs the drift guard's stock
     forwards in its first step); the stagings per step after step 0 not exactly M x (2n - 1) (A0) and 2n - 1 (L1);
     an arm with other than TIMED_REG timed steps; L1's peak above A0's by more than MEM_TERM_GIB.
  2. VOID: a self-pair outside [SELF_LO, SELF_HI]. Across fresh processes it reads process-to-process variance.
  3. LICENSED iff S = min(A0a/L1a, A0b/L1b) >= S_MIN; otherwise NO_GAIN.
REPORTING (never changes the verdict): held-out (AFTER) |L1 - A0| <= HELD_TOL; S inside the band [BAND_LO, BAND_HI].

    python p146_reduce.py --dir RUN --out verdict.json --e4b-sha SHA [--dry]
    python p146_reduce.py --self-test
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
TIMED_REG = 32
ACCUM = 4
# train.py's own recipe at its defaults, with the lane's STEPS (2 warm + 32 timed) and DO_GEN=0
RECIPE = {"seq": 192, "token_budget": 2048, "grad_accum": ACCUM, "steps": 34, "warm": 2, "timed": TIMED_REG, "lr": 2e-4,
          "r": 8, "alpha": 16, "n_train": 2000, "eval_every": 50, "seed": 0, "train_attention": True,
          "train_experts": True, "fast_train": True, "offload": True}
DATASET_CALLS = ["train[:2000]", "train[2000:2064]"]
QWEN = "Qwen/Qwen3-30B-A3B"
QWEN_REV = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
TINY = "p146-tiny"
N_LAYERS = {"reading": 48, "dry": 2}     # the registered n: Qwen3-30B-A3B's 48 MoE layers; the fixture's 2
GIB = 2 ** 30


def _ratio(a, b):
    return a / b if b else float("inf")


def reduce(recs, e4b_sha, dry=False):
    """recs: tag -> record dict, or None when the file is missing. Returns the verdict dict."""
    missing = [t for t in TAGS if recs.get(t) is None]
    if missing:
        return {"verdict": "VOID", "reasons": [f"arm record missing: {t}" for t in missing]}
    r = recs
    reasons = []
    n_reg = N_LAYERS["dry" if dry else "reading"]
    for t in TAGS:
        lm = t[:2] == "L1"
        v = r[t].get("versions", {})
        if v.get("e4b_sha") != e4b_sha:
            reasons.append(f"{t}: e4b {v.get('e4b_sha')} is not the pin {e4b_sha}")
        model = os.path.basename(str(r[t].get("model", "")).rstrip("/"))
        is_tiny = model == TINY
        if is_tiny != dry:
            reasons.append(f"{t}: a {'dry' if is_tiny else 'reading'} record in a {'dry run' if dry else 'reading'}")
        if not dry and QWEN_REV not in str(r[t].get("model", "")):
            reasons.append(f"{t}: the model path does not carry the pinned revision {QWEN_REV[:8]}")
        if not (r[t].get("n_fast") == r[t].get("n_offloaded") == n_reg):
            reasons.append(f"{t}: patched {r[t].get('n_fast')} / offloaded {r[t].get('n_offloaded')} layers, registered {n_reg}")
        ck = r[t].get("ckpt") or {}
        if ck.get("layers") != n_reg or ck.get("mode") != "reentrant" or ck.get("skipped"):
            reasons.append(f"{t}: checkpoint {ck}, registered e4b's reentrant checkpoint on {n_reg} layers")
        if r[t].get("loop") != ("layer_major" if lm else "stock"):
            reasons.append(f"{t}: loop {r[t].get('loop')} is not its arm's")
        want_sw = {"OFFLOAD_EXPERTS": "1", "FAST_TRAIN": None, "LAYER_MAJOR": "1"} if lm else \
            {"OFFLOAD_EXPERTS": "1", "FAST_TRAIN": "1", "LAYER_MAJOR": None}
        if r[t].get("switches") != want_sw:
            reasons.append(f"{t}: switches {r[t].get('switches')}, registered {want_sw}")
        want_rec = {**RECIPE, "layer_major": lm}
        if r[t].get("recipe") != want_rec:
            reasons.append(f"{t}: recipe {r[t].get('recipe')} is not the registered CLI recipe {want_rec}")
        if r[t].get("token_budget_effective") != RECIPE["token_budget"]:
            reasons.append(f"{t}: train.py's OOM back-off ran (effective budget {r[t].get('token_budget_effective')})")
        if r[t].get("dataset_calls") != DATASET_CALLS:
            reasons.append(f"{t}: alpaca requested as {r[t].get('dataset_calls')}, registered {DATASET_CALLS}")
        nt = len(r[t].get("summary", {}).get("timed_walls") or [])
        if nt != TIMED_REG:
            reasons.append(f"{t}: {nt} timed steps, registered {TIMED_REG}")
    for key, what in (("adapter_sha_at_start", "adapters at the optimizer's construction"),
                      ("eval_before_hex", "BEFORE held-out losses"), ("step0_loss_hex", "step-0 losses")):
        vals = {t: r[t].get(key) for t in TAGS}
        if None in vals.values() or len(set(vals.values())) != 1:
            reasons.append(f"the arms' {what} differ: {vals}")
    stacks = {json.dumps({k: v for k, v in r[t].get("versions", {}).items() if k != "gpu"}, sort_keys=True) for t in TAGS}
    if len(stacks) != 1:
        reasons.append(f"the arms ran on {len(stacks)} different stacks")
    if len({r[t].get("model") for t in TAGS}) != 1:
        reasons.append("the arms differ in model")
    s0 = {t: r[t].get("summary", {}).get("stagings_step0") for t in TAGS}
    cold_a0, cold_l1 = ACCUM * (2 * n_reg - 1) + 1, 2 * n_reg
    if not (s0["A0a"] == s0["A0b"] == cold_a0 and s0["L1a"] == s0["L1b"] and (s0["L1a"] or 0) > cold_l1):
        reasons.append(f"step 0 not cold: stagings {s0}, registered A0 {cold_a0}, L1 equal and above {cold_l1}")
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
                 "peak_gib": {t: peak[t] / GIB for t in TAGS}, "load_s": {t: r[t].get("load_s") for t in TAGS}}
    base = {"medians_s": med, "self_pairs": selfp, "pairs": pairs, "S": s, "reporting": reporting}
    noisy = [k for k, v in selfp.items() if not SELF_LO <= v <= SELF_HI]
    if noisy:
        return {"verdict": "VOID", "reasons": [f"{k} = {selfp[k]:.4f} outside [{SELF_LO}, {SELF_HI}]" for k in noisy], **base}
    return {"verdict": "LICENSED" if s >= S_MIN else "NO_GAIN", "reasons": [], **base}


def _fake(tag, med, n=48, model=f"/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/{QWEN_REV}",
          e4b="e" * 40, peak=8 * GIB, step0="0x1.0p+1", loop=None, **kw):
    lm = tag[:2] == "L1"
    steady = 2 * n - 1 if lm else ACCUM * (2 * n - 1)
    rec = {"tag": tag, "loop": loop or ("layer_major" if lm else "stock"), "model": model, "n_fast": n, "n_offloaded": n,
           "ckpt": {"layers": n, "skipped": None, "mode": "reentrant"},
           "switches": {"OFFLOAD_EXPERTS": "1", "FAST_TRAIN": None, "LAYER_MAJOR": "1"} if lm else
           {"OFFLOAD_EXPERTS": "1", "FAST_TRAIN": "1", "LAYER_MAJOR": None},
           "recipe": {**RECIPE, "layer_major": lm}, "token_budget_effective": 2048, "dataset_calls": list(DATASET_CALLS),
           "adapter_sha_at_start": "a" * 64, "eval_before_hex": "0x1.8p+1", "step0_loss_hex": step0, "heldout": 1.0,
           "load_s": 65.0, "versions": {"e4b_sha": e4b, "torch": "2.8.0+cu128", "gpu": "RTX 4090"},
           "summary": {"median_s": med, "stagings_steady": [steady], "peak_alloc": peak, "timed_walls": [med] * TIMED_REG,
                       "stagings_step0": 2 * n + 240 if lm else ACCUM * (2 * n - 1) + 1}}
    rec.update(kw)
    return rec


def self_test():
    E = "e" * 40
    ok = {"A0a": _fake("A0a", 6.0), "L1a": _fake("L1a", 3.0), "L1b": _fake("L1b", 3.02), "A0b": _fake("A0b", 5.95)}
    cases = []

    def case(name, recs, want, dry=False, e4b=E):
        v = reduce(recs, e4b, dry=dry)
        assert v["verdict"] == want, (name, v)
        cases.append(name)
        return v

    def sm(t, **kw):
        return {**ok[t], "summary": {**ok[t]["summary"], **kw}}

    v = case("licensed", ok, "LICENSED")
    assert abs(v["S"] - 5.95 / 3.02) < 1e-12 and v["reporting"]["in_band"] and v["reporting"]["heldout_ok"]
    case("no gain", {**ok, "L1a": _fake("L1a", 5.8), "L1b": _fake("L1b", 5.75)}, "NO_GAIN")
    case("just licensed", {**ok, "L1a": _fake("L1a", 6.0 / 1.10), "L1b": _fake("L1b", 5.95 / 1.10)}, "LICENSED")
    case("just under", {**ok, "L1a": _fake("L1a", 6.0 / 1.0999), "L1b": _fake("L1b", 5.95 / 1.0999)}, "NO_GAIN")
    case("a self-pair off (A0)", {**ok, "A0b": _fake("A0b", 6.3)}, "VOID")
    case("a self-pair off (L1)", {**ok, "L1b": _fake("L1b", 3.2)}, "VOID")
    case("missing (a crashed or timed-out process)", {k: v for k, v in ok.items() if k != "L1b"}, "VOID")
    case("pin", ok, "VOID", e4b="f" * 40)
    case("one arm off pin", {**ok, "A0b": _fake("A0b", 5.95, e4b="f" * 40)}, "VOID")
    case("no fast path", {**ok, "L1a": {**ok["L1a"], "n_fast": 0}}, "VOID")
    case("not offloaded", {**ok, "A0a": {**ok["A0a"], "n_offloaded": 0}}, "VOID")
    case("n = 47", {t: _fake(t, m, n=47) for t, m in (("A0a", 6.0), ("L1a", 3.0), ("L1b", 3.02), ("A0b", 5.95))}, "VOID")
    case("checkpoint not routed (Hugging Face's back)", {**ok, "A0a": {**ok["A0a"], "ckpt": {"layers": 0, "skipped": None,
                                                                                         "mode": None}}}, "VOID")
    case("checkpoint non-reentrant", {**ok, "L1b": {**ok["L1b"], "ckpt": {"layers": 48, "skipped": None,
                                                                          "mode": "non-reentrant"}}}, "VOID")
    case("checkpoint skipped", {**ok, "L1a": {**ok["L1a"], "ckpt": {"layers": 48, "skipped": "dense offload",
                                                                    "mode": "reentrant"}}}, "VOID")
    case("dry record in a reading", {**ok, "A0a": _fake("A0a", 6.0, model="/root/p146/p146-tiny")}, "VOID")
    case("reading record in a dry run", ok, "VOID", dry=True)
    dry = {t: _fake(t, m, model="/root/p146/p146-tiny", n=2) for t, m in
           (("A0a", 1.0), ("L1a", 0.8), ("L1b", 0.8), ("A0b", 1.0))}
    case("dry run reduces", dry, "LICENSED", dry=True)
    case("unpinned revision", {**ok, "A0b": _fake("A0b", 5.95, model="/x/models--Qwen--Qwen3-30B-A3B/snapshots/" + "0" * 40)},
         "VOID")
    case("wrong loop", {**ok, "L1a": _fake("L1a", 3.0, loop="stock")}, "VOID")
    case("A0 without FAST_TRAIN (the reference loop)", {**ok, "A0a": {**ok["A0a"], "switches": {
        "OFFLOAD_EXPERTS": "1", "FAST_TRAIN": None, "LAYER_MAJOR": None}}}, "VOID")
    case("L1 with FAST_TRAIN=0", {**ok, "L1b": {**ok["L1b"], "switches": {
        "OFFLOAD_EXPERTS": "1", "FAST_TRAIN": "0", "LAYER_MAJOR": "1"}}}, "VOID")
    case("recipe off the CLI default (SEQ)", {**ok, "A0b": {**ok["A0b"], "recipe": {**ok["A0b"]["recipe"], "seq": 2048}}},
         "VOID")
    case("recipe's layer_major not its arm's", {**ok, "L1a": {**ok["L1a"], "recipe": {**RECIPE, "layer_major": False}}},
         "VOID")
    case("OOM back-off halved the budget", {**ok, "L1a": {**ok["L1a"], "token_budget_effective": 1024}}, "VOID")
    case("alpaca requested otherwise", {**ok, "A0a": {**ok["A0a"], "dataset_calls": ["train[:400]"]}}, "VOID")
    case("adapters differ at the start", {**ok, "L1b": {**ok["L1b"], "adapter_sha_at_start": "b" * 64}}, "VOID")
    case("no adapter hash", {**ok, "A0a": {**ok["A0a"], "adapter_sha_at_start": None}}, "VOID")
    case("BEFORE held-out differs", {**ok, "A0b": {**ok["A0b"], "eval_before_hex": "0x1.8000000000001p+1"}}, "VOID")
    case("step-0 loss differs", {**ok, "L1a": _fake("L1a", 3.0, step0="0x1.0000000000001p+1")}, "VOID")
    case("stack differs", {**ok, "A0b": {**ok["A0b"], "versions": {**ok["A0b"]["versions"], "torch": "2.9.0"}}}, "VOID")
    case("model differs", {**ok, "A0b": {**ok["A0b"], "model": ok["A0b"]["model"] + "/"}}, "VOID")
    case("A0 step 0 warm", {**ok, "A0a": sm("A0a", stagings_step0=380)}, "VOID")
    case("L1 step 0 without the guard", {**ok, "L1a": sm("L1a", stagings_step0=96), "L1b": sm("L1b", stagings_step0=96)},
         "VOID")
    case("L1 step 0 differs between processes", {**ok, "L1b": sm("L1b", stagings_step0=300)}, "VOID")
    case("stagings off (A0)", {**ok, "A0a": sm("A0a", stagings_steady=[379, 380])}, "VOID")
    case("stagings off (L1)", {**ok, "L1a": sm("L1a", stagings_steady=[96])}, "VOID")
    case("8 timed steps", {**ok, "A0b": sm("A0b", timed_walls=[5.95] * 8)}, "VOID")
    case("memory term", {**ok, "L1a": _fake("L1a", 3.0, peak=8 * GIB + int(4.26 * GIB))}, "VOID")
    case("memory inside the term", {**ok, "L1a": _fake("L1a", 3.0, peak=8 * GIB + int(4.24 * GIB))}, "LICENSED")
    v = case("held-out is reporting only", {**ok, "L1a": {**ok["L1a"], "heldout": 1.5}, "L1b": {**ok["L1b"], "heldout": 1.5}},
             "LICENSED")
    assert v["reporting"]["heldout_ok"] is False
    v = case("held-out missing is reporting only", {**ok, "A0a": {**ok["A0a"], "heldout": None}}, "LICENSED")
    assert v["reporting"]["heldout_ok"] is None
    v = case("out of band is reporting only", {**ok, "L1a": _fake("L1a", 2.0), "L1b": _fake("L1b", 2.0)}, "LICENSED")
    assert v["reporting"]["in_band"] is False
    print(f"p146_reduce self-test OK ({len(cases)} cases)")
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
    print("P146_VERDICT " + json.dumps({k: v.get(k) for k in ("verdict", "S", "self_pairs", "reasons")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
