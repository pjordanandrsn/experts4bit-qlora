#!/usr/bin/env python3
"""fam_reduce.py -- lane FAM's registered rule (bench/fam/PREREG-fam.md; e4b#1362). Reads the box's records and writes
one verdict per (family, ON config): the first rung that applies.

1. VOID -- an integrity fault: a record missing or not ok; another model revision, e4b or grouped-nf4-gemm commit; a
   cell missing; the configs scored different windows; an arm short of its windows; a census off its registration (OFF
   all zero); a pass whose decode attention calls, decode-shaped forwards or glue-kernel calls are not the registered
   count (OFF passes call no glue kernel); a wrapped arm whose wrapper did not touch every decode attention call; the
   expert store not the path's (the default path: no int4 / MXFP4 store); no peak memory; a cell with no floor draw
   (every floor arm bit-identical to R); ``mutant_scale`` passing the gate in any cell (the gate cannot fail).
2. UNRESOLVED -- the graded mutant ``mut098`` passes the gate in any gated cell: the instrument lacks resolution at that
   size, and no gate is licensed for the family.
3. FAIL -- the ON config fails the gate in any gated cell.
4. PASS -- otherwise.

The floor of a (text, shape): every floor draw (``chunk``, ``half``, ``split1``, and ``rep`` when it did not repeat R
bit for bit) on every set. ``B_f`` is the largest |bias|, ``S_f`` the largest spread, ``A_f`` the smallest argmax
agreement: the WORST draw. An arm passes a cell iff on EVERY set (its worst of three):
  |bias| <= B_f + BIAS_MARGIN, spread <= SPREAD_MULT * max(S_f, SPREAD_MIN), agree >= A_f - AGREE_MARGIN,
  and the absolute backstop |bias| <= BACKSTOP_BIAS, agree >= BACKSTOP_AGREE.
bias = mean over windows of d, d = the arm's mean continuation NLL minus R's; spread = mean |d|; agree = mean argmax
agreement with R; kl = mean KL(R || arm) (reported).

The anchor (gpt-oss on SC2g's path, wikitext, shape 12, set A: Phase C's setting) is reported, never gated.

    python fam_reduce.py --dir RUN --out verdict.json --e4b-sha SHA [--proof]
    python fam_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fam_box  # noqa: E402

MODELS = {
    "granite": ("ibm-granite/granite-3.1-3b-a800m-instruct", "a02780686e08a03fe0d2679a293b5c74a90efa89"),
    "gptoss": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
    "qw36": ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
}
FAMILY_CONFIGS = {"granite": ("OFF", "ON_auto"), "gptoss": ("OFF", "ON_glue", "ON_r2", "ON_epi", "ON_auto"),
                  "qw36": ("OFF", "ON_auto")}
ANCHOR = {"family": "gptoss", "configs": ("OFF", "ON_auto"), "cell": ("wikitext", 12, "A")}
#: the proving rental (PREREG "The premise and the proving rental"): Granite alone, every process kind, 32 positions
PROOF = {"families": ("granite",), "configs": {"granite": ("OFF", "ON_epi", "ON_auto")}, "anchor_family": "granite",
         "cont": 32}
CONT = 128                                    # registered teacher-forced positions per window (the reading)
#: registered prompt, prefill chunk and floor chunk: every prefill forward carries more than 64 rows, so only the decode
#: steps (and a hybrid's one-token warm-up) are decode-shaped -- the per-step count below depends on it
PREFILL = (512, 512, 256)
PHASE_C_ANCHOR_AGREE = 0.924                  # Phase C's SANE argmax agreement on gpt-oss (SC2g path, T == 12, set A)

ATTN_LAYERS = {"granite": 32, "gptoss": 24, "qw36": 10}
#: decode-shaped forwards per pass beyond the decode steps: a hybrid's one-token linear-state warm-up, once per padded
#: pass (``PagedModelRunner._warm_linear_state``; Phase C's qw36 receipts: 128 router-epilogue forwards a pass)
EXTRA_DECODE_FORWARDS = {"granite": 0, "gptoss": 0, "qw36": 1}
CENSUS = {
    ("granite", "OFF"): [0, 0, [0, 0], 0], ("granite", "ON_auto"): [0, 65, [32, 32], 32],
    ("granite", "ON_epi"): [0, 0, [0, 0], 32],
    ("gptoss", "OFF"): [0, 0, [0, 0], 0], ("gptoss", "ON_glue"): [0, 49, [0, 0], 0],
    ("gptoss", "ON_r2"): [0, 0, [24, 0], 0], ("gptoss", "ON_epi"): [0, 0, [0, 0], 24],
    ("gptoss", "ON_auto"): [0, 49, [24, 0], 24],
    ("qw36", "OFF"): [0, 0, [0, 0], 0], ("qw36", "ON_auto"): [0, 0, [0, 0], 40],
}
#: glue-kernel calls per decode-shaped forward (P115's KernelCounters names)
PER_STEP = {
    ("granite", "ON_auto"): {"rmsnorm_rows": 33, "rmsnorm_resid_rows": 32, "scaled_resid_add_rows": 32,
                             "rope_heads": 64, "router_epilogue": 32},
    ("granite", "ON_epi"): {"router_epilogue": 32},
    ("gptoss", "ON_glue"): {"rmsnorm_rows": 49},
    ("gptoss", "ON_r2"): {"rmsnorm_resid_rows": 24},
    ("gptoss", "ON_epi"): {"router_epilogue": 24},
    ("gptoss", "ON_auto"): {"rmsnorm_rows": 25, "rmsnorm_resid_rows": 24, "router_epilogue": 24},
    ("qw36", "ON_auto"): {"router_epilogue": 40},
}
CENSUS_KEYS = fam_box.CENSUS_KEYS

BIAS_MARGIN, SPREAD_MULT, SPREAD_MIN, AGREE_MARGIN = 0.010, 2.0, 0.005, 0.005
BACKSTOP_BIAS, BACKSTOP_AGREE = 0.020, 0.90
WINDOWS_PER_SET = 12
FLOOR = fam_box.FLOOR_ARMS


def _census(rec):
    return [rec["census"].get(k) for k in CENSUS_KEYS]


def stats(arm_rows, ref_rows):
    """bias, spread, agree, kl and identity of one arm against R over one cell's windows."""
    ref = {x["window"]: x["nll"] for x in ref_rows}
    d = [x["nll"] - ref[x["window"]] for x in arm_rows]
    kl = [x.get("kl", 0.0) for x in arm_rows]
    agree = [x["argmax_agree"] for x in arm_rows]
    n = len(d)
    return {"n": n, "bias": sum(d) / n, "spread": sum(abs(v) for v in d) / n, "agree": sum(agree) / n,
            "kl": sum(kl) / n, "identical": all(v == 0.0 for v in d) and all(v == 0.0 for v in kl)
            and all(v == 1.0 for v in agree)}


def passes(s, floor):
    """The registered gate on one set's stats against the (text, shape) floor."""
    return (abs(s["bias"]) <= floor["B"] + BIAS_MARGIN and s["spread"] <= SPREAD_MULT * max(floor["S"], SPREAD_MIN)
            and s["agree"] >= floor["A"] - AGREE_MARGIN and abs(s["bias"]) <= BACKSTOP_BIAS
            and s["agree"] >= BACKSTOP_AGREE)


def _per_window(rec, cell, arm):
    """The arm's per-window rows in one cell (base record or a wrapped extra arm), its engagement list, and the
    wrapped-call count (None for base arms)."""
    c = rec["cells"][cell]
    text = cell.split("|")[0]
    if arm in c["extra"]:
        r = c["extra"][arm]["record"]
        return r["per_window"][text][arm], r["engagement"][text][arm], c["extra"][arm]["wrapped_calls"], r
    return c["base"]["per_window"][text].get(arm), c["base"]["engagement"][text].get(arm), None, c["base"]


def _check_engagement(fam, config, rec, cell, arm, why):
    """Every pass of ``arm`` in ``cell``: decode attention calls, decode-shaped forwards and glue-kernel calls."""
    rows, eng, wrapped, sub = _per_window(rec, cell, arm)
    text, shape, _ = cell.split("|")
    shape = int(shape)
    P, C = sub["prompt"], sub["cont"]
    chunk = sub["floor_chunk"] if arm == "chunk" else sub["chunk"]
    halves = 2 if arm == "half" else 1
    per_pass_windows = shape
    step = PER_STEP.get((fam, config), {}) if config != "OFF" else {}
    total_decode = 0
    for e in eng or []:
        want_calls = (C - 1) * ATTN_LAYERS[fam] * halves
        if e.get("decode_calls") != want_calls:
            why.append(f"{config} {cell} {arm}: decode attention calls {e.get('decode_calls')} != {want_calls}")
        prefill = per_pass_windows * math.ceil(P / chunk)
        dec = e.get("forwards", -1) - prefill
        want_dec = (C - 1) * halves + EXTRA_DECODE_FORWARDS[fam]
        if dec != want_dec:
            why.append(f"{config} {cell} {arm}: decode-shaped forwards {dec} != {want_dec}")
        k = {n: v for n, v in (e.get("kernels") or {}).items() if v}
        want_k = {n: v * want_dec for n, v in step.items()}
        if k != want_k:
            why.append(f"{config} {cell} {arm}: glue-kernel calls {k} != {want_k}")
        total_decode += e.get("decode_calls") or 0
    if wrapped is not None and wrapped != total_decode:
        why.append(f"{config} {cell} {arm}: the wrapper touched {wrapped} of {total_decode} decode attention calls")


def family_checks(fam, recs, e4b_sha, cells, why, configs=None, cont=CONT):
    """VOID checks on one family's default-path records; returns the per-cell windows digests."""
    model, rev = MODELS[fam]
    gnf4 = set()
    digests = {}
    for config in configs or FAMILY_CONFIGS[fam]:
        rec = recs.get(config)
        if not rec or rec.get("status") != "ok":
            why.append(f"{config}: record missing or not ok")
            continue
        if (rec.get("model"), rec.get("revision")) != (model, rev):
            why.append(f"{config}: model {rec.get('model')}@{rec.get('revision')}, registered {model}@{rev}")
        if rec.get("e4b_sha") != e4b_sha:
            why.append(f"{config}: e4b {rec.get('e4b_sha')} != {e4b_sha}")
        gnf4.add(rec.get("gnf4_sha"))
        if rec.get("path") != "default":
            why.append(f"{config}: path {rec.get('path')}")
        store = rec.get("store") or {}
        if store.get("int4_expert_layers") or store.get("int4_store_kinds"):
            why.append(f"{config}: the default path carries an int4 / MXFP4 store {store.get('int4_store_kinds')}")
        if not isinstance(rec.get("max_mem_gb"), (int, float)):
            why.append(f"{config}: no peak memory")
        if _census(rec) != CENSUS[(fam, config)]:
            why.append(f"{config}: census {_census(rec)} != {CENSUS[(fam, config)]}")
        for cell in cells:
            c = rec.get("cells", {}).get(cell)
            if c is None:
                why.append(f"{config}: cell {cell} missing")
                continue
            text, shape, set_ = cell.split("|")
            base = c["base"]
            if base.get("cont") != cont:
                why.append(f"{config} {cell}: {base.get('cont')} teacher-forced positions, registered {cont}")
            if (base.get("prompt"), base.get("chunk"), base.get("floor_chunk")) != PREFILL:
                why.append(f"{config} {cell}: prompt / chunk / floor chunk "
                           f"{(base.get('prompt'), base.get('chunk'), base.get('floor_chunk'))}, registered {PREFILL}")
            if base.get("group") != int(shape):
                why.append(f"{config} {cell}: group {base.get('group')} != {shape}")
            dg = base.get("windows_sha256", {}).get(text)
            if digests.setdefault(cell, dg) != dg:
                why.append(f"{config} {cell}: windows differ from the other configs'")
            arms = (fam_box.off_arms(int(shape)) + fam_box.extra_arms(set_)) if config == "OFF" else ("ON",)
            for arm in arms:
                rows, eng, wrapped, _ = _per_window(rec, cell, arm)
                want = int(shape) if arm == "rep" else WINDOWS_PER_SET
                if not rows or len(rows) != want or sorted(x["window"] for x in rows) != list(range(want)):
                    why.append(f"{config} {cell} {arm}: {len(rows or [])} windows, registered {want}")
                    continue
                _check_engagement(fam, config, rec, cell, arm, why)
    if len(gnf4) > 1:
        why.append(f"grouped-nf4-gemm commits differ across processes: {sorted(map(str, gnf4))}")
    return digests


def floors(off, cells):
    """``{(text, shape): {"B", "S", "A", "K", "draws"}}`` from the OFF record's floor arms; a draw bit-identical to R is
    not a draw."""
    out = {}
    for cell in cells:
        text, shape, set_ = cell.split("|")
        ref = off["cells"][cell]["base"]["per_window"][text]["R"]
        f = out.setdefault((text, int(shape)), {"B": 0.0, "S": 0.0, "A": 1.0, "K": 0.0, "draws": [], "identical": []})
        for arm in FLOOR:
            if arm == "half" and int(shape) == 1:
                continue
            rows, _e, _w, _s = _per_window(off, cell, arm)
            if not rows:
                continue
            s = stats(rows, ref)
            if s["identical"]:
                f["identical"].append(f"{arm}/{set_}")
                continue
            f["draws"].append({"arm": arm, "set": set_, **{k: s[k] for k in ("bias", "spread", "agree", "kl")}})
            f["B"], f["S"] = max(f["B"], abs(s["bias"])), max(f["S"], s["spread"])
            f["A"], f["K"] = min(f["A"], s["agree"]), max(f["K"], s["kl"])
    return out


def judge(rec, arm, cells, fl):
    """Per (text, shape): pass/fail of ``arm`` (worst of the sets present) with each set's stats."""
    out = {}
    for cell in cells:
        text, shape, set_ = cell.split("|")
        rows, _e, _w, _s = _per_window(rec, cell, arm)
        if rows is None:
            continue
        ref_rows = rec["_ref"][cell]
        s = stats(rows, ref_rows)
        o = out.setdefault(f"{text}|{shape}", {"pass": True, "sets": {}})
        ok = passes(s, fl[(text, int(shape))])
        o["sets"][set_] = {**{k: round(s[k], 6) for k in ("bias", "spread", "agree", "kl")}, "pass": ok}
        o["pass"] = o["pass"] and ok
    return out


def reduce_family(fam, recs, e4b_sha, proof=False):
    cells = [fam_box.cell_key(t, s, n) for t in fam_box.TEXTS for s in fam_box.SHAPES for n in fam_box.SETS]
    configs = PROOF["configs"][fam] if proof else FAMILY_CONFIGS[fam]
    why = []
    family_checks(fam, recs, e4b_sha, cells, why, configs, PROOF["cont"] if proof else CONT)
    report = {"configs": {}}
    if why:
        return {"verdict": {c: "VOID" for c in configs if c != "OFF"}, "why": why, "report": report}
    off = recs["OFF"]
    refs = {cell: off["cells"][cell]["base"]["per_window"][cell.split("|")[0]]["R"] for cell in cells}
    fl = floors(off, cells)
    report["floor"] = {f"{t}|{s}": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in f.items()}
                       for (t, s), f in fl.items()}
    for key, f in fl.items():
        if not f["draws"]:
            why.append(f"floor {key}: no draw (every floor arm bit-identical to R)")
    off = {**off, "_ref": refs}
    scale = judge(off, "mutant_scale", cells, fl)
    report["mutant_scale"] = scale
    if any(v["pass"] for v in scale.values()):
        why.append(f"mutant_scale passes the gate in {[k for k, v in scale.items() if v['pass']]}: the gate cannot fail")
    if why:
        return {"verdict": {c: "VOID" for c in configs if c != "OFF"}, "why": why, "report": report}
    graded = judge(off, "mut098", cells, fl)
    report["mut098"] = graded
    report["ladder"] = {arm: judge(off, arm, [c for c in cells if c.endswith("|A")], fl) for arm in ("mut095", "mut090")}
    unresolved = [k for k, v in graded.items() if v["pass"]]
    verdict = {}
    for config in configs:
        if config == "OFF":
            continue
        j = judge({**recs[config], "_ref": refs}, "ON", cells, fl)
        report["configs"][config] = j
        if unresolved:
            verdict[config] = "UNRESOLVED"
        else:
            verdict[config] = "PASS" if all(v["pass"] for v in j.values()) else "FAIL"
    if unresolved:
        why.append(f"mut098 passes the gate in {unresolved}: no gate licensed for {fam}")
    return {"verdict": verdict, "why": why, "report": report}


def reduce_anchor(recs):
    """Reported only: the anchor cell's own floor (set A's draws) and where ON_auto and Phase C's 0.924 sit."""
    text, shape, set_ = ANCHOR["cell"]
    cell = fam_box.cell_key(text, shape, set_)
    off, on = recs.get("OFF"), recs.get("ON_auto")
    if not off or not on or off.get("status") != "ok" or on.get("status") != "ok":
        return {"status": "missing"}
    if off.get("path") != "sc2g" or on.get("path") != "sc2g":
        return {"status": "not the SC2g path"}
    refs = {cell: off["cells"][cell]["base"]["per_window"][text]["R"]}
    fl = floors(off, [cell])
    j = judge({**on, "_ref": refs}, "ON", [cell], fl)
    f = fl[(text, shape)]
    return {"status": "ok", "floor": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in f.items()},
            "on_auto": j, "store": on.get("store"),
            "phase_c_0924_within_floor": PHASE_C_ANCHOR_AGREE >= f["A"] - AGREE_MARGIN}


def load_dir(d, proof=False):
    out = {}
    fams = PROOF["families"] if proof else tuple(MODELS)
    for fam in fams:
        out[fam] = {}
        for config in (PROOF["configs"][fam] if proof else FAMILY_CONFIGS[fam]):
            p = os.path.join(d, f"fam_{fam}_{config}.json")
            out[fam][config] = json.load(open(p)) if os.path.exists(p) else None
    out["anchor"] = {}
    af = PROOF["anchor_family"] if proof else ANCHOR["family"]
    for config in ANCHOR["configs"]:
        p = os.path.join(d, f"fam_{af}_anchor_{config}.json")
        out["anchor"][config] = json.load(open(p)) if os.path.exists(p) else None
    return out


def reduce_all(recs, e4b_sha, proof=False):
    fams = PROOF["families"] if proof else tuple(MODELS)
    out = {"proof": bool(proof), "families": {f: reduce_family(f, recs.get(f) or {}, e4b_sha, proof) for f in fams}}
    out["anchor"] = reduce_anchor(recs.get("anchor") or {})
    out["constants"] = {"BIAS_MARGIN": BIAS_MARGIN, "SPREAD_MULT": SPREAD_MULT, "SPREAD_MIN": SPREAD_MIN,
                        "AGREE_MARGIN": AGREE_MARGIN, "BACKSTOP_BIAS": BACKSTOP_BIAS, "BACKSTOP_AGREE": BACKSTOP_AGREE}
    return out


# ------------------------------------------------------------------------------------------------- self-test --

def _synthetic(fam, config, *, floor_d=0.002, floor_agree=0.95, on_d=0.0, on_agree=0.97, mut098_d=0.05,
               mut098_agree=0.80, scale_d=1.0, e4b="E", seed=0, cont=CONT):
    """A record that passes every integrity check; the arms' scores are set by the arguments."""
    import random
    rng = random.Random(seed)
    P, C, chunk, floor_chunk = 512, cont, 512, 256
    cells = {}

    def rows(n, d, agree, kl):
        return [{"window": w, "nll": 2.0 + d + (rng.uniform(-1, 1) * abs(d) * 0.5 if d else 0.0),
                 "argmax_agree": agree, "kl": kl} for w in range(n)]

    def eng(n_pass, arm, shape):
        halves = 2 if arm == "half" else 1
        ch = floor_chunk if arm == "chunk" else chunk
        dec = (C - 1) * halves + EXTRA_DECODE_FORWARDS[fam]
        step = PER_STEP.get((fam, config), {}) if config != "OFF" else {}
        k = dict.fromkeys(("rmsnorm_rows", "rmsnorm_resid_rows", "scaled_resid_add_rows", "rope_norm_heads",
                           "rope_heads", "router_epilogue"), 0)
        k.update({n: v * dec for n, v in step.items()})
        return [{"decode_calls": (C - 1) * ATTN_LAYERS[fam] * halves, "forwards": shape * math.ceil(P / ch) + dec,
                 "kernels": k} for _ in range(n_pass)]

    for t in fam_box.TEXTS:
        for s in fam_box.SHAPES:
            for name in fam_box.SETS:
                cell = fam_box.cell_key(t, s, name)
                npass = WINDOWS_PER_SET // s
                sub = {"prompt": P, "cont": C, "chunk": chunk, "floor_chunk": floor_chunk, "group": s,
                       "windows_sha256": {t: f"{t}{name}"}}
                if config == "OFF":
                    arms = {"R": rows(12, 0.0, 1.0, 0.0), "rep": rows(s, 0.0, 1.0, 0.0),
                            "chunk": rows(12, floor_d, floor_agree, 0.008), "mutant_scale": rows(12, scale_d, 0.3, 2.0)}
                    if s > 1:
                        arms["half"] = rows(12, -floor_d, floor_agree + 0.005, 0.008)
                    base = {**sub, "per_window": {t: arms},
                            "engagement": {t: {a: eng(1 if a == "rep" else npass, a, s) for a in arms}}}
                    extra = {}
                    for arm in fam_box.extra_arms(name):
                        d, ag = (floor_d / 2, floor_agree + 0.01) if arm == "split1" else (mut098_d, mut098_agree)
                        if arm in ("mut095", "mut090"):
                            d, ag = mut098_d * 3, mut098_agree - 0.05
                        e = eng(npass, arm, s)
                        extra[arm] = {"record": {**sub, "per_window": {t: {arm: rows(12, d, ag, 0.01)}},
                                                 "engagement": {t: {arm: e}}},
                                      "wrapped_calls": sum(x["decode_calls"] for x in e)}
                else:
                    base = {**sub, "per_window": {t: {"ON": rows(12, on_d, on_agree, 0.009)}},
                            "engagement": {t: {"ON": eng(npass, "ON", s)}}}
                    extra = {}
                cells[cell] = {"base": base, "extra": extra, "dispatch": None}
    model, rev = MODELS[fam]
    census = dict(zip(CENSUS_KEYS, CENSUS[(fam, config)]))
    return {"config": config, "path": "default", "model": model, "revision": rev, "e4b_sha": e4b, "gnf4_sha": "G",
            "census": census, "store": {"int4_expert_layers": 0, "int4_store_kinds": {}}, "max_mem_gb": 20.0,
            "cells": cells, "status": "ok"}


def _fam_recs(fam, configs=None, **kw):
    return {c: _synthetic(fam, c, **kw) for c in (configs or FAMILY_CONFIGS[fam])}


def self_test() -> int:
    cases = []

    def case(name, fam, recs, want):
        got = reduce_family(fam, recs, "E")["verdict"]
        ok = all(v == want for v in got.values()) if isinstance(want, str) else got == want
        cases.append((name, ok, got))

    base = _fam_recs("gptoss")
    case("clean pass", "gptoss", base, "PASS")
    pr = _fam_recs("granite", PROOF["configs"]["granite"], cont=PROOF["cont"])
    got = reduce_family("granite", pr, "E", proof=True)["verdict"]
    cases.append(("the proof's configs at 32 positions", got == {"ON_epi": "PASS", "ON_auto": "PASS"}, got))
    got = reduce_family("granite", _fam_recs("granite", PROOF["configs"]["granite"]), "E", proof=True)["verdict"]
    cases.append(("the proof at the reading's positions VOIDs", set(got.values()) == {"VOID"}, got))
    case("a reading at the proof's positions VOIDs", "granite", _fam_recs("granite", cont=PROOF["cont"]), "VOID")
    case("granite clean pass", "granite", _fam_recs("granite"), "PASS")
    case("qw36 clean pass (an extra decode-shaped forward)", "qw36", _fam_recs("qw36"), "PASS")
    r = copy.deepcopy(base)
    on = r["ON_glue"]
    rows = on["cells"]["wikitext|1|B"]["base"]["per_window"]["wikitext"]["ON"]
    for x in rows:
        x["argmax_agree"] = 0.90
    case("one set's agreement under the floor fails that knob only", "gptoss", r,
         {"ON_glue": "FAIL", "ON_r2": "PASS", "ON_epi": "PASS", "ON_auto": "PASS"})
    r = _fam_recs("gptoss", on_d=0.015)
    case("bias over B_f + margin fails", "gptoss", r, "FAIL")
    r = _fam_recs("gptoss", floor_d=0.015, floor_agree=0.85, on_d=0.022, on_agree=0.86)
    case("a wide floor still meets the backstop", "gptoss", r, "FAIL")
    r = _fam_recs("gptoss", floor_agree=0.88, on_agree=0.885)
    case("argmax under the 0.90 backstop fails", "gptoss", r, "FAIL")
    r = _fam_recs("gptoss", mut098_d=0.001, mut098_agree=0.96)
    case("a passing graded mutant is UNRESOLVED", "gptoss", r, "UNRESOLVED")
    r = _fam_recs("gptoss", scale_d=0.001)
    for c in r["OFF"]["cells"].values():
        for t, arms in c["base"]["per_window"].items():
            for x in arms["mutant_scale"]:
                x["argmax_agree"] = 0.99
    case("a passing scale mutant VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    del r["ON_epi"]
    case("a missing record VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["ON_auto"]["census"]["fuse_router_epilogue_n"] = 23
    case("a census off its registration VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["OFF"]["census"]["fuse_t1_glue_n"] = 49
    case("an OFF census that is not zero VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["ON_epi"]["cells"]["c4val1|12|C"]["base"]["engagement"]["c4val1"]["ON"][0]["kernels"]["router_epilogue"] -= 24
    case("one decode step short of glue-kernel calls VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["OFF"]["cells"]["wikitext|1|A"]["base"]["engagement"]["wikitext"]["R"][3]["kernels"]["rmsnorm_rows"] = 1
    case("an OFF pass that called a glue kernel VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["OFF"]["cells"]["wikitext|12|B"]["base"]["engagement"]["wikitext"]["chunk"][0]["decode_calls"] -= 1
    case("a decode attention call missing VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["OFF"]["cells"]["c4val1|1|A"]["extra"]["split1"]["wrapped_calls"] -= 1
    case("split1 that missed one attention call VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["ON_r2"]["cells"]["wikitext|12|A"]["base"]["per_window"]["wikitext"]["ON"].pop()
    case("an arm short of a window VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["ON_auto"]["cells"]["c4val1|1|C"]["base"]["windows_sha256"]["c4val1"] = "other"
    case("different windows VOID", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["ON_glue"]["e4b_sha"] = "F"
    case("another e4b commit VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["ON_glue"]["revision"] = "0" * 40
    case("another model revision VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["ON_glue"]["store"] = {"int4_expert_layers": 24, "int4_store_kinds": {"mxfp4": 24}}
    case("an MXFP4 store on the default path VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["OFF"]["max_mem_gb"] = None
    case("no peak memory VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    del r["ON_glue"]["cells"]["wikitext|12|C"]
    case("a missing cell VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["OFF"]["cells"]["wikitext|1|B"]["base"]["group"] = 12
    case("a T == 1 cell run at another group VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    for cell, c in r["OFF"]["cells"].items():
        if cell.startswith("c4val1|1|"):
            for x in c["base"]["per_window"]["c4val1"]["chunk"]:
                x.update(nll=2.0, argmax_agree=1.0, kl=0.0)
            for x in c["extra"]["split1"]["record"]["per_window"]["c4val1"]["split1"]:
                x.update(nll=2.0, argmax_agree=1.0, kl=0.0)
    case("a (text, shape) with no floor draw VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["ON_auto"]["gnf4_sha"] = "H"
    case("another grouped-nf4-gemm commit VOIDs", "gptoss", r, "VOID")
    r = copy.deepcopy(base)
    r["ON_r2"]["cells"]["c4val1|1|B"]["base"]["prompt"] = 64
    case("a prompt at or under the folds' 64-row bound VOIDs", "gptoss", r, "VOID")
    ok = [c[1] for c in cases]
    # gate arithmetic, by hand
    fl = {"B": 0.002, "S": 0.010, "A": 0.95}
    ok += [passes({"bias": 0.012, "spread": 0.02, "agree": 0.945}, fl),
           not passes({"bias": 0.0121, "spread": 0.02, "agree": 0.95}, fl),
           not passes({"bias": 0.0, "spread": 0.0201, "agree": 0.95}, fl),
           not passes({"bias": 0.0, "spread": 0.0, "agree": 0.9449}, fl),
           passes({"bias": 0.0, "spread": 0.0099, "agree": 0.95}, {"B": 0.0, "S": 0.001, "A": 0.95}),  # SPREAD_MIN
           not passes({"bias": 0.021, "spread": 0.03, "agree": 0.95}, {"B": 0.05, "S": 0.05, "A": 0.95}),  # backstop
           stats([{"window": 0, "nll": 2.0, "argmax_agree": 1.0, "kl": 0.0}],
                 [{"window": 0, "nll": 2.0}])["identical"],
           not stats([{"window": 0, "nll": 2.0, "argmax_agree": 0.99, "kl": 0.0}],
                     [{"window": 0, "nll": 2.0}])["identical"]]
    an = {c: copy.deepcopy(base[c]) for c in ANCHOR["configs"]}
    for c in an.values():
        c["path"] = "sc2g"
    rep = reduce_anchor(an)
    ok.append(rep["status"] == "ok" and "phase_c_0924_within_floor" in rep)
    ok.append(reduce_anchor({"OFF": base["OFF"], "ON_auto": base["ON_auto"]})["status"] == "not the SC2g path")
    for name, good, got in cases:
        if not good:
            print(f"  FAILED case: {name}: {got}")
    print(f"fam_reduce self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--e4b-sha")
    p.add_argument("--proof", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.dir and a.out and a.e4b_sha):
        raise SystemExit("REFUSED: --dir, --out and --e4b-sha are required")
    v = reduce_all(load_dir(a.dir, a.proof), a.e4b_sha, a.proof)
    with open(a.out, "w") as f:
        json.dump(v, f, indent=1)
    for fam, r in v["families"].items():
        print(f"FAM_VERDICT {fam} {r['verdict']}" + (f" -- {r['why'][:3]}" if r["why"] else ""))
    print(f"FAM_ANCHOR {json.dumps(v['anchor'].get('floor', v['anchor']), default=str)[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
