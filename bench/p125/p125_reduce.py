#!/usr/bin/env python3
"""p125_reduce.py -- lane P125's verdict (bench/p125/PREREG-p125.md; e4b#1313). Pure function of the box's records.

Inputs in --dir: the speed arms ``arm_{A1,B1,C1,C2,B2,A2}.json`` (palindromic) and the quality arms
``quality_{A,B,C,M,K}.json`` (p125_box.py).

THE GATES (the maintainer's ruling, 2026-10-08T01:13Z). Per quality gate (``t1``: wikitext at one window per pass, the
one-row activation-quantised path; ``k16``: wikitext in 16-row pieces, K16), per arm against A:
  d       per window, the arm's mean continuation NLL minus A's; mean d, SD, SE = SD / sqrt(n)
  bound   ln(1 + 0.05 / ppl_A), ppl_A = exp(A's mean NLL over that gate's own windows): K8's budget in nats
  FAIL          mean argmax agreement < 0.95; or mean d - 2 SE > bound (decisive at any power)
  UNDERPOWERED  SE > bound / 3 (not licensed, not failed; reported with the window count it would have needed)
  FAIL          mean d > bound
  PASS          otherwise
  (one-sided in d, as K8's calibrated rule: an improvement is not a failure, and is not claimed)
LICENCES: B LICENSED iff t1 and k16 PASS and its calibration is deterministic across its three builds; C LICENSED iff
its own gates PASS, its builds agree and B is LICENSED. Otherwise NOT_LICENSED (a FAIL) or UNDERPOWERED.

THE VERDICT, the first rung that applies:
  VOID   a record missing or not ok; another e4b / grouped-nf4-gemm commit or model revision; not the shipped default
         (a fusion or GEMV knob set, the census or its sources off the family's default, the build's GEMV dispatch off
         its registration); an int4 census off its arm (projections, modules, the head); a gate's window count or
         group off; a pass's route counts off shape (t1: gemv at 1 row only, exactly windows x (cont - 1) x modules,
         no K16 or wide; k16 and c4: K16 at 16 rows only, exactly passes x (cont - 1) x modules, no gemv or wide);
         the sure-fail mutant K not FAILING both gates
  READ   otherwise; the speed self-pairs outside [0.97, 1.03] mark the speed NOISY (the licences stand)
Reported, never ruled: RTN (M) against its written prediction (FAIL at t1), the K8-style ppl deltas (wikitext from
t1, c4val1 from c4; neither method runs the one-row path for c4val1), speed ratios g1 / g16 for B and C (the minimum
over the two palindromic pairs), memory per arm and the auto slot counts (the slot trade), calibration seconds.

    python p125_reduce.py --dir RUN/p125 --out verdict_p125.json --e4b-sha SHA [--proof]
    python p125_reduce.py --self-test
"""
import argparse
import json
import math
import os
import statistics
import sys

GNF4_SHA = "6ee2e10408161a9d3c874975c9191a7f2957e6f4"
QWEN, GRAN = "Qwen/Qwen3-30B-A3B", "ibm-granite/granite-3.1-3b-a800m-instruct"
REVS = {QWEN: "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", GRAN: "a02780686e08a03fe0d2679a293b5c74a90efa89"}
LAYERS = {QWEN: 48, GRAN: 32}
#: the family's shipped default: Qwen3-MoE fuses q/k/v (two int4 modules a layer: qkv, o), Granite-MoE does not (four)
DEFAULT = {QWEN: {"source": "default-allowlisted", "census": {"fuse_qkv_n": 48, "fuse_t1_glue_n": 193,
                                                              "fuse_t1_glue_r2_n": [48, 48], "fuse_router_epilogue_n": 48},
                  "mods_per_layer": 2, "gemv": "bw"},
           GRAN: {"source": "default-off", "census": {"fuse_qkv_n": 0, "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": [0, 0],
                                                      "fuse_router_epilogue_n": 0},
                  "mods_per_layer": 4, "gemv": "scalar"}}
KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
GEMV_KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE")
SPEED_TAGS = ("A1", "B1", "C1", "C2", "B2", "A2")
QUALITY_ARMS = ("A", "B", "C", "M", "K")
#: (gate, text, group, windows, K's windows) -- p125_box.GATES
GATES = {"t1": ("wikitext", 1, 108, 12), "k16": ("wikitext", 16, 112, 16), "c4": ("c4val1", 16, 16, 16)}
RULED = ("t1", "k16")
K8_BUDGET, ARGMAX_MIN, POWER = 0.05, 0.95, 3.0
SELF_PAIR = (0.97, 1.03)
CONT = 128
#: the prior the window counts were sized on (Phase D, p115d-5090-1: 12 wikitext windows at T == 1)
PRIOR = {"sd_d": 0.01957, "ppl_R": 8.762, "bound": 0.00569, "windows_needed": 107}
PREDICTED = {"g1_B": (1.10, 1.30), "g1_C": (1.15, 1.40), "g16_B": (1.00, 1.08), "g16_C": (1.00, 1.10),
             "M_t1": "FAIL", "B_mem_gib": (0.4, 0.8), "C_mem_gib": (0.5, 1.0)}


def bound_of(nll_a) -> float:
    return math.log(1.0 + K8_BUDGET / math.exp(statistics.mean(nll_a)))


def gate(nll_a, nll_x, argmax):
    """One arm at one gate against A: the registered rule, rung by rung."""
    n = len(nll_a)
    d = [x - a for a, x in zip(nll_a, nll_x)]
    md = statistics.mean(d)
    sd = statistics.stdev(d) if n > 1 else float("inf")
    se = sd / math.sqrt(n)
    b = bound_of(nll_a)
    am = statistics.mean(argmax)
    if am < ARGMAX_MIN:
        out, why = "FAIL", f"argmax {am:.4f} < {ARGMAX_MIN}"
    elif md - 2 * se > b:
        out, why = "FAIL", f"d {md:+.5f} - 2 SE {se:.5f} > bound {b:.5f}"
    elif se > b / POWER:
        out, why = "UNDERPOWERED", f"SE {se:.5f} > bound/3 {b / POWER:.5f}"
    elif md > b:
        out, why = "FAIL", f"d {md:+.5f} > bound {b:.5f}"
    else:
        out, why = "PASS", f"d {md:+.5f} <= bound {b:.5f}, argmax {am:.4f}, SE {se:.5f}"
    need = math.ceil((sd / (b / POWER)) ** 2) if math.isfinite(sd) else None
    return {"outcome": out, "why": why, "windows": n, "d": round(md, 6), "sd": round(sd, 6), "se": round(se, 6),
            "bound": round(b, 6), "argmax": round(am, 5), "windows_needed": need}


def _per(rec, g, arm):
    pw = rec["gates"][g]["per_window"][GATES[g][0]][arm]
    return sorted(pw, key=lambda r: r["window"])


def mods_expected(model, arm):
    if arm == "A":
        return 0
    n = LAYERS[model] * DEFAULT[model]["mods_per_layer"]
    return n + (1 if arm == "C" else 0)


def int4_faults(tag, r, model, arm) -> list:
    out = []
    i4 = r.get("int4") or {}
    n_proj = LAYERS[model] * 4
    mods = len(r.get("int4_modules") or [])
    want = mods_expected(model, arm)
    if mods != want:
        out.append(f"{tag}: {mods} Int4Linear modules, the arm's default build has {want}")
    if arm == "A" and any(i4.get(k) for k in i4):
        out.append(f"{tag}: int4 census {i4} on the default arm")
    if arm in ("B", "K") and i4.get("attn_int4_calib_projections") != n_proj:
        out.append(f"{tag}: {i4.get('attn_int4_calib_projections')} calibrated projections, not {n_proj}")
    if arm == "C" and i4.get("attn_int4_calib_projections") != n_proj + 1:
        out.append(f"{tag}: {i4.get('attn_int4_calib_projections')} calibrated projections, not {n_proj + 1} (the head)")
    if arm == "C" and r.get("head_type") != "Int4Linear":
        out.append(f"{tag}: the head is {r.get('head_type')}")
    if arm != "C" and r.get("head_type") == "Int4Linear":
        out.append(f"{tag}: the head is int4 off arm C")
    if arm == "M" and i4.get("attn_int4_rtn_projections") != n_proj:
        out.append(f"{tag}: {i4.get('attn_int4_rtn_projections')} RTN projections, not {n_proj}")
    if arm == "K" and r.get("rolled_modules") != want:
        out.append(f"{tag}: the mutant rolled {r.get('rolled_modules')} modules, not {want}")
    return out


def default_faults(tag, r, e4b_sha, model) -> list:
    out = []
    if r.get("e4b_sha") != e4b_sha or r.get("gnf4_sha") != GNF4_SHA:
        out.append(f"{tag}: e4b {r.get('e4b_sha')} gnf4 {r.get('gnf4_sha')}")
    if r.get("model") != model or r.get("revision") != REVS[model]:
        out.append(f"{tag}: model {r.get('model')}@{r.get('revision')}")
    env = r.get("env") or {}
    if any((env.get(k) or "").strip() for k in KNOBS + GEMV_KNOBS):
        out.append(f"{tag}: a fusion or GEMV knob set: { {k: env.get(k) for k in KNOBS + GEMV_KNOBS if env.get(k)} }")
    if r.get("fusions") != DEFAULT[model]["census"]:
        out.append(f"{tag}: fusion census {r.get('fusions')}, the default's is {DEFAULT[model]['census']}")
    if set((r.get("fusion_sources") or {}).values()) != {DEFAULT[model]["source"]}:
        out.append(f"{tag}: fusion sources {r.get('fusion_sources')}")
    db = r.get("dispatch_build") or {}
    if r.get("mode") == "speed":
        bw = sum(v for k, v in db.items() if k.startswith("bw_"))
        if DEFAULT[model]["gemv"] == "bw" and (db.get("bw_prmt32", 0) <= 0 or db.get("dotpad", 0) or db.get("dotpad_splitk", 0)):
            out.append(f"{tag}: the build's GEMV dispatch {db} is not bw_prmt32 without dot-pad")
        if DEFAULT[model]["gemv"] == "scalar" and (bw or db.get("scalar", 0) <= 0):
            out.append(f"{tag}: the build's GEMV dispatch {db} is not the scalar route")
    return out


def route_faults(tag, gname, g, n_mods, windows) -> list:
    rt = g.get("routes") or {}
    group = GATES[gname][1]
    if n_mods == 0:
        return [f"{tag} {gname}: routes {rt} on the default arm"] if rt else []
    out = []
    keys = {k.split(":")[0] for k in rt}
    if group == 1:
        want = windows * (CONT - 1) * n_mods
        if rt.get("gemv:1") != want or keys - {"gemv", "bf16"} or any(k.startswith("gemv:") and k != "gemv:1" for k in rt):
            out.append(f"{tag} {gname}: routes {rt}, want gemv:1 == {want} and no K16 or wide")
    else:
        passes = -(-windows // group)
        want = passes * (CONT - 1) * n_mods
        if (rt.get(f"k16:{group}") != want or keys - {"k16", "bf16"}
                or any(k.startswith("k16:") and k != f"k16:{group}" for k in rt)):
            out.append(f"{tag} {gname}: routes {rt}, want k16:{group} == {want} and no gemv or wide")
    if not any(k.startswith("bf16:") for k in rt):
        out.append(f"{tag} {gname}: no prefill on the bf16 copy ({rt})")
    return out


def reduce(speed, quality, e4b_sha, proof=False, cap=0):
    model = GRAN if proof else QWEN
    out = {"lane": "P125", "proof": proof, "model": model, "verdict": None, "reasons": [], "prior": PRIOR}
    missing = [f"arm_{t}" for t in SPEED_TAGS if not speed.get(t) or speed[t].get("status") != "ok"]
    missing += [f"quality_{a}" for a in QUALITY_ARMS if not quality.get(a) or quality[a].get("status") != "ok"]
    if missing:
        out.update(verdict="VOID", reasons=[f"missing or not ok: {missing}"])
        return out
    void = []
    for t, r in speed.items():
        void += default_faults(f"arm_{t}", r, e4b_sha, model) + int4_faults(f"arm_{t}", r, model, t[0])
    for a, r in quality.items():
        void += default_faults(f"quality_{a}", r, e4b_sha, model) + int4_faults(f"quality_{a}", r, model, a)
        for gname, (text, group, n, n_k) in GATES.items():
            g = (r.get("gates") or {}).get(gname)
            want_n = n_k if a == "K" else n
            want_n = min(want_n, cap) if cap else want_n
            if not g or g.get("group") != group or g.get("windows") != want_n or g.get("text") != text:
                void.append(f"quality_{a} {gname}: {None if not g else (g.get('text'), g.get('group'), g.get('windows'))}, "
                            f"want ({text}, {group}, {want_n})")
                continue
            void += route_faults(f"quality_{a}", gname, g, mods_expected(model, a), want_n)
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    gates = {}
    for a in ("B", "C", "M", "K"):
        gates[a] = {}
        for gname in GATES:
            ra = _per(quality["A"], gname, "R")
            rx = _per(quality[a], gname, "ON")
            ra = ra[:len(rx)]
            if [r["window"] for r in ra] != [r["window"] for r in rx]:
                out.update(verdict="VOID", reasons=[f"quality_{a} {gname}: windows do not pair with A's"])
                return out
            gates[a][gname] = gate([r["nll"] for r in ra], [r["nll"] for r in rx], [r["argmax_agree"] for r in rx])
    out["gates"] = gates
    dead = [g for g in RULED if gates["K"][g]["outcome"] != "FAIL"]
    if dead:
        out.update(verdict="VOID", reasons=[f"the sure-fail mutant K did not FAIL {dead}: {gates['K']}"])
        return out

    def digests(r):
        return {n: h for n, _N, _K, h in r["int4_modules"]}
    det = {a: [digests(speed[f"{a}1"]), digests(speed[f"{a}2"]), digests(quality[a])] for a in ("B", "C")}
    det_ok = {a: all(x == det[a][0] for x in det[a]) for a in det}
    attn_same = {n: h for n, h in det["C"][0].items() if n in det["B"][0]} == {n: h for n, h in det["B"][0].items()
                                                                                if n in det["C"][0]}
    out["calibration"] = {"deterministic": det_ok, "C_attention_equals_B": attn_same,
                          "seconds": {a: round(statistics.mean([speed[f"{a}1"]["load_s"], speed[f"{a}2"]["load_s"]])
                                               - statistics.mean([speed["A1"]["load_s"], speed["A2"]["load_s"]]), 1)
                                      for a in ("B", "C")}}
    lic = {}
    for a in ("B", "C"):
        oc = [gates[a][g]["outcome"] for g in RULED]
        if "FAIL" in oc:
            lic[a] = "NOT_LICENSED"
        elif "UNDERPOWERED" in oc:
            lic[a] = "UNDERPOWERED"
        elif not det_ok[a]:
            lic[a] = "NOT_LICENSED (calibration not deterministic across builds)"
        else:
            lic[a] = "LICENSED"
    if lic["C"] == "LICENSED" and lic["B"] != "LICENSED":
        lic["C"] = f"NOT_LICENSED (B {lic['B']})"
    out["licences"] = lic
    out["rtn"] = {"t1": gates["M"]["t1"]["outcome"], "predicted": PREDICTED["M_t1"],
                  "held": gates["M"]["t1"]["outcome"] == PREDICTED["M_t1"]}
    k8 = {}
    for a in ("B", "C", "M"):
        k8[a] = {}
        for gname, text in (("t1", "wikitext"), ("c4", "c4val1")):
            ra, rx = _per(quality["A"], gname, "R"), _per(quality[a], gname, "ON")
            pa = math.exp(statistics.mean(r["nll"] for r in ra))
            px = math.exp(statistics.mean(r["nll"] for r in rx))
            k8[a][text] = {"ppl_A": round(pa, 4), "ppl": round(px, 4), "delta": round(px - pa, 4),
                           "within_budget": px - pa <= K8_BUDGET}
    out["k8_style"] = k8

    def tok_s(t, w):
        return speed[t]["workloads"][w]["decode_tok_s"]
    sp = {"decode_tok_s": {t: {w: tok_s(t, w) for w in ("W1", "W16")} for t in SPEED_TAGS}}
    noisy = []
    for a in ("A", "B", "C"):
        for w in ("W1", "W16"):
            x = tok_s(f"{a}2", w) / tok_s(f"{a}1", w)
            sp.setdefault("self_pairs", {})[f"{a}2/{a}1 {w}"] = round(x, 4)
            if not SELF_PAIR[0] <= x <= SELF_PAIR[1]:
                noisy.append(f"{a}2/{a}1 {w} {x:.4f}")
    for a in ("B", "C"):
        for w, key in (("W1", "g1"), ("W16", "g16")):
            sp[f"{key}_{a}"] = round(min(tok_s(f"{a}1", w) / tok_s("A1", w), tok_s(f"{a}2", w) / tok_s("A2", w)), 4)
    sp["noisy"] = noisy
    out["speed"] = sp
    mem = {}
    for t in SPEED_TAGS:
        m = speed[t]["memory"]
        mem[t] = {k: round(m[k] / 2**30, 3) for k in ("peak_after_build", "peak_first_prefill", "peak_runs")}
        mem[t]["slots"] = {k: speed[t]["slots"][k]["corrected_slots"] for k in ("2048", "4096")}
    base = {k: mem["A1"]["slots"][k] for k in ("2048", "4096")}
    out["memory"] = mem
    out["slot_cost"] = {t: {k: mem[t]["slots"][k] - base[k] for k in base} for t in SPEED_TAGS
                        if any(mem[t]["slots"][k] != base[k] for k in base)}
    out["predictions"] = {k: {"predicted": v, "measured": sp.get(k), "held": (v[0] <= sp[k] <= v[1])}
                          for k, v in PREDICTED.items() if k in sp}
    out["verdict"] = "READ"
    out["reasons"] = [f"B {lic['B']}, C {lic['C']}; RTN t1 {gates['M']['t1']['outcome']}; "
                      f"g1_B {sp['g1_B']}, g1_C {sp['g1_C']}" + ("; speed NOISY" if noisy else "")]
    return out


# ---------------------------------------------------------------- fakes, as the box writes its records
E = "e" * 40


def fake_speed(tag, model=QWEN, w1=220.0, w16=880.0, load_s=40.0, digest_seed="x", slots=(64, 32)):
    arm = tag[0]
    n = mods_expected(model, arm)
    proj = LAYERS[model] * 4
    mods = [[f"model.layers.{i // 2}.m{i % 2}", 1, 1, f"{digest_seed}{i}"] for i in range(n - (1 if arm == "C" else 0))]
    if arm == "C":
        mods.append(["lm_head", 1, 1, f"{digest_seed}head"])
    int4 = {"int4_attn_projections": n or None, "attn_int4_calib_projections": (proj + (arm == "C")) if arm in "BC" else None,
            "attn_int4_rtn_projections": None}
    gemv = {"bw_prmt32": 288, "dotpad": 0, "dotpad_splitk": 0, "scalar": 0} if DEFAULT[model]["gemv"] == "bw" else \
        {"bw_prmt32": 0, "dotpad": 0, "dotpad_splitk": 0, "scalar": 96}
    return {"mode": "speed", "tag": tag, "arm": arm, "e4b_sha": E, "gnf4_sha": GNF4_SHA, "model": model,
            "revision": REVS[model], "env": {k: None for k in KNOBS + GEMV_KNOBS}, "load_s": load_s,
            "fusions": dict(DEFAULT[model]["census"]),
            "fusion_sources": {k: DEFAULT[model]["source"] for k in KNOBS}, "int4": int4, "int4_modules": mods,
            "head_type": "Int4Linear" if arm == "C" else "Linear", "dispatch_build": gemv,
            "workloads": {"W1": {"decode_tok_s": w1}, "W16": {"decode_tok_s": w16}},
            "memory": {"free_before_load": 31 * 2**30, "total": 32 * 2**30, "peak_after_build": 20 * 2**30,
                       "peak_first_prefill": 21 * 2**30, "peak_runs": 22 * 2**30},
            "slots": {"2048": {"corrected_slots": slots[0]}, "4096": {"corrected_slots": slots[1]}}, "status": "ok"}


def fake_quality(arm, model=QWEN, shift=0.0, argmax=0.97, noise=0.002, digest_seed="x", cap=0, routes=None):
    n_mods = mods_expected(model, arm)
    base = fake_speed(f"{arm}1" if arm in "ABC" else "B1", model, digest_seed=digest_seed)
    if arm in ("M", "K"):
        base["arm"] = arm
    int4 = dict(base["int4"])
    if arm == "M":
        int4.update(attn_int4_calib_projections=None, attn_int4_rtn_projections=LAYERS[model] * 4)
    rec = {**base, "mode": "quality", "arm": arm, "int4": int4, "rolled_modules": n_mods if arm == "K" else 0,
           "gates": {}, "status": "ok"}
    for speed_only in ("workloads", "tag", "slots", "memory"):           # the quality record carries none of these
        rec.pop(speed_only)
    for gname, (text, group, n, n_k) in GATES.items():
        n = n_k if arm == "K" else n
        n = min(n, cap) if cap else n
        pw = []
        for i in range(n):
            nll = 2.17 + 0.01 * ((i * 7) % 5)
            if arm != "A":
                nll += shift + noise * (1 if i % 2 else -1)
            pw.append({"window": i, "nll": nll, "argmax_agree": 1.0 if arm == "A" else argmax})
        if n_mods:
            if group == 1:
                rt = {"gemv:1": n * (CONT - 1) * n_mods, "bf16:512": n * n_mods}
            else:
                rt = {f"k16:{group}": -(-n // group) * (CONT - 1) * n_mods, "bf16:512": n * n_mods}
        else:
            rt = {}
        if routes and gname in routes:
            rt = routes[gname]
        rec["gates"][gname] = {"text": text, "group": group, "windows": n, "routes": rt,
                               "per_window": {text: {("R" if arm == "A" else "ON"): pw}}}
    return rec


def fakes(model=QWEN, cap=0, **kw):
    speed = {t: fake_speed(t, model, w1={"A": 220.0, "B": 270.0, "C": 290.0}[t[0]],
                           w16={"A": 880.0, "B": 920.0, "C": 930.0}[t[0]],
                           load_s={"A": 40.0, "B": 130.0, "C": 160.0}[t[0]]) for t in SPEED_TAGS}
    quality = {"A": fake_quality("A", model, cap=cap), "B": fake_quality("B", model, shift=0.001, cap=cap),
               "C": fake_quality("C", model, shift=0.002, cap=cap), "M": fake_quality("M", model, shift=0.02, cap=cap),
               "K": fake_quality("K", model, shift=1.5, argmax=0.4, cap=cap)}
    for k, v in kw.items():
        if k in speed:
            speed[k] = v
        else:
            quality[k] = v
    return speed, quality


def self_test() -> int:
    def run(model=QWEN, proof=False, cap=0, **kw):
        s, q = fakes(model, cap=cap, **kw)
        return reduce(s, q, E, proof=proof, cap=cap)

    ok = run()
    bound = bound_of([2.17 + 0.01 * ((i * 7) % 5) for i in range(108)])
    cases = [
        ("a clean reading reads", ok["verdict"] == "READ" and ok["licences"] == {"B": "LICENSED", "C": "LICENSED"}),
        ("the bound is K8's budget in nats", abs(bound - math.log(1 + 0.05 / math.exp(statistics.mean(
            [2.17 + 0.01 * ((i * 7) % 5) for i in range(108)])))) < 1e-12 and 0.004 < bound < 0.007),
        ("the prior's arithmetic", math.ceil((PRIOR["sd_d"] / (PRIOR["bound"] / 3)) ** 2) == PRIOR["windows_needed"]
         and abs(math.log(1 + 0.05 / PRIOR["ppl_R"]) - PRIOR["bound"]) < 5e-6),
        ("RTN fails as predicted", ok["rtn"] == {"t1": "FAIL", "predicted": "FAIL", "held": True}),
        ("the mutant fails both gates", all(ok["gates"]["K"][g]["outcome"] == "FAIL" for g in RULED)),
        ("speed ratios: the minimum over the pairs", ok["speed"]["g1_B"] == round(270 / 220, 4)),
        ("B over the bound is NOT_LICENSED", run(B=fake_quality("B", shift=0.008))["licences"]["B"] == "NOT_LICENSED"),
        ("C needs B", run(B=fake_quality("B", shift=0.008))["licences"]["C"].startswith("NOT_LICENSED")),
        ("argmax under 0.95 fails", run(B=fake_quality("B", argmax=0.94))["licences"]["B"] == "NOT_LICENSED"),
        ("a wide spread is UNDERPOWERED", run(B=fake_quality("B", noise=0.03))["licences"]["B"] == "UNDERPOWERED"),
        ("a decisive fail at any power", run(B=fake_quality("B", shift=0.2, noise=0.05))["gates"]["B"]["t1"]["outcome"] == "FAIL"),
        ("an improvement is not a failure", run(B=fake_quality("B", shift=-0.01))["licences"]["B"] == "LICENSED"),
        ("a mutant that passes VOIDs", run(K=fake_quality("K", shift=0.0, argmax=0.97))["verdict"] == "VOID"),
        ("a T == 1 pass off the gemv VOIDs", run(B=fake_quality("B", routes={"t1": {"k16:1": 5, "bf16:512": 1}}))["verdict"] == "VOID"),
        ("a K16 pass off K16 VOIDs", run(B=fake_quality("B", routes={"k16": {"gemv:1": 5, "bf16:512": 1}}))["verdict"] == "VOID"),
        ("a gate's window count off VOIDs", run(A=fake_quality("A", cap=100))["verdict"] == "VOID"),
        ("a missing record VOIDs", reduce({}, {}, E)["verdict"] == "VOID"),
        ("another e4b commit VOIDs", reduce(*fakes(), "f" * 40)["verdict"] == "VOID"),
        ("the census off the default VOIDs", run(A1={**fake_speed("A1"), "fusions": {**DEFAULT[QWEN]["census"], "fuse_qkv_n": 0}})["verdict"] == "VOID"),
        ("an int4 arm without its modules VOIDs", run(B1={**fake_speed("B1"), "int4_modules": []})["verdict"] == "VOID"),
        ("calibration that differs across builds is not licensed",
         run(B2=fake_speed("B2", digest_seed="y", w1=270.0, w16=920.0, load_s=130.0))["licences"]["B"].startswith("NOT_LICENSED")),
        ("a noisy self-pair marks the speed NOISY, the licences stand",
         (lambda x: x["verdict"] == "READ" and x["speed"]["noisy"] and x["licences"]["B"] == "LICENSED")(
             run(A2=fake_speed("A2", w1=200.0)))),
        ("a slot cost is named", run(C1=fake_speed("C1", w1=290.0, w16=930.0, load_s=160.0, slots=(32, 32)))["slot_cost"]
         == {"C1": {"2048": -32, "4096": 0}}),
        ("the proof: Granite unfused, capped windows", run(GRAN, proof=True, cap=16)["verdict"] == "READ"),
        ("the proof's mutant still fails at 12 windows", run(GRAN, proof=True, cap=16)["gates"]["K"]["t1"]["outcome"] == "FAIL"),
    ]
    bad = [name for name, good in cases if not good]
    print(f"p125_reduce self-test {'OK' if not bad else 'FAILED: ' + '; '.join(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def _load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--e4b-sha")
    p.add_argument("--proof", action="store_true")
    p.add_argument("--cap", type=int, default=0, help="the proof's window cap (p125_box --max-windows)")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    speed = {t: r for t in SPEED_TAGS if (r := _load(a.dir, f"arm_{t}.json"))}
    quality = {x: r for x in QUALITY_ARMS if (r := _load(a.dir, f"quality_{x}.json"))}
    v = reduce(speed, quality, a.e4b_sha, proof=a.proof, cap=a.cap)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P125_VERDICT {v['verdict']} {json.dumps(v['reasons'])[:600]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
