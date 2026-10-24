#!/usr/bin/env python3
"""p233_reduce.py -- lane P233's registered rule (bench/p233/PREREG-p233.md, MAP 2), fixed before data: does serving at
``E4B_PAGED_PLACEMENT=solver`` keep teacher-forced quality against the same server at ``all-vram``? P232's reducer
(bench/p232/p232_reduce.py) by counted edits: its pooled argmax counts (``integral_agree``, ``agree_counts``,
``agree_at_least``), its NLL comparison, its tripwire and its shakedown rung; no speed rungs; the placement PROOF per arm
in place of P232's head engagement; a per-T SANE (P153's bars) and SOLVER_SANE / SOLVER_NOT_SANE in place of its flips.

Records (``p233_box.py``): ``quality_A.json`` (all-vram, the reference: R saved, its ``rep`` floor draw), ``quality_S.json``
(solver at 8.0 / 3.0 GiB, all three tiers: the ruling arm) and ``quality_C.json`` (solver at 8.0 / 0: VRAM and NVMe only,
reported). Each phase reads group T = 1 and 8, 24 wikitext windows x 128 positions after a 512-token prompt, the pass KV at
the server's 4096 tokens; S and C score against A's saved log-probs.

NO REAL FIXTURE YET: the shakedown's records (``p233-shake-N``) become this reducer's real fixture by an amendment before
the reading. Every case below is synthetic, built to the box's record shape (p233_box.py).

The rule, first rung that applies:
  VOID             a phase record missing or not ok, or the records mixing a shakedown and a reading; the pins (e4b,
                   grouped-nf4-gemm v0.50.0, torch, the model and revision), the card (cc 12.0, 170 SMs); the windows
                   (24 per T, 128 positions after 512, group T as registered, one window set by sha in every phase); the
                   same model in every arm (the checkpoint revision and the NF4 arena's sha); the PLACEMENT PROOF per arm
                   from the record (the tier byte and expert counts as the solver resolved them, and the MoE-layer path
                   counts from the engagement at every T):
                     A  every expert in VRAM, nothing in DRAM or on NVMe, every MoE layer on the collapse;
                     S  non-zero bytes on VRAM, DRAM and NVMe, budgets 8.0 / 3.0, at least one layer on the tiered forward;
                     C  DRAM empty, NVMe non-empty, budgets 8.0 / 0, at least one layer tiered;
                   graphs on in any arm or any decode bucket captured; the head not the int8 head at rows 8 (both
                   ``explicit``, Amendment 1), or the head, the split cap or the fusions resolving differently between
                   phases; the pass KV off the server's 4096 tokens; a missing per-window NLL, argmax or KL on a scored
                   arm (S or C), or a missing NLL on A's R (never read as 0); an agreement that is not k/128.
  per T            SANE when S against A holds |mean NLL(S) - mean NLL(A)| <= 0.02 and argmax agreement >= 0.95 (exact
                   pooled counts), both unrounded; NOT_SANE otherwise, each failed bar named.
  verdict          SOLVER_SANE (SANE at T = 1 and at T = 8) / SOLVER_NOT_SANE (each failed bar named).
  TRIPWIRE         (reported, named, never changing the verdict) at any T: |NLL gap| > 0.005, argmax < 0.98, or the S
                   against A KL above max(3 x A's rep floor, 0.002), the floor being A's ``rep`` KL at that T (a zero or
                   absent floor falls to 0.002). ``TRIPWIRE none``, every condition that fired, or ``TRIPWIRE unread`` on
                   a VOID.
A shakedown (``p233-shake-N``, every record ``shakedown``) reads ``SHAKEDOWN`` after integrity, never a verdict: its card
is any that holds the arms and its window count is the reduced 8; the rungs below are reported as ``would_read``.

Reported, never ruled: C against A (gap, argmax, KL, the bars printed beside them) and S against C (the NLL gap: the DRAM
tier's share; argmax agreement from the recorded argmax); S's KL against A per window (its spread); the positions where
S's and C's argmax differ from A's; A's own floor arms; each arm's split (experts, bytes, resident bytes), layer paths
and build time; the host CPU, its ISA flags and the CPU kernels' threads; the wall times.

    python p233_reduce.py --dir RUN_DIR --out verdict_p233.json --e4b-sha SHA
    python p233_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ARMS = ("A", "S", "C")
RULING, REFERENCE, REPORTED = "S", "A", "C"
GROUPS = (1, 8)                                           # T, each read on its own
TIERS = ("vram", "dram", "nvme")
PLACEMENT = {"A": "all-vram", "S": "solver", "C": "solver"}
BUDGET_GB = {"S": (8.0, 3.0), "C": (8.0, 0.0)}             # (E4B_PAGED_VRAM_GB, E4B_PAGED_DRAM_GB); A: unused
#: what measure_phase scores per phase at p115_quality.main's keywords (its OFF_ARMS on A, ON_ARMS on S and C)
PHASE_ARMS = {"A": ("R", "rep", "half", "chunk", "mutant_scale"), "S": ("ON",), "C": ("ON",)}
GNF4_SHA = "41ef086e6a3e2be5b158ff1bb38e0de67862a949"     # grouped-nf4-gemm v0.50.0, released
TORCH = "2.8.0+cu128"
MODEL, REV = "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
CC, SMS = "12.0", 170                                     # the RTX 5090
HEAD_SHAPE = [151936, 2048]                               # Qwen3-30B-A3B's lm_head (N, K), untied
HEAD_MODE, HEAD_SOURCE, ROWS_MAX, ROWS_SOURCE = "1", "explicit", 8, "explicit"   # Amendment 1, every arm
KNOBS = {"E4B_PAGED_GRAPHS": "0", "E4B_INT8_HEAD_BW": "1", "E4B_INT8_HEAD_ROWS": "8"}
Q_WINDOWS, SHAKE_WINDOWS, Q_PROMPT, Q_CONT = 24, 8, 512, 128   # 24 (and the shakedown's 8) divide by 1 and 8
SERVED_TOKENS, SERVED_BLOCKS = 4096, 256                  # the server's KV per sequence (16-token blocks)
Q_BIAS, Q_ARGMAX = 0.02, 0.95                             # P153's bars, at each T on its own
TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.005, 0.98, 3.0, 0.002
VERDICTS_READ = ("SOLVER_SANE", "SOLVER_NOT_SANE")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _mean(xs):
    return sum(xs) / len(xs)


def integral_agree(frac: float) -> bool:
    """measure_phase's agreement is matched/positions; on the registered Q_CONT positions it is k/Q_CONT exactly."""
    k = frac * Q_CONT
    return abs(k - round(k)) < 1e-6


def agree_counts(windows: list) -> tuple:
    """Argmax agreement as exact pooled counts: each window's fraction is matched/Q_CONT (measure_phase), so the
    positions that agree are recovered as integers and the bars compare counts, never a float mean that can land a
    hair under 0.95."""
    return sum(round(x["argmax_agree"] * Q_CONT) for x in windows), len(windows) * Q_CONT


def agree_at_least(counts: tuple, bar: float) -> bool:
    """counts[0] / counts[1] >= bar, in integers (the bars are whole percents)."""
    return counts[0] * 100 >= round(bar * 100) * counts[1]


def compare(a: list, b: list) -> dict:
    """Arm ``a`` against arm ``b`` over the same windows (per-window means): the mean NLL gap a - b."""
    ref = {x["window"]: x["nll"] for x in b}
    return {"bias": _mean([x["nll"] - ref[x["window"]] for x in a]), "n": len(a)}


def record_faults(name, r, e4b_sha, arm, shake=False) -> list:
    """One phase's pins, card, config and head, as registered."""
    out = []
    if r.get("e4b_sha") != e4b_sha:
        out.append(f"{name}: e4b {r.get('e4b_sha')} != {e4b_sha}")
    if r.get("gnf4_sha") != GNF4_SHA:
        out.append(f"{name}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
    if r.get("torch") != TORCH:
        out.append(f"{name}: torch {r.get('torch')} != {TORCH}")
    if (r.get("model"), r.get("revision")) != (MODEL, REV):
        out.append(f"{name}: model {r.get('model')}@{r.get('revision')} is not {MODEL}@{REV}")
    if r.get("arm") != arm:
        out.append(f"{name}: the record says arm {r.get('arm')}")
    if not shake and (r.get("cc"), r.get("sm_count")) != (CC, SMS):
        out.append(f"{name}: card cc {r.get('cc')} with {r.get('sm_count')} SMs, registered the RTX 5090's cc {CC}, {SMS} SMs")
    cfg = r.get("config") or {}
    if cfg.get("placement") != PLACEMENT[arm]:
        out.append(f"{name}: placement {cfg.get('placement')}, registered {PLACEMENT[arm]}")
    if arm in BUDGET_GB and (cfg.get("vram_gb"), cfg.get("dram_gb")) != BUDGET_GB[arm]:
        out.append(f"{name}: budgets {cfg.get('vram_gb')} / {cfg.get('dram_gb')} GiB, registered {BUDGET_GB[arm][0]} / "
                   f"{BUDGET_GB[arm][1]}")
    if cfg.get("graphs") is not False:
        out.append(f"{name}: graphs {cfg.get('graphs')}; every arm is registered eager")
    g = r.get("graph_status") or {}
    if any(v == "graph" for v in g.values()):
        out.append(f"{name}: the build captured decode buckets {sorted(b for b, v in g.items() if v == 'graph')}")
    if cfg.get("max_tokens_per_seq") != SERVED_TOKENS:
        out.append(f"{name}: {cfg.get('max_tokens_per_seq')} tokens per sequence, the server's {SERVED_TOKENS}")
    kn = r.get("knobs") or {}
    if any((kn.get(k) or "").strip() != v for k, v in KNOBS.items()):
        out.append(f"{name}: knobs {({k: kn.get(k) for k in KNOBS})}, registered {KNOBS} in every arm (Amendment 1)")
    out += head_faults(name, r)
    return out


def head_faults(name, r) -> list:
    """Amendment 1: the one lm_head swapped to the int8 head by the explicit knob, its rows resolved to 8 explicitly."""
    d = (r.get("levers") or {}).get("int8_head_bw") or {}
    rows = d.get("rows") or {}
    out = []
    if (d.get("mode"), d.get("source"), d.get("modules"), d.get("shape")) != (HEAD_MODE, HEAD_SOURCE, 1, HEAD_SHAPE):
        out.append(f"{name}: the head records mode {d.get('mode')} from {d.get('source')} ({d.get('modules')} module(s) at "
                   f"{d.get('shape')}), registered the int8 head by the explicit E4B_INT8_HEAD_BW=1 at {HEAD_SHAPE}")
    if (rows.get("rows_max"), rows.get("source")) != (ROWS_MAX, ROWS_SOURCE):
        out.append(f"{name}: the head's rows resolved {rows.get('rows_max')} from {rows.get('source')}, registered "
                   f"{ROWS_MAX} by the explicit E4B_INT8_HEAD_ROWS=8")
    return out


def placement_faults(name, r, arm) -> list:
    """The split as the solver resolved it (the record's ``placement``): the arm's tiers, its experts and bytes
    consistent, and its static layer paths."""
    p = r.get("placement") or {}
    ex, by = p.get("experts") or {}, p.get("bytes") or {}
    L, epl, bpe = p.get("moe_layers"), p.get("experts_per_layer"), p.get("bytes_per_expert")
    per = p.get("per_layer") or []
    out = []
    if not (isinstance(L, int) and L > 0 and isinstance(epl, list) and len(epl) == 1 and _num(bpe) and bpe > 0
            and len(per) == L and all(_num(ex.get(t)) and _num(by.get(t)) for t in TIERS)):
        return [f"{name}: the placement record is incomplete ({ {k: p.get(k) for k in ('moe_layers', 'experts_per_layer', 'bytes_per_expert', 'experts', 'bytes')} })"]
    if sum(ex[t] for t in TIERS) != L * epl[0] or any(by[t] != ex[t] * bpe for t in TIERS):
        out.append(f"{name}: experts {ex} and bytes {by} do not account for {L} x {epl[0]} experts at {bpe} B each")
    if any((x.get("vram", 0) + x.get("dram", 0) + x.get("nvme", 0)) != x.get("experts") for x in per) \
            or [sum(x.get(t, 0) for x in per) for t in TIERS] != [ex[t] for t in TIERS]:
        out.append(f"{name}: the per-layer split does not sum to the totals {ex}")
    paths = [x.get("path") for x in per]
    tiered = sum(1 for x in paths if x == "tiered")
    if arm == "A":
        if ex["vram"] != L * epl[0] or ex["dram"] or ex["nvme"] or by["dram"] or by["nvme"]:
            out.append(f"{name}: A holds experts {ex} (bytes {by}); registered every expert in VRAM, none in DRAM or on NVMe")
        if any(x != "collapse" for x in paths):
            out.append(f"{name}: {tiered} of {L} MoE layers not on the collapse at the build; registered every layer")
    else:
        if arm == "S" and not (by["vram"] > 0 and by["dram"] > 0 and by["nvme"] > 0):
            out.append(f"{name}: S's bytes per tier {by}; registered non-zero on VRAM, DRAM and NVMe")
        if arm == "C" and not (by["dram"] == 0 and ex["dram"] == 0 and by["nvme"] > 0):
            out.append(f"{name}: C's bytes per tier {by}; registered DRAM empty and NVMe non-empty")
        if tiered < 1:
            out.append(f"{name}: no MoE layer on the tiered forward at the build; registered at least one")
        if (p.get("budget_gb") or {}) != {"vram": BUDGET_GB[arm][0], "dram": BUDGET_GB[arm][1]}:
            out.append(f"{name}: the placement's budgets {p.get('budget_gb')}, registered {BUDGET_GB[arm]}")
    return out


def path_faults(name, paths, placement, arm) -> list:
    """The MoE-layer path counts from the engagement at one T, against the build's static paths: every layer called;
    a collapse layer took no tiered call and a tiered layer no collapse call; A every layer on the collapse; S and C at
    least one layer on the tiered forward."""
    per = (paths or {}).get("per_layer")
    static = [x.get("path") for x in (placement or {}).get("per_layer") or []]
    if not isinstance(per, list) or len(per) != len(static) or not per:
        return [f"{name}: path counts for {len(per) if isinstance(per, list) else None} layers, the placement names "
                f"{len(static)} (a missing count is unread, never 0)"]
    out = []
    if any(not _num(x.get(k)) for x in per for k in ("calls", "collapse", "tiered")):
        return [f"{name}: a layer's path counts are incomplete"]
    idle = [i for i, x in enumerate(per) if not x["calls"]]
    if idle:
        out.append(f"{name}: MoE layer(s) {idle[:6]} never called during the measurement")
    off = [i for i, (x, s) in enumerate(zip(per, static))
           if (s == "collapse" and x["tiered"]) or (s == "tiered" and x["collapse"]) or s not in ("collapse", "tiered")]
    if off:
        out.append(f"{name}: layer(s) {off[:6]} took a path other than the build's static one")
    tiered = sum(1 for x in per if x["tiered"] and not x["collapse"])
    if arm == "A" and any(x["tiered"] or not x["collapse"] for x in per):
        out.append(f"{name}: A's engagement shows {sum(1 for x in per if x['tiered'])} layer(s) on the tiered forward; "
                   "registered every layer on the collapse")
    if arm != "A" and tiered < 1:
        out.append(f"{name}: {arm}'s engagement shows no layer on the tiered forward; registered at least one")
    return out


def quality_faults(name, rec, arm, W) -> list:
    """Each T as registered: the group, the instrument's arms, the windows, every scored value present and k/128, no
    decode bucket captured in any pass, the pass KV at the server's tokens with the phase's cap, the path counts."""
    phases = rec.get("phases") or {}
    if sorted(phases) != sorted(str(T) for T in GROUPS):
        return [f"{name}: groups {sorted(phases)}, registered {list(GROUPS)}"]
    out = []
    for T in GROUPS:
        p = phases[str(T)] or {}
        n = f"{name} T={T}"
        if p.get("group") != T or p.get("arms") != list(PHASE_ARMS[arm]):
            out.append(f"{n}: group {p.get('group')} arms {p.get('arms')}, registered {T} {list(PHASE_ARMS[arm])}")
        if (p.get("windows"), p.get("prompt"), p.get("cont")) != ({"wikitext": W}, Q_PROMPT, Q_CONT):
            out.append(f"{n}: windows {p.get('windows')} x {p.get('cont')} after {p.get('prompt')}, registered {W} wikitext "
                       f"windows x {Q_CONT} after {Q_PROMPT}")
        per = (p.get("per_window") or {}).get("wikitext") or {}
        keys = ("R", "rep") if arm == "A" else ("ON",)
        for key in keys:
            xs = per.get(key) or []
            want = T if key == "rep" else W               # measure_phase draws the floor on the first group only
            if len(xs) != want:
                out.append(f"{n}: {len(xs)} {key} windows, registered {want}")
            if key == "rep":
                continue
            if sorted(x.get("window") for x in xs if isinstance(x.get("window"), int)) != list(range(W)):
                out.append(f"{n}: the {key} windows are not 0..{W - 1} once each")
            if any(not _num(x.get("nll")) for x in xs):
                out.append(f"{n}: a {key} window carries no NLL (a missing NLL is unread, never 0)")
            if key == "ON":
                if any(not _num(x.get("kl")) for x in xs):
                    out.append(f"{n}: a {key} window carries no KL (a missing KL is unread, never 0)")
                if any(not _num(x.get("argmax_agree")) for x in xs):
                    out.append(f"{n}: a {key} window carries no argmax agreement")
                else:
                    bad = [x.get("window") for x in xs if not integral_agree(x["argmax_agree"])]
                    if bad:
                        out.append(f"{n}: argmax agreement in window(s) {bad[:4]} is not k/{Q_CONT} for an integer k: the "
                                   f"window was not scored on the registered {Q_CONT} positions, so its counts are unread")
        eng = (p.get("engagement") or {}).get("wikitext") or {}
        passes = [e for a_ in eng.values() for e in (a_ or [])]
        if not passes:
            out.append(f"{n}: no pass engagement recorded (the captures are unread)")
        for e in passes:
            gs = e.get("graph_status") or {}
            st = e.get("graph_stats") or {}
            if any(v == "graph" for v in gs.values()) or any((v or {}).get("replays") for v in st.values()):
                out.append(f"{n}: a pass captured or replayed a decode bucket ({gs})")
                break
        sc = p.get("attn_split_cap") or {}
        qk = p.get("quality_kv") or {}
        if (qk.get("max_tokens_per_seq"), qk.get("blocks_per_seq")) != ([SERVED_TOKENS], [SERVED_BLOCKS]) \
                or (qk.get("passes") or 0) < W // T or qk.get("plan_clash") or qk.get("split_cap") != sc.get("cap"):
            out.append(f"{n}: the pass's KV {qk}: registered {SERVED_TOKENS} tokens ({SERVED_BLOCKS} blocks) with the "
                       f"server's cap {sc.get('cap')}, one KV per group of {T}")
        out += path_faults(n, p.get("paths"), rec.get("placement"), arm)
    return out


def _levers(r) -> dict:
    lv = r.get("levers") or {}
    h = lv.get("int8_head_bw") or {}
    sc = lv.get("attn_split_cap") or {}
    return {"head": [h.get("mode"), h.get("source"), (h.get("rows") or {}).get("rows_max"), (h.get("rows") or {}).get("source")],
            "split_cap": [sc.get("cap"), sc.get("source")],
            "fusion_modes": lv.get("fusion_modes"), "fusion_sources": lv.get("fusion_sources")}


def cross_faults(q: dict) -> list:
    """The same model, windows and resolved defaults in every phase."""
    out = []
    shas = {json.dumps(q[a]["phases"][str(T)].get("windows_sha256"), sort_keys=True) for a in ARMS for T in GROUPS
            if str(T) in (q[a].get("phases") or {})}
    shas |= {json.dumps({"wikitext": q[a].get("windows_sha256")}, sort_keys=True) for a in ARMS}
    if len(shas) != 1 or "null" in next(iter(shas)):
        out.append(f"the phases read different (or unrecorded) windows: {sorted(shas)}")
    revs = {(q[a].get("revision"), q[a].get("served_revision")) for a in ARMS}
    if len(revs) != 1:
        out.append(f"the phases loaded different checkpoint revisions: {sorted(map(str, revs))}")
    ar = {(q[a].get("arena") or {}).get("sha256") for a in ARMS}
    if len(ar) != 1 or None in ar:
        out.append(f"the phases served different (or unhashed) NF4 arenas: {sorted(map(str, ar))}")
    lv = {a: _levers(q[a]) for a in ARMS}
    for k in ("head", "split_cap", "fusion_modes", "fusion_sources"):
        if len({json.dumps(lv[a][k], sort_keys=True) for a in ARMS}) != 1:
            out.append(f"the {k.replace('_', ' ')} resolved differently between phases: { {a: lv[a][k] for a in ARMS} }")
    src = [lv["A"]["split_cap"][1]] + list((lv["A"]["fusion_sources"] or {}).values())
    if not lv["A"]["fusion_sources"] or any(not str(s).startswith("default") for s in src):
        out.append(f"the split cap and the fusions must stay at the e4b default (unset knobs): sources {src}")
    return out


def _vs(on: list, ref: list) -> dict:
    """An ON arm against A's R: the mean NLL gap, argmax as pooled counts, the mean KL and its per-window spread."""
    ag = agree_counts(on)
    kls = [x["kl"] for x in on]
    return {"bias": compare(on, ref)["bias"], "argmax_agree": ag[0] / ag[1], "argmax_counts": list(ag), "kl": _mean(kls),
            "kl_spread": {"min": min(kls), "median": statistics.median(kls), "max": max(kls)}, "n": len(on)}


def _argmax_vs(a: dict, b: dict):
    """S against C by the recorded argmax (reported): pooled agreement, or None when either is unread."""
    if not (isinstance(a, dict) and isinstance(b, dict) and a.get("argmax") and b.get("argmax")):
        return None
    same = total = 0
    for w, x in a["argmax"].items():
        y = b["argmax"].get(w) or []
        same += sum(1 for p, q in zip(x, y) if p == q)
        total += max(len(x), len(y))
    return same / total if total else None


def quality_read(q: dict) -> dict:
    """Per T: S against A (the ruling numbers), A's rep floor, and the reported C against A and S against C."""
    out = {}
    for T in GROUPS:
        t = str(T)
        wa = q["A"]["phases"][t]["per_window"]["wikitext"]
        rr, rep = wa["R"], wa.get("rep") or []
        ss = q["S"]["phases"][t]["per_window"]["wikitext"]["ON"]
        cc = q["C"]["phases"][t]["per_window"]["wikitext"]["ON"]
        kl_rep = [x["kl"] for x in rep if _num(x.get("kl"))]
        floor = _mean(kl_rep) if kl_rep else None
        s, c = _vs(ss, rr), _vs(cc, rr)
        am_s, am_c = q["S"]["phases"][t].get("argmax"), q["C"]["phases"][t].get("argmax")
        out[t] = {**s, "floor_kl": floor, "rep_identical": q["A"]["phases"][t].get("rep_identical"),
                  "C_vs_A": {**c, "bars": {"bias": Q_BIAS, "argmax": Q_ARGMAX}},
                  "S_vs_C": {"bias": compare(ss, cc)["bias"], "argmax_agree": _argmax_vs(am_s, am_c),
                             "kl": None, "kl_unread": "the registered phases score S and C against A only"},
                  "argmax_diff_positions": {a: (x or {}).get("diff_positions") for a, x in (("S", am_s), ("C", am_c))},
                  "argmax_consistent": {a: (x or {}).get("consistent") for a, x in (("S", am_s), ("C", am_c))},
                  "A_floor_arms": {k: {"bias": compare(wa[k], rr)["bias"],
                                       "kl": _mean([x["kl"] for x in wa[k] if _num(x.get("kl"))]) if any(
                                           _num(x.get("kl")) for x in wa[k]) else None}
                                   for k in ("half", "chunk", "mutant_scale") if wa.get(k)}}
    return out


def tripwire(qr: dict) -> list:
    """Every condition that fired, at every T (empty: none)."""
    fired = []
    for T in GROUPS:
        x = qr[str(T)]
        if abs(x["bias"]) > TRIP_BIAS:
            fired.append(f"T={T}: |NLL gap| {abs(x['bias']):.6f} > {TRIP_BIAS}")
        if not agree_at_least(tuple(x["argmax_counts"]), TRIP_ARGMAX):
            fired.append(f"T={T}: argmax agreement {x['argmax_agree']:.6f} < {TRIP_ARGMAX}")
        bound = max(TRIP_KL_MULT * (x["floor_kl"] or 0.0), TRIP_KL_ABS)   # a zero or absent floor: the absolute bound
        if x["kl"] > bound:
            fired.append(f"T={T}: KL {x['kl']:.6f} > max({TRIP_KL_MULT:g} x floor {x['floor_kl']}, {TRIP_KL_ABS}) = {bound:.6f}")
    return fired


def reported(q: dict) -> dict:
    """Each arm's split, layer paths, build time, host CPU and wall times (reported, never ruled)."""
    out = {}
    for a in ARMS:
        r, p = q[a], q[a].get("placement") or {}
        out[a] = {"experts": p.get("experts"), "bytes": p.get("bytes"), "resident_bytes": p.get("resident_bytes"),
                  "layers": p.get("layers"), "load_s": r.get("load_s"), "wall_s": r.get("wall_s"),
                  "host_cpu": r.get("host_cpu"),
                  "paths": {t: (ph.get("paths") or {}).get("totals") for t, ph in (r.get("phases") or {}).items()},
                  "seconds": {t: ph.get("seconds") for t, ph in (r.get("phases") or {}).items()}}
    return out


def reduce(quality: dict, e4b_sha: str) -> dict:
    out = {"lane": "P233", "verdict": None, "reasons": [], "tripwire": None,
           "bars": {"quality_bias": Q_BIAS, "quality_argmax": Q_ARGMAX, "q_windows": Q_WINDOWS,
                    "tripwire": {"bias": TRIP_BIAS, "argmax": TRIP_ARGMAX, "kl_mult": TRIP_KL_MULT, "kl_abs": TRIP_KL_ABS}}}
    missing = [f"quality_{a}" for a in ARMS if not quality.get(a) or quality[a].get("status") != "ok"]
    if missing:
        out.update(verdict="VOID", reasons=[f"record(s) missing or not ok: {missing}"], tripwire=["unread"])
        return out
    flags = {bool(quality[a].get("shakedown")) for a in ARMS}
    if len(flags) != 1:
        out.update(verdict="VOID", reasons=["the records mix shakedown and reading processes"], tripwire=["unread"])
        return out
    shake = flags.pop()
    out["shakedown"] = shake
    W = SHAKE_WINDOWS if shake else Q_WINDOWS
    void = []
    for a in ARMS:
        n = f"quality_{a}"
        void += record_faults(n, quality[a], e4b_sha, a, shake)
        void += placement_faults(n, quality[a], a)
        void += quality_faults(n, quality[a], a, W)
    if not void:
        void += cross_faults(quality)
    if void:
        out.update(verdict="VOID", reasons=void, tripwire=["unread"])
        return out
    # ---- read once integrity holds: the per-T bars' numbers and the tripwire (reported whatever the verdict)
    qr = quality_read(quality)
    out["quality"] = {t: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()} for t, x in qr.items()}
    out["quality_unrounded"] = qr
    fired = tripwire(qr)
    out["tripwire"] = fired
    out["reported"] = reported(quality)
    per_t, why_all = {}, []
    for T in GROUPS:
        x = qr[str(T)]
        why = []
        if abs(x["bias"]) > Q_BIAS:
            why.append(f"|NLL gap| {abs(x['bias']):.6f} > {Q_BIAS}")
        if not agree_at_least(tuple(x["argmax_counts"]), Q_ARGMAX):
            why.append(f"argmax agreement {x['argmax_agree']:.6f} < {Q_ARGMAX}")
        per_t[str(T)] = {"result": "SANE" if not why else "NOT_SANE", "failed": why}
        why_all += [f"T={T}: {y}" for y in why]
    out["per_T"] = per_t
    tail = (["-- after the written look the TRIPWIRE asks for"] if fired else [])
    if not why_all:
        rung = ("SOLVER_SANE", [f"T={T}: SANE" for T in GROUPS]
                + ["a doc or lane may propose serving at solver as an opt-in placement, citing this read; never a default, "
                   "and any speed claim needs a read of its own"] + tail)
    else:
        rung = ("SOLVER_NOT_SANE", why_all + ["no doc or lane proposes serving at solver until a later registration "
                                              "reads it SANE"])
    if shake:
        out.update(verdict="SHAKEDOWN", would_read=rung[0], reasons=["a shakedown record is never a reading"] + rung[1])
    else:
        out.update(verdict=rung[0], reasons=rung[1])
    return out


def tripwire_line(v: dict) -> str:
    t = v.get("tripwire")
    if t == ["unread"] or t is None:
        return "TRIPWIRE unread (VOID: nothing was read)"
    return "TRIPWIRE none" if not t else "TRIPWIRE " + "; ".join(t)


# ------------------------------------------------------------------ self-test --
E = "a" * 40
L_, E_, BPE = 48, 128, 2654208                            # Qwen3-30B-A3B's MoE layers and experts; the arena's row bytes
SPLIT = {"A": {"vram": L_ * E_, "dram": 0, "nvme": 0}, "S": {"vram": 3236, "dram": 1213, "nvme": 1695},
         "C": {"vram": 3236, "dram": 0, "nvme": 2908}}   # the explore box's splits


def _split_layers(tot: dict, tiered: int) -> list:
    """``tot`` spread over the 48 layers: the first ``tiered`` layers carry every non-VRAM expert."""
    per = [{"layer": i, "experts": E_, "vram": E_, "dram": 0, "nvme": 0} for i in range(L_)]
    for t in ("dram", "nvme"):
        left = tot[t]
        for i in range(tiered):
            take = min(left, per[i]["vram"] - 1, -(-tot[t] // tiered))
            per[i][t] += take
            per[i]["vram"] -= take
            left -= take
        assert left == 0, (t, left)
    for x in per:
        x["path"] = "collapse" if x["vram"] == E_ else "tiered"
    return per


def _placement(arm, *, experts=None, tiered=None, budget=None, bpe=BPE, paths=None, per=None):
    ex = dict(experts or SPLIT[arm])
    if tiered is None:
        tiered = 0 if not (ex["dram"] or ex["nvme"]) else 40
    layers = per if per is not None else _split_layers(ex, tiered) if (ex["dram"] or ex["nvme"]) else [
        {"layer": i, "experts": E_, "vram": E_, "dram": 0, "nvme": 0, "path": "collapse"} for i in range(L_)]
    if paths is not None:
        for x, p in zip(layers, paths):
            x["path"] = p
    if experts is not None and per is None and not (ex["dram"] or ex["nvme"]):
        layers[0]["vram"] = ex["vram"] - E_ * (L_ - 1)
    b = budget if budget is not None else ({"vram": 8.0, "dram": 3.0} if arm == "S" else {"vram": 8.0, "dram": 0.0}
                                           if arm == "C" else {"vram": 1.2, "dram": 6.0})
    return {"placement": PLACEMENT[arm], "budget_gb": b, "moe_layers": L_, "experts_per_layer": [E_],
            "bytes_per_expert": bpe, "experts": ex, "bytes": {t: ex[t] * BPE for t in TIERS},
            "resident_bytes": {"vram": ex["vram"] * BPE, "dram": ex["dram"] * BPE},
            "layers": {"collapse": sum(x["path"] == "collapse" for x in layers), "tiered": sum(x["path"] == "tiered" for x in layers)},
            "per_layer": layers}


def _paths(placement, *, drop=None, swap=None, idle=None):
    per = []
    for i, x in enumerate(placement.get("per_layer") or []):
        c = {"calls": 10, "collapse": 10 if x["path"] == "collapse" else 0, "tiered": 0 if x["path"] == "collapse" else 10,
             "dram_cpu": 6 if x["dram"] else 0, "dram_gpu": 4 if x["dram"] else 0}
        if swap == i:
            c["collapse"], c["tiered"] = c["tiered"], c["collapse"]
        if idle == i:
            c = {k: 0 for k in c}
        per.append(c)
    if drop:
        per = per[:-drop]
    return {"per_layer": per, "totals": {k: sum(x[k] for x in per) for k in ("calls", "collapse", "tiered")},
            "layers_collapse": sum(1 for x in per if x["collapse"] and not x["tiered"]),
            "layers_tiered": sum(1 for x in per if x["tiered"] and not x["collapse"])}


def _head(*, mode="1", source="explicit", rows=8, rows_source="explicit", shape=None, modules=1):
    return {"mode": mode, "modules": modules, "shape": list(shape or HEAD_SHAPE), "source": source,
            "rows": {"rows_max": rows, "source": rows_source}}


def _fake_phase(arm, T, W, *, bias=0.0, agree=1.0, kl=0.0, floor=0.0, n=None, group=None, tokens=SERVED_TOKENS,
                rep=True, sha="w", cont=Q_CONT, captured=False, placement=None, paths=None, cap=16, drop_field=None,
                kv_cap="same"):
    n = W if n is None else n
    keys = PHASE_ARMS[arm]
    per = {}
    for key in keys:
        cnt = T if key == "rep" else n
        per[key] = [{"window": i, "nll": 2.0 + 0.01 * i + (bias if key == "ON" else 0.0),
                     "argmax_agree": agree if key == "ON" else 1.0,
                     **({} if key == "R" else {"kl": kl if key == "ON" else floor})} for i in range(cnt)]
    if not rep and "rep" in per:
        per["rep"] = [{k: v for k, v in x.items() if k != "kl"} for x in per["rep"]]
    if drop_field:
        key, field = drop_field
        per[key][0].pop(field)
    blocks = -(-tokens // 16)
    gs = {str(b): ("graph" if captured else "eager: capture=False") for b in (1, 2, 4, 8, 16)}
    eng = {k: [{"graph_status": gs, "graph_stats": {str(b): {"replays": 0, "eager_steps": 127} for b in (1, 2, 4, 8, 16)}}]
           for k in keys}
    pl = placement or _placement(arm)
    am = None
    if arm != "A":
        k_ = round(agree * Q_CONT)
        am = {"argmax": {str(i): [0] * Q_CONT for i in range(n)},
              "diff_positions": {str(i): list(range(Q_CONT - k_)) for i in range(n)}, "consistent": True}
    return {"phase": "off" if arm == "A" else "on", "arms": list(keys), "windows": {"wikitext": n},
            "windows_sha256": {"wikitext": sha}, "group": group or T, "prompt": Q_PROMPT, "cont": cont,
            "per_window": {"wikitext": per}, "engagement": {"wikitext": eng},
            "rep_identical": {"wikitext": True} if arm == "A" else {}, "seconds": 60.0,
            "attn_split_cap": {"cap": cap, "source": "default-allowlisted", "plans": {}},
            "quality_kv": {"passes": n // T, "asked": [656], "max_tokens_per_seq": [tokens], "blocks_per_seq": [blocks],
                           "split_cap": cap if kv_cap == "same" else kv_cap, "plan_clash": [], "plan_bT": 11},
            "paths": paths if paths is not None else _paths(pl), "argmax": am}


def _fake_quality(arm, *, per_t=None, shake=False, e4b=E, sms=SMS, cc=CC, torch_v=TORCH, gnf4=GNF4_SHA, rev=REV,
                  served=REV, arena="f" * 64, wsha="w", placement=None, config=None, head=None, knobs=None,
                  cap=16, cap_source="default-allowlisted", modes=None, sources=None, graph_status=None, drop_t=None,
                  status="ok"):
    W = SHAKE_WINDOWS if shake else Q_WINDOWS
    pl = placement or _placement(arm)
    per_t = per_t or {}
    phases = {str(T): _fake_phase(arm, T, W, placement=pl, **{"sha": wsha, **per_t.get(T, {})})
              for T in GROUPS if T != drop_t}
    budget = BUDGET_GB.get(arm, (1.2, 6.0))
    fk = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
    return {"mode": "quality", "arm": arm, "status": status, "shakedown": shake, "e4b_sha": e4b, "gnf4_sha": gnf4,
            "torch": torch_v, "model": MODEL, "revision": rev, "served_revision": served, "sm_count": sms, "cc": cc,
            "config": {"placement": PLACEMENT[arm], "vram_gb": budget[0], "dram_gb": budget[1], "graphs": False,
                       "max_tokens_per_seq": SERVED_TOKENS, **(config or {})},
            "knobs": {**KNOBS, **(knobs or {})},
            "levers": {"int8_head_bw": head if head is not None else _head(),
                       "attn_split_cap": {"cap": cap, "source": cap_source},
                       "fusion_modes": modes or {k: "auto" for k in fk},
                       "fusion_sources": sources or {k: "default-allowlisted" for k in fk}},
            "graph_status": graph_status, "arena": {"sha256": arena}, "windows_sha256": wsha, "placement": pl,
            "host_cpu": {"model": "AMD EPYC 9554", "isa": {"avx2": True, "avx512": True, "amx": False},
                         "cpu_kernel_threads": 64},
            "load_s": 30.0, "wall_s": 900.0, "phases": phases}


def _cases() -> list:
    def qual(**over):
        return {a: _fake_quality(a, **over.get(a, {})) for a in ARMS}

    def r(q=None):
        return reduce(qual() if q is None else q, E)

    def v(q=None):
        return r(q)["verdict"]

    def sq(per_t):
        return qual(S={"per_t": per_t})

    def one(arm, **kw):                                  # one arm's record replaced, the others clean
        return {**qual(), arm: _fake_quality(arm, **kw)}

    pa = _placement("A")
    ps = _placement("S")
    pc = _placement("C")
    cases = [
        ("clean at both T: SOLVER_SANE, TRIPWIRE none",
         (lambda x: x["verdict"] == "SOLVER_SANE" and x["tripwire"] == [] and tripwire_line(x) == "TRIPWIRE none")(r())),
        ("the gap over 0.02 at T = 1 is SOLVER_NOT_SANE, named",
         (lambda x: x["verdict"] == "SOLVER_NOT_SANE" and any("T=1: |NLL gap|" in y for y in x["reasons"]))(r(sq({1: {"bias": 0.021}})))),
        ("the gap over 0.02 at T = 8 alone fails the verdict", v(sq({8: {"bias": -0.0201}})) == "SOLVER_NOT_SANE"),
        ("a gap just under 0.02 passes (and fires the tripwire)",
         (lambda x: x["verdict"] == "SOLVER_SANE" and x["tripwire"] and "written look" in x["reasons"][-1])(
             r(sq({1: {"bias": 0.0199999}})))),
        ("argmax 121/128 (0.9453) at T = 8 is NOT_SANE",
         (lambda x: x["verdict"] == "SOLVER_NOT_SANE" and x["per_T"]["8"]["result"] == "NOT_SANE"
          and x["per_T"]["1"]["result"] == "SANE")(r(sq({8: {"agree": 121 / 128}})))),
        ("argmax 122/128 (0.9531) passes", v(sq({1: {"agree": 122 / 128}})) == "SOLVER_SANE"),
        ("both bars failing are both named", len(r(sq({1: {"bias": 0.05, "agree": 0.5}}))["per_T"]["1"]["failed"]) == 2),
        ("the bars read the UNROUNDED gap: 0.0200004 (rounds to 0.02) fails",
         (lambda x: x["verdict"] == "SOLVER_NOT_SANE" and x["quality"]["1"]["bias"] == 0.02)(r(sq({1: {"bias": 0.0200004}})))),
        ("a window scored on 127 positions (121/127) is a VOID, never rounded", v(sq({8: {"agree": 121 / 127}})) == "VOID"),
        ("the integrality check: 121/128 is integral, 121/127 is not",
         integral_agree(121 / 128) and not integral_agree(121 / 127)),
        ("pooled counts at the 0.95 edge: 2919/3072 passes, 2918 misses",
         agree_at_least((2919, 3072), Q_ARGMAX) and not agree_at_least((2918, 3072), Q_ARGMAX)),
        ("pooled counts at the 0.98 edge: 3011/3072 holds, 3010 fires",
         agree_at_least((3011, 3072), TRIP_ARGMAX) and not agree_at_least((3010, 3072), TRIP_ARGMAX)),
        ("S missing a KL is a VOID, never read as 0", v(sq({1: {"drop_field": ("ON", "kl")}})) == "VOID"),
        ("S missing an NLL is a VOID", v(sq({8: {"drop_field": ("ON", "nll")}})) == "VOID"),
        ("S missing an argmax agreement is a VOID", v(sq({8: {"drop_field": ("ON", "argmax_agree")}})) == "VOID"),
        ("C missing a KL is a VOID (a scored arm)", v(qual(C={"per_t": {1: {"drop_field": ("ON", "kl")}}})) == "VOID"),
        ("A's R missing an NLL is a VOID", v(qual(A={"per_t": {8: {"drop_field": ("R", "nll")}}})) == "VOID"),
        ("a window index repeated is a VOID",
         (lambda q: (q["S"]["phases"]["1"]["per_window"]["wikitext"]["ON"][1].update(window=0), v(q))[1])(qual()) == "VOID"),
        ("S at 23 windows is a VOID", v(sq({1: {"n": 23}})) == "VOID"),
        ("A's rep on two windows at T = 1 is a VOID",
         (lambda q: (q["A"]["phases"]["1"]["per_window"]["wikitext"]["rep"].append(
             dict(q["A"]["phases"]["1"]["per_window"]["wikitext"]["rep"][0])), v(q))[1])(qual()) == "VOID"),
        # ---- C is reported, never ruling
        ("C failing both bars never rules", (lambda x: x["verdict"] == "SOLVER_SANE" and x["quality"]["1"]["C_vs_A"]["bias"] == 0.5)(
            r(qual(C={"per_t": {T: {"bias": 0.5, "agree": 0.5, "kl": 0.5} for T in GROUPS}})))),
        ("S against C is reported: the DRAM tier's share",
         (lambda x: abs(x["quality_unrounded"]["8"]["S_vs_C"]["bias"] - 0.003) < 1e-12 and x["verdict"] == "SOLVER_SANE")(
             r(qual(S={"per_t": {8: {"bias": 0.004}}}, C={"per_t": {8: {"bias": 0.001}}})))),
        ("S against C argmax read from the recorded argmax", r()["quality"]["1"]["S_vs_C"]["argmax_agree"] == 1.0),
        ("the KL spread is reported", r(sq({8: {"kl": 0.0007}}))["quality"]["8"]["kl_spread"]["max"] == 0.0007),
        ("the argmax-difference positions are reported",
         r(sq({1: {"agree": 125 / 128}}))["quality"]["1"]["argmax_diff_positions"]["S"]["0"] == [0, 1, 2]),
        ("each arm's split is reported", r()["reported"]["S"]["experts"] == SPLIT["S"]),
        ("the host CPU is reported", r()["reported"]["S"]["host_cpu"]["isa"]["avx512"] is True),
        ("A's floor arms are reported", "mutant_scale" in r()["quality"]["1"]["A_floor_arms"]),
        # ---- the tripwire
        ("|NLL gap| 0.0051 fires and the verdict holds",
         (lambda x: x["verdict"] == "SOLVER_SANE" and len(x["tripwire"]) == 1 and "T=8: |NLL gap|" in x["tripwire"][0])(
             r(sq({8: {"bias": -0.0051}})))),
        ("|NLL gap| just under 0.005 does not fire", r(sq({1: {"bias": 0.0049999}}))["tripwire"] == []),
        ("argmax 125/128 (0.9766) fires", any("argmax" in y for y in r(sq({1: {"agree": 125 / 128}}))["tripwire"])),
        ("argmax 126/128 (0.9844) does not fire", r(sq({1: {"agree": 126 / 128}}))["tripwire"] == []),
        ("a zero floor falls to 0.002: KL 0.0019 does not fire", r(sq({8: {"kl": 0.0019}}))["tripwire"] == []),
        ("a zero floor: KL 0.0021 fires", any("KL" in y for y in r(sq({8: {"kl": 0.0021}}))["tripwire"])),
        ("an absent floor falls to 0.002 (no rep KL)",
         r(qual(A={"per_t": {1: {"rep": False}}}, S={"per_t": {1: {"kl": 0.0019}}}))["tripwire"] == []
         and r(qual(A={"per_t": {1: {"rep": False}}}, S={"per_t": {1: {"kl": 0.0021}}}))["tripwire"] != []),
        ("a floor of 0.001: KL 0.0029 holds, 0.0031 fires",
         r(qual(A={"per_t": {8: {"floor": 0.001}}}, S={"per_t": {8: {"kl": 0.0029}}}))["tripwire"] == []
         and r(qual(A={"per_t": {8: {"floor": 0.001}}}, S={"per_t": {8: {"kl": 0.0031}}}))["tripwire"] != []),
        ("the floor is read per T: T = 1's floor does not lift T = 8's bound",
         r(qual(A={"per_t": {1: {"floor": 0.01}}}, S={"per_t": {8: {"kl": 0.0021}}}))["tripwire"] != []),
        ("every fired condition is listed, at every T",
         len(r(sq({1: {"bias": 0.006, "agree": 124 / 128, "kl": 0.01}, 8: {"bias": -0.006}}))["tripwire"]) == 4),
        ("the tripwire is reported on SOLVER_NOT_SANE too",
         (lambda x: x["verdict"] == "SOLVER_NOT_SANE" and x["tripwire"])(r(sq({1: {"bias": 0.03}})))),
        ("the tripwire never changes the verdict (SOLVER_SANE with all three fired)",
         v(sq({T: {"bias": 0.01, "agree": 123 / 128, "kl": 0.05} for T in GROUPS})) == "SOLVER_SANE"),
        ("C never fires the tripwire", r(qual(C={"per_t": {1: {"bias": 0.5, "kl": 0.5}}}))["tripwire"] == []),
        ("TRIPWIRE prints each condition", tripwire_line(r(sq({1: {"bias": 0.006}}))).startswith("TRIPWIRE T=1: |NLL gap|")),
        ("a VOID's tripwire is unread, never none",
         tripwire_line(r(sq({1: {"n": 23}}))) == "TRIPWIRE unread (VOID: nothing was read)"),
        # ---- VOID: records, pins, card
        ("a missing phase (C) is VOID", v({a: x for a, x in qual().items() if a != "C"}) == "VOID"),
        ("a missing phase (A) is VOID", v({a: x for a, x in qual().items() if a != "A"}) == "VOID"),
        ("a phase not ok is VOID", v(one("S", status="failed")) == "VOID"),
        ("a phase missing T = 8 is VOID", v(one("S", drop_t=8)) == "VOID"),
        ("the records mix a shakedown and a reading", v(one("C", shake=True)) == "VOID"
         and v({**qual(), "C": {**_fake_quality("C"), "shakedown": True}}) == "VOID"),
        ("wrong commit", v(one("S", e4b="b" * 40)) == "VOID"),
        ("another grouped-nf4-gemm", v(one("A", gnf4="929a6906c12deba256a4ad4b18162681719f5cd1")) == "VOID"),
        ("another torch", v(one("C", torch_v="2.9.0+cu128")) == "VOID"),
        ("another revision in every phase", v(qual(**{a: {"rev": "b" * 40} for a in ARMS})) == "VOID"),
        ("a card that is not the 5090 (188 SMs)", v(one("S", sms=188)) == "VOID"),
        ("a card at cc 8.9", v(one("A", cc="8.9")) == "VOID"),
        ("graphs on in an arm", v(one("S", config={"graphs": True})) == "VOID"),
        ("graphs unrecorded", v(one("A", config={"graphs": None})) == "VOID"),
        ("a bucket captured at the build", v(one("A", graph_status={"1": "graph"})) == "VOID"),
        ("a pass that captured a bucket", v(qual(C={"per_t": {8: {"captured": True}}})) == "VOID"),
        ("a pass that replayed a bucket",
         (lambda q: (q["A"]["phases"]["1"]["engagement"]["wikitext"]["R"][0]["graph_stats"]["1"].update(replays=3), v(q))[1])(
             qual()) == "VOID"),
        ("no pass engagement recorded",
         (lambda q: (q["S"]["phases"]["8"].update(engagement={}), v(q))[1])(qual()) == "VOID"),
        ("the KV at P115's 656 tokens", v(sq({8: {"tokens": 656}})) == "VOID"),
        ("the pass KV without the server's cap", v(sq({1: {"kv_cap": None}})) == "VOID"),
        ("128 positions registered: 127 is VOID", v(sq({1: {"cont": 127}})) == "VOID"),
        ("a group other than T", v(sq({8: {"group": 4}})) == "VOID"),
        ("A scoring P232's two arms, not the instrument's five",
         (lambda q: (q["A"]["phases"]["1"].update(arms=["R", "rep"]), v(q))[1])(qual()) == "VOID"),
        # ---- VOID: the same model and windows in every arm
        ("the phases on different windows", v(qual(C={"wsha": "x"})) == "VOID"),
        ("unrecorded windows", v(qual(**{a: {"wsha": None} for a in ARMS})) == "VOID"),
        ("one phase on another served revision", v(one("S", served="c" * 40)) == "VOID"),
        ("one phase on another arena", v(one("C", arena="e" * 64)) == "VOID"),
        ("an unhashed arena", v(qual(**{a: {"arena": None} for a in ARMS})) == "VOID"),
        # ---- VOID: the head and the defaults
        ("the head off in a phase", v(one("S", head=_head(mode="0", source="explicit"))) == "VOID"),
        ("the head on by the default, not the explicit knob", v(qual(**{a: {"head": _head(source="default-allowlisted")}
                                                                    for a in ARMS})) == "VOID"),
        ("the rows at 1", v(qual(**{a: {"head": _head(rows=1)} for a in ARMS})) == "VOID"),
        ("the rows at 8 by the default, not the explicit knob",
         v(qual(**{a: {"head": _head(rows_source="default-allowlisted")} for a in ARMS})) == "VOID"),
        ("the head at another shape", v(one("A", head=_head(shape=[151936, 4096]))) == "VOID"),
        ("a knob record off the registered value", v(one("C", knobs={"E4B_INT8_HEAD_ROWS": None})) == "VOID"),
        ("the split cap resolved differently between phases", v(one("S", cap=8)) == "VOID"),
        ("the split cap named explicitly in every phase", v(qual(**{a: {"cap_source": "explicit"} for a in ARMS})) == "VOID"),
        ("a fusion mode resolved differently", v(one("C", modes={"E4B_PAGED_FUSE_QKV": "0", "E4B_FUSE_T1_GLUE": "auto",
                                                               "E4B_FUSE_T1_GLUE_R2": "auto", "E4B_FUSE_ROUTER_EPI": "auto"})) == "VOID"),
        ("a fusion named explicitly in every phase",
         v(qual(**{a: {"sources": {"E4B_PAGED_FUSE_QKV": "explicit", "E4B_FUSE_T1_GLUE": "default-allowlisted",
                                   "E4B_FUSE_T1_GLUE_R2": "default-allowlisted", "E4B_FUSE_ROUTER_EPI": "default-allowlisted"}}
                   for a in ARMS})) == "VOID"),
        # ---- VOID: the placement proof, A
        ("A with experts in DRAM", v(one("A", placement=_placement("A", experts={"vram": 6100, "dram": 44, "nvme": 0}, tiered=4,
                                                                     paths=["collapse"] * L_)))
         == "VOID"),
        ("A with an expert on NVMe", v(one("A", placement=_placement("A", experts={"vram": 6143, "dram": 0, "nvme": 1}, tiered=1,
                                                                       paths=["collapse"] * L_)))
         == "VOID"),
        ("A with a layer off the collapse at the build",
         v(one("A", placement=_placement("A", paths=["tiered"] + ["collapse"] * (L_ - 1)))) == "VOID"),
        ("A's engagement with a tiered call", v(qual(A={"per_t": {8: {"paths": _paths(pa, swap=3)}}})) == "VOID"),
        ("A with a layer never called", v(qual(A={"per_t": {1: {"paths": _paths(pa, idle=0)}}})) == "VOID"),
        ("S with a resident layer never called", v(qual(S={"per_t": {1: {"paths": _paths(ps, idle=47)}}})) == "VOID"),
        ("A's path counts for 47 layers", v(qual(A={"per_t": {1: {"paths": _paths(pa, drop=1)}}})) == "VOID"),
        ("A's path counts missing", v(qual(A={"per_t": {8: {"paths": {}}}})) == "VOID"),
        # ---- VOID: the placement proof, S
        ("S with DRAM empty", v(one("S", placement=_placement("S", experts={"vram": 3236, "dram": 0, "nvme": 2908}))) == "VOID"),
        ("S with NVMe empty", v(one("S", placement=_placement("S", experts={"vram": 3236, "dram": 2908, "nvme": 0}))) == "VOID"),
        ("S with VRAM empty", v(one("S", placement=_placement("S", experts={"vram": 0, "dram": 3000, "nvme": 3144},
                                                               per=[{"layer": i, "experts": E_, "vram": 0, "dram": 62 + (i < 24),
                                                                     "nvme": 66 - (i < 24), "path": "tiered"} for i in range(L_)])))
         == "VOID"),
        ("S at another DRAM budget", v(one("S", config={"dram_gb": 2.0})) == "VOID"),
        ("S's placement record at another budget", v(one("S", placement=_placement("S", budget={"vram": 8.0, "dram": 2.0})))
         == "VOID"),
        ("S with no tiered layer at the build",
         v(one("S", placement=_placement("S", paths=["collapse"] * L_))) == "VOID"),
        ("S's engagement with no tiered call",
         v(qual(S={"per_t": {1: {"paths": {"per_layer": [{"calls": 10, "collapse": 10, "tiered": 0}] * L_}}}})) == "VOID"),
        ("a tiered layer taking a collapse call", v(qual(S={"per_t": {8: {"paths": _paths(ps, swap=0)}}})) == "VOID"),
        ("bytes that do not match experts x bytes per expert",
         v(one("S", placement={**_placement("S"), "bytes_per_expert": BPE + 1})) == "VOID"),
        ("experts that do not sum to 48 x 128",
         v(one("S", placement=_placement("S", experts={"vram": 3236, "dram": 1213, "nvme": 1694}))) == "VOID"),
        ("an incomplete placement record", v(one("S", placement={"experts": SPLIT["S"]})) == "VOID"),
        # ---- VOID: the placement proof, C
        ("C with DRAM non-empty", v(one("C", placement=_placement("C", experts=SPLIT["S"]))) == "VOID"),
        ("C with NVMe empty", v(one("C", placement=_placement("C", experts={"vram": L_ * E_, "dram": 0, "nvme": 0}))) == "VOID"),
        ("C at S's budgets", v(one("C", config={"dram_gb": 3.0})) == "VOID"),
        ("C with no tiered layer", v(qual(C={"per_t": {8: {"paths": {"per_layer": [{"calls": 4, "collapse": 4, "tiered": 0}] * L_}}}}))
         == "VOID"),
        ("C at all-vram", v(one("C", config={"placement": "all-vram"})) == "VOID"),
        # ---- the shakedown
        ("a shakedown (8 windows, another card) reads SHAKEDOWN, never a verdict",
         (lambda x: x["verdict"] == "SHAKEDOWN" and x["would_read"] == "SOLVER_SANE" and x["tripwire"] == [])(
             reduce({a: _fake_quality(a, shake=True, sms=84, cc="8.9") for a in ARMS}, E))),
        ("a shakedown's would-read names a failed bar",
         (lambda x: x["verdict"] == "SHAKEDOWN" and x["would_read"] == "SOLVER_NOT_SANE")(
             reduce({a: _fake_quality(a, shake=True, **({"per_t": {1: {"bias": 0.03}}} if a == "S" else {})) for a in ARMS}, E))),
        ("a shakedown at 24 windows is VOID (its registered count is 8)",
         reduce({a: {**_fake_quality(a), "shakedown": True} for a in ARMS}, E)["verdict"] == "VOID"),
        ("a shakedown still reads the placement proof",
         reduce({a: _fake_quality(a, shake=True, **({"placement": _placement("S", experts=SPLIT["C"])} if a == "S" else {}))
                 for a in ARMS}, E)["verdict"] == "VOID"),
        ("the clean fixtures are what they claim", (pa["layers"], ps["layers"]["tiered"] >= 1, pc["experts"]["dram"])
         == ({"collapse": L_, "tiered": 0}, True, 0)),
    ]
    return cases


def self_test() -> int:
    try:
        cases = _cases()
    except Exception as e:  # noqa: BLE001 -- a rule that crashes on a registered case fails its self-test
        print(f"p233_reduce self-test FAILED [{type(e).__name__}: {e}]")
        return 1
    bad = [n for n, ok in cases if not ok]
    print(f"p233_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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

    def load(name):
        f = os.path.join(a.dir, name)
        return json.load(open(f)) if os.path.exists(f) else None
    quality = {q: r for q in ARMS if (r := load(f"quality_{q}.json"))}
    v = reduce(quality, a.e4b_sha)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P233_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)[:1500]}")
    print(tripwire_line(v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
