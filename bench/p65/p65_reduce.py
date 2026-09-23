#!/usr/bin/env python3
"""bench/p65/p65_reduce.py -- P65's comparison: which per-expert ranking survives wikitext -> c4val1, and does activation
entropy earn a place in S-C's selector (bench/p65/P65-PREREG.md). The rules below are the registration's, in code.

    python bench/p65/p65_reduce.py <run-dir> [--md RESULTS-p65.md] [--json p65_verdicts.json]

``<run-dir>`` holds ``census_<family>.json`` from ``p65_census.py``. A missing family is NOT_READ, never a pass.

Signals, per (layer, expert), per census (text in {wikitext, c4val1} x half in {0, 1, full}):
  * ``entropy``   -- Colla-Q's rho of the expert OUTPUT (the dn row's ``entropy.rho``); high = weak = wants bits;
  * ``rel_act``   -- RTN int4-b32 activation-weighted relative error, both projections: sqrt(rel_gu^2 + rel_dn^2);
  * ``freq``      -- token-slots routed (the rows' ``rows``); its top-k set comes from the package's
                     ``hot_sets_from_profile`` over a profile written from the census, so the routing ranking IS the
                     residency engine's;
  * ``rho_x_err`` -- rho * rel_act^2, Colla-Q's objective L = rho x ||e - e~||^2 in relative form (the arm an entropy
                     selector would rank by).
An expert enters a comparison only if it saw >= MIN_ROWS rows in EVERY census that comparison reads.

Statistics (all WITHIN a layer, then averaged over layers -- Colla-Q allocates per MoE block and so does
``hot_sets_from_profile``; a pooled ranking would be inflated by layer identity, which no text changes):
  * r_split(s)      -- Spearman(half 0, half 1) of the SAME text, averaged over the two texts: the resampling ceiling;
  * r_cross_half(s) -- Spearman(wikitext half i, c4val1 half j) over the four (i, j): the domain shift at the SAME
                       sample size as r_split, so the two are comparable;
  * penalty(s)      -- r_split - r_cross_half: what the domain costs beyond what resampling already costs;
  * r_cross_full(s), and Colla-Q's own Table-4 statistic (per-layer cosine, averaged) -- reported, not decided on:
    the cosine of two positive, near-constant vectors is near 1 whatever their order.
Bootstrap CIs resample LAYERS (B = 2000, seed 65), deterministic.

Registered rules:
  * P0: a family is NOT_READ unless its census's selfcheck passed and it is complete (every requested layer, both
    texts, both halves, the full rows) -- a census the arm alarm cut is never read on the layers it reached.
  * s SURVIVES iff r_cross_half(s) >= 0.5 AND penalty(s) <= 0.15.
  * entropy is REDUNDANT with rel_act iff |Spearman(entropy, rel_act)| > 0.8 within layers on either full text.
  * selector (the issue's step 3), per family where the per-expert premise holds:
      entropy survives, not redundant, rho_x_err survives, rel_act survives -> TWO_ARMS (rel_act; rho_x_err)
      entropy survives, not redundant, rel_act does NOT survive            -> ENTROPY_ONLY
      entropy redundant, rel_act survives                                   -> REL_ACT_ONLY (entropy redundant)
      entropy does not survive, rel_act survives                            -> REL_ACT_ONLY (entropy refused)
      (both fail)                                                           -> NO_ONE_TEXT_SELECTOR
    (TWO_ARMS whose product does not survive falls to REL_ACT_ONLY, said so.)
  * Colla-Q's comparative claim: paired per-layer r_cross_half(entropy) - r_cross_half(freq), bootstrap 95 % CI:
    REPLICATES if the CI is above 0, REVERSED if below, UNRESOLVED otherwise.
  * the per-expert premise: P44-a's P3 (Granite HOLDS, Mixtral REFUTED, both read in RESULTS-p44.md); for a family P44
    did not read (OLMoE), P3 on this census's RTN c4val1 full rows (P44's own ``tail_fraction``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "p44"))
from p44_reduce import P3_TAIL_FRACTION, expert_errors, tail_fraction  # noqa: E402

SIGNALS = ("entropy", "rel_act", "freq", "rho_x_err")
# reported beside the registered four and read by NO rule: the two projections' errors separately (the A2000 rehearsal
# found them anti-correlated within a layer, so the combination is visible, not hidden), and the two descriptive
# entropies expert_entropy.py records (input rho, output energy entropy)
DESCRIPTIVE = ("rel_gu", "rel_dn", "rho_in", "h_energy")
TEXTS = ("wikitext", "c4val1")
MIN_ROWS = 32
TOP_FRACTION = 0.10
SURVIVE_MIN = 0.5
SURVIVE_PENALTY = 0.15
REDUNDANT = 0.8
BOOT_B = 2000
BOOT_SEED = 65
FAMILIES = ("granite", "olmoe", "mixtral")
# P44-a's P3 read per family (RESULTS-p44.md:20-21): the premise of a per-expert precision dial
P44_PREMISE = {"granite": ("HOLDS", "P44-a P3 0.0992 (recipe/GPTQ, 32/32 layers)"),
               "mixtral": ("REFUTED", "P44-a P3 0.2266 (RTN, 16/32 layers)")}


# ------------------------------------------------------------------------------------------- signals
def expert_signals(census: dict, text: str, half) -> dict:
    """{(layer, expert): {rows, entropy, rel_act, rel_gu, rel_dn, freq, rho_x_err, rho_in, h_energy}} for one census."""
    by = {}
    for r in census["rows"]:
        if r.get("text") != text or r.get("half") != half:
            continue
        d = by.setdefault((int(r["layer"]), int(r["expert"])), {"rows": int(r.get("rows", 0))})
        rel = (r.get("rtn") or {}).get("rel_act")
        ent = r.get("entropy") or {}
        if r["role"] == "gu":
            d["rel_gu"] = rel
            d["rho_in"] = ent.get("rho")
        elif r["role"] == "dn":
            d["rel_dn"] = rel
            d["entropy"] = ent.get("rho")
            d["h_energy"] = ent.get("h_energy")
    for d in by.values():
        g, n = d.get("rel_gu"), d.get("rel_dn")
        d["rel_act"] = math.sqrt(g * g + n * n) if g is not None and n is not None else None
        d["freq"] = d["rows"]
        e = d.get("entropy")
        d["rho_x_err"] = e * d["rel_act"] ** 2 if e is not None and d["rel_act"] is not None else None
    return by


def censuses(census: dict) -> dict:
    return {(t, h): expert_signals(census, t, h) for t in TEXTS for h in (0, 1, "full")}


# ------------------------------------------------------------------------------------------- statistics
def ranks(v):
    """Average ranks (1-based), ties shared."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def spearman(x, y):
    return pearson(ranks(x), ranks(y)) if len(x) >= 3 else None


def cosine(x, y):
    nx = math.sqrt(sum(a * a for a in x))
    ny = math.sqrt(sum(b * b for b in y))
    if nx <= 0 or ny <= 0:
        return None
    return sum(a * b for a, b in zip(x, y)) / (nx * ny)


def _eligible(cs, keys, sig):
    """Experts valid for `sig` in every census named by `keys` (and routed >= MIN_ROWS in each)."""
    common = None
    for k in keys:
        ok = {e for e, d in cs[k].items() if d["rows"] >= MIN_ROWS and d.get(sig) is not None}
        common = ok if common is None else common & ok
    return common or set()


def per_layer(cs, ka, kb, sig_a, sig_b=None, fn=spearman) -> dict:
    """{layer: fn(sig_a in census ka, sig_b in census kb)} over the experts eligible in both."""
    sig_b = sig_b or sig_a
    keep = _eligible(cs, (ka,), sig_a) & _eligible(cs, (kb,), sig_b)
    layers = {}
    for (layer, e) in keep:
        layers.setdefault(layer, []).append(e)
    out = {}
    for layer, es in sorted(layers.items()):
        es.sort()
        v = fn([cs[ka][(layer, e)][sig_a] for e in es], [cs[kb][(layer, e)][sig_b] for e in es])
        if v is not None:
            out[layer] = v
    return out


def mean_over(d: dict):
    return sum(d.values()) / len(d) if d else None


def avg_maps(maps):
    """Per-layer average of several {layer: value} maps (layers present in all)."""
    maps = [m for m in maps if m]
    if not maps:
        return {}
    common = set.intersection(*(set(m) for m in maps))
    return {layer: sum(m[layer] for m in maps) / len(maps) for layer in sorted(common)}


def bootstrap_ci(d: dict, B: int = BOOT_B, seed: int = BOOT_SEED):
    """95 % CI of the mean over layers, resampling layers with replacement; deterministic."""
    vals = [d[k] for k in sorted(d)]
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [means[int(0.025 * B)], means[int(0.975 * B) - 1]]


def stability(cs, sig) -> dict:
    split = avg_maps([per_layer(cs, (t, 0), (t, 1), sig) for t in TEXTS])
    cross = avg_maps([per_layer(cs, ("wikitext", i), ("c4val1", j), sig) for i in (0, 1) for j in (0, 1)])
    pen = {layer: split[layer] - cross[layer] for layer in split if layer in cross}
    full = per_layer(cs, ("wikitext", "full"), ("c4val1", "full"), sig)
    cos = per_layer(cs, ("wikitext", "full"), ("c4val1", "full"), sig, fn=cosine)
    r_cross, penalty = mean_over(cross), mean_over(pen)
    survives = (r_cross is not None and penalty is not None
                and r_cross >= SURVIVE_MIN and penalty <= SURVIVE_PENALTY)
    return {"r_split": mean_over(split), "r_cross_half": r_cross, "r_cross_half_ci": bootstrap_ci(cross),
            "penalty": penalty, "penalty_ci": bootstrap_ci(pen), "r_cross_full": mean_over(full),
            "collaq_cosine": mean_over(cos), "layers": len(cross), "survives": survives,
            "_cross": cross}


def layer_eta2(cs, key, sig):
    """Share of the signal's variance that is between layers (1 = the layer alone decides a pooled ranking)."""
    vals = {}
    for (layer, _e), d in cs[key].items():
        if d["rows"] >= MIN_ROWS and d.get(sig) is not None:
            vals.setdefault(layer, []).append(d[sig])
    allv = [v for vs in vals.values() for v in vs]
    if len(allv) < 2:
        return None
    m = sum(allv) / len(allv)
    tot = sum((v - m) ** 2 for v in allv)
    between = sum(len(vs) * (sum(vs) / len(vs) - m) ** 2 for vs in vals.values())
    return between / tot if tot > 0 else None


def top_set(cs, key, sig, frac=TOP_FRACTION, eligible=None):
    items = [(d[sig], e) for e, d in cs[key].items()
             if d["rows"] >= MIN_ROWS and d.get(sig) is not None and (eligible is None or e in eligible)]
    k = max(1, round(frac * len(items)))
    items.sort(key=lambda t: (-t[0], t[1]))
    return {e for _v, e in items[:k]}, len(items)


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else None


def chance_jaccard(k, n):
    """Expected Jaccard of two independent random k-subsets of n (to first order): I = k^2/n, J = I / (2k - I)."""
    if n <= 0 or k <= 0:
        return None
    i = k * k / n
    return i / (2 * k - i)


def freq_hot_sets(census: dict, text: str, half, hot_per_layer: int):
    """The package's own routing ranking: write the census's routing as an ``E4B_EXPERT_PROFILE`` JSONL and ask
    ``hot_sets_from_profile``."""
    from experts4bit_qlora.engines.expert_profile import hot_sets_from_profile
    sig = expert_signals(census, text, half)
    n_exp = int(census.get("experts") or (1 + max(e for _l, e in sig)))
    layers = sorted({layer for layer, _e in sig})
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "profile.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"row": "meta"}) + "\n")
            for layer in layers:
                f.write(json.dumps({"row": "layer", "layer_id": layer, "num_experts": n_exp}) + "\n")
            for (layer, e), d in sorted(sig.items()):
                if d["rows"] > 0:
                    f.write(json.dumps({"row": "expert", "layer_id": layer, "expert_id": e, "hits": 1,
                                        "tokens_routed": d["rows"]}) + "\n")
        return hot_sets_from_profile(p, hot_per_layer)


# ------------------------------------------------------------------------------------------- the read
def premise(family: str, census: dict) -> dict:
    """The per-expert premise: P44-a's P3 read where it exists, else P3 on this census (RTN, c4val1 full). This
    census's RTN tail is reported for EVERY family as ``rtn_tail_c4val1`` (descriptive where P44-a's read decides:
    Granite's P44-a read was GPTQ, this census is RTN)."""
    sub = {"rows": [r for r in census["rows"] if r.get("text") == "c4val1" and r.get("half") == "full"]}
    tf = tail_fraction(expert_errors(sub, "rtn")["errors"])
    f = tf.get("fraction_for_share")
    if family in P44_PREMISE:
        v, why = P44_PREMISE[family]
        return {"premise": v == "HOLDS", "source": why, "rtn_tail_c4val1": tf}
    return {"premise": f is not None and f <= P3_TAIL_FRACTION,
            "source": f"P3 on this census (RTN, c4val1 full): fraction {f}", **tf, "rtn_tail_c4val1": tf}


def selector(st: dict, redundant: bool) -> str:
    e, r, p = st["entropy"]["survives"], st["rel_act"]["survives"], st["rho_x_err"]["survives"]
    if e and not redundant:
        if r:
            return "TWO_ARMS" if p else "REL_ACT_ONLY (rho_x_err does not survive although both parts do)"
        return "ENTROPY_ONLY"
    if r:
        return "REL_ACT_ONLY (entropy redundant)" if e and redundant else "REL_ACT_ONLY (entropy refused)"
    return "NO_ONE_TEXT_SELECTOR"


def reduce_family(family: str, census: dict) -> dict:
    cs = censuses(census)
    st = {s: stability(cs, s) for s in SIGNALS}
    over = {}
    for t in TEXTS:
        k = (t, "full")
        for a, b in (("entropy", "rel_act"), ("entropy", "freq"), ("rel_act", "freq"), ("rel_gu", "rel_dn")):
            elig = _eligible(cs, (k,), a) & _eligible(cs, (k,), b)
            sa, n = top_set(cs, k, a, eligible=elig)
            sb, _ = top_set(cs, k, b, eligible=elig)
            over[f"{t}:{a}~{b}"] = {"spearman_within_layer": mean_over(per_layer(cs, k, k, a, b)),
                                    "top10_jaccard_pooled": jaccard(sa, sb),
                                    "chance_jaccard": chance_jaccard(len(sa), n), "experts": n}
    red_r = [over[f"{t}:entropy~rel_act"]["spearman_within_layer"] for t in TEXTS]
    redundant = any(r is not None and abs(r) > REDUNDANT for r in red_r)
    diff = {layer: st["entropy"]["_cross"][layer] - st["freq"]["_cross"][layer]
            for layer in st["entropy"]["_cross"] if layer in st["freq"]["_cross"]}
    ci = bootstrap_ci(diff)
    claim = ("UNRESOLVED" if ci is None else "REPLICATES" if ci[0] > 0 else "REVERSED" if ci[1] < 0 else "UNRESOLVED")
    pooled = {}
    for s in SIGNALS:
        elig = _eligible(cs, (("wikitext", "full"), ("c4val1", "full")), s)
        a, n = top_set(cs, ("wikitext", "full"), s, eligible=elig)
        b, _ = top_set(cs, ("c4val1", "full"), s, eligible=elig)
        pooled[s] = {"top10_jaccard_wikitext_vs_c4val1": jaccard(a, b), "chance": chance_jaccard(len(a), n),
                     "experts": n, "layer_eta2_c4val1": layer_eta2(cs, ("c4val1", "full"), s)}
    E = int(census.get("experts") or 0)
    k = max(1, round(TOP_FRACTION * E)) if E else 1
    hs_w = freq_hot_sets(census, "wikitext", "full", k)
    hs_c = freq_hot_sets(census, "c4val1", "full", k)
    per_layer_j = [jaccard(set(a), set(b)) for a, b in zip(hs_w, hs_c) if a or b]
    prem = premise(family, census)
    out = {"family": family, "layers": census.get("layers_censused"), "experts_per_layer": E,
           "selfcheck_ok": (census.get("selfcheck") or {}).get("ok"),
           "stability": {s: {k2: v for k2, v in st[s].items() if not k2.startswith("_")} for s in SIGNALS},
           "stability_descriptive": {s: {k2: v for k2, v in stability(cs, s).items() if not k2.startswith("_")}
                                     for s in DESCRIPTIVE},
           "overlap": over, "pooled": pooled,
           "hot_sets_from_profile": {"hot_per_layer": k, "mean_jaccard_wikitext_vs_c4val1":
                                     (sum(per_layer_j) / len(per_layer_j)) if per_layer_j else None},
           "entropy_redundant_with_rel_act": redundant,
           "collaq_claim": {"verdict": claim, "mean_diff": mean_over(diff), "ci": ci},
           "premise": prem,
           "selector": selector(st, redundant) if prem["premise"] else "NOT_WRITTEN (no per-expert premise)",
           "selector_if_premise": selector(st, redundant)}
    return out


def load_census(run_dir: str, fam: str):
    """``census_<fam>.json``, or its gzip (how committed receipts carry it); None when neither exists."""
    p = os.path.join(run_dir, f"census_{fam}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    if os.path.exists(p + ".gz"):
        import gzip
        with gzip.open(p + ".gz", "rt") as f:
            return json.load(f)
    return None


def reduce(run_dir: str) -> dict:
    out = {}
    for fam in FAMILIES:
        c = load_census(run_dir, fam)
        if c is None:
            out[fam] = {"verdict": "NOT_READ", "reason": "no census file"}
            continue
        gate = p0_gate(c)
        if gate:
            out[fam] = {"verdict": "NOT_READ", "reason": gate}
            continue
        out[fam] = reduce_family(fam, c)
    return out


def p0_gate(census: dict) -> str | None:
    """P0 (the instrument gate): the selfcheck passed and the census is complete -- every requested layer, both texts,
    both halves and the full rows. Returns the reason a family is NOT_READ, or None."""
    if not (census.get("selfcheck") or {}).get("ok"):
        return "the census's selfcheck did not pass"
    req = census.get("layers_requested")
    if req is None or census.get("layers_censused") != req:
        return f"incomplete: {len(census.get('layers_censused') or [])} of {len(req or [])} requested layers censused"
    cells = {(r.get("text"), r.get("half")) for r in census.get("rows", [])}
    want = {(t, h) for t in TEXTS for h in (0, 1, "full")}
    if cells != want:
        return f"incomplete: census cells {sorted(map(str, cells))} are not {sorted(map(str, want))}"
    return None


def _f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(v: dict) -> str:
    lines = ["| family | signal | r_split | r_cross_half [95 % CI] | penalty | r_cross_full | Colla-Q cosine | "
             "top-10 % Jaccard (chance) | survives |", "|---|---|---|---|---|---|---|---|---|"]
    for fam, r in v.items():
        if "stability" not in r:
            lines.append(f"| {fam} | — | {r.get('verdict')}: {r.get('reason')} | | | | | | |")
            continue
        for s in SIGNALS:
            st, po = r["stability"][s], r["pooled"][s]
            ci = st["r_cross_half_ci"]
            ci_s = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            lines.append(f"| {fam} | {s} | {_f(st['r_split'])} | {_f(st['r_cross_half'])}{ci_s} | {_f(st['penalty'])} | "
                         f"{_f(st['r_cross_full'])} | {_f(st['collaq_cosine'], 4)} | "
                         f"{_f(po['top10_jaccard_wikitext_vs_c4val1'])} ({_f(po['chance'])}) | "
                         f"{'yes' if st['survives'] else 'no'} |")
    lines += ["", "Descriptive, read by no rule:", "",
              "| family | signal | r_split | r_cross_half | penalty | rel_gu~rel_dn (wiki / c4) |", "|---|---|---|---|---|---|"]
    for fam, r in v.items():
        if "stability_descriptive" not in r:
            continue
        o = r["overlap"]
        gd = (f"{_f(o['wikitext:rel_gu~rel_dn']['spearman_within_layer'])} / "
              f"{_f(o['c4val1:rel_gu~rel_dn']['spearman_within_layer'])}")
        for s in DESCRIPTIVE:
            st = r["stability_descriptive"][s]
            lines.append(f"| {fam} | {s} | {_f(st['r_split'])} | {_f(st['r_cross_half'])} | {_f(st['penalty'])} | {gd} |")
    lines += ["", "| family | entropy~rel_act (wiki / c4) | entropy redundant | Colla-Q claim (entropy − freq) | "
              "RTN tail, c4val1 (P3 statistic) | premise | selector |", "|---|---|---|---|---|---|---|"]
    for fam, r in v.items():
        if "stability" not in r:
            continue
        o = r["overlap"]
        c = r["collaq_claim"]
        ci = c["ci"]
        lines.append(f"| {fam} | {_f(o['wikitext:entropy~rel_act']['spearman_within_layer'])} / "
                     f"{_f(o['c4val1:entropy~rel_act']['spearman_within_layer'])} | "
                     f"{'yes' if r['entropy_redundant_with_rel_act'] else 'no'} | {c['verdict']} "
                     f"({_f(c['mean_diff'])}{f' [{ci[0]:.3f}, {ci[1]:.3f}]' if ci else ''}) | "
                     f"{_f(r['premise']['rtn_tail_c4val1'].get('fraction_for_share'), 4)} | "
                     f"{'holds' if r['premise']['premise'] else 'no'} ({r['premise']['source']}) | {r['selector']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="P65 reducer")
    ap.add_argument("run_dir")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    v = reduce(a.run_dir)
    md = render_md(v)
    print(md)
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(v, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
