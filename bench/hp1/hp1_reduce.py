#!/usr/bin/env python3
"""HP1's reducer (HP1-PREREG.md, "Readings and decision rules"): the K readout that picks the cell E between the blocks, and the
P readout that reads E against the stronger competitor after them. Mechanical: no human choice enters either subcommand.

Inputs: one or more evaluator JSONs written by `bench/hp1/hp1_eval.py` (the K and P blocks may be evaluated separately). Their
`arms` lists are merged; a tag seen twice (in one file or across files) refuses the whole input. Per arm, y is:
  - status SCORED        -> `evaluator.row_mean` (the one evaluator on the arm's training base);
  - any other status     -> no y (the run does not exist for the readout). Every verdict uses EVALUATOR numbers only: an in-loop
                            number never becomes a y (an older evaluator's OWN_BASE_ONLY arm is unusable here).
Lower held-out loss is better; every one-sided test points that way.

Arm tags (the driver's naming; see ARM_ROLES below): K cells `hp1_k<init><dtype>_s<seed>` on the SELECTION seeds; on the
CONFIRMATION seeds, the selected cell `hp1_e_<cell>_s<seed>`, K00 as shipped `hp1_k00_s<seed>`, and the competitors
`hp1_unsloth_s<seed>` / `hp1_axolotl_s<seed>`. A tag that matches no role (or a role on the wrong seed set) is listed under
`ignored` with the reason and never enters a number.

`select` (K readout, run on the box between the blocks):
  Needs all 12 K runs (4 cells x 3 selection seeds) with a y, else verdict K_INCOMPLETE, exit 3, nothing selected (and no
  HP1_SELECTED line). Two-way ANOVA with interaction on the balanced 2x2x3 design (df 1/1/1/8). Main effects are
  native - kaiming and bf16 - fp32 (positive = the shipped setting gives the higher loss), SE = sqrt(MSE (1/6 + 1/6)),
  one-sided p for effect > 0 from t(df 8). A knob is CAUSAL iff its effect > 0 at one-sided p < 0.05, else NOT-SHOWN. The
  interaction contrast L = k00 - k10 - k01 + k11 (init effect at bf16 minus init effect at fp32), SE = sqrt(MSE 4/n), two-sided
  p from F(1, 8). sigma_K = sqrt(MSE). Regime: interaction p >= 0.05 -> "main-effects", else "cell-means" (main effects
  reported, not deciding). In BOTH regimes E = the cheapest cell whose mean is within 1 sigma_K of the best (lowest) cell mean,
  cheapness K00 < K10 < K01 < K11: the prereg states the no-interaction rule as that same rule, so the regimes choose
  identically and the regime is recorded. Prints exactly one stdout line `HP1_SELECTED=<cell>`.

`verdict` (P readout, after the P block):
  Re-derives E from the K arms with the rule above (a `hp1_e_<cell>` tag naming another cell refuses the input). E needs all 3
  confirmation seeds, else P_INCOMPLETE (exit 3). A competitor without all 3 confirmation seeds usable is EXCLUDED (reported);
  both excluded -> P_INCOMPLETE. C* = the included competitor with the lower mean over the confirmation seeds. Paired by seed,
  delta_s = y(E, s) - y(C*, s); mean, SE = sd(delta, ddof 1)/sqrt(3), CI90 = mean +- t(0.95, df 2) SE. sigma_seed = the pooled
  within-arm SD over EVERY arm's runs with a y (each K cell on the selection seeds, each P arm on the confirmation seeds, each
  its own group): sqrt(sum of squared deviations / sum of (n_i - 1)). m = min(max(sigma_seed, 0.005), 0.02). Verdict, checked
  in this order: BEATS iff CI upper < 0 AND |mean| > sigma_seed; BEHIND iff CI lower > 0 AND |mean| > sigma_seed; PARITY iff
  CI lower >= -m AND CI upper <= +m; else INCONCLUSIVE. The same reading is made for E vs C* (the headline), K00 vs C* on the
  confirmation seeds (the same reading when E = K00) and E vs each competitor separately. Writes JSON (--out) and a markdown
  table (--md); prints `HP1_VERDICT=<verdict>`.

Data order (peer review): every arm carries `order` = {order_seed, n_rows, order_sha} (or null). A comparison is only paired if
the arms saw the same data order. `select`: each K run's `order.order_seed` equals its tag's seed and the 4 cells at a seed share
one `order_sha`, else K_INCOMPLETE. `verdict`: at each confirmation seed s, every P run with a y (E, K00, unsloth, axolotl) has a
non-null `order` with order_seed == s and one shared `order_sha`, else P_INCOMPLETE naming the arm and seed (exit 3).

Exit codes: 0 read; 2 refused input (duplicate tag, tag/seed mismatch, unreadable JSON, E-tag mismatch); 3 incomplete.
The t distribution is computed here (regularized incomplete beta by Lentz's continued fraction), stdlib only.
"""

import argparse
import json
import math
import re
import sys

SELECTION_SEEDS = (3407, 3408, 3409)
CONFIRMATION_SEEDS = (3410, 3411, 3412)

# The 2x2 K cells: k<init><dtype>. init 0 = native N(0, 1/r), 1 = kaiming (PEFT's); dtype 0 = bf16 expert adapters, 1 = fp32.
CELLS = {
    "k00": {"init": "native", "dtype": "bf16", "flags": "--lora-init native --adapter-dtype native", "note": "shipped"},
    "k10": {"init": "kaiming", "dtype": "bf16", "flags": "--lora-init matched:s --adapter-dtype native", "note": ""},
    "k01": {"init": "native", "dtype": "fp32", "flags": "--lora-init native --adapter-dtype fp32", "note": ""},
    "k11": {"init": "kaiming", "dtype": "fp32", "flags": "--lora-init matched:s --adapter-dtype fp32", "note": "TC1b matched"},
}
CHEAPNESS = ("k00", "k10", "k01", "k11")     # bf16 adapters save memory and time; init is free
SHIPPED = "k00"
COMPETITORS = ("unsloth", "axolotl")

# Tag -> role. Each pattern's groups are (name, seed); `seeds` is the seed set the role lives on.
ARM_ROLES = (
    {"role": "K", "pattern": re.compile(r"^hp1_(k[01][01])_s(\d+)$"), "seeds": SELECTION_SEEDS,
     "what": "K cell on a selection seed"},
    {"role": "P", "pattern": re.compile(r"^hp1_(k00)_s(\d+)$"), "seeds": CONFIRMATION_SEEDS,
     "what": "K00 as shipped on a confirmation seed"},
    {"role": "P", "pattern": re.compile(r"^hp1_e_(k[01][01])_s(\d+)$"), "seeds": CONFIRMATION_SEEDS, "prefix": "E:",
     "what": "the selected cell E on a confirmation seed"},
    {"role": "P", "pattern": re.compile(r"^hp1_(unsloth|axolotl)_s(\d+)$"), "seeds": CONFIRMATION_SEEDS,
     "what": "competitor on a confirmation seed"},
)

ALPHA = 0.05               # one-sided for the knobs, two-sided for the interaction
CI_TAIL = 0.95             # two-sided 90 % CI
MARGIN_FLOOR, MARGIN_CAP = 0.005, 0.02
EXIT_OK, EXIT_REFUSED, EXIT_INCOMPLETE = 0, 2, 3


class RefusedInput(Exception):
    """The input cannot be read under the registered rules (exit 2)."""


# ----------------------------------------------------------------------------------------------------------------- t / F
def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta (modified Lentz; Numerical Recipes betacf)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError(f"betacf did not converge (a={a}, b={b}, x={x})")


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b), 0 <= x <= 1."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T > t) for Student's t with df degrees of freedom: tail = I_{df/(df+t^2)}(df/2, 1/2) / 2."""
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def t_cdf(t, df):
    """P(T <= t)."""
    return 1.0 - t_sf(t, df)


def t_ppf(p, df):
    """The p-quantile of t(df), by bisection on the CDF (monotone; 200 halvings is far below float resolution)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"t_ppf needs 0 < p < 1, got {p}")
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def f_sf(f, d1, d2):
    """P(F > f) for F(d1, d2) = I_{d2/(d2 + d1 f)}(d2/2, d1/2)."""
    if f <= 0.0:
        return 1.0
    if math.isinf(f):
        return 0.0
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def _ratio(num, den):
    """num/den where a zero denominator (no within-cell spread) reads as +-inf, and 0/0 as 0."""
    if den > 0:
        return num / den
    return 0.0 if num == 0 else math.copysign(math.inf, num)


def _mean(xs):
    return sum(xs) / len(xs)


def _ss(xs):
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs)


def _num(x):
    """JSON-safe float (inf is not JSON)."""
    return None if x is None or math.isinf(x) or math.isnan(x) else x


# ------------------------------------------------------------------------------------------------------------- inputs
def usable_y(arm):
    """(y, source, None) for an arm the readout can use, else (None, None, why)."""
    status = arm.get("status")
    if status == "SCORED":
        y = (arm.get("evaluator") or {}).get("row_mean")
        return (y, "evaluator.row_mean", None) if y is not None else (None, None, "SCORED without evaluator.row_mean")
    if status == "OWN_BASE_ONLY":
        return None, None, "OWN_BASE_ONLY: every verdict uses evaluator numbers only"
    return None, None, f"status {status}" + (f": {arm['reason']}" if arm.get("reason") else "")


def classify_tag(tag):
    """(role, group, seed, None) for a registered tag on its own seed set, else (None, None, None, why)."""
    if not isinstance(tag, str):
        return None, None, None, "no tag"
    why = "tag matches no HP1 role"
    for spec in ARM_ROLES:
        mt = spec["pattern"].match(tag)
        if mt:
            seed = int(mt.group(2))
            if seed in spec["seeds"]:
                return spec["role"], spec.get("prefix", "") + mt.group(1), seed, None
            why = f"{spec['what']}, but seed {seed} is not in {list(spec['seeds'])}"
    return None, None, None, why


def load_arms(paths):
    """Merge the evaluators' arms. Returns (index {(role, group): {seed: rec}}, records, ignored). Refuses duplicate tags."""
    seen, records, ignored, index = {}, [], [], {}
    for path in paths:
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as exc:
            raise RefusedInput(f"cannot read evaluator JSON {path}: {exc}") from exc
        if not isinstance(doc.get("arms"), list):
            raise RefusedInput(f"{path}: no top-level `arms` list (not an hp1_eval.py output)")
        for arm in doc["arms"]:
            tag = arm.get("tag")
            if tag in seen:
                raise RefusedInput(f"duplicate tag {tag!r}: in {seen[tag]} and {path}")
            seen[tag] = path
            role, group, seed, why = classify_tag(tag)
            if role is None:
                ignored.append({"tag": tag, "file": path, "why": why})
                continue
            if arm.get("seed") is not None and int(arm["seed"]) != seed:
                raise RefusedInput(f"tag {tag!r} names seed {seed} but the arm's receipt says seed {arm['seed']}")
            y, source, unusable = usable_y(arm)
            rec = {"tag": tag, "file": path, "framework": arm.get("framework"), "role": role, "group": group, "seed": seed,
                   "status": arm.get("status"), "y": y, "y_source": source, "unusable": unusable,
                   "own_base_only": arm.get("status") == "OWN_BASE_ONLY" and y is not None, "order": arm.get("order")}
            records.append(rec)
            index.setdefault((role, group), {})[seed] = rec
    return index, records, ignored


def order_problems(recs):
    """Same-data-order check over runs used together: per seed, every run's `order` is non-null with order_seed == its seed,
    and all runs at that seed share one order_sha. Returns a list of {tag, seed, why} (empty = paired)."""
    problems, by_seed = [], {}
    for rec in recs:
        by_seed.setdefault(rec["seed"], []).append(rec)
    for seed in sorted(by_seed):
        shas = {}
        for rec in by_seed[seed]:
            order = rec["order"]
            if not isinstance(order, dict) or not order.get("order_sha"):
                problems.append({"tag": rec["tag"], "seed": seed, "why": "no data order recorded (order is null or has no order_sha)"})
                continue
            try:
                order_seed = int(order.get("order_seed"))
            except (TypeError, ValueError):
                order_seed = None
            if order_seed != seed:
                problems.append({"tag": rec["tag"], "seed": seed, "why": f"order_seed {order.get('order_seed')} != seed {seed}"})
                continue
            shas[rec["tag"]] = order["order_sha"]
        if len(set(shas.values())) > 1:
            listing = {tag: str(sha)[:12] for tag, sha in sorted(shas.items())}
            problems += [{"tag": tag, "seed": seed, "why": f"order_sha differs across arms at seed {seed}: {listing}"}
                         for tag in sorted(shas)]
    return problems


def series(index, role, group, seeds):
    """({seed: y}, [recs], missing) for one arm over a seed set; complete iff `missing` is empty."""
    runs = index.get((role, group), {})
    ys, recs, missing = {}, [], []
    for s in seeds:
        rec = runs.get(s)
        if rec is None:
            missing.append({"arm": group, "seed": s, "why": "no run"})
        elif rec["y"] is None:
            missing.append({"arm": group, "seed": s, "tag": rec["tag"], "why": rec["unusable"]})
        else:
            ys[s] = rec["y"]
            recs.append(rec)
    return ys, recs, missing


# ---------------------------------------------------------------------------------------------------------- K readout
def two_way_anova(cell_ys):
    """Balanced 2x2 ANOVA with interaction on {cell: [y per seed]}; init = first digit, dtype = second (CELLS)."""
    n = len(cell_ys["k00"])
    if n < 2 or any(len(cell_ys[c]) != n for c in CHEAPNESS):
        raise ValueError("two_way_anova needs a balanced design with >= 2 runs per cell")
    mean = {c: _mean(cell_ys[c]) for c in CHEAPNESS}
    grand = _mean(list(mean.values()))
    native, kaiming = (mean["k00"] + mean["k01"]) / 2, (mean["k10"] + mean["k11"]) / 2
    bf16, fp32 = (mean["k00"] + mean["k10"]) / 2, (mean["k01"] + mean["k11"]) / 2
    init_mean = {"k00": native, "k01": native, "k10": kaiming, "k11": kaiming}
    dtype_mean = {"k00": bf16, "k10": bf16, "k01": fp32, "k11": fp32}
    ss_init = 2 * n * ((native - grand) ** 2 + (kaiming - grand) ** 2)
    ss_dtype = 2 * n * ((bf16 - grand) ** 2 + (fp32 - grand) ** 2)
    ss_inter = n * sum((mean[c] - init_mean[c] - dtype_mean[c] + grand) ** 2 for c in CHEAPNESS)
    ss_error = sum(_ss(cell_ys[c]) for c in CHEAPNESS)
    df_error = 4 * (n - 1)
    mse = ss_error / df_error
    se_main = math.sqrt(mse * (1 / (2 * n) + 1 / (2 * n)))

    def main_effect(name, effect, ss):
        t = _ratio(effect, se_main)
        f = _ratio(ss, mse)
        p_one = t_sf(t, df_error)
        return {"knob": name, "effect": effect, "se": se_main, "t": _num(t), "df": df_error, "p_one_sided": p_one,
                "F": _num(f), "p_F_two_sided": f_sf(f, 1, df_error), "ss": ss,
                "reading": "CAUSAL" if effect > 0 and p_one < ALPHA else "NOT-SHOWN"}

    contrast = mean["k00"] - mean["k10"] - mean["k01"] + mean["k11"]
    se_inter = math.sqrt(mse * 4 / n)
    f_inter = _ratio(ss_inter, mse)
    return {
        "n_per_cell": n, "cell_means": mean, "grand_mean": grand,
        "anova": {"ss_init": ss_init, "ss_dtype": ss_dtype, "ss_interaction": ss_inter, "ss_error": ss_error,
                  "df_init": 1, "df_dtype": 1, "df_interaction": 1, "df_error": df_error, "mse": mse},
        "init": main_effect("init (native - kaiming)", native - kaiming, ss_init),
        "dtype": main_effect("dtype (bf16 - fp32)", bf16 - fp32, ss_dtype),
        "interaction": {"contrast": "k00 - k10 - k01 + k11 (init effect at bf16 minus init effect at fp32)",
                        "estimate": contrast, "se": se_inter, "t": _num(_ratio(contrast, se_inter)), "F": _num(f_inter),
                        "df": [1, df_error], "p_two_sided": f_sf(f_inter, 1, df_error)},
        "sigma_K": math.sqrt(mse),
    }


def select_cell(cell_means, sigma):
    """The cheapest cell (CHEAPNESS order) whose mean is within `sigma` of the best (lowest) cell mean."""
    best_cell = min(CHEAPNESS, key=lambda c: (cell_means[c], CHEAPNESS.index(c)))
    bound = cell_means[best_cell] + sigma
    eligible = [c for c in CHEAPNESS if cell_means[c] <= bound]
    return eligible[0], best_cell, bound, eligible


def k_readout(index):
    """The K rule on the selection seeds. verdict OK (with `selected`) or K_INCOMPLETE (with `missing`)."""
    cell_ys, recs, missing = {}, [], []
    for cell in CHEAPNESS:
        ys, rs, miss = series(index, "K", cell, SELECTION_SEEDS)
        cell_ys[cell] = [ys[s] for s in SELECTION_SEEDS if s in ys]
        recs += rs
        missing += miss
    out = {"seeds": list(SELECTION_SEEDS), "cells": CELLS, "cheapness": list(CHEAPNESS),
           "own_base_only": sorted(r["tag"] for r in recs if r["own_base_only"])}
    out["order_problems"] = order_problems(recs)
    if missing or out["order_problems"]:
        out.update(verdict="K_INCOMPLETE", missing=missing, selected=None)
        return out
    out["per_run"] = {c: dict(zip(map(str, SELECTION_SEEDS), cell_ys[c])) for c in CHEAPNESS}
    out.update(two_way_anova(cell_ys))
    p_inter = out["interaction"]["p_two_sided"]
    regime = "main-effects" if p_inter >= ALPHA else "cell-means"
    selected, best, bound, eligible = select_cell(out["cell_means"], out["sigma_K"])
    out.update(verdict="OK", regime=regime, main_effects_decide=regime == "main-effects", best_cell=best,
               within_bound=bound, eligible=eligible, selected=selected,
               rule="E = cheapest cell (k00<k10<k01<k11) with mean <= best mean + sigma_K; same rule in both regimes")
    return out


# ---------------------------------------------------------------------------------------------------------- P readout
def pooled_sd(groups):
    """sqrt(sum_i SS_i / sum_i (n_i - 1)) over groups with >= 2 values; (sd, df, per-group detail)."""
    ss_tot, df_tot, detail = 0.0, 0, []
    for name, ys in groups:
        if len(ys) >= 2:
            ss_tot += _ss(ys)
            df_tot += len(ys) - 1
        detail.append({"group": name, "n": len(ys), "sd": math.sqrt(_ss(ys) / (len(ys) - 1)) if len(ys) >= 2 else None})
    return (math.sqrt(ss_tot / df_tot) if df_tot else None), df_tot, detail


def paired_reading(name, a_ys, a_recs, b_ys, b_recs, sigma, margin):
    """Delta_s = y(a, s) - y(b, s) over the confirmation seeds; CI90 by t(df n-1); the verdict rule in the registered order."""
    seeds = list(CONFIRMATION_SEEDS)
    deltas = [a_ys[s] - b_ys[s] for s in seeds]
    k = len(deltas)
    mean = _mean(deltas)
    sd = math.sqrt(_ss(deltas) / (k - 1))
    se = sd / math.sqrt(k)
    tq = t_ppf(CI_TAIL, k - 1)
    lo, hi = mean - tq * se, mean + tq * se
    if hi < 0 and abs(mean) > sigma:
        verdict = "BEATS"
    elif lo > 0 and abs(mean) > sigma:
        verdict = "BEHIND"
    elif lo >= -margin and hi <= margin:
        verdict = "PARITY"
    else:
        verdict = "INCONCLUSIVE"
    obo = sorted(r["tag"] for r in a_recs + b_recs if r["own_base_only"])
    return {"reading": name, "seeds": seeds, "deltas": dict(zip(map(str, seeds), deltas)), "mean": mean, "sd": sd, "se": se,
            "df": k - 1, "t_crit": tq, "ci90": [lo, hi], "m": margin, "sigma_seed": sigma, "verdict": verdict,
            "any_own_base_only": bool(obo), "own_base_only": obo}


def p_readout(index, k):
    """The P rule. Returns (result, exit code)."""
    cell = k["selected"]
    e_groups = sorted(g for (role, g) in index if role == "P" and g.startswith("E:"))
    wrong = [g for g in e_groups if g != "E:" + cell]
    if wrong:
        raise RefusedInput(f"the K rule selects {cell}, but the P block carries E tags for {[g[2:] for g in wrong]}")
    e_group = "E:" + cell if (cell != SHIPPED or ("P", "E:" + cell) in index) else SHIPPED
    e_ys, e_recs, e_miss = series(index, "P", e_group, CONFIRMATION_SEEDS)
    k00_ys, k00_recs, k00_miss = series(index, "P", SHIPPED, CONFIRMATION_SEEDS)
    comp = {}
    for c in COMPETITORS:
        ys, recs, miss = series(index, "P", c, CONFIRMATION_SEEDS)
        comp[c] = {"ys": ys, "recs": recs, "missing": miss, "included": not miss,
                   "mean": _mean([ys[s] for s in CONFIRMATION_SEEDS]) if not miss else None}

    # sigma_seed: every arm's runs with a y, each K cell (selection seeds) and each P arm (confirmation seeds) its own group.
    groups = [(f"K:{c}", list(series(index, "K", c, SELECTION_SEEDS)[0].values())) for c in CHEAPNESS]
    p_groups = sorted({g for (role, g) in index if role == "P"} | {e_group})
    groups += [(f"P:{g}", list(series(index, "P", g, CONFIRMATION_SEEDS)[0].values())) for g in p_groups]
    sigma, sigma_df, sigma_detail = pooled_sd(groups)
    margin = min(max(sigma, MARGIN_FLOOR), MARGIN_CAP) if sigma is not None else None

    out = {"selected": cell, "e_arm": e_group, "seeds": list(CONFIRMATION_SEEDS),
           "sigma_seed": sigma, "sigma_seed_df": sigma_df, "sigma_seed_groups": sigma_detail, "m": margin,
           "competitors": {c: {"included": v["included"], "mean": v["mean"], "missing": v["missing"],
                               "per_seed": {str(s): y for s, y in v["ys"].items()},
                               "own_base_only": sorted(r["tag"] for r in v["recs"] if r["own_base_only"])}
                           for c, v in comp.items()},
           "excluded_competitors": [c for c in COMPETITORS if not comp[c]["included"]],
           "e_per_seed": {str(s): y for s, y in e_ys.items()}, "e_missing": e_miss}
    included = [c for c in COMPETITORS if comp[c]["included"]]
    problems = []
    if e_miss:
        problems.append(f"E ({e_group}) lacks confirmation seeds: {e_miss}")
    for c in COMPETITORS:                          # registered: the headline needs BOTH competitors' confirmation cells
        if not comp[c]["included"]:
            problems.append(f"competitor {c} lacks confirmation seeds {comp[c]['missing']}: no headline (HP1-PREREG, P readout)")
    if sigma is None:
        problems.append("sigma_seed undefined (no arm has two runs with a y)")
    p_recs = [rec for (role, _g), runs in index.items() if role == "P" for rec in runs.values() if rec["y"] is not None]
    out["order_problems"] = order_problems(p_recs)
    problems += [f"data order: {x['tag']} at seed {x['seed']}: {x['why']}" for x in out["order_problems"]]
    if problems:
        side = []                                  # the complete competitor's reading, reported and labelled, never the headline
        if not e_miss and sigma is not None and not out["order_problems"]:
            for c in included:
                side.append(dict(paired_reading(f"E ({cell}) vs {c}", e_ys, e_recs, comp[c]["ys"], comp[c]["recs"], sigma, margin),
                                 role="per-competitor (no headline)"))
        out.update(verdict="P_INCOMPLETE", problems=problems, c_star=None, readings=side)
        return out, EXIT_INCOMPLETE

    c_star = min(included, key=lambda c: (comp[c]["mean"], COMPETITORS.index(c)))
    out["c_star"] = c_star
    readings = [dict(paired_reading(f"E ({cell}) vs C* ({c_star})", e_ys, e_recs, comp[c_star]["ys"], comp[c_star]["recs"],
                                    sigma, margin), role="headline")]
    if e_group == SHIPPED or cell == SHIPPED:
        readings.append({"reading": f"K00 vs C* ({c_star})", "role": "k00", "same_as_headline": True,
                         "verdict": readings[0]["verdict"]})
    elif k00_miss:
        readings.append({"reading": f"K00 vs C* ({c_star})", "role": "k00", "verdict": "INCOMPLETE", "missing": k00_miss})
    else:
        readings.append(dict(paired_reading(f"K00 vs C* ({c_star})", k00_ys, k00_recs, comp[c_star]["ys"],
                                            comp[c_star]["recs"], sigma, margin), role="k00"))
    for c in COMPETITORS:
        name = f"E ({cell}) vs {c}"
        if comp[c]["included"]:
            readings.append(dict(paired_reading(name, e_ys, e_recs, comp[c]["ys"], comp[c]["recs"], sigma, margin),
                                 role="per-competitor"))
        else:
            readings.append({"reading": name, "role": "per-competitor", "verdict": "EXCLUDED", "missing": comp[c]["missing"]})
    out.update(verdict=readings[0]["verdict"], readings=readings)
    return out, EXIT_OK


# ----------------------------------------------------------------------------------------------------------- output
def _f(x, nd=5):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def render_md(k, p):
    lines = ["# HP1 readout (bench/hp1/hp1_reduce.py)", ""]
    if k.get("verdict") == "OK":
        lines += ["## K readout (selection seeds 3407-3409)", "",
                  "| cell | init | dtype | mean y |", "|---|---|---|---|"]
        lines += [f"| {c} | {CELLS[c]['init']} | {CELLS[c]['dtype']} | {k['cell_means'][c]:.5f} |" for c in CHEAPNESS]
        lines += [""]
        for key in ("init", "dtype"):
            e = k[key]
            lines.append(f"- {e['knob']}: {_f(e['effect'])} +- {e['se']:.5f}, one-sided p = {e['p_one_sided']:.4g} -> "
                         f"**{e['reading']}**")
        it = k["interaction"]
        lines += [f"- interaction ({it['contrast']}): {_f(it['estimate'])} +- {it['se']:.5f}, p = {it['p_two_sided']:.4g}",
                  f"- sigma_K = {k['sigma_K']:.5f} (df {k['anova']['df_error']}); regime **{k['regime']}**; best "
                  f"{k['best_cell']}; selected **E = {k['selected']}**", ""]
    else:
        lines += [f"## K readout: {k.get('verdict')}", ""]
        lines += [f"- missing: {x['arm']} @ {x['seed']} ({x['why']})" for x in k.get("missing", [])]
        lines += [f"- data order: {x['tag']} @ {x['seed']} ({x['why']})" for x in k.get("order_problems", [])] + [""]
    lines += ["## P readout (confirmation seeds 3410-3412)", ""]
    if p is None:
        return "\n".join(lines) + "\n"
    lines += [f"- sigma_seed = {_f(p['sigma_seed'])} (pooled, df {p['sigma_seed_df']}); m = {_f(p['m'])}; "
              f"C* = {p.get('c_star')}; excluded competitors: {p['excluded_competitors'] or 'none'}"]
    if p["verdict"] == "P_INCOMPLETE":
        lines += ["", "**P_INCOMPLETE**: " + "; ".join(p["problems"])]
        return "\n".join(lines) + "\n"
    lines += ["", "| reading | delta 3410 / 3411 / 3412 | mean | SE | CI90 | m | sigma_seed | verdict | OWN_BASE_ONLY |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in p["readings"]:
        if "deltas" not in r:
            note = "same as headline" if r.get("same_as_headline") else "missing: " + ", ".join(
                f"{x['arm']}@{x['seed']}" for x in r.get("missing", []))
            lines.append(f"| {r['reading']} | {note} | | | | | | {r['verdict']} | |")
            continue
        ds = " / ".join(_f(r["deltas"][str(s)]) for s in CONFIRMATION_SEEDS)
        lines.append(f"| {r['reading']} | {ds} | {_f(r['mean'])} | {r['se']:.5f} | [{_f(r['ci90'][0])}, {_f(r['ci90'][1])}] | "
                     f"{r['m']:.5f} | {r['sigma_seed']:.5f} | **{r['verdict']}** | "
                     f"{', '.join(r['own_base_only']) if r['any_own_base_only'] else 'no'} |")
    return "\n".join(lines) + "\n"


def _write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, allow_nan=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    vp = sub.add_parser("voids", help="the tags of the VOID arms in evaluator JSON(s), one per line (the driver redraws each once)")
    vp.add_argument("evals", nargs="+")
    for name in ("select", "verdict"):
        sp = sub.add_parser(name)
        sp.add_argument("evals", nargs="+", help="hp1_eval.py output JSON(s); arms are merged, duplicate tags refused")
        sp.add_argument("--out", required=True, help="readout JSON")
        if name == "verdict":
            sp.add_argument("--md", required=True, help="human-readable markdown table")
    a = ap.parse_args(argv)
    if a.cmd == "voids":                            # registered redraw rule: a RUN-level VOID is redrawn once on the same box;
        for path in a.evals:                        # a base-level VOID is a systematic miss and is never redrawn
            for arm in json.load(open(path)).get("arms", []):
                if arm.get("status") == "VOID" and arm.get("void_level") != "base":
                    print(arm["tag"])
        return EXIT_OK
    try:
        index, records, ignored = load_arms(a.evals)
        k = k_readout(index)
        base = {"reducer": "bench/hp1/hp1_reduce.py", "subcommand": a.cmd, "inputs": a.evals, "ignored": ignored,
                "runs": [{x: r[x] for x in ("tag", "role", "group", "seed", "status", "y", "y_source", "unusable", "file")}
                         for r in records]}
        if a.cmd == "select":
            _write_json(a.out, dict(base, verdict=k["verdict"], k=k))
            if k["verdict"] != "OK":
                print(f"K_INCOMPLETE: {len(k['missing'])} K run(s) without a y, {len(k['order_problems'])} data-order "
                      f"problem(s) {k['order_problems']}; nothing selected", file=sys.stderr)
                return EXIT_INCOMPLETE
            print(f"HP1_SELECTED={k['selected']}")
            return EXIT_OK
        if k["verdict"] != "OK":
            _write_json(a.out, dict(base, verdict="K_INCOMPLETE", k=k, p=None))
            with open(a.md, "w") as fh:
                fh.write(render_md(k, None))
            print("K_INCOMPLETE: E cannot be derived", file=sys.stderr)
            return EXIT_INCOMPLETE
        p, code = p_readout(index, k)
        _write_json(a.out, dict(base, verdict=p["verdict"], k=k, p=p))
        with open(a.md, "w") as fh:
            fh.write(render_md(k, p))
        print(f"HP1_VERDICT={p['verdict']}")
        return code
    except RefusedInput as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
