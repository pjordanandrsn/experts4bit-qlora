#!/usr/bin/env python3
"""p68_reduce.py -- lane P68's registered read (bench/p68/P68-PREREG.md) over ``<out>/<stack>/p68_arm.json``.

Pure Python and deterministic (the bootstrap is seeded). It reads, in order:

- **G0** (validity, per stack): the unforced control repeats bit for bit; every forced arm's control equals its
  backend's unforced control (a forcing is inert at T = 1); every registered forcing engaged (split calls > 0 in the
  arm); and a ``core`` arm's T = 1 decode calls carried no attention mask (otherwise the per-row no-mask call is not the
  call decode makes). A stack failing G0 is read for nothing.
- **Q1-Q7** (the ablation, per stack): the first differing (layer, site) of every verify and prefill position under
  each forcing arm, against the registered predictions.
- **S** (the size reading): per stack, backend and mode, KL mean and top-1 agreement with 95 % confidence intervals from
  a bootstrap over (row, window) clusters, classed against the shipped bar (KL <= 0.10, top-1 >= 0.93).

    python p68_reduce.py <out-dir> [--md RESULTS-p68-generated.md] [--json p68_rep.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys

STACKS = ("nf4", "int4")
ATTN_SITES_L0 = {"attn_in", "attn_core", "attn_out"}
L0_ATTN_CORE_OUT = {"attn_core", "attn_out"}   # Q2/Q3: a multi-row cuBLAS row can match its M = 1 call bit for bit
BAR = {"kl_mean_max": 0.10, "top1_min": 0.93}
BOOT_B, BOOT_SEED = 2000, 68

# The registered predictions (P68-PREREG.md "Predictions"). Each maps (arm, mode) to the expected outcome:
#   "exact"       -- every position bit-equal (sites, final norm, logits)
#   {"sites": S}  -- not exact, and every position's first difference is at layer 0 in a site of S
#   {"not": S}    -- every position's first difference is somewhere OUTSIDE layer 0's sites S (or none)
#   {"majority": S} -- no position exact, and more than half of the positions' first differences are at layer 0 in S
#   None          -- read and reported, no prediction
VERIFY = ("verify17", "verify16")
PRED = {
    "nf4": {
        ("hf.base", "verify17"): {"sites": {"attn_core"}}, ("hf.base", "verify16"): {"sites": {"attn_core"}},
        ("hf.base", "prefill"): {"sites": {"attn_core"}},
        ("hf.proj", "verify17"): {"majority": L0_ATTN_CORE_OUT}, ("hf.proj", "verify16"): {"majority": L0_ATTN_CORE_OUT},
        ("hf.core", "verify17"): {"majority": L0_ATTN_CORE_OUT}, ("hf.core", "verify16"): {"majority": L0_ATTN_CORE_OUT},
        ("hf.proj_core", "verify17"): {"not": ATTN_SITES_L0}, ("hf.proj_core", "verify16"): {"not": ATTN_SITES_L0},
        ("hf.proj_core", "prefill"): {"not": ATTN_SITES_L0},
        ("hf.all", "verify17"): "exact", ("hf.all", "verify16"): "exact", ("hf.all", "prefill"): "exact",
        ("paged.proj", "prefill"): {"sites": {"attn_core"}},
    },
    "int4": {
        ("hf.base", "verify17"): {"sites": {"attn_core"}}, ("hf.base", "verify16"): {"sites": {"attn_core"}},
        ("hf.base", "prefill"): {"sites": {"attn_in"}},
        ("hf.proj", "verify17"): {"majority": L0_ATTN_CORE_OUT}, ("hf.proj", "verify16"): {"majority": L0_ATTN_CORE_OUT},
        ("hf.core", "verify17"): {"majority": L0_ATTN_CORE_OUT}, ("hf.core", "verify16"): {"majority": L0_ATTN_CORE_OUT},
        ("hf.proj_core", "verify17"): {"not": ATTN_SITES_L0}, ("hf.proj_core", "verify16"): {"not": ATTN_SITES_L0},
        ("hf.proj_core", "prefill"): {"sites": {"attn_in"}},
        ("hf.all", "verify17"): "exact", ("hf.all", "verify16"): "exact",
        ("hf.all", "prefill"): {"sites": {"attn_core"}},
        ("paged.proj", "prefill"): {"sites": {"attn_in"}},
    },
}
# paged.proj verify carries NO prediction (P68-PREREG Q7): the verify path calls the fp8 kernel with T query rows as a
# batch, whose split plan and T-row K/V append can differ from decode's one-row calls; the sm_86 rehearsal read it
# not exact, and the 5090 plans by its own SM count
#: S -- the size prediction: every cell WITHIN the bar's KL half (CI upper < 0.05) and top-1 point estimate >= 0.93
SIZE_PRED = {"kl_ci_hi_max": 0.05, "top1_min": 0.93}


def load_arm(d: str) -> dict | None:
    p = os.path.join(d, "p68_arm.json")
    if os.path.exists(p):
        return json.load(open(p))
    if os.path.exists(p + ".gz"):
        with gzip.open(p + ".gz", "rt") as f:
            return json.load(f)
    return None


# ------------------------------------------------------------------------------------------------------------- G0 --
def gate(arm: dict) -> dict:
    why = []
    for k, v in (arm.get("g0") or {}).items():
        if not v:
            why.append(f"G0 failed: {k}")
    if not arm.get("g0"):
        why.append("G0 not recorded (no hf.base control)")
    for name, res in arm.get("arms", {}).items():
        if "error" in res:
            why.append(f"{name}: errored ({res['error'][:120]})")
            continue
        sa = res.get("arm", {})
        counts = res.get("counts") or {}
        for kind in sa.get("force", []):
            c = counts.get(kind, {})
            if c.get("split_calls", 0) <= 0:
                why.append(f"{name}: forcing '{kind}' never engaged (0 split calls)")
            if kind == "core" and c.get("t1_mask_present", 0) > 0:
                why.append(f"{name}: T = 1 attention calls carried a mask ({c['t1_mask_present']}): the per-row core "
                           "call is not decode's call")
    return {"ok": not why, "why": why}


# ------------------------------------------------------------------------------------------------------- ablation --
def first_diffs(mode: dict) -> list:
    """Per position: None (exact) or (layer, site); ``final``/``logits`` differences carry layer None."""
    out = []
    for r in mode["per_position"]:
        fd = r["first_diff"]
        out.append(None if fd is None else (fd[0] if isinstance(fd[0], int) else None, fd[1]))
    return out


def judge(pred, mode: dict) -> dict:
    fds = first_diffs(mode)
    P = len(fds)
    exact = sum(1 for f in fds if f is None)
    hist = {}
    for f in fds:
        key = "exact" if f is None else (f"L{f[0]}.{f[1]}" if f[0] is not None else f[1])
        hist[key] = hist.get(key, 0) + 1
    obs = {"positions": P, "exact": exact, "first_diff": dict(sorted(hist.items(), key=lambda kv: -kv[1])[:6]),
           "kl_mean": mode.get("kl_mean"), "flips": mode.get("argmax_flips")}
    if pred is None:
        return {**obs, "predicted": None, "verdict": "INFO"}
    if pred == "exact":
        ok = exact == P and P > 0
        return {**obs, "predicted": "EXACT", "verdict": "HELD" if ok else "REFUTED"}
    if "sites" in pred:
        ok = exact == 0 and all(f is not None and f[0] == 0 and f[1] in pred["sites"] for f in fds)
        return {**obs, "predicted": f"not exact; first diff at L0 in {sorted(pred['sites'])}",
                "verdict": "HELD" if ok else "REFUTED"}
    if "majority" in pred:
        hits = sum(1 for f in fds if f is not None and f[0] == 0 and f[1] in pred["majority"])
        ok = exact == 0 and P > 0 and hits * 2 > P
        return {**obs, "predicted": f"not exact; > 50 % first diff at L0 in {sorted(pred['majority'])}",
                "verdict": "HELD" if ok else "REFUTED"}
    if "not" in pred:
        ok = all(f is None or not (f[0] == 0 and f[1] in pred["not"]) for f in fds) and P > 0
        return {**obs, "predicted": f"no first diff at L0 in {sorted(pred['not'])}",
                "verdict": "HELD" if ok else "REFUTED"}
    raise ValueError(pred)


def ablation(stack: str, arm: dict) -> list:
    rows = []
    for name, res in arm.get("arms", {}).items():
        if "error" in res or res.get("arm", {}).get("kind") != "ablate":
            continue
        for mode, m in res.get("modes", {}).items():
            j = judge(PRED.get(stack, {}).get((name, mode)), m)
            rows.append({"stack": stack, "arm": name, "mode": mode, **j})
    return rows


# ---------------------------------------------------------------------------------------------------------- size --
def _clusters(rows: list, mode: str) -> list:
    """One cluster per (row, window): [(sum_kl, flips, n)]."""
    cl = {}
    for r in rows:
        m = r["modes"].get(mode)
        if not m:
            continue
        for kl, fl, w in zip(m["kl"], m["flip"], m["window"]):
            c = cl.setdefault((r["row"], w), [0.0, 0, 0])
            c[0] += kl
            c[1] += fl
            c[2] += 1
    return list(cl.values())


def boot_ci(clusters: list, stat, b: int = BOOT_B, seed: int = BOOT_SEED) -> tuple:
    rng = random.Random(seed)
    n = len(clusters)
    vals = []
    for _ in range(b):
        s = [clusters[rng.randrange(n)] for _ in range(n)]
        vals.append(stat(s))
    vals.sort()
    return vals[int(0.025 * b)], vals[int(0.975 * b) - 1]


def _kl_mean(cs):
    n = sum(c[2] for c in cs)
    return sum(c[0] for c in cs) / n if n else float("nan")


def _top1(cs):
    n = sum(c[2] for c in cs)
    return 1.0 - sum(c[1] for c in cs) / n if n else float("nan")


def size_cell(clusters: list) -> dict:
    kl = _kl_mean(clusters)
    t1 = _top1(clusters)
    kl_lo, kl_hi = boot_ci(clusters, _kl_mean)
    t1_lo, t1_hi = boot_ci(clusters, _top1)
    if kl_hi <= BAR["kl_mean_max"] and t1_lo >= BAR["top1_min"]:
        cls = "WITHIN-BAR"
    elif kl_lo > BAR["kl_mean_max"] or t1_hi < BAR["top1_min"]:
        cls = "OVER-BAR"
    else:
        cls = "UNRESOLVED"
    held = kl_hi < SIZE_PRED["kl_ci_hi_max"] and t1 >= SIZE_PRED["top1_min"]
    return {"positions": sum(c[2] for c in clusters), "clusters": len(clusters), "kl_mean": kl,
            "kl_ci": [kl_lo, kl_hi], "top1": t1, "top1_ci": [t1_lo, t1_hi], "class": cls,
            "verdict": "HELD" if held else "REFUTED"}


def size(stack: str, arm: dict) -> list:
    out = []
    for name, res in arm.get("arms", {}).items():
        if "error" in res or res.get("arm", {}).get("kind") != "size":
            continue
        modes = sorted({m for r in res["rows"] for m in r["modes"]})
        for mode in modes:
            cs = _clusters(res["rows"], mode)
            if cs:
                out.append({"stack": stack, "arm": name, "mode": mode, **size_cell(cs)})
    return out


# ------------------------------------------------------------------------------------------------------ the read --
def reduce(out_dir: str) -> dict:
    rep = {"stacks_present": [], "gates": {}, "ablation": [], "size": [], "census": {}}
    for s in STACKS:
        arm = load_arm(os.path.join(out_dir, s))
        if arm is None:
            continue
        rep["stacks_present"].append(s)
        g = gate(arm)
        rep["gates"][s] = g
        rep["census"][s] = {n: r.get("counts") for n, r in arm.get("arms", {}).items() if "counts" in r}
        if not g["ok"]:
            continue
        rep["ablation"] += ablation(s, arm)
        rep["size"] += size(s, arm)
    ab = [r for r in rep["ablation"] if r["verdict"] in ("HELD", "REFUTED")]
    rep["verdicts"] = {
        "ablation_held": sum(1 for r in ab if r["verdict"] == "HELD"),
        "ablation_refuted": [f"{r['stack']} {r['arm']} {r['mode']}" for r in ab if r["verdict"] == "REFUTED"],
        "exact_verify_held": {s: all(r["verdict"] == "HELD" for r in ab if r["stack"] == s and r["arm"] == "hf.all"
                                     and r["mode"] in VERIFY) and any(r["stack"] == s and r["arm"] == "hf.all"
                                                                      for r in ab)
                              for s in rep["stacks_present"] if rep["gates"][s]["ok"]},
        "size_classes": sorted({r["class"] for r in rep["size"]}),
        "size_refuted": [f"{r['stack']} {r['arm']} {r['mode']}" for r in rep["size"] if r["verdict"] == "REFUTED"],
    }
    return rep


def to_md(rep: dict) -> str:
    L = ["# P68 -- generated results (p68_reduce.py; read against P68-PREREG.md)", ""]
    L.append("## G0 -- validity")
    for s, g in rep["gates"].items():
        L.append(f"- **{s}**: " + ("OK" if g["ok"] else "FAILED -- " + "; ".join(g["why"])))
    L += ["", "## The ablation (first differing layer.site per position)", "",
          "| stack | arm | mode | exact / positions | first differences | KL mean | flips | predicted | verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rep["ablation"]:
        kl = f"{r['kl_mean']:.2e}" if r.get("kl_mean") is not None else ""
        L.append(f"| {r['stack']} | {r['arm']} | {r['mode']} | {r['exact']}/{r['positions']} | {r['first_diff']} | {kl} "
                 f"| {r['flips']} | {r['predicted']} | {r['verdict']} |")
    L += ["", "## The size reading (served default, unforced; 95 % CI over (row, window) clusters)", "",
          "| stack | arm | mode | positions | KL mean [CI] | top-1 [CI] | class | prediction |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rep["size"]:
        L.append(f"| {r['stack']} | {r['arm']} | {r['mode']} | {r['positions']} | {r['kl_mean']:.3e} "
                 f"[{r['kl_ci'][0]:.3e}, {r['kl_ci'][1]:.3e}] | {r['top1']:.4f} [{r['top1_ci'][0]:.4f}, "
                 f"{r['top1_ci'][1]:.4f}] | {r['class']} | {r['verdict']} |")
    v = rep["verdicts"]
    L += ["", "## Verdicts", f"- ablation predictions held: {v['ablation_held']}; refuted: {v['ablation_refuted'] or 'none'}",
          f"- a verify built from T = 1 kernels row by row is bit-exact (hf.all, verify): {v['exact_verify_held']}",
          f"- size classes: {v['size_classes']}; size prediction refuted: {v['size_refuted'] or 'none'}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rep = reduce(a.out_dir)
    md = to_md(rep)
    if a.md:
        open(a.md, "w").write(md)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, sort_keys=True)
    print(md)
    if not rep["stacks_present"]:
        return 43
    return 14 if not all(g["ok"] for g in rep["gates"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
