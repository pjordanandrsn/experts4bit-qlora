#!/usr/bin/env python3
"""p70_reduce.py -- read P70 (bench/p70/P70-PREREG.md) from a run directory.

    python bench/p70/p70_reduce.py <run-dir> [--md RESULTS-p70-generated.md] [--json p70_rep.json]

<run-dir> holds `out/<pass>.<text>.pt` + `.census.json` (kl_router.py's passes), `out/build.json`, `prompts_<text>.json`
(the rows: the NLL targets), `pack.json` and, from the runner, `REHEARSAL` when a knob was off its registered default.
Every KL is kl_fidelity's (fp64, full vocabulary, token-weighted; `p59_reduce.kl`); classification, the NLL, the
paired bootstrap and the verdict rule are P64's (`p64_reduce`, imported unchanged).

The registered read, per text t:
    G(t)     = KL(ep32 || ep16)                          what the cast changes at decode -- PRIMARY
    F(t)     = mean(KL(ep32 || ep32_pc96), KL(ep32 || ep32_pc384))   the arithmetic-order floor; both samples keep the
                                                          primary's router function in the prefill (> 64-row forwards)
    dNLL(t)  = NLL(ep16) - NLL(ep32)                      signed: > 0 means the cast makes the model worse
    F_NLL(t) = mean(|NLL(ep32_pc96) - NLL(ep32)|, |NLL(ep32_pc384) - NLL(ep32)|)
  classes and MATERIAL exactly as P64's rule (factor 2 over F; material = distinguishable, dNLL > F_NLL and the
  bootstrap interval excludes 0).
Informational: P(t) = KL(ep32 || ep32_pc64), P64's floor sample, whose prefill ran the fused router: P(t) / F(t).
Validity (else NOTHING IS READ): ep32 || ep32_rep exactly 0 on every decode token of both texts; the prefill-last
logits of ep16 and ep32_rep bit-identical to ep32's (the cast touches no > 64-row forward); every pass's counts exactly
the registered ones (P64's decode counts and every router class x dtype x phase); the same prompt rows in every pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "p64"), os.path.join(HERE, "..", "p59")):
    if cand not in sys.path:
        sys.path.insert(0, cand)
from p59_reduce import kl  # noqa: E402
from p64_reduce import (NLL_FLOOR_K8, RECORD_FLOOR, RECORD_FLOOR_SRC, _load, classify, is_material,  # noqa: E402
                        lane_verdict, paired_bootstrap, row_nll)

TEXTS = ("wikitext", "c4val1")
PASSES = ("ep32", "ep16", "ep32_rep", "ep32_pc96", "ep32_pc384", "ep32_pc64")
CHUNK = {"ep32": 128, "ep16": 128, "ep32_rep": 128, "ep32_pc96": 96, "ep32_pc384": 384, "ep32_pc64": 64}
CAST = {"ep16"}
FLOOR_PASSES = ("ep32_pc96", "ep32_pc384")
PREFIX, TOKENS, MAX_DECODE_ROWS = 384, 512, 64
REQUIRED = ("ep32", "ep16", "ep32_rep", "ep32_pc96")      # the registered read cannot be made without these, both texts


def expected_counts(pass_name: str, build: dict, steps: int) -> dict:
    """kl_router.expected_counts, restated so the reducer checks the census without importing the scorer
    (tests/test_p70_reduce.py holds the two equal on every pass)."""
    L, n_attn = build["moe_layers"], build.get("int4_attn_projections", 0)
    H = (build.get("counters") or {}).get("router_hooks", 0)
    chunk = CHUNK[pass_name]
    rows = steps // (TOKENS - PREFIX)
    n_prefill = rows * -(-PREFIX // chunk)
    small = chunk <= MAX_DECODE_ROWS
    c = {"expert_quant_decode": 2 * L * steps, "expert_dequant_decode": 0, "attn_quant_decode": n_attn * steps}
    for size in ("le64", "gt64"):
        for dt in ("fp32", "bf16"):
            for ph in ("decode", "prefill"):
                c[f"router_{size}_{dt}_{ph}"] = 0
    fused_dt = "bf16" if pass_name in CAST else "fp32"
    c[f"router_le64_{fused_dt}_decode"] = H * steps
    c[f"router_le64_{fused_dt}_prefill" if small else "router_gt64_bf16_prefill"] = H * n_prefill
    return c


def reduce(D: str) -> tuple[dict, list[str]]:
    out_d = os.path.join(D, "out")
    build = json.load(open(os.path.join(out_d, "build.json"))) if os.path.exists(os.path.join(out_d, "build.json")) else {}
    pack = json.load(open(os.path.join(D, "pack.json"))) if os.path.exists(os.path.join(D, "pack.json")) else {}
    rehearsal = os.path.exists(os.path.join(D, "REHEARSAL"))
    rep = {"rehearsal": rehearsal, "pack": pack,
           "build": {k: build.get(k) for k in ("moe_layers", "top_k", "int4_expert_layers", "int4_attn_projections",
                                               "fuse_router_epilogue_n", "attn_pack_sha256")},
           "router_hooks": (build.get("counters") or {}).get("router_hooks"), "texts": {}, "validity": {}, "verdict": None}
    L = ["# Results -- P70: what rounding the fused router's decode weights to bf16 costs (e4b#726)"
         + (" -- REHEARSAL, NOT A READING" if rehearsal else ""), "",
         "Pre-registration: [`P70-PREREG.md`](P70-PREREG.md). Every number below is read from the run's receipts by `p70_reduce.py`.", ""]
    if rehearsal:
        L += ["**REHEARSAL -- NOT a reading.** A runner knob was off its registered default (the run's `summary.txt` KNOBS line).", ""]
    L += [f"Pack: `{pack.get('fingerprint')}` -- {'the licensed pack (P55x)' if pack.get('licensed') else 'NOT the licensed pack: the read is labelled so'}; "
          f"routers patched {build.get('fuse_router_epilogue_n')}, hooked {rep['router_hooks']}; MoE layers {build.get('moe_layers')}.", ""]
    ok_det, ok_pre, ok_cnt, ok_prompts, ok_hooks = True, True, True, True, True
    problems = []
    if build and rep["router_hooks"] != build.get("moe_layers"):
        ok_hooks = False
        problems.append(f"router hooks {rep['router_hooks']} != MoE layers {build.get('moe_layers')}: not every router was patched and counted")
    for text in TEXTS:
        T = rep["texts"][text] = {"passes": {}, "pairs": {}, "nll": {}}
        pf = os.path.join(D, f"prompts_{text}.json")
        ids = torch.tensor(json.load(open(pf))["prompts"], dtype=torch.long) if os.path.exists(pf) else None
        data = {p: _load(out_d, p, text) for p in PASSES}
        shas = set()
        for p, (lg, c) in data.items():
            if c is None or lg is None:
                T["passes"][p] = None if c is None else {"skipped": c.get("skipped")}
                continue
            steps = c["decode_steps"]
            want = expected_counts(p, build, steps) if build else c.get("expected_decode_counts", {})
            got = c.get("counts", {})
            engaged = all(got.get(k, 0) == v for k, v in want.items())
            T["passes"][p] = {"engaged": engaged, "counts": {k: got.get(k, 0) for k in want if got.get(k, 0) or want[k]},
                              "expected_nonzero": {k: v for k, v in want.items() if v}, "wall_s": c.get("wall_s"),
                              "rows": c.get("rows"), "prefill_chunk": c.get("prefill_chunk"),
                              "cast_during_pass": (c.get("toggle_state_during_pass") or {}).get("CAST_WEIGHTS")}
            shas.add(c.get("prompts_sha256"))
            if not engaged:
                ok_cnt = False
                problems.append(f"{p}/{text}: counts {T['passes'][p]['counts']} != registered {T['passes'][p]['expected_nonzero']}")
        if len(shas) > 1:
            ok_prompts = False
            problems.append(f"{text}: passes scored different prompt rows")

        def pair(name, r, t):
            if data[r][0] is None or data[t][0] is None:
                T["pairs"][name] = None
                return None
            s = kl(data[r][0], data[t][0], "decode")
            T["pairs"][name] = s
            return s
        G = pair("G", "ep32", "ep16")
        F1 = pair("F_pc96", "ep32", "ep32_pc96")
        F2 = pair("F_pc384", "ep32", "ep32_pc384")
        P = pair("P64_floor_pc64", "ep32", "ep32_pc64")
        Dt = pair("determinism", "ep32", "ep32_rep")
        if Dt is None or not Dt["exactly_zero"]:
            ok_det = False
            problems.append(f"{text}: ep32 || ep32_rep is {'missing' if Dt is None else 'not exactly 0 (max %.3e)' % Dt['kl_max_per_token']}")
        base = data["ep32"][0]
        for p in ("ep16", "ep32_rep"):
            if base is None or data[p][0] is None:
                ok_pre = False
                problems.append(f"{text}: prefill-last control ep32 vs {p} unreadable (missing)")
                continue
            same = torch.equal(base["prefill_last"], data[p][0]["prefill_last"])
            T.setdefault("prefill_last_identical", {})[p] = same
            if not same:
                ok_pre = False
                problems.append(f"{text}: prefill-last logits of {p} differ from ep32's -- the pass touched a > 64-row forward")
        floors = [s["kl_mean"] for s in (F1, F2) if s is not None]
        F = sum(floors) / len(floors) if floors else None
        src = "in-lane (mean of the measured floor pairs)"
        if F is not None and F == 0.0:
            F, src = RECORD_FLOOR, f"ON RECORD -- the in-lane floor read exactly 0 (unmeasured): {RECORD_FLOOR_SRC}"
        T["floor"] = {"F": F, "source": src, "samples": floors}
        if ids is not None:
            for p, (lg, c) in data.items():
                if lg is not None:
                    T["nll"][p] = row_nll(lg["decode"], ids, lg.get("prefix", PREFIX))
        nl = T["nll"]
        fnl = [abs(float(nl[f].mean() - nl["ep32"].mean())) for f in FLOOR_PASSES if f in nl and "ep32" in nl]
        T["floor"]["F_NLL"] = sum(fnl) / len(fnl) if fnl else None
        if G is None or F is None:
            T["stats"] = None
        else:
            cls = classify(G["kl_mean"], F)
            if "ep32" in nl and "ep16" in nl:
                d_rows = nl["ep16"] - nl["ep32"]
                dn, (lo, hi) = float(d_rows.mean()), paired_bootstrap(d_rows)
            else:
                dn, lo, hi = float("nan"), float("nan"), float("nan")
            f_nll = T["floor"]["F_NLL"] if T["floor"]["F_NLL"] is not None else NLL_FLOOR_K8
            T["stats"] = {"G": G["kl_mean"], "top1": G["top1_agreement"], "ratio_to_floor": G["kl_mean"] / F if F else None,
                          "class": cls, "dNLL": dn, "dNLL_ci95": [lo, hi], "F_NLL": f_nll,
                          "material": (not math.isnan(dn)) and is_material(cls, dn, lo, f_nll)}
        T["p64_floor_informational"] = ({"KL": P["kl_mean"], "ratio_to_F": P["kl_mean"] / F if F else None}
                                        if P is not None and F is not None else None)
        for k, v in list(nl.items()):
            nl[k] = float(v.mean())
    valid = ok_det and ok_pre and ok_cnt and ok_prompts and ok_hooks
    rep["validity"] = {"status": "VALID" if valid else "NOTHING READ", "determinism": ok_det, "prefill_last": ok_pre,
                       "engagement": ok_cnt, "same_prompts": ok_prompts, "router_hooks": ok_hooks, "problems": problems}
    per = {t: rep["texts"][t]["stats"] for t in TEXTS}
    if not valid:
        rep["verdict"] = "NOTHING READ (validity)"
    elif any(v is None for v in per.values()):
        rep["verdict"] = "UNREAD (a text is missing)"
    else:
        rep["verdict"] = lane_verdict(per)
    # ---- tables
    L += ["## Validity (read first)", "",
          f"**VALIDITY: {rep['validity']['status']}** -- determinism {ok_det}, prefill-last untouched {ok_pre}, engagement {ok_cnt}, "
          f"same rows {ok_prompts}, every router hooked {ok_hooks}", ""]
    L += [f"- {pr}" for pr in problems]
    L += ["", "## Passes (census against the registered counts; router columns are calls by row class and returned dtype)", "",
          "| text | pass | engaged | chunk | cast | le64 fp32 decode | le64 bf16 decode | le64 fp32 prefill | gt64 bf16 prefill | wall s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for text in TEXTS:
        for p in PASSES:
            v = rep["texts"][text]["passes"].get(p)
            if not v or "counts" not in v:
                L.append(f"| {text} | {p} | {'SKIPPED' if v else 'MISSING'} | | | | | | | |")
                continue
            c = v["counts"]
            L.append(f"| {text} | {p} | {v['engaged']} | {v['prefill_chunk']} | {v['cast_during_pass']} | {c.get('router_le64_fp32_decode', 0)} | "
                     f"{c.get('router_le64_bf16_decode', 0)} | {c.get('router_le64_fp32_prefill', 0)} | {c.get('router_gt64_bf16_prefill', 0)} | {v['wall_s']} |")
    L += ["", "## KL pairs (nats/token, fp64, full vocabulary)", "",
          "| text | pair (ref || test) | positions | kl_mean | median | p95 | max | top-1 | exactly 0 |", "|---|---|---|---|---|---|---|---|---|"]
    names = {"G": "ep32 || ep16 (G, the cast)", "F_pc96": "ep32 || ep32_pc96 (floor)", "F_pc384": "ep32 || ep32_pc384 (floor)",
             "determinism": "ep32 || ep32_rep (determinism)", "P64_floor_pc64": "ep32 || ep32_pc64 (P64's floor sample; informational)"}
    for text in TEXTS:
        for k, nm in names.items():
            s = rep["texts"][text]["pairs"].get(k)
            if s is None:
                L.append(f"| {text} | {nm} | -- | MISSING | | | | | |")
            else:
                L.append(f"| {text} | {nm} | {s['n_tokens_scored']} | {s['kl_mean']:.6f} | {s['kl_median']:.6f} | {s['kl_p95']:.6f} | "
                         f"{s['kl_max_per_token']:.4f} | {s['top1_agreement']:.4f} | {s['exactly_zero']} |")
    L += ["", "## NLL (nats/token over the scored positions)", "", "| text | " + " | ".join(PASSES) + " |", "|---" * (len(PASSES) + 1) + "|"]
    for text in TEXTS:
        nl = rep["texts"][text]["nll"]
        L.append(f"| {text} | " + " | ".join(f"{nl[p]:.5f}" if p in nl else "--" for p in PASSES) + " |")
    L += ["", "## The read (P64's rule: factor 2 over the measured floor)", "",
          "| text | G | floor F (source) | G / F | class | dNLL = NLL(ep16) - NLL(ep32) (95% CI) | F_NLL | material | P64 floor sample / F |",
          "|---|---|---|---|---|---|---|---|---|"]
    for text in TEXTS:
        T = rep["texts"][text]
        s, pf = T["stats"], T["p64_floor_informational"]
        if s is None:
            L.append(f"| {text} | UNREAD | | | | | | | |")
            continue
        L.append(f"| {text} | {s['G']:.6f} | {T['floor']['F']:.6f} ({'in-lane' if T['floor']['source'].startswith('in-lane') else 'on record'}) | "
                 f"{s['ratio_to_floor']:.2f} | {s['class']} | {s['dNLL']:+.5f} ({s['dNLL_ci95'][0]:+.5f}, {s['dNLL_ci95'][1]:+.5f}) | "
                 f"{s['F_NLL']:.5f} | {s['material']} | {('%.2f' % pf['ratio_to_F']) if pf else '--'} |")
    L += ["", f"**VERDICT (the cast, G): {rep['verdict']}**" + (" -- REHEARSAL, NOT A READING" if rehearsal else ""),
          "- floor sources: " + "; ".join(f"{t}: {rep['texts'][t]['floor']['source']}" for t in TEXTS),
          f"- the family's registered K8 |dNLL| floor, quoted beside: {NLL_FLOOR_K8} nats (METHODOLOGY section 13.1)"]
    if not pack.get("licensed", False):
        L.append("- the pack is NOT the licensed one: whatever the verdict, it is a reading on this box's pack, not on the licensed stack")
    return rep, L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P70 reducer (e4b#726)")
    ap.add_argument("run_dir")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if not os.path.isdir(os.path.join(a.run_dir, "out")):
        print(f"no out/ under {a.run_dir}: nothing to reduce", file=sys.stderr)
        return 43
    rep, lines = reduce(a.run_dir)
    text = "\n".join(lines) + "\n"
    if a.md:
        open(a.md, "w").write(text)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, default=str)
    print(text)
    return 0 if rep["validity"]["status"] == "VALID" else 14


if __name__ == "__main__":
    sys.exit(main())
