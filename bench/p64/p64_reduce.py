#!/usr/bin/env python3
"""p64_reduce.py -- read P64 (bench/p64/P64-PREREG.md) from a run directory.

    python bench/p64/p64_reduce.py <run-dir> [--md RESULTS-p64-generated.md] [--json p64_rep.json]

<run-dir> holds `out/<pass>.<text>.pt` + `.census.json` (the int4 stack's passes, kl_a16.py), `out/build.json`,
`out_nf4/nf4.<text>.*` (the anchor), `prompts_<text>.json` (the rows: the NLL targets) and, from the runner,
`pack.json` and `REHEARSAL` when a knob was off its registered default. Every KL is kl_fidelity's (fp64, full
vocabulary, token-weighted over rows x decode positions; `p59_reduce.kl`); NLL is the ground-truth next token's
negative log-probability at the decode positions that have one (127 of 128 per row).

The registered read (P64-PREREG "Decision rule"), per text t:
    G_exp(t)  = KL(a16 || a8)          the expert int8 step (the lever)             -- PRIMARY
    G_all(t)  = KL(a16_all || a8)      the whole lane's int8 step (experts + attention)
    G_attn(t) = KL(a16_all || a16)     the attention int8 step, experts already bf16
    F(t)      = mean(KL(a8 || a8_pc64), KL(a8 || a8_pc384))   the arithmetic-order floor, this instrument, this box
    dNLL(t)   = NLL(a8) - NLL(a16)     signed: > 0 means the int8 step makes the model worse; paired row bootstrap
    F_NLL(t)  = mean(|NLL(a8_pc64) - NLL(a8)|, |NLL(a8_pc384) - NLL(a8)|)
  G <= F: BELOW FLOOR; F < G <= 2F: NOT DISTINGUISHABLE; G > 2F: DISTINGUISHABLE.
  MATERIAL on t: DISTINGUISHABLE and dNLL(t) > F_NLL(t) and the bootstrap 95% interval of dNLL(t) excludes 0.
  Lane verdict for a statistic: MATERIAL if material on either text; else DISTINGUISHABLE, NOT MATERIAL if
  distinguishable on either; else INDISTINGUISHABLE.
Validity (else NOTHING IS READ): a8 || a8_rep exactly 0 on every decode token of both texts; the prefill-last logits
of a8, a16, a16_all and a8_rep bit-identical (the flag touches no T > 1 forward); every pass's decode-phase counts
exactly the registered ones; the same prompt rows in every pass. A floor that reads exactly 0 on a text is
unmeasured there, and that text's read uses the on-record reorder reading for this family and scorer (P59b,
0.0044 nats/token), stated on the row.
stdlib + torch; nothing is licensed by this file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "p59")):
    if cand not in sys.path:
        sys.path.insert(0, cand)
from p59_reduce import kl  # noqa: E402  (kl_fidelity's accumulator over rows, P59's reducer)

TEXTS = ("wikitext", "c4val1")
PASSES = ("a8", "a16", "a16_all", "a8_rep", "a8_pc64", "a8_pc384")
FLOOR_PASSES = ("a8_pc64", "a8_pc384")
FACTOR = 2.0
RECORD_FLOOR = 0.0044          # P59b: KL(int4 || int4_fqkv), a reorder of the prefill through this scorer, Qwen3, wikitext rows
RECORD_FLOOR_SRC = "e4b.serve.p59b.qwen3.b16.fqkv-kl.5090.2026-09-22 (0.0044 nats/token, a reorder-class change through P59's scorer)"
NLL_FLOOR_K8 = 0.0095          # Qwen3-30B-A3B's registered |dNLL| arithmetic-order floor (METHODOLOGY section 13.1), quoted beside
BOOT_N, BOOT_SEED = 10000, 1689


def classify(g: float, f: float, factor: float = FACTOR) -> str:
    if g <= f:
        return "BELOW FLOOR"
    if g <= factor * f:
        return "NOT DISTINGUISHABLE"
    return "DISTINGUISHABLE"


def is_material(cls: str, dnll: float, ci_lo: float, f_nll: float) -> bool:
    return cls == "DISTINGUISHABLE" and dnll > f_nll and ci_lo > 0.0


def lane_verdict(per_text: dict) -> str:
    """per_text: {text: {"class": ..., "material": bool}} -> the registered lane verdict."""
    if any(v["material"] for v in per_text.values()):
        return "MATERIAL"
    if any(v["class"] == "DISTINGUISHABLE" for v in per_text.values()):
        return "DISTINGUISHABLE, NOT MATERIAL"
    return "INDISTINGUISHABLE"


def row_nll(dec: torch.Tensor, ids: torch.Tensor, prefix: int) -> torch.Tensor:
    """Per-row mean NLL (fp64) over the decode positions with a target: logit j (fed ids[prefix + j]) predicts
    ids[prefix + j + 1]; the last position predicts past the row and is not scored."""
    R, S, _ = dec.shape
    n = min(S, ids.shape[1] - prefix - 1)
    tgt = ids[:R, prefix + 1:prefix + 1 + n]
    lp = torch.log_softmax(dec[:, :n].to(torch.float64), dim=-1)
    return -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean(dim=1)


def paired_bootstrap(diff_rows: torch.Tensor, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% interval of the mean of per-row differences (equal tokens per row), rows resampled with replacement."""
    g = torch.Generator().manual_seed(seed)
    R = diff_rows.numel()
    idx = torch.randint(0, R, (n, R), generator=g)
    means = diff_rows.to(torch.float64)[idx].mean(dim=1)
    return float(means.quantile(0.025)), float(means.quantile(0.975))


def expected_counts(pass_name: str, build: dict, steps: int) -> dict:
    """kl_a16.expected_counts, restated so the reducer checks the census without importing the scorer."""
    L, k, n_attn = build["moe_layers"], build["top_k"], build.get("int4_attn_projections", 0)
    exp_a16 = pass_name in ("a16", "a16_all")
    attn_a16 = pass_name == "a16_all"
    if pass_name == "nf4":
        return {"expert_quant_decode": 0, "expert_dequant_decode": 0, "attn_quant_decode": 0}
    return {"expert_quant_decode": 0 if exp_a16 else 2 * L * steps,
            "expert_dequant_decode": 2 * L * k * steps if exp_a16 else 0,
            "attn_quant_decode": 0 if attn_a16 else n_attn * steps}


def _load(d, name, text):
    p, c = os.path.join(d, f"{name}.{text}.pt"), os.path.join(d, f"{name}.{text}.census.json")
    if not (os.path.exists(p) and os.path.exists(c)):
        return None, (json.load(open(c)) if os.path.exists(c) else None)
    return torch.load(p, map_location="cpu"), json.load(open(c))


def reduce(D: str) -> tuple[dict, list[str]]:
    out_d, nf4_d = os.path.join(D, "out"), os.path.join(D, "out_nf4")
    build = json.load(open(os.path.join(out_d, "build.json"))) if os.path.exists(os.path.join(out_d, "build.json")) else {}
    pack = json.load(open(os.path.join(D, "pack.json"))) if os.path.exists(os.path.join(D, "pack.json")) else {}
    rehearsal = os.path.exists(os.path.join(D, "REHEARSAL"))
    rep = {"rehearsal": rehearsal, "pack": pack, "build": {k: build.get(k) for k in ("moe_layers", "top_k", "int4_expert_layers",
           "int4_store_kinds", "int4_attn_projections", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n", "attn_pack_sha256")},
           "texts": {}, "validity": {}, "verdicts": {}}
    L = []
    title = "# Results -- P64: what the int4 serve lane's int8 activation step costs at decode (e4b#709)"
    L += [title + (" -- REHEARSAL, NOT A READING" if rehearsal else ""), "",
          "Pre-registration: [`P64-PREREG.md`](P64-PREREG.md). Every number below is read from the run's receipts by `p64_reduce.py`.", ""]
    if rehearsal:
        L += ["**REHEARSAL -- NOT a reading.** A runner knob was off its registered default (the run's `summary.txt` KNOBS line).", ""]
    L += [f"Pack: `{pack.get('fingerprint')}` -- {'the licensed pack (P55x)' if pack.get('licensed') else 'NOT the licensed pack: the read is labelled so'}; "
          f"attention pack `{build.get('attn_pack_sha256')}`; int4 expert layers {build.get('int4_expert_layers')}, Int4Linear {build.get('int4_attn_projections')}.", ""]
    ok_det, ok_pre, ok_cnt, ok_prompts = True, True, True, True
    problems = []
    for text in TEXTS:
        T = rep["texts"][text] = {"passes": {}, "pairs": {}, "nll": {}}
        pf = os.path.join(D, f"prompts_{text}.json")
        ids = torch.tensor(json.load(open(pf))["prompts"], dtype=torch.long) if os.path.exists(pf) else None
        data = {}
        for p in PASSES:
            lg, c = _load(out_d, p, text)
            data[p] = (lg, c)
        data["nf4"] = _load(nf4_d, "nf4", text)
        shas = set()
        for p, (lg, c) in data.items():
            if c is None or lg is None:
                T["passes"][p] = None if c is None else {"skipped": c.get("skipped")}
                continue
            steps = c["decode_steps"]
            want = expected_counts(p, build, steps) if build else c.get("expected_decode_counts", {})
            got = c.get("counts", {})
            engaged = all(got.get(k, 0) == v for k, v in want.items())
            T["passes"][p] = {"engaged": engaged, "counts": {k: got.get(k, 0) for k in want}, "expected": want, "wall_s": c.get("wall_s"),
                              "s_per_step": c.get("s_per_decode_step_incl_prefill"), "rows": c.get("rows"), "prefill_chunk": c.get("prefill_chunk")}
            shas.add(c.get("prompts_sha256"))
            if not engaged:
                ok_cnt = False
                problems.append(f"{p}/{text}: decode counts {T['passes'][p]['counts']} != registered {want}")
        if len(shas) > 1:
            ok_prompts = False
            problems.append(f"{text}: passes scored different prompt rows")

        def pair(name, r, t, key="decode"):
            if data[r][0] is None or data[t][0] is None:
                T["pairs"][name] = None
                return None
            s = kl(data[r][0], data[t][0], key)
            T["pairs"][name] = s
            return s
        G = pair("G_exp", "a16", "a8")
        Ga = pair("G_all", "a16_all", "a8")
        Gt = pair("G_attn", "a16_all", "a16")
        F1 = pair("F_pc64", "a8", "a8_pc64")
        F2 = pair("F_pc384", "a8", "a8_pc384")
        Dt = pair("determinism", "a8", "a8_rep")
        pair("anchor_a8", "nf4", "a8")
        pair("anchor_a16", "nf4", "a16")
        pair("anchor_a16_all", "nf4", "a16_all")
        if Dt is None or not Dt["exactly_zero"]:
            ok_det = False
            problems.append(f"{text}: a8 || a8_rep is {'missing' if Dt is None else 'not exactly 0 (max %.3e)' % Dt['kl_max_per_token']}")
        # prefill-last: the flag and the attention toggle must not touch a T > 1 forward
        base = data["a8"][0]
        for p in ("a16", "a16_all", "a8_rep"):
            if base is None or data[p][0] is None:
                if p in ("a16", "a8_rep"):
                    ok_pre = False
                    problems.append(f"{text}: prefill-last control a8 vs {p} unreadable (missing)")
                continue
            same = torch.equal(base["prefill_last"], data[p][0]["prefill_last"])
            T.setdefault("prefill_last_identical", {})[p] = same
            if not same:
                ok_pre = False
                problems.append(f"{text}: prefill-last logits of {p} differ from a8's -- the pass touched a T > 1 forward")
        floors = [s["kl_mean"] for s in (F1, F2) if s is not None]
        F = sum(floors) / len(floors) if floors else None
        floor_src = "in-lane (mean of the measured floor pairs)"
        if F is not None and F == 0.0:
            F, floor_src = RECORD_FLOOR, f"ON RECORD -- the in-lane floor read exactly 0 (unmeasured): {RECORD_FLOOR_SRC}"
        T["floor"] = {"F": F, "source": floor_src, "samples": floors}
        # NLL
        if ids is not None:
            prefix = next((d[0]["prefix"] for d in data.values() if d[0] is not None), 384)
            for p, (lg, c) in data.items():
                if lg is not None:
                    T["nll"][p] = row_nll(lg["decode"], ids, prefix)
        nl = T["nll"]
        fnl = [abs(float(nl[f].mean() - nl["a8"].mean())) for f in FLOOR_PASSES if f in nl and "a8" in nl]
        T["floor"]["F_NLL"] = sum(fnl) / len(fnl) if fnl else None
        T["stats"] = {}
        for gname, s, ref_pass in (("G_exp", G, "a16"), ("G_all", Ga, "a16_all"), ("G_attn", Gt, None)):
            if s is None or F is None:
                T["stats"][gname] = None
                continue
            cls = classify(s["kl_mean"], F)
            if ref_pass and "a8" in nl and ref_pass in nl:
                d_rows = nl["a8"] - nl[ref_pass]
                dn, (lo, hi) = float(d_rows.mean()), paired_bootstrap(d_rows)
            elif gname == "G_attn" and "a16" in nl and "a16_all" in nl:
                d_rows = nl["a16"] - nl["a16_all"]
                dn, (lo, hi) = float(d_rows.mean()), paired_bootstrap(d_rows)
            else:
                dn, lo, hi = float("nan"), float("nan"), float("nan")
            f_nll = T["floor"]["F_NLL"] if T["floor"]["F_NLL"] is not None else NLL_FLOOR_K8
            T["stats"][gname] = {"G": s["kl_mean"], "top1": s["top1_agreement"], "ratio_to_floor": s["kl_mean"] / F if F else None,
                                 "class": cls, "dNLL": dn, "dNLL_ci95": [lo, hi], "F_NLL": f_nll,
                                 "material": (not math.isnan(dn)) and is_material(cls, dn, lo, f_nll)}
        for k, v in list(nl.items()):
            nl[k] = float(v.mean())
    valid = ok_det and ok_pre and ok_cnt and ok_prompts
    rep["validity"] = {"status": "VALID" if valid else "NOTHING READ", "determinism": ok_det, "prefill_last": ok_pre,
                       "engagement": ok_cnt, "same_prompts": ok_prompts, "problems": problems}
    # ---- tables
    L += ["## Validity (read first)", ""]
    L += [f"**VALIDITY: {rep['validity']['status']}** -- determinism {ok_det}, prefill-last untouched {ok_pre}, engagement {ok_cnt}, same rows {ok_prompts}", ""]
    for pr in problems:
        L.append(f"- {pr}")
    L += ["", "## Passes (census: decode-phase counts against the registered ones)", "",
          "| text | pass | engaged | expert quant | expert dequant | attn quant | s/step | prefill chunk |", "|---|---|---|---|---|---|---|---|"]
    for text in TEXTS:
        for p in PASSES + ("nf4",):
            v = rep["texts"][text]["passes"].get(p)
            if not v or "counts" not in v:
                L.append(f"| {text} | {p} | {'SKIPPED' if v else 'MISSING'} | | | | | |")
                continue
            c = v["counts"]
            L.append(f"| {text} | {p} | {v['engaged']} | {c.get('expert_quant_decode')} | {c.get('expert_dequant_decode')} | {c.get('attn_quant_decode')} | {v['s_per_step']} | {v['prefill_chunk']} |")
    L += ["", "## KL pairs (nats/token, fp64, full vocabulary)", "",
          "| text | pair (ref || test) | positions | kl_mean | median | p95 | max | top-1 | exactly 0 |", "|---|---|---|---|---|---|---|---|---|"]
    names = {"G_exp": "a16 || a8 (G_exp, the lever)", "G_all": "a16_all || a8 (G_all)", "G_attn": "a16_all || a16 (G_attn)",
             "F_pc64": "a8 || a8_pc64 (floor)", "F_pc384": "a8 || a8_pc384 (floor)", "determinism": "a8 || a8_rep (determinism)",
             "anchor_a8": "nf4 || a8 (anchor)", "anchor_a16": "nf4 || a16 (anchor)", "anchor_a16_all": "nf4 || a16_all (anchor)"}
    for text in TEXTS:
        for k, nm in names.items():
            s = rep["texts"][text]["pairs"].get(k)
            if s is None:
                L.append(f"| {text} | {nm} | -- | MISSING | | | | | |")
            else:
                L.append(f"| {text} | {nm} | {s['n_tokens_scored']} | {s['kl_mean']:.6f} | {s['kl_median']:.6f} | {s['kl_p95']:.6f} | {s['kl_max_per_token']:.4f} | {s['top1_agreement']:.4f} | {s['exactly_zero']} |")
    L += ["", "## NLL (nats/token over the scored positions; the ground-truth next token)", "", "| text | " + " | ".join(PASSES + ("nf4",)) + " |", "|---" * (len(PASSES) + 2) + "|"]
    for text in TEXTS:
        nl = rep["texts"][text]["nll"]
        L.append(f"| {text} | " + " | ".join(f"{nl[p]:.5f}" if p in nl else "--" for p in PASSES + ("nf4",)) + " |")
    L += ["", "## The read (registered rule; factor 2 over the measured floor)", "",
          "| text | statistic | G | floor F (source) | G / F | class | dNLL (95% CI) | F_NLL | material |", "|---|---|---|---|---|---|---|---|---|"]
    for text in TEXTS:
        T = rep["texts"][text]
        for g in ("G_exp", "G_all", "G_attn"):
            s = T["stats"].get(g)
            if s is None:
                L.append(f"| {text} | {g} | UNREAD | | | | | | |")
                continue
            L.append(f"| {text} | {g} | {s['G']:.6f} | {T['floor']['F']:.6f} ({'in-lane' if T['floor']['source'].startswith('in-lane') else 'on record'}) | "
                     f"{s['ratio_to_floor']:.2f} | {s['class']} | {s['dNLL']:+.5f} ({s['dNLL_ci95'][0]:+.5f}, {s['dNLL_ci95'][1]:+.5f}) | {s['F_NLL']:.5f} | {s['material']} |")
    L += [""]
    for g in ("G_exp", "G_all", "G_attn"):
        per = {t: rep["texts"][t]["stats"].get(g) for t in TEXTS}
        if not valid:
            rep["verdicts"][g] = "NOTHING READ (validity)"
        elif any(v is None for v in per.values()):
            rep["verdicts"][g] = "UNREAD (a text is missing)"
        else:
            rep["verdicts"][g] = lane_verdict(per)
    L.append(f"**VERDICT (expert int8 step, G_exp): {rep['verdicts']['G_exp']}**" + (" -- REHEARSAL, NOT A READING" if rehearsal else ""))
    L.append(f"- **G_all (experts + attention): {rep['verdicts']['G_all']}**")
    L.append(f"- **G_attn (attention, experts bf16): {rep['verdicts']['G_attn']}**")
    L.append("- floor sources: " + "; ".join(f"{t}: {rep['texts'][t]['floor']['source']}" for t in TEXTS))
    L.append(f"- the family's registered K8 |dNLL| floor, quoted beside: {NLL_FLOOR_K8} nats (METHODOLOGY section 13.1)")
    if not pack.get("licensed", False):
        L.append("- the pack is NOT the licensed one: whatever the verdict, it is a reading on this box's pack, not on the licensed stack")
    return rep, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    rep, L = reduce(a.run_dir)
    text = "\n".join(L)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
