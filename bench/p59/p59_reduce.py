#!/usr/bin/env python3
"""p59_reduce.py -- read P59 (bench/p59/P59-PREREG.md) from the fetched run directory.

    python bench/p59/p59_reduce.py <run-dir> [--md RESULTS-p59.md] [--json out.json]

Arms: `nf4` (the register's NF4 control), `int4` (P54's stack, unfused q/k/v -- the shipped B=16 configuration),
`int4_fqkv` (the same with q/k/v fused: the lever under test), `int4_aa` (`int4` built again in a fresh process --
the determinism control). Each arm dir holds logits.pt ([16, 128, V] fp32 decode logits + the prefill-last logits)
and census.json. Every KL is kl_fidelity's: fp64, full vocabulary, token-weighted over the 16 rows.

  P1  KL(int4 || int4_fqkv), decode positions: mean <= 0.01 nats/token AND top-1 >= 0.97   (refuted: mean > 0.03 or top-1 < 0.95)
  P2  |KL(nf4 || int4_fqkv) - KL(nf4 || int4)| <= 0.005                                    (the fusion does not move the stack's distance from the NF4 anchor)
  P3  KL(int4 || int4_aa) exactly 0 on every token                                          (the int4 stack is deterministic across processes)
  P4  prefill control: KL(int4 || int4_fqkv) over the prefill-last logits mean <= 1e-4       (the shared >16-row path is unchanged by the fusion)
  census: the fused arm fused 48 modules and carries 96 Int4Linear (48 qkv + 48 o) vs 192 in int4; nf4 carries 0.
Nothing is licensed by this file; the decision rule in the pre-registration reads the verdicts. stdlib + torch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (HERE, os.path.join(HERE, "..")):
    if cand not in sys.path:
        sys.path.insert(0, cand)
import kl_fidelity  # noqa: E402

ARMS = ("nf4", "int4", "int4_fqkv", "int4_aa")


def load(d: str, arm: str):
    p = os.path.join(d, arm, "logits.pt")
    c = os.path.join(d, arm, "census.json")
    if not (os.path.exists(p) and os.path.exists(c)):
        return None, None
    return torch.load(p, map_location="cpu"), json.load(open(c))


def kl(ref, test, key="decode") -> dict:
    acc = kl_fidelity.KLAccumulator()
    a, b = ref[key], test[key]
    if a.dim() == 2:                       # prefill-last: [B, V] -> one "token" per row
        a, b = a.unsqueeze(1), b.unsqueeze(1)
    for r in range(a.shape[0]):
        acc.add(a[r], b[r])
    return acc.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    D = a.run_dir
    L, rep = [], {"arms": {}, "pairs": {}, "verdicts": {}}
    L += ["# Results -- P59: KL at B=16 between the unfused and fused-q/k/v int4 stacks (the gate on the B=16 fusion default)", "",
          "Pre-registration: [`P59-PREREG.md`](P59-PREREG.md). Every number below is read from the fetched receipts by `p59_reduce.py`.", ""]
    data = {}
    L += ["## Arms (census)", "", "| arm | built | fuse_qkv_n | Int4Linear (before -> after fuse) | int4 expert layers | glue r1/r2/epi | decode logits | build s | score s |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        lg, c = load(D, arm)
        data[arm] = (lg, c)
        if c is None:
            L.append(f"| {arm} | MISSING | | | | | | | |")
            rep["arms"][arm] = None
            continue
        rep["arms"][arm] = {k: c.get(k) for k in ("fuse_qkv_n", "int4_attn_projections_before_fuse", "int4_attn_projections", "int4_expert_layers",
                                                    "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n", "decode_logits_shape", "build_s", "score_s", "prompts_sha256")}
        L.append(f"| {arm} | yes | {c.get('fuse_qkv_n')} | {c.get('int4_attn_projections_before_fuse')} -> {c.get('int4_attn_projections')} | {c.get('int4_expert_layers')} | "
                 f"{c.get('fuse_t1_glue_n')}/{c.get('fuse_t1_glue_r2_n')}/{c.get('fuse_router_epilogue_n')} | {c.get('decode_logits_shape')} | {c.get('build_s')} | {c.get('score_s')} |")
    shas = {arm: c.get("prompts_sha256") for arm, (lg, c) in data.items() if c}
    same_prompts = len(set(shas.values())) == 1
    L += ["", f"prompts_sha256 identical across arms: **{same_prompts}** ({list(set(shas.values()))[0][:16] if shas else 'n/a'})", ""]

    def pair(name, r, t, key="decode"):
        if data[r][0] is None or data[t][0] is None:
            rep["pairs"][name] = None
            return None
        s = kl(data[r][0], data[t][0], key)
        rep["pairs"][name] = s
        return s

    L += ["## KL pairs (fp64, full vocab, token-weighted over 16 rows x 128 decode positions unless noted)", "",
          "| pair (ref || test) | positions | kl_mean | kl_median | kl_p95 | kl_max | top-1 | exactly zero |", "|---|---|---|---|---|---|---|---|"]
    rows = [("int4 || int4_fqkv (P1)", "int4", "int4_fqkv", "decode"), ("nf4 || int4", "nf4", "int4", "decode"), ("nf4 || int4_fqkv", "nf4", "int4_fqkv", "decode"),
            ("int4 || int4_aa (P3, determinism)", "int4", "int4_aa", "decode"), ("int4 || int4_fqkv, PREFILL-last (P4)", "int4", "int4_fqkv", "prefill_last"),
            ("nf4 || int4_aa", "nf4", "int4_aa", "decode")]
    for name, r, t, key in rows:
        s = pair(name, r, t, key)
        if s is None:
            L.append(f"| {name} | -- | MISSING | | | | | |")
        else:
            L.append(f"| {name} | {s['n_tokens_scored']} | {s['kl_mean']:.6f} | {s['kl_median']:.6f} | {s['kl_p95']:.6f} | {s['kl_max_per_token']:.4f} | {s['top1_agreement']:.4f} | {s['exactly_zero']} |")
    P = rep["pairs"]
    v = rep["verdicts"]
    p1 = P.get("int4 || int4_fqkv (P1)")
    if p1:
        v["P1"] = ("HOLDS" if (p1["kl_mean"] <= 0.01 and p1["top1_agreement"] >= 0.97) else
                   ("REFUTED" if (p1["kl_mean"] > 0.03 or p1["top1_agreement"] < 0.95) else "BETWEEN BANDS (not held, not refuted)"))
        v["P1_detail"] = f"kl_mean {p1['kl_mean']:.5f} (hold <= 0.01, refute > 0.03), top-1 {p1['top1_agreement']:.4f} (hold >= 0.97, refute < 0.95)"
    a1, a2 = P.get("nf4 || int4"), P.get("nf4 || int4_fqkv")
    if a1 and a2:
        d = a2["kl_mean"] - a1["kl_mean"]
        v["P2"] = "HOLDS" if abs(d) <= 0.005 else "REFUTED"
        v["P2_detail"] = f"KL(nf4||int4_fqkv) - KL(nf4||int4) = {d:+.5f} (band +-0.005); anchors {a1['kl_mean']:.5f} / {a2['kl_mean']:.5f}"
    p3 = P.get("int4 || int4_aa (P3, determinism)")
    if p3:
        v["P3"] = "HOLDS" if p3["exactly_zero"] else "REFUTED"
        v["P3_detail"] = f"exactly_zero {p3['exactly_zero']}, kl_max {p3['kl_max_per_token']:.3e}"
    p4 = P.get("int4 || int4_fqkv, PREFILL-last (P4)")
    if p4:
        v["P4"] = "HOLDS" if p4["kl_mean"] <= 1e-4 else "REFUTED"
        v["P4_detail"] = f"prefill-last kl_mean {p4['kl_mean']:.3e} (hold <= 1e-4)"
    ci, cf, cn = rep["arms"].get("int4"), rep["arms"].get("int4_fqkv"), rep["arms"].get("nf4")
    if ci and cf and cn:
        ok = (cf["fuse_qkv_n"] == 48 and cf["int4_attn_projections"] == 96 and ci["int4_attn_projections"] == 192 and cn["int4_attn_projections"] == 0
              and ci["fuse_qkv_n"] == 0 and same_prompts)
        v["census"] = "OK" if ok else "FAILED"
        v["census_detail"] = f"fused n={cf['fuse_qkv_n']} Int4Linear {cf['int4_attn_projections']}; int4 {ci['int4_attn_projections']}; nf4 {cn['int4_attn_projections']}; same prompts {same_prompts}"
    L += ["", "## Verdicts (pre-registered rules)", ""]
    for k in ("P1", "P2", "P3", "P4", "census"):
        if k in v:
            L.append(f"- **{k}: {v[k]}** -- {v.get(k + '_detail', '')}")
        else:
            L.append(f"- **{k}: UNREAD** (arm missing)")
    dec = ("P1 and P2 hold and the census is OK -> --fuse-qkv becomes the default on the int4 lanes at B=16 too (P54's fused B=16 row is the quoted position)"
           if v.get("P1") == "HOLDS" and v.get("P2") == "HOLDS" and v.get("census") == "OK" else
           "NOT licensed: --fuse-qkv stays opt-in at B=16 (see the verdicts above)")
    L += ["", f"**Decision rule:** {dec}"]
    text = "\n".join(L)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
